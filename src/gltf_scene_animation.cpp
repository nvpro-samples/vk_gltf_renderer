/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// AnimationSystem: drives glTF animations (skeletal, morph targets,
// KHR_animation_pointer properties) by evaluating channels, interpolating
// keyframes (step/linear/cubic-spline), and applying the results back
// to scene nodes and material properties each frame.
//

#include "gltf_scene_animation.hpp"
#include "gltf_scene_editor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

#include <tinygltf/tiny_gltf.h>

#include <nvutils/logger.hpp>
#include "nvutils/parallel_work.hpp"

#include "tinygltf_utils.hpp"

namespace nvvkgltf {

//--------------------------------------------------------------------------------------------------
// Construct the animation system bound to a scene. Initializes the animation pointer subsystem
// for KHR_animation_pointer support.
AnimationSystem::AnimationSystem(Scene& scene)
    : m_scene(scene)
    , m_animationPointer(scene.getModel())
{
}

//--------------------------------------------------------------------------------------------------
// Release all parsed animation data, morph targets, skin tasks, and reset the animation pointer.
void AnimationSystem::clear()
{
  m_animations.clear();
  m_morphPrimitives.clear();
  m_skinTasks.clear();
  m_morphResults.clear();
  m_skinToNodeIndices.clear();
  m_animationPointer.reset();
}

//--------------------------------------------------------------------------------------------------
// Reset the animation pointer subsystem without clearing parsed animations.
void AnimationSystem::resetPointer()
{
  m_animationPointer.reset();
}

//--------------------------------------------------------------------------------------------------
// Parse all animations from the glTF model into internal structures.
//
// For each glTF animation, extracts samplers (keyframe times + output values) and channels
// (target node/path bindings). Supports translation, rotation, scale, weights, and
// KHR_animation_pointer channels. Sampler outputs are stored typed (vec2/vec3/vec4/float)
// based on the accessor type.
//
// Also scans all render primitives to identify morph target and skinning work:
//  - Morph: caches base geometry and allocates blending output for primitives with morph targets.
//  - Skin:  caches static vertex attributes (weights, joints, positions, normals, tangents)
//           and inverse bind matrices, then pre-allocates output vectors for each skinned primitive.
void AnimationSystem::parseAnimations()
{
  parseSamplersAndChannels();
  parseMorphPrimitives();
  parseSkinTasks();
}

//--------------------------------------------------------------------------------------------------
// Parse all glTF animation samplers (keyframe times + output values) and channels
// (target node/path bindings) into internal Animation structs.
void AnimationSystem::parseSamplersAndChannels()
{
  tinygltf::Model& m_model = m_scene.getModel();

  m_animations.clear();
  m_animations.reserve(m_model.animations.size());
  for(tinygltf::Animation& anim : m_model.animations)
  {
    Animation animation;
    animation.info.name = anim.name;
    if(animation.info.name.empty())
    {
      animation.info.name = "Animation" + std::to_string(m_animations.size());
    }

    for(auto& samp : anim.samplers)
    {
      AnimationSampler sampler;

      if(samp.interpolation == "LINEAR")
        sampler.interpolation = AnimationSampler::InterpolationType::eLinear;
      else if(samp.interpolation == "STEP")
        sampler.interpolation = AnimationSampler::InterpolationType::eStep;
      else if(samp.interpolation == "CUBICSPLINE")
        sampler.interpolation = AnimationSampler::InterpolationType::eCubicSpline;

      {
        const tinygltf::Accessor& accessor = m_model.accessors[samp.input];
        if(!tinygltf::utils::copyAccessorData(m_model, accessor, sampler.inputs))
        {
          LOGW("Invalid data type for animation input");
          continue;
        }
        for(auto input : sampler.inputs)
        {
          if(input < animation.info.start)
            animation.info.start = input;
          if(input > animation.info.end)
            animation.info.end = input;
        }
      }

      {
        const tinygltf::Accessor& accessor = m_model.accessors[samp.output];
        switch(accessor.type)
        {
          case TINYGLTF_TYPE_VEC2:
            tinygltf::utils::copyAccessorData(m_model, accessor, sampler.outputsVec2);
            break;
          case TINYGLTF_TYPE_VEC3:
            tinygltf::utils::copyAccessorData(m_model, accessor, sampler.outputsVec3);
            break;
          case TINYGLTF_TYPE_VEC4:
            tinygltf::utils::copyAccessorData(m_model, accessor, sampler.outputsVec4);
            break;
          case TINYGLTF_TYPE_SCALAR: {
            sampler.outputsFloat.resize(sampler.inputs.size());
            const size_t           elemPerKey = accessor.count / sampler.inputs.size();
            std::vector<float>     storage;
            std::span<const float> val     = tinygltf::utils::getAccessorData(m_model, accessor, &storage);
            const float*           dataPtr = val.data();
            for(size_t i = 0; i < sampler.inputs.size(); i++)
            {
              for(int j = 0; j < elemPerKey; j++)
                sampler.outputsFloat[i].push_back(*dataPtr++);
            }
            break;
          }
          default:
            LOGW("Unknown animation type: %d\n", accessor.type);
            break;
        }
      }

      animation.samplers.emplace_back(sampler);
    }

    for(auto& source : anim.channels)
    {
      AnimationChannel channel;
      if(source.target_path == "rotation")
        channel.path = AnimationChannel::PathType::eRotation;
      else if(source.target_path == "translation")
        channel.path = AnimationChannel::PathType::eTranslation;
      else if(source.target_path == "scale")
        channel.path = AnimationChannel::PathType::eScale;
      else if(source.target_path == "weights")
        channel.path = AnimationChannel::PathType::eWeights;
      else if(source.target_path == "pointer")
      {
        channel.path = AnimationChannel::PathType::ePointer;
        assert(tinygltf::utils::hasElementName(source.target_extensions, KHR_ANIMATION_POINTER));
        const tinygltf::Value& ext = tinygltf::utils::getElementValue(source.target_extensions, KHR_ANIMATION_POINTER);
        tinygltf::utils::getValue(ext, "pointer", channel.pointerPath);
      }
      channel.samplerIndex = source.sampler;
      channel.node         = source.target_node;
      animation.channels.emplace_back(channel);
    }

    animation.info.reset();
    m_animations.emplace_back(animation);
  }
}

//--------------------------------------------------------------------------------------------------
// Scan all render primitives for morph targets. For each primitive that has targets and
// mesh weights, cache the base positions/normals/tangents for blending.
void AnimationSystem::parseMorphPrimitives()
{
  m_morphPrimitives.clear();
  m_morphResults.clear();
  for(size_t renderPrimID = 0; renderPrimID < m_scene.getRenderPrimitives().size(); renderPrimID++)
  {
    const auto&                renderPrimitive = m_scene.getRenderPrimitive(renderPrimID);
    const tinygltf::Primitive& primitive       = *renderPrimitive.pPrimitive;
    const tinygltf::Mesh&      mesh            = m_scene.getModel().meshes[renderPrimitive.meshID];
    if(!primitive.targets.empty() && !mesh.weights.empty())
    {
      m_morphPrimitives.push_back(uint32_t(renderPrimID));

      MorphResult            mr;
      const tinygltf::Model& mdl = m_scene.getModel();
      mr.renderPrimID            = static_cast<int>(renderPrimID);

      auto posIt = primitive.attributes.find("POSITION");
      if(posIt != primitive.attributes.end() && posIt->second >= 0 && posIt->second < static_cast<int>(mdl.accessors.size()))
      {
        std::vector<glm::vec3> tempStorage;
        const std::span<const glm::vec3> posData = tinygltf::utils::getAccessorData(mdl, mdl.accessors[posIt->second], &tempStorage);
        mr.basePositions.assign(posData.begin(), posData.end());
      }

      bool hasNormalTargets  = false;
      bool hasTangentTargets = false;
      for(const auto& target : primitive.targets)
      {
        if(target.count("NORMAL"))
          hasNormalTargets = true;
        if(target.count("TANGENT"))
          hasTangentTargets = true;
      }

      if(hasNormalTargets)
      {
        auto nrmIt = primitive.attributes.find("NORMAL");
        if(nrmIt != primitive.attributes.end() && nrmIt->second >= 0 && nrmIt->second < static_cast<int>(mdl.accessors.size()))
        {
          std::vector<glm::vec3>           tempStorage;
          const std::span<const glm::vec3> nrmData =
              tinygltf::utils::getAccessorData(mdl, mdl.accessors[nrmIt->second], &tempStorage);
          mr.baseNormals.assign(nrmData.begin(), nrmData.end());
        }
      }

      if(hasTangentTargets)
      {
        auto tanIt = primitive.attributes.find("TANGENT");
        if(tanIt != primitive.attributes.end() && tanIt->second >= 0 && tanIt->second < static_cast<int>(mdl.accessors.size()))
        {
          std::vector<glm::vec4>           tempStorage;
          const std::span<const glm::vec4> tanData =
              tinygltf::utils::getAccessorData(mdl, mdl.accessors[tanIt->second], &tempStorage);
          mr.baseTangents.assign(tanData.begin(), tanData.end());
        }
      }

      m_morphResults.push_back(std::move(mr));
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Scan all render nodes for skinned primitives. For each unique skinned primitive, cache the
// static vertex attributes (weights, joints, positions, normals, tangents) and inverse bind
// matrices for use by both the GPU compute path and the CPU fallback.
void AnimationSystem::parseSkinTasks()
{
  m_skinTasks.clear();

  std::unordered_set<int> seenPrimIDs;
  const auto&             rnodes = m_scene.getRenderNodeRegistry().getRenderNodes();
  const tinygltf::Model&  model  = m_scene.getModel();
  for(size_t rnID = 0; rnID < rnodes.size(); rnID++)
  {
    const auto& rn = rnodes[rnID];
    if(rn.skinID < 0 || !seenPrimIDs.insert(rn.renderPrimID).second)
      continue;

    SkinTask task;
    task.renderPrimID = rn.renderPrimID;
    task.skinID       = rn.skinID;
    task.refNodeID    = rn.refNodeID;

    const tinygltf::Primitive& primitive = *m_scene.getRenderPrimitive(rn.renderPrimID).pPrimitive;

    std::vector<glm::vec4> tempW;
    auto                   wSpan = tinygltf::utils::getAttributeData3(model, primitive, "WEIGHTS_0", &tempW);
    task.weights.assign(wSpan.begin(), wSpan.end());
    std::vector<glm::ivec4> tempJ;
    auto                    jSpan = tinygltf::utils::getAttributeData3(model, primitive, "JOINTS_0", &tempJ);
    task.joints.assign(jSpan.begin(), jSpan.end());
    std::vector<glm::vec3> tempP;
    auto                   pSpan = tinygltf::utils::getAttributeData3(model, primitive, "POSITION", &tempP);
    task.basePositions.assign(pSpan.begin(), pSpan.end());
    std::vector<glm::vec3> tempN;
    auto                   nSpan = tinygltf::utils::getAttributeData3(model, primitive, "NORMAL", &tempN);
    task.baseNormals.assign(nSpan.begin(), nSpan.end());
    std::vector<glm::vec4> tempT;
    auto                   tSpan = tinygltf::utils::getAttributeData3(model, primitive, "TANGENT", &tempT);
    task.baseTangents.assign(tSpan.begin(), tSpan.end());

    if(rn.skinID >= 0 && rn.skinID < static_cast<int>(model.skins.size()))
    {
      const tinygltf::Skin& skin = model.skins[rn.skinID];
      if(skin.inverseBindMatrices >= 0 && skin.inverseBindMatrices < static_cast<int>(model.accessors.size()))
      {
        std::vector<glm::mat4>     ibmStorage;
        std::span<const glm::mat4> ibm =
            tinygltf::utils::getAccessorData(model, model.accessors[skin.inverseBindMatrices], &ibmStorage);
        task.inverseBindMatrices.assign(ibm.begin(), ibm.end());
      }
    }

    m_skinTasks.push_back(std::move(task));
  }

  buildSkinToNodeMap();
}

//--------------------------------------------------------------------------------------------------
// Build reverse lookup: skinID → list of node indices that reference that skin.
// Replaces the O(skins * nodes) brute-force scan in updateAnimation().
void AnimationSystem::buildSkinToNodeMap()
{
  const tinygltf::Model& model = m_scene.getModel();
  m_skinToNodeIndices.clear();
  m_skinToNodeIndices.resize(model.skins.size());
  for(int nodeIdx = 0; nodeIdx < static_cast<int>(model.nodes.size()); ++nodeIdx)
  {
    int skinIdx = model.nodes[nodeIdx].skin;
    if(skinIdx >= 0 && skinIdx < static_cast<int>(model.skins.size()))
      m_skinToNodeIndices[skinIdx].push_back(nodeIdx);
  }
}

//--------------------------------------------------------------------------------------------------
// Advance a single animation and apply its results to the scene.
//
// Evaluates all channels of the given animation at its current time, writing interpolated
// values back to glTF node transforms and mesh weights. Pointer channels are forwarded to
// the AnimationPointerSystem for material/light property animation.
//
// After channel evaluation, propagates dirty state: nodes whose joints moved are marked
// dirty (triggering world matrix recomputation), and animated materials/lights are flagged
// for GPU re-upload.
//
// Returns true if any scene state changed (nodes, materials, lights, or morph weights).
bool AnimationSystem::updateAnimation(uint32_t animationIndex)
{
  tinygltf::Model& m_model  = m_scene.getModel();
  const size_t     numNodes = m_model.nodes.size();

  Animation& animation = m_animations[animationIndex];
  float      time      = animation.info.currentTime;

  // Flat bit vector for O(1) dirty checks, with companion list for O(dirty) iteration
  std::vector<bool> dirtyBits(numNodes, false);
  std::vector<int>  dirtyList;
  bool              hadWeightsChannel = false;

  auto setDirty = [&](int idx) {
    if(!dirtyBits[idx])
    {
      dirtyBits[idx] = true;
      dirtyList.push_back(idx);
    }
  };

  for(auto& channel : animation.channels)
  {
    AnimationSampler& sampler = animation.samplers[channel.samplerIndex];

    if(channel.path == AnimationChannel::PathType::ePointer)
    {
      processAnimationChannel(nullptr, sampler, channel, time);
      continue;
    }

    if(channel.node < 0 || channel.node >= static_cast<int>(numNodes))
      continue;

    tinygltf::Node& gltfNode = m_model.nodes[channel.node];
    processAnimationChannel(&gltfNode, sampler, channel, time);
    setDirty(channel.node);
    if(channel.path == AnimationChannel::PathType::eWeights)
      hadWeightsChannel = true;
  }

  m_animationPointer.syncToModel();
  const auto& animDirtyNodes = m_animationPointer.getDirtyNodes();
  for(int nodeIndex : animDirtyNodes)
  {
    if(nodeIndex >= 0 && static_cast<size_t>(nodeIndex) < numNodes)
      setDirty(nodeIndex);
    m_scene.editor().updateVisibility(nodeIndex);
  }

  // Use precomputed skin-to-node map instead of O(skins*nodes) brute-force scan
  for(size_t skinIdx = 0; skinIdx < m_model.skins.size(); ++skinIdx)
  {
    const auto& skin          = m_model.skins[skinIdx];
    bool        jointAnimated = false;
    for(int jointNodeId : skin.joints)
    {
      if(jointNodeId >= 0 && static_cast<size_t>(jointNodeId) < numNodes && dirtyBits[jointNodeId])
      {
        jointAnimated = true;
        break;
      }
    }
    if(!jointAnimated)
      continue;
    if(skinIdx < m_skinToNodeIndices.size())
    {
      for(int nodeIdx : m_skinToNodeIndices[skinIdx])
        setDirty(nodeIdx);
    }
  }

  for(int nodeIdx : dirtyList)
    m_scene.markNodeDirty(nodeIdx);

  bool hadPointerDirty = m_animationPointer.hasDirty();
  if(hadPointerDirty)
  {
    for(int matIdx : m_animationPointer.getDirtyMaterials())
      m_scene.markMaterialDirty(matIdx);
    for(int lightIdx : m_animationPointer.getDirtyLights())
      m_scene.markLightDirty(lightIdx);
    m_animationPointer.clearDirty();
  }

  return !dirtyList.empty() || hadPointerDirty || hadWeightsChannel;
}

//--------------------------------------------------------------------------------------------------
// Evaluate a single animation channel at the given time. Finds the keyframe segment that
// contains `time`, computes the interpolation factor, and dispatches to the appropriate
// interpolation handler (linear, step, or cubic spline). Returns true if the channel was
// animated (i.e. time fell within the sampler's keyframe range).
bool AnimationSystem::processAnimationChannel(tinygltf::Node* gltfNode, AnimationSampler& sampler, const AnimationChannel& channel, float time)
{
  if(sampler.inputs.size() < 2)
    return false;

  // Binary search: find first keyframe strictly after `time`
  auto it = std::upper_bound(sampler.inputs.begin(), sampler.inputs.end(), time);
  if(it == sampler.inputs.begin())
    return false;

  size_t i = static_cast<size_t>(std::distance(sampler.inputs.begin(), it)) - 1;
  if(i + 1 >= sampler.inputs.size())
    i = sampler.inputs.size() - 2;

  float inputStart = sampler.inputs[i];
  float inputEnd   = sampler.inputs[i + 1];
  if(time < inputStart || time > inputEnd)
    return false;

  float t = calculateInterpolationFactor(inputStart, inputEnd, time);
  switch(sampler.interpolation)
  {
    case AnimationSampler::InterpolationType::eLinear:
      handleLinearInterpolation(gltfNode, sampler, channel, t, i);
      break;
    case AnimationSampler::InterpolationType::eStep:
      handleStepInterpolation(gltfNode, sampler, channel, i);
      break;
    case AnimationSampler::InterpolationType::eCubicSpline:
      handleCubicSplineInterpolation(gltfNode, sampler, channel, t, inputEnd - inputStart, i);
      break;
  }
  return true;
}

//--------------------------------------------------------------------------------------------------
// Compute the normalized interpolation factor t in [0,1] for a time within a keyframe segment.
// Returns 0 if the segment duration is effectively zero.
float AnimationSystem::calculateInterpolationFactor(float inputStart, float inputEnd, float time)
{
  float keyDelta = inputEnd - inputStart;
  if(std::abs(keyDelta) < std::numeric_limits<float>::epsilon())
    return 0.0f;
  return std::clamp((time - inputStart) / keyDelta, 0.0f, 1.0f);
}

//--------------------------------------------------------------------------------------------------
// Cubic Hermite spline interpolation as defined by glTF 2.0 spec section 3.8.4.4.
// Each keyframe stores three values: in-tangent (A), value (V), and out-tangent (B).
// The formula evaluates: v0*cV0 + a*cA + b*cB + v1*cV1 using Hermite basis functions.
namespace {
template <class T>
T computeCubicInterpolation(const T* values, float t, float keyDelta, size_t index)
{
  const float  tSq       = t * t;
  const float  tCb       = tSq * t;
  const float  tD        = keyDelta;
  const float  cV1       = -2 * tCb + 3 * tSq;
  const float  cV0       = 1 - cV1;
  const float  cA        = tD * (tCb - tSq);
  const float  cB        = tD * (tCb - 2 * tSq + t);
  const size_t prevIndex = index * 3;
  const size_t nextIndex = (index + 1) * 3;
  const size_t A = 0, V = 1, B = 2;
  const T&     v0 = values[prevIndex + V];
  const T&     a  = values[nextIndex + A];
  const T&     b  = values[prevIndex + B];
  const T&     v1 = values[nextIndex + V];
  return v0 * cV0 + a * cA + b * cB + v1 * cV1;
}
}  // namespace

//--------------------------------------------------------------------------------------------------
// Apply linear interpolation for a channel at the given keyframe index with factor t.
// Rotation uses quaternion slerp; translation, scale, and weights use component-wise lerp.
// Pointer channels delegate to the AnimationPointerSystem based on output dimensionality.
void AnimationSystem::handleLinearInterpolation(tinygltf::Node*         gltfNode,
                                                AnimationSampler&       sampler,
                                                const AnimationChannel& channel,
                                                float                   t,
                                                size_t                  index)
{
  tinygltf::Model& m_model = m_scene.getModel();

  switch(channel.path)
  {
    case AnimationChannel::PathType::eRotation: {
      if(index + 1 >= sampler.outputsVec4.size())
        break;
      const glm::quat q1 = glm::make_quat(glm::value_ptr(sampler.outputsVec4[index]));
      const glm::quat q2 = glm::make_quat(glm::value_ptr(sampler.outputsVec4[index + 1]));
      glm::quat       q  = glm::normalize(glm::slerp(q1, q2, t));
      if(gltfNode)
        gltfNode->rotation = {q.x, q.y, q.z, q.w};
      break;
    }
    case AnimationChannel::PathType::eTranslation: {
      if(index + 1 >= sampler.outputsVec3.size())
        break;
      glm::vec3 trans = glm::mix(sampler.outputsVec3[index], sampler.outputsVec3[index + 1], t);
      if(gltfNode)
        gltfNode->translation = {trans.x, trans.y, trans.z};
      break;
    }
    case AnimationChannel::PathType::eScale: {
      if(index + 1 >= sampler.outputsVec3.size())
        break;
      glm::vec3 s = glm::mix(sampler.outputsVec3[index], sampler.outputsVec3[index + 1], t);
      if(gltfNode)
        gltfNode->scale = {s.x, s.y, s.z};
      break;
    }
    case AnimationChannel::PathType::eWeights: {
      if(gltfNode && gltfNode->mesh >= 0 && (size_t)gltfNode->mesh < m_model.meshes.size()
         && index + 1 < sampler.outputsFloat.size()
         && sampler.outputsFloat[index].size() == sampler.outputsFloat[index + 1].size())
      {
        tinygltf::Mesh& mesh = m_model.meshes[gltfNode->mesh];
        if(mesh.weights.size() != sampler.outputsFloat[index].size())
          mesh.weights.resize(sampler.outputsFloat[index].size());
        for(size_t j = 0; j < mesh.weights.size(); j++)
          mesh.weights[j] = glm::mix(sampler.outputsFloat[index][j], sampler.outputsFloat[index + 1][j], t);
      }
      break;
    }
    case AnimationChannel::PathType::ePointer: {
      if(!sampler.outputsVec4.empty() && index + 1 < sampler.outputsVec4.size())
        m_animationPointer.applyValue(channel.pointerPath,
                                      glm::mix(sampler.outputsVec4[index], sampler.outputsVec4[index + 1], t));
      else if(!sampler.outputsVec3.empty() && index + 1 < sampler.outputsVec3.size())
        m_animationPointer.applyValue(channel.pointerPath,
                                      glm::mix(sampler.outputsVec3[index], sampler.outputsVec3[index + 1], t));
      else if(!sampler.outputsVec2.empty() && index + 1 < sampler.outputsVec2.size())
        m_animationPointer.applyValue(channel.pointerPath,
                                      glm::mix(sampler.outputsVec2[index], sampler.outputsVec2[index + 1], t));
      else if(!sampler.outputsFloat.empty() && index + 1 < sampler.outputsFloat.size() && !sampler.outputsFloat[index].empty())
        m_animationPointer.applyValue(channel.pointerPath,
                                      glm::mix(sampler.outputsFloat[index][0], sampler.outputsFloat[index + 1][0], t));
      break;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Apply step interpolation: use the value at the start of the keyframe segment without blending.
void AnimationSystem::handleStepInterpolation(tinygltf::Node* gltfNode, AnimationSampler& sampler, const AnimationChannel& channel, size_t index)
{
  switch(channel.path)
  {
    case AnimationChannel::PathType::eRotation: {
      if(index >= sampler.outputsVec4.size())
        break;
      glm::quat q(sampler.outputsVec4[index]);
      if(gltfNode)
        gltfNode->rotation = {q.x, q.y, q.z, q.w};
      break;
    }
    case AnimationChannel::PathType::eTranslation: {
      if(index >= sampler.outputsVec3.size())
        break;
      glm::vec3 trans(sampler.outputsVec3[index]);
      if(gltfNode)
        gltfNode->translation = {trans.x, trans.y, trans.z};
      break;
    }
    case AnimationChannel::PathType::eScale: {
      if(index >= sampler.outputsVec3.size())
        break;
      glm::vec3 s(sampler.outputsVec3[index]);
      if(gltfNode)
        gltfNode->scale = {s.x, s.y, s.z};
      break;
    }
    case AnimationChannel::PathType::eWeights:
      break;
    case AnimationChannel::PathType::ePointer: {
      if(!sampler.outputsVec4.empty() && index < sampler.outputsVec4.size())
        m_animationPointer.applyValue(channel.pointerPath, sampler.outputsVec4[index]);
      else if(!sampler.outputsVec3.empty() && index < sampler.outputsVec3.size())
        m_animationPointer.applyValue(channel.pointerPath, sampler.outputsVec3[index]);
      else if(!sampler.outputsVec2.empty() && index < sampler.outputsVec2.size())
        m_animationPointer.applyValue(channel.pointerPath, sampler.outputsVec2[index]);
      else if(!sampler.outputsFloat.empty() && index < sampler.outputsFloat.size() && !sampler.outputsFloat[index].empty())
        m_animationPointer.applyValue(channel.pointerPath, sampler.outputsFloat[index][0]);
      break;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Apply cubic spline interpolation using the Hermite basis. Rotation results are re-normalized
// to maintain unit quaternion validity. Pointer channels are dispatched by output dimensionality.
void AnimationSystem::handleCubicSplineInterpolation(tinygltf::Node*         gltfNode,
                                                     AnimationSampler&       sampler,
                                                     const AnimationChannel& channel,
                                                     float                   t,
                                                     float                   keyDelta,
                                                     size_t                  index)
{
  const size_t maxRequiredIndex = (index + 1) * 3 + 1;

  if(channel.path == AnimationChannel::PathType::ePointer)
  {
    if(!sampler.outputsVec4.empty() && sampler.outputsVec4.size() > maxRequiredIndex)
      m_animationPointer.applyValue(channel.pointerPath,
                                    computeCubicInterpolation<glm::vec4>(sampler.outputsVec4.data(), t, keyDelta, index));
    else if(!sampler.outputsVec3.empty() && sampler.outputsVec3.size() > maxRequiredIndex)
      m_animationPointer.applyValue(channel.pointerPath,
                                    computeCubicInterpolation<glm::vec3>(sampler.outputsVec3.data(), t, keyDelta, index));
    else if(!sampler.outputsVec2.empty() && sampler.outputsVec2.size() > maxRequiredIndex)
      m_animationPointer.applyValue(channel.pointerPath,
                                    computeCubicInterpolation<glm::vec2>(sampler.outputsVec2.data(), t, keyDelta, index));
    return;
  }

  if(!gltfNode)
    return;

  if(channel.path == AnimationChannel::PathType::eRotation)
  {
    if(sampler.outputsVec4.size() > maxRequiredIndex)
    {
      glm::vec4 result     = computeCubicInterpolation<glm::vec4>(sampler.outputsVec4.data(), t, keyDelta, index);
      glm::quat quatResult = glm::make_quat(glm::value_ptr(result));
      quatResult           = glm::normalize(quatResult);
      gltfNode->rotation   = {quatResult.x, quatResult.y, quatResult.z, quatResult.w};
    }
  }
  else if(sampler.outputsVec3.size() > maxRequiredIndex)
  {
    glm::vec3 result = computeCubicInterpolation<glm::vec3>(sampler.outputsVec3.data(), t, keyDelta, index);
    if(channel.path == AnimationChannel::PathType::eTranslation)
      gltfNode->translation = {result.x, result.y, result.z};
    else if(channel.path == AnimationChannel::PathType::eScale)
      gltfNode->scale = {result.x, result.y, result.z};
  }
}

//========== Deferred CPU Output Allocation ==========

//--------------------------------------------------------------------------------------------------
// Allocate the CPU-side skinning output vectors (positions, normals, tangents) if not already
// sized. Only called by the CPU fallback path (computeSkinning); the GPU compute path skips
// this entirely, avoiding the memory cost of duplicate vertex data.
void SkinTask::ensureCpuOutput()
{
  size_t vertexCount = weights.size();
  if(result.positions.size() == vertexCount)
    return;
  result.positions.resize(vertexCount);
  if(!baseNormals.empty())
    result.normals.resize(vertexCount);
  if(!baseTangents.empty())
    result.tangents.resize(vertexCount);
}

//--------------------------------------------------------------------------------------------------
// Allocate the CPU-side morph blending output vectors if not already sized. Only called by
// the CPU fallback path (computeMorphTargets); the GPU compute path skips this entirely.
void MorphResult::ensureCpuOutput()
{
  if(blendedPositions.size() == basePositions.size())
    return;
  blendedPositions.resize(basePositions.size());
  if(!baseNormals.empty())
    blendedNormals.resize(baseNormals.size());
  if(!baseTangents.empty())
    blendedTangents.resize(baseTangents.size());
}

//========== CPU Fallback Animation ==========

//--------------------------------------------------------------------------------------------------
// CPU skinning fallback: transform vertices by their joint influences for all skinned primitives.
//
// For each SkinTask, computes per-joint matrices as: inverse(nodeWorld) * jointWorld * IBM,
// then derives normal matrices via inverse-transpose of the upper 3x3.  Vertices are
// transformed in parallel batches using up to 4 joint influences per vertex.  Results are
// written directly into the SkinTask::result vectors.
void AnimationSystem::computeSkinning()
{
  const tinygltf::Model&        model        = m_scene.getModel();
  const std::vector<glm::mat4>& nodeMatrices = m_scene.getNodesWorldMatrices();

  for(SkinTask& task : m_skinTasks)
  {
    if(task.skinID < 0 || task.skinID >= static_cast<int>(model.skins.size()))
      continue;
    if(task.refNodeID < 0 || task.refNodeID >= static_cast<int>(nodeMatrices.size()))
      continue;

    task.ensureCpuOutput();

    const tinygltf::Skin& skin        = model.skins[task.skinID];
    const size_t          numJoints   = skin.joints.size();
    const size_t          vertexCount = task.weights.size();
    const bool            hasNormals  = !task.baseNormals.empty();
    const bool            hasTangents = !task.baseTangents.empty();

    // Compute joint matrices (only per-frame work: invNode * worldJoint * IBM)
    m_jointMatrices.resize(numJoints);
    glm::mat4 invNode = glm::inverse(nodeMatrices[task.refNodeID]);
    for(size_t i = 0; i < numJoints; ++i)
    {
      int jointNodeID = skin.joints[i];
      if(jointNodeID < 0 || jointNodeID >= static_cast<int>(nodeMatrices.size()))
      {
        m_jointMatrices[i] = glm::mat4(1);
        continue;
      }
      m_jointMatrices[i] = invNode * nodeMatrices[jointNodeID]
                           * (i < task.inverseBindMatrices.size() ? task.inverseBindMatrices[i] : glm::mat4(1));
    }

    // Pre-compute normal matrices (inverse-transpose of upper 3x3)
    m_normalMatrices.resize(numJoints);
    for(size_t i = 0; i < numJoints; ++i)
      m_normalMatrices[i] = glm::transpose(glm::inverse(glm::mat3(m_jointMatrices[i])));

    // Apply skinning directly into pre-allocated result vectors
    SkinningResult& result        = task.result;
    const auto&     jointMatrices = m_jointMatrices;
    const auto&     normalMats    = m_normalMatrices;

    nvutils::parallel_batches<2048>(vertexCount, [&](uint64_t v) {
      const glm::vec4&  w = task.weights[v];
      const glm::ivec4& j = task.joints[v];

      glm::vec3 skinnedPos(0.0f);
      glm::vec3 skinnedNrm(0.0f);
      glm::vec3 skinnedTan(0.0f);

      for(int i = 0; i < 4; ++i)
      {
        const float jointWeight = w[i];
        if(jointWeight > 0.0f)
        {
          const int        jointIndex = j[i];
          const bool       validJoint = jointIndex >= 0 && static_cast<size_t>(jointIndex) < numJoints;
          const glm::mat4& jMat       = validJoint ? jointMatrices[jointIndex] : glm::identity<glm::mat4>();
          const glm::mat3& nMat       = validJoint ? normalMats[jointIndex] : glm::identity<glm::mat3>();

          skinnedPos += jointWeight * glm::vec3(jMat * glm::vec4(task.basePositions[v], 1.0f));
          if(hasNormals)
            skinnedNrm += jointWeight * (nMat * task.baseNormals[v]);
          if(hasTangents)
            skinnedTan += jointWeight * (glm::mat3(jMat) * glm::vec3(task.baseTangents[v]));
        }
      }

      result.positions[v] = skinnedPos;
      if(hasNormals)
        result.normals[v] = glm::normalize(skinnedNrm);
      if(hasTangents)
        result.tangents[v] = glm::vec4(glm::normalize(skinnedTan), task.baseTangents[v].w);
    });
  }
}

//--------------------------------------------------------------------------------------------------
// Return the skinning result (positions, normals, tangents) for the given skin task index.
const SkinningResult& AnimationSystem::getSkinningResult(size_t skinTaskIndex) const
{
  return m_skinTasks[skinTaskIndex].result;
}

//--------------------------------------------------------------------------------------------------
// CPU morph target blending fallback: accumulate weighted deltas from all active morph targets.
//
// For each morph primitive, starts from the cached base geometry and adds position, normal,
// and tangent deltas scaled by their respective mesh weights. Deltas are read directly from
// glTF accessors; blending is parallelized per-vertex. Normals are re-normalized after
// accumulation. Results are stored in MorphResult::blendedPositions/Normals/Tangents.
void AnimationSystem::computeMorphTargets()
{
  const tinygltf::Model& model = m_scene.getModel();

  for(size_t mi = 0; mi < m_morphResults.size(); mi++)
  {
    MorphResult& mr = m_morphResults[mi];
    if(mr.renderPrimID < 0 || mr.basePositions.empty())
      continue;

    mr.ensureCpuOutput();

    const RenderPrimitive& renderPrimitive = m_scene.getRenderPrimitive(mr.renderPrimID);
    if(renderPrimitive.meshID < 0 || renderPrimitive.meshID >= static_cast<int>(model.meshes.size()))
      continue;
    const tinygltf::Primitive& primitive = *renderPrimitive.pPrimitive;
    const tinygltf::Mesh&      mesh      = model.meshes[renderPrimitive.meshID];

    const bool hasNormals  = !mr.baseNormals.empty();
    const bool hasTangents = !mr.baseTangents.empty();

    std::copy(mr.basePositions.begin(), mr.basePositions.end(), mr.blendedPositions.begin());
    if(hasNormals)
      std::copy(mr.baseNormals.begin(), mr.baseNormals.end(), mr.blendedNormals.begin());
    if(hasTangents)
      std::copy(mr.baseTangents.begin(), mr.baseTangents.end(), mr.blendedTangents.begin());

    for(size_t targetIndex = 0; targetIndex < primitive.targets.size(); ++targetIndex)
    {
      if(targetIndex >= mesh.weights.size())
        continue;
      float weight = float(mesh.weights[targetIndex]);
      if(weight == 0.0f)
        continue;

      const auto& target = primitive.targets[targetIndex];

      auto posFind = target.find("POSITION");
      if(posFind != target.end())
      {
        int idx = posFind->second;
        if(idx >= 0 && idx < static_cast<int>(model.accessors.size()))
        {
          const std::span<const glm::vec3> deltas = tinygltf::utils::getAccessorData(model, model.accessors[idx], &m_morphTempVec3);
          if(deltas.size() == mr.blendedPositions.size())
            nvutils::parallel_batches(mr.blendedPositions.size(),
                                      [&](uint64_t v) { mr.blendedPositions[v] += weight * deltas[v]; });
        }
      }

      if(hasNormals)
      {
        auto nrmFind = target.find("NORMAL");
        if(nrmFind != target.end())
        {
          int idx = nrmFind->second;
          if(idx >= 0 && idx < static_cast<int>(model.accessors.size()))
          {
            const std::span<const glm::vec3> deltas =
                tinygltf::utils::getAccessorData(model, model.accessors[idx], &m_morphTempVec3);
            if(deltas.size() == mr.blendedNormals.size())
              nvutils::parallel_batches(mr.blendedNormals.size(),
                                        [&](uint64_t v) { mr.blendedNormals[v] += weight * deltas[v]; });
          }
        }
      }

      if(hasTangents)
      {
        auto tanFind = target.find("TANGENT");
        if(tanFind != target.end())
        {
          int idx = tanFind->second;
          if(idx >= 0 && idx < static_cast<int>(model.accessors.size()))
          {
            const std::span<const glm::vec3> deltas =
                tinygltf::utils::getAccessorData(model, model.accessors[idx], &m_morphTempVec3);
            if(deltas.size() == mr.blendedTangents.size())
              nvutils::parallel_batches(mr.blendedTangents.size(), [&](uint64_t v) {
                mr.blendedTangents[v] =
                    glm::vec4(glm::vec3(mr.blendedTangents[v]) + weight * deltas[v], mr.blendedTangents[v].w);
              });
          }
        }
      }
    }

    if(hasNormals)
      nvutils::parallel_batches(mr.blendedNormals.size(),
                                [&](uint64_t v) { mr.blendedNormals[v] = glm::normalize(mr.blendedNormals[v]); });
  }
}

//--------------------------------------------------------------------------------------------------
// Return the morph blending result (base + blended geometry) for the given morph task index.
const MorphResult& AnimationSystem::getMorphResult(size_t morphTaskIndex) const
{
  return m_morphResults[morphTaskIndex];
}

}  // namespace nvvkgltf
