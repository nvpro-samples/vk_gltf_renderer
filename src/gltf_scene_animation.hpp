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

#pragma once

#include <cstdint>
#include <vector>

#include "gltf_animation_pointer.hpp"
#include "gltf_scene.hpp"

namespace nvvkgltf {

struct SkinningResult
{
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec4> tangents;
};

// Per-primitive skinning task with cached static data (read once at parse time).
// The cached vertex attributes and inverse bind matrices are exactly the data a future
// compute shader would upload as GPU SSBOs. The result vectors are pre-allocated and
// rewritten in-place each frame (no per-frame allocation).
struct SkinTask
{
  int renderPrimID;
  int skinID;
  int refNodeID;

  // Static data cached at parse time (never changes per frame)
  std::vector<glm::vec4>  weights;              // WEIGHTS_0
  std::vector<glm::ivec4> joints;               // JOINTS_0
  std::vector<glm::vec3>  basePositions;        // POSITION
  std::vector<glm::vec3>  baseNormals;          // NORMAL (empty if absent)
  std::vector<glm::vec4>  baseTangents;         // TANGENT (empty if absent)
  std::vector<glm::mat4>  inverseBindMatrices;  // IBM for this skin

  // Pre-allocated output (sized once, rewritten each frame)
  SkinningResult result;
};

struct MorphResult
{
  int                    renderPrimID = -1;
  std::vector<glm::vec3> basePositions;     // Cached at parse time
  std::vector<glm::vec3> baseNormals;       // Cached at parse time (empty if absent)
  std::vector<glm::vec4> baseTangents;      // Cached at parse time (empty if absent)
  std::vector<glm::vec3> blendedPositions;  // Rewritten each frame
  std::vector<glm::vec3> blendedNormals;    // Rewritten each frame (empty if no normal targets)
  std::vector<glm::vec4> blendedTangents;   // Rewritten each frame (empty if no tangent targets)
};

/*-------------------------------------------------------------------------------------------------
# class nvvkgltf::AnimationSystem

>  Handles animation parsing and update: parseAnimations, updateAnimation, interpolation.
   Friend of Scene for direct access to model and dirty tracking. Scene retains forwarding
   wrappers so external call sites are unchanged.

 -------------------------------------------------------------------------------------------------*/
class AnimationSystem
{
public:
  explicit AnimationSystem(Scene& scene);

  void parseAnimations();
  void clear();
  void resetPointer();

  [[nodiscard]] bool updateAnimation(uint32_t animationIndex);

  [[nodiscard]] int                       getNumAnimations() const { return static_cast<int>(m_animations.size()); }
  [[nodiscard]] bool                      hasAnimation() const { return !m_animations.empty(); }
  nvvkgltf::AnimationInfo&                getAnimationInfo(int index) { return m_animations[index].info; }
  nvvkgltf::AnimationPointerSystem&       getAnimationPointer() { return m_animationPointer; }
  const nvvkgltf::AnimationPointerSystem& getAnimationPointer() const { return m_animationPointer; }
  const std::vector<uint32_t>&            getMorphPrimitives() const { return m_morphPrimitives; }
  bool                                    hasMorphTargets() const { return !m_morphPrimitives.empty(); }
  void                                    computeMorphTargets();
  const MorphResult&                      getMorphResult(size_t morphTaskIndex) const;

  const std::vector<SkinTask>& getSkinTasks() const { return m_skinTasks; }
  bool                         hasSkinning() const { return !m_skinTasks.empty(); }
  void                         computeSkinning();
  const SkinningResult&        getSkinningResult(size_t skinTaskIndex) const;

private:
  Scene& m_scene;

  struct AnimationChannel
  {
    enum PathType
    {
      eTranslation,
      eRotation,
      eScale,
      eWeights,
      ePointer
    };
    PathType    path         = eTranslation;
    int         node         = -1;
    uint32_t    samplerIndex = 0;
    std::string pointerPath;
  };

  struct AnimationSampler
  {
    enum InterpolationType
    {
      eLinear,
      eStep,
      eCubicSpline
    };
    InterpolationType               interpolation = eLinear;
    std::vector<float>              inputs;
    std::vector<glm::vec2>          outputsVec2;
    std::vector<glm::vec3>          outputsVec3;
    std::vector<glm::vec4>          outputsVec4;
    std::vector<std::vector<float>> outputsFloat;
  };

  struct Animation
  {
    AnimationInfo                 info;
    std::vector<AnimationSampler> samplers;
    std::vector<AnimationChannel> channels;
  };

  std::vector<Animation>           m_animations;
  nvvkgltf::AnimationPointerSystem m_animationPointer;
  std::vector<uint32_t>            m_morphPrimitives;
  std::vector<SkinTask>            m_skinTasks;

  std::vector<MorphResult> m_morphResults;

  // Reused across morph targets each frame to avoid per-target allocations
  std::vector<glm::vec3> m_morphTempVec3;

  // Reused across skin tasks each frame (only normalMatrices and jointMatrices need per-frame workspace)
  std::vector<glm::mat3> m_normalMatrices;
  std::vector<glm::mat4> m_jointMatrices;

  bool processAnimationChannel(tinygltf::Node* gltfNode, AnimationSampler& sampler, const AnimationChannel& channel, float time);
  float calculateInterpolationFactor(float inputStart, float inputEnd, float time);
  void handleLinearInterpolation(tinygltf::Node* gltfNode, AnimationSampler& sampler, const AnimationChannel& channel, float t, size_t index);
  void handleStepInterpolation(tinygltf::Node* gltfNode, AnimationSampler& sampler, const AnimationChannel& channel, size_t index);
  void handleCubicSplineInterpolation(tinygltf::Node*         gltfNode,
                                      AnimationSampler&       sampler,
                                      const AnimationChannel& channel,
                                      float                   t,
                                      float                   keyDelta,
                                      size_t                  index);
};

}  // namespace nvvkgltf
