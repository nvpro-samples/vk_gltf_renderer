/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Merges a source glTF model into a base model by appending all resources
// (nodes, meshes, materials, textures, buffers, skins, animations, etc.)
// and remapping every internal index so both models coexist correctly
// within a single tinygltf::Model.
//

#include "gltf_scene_merger.hpp"
#include "tinygltf_utils.hpp"
#include <algorithm>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>

namespace {

using namespace nvvkgltf;

IndexRemapping computeOffsets(const tinygltf::Model& baseModel)
{
  IndexRemapping o;
  o.nodes       = static_cast<int>(baseModel.nodes.size());
  o.meshes      = static_cast<int>(baseModel.meshes.size());
  o.materials   = static_cast<int>(baseModel.materials.size());
  o.textures    = static_cast<int>(baseModel.textures.size());
  o.images      = static_cast<int>(baseModel.images.size());
  o.samplers    = static_cast<int>(baseModel.samplers.size());
  o.accessors   = static_cast<int>(baseModel.accessors.size());
  o.bufferViews = static_cast<int>(baseModel.bufferViews.size());
  o.buffers     = static_cast<int>(baseModel.buffers.size());
  o.skins       = static_cast<int>(baseModel.skins.size());
  o.cameras     = static_cast<int>(baseModel.cameras.size());
  o.animations  = static_cast<int>(baseModel.animations.size());
  o.lights      = 0;
  if(baseModel.extensions.count(KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME))
  {
    const tinygltf::Value& ext = baseModel.extensions.at(KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME);
    if(ext.Has("lights") && ext.Get("lights").IsArray())
    {
      o.lights = static_cast<int>(ext.Get("lights").ArrayLen());
    }
  }
  return o;
}

static void remapTextureInfo(tinygltf::TextureInfo& info, int textureOffset)
{
  if(info.index >= 0)
    info.index += textureOffset;
}

static void remapNormalTextureInfo(tinygltf::NormalTextureInfo& info, int textureOffset)
{
  if(info.index >= 0)
    info.index += textureOffset;
}

static void remapOcclusionTextureInfo(tinygltf::OcclusionTextureInfo& info, int textureOffset)
{
  if(info.index >= 0)
    info.index += textureOffset;
}

// Recursively remap every "index" in material extension JSON (all are texture indices).
static void remapTextureIndicesInValue(tinygltf::Value& val, int textureOffset)
{
  if(val.IsObject())
  {
    auto& obj = val.Get<tinygltf::Value::Object>();
    for(auto& kv : obj)
    {
      if(kv.first == "index" && (kv.second.IsInt() || kv.second.IsNumber()))
      {
        int idx = kv.second.GetNumberAsInt();
        if(idx >= 0)
          kv.second = tinygltf::Value(idx + textureOffset);
      }
      else
      {
        remapTextureIndicesInValue(kv.second, textureOffset);
      }
    }
  }
  else if(val.IsArray())
  {
    for(auto& el : val.Get<tinygltf::Value::Array>())
      remapTextureIndicesInValue(el, textureOffset);
  }
}

static void remapAndAppendBuffers(tinygltf::Model& base, const tinygltf::Model& imported)
{
  // Collect existing buffer URIs to detect clashes
  std::unordered_set<std::string> existingUris;
  for(const auto& buf : base.buffers)
  {
    if(!buf.uri.empty())
      existingUris.insert(buf.uri);
  }

  for(const auto& buffer : imported.buffers)
  {
    tinygltf::Buffer buf = buffer;

    // If the buffer URI already exists, make it unique by adding a suffix
    if(!buf.uri.empty() && existingUris.count(buf.uri))
    {
      // Make unique: insert "_N" before the extension (e.g. scene.bin → scene_2.bin)
      std::filesystem::path p(buf.uri);
      std::string           stem = p.stem().string();
      std::string           ext  = p.extension().string();
      int                   n    = 2;
      do
      {
        buf.uri = stem + "_" + std::to_string(n++) + ext;
      } while(existingUris.count(buf.uri));
    }
    if(!buf.uri.empty())
      existingUris.insert(buf.uri);
    base.buffers.push_back(std::move(buf));
  }
}

static void remapAndAppendBufferViews(tinygltf::Model& base, const tinygltf::Model& imported, const IndexRemapping& offsets)
{
  for(const auto& bufferView : imported.bufferViews)
  {
    tinygltf::BufferView bv = bufferView;
    if(bv.buffer >= 0)
      bv.buffer += offsets.buffers;
    if(bv.extensions.count("EXT_meshopt_compression"))
    {
      tinygltf::Value& ext = bv.extensions["EXT_meshopt_compression"];
      if(ext.Has("buffer"))
      {
        int bufIdx = ext.Get("buffer").GetNumberAsInt();
        if(bufIdx >= 0)
        {
          auto& obj     = ext.Get<tinygltf::Value::Object>();
          obj["buffer"] = tinygltf::Value(bufIdx + offsets.buffers);
        }
      }
    }
    base.bufferViews.push_back(std::move(bv));
  }
}

static void remapAndAppendAccessors(tinygltf::Model& base, const tinygltf::Model& imported, const IndexRemapping& offsets)
{
  for(const auto& accessor : imported.accessors)
  {
    tinygltf::Accessor acc = accessor;
    if(acc.bufferView >= 0)
      acc.bufferView += offsets.bufferViews;
    if(acc.sparse.isSparse)
    {
      if(acc.sparse.indices.bufferView >= 0)
        acc.sparse.indices.bufferView += offsets.bufferViews;
      if(acc.sparse.values.bufferView >= 0)
        acc.sparse.values.bufferView += offsets.bufferViews;
    }
    base.accessors.push_back(std::move(acc));
  }
}

static void remapAndAppendImages(tinygltf::Model& base, const tinygltf::Model& imported, const IndexRemapping& offsets)
{
  for(const auto& image : imported.images)
  {
    tinygltf::Image img = image;
    // Remap bufferView index for embedded images (GLB files)
    if(img.bufferView >= 0)
      img.bufferView += offsets.bufferViews;
    base.images.push_back(std::move(img));
  }
}

static void remapAndAppendSamplers(tinygltf::Model& base, const tinygltf::Model& imported)
{
  for(const auto& sampler : imported.samplers)
  {
    base.samplers.push_back(sampler);
  }
}

static void remapAndAppendTextures(tinygltf::Model& base, const tinygltf::Model& imported, const IndexRemapping& offsets)
{
  for(const auto& texture : imported.textures)
  {
    tinygltf::Texture tex = texture;
    if(tex.source >= 0)
      tex.source += offsets.images;
    if(tex.sampler >= 0)
      tex.sampler += offsets.samplers;
    tinygltf::utils::offsetTextureExtensionImageSources(tex, offsets.images);
    base.textures.push_back(std::move(tex));
  }
}

static void remapAndAppendMaterials(tinygltf::Model& base, const tinygltf::Model& imported, const IndexRemapping& offsets)
{
  for(const auto& material : imported.materials)
  {
    tinygltf::Material mat = material;
    remapTextureInfo(mat.pbrMetallicRoughness.baseColorTexture, offsets.textures);
    remapTextureInfo(mat.pbrMetallicRoughness.metallicRoughnessTexture, offsets.textures);
    remapNormalTextureInfo(mat.normalTexture, offsets.textures);
    remapOcclusionTextureInfo(mat.occlusionTexture, offsets.textures);
    remapTextureInfo(mat.emissiveTexture, offsets.textures);
    for(auto& ext : mat.extensions)
      remapTextureIndicesInValue(ext.second, offsets.textures);
    base.materials.push_back(std::move(mat));
  }
}

static void remapAndAppendMeshes(tinygltf::Model& base, const tinygltf::Model& imported, const IndexRemapping& offsets)
{
  for(const auto& mesh : imported.meshes)
  {
    tinygltf::Mesh m = mesh;
    for(auto& primitive : m.primitives)
    {
      if(primitive.material >= 0)
        primitive.material += offsets.materials;
      if(primitive.indices >= 0)
        primitive.indices += offsets.accessors;
      for(auto& pair : primitive.attributes)
      {
        if(pair.second >= 0)
          pair.second += offsets.accessors;
      }
      for(auto& pair : primitive.targets)
      {
        for(auto& attr : pair)
        {
          if(attr.second >= 0)
            attr.second += offsets.accessors;
        }
      }
      // KHR_materials_variants: mapping "material" indices are in imported space; remap to combined model.
      if(primitive.extensions.count("KHR_materials_variants"))
      {
        const auto& ext = primitive.extensions["KHR_materials_variants"];
        if(ext.Has("mappings") && ext.Get("mappings").IsArray())
        {
          const auto&            oldMappings = ext.Get("mappings");
          tinygltf::Value::Array newMappings;
          for(size_t i = 0; i < oldMappings.ArrayLen(); ++i)
          {
            const auto&             mapVal = oldMappings.Get(static_cast<int>(i));
            tinygltf::Value::Object mapObj;
            if(mapVal.Has("variants"))
              mapObj["variants"] = mapVal.Get("variants");
            if(mapVal.Has("material"))
            {
              int matIdx = mapVal.Get("material").Get<int>();
              if(matIdx >= 0)
                matIdx += offsets.materials;
              mapObj["material"] = tinygltf::Value(matIdx);
            }
            newMappings.push_back(tinygltf::Value(std::move(mapObj)));
          }
          tinygltf::Value::Object extObj;
          extObj["mappings"]                             = tinygltf::Value(std::move(newMappings));
          primitive.extensions["KHR_materials_variants"] = tinygltf::Value(std::move(extObj));
        }
      }
    }
    base.meshes.push_back(std::move(m));
  }
}

static void remapAndAppendSkins(tinygltf::Model& base, const tinygltf::Model& imported, const IndexRemapping& offsets)
{
  for(const auto& skin : imported.skins)
  {
    tinygltf::Skin s = skin;
    if(s.inverseBindMatrices >= 0)
      s.inverseBindMatrices += offsets.accessors;
    if(s.skeleton >= 0)
      s.skeleton += offsets.nodes;
    for(int& joint : s.joints)
    {
      if(joint >= 0)
        joint += offsets.nodes;
    }
    base.skins.push_back(std::move(s));
  }
}

static void remapAndAppendCameras(tinygltf::Model& base, const tinygltf::Model& imported)
{
  for(const auto& camera : imported.cameras)
  {
    base.cameras.push_back(camera);
  }
}

static void remapAndAppendNodes(tinygltf::Model& base, const tinygltf::Model& imported, const IndexRemapping& offsets)
{
  for(const auto& node : imported.nodes)
  {
    tinygltf::Node n = node;
    for(int& child : n.children)
    {
      if(child >= 0)
        child += offsets.nodes;
    }
    if(n.mesh >= 0)
      n.mesh += offsets.meshes;
    if(n.camera >= 0)
      n.camera += offsets.cameras;
    if(n.skin >= 0)
      n.skin += offsets.skins;
    if(n.extensions.count(KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME))
    {
      tinygltf::Value& ext = n.extensions[KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME];
      if(ext.Has("light"))
      {
        int lightIdx = ext.Get("light").GetNumberAsInt();
        if(lightIdx >= 0)
        {
          ext     = tinygltf::Value(lightIdx + offsets.lights);
          n.light = lightIdx + offsets.lights;  // Scene uses node.light, not only extension
        }
      }
    }
    if(n.extensions.count(EXT_MESH_GPU_INSTANCING_EXTENSION_NAME))
    {
      tinygltf::Value& ext = n.extensions[EXT_MESH_GPU_INSTANCING_EXTENSION_NAME];
      if(ext.Has("attributes") && ext.Get("attributes").IsObject())
      {
        auto& attrs = ext.Get<tinygltf::Value::Object>()["attributes"].Get<tinygltf::Value::Object>();
        for(auto& attr : attrs)
        {
          if(attr.second.IsInt() || attr.second.IsNumber())
          {
            int accIdx = attr.second.GetNumberAsInt();
            if(accIdx >= 0)
              attr.second = tinygltf::Value(accIdx + offsets.accessors);
          }
        }
      }
    }
    base.nodes.push_back(std::move(n));
  }
}

static void remapAndAppendAnimations(tinygltf::Model& base, const tinygltf::Model& imported, const IndexRemapping& offsets)
{
  for(const auto& animation : imported.animations)
  {
    tinygltf::Animation anim = animation;
    for(auto& channel : anim.channels)
    {
      if(channel.target_node >= 0)
        channel.target_node += offsets.nodes;
    }
    for(auto& sampler : anim.samplers)
    {
      if(sampler.input >= 0)
        sampler.input += offsets.accessors;
      if(sampler.output >= 0)
        sampler.output += offsets.accessors;
    }
    base.animations.push_back(std::move(anim));
  }
}

static void remapAndAppendLights(tinygltf::Model& base, const tinygltf::Model& imported)
{
  const char* extName = KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME;
  if(!imported.extensions.count(extName))
    return;
  const tinygltf::Value& importedExt = imported.extensions.at(extName);
  if(!importedExt.Has("lights") || !importedExt.Get("lights").IsArray())
    return;

  tinygltf::Value baseLights;
  if(base.extensions.count(extName) && base.extensions.at(extName).Has("lights")
     && base.extensions.at(extName).Get("lights").IsArray())
  {
    baseLights = base.extensions.at(extName).Get("lights");
  }

  tinygltf::Value::Array arr;
  if(baseLights.IsArray())
  {
    for(size_t i = 0; i < baseLights.ArrayLen(); ++i)
      arr.push_back(baseLights.Get(static_cast<int>(i)));
  }
  const tinygltf::Value& importedLights = importedExt.Get("lights");
  for(size_t i = 0; i < importedLights.ArrayLen(); ++i)
  {
    arr.push_back(importedLights.Get(static_cast<int>(i)));
  }

  tinygltf::Value         newLights(std::move(arr));
  tinygltf::Value::Object extObj;
  extObj["lights"]         = newLights;
  base.extensions[extName] = tinygltf::Value(std::move(extObj));

  // Keep model.lights in sync with the extension (Scene uses model.lights; loader fills it from extension).
  for(const tinygltf::Light& light : imported.lights)
    base.lights.push_back(light);
}

// Append every imported resource array (buffers..lights) into base, remapping indices by
// `offsets`, then union the imported extension-used/required lists into base. Shared by both
// merge() (wrapper-node variant) and mergeIntoNode() (external-asset variant). Does NOT touch
// base.nodes' scene wiring -- the caller decides how the imported roots are attached.
static void appendResourcesAndExtensions(tinygltf::Model& base, const tinygltf::Model& imported, const IndexRemapping& offsets)
{
  remapAndAppendBuffers(base, imported);
  remapAndAppendBufferViews(base, imported, offsets);
  remapAndAppendAccessors(base, imported, offsets);
  remapAndAppendImages(base, imported, offsets);
  remapAndAppendSamplers(base, imported);
  remapAndAppendTextures(base, imported, offsets);
  remapAndAppendMaterials(base, imported, offsets);
  remapAndAppendMeshes(base, imported, offsets);
  remapAndAppendSkins(base, imported, offsets);
  remapAndAppendCameras(base, imported);
  remapAndAppendNodes(base, imported, offsets);
  remapAndAppendAnimations(base, imported, offsets);
  remapAndAppendLights(base, imported);

  // Merge extension lists so decompression and other logic see imported extensions
  for(const std::string& ext : imported.extensionsUsed)
  {
    if(std::find(base.extensionsUsed.begin(), base.extensionsUsed.end(), ext) == base.extensionsUsed.end())
      base.extensionsUsed.push_back(ext);
  }
  for(const std::string& ext : imported.extensionsRequired)
  {
    if(std::find(base.extensionsRequired.begin(), base.extensionsRequired.end(), ext) == base.extensionsRequired.end())
      base.extensionsRequired.push_back(ext);
  }
}

}  // anonymous namespace

namespace nvvkgltf {

int SceneMerger::merge(tinygltf::Model&        baseModel,
                       const tinygltf::Model&  importedModel,
                       const std::string&      importedSceneName,
                       std::optional<uint32_t> maxTextureCount)
{
  if(importedModel.nodes.empty())
    return -1;
  if(importedModel.scenes.empty())
    return -1;

  if(maxTextureCount.has_value())
  {
    const size_t combined = baseModel.textures.size() + importedModel.textures.size();
    if(combined > static_cast<size_t>(*maxTextureCount))
      return -1;
  }

  IndexRemapping offsets = computeOffsets(baseModel);
  appendResourcesAndExtensions(baseModel, importedModel, offsets);

  int                    defaultScene  = importedModel.defaultScene >= 0 ? importedModel.defaultScene : 0;
  const tinygltf::Scene& importedScene = importedModel.scenes[defaultScene];

  tinygltf::Node wrapperNode;
  wrapperNode.name = importedSceneName;
  for(int nodeIdx : importedScene.nodes)
  {
    wrapperNode.children.push_back(offsets.nodes + nodeIdx);
  }

  int wrapperIdx = static_cast<int>(baseModel.nodes.size());
  baseModel.nodes.push_back(std::move(wrapperNode));

  // Add the imported scene to the base model
  if(baseModel.scenes.empty())
    baseModel.scenes.emplace_back();
  int currentScene = baseModel.defaultScene >= 0 ? baseModel.defaultScene : 0;
  baseModel.scenes[currentScene].nodes.push_back(wrapperIdx);

  return wrapperIdx;
}

MergeResult SceneMerger::mergeIntoNode(tinygltf::Model&        baseModel,
                                       const tinygltf::Model&  importedModel,
                                       int                     targetNodeIndex,
                                       std::optional<uint32_t> maxTextureCount)
{
  if(importedModel.nodes.empty() || importedModel.scenes.empty())
    return {};
  if(targetNodeIndex < 0 || targetNodeIndex >= static_cast<int>(baseModel.nodes.size()))
    return {};

  if(maxTextureCount.has_value())
  {
    const size_t combined = baseModel.textures.size() + importedModel.textures.size();
    if(combined > static_cast<size_t>(*maxTextureCount))
      return {};
  }

  IndexRemapping offsets = computeOffsets(baseModel);
  appendResourcesAndExtensions(baseModel, importedModel, offsets);

  MergeResult result;
  result.firstNode = offsets.nodes;
  result.lastNode  = static_cast<int>(baseModel.nodes.size());

  // Attach the imported default-scene roots as children of the existing target node.
  // (glTF 2.1: "append the externalAsset root node(s) to the array of children for the node.")
  // Guard against malformed assets: clamp an out-of-range defaultScene to 0 (scenes is non-empty,
  // checked above), and skip scene-root entries that fall outside the imported node range so we
  // never attach dangling children or push invalid roots.
  int defaultScene = importedModel.defaultScene >= 0 ? importedModel.defaultScene : 0;
  if(defaultScene >= static_cast<int>(importedModel.scenes.size()))
    defaultScene = 0;
  const tinygltf::Scene& importedScene     = importedModel.scenes[defaultScene];
  const int              importedNodeCount = static_cast<int>(importedModel.nodes.size());
  for(int nodeIdx : importedScene.nodes)
  {
    if(nodeIdx < 0 || nodeIdx >= importedNodeCount)
      continue;
    const int rootIdx = offsets.nodes + nodeIdx;
    baseModel.nodes[targetNodeIndex].children.push_back(rootIdx);
    result.roots.push_back(rootIdx);
  }

  return result;
}

MergeResult SceneMerger::instanceSubtree(tinygltf::Model& baseModel, const MergeResult& source, int targetNodeIndex)
{
  if(!source.valid())
    return {};
  if(targetNodeIndex < 0 || targetNodeIndex >= static_cast<int>(baseModel.nodes.size()))
    return {};
  if(source.lastNode > static_cast<int>(baseModel.nodes.size()))
    return {};

  // The source subtree occupies the contiguous block [firstNode, lastNode); every child/joint
  // index inside the block refers to another node in the block. A copy therefore only needs to
  // shift in-block references by a constant delta.
  const int firstNode = source.firstNode;
  const int lastNode  = source.lastNode;
  const int newFirst  = static_cast<int>(baseModel.nodes.size());
  const int delta     = newFirst - firstNode;

  auto inBlock = [&](int idx) { return idx >= firstNode && idx < lastNode; };

  // Duplicate skins referenced by the block so skinned instances animate independently. Meshes,
  // materials and accessors are intentionally shared (=> shared render primitives / BLAS).
  std::unordered_map<int, int> skinMap;
  for(int i = firstNode; i < lastNode; ++i)
  {
    const int s = baseModel.nodes[i].skin;
    if(s >= 0 && s < static_cast<int>(baseModel.skins.size()) && skinMap.find(s) == skinMap.end())
    {
      tinygltf::Skin ns = baseModel.skins[s];
      for(int& j : ns.joints)
      {
        if(inBlock(j))
          j += delta;
      }
      if(inBlock(ns.skeleton))
        ns.skeleton += delta;
      skinMap[s] = static_cast<int>(baseModel.skins.size());
      baseModel.skins.push_back(std::move(ns));
    }
  }

  // Duplicate the nodes (indexed access stays valid across push_back reallocations).
  for(int i = firstNode; i < lastNode; ++i)
  {
    tinygltf::Node n = baseModel.nodes[i];
    for(int& c : n.children)
    {
      if(inBlock(c))
        c += delta;
    }
    if(n.skin >= 0)
    {
      auto it = skinMap.find(n.skin);
      if(it != skinMap.end())
        n.skin = it->second;
    }
    baseModel.nodes.push_back(std::move(n));
  }

  // Duplicate animation channels that target the block, retargeted to the copies (samplers shared).
  for(auto& anim : baseModel.animations)
  {
    const size_t originalChannelCount = anim.channels.size();
    for(size_t ci = 0; ci < originalChannelCount; ++ci)
    {
      if(inBlock(anim.channels[ci].target_node))
      {
        tinygltf::AnimationChannel nc = anim.channels[ci];
        nc.target_node += delta;
        anim.channels.push_back(nc);
      }
    }
  }

  MergeResult result;
  result.firstNode = newFirst;
  result.lastNode  = static_cast<int>(baseModel.nodes.size());
  for(int r : source.roots)
  {
    const int rootIdx = r + delta;
    baseModel.nodes[targetNodeIndex].children.push_back(rootIdx);
    result.roots.push_back(rootIdx);
  }

  return result;
}

}  // namespace nvvkgltf
