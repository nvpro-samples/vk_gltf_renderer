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

//--------------------------------------------------------------------------------------------------
// Scene::compactModel() - Comprehensive scene compaction
//
// This file contains the implementation of Scene::compactModel() which removes all orphaned
// resources from the glTF model after operations like delete, import/merge, and duplicate.
//--------------------------------------------------------------------------------------------------

#include "gltf_scene.hpp"
#include "gltf_compact_model.hpp"
#include "tinygltf_utils.hpp"
#include <nvutils/logger.hpp>
#include <algorithm>
#include <set>
#include <vector>

namespace nvvkgltf {

namespace {

// Helper to collect texture index from any TextureInfo-like structure (TextureInfo, NormalTextureInfo, OcclusionTextureInfo)
template <typename T>
void collectTextureIndex(const T& texInfo, std::set<int>& usedTextures)
{
  if(texInfo.index >= 0)
    usedTextures.insert(texInfo.index);
}

// Collect texture indices from a material's extensions
void collectMaterialExtensionTextures(const tinygltf::Material& material, std::set<int>& usedTextures)
{
  // Helper lambda to safely get texture index from extension object
  auto getTextureIndex = [](const tinygltf::Value& obj, const std::string& key) -> int {
    if(obj.Has(key) && obj.Get(key).IsObject())
    {
      const tinygltf::Value& texInfo = obj.Get(key);
      if(texInfo.Has("index") && texInfo.Get("index").IsInt())
      {
        return texInfo.Get("index").GetNumberAsInt();
      }
    }
    return -1;
  };

  // KHR_materials_anisotropy
  if(material.extensions.count("KHR_materials_anisotropy"))
  {
    const tinygltf::Value& ext = material.extensions.at("KHR_materials_anisotropy");
    int                    idx = getTextureIndex(ext, "anisotropyTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
  }

  // KHR_materials_clearcoat
  if(material.extensions.count("KHR_materials_clearcoat"))
  {
    const tinygltf::Value& ext = material.extensions.at("KHR_materials_clearcoat");
    int                    idx = getTextureIndex(ext, "clearcoatTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
    idx = getTextureIndex(ext, "clearcoatRoughnessTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
    idx = getTextureIndex(ext, "clearcoatNormalTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
  }

  // KHR_materials_emissive_strength (no textures, just a strength value)

  // KHR_materials_iridescence
  if(material.extensions.count("KHR_materials_iridescence"))
  {
    const tinygltf::Value& ext = material.extensions.at("KHR_materials_iridescence");
    int                    idx = getTextureIndex(ext, "iridescenceTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
    idx = getTextureIndex(ext, "iridescenceThicknessTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
  }

  // KHR_materials_sheen
  if(material.extensions.count("KHR_materials_sheen"))
  {
    const tinygltf::Value& ext = material.extensions.at("KHR_materials_sheen");
    int                    idx = getTextureIndex(ext, "sheenColorTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
    idx = getTextureIndex(ext, "sheenRoughnessTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
  }

  // KHR_materials_specular
  if(material.extensions.count("KHR_materials_specular"))
  {
    const tinygltf::Value& ext = material.extensions.at("KHR_materials_specular");
    int                    idx = getTextureIndex(ext, "specularTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
    idx = getTextureIndex(ext, "specularColorTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
  }

  // KHR_materials_transmission
  if(material.extensions.count("KHR_materials_transmission"))
  {
    const tinygltf::Value& ext = material.extensions.at("KHR_materials_transmission");
    int                    idx = getTextureIndex(ext, "transmissionTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
  }

  // KHR_materials_volume
  if(material.extensions.count("KHR_materials_volume"))
  {
    const tinygltf::Value& ext = material.extensions.at("KHR_materials_volume");
    int                    idx = getTextureIndex(ext, "thicknessTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
  }

  // KHR_materials_diffuse_transmission
  if(material.extensions.count("KHR_materials_diffuse_transmission"))
  {
    const tinygltf::Value& ext = material.extensions.at("KHR_materials_diffuse_transmission");
    int                    idx = getTextureIndex(ext, "diffuseTransmissionTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
    idx = getTextureIndex(ext, "diffuseTransmissionColorTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
  }

  // KHR_materials_retroreflection
  if(material.extensions.count(KHR_MATERIALS_RETROREFLECTION_EXTENSION_NAME))
  {
    const tinygltf::Value& ext = material.extensions.at(KHR_MATERIALS_RETROREFLECTION_EXTENSION_NAME);
    int                    idx = getTextureIndex(ext, "retroreflectionTexture");
    if(idx >= 0)
      usedTextures.insert(idx);
  }
}

// Remap texture indices in material extensions
void remapMaterialExtensionTextures(tinygltf::Material& material, const std::vector<int>& textureRemap)
{
  // Helper lambda to remap texture index in extension object
  auto remapTextureIndex = [&](tinygltf::Value& obj, const std::string& key) {
    if(obj.Has(key) && obj.Get(key).IsObject())
    {
      const tinygltf::Value& texInfo = obj.Get(key);
      if(texInfo.Has("index") && texInfo.Get("index").IsInt())
      {
        int oldIdx = texInfo.Get("index").GetNumberAsInt();
        if(oldIdx >= 0 && oldIdx < static_cast<int>(textureRemap.size()) && textureRemap[oldIdx] >= 0)
        {
          obj.Get<tinygltf::Value::Object>()[key].Get<tinygltf::Value::Object>()["index"] = tinygltf::Value(textureRemap[oldIdx]);
        }
      }
    }
  };

  // KHR_materials_anisotropy
  if(material.extensions.count("KHR_materials_anisotropy"))
  {
    tinygltf::Value& ext = material.extensions.at("KHR_materials_anisotropy");
    remapTextureIndex(ext, "anisotropyTexture");
  }

  // KHR_materials_clearcoat
  if(material.extensions.count("KHR_materials_clearcoat"))
  {
    tinygltf::Value& ext = material.extensions.at("KHR_materials_clearcoat");
    remapTextureIndex(ext, "clearcoatTexture");
    remapTextureIndex(ext, "clearcoatRoughnessTexture");
    remapTextureIndex(ext, "clearcoatNormalTexture");
  }

  // KHR_materials_iridescence
  if(material.extensions.count("KHR_materials_iridescence"))
  {
    tinygltf::Value& ext = material.extensions.at("KHR_materials_iridescence");
    remapTextureIndex(ext, "iridescenceTexture");
    remapTextureIndex(ext, "iridescenceThicknessTexture");
  }

  // KHR_materials_sheen
  if(material.extensions.count("KHR_materials_sheen"))
  {
    tinygltf::Value& ext = material.extensions.at("KHR_materials_sheen");
    remapTextureIndex(ext, "sheenColorTexture");
    remapTextureIndex(ext, "sheenRoughnessTexture");
  }

  // KHR_materials_specular
  if(material.extensions.count("KHR_materials_specular"))
  {
    tinygltf::Value& ext = material.extensions.at("KHR_materials_specular");
    remapTextureIndex(ext, "specularTexture");
    remapTextureIndex(ext, "specularColorTexture");
  }

  // KHR_materials_transmission
  if(material.extensions.count("KHR_materials_transmission"))
  {
    tinygltf::Value& ext = material.extensions.at("KHR_materials_transmission");
    remapTextureIndex(ext, "transmissionTexture");
  }

  // KHR_materials_volume
  if(material.extensions.count("KHR_materials_volume"))
  {
    tinygltf::Value& ext = material.extensions.at("KHR_materials_volume");
    remapTextureIndex(ext, "thicknessTexture");
  }

  // KHR_materials_diffuse_transmission
  if(material.extensions.count("KHR_materials_diffuse_transmission"))
  {
    tinygltf::Value& ext = material.extensions.at("KHR_materials_diffuse_transmission");
    remapTextureIndex(ext, "diffuseTransmissionTexture");
    remapTextureIndex(ext, "diffuseTransmissionColorTexture");
  }

  // KHR_materials_retroreflection
  if(material.extensions.count(KHR_MATERIALS_RETROREFLECTION_EXTENSION_NAME))
  {
    tinygltf::Value& ext = material.extensions.at(KHR_MATERIALS_RETROREFLECTION_EXTENSION_NAME);
    remapTextureIndex(ext, "retroreflectionTexture");
  }
}

// Holds the sets of resource indices that are actively referenced by the scene graph.
struct UsedResources
{
  std::set<int> meshes, materials, textures, images, samplers, skins, cameras, lights, animations;
};

// Phase 1: Walk the scene graph top-down and collect every referenced resource index.
// Cascade: scenes → nodes → meshes → materials → textures → images/samplers; plus skins, cameras, lights, animations.
void collectReferencedResources(const tinygltf::Model& model, UsedResources& used)
{
  // 1. Traverse all scenes → nodes to collect mesh/skin/camera/light references
  for(const auto& scene : model.scenes)
  {
    for(int rootNodeIdx : scene.nodes)
    {
      std::function<void(int)> traverseNode = [&](int nodeIdx) {
        if(nodeIdx < 0 || nodeIdx >= static_cast<int>(model.nodes.size()))
          return;

        const tinygltf::Node& node = model.nodes[nodeIdx];

        if(node.mesh >= 0)
          used.meshes.insert(node.mesh);
        if(node.skin >= 0)
          used.skins.insert(node.skin);
        if(node.camera >= 0)
          used.cameras.insert(node.camera);

        if(node.extensions.count("KHR_lights_punctual"))
        {
          const tinygltf::Value& ext = node.extensions.at("KHR_lights_punctual");
          if(ext.Has("light") && ext.Get("light").IsInt())
          {
            int lightIdx = ext.Get("light").GetNumberAsInt();
            if(lightIdx >= 0)
              used.lights.insert(lightIdx);
          }
        }

        for(int childIdx : node.children)
          traverseNode(childIdx);
      };
      traverseNode(rootNodeIdx);
    }
  }

  // 2. Collect material references from used meshes
  for(int meshIdx : used.meshes)
  {
    if(meshIdx >= static_cast<int>(model.meshes.size()))
      continue;

    for(const auto& primitive : model.meshes[meshIdx].primitives)
    {
      if(primitive.material >= 0)
        used.materials.insert(primitive.material);

      if(primitive.extensions.count("KHR_materials_variants"))
      {
        const tinygltf::Value& ext = primitive.extensions.at("KHR_materials_variants");
        if(ext.Has("mappings") && ext.Get("mappings").IsArray())
        {
          const tinygltf::Value& mappings = ext.Get("mappings");
          for(size_t i = 0; i < mappings.ArrayLen(); ++i)
          {
            const tinygltf::Value& mapping = mappings.Get(static_cast<int>(i));
            if(mapping.Has("material") && mapping.Get("material").IsInt())
            {
              int matIdx = mapping.Get("material").GetNumberAsInt();
              if(matIdx >= 0)
                used.materials.insert(matIdx);
            }
          }
        }
      }
    }
  }

  // 3. Collect texture references from used materials
  for(int matIdx : used.materials)
  {
    if(matIdx >= static_cast<int>(model.materials.size()))
      continue;

    const tinygltf::Material& material = model.materials[matIdx];

    if(material.pbrMetallicRoughness.baseColorTexture.index >= 0)
      used.textures.insert(material.pbrMetallicRoughness.baseColorTexture.index);
    if(material.pbrMetallicRoughness.metallicRoughnessTexture.index >= 0)
      used.textures.insert(material.pbrMetallicRoughness.metallicRoughnessTexture.index);

    collectTextureIndex(material.normalTexture, used.textures);
    collectTextureIndex(material.occlusionTexture, used.textures);
    collectTextureIndex(material.emissiveTexture, used.textures);
    collectMaterialExtensionTextures(material, used.textures);
  }

  // 4. Collect image and sampler references from used textures
  for(int texIdx : used.textures)
  {
    if(texIdx >= static_cast<int>(model.textures.size()))
      continue;

    const tinygltf::Texture& texture  = model.textures[texIdx];
    const int                imageIdx = tinygltf::utils::getTextureImageIndex(texture);
    if(imageIdx >= 0)
      used.images.insert(imageIdx);
    if(texture.sampler >= 0)
      used.samplers.insert(texture.sampler);
  }

  // 5. Keep animations that reference valid nodes or materials/lights
  for(size_t animIdx = 0; animIdx < model.animations.size(); ++animIdx)
  {
    bool isUsed = false;
    for(const auto& channel : model.animations[animIdx].channels)
    {
      if(channel.target_node >= 0 && channel.target_node < static_cast<int>(model.nodes.size()))
      {
        isUsed = true;
        break;
      }
      if(channel.target_path == "pointer" && channel.extensions.count("KHR_animation_pointer"))
      {
        const tinygltf::Value& ext = channel.extensions.at("KHR_animation_pointer");
        if(ext.Has("pointer") && ext.Get("pointer").IsString())
        {
          std::string pointer = ext.Get("pointer").Get<std::string>();
          if(pointer.find("/materials/") != std::string::npos
             || pointer.find("/extensions/KHR_lights_punctual/lights/") != std::string::npos)
          {
            isUsed = true;
            break;
          }
        }
      }
    }
    if(isUsed)
      used.animations.insert(static_cast<int>(animIdx));
  }
}

// Build old→new index remapping: entries in usedSet get consecutive new indices; others stay -1.
std::vector<int> buildRemapTable(const std::set<int>& usedSet, size_t originalSize)
{
  std::vector<int> remap(originalSize, -1);
  int              newIdx = 0;
  for(int oldIdx : usedSet)
  {
    // Skip malformed/partially-edited references that fall outside the model array; without this a
    // stray index (e.g. node.mesh past meshes.size()) would write out of bounds.
    if(oldIdx < 0 || oldIdx >= static_cast<int>(originalSize))
      continue;
    remap[oldIdx] = newIdx++;
  }
  return remap;
}

// Copy only the elements whose old indices appear in usedSet, preserving the order of usedSet.
template <typename T>
std::vector<T> extractUsedElements(const std::vector<T>& original, const std::set<int>& usedSet)
{
  std::vector<T> result;
  result.reserve(usedSet.size());
  for(int oldIdx : usedSet)
  {
    // Skip out-of-range indices in lockstep with buildRemapTable so the new-index numbering the two
    // produce stays consistent and no out-of-bounds element is read.
    if(oldIdx < 0 || oldIdx >= static_cast<int>(original.size()))
      continue;
    result.push_back(original[oldIdx]);
  }
  return result;
}

// Phase 4: Rewrite every index reference in nodes, meshes, materials, textures, and lights to use new indices.
void updateAllReferences(tinygltf::Model&                 model,
                         std::vector<tinygltf::Mesh>&     newMeshes,
                         std::vector<tinygltf::Material>& newMaterials,
                         std::vector<tinygltf::Texture>&  newTextures,
                         const std::vector<int>&          meshRemap,
                         const std::vector<int>&          materialRemap,
                         const std::vector<int>&          textureRemap,
                         const std::vector<int>&          imageRemap,
                         const std::vector<int>&          samplerRemap,
                         const std::vector<int>&          skinRemap,
                         const std::vector<int>&          cameraRemap,
                         const std::vector<int>&          lightRemap,
                         const std::set<int>&             usedLights)
{
  auto safeRemap = [](int oldIdx, const std::vector<int>& remap) -> int {
    return (oldIdx >= 0 && oldIdx < static_cast<int>(remap.size())) ? remap[oldIdx] : oldIdx;
  };

  // Update nodes
  for(auto& node : model.nodes)
  {
    node.mesh   = safeRemap(node.mesh, meshRemap);
    node.skin   = safeRemap(node.skin, skinRemap);
    node.camera = safeRemap(node.camera, cameraRemap);

    if(node.extensions.count("KHR_lights_punctual"))
    {
      tinygltf::Value& ext = node.extensions.at("KHR_lights_punctual");
      if(ext.Has("light") && ext.Get("light").IsInt())
      {
        int oldIdx = ext.Get("light").GetNumberAsInt();
        if(oldIdx >= 0 && oldIdx < static_cast<int>(lightRemap.size()) && lightRemap[oldIdx] >= 0)
          ext.Get<tinygltf::Value::Object>()["light"] = tinygltf::Value(lightRemap[oldIdx]);
      }
    }
  }

  // Update mesh primitives
  for(auto& mesh : newMeshes)
  {
    for(auto& primitive : mesh.primitives)
    {
      primitive.material = safeRemap(primitive.material, materialRemap);

      if(primitive.extensions.count("KHR_materials_variants"))
      {
        tinygltf::Value& ext = primitive.extensions.at("KHR_materials_variants");
        if(ext.Has("mappings") && ext.Get("mappings").IsArray())
        {
          const tinygltf::Value& mappings = ext.Get("mappings");
          for(size_t i = 0; i < mappings.ArrayLen(); ++i)
          {
            const tinygltf::Value& mapping = mappings.Get(static_cast<int>(i));
            if(mapping.Has("material") && mapping.Get("material").IsInt())
            {
              int oldIdx = mapping.Get("material").GetNumberAsInt();
              if(oldIdx >= 0 && oldIdx < static_cast<int>(materialRemap.size()) && materialRemap[oldIdx] >= 0)
              {
                ext.Get<tinygltf::Value::Object>()["mappings"].Get<tinygltf::Value::Array>()[i].Get<tinygltf::Value::Object>()["material"] =
                    tinygltf::Value(materialRemap[oldIdx]);
              }
            }
          }
        }
      }
    }
  }

  // Update materials (texture references in base + extensions)
  for(auto& material : newMaterials)
  {
    material.pbrMetallicRoughness.baseColorTexture.index =
        safeRemap(material.pbrMetallicRoughness.baseColorTexture.index, textureRemap);
    material.pbrMetallicRoughness.metallicRoughnessTexture.index =
        safeRemap(material.pbrMetallicRoughness.metallicRoughnessTexture.index, textureRemap);
    material.normalTexture.index    = safeRemap(material.normalTexture.index, textureRemap);
    material.occlusionTexture.index = safeRemap(material.occlusionTexture.index, textureRemap);
    material.emissiveTexture.index  = safeRemap(material.emissiveTexture.index, textureRemap);
    remapMaterialExtensionTextures(material, textureRemap);
  }

  // Update textures
  for(auto& texture : newTextures)
  {
    texture.source  = safeRemap(texture.source, imageRemap);
    texture.sampler = safeRemap(texture.sampler, samplerRemap);
    tinygltf::utils::remapTextureExtensionImageSources(texture, imageRemap);
  }

  // Update KHR_lights_punctual extension
  if(model.extensions.count("KHR_lights_punctual") && !usedLights.empty())
  {
    tinygltf::Value& ext = model.extensions.at("KHR_lights_punctual");
    if(ext.Has("lights") && ext.Get("lights").IsArray())
    {
      const tinygltf::Value& oldLights = ext.Get("lights");
      tinygltf::Value        newLights = tinygltf::Value(tinygltf::Value::Array());
      for(int oldIdx : usedLights)
      {
        if(oldIdx < static_cast<int>(oldLights.ArrayLen()))
          newLights.Get<tinygltf::Value::Array>().push_back(oldLights.Get(oldIdx));
      }
      ext.Get<tinygltf::Value::Object>()["lights"] = std::move(newLights);
    }
  }
  else if(model.extensions.count("KHR_lights_punctual") && usedLights.empty())
  {
    model.extensions.erase("KHR_lights_punctual");
  }
}

void logCompactionResults(size_t                 origMesh,
                          size_t                 origMat,
                          size_t                 origTex,
                          size_t                 origImg,
                          size_t                 origSamp,
                          size_t                 origSkin,
                          size_t                 origCam,
                          size_t                 origAnim,
                          size_t                 origLight,
                          const tinygltf::Model& model,
                          size_t                 finalLightCount)
{
  LOGI("=== Scene Compaction Results ===\n");
  auto logChange = [](const char* name, size_t original, size_t final_) {
    if(original > final_)
    {
      size_t removed = original - final_;
      double pct     = original > 0 ? (double(removed) / double(original)) * 100.0 : 0.0;
      LOGI("  %-12s %3zu -> %3zu  (-%zu, %.1f%% reduction)\n", name, original, final_, removed, pct);
    }
    else if(original == final_)
    {
      LOGI("  %-12s %3zu (no change)\n", name, original);
    }
  };
  logChange("Meshes:", origMesh, model.meshes.size());
  logChange("Materials:", origMat, model.materials.size());
  logChange("Textures:", origTex, model.textures.size());
  logChange("Images:", origImg, model.images.size());
  logChange("Samplers:", origSamp, model.samplers.size());
  logChange("Skins:", origSkin, model.skins.size());
  logChange("Cameras:", origCam, model.cameras.size());
  logChange("Animations:", origAnim, model.animations.size());
  logChange("Lights:", origLight, finalLightCount);
}

}  // anonymous namespace

//--------------------------------------------------------------------------------------------------
// removeExternalAssetContent() - strip merged external-asset subtrees for re-externalization on save
//--------------------------------------------------------------------------------------------------
void removeExternalAssetContent(tinygltf::Model& model)
{
  const int nodeCount = static_cast<int>(model.nodes.size());

  // Flag the merged (marked) nodes for removal.
  std::vector<char> keep(nodeCount, 1);
  bool              anyRemoved = false;
  for(int i = 0; i < nodeCount; ++i)
  {
    if(nvvkgltf::Scene::hasExternalAssetMarker(model.nodes[i]))
    {
      keep[i]    = 0;
      anyRemoved = true;
    }
  }
  if(!anyRemoved)
    return;  // no external-asset content merged in; nothing to strip

  // old -> new node index for the survivors.
  std::vector<int> newIdx(nodeCount, -1);
  int              next = 0;
  for(int i = 0; i < nodeCount; ++i)
    if(keep[i])
      newIdx[i] = next++;
  auto remap = [&](int old) -> int { return (old >= 0 && old < nodeCount && keep[old]) ? newIdx[old] : -1; };

  // Rebuild the node array (survivors only), remapping children.
  std::vector<tinygltf::Node> newNodes;
  newNodes.reserve(static_cast<size_t>(next));
  for(int i = 0; i < nodeCount; ++i)
  {
    if(!keep[i])
      continue;
    tinygltf::Node   node = std::move(model.nodes[i]);
    std::vector<int> children;
    children.reserve(node.children.size());
    for(int c : node.children)
    {
      const int m = remap(c);
      if(m >= 0)
        children.push_back(m);
    }
    node.children = std::move(children);
    newNodes.push_back(std::move(node));
  }
  model.nodes = std::move(newNodes);

  // Scene roots.
  for(auto& scene : model.scenes)
  {
    std::vector<int> roots;
    roots.reserve(scene.nodes.size());
    for(int c : scene.nodes)
    {
      const int m = remap(c);
      if(m >= 0)
        roots.push_back(m);
    }
    scene.nodes = std::move(roots);
  }

  // Skin joints / skeleton (skins that become orphaned are dropped by the compaction pass below).
  for(auto& skin : model.skins)
  {
    std::vector<int> joints;
    joints.reserve(skin.joints.size());
    for(int j : skin.joints)
    {
      const int m = remap(j);
      if(m >= 0)
        joints.push_back(m);
    }
    skin.joints   = std::move(joints);
    skin.skeleton = remap(skin.skeleton);
  }

  // Animation channels: drop those targeting removed nodes; keep only animations that still have
  // channels by rebuilding the list.
  std::vector<tinygltf::Animation> keptAnimations;
  keptAnimations.reserve(model.animations.size());
  for(auto& anim : model.animations)
  {
    std::vector<tinygltf::AnimationChannel> channels;
    channels.reserve(anim.channels.size());
    for(auto& ch : anim.channels)
    {
      const int m = remap(ch.target_node);
      if(m >= 0)
      {
        ch.target_node = m;
        channels.push_back(ch);
      }
    }
    anim.channels = std::move(channels);
    if(!anim.channels.empty())
      keptAnimations.push_back(std::move(anim));
  }
  model.animations = std::move(keptAnimations);

  // Cascade-remove the now-orphaned meshes/materials/textures/images/samplers/skins/cameras/
  // animations/lights (reachability from the surviving scene graph), then compact geometry buffers.
  UsedResources used;
  collectReferencedResources(model, used);

  const size_t origMesh  = model.meshes.size();
  const size_t origMat   = model.materials.size();
  const size_t origTex   = model.textures.size();
  const size_t origImg   = model.images.size();
  const size_t origSamp  = model.samplers.size();
  const size_t origSkin  = model.skins.size();
  const size_t origCam   = model.cameras.size();
  const size_t origAnim  = model.animations.size();
  size_t       origLight = 0;
  if(model.extensions.count("KHR_lights_punctual"))
  {
    const tinygltf::Value& ext = model.extensions.at("KHR_lights_punctual");
    if(ext.Has("lights") && ext.Get("lights").IsArray())
      origLight = ext.Get("lights").ArrayLen();
  }

  const std::vector<int> meshRemap     = buildRemapTable(used.meshes, origMesh);
  const std::vector<int> materialRemap = buildRemapTable(used.materials, origMat);
  const std::vector<int> textureRemap  = buildRemapTable(used.textures, origTex);
  const std::vector<int> imageRemap    = buildRemapTable(used.images, origImg);
  const std::vector<int> samplerRemap  = buildRemapTable(used.samplers, origSamp);
  const std::vector<int> skinRemap     = buildRemapTable(used.skins, origSkin);
  const std::vector<int> cameraRemap   = buildRemapTable(used.cameras, origCam);
  const std::vector<int> lightRemap    = buildRemapTable(used.lights, origLight);

  std::vector<tinygltf::Mesh>      newMeshes     = extractUsedElements(model.meshes, used.meshes);
  std::vector<tinygltf::Material>  newMaterials  = extractUsedElements(model.materials, used.materials);
  std::vector<tinygltf::Texture>   newTextures   = extractUsedElements(model.textures, used.textures);
  std::vector<tinygltf::Image>     newImages     = extractUsedElements(model.images, used.images);
  std::vector<tinygltf::Sampler>   newSamplers   = extractUsedElements(model.samplers, used.samplers);
  std::vector<tinygltf::Skin>      newSkins      = extractUsedElements(model.skins, used.skins);
  std::vector<tinygltf::Camera>    newCameras    = extractUsedElements(model.cameras, used.cameras);
  std::vector<tinygltf::Animation> newAnimations = extractUsedElements(model.animations, used.animations);

  updateAllReferences(model, newMeshes, newMaterials, newTextures, meshRemap, materialRemap, textureRemap, imageRemap,
                      samplerRemap, skinRemap, cameraRemap, lightRemap, used.lights);

  model.meshes     = std::move(newMeshes);
  model.materials  = std::move(newMaterials);
  model.textures   = std::move(newTextures);
  model.images     = std::move(newImages);
  model.samplers   = std::move(newSamplers);
  model.skins      = std::move(newSkins);
  model.cameras    = std::move(newCameras);
  model.animations = std::move(newAnimations);

  ::compactModel(model);

  // ::compactModel always leaves a single merged buffer, even when no geometry remains. A complex
  // scene that references only external assets has no buffer data, and an empty buffer (byteLength
  // 0) is invalid glTF, so drop any buffer that no buffer view references and remap the rest.
  {
    std::vector<char> bufferUsed(model.buffers.size(), 0);
    for(const auto& bv : model.bufferViews)
    {
      if(bv.buffer >= 0 && bv.buffer < static_cast<int>(model.buffers.size()))
        bufferUsed[bv.buffer] = 1;
    }
    size_t usedCount = 0;
    for(char u : bufferUsed)
      usedCount += (u != 0) ? 1 : 0;

    // Only rebuild when at least one buffer is unreferenced. The rebuild moves buffers OUT of
    // model.buffers; doing it when nothing is dropped (survivors == total) would leave every survivor
    // in a moved-from state (empty data), which wiped the geometry .bin on save.
    if(usedCount != model.buffers.size())
    {
      std::vector<int>              bufferRemap(model.buffers.size(), -1);
      std::vector<tinygltf::Buffer> newBuffers;
      for(size_t i = 0; i < model.buffers.size(); ++i)
      {
        if(bufferUsed[i])
        {
          bufferRemap[i] = static_cast<int>(newBuffers.size());
          newBuffers.push_back(std::move(model.buffers[i]));
        }
      }
      for(auto& bv : model.bufferViews)
        if(bv.buffer >= 0 && bv.buffer < static_cast<int>(bufferRemap.size()))
          bv.buffer = bufferRemap[bv.buffer];
      model.buffers = std::move(newBuffers);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// flattenExternalAssets() - keep merged content inline, strip every external-asset reference
//--------------------------------------------------------------------------------------------------
void flattenExternalAssets(tinygltf::Model& model)
{
  for(tinygltf::Node& node : model.nodes)
  {
    // Drop the external-asset link so the (former) instance node becomes an ordinary transform node,
    // and remove the read-only provenance marker so the merged content is fully owned/editable.
    node.externalAsset = -1;
    nvvkgltf::Scene::clearExternalAssetMarker(node);
  }

  // The merged geometry now stands on its own; the external-asset tables are no longer referenced.
  model.externalAssets.clear();
  model.files.clear();
}

//--------------------------------------------------------------------------------------------------
// compactExternalAssetReferences() - drop orphaned externalAssets/files entries and remap survivors
//--------------------------------------------------------------------------------------------------
void compactExternalAssetReferences(tinygltf::Model& model)
{
  const size_t origExternalAssets = model.externalAssets.size();
  const size_t origFiles          = model.files.size();
  if(origExternalAssets == 0 && origFiles == 0)
    return;

  // Phase 1: collect used externalAssets (still referenced by a node) and the files reachable from
  // them -- directly via ExternalAsset::file, then transitively through each surviving file's
  // aliases. std::set keeps indices sorted so the remap/extract below match the rest of this file.
  std::set<int> usedExternalAssets;
  for(const tinygltf::Node& n : model.nodes)
    if(n.externalAsset >= 0 && n.externalAsset < static_cast<int>(origExternalAssets))
      usedExternalAssets.insert(n.externalAsset);

  std::set<int>    usedFiles;
  std::vector<int> worklist;
  auto             markFile = [&](int f) {
    if(f >= 0 && f < static_cast<int>(origFiles) && usedFiles.insert(f).second)
      worklist.push_back(f);
  };
  for(int e : usedExternalAssets)
    markFile(model.externalAssets[e].file);
  while(!worklist.empty())
  {
    const tinygltf::File& f = model.files[worklist.back()];
    worklist.pop_back();
    for(const tinygltf::FileAlias& a : f.aliases)
      markFile(a.file);
  }

  if(usedExternalAssets.size() == origExternalAssets && usedFiles.size() == origFiles)
    return;  // everything still referenced; nothing to compact

  // Phase 2: build old->new remap tables and extract the survivors (same helpers as compactModel()).
  const std::vector<int> externalAssetRemap = buildRemapTable(usedExternalAssets, origExternalAssets);
  const std::vector<int> fileRemap          = buildRemapTable(usedFiles, origFiles);

  std::vector<tinygltf::ExternalAsset> newExternalAssets = extractUsedElements(model.externalAssets, usedExternalAssets);
  std::vector<tinygltf::File> newFiles = extractUsedElements(model.files, usedFiles);

  // Phase 3: rewrite every index reference to the new indices.
  for(tinygltf::Node& n : model.nodes)
    if(n.externalAsset >= 0 && n.externalAsset < static_cast<int>(origExternalAssets))
      n.externalAsset = externalAssetRemap[n.externalAsset];

  for(tinygltf::ExternalAsset& a : newExternalAssets)
    if(a.file >= 0 && a.file < static_cast<int>(origFiles))
      a.file = fileRemap[a.file];

  for(tinygltf::File& f : newFiles)
    for(tinygltf::FileAlias& a : f.aliases)
      if(a.file >= 0 && a.file < static_cast<int>(origFiles))
        a.file = fileRemap[a.file];

  model.externalAssets = std::move(newExternalAssets);
  model.files          = std::move(newFiles);
}

//--------------------------------------------------------------------------------------------------
// Scene::compactModel() - Remove all orphaned resources
//--------------------------------------------------------------------------------------------------
bool Scene::compactModel()
{
  if(!m_validSceneParsed)
  {
    LOGW("Cannot compact model: no valid scene loaded\n");
    return false;
  }

  const size_t origMesh  = m_model.meshes.size();
  const size_t origMat   = m_model.materials.size();
  const size_t origTex   = m_model.textures.size();
  const size_t origImg   = m_model.images.size();
  const size_t origSamp  = m_model.samplers.size();
  const size_t origSkin  = m_model.skins.size();
  const size_t origCam   = m_model.cameras.size();
  const size_t origAnim  = m_model.animations.size();
  size_t       origLight = 0;
  if(m_model.extensions.count("KHR_lights_punctual"))
  {
    const tinygltf::Value& ext = m_model.extensions.at("KHR_lights_punctual");
    if(ext.Has("lights") && ext.Get("lights").IsArray())
      origLight = ext.Get("lights").ArrayLen();
  }

  // Phase 1: Collect all referenced resources via top-down scene graph traversal
  UsedResources used;
  collectReferencedResources(m_model, used);

  bool needsCompaction =
      (used.meshes.size() < origMesh) || (used.materials.size() < origMat) || (used.textures.size() < origTex)
      || (used.images.size() < origImg) || (used.samplers.size() < origSamp) || (used.skins.size() < origSkin)
      || (used.cameras.size() < origCam) || (used.animations.size() < origAnim) || (used.lights.size() < origLight);
  if(!needsCompaction)
  {
    LOGI("Scene compaction: No orphaned resources found\n");
    return false;
  }

  // Phase 2: Build old→new index remapping tables
  const std::vector<int> meshRemap      = buildRemapTable(used.meshes, origMesh);
  const std::vector<int> materialRemap  = buildRemapTable(used.materials, origMat);
  const std::vector<int> textureRemap   = buildRemapTable(used.textures, origTex);
  const std::vector<int> imageRemap     = buildRemapTable(used.images, origImg);
  const std::vector<int> samplerRemap   = buildRemapTable(used.samplers, origSamp);
  const std::vector<int> skinRemap      = buildRemapTable(used.skins, origSkin);
  const std::vector<int> cameraRemap    = buildRemapTable(used.cameras, origCam);
  const std::vector<int> animationRemap = buildRemapTable(used.animations, origAnim);
  const std::vector<int> lightRemap     = buildRemapTable(used.lights, origLight);

  // Phase 3: Extract only the used elements into compacted arrays
  std::vector<tinygltf::Mesh>      newMeshes     = extractUsedElements(m_model.meshes, used.meshes);
  std::vector<tinygltf::Material>  newMaterials  = extractUsedElements(m_model.materials, used.materials);
  std::vector<tinygltf::Texture>   newTextures   = extractUsedElements(m_model.textures, used.textures);
  std::vector<tinygltf::Image>     newImages     = extractUsedElements(m_model.images, used.images);
  std::vector<tinygltf::Sampler>   newSamplers   = extractUsedElements(m_model.samplers, used.samplers);
  std::vector<tinygltf::Skin>      newSkins      = extractUsedElements(m_model.skins, used.skins);
  std::vector<tinygltf::Camera>    newCameras    = extractUsedElements(m_model.cameras, used.cameras);
  std::vector<tinygltf::Animation> newAnimations = extractUsedElements(m_model.animations, used.animations);

  // Phase 4: Rewrite all index references to use new indices
  updateAllReferences(m_model, newMeshes, newMaterials, newTextures, meshRemap, materialRemap, textureRemap, imageRemap,
                      samplerRemap, skinRemap, cameraRemap, lightRemap, used.lights);

  // Phase 5: Replace model arrays and compact underlying geometry buffers
  m_model.meshes     = std::move(newMeshes);
  m_model.materials  = std::move(newMaterials);
  m_model.textures   = std::move(newTextures);
  m_model.images     = std::move(newImages);
  m_model.samplers   = std::move(newSamplers);
  m_model.skins      = std::move(newSkins);
  m_model.cameras    = std::move(newCameras);
  m_model.animations = std::move(newAnimations);

  // Compact geometry (accessors → bufferViews → buffers) now that orphaned meshes are gone.
  // Merges all surviving data into a single buffer[0].
  ::compactModel(m_model);

  logCompactionResults(origMesh, origMat, origTex, origImg, origSamp, origSkin, origCam, origAnim, origLight, m_model,
                       used.lights.size());

  LOGI("Scene compaction complete - caller should rebuild GPU resources\n");
  return true;
}

}  // namespace nvvkgltf
