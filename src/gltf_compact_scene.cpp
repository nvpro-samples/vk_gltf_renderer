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

    const tinygltf::Texture& texture = model.textures[texIdx];
    if(texture.source >= 0)
      used.images.insert(texture.source);
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
    remap[oldIdx] = newIdx++;
  return remap;
}

// Copy only the elements whose old indices appear in usedSet, preserving the order of usedSet.
template <typename T>
std::vector<T> extractUsedElements(const std::vector<T>& original, const std::set<int>& usedSet)
{
  std::vector<T> result;
  result.reserve(usedSet.size());
  for(int oldIdx : usedSet)
    result.push_back(original[oldIdx]);
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
  auto meshRemap      = buildRemapTable(used.meshes, origMesh);
  auto materialRemap  = buildRemapTable(used.materials, origMat);
  auto textureRemap   = buildRemapTable(used.textures, origTex);
  auto imageRemap     = buildRemapTable(used.images, origImg);
  auto samplerRemap   = buildRemapTable(used.samplers, origSamp);
  auto skinRemap      = buildRemapTable(used.skins, origSkin);
  auto cameraRemap    = buildRemapTable(used.cameras, origCam);
  auto animationRemap = buildRemapTable(used.animations, origAnim);
  auto lightRemap     = buildRemapTable(used.lights, origLight);

  // Phase 3: Extract only the used elements into compacted arrays
  auto newMeshes     = extractUsedElements(m_model.meshes, used.meshes);
  auto newMaterials  = extractUsedElements(m_model.materials, used.materials);
  auto newTextures   = extractUsedElements(m_model.textures, used.textures);
  auto newImages     = extractUsedElements(m_model.images, used.images);
  auto newSamplers   = extractUsedElements(m_model.samplers, used.samplers);
  auto newSkins      = extractUsedElements(m_model.skins, used.skins);
  auto newCameras    = extractUsedElements(m_model.cameras, used.cameras);
  auto newAnimations = extractUsedElements(m_model.animations, used.animations);

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
