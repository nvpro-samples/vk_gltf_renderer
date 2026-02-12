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

#include <charconv>
#include <string_view>

#include <nvutils/logger.hpp>

#include "gltf_animation_pointer.hpp"
#include "tinygltf_utils.hpp"

namespace nvvkgltf {

namespace {
// Helper: Recursively merge tinygltf::Value objects (updates target with source properties)
void mergeValue(tinygltf::Value& target, const tinygltf::Value& source)
{
  // If source is not an object, replace target entirely
  if(!source.IsObject() || !target.IsObject())
  {
    target = source;
    return;
  }

  // Both are objects - merge recursively
  // Safe: we iterate over source (const), modify target directly
  auto&       targetObj = target.Get<tinygltf::Value::Object>();
  const auto& sourceObj = source.Get<tinygltf::Value::Object>();

  for(const auto& [key, value] : sourceObj)
  {
    auto it = targetObj.find(key);
    if(it != targetObj.end() && value.IsObject() && it->second.IsObject())
    {
      // Both have this key and both are objects - recurse
      mergeValue(it->second, value);
    }
    else
    {
      // Key doesn't exist or is not an object - replace/add
      targetObj[key] = value;
    }
  }
}
}  // namespace

AnimationPointerSystem::AnimationPointerSystem(tinygltf::Model& model)
    : m_model(model)
{
  // Initialize with empty JSON object - we'll populate lazily as animations target properties
  m_jsonModel = nlohmann::json::object();
}

//--------------------------------------------------------------------------------------------------
// Core optimization: Get or create cached path info
// This is called once per unique path; subsequent calls return the cached entry.
// Avoids: re-parsing json_pointer, re-computing resource type/index
//--------------------------------------------------------------------------------------------------
AnimationPointerSystem::CachedPathInfo& AnimationPointerSystem::getOrCreateCachedPath(const std::string& jsonPointerPath)
{
  auto [it, inserted] = m_pathCache.try_emplace(jsonPointerPath);
  if(inserted)
  {
    // First time seeing this path - initialize cache entry
    CachedPathInfo& cached = it->second;

    // Parse the json_pointer once (this allocates reference_tokens internally)
    cached.ptr = nlohmann::json::json_pointer(jsonPointerPath);

    // Parse resource type and index for dirty tracking
    parseResourceInfo(jsonPointerPath, cached);
  }
  return it->second;
}

//--------------------------------------------------------------------------------------------------
// Parse resource type and index from path (done once per unique path)
// Uses std::from_chars for zero-allocation integer parsing
//--------------------------------------------------------------------------------------------------
void AnimationPointerSystem::parseResourceInfo(const std::string& path, CachedPathInfo& cached)
{
  cached.resourceType  = ResourceType::eNone;
  cached.resourceIndex = -1;

  // Path prefixes and their corresponding resource types
  struct PrefixMapping
  {
    std::string_view prefix;
    ResourceType     type;
  };
  constexpr PrefixMapping mappings[] = {{"/materials/", ResourceType::eMaterial},
                                        {"/extensions/KHR_lights_punctual/lights/", ResourceType::eLight},
                                        {"/cameras/", ResourceType::eCamera},
                                        {"/nodes/", ResourceType::eNode}};

  for(const auto& [prefix, type] : mappings)
  {
    if(path.rfind(prefix, 0) == 0)
    {
      // Parse index after prefix using std::from_chars (zero allocations)
      size_t startIdx = prefix.size();
      size_t endIdx   = path.find('/', startIdx);
      if(endIdx == std::string::npos)
        endIdx = path.size();

      int                    index  = 0;
      const char*            first  = path.data() + startIdx;
      const char*            last   = path.data() + endIdx;
      std::from_chars_result result = std::from_chars(first, last, index);

      cached.resourceType  = type;
      cached.resourceIndex = (result.ec == std::errc{} && result.ptr == last) ? index : -1;
      return;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Apply animated values - all overloads use the cached path info
// Note: nlohmann::json automatically creates the path structure when assigning via json_pointer.
// The structure won't be "proper" glTF (objects with numeric keys instead of arrays), but
// that's fine - we just need consistent store/retrieve, and json_pointer handles both identically.
//--------------------------------------------------------------------------------------------------
bool AnimationPointerSystem::applyValue(const std::string& jsonPointerPath, float value)
{
  try
  {
    CachedPathInfo& cached = getOrCreateCachedPath(jsonPointerPath);

    // Special handling: Some properties are boolean in glTF spec but animated as floats
    // Convert float to boolean for known boolean properties (e.g., KHR_node_visibility/visible)
    if(jsonPointerPath.ends_with("/visible"))
    {
      m_jsonModel[cached.ptr] = (value != 0.0f);
    }
    else
    {
      // Apply value using cached pointer (nlohmann creates path automatically)
      m_jsonModel[cached.ptr] = value;
    }

    // Mark dirty using cached resource info (no parsing needed)
    markDirtyFromCache(cached);

    return true;
  }
  catch(const nlohmann::json::exception& e)
  {
    LOGW("Failed to apply float to path '%s': %s", jsonPointerPath.c_str(), e.what());
    return false;
  }
}

bool AnimationPointerSystem::applyValue(const std::string& jsonPointerPath, const glm::vec2& value)
{
  try
  {
    CachedPathInfo& cached  = getOrCreateCachedPath(jsonPointerPath);
    m_jsonModel[cached.ptr] = nlohmann::json::array({value.x, value.y});
    markDirtyFromCache(cached);
    return true;
  }
  catch(const nlohmann::json::exception& e)
  {
    LOGW("Failed to apply vec2 to path '%s': %s", jsonPointerPath.c_str(), e.what());
    return false;
  }
}

bool AnimationPointerSystem::applyValue(const std::string& jsonPointerPath, const glm::vec3& value)
{
  try
  {
    CachedPathInfo& cached  = getOrCreateCachedPath(jsonPointerPath);
    m_jsonModel[cached.ptr] = nlohmann::json::array({value.x, value.y, value.z});
    markDirtyFromCache(cached);
    return true;
  }
  catch(const nlohmann::json::exception& e)
  {
    LOGW("Failed to apply vec3 to path '%s': %s", jsonPointerPath.c_str(), e.what());
    return false;
  }
}

bool AnimationPointerSystem::applyValue(const std::string& jsonPointerPath, const glm::vec4& value)
{
  try
  {
    CachedPathInfo& cached  = getOrCreateCachedPath(jsonPointerPath);
    m_jsonModel[cached.ptr] = nlohmann::json::array({value.x, value.y, value.z, value.w});
    markDirtyFromCache(cached);
    return true;
  }
  catch(const nlohmann::json::exception& e)
  {
    LOGW("Failed to apply vec4 to path '%s': %s", jsonPointerPath.c_str(), e.what());
    return false;
  }
}

//--------------------------------------------------------------------------------------------------
// Mark resources dirty using cached info - O(1), no string parsing
//--------------------------------------------------------------------------------------------------
void AnimationPointerSystem::markDirtyFromCache(const CachedPathInfo& cached)
{
  if(cached.resourceIndex < 0)
    return;

  switch(cached.resourceType)
  {
    case ResourceType::eMaterial:
      m_dirtyMaterials.insert(cached.resourceIndex);
      break;
    case ResourceType::eLight:
      m_dirtyLights.insert(cached.resourceIndex);
      break;
    case ResourceType::eCamera:
      m_dirtyCameras.insert(cached.resourceIndex);
      break;
    case ResourceType::eNode:
      m_dirtyNodes.insert(cached.resourceIndex);
      break;
    case ResourceType::eNone:
      break;
  }
}

void AnimationPointerSystem::syncToModel()
{
  // Sync dirty materials back to tinygltf::Model
  for(int materialIndex : m_dirtyMaterials)
  {
    syncMaterial(materialIndex);
  }

  for(int lightIndex : m_dirtyLights)
  {
    syncLight(lightIndex);
  }

  for(int cameraIndex : m_dirtyCameras)
  {
    syncCamera(cameraIndex);
  }

  for(int nodeIndex : m_dirtyNodes)
  {
    syncNode(nodeIndex);
  }
}

void AnimationPointerSystem::syncMaterial(int materialIndex)
{
  if(materialIndex < 0 || materialIndex >= static_cast<int>(m_model.materials.size()))
    return;

  try
  {
    nlohmann::json::json_pointer ptr("/materials/" + std::to_string(materialIndex));
    mergeJsonIntoMaterial(m_jsonModel.at(ptr), m_model.materials[materialIndex]);
  }
  catch(const nlohmann::json::exception& e)
  {
    LOGW("Failed to sync material %d: %s", materialIndex, e.what());
  }
}

void AnimationPointerSystem::syncLight(int lightIndex)
{
  if(lightIndex < 0 || lightIndex >= static_cast<int>(m_model.lights.size()))
    return;

  try
  {
    nlohmann::json::json_pointer ptr("/extensions/KHR_lights_punctual/lights/" + std::to_string(lightIndex));
    const nlohmann::json&        lightJson = m_jsonModel.at(ptr);
    tinygltf::Light&             light     = m_model.lights[lightIndex];

    if(lightJson.contains("color") && lightJson["color"].is_array())
    {
      const auto& arr = lightJson["color"];
      light.color.resize(3);
      for(size_t i = 0; i < 3 && i < arr.size(); ++i)
        light.color[i] = arr[i].get<double>();
    }
    if(lightJson.contains("intensity"))
      light.intensity = lightJson["intensity"].get<double>();
    if(lightJson.contains("range"))
      light.range = lightJson["range"].get<double>();
    if(lightJson.contains("spot") && lightJson["spot"].is_object())
    {
      const auto& spot = lightJson["spot"];
      if(spot.contains("innerConeAngle"))
        light.spot.innerConeAngle = spot["innerConeAngle"].get<double>();
      if(spot.contains("outerConeAngle"))
        light.spot.outerConeAngle = spot["outerConeAngle"].get<double>();
    }
  }
  catch(const nlohmann::json::exception& e)
  {
    LOGW("Failed to sync light %d: %s", lightIndex, e.what());
  }
}

void AnimationPointerSystem::syncCamera(int cameraIndex)
{
  if(cameraIndex < 0 || cameraIndex >= static_cast<int>(m_model.cameras.size()))
    return;

  try
  {
    nlohmann::json::json_pointer ptr("/cameras/" + std::to_string(cameraIndex));
    const nlohmann::json&        camJson = m_jsonModel.at(ptr);
    tinygltf::Camera&            cam     = m_model.cameras[cameraIndex];

    if(cam.type == "perspective" && camJson.contains("perspective"))
    {
      const auto& persp = camJson["perspective"];
      if(persp.contains("yfov"))
        cam.perspective.yfov = persp["yfov"].get<double>();
      if(persp.contains("aspectRatio"))
        cam.perspective.aspectRatio = persp["aspectRatio"].get<double>();
      if(persp.contains("znear"))
        cam.perspective.znear = persp["znear"].get<double>();
      if(persp.contains("zfar"))
        cam.perspective.zfar = persp["zfar"].get<double>();
    }
    else if(cam.type == "orthographic" && camJson.contains("orthographic"))
    {
      const auto& ortho = camJson["orthographic"];
      if(ortho.contains("xmag"))
        cam.orthographic.xmag = ortho["xmag"].get<double>();
      if(ortho.contains("ymag"))
        cam.orthographic.ymag = ortho["ymag"].get<double>();
      if(ortho.contains("znear"))
        cam.orthographic.znear = ortho["znear"].get<double>();
      if(ortho.contains("zfar"))
        cam.orthographic.zfar = ortho["zfar"].get<double>();
    }
  }
  catch(const nlohmann::json::exception& e)
  {
    LOGW("Failed to sync camera %d: %s", cameraIndex, e.what());
  }
}

void AnimationPointerSystem::syncNode(int nodeIndex)
{
  if(nodeIndex < 0 || nodeIndex >= static_cast<int>(m_model.nodes.size()))
    return;
  try
  {
    nlohmann::json::json_pointer ptr("/nodes/" + std::to_string(nodeIndex));
    const nlohmann::json&        nodeJson = m_jsonModel.at(ptr);
    tinygltf::Node&              node     = m_model.nodes[nodeIndex];

    // Manually merge extensions (like KHR_node_visibility)
    if(nodeJson.contains("extensions") && nodeJson["extensions"].is_object())
    {
      for(auto& [extName, extValue] : nodeJson["extensions"].items())
      {
        node.extensions[extName] = jsonToTinyGltfValue(extValue);
      }
    }

    // Could also handle other node properties if needed:
    // rotation, scale, translation, matrix, weights, etc.
  }
  catch(const nlohmann::json::exception& e)
  {
    LOGW("Failed to sync node %d: %s", nodeIndex, e.what());
  }
}


// Helper: Convert nlohmann::json to tinygltf::Value
tinygltf::Value AnimationPointerSystem::jsonToTinyGltfValue(const nlohmann::json& j)
{
  if(j.is_null())
  {
    return {};
  }

  if(j.is_boolean())
  {
    return tinygltf::Value(j.get<bool>());
  }
  else if(j.is_number_integer())
  {
    return tinygltf::Value(j.get<int>());
  }
  else if(j.is_number_float())
  {
    return tinygltf::Value(j.get<double>());
  }
  else if(j.is_string())
  {
    return tinygltf::Value(j.get<std::string>());
  }
  else if(j.is_array())
  {
    tinygltf::Value::Array arr;
    for(const auto& elem : j)
    {
      arr.push_back(jsonToTinyGltfValue(elem));
    }
    return tinygltf::Value(arr);
  }
  else if(j.is_object())
  {
    tinygltf::Value::Object obj;
    for(auto& [key, value] : j.items())
    {
      obj[key] = jsonToTinyGltfValue(value);
    }
    return tinygltf::Value(obj);
  }

  return {};
}

void AnimationPointerSystem::clearDirty()
{
  m_dirtyMaterials.clear();
  m_dirtyLights.clear();
  m_dirtyCameras.clear();
  m_dirtyNodes.clear();
}

bool AnimationPointerSystem::hasDirty() const
{
  return !m_dirtyMaterials.empty() || !m_dirtyLights.empty() || !m_dirtyCameras.empty() || !m_dirtyNodes.empty();
}

//--------------------------------------------------------------------------------------------------
// Reset all cached state when the model is replaced
// This must be called when takeModel() or load() replaces m_model to ensure cached indices,
// JSON pointers, and shadow JSON data don't reference stale model data.
//--------------------------------------------------------------------------------------------------
void AnimationPointerSystem::reset()
{
  m_pathCache.clear();
  m_jsonModel = nlohmann::json::object();
  clearDirty();
}


// Generic merge function: Updates tinygltf::Material from JSON shadow
// This handles any property at any depth: colors, factors, textures, extensions, nested extensions, etc.
void AnimationPointerSystem::mergeJsonIntoMaterial(const nlohmann::json& json, tinygltf::Material& mat)
{
  // Helper lambda: Merge JSON extensions into any object with extensions field
  auto mergeExtensions = [this](const nlohmann::json& json, auto& target) {
    if(json.contains("extensions") && json["extensions"].is_object())
    {
      for(auto& [extName, extValue] : json["extensions"].items())
      {
        target.extensions[extName] = jsonToTinyGltfValue(extValue);
      }
    }
  };

  // Merge pbrMetallicRoughness
  if(json.contains("pbrMetallicRoughness") && json["pbrMetallicRoughness"].is_object())
  {
    const auto& pbr = json["pbrMetallicRoughness"];

    // Scalars
    if(pbr.contains("roughnessFactor") && pbr["roughnessFactor"].is_number())
      mat.pbrMetallicRoughness.roughnessFactor = pbr["roughnessFactor"].get<double>();

    if(pbr.contains("metallicFactor") && pbr["metallicFactor"].is_number())
      mat.pbrMetallicRoughness.metallicFactor = pbr["metallicFactor"].get<double>();

    // Arrays (colors)
    if(pbr.contains("baseColorFactor") && pbr["baseColorFactor"].is_array())
    {
      const auto& arr = pbr["baseColorFactor"];
      if(arr.size() >= 4)
      {
        for(size_t i = 0; i < 4; ++i)
          mat.pbrMetallicRoughness.baseColorFactor[i] = arr[i].get<double>();
      }
    }

    // Textures (generic - handles extensions like KHR_texture_transform)
    if(pbr.contains("baseColorTexture"))
      mergeExtensions(pbr["baseColorTexture"], mat.pbrMetallicRoughness.baseColorTexture);

    if(pbr.contains("metallicRoughnessTexture"))
      mergeExtensions(pbr["metallicRoughnessTexture"], mat.pbrMetallicRoughness.metallicRoughnessTexture);
  }

  // Merge emissive properties
  if(json.contains("emissiveFactor") && json["emissiveFactor"].is_array())
  {
    const auto& arr = json["emissiveFactor"];
    if(arr.size() >= 3)
    {
      for(size_t i = 0; i < 3; ++i)
        mat.emissiveFactor[i] = arr[i].get<double>();
    }
  }

  if(json.contains("emissiveTexture"))
    mergeExtensions(json["emissiveTexture"], mat.emissiveTexture);

  // Merge normal texture
  if(json.contains("normalTexture"))
  {
    if(json["normalTexture"].contains("scale") && json["normalTexture"]["scale"].is_number())
      mat.normalTexture.scale = json["normalTexture"]["scale"].get<double>();
    mergeExtensions(json["normalTexture"], mat.normalTexture);
  }

  // Merge occlusion texture
  if(json.contains("occlusionTexture"))
  {
    if(json["occlusionTexture"].contains("strength") && json["occlusionTexture"]["strength"].is_number())
      mat.occlusionTexture.strength = json["occlusionTexture"]["strength"].get<double>();
    mergeExtensions(json["occlusionTexture"], mat.occlusionTexture);
  }

  // Merge alphaCutoff, alphaMode, doubleSided
  if(json.contains("alphaCutoff") && json["alphaCutoff"].is_number())
    mat.alphaCutoff = json["alphaCutoff"].get<double>();

  if(json.contains("alphaMode") && json["alphaMode"].is_string())
    mat.alphaMode = json["alphaMode"].get<std::string>();

  if(json.contains("doubleSided") && json["doubleSided"].is_boolean())
    mat.doubleSided = json["doubleSided"].get<bool>();

  // GENERIC: Merge ALL material extensions (handles ANY extension with nested textures/transforms)
  if(json.contains("extensions") && json["extensions"].is_object())
  {
    for(auto& [extName, extValue] : json["extensions"].items())
    {
      // Convert the sparse JSON extension to tinygltf::Value
      tinygltf::Value newExtValue = jsonToTinyGltfValue(extValue);

      // MERGE instead of REPLACE to preserve existing extension data
      if(mat.extensions.find(extName) != mat.extensions.end())
      {
        mergeValue(mat.extensions[extName], newExtValue);
      }
      else
      {
        mat.extensions[extName] = newExtValue;
      }
    }
  }
}

}  // namespace nvvkgltf
