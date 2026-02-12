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

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <glm/glm.hpp>
#include <tinygltf/json.hpp>
#include <tinygltf/tiny_gltf.h>

namespace nvvkgltf {

//----------------------------------------------------------------------------------------------
// Animation pointer system using nlohmann::json
//
// 1. Stores animated properties in a nlohmann::json shadow structure
// 2. Uses nlohmann::json::json_pointer directly for path resolution
// 3. Syncs changed values back to tinygltf::Model when needed
//
// Optimization: Caches json_pointer objects and path metadata to avoid repeated parsing.
// Animation systems call applyValue with the same paths every frame, so caching is highly effective.
//----------------------------------------------------------------------------------------------
class AnimationPointerSystem
{
public:
  AnimationPointerSystem(tinygltf::Model& model);

  // Apply animated value to a JSON pointer path
  bool applyValue(const std::string& jsonPointerPath, float value);
  bool applyValue(const std::string& jsonPointerPath, const glm::vec2& value);
  bool applyValue(const std::string& jsonPointerPath, const glm::vec3& value);
  bool applyValue(const std::string& jsonPointerPath, const glm::vec4& value);

  // Sync changed properties back to tinygltf::Model
  void syncToModel();

  // Get dirty tracking info
  const std::unordered_set<int>& getDirtyMaterials() const { return m_dirtyMaterials; }
  const std::unordered_set<int>& getDirtyLights() const { return m_dirtyLights; }
  const std::unordered_set<int>& getDirtyCameras() const { return m_dirtyCameras; }
  const std::unordered_set<int>& getDirtyNodes() const { return m_dirtyNodes; }

  void clearDirty();
  bool hasDirty() const;

  // Reset all cached state when the model is replaced
  void reset();

private:
  // Cached information for a JSON pointer path - computed once, reused every frame
  enum class ResourceType : uint8_t
  {
    eNone,
    eMaterial,
    eLight,
    eCamera,
    eNode,
  };

  struct CachedPathInfo
  {
    nlohmann::json::json_pointer ptr;                 // Parsed pointer (avoids re-parsing path string)
    int                          resourceIndex = -1;  // Index of material/light/camera/node
    ResourceType                 resourceType  = ResourceType::eNone;
  };

  tinygltf::Model& m_model;
  nlohmann::json   m_jsonModel;  // Shadow JSON structure for animated properties

  // Path cache: maps path string -> cached info (computed once per unique path)
  std::unordered_map<std::string, CachedPathInfo> m_pathCache;

  // Dirty tracking
  std::unordered_set<int> m_dirtyMaterials;
  std::unordered_set<int> m_dirtyLights;
  std::unordered_set<int> m_dirtyCameras;
  std::unordered_set<int> m_dirtyNodes;

  // Get or create cached info for a path (main optimization entry point)
  CachedPathInfo& getOrCreateCachedPath(const std::string& jsonPointerPath);

  // Helper: Parse resource type and index from path (stores in cached)
  static void parseResourceInfo(const std::string& path, CachedPathInfo& cached);

  // Helper: Mark resources as dirty using cached info (no parsing needed)
  void markDirtyFromCache(const CachedPathInfo& cached);

  // Sync helpers
  void syncMaterial(int materialIndex);
  void syncLight(int lightIndex);
  void syncCamera(int cameraIndex);
  void syncNode(int nodeIndex);

  // Generic merge helper (updates tinygltf from JSON shadow - handles ANY property/extension)
  void mergeJsonIntoMaterial(const nlohmann::json& json, tinygltf::Material& mat);

  // Conversion helper (recursive - handles nested structures)
  static tinygltf::Value jsonToTinyGltfValue(const nlohmann::json& j);
};

}  // namespace nvvkgltf
