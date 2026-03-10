/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

//
// Standalone parser for .scene.json / .glXf descriptor files.
//
// Descriptor files define a set of glTF models and their instances (transform, name).
// Supports RTXPT-style "graph" nodes and a simple "models" + "instances" array format.
// No dependency on Scene or Vulkan types; used by the application to load and place scenes.
//

// One glTF model entry from the descriptor: path (as in the file) and resolved filesystem path.
struct SceneDescriptorModel
{
  std::string           path;          // Relative or absolute path to the glTF file
  std::filesystem::path resolvedPath;  // Resolved (absolute) path for loading
};

// One placed instance of a model: index into the models array plus TRS transform and optional name.
struct SceneDescriptorInstance
{
  int         modelIndex = 0;   // Index into SceneDescriptor::models
  glm::vec3   translation{0.f}; // Position
  glm::quat   rotation{1.f, 0.f, 0.f, 0.f}; // Orientation (glTF order: xyzw)
  glm::vec3   scale{1.f};       // Scale (uniform or per-axis)
  std::string name;             // Optional instance name
};

// Aggregated result of a scene descriptor: list of models and list of instances.
struct SceneDescriptor
{
  std::vector<SceneDescriptorModel>    models;    // glTF files to load
  std::vector<SceneDescriptorInstance> instances; // Placed instances (model index + transform)

  // Groups instances by model index. Key = model index, value = pointers into instances.
  std::unordered_map<int, std::vector<const SceneDescriptorInstance*>> getInstancesByModel() const
  {
    std::unordered_map<int, std::vector<const SceneDescriptorInstance*>> result;
    for(const auto& inst : instances)
      result[inst.modelIndex].push_back(&inst);
    return result;
  }
};

// Parses a .scene.json or .glXf descriptor file. Model paths are resolved relative to the
// descriptor's directory. Returns true on success; on failure, logs the error and returns false.
bool loadSceneDescriptor(const std::filesystem::path& descriptorPath, SceneDescriptor& outDescriptor);
