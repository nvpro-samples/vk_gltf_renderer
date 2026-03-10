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

//
// Parses .scene.json / .glXf descriptor files that define multiple glTF models
// and their instances (path, transform, name). Supports both RTXPT-style
// "graph" nodes and a simple "models" + "instances" array format. No dependency
// on Scene or Vulkan; output is used by the application to load and place scenes.
//

#include "scene_descriptor.hpp"

#include <fstream>
#include <tinygltf/json.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>

using json = nlohmann::json;

//--------------------------------------------------------------------------------------------------
// JSON parsing helpers
//--------------------------------------------------------------------------------------------------

// Returns a 3-component vector from a JSON array at key, or defaultVal if missing/invalid.
static glm::vec3 parseVec3(const json& j, const char* key, const glm::vec3& defaultVal)
{
  if(!j.contains(key) || !j[key].is_array() || j[key].size() < 3)
    return defaultVal;
  return {j[key][0].get<float>(), j[key][1].get<float>(), j[key][2].get<float>()};
}

// Returns a scale vector: accepts a single number (uniform) or a 3-component array.
// Tries key1 first, then key2; returns defaultVal if neither is present or valid.
static glm::vec3 parseScale(const json& j, const char* key1, const char* key2, const glm::vec3& defaultVal)
{
  const char* key = j.contains(key1) ? key1 : (j.contains(key2) ? key2 : nullptr);
  if(!key)
    return defaultVal;
  const auto& val = j[key];
  if(val.is_number())
  {
    float s = val.get<float>();
    return {s, s, s};
  }
  if(val.is_array() && val.size() >= 3)
    return {val[0].get<float>(), val[1].get<float>(), val[2].get<float>()};
  return defaultVal;
}

// Returns a quaternion from a JSON array at key (glTF order: x, y, z, w), or defaultVal if missing/invalid.
static glm::quat parseQuat(const json& j, const char* key, const glm::quat& defaultVal)
{
  if(!j.contains(key) || !j[key].is_array() || j[key].size() < 4)
    return defaultVal;
  // glTF convention: [x, y, z, w]
  return glm::quat(j[key][3].get<float>(), j[key][0].get<float>(), j[key][1].get<float>(), j[key][2].get<float>());
}

//--------------------------------------------------------------------------------------------------
// Load scene descriptor
//--------------------------------------------------------------------------------------------------
// Parses a .scene.json or .glXf file: reads "models" (glTF paths) and "graph" or "instances"
// (model index, translation, rotation, scale, name). Model paths are resolved relative to the
// descriptor's directory. On success, out is populated and true is returned; on failure, an
// error is logged and false is returned. If no instances are present, one identity instance
// per model is created.
//--------------------------------------------------------------------------------------------------
bool loadSceneDescriptor(const std::filesystem::path& descriptorPath, SceneDescriptor& out)
{
  std::ifstream file(descriptorPath);
  if(!file.is_open())
  {
    LOGE("Cannot open scene descriptor: %s\n", nvutils::utf8FromPath(descriptorPath).c_str());
    return false;
  }

  std::string jsonStr((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  json doc;
  try
  {
    doc = json::parse(jsonStr, {}, true, true, true);
  }
  catch(const json::parse_error& e)
  {
    LOGE("JSON parse error in %s: %s\n", nvutils::utf8FromPath(descriptorPath).c_str(), e.what());
    return false;
  }

  std::filesystem::path baseDir = descriptorPath.parent_path();

  out = {};

  if(!doc.contains("models") || !doc["models"].is_array())
  {
    LOGE("Scene descriptor missing 'models' array: %s\n", nvutils::utf8FromPath(descriptorPath).c_str());
    return false;
  }

  for(const auto& m : doc["models"])
  {
    SceneDescriptorModel model;
    if(m.is_string())
      model.path = m.get<std::string>();  // RTXPT format: bare string
    else if(m.is_object())
      model.path = m.value("path", "");  // Object format: { "path": "..." }
    if(model.path.empty())
      LOGW("Scene descriptor: model entry %zu has empty path\n", out.models.size());
    else
      model.resolvedPath = baseDir / model.path;
    out.models.push_back(std::move(model));  // Always push to keep indices aligned with "model" references
  }

  // Parse instances from "graph" (RTXPT format) or "instances" array.
  // RTXPT "graph" nodes with a "model" key are model instances; others (lights, cameras, settings) are skipped.
  const char* instancesKey = doc.contains("graph") ? "graph" : "instances";

  if(doc.contains(instancesKey) && doc[instancesKey].is_array())
  {
    for(const auto& node : doc[instancesKey])
    {
      if(!node.is_object())
        continue;

      // Skip non-model graph nodes (lights, cameras, settings)
      if(!node.contains("model"))
        continue;

      SceneDescriptorInstance instance;
      instance.modelIndex  = node.value("model", 0);
      instance.translation = parseVec3(node, "translation", glm::vec3(0.f));
      instance.rotation    = parseQuat(node, "rotation", glm::quat(1.f, 0.f, 0.f, 0.f));
      instance.scale       = parseScale(node, "scale", "scaling", glm::vec3(1.f));
      instance.name        = node.value("name", "");

      if(instance.modelIndex < 0 || instance.modelIndex >= static_cast<int>(out.models.size()))
      {
        LOGW("Scene descriptor: instance references invalid model index %d, skipping\n", instance.modelIndex);
        continue;
      }
      out.instances.push_back(std::move(instance));
    }
  }

  // If no instances were found, create one default instance per model at identity
  if(out.instances.empty())
  {
    for(int i = 0; i < static_cast<int>(out.models.size()); i++)
    {
      SceneDescriptorInstance instance;
      instance.modelIndex = i;
      out.instances.push_back(std::move(instance));
    }
  }

  LOGI("Scene descriptor: %zu models, %zu instances from %s\n", out.models.size(), out.instances.size(),
       nvutils::utf8FromPath(descriptorPath.filename()).c_str());
  return true;
}
