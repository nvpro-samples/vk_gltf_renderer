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

//--------------------------------------------------------------------------------------------------
// The text layer for `.sky.json`. See the header for what the format is and why it is not ours.
//
// tinygltf::Value and nlohmann::json are the same shape -- object, array, string, number, bool --
// so the conversion (tinygltf::utils::valueToJson / valueFromJson) is the obvious recursion and
// nothing more. It exists because tinygltf will only serialize a Value as part of a whole glTF
// document, and a preset is one fragment of one.
//

#include <algorithm>
#include <fstream>

#include <tinygltf/json.hpp>  // the nlohmann single-header tinygltf already vendors
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>

#include "sky_preset.hpp"
#include "tinygltf_utils.hpp"

namespace {

constexpr const char* kSunRotation = "sunRotation";

}  // namespace

namespace sky_preset {

bool save(const std::filesystem::path& path, const Preset& preset)
{
  nlohmann::json json = tinygltf::utils::valueToJson(gltf_environment_sky::toValue(preset.sky));
  if(!json.is_object())
  {
    LOGE("Sky preset: nothing to write for '%s'.\n", nvutils::utf8FromPath(path).c_str());
    return false;
  }

  if(preset.sunRotation.has_value())
  {
    // glTF's quaternion order, xyzw, so the number reads the same here as it does on a node.
    const glm::quat& q = *preset.sunRotation;
    json[kSunRotation] = {q.x, q.y, q.z, q.w};
  }

  std::ofstream file(path, std::ios::binary);
  if(!file)
  {
    LOGE("Sky preset: could not open '%s' for writing.\n", nvutils::utf8FromPath(path).c_str());
    return false;
  }
  // Indented, because a preset is a file people are meant to read, diff and hand-edit.
  file << json.dump(2) << '\n';
  if(!file)
  {
    LOGE("Sky preset: failed while writing '%s'.\n", nvutils::utf8FromPath(path).c_str());
    return false;
  }

  LOGI("Sky preset: wrote %s\n", nvutils::utf8FromPath(path).c_str());
  return true;
}

std::optional<Preset> load(const std::filesystem::path& path)
{
  std::ifstream file(path, std::ios::binary);
  if(!file)
  {
    LOGE("Sky preset: could not open '%s'.\n", nvutils::utf8FromPath(path).c_str());
    return std::nullopt;
  }

  nlohmann::json json;
  try
  {
    file >> json;
  }
  catch(const nlohmann::json::exception& e)
  {
    LOGE("Sky preset: '%s' is not valid JSON (%s).\n", nvutils::utf8FromPath(path).c_str(), e.what());
    return std::nullopt;
  }

  if(!json.is_object())
  {
    LOGE("Sky preset: '%s' is not a sky object.\n", nvutils::utf8FromPath(path).c_str());
    return std::nullopt;
  }

  // The sun rotation is the preset's own key, not the sky's. Taken out before the rest becomes the
  // descriptor, whose `raw` is written back verbatim -- left in, it would ride into every glTF
  // saved after this preset was loaded, as a property OMI_environment_sky does not define.
  nlohmann::json sunRotation;
  if(auto it = json.find(kSunRotation); it != json.end())
  {
    sunRotation = std::move(*it);
    json.erase(it);
  }

  EnvironmentState sky = gltf_environment_sky::fromValue(tinygltf::utils::valueFromJson(json));
  if(!sky.has_value())
  {
    LOGE("Sky preset: '%s' does not describe a sky.\n", nvutils::utf8FromPath(path).c_str());
    return std::nullopt;
  }

  Preset preset{.sky = *sky};

  // Optional, and quietly ignored if malformed: a preset that restores the atmosphere but not the
  // sun is still worth having, which is not true of one that refuses to load over a bad array.
  const bool validRotation =
      sunRotation.is_array() && sunRotation.size() == 4
      && std::all_of(sunRotation.begin(), sunRotation.end(), [](const nlohmann::json& v) { return v.is_number(); });
  if(validRotation)
  {
    preset.sunRotation = glm::quat(sunRotation[3].get<float>(), sunRotation[0].get<float>(),
                                   sunRotation[1].get<float>(), sunRotation[2].get<float>());
  }
  else if(!sunRotation.is_null())
  {
    LOGW("Sky preset: '%s' has a malformed `%s` (expected four numbers); the sun is left where it is.\n",
         nvutils::utf8FromPath(path).c_str(), kSunRotation);
  }

  return preset;
}

bool isPresetPath(const std::filesystem::path& path)
{
  // The full `.sky.json`, not merely `.json`. The viewport's drop handler already claims
  // `.scene.json` for scene descriptors, and a rule of "any .json is a sky" would quietly take
  // those over -- the kind of collision that only shows up when someone drops the other file.
  std::string name = nvutils::utf8FromPath(path.filename());
  for(char& c : name)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return name.ends_with(".sky.json");
}

}  // namespace sky_preset
