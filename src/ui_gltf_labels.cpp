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

#include "ui_gltf_labels.hpp"

#include <imgui.h>

namespace uigltf {

// glTF sampler enum/name tables (single source of truth for both the list summary and the editors).
static const int kWrapEnum[] = {TINYGLTF_TEXTURE_WRAP_REPEAT, TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE, TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT};
static const char* const kWrapNames[] = {"Repeat", "Clamp to edge", "Mirrored repeat"};
static const int         kMagEnum[]   = {-1, TINYGLTF_TEXTURE_FILTER_NEAREST, TINYGLTF_TEXTURE_FILTER_LINEAR};
static const char* const kMagNames[]  = {"Default", "Nearest", "Linear"};
static const int         kMinEnum[]   = {-1,
                                         TINYGLTF_TEXTURE_FILTER_NEAREST,
                                         TINYGLTF_TEXTURE_FILTER_LINEAR,
                                         TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST,
                                         TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST,
                                         TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR,
                                         TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR};
static const char* const kMinNames[]  = {
    "Default",          "Nearest", "Linear", "Nearest/Nearest mip", "Linear/Nearest mip", "Nearest/Linear mip",
    "Linear/Linear mip"};

template <size_t N>
static const char* enumName(int value, const int (&enums)[N], const char* const (&names)[N])
{
  for(size_t j = 0; j < N; ++j)
    if(enums[j] == value)
      return names[j];
  return "?";
}

// Combo over parallel enum/name arrays; *value holds the glTF enum. Returns true when changed.
template <size_t N>
static bool enumCombo(const char* label, int* value, const int (&enums)[N], const char* const (&names)[N])
{
  int cur = 0;
  for(size_t j = 0; j < N; ++j)
    if(enums[j] == *value)
      cur = static_cast<int>(j);
  if(ImGui::Combo(label, &cur, names, static_cast<int>(N)))
  {
    *value = enums[cur];
    return true;
  }
  return false;
}

const char* wrapName(int value)
{
  return enumName(value, kWrapEnum, kWrapNames);
}
const char* magName(int value)
{
  return enumName(value, kMagEnum, kMagNames);
}
const char* minName(int value)
{
  return enumName(value, kMinEnum, kMinNames);
}

std::string imageDisplayName(const tinygltf::Model& model, int imageIndex)
{
  if(imageIndex < 0 || imageIndex >= static_cast<int>(model.images.size()))
    return {};
  const tinygltf::Image& image = model.images[imageIndex];
  if(!image.uri.empty())
    return image.uri;
  if(!image.name.empty())
    return image.name;
  return "Embedded image " + std::to_string(imageIndex);
}

std::string samplerSummary(const tinygltf::Sampler& s)
{
  return std::string(wrapName(s.wrapS)) + " / " + minName(s.minFilter);
}

bool renderSamplerFields(const tinygltf::Sampler& cur, const std::function<void(const tinygltf::Sampler&)>& commit)
{
  bool changed = false;
  // One field per combo; a change hands back a copy of cur with that field applied (one field/frame).
  auto edit = [&](const char* label, int tinygltf::Sampler::* field, const auto& enums, const auto& names) {
    ImGui::SetNextItemWidth(180.0f);
    int v = cur.*field;
    if(enumCombo(label, &v, enums, names))
    {
      tinygltf::Sampler edited = cur;
      edited.*field            = v;
      commit(edited);
      changed = true;
    }
  };
  edit("Wrap S", &tinygltf::Sampler::wrapS, kWrapEnum, kWrapNames);
  edit("Wrap T", &tinygltf::Sampler::wrapT, kWrapEnum, kWrapNames);
  edit("Mag filter", &tinygltf::Sampler::magFilter, kMagEnum, kMagNames);
  edit("Min filter", &tinygltf::Sampler::minFilter, kMinEnum, kMinNames);
  return changed;
}

long long primitiveTriangleCountForMode(int mode, long long count)
{
  switch(mode)
  {
    case TINYGLTF_MODE_TRIANGLES:
      return count / 3;
    case TINYGLTF_MODE_TRIANGLE_STRIP:
    case TINYGLTF_MODE_TRIANGLE_FAN:
      return count >= 3 ? count - 2 : 0;
    default:  // POINTS, LINE, LINE_LOOP, LINE_STRIP
      return 0;
  }
}

}  // namespace uigltf
