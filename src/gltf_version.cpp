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

#include "gltf_version.hpp"

#include <charconv>
#include <map>
#include <vector>

namespace nvvkgltf {

namespace {

// One version component: "0", or a non-zero digit followed by up to eight digits (spec "Asset").
std::optional<uint32_t> parseComponent(std::string_view text)
{
  if(text.empty() || text.size() > 9 || (text.size() > 1 && text[0] == '0'))
    return std::nullopt;
  for(char c : text)
  {
    if(c < '0' || c > '9')
      return std::nullopt;
  }
  uint32_t value = 0;
  std::from_chars(text.data(), text.data() + text.size(), value);
  return value;
}

// "TEXCOORD_3" with prefix "TEXCOORD_" -> 3; anything else -> -1.
int64_t semanticIndex(const std::string& name, std::string_view prefix)
{
  if(name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix) != 0)
    return -1;
  const std::optional<uint32_t> index = parseComponent(std::string_view(name).substr(prefix.size()));
  return index ? int64_t(*index) : -1;
}

// glTF 2.0 requires TEXCOORD_n / COLOR_n sets to start at 0 without gaps; 2.1 lifts that.
bool hasNonSequentialSets(const std::map<std::string, int>& attributes, std::string_view prefix)
{
  std::vector<int64_t> indices;
  for(const auto& [name, accessor] : attributes)
  {
    const int64_t index = semanticIndex(name, prefix);
    if(index >= 0)
      indices.push_back(index);
  }
  // Sets 0..k-1 are all present exactly when every index is below the number of sets.
  for(size_t i = 0; i < indices.size(); i++)
  {
    if(indices[i] >= int64_t(indices.size()))
      return true;
  }
  return false;
}

// Component types glTF 2.0 does not define: int32, float64, float16, int64, uint64.
bool isGltf21ComponentType(int componentType)
{
  switch(componentType)
  {
    case TINYGLTF_COMPONENT_TYPE_INT:
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
    case TINYGLTF_COMPONENT_TYPE_HALF_FLOAT:
    case TINYGLTF_COMPONENT_TYPE_INT64:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT64:
      return true;
    default:
      return false;
  }
}

template <typename T>
bool anyUid(const std::vector<T>& objects)
{
  for(size_t i = 0; i < objects.size(); i++)
  {
    if(!objects[i].uid.empty())
      return true;
  }
  return false;
}

}  // namespace

std::string GltfVersion::toString() const
{
  return std::to_string(majorVersion) + "." + std::to_string(minorVersion);
}

std::optional<GltfVersion> parseGltfVersion(std::string_view text)
{
  const size_t dot = text.find('.');
  if(dot == std::string_view::npos)
    return std::nullopt;
  const std::optional<uint32_t> majorPart = parseComponent(text.substr(0, dot));
  const std::optional<uint32_t> minorPart = parseComponent(text.substr(dot + 1));
  if(!majorPart || !minorPart)
    return std::nullopt;
  return GltfVersion{*majorPart, *minorPart};
}

bool isGltfVersionLoadable(const tinygltf::Asset& asset, std::string* reason)
{
  auto fail = [reason](std::string message) {
    if(reason)
      *reason = std::move(message);
    return false;
  };

  const std::optional<GltfVersion> version = parseGltfVersion(asset.version);
  if(!version)
    return fail("invalid asset.version \"" + asset.version + "\"");

  if(!asset.minVersion.empty())
  {
    const std::optional<GltfVersion> minVersion = parseGltfVersion(asset.minVersion);
    if(!minVersion)
      return fail("invalid asset.minVersion \"" + asset.minVersion + "\"");
    if(*minVersion > *version)
      return fail("asset.minVersion \"" + asset.minVersion + "\" exceeds asset.version \"" + asset.version + "\"");
    if(*minVersion > kMaxSupportedGltfVersion)
      return fail("asset requires glTF " + minVersion->toString() + " (asset.minVersion); this renderer supports up to "
                  + kMaxSupportedGltfVersion.toString());
  }

  if(version->majorVersion != kMaxSupportedGltfVersion.majorVersion)
    return fail("unsupported glTF major version " + version->toString());

  return true;
}

std::string findGltf21Feature(const tinygltf::Model& model)
{
  // PROVISIONAL(KhronosGroup/glTF#2585): the 2.1 additions as of the current draft. Promoted
  // extensions (Khronos MR 19) are missing until tinygltf parses them.
  if(!model.files.empty())
    return "files";
  if(!model.externalAssets.empty())
    return "externalAssets";
  if(!model.shapes.empty())
    return "shapes";
  if(model.asset.thumbnail >= 0)
    return "asset.thumbnail";
  for(size_t i = 0; i < model.nodes.size(); i++)
  {
    const tinygltf::Node& node = model.nodes[i];
    if(node.externalAsset >= 0)
      return "node.externalAsset";
    if(node.boundingVolume.shape >= 0)
      return "node.boundingVolume";
  }

  if(anyUid(model.nodes) || anyUid(model.meshes) || anyUid(model.materials) || anyUid(model.images)
     || anyUid(model.textures) || anyUid(model.samplers) || anyUid(model.scenes) || anyUid(model.animations)
     || anyUid(model.skins) || anyUid(model.cameras) || anyUid(model.lights))
    return "uid";

  // This also covers int32 vertex attributes. uint32 ones are not a 2.1 feature: 2.0 only says
  // uint32 SHOULD NOT be used outside indices, so such a file is still valid 2.0.
  for(size_t i = 0; i < model.accessors.size(); i++)
  {
    const int componentType = model.accessors[i].componentType;
    if(isGltf21ComponentType(componentType))
      return "accessor component type " + std::to_string(componentType);
  }

  for(size_t m = 0; m < model.meshes.size(); m++)
  {
    for(size_t p = 0; p < model.meshes[m].primitives.size(); p++)
    {
      const tinygltf::Primitive& primitive = model.meshes[m].primitives[p];
      if(hasNonSequentialSets(primitive.attributes, "TEXCOORD_"))
        return "non-sequential TEXCOORD_n";
      if(hasNonSequentialSets(primitive.attributes, "COLOR_"))
        return "non-sequential COLOR_n";
    }
  }
  return {};
}

std::string gltfVersionForSave(const tinygltf::Model& model)
{
  // Loading rejects a minVersion above kMaxSupportedGltfVersion, so honoring it never asks for more
  // than we write. The spec requires minVersion <= version.
  const std::optional<GltfVersion> minVersion = parseGltfVersion(model.asset.minVersion);
  if(minVersion && *minVersion >= GltfVersion{2, 1})
    return "2.1";
  return findGltf21Feature(model).empty() ? "2.0" : "2.1";
}

}  // namespace nvvkgltf
