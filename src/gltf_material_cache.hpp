/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use it except in compliance with the License.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include <tinygltf/tiny_gltf.h>

#include "shaders/gltf_scene_io.h.slang"

namespace nvvkgltf {

// Contiguous span of texture infos used by a material (for surgical upload)
struct TextureInfoSpan
{
  uint16_t minIdx = std::numeric_limits<uint16_t>::max();
  uint16_t maxIdx = 0;
  uint16_t count  = 0;

  bool hasAny() const { return count > 0; }

  // Returns the number of slots in [minIdx, maxIdx] (inclusive). Returns 0 when empty (count == 0)
  // to avoid underflow when minIdx is default (uint16_t max) and maxIdx is 0.
  size_t spanSize() const
  {
    if(!hasAny() || minIdx > maxIdx)
      return 0;
    return static_cast<size_t>(maxIdx) - static_cast<size_t>(minIdx) + 1;
  }
};

// Result of updating a single cached material
struct MaterialUpdateResult
{
  TextureInfoSpan span{};
  bool            topologyChanged = false;  // Texture slots added/removed; caller should rebuild cache
};

// Per-glTF-texture data resolved by SceneVk and baked into each GltfTextureInfo. Both vectors are
// indexed by glTF texture index and may be shorter than model.textures (or empty), in which case the
// accessors below fall back to the neutral default.
struct TextureSlotTable
{
  // Resolved GPU sampler slot (SceneVk::samplers(); slot 0 = default). Pure model data, so it is
  // available before the images are decoded.
  std::vector<int> samplerSlots;

  // Set when the texture's source image decoded to a format with fewer than three color components
  // (BC5, R8G8, ...). Only known once the images exist, hence the separate vector.
  std::vector<uint8_t> twoChannelSource;

  int samplerSlot(int texIndex) const
  {
    return (texIndex >= 0 && texIndex < static_cast<int>(samplerSlots.size())) ? samplerSlots[texIndex] : 0;
  }

  bool isTwoChannelSource(int texIndex) const
  {
    return texIndex >= 0 && texIndex < static_cast<int>(twoChannelSource.size()) && twoChannelSource[texIndex] != 0;
  }

  void clear()
  {
    samplerSlots.clear();
    twoChannelSource.clear();
  }
};

/*-------------------------------------------------------------------------------------------------
# class nvvkgltf::MaterialCache

>  CPU cache of shader materials and texture infos converted from tinygltf::Material.
   Used by SceneVk to upload material/texture buffers. Supports full rebuild and
   per-material update with topology change detection.
--------------------------------------------------------------------------------------------------*/
class MaterialCache
{
public:
  // textureSlots carries the per-texture data baked into each GltfTextureInfo (sampler slot and
  // source-format flags); see TextureSlotTable.
  void buildFromMaterials(const std::vector<tinygltf::Material>& materials, const TextureSlotTable& textureSlots);

  // Update one cached material in place. Returns span of texture infos and whether
  // texture slot topology changed (if true, caller should rebuild cache).
  [[nodiscard]] MaterialUpdateResult updateMaterial(int index, const tinygltf::Material& srcMat, const TextureSlotTable& textureSlots);

  void clear();

  const std::vector<shaderio::GltfShadeMaterial>& getShadeMaterials() const { return m_shadeMaterials; }
  const std::vector<shaderio::GltfTextureInfo>&   getTextureInfos() const { return m_textureInfos; }

private:
  std::vector<shaderio::GltfShadeMaterial> m_shadeMaterials;
  std::vector<shaderio::GltfTextureInfo>   m_textureInfos;
};

}  // namespace nvvkgltf
