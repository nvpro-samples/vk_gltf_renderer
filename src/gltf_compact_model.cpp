/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Low-level utilities for compacting a tinygltf::Model in place.
// Removes unused buffers, buffer views, accessors, images, textures,
// samplers, and other resources, then remaps all indices so the model
// remains valid after orphaned elements are stripped.
//

#include "gltf_compact_model.hpp"

#include <set>
#include <vector>

#include "tinygltf_utils.hpp"
#include "nvutils/logger.hpp"

#define KHR_DRACO_MESH_COMPRESSION_EXTENSION_NAME "KHR_draco_mesh_compression"

namespace {

inline size_t align4(size_t offset)
{
  return (offset + 3) & ~size_t(3);
}

inline bool isDracoCompressed(const tinygltf::Primitive& primitive)
{
  return tinygltf::utils::hasElementName(primitive.extensions, KHR_DRACO_MESH_COMPRESSION_EXTENSION_NAME);
}

void collectPrimitiveAccessors(const tinygltf::Primitive& primitive, std::set<int>& usedAccessors)
{
  if(isDracoCompressed(primitive))
    return;

  for(const auto& [name, accessorIdx] : primitive.attributes)
  {
    if(accessorIdx >= 0)
      usedAccessors.insert(accessorIdx);
  }

  if(primitive.indices >= 0)
    usedAccessors.insert(primitive.indices);

  for(const auto& target : primitive.targets)
  {
    for(const auto& [name, accessorIdx] : target)
    {
      if(accessorIdx >= 0)
        usedAccessors.insert(accessorIdx);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Phase 1: Collect all accessor indices referenced by meshes, skins, and animations.
//--------------------------------------------------------------------------------------------------
std::set<int> collectUsedAccessors(const tinygltf::Model& model)
{
  std::set<int> usedAccessors;

  for(const auto& mesh : model.meshes)
  {
    for(const auto& primitive : mesh.primitives)
    {
      collectPrimitiveAccessors(primitive, usedAccessors);
    }
  }

  for(const auto& skin : model.skins)
  {
    if(skin.inverseBindMatrices >= 0)
      usedAccessors.insert(skin.inverseBindMatrices);
  }

  for(const auto& animation : model.animations)
  {
    for(const auto& sampler : animation.samplers)
    {
      if(sampler.input >= 0)
        usedAccessors.insert(sampler.input);
      if(sampler.output >= 0)
        usedAccessors.insert(sampler.output);
    }
  }

  return usedAccessors;
}

//--------------------------------------------------------------------------------------------------
// Phase 2: Collect all buffer view indices referenced by used accessors and images.
//--------------------------------------------------------------------------------------------------
std::set<int> collectUsedBufferViews(const tinygltf::Model& model, const std::set<int>& usedAccessors)
{
  std::set<int> usedBufferViews;

  for(int accIdx : usedAccessors)
  {
    if(accIdx < 0 || accIdx >= static_cast<int>(model.accessors.size()))
      continue;

    const auto& accessor = model.accessors[accIdx];
    if(accessor.bufferView >= 0)
      usedBufferViews.insert(accessor.bufferView);

    if(accessor.sparse.isSparse)
    {
      if(accessor.sparse.indices.bufferView >= 0)
        usedBufferViews.insert(accessor.sparse.indices.bufferView);
      if(accessor.sparse.values.bufferView >= 0)
        usedBufferViews.insert(accessor.sparse.values.bufferView);
    }
  }

  for(const auto& image : model.images)
  {
    if(image.bufferView >= 0)
      usedBufferViews.insert(image.bufferView);
  }

  return usedBufferViews;
}

//--------------------------------------------------------------------------------------------------
// Phase 3: Collect all buffer indices referenced by used buffer views (including meshopt extension).
//--------------------------------------------------------------------------------------------------
std::set<int> collectUsedBuffers(const tinygltf::Model& model, const std::set<int>& usedBufferViews)
{
  std::set<int> usedBuffers;

  for(int bvIdx : usedBufferViews)
  {
    if(bvIdx < 0 || bvIdx >= static_cast<int>(model.bufferViews.size()))
      continue;

    const auto& bv = model.bufferViews[bvIdx];
    if(bv.buffer >= 0)
      usedBuffers.insert(bv.buffer);

    KHR_meshopt_compression mcomp;
    if(tinygltf::utils::getMeshoptCompression(bv, mcomp) && mcomp.buffer >= 0)
      usedBuffers.insert(mcomp.buffer);
  }

  return usedBuffers;
}

//--------------------------------------------------------------------------------------------------
// Determine whether compaction would save meaningful space.
//--------------------------------------------------------------------------------------------------
bool isCompactionNeeded(const tinygltf::Model& model,
                        const std::set<int>&   usedAccessors,
                        const std::set<int>&   usedBufferViews,
                        const std::set<int>&   usedBuffers)
{
  if(usedAccessors.size() < model.accessors.size() || usedBufferViews.size() < model.bufferViews.size()
     || usedBuffers.size() < model.buffers.size())
    return true;

  size_t usedDataSize = 0;
  for(int bvIdx : usedBufferViews)
    usedDataSize += model.bufferViews[bvIdx].byteLength;

  size_t totalBufferSize = 0;
  for(const auto& buffer : model.buffers)
    totalBufferSize += buffer.data.size();

  size_t alignmentOverhead = usedBufferViews.size() * 4;
  return (usedDataSize + alignmentOverhead) < totalBufferSize * 0.95;
}

//--------------------------------------------------------------------------------------------------
// Phase 5: Merge all used buffer view data into a single compact buffer with 4-byte alignment.
//--------------------------------------------------------------------------------------------------
struct CompactBufferResult
{
  tinygltf::Buffer    buffer;
  std::vector<size_t> newOffsets;
};

CompactBufferResult buildCompactBuffer(const tinygltf::Model& model, const std::set<int>& usedBufferViews)
{
  CompactBufferResult result;
  result.buffer.name = model.buffers.empty() ? "" : model.buffers[0].name;
  result.newOffsets.reserve(usedBufferViews.size());

  for(int oldBvIdx : usedBufferViews)
  {
    const auto& oldBv = model.bufferViews[oldBvIdx];

    if(oldBv.buffer < 0 || oldBv.buffer >= static_cast<int>(model.buffers.size()))
    {
      result.newOffsets.push_back(result.buffer.data.size());
      continue;
    }

    const auto& oldBuffer = model.buffers[oldBv.buffer];

    if(oldBv.byteOffset + oldBv.byteLength > oldBuffer.data.size())
    {
      result.newOffsets.push_back(result.buffer.data.size());
      continue;
    }

    size_t alignedOffset = align4(result.buffer.data.size());
    if(alignedOffset > result.buffer.data.size())
      result.buffer.data.resize(alignedOffset, 0);

    result.newOffsets.push_back(alignedOffset);

    const unsigned char* srcData = oldBuffer.data.data() + oldBv.byteOffset;
    result.buffer.data.insert(result.buffer.data.end(), srcData, srcData + oldBv.byteLength);
  }

  result.buffer.byteLength = result.buffer.data.size();
  return result;
}

//--------------------------------------------------------------------------------------------------
// Phase 6: Create new buffer views pointing into the compact buffer.
//--------------------------------------------------------------------------------------------------
std::vector<tinygltf::BufferView> buildCompactBufferViews(const tinygltf::Model&     model,
                                                          const std::set<int>&       usedBufferViews,
                                                          const std::vector<size_t>& newOffsets)
{
  std::vector<tinygltf::BufferView> newBufferViews;
  newBufferViews.reserve(usedBufferViews.size());

  size_t bvIndex = 0;
  for(int oldBvIdx : usedBufferViews)
  {
    tinygltf::BufferView newBv = model.bufferViews[oldBvIdx];
    newBv.buffer               = 0;
    newBv.byteOffset           = newOffsets[bvIndex++];

    const char* extName = nullptr;
    if(tinygltf::utils::hasElementName(newBv.extensions, KHR_MESHOPT_COMPRESSION_EXTENSION_NAME))
      extName = KHR_MESHOPT_COMPRESSION_EXTENSION_NAME;
    else if(tinygltf::utils::hasElementName(newBv.extensions, EXT_MESHOPT_COMPRESSION_EXTENSION_NAME))
      extName = EXT_MESHOPT_COMPRESSION_EXTENSION_NAME;

    if(extName)
    {
      tinygltf::Value& ext = newBv.extensions[extName];
      if(ext.Has("buffer"))
        ext.Get<tinygltf::Value::Object>()["buffer"] = tinygltf::Value(0);
    }

    newBufferViews.push_back(std::move(newBv));
  }

  return newBufferViews;
}

//--------------------------------------------------------------------------------------------------
// Phase 7: Create new accessors with remapped buffer view indices.
//--------------------------------------------------------------------------------------------------
std::vector<tinygltf::Accessor> buildCompactAccessors(const tinygltf::Model&  model,
                                                      const std::set<int>&    usedAccessors,
                                                      const std::vector<int>& bufferViewRemap)
{
  std::vector<tinygltf::Accessor> newAccessors;
  newAccessors.reserve(usedAccessors.size());

  for(int oldAccIdx : usedAccessors)
  {
    tinygltf::Accessor newAcc = model.accessors[oldAccIdx];

    if(newAcc.bufferView >= 0)
      newAcc.bufferView = bufferViewRemap[newAcc.bufferView];

    if(newAcc.sparse.isSparse)
    {
      if(newAcc.sparse.indices.bufferView >= 0)
        newAcc.sparse.indices.bufferView = bufferViewRemap[newAcc.sparse.indices.bufferView];
      if(newAcc.sparse.values.bufferView >= 0)
        newAcc.sparse.values.bufferView = bufferViewRemap[newAcc.sparse.values.bufferView];
    }

    newAccessors.push_back(std::move(newAcc));
  }

  return newAccessors;
}

//--------------------------------------------------------------------------------------------------
// Phase 8: Update all accessor references in meshes, skins, animations, and images.
//--------------------------------------------------------------------------------------------------
void updateModelReferences(tinygltf::Model& model, const std::vector<int>& accessorRemap, const std::vector<int>& bufferViewRemap)
{
  for(auto& mesh : model.meshes)
  {
    for(auto& primitive : mesh.primitives)
    {
      if(isDracoCompressed(primitive))
        continue;

      for(auto& [name, accessorIdx] : primitive.attributes)
      {
        if(accessorIdx >= 0 && accessorIdx < static_cast<int>(accessorRemap.size()))
          accessorIdx = accessorRemap[accessorIdx];
      }

      if(primitive.indices >= 0 && primitive.indices < static_cast<int>(accessorRemap.size()))
        primitive.indices = accessorRemap[primitive.indices];

      for(auto& target : primitive.targets)
      {
        for(auto& [name, accessorIdx] : target)
        {
          if(accessorIdx >= 0 && accessorIdx < static_cast<int>(accessorRemap.size()))
            accessorIdx = accessorRemap[accessorIdx];
        }
      }
    }
  }

  for(auto& skin : model.skins)
  {
    if(skin.inverseBindMatrices >= 0 && skin.inverseBindMatrices < static_cast<int>(accessorRemap.size()))
      skin.inverseBindMatrices = accessorRemap[skin.inverseBindMatrices];
  }

  for(auto& animation : model.animations)
  {
    for(auto& sampler : animation.samplers)
    {
      if(sampler.input >= 0 && sampler.input < static_cast<int>(accessorRemap.size()))
        sampler.input = accessorRemap[sampler.input];
      if(sampler.output >= 0 && sampler.output < static_cast<int>(accessorRemap.size()))
        sampler.output = accessorRemap[sampler.output];
    }
  }

  for(auto& image : model.images)
  {
    if(image.bufferView >= 0 && image.bufferView < static_cast<int>(bufferViewRemap.size()))
      image.bufferView = bufferViewRemap[image.bufferView];
  }
}

}  // anonymous namespace

//--------------------------------------------------------------------------------------------------
// compactModel - Removes orphaned geometry data (accessors, buffer views, buffers) from a glTF model.
//--------------------------------------------------------------------------------------------------
bool compactModel(tinygltf::Model& model)
{
  if(model.accessors.empty() && model.bufferViews.empty() && model.buffers.empty())
    return false;

  const size_t originalAccessorCount   = model.accessors.size();
  const size_t originalBufferViewCount = model.bufferViews.size();
  size_t       originalBufferSize      = 0;
  for(const auto& buffer : model.buffers)
    originalBufferSize += buffer.data.size();

  // Phase 1-3: Collect all referenced resources
  auto usedAccessors   = collectUsedAccessors(model);
  auto usedBufferViews = collectUsedBufferViews(model, usedAccessors);
  auto usedBuffers     = collectUsedBuffers(model, usedBufferViews);

  if(!isCompactionNeeded(model, usedAccessors, usedBufferViews, usedBuffers))
    return false;

  // Phase 4: Build remapping tables
  std::vector<int> accessorRemap(model.accessors.size(), -1);
  {
    int newIdx = 0;
    for(int oldIdx : usedAccessors)
      accessorRemap[oldIdx] = newIdx++;
  }

  std::vector<int> bufferViewRemap(model.bufferViews.size(), -1);
  {
    int newIdx = 0;
    for(int oldIdx : usedBufferViews)
      bufferViewRemap[oldIdx] = newIdx++;
  }

  // Phase 5: Create compact buffer
  auto [newBuffer, newOffsets] = buildCompactBuffer(model, usedBufferViews);

  // Phase 6: Create compact buffer views
  auto newBufferViews = buildCompactBufferViews(model, usedBufferViews, newOffsets);

  // Phase 7: Create compact accessors
  auto newAccessors = buildCompactAccessors(model, usedAccessors, bufferViewRemap);

  // Phase 8: Update all references
  updateModelReferences(model, accessorRemap, bufferViewRemap);

  // Phase 9: Replace model data
  const size_t newAccessorCount   = newAccessors.size();
  const size_t newBufferViewCount = newBufferViews.size();
  const size_t newBufferSize      = newBuffer.data.size();

  model.accessors   = std::move(newAccessors);
  model.bufferViews = std::move(newBufferViews);
  model.buffers.clear();
  model.buffers.push_back(std::move(newBuffer));

  LOGI("Accessors %zu -> %zu, BufferViews %zu -> %zu, Buffer %.2f KB -> %.2f KB (%.1f%% reduction)\n", originalAccessorCount,
       newAccessorCount, originalBufferViewCount, newBufferViewCount, originalBufferSize / 1024.0, newBufferSize / 1024.0,
       originalBufferSize > 0 ? (1.0 - double(newBufferSize) / double(originalBufferSize)) * 100.0 : 0.0);

  return true;
}
