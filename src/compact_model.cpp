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

#include "compact_model.hpp"

#include <set>
#include <vector>

#include "tinygltf_utils.hpp"
#include "nvutils/logger.hpp"

//-------------------------------------------------------------------------------------------------
// compactModel - Removes orphaned data from a glTF model
//-------------------------------------------------------------------------------------------------

// Extension name for Draco compression
#define KHR_DRACO_MESH_COMPRESSION_EXTENSION_NAME "KHR_draco_mesh_compression"

namespace {

// Align offset to 4-byte boundary (glTF requirement for accessor data)
inline size_t align4(size_t offset)
{
  return (offset + 3) & ~size_t(3);
}

// Check if a primitive uses Draco compression
inline bool isDracoCompressed(const tinygltf::Primitive& primitive)
{
  return tinygltf::utils::hasElementName(primitive.extensions, KHR_DRACO_MESH_COMPRESSION_EXTENSION_NAME);
}

// Collect all accessor indices referenced by a primitive
void collectPrimitiveAccessors(const tinygltf::Primitive& primitive, std::set<int>& usedAccessors)
{
  // Skip Draco-compressed primitives (accessors are placeholders, data is in extension)
  if(isDracoCompressed(primitive))
  {
    return;
  }

  // Primitive attributes (POSITION, NORMAL, TANGENT, TEXCOORD_*, COLOR_*, WEIGHTS_*, JOINTS_*)
  for(const auto& [name, accessorIdx] : primitive.attributes)
  {
    if(accessorIdx >= 0)
    {
      usedAccessors.insert(accessorIdx);
    }
  }

  // Primitive indices
  if(primitive.indices >= 0)
  {
    usedAccessors.insert(primitive.indices);
  }

  // Morph targets
  for(const auto& target : primitive.targets)
  {
    for(const auto& [name, accessorIdx] : target)
    {
      if(accessorIdx >= 0)
      {
        usedAccessors.insert(accessorIdx);
      }
    }
  }
}

}  // anonymous namespace

//--------------------------------------------------------------------------------------------------
// This compact the glTF buffer, removing unused references
//
bool compactModel(tinygltf::Model& model)
{
  // Early out if model has no data to compact
  if(model.accessors.empty() && model.bufferViews.empty() && model.buffers.empty())
  {
    return false;
  }

  // Capture original sizes for logging
  const size_t originalAccessorCount   = model.accessors.size();
  const size_t originalBufferViewCount = model.bufferViews.size();
  size_t       originalBufferSize      = 0;
  for(const auto& buffer : model.buffers)
  {
    originalBufferSize += buffer.data.size();
  }

  //-------------------------------------------------------------------------
  // Phase 1: Collect all referenced accessors
  //-------------------------------------------------------------------------
  std::set<int> usedAccessors;

  // 1. From mesh primitives
  for(const auto& mesh : model.meshes)
  {
    for(const auto& primitive : mesh.primitives)
    {
      collectPrimitiveAccessors(primitive, usedAccessors);
    }
  }

  // 2. From skins (inverseBindMatrices)
  for(const auto& skin : model.skins)
  {
    if(skin.inverseBindMatrices >= 0)
    {
      usedAccessors.insert(skin.inverseBindMatrices);
    }
  }

  // 3. From animations (sampler inputs/outputs)
  for(const auto& animation : model.animations)
  {
    for(const auto& sampler : animation.samplers)
    {
      if(sampler.input >= 0)
      {
        usedAccessors.insert(sampler.input);
      }
      if(sampler.output >= 0)
      {
        usedAccessors.insert(sampler.output);
      }
    }
  }

  //-------------------------------------------------------------------------
  // Phase 2: Collect all referenced buffer views
  //-------------------------------------------------------------------------
  std::set<int> usedBufferViews;

  // From used accessors
  for(int accIdx : usedAccessors)
  {
    if(accIdx < 0 || accIdx >= static_cast<int>(model.accessors.size()))
    {
      continue;
    }

    const auto& accessor = model.accessors[accIdx];

    // Main bufferView
    if(accessor.bufferView >= 0)
    {
      usedBufferViews.insert(accessor.bufferView);
    }

    // Sparse accessor bufferViews
    if(accessor.sparse.isSparse)
    {
      if(accessor.sparse.indices.bufferView >= 0)
      {
        usedBufferViews.insert(accessor.sparse.indices.bufferView);
      }
      if(accessor.sparse.values.bufferView >= 0)
      {
        usedBufferViews.insert(accessor.sparse.values.bufferView);
      }
    }
  }

  // From images (embedded images with bufferView)
  for(const auto& image : model.images)
  {
    if(image.bufferView >= 0)
    {
      usedBufferViews.insert(image.bufferView);
    }
  }

  //-------------------------------------------------------------------------
  // Phase 3: Collect all referenced buffers and handle EXT_meshopt_compression
  //-------------------------------------------------------------------------
  std::set<int> usedBuffers;

  // From used buffer views
  for(int bvIdx : usedBufferViews)
  {
    if(bvIdx < 0 || bvIdx >= static_cast<int>(model.bufferViews.size()))
    {
      continue;
    }

    const auto& bv = model.bufferViews[bvIdx];
    if(bv.buffer >= 0)
    {
      usedBuffers.insert(bv.buffer);
    }

    // Also check EXT_meshopt_compression extension for additional buffer references
    EXT_meshopt_compression mcomp;
    if(tinygltf::utils::getMeshoptCompression(bv, mcomp) && mcomp.buffer >= 0)
    {
      usedBuffers.insert(mcomp.buffer);
    }
  }

  //-------------------------------------------------------------------------
  // Check if compaction is needed
  //-------------------------------------------------------------------------
  bool needsCompaction = (usedAccessors.size() < model.accessors.size()) || (usedBufferViews.size() < model.bufferViews.size())
                         || (usedBuffers.size() < model.buffers.size());

  // Also check if buffer data can be reduced
  if(!needsCompaction)
  {
    // Calculate total used buffer data size
    size_t usedDataSize = 0;
    for(int bvIdx : usedBufferViews)
    {
      usedDataSize += model.bufferViews[bvIdx].byteLength;
    }

    // Calculate total current buffer size
    size_t totalBufferSize = 0;
    for(const auto& buffer : model.buffers)
    {
      totalBufferSize += buffer.data.size();
    }

    // Account for alignment overhead (rough estimate)
    size_t alignmentOverhead = usedBufferViews.size() * 4;
    needsCompaction          = (usedDataSize + alignmentOverhead) < totalBufferSize * 0.95;  // 5% threshold
  }

  if(!needsCompaction)
  {
    return false;  // Model is already compact
  }

  //-------------------------------------------------------------------------
  // Phase 4: Build remapping tables
  //-------------------------------------------------------------------------
  std::vector<int> accessorRemap(model.accessors.size(), -1);
  std::vector<int> bufferViewRemap(model.bufferViews.size(), -1);

  // Create accessor remapping (old index -> new index)
  {
    int newIdx = 0;
    for(int oldIdx : usedAccessors)
    {
      accessorRemap[oldIdx] = newIdx++;
    }
  }

  // Create buffer view remapping (old index -> new index)
  {
    int newIdx = 0;
    for(int oldIdx : usedBufferViews)
    {
      bufferViewRemap[oldIdx] = newIdx++;
    }
  }

  //-------------------------------------------------------------------------
  // Phase 5: Create new compact buffer with all used data
  //-------------------------------------------------------------------------
  tinygltf::Buffer newBuffer;
  newBuffer.name = model.buffers.empty() ? "" : model.buffers[0].name;

  // Track new byte offsets for each buffer view
  std::vector<size_t> newBufferViewOffsets;
  newBufferViewOffsets.reserve(usedBufferViews.size());

  for(int oldBvIdx : usedBufferViews)
  {
    const auto& oldBv = model.bufferViews[oldBvIdx];

    // Validate buffer index
    if(oldBv.buffer < 0 || oldBv.buffer >= static_cast<int>(model.buffers.size()))
    {
      // Invalid buffer reference, skip (shouldn't happen in valid glTF)
      newBufferViewOffsets.push_back(newBuffer.data.size());
      continue;
    }

    const auto& oldBuffer = model.buffers[oldBv.buffer];

    // Validate byte range
    if(oldBv.byteOffset + oldBv.byteLength > oldBuffer.data.size())
    {
      // Invalid byte range, skip
      newBufferViewOffsets.push_back(newBuffer.data.size());
      continue;
    }

    // Align to 4 bytes (glTF requirement for accessor data)
    size_t alignedOffset = align4(newBuffer.data.size());

    // Pad with zeros if needed for alignment
    if(alignedOffset > newBuffer.data.size())
    {
      newBuffer.data.resize(alignedOffset, 0);
    }

    // Record the new offset for this buffer view
    newBufferViewOffsets.push_back(alignedOffset);

    // Copy the data
    const unsigned char* srcData = oldBuffer.data.data() + oldBv.byteOffset;
    newBuffer.data.insert(newBuffer.data.end(), srcData, srcData + oldBv.byteLength);
  }

  newBuffer.byteLength = newBuffer.data.size();

  //-------------------------------------------------------------------------
  // Phase 6: Create new buffer views with updated offsets
  //-------------------------------------------------------------------------
  std::vector<tinygltf::BufferView> newBufferViews;
  newBufferViews.reserve(usedBufferViews.size());

  size_t bvIndex = 0;
  for(int oldBvIdx : usedBufferViews)
  {
    tinygltf::BufferView newBv = model.bufferViews[oldBvIdx];  // Copy all metadata

    // Update to point to the merged buffer
    newBv.buffer     = 0;
    newBv.byteOffset = newBufferViewOffsets[bvIndex++];
    // byteLength, byteStride, target remain the same

    // Handle EXT_meshopt_compression extension - update buffer reference if present
    if(tinygltf::utils::hasElementName(newBv.extensions, EXT_MESHOPT_COMPRESSION_EXTENSION_NAME))
    {
      // The compressed data buffer is also merged into buffer 0
      // Update the buffer index in the extension
      tinygltf::Value& ext = newBv.extensions[EXT_MESHOPT_COMPRESSION_EXTENSION_NAME];
      if(ext.Has("buffer"))
      {
        ext.Get<tinygltf::Value::Object>()["buffer"] = tinygltf::Value(0);
      }
    }

    newBufferViews.push_back(std::move(newBv));
  }

  //-------------------------------------------------------------------------
  // Phase 7: Create new accessors with remapped buffer view indices
  //-------------------------------------------------------------------------
  std::vector<tinygltf::Accessor> newAccessors;
  newAccessors.reserve(usedAccessors.size());

  for(int oldAccIdx : usedAccessors)
  {
    tinygltf::Accessor newAcc = model.accessors[oldAccIdx];  // Copy all metadata

    // Remap main bufferView
    if(newAcc.bufferView >= 0)
    {
      newAcc.bufferView = bufferViewRemap[newAcc.bufferView];
    }

    // Remap sparse accessor bufferViews
    if(newAcc.sparse.isSparse)
    {
      if(newAcc.sparse.indices.bufferView >= 0)
      {
        newAcc.sparse.indices.bufferView = bufferViewRemap[newAcc.sparse.indices.bufferView];
      }
      if(newAcc.sparse.values.bufferView >= 0)
      {
        newAcc.sparse.values.bufferView = bufferViewRemap[newAcc.sparse.values.bufferView];
      }
    }

    newAccessors.push_back(std::move(newAcc));
  }

  //-------------------------------------------------------------------------
  // Phase 8: Update all references in the model
  //-------------------------------------------------------------------------

  // Update mesh primitives
  for(auto& mesh : model.meshes)
  {
    for(auto& primitive : mesh.primitives)
    {
      // Skip Draco-compressed primitives
      if(isDracoCompressed(primitive))
      {
        continue;
      }

      // Update attributes
      for(auto& [name, accessorIdx] : primitive.attributes)
      {
        if(accessorIdx >= 0 && accessorIdx < static_cast<int>(accessorRemap.size()))
        {
          accessorIdx = accessorRemap[accessorIdx];
        }
      }

      // Update indices
      if(primitive.indices >= 0 && primitive.indices < static_cast<int>(accessorRemap.size()))
      {
        primitive.indices = accessorRemap[primitive.indices];
      }

      // Update morph targets
      for(auto& target : primitive.targets)
      {
        for(auto& [name, accessorIdx] : target)
        {
          if(accessorIdx >= 0 && accessorIdx < static_cast<int>(accessorRemap.size()))
          {
            accessorIdx = accessorRemap[accessorIdx];
          }
        }
      }
    }
  }

  // Update skins
  for(auto& skin : model.skins)
  {
    if(skin.inverseBindMatrices >= 0 && skin.inverseBindMatrices < static_cast<int>(accessorRemap.size()))
    {
      skin.inverseBindMatrices = accessorRemap[skin.inverseBindMatrices];
    }
  }

  // Update animations
  for(auto& animation : model.animations)
  {
    for(auto& sampler : animation.samplers)
    {
      if(sampler.input >= 0 && sampler.input < static_cast<int>(accessorRemap.size()))
      {
        sampler.input = accessorRemap[sampler.input];
      }
      if(sampler.output >= 0 && sampler.output < static_cast<int>(accessorRemap.size()))
      {
        sampler.output = accessorRemap[sampler.output];
      }
    }
  }

  // Update images
  for(auto& image : model.images)
  {
    if(image.bufferView >= 0 && image.bufferView < static_cast<int>(bufferViewRemap.size()))
    {
      image.bufferView = bufferViewRemap[image.bufferView];
    }
  }

  //-------------------------------------------------------------------------
  // Phase 9: Replace model data with compacted versions
  //-------------------------------------------------------------------------
  const size_t newAccessorCount   = newAccessors.size();
  const size_t newBufferViewCount = newBufferViews.size();
  const size_t newBufferSize      = newBuffer.data.size();

  model.accessors   = std::move(newAccessors);
  model.bufferViews = std::move(newBufferViews);
  model.buffers.clear();
  model.buffers.push_back(std::move(newBuffer));

  // Log compaction results
  LOGI("Accessors %zu -> %zu, BufferViews %zu -> %zu, Buffer %.2f KB -> %.2f KB (%.1f%% reduction)\n", originalAccessorCount,
       newAccessorCount, originalBufferViewCount, newBufferViewCount, originalBufferSize / 1024.0, newBufferSize / 1024.0,
       originalBufferSize > 0 ? (1.0 - double(newBufferSize) / double(originalBufferSize)) * 100.0 : 0.0);

  return true;  // Data was compacted
}
