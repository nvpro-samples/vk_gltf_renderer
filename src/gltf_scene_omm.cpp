/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Builds Vulkan opacity micromaps (VK_EXT_opacity_micromap) from the pre-baked
// EXT_mesh_opacity_micromap glTF extension. See gltf_scene_omm.hpp for design.
//

#include <algorithm>
#include <cstring>
#include <span>

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "gltf_scene_omm.hpp"
#include "tinygltf_utils.hpp"

namespace {
// Byte size of a glTF accessor component type.
uint32_t componentByteSize(int componentType)
{
  switch(componentType)
  {
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      return 1;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      return 2;
    case TINYGLTF_COMPONENT_TYPE_INT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
      return 4;
    default:
      return 0;
  }
}

// Map a glTF micromapIndices component type to the Vulkan index type used by the OMM linkage.
// Signed special indices (-1..-4) share the two's-complement bit pattern of their unsigned
// counterparts, so the unsigned index type reproduces them exactly.
VkIndexType toIndexType(int componentType)
{
  switch(componentType)
  {
    case TINYGLTF_COMPONENT_TYPE_BYTE:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      return VK_INDEX_TYPE_UINT8_EXT;
    case TINYGLTF_COMPONENT_TYPE_SHORT:
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      return VK_INDEX_TYPE_UINT16;
    default:
      return VK_INDEX_TYPE_UINT32;
  }
}
}  // namespace

//--------------------------------------------------------------------------------------------------
void nvvkgltf::SceneOmm::init(nvvk::ResourceAllocator* alloc)
{
  assert(!m_alloc);
  m_alloc  = alloc;
  m_device = alloc->getDevice();
}

//--------------------------------------------------------------------------------------------------
void nvvkgltf::SceneOmm::deinit()
{
  if(!m_alloc)
    return;
  destroy();
  m_alloc = nullptr;
}

//--------------------------------------------------------------------------------------------------
void nvvkgltf::SceneOmm::destroy()
{
  for(Micromap& mm : m_micromaps)
  {
    vkDestroyMicromapEXT(m_device, mm.micromap, nullptr);
    m_alloc->destroyBuffer(mm.storage);
    m_alloc->destroyBuffer(mm.data);
    m_alloc->destroyBuffer(mm.triangles);
    m_alloc->destroyBuffer(mm.scratch);
  }
  m_micromaps.clear();

  for(nvvk::Buffer& b : m_indexBuffers)
    m_alloc->destroyBuffer(b);
  m_indexBuffers.clear();

  m_primitives.clear();
}

//--------------------------------------------------------------------------------------------------
bool nvvkgltf::SceneOmm::has(uint32_t renderPrimID) const
{
  return renderPrimID < m_primitives.size() && m_primitives[renderPrimID].valid;
}

//--------------------------------------------------------------------------------------------------
const nvvkgltf::SceneOmm::PrimitiveOmm& nvvkgltf::SceneOmm::get(uint32_t renderPrimID) const
{
  assert(renderPrimID < m_primitives.size());
  return m_primitives[renderPrimID];
}

//--------------------------------------------------------------------------------------------------
// Parse EXT_mesh_opacity_micromap, upload build inputs, and build the micromaps on `cmd`.
void nvvkgltf::SceneOmm::create(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scene)
{
  destroy();

  if(!m_enabled)
    return;

  const tinygltf::Model& model = scene.getModel();

  // Root extension holds the array of micromap build inputs.
  const tinygltf::Value* rootExt = tinygltf::utils::findExtension(model.extensions, EXT_MESH_OPACITY_MICROMAP_EXTENSION_NAME);
  if(rootExt == nullptr || !rootExt->Has("micromaps"))
    return;

  nvutils::ScopedTimer st(__FUNCTION__);

  const std::vector<nvvkgltf::RenderPrimitive>& renderPrimitives = scene.getRenderPrimitives();
  m_primitives.resize(renderPrimitives.size());
  m_indexBuffers.resize(renderPrimitives.size());

  // Helper: true if a bufferView index (and the buffer it points at) is in range. Extension-provided
  // indices are untrusted, so callers must validate with this before dereferencing via bufferViewSpan.
  auto validBufferView = [&model](int bufferViewIndex) {
    return bufferViewIndex >= 0 && size_t(bufferViewIndex) < model.bufferViews.size()
           && model.bufferViews[bufferViewIndex].buffer >= 0
           && size_t(model.bufferViews[bufferViewIndex].buffer) < model.buffers.size();
  };

  // Helper: raw pointer + length of a bufferView. Caller must have checked validBufferView() first.
  auto bufferViewSpan = [&model](int bufferViewIndex) -> std::span<const uint8_t> {
    const tinygltf::BufferView& bv  = model.bufferViews[bufferViewIndex];
    const tinygltf::Buffer&     buf = model.buffers[bv.buffer];
    return std::span<const uint8_t>(buf.data.data() + bv.byteOffset, bv.byteLength);
  };

  // --- Build one VkMicromapEXT per root micromaps[] entry ---
  const tinygltf::Value& micromapsArray = rootExt->Get("micromaps");
  m_micromaps.resize(micromapsArray.ArrayLen());

  // The GPU builds are deferred until after the upload barrier, so the build inputs are resident
  // before vkCmdBuildMicromapsEXT reads them. Reserve so pUsageCounts pointers stay stable.
  struct PendingBuild
  {
    VkMicromapBuildInfoEXT          buildInfo;
    Micromap*                       micromap;
    uint32_t                        triangleStride;
    std::vector<VkMicromapUsageEXT> usages;
  };
  std::vector<PendingBuild> pendingBuilds;
  pendingBuilds.reserve(micromapsArray.ArrayLen());

  for(size_t i = 0; i < micromapsArray.ArrayLen(); i++)
  {
    const tinygltf::Value& mmDef = micromapsArray.Get(int(i));
    Micromap&              mm    = m_micromaps[i];

    // Required fields per the extension spec. Missing any of them would make Get<int>()/ArrayLen()
    // below silently return a garbage buffer-view index or an empty usage histogram, so skip the
    // entry (leaving m_micromaps[i] as a null handle) rather than build from bad data.
    if(!mmDef.Has("data") || !mmDef.Has("triangles") || !mmDef.Has("usageCounts") || !mmDef.Has("usageLevels")
       || !mmDef.Has("usageFormats"))
    {
      LOGW("EXT_mesh_opacity_micromap: micromaps[%zu] missing a required field (data/triangles/usageCounts/usageLevels/usageFormats) - skipping\n",
           i);
      continue;
    }

    const int dataView      = mmDef.Get("data").Get<int>();
    const int trianglesView = mmDef.Get("triangles").Get<int>();
    if(!validBufferView(dataView) || !validBufferView(trianglesView))
    {
      LOGW("EXT_mesh_opacity_micromap: micromaps[%zu] references an out-of-range data/triangles bufferView - skipping\n", i);
      continue;
    }

    // Usage histogram: parallel usageCounts / usageLevels / usageFormats arrays. They must be arrays
    // of equal length, otherwise the per-entry reads below would run off the end of a shorter one.
    const tinygltf::Value& counts  = mmDef.Get("usageCounts");
    const tinygltf::Value& levels  = mmDef.Get("usageLevels");
    const tinygltf::Value& formats = mmDef.Get("usageFormats");
    if(!counts.IsArray() || !levels.IsArray() || !formats.IsArray() || counts.ArrayLen() != levels.ArrayLen()
       || counts.ArrayLen() != formats.ArrayLen())
    {
      LOGW("EXT_mesh_opacity_micromap: micromaps[%zu] has misaligned usageCounts/usageLevels/usageFormats arrays - skipping\n", i);
      continue;
    }
    std::vector<VkMicromapUsageEXT> usages(counts.ArrayLen());
    for(size_t u = 0; u < usages.size(); u++)
    {
      usages[u].count            = uint32_t(counts.Get(int(u)).Get<int>());
      usages[u].subdivisionLevel = uint32_t(levels.Get(int(u)).Get<int>());
      usages[u].format           = uint32_t(formats.Get(int(u)).Get<int>());
    }

    // Upload the packed opacity bits and the per-triangle records (layout-compatible with
    // VkMicromapTriangleEXT). The triangles bufferView may specify a stride (>= 8, multiple of 4).
    const std::span<const uint8_t> dataBytes      = bufferViewSpan(dataView);
    const std::span<const uint8_t> trianglesBytes = bufferViewSpan(trianglesView);
    const uint32_t                 triangleStride = model.bufferViews[trianglesView].byteStride ?
                                                        uint32_t(model.bufferViews[trianglesView].byteStride) :
                                                        uint32_t(sizeof(VkMicromapTriangleEXT));

    // vkCmdBuildMicromapsEXT requires data.deviceAddress and triangleArray.deviceAddress to be
    // multiples of 256 (VUID-vkCmdBuildMicromapsEXT-pInfos-07515). Force the buffers' base device
    // address to that alignment; otherwise VMA can suballocate them at a smaller offset (the exact
    // offset depends on prior allocations, so this only surfaces after e.g. a scene merge).
    constexpr VkDeviceSize kMicromapInputAlignment = 256;
    NVVK_CHECK(m_alloc->createBuffer(mm.data, dataBytes.size_bytes(),
                                     VK_BUFFER_USAGE_2_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                     VMA_MEMORY_USAGE_AUTO, {}, kMicromapInputAlignment));
    NVVK_CHECK(staging.appendBuffer(mm.data, 0, dataBytes));
    NVVK_DBG_NAME(mm.data.buffer);

    NVVK_CHECK(m_alloc->createBuffer(mm.triangles, trianglesBytes.size_bytes(),
                                     VK_BUFFER_USAGE_2_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                     VMA_MEMORY_USAGE_AUTO, {}, kMicromapInputAlignment));
    NVVK_CHECK(staging.appendBuffer(mm.triangles, 0, trianglesBytes));
    NVVK_DBG_NAME(mm.triangles.buffer);

    // Query build sizes, then create the micromap storage + scratch buffers and the micromap object.
    VkMicromapBuildInfoEXT buildInfo{VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT};
    buildInfo.type             = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
    buildInfo.flags            = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT;
    buildInfo.mode             = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
    buildInfo.usageCountsCount = uint32_t(usages.size());
    buildInfo.pUsageCounts     = usages.data();

    VkMicromapBuildSizesInfoEXT sizeInfo{VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT};
    vkGetMicromapBuildSizesEXT(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &sizeInfo);
    assert(sizeInfo.micromapSize > 0);

    NVVK_CHECK(m_alloc->createBuffer(mm.storage, sizeInfo.micromapSize,
                                     VK_BUFFER_USAGE_2_MICROMAP_STORAGE_BIT_EXT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_DBG_NAME(mm.storage.buffer);

    const VkDeviceSize scratchSize = std::max(sizeInfo.buildScratchSize, VkDeviceSize(4));
    NVVK_CHECK(m_alloc->createBuffer(mm.scratch, scratchSize,
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                         | VK_BUFFER_USAGE_2_MICROMAP_STORAGE_BIT_EXT));
    NVVK_DBG_NAME(mm.scratch.buffer);

    VkMicromapCreateInfoEXT createInfo{VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT};
    createInfo.buffer = mm.storage.buffer;
    createInfo.size   = sizeInfo.micromapSize;
    createInfo.type   = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
    NVVK_CHECK(vkCreateMicromapEXT(m_device, &createInfo, nullptr, &mm.micromap));

    // Store the finalized build info to record after the upload barrier.
    pendingBuilds.push_back({buildInfo, &mm, triangleStride, std::move(usages)});
  }

  // --- Upload the per-primitive micromap index buffers and record the linkage ---
  for(uint32_t primID = 0; primID < renderPrimitives.size(); primID++)
  {
    const tinygltf::Primitive* prim = renderPrimitives[primID].pPrimitive;
    if(prim == nullptr)
      continue;

    const tinygltf::Value* primExt = tinygltf::utils::findExtension(prim->extensions, EXT_MESH_OPACITY_MICROMAP_EXTENSION_NAME);
    if(primExt == nullptr || !primExt->Has("micromap"))
      continue;

    const int micromapIdx = primExt->Get("micromap").Get<int>();
    if(micromapIdx < 0 || micromapIdx >= int(m_micromaps.size()))
      continue;
    // The referenced entry may have been skipped above (malformed build input), leaving a null
    // handle. Linking that into the BLAS would be invalid, so skip the primitive too.
    if(m_micromaps[micromapIdx].micromap == VK_NULL_HANDLE)
      continue;

    // Optional; defaults to 0. Guard against a negative value: uint32_t(negative) wraps to a huge
    // baseTriangle that would trip a validation VUID or mislink the BLAS.
    int baseTriangle = primExt->Has("micromapBaseTriangle") ? primExt->Get("micromapBaseTriangle").Get<int>() : 0;
    if(baseTriangle < 0)
    {
      LOGW("EXT_mesh_opacity_micromap: primitive %u has a negative micromapBaseTriangle - skipping\n", primID);
      continue;
    }

    PrimitiveOmm& out = m_primitives[primID];
    out.micromap      = m_micromaps[micromapIdx].micromap;
    out.baseTriangle  = uint32_t(baseTriangle);

    // Per-triangle lookup values. When absent, the lookup is the implicit identity (index == triangle).
    if(primExt->Has("micromapIndices"))
    {
      const int accessorIdx = primExt->Get("micromapIndices").Get<int>();
      if(accessorIdx < 0 || size_t(accessorIdx) >= model.accessors.size())
      {
        LOGW("EXT_mesh_opacity_micromap: primitive %u references an out-of-range micromapIndices accessor - skipping\n", primID);
        continue;
      }
      const tinygltf::Accessor& accessor = model.accessors[accessorIdx];
      const uint32_t            compSize = componentByteSize(accessor.componentType);
      if(compSize == 0)
      {
        LOGW("EXT_mesh_opacity_micromap: primitive %u has unsupported micromapIndices component type %d - skipping\n",
             primID, accessor.componentType);
        continue;
      }
      if(!validBufferView(accessor.bufferView))
      {
        LOGW("EXT_mesh_opacity_micromap: primitive %u micromapIndices accessor references an out-of-range bufferView - skipping\n",
             primID);
        continue;
      }

      const tinygltf::BufferView& bv        = model.bufferViews[accessor.bufferView];
      const tinygltf::Buffer&     buf       = model.buffers[bv.buffer];
      const size_t                srcStride = bv.byteStride ? size_t(bv.byteStride) : size_t(compSize);
      const size_t                srcBegin  = size_t(bv.byteOffset) + size_t(accessor.byteOffset);
      // Ensure the strided read of `accessor.count` elements stays inside the source buffer.
      if(accessor.count > 0 && srcBegin + (size_t(accessor.count) - 1) * srcStride + compSize > buf.data.size())
      {
        LOGW("EXT_mesh_opacity_micromap: primitive %u micromapIndices read exceeds its buffer - skipping\n", primID);
        continue;
      }
      const uint8_t* base = buf.data.data() + srcBegin;

      // Repack tightly (de-interleave if the accessor is strided).
      std::vector<uint8_t> tight(size_t(accessor.count) * compSize);
      for(size_t e = 0; e < accessor.count; e++)
        std::memcpy(&tight[e * compSize], base + e * srcStride, compSize);

      nvvk::Buffer& indexBuffer = m_indexBuffers[primID];
      NVVK_CHECK(m_alloc->createBuffer(indexBuffer, std::span(tight).size_bytes(),
                                       VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                           | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
      NVVK_CHECK(staging.appendBuffer(indexBuffer, 0, std::span(tight)));
      NVVK_DBG_NAME(indexBuffer.buffer);

      out.indexAddress = indexBuffer.address;
      out.indexType    = toIndexType(accessor.componentType);
      out.indexStride  = compSize;
    }

    out.valid = true;
  }

  // --- Flush uploads, then record the micromap builds (build inputs must be resident first) ---
  staging.cmdUploadAppended(cmd);

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_MICROMAP_READ_BIT_EXT);

  for(PendingBuild& pb : pendingBuilds)
  {
    pb.buildInfo.pUsageCounts                = pb.usages.data();  // Re-point after the move into the vector
    pb.buildInfo.dstMicromap                 = pb.micromap->micromap;
    pb.buildInfo.data.deviceAddress          = pb.micromap->data.address;
    pb.buildInfo.triangleArray.deviceAddress = pb.micromap->triangles.address;
    pb.buildInfo.triangleArrayStride         = pb.triangleStride;
    pb.buildInfo.scratchData.deviceAddress   = pb.micromap->scratch.address;
    vkCmdBuildMicromapsEXT(cmd, 1, &pb.buildInfo);
  }

  // Micromap build results and the transfer-written index buffers are consumed by the BLAS build
  // (recorded on a later command buffer on the same queue, so this barrier synchronizes it).
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_MICROMAP_READ_BIT_EXT);
}
