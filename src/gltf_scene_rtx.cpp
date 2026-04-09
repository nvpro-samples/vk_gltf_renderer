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
// Builds and manages Vulkan ray tracing acceleration structures (BLAS/TLAS)
// for glTF scenes. Creates bottom-level structures per mesh primitive,
// compacts them for optimal memory use, and assembles top-level structures
// from the flattened node instances. Supports animated geometry rebuilds.
//

#include <cinttypes>
#include <numeric>

#include <nvutils/alignment.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "gltf_scene_rtx.hpp"
#include "gltf_scene_vk.hpp"
#include "gltf_scene_animation.hpp"
#include "tinygltf_utils.hpp"

// GPU memory category names for RTX resources
namespace {
constexpr std::string_view kMemCategoryBLAS      = "BLAS";
constexpr std::string_view kMemCategoryTLAS      = "TLAS";
constexpr std::string_view kMemCategoryScratch   = "Scratch";
constexpr std::string_view kMemCategoryInstances = "Instances";
}  // namespace

//--------------------------------------------------------------------------------------------------
// SceneRtx implementation grouped by work area:
//   1. Init / Deinit  - allocator setup and teardown
//   2. Create        - BLAS/TLAS build (create, createBottomLevel, cmdBuild, cmdCreateBuildTLAS, compact, trackBlasMemory, tlas)
//   3. Sync / Update  - TLAS/BLAS updates (syncTopLevelAS, rebuildTopLevelAS, updateBottomLevelAS)
//   4. Destroy        - release acceleration structures and scratch (destroy, destroyScratchBuffers, destroyTlasResources)
//--------------------------------------------------------------------------------------------------

//========== Init / Deinit ==========

//--------------------------------------------------------------------------------------------------
// Initialize ray tracing: allocator and RT properties. Must be called before create().
void nvvkgltf::SceneRtx::init(nvvk::ResourceAllocator* alloc)
{
  assert(!m_alloc);

  m_device         = alloc->getDevice();
  m_physicalDevice = alloc->getPhysicalDevice();
  m_alloc          = alloc;
  m_memoryTracker.init(alloc);

  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  m_rtProperties.pNext = &m_rtASProperties;
  prop2.pNext          = &m_rtProperties;
  vkGetPhysicalDeviceProperties2(m_physicalDevice, &prop2);
}

//--------------------------------------------------------------------------------------------------
// Tear down allocator reference and release all resources. Idempotent if already deinit.
void nvvkgltf::SceneRtx::deinit()
{
  if(!m_alloc)
  {
    return;
  }

  destroy();

  m_alloc = nullptr;
}

//========== Create ==========

//--------------------------------------------------------------------------------------------------
// Create BLAS (and optionally build in a loop) then TLAS. Calls destroy() first.
void nvvkgltf::SceneRtx::create(VkCommandBuffer                      cmd,
                                nvvk::StagingUploader&               staging,
                                const nvvkgltf::Scene&               scn,
                                const SceneVk&                       scnVk,
                                VkBuildAccelerationStructureFlagsKHR flags)
{
  // Create the bottom-level acceleration structure
  createBottomLevelAccelerationStructure(scn, scnVk, flags);

  bool finished = false;
  do
  {
    // This won't compact the BLAS, but will create the acceleration structure
    finished = cmdBuildBottomLevelAccelerationStructure(cmd, 512'000'000);
  } while(!finished);

  // Track all BLAS allocations after they're all built
  trackBlasMemory();

  cmdCreateBuildTopLevelAccelerationStructure(cmd, staging, scn);
}

//--------------------------------------------------------------------------------------------------
// Return the top-level acceleration structure handle for binding in ray tracing pipelines.
VkAccelerationStructureKHR nvvkgltf::SceneRtx::topLevelAS()
{
  return m_tlasAccel.accel;
}

//--------------------------------------------------------------------------------------------------
// Register all BLAS allocations with the memory tracker. Call after all BLAS are built.
void nvvkgltf::SceneRtx::trackBlasMemory()
{
  for(const auto& blas : m_blasAccel)
  {
    if(blas.accel != VK_NULL_HANDLE && blas.buffer.allocation)
    {
      m_memoryTracker.track(kMemCategoryBLAS, blas.buffer.allocation);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Convert a RenderPrimitive to AccelerationStructureGeometryInfo for BLAS build.
nvvk::AccelerationStructureGeometryInfo nvvkgltf::SceneRtx::renderPrimitiveToAsGeometry(const nvvkgltf::RenderPrimitive& prim,
                                                                                        VkDeviceAddress vertexAddress,
                                                                                        VkDeviceAddress indexAddress)
{
  nvvk::AccelerationStructureGeometryInfo result;
  uint32_t                                numTriangles = prim.indexCount / 3;

  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
  triangles.vertexData.deviceAddress = vertexAddress;
  triangles.vertexStride             = sizeof(glm::vec3);
  triangles.indexType                = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress  = indexAddress;
  triangles.maxVertex                = prim.vertexCount - 1;
  //triangles.transformData; // Identity

  // Identify the above data as containing opaque triangles.
  result.geometry.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  result.geometry.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  result.geometry.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
  result.geometry.geometry.triangles = triangles;

  result.rangeInfo.firstVertex     = 0;
  result.rangeInfo.primitiveCount  = numTriangles;
  result.rangeInfo.primitiveOffset = 0;
  result.rangeInfo.transformOffset = 0;

  return result;
}

//--------------------------------------------------------------------------------------------------
// Build BLAS build data for all primitives and init BLAS builder. Does not run GPU build (use cmdBuildBottomLevelAccelerationStructure).
void nvvkgltf::SceneRtx::createBottomLevelAccelerationStructure(const nvvkgltf::Scene&               scene,
                                                                const SceneVk&                       sceneVk,
                                                                VkBuildAccelerationStructureFlagsKHR flags)
{
  nvutils::ScopedTimer st(__FUNCTION__);

  destroy();  // Make sure not to leave allocated buffers

  auto& renderPrimitives = scene.getRenderPrimitives();

  // BLAS - Storing each primitive in a geometry
  m_blasBuildData.resize(renderPrimitives.size());
  m_blasAccel.resize(m_blasBuildData.size());

  // Retrieve the array of primitive buffers (see in SceneVk)
  const auto& vertexBuffers = sceneVk.vertexBuffers();
  const auto& indices       = sceneVk.indices();

  for(uint32_t p_idx = 0; p_idx < renderPrimitives.size(); p_idx++)
  {
    auto& blasData  = m_blasBuildData[p_idx];
    blasData.asType = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    // Converting each primitive into the required format for the BLAS
    VkDeviceAddress vertexAddress = vertexBuffers[p_idx].position.address;
    VkDeviceAddress indexAddress  = indices[p_idx].address;
    // Fill the BLAS information
    auto geo = renderPrimitiveToAsGeometry(renderPrimitives[p_idx], vertexAddress, indexAddress);
    blasData.addGeometry(geo);
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = blasData.finalizeGeometry(m_device, flags);  // Will query the size of the resulting BLAS
  }

  // Create the bottom-level acceleration structure Builder and query pool for compaction
  m_blasBuilder = std::make_unique<nvvk::AccelerationStructureBuilder>();
  m_blasBuilder->init(m_alloc);
}

//--------------------------------------------------------------------------------------------------
// Record BLAS build on GPU. Returns true when all BLAS are built; false if budget exceeded (call again).
bool nvvkgltf::SceneRtx::cmdBuildBottomLevelAccelerationStructure(VkCommandBuffer cmd, VkDeviceSize hintMaxBudget /*= 512'000'000*/)
{
  nvutils::ScopedTimer st(__FUNCTION__);
  assert(m_blasBuilder);

  destroyScratchBuffers();

  // 1) finding the largest scratch size
  VkDeviceSize scratchSize = m_blasBuilder->getScratchSize(hintMaxBudget, m_blasBuildData);

  // 2) allocating the scratch buffer
  NVVK_CHECK(m_alloc->createBuffer(m_blasScratchBuffer, scratchSize,
                                   VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT
                                       | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                   VMA_MEMORY_USAGE_AUTO, {}, m_blasBuilder->getScratchAlignment()));
  NVVK_DBG_NAME(m_blasScratchBuffer.buffer);
  m_memoryTracker.track(kMemCategoryScratch, m_blasScratchBuffer.allocation);

  std::span<nvvk::AccelerationStructureBuildData> blasBuildData(m_blasBuildData);
  std::span<nvvk::AccelerationStructure>          blasAccel(m_blasAccel);

  // Building all BLAS in parallel, as long as there are enough budget
  VkResult result = m_blasBuilder->cmdCreateBlas(cmd, blasBuildData, blasAccel, m_blasScratchBuffer.address,
                                                 m_blasScratchBuffer.bufferSize, hintMaxBudget);
  if(result != VK_SUCCESS && result != VK_INCOMPLETE)
  {
    nvvk::CheckError::getInstance().check(result, "m_blasBuilder->cmdCreateBlas", __FILE__, __LINE__);
    // must return true, otherwise calling function can do infinite loop
    // although above should actually terminate the app
    return true;
  }
  return result == VK_SUCCESS;
}

// Instance flags for TLAS (opaque / double-sided) from material.
static VkGeometryInstanceFlagsKHR getInstanceFlag(const tinygltf::Material& mat)
{
  VkGeometryInstanceFlagsKHR         instanceFlags{};
  KHR_materials_transmission         transmission        = tinygltf::utils::getTransmission(mat);
  KHR_materials_volume               volume              = tinygltf::utils::getVolume(mat);
  KHR_materials_diffuse_transmission diffuseTransmission = tinygltf::utils::getDiffuseTransmission(mat);

  // Check if the material is opaque, if so, we can skip the anyhit
  bool isOpaque = ((transmission.factor == 0.0f) && (mat.alphaMode == "OPAQUE")
                   && (diffuseTransmission.diffuseTransmissionFactor == 0.0F));

  // Always opaque, no need to use ANYHIT (faster)
  if(isOpaque)
  {
    instanceFlags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
  }

  // Need to skip the cull flag in traceray_rtx for double sided materials
  if(mat.doubleSided == 1 || volume.thicknessFactor > 0.0F || transmission.factor > 0.0F)
  {
    instanceFlags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  }

  return instanceFlags;
}

//--------------------------------------------------------------------------------------------------
// Build TLAS from scene render nodes.
void nvvkgltf::SceneRtx::cmdCreateBuildTopLevelAccelerationStructure(VkCommandBuffer        cmd,
                                                                     nvvk::StagingUploader& staging,
                                                                     const nvvkgltf::Scene& scene)
{
  nvutils::ScopedTimer st(__FUNCTION__);
  const auto&          drawObjects = scene.getRenderNodes();
  const auto&          materials   = scene.getModel().materials;

  const uint32_t instanceCount = static_cast<uint32_t>(drawObjects.size());

  // Rebuild the full instance flags cache (initial build or size-mismatch rebuild).
  m_instanceFlagsCache.resize(materials.size());
  for(size_t i = 0; i < materials.size(); i++)
    m_instanceFlagsCache[i] = getInstanceFlag(materials[i]);

  m_tlasInstances.clear();
  m_tlasInstances.reserve(instanceCount);
  m_numVisibleElement = 0;
  for(const auto& object : drawObjects)
  {
    VkDeviceAddress blasAddress = m_blasAccel[object.renderPrimID].address;
    if(!object.visible)
      blasAddress = 0;

    m_numVisibleElement += object.visible ? 1 : 0;

    VkAccelerationStructureInstanceKHR asInstance{};
    asInstance.transform                              = nvvk::toTransformMatrixKHR(object.worldMatrix);
    asInstance.instanceCustomIndex                    = object.renderPrimID;
    asInstance.accelerationStructureReference         = blasAddress;
    asInstance.instanceShaderBindingTableRecordOffset = 0;
    asInstance.mask                                   = 0x01;
    asInstance.flags                                  = m_instanceFlagsCache[object.materialID];

    m_tlasInstances.push_back(asInstance);
  }

  // Vulkan requires at least one instance for TLAS builds. Use a local invisible dummy
  // so m_tlasInstances always reflects the true scene state (empty when no render nodes).
  VkAccelerationStructureInstanceKHR                  dummyInstance{};
  std::span<const VkAccelerationStructureInstanceKHR> buildInstances = m_tlasInstances;
  if(m_tlasInstances.empty())
  {
    dummyInstance.transform = nvvk::toTransformMatrixKHR(glm::mat4(1.0f));
    dummyInstance.mask      = 0x00;
    buildInstances          = {&dummyInstance, 1};
  }

  VkBuildAccelerationStructureFlagsKHR buildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  buildFlags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

  constexpr VmaAllocationCreateFlags instanceAllocFlags   = 0;
  constexpr VkDeviceSize             instanceMinAlignment = 16;
  NVVK_CHECK(m_alloc->createBuffer(m_instancesBuffer, buildInstances.size_bytes(),
                                   VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                       | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                   VMA_MEMORY_USAGE_AUTO, instanceAllocFlags, instanceMinAlignment));
  NVVK_CHECK(staging.appendBuffer(m_instancesBuffer, 0, buildInstances));
  NVVK_DBG_NAME(m_instancesBuffer.buffer);
  m_memoryTracker.track(kMemCategoryInstances, m_instancesBuffer.allocation);

  m_tlasBuildData.asType = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  auto geo               = m_tlasBuildData.makeInstanceGeometry(buildInstances.size(), m_instancesBuffer.address);
  m_tlasBuildData.addGeometry(geo);

  staging.cmdUploadAppended(cmd);

  nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT);

  VkAccelerationStructureBuildSizesInfoKHR sizeInfo = m_tlasBuildData.finalizeGeometry(m_device, buildFlags);

  NVVK_CHECK(m_alloc->createBuffer(m_tlasScratchBuffer, sizeInfo.buildScratchSize,
                                   VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT
                                       | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR));
  NVVK_DBG_NAME(m_tlasScratchBuffer.buffer);
  m_memoryTracker.track(kMemCategoryScratch, m_tlasScratchBuffer.allocation);

  VkAccelerationStructureCreateInfoKHR createInfo = m_tlasBuildData.makeCreateInfo();
  NVVK_CHECK(m_alloc->createAcceleration(m_tlasAccel, createInfo));
  NVVK_DBG_NAME(m_tlasAccel.accel);
  m_memoryTracker.track(kMemCategoryTLAS, m_tlasAccel.buffer.allocation);

  m_tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlasAccel.accel, m_tlasScratchBuffer.address);

  nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}

//========== Sync / Update ==========

//--------------------------------------------------------------------------------------------------
// Update the per-material instance flags cache. Uses df.materials for surgical updates when only
// a few materials changed; falls back to full rebuild when the material count changes.
// Must be called before syncFromScene() clears df.materials.
void nvvkgltf::SceneRtx::updateInstanceFlagsCache(const nvvkgltf::Scene& scene)
{
  const auto& materials = scene.getModel().materials;
  const auto& dirtyMats = scene.getDirtyFlags().materials;

  if(m_instanceFlagsCache.size() != materials.size())
  {
    m_instanceFlagsCache.resize(materials.size());
    for(size_t i = 0; i < materials.size(); i++)
      m_instanceFlagsCache[i] = getInstanceFlag(materials[i]);
  }
  else if(!dirtyMats.empty())
  {
    for(int idx : dirtyMats)
    {
      if(idx >= 0 && idx < static_cast<int>(materials.size()))
        m_instanceFlagsCache[idx] = getInstanceFlag(materials[idx]);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Rebuild or update TLAS. dirtyRenderNodes empty = full rebuild from scratch.
void nvvkgltf::SceneRtx::rebuildTopLevelAS(VkCommandBuffer                cmd,
                                           nvvk::StagingUploader&         staging,
                                           const nvvkgltf::Scene&         scene,
                                           const std::unordered_set<int>& dirtyRenderNodes)
{
  const auto& drawObjects = scene.getRenderNodes();

  int32_t numVisibleElement = dirtyRenderNodes.empty() ? 0 : m_numVisibleElement;

  // If the number of render nodes changed, we need to recreate the TLAS from scratch,
  // as well as the instance buffer.
  // This is because the TLAS build requires a contiguous array of instances, and we don't
  // want to manage holes in that array from removed nodes.
  if(m_tlasInstances.size() != drawObjects.size())
  {
    destroyTlasResourcesDeferred();
    cmdCreateBuildTopLevelAccelerationStructure(cmd, staging, scene);
    return;
  }

  // Lambda to update a single instance in the TLAS instance array and return its previous and current visibility.
  auto updateInstance = [&](int idx) {
    const auto&     object      = drawObjects[idx];
    VkDeviceAddress blasAddress = m_blasAccel[object.renderPrimID].address;
    const bool      isVisible   = object.visible && (blasAddress != 0);
    const bool      wasVisible  = (m_tlasInstances[idx].accelerationStructureReference != 0);

    m_tlasInstances[idx].transform                      = nvvk::toTransformMatrixKHR(object.worldMatrix);
    m_tlasInstances[idx].flags                          = m_instanceFlagsCache[object.materialID];
    m_tlasInstances[idx].accelerationStructureReference = isVisible ? blasAddress : 0;
    m_tlasInstances[idx].instanceCustomIndex            = object.renderPrimID;

    return std::pair<bool, bool>{wasVisible, isVisible};
  };

  if(dirtyRenderNodes.empty())
  {
    for(size_t i = 0; i < drawObjects.size(); i++)
    {
      auto visibility = updateInstance(static_cast<int>(i));
      numVisibleElement += visibility.second ? 1 : 0;
    }
    staging.appendBuffer(m_instancesBuffer, 0, std::span(m_tlasInstances));
  }
  else
  {
    for(int idx : dirtyRenderNodes)
    {
      if(idx < 0 || idx >= static_cast<int>(drawObjects.size()))
        continue;
      auto visibility = updateInstance(idx);
      if(visibility.first != visibility.second)
        numVisibleElement += visibility.second ? 1 : -1;

      const VkDeviceSize offset = static_cast<VkDeviceSize>(idx) * sizeof(VkAccelerationStructureInstanceKHR);
      staging.appendBuffer(m_instancesBuffer, offset, std::span(&m_tlasInstances[idx], 1));
    }
  }

  assert(numVisibleElement >= 0 && numVisibleElement <= static_cast<int32_t>(drawObjects.size()));

  staging.cmdUploadAppended(cmd);

  nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT);

  if(m_tlasScratchBuffer.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(m_tlasScratchBuffer, m_tlasBuildData.sizeInfo.buildScratchSize,
                                     VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
    NVVK_DBG_NAME(m_tlasScratchBuffer.buffer);
    m_memoryTracker.track(kMemCategoryScratch, m_tlasScratchBuffer.allocation);
  }

  if(m_numVisibleElement != numVisibleElement)
  {
    m_tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlasAccel.accel, m_tlasScratchBuffer.address);
  }
  else
  {
    m_tlasBuildData.cmdUpdateAccelerationStructure(cmd, m_tlasAccel.accel, m_tlasScratchBuffer.address);
  }

  m_numVisibleElement = numVisibleElement;

  nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
// In-place TLAS update after compute wrote instance transforms into `m_instancesBuffer` (CPU shadow vector may be stale).
void nvvkgltf::SceneRtx::cmdUpdateTlasFromInstanceBuffer(VkCommandBuffer cmd)
{
  if(m_tlasAccel.accel == VK_NULL_HANDLE || m_instancesBuffer.buffer == VK_NULL_HANDLE || m_tlasInstances.empty())
    return;

  if(m_tlasScratchBuffer.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(m_tlasScratchBuffer, m_tlasBuildData.sizeInfo.buildScratchSize,
                                     VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
    NVVK_DBG_NAME(m_tlasScratchBuffer.buffer);
    m_memoryTracker.track(kMemCategoryScratch, m_tlasScratchBuffer.allocation);
  }

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);

  m_tlasBuildData.cmdUpdateAccelerationStructure(cmd, m_tlasAccel.accel, m_tlasScratchBuffer.address);

  nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
// Sync TLAS from Scene dirty flags; clears renderNodesRtx. Returns true if TLAS was updated.
bool nvvkgltf::SceneRtx::syncTopLevelAS(VkCommandBuffer cmd, nvvk::StagingUploader& staging, nvvkgltf::Scene& scene)
{
  auto&       df          = scene.getDirtyFlags();
  const auto& dirty       = df.renderNodesRtx;
  const auto& renderNodes = scene.getRenderNodes();

  if(!df.allRenderNodesDirty && dirty.empty() && m_tlasInstances.size() == renderNodes.size())
    return false;

  // Check if we need to do a full update (rebuild) or if we can do a surgical update of the existing TLAS.
  // If the ratio of dirty nodes is high, it's more efficient to do a full rebuild.
  const bool useFullUpdate = df.allRenderNodesDirty || renderNodes.empty()
                             || float(dirty.size()) / float(renderNodes.size()) >= nvvkgltf::kFullUpdateRatio;

  rebuildTopLevelAS(cmd, staging, scene, useFullUpdate ? std::unordered_set<int>{} : dirty);
  df.renderNodesRtx.clear();

  return true;
}

//--------------------------------------------------------------------------------------------------
// Update BLAS for morph targets and skinned primitives (vertex data changed).
void nvvkgltf::SceneRtx::updateBottomLevelAS(VkCommandBuffer cmd, const nvvkgltf::Scene& scene)
{
  for(auto& primID : scene.animation().getMorphPrimitives())
  {
    m_blasBuildData[primID].cmdUpdateAccelerationStructure(cmd, m_blasAccel[primID].accel, m_blasScratchBuffer.address);
    // Add synchronization between consecutive acceleration structure updates that use the same scratch buffer
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                       VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
  }
  for(const auto& task : scene.animation().getSkinTasks())
  {
    m_blasBuildData[task.renderPrimID].cmdUpdateAccelerationStructure(cmd, m_blasAccel[task.renderPrimID].accel,
                                                                      m_blasScratchBuffer.address);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                       VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
  }
}

//--------------------------------------------------------------------------------------------------
// Compact BLAS to reduce memory. Call after cmdBuildBottomLevelAccelerationStructure; then destroyNonCompactedBlas().
VkResult nvvkgltf::SceneRtx::cmdCompactBlas(VkCommandBuffer cmd)
{
  nvutils::ScopedTimer st(__FUNCTION__ + std::string("\n"));

  std::span<nvvk::AccelerationStructureBuildData> blasBuildData(m_blasBuildData);
  std::span<nvvk::AccelerationStructure>          blasAccel(m_blasAccel);

  VkResult result = m_blasBuilder->cmdCompactBlas(cmd, blasBuildData, blasAccel);

  auto stats = m_blasBuilder->getStatistics().toString();
  LOGI("%s%s\n", nvutils::ScopedTimer::indent().c_str(), stats.c_str());
  return result;
}

//--------------------------------------------------------------------------------------------------
// Release non-compacted BLAS after compaction. Call after cmdCompactBlas().
void nvvkgltf::SceneRtx::destroyNonCompactedBlas()
{
  m_blasBuilder->destroyNonCompactedBlas();
}

//========== Destroy ==========

//--------------------------------------------------------------------------------------------------
// Release all acceleration structures, scratch buffers, and BLAS builder. Idempotent.
void nvvkgltf::SceneRtx::destroy()
{
  for(auto& blas : m_blasAccel)
  {
    if(blas.accel != VK_NULL_HANDLE && blas.buffer.allocation)
    {
      m_memoryTracker.untrack(kMemCategoryBLAS, blas.buffer.allocation);
      m_alloc->destroyAcceleration(blas);
    }
  }

  destroyTlasResources();
  destroyScratchBuffers();

  m_blasAccel          = {};
  m_blasBuildData      = {};
  m_instanceFlagsCache = {};
  if(m_blasBuilder)
  {
    m_blasBuilder->deinit();
  }
  m_blasBuilder.reset();
}

//--------------------------------------------------------------------------------------------------
// Release BLAS and TLAS scratch buffers.
void nvvkgltf::SceneRtx::destroyScratchBuffers()
{
  if(m_tlasScratchBuffer.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategoryScratch, m_tlasScratchBuffer.allocation);
    m_alloc->destroyBuffer(m_tlasScratchBuffer);
  }
  if(m_blasScratchBuffer.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategoryScratch, m_blasScratchBuffer.allocation);
    m_alloc->destroyBuffer(m_blasScratchBuffer);
  }
}

//--------------------------------------------------------------------------------------------------
// Deferred destruction of TLAS resources: captures current handles by value and schedules cleanup
// through m_deferredFree (frame-cycle-aware) or falls back to queue/device wait.
void nvvkgltf::SceneRtx::destroyTlasResourcesDeferred()
{
  nvvk::Buffer                oldInstances = m_instancesBuffer;
  nvvk::Buffer                oldScratch   = m_tlasScratchBuffer;
  nvvk::AccelerationStructure oldTlas      = m_tlasAccel;
  m_instancesBuffer                        = {};
  m_tlasScratchBuffer                      = {};
  m_tlasAccel                              = {};
  m_tlasBuildData                          = {};

  auto* alloc   = m_alloc;
  auto* tracker = &m_memoryTracker;
  auto  cleanup = [=]() mutable {
    if(oldInstances.buffer != VK_NULL_HANDLE)
    {
      tracker->untrack(kMemCategoryInstances, oldInstances.allocation);
      alloc->destroyBuffer(oldInstances);
    }
    if(oldScratch.buffer != VK_NULL_HANDLE)
    {
      tracker->untrack(kMemCategoryScratch, oldScratch.allocation);
      alloc->destroyBuffer(oldScratch);
    }
    if(oldTlas.accel != VK_NULL_HANDLE && oldTlas.buffer.allocation)
    {
      tracker->untrack(kMemCategoryTLAS, oldTlas.buffer.allocation);
      alloc->destroyAcceleration(oldTlas);
    }
  };

  if(m_deferredFree)
  {
    m_deferredFree(std::move(cleanup));
  }
  else
  {
    if(m_graphicsQueue)
      vkQueueWaitIdle(m_graphicsQueue);
    else
      vkDeviceWaitIdle(m_device);
    cleanup();
  }
}

//--------------------------------------------------------------------------------------------------
// Release TLAS, instance buffer, and TLAS scratch. BLAS scratch kept for updateBottomLevelAS().
void nvvkgltf::SceneRtx::destroyTlasResources()
{
  if(m_instancesBuffer.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategoryInstances, m_instancesBuffer.allocation);
    m_alloc->destroyBuffer(m_instancesBuffer);
  }

  if(m_tlasScratchBuffer.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategoryScratch, m_tlasScratchBuffer.allocation);
    m_alloc->destroyBuffer(m_tlasScratchBuffer);
  }

  if(m_tlasAccel.accel != VK_NULL_HANDLE && m_tlasAccel.buffer.allocation)
  {
    m_memoryTracker.untrack(kMemCategoryTLAS, m_tlasAccel.buffer.allocation);
    m_alloc->destroyAcceleration(m_tlasAccel);
  }
  m_tlasAccel     = {};
  m_tlasBuildData = {};
}
