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

#include <cinttypes>
#include <numeric>

#include <nvutils/alignment.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "gltf_scene_rtx.hpp"
#include "gltf_scene_vk.hpp"
#include "tinygltf_utils.hpp"

// GPU memory category names for RTX resources
namespace {
constexpr std::string_view kMemCategoryBLAS      = "BLAS";
constexpr std::string_view kMemCategoryTLAS      = "TLAS";
constexpr std::string_view kMemCategoryScratch   = "Scratch";
constexpr std::string_view kMemCategoryInstances = "Instances";
}  // namespace

// Initialize the scene for ray tracing
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

void nvvkgltf::SceneRtx::deinit()
{
  if(!m_alloc)
  {
    return;
  }

  destroy();

  m_alloc = nullptr;
}

// Create the acceleration structure
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

// Get the top-level acceleration structure
VkAccelerationStructureKHR nvvkgltf::SceneRtx::tlas()
{
  return m_tlasAccel.accel;
}

// Track all BLAS allocations - call this after all BLAS are built
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

// Destroy the acceleration structure
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

  if(m_instancesBuffer.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategoryInstances, m_instancesBuffer.allocation);
    m_alloc->destroyBuffer(m_instancesBuffer);
  }
  destroyScratchBuffers();

  if(m_tlasAccel.accel != VK_NULL_HANDLE && m_tlasAccel.buffer.allocation)
  {
    m_memoryTracker.untrack(kMemCategoryTLAS, m_tlasAccel.buffer.allocation);
    m_alloc->destroyAcceleration(m_tlasAccel);
  }
  m_blasAccel     = {};
  m_blasBuildData = {};
  m_tlasAccel     = {};
  m_tlasBuildData = {};
  if(m_blasBuilder)
  {
    m_blasBuilder->deinit();
  }
  m_blasBuilder.reset();
}

// Destroy the scratch buffers
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
// Converting a PrimitiveMesh as input for BLAS
//
// Convert a RenderPrimitive to an AccelerationStructureGeometryInfo
// This function takes a RenderPrimitive and converts it into an AccelerationStructureGeometryInfo
// The resulting structure is used to build the bottom-level acceleration structure (BLAS)
//
nvvk::AccelerationStructureGeometryInfo nvvkgltf::SceneRtx::renderPrimitiveToAsGeometry(const nvvkgltf::RenderPrimitive& prim,  // The RenderPrimitive to convert
                                                                                        VkDeviceAddress vertexAddress,  // The device address of the vertex buffer
                                                                                        VkDeviceAddress indexAddress)  // The device address of the index buffer
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
// Create the bottom-level acceleration structure
//
// This function creates the bottom-level acceleration structure (BLAS)
// It takes a nvvkgltf::Scene and a SceneVk object as input
//

void nvvkgltf::SceneRtx::createBottomLevelAccelerationStructure(const nvvkgltf::Scene& scene,  // The nvvkgltf::Scene object
                                                                const SceneVk& sceneVk,        // The SceneVk object
                                                                VkBuildAccelerationStructureFlagsKHR flags)  // The flags for the acceleration structure
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
// Build the bottom-level acceleration structure
//
// This function builds the bottom-level acceleration structure (BLAS)
// It takes a VkCommandBuffer and a VkDeviceSize as input
//
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

//--------------------------------------------------------------------------------------------------
// Get the instance flag,
// The instance flag is used to determine if the material is opaque or not, and if the material is double sided or not
//
VkGeometryInstanceFlagsKHR getInstanceFlag(const tinygltf::Material& mat)
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
// Create the top-level acceleration structure from all the BLAS
//
void nvvkgltf::SceneRtx::cmdCreateBuildTopLevelAccelerationStructure(VkCommandBuffer        cmd,
                                                                     nvvk::StagingUploader& staging,
                                                                     const nvvkgltf::Scene& scene)
{
  nvutils::ScopedTimer st(__FUNCTION__);
  const auto&          materials   = scene.getModel().materials;
  const auto&          drawObjects = scene.getRenderNodes();

  uint32_t instanceCount = static_cast<uint32_t>(drawObjects.size());

  m_tlasInstances.clear();
  m_tlasInstances.reserve(instanceCount);
  m_numVisibleElement = 0;  // Number of visible elements
  for(const auto& object : drawObjects)
  {
    const tinygltf::Material&  mat           = materials[object.materialID];
    VkGeometryInstanceFlagsKHR instanceFlags = getInstanceFlag(mat);

    VkDeviceAddress blasAddress = m_blasAccel[object.renderPrimID].address;
    if(!object.visible)
      blasAddress = 0;  // The instance is added, but the BLAS is set to null making it invisible

    // Update the number of visible elements
    m_numVisibleElement += object.visible ? 1 : 0;


    VkAccelerationStructureInstanceKHR asInstance{};
    asInstance.transform           = nvvk::toTransformMatrixKHR(object.worldMatrix);  // Position of the instance
    asInstance.instanceCustomIndex = object.renderPrimID;                             // gl_InstanceCustomIndexEXT
    asInstance.accelerationStructureReference         = blasAddress;                  // The reference to the BLAS
    asInstance.instanceShaderBindingTableRecordOffset = 0;     // We will use the same hit group for all objects
    asInstance.mask                                   = 0x01;  // Visibility mask
    asInstance.flags                                  = instanceFlags;

    // Storing the instance
    m_tlasInstances.push_back(asInstance);
  }

  VkBuildAccelerationStructureFlagsKHR buildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  //if(scene.hasAnimation())
  {
    buildFlags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  }

  // Create a buffer holding the actual instance data (matrices++) for use by the AS builder.
  // Instance buffer device addresses must be aligned to 16 bytes according to
  // https://vulkan.lunarg.com/doc/view/1.4.328.1/windows/antora/spec/latest/chapters/accelstructures.html#VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03717 .
  constexpr VmaAllocationCreateFlags instanceAllocFlags   = 0;
  constexpr VkDeviceSize             instanceMinAlignment = 16;
  NVVK_CHECK(m_alloc->createBuffer(m_instancesBuffer, std::span(m_tlasInstances).size_bytes(),
                                   VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                   VMA_MEMORY_USAGE_AUTO, instanceAllocFlags, instanceMinAlignment));
  NVVK_CHECK(staging.appendBuffer(m_instancesBuffer, 0, std::span(m_tlasInstances)));
  NVVK_DBG_NAME(m_instancesBuffer.buffer);
  m_memoryTracker.track(kMemCategoryInstances, m_instancesBuffer.allocation);

  m_tlasBuildData.asType = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  auto geo               = m_tlasBuildData.makeInstanceGeometry(m_tlasInstances.size(), m_instancesBuffer.address);
  m_tlasBuildData.addGeometry(geo);

  staging.cmdUploadAppended(cmd);

  // Make sure the copy of the instance buffer are copied before triggering the acceleration structure build
  nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT);

  // Calculate the amount of scratch memory needed to build the TLAS
  VkAccelerationStructureBuildSizesInfoKHR sizeInfo = m_tlasBuildData.finalizeGeometry(m_device, buildFlags);

  // Create the scratch buffer needed during build of the TLAS
  NVVK_CHECK(m_alloc->createBuffer(m_tlasScratchBuffer, sizeInfo.buildScratchSize,
                                   VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT
                                       | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR));
  NVVK_DBG_NAME(m_tlasScratchBuffer.buffer);
  m_memoryTracker.track(kMemCategoryScratch, m_tlasScratchBuffer.allocation);

  VkAccelerationStructureCreateInfoKHR createInfo = m_tlasBuildData.makeCreateInfo();
  NVVK_CHECK(m_alloc->createAcceleration(m_tlasAccel, createInfo));
  NVVK_DBG_NAME(m_tlasAccel.accel);
  m_memoryTracker.track(kMemCategoryTLAS, m_tlasAccel.buffer.allocation);


  // Build the TLAS
  m_tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlasAccel.accel, m_tlasScratchBuffer.address);

  // Make sure to have the TLAS ready before using it
  nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}


// Partial update: only update instances for the dirty render nodes
void nvvkgltf::SceneRtx::updateTopLevelAS(VkCommandBuffer                cmd,
                                          nvvk::StagingUploader&         staging,
                                          const nvvkgltf::Scene&         scene,
                                          const std::unordered_set<int>& dirtyRenderNodes)
{
  // nvutils::ScopedTimer st(__FUNCTION__);

  const auto& drawObjects = scene.getRenderNodes();
  const auto& materials   = scene.getModel().materials;

  // Number of visible elements before the update.
  // If the dirtyRenderNodes is empty, we do a full update, so the number of visible elements is 0.
  // Otherwise, we do a partial update, so the number of visible elements is the number of visible elements before the update
  // which can be reduced if no longer visible.
  int32_t numVisibleElement = dirtyRenderNodes.empty() ? 0 : m_numVisibleElement;


  auto updateInstance = [&](int idx) {
    const auto&               object      = drawObjects[idx];
    const tinygltf::Material& mat         = materials[object.materialID];
    VkDeviceAddress           blasAddress = m_blasAccel[object.renderPrimID].address;
    const bool                isVisible   = object.visible && (blasAddress != 0);
    const bool                wasVisible  = (m_tlasInstances[idx].accelerationStructureReference != 0);

    m_tlasInstances[idx].transform                      = nvvk::toTransformMatrixKHR(object.worldMatrix);
    m_tlasInstances[idx].flags                          = getInstanceFlag(mat);
    m_tlasInstances[idx].accelerationStructureReference = isVisible ? blasAddress : 0;
    return std::pair<bool, bool>{wasVisible, isVisible};
  };

  if(dirtyRenderNodes.empty())
  {
    // Full update
    for(size_t i = 0; i < drawObjects.size(); i++)
    {
      auto visibility = updateInstance(static_cast<int>(i));
      numVisibleElement += visibility.second ? 1 : 0;
    }

    // Update the full instance buffer
    staging.appendBuffer(m_instancesBuffer, 0, std::span(m_tlasInstances));
  }
  else
  {
    // Partial update: only dirty indices
    for(int idx : dirtyRenderNodes)
    {
      if(idx < 0 || idx >= static_cast<int>(drawObjects.size()))
        continue;

      // Change the number of visible elements, if needed
      auto visibility = updateInstance(idx);
      if(visibility.first != visibility.second)           // Was visible != Now visible
        numVisibleElement += visibility.second ? 1 : -1;  // Add or remove

      const VkDeviceSize offset = static_cast<VkDeviceSize>(idx) * sizeof(VkAccelerationStructureInstanceKHR);
      staging.appendBuffer(m_instancesBuffer, offset, std::span(&m_tlasInstances[idx], 1));
    }
  }

  // Sanity check
  assert(numVisibleElement >= 0 && numVisibleElement <= static_cast<int32_t>(drawObjects.size()));

  staging.cmdUploadAppended(cmd);

  // Make sure the copy of the instance buffer are copied before triggering the acceleration structure build
  nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT);

  if(m_tlasScratchBuffer.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(m_tlasScratchBuffer, m_tlasBuildData.sizeInfo.buildScratchSize,
                                     VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
    NVVK_DBG_NAME(m_tlasScratchBuffer.buffer);
    m_memoryTracker.track(kMemCategoryScratch, m_tlasScratchBuffer.allocation);
  }

  // Building or updating the top-level acceleration structure
  if(m_numVisibleElement != numVisibleElement)
  {
    m_tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlasAccel.accel, m_tlasScratchBuffer.address);
  }
  else
  {
    m_tlasBuildData.cmdUpdateAccelerationStructure(cmd, m_tlasAccel.accel, m_tlasScratchBuffer.address);
  }

  m_numVisibleElement = numVisibleElement;

  // Make sure to have the TLAS ready before using it
  nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}

void nvvkgltf::SceneRtx::updateBottomLevelAS(VkCommandBuffer cmd, const nvvkgltf::Scene& scene)
{
  // nvutils::ScopedTimer st(__FUNCTION__);
  // #TODO - Check that primID aren't duplicated
  for(auto& primID : scene.getMorphPrimitives())
  {
    m_blasBuildData[primID].cmdUpdateAccelerationStructure(cmd, m_blasAccel[primID].accel, m_blasScratchBuffer.address);
    // Add synchronization between consecutive acceleration structure updates that use the same scratch buffer
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                       VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
  }
  for(auto& skinNode : scene.getSkinNodes())  // Update the BLAS
  {
    int primID = scene.getRenderNodes()[skinNode].renderPrimID;
    m_blasBuildData[primID].cmdUpdateAccelerationStructure(cmd, m_blasAccel[primID].accel, m_blasScratchBuffer.address);
    // Add synchronization between consecutive acceleration structure updates that use the same scratch buffer
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                       VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
  }
}

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

void nvvkgltf::SceneRtx::destroyNonCompactedBlas()
{
  m_blasBuilder->destroyNonCompactedBlas();
}
