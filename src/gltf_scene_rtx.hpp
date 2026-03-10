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

#pragma once


#include <nvvk/acceleration_structures.hpp>

#include "gltf_scene_vk.hpp"
#include "gpu_memory_tracker.hpp"

/*-------------------------------------------------------------------------------------------------
# class nvvkgltf::SceneRtx

>  This class is responsible for the ray tracing acceleration structure. 

It is using the `nvvkgltf::Scene` and `nvvkgltf::SceneVk` information to create the acceleration structure.

 -------------------------------------------------------------------------------------------------*/
namespace nvvkgltf {

class SceneRtx
{
public:
  SceneRtx() = default;
  ~SceneRtx() { assert(m_blasAccel.empty()); }  // Missing deinit call

  // ---------- Init / Deinit ----------
  void init(nvvk::ResourceAllocator* alloc);
  void deinit();

  // Optional: set the graphics queue for finer-grained sync in the fallback path.
  void setGraphicsQueue(VkQueue queue) { m_graphicsQueue = queue; }

  // Optional: set a callback for deferred GPU resource destruction (same pattern as SceneVk).
  using DeferredFreeFunc = std::function<void(std::function<void()>&&)>;
  void setDeferredFree(DeferredFreeFunc func) { m_deferredFree = std::move(func); }

  // ---------- Create ----------
  // Create BLAS and TLAS for the scene (no compaction). Calls destroy() first.
  void create(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn, const SceneVk& scnVk, VkBuildAccelerationStructureFlagsKHR flags);
  void createBottomLevelAccelerationStructure(const nvvkgltf::Scene& scene, const SceneVk& sceneVk, VkBuildAccelerationStructureFlagsKHR flags);
  // Build BLAS on GPU; may return false if budget exceeded (call again to continue). Returns true when done.
  [[nodiscard]] bool cmdBuildBottomLevelAccelerationStructure(VkCommandBuffer cmd, VkDeviceSize hintMaxBudget = 512'000'000);
  void cmdCreateBuildTopLevelAccelerationStructure(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scene);
  VkResult                                 cmdCompactBlas(VkCommandBuffer cmd);
  void                                     destroyNonCompactedBlas();
  void                                     trackBlasMemory();
  [[nodiscard]] VkAccelerationStructureKHR topLevelAS();

  // ---------- Sync / Update ----------
  // Sync TLAS from Scene dirty flags (reads + clears renderNodesRtx). Returns true if TLAS was updated.
  [[nodiscard]] bool syncTopLevelAS(VkCommandBuffer cmd, nvvk::StagingUploader& staging, nvvkgltf::Scene& scene);
  // Rebuild or update TLAS. dirtyRenderNodes empty = full update.
  void rebuildTopLevelAS(VkCommandBuffer                cmd,
                         nvvk::StagingUploader&         staging,
                         const nvvkgltf::Scene&         scene,
                         const std::unordered_set<int>& dirtyRenderNodes = {});
  // Update BLAS for morph targets and skinned primitives.
  void updateBottomLevelAS(VkCommandBuffer cmd, const nvvkgltf::Scene& scene);

  // ---------- Destroy ----------
  void destroy();
  void destroyScratchBuffers();

  // ---------- Memory tracking ----------
  const GpuMemoryTracker& getMemoryTracker() const { return m_memoryTracker; }
  GpuMemoryTracker&       getMemoryTracker() { return m_memoryTracker; }

  const std::vector<VkAccelerationStructureInstanceKHR>& getTlasInstances() const { return m_tlasInstances; }

protected:
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_rtASProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

  nvvk::AccelerationStructureGeometryInfo renderPrimitiveToAsGeometry(const nvvkgltf::RenderPrimitive& prim,
                                                                      VkDeviceAddress                  vertexAddress,
                                                                      VkDeviceAddress                  indexAddress);

  void destroyTlasResources();
  void destroyTlasResourcesDeferred();

  VkDevice         m_device         = VK_NULL_HANDLE;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
  VkQueue          m_graphicsQueue  = VK_NULL_HANDLE;

  nvvk::ResourceAllocator* m_alloc = nullptr;

  std::unique_ptr<nvvk::AccelerationStructureBuilder> m_blasBuilder;
  std::vector<nvvk::AccelerationStructureBuildData>   m_blasBuildData;
  std::vector<nvvk::AccelerationStructure>            m_blasAccel;

  nvvk::AccelerationStructureBuildData            m_tlasBuildData;
  nvvk::AccelerationStructure                     m_tlasAccel;
  std::vector<VkAccelerationStructureInstanceKHR> m_tlasInstances;

  nvvk::Buffer m_blasScratchBuffer;
  nvvk::Buffer m_tlasScratchBuffer;
  nvvk::Buffer m_instancesBuffer;

  int32_t m_numVisibleElement = 0;  // Keep track of the number of visible elements in the TLAS

  DeferredFreeFunc m_deferredFree;   // Optional: schedules deferred GPU resource destruction
  GpuMemoryTracker m_memoryTracker;  // GPU memory tracking
};

}  // namespace nvvkgltf
