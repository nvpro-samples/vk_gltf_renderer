/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * GPU propagation of node world matrices and direct writes to SceneVk render-node SSBO + TLAS instance
 * buffer. Avoids staging 100k+ matrices when only a few locals change; CPU still runs
 * updateNodeWorldMatrices() for gizmo, lights, and picking.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>

#include "gltf_scene.hpp"
#include "gltf_scene_vk.hpp"
#include "gltf_scene_rtx.hpp"
#include "gpu_memory_tracker.hpp"
#include "shaders/world_matrix_io.h.slang"

namespace nvvkgltf {

class TransformComputeVk;

// Returns true when transform-only GPU path is valid (no structural/material-RTX changes this frame).
[[nodiscard]] bool canUseGpuTransformPath(const TransformComputeVk& tc, const Scene& scn, const SceneRtx& rtx);

class TransformComputeVk
{
public:
  TransformComputeVk()                                     = default;
  ~TransformComputeVk()                                    = default;
  TransformComputeVk(const TransformComputeVk&)            = delete;
  TransformComputeVk& operator=(const TransformComputeVk&) = delete;
  TransformComputeVk(TransformComputeVk&&)                 = delete;
  TransformComputeVk& operator=(TransformComputeVk&&)      = delete;

  void init(nvvk::ResourceAllocator* alloc);
  void deinit();

  // Same pattern as SceneVk / SceneRtx: schedule VkBuffer destruction after the GPU finishes
  // using them (see Application::submitResourceFree). Required when createGpuBuffers runs mid-frame.
  void setGraphicsQueue(VkQueue queue) { m_graphicsQueue = queue; }
  void setDeferredFree(SceneVk::DeferredFreeFunc func) { m_deferredFree = std::move(func); }

  void createGpuBuffers(nvvk::StagingUploader& staging, const Scene& scn);
  void destroyGpuBuffers();

  // Call when the renderer used the CPU upload path — next GPU frame must re-upload all locals.
  void markGpuStale();

  void dispatchTransformUpdate(VkCommandBuffer cmd, nvvk::StagingUploader& staging, Scene& scn, const SceneVk& scnVk, SceneRtx& scnRtx);

  [[nodiscard]] bool isInitialized() const { return m_alloc != nullptr; }
  [[nodiscard]] bool hasSceneGpuBuffers() const { return m_bLocalMatrices.buffer != VK_NULL_HANDLE; }

  const GpuMemoryTracker& getMemoryTracker() const { return m_memoryTracker; }
  GpuMemoryTracker&       getMemoryTracker() { return m_memoryTracker; }

private:
  void createPipelines();
  void destroyPipelines();
  void ensureGpuBuffersMatchScene(nvvk::StagingUploader& staging, const Scene& scn);
  void destroyGpuBuffersImmediate();

  nvvk::ResourceAllocator*  m_alloc = nullptr;
  SceneVk::DeferredFreeFunc m_deferredFree;
  VkQueue                   m_graphicsQueue{};
  bool                      m_gpuNeedsFullSync         = false;
  uint64_t                  m_cachedSceneGraphRevision = 0;
  size_t                    m_cachedNumRenderNodes     = 0;
  size_t                    m_cachedNumNodes           = 0;

  VkPipeline       m_propagatePipeline{};
  VkPipeline       m_updatePipeline{};
  VkPipelineLayout m_propagateLayout{};
  VkPipelineLayout m_updateLayout{};

  nvvk::Buffer m_bNodeParents;         // Node parents (node index -> parent index)
  nvvk::Buffer m_bTopoNodeOrder;       // Topological node order (node index -> topological node index)
  nvvk::Buffer m_bLocalMatrices;       // Local matrices (node index -> local matrix)
  nvvk::Buffer m_bWorldMatrices;       // World matrices (node index -> world matrix)
  nvvk::Buffer m_bRenderNodeMappings;  // Render node mappings (render node index -> node index)
  nvvk::Buffer m_bGpuInstLocalMatrices;  // GPU instance local matrices (instance index -> local matrix) KHR_mesh_gpu_instancing

  GpuMemoryTracker m_memoryTracker;
};

}  // namespace nvvkgltf
