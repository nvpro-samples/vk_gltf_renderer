/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * GPU propagation of node world matrices and direct writes to SceneVk render-node SSBO + TLAS instance
 * buffer. Avoids staging 100k+ matrices when only a few locals change; CPU still runs
 * updateNodeWorldMatrices() for gizmo, lights, and picking.
 *
 * TLAS visibility (KHR_node_visibility): hidden instances clear their BLAS reference on the CPU
 * via SceneRtx::syncTopLevelAS when Scene::DirtyFlags::tlasVisibilityNeedsCpuSync is set (see
 * SceneEditor::updateVisibility), avoiding huge mapping SSBO uploads on the GPU transform path.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*

What problem does it solve?
A glTF scene is a tree of nodes. Each node has a local transform (position/rotation/scale relative to its parent). To render, you need the world transform (absolute position in the scene). Traditionally this is computed on the CPU and uploaded — but with 100k+ nodes, that's slow. This system does it entirely on the GPU.

The three-step GPU pipeline

1. Upload dirty locals
Every frame, only the changed local matrices are uploaded (sparse patch). If markGpuStale() was called (e.g. after a CPU-side upload), all matrices are re-uploaded.

2. Propagate world matrices — world_matrix_propagate.comp
The scene graph is pre-sorted in BFS (breadth-first) topological order, level by level. The shader runs one dispatch per level:

worldMatrix[node] = worldMatrix[parent] × localMatrix[node]
A memory barrier between levels ensures parents are fully written before children read them. Root nodes (parent = -1) just copy their local matrix.

3. Write render instances — update_render_instances.comp
For each RenderNode (a mesh instance), the shader computes the final object-to-world transform and writes it directly into:

The SceneVk render-node SSBO (used by rasterization/ray-query shaders)
The TLAS instance buffer (used by ray tracing — BLAS reference is preserved from the GPU, never touched by CPU)
Then cmdUpdateTlasFromInstanceBuffer() does a fast in-place TLAS rebuild.


CPU writes once (at load / topo change):
  NodeParents[]          — who is each node's parent
  TopoNodeOrder[]        — BFS-sorted node indices
  RenderNodeMappings[]   — which node/material/prim each RenderNode points to
  InstLocalMatrices[]    — extra local offset (KHR_mesh_gpu_instancing)

CPU patches each frame:
  LocalMatrices[]        — dirty node local matrices

GPU writes each frame:
  WorldMatrices[]        — output of Phase 1
  SceneVk.renderNodes[]  — output of Phase 2 (objectToWorld + IDs)
  TLAS instance buffer   — output of Phase 2 (transform only, BLAS ref kept)


init()                    — create the 3 compute pipelines
  │
  ├─ createGpuBuffers()   — on scene load or topology change
  │                         (deferred-destroy old buffers so in-flight frames aren't broken)
  │
  ├─ [every frame]
  │   ├─ cmdSnapshotPrevObjectToWorld()   — DLSS: save prev transforms
  │   └─ dispatchTransformUpdate()        — (1,2,3) + TLAS rebuild
  │
  └─ deinit()             — vkDeviceWaitIdle → destroy everything
*/


#pragma once

#include <cstdint>

#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>

#include "gltf_scene.hpp"
#include "gltf_scene_vk.hpp"
#include "gltf_scene_rtx.hpp"
#include "gpu_memory_tracker.hpp"

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

  // DLSS instance motion vectors: snapshot the current render-node objectToWorld matrices
  void cmdSnapshotPrevObjectToWorld(VkCommandBuffer cmd, const SceneVk& scnVk, size_t numRenderNodes);

  // Device address of the previous-frame objectToWorld buffer (0 if not yet allocated).
  [[nodiscard]] VkDeviceAddress prevObjectToWorldAddress() const { return m_bPrevRenderNodeO2W.address; }

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
  size_t                    m_cachedTopoOrderSize      = 0;

  VkPipeline       m_propagatePipeline{};
  VkPipeline       m_updatePipeline{};
  VkPipeline       m_snapshotPipeline{};
  VkPipelineLayout m_propagateLayout{};
  VkPipelineLayout m_updateLayout{};
  VkPipelineLayout m_snapshotLayout{};

  nvvk::Buffer m_bNodeParents;         // Node parents (node index -> parent index)
  nvvk::Buffer m_bTopoNodeOrder;       // Topological node order (node index -> topological node index)
  nvvk::Buffer m_bLocalMatrices;       // Local matrices (node index -> local matrix)
  nvvk::Buffer m_bWorldMatrices;       // World matrices (node index -> world matrix)
  nvvk::Buffer m_bRenderNodeMappings;  // Render node mappings (render node index -> node index)
  nvvk::Buffer m_bGpuInstLocalMatrices;  // GPU instance local matrices (instance index -> local matrix) KHR_mesh_gpu_instancing

  // DLSS instance motion: previous-frame objectToWorld per render node (lazily allocated, DLSS only).
  nvvk::Buffer m_bPrevRenderNodeO2W;
  size_t       m_cachedPrevO2WCount = 0;

  GpuMemoryTracker m_memoryTracker;
};

}  // namespace nvvkgltf
