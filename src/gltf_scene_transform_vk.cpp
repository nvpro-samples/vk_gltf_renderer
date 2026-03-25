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

//
// GPU-accelerated scene-graph transform propagation via compute shaders.
// Uploads dirty local matrices, dispatches a per-level world-matrix
// propagation pass, then writes final render-node transforms and TLAS
// instance data directly on the GPU. See gltf_scene_transform_vk.hpp
// for design rationale.
//

#include "gltf_scene_transform_vk.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>

#include <glm/glm.hpp>

#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include <vulkan/vulkan_core.h>

#include "_autogen/update_render_instances.comp.slang.h"
#include "_autogen/world_matrix_propagate.comp.slang.h"

#include "gltf_scene_vk.hpp"

namespace nvvkgltf {

namespace {

constexpr VkBufferUsageFlags kSsboUsage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT;
constexpr const char* kMemCategoryGraph    = "Xform/Graph";
constexpr const char* kMemCategoryMatrices = "Xform/Matrices";

// Build a per-render-node array of GPU instance-local matrices (identity for nodes without one).
static void fillPerRenderNodeInstanceLocals(const Scene& scn, std::vector<glm::mat4>& out)
{
  const auto& rns    = scn.getRenderNodes();
  const auto& reg    = scn.getRenderNodeRegistry();
  const auto& gpuMap = scn.getGpuInstanceLocalMatrices();
  out.assign(rns.size(), glm::mat4(1.f));

  for(size_t i = 0; i < rns.size(); ++i)
  {
    int nodeID = rns[i].refNodeID;
    if(nodeID < 0)
      continue;
    auto it = gpuMap.find(nodeID);
    if(it == gpuMap.end() || it->second.empty())
      continue;

    const auto& rnList = reg.getRenderNodesForNode(nodeID);
    for(size_t k = 0; k < rnList.size(); ++k)
    {
      if(rnList[k] == static_cast<int>(i))
      {
        out[i] = it->second[k % it->second.size()];
        break;
      }
    }
  }
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// Returns true when the fast GPU transform path can replace the full CPU upload.
// Requires: initialized GPU buffers, only node transforms dirty (no material or structural changes),
// and a valid TLAS instance array matching the current render-node count.
bool canUseGpuTransformPath(const TransformComputeVk& tc, const Scene& scn, const SceneRtx& rtx)
{
  if(!tc.isInitialized() || !tc.hasSceneGpuBuffers())
    return false;

  const auto& df = scn.getDirtyFlags();
  if(df.primitivesChanged || df.allRenderNodesDirty)
    return false;
  if(df.nodes.empty())
    return false;
  if(!df.materials.empty())
    return false;

  const auto& rns  = scn.getRenderNodes();
  const auto& tlas = rtx.getTlasInstances();
  if(rns.empty() || tlas.size() != rns.size())
    return false;
  if(scn.getTopoNodeOrder().empty())
    return false;

  return true;
}

//--------------------------------------------------------------------------------------------------
// Initialize the transform compute system: store allocator and create compute pipelines.
void TransformComputeVk::init(nvvk::ResourceAllocator* alloc)
{
  m_alloc = alloc;
  m_memoryTracker.init(alloc);
  createPipelines();
}

//--------------------------------------------------------------------------------------------------
// Tear down everything: drain GPU, destroy buffers and pipelines, clear allocator.
void TransformComputeVk::deinit()
{
  if(!m_alloc)
    return;
  // Drain GPU work before destroying buffers synchronously (pending deferred frees may still hold old handles).
  vkDeviceWaitIdle(m_alloc->getDevice());
  destroyGpuBuffersImmediate();
  m_deferredFree  = {};
  m_graphicsQueue = VK_NULL_HANDLE;
  destroyPipelines();
  m_alloc = nullptr;
}

//--------------------------------------------------------------------------------------------------
// Create the two compute pipelines (world-matrix propagation and render-instance update).
// Both use push constants only (no descriptor sets); buffer addresses are passed via BDA.
void TransformComputeVk::createPipelines()
{
  VkDevice device = m_alloc->getDevice();

  {
    VkPushConstantRange        pushRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                         .offset     = 0,
                                         .size       = sizeof(shaderio::PropagateWorldMatricesPushConstant)};
    VkPipelineLayoutCreateInfo layoutInfo{.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                          .pushConstantRangeCount = 1,
                                          .pPushConstantRanges    = &pushRange};
    NVVK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_propagateLayout));
    NVVK_DBG_NAME(m_propagateLayout);

    VkShaderModuleCreateInfo shaderInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderInfo.codeSize = world_matrix_propagate_comp_slang_sizeInBytes;
    shaderInfo.pCode    = world_matrix_propagate_comp_slang;

    VkComputePipelineCreateInfo pipeInfo{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.pName = "main";
    pipeInfo.stage.pNext = &shaderInfo;
    pipeInfo.layout      = m_propagateLayout;

    NVVK_CHECK(vkCreateComputePipelines(device, nullptr, 1, &pipeInfo, nullptr, &m_propagatePipeline));
    NVVK_DBG_NAME(m_propagatePipeline);
  }

  {
    VkPushConstantRange        pushRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                         .offset     = 0,
                                         .size       = sizeof(shaderio::UpdateRenderInstancesPushConstant)};
    VkPipelineLayoutCreateInfo layoutInfo{.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                          .pushConstantRangeCount = 1,
                                          .pPushConstantRanges    = &pushRange};
    NVVK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_updateLayout));
    NVVK_DBG_NAME(m_updateLayout);

    VkShaderModuleCreateInfo shaderInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shaderInfo.codeSize = update_render_instances_comp_slang_sizeInBytes;
    shaderInfo.pCode    = update_render_instances_comp_slang;

    VkComputePipelineCreateInfo pipeInfo{.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.pName = "main";
    pipeInfo.stage.pNext = &shaderInfo;
    pipeInfo.layout      = m_updateLayout;

    NVVK_CHECK(vkCreateComputePipelines(device, nullptr, 1, &pipeInfo, nullptr, &m_updatePipeline));
    NVVK_DBG_NAME(m_updatePipeline);
  }
}

//--------------------------------------------------------------------------------------------------
// Destroy compute pipelines and their layouts.
void TransformComputeVk::destroyPipelines()
{
  VkDevice device = m_alloc ? m_alloc->getDevice() : VkDevice{VK_NULL_HANDLE};
  if(device == VK_NULL_HANDLE)
    return;

  if(m_propagatePipeline)
    vkDestroyPipeline(device, m_propagatePipeline, nullptr);
  if(m_updatePipeline)
    vkDestroyPipeline(device, m_updatePipeline, nullptr);
  if(m_propagateLayout)
    vkDestroyPipelineLayout(device, m_propagateLayout, nullptr);
  if(m_updateLayout)
    vkDestroyPipelineLayout(device, m_updateLayout, nullptr);

  m_propagatePipeline = {};
  m_updatePipeline    = {};
  m_propagateLayout   = {};
  m_updateLayout      = {};
}

//--------------------------------------------------------------------------------------------------
// Flag that the GPU local-matrix buffer is out of date (e.g. after a CPU upload path was used).
// The next dispatchTransformUpdate will re-upload all local matrices.
void TransformComputeVk::markGpuStale()
{
  m_gpuNeedsFullSync = true;
}

//--------------------------------------------------------------------------------------------------
// Recreate GPU buffers if the scene topology changed (node/render-node count or graph revision).
void TransformComputeVk::ensureGpuBuffersMatchScene(nvvk::StagingUploader& staging, const Scene& scn)
{
  const size_t   nRn = scn.getRenderNodes().size();
  const size_t   nNd = scn.getModel().nodes.size();
  const uint64_t rev = scn.getSceneGraphRevision();
  if(!hasSceneGpuBuffers() || nRn != m_cachedNumRenderNodes || nNd != m_cachedNumNodes || rev != m_cachedSceneGraphRevision)
  {
    createGpuBuffers(staging, scn);
  }
}

//--------------------------------------------------------------------------------------------------
// Schedule deferred destruction of all scene-graph GPU buffers. Old handles are captured by value
// and freed via the deferred-free callback so in-flight command buffers are not invalidated.
void TransformComputeVk::destroyGpuBuffers()
{
  if(!m_alloc)
    return;

  // Untrack memory before handles are captured and members are cleared.
  m_memoryTracker.untrack(kMemCategoryGraph, m_bNodeParents.allocation);
  m_memoryTracker.untrack(kMemCategoryGraph, m_bTopoNodeOrder.allocation);
  m_memoryTracker.untrack(kMemCategoryGraph, m_bRenderNodeMappings.allocation);
  m_memoryTracker.untrack(kMemCategoryMatrices, m_bLocalMatrices.allocation);
  m_memoryTracker.untrack(kMemCategoryMatrices, m_bWorldMatrices.allocation);
  m_memoryTracker.untrack(kMemCategoryMatrices, m_bGpuInstLocalMatrices.allocation);

  // Capture handles by value, clear members immediately so createGpuBuffers can allocate replacements.
  // Actual vkDestroyBuffer must be deferred while command buffers may still reference the old SSBOs
  // (same pattern as SceneRtx::destroyTlasResourcesDeferred / SceneVk::destroyBufferDeferred).
  nvvk::Buffer oldParents    = m_bNodeParents;
  nvvk::Buffer oldTopo       = m_bTopoNodeOrder;
  nvvk::Buffer oldLocals     = m_bLocalMatrices;
  nvvk::Buffer oldWorlds     = m_bWorldMatrices;
  nvvk::Buffer oldMappings   = m_bRenderNodeMappings;
  nvvk::Buffer oldInstLocals = m_bGpuInstLocalMatrices;

  m_bNodeParents          = {};
  m_bTopoNodeOrder        = {};
  m_bLocalMatrices        = {};
  m_bWorldMatrices        = {};
  m_bRenderNodeMappings   = {};
  m_bGpuInstLocalMatrices = {};

  m_cachedSceneGraphRevision = 0;
  m_cachedNumRenderNodes     = 0;
  m_cachedNumNodes           = 0;

  // Deferred cleanup lambda, so buffers are only destroyed after GPU work referencing them has completed.
  nvvk::ResourceAllocator* alloc   = m_alloc;
  auto                     cleanup = [=]() mutable {
    alloc->destroyBuffer(oldParents);
    alloc->destroyBuffer(oldTopo);
    alloc->destroyBuffer(oldLocals);
    alloc->destroyBuffer(oldWorlds);
    alloc->destroyBuffer(oldMappings);
    alloc->destroyBuffer(oldInstLocals);
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
      vkDeviceWaitIdle(alloc->getDevice());
    cleanup();
  }
}

//--------------------------------------------------------------------------------------------------
// Immediately destroy all scene-graph GPU buffers (used during deinit after draining the GPU).
void TransformComputeVk::destroyGpuBuffersImmediate()
{
  if(!m_alloc)
    return;

  auto destroyBuf = [this](nvvk::Buffer& b, const char* category) {
    if(b.buffer)
    {
      m_memoryTracker.untrack(category, b.allocation);
      m_alloc->destroyBuffer(b);
    }
    b = {};
  };

  destroyBuf(m_bNodeParents, kMemCategoryGraph);
  destroyBuf(m_bTopoNodeOrder, kMemCategoryGraph);
  destroyBuf(m_bRenderNodeMappings, kMemCategoryGraph);
  destroyBuf(m_bLocalMatrices, kMemCategoryMatrices);
  destroyBuf(m_bWorldMatrices, kMemCategoryMatrices);
  destroyBuf(m_bGpuInstLocalMatrices, kMemCategoryMatrices);

  m_cachedSceneGraphRevision = 0;
  m_cachedNumRenderNodes     = 0;
  m_cachedNumNodes           = 0;
}

//--------------------------------------------------------------------------------------------------
// Allocate and upload all scene-graph SSBOs: node parents, topological order, render-node
// mappings, per-instance local matrices, and the local/world matrix pair. Called once on
// scene load and again whenever the graph topology changes.
void TransformComputeVk::createGpuBuffers(nvvk::StagingUploader& staging, const Scene& scn)
{
  destroyGpuBuffers();

  const size_t numNodes       = scn.getModel().nodes.size();
  const size_t numRenderNodes = scn.getRenderNodes().size();

  if(numNodes == 0 || numRenderNodes == 0)
    return;

  const auto& parents   = scn.getNodeParents();
  const auto& topoOrder = scn.getTopoNodeOrder();

  NVVK_CHECK(m_alloc->createBuffer(m_bNodeParents, parents.size() * sizeof(int32_t), kSsboUsage));
  NVVK_CHECK(staging.appendBuffer(m_bNodeParents, 0, std::span(parents)));

  NVVK_CHECK(m_alloc->createBuffer(m_bTopoNodeOrder, topoOrder.size() * sizeof(int32_t), kSsboUsage));
  NVVK_CHECK(staging.appendBuffer(m_bTopoNodeOrder, 0, std::span(topoOrder)));

  std::vector<shaderio::RenderNodeGpuMapping> mappings(numRenderNodes);
  const auto&                                 rns = scn.getRenderNodes();
  for(size_t i = 0; i < numRenderNodes; ++i)
  {
    mappings[i].nodeID       = rns[i].refNodeID;
    mappings[i].pad0         = 0;
    mappings[i].materialID   = rns[i].materialID;
    mappings[i].renderPrimID = rns[i].renderPrimID;
  }

  NVVK_CHECK(m_alloc->createBuffer(m_bRenderNodeMappings, std::span(mappings).size_bytes(), kSsboUsage));
  NVVK_CHECK(staging.appendBuffer(m_bRenderNodeMappings, 0, std::span(mappings)));

  std::vector<glm::mat4> instLocals;
  fillPerRenderNodeInstanceLocals(scn, instLocals);
  NVVK_CHECK(m_alloc->createBuffer(m_bGpuInstLocalMatrices, std::span(instLocals).size_bytes(), kSsboUsage));
  NVVK_CHECK(staging.appendBuffer(m_bGpuInstLocalMatrices, 0, std::span(instLocals)));

  const VkDeviceSize matBytes = numNodes * sizeof(glm::mat4);
  NVVK_CHECK(m_alloc->createBuffer(m_bLocalMatrices, matBytes, kSsboUsage));
  NVVK_CHECK(m_alloc->createBuffer(m_bWorldMatrices, matBytes, kSsboUsage));

  NVVK_CHECK(staging.appendBuffer(m_bLocalMatrices, 0, std::span(scn.getNodesLocalMatrices())));

  NVVK_DBG_NAME(m_bNodeParents.buffer);
  NVVK_DBG_NAME(m_bTopoNodeOrder.buffer);
  NVVK_DBG_NAME(m_bRenderNodeMappings.buffer);
  NVVK_DBG_NAME(m_bGpuInstLocalMatrices.buffer);
  NVVK_DBG_NAME(m_bLocalMatrices.buffer);
  NVVK_DBG_NAME(m_bWorldMatrices.buffer);

  m_memoryTracker.track(kMemCategoryGraph, m_bNodeParents.allocation);
  m_memoryTracker.track(kMemCategoryGraph, m_bTopoNodeOrder.allocation);
  m_memoryTracker.track(kMemCategoryGraph, m_bRenderNodeMappings.allocation);
  m_memoryTracker.track(kMemCategoryMatrices, m_bLocalMatrices.allocation);
  m_memoryTracker.track(kMemCategoryMatrices, m_bWorldMatrices.allocation);
  m_memoryTracker.track(kMemCategoryMatrices, m_bGpuInstLocalMatrices.allocation);

  m_gpuNeedsFullSync         = false;
  m_cachedSceneGraphRevision = scn.getSceneGraphRevision();
  m_cachedNumRenderNodes     = numRenderNodes;
  m_cachedNumNodes           = numNodes;
}

//--------------------------------------------------------------------------------------------------
// Record the full GPU transform update into `cmd`:
//   1. Upload dirty (or all) local matrices via the staging uploader.
//   2. Phase 1 — propagate world matrices level-by-level through the scene-graph topology.
//   3. Phase 2 — write final render-node transforms and TLAS instance data.
//   4. Rebuild the TLAS from the updated instance buffer.
void TransformComputeVk::dispatchTransformUpdate(VkCommandBuffer cmd, nvvk::StagingUploader& staging, Scene& scn, const SceneVk& scnVk, SceneRtx& scnRtx)
{
  if(!m_alloc)
    return;

  ensureGpuBuffersMatchScene(staging, scn);
  if(!hasSceneGpuBuffers())
    return;

  const size_t numNodes       = scn.getModel().nodes.size();
  const size_t numRenderNodes = scn.getRenderNodes().size();
  if(numNodes == 0 || numRenderNodes == 0)
    return;

  auto& df = scn.getDirtyFlags();

  const auto& locals = scn.getNodesLocalMatrices();
  if(locals.size() < numNodes)
    return;

  if(m_gpuNeedsFullSync)
  {
    NVVK_CHECK(staging.appendBuffer(m_bLocalMatrices, 0, std::span(locals.data(), numNodes)));
  }
  else
  {
    for(int nodeID : df.nodes)
    {
      if(nodeID < 0 || static_cast<size_t>(nodeID) >= numNodes)
        continue;
      const VkDeviceSize offset = static_cast<VkDeviceSize>(nodeID) * sizeof(glm::mat4);
      NVVK_CHECK(staging.appendBuffer(m_bLocalMatrices, offset, std::span(&locals[static_cast<size_t>(nodeID)], 1)));
    }
  }

  staging.cmdUploadAppended(cmd);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

  // Phase 1: propagate world matrices per topological level.
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_propagatePipeline);

  const auto& levels = scn.getTopoLevels();
  for(size_t li = 0; li < levels.size(); ++li)
  {
    const auto [levelOffset, levelCount] = levels[li];
    if(levelCount == 0)
      continue;

    shaderio::PropagateWorldMatricesPushConstant pc{};
    pc.localMatrices = reinterpret_cast<glm::mat4*>(m_bLocalMatrices.address);
    pc.worldMatrices = reinterpret_cast<glm::mat4*>(m_bWorldMatrices.address);
    pc.parentIndices = reinterpret_cast<int*>(m_bNodeParents.address);
    pc.topoNodeOrder = reinterpret_cast<int*>(m_bTopoNodeOrder.address);
    pc.levelOffset   = static_cast<uint32_t>(levelOffset);
    pc.levelCount    = static_cast<uint32_t>(levelCount);

    vkCmdPushConstants(cmd, m_propagateLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (levelCount + WORLD_MATRIX_WORKGROUP_SIZE - 1) / WORLD_MATRIX_WORKGROUP_SIZE, 1, 1);

    if(li + 1 < levels.size())
    {
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    }
  }

  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

  // Phase 2: render nodes + TLAS instance transforms.
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_updatePipeline);

  shaderio::UpdateRenderInstancesPushConstant upc{};
  upc.worldMatrices        = reinterpret_cast<glm::mat4*>(m_bWorldMatrices.address);
  upc.mappings             = reinterpret_cast<shaderio::RenderNodeGpuMapping*>(m_bRenderNodeMappings.address);
  upc.gpuInstLocalMatrices = reinterpret_cast<glm::mat4*>(m_bGpuInstLocalMatrices.address);
  upc.outRenderNodes       = reinterpret_cast<shaderio::GltfRenderNode*>(scnVk.renderNodeBuffer().address);
  upc.outInstances         = reinterpret_cast<shaderio::TlasInstance*>(scnRtx.getInstancesBufferAddress());
  upc.renderNodeCount      = static_cast<uint32_t>(numRenderNodes);
  upc.pad0                 = 0;

  vkCmdPushConstants(cmd, m_updateLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(upc), &upc);
  vkCmdDispatch(cmd, static_cast<uint32_t>((numRenderNodes + WORLD_MATRIX_WORKGROUP_SIZE - 1) / WORLD_MATRIX_WORKGROUP_SIZE), 1, 1);

  scnRtx.cmdUpdateTlasFromInstanceBuffer(cmd);

#ifndef NDEBUG
  const_cast<SceneVk&>(scnVk).debugUpdateShadowCopy(scn);  // material/prim IDs — matrices validated separately
#endif

  m_gpuNeedsFullSync = false;

  df.renderNodesVk.clear();
  df.renderNodesRtx.clear();
}

}  // namespace nvvkgltf
