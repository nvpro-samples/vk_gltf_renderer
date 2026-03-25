/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Coordinator that sequences SceneVk, AnimationVk, and SceneRtx for
// scene creation, destruction, rebuild, and per-frame animation dispatch.
// See gltf_scene_gpu.hpp for design rationale.
//

#include "gltf_scene_gpu.hpp"
#include "gltf_scene_animation.hpp"

namespace nvvkgltf {

//--------------------------------------------------------------------------------------------------
// Initialize all three GPU subsystems with shared allocator, queue, and deferred-free callback.
// SceneVk additionally requires a sampler pool for texture creation.
void SceneGpu::init(nvvk::ResourceAllocator* alloc, nvvk::SamplerPool* samplerPool, VkQueue graphicsQueue, SceneVk::DeferredFreeFunc deferredFree)
{
  m_sceneVk.init(alloc, samplerPool);
  m_sceneVk.setGraphicsQueue(graphicsQueue);
  m_sceneVk.setDeferredFree(deferredFree);

  m_sceneRtx.init(alloc);
  m_sceneRtx.setGraphicsQueue(graphicsQueue);
  m_sceneRtx.setDeferredFree(std::move(deferredFree));

  m_animationVk.init(alloc);
}

//--------------------------------------------------------------------------------------------------
// Tear down all three subsystems. Releases pipelines, allocator references, and any remaining
// GPU resources. Safe to call even if destroy() was already called.
void SceneGpu::deinit()
{
  m_animationVk.deinit();
  m_sceneVk.deinit();
  m_sceneRtx.deinit();
}

//--------------------------------------------------------------------------------------------------
// Upload all GPU resources for a scene and apply the initial morph/skinning deformation.
// Sequence: SceneVk geometry/textures -> AnimationVk static SSBOs -> initial deformation -> flush.
// Does NOT build acceleration structures; the caller handles BLAS/TLAS separately since those
// use the async command buffer queue.
void SceneGpu::create(VkCommandBuffer cmd, Scene& scn, bool generateMipmaps)
{
  m_sceneVk.create(cmd, m_staging, scn, generateMipmaps);
  m_animationVk.createGpuBuffers(m_staging, scn);
  applyAnimation(cmd, scn);
  m_staging.cmdUploadAppended(cmd);
}

//--------------------------------------------------------------------------------------------------
// Rebuild GPU resources after geometry or model changes.
// For full rebuilds (rebuildTextures=true): destroys everything, then recreates via create().
// For geometry-only rebuilds: destroys only geometry + animation + RTX, preserves textures.
void SceneGpu::rebuild(VkCommandBuffer cmd, Scene& scn, bool rebuildTextures)
{
  m_animationVk.destroyGpuBuffers();
  m_sceneRtx.destroy();

  if(rebuildTextures)
  {
    m_sceneVk.create(cmd, m_staging, scn, true);
  }
  else
  {
    m_sceneVk.destroyGeometry();
    m_sceneVk.createGeometry(cmd, m_staging, scn);
  }

  m_animationVk.createGpuBuffers(m_staging, scn);
  applyAnimation(cmd, scn);
  m_staging.cmdUploadAppended(cmd);
}

//--------------------------------------------------------------------------------------------------
// Release scene-level GPU resources across all three subsystems.
// Order: animation SSBOs -> scene buffers/textures -> acceleration structures.
void SceneGpu::destroy()
{
  m_animationVk.destroyGpuBuffers();
  m_sceneVk.destroy();
  m_sceneRtx.destroy();
}

//--------------------------------------------------------------------------------------------------
// Apply morph target blending and skeletal skinning to vertex buffers.
// Dispatches GPU compute shaders when useComputeAnimation is set and the compute pipelines
// are initialized; otherwise falls back to CPU-side computation via SceneVk::uploadPrimitives.
// No-op if the scene has no morph targets or skinning data.
void SceneGpu::applyAnimation(VkCommandBuffer cmd, Scene& scn)
{
  if(!scn.animation().hasMorphTargets() && !scn.animation().hasSkinning())
    return;

  if(useComputeAnimation && m_animationVk.isInitialized())
    m_animationVk.dispatchAnimation(cmd, m_staging, scn, m_sceneVk);
  else
    m_sceneVk.uploadPrimitives(cmd, m_staging, scn);
}

//--------------------------------------------------------------------------------------------------
// Check whether the GPU compute transform path should be used this frame.
// Combines the user-facing toggle with the technical prerequisites checked by canUseGpuTransformPath.
bool SceneGpu::shouldUseGpuTransform(const Scene& scn) const
{
  return useComputeTransformation && canUseGpuTransformPath(m_transformCompute, scn, m_sceneRtx);
}

}  // namespace nvvkgltf
