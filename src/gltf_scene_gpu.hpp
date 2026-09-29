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

#pragma once

#include "gltf_scene_vk.hpp"
#include "gltf_scene_animation_vk.hpp"
#include "gltf_scene_rtx.hpp"
#include "gltf_scene_transform_vk.hpp"

/*-------------------------------------------------------------------------------------------------
# class nvvkgltf::SceneGpu

>  Coordinator that sequences SceneVk, AnimationVk, and SceneRtx for scene creation,
   destruction, rebuild, and per-frame animation dispatch.

Holds references to the three subsystems and a staging uploader, and ensures they are
called in the correct order. `init()` / `deinit()` consolidate the subsystem initialization
that would otherwise be scattered across the renderer.

The `useComputeAnimation` toggle lives here because the GPU-vs-CPU dispatch decision is
purely an orchestration concern that neither SceneVk nor AnimationVk should know about.

Usage:
  sceneGpu.init(alloc, samplerPool, staging, queue, deferredFree);
  sceneGpu.create(cmd, scene);           // geometry + animation SSBOs + initial deformation
  sceneGpu.applyAnimation(cmd, scene);   // per-frame morph/skin dispatch
  sceneGpu.destroy();                    // release scene-level GPU resources
  sceneGpu.deinit();                     // release pipelines and allocator references

 -------------------------------------------------------------------------------------------------*/
namespace nvvkgltf {

class SceneGpu
{
public:
  SceneGpu(SceneVk& sceneVk, AnimationVk& animationVk, SceneRtx& sceneRtx, TransformComputeVk& transformCompute, nvvk::CmdUploaderInterface& staging)
      : m_sceneVk(sceneVk)
      , m_animationVk(animationVk)
      , m_sceneRtx(sceneRtx)
      , m_transformCompute(transformCompute)
      , m_staging(staging)
  {
  }

  SceneGpu(const SceneGpu&)            = delete;
  SceneGpu& operator=(const SceneGpu&) = delete;

  // --- Initialization (once at startup) ---

  // Initialize all three subsystems: SceneVk, SceneRtx, and AnimationVk.
  // Sets up allocator references, graphics queue, and deferred-free callbacks.
  void init(nvvk::ResourceAllocator* alloc, nvvk::SamplerPool* samplerPool, VkQueue graphicsQueue, SceneVk::DeferredFreeFunc deferredFree);

  // Tear down all three subsystems. Must be called before destruction.
  void deinit();

  // --- Lifecycle (once per scene load/rebuild) ---

  // Upload all GPU resources for a scene: geometry, textures, materials, animation SSBOs,
  // and apply the initial morph/skinning deformation. Flushes staging at the end.
  // Does NOT build acceleration structures -- the caller handles BLAS/TLAS separately.
  void create(VkCommandBuffer cmd, Scene& scn, bool generateMipmaps = true);

  // How rebuild() reconciles the GPU scene with the (possibly changed) model.
  enum class RebuildMode
  {
    eFull,  // Destroy and recreate everything, re-reading every image from disk (scene replace, structural texture change).
    eGeometryOnly,  // Rebuild geometry only; textures and materials are left untouched (mesh/tangent edits).
    eMergeAppend,  // Rebuild geometry + materials, preserve existing textures and append only the new tail images (merge/reference).
  };

  // Rebuild GPU resources after geometry or model changes. Destroys old animation + RTX resources, then
  // recreates SceneVk according to mode. eMergeAppend requires a tail-only, non-empty-base texture set
  // (the merge/reference paths guarantee this); the caller writes the texture descriptors afterwards.
  void rebuild(VkCommandBuffer cmd, Scene& scn, RebuildMode mode);

  // Release scene-level GPU resources across all three subsystems (buffers, textures, AS).
  // Does NOT release pipelines or allocator references -- call deinit() for full teardown.
  void destroy();

  // --- Per-frame animation ---

  // Apply morph target blending and skeletal skinning to vertex buffers.
  // Dispatches GPU compute shaders or the CPU fallback based on useComputeAnimation.
  // No-op if the scene has no morph targets or skinning.
  void applyAnimation(VkCommandBuffer cmd, Scene& scn);

  // Returns true when the GPU compute transform path should be used this frame.
  // Checks the user toggle and technical prerequisites (buffers, dirty flags, TLAS state).
  [[nodiscard]] bool shouldUseGpuTransform(const Scene& scn) const;

  bool useComputeAnimation      = true;
  bool useComputeTransformation = true;

private:
  SceneVk&                    m_sceneVk;
  AnimationVk&                m_animationVk;
  SceneRtx&                   m_sceneRtx;
  TransformComputeVk&         m_transformCompute;
  nvvk::CmdUploaderInterface& m_staging;
};

}  // namespace nvvkgltf
