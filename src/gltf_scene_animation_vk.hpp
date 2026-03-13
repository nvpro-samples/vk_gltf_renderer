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

#include <vector>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>

#include "gltf_scene.hpp"
#include "gltf_scene_vk.hpp"
#include "shaders/animation_io.h.slang"

/*-------------------------------------------------------------------------------------------------
# class nvvkgltf::AnimationVk

>  GPU-accelerated skeletal skinning and morph target blending via compute shaders.

Manages two compute pipelines (skinning + morph) and the GPU SSBOs that feed them.
Static data (base geometry, skin weights, joint indices, IBMs, morph deltas) is uploaded
once at scene load via `createGpuBuffers()`. Each frame, `dispatchAnimation()` computes
joint matrices on CPU, uploads the small per-frame data (joint matrices, morph weights),
and dispatches compute shaders that write transformed vertices directly into SceneVk's
existing vertex buffers via buffer device addresses (BDA push constants).

The CPU fallback path in `SceneVk::uploadPrimitives()` is preserved and selectable via
`SceneGpu::useComputeAnimation`.

 -------------------------------------------------------------------------------------------------*/
namespace nvvkgltf {

class AnimationVk
{
public:
  AnimationVk() = default;
  ~AnimationVk() { assert(!m_alloc); }

  // --- Lifecycle ---
  void init(nvvk::ResourceAllocator* alloc);  // Create compute pipelines
  void deinit();                              // Destroy everything

  // --- Scene data (call once per scene load/rebuild) ---
  void createGpuBuffers(nvvk::StagingUploader& staging, const Scene& scn);  // Upload static SSBOs
  void destroyGpuBuffers();                                                 // Release static + workspace SSBOs

  // --- Per-frame animation (call each frame when animation is active) ---
  void dispatchAnimation(VkCommandBuffer cmd, nvvk::StagingUploader& staging, Scene& scn, const SceneVk& scnVk);

  [[nodiscard]] bool isInitialized() const { return m_alloc != nullptr && m_skinPipeline != VK_NULL_HANDLE; }

private:
  nvvk::ResourceAllocator* m_alloc = nullptr;

  VkPipeline       m_skinPipeline{};
  VkPipeline       m_morphPipeline{};
  VkPipelineLayout m_skinPipelineLayout{};
  VkPipelineLayout m_morphPipelineLayout{};

  // Per-SkinTask GPU buffers (static, uploaded once)
  struct SkinGpuData
  {
    nvvk::Buffer basePositions;
    nvvk::Buffer baseNormals;
    nvvk::Buffer baseTangents;
    nvvk::Buffer weights;
    nvvk::Buffer joints;
    nvvk::Buffer inverseBindMatrices;
    size_t       jointMatOffset  = 0;  // Byte offset into packed m_jointMatricesBuffer
    size_t       normalMatOffset = 0;  // Byte offset into packed m_normalMatricesBuffer
  };
  std::vector<SkinGpuData> m_skinGpuData;

  // Per-MorphResult GPU buffers (static, uploaded once)
  struct MorphGpuData
  {
    nvvk::Buffer basePositions;
    nvvk::Buffer baseNormals;
    nvvk::Buffer baseTangents;
    nvvk::Buffer positionDeltas;  // all targets packed: [numTargets * vertexCount]
    nvvk::Buffer normalDeltas;
    nvvk::Buffer tangentDeltas;
    uint32_t     numTargets    = 0;
    size_t       weightsOffset = 0;  // Byte offset into packed m_morphWeightsBuffer
  };
  std::vector<MorphGpuData> m_morphGpuData;

  // Packed per-frame buffers: all tasks' matrices/weights in one buffer, uploaded in a single transfer.
  // Each task's data is at its stored byte offset. Allocated once at createGpuBuffers() to the total size.
  nvvk::Buffer m_jointMatricesBuffer;
  nvvk::Buffer m_normalMatricesBuffer;
  nvvk::Buffer m_morphWeightsBuffer;

  void createPipelines();
  void destroyPipelines();

  template <typename T>
  nvvk::Buffer createBufferFromSpan(nvvk::StagingUploader& staging, std::span<const T> data, const char* debugName = nullptr);

  nvvk::Buffer createBufferFromData(nvvk::StagingUploader& staging, const void* data, size_t byteSize, const char* debugName = nullptr);
};

}  // namespace nvvkgltf
