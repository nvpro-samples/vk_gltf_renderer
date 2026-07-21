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

#include <cassert>
#include <vector>

#include <vulkan/vulkan_core.h>

#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>

#include "gltf_scene.hpp"

/*-------------------------------------------------------------------------------------------------
# class nvvkgltf::SceneOmm

>  Builds Vulkan opacity micromaps (VK_EXT_opacity_micromap) from the pre-baked
   EXT_mesh_opacity_micromap glTF extension.

This extension stores *pre-build* opacity micromap source data (packed opacity bits, per-triangle
records and usage histograms) that is layout-compatible with the Vulkan build inputs. This class
uploads that data, builds one `VkMicromapEXT` per root `micromaps[]` entry, and uploads the
per-primitive micromap index buffers. `SceneRtx` then attaches the result to the BLAS geometry
(`VkAccelerationStructureTrianglesOpacityMicromapEXT` in `triangles.pNext`).

The per-primitive results are indexed by `renderPrimID` so they align with the BLAS ordering used
by `SceneRtx`. When disabled (device extension unsupported), `create()` is a no-op and the glTF
extension is ignored, so rendering falls back to the regular alpha path.
-------------------------------------------------------------------------------------------------*/
namespace nvvkgltf {

class SceneOmm
{
public:
  // Per-renderPrimID opacity micromap linkage, consumed by SceneRtx at BLAS build time.
  struct PrimitiveOmm
  {
    VkMicromapEXT   micromap     = VK_NULL_HANDLE;
    VkDeviceAddress indexAddress = 0;
    VkIndexType     indexType    = VK_INDEX_TYPE_UINT16;
    uint32_t        indexStride  = 0;
    uint32_t        baseTriangle = 0;
    bool            valid        = false;  // true when this primitive has a usable opacity micromap
  };

  SceneOmm() = default;
  ~SceneOmm() { assert(!m_alloc); }  // Missing deinit call

  void init(nvvk::ResourceAllocator* alloc);
  void deinit();

  // Enable/disable the OMM path. Driven from the VK_EXT_opacity_micromap availability flag.
  void setEnabled(bool enabled) { m_enabled = enabled; }
  bool isEnabled() const { return m_enabled; }

  // Parse EXT_mesh_opacity_micromap, upload build inputs, and record the micromap builds on `cmd`.
  // No-op unless enabled and the scene actually uses the extension.
  void create(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scene);

  // Release all micromaps and buffers.
  void destroy();

  // Per-renderPrimID query API (aligned with SceneRtx BLAS ordering).
  bool                has(uint32_t renderPrimID) const;
  const PrimitiveOmm& get(uint32_t renderPrimID) const;

private:
  // A single built opacity micromap (one per root micromaps[] entry).
  struct Micromap
  {
    VkMicromapEXT micromap = VK_NULL_HANDLE;
    nvvk::Buffer  storage;    // Backing storage of the built micromap (kept for BLAS lifetime)
    nvvk::Buffer  data;       // Packed opacity bits (build input)
    nvvk::Buffer  triangles;  // VkMicromapTriangleEXT records (build input)
    nvvk::Buffer  scratch;    // Build scratch
  };

  VkDevice                 m_device  = VK_NULL_HANDLE;
  nvvk::ResourceAllocator* m_alloc   = nullptr;
  bool                     m_enabled = false;

  std::vector<Micromap>     m_micromaps;     // One per root micromaps[] entry
  std::vector<nvvk::Buffer> m_indexBuffers;  // Per-renderPrimID micromap index buffers (owned)
  std::vector<PrimitiveOmm> m_primitives;    // Per-renderPrimID linkage
};

}  // namespace nvvkgltf
