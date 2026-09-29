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

#pragma once

#include <algorithm>
#include <span>

#include <vulkan/vulkan_core.h>

#include <nvvk/uploader_interface.hpp>

// Shared configuration for `nvvk::FrameUploader` (Resources::staging).
//
// This constant is owned by the app (not by `nvvk::FrameUploader`), because:
//   * `Resources::staging.init()` in renderer.cpp passes it as `blockSize`.
//   * `SceneVk::createImage` in gltf_scene_vk.cpp must know it to chunk oversized
//     mip uploads into row-strips (`BufferCircularAllocator::subAllocate`
//     asserts `size <= blockSize`, so any single append must stay under it).
//
// Keeping both call sites bound to this one symbol guarantees they cannot drift.
//
// Rationale for the current value:
//   * Must be large enough for the biggest typical single upload we stage in one
//     shot (index/vertex buffers of large scenes, single mip levels of large
//     textures).
//   * Kept below 512 MiB so cold init doesn't reserve a huge host-visible chunk
//     up front. Any single mip that exceeds this is uploaded in row-strips by
//     `SceneVk::createImage`, so this is not a hard cap on texture size.
constexpr VkDeviceSize kFrameUploaderBlockSize = 256ull * 1024 * 1024;

// Append a large array in pieces that each fit one staging block. A single append must stay
// under kFrameUploaderBlockSize (see above), but the uploader accepts several appends before a
// flush, so arrays that scale with the scene (render nodes, TLAS instances) go through here:
// 136-byte render nodes would otherwise overflow the block beyond ~1.97M entries.
template <typename T>
VkResult appendBufferChunked(nvvk::CmdUploaderInterface& staging, const nvvk::Buffer& buffer, size_t offset, std::span<const T> data)
{
  constexpr size_t kChunkCount = std::max<size_t>(1, static_cast<size_t>(kFrameUploaderBlockSize) / sizeof(T));
  for(size_t first = 0; first < data.size(); first += kChunkCount)
  {
    const size_t count = std::min(kChunkCount, data.size() - first);
    const VkResult result = staging.appendBuffer(buffer, offset + first * sizeof(T), count * sizeof(T), data.data() + first);
    if(result != VK_SUCCESS)
      return result;
  }
  return VK_SUCCESS;
}
