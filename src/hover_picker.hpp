/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <deque>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

#include <nvvk/resources.hpp>
#include <nvvk/semaphore.hpp>

struct Resources;

//--------------------------------------------------------------------------------------------------
// Async, non-blocking readback of a single pixel from Resources::eImgSelection (the per-pixel
// render-node "ObjectID" G-buffer already produced every frame for the silhouette pass), used to
// detect what's under the mouse cursor for KHR_interactivity hover events - without casting any
// extra rays. See docs/interactivity.md's "Phase E design" note for the full rationale (why this
// beats a per-frame nvvk::RayPicker query) and gltf_scene.hpp's notifyNodeHoverChanged() for what
// consumes the result.
//
// Never blocks: requestReadback() only records a copy; pollResult() only polls a timeline
// semaphore. A small fixed pool of readback buffers absorbs the GPU's in-flight latency; if the
// GPU falls behind and the pool is briefly exhausted, a frame's hover sample is silently skipped
// (never a correctness issue - hover simply updates one frame later).
//--------------------------------------------------------------------------------------------------
class HoverPicker
{
public:
  HoverPicker() = default;
  ~HoverPicker() { assert(m_pool.empty() && "deinit must be called"); }

  void init(Resources& res);
  void deinit(Resources& res);

  // Records a 1x1 copy of eImgSelection at `cursorPixel` (clamped to the image bounds) into a
  // free pool slot. Call once per frame, after the path tracer/rasterizer dispatch and after the
  // silhouette pass specifically - by then eImgSelection is a proven-safe read (silhouette already
  // reads it), so this copy is a read-after-read needing no additional image-layout handling.
  // No-op (silently skipped) if the pool is temporarily exhausted (GPU running behind).
  void requestReadback(VkCommandBuffer cmd, Resources& resources, glm::ivec2 cursorPixel);

  // Non-blocking. Drains every pool slot whose copy has completed (oldest first, freeing each for
  // reuse) and returns the render-node ID from the most recently completed one, if any. Returns
  // nullopt if nothing has completed since the last call, or if the pixel sampled the background
  // (no render node - see the sentinel comment in the .cpp for the exact encoding).
  std::optional<int32_t> pollResult(VkDevice device);

private:
  struct Slot
  {
    nvvk::Buffer buffer;  // 4 bytes, host-visible, persistently mapped
  };
  std::vector<Slot>   m_pool;
  std::vector<size_t> m_freeSlots;

  struct PendingCopy
  {
    size_t               poolIndex;
    nvvk::SemaphoreState semaphoreState;
  };
  std::deque<PendingCopy> m_pending;  // FIFO: oldest submitted first
};
