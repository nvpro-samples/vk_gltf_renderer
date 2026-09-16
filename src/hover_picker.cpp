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

//
// Async, non-blocking readback of a single pixel from Resources::eImgSelection (the per-pixel
// render-node "ObjectID" G-buffer already produced every frame for the silhouette pass), used to
// detect what's under the mouse cursor for KHR_interactivity hover events without casting any
// extra rays. Never blocks: requestReadback() only records a copy, pollResult() only polls a
// timeline semaphore. See docs/interactivity.md's "Phase E design" note.
//

#include <cstring>

#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "hover_picker.hpp"
#include "resources.hpp"

namespace {
// Small fixed pool of readback buffers - comfortably above the app's 2 frames-in-flight
// (nvapp::Application runs double buffering; see docs/interactivity.md's Phase E design note) so
// a copy normally has a full ring cycle to complete before its slot is needed again.
constexpr size_t kPoolSize = 4;
}  // namespace

void HoverPicker::init(Resources& res)
{
  m_pool.resize(kPoolSize);
  m_freeSlots.reserve(kPoolSize);
  for(size_t i = 0; i < kPoolSize; ++i)
  {
    NVVK_CHECK(res.allocator.createBuffer(m_pool[i].buffer, sizeof(float), VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO,
                                          VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
    NVVK_DBG_NAME(m_pool[i].buffer.buffer);
    m_freeSlots.push_back(i);
  }
}

void HoverPicker::deinit(Resources& res)
{
  for(Slot& slot : m_pool)
    res.allocator.destroyBuffer(slot.buffer);
  m_pool.clear();
  m_freeSlots.clear();
  m_pending.clear();
}

void HoverPicker::requestReadback(VkCommandBuffer cmd, Resources& resources, glm::ivec2 cursorPixel)
{
  if(m_freeSlots.empty())
    return;  // GPU running behind - drop this frame's hover sample, not a correctness issue.

  const VkExtent2D size = resources.gBuffers.getSize();
  if(cursorPixel.x < 0 || cursorPixel.y < 0 || cursorPixel.x >= static_cast<int>(size.width)
     || cursorPixel.y >= static_cast<int>(size.height))
    return;  // Cursor outside the render target (e.g. over an ImGui panel) - nothing to sample.

  const size_t poolIndex = m_freeSlots.back();
  m_freeSlots.pop_back();

  // eImgSelection is kept in VK_IMAGE_LAYOUT_GENERAL throughout (nvvk::RenderTarget's documented
  // "Compute MRT" contract), which vkCmdCopyImageToBuffer accepts directly - no layout transition.
  // No additional barrier here: this call is placed after the silhouette pass's own read of the
  // same image (see the call site in renderer.cpp), so this is a read-after-read of an image the
  // silhouette pass has already proven synchronized - only a write-then-read transition needs a
  // barrier, and that one already happened before silhouette's dispatch.
  VkBufferImageCopy region{
      .bufferOffset      = 0,
      .bufferRowLength   = 0,
      .bufferImageHeight = 0,
      .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageOffset       = {cursorPixel.x, cursorPixel.y, 0},
      .imageExtent       = {1, 1, 1},
  };
  vkCmdCopyImageToBuffer(cmd, resources.gBuffers.getSampleImage(Resources::eImgSelection), VK_IMAGE_LAYOUT_GENERAL,
                         m_pool[poolIndex].buffer.buffer, 1, &region);

  m_pending.push_back({poolIndex, nvvk::SemaphoreState::makeFixed(resources.app->getFrameSignalSemaphore())});
}

std::optional<int32_t> HoverPicker::pollResult(VkDevice device)
{
  std::optional<int32_t> result;
  while(!m_pending.empty() && m_pending.front().semaphoreState.testSignaled(device))
  {
    const size_t poolIndex = m_pending.front().poolIndex;
    m_pending.pop_front();

    float raw;
    std::memcpy(&raw, m_pool[poolIndex].buffer.mapping, sizeof(float));
    m_freeSlots.push_back(poolIndex);

    // Shader-side encoding (gltf_pathtrace.slang's firstFrame block): the render-node index is
    // bit-cast (not numerically converted - GLSL/Slang `asfloat`) into the float channel, offset
    // by +1 so 0 can mean "nothing hit" (see traceSelectionRay, pathtrace_functions.h.slang).
    uint32_t bits;
    std::memcpy(&bits, &raw, sizeof(bits));
    result = static_cast<int32_t>(bits) - 1;  // -1 (from encoded 0) means "nothing under the cursor"
  }
  return result;
}
