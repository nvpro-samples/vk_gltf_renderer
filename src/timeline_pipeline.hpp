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

/*-------------------------------------------------------------------------------------------------
# class TimelinePipeline

>  Non-blocking GPU command pipeline using a Vulkan timeline semaphore.

Instead of submitting command buffers with fences and blocking on each one,
this pipeline submits with monotonically increasing timeline values, then polls
for completion with vkGetSemaphoreCounterValue (non-blocking). The GPU runs
ahead while the CPU stays free to render frames and handle UI.

Callbacks may call enqueue() to chain dependent work (e.g. BLAS compaction
after a BLAS build). poll() and drain() naturally pick up the new work.

Usage:
  pipeline.init(device, queue, cmdPool);

  // From any thread: record commands, end the buffer, then enqueue
  vkEndCommandBuffer(cmd);
  pipeline.enqueue(cmd);                                // fire-and-forget
  pipeline.enqueue(cmd, [&]{ releaseStaging(); });      // with completion callback

  // Each frame on the main thread
  bool loading = pipeline.poll();

  // Headless / testing: block until all work completes
  pipeline.drain();

  // On scene change (after vkQueueWaitIdle): free everything
  pipeline.clear();

  pipeline.destroy();
-------------------------------------------------------------------------------------------------*/

#include <vulkan/vulkan_core.h>

#include <functional>
#include <mutex>
#include <queue>
#include <deque>

class TimelinePipeline
{
public:
  void init(VkDevice device, VkQueue queue, VkCommandPool cmdPool);
  void destroy();

  // Enqueue an ended command buffer for submission. Thread-safe.
  // onComplete is called on the main thread once the GPU has finished this buffer.
  void enqueue(VkCommandBuffer cmd, std::function<void()> onComplete = {});

  // Submit queued buffers, poll completed ones, run callbacks.
  // Call once per frame from the main thread.
  // Returns true if work is still pending or in-flight.
  bool poll();

  // Block until all in-flight and queued work completes (for headless / testing).
  void drain();

  // Free all queued and pending command buffers. Call after vkQueueWaitIdle.
  void clear();

private:
  VkDevice      m_device{};
  VkQueue       m_queue{};
  VkCommandPool m_cmdPool{};
  VkSemaphore   m_timeline{VK_NULL_HANDLE};
  uint64_t      m_nextValue{1};

  struct Queued
  {
    VkCommandBuffer       cmd;
    std::function<void()> onComplete;
  };
  std::queue<Queued> m_submitQueue;
  std::mutex         m_submitMutex;

  struct Pending
  {
    uint64_t              timelineValue;
    VkCommandBuffer       cmd;
    std::function<void()> onComplete;
  };
  std::deque<Pending> m_pending;

  void submitQueued();
  void processCompleted(uint64_t gpuValue);
};
