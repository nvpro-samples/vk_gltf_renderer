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
// Non-blocking GPU command pipeline using a Vulkan timeline semaphore.
// See timeline_pipeline.hpp for design rationale and usage.
//

#include "timeline_pipeline.hpp"
#include <volk.h>
#include <nvvk/check_error.hpp>

//--------------------------------------------------------------------------------------------------
// Create the timeline semaphore and store the device, queue, and command pool for later use.
void TimelinePipeline::init(VkDevice device, VkQueue queue, VkCommandPool cmdPool)
{
  m_device    = device;
  m_queue     = queue;
  m_cmdPool   = cmdPool;
  m_nextValue = 1;

  VkSemaphoreTypeCreateInfo typeInfo{.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                     .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                                     .initialValue  = 0};
  VkSemaphoreCreateInfo     semInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &typeInfo};
  NVVK_CHECK(vkCreateSemaphore(m_device, &semInfo, nullptr, &m_timeline));
}

//--------------------------------------------------------------------------------------------------
// Free all pending work and destroy the timeline semaphore.
void TimelinePipeline::destroy()
{
  clear();
  if(m_timeline != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(m_device, m_timeline, nullptr);
    m_timeline = VK_NULL_HANDLE;
  }
}

//--------------------------------------------------------------------------------------------------
// Thread-safe enqueue: push an ended command buffer (and optional callback) for later submission.
void TimelinePipeline::enqueue(VkCommandBuffer cmd, std::function<void()> onComplete)
{
  std::lock_guard<std::mutex> lock(m_submitMutex);
  m_submitQueue.push({cmd, std::move(onComplete)});
}

//--------------------------------------------------------------------------------------------------
// Main-thread poll: submit any queued buffers, then check which ones the GPU has finished.
// Completed entries get their callback called and their command buffer freed.
bool TimelinePipeline::poll()
{
  submitQueued();
  if(m_pending.empty())
    return false;

  uint64_t gpuValue = 0;
  NVVK_CHECK(vkGetSemaphoreCounterValue(m_device, m_timeline, &gpuValue));
  processCompleted(gpuValue);

  if(!m_pending.empty())
    return true;

  // Check the submit queue: a callback may have enqueued follow-up work (e.g. BLAS
  // compaction records a new command buffer and enqueues it). Without this check we would
  // return false ("done"), letting the renderer use resources before the chained GPU work
  // has even been submitted.
  std::lock_guard<std::mutex> lock(m_submitMutex);
  return !m_submitQueue.empty();
}

//--------------------------------------------------------------------------------------------------
// Blocking drain: submit and wait until every pending entry completes.
// Loops because callbacks may enqueue new work (e.g. compaction after BLAS build).
// Useful in headless mode to wait for all GPU work to complete
void TimelinePipeline::drain()
{
  while(true)
  {
    submitQueued();
    if(m_pending.empty())
    {
      std::lock_guard<std::mutex> lock(m_submitMutex);
      if(m_submitQueue.empty())
        return;
      continue;
    }

    const uint64_t      lastValue = m_nextValue - 1;
    VkSemaphoreWaitInfo waitInfo{
        .sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores    = &m_timeline,
        .pValues        = &lastValue,
    };
    NVVK_CHECK(vkWaitSemaphores(m_device, &waitInfo, UINT64_MAX));

    uint64_t gpuValue = 0;
    NVVK_CHECK(vkGetSemaphoreCounterValue(m_device, m_timeline, &gpuValue));
    processCompleted(gpuValue);
  }
}

//--------------------------------------------------------------------------------------------------
// Cancel and free everything. Call after vkQueueWaitIdle so no in-flight GPU work references
// the command buffers we are about to free.
void TimelinePipeline::clear()
{
  std::lock_guard<std::mutex> lock(m_submitMutex);
  while(!m_submitQueue.empty())
  {
    VkCommandBuffer cmd = m_submitQueue.front().cmd;
    m_submitQueue.pop();
    if(cmd != VK_NULL_HANDLE)
      vkFreeCommandBuffers(m_device, m_cmdPool, 1, &cmd);
  }
  for(auto& e : m_pending)
  {
    if(e.cmd != VK_NULL_HANDLE)
      vkFreeCommandBuffers(m_device, m_cmdPool, 1, &e.cmd);
  }
  m_pending.clear();
}

//--------------------------------------------------------------------------------------------------
// Drain the thread-safe submit queue and submit each buffer with a unique timeline signal value.
// Each submitted buffer becomes a Pending entry tracked until the GPU reaches its timeline value.
//
// Each submit only *signals* the timeline -- there is no wait semaphore. The GPU is free to
// overlap work across submissions. Same-queue ordering guarantees values are reached in order,
// so when value N is signaled, all entries up to N are complete. This gives us non-blocking
// CPU-side progress tracking without serializing GPU work.
void TimelinePipeline::submitQueued()
{
  std::vector<Queued> batch;
  {
    std::lock_guard<std::mutex> lock(m_submitMutex);
    while(!m_submitQueue.empty())
    {
      batch.push_back(std::move(m_submitQueue.front()));
      m_submitQueue.pop();
    }
  }
  for(auto& q : batch)
  {
    const uint64_t        signalValue = m_nextValue++;
    VkSemaphoreSubmitInfo signalInfo{
        .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = m_timeline,
        .value     = signalValue,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    };
    VkCommandBufferSubmitInfo cmdInfo{
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = q.cmd,
    };
    VkSubmitInfo2 submitInfo{
        .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .commandBufferInfoCount   = 1,
        .pCommandBufferInfos      = &cmdInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos    = &signalInfo,
    };
    NVVK_CHECK(vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE));
    m_pending.push_back({signalValue, q.cmd, std::move(q.onComplete)});
  }
}

//--------------------------------------------------------------------------------------------------
// Walk the pending list front-to-back (in timeline order) and process every entry whose
// timeline value the GPU has reached: run the callback, then free the command buffer.
// The callback runs only after the GPU has fully completed the associated command buffer,
// so it is safe to read back results (e.g. compaction sizes) or chain dependent work.
void TimelinePipeline::processCompleted(uint64_t gpuValue)
{
  while(!m_pending.empty() && m_pending.front().timelineValue <= gpuValue)
  {
    Pending entry = std::move(m_pending.front());
    m_pending.pop_front();
    if(entry.onComplete)
      entry.onComplete();  // execute the callback function
    if(entry.cmd != VK_NULL_HANDLE)
      vkFreeCommandBuffers(m_device, m_cmdPool, 1, &entry.cmd);
  }
}
