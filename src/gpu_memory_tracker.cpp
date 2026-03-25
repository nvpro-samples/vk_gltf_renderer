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

#include "gpu_memory_tracker.hpp"

#include <nvvk/gbuffers.hpp>

namespace nvvkgltf {

void GpuMemoryTracker::track(std::string_view category, const nvvk::GBuffer& gbuffer, uint32_t colorCount)
{
  VkExtent2D sz = gbuffer.getSize();
  if(sz.width == 0 || sz.height == 0)
    return;

  for(uint32_t i = 0; i < colorCount; ++i)
    track(category, gbuffer.getColorNvvkImage(i).allocation);

  if(gbuffer.getDepthImage() != VK_NULL_HANDLE)
    track(category, gbuffer.getDepthNvvkImage().allocation);
}

void GpuMemoryTracker::track(std::string_view category, VmaAllocation allocation)
{
  if(!m_alloc || !allocation)
    return;

  VmaAllocationInfo allocInfo;
  vmaGetAllocationInfo(*m_alloc, allocation, &allocInfo);

  std::lock_guard<std::mutex> lock(m_mutex);
  std::string                 categoryStr(category);
  auto&                       stats = m_stats[categoryStr];
  stats.currentBytes += allocInfo.size;
  stats.currentCount += 1;
  stats.totalAllocations += 1;

  // Update peaks
  if(stats.currentBytes > stats.peakBytes)
    stats.peakBytes = stats.currentBytes;
  if(stats.currentCount > stats.peakCount)
    stats.peakCount = stats.currentCount;
}

void GpuMemoryTracker::untrack(std::string_view category, const nvvk::GBuffer& gbuffer, uint32_t colorCount)
{
  VkExtent2D sz = gbuffer.getSize();
  if(sz.width == 0 || sz.height == 0)
    return;

  for(uint32_t i = 0; i < colorCount; ++i)
    untrack(category, gbuffer.getColorNvvkImage(i).allocation);

  if(gbuffer.getDepthImage() != VK_NULL_HANDLE)
    untrack(category, gbuffer.getDepthNvvkImage().allocation);
}

void GpuMemoryTracker::untrack(std::string_view category, VmaAllocation allocation)
{
  if(!m_alloc || !allocation)
    return;

  VmaAllocationInfo allocInfo;
  vmaGetAllocationInfo(*m_alloc, allocation, &allocInfo);

  std::lock_guard<std::mutex> lock(m_mutex);
  std::string                 categoryStr(category);
  auto                        it = m_stats.find(categoryStr);
  if(it == m_stats.end())
    return;

  auto& stats = it->second;
  if(stats.currentBytes >= allocInfo.size)
    stats.currentBytes -= allocInfo.size;
  if(stats.currentCount > 0)
    stats.currentCount -= 1;
  stats.totalDeallocations += 1;
}

nvvkgltf::GpuMemoryStats GpuMemoryTracker::getStats(std::string_view category) const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  std::string                 categoryStr(category);
  auto                        it = m_stats.find(categoryStr);
  if(it != m_stats.end())
    return it->second;
  return GpuMemoryStats{};  // Return empty stats if category doesn't exist
}

nvvkgltf::GpuMemoryStats GpuMemoryTracker::getTotalStats() const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  GpuMemoryStats              total{};
  for(const auto& [name, stats] : m_stats)
  {
    total.currentBytes += stats.currentBytes;
    total.currentCount += stats.currentCount;
    total.totalAllocations += stats.totalAllocations;
    total.totalDeallocations += stats.totalDeallocations;
    total.peakBytes += stats.peakBytes;
    total.peakCount += stats.peakCount;
  }
  return total;
}

std::vector<std::string> GpuMemoryTracker::getActiveCategories(CategorySortBy sortBy /*= CategorySortBy::eName*/,
                                                               bool           ascending /*= true*/) const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  std::vector<std::string>    categories;
  for(const auto& [name, stats] : m_stats)
  {
    if(stats.currentBytes > 0)
      categories.push_back(name);
  }

  // Sort based on specified criteria
  switch(sortBy)
  {
    case CategorySortBy::eName:
      std::sort(categories.begin(), categories.end(),
                [ascending](const std::string& a, const std::string& b) { return ascending ? (a < b) : (a > b); });
      break;
    case CategorySortBy::eCurrentBytes:
      std::sort(categories.begin(), categories.end(), [this, ascending](const std::string& a, const std::string& b) {
        return ascending ? (m_stats.at(a).currentBytes < m_stats.at(b).currentBytes) :
                           (m_stats.at(a).currentBytes > m_stats.at(b).currentBytes);
      });
      break;
    case CategorySortBy::eCurrentCount:
      std::sort(categories.begin(), categories.end(), [this, ascending](const std::string& a, const std::string& b) {
        return ascending ? (m_stats.at(a).currentCount < m_stats.at(b).currentCount) :
                           (m_stats.at(a).currentCount > m_stats.at(b).currentCount);
      });
      break;
  }

  return categories;
}

void GpuMemoryTracker::reset()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  for(auto& [name, stats] : m_stats)
  {
    stats.currentBytes = 0;
    stats.currentCount = 0;
    stats.peakBytes    = 0;
    stats.peakCount    = 0;
    // Keep stats.totalAllocations and stats.totalDeallocations for lifetime tracking
  }
}

void GpuMemoryTracker::resetAll()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_stats.clear();
}

}  // namespace nvvkgltf
