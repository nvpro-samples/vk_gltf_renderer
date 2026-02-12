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

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <vk_mem_alloc.h>

namespace nvvk {
class ResourceAllocator;
}

namespace nvvkgltf {

// Sort criteria for category listing
enum class CategorySortBy
{
  eName,          // Sort by category name (alphabetically)
  eCurrentBytes,  // Sort by current bytes allocated (largest first)
  eCurrentCount   // Sort by current allocation count (largest first)
};

// Statistics for a single memory category
struct GpuMemoryStats
{
  uint64_t currentBytes       = 0;  // Currently allocated bytes
  uint32_t currentCount       = 0;  // Number of active allocations
  uint64_t totalAllocations   = 0;  // Lifetime allocation count
  uint64_t totalDeallocations = 0;  // Lifetime deallocation count
  uint64_t peakBytes          = 0;  // High water mark for bytes
  uint32_t peakCount          = 0;  // Maximum concurrent allocations
};

// GPU memory tracker for monitoring allocations
class GpuMemoryTracker
{
public:
  GpuMemoryTracker() = default;

  // Initialize with resource allocator to query allocation sizes
  void init(nvvk::ResourceAllocator* alloc) { m_alloc = alloc; }

  // Track an allocation - queries VMA for actual size
  void track(std::string_view category, VmaAllocation allocation)
  {
    if(!m_alloc || !allocation)
      return;

    VmaAllocationInfo allocInfo;
    vmaGetAllocationInfo(*m_alloc, allocation, &allocInfo);

    std::string categoryStr(category);
    auto&       stats = m_stats[categoryStr];
    stats.currentBytes += allocInfo.size;
    stats.currentCount += 1;
    stats.totalAllocations += 1;

    // Update peaks
    if(stats.currentBytes > stats.peakBytes)
      stats.peakBytes = stats.currentBytes;
    if(stats.currentCount > stats.peakCount)
      stats.peakCount = stats.currentCount;
  }

  // Untrack a deallocation - queries VMA for actual size
  void untrack(std::string_view category, VmaAllocation allocation)
  {
    if(!m_alloc || !allocation)
      return;

    VmaAllocationInfo allocInfo;
    vmaGetAllocationInfo(*m_alloc, allocation, &allocInfo);

    std::string categoryStr(category);
    auto        it = m_stats.find(categoryStr);
    if(it == m_stats.end())
      return;

    auto& stats = it->second;
    if(stats.currentBytes >= allocInfo.size)
      stats.currentBytes -= allocInfo.size;
    if(stats.currentCount > 0)
      stats.currentCount -= 1;
    stats.totalDeallocations += 1;
  }

  // Get statistics for a specific category
  GpuMemoryStats getStats(std::string_view category) const
  {
    std::string categoryStr(category);
    auto        it = m_stats.find(categoryStr);
    if(it != m_stats.end())
      return it->second;
    return GpuMemoryStats{};  // Return empty stats if category doesn't exist
  }

  // Get statistics for all categories combined
  GpuMemoryStats getTotalStats() const
  {
    GpuMemoryStats total{};
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

  // Get all category names that have non-zero current bytes (for UI iteration)
  // sortBy: How to sort the returned categories
  // ascending: true for ascending order, false for descending (default depends on sort type)
  std::vector<std::string> getActiveCategories(CategorySortBy sortBy = CategorySortBy::eName, bool ascending = true) const
  {
    std::vector<std::string> categories;
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

  // Reset statistics (typically called when loading a new scene)
  // Keeps total allocation/deallocation counts but resets current and peak
  void reset()
  {
    for(auto& [name, stats] : m_stats)
    {
      stats.currentBytes = 0;
      stats.currentCount = 0;
      stats.peakBytes    = 0;
      stats.peakCount    = 0;
      // Keep stats.totalAllocations and stats.totalDeallocations for lifetime tracking
    }
  }

  // Complete reset - clears all statistics including totals
  void resetAll() { m_stats.clear(); }

private:
  nvvk::ResourceAllocator*                        m_alloc = nullptr;
  std::unordered_map<std::string, GpuMemoryStats> m_stats;
};

}  // namespace nvvkgltf
