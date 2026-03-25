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

//
// Tracks GPU memory allocations by named category (e.g. "Geometry",
// "Images", "BLAS"). Records current and peak byte counts per category
// so the application can display memory usage breakdowns and detect
// allocation regressions during development.
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <vk_mem_alloc.h>

namespace nvvk {
class ResourceAllocator;
class GBuffer;
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

// GPU memory tracker for monitoring allocations.
// Thread-safe: track(), untrack(), getStats(), getTotalStats(), getActiveCategories(),
// reset(), and resetAll() may be called concurrently from multiple threads.
class GpuMemoryTracker
{
public:
  GpuMemoryTracker() = default;

  // Initialize with resource allocator to query allocation sizes
  void init(nvvk::ResourceAllocator* alloc) { m_alloc = alloc; }

  // Track an allocation - queries VMA for actual size
  void track(std::string_view category, VmaAllocation allocation);

  // Untrack a deallocation - queries VMA for actual size
  void untrack(std::string_view category, VmaAllocation allocation);

  // Track all allocations (color + depth) in a GBuffer.
  // colorCount: number of color attachments (GBuffer doesn't expose this).
  void track(std::string_view category, const nvvk::GBuffer& gbuffer, uint32_t colorCount);

  // Untrack all allocations (color + depth) in a GBuffer.
  void untrack(std::string_view category, const nvvk::GBuffer& gbuffer, uint32_t colorCount);

  // Get statistics for a specific category
  GpuMemoryStats getStats(std::string_view category) const;

  // Get statistics for all categories combined
  GpuMemoryStats getTotalStats() const;

  // Get all category names that have non-zero current bytes (for UI iteration)
  // sortBy: How to sort the returned categories
  // ascending: true for ascending order, false for descending (default depends on sort type)
  std::vector<std::string> getActiveCategories(CategorySortBy sortBy = CategorySortBy::eName, bool ascending = true) const;

  // Reset statistics (typically called when loading a new scene)
  // Keeps total allocation/deallocation counts but resets current and peak
  void reset();

  // Complete reset - clears all statistics including totals
  void resetAll();

private:
  nvvk::ResourceAllocator*                        m_alloc = nullptr;
  std::unordered_map<std::string, GpuMemoryStats> m_stats;
  mutable std::mutex                              m_mutex;
};

}  // namespace nvvkgltf
