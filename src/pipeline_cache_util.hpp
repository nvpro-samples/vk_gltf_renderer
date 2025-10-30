/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <filesystem>
#include <vulkan/vulkan_core.h>

//--------------------------------------------------------------------------------------------------
// Pipeline Cache Manager
//
// Utility class for managing Vulkan pipeline cache with file persistence.
// This class simplifies the creation, loading, and saving of Vulkan pipeline caches.
// Pipeline caches can significantly speed up pipeline creation on subsequent runs.
//--------------------------------------------------------------------------------------------------

namespace nvvk {

class PipelineCacheManager
{
public:
  PipelineCacheManager() = default;
  ~PipelineCacheManager() { deinit(); }

  // Non-copyable
  PipelineCacheManager(const PipelineCacheManager&)            = delete;
  PipelineCacheManager& operator=(const PipelineCacheManager&) = delete;

  // Create a pipeline cache, optionally loading from a file.
  // Returns VK_SUCCESS on success, error code otherwise.
  VkResult init(VkDevice device, const std::filesystem::path& cacheFilePath = "pipeline_cache.bin");

  // Save the cache to disk and destroy the cache object
  void deinit();

  // Save the current cache to disk.
  // Returns true on success, false on failure.
  bool save();

  // Get the VkPipelineCache handle.
  // Returns the pipeline cache handle, or VK_NULL_HANDLE if not initialized.
  VkPipelineCache getCache() const { return m_cache; }

  // Implicit conversion to VkPipelineCache for convenience
  operator VkPipelineCache() const { return m_cache; }

private:
  VkDevice              m_device{VK_NULL_HANDLE};
  VkPipelineCache       m_cache{VK_NULL_HANDLE};
  std::filesystem::path m_cacheFilePath;
};

}  // namespace nvvk
