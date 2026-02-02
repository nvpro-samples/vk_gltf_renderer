/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pipeline_cache_util.hpp"
#include <fstream>
#include <vector>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvutils/logger.hpp>

namespace nvvk {

VkResult PipelineCacheManager::init(VkDevice device, const std::filesystem::path& cacheFilePath)
{
  m_device        = device;
  m_cacheFilePath = cacheFilePath;

  std::vector<char> cacheData;

  // Try to load existing cache data from disk
  if(std::filesystem::exists(m_cacheFilePath))
  {
    try
    {
      std::ifstream file(m_cacheFilePath, std::ios::binary | std::ios::ate);
      if(file.is_open())
      {
        size_t fileSize = static_cast<size_t>(file.tellg());
        if(fileSize > 0)
        {
          cacheData.resize(fileSize);
          file.seekg(0);
          file.read(cacheData.data(), fileSize);
          file.close();
          LOGI("Loaded pipeline cache from %s (%zu bytes)\n", m_cacheFilePath.string().c_str(), fileSize);
        }
      }
    }
    catch(const std::exception& e)
    {
      LOGW("Failed to load pipeline cache: %s\n", e.what());
      cacheData.clear();
    }
  }

  // Create the pipeline cache
  VkPipelineCacheCreateInfo cacheInfo{
      .sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
      .initialDataSize = cacheData.size(),
      .pInitialData    = cacheData.empty() ? nullptr : cacheData.data(),
  };

  VkResult result = vkCreatePipelineCache(m_device, &cacheInfo, nullptr, &m_cache);
  if(result == VK_SUCCESS)
  {
    NVVK_DBG_NAME(m_cache);
    if(cacheData.empty())
    {
      LOGI("Created new pipeline cache\n");
    }
    else
    {
      LOGI("Created pipeline cache with existing data\n");
    }
  }
  else
  {
    LOGE("Failed to create pipeline cache: %d\n", result);
  }

  return result;
}

void PipelineCacheManager::deinit()
{
  if(m_cache != VK_NULL_HANDLE)
  {
    save();
    vkDestroyPipelineCache(m_device, m_cache, nullptr);
    m_cache  = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
  }
}

bool PipelineCacheManager::save()
{
  if(m_cache == VK_NULL_HANDLE)
    return false;

  // Get the size of the cache data
  size_t cacheSize = 0;
  if(vkGetPipelineCacheData(m_device, m_cache, &cacheSize, nullptr) != VK_SUCCESS)
  {
    LOGW("Failed to get pipeline cache size\n");
    return false;
  }

  if(cacheSize == 0)
  {
    LOGI("Pipeline cache is empty, not saving\n");
    return true;  // Not an error, just nothing to save
  }

  // Allocate memory and get the cache data
  std::vector<char> cacheData(cacheSize);
  if(vkGetPipelineCacheData(m_device, m_cache, &cacheSize, cacheData.data()) != VK_SUCCESS)
  {
    LOGW("Failed to retrieve pipeline cache data\n");
    return false;
  }

  // Write to disk
  try
  {
    std::ofstream file(m_cacheFilePath, std::ios::binary);
    if(file.is_open())
    {
      file.write(cacheData.data(), cacheSize);
      file.close();
      LOGI("Saved pipeline cache to %s (%zu bytes)\n", m_cacheFilePath.string().c_str(), cacheSize);
      return true;
    }
    else
    {
      LOGW("Failed to open pipeline cache file for writing: %s\n", m_cacheFilePath.string().c_str());
      return false;
    }
  }
  catch(const std::exception& e)
  {
    LOGW("Failed to save pipeline cache: %s\n", e.what());
    return false;
  }
}


//--------------------------------------------------------------------------------------------------
// Usage example
//--------------------------------------------------------------------------------------------------
static void usage_PipelineCacheManager()
{
  VkDevice device = nullptr;  // EX: get the device from the app (m_app->getDevice())

  nvvk::PipelineCacheManager pipelineCacheManager;

  // Initialize the pipeline cache, loading from file if it exists
  pipelineCacheManager.init(device, "pipeline_cache.bin");

  // Use the cache when creating pipelines
  VkPipelineCache cache = pipelineCacheManager.getCache();

  VkGraphicsPipelineCreateInfo pipelineInfo = {};  // EX: fill in pipeline create info
  VkPipeline                   pipeline     = VK_NULL_HANDLE;
  vkCreateGraphicsPipelines(device, cache, 1, &pipelineInfo, nullptr, &pipeline);

  // The cache can also be used implicitly via conversion operator
  vkCreateGraphicsPipelines(device, pipelineCacheManager, 1, &pipelineInfo, nullptr, &pipeline);

  // Save the cache manually (optional, as it's saved automatically on deinit)
  pipelineCacheManager.save();

  // Cleanup - this will save the cache to disk and destroy it
  pipelineCacheManager.deinit();
}

}  // namespace nvvk
