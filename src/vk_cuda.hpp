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


/***********************************************************************

  This file is contains many functions to share resources between
  Vulkan and CUDA.

  Besides Semaphore, all Vulkan resources must be created using the 
  export flag. Then, this Vulkan resource can be pass in one of the
  function below, to create its CUDA counterpart. 

***********************************************************************/

#pragma once

#include <array>
#include <unordered_map>
#include <atomic>
#include <cassert>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <volk.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvvk/resource_allocator.hpp>


#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    cudaError_t res = call;                                                                                            \
    if(res != cudaSuccess)                                                                                             \
    {                                                                                                                  \
      LOGE("Cuda call (%s) failed with code %d (%s:%d)\n", #call, res, __FILE__, __LINE__);                            \
      assert(res == cudaSuccess);                                                                                      \
    }                                                                                                                  \
  } while(false)

#endif  // CUDA_CHECK

namespace vkcuda {

//------------------------------------------------------------------------------------------------
// Memory object manager for Vulkan-CUDA interop with reference counting
// Manages CUDA external memory objects with automatic cleanup when reference count reaches zero
class MemoryObjectManager
{
public:
  ~MemoryObjectManager() { assert(m_importedMemoryObjects.empty() && "Missing to call clear()"); }

  // Acquire an external memory object for the given VMA allocation
  // Returns the CUDA external memory handle
  cudaExternalMemory_t acquireExternalMemory(VmaAllocation allocation, nvvk::ResourceAllocator& allocator)
  {
    VmaAllocationInfo2 allocationInfo2{};
    vmaGetAllocationInfo2(allocator, allocation, &allocationInfo2);

    VkDeviceMemory deviceMemory = allocationInfo2.allocationInfo.deviceMemory;

    // Check if already imported
    auto it = m_importedMemoryObjects.find(deviceMemory);
    if(it != m_importedMemoryObjects.end())
    {
      // Increment reference count
      auto refIt = m_refCounts.find(it->second);
      if(refIt != m_refCounts.end())
      {
        refIt->second.fetch_add(1);
      }
      return it->second;
    }

    // Create new external memory object
    cudaExternalMemory_t extMemory{};

#ifdef WIN32
    HANDLE handle;
    vmaGetMemoryWin32Handle(allocator, allocation, nullptr, &handle);
    // Store the handle for later cleanup
    m_win32Handles[extMemory] = handle;
#else
    VkMemoryGetFdInfoKHR getInfo{};
    getInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    getInfo.memory     = deviceMemory;
    getInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    int fd{-1};
    vkGetMemoryFdKHR(allocator.getDevice(), &getInfo, &fd);
#endif

    // Import the entire memory block using blockSize
    cudaExternalMemoryHandleDesc cuda_ext_mem_handle_desc{};
    cuda_ext_mem_handle_desc.size = allocationInfo2.blockSize;
#ifdef WIN32
    cuda_ext_mem_handle_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
    cuda_ext_mem_handle_desc.handle.win32.handle = handle;
#else
    cuda_ext_mem_handle_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    cuda_ext_mem_handle_desc.handle.fd = fd;
#endif

    CUDA_CHECK(cudaImportExternalMemory(&extMemory, &cuda_ext_mem_handle_desc));

    // Store the mapping and initialize reference count
    m_importedMemoryObjects[deviceMemory] = extMemory;
    m_refCounts[extMemory]                = 1;

    return extMemory;
  }

  // Release an external memory object (decrement reference count)
  void releaseExternalMemory(cudaExternalMemory_t extMemory)
  {
    auto it = m_refCounts.find(extMemory);
    if(it != m_refCounts.end())
    {
      uint64_t newCount = it->second.fetch_sub(1) - 1;
      if(newCount == 0)
      {
        // Reference count reached zero, destroy the CUDA external memory
        cudaDestroyExternalMemory(extMemory);

#ifdef WIN32
        // Close the Windows handle if it exists
        auto handleIt = m_win32Handles.find(extMemory);
        if(handleIt != m_win32Handles.end())
        {
          CloseHandle(handleIt->second);
          m_win32Handles.erase(handleIt);
        }
#endif

        // Remove from both maps
        m_refCounts.erase(it);

        // Find and remove from importedMemoryObjects map
        for(auto memIt = m_importedMemoryObjects.begin(); memIt != m_importedMemoryObjects.end(); ++memIt)
        {
          if(memIt->second == extMemory)
          {
            m_importedMemoryObjects.erase(memIt);
            break;
          }
        }
      }
    }
  }

  // Clear all memory objects (useful for cleanup)
  void clear()
  {
    for(auto& [extMemory, refCount] : m_refCounts)
    {
      if(extMemory != nullptr)
      {
        cudaDestroyExternalMemory(extMemory);
      }
    }

#ifdef WIN32
    // Close all Windows handles
    for(auto& [extMemory, handle] : m_win32Handles)
    {
      if(handle != nullptr)
      {
        CloseHandle(handle);
      }
    }
    m_win32Handles.clear();
#endif

    m_importedMemoryObjects.clear();
    m_refCounts.clear();
  }

  // Remove a specific memory object by device memory
  void remove(VkDeviceMemory deviceMemory)
  {
    auto it = m_importedMemoryObjects.find(deviceMemory);
    if(it != m_importedMemoryObjects.end())
    {
      releaseExternalMemory(it->second);
    }
  }

private:
  std::unordered_map<VkDeviceMemory, cudaExternalMemory_t>       m_importedMemoryObjects;
  std::unordered_map<cudaExternalMemory_t, std::atomic_uint64_t> m_refCounts;
#ifdef WIN32
  std::unordered_map<cudaExternalMemory_t, HANDLE> m_win32Handles;
#endif
};

// Meyer's Singleton accessor for the global CUDA memory object manager
// Returns the single instance shared across all translation units
inline MemoryObjectManager& getCudaMemoryObjectManager()
{
  static MemoryObjectManager instance;
  return instance;
}

// Utility function to clear the global CUDA memory object manager
// Call this when cleaning up the application or when you want to free all cached memory objects
inline void clearCudaMemoryObjectManager()
{
  getCudaMemoryObjectManager().clear();
}

// -- Initialization --

// Setting the CUDA device to match the Vulkan physical device
// This function will set the CUDA device to the first device that matches the Vulkan physical device.
cudaError_t setCudaDevice(VkPhysicalDevice physicalDevice);


// -- Resources --

// Buffers: holding the buffer for CUDA interop
// Note: cuMemory is managed by MemoryObjectManager and should not be directly destroyed
struct Buffer
{
  void*                cuPtr    = nullptr;
  cudaExternalMemory_t cuMemory = nullptr;  // Reference to cached memory (owned by manager)
};


// Images : holding the image for CUDA interop
// Note: cuMemory is managed by MemoryObjectManager and should not be directly destroyed
struct Image
{
  cudaMipmappedArray_t cuImage   = nullptr;
  cudaSurfaceObject_t  cuSurface = 0ULL;
  cudaExternalMemory_t cuMemory  = nullptr;  // Reference to cached memory (owned by manager)
};

// CUDA-only buffer (no Vulkan interop, simple CUDA memory allocation)
struct CudaBuffer
{
  CUdeviceptr ptr  = 0;
  size_t      size = 0;

  cudaError_t allocate(size_t bufferSize)
  {
    size = bufferSize;
    return cudaMalloc((void**)&ptr, size);
  }

  void free()
  {
    if(ptr != 0)
    {
      cudaFree((void*)ptr);
      ptr  = 0;
      size = 0;
    }
  }
};

// Helper to create a buffer  or image with the export flag
Buffer createCudaBuffer(nvvk::ResourceAllocator& allocator, nvvk::Buffer& buffer);
Image  createCudaImage2D(nvvk::ResourceAllocator& allocator, nvvk::Image image, const VkImageCreateInfo& imgInfo);
void   destroyCudaBuffer(Buffer& buffer);
void   destroyCudaImage(Image& image);


// -- Semaphore --

// Semaphore : holding the semaphore for CUDA interop
struct Semaphore
{
  VkSemaphore             vk{};  // Vulkan
  cudaExternalSemaphore_t cu{};  // CUDA version
#ifdef WIN32
  HANDLE handle{INVALID_HANDLE_VALUE};
#else
  int handle{-1};
#endif
};

// Create a Vulkan/Cuda semaphore from a Vk semaphore, timeline is for Vulkan timeline semaphore
vkcuda::Semaphore                     createCudaSemaphore(VkDevice device, bool timeline = false);
void                                  destroySemaphore(VkDevice device, Semaphore& cudaSemaphore);
VkSemaphoreSubmitInfoKHR              cudaSignalSemaphore(uint64_t fenceValue, Semaphore& cudaSemaphore);
VkExternalSemaphoreHandleTypeFlagBits getSemaphoreExportHandleType();


// ---  Utilities ---
cudaChannelFormatDesc getCudaChannelFormat(VkFormat format);

}  // namespace vkcuda
