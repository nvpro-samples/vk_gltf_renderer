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

#include "vk_cuda.hpp"

#ifndef WIN32
#include <unistd.h>
#endif
#include "nvutils/logger.hpp"
#include "nvvk/check_error.hpp"
#include "nvvk/debug_util.hpp"

#ifdef WIN32
#include <delayimp.h>
#endif

/***********************************************************************

  This file is contains many functions to share resources between
  Vulkan and CUDA.

  Besides Semaphore, all Vulkan resources must be created using the 
  export flag. Then, this Vulkan resource can be pass in one of the
  function below, to create its CUDA counterpart. 

***********************************************************************/

//--------------------------------------------------------------------------------------------------
// Check if CUDA runtime is available (DLL can be loaded)
// This safely handles the delay-load case where cudart64_XX.dll may not be present
//
bool vkcuda::isCudaRuntimeAvailable()
{
  static bool s_checked   = false;
  static bool s_available = false;

  if(s_checked)
    return s_available;

  s_checked = true;

#ifdef WIN32
  // On Windows, use SEH to catch the delay-load exception if the DLL is not found
  __try
  {
    int         deviceCount = 0;
    cudaError_t err         = cudaGetDeviceCount(&deviceCount);
    s_available             = (err == cudaSuccess || err == cudaErrorNoDevice);
  }
  __except(GetExceptionCode() == VcppException(ERROR_SEVERITY_ERROR, ERROR_MOD_NOT_FOUND) ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH)
  {
    // Delay-load failed - DLL not found
    LOGW("CUDA runtime DLL not found. OptiX denoiser will be unavailable.\n");
    s_available = false;
  }
#else
  // On Linux, delay-load is not typically used, so just try to call CUDA
  int         deviceCount = 0;
  cudaError_t err         = cudaGetDeviceCount(&deviceCount);
  s_available             = (err == cudaSuccess || err == cudaErrorNoDevice);
#endif

  return s_available;
}

VkExternalSemaphoreHandleTypeFlagBits vkcuda::getSemaphoreExportHandleType()
{
#ifdef WIN32
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}

static VkSemaphore createVkSemaphore(VkDevice device, bool timeline)
{
  VkSemaphore semaphore{};

  VkExportSemaphoreCreateInfo export_semaphore_info{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR};
  export_semaphore_info.handleTypes = vkcuda::getSemaphoreExportHandleType();

  VkSemaphoreTypeCreateInfo semaphore_type_info{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  semaphore_type_info.semaphoreType = timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY;
  semaphore_type_info.pNext         = &export_semaphore_info;

  VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  semaphore_info.pNext = &semaphore_type_info;

  NVVK_CHECK(vkCreateSemaphore(device, &semaphore_info, nullptr, &semaphore));
  return semaphore;
}

vkcuda::Semaphore vkcuda::createCudaSemaphore(VkDevice device, bool timeline)
{
  Semaphore semaphore;

  semaphore.vk = createVkSemaphore(device, timeline);

#ifdef WIN32
  VkSemaphoreGetWin32HandleInfoKHR handle_info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
  handle_info.handleType = vkcuda::getSemaphoreExportHandleType();
  handle_info.semaphore  = semaphore.vk;
  vkGetSemaphoreWin32HandleKHR(device, &handle_info, &semaphore.handle);
#else
  VkSemaphoreGetFdInfoKHR handle_info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
  handle_info.handleType = getSemaphoreExportHandleType();
  handle_info.semaphore  = semaphore.vk;
  vkGetSemaphoreFdKHR(device, &handle_info, &semaphore.handle);
#endif

  cudaExternalSemaphoreHandleDesc external_semaphore_handle_desc{};
  std::memset(&external_semaphore_handle_desc, 0, sizeof(external_semaphore_handle_desc));
  external_semaphore_handle_desc.flags = 0;
#ifdef WIN32
  external_semaphore_handle_desc.type =
      timeline ? cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 : cudaExternalSemaphoreHandleTypeOpaqueWin32;
  external_semaphore_handle_desc.handle.win32.handle = static_cast<void*>(semaphore.handle);
#else
  external_semaphore_handle_desc.type =
      timeline ? cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd : cudaExternalSemaphoreHandleTypeOpaqueFd;
  external_semaphore_handle_desc.handle.fd = semaphore.handle;
#endif

  CUDA_CHECK(cudaImportExternalSemaphore(&semaphore.cu, &external_semaphore_handle_desc));

  return semaphore;
}

void vkcuda::destroySemaphore(VkDevice device, Semaphore& cudaSemaphore)
{
  vkDestroySemaphore(device, cudaSemaphore.vk, nullptr);
  cudaSemaphore.vk = VK_NULL_HANDLE;
#ifdef WIN32
  if(cudaSemaphore.handle != nullptr)
  {
    CloseHandle(cudaSemaphore.handle);
    cudaSemaphore.handle = nullptr;
  }
#else
  if(cudaSemaphore.handle != -1)
  {
    close(cudaSemaphore.handle);
    cudaSemaphore.handle = -1;
  }
#endif
}


VkSemaphoreSubmitInfoKHR vkcuda::cudaSignalSemaphore(uint64_t fenceValue, Semaphore& cudaSemaphore)
{
  // CUDA is signaling, then Vulkan will wait
  cudaExternalSemaphoreSignalParams sig_params{};
  sig_params.flags              = 0;
  sig_params.params.fence.value = fenceValue;
  cudaSignalExternalSemaphoresAsync(&cudaSemaphore.cu, &sig_params, 1);

  // Adding a wait semaphore to the application, such that the frame command buffer,
  // will wait for the end of CUDA execution before executing the command buffer.
  VkSemaphoreSubmitInfoKHR wait_semaphore{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR};
  wait_semaphore.semaphore = cudaSemaphore.vk;
  wait_semaphore.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
  wait_semaphore.value     = fenceValue;
  return wait_semaphore;
}

vkcuda::Image vkcuda::createCudaImage2D(nvvk::ResourceAllocator& allocator, nvvk::Image image, const VkImageCreateInfo& imgInfo)
{
  vkcuda::Image imgCuda;

  // Get allocation info to determine offset
  VmaAllocationInfo2 allocationInfo2{};
  vmaGetAllocationInfo2(allocator, image.allocation, &allocationInfo2);

  // Use reference-counted memory object manager to acquire or reuse the external memory
  // This imports the entire memory block using blockSize
  imgCuda.cuMemory = getCudaMemoryObjectManager().acquireExternalMemory(image.allocation, allocator);

  cudaExternalMemoryMipmappedArrayDesc mipmapDesc = {};
  mipmapDesc.extent.width                         = imgInfo.extent.width;
  mipmapDesc.extent.height                        = imgInfo.extent.height;
  mipmapDesc.extent.depth                         = 0;
  mipmapDesc.flags                                = cudaArrayDefault;
  mipmapDesc.formatDesc                           = getCudaChannelFormat(imgInfo.format);
  mipmapDesc.numLevels                            = 1;

  // Set the offset for the image within the device memory
  mipmapDesc.offset = allocationInfo2.allocationInfo.offset;

  CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&imgCuda.cuImage, imgCuda.cuMemory, &mipmapDesc));

  // Take only the first Layer, otherwise not working for non-power of two
  cudaArray_t levelArray;
  CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, imgCuda.cuImage, 0));

  cudaSurfaceObject_t SurfObject = {};
  cudaResourceDesc    ResDesc    = {};
  ResDesc.res.array.array        = levelArray;
  ResDesc.resType                = cudaResourceTypeArray;
  CUDA_CHECK(cudaCreateSurfaceObject(&imgCuda.cuSurface, &ResDesc));

  return imgCuda;
}

void vkcuda::destroyCudaImage(vkcuda::Image& image)
{
  if(image.cuSurface != 0ULL)
  {
    CUDA_CHECK(cudaDestroySurfaceObject(image.cuSurface));
    image.cuSurface = 0ULL;
  }

  // Release the external memory reference (managed by MemoryObjectManager)
  if(image.cuMemory != nullptr)
  {
    getCudaMemoryObjectManager().releaseExternalMemory(image.cuMemory);
    image.cuMemory = nullptr;
  }
}

cudaError_t vkcuda::setCudaDevice(VkPhysicalDevice physicalDevice)
{
  cudaError_t cudaStatus = cudaSuccess;

  // One way to do this is to match up PCIe IDs.
  // First, get the Vulkan PCI bus ID:
  VkPhysicalDevicePCIBusInfoPropertiesEXT pciProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
  VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &pciProperties};
  vkGetPhysicalDeviceProperties2(physicalDevice, &properties);
  const uint32_t vkPciBus = pciProperties.pciBus;
  // Then iterate over CUDA devices and try to find a matching one.
  int numCudaDevices = 0;
  cudaStatus         = cudaGetDeviceCount(&numCudaDevices);
  if(cudaSuccess != cudaStatus)
  {
    LOGW("cudaGetDeviceCount failed! CUDA might not be enabled or supported on this system.\n");
    return cudaStatus;
  }

  for(int i = 0; i < numCudaDevices; i++)
  {
    int cudaPciBus = 0;
    cudaStatus     = cudaDeviceGetAttribute(&cudaPciBus, cudaDevAttrPciBusId, i);
    if(cudaSuccess != cudaStatus)
    {
      LOGW("Could not query cudaDevAttrPciBusId for CUDA device %i.\n", i);
      continue;
    }
    if(vkPciBus == cudaPciBus)
    {
      // This is our device! Select it:
      LOGI("Selected CUDA device %i.\n", i);
      cudaStatus = cudaSetDevice(i);
      if(cudaSuccess != cudaStatus)
      {
        LOGE("Calling cudaSetDevice() failed!\n");
      }
      return cudaStatus;
    }
  }
  LOGW("Could not find a corresponding CUDA device for the Vulkan device on PCIe bus %u.\n", vkPciBus);
  return cudaErrorNoDevice;
}

vkcuda::Buffer vkcuda::createCudaBuffer(nvvk::ResourceAllocator& allocator, nvvk::Buffer& buffer)
{
  vkcuda::Buffer cudaBuffer{};

  // Get allocation info to determine offset
  VmaAllocationInfo2 allocationInfo2{};
  vmaGetAllocationInfo2(allocator, buffer.allocation, &allocationInfo2);

  // Use reference-counted memory object manager to acquire or reuse the external memory
  // This imports the entire memory block using blockSize
  cudaBuffer.cuMemory = getCudaMemoryObjectManager().acquireExternalMemory(buffer.allocation, allocator);

  // The external memory object is created, now we can create the external memory buffer object
  // The offset accounts for where the buffer is located within the device memory allocation
  cudaExternalMemoryBufferDesc cudaExtBufferDesc{};
  cudaExtBufferDesc.size   = buffer.bufferSize;                      // Size of the buffer
  cudaExtBufferDesc.offset = allocationInfo2.allocationInfo.offset;  // Offset within the imported memory

  CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&cudaBuffer.cuPtr, cudaBuffer.cuMemory, &cudaExtBufferDesc));

  return cudaBuffer;
}

void vkcuda::destroyCudaBuffer(vkcuda::Buffer& buffer)
{
  // Release the external memory reference (managed by MemoryObjectManager)
  if(buffer.cuMemory != nullptr)
  {
    getCudaMemoryObjectManager().releaseExternalMemory(buffer.cuMemory);
    buffer.cuMemory = nullptr;
  }
}


cudaChannelFormatDesc vkcuda::getCudaChannelFormat(VkFormat format)
{
  switch(format)
  {
    case VK_FORMAT_R8_UNORM:
      return {8, 0, 0, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R8_SNORM:
      return {8, 0, 0, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R8_UINT:
      return {8, 0, 0, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R8_SINT:
      return {8, 0, 0, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R8G8_UNORM:
      return {8, 8, 0, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R8G8_SNORM:
      return {8, 8, 0, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R8G8_UINT:
      return {8, 8, 0, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R8G8_SINT:
      return {8, 8, 0, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R8G8B8_UNORM:
      return {8, 8, 8, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R8G8B8_SNORM:
      return {8, 8, 8, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R8G8B8_UINT:
      return {8, 8, 8, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R8G8B8_SINT:
      return {8, 8, 8, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R8G8B8A8_UNORM:
      return {8, 8, 8, 8, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R8G8B8A8_SNORM:
      return {8, 8, 8, 8, cudaChannelFormatKindSigned};
    case VK_FORMAT_R8G8B8A8_UINT:
      return {8, 8, 8, 8, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R8G8B8A8_SINT:
      return {8, 8, 8, 8, cudaChannelFormatKindSigned};
    case VK_FORMAT_B8G8R8A8_UNORM:
      return {8, 8, 8, 8, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_B8G8R8A8_SNORM:
      return {8, 8, 8, 8, cudaChannelFormatKindSigned};
    case VK_FORMAT_B8G8R8A8_UINT:
      return {8, 8, 8, 8, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_B8G8R8A8_SINT:
      return {8, 8, 8, 8, cudaChannelFormatKindSigned};
    case VK_FORMAT_R16_UNORM:
      return {16, 0, 0, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R16_SNORM:
      return {16, 0, 0, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R16_SFLOAT:
      return {16, 0, 0, 0, cudaChannelFormatKindFloat};
    case VK_FORMAT_R16G16_UNORM:
      return {16, 16, 0, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R16G16_SNORM:
      return {16, 16, 0, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R16G16_SFLOAT:
      return {16, 16, 0, 0, cudaChannelFormatKindFloat};
    case VK_FORMAT_R16G16B16_UNORM:
      return {16, 16, 16, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R16G16B16_SNORM:
      return {16, 16, 16, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R16G16B16_SFLOAT:
      return {16, 16, 16, 0, cudaChannelFormatKindFloat};
    case VK_FORMAT_R16G16B16A16_UNORM:
      return {16, 16, 16, 16, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R16G16B16A16_SNORM:
      return {16, 16, 16, 16, cudaChannelFormatKindSigned};
    case VK_FORMAT_R16G16B16A16_SFLOAT:
      return {16, 16, 16, 16, cudaChannelFormatKindFloat};
    case VK_FORMAT_R32_UINT:
      return {32, 0, 0, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R32_SINT:
      return {32, 0, 0, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R32_SFLOAT:
      return {32, 0, 0, 0, cudaChannelFormatKindFloat};
    case VK_FORMAT_R32G32_UINT:
      return {32, 32, 0, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R32G32_SINT:
      return {32, 32, 0, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R32G32_SFLOAT:
      return {32, 32, 0, 0, cudaChannelFormatKindFloat};
    case VK_FORMAT_R32G32B32_UINT:
      return {32, 32, 32, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R32G32B32_SINT:
      return {32, 32, 32, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R32G32B32_SFLOAT:
      return {32, 32, 32, 0, cudaChannelFormatKindFloat};
    case VK_FORMAT_R32G32B32A32_UINT:
      return {32, 32, 32, 32, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R32G32B32A32_SINT:
      return {32, 32, 32, 32, cudaChannelFormatKindSigned};
    case VK_FORMAT_R32G32B32A32_SFLOAT:
      return {32, 32, 32, 32, cudaChannelFormatKindFloat};
    case VK_FORMAT_R64_UINT:
      return {64, 0, 0, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R64_SINT:
      return {64, 0, 0, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R64_SFLOAT:
      return {64, 0, 0, 0, cudaChannelFormatKindFloat};
    case VK_FORMAT_R64G64_UINT:
      return {64, 64, 0, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R64G64_SINT:
      return {64, 64, 0, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R64G64_SFLOAT:
      return {64, 64, 0, 0, cudaChannelFormatKindFloat};
    case VK_FORMAT_R64G64B64_UINT:
      return {64, 64, 64, 0, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R64G64B64_SINT:
      return {64, 64, 64, 0, cudaChannelFormatKindSigned};
    case VK_FORMAT_R64G64B64_SFLOAT:
      return {64, 64, 64, 0, cudaChannelFormatKindFloat};
    case VK_FORMAT_R64G64B64A64_UINT:
      return {64, 64, 64, 64, cudaChannelFormatKindUnsigned};
    case VK_FORMAT_R64G64B64A64_SINT:
      return {64, 64, 64, 64, cudaChannelFormatKindSigned};
    case VK_FORMAT_R64G64B64A64_SFLOAT:
      return {64, 64, 64, 64, cudaChannelFormatKindFloat};
    default:
      break;
  }
  assert(!"Unsupported");
  return {0, 0, 0, cudaChannelFormatKindSigned};
}
