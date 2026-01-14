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

#include "optix_denoiser.hpp"

// Define OptiX function table (must be in exactly one translation unit)
#include <optix_function_table_definition.h>

#include "nvutils/logger.hpp"
#include "nvutils/parameter_registry.hpp"
#include "nvvk/commands.hpp"
#include "nvvk/descriptors.hpp"
#include "nvvk/resource_allocator.hpp"

#include "_autogen/optix_image_to_buffer.slang.h"

#ifdef CUDA_FOUND
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include "nvvk/compute_pipeline.hpp"
#include "nvvk/default_structs.hpp"
#include <nvvk/debug_util.hpp>

// OptiX error checking macro
#define OPTIX_CHECK(call)                                                                                              \
  do                                                                                                                   \
  {                                                                                                                    \
    OptixResult res = call;                                                                                            \
    if(res != OPTIX_SUCCESS)                                                                                           \
    {                                                                                                                  \
      LOGE("Optix call (%s) failed with code %d (%s:%d)\n", #call, res, __FILE__, __LINE__);                           \
      return false;                                                                                                    \
    }                                                                                                                  \
  } while(false)

// CUDA error checking macro
#define CUDA_CHECK_BOOL(call)                                                                                          \
  do                                                                                                                   \
  {                                                                                                                    \
    cudaError_t err = call;                                                                                            \
    if(err != cudaSuccess)                                                                                             \
    {                                                                                                                  \
      LOGE("CUDA call (%s) failed with code %d (%s:%d)\n", #call, err, __FILE__, __LINE__);                            \
      return false;                                                                                                    \
    }                                                                                                                  \
  } while(false)

#define NVVK_DBG_NAME_STR(obj, name) nvvk::DebugUtil::getInstance().setObjectName(obj, name)

void OptiXDenoiser::init(Resources& resources)
{
  m_availability                  = Availability::eUnavailable;
  VkPhysicalDevice physicalDevice = resources.allocator.getPhysicalDevice();
  m_device                        = resources.allocator.getDevice();

  // Check if CUDA runtime is available (handles delay-load failure gracefully)
  if(!vkcuda::isCudaRuntimeAvailable())
  {
    LOGW("CUDA runtime not available. OptiX denoiser disabled.\n");
    return;
  }

  // Create GBuffers for denoiser output and guides
  resources.samplerPool.acquireSampler(m_linearSampler);
  m_inputOutputGbuffers.init({.allocator = &resources.allocator,
                              .colorFormats =
                                  {
                                      VK_FORMAT_R32G32B32A32_SFLOAT,  // Output denoised image (index 0)
                                      VK_FORMAT_R32G32B32A32_SFLOAT,  // OptiX Albedo+Normal (index 1)
                                  },
                              .imageSampler   = m_linearSampler,
                              .descriptorPool = resources.descriptorPool});


  // Create export allocator for Vulkan-CUDA interop
  VmaAllocatorCreateInfo allocatorInfo = {
      .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice   = physicalDevice,
      .device           = m_device,
      .instance         = resources.instance,
      .vulkanApiVersion = VK_API_VERSION_1_4,
  };
  m_allocExport.init(allocatorInfo);

  // Set CUDA device to match Vulkan device
  if(cudaSuccess != vkcuda::setCudaDevice(physicalDevice))
    return;

  // Initialize OptiX
  if(!initOptiX(resources))
    return;

  m_availability = Availability::eAvailable;
}

void OptiXDenoiser::deinit(Resources& resources)
{
  cleanupOptiX();
}

bool OptiXDenoiser::initOptiX(Resources& resources)
{
  if(!initOptiXContext())
  {
    LOGE("Failed to initialize OptiX context\n");
    return false;
  }

  if(!initOptiXDenoiser())
  {
    LOGE("Failed to initialize OptiX denoiser\n");
    return false;
  }

  return true;
}

bool OptiXDenoiser::initOptiXContext()
{
  // Initialize OptiX
  OPTIX_CHECK(optixInit());

  // Create OptiX device context
  OptixDeviceContextOptions contextOptions = {};
  contextOptions.logCallbackFunction       = nullptr;
  contextOptions.logCallbackLevel          = 0;

  // Get current CUDA context (already set by vkcuda::setCudaDevice)
  CUcontext cudaContext = nullptr;
  CUresult  cudaResult  = cuCtxGetCurrent(&cudaContext);
  if(cudaResult != CUDA_SUCCESS)
  {
    LOGE("Failed to get CUDA context for OptiX\n");
    return false;
  }

  OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &contextOptions, &m_optixContext));

  // Create a dedicated CUDA stream for denoising operations
  // This keeps GPU resources allocated and can help with performance consistency
  cudaError_t err = cudaStreamCreate(&m_cudaStream);
  if(err != cudaSuccess)
  {
    LOGE("Failed to create CUDA stream: %s\n", cudaGetErrorString(err));
    m_cudaStream = nullptr;  // Fall back to default stream
  }

  return true;
}

bool OptiXDenoiser::initOptiXDenoiser()
{
  // Set up denoiser options with albedo and normal guides
  m_denoiserOptions.guideAlbedo  = 1;
  m_denoiserOptions.guideNormal  = 1;
  m_denoiserOptions.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;

  // Use AOV model kind - since OptiX r575+, HDR/LDR models are internally mapped to AOV
  // using kernel prediction, making AOV the recommended model for all use cases
  OptixDenoiserModelKind modelKind = OPTIX_DENOISER_MODEL_KIND_AOV;

  // Create denoiser
  OPTIX_CHECK(optixDenoiserCreate(m_optixContext, modelKind, &m_denoiserOptions, &m_denoiser));

  // Get denoiser memory requirements
  OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, m_outputSize.width, m_outputSize.height, &m_denoiserSizes));

  // Set denoiser parameters
  m_denoiserParams.blendFactor = 0;  //m_settings.blendFactor;

  return true;
}

bool OptiXDenoiser::createSharedBuffers()
{
  SCOPED_TIMER("Optix: Create buffers");
  if(m_outputSize.width == 0 || m_outputSize.height == 0)
  {
    return true;  // Not an error, just nothing to do yet
  }

  // Calculate buffer sizes
  size_t pixelSize  = sizeof(float) * 4;  // RGBA float
  size_t bufferSize = m_outputSize.width * m_outputSize.height * pixelSize;

  // Create shared buffers with export flags and DEDICATED memory to ensure each gets its own memory block
  VkBufferUsageFlags2KHR usage =
      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT;

  // RGB buffer
  m_allocExport.createBufferExport(m_rgbBuffer.vkBuffer, bufferSize, usage);
  m_rgbBuffer.cudaBuffer = vkcuda::createCudaBuffer(m_allocExport, m_rgbBuffer.vkBuffer);
  NVVK_DBG_NAME_STR(m_rgbBuffer.vkBuffer.buffer, "Optix::m_rgbBuffer");

  // Albedo buffer
  m_allocExport.createBufferExport(m_albedoBuffer.vkBuffer, bufferSize, usage);
  m_albedoBuffer.cudaBuffer = vkcuda::createCudaBuffer(m_allocExport, m_albedoBuffer.vkBuffer);
  NVVK_DBG_NAME_STR(m_albedoBuffer.vkBuffer.buffer, "Optix::m_albedoBuffer");

  // Normal buffer
  m_allocExport.createBufferExport(m_normalBuffer.vkBuffer, bufferSize, usage);
  m_normalBuffer.cudaBuffer = vkcuda::createCudaBuffer(m_allocExport, m_normalBuffer.vkBuffer);
  NVVK_DBG_NAME_STR(m_normalBuffer.vkBuffer.buffer, "Optix::m_normalBuffer");

  // Output buffer
  m_allocExport.createBufferExport(m_outputBuffer.vkBuffer, bufferSize, usage);
  m_outputBuffer.cudaBuffer = vkcuda::createCudaBuffer(m_allocExport, m_outputBuffer.vkBuffer);
  NVVK_DBG_NAME_STR(m_outputBuffer.vkBuffer.buffer, "Optix::m_outputBuffer");

  // CUDA-only buffers (OptiX internal)
  CUDA_CHECK_BOOL(m_stateBuffer.allocate(m_denoiserSizes.stateSizeInBytes));
  CUDA_CHECK_BOOL(m_scratchBuffer.allocate(m_denoiserSizes.withoutOverlapScratchSizeInBytes));

  // Setup the denoiser with the allocated buffers
  if(!m_denoiser)
  {
    LOGE("ERROR: Cannot setup denoiser - denoiser is null!\n");
    return false;
  }

  // Setup denoiser with maximum scratch size (for both intensity and denoising operations)
  OPTIX_CHECK(optixDenoiserSetup(m_denoiser, m_cudaStream, m_outputSize.width, m_outputSize.height, m_stateBuffer.ptr,
                                 m_stateBuffer.size, m_scratchBuffer.ptr, m_scratchBuffer.size));

  // Wait for the setup to complete - it's asynchronous!
  CUDA_CHECK_BOOL(cudaStreamSynchronize(m_cudaStream));

  return true;
}

//---------------------------------------------------------
// Called by the application when the window size changes
void OptiXDenoiser::updateSize(VkCommandBuffer cmd, VkExtent2D size)
{
  if(m_settings.enable)
  {
    // If the denoiser is active, it needs the guide buffer albedo/normal to be updated
    // Also update the output buffer
    m_inputOutputGbuffers.update(cmd, size);
    NVVK_DBG_NAME_STR(m_inputOutputGbuffers.getColorImage(), "Optix::m_outputImage");
  }

  // If the buffer size has changed, we will need to rebuild the OptiX buffers before denoising.
  if(m_bufferSize.width != size.width || m_bufferSize.height != size.height)
  {
    m_bufferSize         = size;
    m_needRebuitlBuffers = true;
  }
}

//---------------------------------------------------------
// Called when the buffer size has changed, we will need to rebuild the OptiX buffers before denoising.
void OptiXDenoiser::rebuiltBuffers()
{
  m_needRebuitlBuffers = false;
  m_outputSize         = m_bufferSize;

  if(isAvailable())
  {
    // Compute denoiser memory requirements BEFORE creating buffers
    // The buffer allocation sizes depend on m_denoiserSizes
    if(m_denoiser)
    {
      OptixResult res = optixDenoiserComputeMemoryResources(m_denoiser, m_outputSize.width, m_outputSize.height, &m_denoiserSizes);
      if(res != OPTIX_SUCCESS)
      {
        LOGE("optixDenoiserComputeMemoryResources failed with code %d\n", res);
        m_availability = Availability::eUnavailable;
        return;
      }
    }

    // Recreate buffers with new size (uses updated m_denoiserSizes)
    cleanupBuffers();
    if(!createSharedBuffers())
    {
      LOGE("Failed to recreate shared buffers for new size\n");
      m_availability = Availability::eUnavailable;
      return;
    }
  }
}

bool OptiXDenoiser::denoiseOneShot(Resources& resources)
{
  SCOPED_TIMER("Optix: Denoise");
  if(!isEnabled())
  {
    return false;
  }

  // IMPORTANT: Wait for ALL GPU operations to complete before denoising
  // This ensures:
  // 1. The latest accumulated samples are in the rendered image
  // 2. The denoised output image is not being read by the display pipeline
  // 3. All previous frames have finished rendering
  {
    SCOPED_TIMER("OptiX: wait");
    vkQueueWaitIdle(resources.app->getQueue(0).queue);
  }

  // If the OptiX buffers need to be rebuilt, do it now.
  if(m_needRebuitlBuffers)
  {
    rebuiltBuffers();

    // Check if rebuild was successful - rebuiltBuffers() may set m_availability to eUnavailable on failure
    if(!isEnabled())
    {
      LOGE("OptiX buffer rebuild failed, cannot denoise\n");
      return false;
    }
  }

  // Prepare denoising inputs
  {
    SCOPED_TIMER("OptiX: prepareDenoisingInputs");

    VkCommandBuffer cmd = resources.app->createTempCmdBuffer();

    DenoisingInputs inputs = {
        .renderedImage     = resources.gBuffers.getDescriptorImageInfo(Resources::eImgRendered),
        .albedoNormalImage = m_inputOutputGbuffers.getDescriptorImageInfo(eGBufferAlbedoNormal),
    };

    if(!prepareDenoisingInputs(cmd, inputs))
    {
      return false;
    }
    resources.app->submitAndWaitTempCmdBuffer(cmd);
  }

  // Execute OptiX denoising
  {
    SCOPED_TIMER("OptiX: executeDenoising");

    if(!executeDenoising())
    {
      return false;
    }
  }

  // Finalize denoised output
  {
    SCOPED_TIMER("OptiX: finalizeDenoisedOutput");

    VkCommandBuffer cmd = resources.app->createTempCmdBuffer();

    DenoisingOutputs outputs = {.outputImage = m_inputOutputGbuffers.getColorImage(eGBufferDenoised)};

    if(!finalizeDenoisedOutput(cmd, outputs))
    {
      return false;
    }

    resources.app->submitAndWaitTempCmdBuffer(cmd);
  }

  m_hasValidOutput = true;
  return true;
}

bool OptiXDenoiser::prepareDenoisingInputs(VkCommandBuffer cmd, const DenoisingInputs& inputs)
{
  if(!isEnabled())
  {
    return false;
  }

  // Create compute pipeline if needed
  if(m_computePipeline == VK_NULL_HANDLE)
  {
    createComputePipeline();
  }

  // Bind compute pipeline
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);

  using namespace shaderio;
  nvvk::WriteSetContainer writeContainer;
  writeContainer.append(m_bindings.getWriteSet(OptixBindingPoints::eInRgba), inputs.renderedImage);
  writeContainer.append(m_bindings.getWriteSet(OptixBindingPoints::eInAlbedoNormal), inputs.albedoNormalImage);
  writeContainer.append(m_bindings.getWriteSet(OptixBindingPoints::eOutRgba), m_rgbBuffer.vkBuffer);
  writeContainer.append(m_bindings.getWriteSet(OptixBindingPoints::eOutAlbedo), m_albedoBuffer.vkBuffer);
  writeContainer.append(m_bindings.getWriteSet(OptixBindingPoints::eOutNormal), m_normalBuffer.vkBuffer);
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineLayout, 0,
                            static_cast<uint32_t>(writeContainer.size()), writeContainer.data());

  // Push constants (image size)
  vkCmdPushConstants(cmd, m_computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkExtent2D), &m_outputSize);

  // Dispatch compute shader
  VkExtent2D groupCounts = nvvk::getGroupCounts(m_outputSize, VkExtent2D{16, 16});
  vkCmdDispatch(cmd, groupCounts.width, groupCounts.height, 1);

  return true;
}

bool OptiXDenoiser::executeDenoising()
{
  if(!isEnabled())
  {
    return false;
  }

  // Step 2: Perform OptiX denoising (synchronously on CUDA)
  {
    // Set up OptiX denoiser inputs using CUDA pointers
    OptixDenoiserLayer      layers = {};
    OptixDenoiserGuideLayer guide  = {};

    // Pixel format and stride
    OptixImage2D commonFormat = {
        .data               = 0,
        .width              = m_outputSize.width,
        .height             = m_outputSize.height,
        .rowStrideInBytes   = static_cast<unsigned int>(sizeof(float) * 4 * m_outputSize.width),
        .pixelStrideInBytes = sizeof(float) * 4,
        .format             = OPTIX_PIXEL_FORMAT_FLOAT4,
    };

    // RGB input
    layers.input      = commonFormat;
    layers.input.data = (CUdeviceptr)m_rgbBuffer.cudaBuffer.cuPtr;

    // Albedo input (only if guideAlbedo is enabled)
    if(m_denoiserOptions.guideAlbedo != 0u)
    {
      guide.albedo      = commonFormat;
      guide.albedo.data = (CUdeviceptr)m_albedoBuffer.cudaBuffer.cuPtr;
    }

    // Normal input (only if guideNormal is enabled)
    if(m_denoiserOptions.guideNormal != 0u)
    {
      guide.normal      = commonFormat;
      guide.normal.data = (CUdeviceptr)m_normalBuffer.cudaBuffer.cuPtr;
    }

    // Output buffer
    layers.output      = commonFormat;
    layers.output.data = (CUdeviceptr)m_outputBuffer.cudaBuffer.cuPtr;

    // Configure denoiser parameters
    // Note: hdrIntensity and hdrAverageColor are set to 0 (null) to let OptiX auto-calculate them.
    // According to NVIDIA: "They are automatically calculated if the device pointer is zero."
    // Explicit computation is only needed when using tiling for consistent results across tiles.
    OptixDenoiserParams params           = {};
    params.blendFactor                   = 0.0f;  // 0.0 = fully denoised, 1.0 = original noisy
    params.hdrIntensity                  = 0;     // Auto-calculated by OptiX
    params.hdrAverageColor               = 0;     // Auto-calculated by OptiX (for AOV mode)
    params.temporalModeUsePreviousLayers = 0;     // Not using temporal mode

    OPTIX_CHECK(optixDenoiserInvoke(m_denoiser, m_cudaStream, &params, m_stateBuffer.ptr, m_stateBuffer.size, &guide,
                                    &layers, 1, 0, 0, m_scratchBuffer.ptr, m_scratchBuffer.size));

    // Wait for CUDA stream to finish - ensure denoiser completes before returning
    CUDA_CHECK_BOOL(cudaStreamSynchronize(m_cudaStream));
  }

  return true;
}

bool OptiXDenoiser::finalizeDenoisedOutput(VkCommandBuffer cmd, const DenoisingOutputs& outputs)
{
  if(!isEnabled())
  {
    return false;
  }

  // Define subresource range for color image (always the same for denoising output)
  VkImageSubresourceRange subresourceRange = DEFAULT_VkImageSubresourceRange;

  // Transition output image to transfer destination
  nvvk::cmdImageMemoryBarrier(cmd, {outputs.outputImage, VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange});

  // Copy buffer to image
  VkBufferImageCopy copyRegion               = {};
  copyRegion.bufferOffset                    = 0;
  copyRegion.bufferRowLength                 = m_outputSize.width;
  copyRegion.bufferImageHeight               = m_outputSize.height;
  copyRegion.imageSubresource.aspectMask     = subresourceRange.aspectMask;
  copyRegion.imageSubresource.mipLevel       = subresourceRange.baseMipLevel;
  copyRegion.imageSubresource.baseArrayLayer = subresourceRange.baseArrayLayer;
  copyRegion.imageSubresource.layerCount     = subresourceRange.layerCount;
  copyRegion.imageOffset                     = {0, 0, 0};
  copyRegion.imageExtent                     = {m_outputSize.width, m_outputSize.height, 1};

  vkCmdCopyBufferToImage(cmd, m_outputBuffer.vkBuffer.buffer, outputs.outputImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         1, &copyRegion);

  // Transition to general layout
  nvvk::cmdImageMemoryBarrier(cmd, {outputs.outputImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange});

  m_hasValidOutput = true;
  return true;
}

void OptiXDenoiser::createComputePipeline()
{
  // Shader descriptor set layout (following Tonemapper::init pattern)
  using namespace shaderio;
  m_bindings.addBinding(OptixBindingPoints::eInRgba, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // inResultImage
  m_bindings.addBinding(OptixBindingPoints::eInAlbedoNormal, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
                        VK_SHADER_STAGE_COMPUTE_BIT);  // inAlbedoNormalImage
  m_bindings.addBinding(OptixBindingPoints::eOutRgba, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // outRgbBuffer
  m_bindings.addBinding(OptixBindingPoints::eOutAlbedo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // outAlbedoBuffer
  m_bindings.addBinding(OptixBindingPoints::eOutNormal, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);  // outNormalBuffer

  NVVK_CHECK(m_descriptorPack.init(m_bindings, m_device, 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR));
  NVVK_DBG_NAME(m_descriptorPack.getLayout());

  // Push constant for image size
  VkPushConstantRange pushConstantRange{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .size = sizeof(VkExtent2D)};

  // Pipeline layout
  const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = m_descriptorPack.getLayoutPtr(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstantRange,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_computePipelineLayout));
  NVVK_DBG_NAME(m_computePipelineLayout);

  // Create Compute Pipeline
  VkShaderModuleCreateInfo shaderInfo = {
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = optix_image_to_buffer_slang_sizeInBytes,
      .pCode    = optix_image_to_buffer_slang,
  };

  VkPipelineShaderStageCreateInfo stageInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = &shaderInfo,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .pName = "main",
  };

  VkComputePipelineCreateInfo compInfo = {
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage  = stageInfo,
      .layout = m_computePipelineLayout,
  };

  NVVK_CHECK(vkCreateComputePipelines(m_device, nullptr, 1, &compInfo, nullptr, &m_computePipeline));
  NVVK_DBG_NAME(m_computePipeline);
}

VkDescriptorImageInfo OptiXDenoiser::getDescriptorImageInfo(GBufferIndex index) const
{
  return m_inputOutputGbuffers.getDescriptorImageInfo(index);
}

void OptiXDenoiser::cleanupOptiX()
{
  if(m_denoiser)
  {
    optixDenoiserDestroy(m_denoiser);
    m_denoiser = nullptr;
  }

  if(m_cudaStream)
  {
    cudaStreamDestroy(m_cudaStream);
    m_cudaStream = nullptr;
  }

  if(m_optixContext)
  {
    optixDeviceContextDestroy(m_optixContext);
    m_optixContext = nullptr;
  }

  // Cleanup Vulkan compute pipeline resources
  if(m_device != VK_NULL_HANDLE)
  {
    vkDestroyPipeline(m_device, m_computePipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_computePipelineLayout, nullptr);
    m_descriptorPack.deinit();

    m_computePipeline       = VK_NULL_HANDLE;
    m_computePipelineLayout = VK_NULL_HANDLE;
  }

  cleanupBuffers();

  m_allocExport.deinit();

  m_inputOutputGbuffers.deinit();

  m_availability = Availability::eNotChecked;
}


void OptiXDenoiser::cleanupBuffers()
{
  if(!vkcuda::isCudaRuntimeAvailable())
  {
    return;
  }

  // Cleanup shared Vulkan-CUDA buffers
  vkcuda::destroyCudaBuffer(m_rgbBuffer.cudaBuffer);
  vkcuda::destroyCudaBuffer(m_albedoBuffer.cudaBuffer);
  vkcuda::destroyCudaBuffer(m_normalBuffer.cudaBuffer);
  vkcuda::destroyCudaBuffer(m_outputBuffer.cudaBuffer);

  m_allocExport.destroyBuffer(m_rgbBuffer.vkBuffer);
  m_allocExport.destroyBuffer(m_albedoBuffer.vkBuffer);
  m_allocExport.destroyBuffer(m_normalBuffer.vkBuffer);
  m_allocExport.destroyBuffer(m_outputBuffer.vkBuffer);

  // Cleanup CUDA-only buffers
  m_stateBuffer.free();
  m_scratchBuffer.free();
}

void OptiXDenoiser::registerParameters(nvutils::ParameterRegistry* paramReg)
{
  paramReg->add({"optixEnable", "OptiX Denoiser: Enable OptiX denoiser"}, &m_settings.enable);
  paramReg->add({"optixAutoDenoiseEnabled", "OptiX Denoiser: Auto-denoise every N frames"}, &m_settings.autoDenoiseEnabled);
  paramReg->add({"optixAutoDenoiseInterval", "OptiX Denoiser: Auto-denoise interval (frames)"}, &m_settings.autoDenoiseInterval);
}

void OptiXDenoiser::setSettingsHandler(nvgui::SettingsHandler* settingsHandler)
{
  settingsHandler->setSetting("optixEnable", &m_settings.enable);
  settingsHandler->setSetting("optixAutoDenoiseEnabled", &m_settings.autoDenoiseEnabled);
  settingsHandler->setSetting("optixAutoDenoiseInterval", &m_settings.autoDenoiseInterval);
}

void OptiXDenoiser::updateDenoiser(Resources& resources)
{
  if(!m_settings.enable)
  {
    return;
  }

  // Auto-denoise logic: trigger denoising at frame intervals
  if(m_settings.autoDenoiseEnabled && m_settings.autoDenoiseInterval > 0)
  {
    // Reset tracking if frame count went backwards (rendering restarted)
    if(resources.frameCount < m_lastAutoDenoiseFrame)
    {
      m_lastAutoDenoiseFrame = 0;
    }

    bool     shouldDenoise   = false;
    uint64_t currentInterval = resources.frameCount / m_settings.autoDenoiseInterval;
    uint64_t lastInterval    = m_lastAutoDenoiseFrame / m_settings.autoDenoiseInterval;

    // Check if we've crossed an interval boundary
    if(currentInterval > lastInterval && resources.frameCount % m_settings.autoDenoiseInterval == 0)
    {
      shouldDenoise = true;
    }

    if(m_settings.autoDenoiseInterval == 1)
      shouldDenoise = true;

    if(shouldDenoise)
    {
      if(denoiseOneShot(resources))
      {
        m_lastAutoDenoiseFrame = resources.frameCount;
        // Automatically switch to display the denoised result
        resources.settings.displayBuffer = DisplayBuffer::eOptixDenoised;
        LOGI("Auto-denoise triggered at frame %i\n", resources.frameCount);
      }
    }
  }
}

bool OptiXDenoiser::onUi(Resources& resources)
{
  bool changed = false;

  // Check if init failed (e.g. because hardware is missing)
  if(Availability::eUnavailable == m_availability)
  {
    ImGui::BeginDisabled();
    bool dummyEnable = false;
    ImGui::Checkbox("OptiX Denoiser", &dummyEnable);
    ImGui::EndDisabled();
    ImGui::SameLine();

    ImGui::TextDisabled("(OptiX initialization failed; is hardware available?)");
    return changed;
  }

  {
    bool wasEnabled = m_settings.enable;
    changed |= ImGui::Checkbox("OptiX Denoiser", &m_settings.enable);

    // When enabling the denoiser, ensure buffers are properly sized
    if(m_settings.enable && !wasEnabled)
    {
      VkCommandBuffer cmd = resources.app->createTempCmdBuffer();
      updateSize(cmd, resources.gBuffers.getSize());
      resources.app->submitAndWaitTempCmdBuffer(cmd);
    }

    // If the denoiser is disabled switch the display to the standard rendered output.
    if(!m_settings.enable && (resources.settings.displayBuffer == DisplayBuffer::eOptixDenoised))
    {
      resources.settings.displayBuffer = DisplayBuffer::eRendered;
    }

    if(m_settings.enable)
    {
      // Manual denoise button
      ImGui::SameLine();
      if(ImGui::Button("Denoise"))
      {
        if(denoiseOneShot(resources))
        {
          // Automatically switch to display the denoised result
          resources.settings.displayBuffer = DisplayBuffer::eOptixDenoised;
          changed                          = true;
        }
        else
        {
          LOGE("OptiX denoising failed\n");
        }
      }

      // Auto-denoise settings
      namespace PE = nvgui::PropertyEditor;
      PE::begin(__FUNCTION__);
      changed |= PE::Checkbox("Auto-Denoise", &m_settings.autoDenoiseEnabled);
      if(m_settings.autoDenoiseEnabled)
      {
        changed |= PE::SliderInt("Interval (frames)", &m_settings.autoDenoiseInterval, 1, 500);
        if(m_settings.autoDenoiseInterval > 1)
        {
          ImGui::Text("Next denoise at frame: %llu",
                      ((resources.frameCount / m_settings.autoDenoiseInterval) + 1) * m_settings.autoDenoiseInterval);
        }
      }
      PE::end();

      // Show denoised output thumbnail if available
      if(m_hasValidOutput)
      {
        ImGui::Spacing();

        float  aspectRatio   = m_outputSize.width > 0 ? float(m_outputSize.width) / float(m_outputSize.height) : 1.0f;
        ImVec2 thumbnailSize = {100.0f * aspectRatio, 100.0f};

        DisplayBuffer bufferType = DisplayBuffer::eOptixDenoised;
        bool          isActive   = (resources.settings.displayBuffer == bufferType);

        // Highlight active buffer with green border
        if(isActive)
        {
          ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
          ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 3.0f);
        }

        ImGui::Text("Denoised Result%s", isActive ? " (Active)" : "");
        if(ImGui::ImageButton("OptiXDenoised", ImTextureID(m_inputOutputGbuffers.getDescriptorSet(eGBufferDenoised)), thumbnailSize))
        {
          // Toggle back to rendered image
          resources.settings.displayBuffer = isActive ? DisplayBuffer::eRendered : DisplayBuffer::eOptixDenoised;
          changed                          = true;
        }

        if(isActive)
        {
          ImGui::PopStyleVar();
          ImGui::PopStyleColor();
        }
      }
    }
  }

  return false;
}
