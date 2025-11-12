/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include <nvutils/parameter_registry.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/resource_allocator.hpp>

// OptiX includes
#include <optix.h>
#include <optix_stubs.h>

// Vulkan-CUDA interop
#include "vk_cuda.hpp"

#include "resources.hpp"

//-------------------------------------------------------------------------------------------------------
// OptiXDenoiser
//
// This class provides an integration layer between Vulkan rendering and NVIDIA OptiX AI-accelerated
// denoising. It wraps OptiX denoiser functionality to remove Monte Carlo noise from rendered images,
// utilizing guide buffers (albedo and normals) to preserve fine details and improve convergence.
//
// The denoiser operates by copying Vulkan image data to shared CUDA buffers, invoking the OptiX
// denoiser on GPU, and transferring the denoised result back to Vulkan images. This implementation
// uses Vulkan-CUDA interoperability for efficient zero-copy memory sharing where possible.
//
// Usage:
//
// 1. Initialization:
//    OptiXDenoiser denoiser;
//    denoiser.init(resources);
//
// 2. Configure resolution:
//    denoiser.updateSize(cmd, extent);
//
// 3. Provide input G-buffers:
//    denoiser.setInputResources(gBuffers);
//
// 4. One-shot denoising (simple path):
//    denoiser.denoiseOneShot(resources);
//
// 5. Manual denoising (advanced control):
//    DenoisingInputs inputs = {renderedImage, albedoNormalImage};
//    DenoisingOutputs outputs = {outputImage};
//    denoiser.prepareDenoisingInputs(cmd, inputs);
//    denoiser.executeDenoising();
//    denoiser.finalizeDenoisedOutput(cmd, outputs);
//
// 6. Retrieve results:
//    VkDescriptorImageInfo denoisedImage = denoiser.getDescriptorImageInfo(eGBufferDenoised);
//
// 7. Cleanup:
//    denoiser.deinit(resources);
//
// Requirements:
// - OptiX 7.x or later
// - CUDA-enabled GPU with compute capability 5.0+
// - Vulkan external memory support for Vulkan-CUDA interop
//
//-------------------------------------------------------------------------------------------------------

class OptiXDenoiser
{
public:
  // GBuffer indices for OptiX denoiser outputs
  enum GBufferIndex
  {
    eGBufferDenoised     = 0,  // Denoised output
    eGBufferAlbedoNormal = 1,  // Albedo+Normal guide buffer
  };

  enum class Availability
  {
    eNotChecked,   // Haven't attempted initialization yet
    eUnavailable,  // Hardware/runtime missing or initialization failed
    eAvailable,    // Fully checked and ready to use
  };

  struct Settings
  {
    bool enable              = false;
    bool autoDenoiseEnabled  = true;  // Automatically denoise every N frames
    int  autoDenoiseInterval = 50;    // Denoise at frames 50, 100, 150, etc.
  };

  // Input images for denoising
  struct DenoisingInputs
  {
    VkDescriptorImageInfo renderedImage;      // RGBA rendered image
    VkDescriptorImageInfo albedoNormalImage;  // Albedo + normal guide image
  };

  // Output destination for denoised result
  struct DenoisingOutputs
  {
    VkImage outputImage;  // Image to write denoised result to
  };

  OptiXDenoiser() {};
  ~OptiXDenoiser() {};

  // Initialize the denoiser
  void init(Resources& resources);
  void deinit(Resources& resources);

  // Register parameters for UI
  void registerParameters(nvutils::ParameterRegistry* paramReg);

  // Checks if the OptiX denoiser and necessary hardware is available and ready
  bool isAvailable() const { return m_availability == Availability::eAvailable; }

  // Check if denoiser is enabled
  bool isEnabled() const { return m_settings.enable && (m_availability != Availability::eUnavailable); }

  // Get the descriptor for the denoised output
  VkDescriptorImageInfo getDescriptorImageInfo(GBufferIndex index) const;

  // Update size when rendering resolution changes
  void updateSize(VkCommandBuffer cmd, VkExtent2D size);

  // Perform one-shot denoising
  bool denoiseOneShot(Resources& resources);

  // Public API for external control of denoising steps
  bool prepareDenoisingInputs(VkCommandBuffer cmd, const DenoisingInputs& inputs);
  bool executeDenoising();
  bool finalizeDenoisedOutput(VkCommandBuffer cmd, const DenoisingOutputs& outputs);

  // Check if we have a valid denoised output
  bool hasValidDenoisedOutput() const { return m_hasValidOutput; }

  // Update auto-denoise logic (call every frame in render loop)
  void updateDenoiser(Resources& resources);

  // UI controls
  bool onUi(Resources& resources);

private:
  // Initialize OptiX context and denoiser
  bool initOptiX(Resources& resources);

  // Initialize OptiX context
  bool initOptiXContext();

  // Initialize OptiX denoiser
  bool initOptiXDenoiser();

  // Create shared memory buffers between Vulkan and OptiX
  bool createSharedBuffers();

  // Create compute pipeline for image-to-buffer copy
  void createComputePipeline();

  // Cleanup OptiX resources
  void cleanupOptiX();

  // Clean all created buffers
  void cleanupBuffers();

private:
  Settings     m_settings{};
  VkExtent2D   m_outputSize{};
  Availability m_availability         = Availability::eNotChecked;
  bool         m_hasValidOutput       = false;
  uint64_t     m_lastAutoDenoiseFrame = 0;  // Track last frame we auto-denoised

  // OptiX context and denoiser
  OptixDeviceContext m_optixContext = nullptr;
  OptixDenoiser      m_denoiser     = nullptr;

  // OptiX denoiser state
  OptixDenoiserSizes   m_denoiserSizes{};
  OptixDenoiserParams  m_denoiserParams{};
  OptixDenoiserOptions m_denoiserOptions{};

  // Vulkan-CUDA interop resources
  nvvk::ResourceAllocatorExport m_allocExport{};

  // Shared buffers (Vulkan <-> CUDA <-> OptiX)
  struct SharedBuffer
  {
    nvvk::Buffer   vkBuffer;
    vkcuda::Buffer cudaBuffer;
  };

  SharedBuffer m_rgbBuffer;
  SharedBuffer m_albedoBuffer;
  SharedBuffer m_normalBuffer;
  SharedBuffer m_outputBuffer;

  // CUDA-only buffers (OptiX internal, never accessed by Vulkan)
  vkcuda::CudaBuffer m_stateBuffer;
  vkcuda::CudaBuffer m_scratchBuffer;

  // CUDA stream for denoising operations
  CUstream m_cudaStream = nullptr;

  // Vulkan resources
  VkDevice m_device = VK_NULL_HANDLE;

  // Input G-Buffers reference
  const nvvk::GBuffer* m_inputGBuffers = nullptr;

  // Output image
  nvvk::GBuffer m_outputImage;  // See GBufferIndex
  VkSampler     m_linearSampler{};


  // Compute pipeline for image-to-buffer copy
  nvvk::DescriptorBindings m_bindings;
  nvvk::DescriptorPack     m_descriptorPack;
  VkPipeline               m_computePipeline       = VK_NULL_HANDLE;
  VkPipelineLayout         m_computePipelineLayout = VK_NULL_HANDLE;
};
