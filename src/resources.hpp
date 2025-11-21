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

/**
 * Core resource management header for the Vulkan GLTF renderer.
 * 
 * This header defines the main resource structures and settings for a Vulkan-based 3D renderer
 * that supports both path tracing and rasterization. It manages Vulkan resources, GLTF scene data,
 * environment maps, and rendering settings.
 */

#pragma once
#include <bitset>

#include <glm/glm.hpp>
#include <glm/ext/scalar_constants.hpp>

#include "shaders/shaderio.h"  // Shared between host and device

#include <nvgui/sky.hpp>
#include <nvshaders_host/hdr_env_dome.hpp>
#include <nvshaders_host/tonemapper.hpp>
#include <nvslang/slang.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/hdr_ibl.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvkgltf/scene.hpp>
#include <nvvkgltf/scene_rtx.hpp>
#include <nvvkgltf/scene_vk.hpp>
#include <nvapp/application.hpp>


enum class RenderingMode
{
  ePathtracer,
  eRasterizer
};

enum class DisplayBuffer
{
  eRendered,         // Final rendered image
  eAlbedo,           // DLSS Albedo
  eSpecAlbedo,       // DLSS Specular Albedo
  eNormalRoughness,  // DLSS Normal/Roughness
  eMotionVectors,    // DLSS Motion
  eDepth,            // DLSS Depth
  eOptixDenoised,    // OptiX Denoised output
};

// Utility functions for bidirectional mapping between DisplayBuffer and OutputImage enums
constexpr inline shaderio::OutputImage displayBufferToOutputImage(DisplayBuffer buffer)
{
  switch(buffer)
  {
    case DisplayBuffer::eAlbedo:
      return shaderio::eDlssAlbedo;
    case DisplayBuffer::eSpecAlbedo:
      return shaderio::eDlssSpecAlbedo;
    case DisplayBuffer::eNormalRoughness:
      return shaderio::eDlssNormalRoughness;
    case DisplayBuffer::eMotionVectors:
      return shaderio::eDlssMotion;
    case DisplayBuffer::eDepth:
      return shaderio::eDlssDepth;
    case DisplayBuffer::eOptixDenoised:
      return shaderio::eResultImage;  // Special case: handled separately
    default:
      return shaderio::eResultImage;
  }
}

constexpr inline DisplayBuffer outputImageToDisplayBuffer(shaderio::OutputImage image)
{
  switch(image)
  {
    case shaderio::eDlssAlbedo:
      return DisplayBuffer::eAlbedo;
    case shaderio::eDlssSpecAlbedo:
      return DisplayBuffer::eSpecAlbedo;
    case shaderio::eDlssNormalRoughness:
      return DisplayBuffer::eNormalRoughness;
    case shaderio::eDlssMotion:
      return DisplayBuffer::eMotionVectors;
    case shaderio::eDlssDepth:
      return DisplayBuffer::eDepth;
    default:
      return DisplayBuffer::eRendered;
  }
}

// Note: eOptixDenoised doesn't have a direct OutputImage equivalent
// It's handled specially through OptiXDenoiser::getDescriptorImageInfo()

enum DirtyFlags
{
  eVulkanScene,       // When the Vulkan geometry buffers need to be updated
  eVulkanMaterial,    // When the Vulkan material buffers need to be updated
  eVulkanAttributes,  // When the Vulkan attributes need to be updated
  eRtxScene,          // When the RTX acceleration structures need to be updated
  eHdrEnv,            // When the HDR environment needs to be updated
  eNodeVisibility,    // When the node visibility has changed

  eNumDirtyFlags  // Keep last - Number of dirty flags
};

struct Settings
{
  RenderingMode         renderSystem           = RenderingMode::ePathtracer;    // Renderer to use
  shaderio::DebugMethod debugMethod            = shaderio::DebugMethod::eNone;  // Debug method for the rasterizer
  shaderio::EnvSystem   envSystem              = shaderio::EnvSystem::eSky;     // Environment system: Sky or HDR
  bool                  showAxis               = true;                          // Show the axis (bottom left)
  bool                  showMemStats           = false;                         // Show memory statistics window
  float                 hdrEnvIntensity        = 1.0f;                          // Intensity of the environment (HDR)
  float                 hdrEnvRotation         = 0.0f;                          // Rotation of the environment (HDR)
  float                 hdrBlur                = 0.0f;                          // Blur of the environment (HDR)
  glm::vec3             silhouetteColor        = {0.6f, 0.4f, 0.0f};            // Color of the silhouette
  bool                  useSolidBackground     = false;                         // Use solid background color
  glm::vec3             solidBackgroundColor   = {0.0f, 0.0f, 0.0f};            // Solid background color
  int                   maxFrames              = {200};                         // Maximum number of frames to render
  bool                  useInfinitePlane       = false;                         // Use infinite plane
  bool                  isShadowCatcher        = true;                          // Infinite place only catch shadow
  float                 infinitePlaneDistance  = 0;                             // Distance/height of the infinite plane
  glm::vec3             infinitePlaneBaseColor = glm::vec3(0.5, 0.5, 0.5);      // Default gray color
  float                 infinitePlaneMetallic  = 0.0;                           // Default non-metallic
  float                 infinitePlaneRoughness = 0.5;                           // Default medium roughness
  bool                  dlssHardwareAvailable  = false;  // DLSS hardware/extensions available (set at startup)
  DisplayBuffer         displayBuffer          = DisplayBuffer::eRendered;  // Which buffer to display in viewport
};


struct Resources
{
  enum ImageType
  {
    eImgTonemapped,
    eImgRendered,
    eImgSelection,
  };

  nvapp::Application* app{nullptr};

  VkInstance              instance{};
  nvvk::ResourceAllocator allocator{};  // Vulkan Memory Allocator
  nvvk::StagingUploader   staging;

  nvvk::SamplerPool      samplerPool{};    // Texture Sampler Pool
  VkCommandPool          commandPool{};    // Command pool for secondary command buffer
  nvslang::SlangCompiler slangCompiler{};  // Slang compiler

  // Scene
  nvvkgltf::Scene    scene;     // GLTF Scene
  nvvkgltf::SceneVk  sceneVk;   // GLTF Scene buffers
  nvvkgltf::SceneRtx sceneRtx;  // GLTF Scene BLAS/TLAS

  // Resources
  nvvk::HdrIbl                    hdrIbl;  // HDR environment map
  nvshaders::HdrEnvDome           hdrDome;
  nvvk::GBuffer                   gBuffers;      // G-Buffers: color + depth
  nvvk::Buffer                    bFrameInfo;    // Scene/Frame information
  nvvk::Buffer                    bSkyParams;    // Sky parameters
  shaderio::SkyPhysicalParameters skyParams{};   // Sky parameters
  nvshaders::Tonemapper           tonemapper{};  // Tonemapper
  shaderio::TonemapperData        tonemapperData{
             .autoExposure = 1,
  };  // Tonemapper data
  std::shared_ptr<nvutils::CameraManipulator> cameraManip;  // Camera manipulator (owned by GltfRenderer)

  // Pipeline
  std::array<nvvk::DescriptorBindings, 2> descriptorBinding{};    // Descriptor bindings: 0: textures, 1: tlas
  std::array<VkDescriptorSetLayout, 2>    descriptorSetLayout{};  // Descriptor set layout
  VkDescriptorSet                         descriptorSet{};        // Descriptor set for the textures
  VkDescriptorPool                        descriptorPool{};


  int frameCount{0};
  int selectedObject{-1};  // Selected object in the scene

  Settings settings;

  std::bitset<32> dirtyFlags;
};
