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

#pragma once

#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/barriers.hpp>
#include <nvapp/application.hpp>
#include <nvslang/slang.hpp>

#include "gizmo_transform_vk.hpp"
#include "gizmo_grid_vk.hpp"

//--------------------------------------------------------------------------------------------------
// VisualHelpers: Manages 3D transform gizmo and grid visualization helpers
//
// Renders directly onto the provided target image (typically the tonemapped image)
// using a dedicated helper depth buffer for self-occlusion, and samples the scene
// depth for behind-geometry occlusion effects.
//--------------------------------------------------------------------------------------------------
class VisualHelpers
{
public:
  GridHelperVk      grid;
  TransformHelperVk transform;

  struct Resources
  {
    nvapp::Application*      app;
    nvvk::ResourceAllocator* alloc;
    nvvk::StagingUploader*   uploader;
    VkDevice                 device;
    VkSampler                sampler;
    nvslang::SlangCompiler*  slangCompiler;
    VkFormat                 colorFormat;
    VkFormat                 depthFormat;
  };

  void init(const Resources& res);
  void deinit();
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size, VkImage sceneDepth, VkImageView sceneDepthView, VkSampler sampler);

  bool shouldRender() const;
  void render(VkCommandBuffer  cmd,
              VkImage          targetColorImage,
              VkImageView      targetColorImageView,
              VkDescriptorSet  sceneDescriptorSet,
              const glm::mat4& viewMatrix,
              const glm::mat4& projMatrix,
              const glm::vec2& viewportSize,
              const glm::vec2& depthBufferSize);

private:
  nvapp::Application*      m_app         = nullptr;
  nvvk::ResourceAllocator* m_alloc       = nullptr;
  VkDevice                 m_device      = VK_NULL_HANDLE;
  VkSampler                m_sampler     = VK_NULL_HANDLE;
  VkFormat                 m_colorFormat = VK_FORMAT_UNDEFINED;
  VkFormat                 m_depthFormat = VK_FORMAT_UNDEFINED;

  // Helper depth buffer for self-occlusion among helper geometry
  nvvk::Image m_helperDepthImage;
  VkImageView m_helperDepthImageView = VK_NULL_HANDLE;
  VkExtent2D  m_helperExtent{};

  // Descriptor set for scene depth sampling (shared between grid and transform helpers)
  VkDescriptorPool m_helperDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorSet  m_helperDescriptorSet  = VK_NULL_HANDLE;

  void initDescriptorSet(VkImage sceneDepth, VkImageView sceneDepthView);
  void deinitDescriptorSet();
  void createHelperDepthBuffer(VkCommandBuffer cmd, const VkExtent2D& size);
  void destroyHelperDepthBuffer();
};
