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

//
// Aggregates all visual helper overlays (transform gizmo, grid, axis
// indicator) into a single VisualHelpers facade. Manages their shared
// Vulkan resources, coordinates draw order, and forwards render calls
// so the main renderer only interacts with one object.
//

#include "gizmo_visuals_vk.hpp"

#include <nvvk/debug_util.hpp>
#include <nvvk/check_error.hpp>
#include <nvutils/logger.hpp>

//--------------------------------------------------------------------------------------------------
// Lifecycle
//--------------------------------------------------------------------------------------------------

void VisualHelpers::init(const Resources& res)
{
  m_app         = res.app;
  m_alloc       = res.alloc;
  m_device      = res.device;
  m_sampler     = res.sampler;
  m_colorFormat = res.colorFormat;
  m_depthFormat = res.depthFormat;

  TransformHelperVk::Resources transformHelperRes{
      .app      = res.app,
      .alloc    = res.alloc,
      .uploader = res.uploader,
      .device   = res.device,
  };
  transform.init(transformHelperRes);

  GridHelperVk::Resources gridHelperRes = {
      .app                       = res.app,
      .alloc                     = res.alloc,
      .uploader                  = res.uploader,
      .device                    = res.device,
      .slangCompiler             = res.slangCompiler,
      .colorFormat               = res.colorFormat,
      .depthFormat               = res.depthFormat,
      .helperDescriptorSetLayout = transform.getDescriptorSetLayout(),
  };
  grid.init(gridHelperRes);

  LOGI("VisualHelpers initialized\n");
}

void VisualHelpers::deinit()
{
  deinitDescriptorSet();
  destroyHelperDepthBuffer();

  transform.deinit();
  grid.deinit();

  m_app     = nullptr;
  m_alloc   = nullptr;
  m_device  = VK_NULL_HANDLE;
  m_sampler = VK_NULL_HANDLE;

  LOGI("VisualHelpers deinitialized\n");
}

void VisualHelpers::onResize(VkCommandBuffer cmd, const VkExtent2D& size, VkImage sceneDepth, VkImageView sceneDepthView, VkSampler sampler)
{
  m_sampler = sampler;

  createHelperDepthBuffer(cmd, size);

  deinitDescriptorSet();
  initDescriptorSet(sceneDepth, sceneDepthView);
}

//--------------------------------------------------------------------------------------------------
// Rendering
//--------------------------------------------------------------------------------------------------

bool VisualHelpers::shouldRender() const
{
  bool gridVisible  = grid.isVisible();
  bool gizmoVisible = transform.isAttached();

  return gridVisible || gizmoVisible;
}

void VisualHelpers::render(VkCommandBuffer  cmd,
                           VkImage          targetColorImage,
                           VkImageView      targetColorImageView,
                           VkDescriptorSet  sceneDescriptorSet,
                           const glm::mat4& viewMatrix,
                           const glm::mat4& projMatrix,
                           const glm::vec2& viewportSize,
                           const glm::vec2& depthBufferSize)
{
  if(m_helperDescriptorSet == VK_NULL_HANDLE)
    return;

  NVVK_DBG_SCOPE(cmd);

  const VkExtent2D viewportExtent = {static_cast<uint32_t>(viewportSize.x), static_cast<uint32_t>(viewportSize.y)};

  // Transition target color image from GENERAL to COLOR_ATTACHMENT_OPTIMAL
  nvvk::cmdImageMemoryBarrier(cmd, {targetColorImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

  // Begin dynamic rendering onto the target image
  VkRenderingAttachmentInfo colorAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  colorAttachment.imageView                 = targetColorImageView;
  colorAttachment.imageLayout               = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  colorAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_LOAD;
  colorAttachment.storeOp                   = VK_ATTACHMENT_STORE_OP_STORE;

  VkRenderingAttachmentInfo depthAttachment = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  depthAttachment.imageView                 = m_helperDepthImageView;
  depthAttachment.imageLayout               = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depthAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depthAttachment.storeOp                   = VK_ATTACHMENT_STORE_OP_STORE;
  depthAttachment.clearValue.depthStencil   = {1.0f, 0};

  VkRenderingInfo renderingInfo      = {VK_STRUCTURE_TYPE_RENDERING_INFO};
  renderingInfo.renderArea           = {{0, 0}, viewportExtent};
  renderingInfo.layerCount           = 1;
  renderingInfo.colorAttachmentCount = 1;
  renderingInfo.pColorAttachments    = &colorAttachment;
  renderingInfo.pDepthAttachment     = &depthAttachment;

  vkCmdBeginRendering(cmd, &renderingInfo);

  VkViewport viewport{0.0f, 0.0f, viewportSize.x, viewportSize.y, 0.0f, 1.0f};
  vkCmdSetViewportWithCount(cmd, 1, &viewport);

  VkRect2D scissor{{0, 0}, viewportExtent};
  vkCmdSetScissorWithCount(cmd, 1, &scissor);

  // Grid renders behind gizmo
  grid.renderRaster(cmd, m_helperDescriptorSet, viewMatrix, projMatrix, viewportSize, depthBufferSize);

  // Gizmo renders on top
  transform.renderRaster(cmd, m_helperDescriptorSet, viewMatrix, projMatrix, viewportSize, depthBufferSize);

  vkCmdEndRendering(cmd);

  // Transition target color image back to GENERAL for ImGui display
  nvvk::cmdImageMemoryBarrier(cmd, {targetColorImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
}

//--------------------------------------------------------------------------------------------------
// Helper Depth Buffer
//--------------------------------------------------------------------------------------------------

void VisualHelpers::createHelperDepthBuffer(VkCommandBuffer cmd, const VkExtent2D& size)
{
  destroyHelperDepthBuffer();

  m_helperExtent = size;

  VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  imageInfo.imageType         = VK_IMAGE_TYPE_2D;
  imageInfo.format            = m_depthFormat;
  imageInfo.extent            = {size.width, size.height, 1};
  imageInfo.mipLevels         = 1;
  imageInfo.arrayLayers       = 1;
  imageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.tiling            = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.usage             = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  imageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;

  NVVK_CHECK(m_alloc->createImage(m_helperDepthImage, imageInfo));
  NVVK_DBG_NAME(m_helperDepthImage.image);

  VkImageViewCreateInfo viewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  viewInfo.image                           = m_helperDepthImage.image;
  viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format                          = m_depthFormat;
  viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
  viewInfo.subresourceRange.baseMipLevel   = 0;
  viewInfo.subresourceRange.levelCount     = 1;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount     = 1;

  NVVK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &m_helperDepthImageView));
  NVVK_DBG_NAME(m_helperDepthImageView);

  // Transition depth image to attachment-optimal layout
  nvvk::cmdImageMemoryBarrier(cmd, {.image            = m_helperDepthImage.image,
                                    .oldLayout        = VK_IMAGE_LAYOUT_UNDEFINED,
                                    .newLayout        = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                    .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}});
}

void VisualHelpers::destroyHelperDepthBuffer()
{
  if(m_helperDepthImageView != VK_NULL_HANDLE)
  {
    vkDestroyImageView(m_device, m_helperDepthImageView, nullptr);
    m_helperDepthImageView = VK_NULL_HANDLE;
  }
  if(m_helperDepthImage.image != VK_NULL_HANDLE)
  {
    m_alloc->destroyImage(m_helperDepthImage);
    m_helperDepthImage = {};
  }
}

//--------------------------------------------------------------------------------------------------
// Descriptor Set for Scene Depth Sampling
//--------------------------------------------------------------------------------------------------

void VisualHelpers::initDescriptorSet(VkImage sceneDepth, VkImageView sceneDepthView)
{
  VkDescriptorSetLayout helperLayout = transform.getDescriptorSetLayout();

  std::array<VkDescriptorPoolSize, 2> poolSizes = {{
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
      {VK_DESCRIPTOR_TYPE_SAMPLER, 1},
  }};

  VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  poolInfo.maxSets       = 1;
  poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
  poolInfo.pPoolSizes    = poolSizes.data();

  NVVK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_helperDescriptorPool));
  NVVK_DBG_NAME(m_helperDescriptorPool);

  VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocInfo.descriptorPool     = m_helperDescriptorPool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts        = &helperLayout;

  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_helperDescriptorSet));
  NVVK_DBG_NAME(m_helperDescriptorSet);

  VkDescriptorImageInfo depthImageInfo = {};
  depthImageInfo.imageView             = sceneDepthView;
  depthImageInfo.imageLayout           = VK_IMAGE_LAYOUT_GENERAL;

  VkDescriptorImageInfo samplerInfo = {};
  samplerInfo.sampler               = m_sampler;

  std::array<VkWriteDescriptorSet, 2> writes{};

  writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet          = m_helperDescriptorSet;
  writes[0].dstBinding      = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  writes[0].pImageInfo      = &depthImageInfo;

  writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet          = m_helperDescriptorSet;
  writes[1].dstBinding      = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLER;
  writes[1].pImageInfo      = &samplerInfo;

  vkUpdateDescriptorSets(m_device, 2, writes.data(), 0, nullptr);
}

void VisualHelpers::deinitDescriptorSet()
{
  if(m_helperDescriptorPool != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorPool(m_device, m_helperDescriptorPool, nullptr);
    m_helperDescriptorPool = VK_NULL_HANDLE;
    m_helperDescriptorSet  = VK_NULL_HANDLE;
  }
}
