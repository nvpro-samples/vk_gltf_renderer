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

//////////////////////////////////////////////////////////////////////////
/*
    Silhouette Rendering System

    This module implements a compute-based silhouette rendering system that:
    - Highlights selected objects in the scene with a customizable color
    - Uses a compute shader for efficient silhouette generation
    - Supports real-time updates for interactive object selection
    - Integrates with the main renderer's G-Buffer system
    .
*/
//////////////////////////////////////////////////////////////////////////

#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/compute_pipeline.hpp>

#include "silhouette.hpp"

// Pre-compiled shader
#include "_autogen/silhouette.comp.slang.h"

//--------------------------------------------------------------------------------------------------
// Initialize the silhouette rendering system
// This function sets up:
// 1. Push constants for silhouette color
// 2. Descriptor set layout for input/output images
// 3. Pipeline layout with push constants
// 4. Compute shader for silhouette generation
void Silhouette::init(Resources& res)
{
  SCOPED_TIMER(__FUNCTION__);
  VkDevice device      = res.allocator.getDevice();
  m_pushConstant.color = glm::vec3(1.f, 0.f, 0.f);  // Default red color for silhouettes

  // Define push constant range for the compute shader
  VkPushConstantRange pushConstant = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(shaderio::SilhouettePushConstant)};

  // Create descriptor bindings for input/output images
  m_bindings.addBinding(shaderio::SilhouetteBindings::eObjectID, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::SilhouetteBindings::eRGBAIImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);

  // Create descriptor set layout with push descriptor support
  NVVK_CHECK(m_bindings.createDescriptorSetLayout(device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  // Create pipeline layout with push constants
  VkPipelineLayoutCreateInfo plCreateInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_descriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstant,
  };
  NVVK_CHECK(vkCreatePipelineLayout(device, &plCreateInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  // Compute Pipeline
  VkComputePipelineCreateInfo compInfo   = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  VkShaderModuleCreateInfo    shaderInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  compInfo.stage                         = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  compInfo.stage.stage                   = VK_SHADER_STAGE_COMPUTE_BIT;
  compInfo.stage.pName                   = "main";
  compInfo.stage.pNext                   = &shaderInfo;
  compInfo.layout                        = m_pipelineLayout;

  shaderInfo.codeSize = uint32_t(silhouette_comp_slang_sizeInBytes);
  shaderInfo.pCode    = silhouette_comp_slang;

  // Create the compute shader
  vkCreateComputePipelines(device, nullptr, 1, &compInfo, nullptr, &m_pipeline);
  NVVK_DBG_NAME(m_pipeline);
}

//--------------------------------------------------------------------------------------------------
// Clean up silhouette rendering resources
void Silhouette::deinit(Resources& res)
{
  VkDevice device = res.allocator.getDevice();
  vkDestroyPipeline(device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
  m_bindings.clear();
  m_pipeline            = {};
  m_pipelineLayout      = {};
  m_descriptorSetLayout = {};
}

//--------------------------------------------------------------------------------------------------
// Dispatch the silhouette compute shader
// This function:
// 1. Updates descriptor sets with input/output images
// 2. Binds the compute shader
// 3. Pushes constants for silhouette color
// 4. Dispatches the compute shader with appropriate workgroup sizes
void Silhouette::dispatch(VkCommandBuffer cmd, const VkExtent2D& imgSize, std::vector<VkDescriptorImageInfo>& imageDescriptors)
{
  // Update descriptor sets with input/output images
  nvvk::WriteSetContainer writeContainer;
  writeContainer.append(m_bindings.getWriteSet(shaderio::SilhouetteBindings::eObjectID), imageDescriptors[0]);
  writeContainer.append(m_bindings.getWriteSet(shaderio::SilhouetteBindings::eRGBAIImage), imageDescriptors[1]);
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,
                            static_cast<uint32_t>(writeContainer.size()), writeContainer.data());

  // Bind compute shader
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);

  // Push constants for silhouette color
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::SilhouettePushConstant), &m_pushConstant);

  // Calculate and dispatch workgroups
  VkExtent2D group_counts = nvvk::getGroupCounts(imgSize, SILHOUETTE_WORKGROUP_SIZE);
  vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
}

//--------------------------------------------------------------------------------------------------
// Set the silhouette color
// This function updates the push constant color used by the compute shader
// to render the silhouette of selected objects
void Silhouette::setColor(glm::vec3 color)
{
  m_pushConstant.color = color;
}
