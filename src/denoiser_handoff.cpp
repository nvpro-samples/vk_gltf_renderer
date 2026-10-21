/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// DLSS hand-off: the image shown while OptiX or OIDN takes over from DLSS-RR (see denoiser_handoff.hpp).
//

#include "denoiser_handoff.hpp"

#include <nvgui/property_editor.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>

#include "denoiser_schedule.hpp"

// Pre-compiled shader
#include "_autogen/denoiser_handoff.slang.h"

static constexpr const char* kMemoryCategory = "DLSS hand-off";

void DenoiserHandoff::init(Resources& resources)
{
  const VkDevice device = resources.allocator.getDevice();
  m_appMemoryTracker    = &resources.appMemoryTracker;

  // Allocated at the first capture (updated by onResize from then on): only DLSS-RR + OptiX/OIDN uses them
  NVVK_CHECK(m_target.init({.device       = device,
                            .alloc        = &resources.allocator,
                            .colorFormats = {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT},
                            .debugName    = "DenoiserHandoff"}));

  m_bindings.addBinding(shaderio::DenoiserHandoffBindings::eHandoffDlss, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::DenoiserHandoffBindings::eHandoffDenoised, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                        VK_SHADER_STAGE_COMPUTE_BIT);
  m_bindings.addBinding(shaderio::DenoiserHandoffBindings::eHandoffOut, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_bindings.createDescriptorSetLayout(device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  const VkPushConstantRange        pushConstant{.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                .size       = sizeof(shaderio::DenoiserHandoffPushConstant)};
  const VkPipelineLayoutCreateInfo layoutInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_descriptorSetLayout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstant,
  };
  NVVK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  const VkShaderModuleCreateInfo shaderInfo{
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = denoiser_handoff_slang_sizeInBytes,
      .pCode    = denoiser_handoff_slang,
  };
  const VkComputePipelineCreateInfo pipelineInfo{
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage  = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .pNext = &shaderInfo,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                 .pName = "main"},
      .layout = m_pipelineLayout,
  };
  NVVK_CHECK(vkCreateComputePipelines(device, nullptr, 1, &pipelineInfo, nullptr, &m_pipeline));
  NVVK_DBG_NAME(m_pipeline);
}

void DenoiserHandoff::deinit(Resources& resources)
{
  const VkDevice device = resources.allocator.getDevice();
  vkDestroyPipeline(device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(device, m_descriptorSetLayout, nullptr);
  m_bindings.clear();
  m_pipeline            = {};
  m_pipelineLayout      = {};
  m_descriptorSetLayout = {};

  if(m_appMemoryTracker)
    m_appMemoryTracker->untrack(kMemoryCategory, m_target, eImageCount);
  m_target.deinit();
  cancel();
}

void DenoiserHandoff::registerParameters(SettingsRegistry* settings)
{
  // Bounds mirror the UI slider
  settings->add({"dlssHandoffSpp",
                 "Denoisers: With DLSS-RR and OptiX or OIDN, samples per pixel over which the image blends from "
                 "DLSS's to the denoised one (0 = switch at the first denoise)"},
                &m_settings.spp, Persist::eYes, 0, 1000);
}

void DenoiserHandoff::onResize(VkCommandBuffer cmd, VkExtent2D size)
{
  cancel();
  const VkExtent2D current = m_target.getSize();
  if(current.width == 0 || current.height == 0)  // never allocated: the first capture does
    return;
  if(m_appMemoryTracker)
    m_appMemoryTracker->untrack(kMemoryCategory, m_target, eImageCount);
  NVVK_CHECK(m_target.update(cmd, size));
  if(m_appMemoryTracker)
    m_appMemoryTracker->track(kMemoryCategory, m_target, eImageCount);
}

void DenoiserHandoff::cmdCapture(VkCommandBuffer cmd, Resources& resources, const Denoiser& denoiser)
{
  if(m_settings.spp <= 0 || !denoiser.autoDenoise)
    return;
  NVVK_DBG_SCOPE(cmd);
  const VkExtent2D size    = resources.gBuffers.getSize();
  const VkExtent2D current = m_target.getSize();
  if(current.width != size.width || current.height != size.height)
  {
    // The first capture: nothing in flight uses these images yet (a resize goes through onResize)
    NVVK_CHECK(m_target.update(cmd, size));
    if(m_appMemoryTracker)
      m_appMemoryTracker->track(kMemoryCategory, m_target, eImageCount);
  }

  // DLSS wrote eImgRendered in the previous frame, and the tonemapper may still read the kept image
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
  const VkImageSubresourceLayers layers{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1};
  const VkImageCopy copy{.srcSubresource = layers, .dstSubresource = layers, .extent = {size.width, size.height, 1}};
  vkCmdCopyImage(cmd, resources.gBuffers.getColorImage(Resources::eImgRendered), VK_IMAGE_LAYOUT_GENERAL,
                 m_target.getColorImage(eImageDlss), VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
  // The path tracer writes eImgRendered next: its transfer -> shader barrier in PathTracer::onRender
  // orders it after this copy. The tonemapper reads the kept image.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

  m_active                         = true;
  m_captureDenoiseCount            = denoiser.denoiseCount;
  m_weight                         = 0.0f;
  m_shown                          = true;
  m_shownImage                     = m_target.getColorStorageImageInfo(eImageDlss);
  resources.settings.displayBuffer = denoiser.view;
}

void DenoiserHandoff::cmdUpdate(VkCommandBuffer cmd, Resources& resources, const Frame& frame)
{
  const Denoiser* denoiser = frame.denoiser;
  m_shown                  = false;
  if(!m_active)
    return;
  // The denoiser was turned off, or another view picked
  if(!denoiser || resources.settings.displayBuffer != denoiser->view)
  {
    m_active = false;
    return;
  }

  const bool  denoised = denoiser->denoiseCount != m_captureDenoiseCount;  // this accumulation has a denoised image
  const float weight   = handoffWeight(frame.samples, frame.handoffEnd);
  // Done: the denoised image alone. Without one yet, DLSS's image stays while Auto will make it (OptiX,
  // one frame behind, may reach the end first); with Auto turned off, the accumulation is shown.
  if(weight >= 1.0f && (denoised || !denoiser->autoDenoise))
  {
    m_active = false;
    if(!denoised)
      resources.settings.displayBuffer = DisplayBuffer::eRendered;
    return;
  }

  m_shown  = true;
  m_weight = denoised ? weight : 0.0f;
  if(!denoised)  // DLSS's image until the first denoise of the accumulation
  {
    m_shownImage = m_target.getColorStorageImageInfo(eImageDlss);
    return;
  }

  NVVK_DBG_SCOPE(cmd);
  // The denoised image is written (OIDN in this command buffer, OptiX in an earlier submission), and
  // the previous frame's tonemapper may still read the output
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

  nvvk::WriteSetContainer writes;
  writes.append(m_bindings.getWriteSet(shaderio::DenoiserHandoffBindings::eHandoffDlss),
                m_target.getColorStorageImageInfo(eImageDlss));
  writes.append(m_bindings.getWriteSet(shaderio::DenoiserHandoffBindings::eHandoffDenoised), denoiser->image);
  writes.append(m_bindings.getWriteSet(shaderio::DenoiserHandoffBindings::eHandoffOut), m_target.getColorStorageImageInfo(eImageOut));
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, uint32_t(writes.size()), writes.data());
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
  const shaderio::DenoiserHandoffPushConstant pushConstant{.weight = m_weight};
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstant), &pushConstant);
  const VkExtent2D groups = nvvk::getGroupCounts(m_target.getSize(), WORKGROUP_SIZE);
  vkCmdDispatch(cmd, groups.width, groups.height, 1);

  // Read by the tonemapper
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  m_shownImage = m_target.getColorStorageImageInfo(eImageOut);
}

void DenoiserHandoff::cancel()
{
  m_active = false;
  m_shown  = false;
}

void DenoiserHandoff::onUi()
{
  namespace PE = nvgui::PropertyEditor;
  PE::begin("DenoiserHandoff");
  PE::SliderInt("DLSS Hand-off", &m_settings.spp, 0, 1000, "%d spp", ImGuiSliderFlags_Logarithmic,
                "Once the camera stops, blend from DLSS's image to the OptiX / OIDN one over this many samples "
                "per pixel, while the denoised image is still made of few samples. 0 switches at its first denoise.");
  ImGui::TextDisabled("Hand-off: %d%% denoised", int(m_weight * 100.0f + 0.5f));
  PE::end();
}
