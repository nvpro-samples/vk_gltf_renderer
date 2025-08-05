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

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/debug_util.hpp>
#include <imgui/imgui.h>

#include "dlss_denoiser.hpp"

bool DlssDenoiser::ensureInitialized(Resources& resources)
{
  if(!m_initialized)
  {
    initDenoiser(resources);
    return true;
  }
  return false;
}

void DlssDenoiser::registerParameters(nvutils::ParameterRegistry* paramReg)
{
  paramReg->add({"dlssEnable", "DLSS Denoiser: Enable DLSS denoiser"}, &m_settings.enable);
}

void DlssDenoiser::init(Resources& resources)
{

  resources.samplerPool.acquireSampler(m_linearSampler);
  // G-Buffer
  m_dlssGBuffers.init({.allocator      = &resources.allocator,
                       .colorFormats   = m_bufferInfos,
                       .imageSampler   = m_linearSampler,
                       .descriptorPool = resources.descriptorPool});
}

void DlssDenoiser::deinit(Resources& resources)
{
  resources.samplerPool.releaseSampler(m_linearSampler);
  m_dlssGBuffers.deinit();
  m_dlss.deinit();
  m_ngx.deinit();
  m_initialized = false;
}

void DlssDenoiser::initDenoiser(Resources& resources)
{
  if(m_initialized)
    return;
  SCOPED_TIMER("Initializing DLSS Denoiser");

  m_device = resources.allocator.getDevice();

  // #DLSS - Create the DLSS
  NgxContext::InitInfo ngxInitInfo{
      .instance       = resources.instance,
      .physicalDevice = resources.allocator.getPhysicalDevice(),
      .device         = resources.allocator.getDevice(),
  };
  // ngxInitInfo.loggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;

  NVSDK_NGX_Result ngxResult = m_ngx.init(ngxInitInfo);
  if(ngxResult == NVSDK_NGX_Result_Success)
  {
    m_dlssSupported = (m_ngx.isDlssRRAvailable() == NVSDK_NGX_Result_Success);
  }

  if(!m_dlssSupported)
  {
    LOGW("NGX init failed: %d - DLSS unsupported\n", ngxResult);
  }
  m_initialized = true;
}


VkDescriptorImageInfo DlssDenoiser::getDescriptorImageInfo(shaderio::OutputImage name) const
{
  return m_dlssGBuffers.getDescriptorImageInfo(name);
}

bool DlssDenoiser::isEnabled() const
{
  return m_settings.enable;
}

VkExtent2D DlssDenoiser::updateSize(VkCommandBuffer cmd, VkExtent2D size)
{
  if(!m_dlssSupported || !m_initialized)
    return size;

  // Choose the size of the DLSS buffers
  DlssRayReconstruction::SupportedSizes supportedSizes{};
  DlssRayReconstruction::querySupportedInputSizes(m_ngx, {size, NVSDK_NGX_PerfQuality_Value_MaxQuality}, &supportedSizes);
  m_renderingSize = supportedSizes.optimalSize;

  DlssRayReconstruction::InitInfo initInfo{
      .inputSize  = m_renderingSize,
      .outputSize = size,
  };
  m_dlss.deinit();
  vkDeviceWaitIdle(m_device);
  m_dlss.cmdInit(cmd, m_ngx, initInfo);

  // Recreate the G-Buffers
  m_dlssGBuffers.update(cmd, m_renderingSize);

  return m_renderingSize;
}

void DlssDenoiser::setResources()
{
  if(!m_dlssSupported || !m_initialized)
    return;

  auto dlssResourceFromGBufTexture = [&](DlssRayReconstruction::ResourceType resource, shaderio::OutputImage gbufIndex) {
    m_dlss.setResource({resource, m_dlssGBuffers.getColorImage(gbufIndex), m_dlssGBuffers.getColorImageView(gbufIndex),
                        m_dlssGBuffers.getColorFormat(gbufIndex)});
  };

  // #DLSS Fill the user pool with our textures
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eColorIn, shaderio::OutputImage::eResultImage);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eDiffuseAlbedo, shaderio::OutputImage::eDlssAlbedo);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eSpecularAlbedo, shaderio::OutputImage::eDlssSpecAlbedo);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eNormalRoughness, shaderio::OutputImage::eDlssNormalRoughness);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eMotionVector, shaderio::OutputImage::eDlssMotion);
  dlssResourceFromGBufTexture(DlssRayReconstruction::ResourceType::eDepth, shaderio::OutputImage::eDlssDepth);
}

void DlssDenoiser::setResource(DlssRayReconstruction::ResourceType resourceId, VkImage image, VkImageView imageView, VkFormat format)
{
  m_dlss.setResource({resourceId, image, imageView, format});
}

void DlssDenoiser::denoise(VkCommandBuffer cmd, glm::vec2 jitter, const glm::mat4& modelView, const glm::mat4& projection, bool reset /*= false*/)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  m_dlss.cmdDenoise(cmd, m_ngx, {jitter, modelView, projection, reset});
}

void DlssDenoiser::onUi(Resources& resources)
{
  if(!m_dlssSupported && m_initialized)
  {
    ImGui::Text("DLSS is not available");
    return;
  }

  ImGui::Checkbox("Enable DLSS", &m_settings.enable);
  if(!m_initialized)
    return;

  if(!m_settings.enable)
    return;

  ImGui::Text("Resolution: %d x %d", m_renderingSize.width, m_renderingSize.height);

  ImVec2 tumbnailSize = {100 * m_dlssGBuffers.getAspectRatio(), 100};
  int    m_showBuffer = -1;
  auto   showBuffer   = [&](const char* name, shaderio::OutputImage buffer) {
    ImGui::Text("%s", name);
    if(ImGui::ImageButton(name, ImTextureID(m_dlssGBuffers.getDescriptorSet(buffer)), tumbnailSize))
      m_showBuffer = buffer;
  };

  if(ImGui::CollapsingHeader("Guide Images"))
  {
    if(ImGui::BeginTable("thumbnails", 2))
    {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      showBuffer("Color", shaderio::OutputImage::eDlssAlbedo);
      ImGui::TableNextColumn();
      showBuffer("Normal", shaderio::OutputImage::eDlssNormalRoughness);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      showBuffer("Motion", shaderio::OutputImage::eDlssMotion);
      ImGui::TableNextColumn();
      showBuffer("ViewZ", shaderio::OutputImage::eDlssDepth);
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      showBuffer("Specular Albedo", shaderio::OutputImage::eDlssSpecAlbedo);
      ImGui::EndTable();
    }
  }
}
