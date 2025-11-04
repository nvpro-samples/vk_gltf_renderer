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

#include <nvgui/tooltip.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/debug_util.hpp>
#include <imgui/imgui.h>

#include "dlss_denoiser.hpp"

void DlssDenoiser::registerParameters(nvutils::ParameterRegistry* paramReg)
{
  paramReg->add({"dlssEnable", "DLSS Denoiser: Enable DLSS denoiser"}, &m_settings.enable);
}

//--------------------------------------------------------------------------------------------------
// Initialization, not actual creation - called at PathTracer startup
// See also tryInitializeNGX() for expensive NGX init
void DlssDenoiser::init(Resources& resources)
{
  // Early exit if hardware/extensions not available (already logged in main.cpp)
  if(!resources.settings.dlssHardwareAvailable)
  {
    m_state = DlssState::eUnavailable;
    return;  // Don't create GBuffers if DLSS will never work
  }

  // Create GBuffers (fast operation - only done if hardware available)
  resources.samplerPool.acquireSampler(m_linearSampler);
  m_dlssGBuffers.init({.allocator      = &resources.allocator,
                       .colorFormats   = m_bufferInfos,
                       .imageSampler   = m_linearSampler,
                       .descriptorPool = resources.descriptorPool});

  // State remains eNotChecked - NGX initialization will be attempted on first use
}

//--------------------------------------------------------------------------------------------------
void DlssDenoiser::deinit(Resources& resources)
{
  if(m_state != DlssState::eUnavailable)
  {
    resources.samplerPool.releaseSampler(m_linearSampler);
    m_dlssGBuffers.deinit();
    m_dlss.deinit();
    m_ngx.deinit();
  }
  m_state = DlssState::eNotChecked;
}

//--------------------------------------------------------------------------------------------------
// Expensive NGX initialization (2-5 seconds) - lazy called on first DLSS use
// Returns true ONLY on first successful initialization (triggers resource setup)
bool DlssDenoiser::tryInitializeNGX(Resources& resources)
{
  // Already checked?
  if(m_state != DlssState::eNotChecked)
    return false;  // Return false if already initialized (not first time)

  SCOPED_TIMER("DLSS NGX Initialization (may take 2-5 seconds)");
  LOGI("DLSS: Starting NGX initialization (this may take a few seconds)...\n");

  m_device = resources.allocator.getDevice();

  // #DLSS - Initialize NGX (EXPENSIVE: 2-5 seconds)
  NgxContext::InitInfo ngxInitInfo{
      .instance       = resources.instance,
      .physicalDevice = resources.allocator.getPhysicalDevice(),
      .device         = resources.allocator.getDevice(),
  };
  // ngxInitInfo.loggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;

  NVSDK_NGX_Result ngxResult = m_ngx.init(ngxInitInfo);
  if(ngxResult != NVSDK_NGX_Result_Success)
  {
    LOGW("DLSS: NGX initialization failed (error: %d) - DLSS disabled\n", ngxResult);
    m_state = DlssState::eUnavailable;
    return false;
  }

  // Check DLSS Ray Reconstruction support
  if(m_ngx.isDlssRRAvailable() != NVSDK_NGX_Result_Success)
  {
    LOGW("DLSS: Ray Reconstruction not available - DLSS disabled\n");
    m_state = DlssState::eUnavailable;
    return false;
  }

  // Success!
  m_state = DlssState::eAvailable;
  LOGI("DLSS: Successfully initialized and ready\n");
  return true;
}


VkDescriptorImageInfo DlssDenoiser::getDescriptorImageInfo(shaderio::OutputImage name) const
{
  return m_dlssGBuffers.getDescriptorImageInfo(name);
}

//--------------------------------------------------------------------------------------------------
bool DlssDenoiser::isAvailable() const
{
  return m_state == DlssState::eAvailable;
}

//--------------------------------------------------------------------------------------------------
bool DlssDenoiser::isEnabled() const
{
  // Before NGX initialization attempt, return user's setting (allows lazy init trigger)
  if(m_state == DlssState::eNotChecked)
    return m_settings.enable;

  // After NGX check, DLSS is enabled only if:
  // 1. User has enabled it in settings
  // 2. NGX initialization succeeded
  return m_settings.enable && (m_state == DlssState::eAvailable);
}

VkExtent2D DlssDenoiser::updateSize(VkCommandBuffer cmd, VkExtent2D size)
{
  if(m_state != DlssState::eAvailable)
    return size;

  // Query the supported sizes
  DlssRayReconstruction::SupportedSizes supportedSizes{};
  NVSDK_NGX_Result                      result =
      DlssRayReconstruction::querySupportedInputSizes(m_ngx, {size, NVSDK_NGX_PerfQuality_Value_MaxQuality}, &supportedSizes);
  if(NVSDK_NGX_FAILED(result))
  {
    m_renderingSize = size;
    LOGE("DLSS: Failed to query supported input sizes: %d\n", result);
    return m_renderingSize;  // Return the original size if query fails
  }

  // Choose the size based on the selected mode
  switch(m_settings.sizeMode)
  {
    case SizeMode::eMin:
      m_renderingSize = supportedSizes.minSize;
      break;
    case SizeMode::eMax:
      m_renderingSize = supportedSizes.maxSize;
      break;
    case SizeMode::eOptimal:
    default:
      m_renderingSize = supportedSizes.optimalSize;
      break;
  }

  // Update the last used size mode and clear the change flag
  m_sizeModeChanged = false;

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
  if(m_state != DlssState::eAvailable)
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
  if(m_state != DlssState::eAvailable)
    return;
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  reset = reset || m_forceReset;
  m_dlss.cmdDenoise(cmd, m_ngx, {jitter, modelView, projection, reset});
  m_forceReset = false;
}

bool DlssDenoiser::onUi(Resources& resources)
{
  bool changed = false;

  // Check if DLSS is unavailable (hardware missing or NGX init failed)
  if(!resources.settings.dlssHardwareAvailable || m_state == DlssState::eUnavailable)
  {
    ImGui::BeginDisabled();
    bool dummyEnable = false;
    ImGui::Checkbox("Enable DLSS", &dummyEnable);
    ImGui::EndDisabled();
    ImGui::SameLine();

    // Show appropriate reason
    const char* reason = !resources.settings.dlssHardwareAvailable ? "(Hardware/extensions not available)" : "(NGX initialization failed)";
    ImGui::TextDisabled("%s", reason);
    return changed;
  }

  // Note: If eNotChecked, user can still enable - lazy init will happen on first render
  if(ImGui::Checkbox("Enable DLSS", &m_settings.enable))
  {
    m_forceReset = true;  // Force a reset when enabling/disabling DLSS
    changed      = true;
  }

  if(!m_settings.enable || m_state != DlssState::eAvailable)
    return changed;

  // Size mode selection
  const char* sizeModes[]     = {"Min", "Optimal", "Max"};
  int         currentSizeMode = static_cast<int>(m_settings.sizeMode);

  namespace PE = nvgui::PropertyEditor;
  PE::begin();
  if(PE::Checkbox("Skip Transparent Surfaces", &m_useDlssTransp,
                  "Skip transparent surfaces when collecting DLSS auxiliary data. "
                  "Uses the first opaque/diffuse surface instead of the first hit. "
                  "May improve denoiser quality with transparent materials, but results vary by scene."))
  {
    changed = true;
  }
  if(PE::Combo("DLSS Size Mode", &currentSizeMode, sizeModes, IM_ARRAYSIZE(sizeModes)))
  {
    m_settings.sizeMode = static_cast<SizeMode>(currentSizeMode);
    m_sizeModeChanged   = true;  // Mark that size mode has changed
    changed             = true;  // Mark that changes were made
  }
  PE::end();

  ImGui::Text("Current Resolution: %d x %d", m_renderingSize.width, m_renderingSize.height);


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

  return changed;
}

bool DlssDenoiser::needsSizeUpdate() const
{
  if(m_state != DlssState::eAvailable)
    return false;
  return m_sizeModeChanged;
}
