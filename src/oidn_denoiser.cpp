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
// vk_oidn (Open Image Denoise on Vulkan) integration for path-traced images.
// The denoising is recorded in the frame's command buffer (see oidn_denoiser.hpp).
//

#include "oidn_denoiser.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include <nvgui/fonts.hpp>
#include <nvgui/property_editor.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "ui_denoiser_controls.hpp"

// Name of the GPU timer section around the auto-denoise (Profiler window, GPU time in the settings)
static constexpr const char* kProfilerSection = "OIDN denoiser";

void OidnDenoiser::requestDeviceExtensions(VkInstance instance, VkPhysicalDevice physicalDevice, nvvk::ContextInitInfo& info)
{
  // Tensor cores; the feature structure must outlive the device creation
  static VkPhysicalDeviceCooperativeMatrixFeaturesKHR s_coopMatFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};

  std::vector<const char*> extensions;
  vkoidn::getRequiredDeviceExtensions(instance, physicalDevice, vkGetInstanceProcAddr, extensions);
  for(size_t i = 0; i < extensions.size(); i++)
  {
    const char* name    = extensions[i];
    bool        present = false;
    for(size_t j = 0; j < info.deviceExtensions.size() && !present; j++)
      present = strcmp(info.deviceExtensions[j].extensionName, name) == 0;
    if(present)
      continue;
    const bool coopMat = strcmp(name, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0;
    // Not required: without it the device is still created, and vk_oidn reports what it cannot do
    info.deviceExtensions.push_back({.extensionName = name, .feature = coopMat ? &s_coopMatFeature : nullptr, .required = false});
  }
}

void OidnDenoiser::init(Resources& resources)
{
  m_availability     = Availability::eUnavailable;
  m_device           = resources.allocator.getDevice();
  m_appMemoryTracker = &resources.appMemoryTracker;

  // Denoised output and the two guides, allocated once the denoiser is enabled (updateSize / ensureTargets)
  NVVK_CHECK(m_target.init({.device = m_device,
                            .alloc  = &resources.allocator,
                            .colorFormats = {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT},
                            .debugName = "OIDN-IO"}));

  // The context holds the compute pipelines. vk_oidn does not link the Vulkan loader: it gets the
  // functions from vkGet*ProcAddr, and picks its optional features from the enabled extensions.
  std::vector<const char*> extensions;
  for(size_t i = 0; i < resources.deviceExtensions.size(); i++)
    extensions.push_back(resources.deviceExtensions[i].c_str());
  const vkoidn::ContextCreateInfo contextInfo{
      .instance              = resources.instance,
      .physicalDevice        = resources.allocator.getPhysicalDevice(),
      .device                = m_device,
      .getInstanceProcAddr   = vkGetInstanceProcAddr,
      .getDeviceProcAddr     = vkGetDeviceProcAddr,
      .enabledExtensions     = extensions.data(),
      .enabledExtensionCount = uint32_t(extensions.size()),
  };
  const vkoidn::Result result = vkoidn::Context::create(contextInfo, m_context);
  if(result != vkoidn::Result::eSuccess)
  {
    LOGW("OidnDenoiser::init: vk_oidn context creation failed (%s) with %zu device extensions. OIDN denoiser disabled.\n",
         vkoidn::getResultString(result), extensions.size());
    m_context.reset();
    return;
  }
  // The renderer's images are given as storage image views, which need VK_KHR_push_descriptor
  if(!m_context->areImageViewsSupported())
  {
    LOGW("OidnDenoiser::init: vk_oidn cannot use image views on %s (VK_KHR_push_descriptor). OIDN denoiser disabled.\n",
         m_context->getDeviceName());
    m_context.reset();
    return;
  }
  LOGI("vk_oidn %d.%d.%d: tensor cores (VK_KHR_cooperative_matrix): %s\n", VK_OIDN_VERSION_MAJOR, VK_OIDN_VERSION_MINOR,
       VK_OIDN_VERSION_PATCH, m_context->isCooperativeMatrixEnabled() ? "yes" : "no");

  if(resources.app && !resources.app->isHeadless())
    m_imageUi.init({.device = m_device, .colorFormats = {resources.app->getSwapchainFormat()}});

  m_availability = Availability::eAvailable;
}

void OidnDenoiser::deinit(Resources& /*resources*/)
{
  // Release the denoiser and the context before the device
  m_denoiser.reset();
  m_context.reset();

  if(m_appMemoryTracker)
    m_appMemoryTracker->untrack("OIDN/GBuffers", m_target, eGBufferCount);
  m_imageUi.deinit();
  m_target.deinit();

  m_hasValidOutput = false;
  m_guides.restart();
  m_availability = Availability::eNotChecked;
}

void OidnDenoiser::registerParameters(SettingsRegistry* settings)
{
  // Bounds mirror the UI (Interval slider 1..1000, Quality and Guides combos 0..2) so CLI, benchmark
  // sequences, and ImGui.ini restore cannot set values the UI would reject.
  settings->add({"oidnEnable", "OIDN Denoiser: Enable the vk_oidn denoiser"}, &m_settings.enable, Persist::eYes);
  settings->add({"oidnAutoDenoiseEnabled", "OIDN Denoiser: Auto-denoise every N samples per pixel and at the max samples"},
                &m_settings.autoDenoiseEnabled, Persist::eYes);
  settings->add({"oidnAutoDenoiseSpp", "OIDN Denoiser: Auto-denoise interval, in samples per pixel"},
                &m_settings.autoDenoiseSpp, Persist::eYes, 1, 1000);
  settings->add({"oidnQuality", "OIDN Denoiser: Network quality [Fast:0, Balanced:1, High:2]"}, &m_settings.quality,
                Persist::eYes, 0, 2);
  settings->add({"oidnGuides", "OIDN Denoiser: Guide images [Color:0, Color+Albedo:1, Color+Albedo+Normal:2]"},
                &m_settings.guides, Persist::eYes, 0, 2);
}

VkDescriptorImageInfo OidnDenoiser::getDescriptorImageInfo(GBufferIndex index) const
{
  return m_target.getColorStorageImageInfo(index);
}

//---------------------------------------------------------
// (Re)allocates the images; the guides start over.
void OidnDenoiser::resizeTargets(VkCommandBuffer cmd, VkExtent2D size)
{
  if(m_appMemoryTracker)
    m_appMemoryTracker->untrack("OIDN/GBuffers", m_target, eGBufferCount);
  NVVK_CHECK(m_target.update(cmd, size));
  m_target.cmdClear(cmd);
  if(m_appMemoryTracker)
    m_appMemoryTracker->track("OIDN/GBuffers", m_target, eGBufferCount);

  m_guides.restart();
  m_hasValidOutput = false;
}

//---------------------------------------------------------
// The denoiser is created for a size, a quality and a set of guides. Creating it records the
// upload of the network weights into cmd, which must execute before the first cmdDenoise().
// Images larger than vk_oidn's default limit are processed in tiles.
void OidnDenoiser::createDenoiser(VkCommandBuffer cmd)
{
  m_denoiser.reset();
  m_hasValidOutput = false;
  m_denoiseTimeMs  = 0.0;
  m_autoDenoise.forgetDenoised();  // the last denoise came from the previous network
  m_createdQuality = m_settings.quality;
  m_createdGuides  = m_settings.guides;

  const VkExtent2D                 size = m_target.getSize();
  const vkoidn::DenoiserCreateInfo info{
      .width   = size.width,
      .height  = size.height,
      .hdr     = true,  // the accumulated color is HDR radiance
      .albedo  = m_settings.guides >= eGuidesAlbedo,
      .normal  = m_settings.guides >= eGuidesAlbedoNormal,
      .quality = vkoidn::Quality(m_settings.quality),
  };
  const vkoidn::Result result = vkoidn::Denoiser::create(*m_context, cmd, info, m_denoiser);
  if(result != vkoidn::Result::eSuccess)
  {
    LOGE(
        "OidnDenoiser::createDenoiser: vk_oidn denoiser creation failed (%s) for %ux%u, quality %d, guides %d. OIDN "
        "denoiser disabled.\n",
        vkoidn::getResultString(result), size.width, size.height, m_settings.quality, m_settings.guides);
    m_denoiser.reset();
    m_availability = Availability::eUnavailable;
  }
}

//---------------------------------------------------------
// Called by the application when the window size changes.
void OidnDenoiser::updateSize(VkCommandBuffer cmd, VkExtent2D size)
{
  if(!m_settings.enable || !isAvailable() || size.width == 0 || size.height == 0)
    return;
  resizeTargets(cmd, size);
  createDenoiser(cmd);
}

bool OidnDenoiser::ensureTargets(Resources& resources)
{
  if(!m_settings.enable || !isAvailable())
    return false;

  const VkExtent2D size = resources.gBuffers.getSize();
  if(size.width == 0 || size.height == 0)
    return false;
  const VkExtent2D current = m_target.getSize();
  const bool       resize  = current.width != size.width || current.height != size.height;
  if(!resize && m_denoiser && m_createdQuality == m_settings.quality && m_createdGuides == m_settings.guides)
    return false;

  // SYNC NOTE: rare (enable / quality / guides change); the old images and network may still be
  // used by a frame in flight.
  NVVK_CHECK(vkQueueWaitIdle(resources.app->getQueue(0).queue));
  VkCommandBuffer cmd = resources.app->createTempCmdBuffer();
  if(resize)
    resizeTargets(cmd, size);  // a new network alone keeps the accumulated guides
  createDenoiser(cmd);
  resources.app->submitAndWaitTempCmdBuffer(cmd);
  return true;
}

//---------------------------------------------------------
// Records the denoising of the accumulated color (G-buffer eImgRendered) into cmd.
bool OidnDenoiser::cmdDenoise(VkCommandBuffer cmd, Resources& resources, int samples)
{
  if(!m_denoiser)
    return false;
  // Guides the path tracer has not written yet would be garbage (enabled on an idle image)
  if(!m_guides.isCurrent() && m_createdGuides != eGuidesNone)
    return false;

  NVVK_DBG_SCOPE(cmd);
  const VkExtent2D size = m_target.getSize();

  // Before: the color and guides written by the path tracer must be visible, and the previous output
  // must no longer be read (tonemapper, preview thumbnail).
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT
                             | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

  // vk_oidn writes the RGB channels and preserves the alpha of the output: start from the rendered
  // image so the denoised one keeps its coverage alpha (like OptiX's ALPHA_MODE_COPY).
  const VkImageSubresourceLayers layers{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1};
  const VkImageCopy copy{.srcSubresource = layers, .dstSubresource = layers, .extent = {size.width, size.height, 1}};
  vkCmdCopyImage(cmd, resources.gBuffers.getColorImage(Resources::eImgRendered), VK_IMAGE_LAYOUT_GENERAL,
                 m_target.getColorImage(eGBufferDenoised), VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

  // Storage image views in VK_IMAGE_LAYOUT_GENERAL; the RGB channels are used
  constexpr VkFormat    format = VK_FORMAT_R32G32B32A32_SFLOAT;
  vkoidn::DenoiseInputs inputs;
  inputs.colorView  = {resources.gBuffers.getColorAttachmentView(Resources::eImgRendered), format};
  inputs.outputView = {m_target.getColorAttachmentView(eGBufferDenoised), format};
  if(m_createdGuides >= eGuidesAlbedo)
    inputs.albedoView = {m_target.getColorAttachmentView(eGBufferAlbedo), format};
  if(m_createdGuides >= eGuidesAlbedoNormal)
    inputs.normalView = {m_target.getColorAttachmentView(eGBufferNormal), format};
  // inputs.inputScale: NaN (default) = automatic exposure, computed on the GPU

  const vkoidn::Result result = m_denoiser->cmdDenoise(cmd, inputs);
  if(result != vkoidn::Result::eSuccess)
  {
    LOGE("OidnDenoiser::cmdDenoise: vk_oidn cmdDenoise failed (%s) at %d spp\n", vkoidn::getResultString(result), samples);
    return false;
  }

  // After: the denoised image is read by the tonemapper and the preview thumbnail
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

  m_hasValidOutput  = true;
  m_denoisedSamples = samples;
  ++m_denoiseCount;
  return true;
}

// The profiler section is in the frame on every frame OIDN is on, like OptiX's, and empty on the frames
// that do not denoise. A frame's timer is resolved a few frames later, in the slot (subFrame) that the
// current frame's section reuses, so that slot keeps whether its frame denoised: the GPU time shown in
// the settings and reported headless is that of the last denoise, not of an empty frame.
void OidnDenoiser::readGpuTime(uint32_t subFrame)
{
  nvutils::ProfilerTimeline::TimerInfo info;
  std::string                          apiName;
  if(m_denoisedInSubFrame[subFrame] && m_profilerTimeline
     && m_profilerTimeline->getFrameTimerInfo(kProfilerSection, info, apiName) && info.gpu.last > 0.0)
    m_denoiseTimeMs = info.gpu.last / 1000.0;
}

void OidnDenoiser::updateDenoiser(VkCommandBuffer cmd, Resources& resources, nvvk::ProfilerGpuTimer* profiler, int samples, int handoffEnd)
{
  const nvutils::ProfilerTimeline::FrameSectionID section = profiler->cmdFrameBeginSection(cmd, kProfilerSection);
  readGpuTime(section.subFrame);
  // The frame that reaches the samples cap is denoised whatever the interval: its image is the final one
  const bool final = samples >= resources.settings.maxSamples;
  const bool due   = isEnabled() && m_settings.autoDenoiseEnabled
                   && m_autoDenoise.isDue(samples, m_settings.autoDenoiseSpp, final, handoffEnd);
  const bool denoised                    = due && autoDenoise(cmd, resources, samples);
  m_denoisedInSubFrame[section.subFrame] = denoised;
  profiler->cmdFrameEndSection(cmd, section);
}

void OidnDenoiser::denoiseFinal(VkCommandBuffer cmd, Resources& resources, nvvk::ProfilerGpuTimer* profiler)
{
  // The network follows the settings on idle frames too (quality or guides set by MCP or a script once
  // the render is done): a new network forgets the last denoise, so the final image is denoised again.
  // ensureTargets() waits for the queue when it recreates the network: kept out of the profiler section.
  const bool autoDenoiseOn = isEnabled() && m_settings.autoDenoiseEnabled;
  if(autoDenoiseOn)
    ensureTargets(resources);
  const bool due = autoDenoiseOn && m_autoDenoise.isFinalDue(resources.accumulatedSamples);

  const nvutils::ProfilerTimeline::FrameSectionID section = profiler->cmdFrameBeginSection(cmd, kProfilerSection);
  readGpuTime(section.subFrame);
  bool denoised = false;
  if(due && m_denoiser)
  {
    denoised = autoDenoise(cmd, resources, resources.accumulatedSamples);
    m_autoDenoise.markDenoised(resources.accumulatedSamples);  // a failed denoise is not retried every idle frame
  }
  m_denoisedInSubFrame[section.subFrame] = denoised;
  profiler->cmdFrameEndSection(cmd, section);
}

bool OidnDenoiser::autoDenoise(VkCommandBuffer cmd, Resources& resources, int samples)
{
  if(!cmdDenoise(cmd, resources, samples))
    return false;
  m_autoDenoise.markDenoised(samples);
  // Automatically switch to display the denoised result
  resources.settings.displayBuffer = DisplayBuffer::eOidnDenoised;
  return true;
}

bool OidnDenoiser::denoiseNow(Resources& resources)
{
  if(!isEnabled())
    return false;
  ensureTargets(resources);  // can run while the renderer is idle (converged)
  if(!m_denoiser)
    return false;

  // Commands earlier in submission order are covered by the first barrier of cmdDenoise()
  VkCommandBuffer cmd = resources.app->createTempCmdBuffer();
  const bool      ok  = cmdDenoise(cmd, resources, resources.accumulatedSamples);
  resources.app->submitAndWaitTempCmdBuffer(cmd);
  return ok;
}

void OidnDenoiser::disable(Resources& resources)
{
  m_settings.enable = false;
  if(resources.settings.displayBuffer == DisplayBuffer::eOidnDenoised)
    resources.settings.displayBuffer = DisplayBuffer::eRendered;
}

bool OidnDenoiser::onUiActivation(Resources& resources)
{
  bool       changed = false;
  const bool avail   = m_availability != Availability::eUnavailable;

  bool        enabled = m_settings.enable;
  const char* state   = "Off";
  ImVec4      color   = nvsamples::denoiserui::mutedColor();
  if(!avail)
  {
    state = "Unavailable";
    color = nvsamples::denoiserui::unavailableColor();
  }
  else if(m_settings.enable)
  {
    state = "On";
    color = nvsamples::denoiserui::readyColor();
  }

  const bool wasEnabled = m_settings.enable;
  if(nvsamples::denoiserui::featureRow("oidn", "OIDN AI Denoiser", &enabled, avail, state, color, &m_settingsOpen,
                                       avail ? "Enable the Open Image Denoise denoiser (vk_oidn, Vulkan compute)." :
                                               "vk_oidn initialization failed; see the log for the missing device support.",
                                       "Show OIDN denoiser settings."))
  {
    // The images and the network are allocated by ensureTargets() on the next rendered frame or denoise.
    m_settings.enable = enabled;
    if(!m_settings.enable && (resources.settings.displayBuffer == DisplayBuffer::eOidnDenoised))
      resources.settings.displayBuffer = DisplayBuffer::eRendered;
    // Toggling keeps the accumulation: the guides restart on the next frame. Restart it only when
    // enabling on a converged image: the renderer is idle then, and the guides need rendered frames.
    const bool converged = resources.accumulationComplete();
    changed              = m_settings.enable && !wasEnabled && converged;
  }
  return changed;
}

// One labeled image of the preview row; returns its top-left corner. Guides the network does not use
// are dimmed.
ImVec2 OidnDenoiser::thumbnailUi(const char* label, GBufferIndex index, bool used, ImVec2 size, nvapp::ImTextureVisualizer::Settings settings)
{
  ImGui::BeginGroup();
  if(used)
    ImGui::TextUnformatted(label);
  else
  {
    ImGui::TextDisabled("%s (unused)", label);
    settings.scale.w = 0.35f;
  }
  const ImVec2 imgPos = ImGui::GetCursorScreenPos();
  if(m_imageUi.isValid())
    m_imageUi.image(m_target.getUiImageView(index), size, settings);
  else
    ImGui::Dummy(size);
  ImGui::EndGroup();
  return imgPos;
}

bool OidnDenoiser::onUiSettings(Resources& resources)
{
  if(!m_settingsOpen)
    return false;

  const bool avail = m_availability != Availability::eUnavailable;

  ImGui::Indent();
  ImGui::PushID("oidn_settings");

  if(!avail)
  {
    ImGui::TextDisabled("vk_oidn is unavailable on this device; see the log.");
    ImGui::PopID();
    ImGui::Unindent();
    return false;
  }

  ImGui::TextDisabled("OIDN AI Denoiser settings");

  // Settings body: greyed when disabled
  ImGui::BeginDisabled(!m_settings.enable);

  if(ImGui::Button(ICON_MS_AUTO_FIX_HIGH " Denoise Now"))
  {
    if(denoiseNow(resources))
      resources.settings.displayBuffer = DisplayBuffer::eOidnDenoised;
    else
      LOGW("OIDN denoising skipped: no rendered frame with guides yet\n");
  }
  nvsamples::denoiserui::tooltip("Denoise the current accumulation and show the result.");

  namespace PE = nvgui::PropertyEditor;
  PE::begin(__FUNCTION__);

  static const char* s_qualities[] = {"Fast", "Balanced", "High"};
  bool               newNetwork = PE::Combo("Quality", &m_settings.quality, s_qualities, IM_ARRAYSIZE(s_qualities), -1,
                                            "Size of the neural network: Fast is almost twice as fast as High.");
  static const char* s_guides[] = {"Color only", "Color + Albedo", "Color + Albedo + Normal"};
  newNetwork |= PE::Combo("Guides", &m_settings.guides, s_guides, IM_ARRAYSIZE(s_guides), -1,
                          "Guide (auxiliary) images given to the denoiser.");

  PE::Checkbox("Auto", &m_settings.autoDenoiseEnabled, "Denoise automatically every N accumulated samples per pixel.");
  if(m_settings.autoDenoiseEnabled)
  {
    PE::SliderInt("Interval", &m_settings.autoDenoiseSpp, 1, 1000, "%d spp", ImGuiSliderFlags_Logarithmic,
                  "1 denoises every frame, in the same command buffer as the rendering (no lag).");
    if(m_settings.autoDenoiseSpp > 1)
      ImGui::TextDisabled("Next: %d spp", AutoDenoiseSchedule::nextSamples(resources.accumulatedSamples, m_settings.autoDenoiseSpp,
                                                                           resources.settings.maxSamples));
  }
  PE::end();

  // A new network replaces the shown result, also while the renderer is idle (converged)
  if(newNetwork && m_hasValidOutput)
    denoiseNow(resources);

  if(m_denoiser)
  {
    ImGui::TextDisabled("Network: %s, %s", m_denoiser->getWeightsName(),
                        m_context->isCooperativeMatrixEnabled() ? "tensor cores" : "no tensor cores");
    ImGui::TextDisabled("Memory: %.1f MB, tiles: %u", double(m_denoiser->getMemoryUsage()) / (1024.0 * 1024.0),
                        m_denoiser->getTileCount());
    // GPU time of the auto-denoise, updated by updateDenoiser()
    if(m_denoiseTimeMs > 0.0)
      ImGui::TextDisabled("GPU time: %.2f ms", m_denoiseTimeMs);
  }

  ImGui::EndDisabled();

  ImGui::PopID();
  ImGui::Unindent();
  return false;  // Denoising never changes the accumulation
}

bool OidnDenoiser::onUiPreview(Resources& resources)
{
  if(!m_hasValidOutput || !m_settings.enable)
    return false;

  if(!ImGui::TreeNodeEx("OIDN Output Preview"))
    return false;

  const bool isActive = (resources.settings.displayBuffer == DisplayBuffer::eOidnDenoised);

  ImGui::AlignTextToFramePadding();
  ImGui::TextDisabled("Viewport");
  ImGui::SameLine();
  if(nvsamples::denoiserui::activeButton("Rendered", !isActive))
    resources.settings.displayBuffer = DisplayBuffer::eRendered;
  nvsamples::denoiserui::tooltip("Show the main rendered image.");
  ImGui::SameLine();
  if(nvsamples::denoiserui::activeButton("Denoised", isActive))
    resources.settings.displayBuffer = DisplayBuffer::eOidnDenoised;
  nvsamples::denoiserui::tooltip("Show the OIDN denoised image.");
  ImGui::Spacing();

  // One row: the denoised image (click to show it in the viewport) and the accumulated guides
  const VkExtent2D size        = m_target.getSize();
  const float      aspectRatio = size.height > 0 ? float(size.width) / float(size.height) : 1.0f;
  const float      spacing     = ImGui::GetStyle().ItemSpacing.x;
  float            thumbW      = std::min((ImGui::GetContentRegionAvail().x - 2.0f * spacing) / 3.0f, 160.0f);
  float            thumbH      = thumbW / std::max(0.1f, aspectRatio);
  if(thumbH > 100.0f)
  {
    thumbH = 100.0f;
    thumbW = thumbH * aspectRatio;
  }
  const ImVec2 thumbnailSize = {std::max(thumbW, 1.0f), std::max(thumbH, 1.0f)};

  // Linear-to-sRGB gamma so the linear images look right in the ImGui panel; normals mapped to [0,1]
  const nvapp::ImTextureVisualizer::Settings linearToSrgb{.pow = glm::vec4(1.0f / 2.2f, 1.0f / 2.2f, 1.0f / 2.2f, 1.0f)};
  const nvapp::ImTextureVisualizer::Settings normalToColor{.scale = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f),
                                                           .add   = glm::vec4(0.5f, 0.5f, 0.5f, 0.0f)};

  const ImVec2 imgPos =
      thumbnailUi(isActive ? "Denoised (Active)" : "Denoised", eGBufferDenoised, true, thumbnailSize, linearToSrgb);
  if(ImGui::IsItemClicked())
    resources.settings.displayBuffer = isActive ? DisplayBuffer::eRendered : DisplayBuffer::eOidnDenoised;
  nvsamples::denoiserui::tooltip("Click to toggle the denoised image in the viewport.");
  if(isActive)
    ImGui::GetWindowDrawList()->AddRect(imgPos, ImVec2(imgPos.x + thumbnailSize.x, imgPos.y + thumbnailSize.y),
                                        IM_COL32(0, 255, 0, 255), 0.0f, 0, 3.0f);
  ImGui::SameLine();
  thumbnailUi("Albedo", eGBufferAlbedo, m_createdGuides >= eGuidesAlbedo, thumbnailSize, linearToSrgb);
  ImGui::SameLine();
  thumbnailUi("Normal", eGBufferNormal, m_createdGuides >= eGuidesAlbedoNormal, thumbnailSize, normalToColor);

  ImGui::TreePop();
  return false;  // Display buffer toggle is a view-only switch; no accumulation reset needed.
}
