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

#pragma once

#include "settings_registry.hpp"

#include <array>
#include <memory>

#include <nvapp/imgui_texture_visualizer.hpp>
#include <nvutils/profiler.hpp>
#include <nvvk/context.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/render_target.hpp>

// Open Image Denoise on Vulkan (https://github.com/nvpro-samples/vk_oidn)
#include <vk_oidn/vk_oidn.h>

#include "denoiser_schedule.hpp"
#include "gpu_memory_tracker.hpp"
#include "resources.hpp"

//-------------------------------------------------------------------------------------------------------
// OidnDenoiser
//
// Denoises the path-traced image with vk_oidn, which runs the Intel Open Image Denoise networks with
// Vulkan compute shaders (tensor cores through VK_KHR_cooperative_matrix when available).
//
// Unlike the OptiX denoiser, there is no CUDA, no buffer copy and no host wait: cmdDenoise() is
// recorded in the frame's own command buffer, right after the path tracer, and reads the
// accumulated color and guides as storage images. The result of a frame is therefore denoised in
// that same frame, without the one-frame lag of OptiX.
//
// The path tracer accumulates the albedo and normal guides like the color (ePtDenoiserGuides, the
// same shader code as for OptiX), into the images this class owns, so their edges are anti-aliased
// exactly like the color.
//
// Usage:
//   OidnDenoiser::requestDeviceExtensions(...)  // nvvk::ContextInitInfo::postSelectPhysicalDeviceCallback
//   denoiser.init(resources);                    // vkoidn::Context (compute pipelines), once
//   denoiser.updateSize(cmd, size);              // onResize: images + vkoidn::Denoiser (weights upload)
//   denoiser.ensureTargets(resources);           // every frame: follows the settings, however set
//   denoiser.updateDenoiser(cmd, resources, p, spp);  // every frame, after the path tracer: auto-denoise
//   denoiser.denoiseFinal(cmd, resources, p);  // every idle frame, the accumulation complete
//   denoiser.deinit(resources);
//
// Like OptiX, not every frame is denoised: auto-denoise runs every N accumulated samples per pixel
// and on the final image (AutoDenoiseSchedule), and "Denoise Now" denoises the current accumulation
// on demand.
//-------------------------------------------------------------------------------------------------------

class OidnDenoiser
{
public:
  // Images owned by the denoiser (all RGBA32F, storage, at the G-buffer size)
  enum GBufferIndex
  {
    eGBufferDenoised = 0,  // Denoised output
    eGBufferAlbedo   = 1,  // Accumulated albedo guide (written by the path tracer)
    eGBufferNormal   = 2,  // Accumulated world-space normal guide (written by the path tracer)
    eGBufferCount
  };

  // Guide (auxiliary) images given to the network
  enum Guides
  {
    eGuidesNone,          // Color only
    eGuidesAlbedo,        // Color + albedo
    eGuidesAlbedoNormal,  // Color + albedo + normal
  };

  enum class Availability
  {
    eNotChecked,   // Haven't attempted initialization yet
    eUnavailable,  // Device unsupported or initialization failed
    eAvailable,    // Context created and ready to use
  };

  struct Settings
  {
    bool enable             = true;
    bool autoDenoiseEnabled = true;                         // Automatically denoise every N samples per pixel
    int  autoDenoiseSpp     = 50;                           // Denoise at 50, 100, 150, ... spp
    int  quality            = int(vkoidn::Quality::eHigh);  // vkoidn::Quality: eFast, eBalanced, eHigh
    int  guides             = eGuidesAlbedoNormal;          // Guides
  };

  OidnDenoiser()  = default;
  ~OidnDenoiser() = default;

  // Adds the device extensions vk_oidn can use (VK_KHR_cooperative_matrix for the tensor cores, ...).
  // Call from nvvk::ContextInitInfo::postSelectPhysicalDeviceCallback. Its core features
  // (shaderFloat16, storageBuffer16BitAccess, ...) are enabled by nvvk::Context (enableAllFeatures).
  static void requestDeviceExtensions(VkInstance instance, VkPhysicalDevice physicalDevice, nvvk::ContextInitInfo& info);

  void init(Resources& resources);
  void deinit(Resources& resources);

  /// Declare this subsystem's settings (command line + benchmark + MCP + persistence).
  void registerParameters(SettingsRegistry* settings);

  bool isAvailable() const { return m_availability == Availability::eAvailable; }
  bool isEnabled() const { return m_settings.enable && (m_availability != Availability::eUnavailable); }
  bool isAutoDenoiseActive() const { return m_settings.autoDenoiseEnabled; }
  bool hasValidDenoisedOutput() const { return m_hasValidOutput; }
  int  denoisedSamples() const { return m_denoisedSamples; }  // Samples per pixel of the denoised image

  // Denoises made so far (auto and Denoise Now)
  uint32_t denoiseCount() const { return m_denoiseCount; }

  VkDescriptorImageInfo getDescriptorImageInfo(GBufferIndex index) const;

  // The path tracer reports whether it writes the guides this frame; returns the samples already in
  // them (see GuideAccumulation)
  int updateGuideState(bool writtenThisFrame, int totalSamples)
  {
    return m_guides.update(writtenThisFrame, totalSamples);
  }

  // Called when the window size changes
  void updateSize(VkCommandBuffer cmd, VkExtent2D size);
  // The one place the images and the network follow `enable` / `quality` / `guides`, however those
  // were set (UI, command line, sequence script, MCP, ini restore). A few compares when nothing
  // changed. Returns true when it reallocated.
  bool ensureTargets(Resources& resources);

  // Auto-denoise: records the denoising into the frame's command buffer when it is due. Call after
  // the path tracer wrote the color and the guides; `samples` is the spp of that accumulated image.
  // The frame that reaches the samples cap is always denoised. `handoffEnd`: the spp the DLSS hand-off
  // ends at, 0 without (AutoDenoiseSchedule).
  void updateDenoiser(VkCommandBuffer cmd, Resources& resources, nvvk::ProfilerGpuTimer* profiler, int samples, int handoffEnd = 0);
  // Auto-denoise on an idle frame, the accumulation complete: denoises its final image if the last
  // denoise was not of it (e.g. Auto turned on, or the cap lowered below the accumulation)
  void denoiseFinal(VkCommandBuffer cmd, Resources& resources, nvvk::ProfilerGpuTimer* profiler);
  // Denoises the current accumulation now, in its own command buffer (works when the renderer is idle)
  bool denoiseNow(Resources& resources);

  void setProfilerTimeline(nvutils::ProfilerTimeline* timeline) { m_profilerTimeline = timeline; }
  // GPU time of the last auto-denoise in ms (its profiler frame section; the work is in the frame's command buffer)
  double getGpuTimeMs() const { return m_denoiseTimeMs; }

  // UI controls
  bool onUiActivation(Resources& resources);
  // Turn off, as the checkbox does. The accumulation is kept.
  void disable(Resources& resources);
  bool onUiSettings(Resources& resources);
  bool onUiPreview(Resources& resources);

private:
  void resizeTargets(VkCommandBuffer cmd, VkExtent2D size);
  void createDenoiser(VkCommandBuffer cmd);
  bool cmdDenoise(VkCommandBuffer cmd, Resources& resources, int samples);
  bool autoDenoise(VkCommandBuffer cmd, Resources& resources, int samples);
  void readGpuTime(uint32_t subFrame);
  ImVec2 thumbnailUi(const char* label, GBufferIndex index, bool used, ImVec2 size, nvapp::ImTextureVisualizer::Settings settings);

  Settings          m_settings{};
  Availability      m_availability    = Availability::eNotChecked;
  bool              m_hasValidOutput  = false;
  int               m_denoisedSamples = 0;  // Samples per pixel of the image the output was denoised from
  uint32_t          m_denoiseCount    = 0;  // Denoises made so far
  GuideAccumulation m_guides;               // Samples in the guide images
  std::array<bool, nvutils::ProfilerTimeline::MAX_FRAME_DELAY> m_denoisedInSubFrame{};  // See readGpuTime()
  bool                                                         m_settingsOpen = false;
  double m_denoiseTimeMs = 0.0;  // Last measured GPU time of the auto-denoise

  AutoDenoiseSchedule m_autoDenoise;

  // vk_oidn objects: the context holds the compute pipelines (once per device), the denoiser the
  // network weights and intermediate data (per size, quality and guides)
  std::unique_ptr<vkoidn::Context>  m_context;
  std::unique_ptr<vkoidn::Denoiser> m_denoiser;
  int                               m_createdQuality = -1;  // Settings the denoiser was created with
  int                               m_createdGuides  = -1;

  VkDevice                    m_device           = VK_NULL_HANDLE;
  nvvkgltf::GpuMemoryTracker* m_appMemoryTracker = nullptr;
  nvutils::ProfilerTimeline*  m_profilerTimeline = nullptr;

  nvvk::RenderTarget         m_target;  // See GBufferIndex
  nvapp::ImTextureVisualizer m_imageUi{};
};
