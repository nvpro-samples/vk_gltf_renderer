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

#include <nvvk/descriptors.hpp>
#include <nvvk/render_target.hpp>

#include "resources.hpp"
#include "settings_registry.hpp"

//-------------------------------------------------------------------------------------------------------
// DenoiserHandoff
//
// DLSS-RR combined with an accumulated-image denoiser (OptiX or OIDN): DLSS denoises frame 0, the image
// seen while the camera moves, and the accumulation denoiser takes the frames after it, once the camera
// stops. Its first images come from a few samples and look worse than DLSS's, so it does not replace
// DLSS's image at once: the image shown blends from one to the other as the samples accumulate,
//
//   shown = lerp(dlss, denoised, handoffWeight(samples, handoffEnd))      (denoiser_schedule.hpp)
//
// and is the denoised image alone from `handoffEnd` on (the hand-off setting, or Max Samples when lower).
// Until the first denoise of the accumulation, DLSS's image is shown, past `handoffEnd` too: OptiX
// denoises one frame behind, so a frame that reaches the end may have nothing denoised yet. Meanwhile the denoiser's Auto
// schedule denoises each time the samples double (AutoDenoiseSchedule::isDue).
//
//   handoff.cmdCapture(cmd, resources, denoiser);                     // frame 1: keep DLSS's image
//   handoff.cmdUpdate(cmd, resources, {.denoiser = &denoiser, ...});  // each rendered frame, after the denoisers
//   if(const VkDescriptorImageInfo* image = handoff.displayImage())   // the tonemapper's input
//
// The hand-off is shown in place of the denoiser's own view (DisplayBuffer): picking another view in
// the UI, or turning the denoiser off, ends it.
//-------------------------------------------------------------------------------------------------------
class DenoiserHandoff
{
public:
  struct Settings
  {
    int spp = 64;  // Samples per pixel the blend lasts; 0 switches to the denoised image at its first denoise
  };

  // The accumulation denoiser of the frame (OptiX or OIDN)
  struct Denoiser
  {
    DisplayBuffer         view = DisplayBuffer::eRendered;  // Its view: the hand-off is shown in its place
    VkDescriptorImageInfo image{};              // Its denoised image (RGBA32F storage, VK_IMAGE_LAYOUT_GENERAL)
    bool                  autoDenoise = false;  // Its Auto is on: without it, nothing may be denoised to hand off to
    uint32_t denoiseCount = 0;  // OptiX's and OIDN's denoises so far: more than at the capture means this accumulation has one
  };

  // What cmdUpdate() blends for
  struct Frame
  {
    const Denoiser* denoiser   = nullptr;  // The accumulation denoiser, null when none is on
    int             samples    = 0;        // Samples per pixel of the image this frame
    int             handoffEnd = 0;        // Samples per pixel the hand-off ends at
  };

  void init(Resources& resources);
  void deinit(Resources& resources);
  void registerParameters(SettingsRegistry* settings);
  void onResize(VkCommandBuffer cmd, VkExtent2D size);

  int spp() const { return m_settings.spp; }

  // Frame 1 after DLSS's frame 0, before the path tracer overwrites it: copies DLSS's image
  // (eImgRendered) and shows it in place of the denoiser's view. Nothing when the setting is 0 or
  // the denoiser's Auto is off.
  void cmdCapture(VkCommandBuffer cmd, Resources& resources, const Denoiser& denoiser);
  // Each rendered frame, after the denoisers
  void cmdUpdate(VkCommandBuffer cmd, Resources& resources, const Frame& frame);
  // The camera moved, the size changed: the kept image is gone
  void cancel();

  // The image to show instead of the denoiser's (null when the hand-off is not shown)
  const VkDescriptorImageInfo* displayImage() const { return m_shown ? &m_shownImage : nullptr; }

  // The setting, shown when DLSS-RR and an accumulation denoiser are both on
  void onUi();

private:
  enum Image
  {
    eImageDlss,  // DLSS's image, kept (RGBA32F, the format of eImgRendered: copied as is)
    eImageOut,   // The blend shown (RGBA16F)
    eImageCount
  };

  Settings m_settings{};

  bool                  m_active = false;  // From the capture until the denoised image is shown alone
  bool                  m_shown  = false;  // The hand-off is the image shown this frame
  VkDescriptorImageInfo m_shownImage{};
  uint32_t              m_captureDenoiseCount = 0;
  float                 m_weight              = 0.0f;  // Share of the denoised image, for the UI

  nvvk::RenderTarget          m_target;
  nvvkgltf::GpuMemoryTracker* m_appMemoryTracker = nullptr;
  nvvk::DescriptorBindings    m_bindings;
  VkDescriptorSetLayout       m_descriptorSetLayout{};
  VkPipelineLayout            m_pipelineLayout{};
  VkPipeline                  m_pipeline{};
};
