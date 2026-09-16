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

//
// Dlss -- application-side adapter for NVIDIA DLSS (NGX). Drives both DLSS Ray Reconstruction
// (path-tracer denoiser + upscaler, owns an 8-attachment guide GBuffer) and Super Resolution/DLAA
// (rasterizer AA + upscaling, owns a 3-attachment + depth inner GBuffer), bridging the
// project-independent dlss_wrapper (NgxContext + DlssFeature) to Resources, GBuffers, ImGui, and
// the renderer's output image.
//

#include "dlss.hpp"

#if defined(USE_DLSS)

#include <algorithm>
#include <cassert>
#include <span>

#include <imgui/imgui.h>

#include <nvgui/fonts.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/settings_handler.hpp>
#include <nvgui/tooltip.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

// NVSDK_NGX_RayReconstruction_Hint_Render_Preset_* enum values.
#include <nvsdk_ngx_defs_dlssd.h>

#if defined(USE_DLSSNR)
// DLSS-NR (Neural Rendering) — beta feature, separate from RR/SR.
#include <nvsdk_ngx_helpers_dlssnr_vk.h>
#endif

// Halton sequence utility functions
#include "shaders/dlss_util.h"

#include "ui_denoiser_controls.hpp"


// =============================================================================================
// Per-Kind configuration table
// =============================================================================================
//
// All DLSS-RR vs DLSS-SR config differences (log tag, feature ID, GBuffer formats, preset table) are in this one table.
// SR inner formats are handled at runtime; this table just provides constants.

namespace {

// RR guide-buffer formats. Indices line up with shaderio::OutputImage values: the path tracer writes its 8 guide buffers into m_innerGBuffer.getColorImage(eXxx).
inline constexpr VkFormat kRrInnerFormats[] = {
    VK_FORMAT_R32G32B32A32_SFLOAT,  // #DLSS - Rendered image       : eResultImage
    VK_FORMAT_R32_SFLOAT,           // #DLSS - Object ID in .r      : eSelectImage
    VK_FORMAT_R8G8B8A8_UNORM,       // #DLSS - BaseColor            : eDlssAlbedo
    VK_FORMAT_R16G16B16A16_SFLOAT,  // #DLSS - SpecAlbedo           : eDlssSpecAlbedo
    VK_FORMAT_R16G16B16A16_SFLOAT,  // #DLSS - Normal / Roughness   : eDlssNormalRoughness
    VK_FORMAT_R16G16_SFLOAT,        // #DLSS - Motion vectors       : eDlssMotion
    VK_FORMAT_R16_SFLOAT,           // #DLSS - ViewZ                : eDlssDepth
    VK_FORMAT_R16_SFLOAT,           // #DLSS - Specular Hit Dist    : eDlssSpecularHitDist
    VK_FORMAT_R16G16B16A16_SFLOAT,  // #DLSS-NR - Per-material mask : eNrMask
};

// One entry in a preset combo. `value` is the NGX preset enum (RR or SR table; both share the
// same letter -> integer mapping so a single uint32_t works for either side).
struct PresetEntry
{
  uint32_t    value;
  const char* label;
};

// RR preset table (NVSDK_NGX_RayReconstruction_Hint_Render_Preset -> label). Only the presets documented as functional in nvsdk_ngx_defs.h are exposed.
inline constexpr PresetEntry kRrPresets[] = {
    {NVSDK_NGX_RayReconstruction_Hint_Render_Preset_Default, "Default"},
    {NVSDK_NGX_RayReconstruction_Hint_Render_Preset_D, "D (transformer)"},
    {NVSDK_NGX_RayReconstruction_Hint_Render_Preset_E, "E (latest transformer)"},
};

// SR preset table (NVSDK_NGX_DLSS_Hint_Render_Preset). Only the presets documented as functional in nvsdk_ngx_defs_dlssd.h are exposed.
inline constexpr PresetEntry kSrPresets[] = {
    {NVSDK_NGX_DLSS_Hint_Render_Preset_Default, "Default"},
    {NVSDK_NGX_DLSS_Hint_Render_Preset_J, "J- Similar to preset K"},
    {NVSDK_NGX_DLSS_Hint_Render_Preset_K, "K- Default preset for DLAA/Balanced/Quality"},
    {NVSDK_NGX_DLSS_Hint_Render_Preset_L, "L- Default for Ultra Perf mode"},
    {NVSDK_NGX_DLSS_Hint_Render_Preset_M, "M- Default for Perf mode"},
};

// Per-Kind configuration: every constant that differs between DLSS-RR and DLSS-SR collected
// in one struct so the dispatch helpers below (kindConfig()) hand it out by Kind.
struct KindConfig
{
  const char*                  logTag;             // For LOGI / LOGW / LOGE prefix.
  NVSDK_NGX_Feature            ngxFeature;         // NGX feature ID for isFeatureAvailable().
  std::span<const VkFormat>    innerColorFormats;  // RR: full list; SR: empty (computed in initSr()).
  std::span<const PresetEntry> presets;            // User-selectable NGX network presets.
};

inline constexpr KindConfig kRrConfig{
    .logTag            = "DLSS-RR",
    .ngxFeature        = NVSDK_NGX_Feature_RayReconstruction,
    .innerColorFormats = std::span<const VkFormat>(kRrInnerFormats),
    .presets           = std::span<const PresetEntry>(kRrPresets),
};

inline constexpr KindConfig kSrConfig{
    .logTag            = "DLSS-SR",
    .ngxFeature        = NVSDK_NGX_Feature_SuperSampling,
    .innerColorFormats = {},  // SR layout is fallback-dependent; built in initSr().
    .presets           = std::span<const PresetEntry>(kSrPresets),
};

// Hand out the right per-Kind configuration row.
inline const KindConfig& kindConfig(Dlss::Kind k)
{
  return (k == Dlss::Kind::RR) ? kRrConfig : kSrConfig;
}

constexpr int kPendingInitNone              = 0;
constexpr int kPendingInitUnavailable       = 1;
constexpr int kPendingInitAvailable         = 2;
constexpr int kPendingInitAvailableWithNr   = 3;
constexpr int kPendingInitUnavailableWithNr = 4;  // main DLSS feature absent, but NR is available (SR fallback)

}  // namespace


// ============================================================================================
// Constructor
// ============================================================================================

Dlss::Dlss(Kind kind)
    : m_kind(kind)
    , m_dlss(kind)
{
}


// ============================================================================================
// Common lifecycle
// ============================================================================================

void Dlss::init(Resources& resources)
{
  m_appMemoryTracker = &resources.appMemoryTracker;
  m_graphicsQueue    = resources.app ? resources.app->getQueue(0).queue : VK_NULL_HANDLE;
  if(m_linearSampler == VK_NULL_HANDLE)
    resources.samplerPool.acquireSampler(m_linearSampler);

#if defined(USE_DLSSNR)
  // NR scratch GBuffer: SFLOAT for NGX internal compatibility; blitted to eImgTonemapped (UNORM)
  // after evaluation. Allocated at resize time (updateSizeNr); just wire the allocator here.
  NVVK_CHECK(m_nrGBuffer.init({.device       = resources.allocator.getDevice(),
                               .alloc        = &resources.allocator,
                               .colorFormats = {VK_FORMAT_R32G32B32A32_SFLOAT},
                               .debugName    = "DLSS-NR"}));
#endif

  if(m_kind == Kind::RR)
    initRr(resources);
  else
    initSr(resources);

  // Skip the guide-image table when the inner GBuffer wasn't allocated by initRr() -- this
  // happens when DLSS-RR hardware isn't available on the current GPU.
  const bool rrSkipped = (m_kind == Kind::RR && m_state == InitStatus::eUnavailable);
  if(!rrSkipped)
  {
    buildGuideEntries();
    if(resources.app && !resources.app->isHeadless())
      m_guideVisualizer.init({.device = resources.allocator.getDevice(), .colorFormats = {resources.app->getSwapchainFormat()}});
  }

  startAsyncInit(resources);
}

void Dlss::deinit(Resources& resources)
{
  // If an async NGX init is in flight, block until the worker is done before tearing down NGX
  // state -- otherwise the worker could call into a half-destroyed NgxContext.
  if(m_initThread.joinable())
    m_initThread.join();
  m_initInProgress.store(false, std::memory_order_release);
  m_pendingInitResult.store(kPendingInitNone, std::memory_order_release);

  m_guideVisualizer.deinit();
#if defined(USE_DLSSNR)
  destroyNr();
  m_nrGBuffer.deinit();
#endif

  if(m_kind == Kind::RR)
    deinitRr(resources);
  else
    deinitSr(resources);  // reads m_nrAvailable to decide whether to deinit NGX

#if defined(USE_DLSSNR)
  m_nrAvailable = false;
#endif

  if(m_linearSampler != VK_NULL_HANDLE)
  {
    resources.samplerPool.releaseSampler(m_linearSampler);
    m_linearSampler = VK_NULL_HANDLE;
  }

  m_state         = InitStatus::eNotChecked;
  m_selectedGuide = -1;
  m_guideEntries.clear();
}

// NGX init logic, runs on worker thread. Only CPU-side Vulkan calls; result published via m_pendingInitResult.
Dlss::InitStatus Dlss::runNgxInit(const NgxContext::InitInfo& ngxInitInfo, bool& nrAvailable)
{
  const KindConfig& t = kindConfig(m_kind);

  nrAvailable = false;

  if(m_ngx.init(ngxInitInfo) != NVSDK_NGX_Result_Success)
  {
    LOGW("%s: NGX initialization failed - DLSS disabled\n", t.logTag);
    return InitStatus::eUnavailable;
  }

#if defined(USE_DLSSNR)
  // Probe NR before the main feature check: NR can be available even when DLSS-SR is not,
  // so we must discover it before potentially tearing down the NGX context.
  nrAvailable = NVSDK_NGX_SUCCEED(m_ngx.isFeatureAvailable(NVSDK_NGX_Feature_DLSSNR));
  LOGI("DLSS-NR: %s\n", nrAvailable ? "Available" : "Not available on this GPU/driver");
#endif

  if(m_ngx.isFeatureAvailable(t.ngxFeature) != NVSDK_NGX_Result_Success)
  {
    LOGW("%s: feature not available on this driver/hardware - DLSS disabled\n", t.logTag);
    if(!nrAvailable)
      m_ngx.deinit();  // Keep NGX alive when NR will still use it.
    return InitStatus::eUnavailable;
  }

  return InitStatus::eAvailable;
}

void Dlss::startAsyncInit(Resources& resources)
{
  if(m_state != InitStatus::eNotChecked)
    return;
  if(m_pendingInitResult.load(std::memory_order_acquire) != kPendingInitNone)
    return;

  bool expected = false;
  if(!m_initInProgress.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
    return;

  NgxContext::InitInfo ngxInitInfo{
      .instance       = resources.instance,
      .physicalDevice = resources.allocator.getPhysicalDevice(),
      .device         = resources.allocator.getDevice(),
  };

  m_initThread = std::thread([this, ngxInitInfo]() {
    SCOPED_TIMER("DLSS NGX Initialization (async, may take 2-5 seconds)");
    bool             nrAvailable = false;
    const InitStatus result      = runNgxInit(ngxInitInfo, nrAvailable);

    int pending = kPendingInitUnavailable;
    if(result == InitStatus::eAvailable)
      pending = nrAvailable ? kPendingInitAvailableWithNr : kPendingInitAvailable;
    else if(nrAvailable)
      pending = kPendingInitUnavailableWithNr;  // main DLSS feature absent, but NR is available

    // Publish the result first, then clear the in-progress flag so an observer that sees
    // !m_initInProgress also sees the pending result.
    m_pendingInitResult.store(pending, std::memory_order_release);
    m_initInProgress.store(false, std::memory_order_release);
  });
}

bool Dlss::consumePendingInitResult()
{
  const int pending = m_pendingInitResult.exchange(kPendingInitNone, std::memory_order_acq_rel);
  if(pending == kPendingInitNone)
    return false;

  if(m_initThread.joinable())
    m_initThread.join();

  m_state = (pending == kPendingInitUnavailable || pending == kPendingInitUnavailableWithNr) ? InitStatus::eUnavailable :
                                                                                               InitStatus::eAvailable;
#if defined(USE_DLSSNR)
  m_nrAvailable = (pending == kPendingInitAvailableWithNr || pending == kPendingInitUnavailableWithNr);
#endif
  if(m_state == InitStatus::eAvailable)
  {
    LOGI("%s: Successfully initialized and ready\n", kindConfig(m_kind).logTag);
    m_needsRecreate = true;
    return true;
  }

#if defined(USE_DLSSNR)
  if(pending == kPendingInitUnavailableWithNr)
  {
    LOGI("%s: DLSS feature unavailable but DLSS-NR is available\n", kindConfig(m_kind).logTag);
    m_needsRecreate = false;
    return false;
  }
#endif

  LOGW("%s: NGX initialization failed\n", kindConfig(m_kind).logTag);
  m_needsRecreate = false;
  return false;
}

void Dlss::teardownNgx()
{
  m_dlss.deinit();
  m_ngx.deinit();
}


// ============================================================================================
// Common queries -- public State machine + the private predicates feeding it
// ============================================================================================

bool Dlss::isUserEnabled() const
{
  return (m_kind == Kind::RR) ? m_settings.enableRr : (getQuality() != Quality::eOff);
}

Dlss::State Dlss::state() const
{
  // (1) Sticky permanent failure: hardware doesn't support DLSS, or async NGX init failed.
  //     Highest priority -- once we know it can't work, the user's toggle doesn't matter.
  if(m_state == InitStatus::eUnavailable)
    return State::eUnsupported;

  // (2) User intent: if the user disabled DLSS (or the SR Quality is eOff), we're off
  //     regardless of system state.
  if(!isUserEnabled())
    return State::eOff;

  // (3) Async NGX init in flight, or its first result has not been consumed yet.
  if(m_state == InitStatus::eNotChecked || m_initInProgress.load(std::memory_order_acquire))
    return State::eLoading;

  // (4) NGX is up. Brief transient window where the worker just settled but the renderer
  //     hasn't yet called updateSize() to allocate the inner GBuffer images. tick() returns
  //     true the same frame we land here, so the renderer's updateDlssResources() runs and
  //     this branch ceases to fire.
  if(m_innerGBuffer.getSize().width == 0)
    return State::eLoading;

  // (5) SR-only: cmdInit() may have failed at updateSize() time even though NGX itself is up.
  if(m_kind == Kind::SR && !m_dlssCreated)
    return State::eLoading;

  return State::eActive;
}

// Advance the asynchronous NGX initialization state machine and notify the renderer if DLSS
// resources should be (re-)configured this frame. init() starts the normal background prewarm;
// tick() consumes that result, starts a fallback worker if init() could not, and reports either
// first-time NGX readiness or feature recreation requests.
bool Dlss::tick(Resources& resources)
{
  const bool justInitedNgx = consumePendingInitResult();

  const State s = state();
  if(s == State::eUnsupported)
    return false;

  if(m_state == InitStatus::eNotChecked && !m_initInProgress.load(std::memory_order_acquire))
    startAsyncInit(resources);

  // When DLSS-RR is off, keep the NGX prewarm alive but do not allocate its heavy guide
  // buffers until the user enables it. SR is different: even "Off" needs its motion target
  // resized back to native when quality changes.
  if(s == State::eOff && m_kind == Kind::RR)
    return false;

  return justInitedNgx || needsRecreate();
}

bool Dlss::needsRecreate() const
{
  // Only meaningful once NGX is up; before that updateSize() is a no-op.
  return isAvailable() && m_needsRecreate;
}


// ============================================================================================
// Top-level dispatch (updateSize / setResources / onUi)
// ============================================================================================

VkExtent2D Dlss::updateSize(VkCommandBuffer cmd, Resources& resources, VkExtent2D size)
{
  VkExtent2D innerSize = (m_kind == Kind::RR) ? updateSizeRr(cmd, size) : updateSizeSr(cmd, resources, size);
#if defined(USE_DLSSNR)
  // NR always operates at display resolution, independent of inner render resolution.
  updateSizeNr(cmd, size);
#endif
  return innerSize;
}

void Dlss::setResources()
{
  if(m_kind == Kind::RR)
    setResourcesRr();
  else
    setResourcesSr();
}

bool Dlss::onUi(Resources& resources)
{
  bool changed = onUiActivation(resources);
#if defined(USE_DLSSNR)
  changed |= onUiNrActivation();
#endif
  changed |= onUiSettings(resources);
#if defined(USE_DLSSNR)
  changed |= onUiNrSettings();
#endif
  changed |= onUiGuideBuffers();
  return changed;
}

bool Dlss::onUiActivation(Resources& resources)
{
  return (m_kind == Kind::RR) ? onUiActivationRr(resources) : onUiActivationSr(resources);
}

bool Dlss::onUiSettings(Resources& resources)
{
  return (m_kind == Kind::RR) ? onUiSettingsRr() : onUiSettingsSr(resources);
}


#if defined(USE_DLSSNR)
// ============================================================================================
// DLSS-NR (Neural Rendering)
// ============================================================================================

void Dlss::updateSizeNr(VkCommandBuffer cmd, VkExtent2D displaySize)
{
  // Always track the display size so evaluateNr() knows what size to pass to NGX.
  m_nrSize = displaySize;

  if(!m_nrAvailable)
    return;

  // Wait for in-flight GPU work before releasing or recreating the NR handle.
  if(m_nrCreated && m_graphicsQueue != VK_NULL_HANDLE)
    NVVK_CHECK(vkQueueWaitIdle(m_graphicsQueue));

  destroyNr();

  if(!m_nrSettings.enabled)
    return;

  // Create NR feature handle.
  NVSDK_NGX_DLSSNR_Create_Params createParams{displaySize.width, displaySize.height};
  NVSDK_NGX_Result               result =
      NGX_VULKAN_CREATE_DLSSNR_EXT1(m_ngx.getDevice(), cmd, 1, 1, &m_nrHandle, m_ngx.getNgxParams(), &createParams);
  if(NVSDK_NGX_FAILED(result))
  {
    LOGW("DLSS-NR: NGX_VULKAN_CREATE_DLSSNR_EXT1 failed: 0x%x\n", result);
    m_nrHandle  = nullptr;
    m_nrCreated = false;
    return;
  }
  m_nrCreated    = true;
  m_nrForceReset = true;  // fresh handle, drop history on first eval

  // Allocate / reallocate the NR output image.
  NVVK_CHECK(m_nrGBuffer.update(cmd, displaySize));
  m_nrGBuffer.cmdClear(cmd);
}

void Dlss::destroyNr()
{
  if(m_nrHandle)
  {
    NVSDK_NGX_VULKAN_ReleaseFeature(m_nrHandle);
    m_nrHandle = nullptr;
  }
  m_nrCreated = false;
}

void Dlss::evaluateNr(VkCommandBuffer cmd)
{
  if(!isNrActive())
    return;
  if(m_nrInputImage == VK_NULL_HANDLE || m_nrInputView == VK_NULL_HANDLE)
    return;

  NVVK_DBG_SCOPE(cmd);

  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

  auto colorRes = NVSDK_NGX_Create_ImageView_Resource_VK(m_nrInputView, m_nrInputImage, range, m_nrInputFormat,
                                                         m_nrSize.width, m_nrSize.height, false);
  auto outputRes =
      NVSDK_NGX_Create_ImageView_Resource_VK(m_nrGBuffer.getColorAttachmentView(0), m_nrGBuffer.getColorImage(0), range,
                                             m_nrGBuffer.getColorFormat(0), m_nrSize.width, m_nrSize.height, true);

  // Depth and motion vectors — needed for NR to temporally stabilise the local structure effect.
  // Kind::RR: both are colour attachments in the inner GBuffer written by the path tracer
  //   (linearised ViewZ R16_SFLOAT; screen-space motion R16G16_SFLOAT) at render resolution.
  //   MV values are in render-pixel space; scale to display-pixel space so NR tracks correctly.
  NVSDK_NGX_Resource_VK  depthRes{}, motionRes{};
  NVSDK_NGX_Resource_VK* pDepth   = nullptr;
  NVSDK_NGX_Resource_VK* pMotion  = nullptr;
  float                  mvScaleX = 1.0f;
  float                  mvScaleY = 1.0f;
  NVSDK_NGX_Dimensions   auxSubrect{m_nrSize.width, m_nrSize.height};

  const VkImageSubresourceRange depthRange{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};

  if(m_kind == Kind::RR && m_innerGBuffer.getSize().width == 0)
    return;  // GBuffer reacquired (init'd) but not yet sized by updateSizeRr(); skip this frame.

  if(m_kind == Kind::RR)
  {
    // Linearised ViewZ (R16_SFLOAT) and motion (R16G16_SFLOAT) are colour attachments in inner GBuffer.
    const VkExtent2D rs        = getRenderSize();
    const auto       depthIdx  = static_cast<uint32_t>(shaderio::OutputImage::eDlssDepth);
    const auto       motionIdx = static_cast<uint32_t>(shaderio::OutputImage::eDlssMotion);
    depthRes                   = NVSDK_NGX_Create_ImageView_Resource_VK(m_innerGBuffer.getColorAttachmentView(depthIdx),
                                                                        m_innerGBuffer.getColorImage(depthIdx), range,
                                                                        m_innerGBuffer.getColorFormat(depthIdx), rs.width, rs.height, false);
    motionRes  = NVSDK_NGX_Create_ImageView_Resource_VK(m_innerGBuffer.getColorAttachmentView(motionIdx),
                                                        m_innerGBuffer.getColorImage(motionIdx), range,
                                                        m_innerGBuffer.getColorFormat(motionIdx), rs.width, rs.height, false);
    pDepth     = &depthRes;
    pMotion    = &motionRes;
    auxSubrect = {rs.width, rs.height};
    mvScaleX   = rs.width > 0 ? float(m_nrSize.width) / float(rs.width) : 1.0f;
    mvScaleY   = rs.height > 0 ? float(m_nrSize.height) / float(rs.height) : 1.0f;
  }
  else if(m_kind == Kind::SR)
  {
    // Hardware depth attachment (reversed-Z) and motion colour attachment in the SR inner GBuffer.
    const VkExtent2D rs = getRenderSize();
    depthRes   = NVSDK_NGX_Create_ImageView_Resource_VK(getSrImageView(SrSlot::eDepth), getSrImage(SrSlot::eDepth),
                                                        depthRange, getSrFormat(SrSlot::eDepth), rs.width, rs.height, false);
    motionRes  = NVSDK_NGX_Create_ImageView_Resource_VK(getSrImageView(SrSlot::eMotion), getSrImage(SrSlot::eMotion),
                                                        range, getSrFormat(SrSlot::eMotion), rs.width, rs.height, false);
    pDepth     = &depthRes;
    pMotion    = &motionRes;
    auxSubrect = {rs.width, rs.height};
    mvScaleX   = rs.width > 0 ? float(m_nrSize.width) / float(rs.width) : 1.0f;
    mvScaleY   = rs.height > 0 ? float(m_nrSize.height) / float(rs.height) : 1.0f;
  }

  NVSDK_NGX_VK_DLSSNR_Eval_Params params{};
  params.pInColor  = &colorRes;
  params.pInOutput = &outputRes;
  // Per-material NR control mask from the RR inner GBuffer (eNrMask slot, path-tracer only).
  NVSDK_NGX_Resource_VK  maskRes{};
  NVSDK_NGX_Resource_VK* pMask = nullptr;
  NVSDK_NGX_Dimensions   maskSubrect{m_nrSize.width, m_nrSize.height};
  if(m_kind == Kind::RR)
  {
    const VkExtent2D rs      = getRenderSize();
    const auto       maskIdx = static_cast<uint32_t>(shaderio::OutputImage::eNrMask);
    maskRes                  = NVSDK_NGX_Create_ImageView_Resource_VK(m_innerGBuffer.getColorAttachmentView(maskIdx),
                                                                      m_innerGBuffer.getColorImage(maskIdx), range,
                                                                      m_innerGBuffer.getColorFormat(maskIdx), rs.width, rs.height, false);
    pMask                    = &maskRes;
    maskSubrect              = {rs.width, rs.height};
  }

  params.pInDepth                 = pDepth;
  params.pInMVec                  = pMotion;
  params.pInControlMask           = pMask;
  params.InEnabled                = 1;
  params.InReset                  = m_nrForceReset ? 1 : 0;
  m_nrForceReset                  = false;
  params.InIntensity              = m_nrSettings.intensity;
  params.InLocalToneStrength      = m_nrSettings.localToneStrength;
  params.InLocalStructureStrength = m_nrSettings.localStructureStrength;
  params.InGlobalToneStrength     = m_nrSettings.globalToneStrength;
  params.InStyle                  = m_nrSettings.style;
  params.InUseAutoMask            = (pMask != nullptr) ? 0 : (m_nrSettings.useAutoMask ? 1 : 0);
  params.InSkinStructureStrength  = m_nrSettings.skinStructureStrength;
  params.InColorSubrectSize       = {m_nrSize.width, m_nrSize.height};
  params.InOutputSubrectSize      = {m_nrSize.width, m_nrSize.height};
  params.InDepthSubrectSize       = auxSubrect;
  params.InMVecSubrectSize        = auxSubrect;
  params.InControlMaskSubrectSize = maskSubrect;
  params.InMVecScaleX             = mvScaleX;
  params.InMVecScaleY             = mvScaleY;
  // RR: linearised ViewZ — not reversed. SR: hardware depth — reversed-Z (far=0, near=1).
  params.InDepthInverted = (m_kind == Kind::SR) ? 1 : 0;

  NVSDK_NGX_Result result = NGX_VULKAN_EVALUATE_DLSSNR_EXT(cmd, m_nrHandle, m_ngx.getNgxParams(), &params);
  if(NVSDK_NGX_FAILED(result))
  {
    LOGW("DLSS-NR: evaluate failed: 0x%x\n", result);
    return;
  }

  // Blit NR output back to eImgTonemapped (NR ran on the LDR display image).
  nvvk::cmdImageMemoryBarrier(cmd, {m_nrGBuffer.getColorImage(0), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});
  nvvk::cmdImageMemoryBarrier(cmd, {m_nrInputImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});

  VkImageBlit2 region{
      .sType          = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffsets     = {{0, 0, 0}, {int32_t(m_nrSize.width), int32_t(m_nrSize.height), 1}},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffsets     = {{0, 0, 0}, {int32_t(m_nrSize.width), int32_t(m_nrSize.height), 1}},
  };
  VkBlitImageInfo2 blitInfo{
      .sType          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
      .srcImage       = m_nrGBuffer.getColorImage(0),
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .dstImage       = m_nrInputImage,
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount    = 1,
      .pRegions       = &region,
      .filter         = VK_FILTER_NEAREST,
  };
  vkCmdBlitImage2(cmd, &blitInfo);

  nvvk::cmdImageMemoryBarrier(cmd, {m_nrGBuffer.getColorImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
  nvvk::cmdImageMemoryBarrier(cmd, {m_nrInputImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
}

bool Dlss::onUiNrActivation()
{
  consumePendingInitResult();

  bool changed = false;

  // For the path-tracer (Kind::RR), NR requires a clean denoised input — disable the toggle
  // when RR is off so the user can't enable NR on a noisy image.
  const bool initPending     = m_state == InitStatus::eNotChecked || m_initInProgress.load(std::memory_order_acquire);
  const bool nrBlocked       = (m_kind == Kind::RR) && !(isAvailable() && m_settings.enableRr);
  const bool enableAvailable = !initPending && m_nrAvailable && !nrBlocked;

  bool        nrEnabled = m_nrSettings.enabled;
  const char* state     = "Ready";
  ImVec4      color     = nvsamples::denoiserui::readyColor();
  if(initPending)
  {
    state = "Checking";
    color = nvsamples::denoiserui::workingColor();
  }
  else if(!m_nrAvailable)
  {
    state = "Unavailable";
    color = nvsamples::denoiserui::unavailableColor();
  }
  else if(nrBlocked)
  {
    state = "Needs RR";
    color = nvsamples::denoiserui::workingColor();
  }
  else if(m_nrSettings.enabled && isNrActive())
  {
    state = "On";
    color = nvsamples::denoiserui::readyColor();
  }

#ifdef USE_DLSSNR
  if(nvsamples::denoiserui::featureRow(
         "dlss_nr", "DLSS Neural Rendering", &nrEnabled, enableAvailable, state, color, &m_nrSettingsOpen,
         initPending ? "DLSS-NR availability is being checked in the background." :
                       (nrBlocked ? "Requires DLSS Ray Reconstruction to be enabled." : "Enable DLSS-NR neural rendering."),
         "Show DLSS-NR settings."))
  {
    m_nrSettings.enabled = nrEnabled;
    changed              = true;
    m_nrForceReset       = true;
    m_needsRecreate      = true;
  }
#endif
  return changed;
}

bool Dlss::onUiNrSettings()
{
  if(!m_nrSettingsOpen)
    return false;

  bool       changed   = false;
  const bool nrBlocked = (m_kind == Kind::RR) && !(isAvailable() && m_settings.enableRr);

  ImGui::Indent();
  ImGui::PushID("dlss_nr_settings");
  if(!m_nrAvailable)
  {
    ImGui::TextDisabled("Not available on this GPU/driver.");
    ImGui::PopID();
    ImGui::Unindent();
    return false;
  }
  if(nrBlocked)
  {
    ImGui::TextDisabled("Requires DLSS-RR to be enabled (NR needs a denoised input).");
    ImGui::PopID();
    ImGui::Unindent();
    return false;
  }

  ImGui::TextDisabled("DLSS Neural Rendering settings");

  // Settings body — greyed when NR is disabled
  if(!m_nrSettings.enabled)
    ImGui::BeginDisabled();

  // Strength sliders: per-frame weights, no temporal reset needed — changing them mid-stream
  // is intentional and should blend smoothly.
  namespace PE = nvgui::PropertyEditor;
  PE::begin();
  auto sliderF = [&](const char* label, float& val, float lo, float hi, const char* tip) {
    if(PE::SliderFloat(label, &val, lo, hi, "%.2f", 0, tip))
      changed = true;
  };
  sliderF("Intensity", m_nrSettings.intensity, 0.0f, 1.0f, "Overall NR effect strength (0 = off, 1 = full).");
  sliderF("Local Tone", m_nrSettings.localToneStrength, 0.0f, 1.0f, "Strength of local tone-mapping adjustments.");
  sliderF("Local Structure", m_nrSettings.localStructureStrength, 0.0f, 1.0f, "Strength of local structure enhancement.");
  sliderF("Global Tone", m_nrSettings.globalToneStrength, 0.0f, 1.0f, "Strength of global tone-mapping adjustments.");
  sliderF("Skin Structure", m_nrSettings.skinStructureStrength, 0.0f, 1.0f, "Skin texture enhancement strength.");

  static const char* kNrStyleLabels[] = {"Default", "A", "B", "C", "D", "E", "F", "G"};
  int                nrStyle          = static_cast<int>(m_nrSettings.style);
  nrStyle                             = std::clamp(nrStyle, 0, IM_ARRAYSIZE(kNrStyleLabels) - 1);
  if(PE::Combo("Style", &nrStyle, kNrStyleLabels, IM_ARRAYSIZE(kNrStyleLabels), 0, "NR style preset."))
  {
    m_nrSettings.style = static_cast<unsigned int>(nrStyle);
    changed            = true;
    m_nrForceReset     = true;  // switching model preset warrants a history reset
  }

  if(PE::Checkbox("Auto Mask", &m_nrSettings.useAutoMask, "Let NR automatically derive a per-pixel enhancement mask."))
    changed = true;
  PE::end();

  if(!m_nrSettings.enabled)
    ImGui::EndDisabled();

  ImGui::PopID();
  ImGui::Unindent();
  return changed;
}

bool Dlss::onUiNr()
{
  bool changed = onUiNrActivation();
  changed |= onUiNrSettings();
  return changed;
}

#endif  // USE_DLSSNR


// ============================================================================================
// Settings / parameter registration (kind-aware)
// ============================================================================================

void Dlss::registerParameters(SettingsRegistry* settings, std::function<void()> onSettingChanged)
{
  // Attach the caller's invalidation hook (if any) as callbackSuccess so a CLI or benchmark
  // change fires it. The Info-with-callback form is used unconditionally to keep the two
  // branches identical; when no hook was passed the field stays default-empty and behaves
  // like a plain add(). See Dlss::registerParameters doc-comment for the ini-restore caveat.
  nvutils::ParameterBase::CallbackSuccess cb;
  if(onSettingChanged)
    cb = [hook = std::move(onSettingChanged)](const nvutils::ParameterBase* const) { hook(); };

  if(m_kind == Kind::RR)
  {
    settings->add({.name = "dlssEnable", .help = "DLSS Denoiser: Enable DLSS denoiser", .callbackSuccess = cb},
                  &m_settings.enableRr, Persist::eYes);
  }
  else
  {
    settings->add({.name = "dlssQuality",
                   .help = "DLSS Super Resolution mode: 0=Off, 1=DLAA, 2=Quality, 3=Balanced, 4=Performance, 5=UltraPerformance",
                   .callbackSuccess = cb},
                  &m_settings.qualityMode, Persist::eYes);
  }
}


// ============================================================================================
// setOutputImage / evaluate (kind-aware via DlssFeature)
// ============================================================================================

void Dlss::setOutputImage(VkImage image, VkImageView imageView, VkFormat format)
{
  if(!isAvailable())
    return;
  m_dlss.setResource({DlssFeature::ResourceType::eColorOut, image, imageView, format});
}

#if defined(USE_DLSSNR)
void Dlss::setNrImage(VkImage image, VkImageView view, VkFormat format)
{
  m_nrInputImage  = image;
  m_nrInputView   = view;
  m_nrInputFormat = format;
}

#endif  // USE_DLSSNR

Dlss::FrameContext Dlss::beginFrame()
{
  // No-op when DLSS won't actually run this frame.
  if(!isActive())
  {
    m_currentJitter = {0.0f, 0.0f};
    return {};
  }

  ++m_frameIndex;
  // Halton low-discrepancy sequence centered on (0,0).
  m_currentJitter = shaderio::dlssJitter(m_frameIndex);
  return {m_currentJitter, m_frameIndex};
}

void Dlss::notifyReset()
{
  // Restart the Halton sequence so we never feed NGX a partial cycle and arm the next evaluate() to drop temporal history.
  m_frameIndex           = 0;
  m_forceResetUntilFrame = m_globalFrame + 1;
#if defined(USE_DLSSNR)
  m_nrForceReset = true;
#endif
}

// Evaluate the DLSS feature.
void Dlss::evaluate(VkCommandBuffer cmd, const glm::mat4& view, const glm::mat4& proj)
{
  assert(isActive() && "Dlss::evaluate() called when isActive()==false");

  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  const bool reset = (m_globalFrame < m_forceResetUntilFrame);
  ++m_globalFrame;
  m_dlss.cmdEvaluate(cmd, m_ngx, {m_currentJitter, view, proj, reset});
}


// ============================================================================================
// Guide-buffer visualization (encapsulated; non-intrusive)
// ============================================================================================

void Dlss::buildGuideEntries()
{
  m_guideEntries.clear();

  if(m_kind == Kind::RR)
  {
    // RR table mirrors the existing 6 thumbnails. Indices match shaderio::OutputImage values
    // because the path tracer writes its guide buffers directly into those attachment slots.
    m_guideEntries.push_back({"Color", static_cast<uint32_t>(shaderio::OutputImage::eDlssAlbedo)});
    m_guideEntries.push_back({"Specular Albedo", static_cast<uint32_t>(shaderio::OutputImage::eDlssSpecAlbedo)});
    m_guideEntries.push_back({"Normal", static_cast<uint32_t>(shaderio::OutputImage::eDlssNormalRoughness)});
    m_guideEntries.push_back({"Motion", static_cast<uint32_t>(shaderio::OutputImage::eDlssMotion)});
    m_guideEntries.push_back({"ViewZ", static_cast<uint32_t>(shaderio::OutputImage::eDlssDepth)});
    m_guideEntries.push_back({"Specular HitT", static_cast<uint32_t>(shaderio::OutputImage::eDlssSpecularHitDist)});
    return;
  }

  // SR: only color attachments in the inner render target make sense as guide thumbnails
  // (depth is excluded — it is a depth format, not a color/float view).
  if(m_fallback)
  {
    // Fallback mode: only motion lives in the inner GBuffer (color/selection route to outer).
    m_guideEntries.push_back({"Motion", kFallbackMotionIdx});
    return;
  }
  m_guideEntries.push_back({"Color", kInnerColorIdx});
  m_guideEntries.push_back({"Motion", kInnerMotionIdx});
}

std::optional<Dlss::GuideImage> Dlss::activeGuideImage() const
{
  if(m_selectedGuide < 0 || m_selectedGuide >= static_cast<int>(m_guideEntries.size()))
    return std::nullopt;
  if(!isActive())
    return std::nullopt;

  const GuideEntry& entry = m_guideEntries[m_selectedGuide];
  return GuideImage{
      .image  = m_innerGBuffer.getColorSampleDescriptorImageInfo(entry.gbufColorIdx, m_linearSampler),
      .extent = m_innerGBuffer.getSize(),
      .name   = entry.label,
  };
}

bool Dlss::onUiGuideBuffers()
{
  if(m_guideEntries.empty())
    return false;

  if(!ImGui::TreeNodeEx("Developer Guide Buffers"))
    return false;

  if(!isActive())
  {
    ImGui::BeginDisabled();
    ImGui::TextUnformatted("Enable DLSS to view guide buffers.");
    ImGui::EndDisabled();
    ImGui::TreePop();
    return false;
  }

  drawGuideThumbnails();
  ImGui::TreePop();
  return false;  // Guide selection is view-only; no accumulation reset needed.
}

bool Dlss::drawGuideThumbnails()
{
  bool changed = false;
  ImGui::AlignTextToFramePadding();
  ImGui::TextDisabled("Viewport");
  ImGui::SameLine();
  if(nvsamples::denoiserui::activeButton("Rendered", m_selectedGuide < 0))
  {
    if(m_selectedGuide >= 0)
    {
      m_selectedGuide = -1;
      changed         = true;
    }
  }
  nvsamples::denoiserui::tooltip("Show the main rendered image.");
  ImGui::Spacing();

  const float  aspect      = std::max(0.1f, m_innerGBuffer.getAspectRatio());
  const ImVec2 itemSpacing = ImGui::GetStyle().ItemSpacing;
  const float  availableX  = ImGui::GetContentRegionAvail().x;
  const int    columns     = std::max(1, std::min(3, static_cast<int>((availableX + itemSpacing.x) / 124.0f)));

  // Apply linear-to-sRGB gamma so the linear guide buffers look correct in the ImGui panel.
  const nvapp::ImTextureVisualizer::Settings linearToSrgb{.pow = glm::vec4(1.0f / 2.2f, 1.0f / 2.2f, 1.0f / 2.2f, 1.0f)};

  if(ImGui::BeginTable("dlss_guide_thumbnails", columns, ImGuiTableFlags_SizingStretchSame))
  {
    for(int i = 0; i < static_cast<int>(m_guideEntries.size()); ++i)
    {
      if((i % columns) == 0)
        ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::PushID(i);

      const GuideEntry& entry    = m_guideEntries[i];
      const bool        isActive = (m_selectedGuide == i);

      ImGui::Text("%s%s", entry.label, isActive ? " (Active)" : "");
      if(ImGui::IsItemHovered())
        ImGui::SetTooltip("Show %s guide in the viewport.", entry.label);

      const float cellW  = ImGui::GetContentRegionAvail().x;
      float       thumbW = std::min(cellW, 116.0f);
      float       thumbH = thumbW / aspect;
      if(thumbH > 92.0f)
      {
        thumbH = 92.0f;
        thumbW = thumbH * aspect;
      }
      const ImVec2 thumbnailSize = {thumbW, thumbH};

      // Draw the guide image with sRGB gamma correction, then overlay an invisible button for clicks.
      const ImVec2 imgPos = ImGui::GetCursorScreenPos();
      // Skip registering the callback when the inner GBuffer is about to be resized this frame
      // (m_needsRecreate set earlier in onUiSettings). The old VkImageView would be destroyed
      // in onRender() before renderToSwapchain() executes the ImGui callback.
      if(m_guideVisualizer.isValid() && !m_needsRecreate)
        m_guideVisualizer.image(m_innerGBuffer.getUiImageView(entry.gbufColorIdx), thumbnailSize, linearToSrgb);
      else
        ImGui::Dummy(thumbnailSize);
      ImGui::SetCursorScreenPos(imgPos);
      ImGui::InvisibleButton(entry.label, thumbnailSize);
      if(ImGui::IsItemClicked())
      {
        // Toggle: re-clicking the active thumbnail goes back to the rendered image.
        m_selectedGuide = isActive ? -1 : i;
        changed         = true;
      }

      // Highlight the active thumbnail with a green border drawn on the draw list.
      if(isActive)
        ImGui::GetWindowDrawList()->AddRect(imgPos, ImVec2(imgPos.x + thumbnailSize.x, imgPos.y + thumbnailSize.y),
                                            IM_COL32(0, 255, 0, 255), 0.0f, 0, 3.0f);
      ImGui::PopID();
    }
    ImGui::EndTable();
  }
  return changed;
}


Dlss::Quality Dlss::getQuality() const
{
  assertKind(Kind::SR);
  const int raw = m_settings.qualityMode;
  if(raw < 0 || raw >= static_cast<int>(Quality::eCount))
    return Quality::eOff;
  return static_cast<Quality>(raw);
}

bool Dlss::isFallback() const
{
  assertKind(Kind::SR);
  return m_fallback;
}

NVSDK_NGX_PerfQuality_Value Dlss::qualityToNgx(Quality q)
{
  switch(q)
  {
    case Quality::eDLAA:
      return NVSDK_NGX_PerfQuality_Value_DLAA;
    case Quality::eQuality:
      return NVSDK_NGX_PerfQuality_Value_MaxQuality;
    case Quality::eBalanced:
      return NVSDK_NGX_PerfQuality_Value_Balanced;
    case Quality::ePerformance:
      return NVSDK_NGX_PerfQuality_Value_MaxPerf;
    case Quality::eUltraPerformance:
      return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
    case Quality::eOff:
    case Quality::eCount:
    default:
      return NVSDK_NGX_PerfQuality_Value_DLAA;  // never used; eOff path skips feature create
  }
}


void Dlss::blitInnerToOuter(VkCommandBuffer cmd, VkImage innerSrc, VkImage outerDst, VkExtent2D outerExtent) const
{
  assertKind(Kind::SR);
  const VkExtent2D innerSz = getRenderSize();

  nvvk::cmdImageMemoryBarrier(cmd, {innerSrc, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});
  nvvk::cmdImageMemoryBarrier(cmd, {outerDst, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});

  VkImageBlit2 region{
      .sType          = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffsets     = {{0, 0, 0}, {int32_t(innerSz.width), int32_t(innerSz.height), 1}},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffsets     = {{0, 0, 0}, {int32_t(outerExtent.width), int32_t(outerExtent.height), 1}},
  };
  VkBlitImageInfo2 blitInfo{
      .sType          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
      .srcImage       = innerSrc,
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .dstImage       = outerDst,
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount    = 1,
      .pRegions       = &region,
      // NEAREST: selection is an object-ID texture; the no-DLSS color path is 1:1.
      .filter = VK_FILTER_NEAREST,
  };
  vkCmdBlitImage2(cmd, &blitInfo);

  nvvk::cmdImageMemoryBarrier(cmd, {innerSrc, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
  nvvk::cmdImageMemoryBarrier(cmd, {outerDst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
}


// ============================================================================================
// Kind-assertion shim (debug-only; release builds rely on call-site Kind discipline)
// ============================================================================================

void Dlss::assertKind([[maybe_unused]] Kind expected) const
{
  assert(m_kind == expected && "Dlss accessor called on the wrong Kind");
}

namespace {

// Renders DLSS preset combo for the active Kind and updates *currentPreset when changed.
bool drawPresetCombo(Dlss::Kind kind, uint32_t* currentPreset, const char* label, const char* tooltip)
{
  const auto presets = kindConfig(kind).presets;
  if(presets.empty())
    return false;

  // Resolve the current selection's index in the table
  int currentIdx = 0;
  for(int i = 0; i < static_cast<int>(presets.size()); ++i)
  {
    if(presets[i].value == *currentPreset)
    {
      currentIdx = i;
      break;
    }
  }

  // Build the labels array in a stack-friendly buffer (preset counts are tiny -- <= ~8).
  const int   labelCount = static_cast<int>(presets.size());
  const char* labels[16];
  assert(labelCount <= static_cast<int>(std::size(labels)) && "kRr/kSrPresets is larger than the labels[] buffer");
  for(int i = 0; i < labelCount; ++i)
    labels[i] = presets[i].label;

  namespace PE = nvgui::PropertyEditor;
  if(PE::Combo(label, &currentIdx, labels, labelCount, 0, tooltip))
  {
    *currentPreset = presets[currentIdx].value;
    return true;
  }
  return false;
}

}  // namespace

// ============================================================================================
// Common UI helper -- "Reset History" button
// ============================================================================================

bool Dlss::drawResetHistoryButton()
{
  bool clicked = false;
  if(ImGui::Button(ICON_MS_REFRESH " Reset History"))
  {
    notifyReset();
    clicked = true;
  }
  nvgui::tooltip("Discard the denoiser's temporal history on the next frame.");
  return clicked;
}


// ============================================================================================
// RR (Ray Reconstruction / Denoiser): inner GBuffer = 8 guide attachments
// ============================================================================================

void Dlss::initRr(Resources& resources)
{
  // Skip GBuffer + sampler allocation entirely when the hardware can't run RR. Keeping the
  // class otherwise functional (state stays eUnavailable) lets the UI render its disabled
  // checkbox without further branching.
  if(!resources.settings.dlssRrHardwareAvailable)
  {
    m_state = InitStatus::eUnavailable;
    return;
  }

  // RenderTarget CreateInfo wants a std::vector; copy from the file-scope constexpr span so the formats
  // table stays the single source of truth.
  const auto& rrFormats = kindConfig(Kind::RR).innerColorFormats;
  NVVK_CHECK(m_innerGBuffer.init({.device       = resources.allocator.getDevice(),
                                  .alloc        = &resources.allocator,
                                  .colorFormats = std::vector<VkFormat>(rrFormats.begin(), rrFormats.end()),
                                  .debugName    = "DLSS-RR"}));
}

void Dlss::deinitRr(Resources& /*resources*/)
{
  if(m_appMemoryTracker)
    m_appMemoryTracker->untrack("DLSS/GBuffers", m_innerGBuffer,
                                static_cast<uint32_t>(kindConfig(Kind::RR).innerColorFormats.size()));
  m_innerGBuffer.deinit();
  if(m_state != InitStatus::eUnavailable)
    teardownNgx();
}

VkExtent2D Dlss::updateSizeRr(VkCommandBuffer cmd, VkExtent2D size)
{
  if(!isAvailable())
    return size;
  // Skip update if DLSS-RR is disabled to avoid crashes and unnecessary memory use.
  if(!m_settings.enableRr)
    return size;

  // RR-flavored wrapper routes querySupportedInputSizes to NGX_DLSSD_GET_OPTIMAL_SETTINGS.
  DlssFeature::SupportedSizes supportedSizes{};
  NVSDK_NGX_Result result = m_dlss.querySupportedInputSizes(m_ngx, {size, NVSDK_NGX_PerfQuality_Value_MaxQuality}, &supportedSizes);
  if(NVSDK_NGX_FAILED(result))
  {
    LOGE("DLSS-RR: Failed to query supported input sizes: %d\n", result);
    return size;
  }

  VkExtent2D renderingSize{};
  switch(m_settings.sizeMode)
  {
    case SizeMode::eMin:
      renderingSize = supportedSizes.minSize;
      break;
    case SizeMode::eMax:
      renderingSize = supportedSizes.maxSize;
      break;
    case SizeMode::eOptimal:
    default:
      renderingSize = supportedSizes.optimalSize;
      break;
  }

  m_needsRecreate = false;

  DlssFeature::InitInfo initInfo{
      .preset     = m_preset,
      .inputSize  = renderingSize,
      .outputSize = size,
  };
  m_dlss.deinit();
  if(m_graphicsQueue)
    NVVK_CHECK(vkQueueWaitIdle(m_graphicsQueue));
  m_dlss.cmdInit(cmd, m_ngx, initInfo);

  // Recreate the G-Buffers
  const uint32_t dlssColorCount = static_cast<uint32_t>(kindConfig(Kind::RR).innerColorFormats.size());
  if(m_appMemoryTracker)
    m_appMemoryTracker->untrack("DLSS/GBuffers", m_innerGBuffer, dlssColorCount);
  NVVK_CHECK(m_innerGBuffer.update(cmd, renderingSize));
  m_innerGBuffer.cmdClear(cmd);
  if(m_appMemoryTracker)
    m_appMemoryTracker->track("DLSS/GBuffers", m_innerGBuffer, dlssColorCount);

  return renderingSize;
}

void Dlss::setResourcesRr()
{
  if(!isAvailable())
    return;
  // Symmetric guard with updateSizeRr()
  if(!m_settings.enableRr)
    return;

  auto bind = [&](DlssFeature::ResourceType resource, shaderio::OutputImage gbufIndex) {
    m_dlss.setResource({resource, m_innerGBuffer.getColorImage(gbufIndex),
                        m_innerGBuffer.getColorAttachmentView(gbufIndex), m_innerGBuffer.getColorFormat(gbufIndex)});
  };

  bind(DlssFeature::ResourceType::eColorIn, shaderio::OutputImage::eResultImage);
  bind(DlssFeature::ResourceType::eDiffuseAlbedo, shaderio::OutputImage::eDlssAlbedo);
  bind(DlssFeature::ResourceType::eSpecularAlbedo, shaderio::OutputImage::eDlssSpecAlbedo);
  bind(DlssFeature::ResourceType::eNormalRoughness, shaderio::OutputImage::eDlssNormalRoughness);
  bind(DlssFeature::ResourceType::eMotionVector, shaderio::OutputImage::eDlssMotion);
  bind(DlssFeature::ResourceType::eDepth, shaderio::OutputImage::eDlssDepth);
  bind(DlssFeature::ResourceType::eSpecularHitDistance, shaderio::OutputImage::eDlssSpecularHitDist);
}

bool Dlss::onUiActivationRr(Resources& resources)
{
  consumePendingInitResult();
  startAsyncInit(resources);

  bool       changed           = false;
  const bool hardwareAvailable = resources.settings.dlssRrHardwareAvailable;
  const bool initPending       = m_state == InitStatus::eNotChecked || m_initInProgress.load(std::memory_order_acquire);
  const bool avail             = hardwareAvailable && m_state != InitStatus::eUnavailable;

  bool        rrEnabled    = m_settings.enableRr;
  const bool  wasEnabledRr = m_settings.enableRr;
  const char* reason       = !hardwareAvailable ? "Hardware/extensions not available." : "NGX initialization failed.";
  const char* enableTip    = avail ? "Enable DLSS-RR denoising and upscaling." : reason;
  const char* uiState      = "Ready";
  ImVec4      uiStateColor = nvsamples::denoiserui::readyColor();
  if(!avail)
  {
    uiState      = "Unavailable";
    uiStateColor = nvsamples::denoiserui::unavailableColor();
  }
  else if(initPending)
  {
    uiState      = "Loading";
    uiStateColor = nvsamples::denoiserui::workingColor();
    enableTip    = "DLSS-RR is initializing in the background.";
  }
  else if(m_settings.enableRr && state() == State::eActive)
  {
    uiState      = "On";
    uiStateColor = nvsamples::denoiserui::readyColor();
  }

  if(nvsamples::denoiserui::featureRow("dlss_rr", "DLSS Ray Reconstruction", &rrEnabled, avail, uiState, uiStateColor,
                                       &m_settingsOpen, enableTip, "Show DLSS-RR settings."))
  {
    m_settings.enableRr = rrEnabled;
    notifyReset();
    if(wasEnabledRr && !m_settings.enableRr)
      releaseRrInnerGBuffer();
    else if(!wasEnabledRr && m_settings.enableRr)
      reacquireRrInnerGBuffer(resources);
    changed = true;
  }
  return changed;
}

bool Dlss::onUiSettingsRr()
{
  if(!m_settingsOpen)
    return false;

  bool changed = false;
  ImGui::Indent();
  ImGui::PushID("dlss_rr_settings");

  if(m_state == InitStatus::eUnavailable)
  {
    ImGui::TextDisabled("DLSS-RR is unavailable on this GPU/driver.");
    ImGui::PopID();
    ImGui::Unindent();
    return false;
  }

  ImGui::TextDisabled("DLSS Ray Reconstruction settings");

  // Settings body — greyed when RR is disabled
  if(!m_settings.enableRr)
    ImGui::BeginDisabled();

  const char* sizeModes[]     = {"Min", "Optimal", "Max"};
  int         currentSizeMode = static_cast<int>(m_settings.sizeMode);

  namespace PE = nvgui::PropertyEditor;
  PE::begin();
  const char* transparencyModes[] = {"Default (first hit)", "Improved (blended guides)"};
  int         currentTransMode    = static_cast<int>(m_transparencyMode);
  if(PE::Combo("Transparency", &currentTransMode, transparencyModes, IM_ARRAYSIZE(transparencyModes), 0,
               "Controls how DLSS guide buffers are generated for transparent materials."))
  {
    m_transparencyMode = static_cast<TransparencyMode>(currentTransMode);
    changed            = true;
  }
  if(PE::Combo("Input Size", &currentSizeMode, sizeModes, IM_ARRAYSIZE(sizeModes)))
  {
    m_settings.sizeMode = static_cast<SizeMode>(currentSizeMode);
    m_needsRecreate     = true;
    notifyReset();  // restart Halton sequence so the next-frame jitter matches the recreated feature
    changed = true;
  }
  // RR preset combo
  if(drawPresetCombo(Kind::RR, &m_preset, "Preset",
                     "DLSS-RR network preset. Default lets NGX choose the SDK default; Preset E is required when a Depth-of-Field guide is used."))
  {
    m_needsRecreate = true;  // NGX bakes the preset at feature creation; previous history is incompatible.
    notifyReset();
    changed = true;
  }
  PE::end();

  // Manual history reset for cuts/teleports/visible-ghosting recovery.
  if(drawResetHistoryButton())
    changed = true;

  const VkExtent2D renderSize = m_innerGBuffer.getSize();
  if(renderSize.width == 0 || renderSize.height == 0)
    ImGui::TextDisabled("Input resolution: pending");
  else
    ImGui::TextDisabled("Input resolution: %u x %u", renderSize.width, renderSize.height);

  if(!m_settings.enableRr)
    ImGui::EndDisabled();

  ImGui::PopID();
  ImGui::Unindent();
  return changed;
}

bool Dlss::useDlssTransparency() const
{
  assertKind(Kind::RR);
  return m_transparencyMode != TransparencyMode::eDefault;
}

Dlss::TransparencyMode Dlss::getTransparencyMode() const
{
  assertKind(Kind::RR);
  return m_transparencyMode;
}

void Dlss::releaseRrInnerGBuffer()
{
  assertKind(Kind::RR);
  // Drain in-flight references before destroying images
  if(m_graphicsQueue != VK_NULL_HANDLE)
    NVVK_CHECK(vkQueueWaitIdle(m_graphicsQueue));

  if(m_appMemoryTracker)
    m_appMemoryTracker->untrack("DLSS/GBuffers", m_innerGBuffer,
                                static_cast<uint32_t>(kindConfig(Kind::RR).innerColorFormats.size()));

  m_innerGBuffer.deinit();
  m_guideEntries.clear();
  m_selectedGuide = -1;
}

void Dlss::reacquireRrInnerGBuffer(Resources& resources)
{
  assertKind(Kind::RR);
  // Refuse to reacquire when the hardware never supported DLSS-RR
  if(!resources.settings.dlssRrHardwareAvailable)
    return;

  // Clean up the old GBuffer
  m_innerGBuffer.deinit();

  // Create a new inner render target
  const auto& rrFormats = kindConfig(Kind::RR).innerColorFormats;
  NVVK_CHECK(m_innerGBuffer.init({.device       = resources.allocator.getDevice(),
                                  .alloc        = &resources.allocator,
                                  .colorFormats = std::vector<VkFormat>(rrFormats.begin(), rrFormats.end()),
                                  .debugName    = "DLSS-RR"}));

  buildGuideEntries();     // rebuilds the 6-entry RR table; safe -- the GBuffer is init()'d.
  m_needsRecreate = true;  // setupPushConstant -> updateDlssResources -> updateSizeRr next frame.
  notifyReset();           // clean Halton restart + arm reset on the first post-reacquire evaluate.
}

VkImage Dlss::getRrImage(shaderio::OutputImage which) const
{
  assertKind(Kind::RR);
  return m_innerGBuffer.getColorImage(static_cast<uint32_t>(which));
}

VkDescriptorImageInfo Dlss::getRrAttachment(shaderio::OutputImage which) const
{
  assertKind(Kind::RR);
  return m_innerGBuffer.getColorStorageImageInfo(static_cast<uint32_t>(which));
}


// ============================================================================================
// SR (Super Resolution): inner GBuffer = color/selection/motion (+ depth) or motion-only fallback
// ============================================================================================

void Dlss::initSr(Resources& resources)
{
  // Common setup runs regardless of HW availability so the rasterizer's 3-attachment pipeline
  // always has a valid motion attachment to render into.
  m_innerDepthFormat = resources.gBuffers.getDepthFormat();
  m_fallback         = !resources.settings.dlssSrHardwareAvailable;

  // Fallback path: inner GBuffer holds only the motion attachment (no depth, no extra colors).
  // Color/selection/depth route to outer via the get*Image accessors. Saves ~16 B/pixel.
  // Normal path: full inner GBuffer (color + selection + motion + matching depth).
  const std::vector<VkFormat> colorFormats = m_fallback ?
                                                 std::vector<VkFormat>{kSrMotionFormat} :
                                                 std::vector<VkFormat>{kSrColorFormat, kSrSelectionFormat, kSrMotionFormat};
  NVVK_CHECK(m_innerGBuffer.init({.device       = resources.allocator.getDevice(),
                                  .alloc        = &resources.allocator,
                                  .colorFormats = colorFormats,
                                  .depthFormat  = m_fallback ? VK_FORMAT_UNDEFINED : m_innerDepthFormat,
                                  .debugName    = "DLSS-SR"}));

  // Fallback: m_state stays eNotChecked. The normal async probe (startAsyncInit) will run,
  // discover that SR is unavailable, and set m_state = eUnavailable then. NR availability is
  // also discovered in the same probe. Brief "Loading" state is acceptable since this
  // only shows during the one-time startup probe (~2-5 s) on SR-unavailable hardware.
}

void Dlss::deinitSr(Resources& /*resources*/)
{
  destroyImagesSr();
  m_innerGBuffer.deinit();
  if(m_dlssCreated)
  {
    m_dlss.deinit();
    m_dlssCreated = false;
  }
  // In the SR-fallback+NR case: m_state is eUnavailable but NGX is still alive to serve NR.
  // m_nrAvailable is cleared in deinit() AFTER this call, so it still reflects the active state here.
#if defined(USE_DLSSNR)
  const bool nrKeepsNgxAlive = m_nrAvailable;
#else
  const bool nrKeepsNgxAlive = false;
#endif
  if(m_state != InitStatus::eUnavailable || nrKeepsNgxAlive)
    m_ngx.deinit();
}

void Dlss::refreshOuterRefs(Resources& resources)
{
  if(!m_fallback)
    return;
  m_outerRefs.colorImage     = resources.gBuffers.getColorImage(Resources::eImgRendered);
  m_outerRefs.colorView      = resources.gBuffers.getColorAttachmentView(Resources::eImgRendered);
  m_outerRefs.selectionImage = resources.gBuffers.getColorImage(Resources::eImgSelection);
  m_outerRefs.selectionView  = resources.gBuffers.getColorAttachmentView(Resources::eImgSelection);
  m_outerRefs.depthImage     = resources.gBuffers.getDepthImage();
  m_outerRefs.depthView      = resources.gBuffers.getDepthImageView();
}

// Drop the tracker entry for the currently-allocated inner GBuffer (if any)
void Dlss::destroyImagesSr()
{
  if(m_appMemoryTracker && m_innerTracked)
  {
    const uint32_t count = m_fallback ? 1u : kInnerColorCount;
    m_appMemoryTracker->untrack("DLSS-SR/Inner", m_innerGBuffer, count);
    m_innerTracked = false;
  }
}

void Dlss::createImagesSr(VkCommandBuffer cmd, VkExtent2D size)
{
  destroyImagesSr();

  // Create new color + depth images at `size`, transitioning every image to VK_IMAGE_LAYOUT_GENERAL.
  NVVK_CHECK(m_innerGBuffer.update(cmd, size));
  m_innerGBuffer.cmdClear(cmd);

  const uint32_t count = m_fallback ? 1u : kInnerColorCount;

  if(m_appMemoryTracker)
  {
    m_appMemoryTracker->track("DLSS-SR/Inner", m_innerGBuffer, count);
    m_innerTracked = true;
  }
}

VkExtent2D Dlss::updateSizeSr(VkCommandBuffer cmd, Resources& resources, VkExtent2D size)
{
  // size = display (output) extent. Inner extent = NGX-optimal input given size + quality
  const bool hasGpuState = (m_innerGBuffer.getSize().width != 0) || m_dlssCreated;
  if(hasGpuState && m_graphicsQueue != VK_NULL_HANDLE)
    NVVK_CHECK(vkQueueWaitIdle(m_graphicsQueue));

  const Quality                     quality    = getQuality();
  const NVSDK_NGX_PerfQuality_Value ngxQuality = qualityToNgx(quality);

  // Default 1:1 (DLAA / Off / fallback)
  VkExtent2D innerExtent = size;
  if(isAvailable() && quality != Quality::eOff && quality != Quality::eDLAA)
  {
    DlssFeature::SupportedSizes supported{};
    const NVSDK_NGX_Result      result = m_dlss.querySupportedInputSizes(m_ngx, {size, ngxQuality}, &supported);
    if(NVSDK_NGX_SUCCEED(result))
      innerExtent = supported.optimalSize;
    else
      LOGE("DLSS-SR: querySupportedInputSizes failed: %d -- falling back to 1:1 (DLAA)\n", result);
  }

  // Always (re)allocate the inner GBuffer so the rasterizer's pipeline has a valid motion attachment regardless of NGX/HW state.
  createImagesSr(cmd, innerExtent);
  // Fallback mode routes color/selection/depth back to outer; cache those handles each resize (outer GBuffer images are reallocated on viewport resize so views become stale).
  refreshOuterRefs(resources);

  // DLSS not in use (eOff or HW unavailable)
  if(!isAvailable() || quality == Quality::eOff)
  {
    if(m_dlssCreated)
    {
      m_dlss.deinit();
      m_dlssCreated = false;
    }
    m_needsRecreate = false;
    return innerExtent;
  }

  if(m_dlssCreated)
  {
    m_dlss.deinit();
    m_dlssCreated = false;
  }

  DlssFeature::InitInfo initInfo{
      .quality    = ngxQuality,
      .preset     = m_preset,
      .inputSize  = innerExtent,
      .outputSize = size,
  };
  const NVSDK_NGX_Result initRes = m_dlss.cmdInit(cmd, m_ngx, initInfo);
  if(NVSDK_NGX_FAILED(initRes))
  {
    LOGE("DLSS-SR: m_dlss.cmdInit failed: %d (quality=%d, in=%ux%u, out=%ux%u)\n", initRes,
         static_cast<int>(ngxQuality), innerExtent.width, innerExtent.height, size.width, size.height);
    notifyReset();
    return innerExtent;
  }
  m_dlssCreated = true;

  m_needsRecreate = false;
  notifyReset();
  return innerExtent;
}

void Dlss::setResourcesSr()
{
  if(!isAvailable())
    return;

  // ColorIn / MotionVector / Depth come from the inner GBuffer; ColorOut is wired separately via setOutputImage() (the rasterizer's eImgRendered at outer extent).
  m_dlss.setResource({
      .type      = DlssFeature::ResourceType::eColorIn,
      .image     = m_innerGBuffer.getColorImage(kInnerColorIdx),
      .imageView = m_innerGBuffer.getColorAttachmentView(kInnerColorIdx),
      .format    = kSrColorFormat,
  });

  m_dlss.setResource({
      .type      = DlssFeature::ResourceType::eMotionVector,
      .image     = m_innerGBuffer.getColorImage(kInnerMotionIdx),
      .imageView = m_innerGBuffer.getColorAttachmentView(kInnerMotionIdx),
      .format    = kSrMotionFormat,
  });

  m_dlss.setResource({
      .type      = DlssFeature::ResourceType::eDepth,
      .image     = m_innerGBuffer.getDepthImage(),
      .imageView = m_innerGBuffer.getDepthImageView(),
      .format    = m_innerDepthFormat,
      .range     = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
  });
}

VkFormat Dlss::getSrFormat(SrSlot slot) const
{
  assertKind(Kind::SR);
  switch(slot)
  {
    case SrSlot::eColor:
      return kSrColorFormat;
    case SrSlot::eSelection:
      return kSrSelectionFormat;
    case SrSlot::eMotion:
      return kSrMotionFormat;
    case SrSlot::eDepth:
      return m_innerDepthFormat;
  }
  return VK_FORMAT_UNDEFINED;  // unreachable
}

VkImageView Dlss::getSrImageView(SrSlot slot) const
{
  assertKind(Kind::SR);
  switch(slot)
  {
    case SrSlot::eColor:
      return m_fallback ? m_outerRefs.colorView : m_innerGBuffer.getColorAttachmentView(kInnerColorIdx);
    case SrSlot::eSelection:
      return m_fallback ? m_outerRefs.selectionView : m_innerGBuffer.getColorAttachmentView(kInnerSelectionIdx);
    case SrSlot::eMotion:
      return m_innerGBuffer.getColorAttachmentView(m_fallback ? kFallbackMotionIdx : kInnerMotionIdx);
    case SrSlot::eDepth:
      return m_fallback ? m_outerRefs.depthView : m_innerGBuffer.getDepthImageView();
  }
  return VK_NULL_HANDLE;  // unreachable
}

VkImage Dlss::getSrImage(SrSlot slot) const
{
  assertKind(Kind::SR);
  switch(slot)
  {
    case SrSlot::eColor:
      return m_fallback ? m_outerRefs.colorImage : m_innerGBuffer.getColorImage(kInnerColorIdx);
    case SrSlot::eSelection:
      return m_fallback ? m_outerRefs.selectionImage : m_innerGBuffer.getColorImage(kInnerSelectionIdx);
    case SrSlot::eMotion:
      // Motion is the one attachment that lives in the inner GBuffer in both modes
      return m_innerGBuffer.getColorImage(m_fallback ? kFallbackMotionIdx : kInnerMotionIdx);
    case SrSlot::eDepth:
      return m_fallback ? m_outerRefs.depthImage : m_innerGBuffer.getDepthImage();
  }
  return VK_NULL_HANDLE;  // unreachable; switch is exhaustive
}

bool Dlss::onUiActivationSr(Resources& resources)
{
  consumePendingInitResult();
  startAsyncInit(resources);

  bool changed = false;

  const bool hardwareAvailable = resources.settings.dlssSrHardwareAvailable;
  const bool initPending       = m_state == InitStatus::eNotChecked || m_initInProgress.load(std::memory_order_acquire);
  const bool avail             = hardwareAvailable && m_state != InitStatus::eUnavailable;

  bool        srEnabled    = getQuality() != Quality::eOff;
  const char* reason       = !hardwareAvailable ? "Hardware/extensions not available." : "NGX initialization failed.";
  const char* enableTip    = avail ? "Enable DLSS Super Resolution / DLAA." : reason;
  const char* uiState      = "Ready";
  ImVec4      uiStateColor = nvsamples::denoiserui::readyColor();
  if(!avail)
  {
    uiState      = "Unavailable";
    uiStateColor = nvsamples::denoiserui::unavailableColor();
  }
  else if(initPending)
  {
    uiState      = "Loading";
    uiStateColor = nvsamples::denoiserui::workingColor();
    enableTip    = "DLSS-SR is initializing in the background.";
  }
  else if(srEnabled && state() == State::eActive)
  {
    uiState      = "On";
    uiStateColor = nvsamples::denoiserui::readyColor();
  }

  if(nvsamples::denoiserui::featureRow("dlss_sr", "DLSS Super Resolution", &srEnabled, avail, uiState, uiStateColor,
                                       &m_settingsOpen, enableTip, "Show DLSS-SR settings."))
  {
    m_settings.qualityMode = static_cast<int>(srEnabled ? Quality::eDLAA : Quality::eOff);
    m_needsRecreate        = true;
    notifyReset();
    changed = true;
  }
  return changed;
}

bool Dlss::onUiSettingsSr(Resources& resources)
{
  if(!m_settingsOpen)
    return false;

  bool       changed = false;
  const bool avail   = resources.settings.dlssSrHardwareAvailable && m_state != InitStatus::eUnavailable;

  ImGui::Indent();
  ImGui::PushID("dlss_sr_settings");

  if(!avail)
  {
    ImGui::TextDisabled("DLSS-SR is unavailable on this GPU/driver.");
    ImGui::PopID();
    ImGui::Unindent();
    return false;
  }

  ImGui::TextDisabled("DLSS Super Resolution settings");

  // Quality combo: Off, DLAA (native-res AA), or one of the four upscale presets. Switching
  // mode forces an NGX feature recreate (NGX bakes input resolution at create time).
  static const char* kQualityLabels[] = {
      "Off",                 // Quality::eOff
      "DLAA (native res)",   // Quality::eDLAA
      "Quality (~67%)",      // Quality::eQuality
      "Balanced (~58%)",     // Quality::eBalanced
      "Performance (~50%)",  // Quality::ePerformance
      "Ultra Perf (~33%)",   // Quality::eUltraPerformance
  };
  static_assert(IM_ARRAYSIZE(kQualityLabels) == static_cast<int>(Quality::eCount), "Quality combo labels must match Quality enum count");
  int currentQuality = static_cast<int>(getQuality());

  namespace PE = nvgui::PropertyEditor;
  PE::begin();
  if(PE::Combo("Quality", &currentQuality, kQualityLabels, IM_ARRAYSIZE(kQualityLabels), 0,
               "DLSS Super Resolution mode. Off bypasses DLSS; DLAA renders at native resolution; Quality through Ultra Performance render smaller and upscale."))
  {
    m_settings.qualityMode = currentQuality;
    m_needsRecreate        = true;
    notifyReset();
    changed = true;
  }

  if(getQuality() == Quality::eOff)
  {
    PE::end();
    ImGui::PopID();
    ImGui::Unindent();
    return changed;
  }

  // SR preset combo -- the table itself lives at file scope (kSrPresets[] in the KindConfig
  // section above), reusable from any future CLI flag / preset-cycler hotkey.
  if(drawPresetCombo(Kind::SR, &m_preset, "Preset", "DLSS-SR network preset. Default lets NGX choose the SDK default."))
  {
    m_needsRecreate = true;  // NGX bakes the preset at feature creation; previous history is incompatible.
    notifyReset();
    changed = true;
  }
  PE::end();

  const VkExtent2D innerSz = m_innerGBuffer.getSize();
  const VkExtent2D outerSz = resources.gBuffers.getSize();
  if(innerSz.width == 0 || innerSz.height == 0 || !isAvailable())
  {
    ImGui::TextDisabled("Resolution: %u x %u  (pending NGX init)", outerSz.width, outerSz.height);
  }
  else
  {
    const float ratio = (outerSz.width > 0) ? (100.0f * float(innerSz.width) / float(outerSz.width)) : 0.0f;
    ImGui::Text("Inner (DLSS input):  %u x %u  (%.1f%%)", innerSz.width, innerSz.height, ratio);
  }
  nvgui::tooltip("Inner resolution = what the rasterizer actually renders. Outer = the upscaled DLSS output (display target).");

  ImGui::Spacing();

  if(drawResetHistoryButton())
    changed = true;

  ImGui::PopID();
  ImGui::Unindent();
  return changed;
}


#endif  // USE_DLSS
