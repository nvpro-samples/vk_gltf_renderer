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

#include "dlss.hpp"

#if defined(USE_DLSS)

#include <cassert>
#include <span>

#include <imgui/imgui.h>

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

// Halton sequence utility functions
#include "shaders/dlss_util.h"

// Optional BusyWindow integration
#include "ui_busy_window.hpp"


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

  if(m_kind == Kind::RR)
    initRr(resources);
  else
    initSr(resources);

  // Skip the guide-image table when the inner GBuffer wasn't allocated by initRr() -- this
  // happens when DLSS-RR hardware isn't available on the current GPU.
  const bool rrSkipped = (m_kind == Kind::RR && m_state == InitStatus::eUnavailable);
  if(!rrSkipped)
    buildGuideEntries();
}

void Dlss::deinit(Resources& resources)
{
  // If an async NGX init is in flight, block until the worker is done before tearing down NGX
  // state -- otherwise the worker could call into a half-destroyed NgxContext.
  if(m_initThread.joinable())
    m_initThread.join();
  m_initInProgress.store(false, std::memory_order_release);
  m_pendingInitResult.store(0, std::memory_order_release);

  if(m_kind == Kind::RR)
    deinitRr(resources);
  else
    deinitSr(resources);

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
Dlss::InitStatus Dlss::runNgxInit(Resources& resources)
{
  const KindConfig& t = kindConfig(m_kind);

  NgxContext::InitInfo ngxInitInfo{
      .instance       = resources.instance,
      .physicalDevice = resources.allocator.getPhysicalDevice(),
      .device         = resources.allocator.getDevice(),
  };

  if(m_ngx.init(ngxInitInfo) != NVSDK_NGX_Result_Success)
  {
    LOGW("%s: NGX initialization failed - DLSS disabled\n", t.logTag);
    return InitStatus::eUnavailable;
  }

  if(m_ngx.isFeatureAvailable(t.ngxFeature) != NVSDK_NGX_Result_Success)
  {
    LOGW("%s: feature not available on this driver/hardware - DLSS disabled\n", t.logTag);
    m_ngx.deinit();
    return InitStatus::eUnavailable;
  }

  return InitStatus::eAvailable;
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

  // (3) Async NGX init in flight, or worker hasn't been kicked off yet. tick() drives forward.
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

// Advance the asynchronous NGX initialization state machine and notify the renderer if DLSS resources
// should be (re-)configured this frame. The logic below proceeds through the following phases:
//
//   eOff/eUnsupported : DLSS is disabled or unsupported, nothing to do, return false.
//   pending result    : Consume the async worker result on the main thread, update m_state, and return true on success.
//   already settled   : Initialization already completed, nothing changed, return false.
//   worker running    : Initialization still in progress, return false and renderer operates at native resolution.
//   first call        : Launch async worker for NGX init, return false.
//
// In addition, we check needsRecreate() (e.g. triggered by UI changes such as size, mode, or preset)
// so tick() reflects both "first-time successful init" as well as feature recreation events.
bool Dlss::tick(Resources& resources)
{
  // (a) Off / Unsupported: nothing for the worker to do.
  const State s = state();
  if(s == State::eOff || s == State::eUnsupported)
    return false;

  bool justInitedNgx = false;

  // (b) Worker has finished since the last call -- consume the pending result on the main thread.
  const int pending = m_pendingInitResult.exchange(0, std::memory_order_acq_rel);
  if(pending != 0)
  {
    if(m_initThread.joinable())
      m_initThread.join();
    m_state = (pending == 1) ? InitStatus::eAvailable : InitStatus::eUnavailable;
    if(m_state == InitStatus::eAvailable)
    {
      LOGI("%s: Successfully initialized and ready\n", kindConfig(m_kind).logTag);
      justInitedNgx = true;
    }
    else
    {
      LOGW("%s: NGX initialization failed\n", kindConfig(m_kind).logTag);
    }
  }
  // (c) NGX is already up (this tick() didn't observe a fresh settle) and no other work to do.
  else if(m_state != InitStatus::eNotChecked)
  {
    // fall through to needsRecreate() check below
  }
  // (d) Worker still running -- nothing to do here, just keep waiting.
  else if(m_initInProgress.load(std::memory_order_acquire))
  {
    // fall through to needsRecreate() check below (will be false anyway when state is loading)
  }
  // (e) First call when state is eNotChecked -- kick off the worker.
  else
  {
    m_initInProgress.store(true, std::memory_order_release);
    // Show the modal NOW (main-thread call) if the renderer wired one.
    if(m_busyWindow)
      m_busyWindow->start("Loading DLSS network (one-time, 2-5 s)...");
    m_initThread = std::thread([this, &resources]() {
      SCOPED_TIMER("DLSS NGX Initialization (async, may take 2-5 seconds)");
      const InitStatus result = runNgxInit(resources);
      // Publish the result first, then clear the in-progress flag so an observer that sees !m_initInProgress also sees a non-zero pending result.
      m_pendingInitResult.store(result == InitStatus::eAvailable ? 1 : 2, std::memory_order_release);
      m_initInProgress.store(false, std::memory_order_release);
      // Stop the BusyWindow LAST -- this is what unblocks the render thread.
      if(m_busyWindow)
        m_busyWindow->stop();
    });
  }

  // Also signal UI-driven recreates (size mode / preset / quality / re-enable).
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
  return (m_kind == Kind::RR) ? updateSizeRr(cmd, size) : updateSizeSr(cmd, resources, size);
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
  return (m_kind == Kind::RR) ? onUiRr(resources) : onUiSr(resources);
}


// ============================================================================================
// Settings / parameter registration (kind-aware)
// ============================================================================================

void Dlss::registerParameters(nvutils::ParameterRegistry* paramReg)
{
  if(m_kind == Kind::RR)
  {
    paramReg->add({"dlssEnable", "DLSS Denoiser: Enable DLSS denoiser"}, &m_settings.enableRr);
  }
  else
  {
    paramReg->add({"dlssQuality", "DLSS Super Resolution mode: 0=Off, 1=DLAA, 2=Quality, 3=Balanced, 4=Performance, 5=UltraPerformance"},
                  &m_settings.qualityMode);
  }
}

void Dlss::setSettingsHandler(nvgui::SettingsHandler* settingsHandler)
{
  if(m_kind == Kind::RR)
    settingsHandler->setSetting("dlssEnable", &m_settings.enableRr);
  else
    settingsHandler->setSetting("dlssQuality", &m_settings.qualityMode);
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

  // SR: only attachments that are color-format and live in the inner GBuffer make sense as
  // thumbnails (the depth attachment is depth-format and not directly samplable through the
  // GBuffer's ImGui descriptor set). We keep this minimal; depth is excluded by design.
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
  if(m_innerGBuffer.getSize().width == 0)
    return std::nullopt;  // GBuffer not yet allocated (e.g. before first onResize).

  const GuideEntry& entry = m_guideEntries[m_selectedGuide];
  return GuideImage{
      .image  = m_innerGBuffer.getDescriptorImageInfo(entry.gbufColorIdx),
      .extent = m_innerGBuffer.getSize(),
      .name   = entry.label,
  };
}

bool Dlss::drawGuideThumbnails()
{
  if(m_guideEntries.empty())
    return false;

  bool changed = false;
  if(!ImGui::CollapsingHeader("Guide Images"))
    return false;

  ImGui::TextWrapped(
      "Click on a thumbnail to view it in the viewport. Click again to toggle back to the "
      "rendered image.");
  ImGui::Spacing();

  const ImVec2 thumbnailSize = {100 * m_innerGBuffer.getAspectRatio(), 100};

  // 2-column grid; matches the previous DlssDenoiser look. Works for any entry count.
  if(ImGui::BeginTable("dlss_guide_thumbnails", 2))
  {
    for(int i = 0; i < static_cast<int>(m_guideEntries.size()); ++i)
    {
      if((i % 2) == 0)
        ImGui::TableNextRow();
      ImGui::TableNextColumn();

      const GuideEntry& entry    = m_guideEntries[i];
      const bool        isActive = (m_selectedGuide == i);

      if(isActive)
      {
        ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.0f, 1.0f, 0.0f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 3.0f);
      }

      ImGui::Text("%s%s", entry.label, isActive ? " (Active)" : "");
      const VkDescriptorSet thumbDs = m_innerGBuffer.getDescriptorSet(entry.gbufColorIdx);
      if(ImGui::ImageButton(entry.label, ImTextureID(thumbDs), thumbnailSize))
      {
        // Toggle: re-clicking the active thumbnail goes back to the rendered image.
        m_selectedGuide = isActive ? -1 : i;
        changed         = true;
      }

      if(isActive)
      {
        ImGui::PopStyleVar();
        ImGui::PopStyleColor();
      }
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
bool drawPresetCombo(Dlss::Kind kind, uint32_t* currentPreset, const char* tooltip)
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
  if(PE::Combo("DLSS Preset", &currentIdx, labels, labelCount, 0, tooltip))
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
  if(ImGui::Button("Reset History"))
  {
    notifyReset();
    clicked = true;
  }
  nvgui::tooltip("Forces InReset=true on the next DLSS evaluate, discarding all accumulated temporal history.");
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

  // GBufferInitInfo wants a std::vector; copy from the file-scope constexpr span so the formats
  // table stays the single source of truth.
  const auto& rrFormats = kindConfig(Kind::RR).innerColorFormats;
  m_innerGBuffer.init({.allocator      = &resources.allocator,
                       .colorFormats   = std::vector<VkFormat>(rrFormats.begin(), rrFormats.end()),
                       .imageSampler   = m_linearSampler,
                       .descriptorPool = resources.descriptorPool});
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
  m_innerGBuffer.update(cmd, renderingSize);
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
    m_dlss.setResource({resource, m_innerGBuffer.getColorImage(gbufIndex), m_innerGBuffer.getColorImageView(gbufIndex),
                        m_innerGBuffer.getColorFormat(gbufIndex)});
  };

  bind(DlssFeature::ResourceType::eColorIn, shaderio::OutputImage::eResultImage);
  bind(DlssFeature::ResourceType::eDiffuseAlbedo, shaderio::OutputImage::eDlssAlbedo);
  bind(DlssFeature::ResourceType::eSpecularAlbedo, shaderio::OutputImage::eDlssSpecAlbedo);
  bind(DlssFeature::ResourceType::eNormalRoughness, shaderio::OutputImage::eDlssNormalRoughness);
  bind(DlssFeature::ResourceType::eMotionVector, shaderio::OutputImage::eDlssMotion);
  bind(DlssFeature::ResourceType::eDepth, shaderio::OutputImage::eDlssDepth);
  bind(DlssFeature::ResourceType::eSpecularHitDistance, shaderio::OutputImage::eDlssSpecularHitDist);
}

bool Dlss::onUiRr(Resources& resources)
{
  bool changed = false;

  if(!resources.settings.dlssRrHardwareAvailable || m_state == InitStatus::eUnavailable)
  {
    ImGui::BeginDisabled();
    bool dummyEnable = false;
    ImGui::Checkbox("DLSS-RR", &dummyEnable);
    ImGui::EndDisabled();
    ImGui::SameLine();
    const char* reason = !resources.settings.dlssRrHardwareAvailable ? "(Hardware/extensions not available)" :
                                                                       "(NGX initialization failed)";
    ImGui::TextDisabled("%s", reason);
    return changed;
  }

  // Snapshot previous checkbox state to handle DLSS-RR on/off transitions.
  const bool wasEnabledRr = m_settings.enableRr;
  if(ImGui::Checkbox("DLSS-RR", &m_settings.enableRr))
  {
    notifyReset();
    if(wasEnabledRr && !m_settings.enableRr)
    {
      releaseRrInnerGBuffer();
    }
    else if(!wasEnabledRr && m_settings.enableRr)
    {
      // off->on: re-init the GBuffer.
      reacquireRrInnerGBuffer(resources);
    }
    changed = true;
  }
  nvgui::tooltip("DLSS-RR (Ray Reconstruction) is an NVIDIA RTX technology that uses an AI neural network to replace traditional denoisers.");

  // Telemetry for the async lazy NGX init
  if(isInitializing())
  {
    ImGui::SameLine();
    ImGui::TextDisabled("(Loading NGX network...)");
  }

  if(!m_settings.enableRr || !isAvailable())
    return changed;

  const char* sizeModes[]     = {"Min", "Optimal", "Max"};
  int         currentSizeMode = static_cast<int>(m_settings.sizeMode);

  namespace PE = nvgui::PropertyEditor;
  PE::begin();
  const char* transparencyModes[] = {"Default (first hit)", "Improved (blended guides)"};
  int         currentTransMode    = static_cast<int>(m_transparencyMode);
  if(PE::Combo("Transparency Handling", &currentTransMode, transparencyModes, IM_ARRAYSIZE(transparencyModes), 0,
               "Controls how DLSS guide buffers are generated for transparent materials."))
  {
    m_transparencyMode = static_cast<TransparencyMode>(currentTransMode);
    changed            = true;
  }
  if(PE::Combo("DLSS Size Mode", &currentSizeMode, sizeModes, IM_ARRAYSIZE(sizeModes)))
  {
    m_settings.sizeMode = static_cast<SizeMode>(currentSizeMode);
    m_needsRecreate     = true;
    notifyReset();  // restart Halton sequence so the next-frame jitter matches the recreated feature
    changed = true;
  }
  // RR preset combo
  if(drawPresetCombo(Kind::RR, &m_preset,
                     "Selects which NGX DLSS-RR network preset to load. 'Default' lets NGX pick "
                     "the current SDK default (today: D). Preset E is the latest transformer "
                     "model and is required when a Depth-of-Field guide buffer is used. Other "
                     "presets listed in the SDK header are documented as 'do not use' and crash "
                     "on this driver build."))
  {
    m_needsRecreate = true;  // NGX bakes the preset at feature creation; previous history is incompatible.
    notifyReset();
    changed = true;
  }
  PE::end();

  // Manual history reset for cuts/teleports/visible-ghosting recovery.
  if(drawResetHistoryButton())
    changed = true;

  ImGui::Text("Current Resolution: %d x %d", m_innerGBuffer.getSize().width, m_innerGBuffer.getSize().height);

  // Guide-image thumbnails
  if(drawGuideThumbnails())
    changed = true;

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

  // Create a new GBuffer
  const auto& rrFormats = kindConfig(Kind::RR).innerColorFormats;
  m_innerGBuffer.init({.allocator      = &resources.allocator,
                       .colorFormats   = std::vector<VkFormat>(rrFormats.begin(), rrFormats.end()),
                       .imageSampler   = m_linearSampler,
                       .descriptorPool = resources.descriptorPool});

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
  return m_innerGBuffer.getDescriptorImageInfo(static_cast<uint32_t>(which));
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
  m_innerGBuffer.init({.allocator      = &resources.allocator,
                       .colorFormats   = colorFormats,
                       .depthFormat    = m_fallback ? VK_FORMAT_UNDEFINED : m_innerDepthFormat,
                       .imageSampler   = m_linearSampler,
                       .descriptorPool = resources.descriptorPool});

  if(m_fallback)
  {
    m_state = InitStatus::eUnavailable;
  }
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
  if(m_state != InitStatus::eUnavailable)
    m_ngx.deinit();
}

void Dlss::refreshOuterRefs(Resources& resources)
{
  if(!m_fallback)
    return;
  m_outerRefs.colorImage     = resources.gBuffers.getColorImage(Resources::eImgRendered);
  m_outerRefs.colorView      = resources.gBuffers.getColorImageView(Resources::eImgRendered);
  m_outerRefs.selectionImage = resources.gBuffers.getColorImage(Resources::eImgSelection);
  m_outerRefs.selectionView  = resources.gBuffers.getColorImageView(Resources::eImgSelection);
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
  m_innerGBuffer.update(cmd, size);

  if(m_appMemoryTracker)
  {
    const uint32_t count = m_fallback ? 1u : kInnerColorCount;
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
      .imageView = m_innerGBuffer.getColorImageView(kInnerColorIdx),
      .format    = kSrColorFormat,
  });

  m_dlss.setResource({
      .type      = DlssFeature::ResourceType::eMotionVector,
      .image     = m_innerGBuffer.getColorImage(kInnerMotionIdx),
      .imageView = m_innerGBuffer.getColorImageView(kInnerMotionIdx),
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
      return m_fallback ? m_outerRefs.colorView : m_innerGBuffer.getColorImageView(kInnerColorIdx);
    case SrSlot::eSelection:
      return m_fallback ? m_outerRefs.selectionView : m_innerGBuffer.getColorImageView(kInnerSelectionIdx);
    case SrSlot::eMotion:
      return m_innerGBuffer.getColorImageView(m_fallback ? kFallbackMotionIdx : kInnerMotionIdx);
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

bool Dlss::onUiSr(Resources& resources)
{
  bool changed = false;

  if(!resources.settings.dlssSrHardwareAvailable || m_state == InitStatus::eUnavailable)
  {
    ImGui::BeginDisabled();
    bool dummy = false;
    ImGui::Checkbox("DLSS", &dummy);
    ImGui::EndDisabled();
    ImGui::SameLine();
    const char* reason = !resources.settings.dlssSrHardwareAvailable ? "(Hardware/extensions not available)" :
                                                                       "(NGX initialization failed)";
    ImGui::TextDisabled("%s", reason);
    return changed;
  }

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
  if(ImGui::Combo("Quality", &currentQuality, kQualityLabels, IM_ARRAYSIZE(kQualityLabels)))
  {
    m_settings.qualityMode = currentQuality;
    m_needsRecreate        = true;
    notifyReset();
    changed = true;
  }
  // Telemetry for the async lazy NGX init -- renderer keeps drawing without DLSS during init.
  if(isInitializing())
  {
    ImGui::SameLine();
    ImGui::TextDisabled("(Loading NGX network...)");
  }
  nvgui::tooltip(
      "DLSS Super Resolution mode:\n"
      "  Off          - bypass DLSS, raster output goes straight to the display target.\n"
      "  DLAA         - render at native resolution, AI temporal antialiasing only.\n"
      "  Quality..Ultra - render at a smaller inner resolution, AI upscale to display res.");

  if(getQuality() == Quality::eOff)
    return changed;

  // SR preset combo -- the table itself lives at file scope (kSrPresets[] in the KindConfig
  // section above), reusable from any future CLI flag / preset-cycler hotkey.
  namespace PE = nvgui::PropertyEditor;
  PE::begin();
  if(drawPresetCombo(Kind::SR, &m_preset, "Selects which NGX DLSS-SR network preset to load.\n"))
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

  // Guide-image thumbnails (Color / Motion in normal mode). Click to route the viewport through
  // activeGuideImage(); click again to clear.
  if(drawGuideThumbnails())
    changed = true;

  return changed;
}

#endif  // USE_DLSS
