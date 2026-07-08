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

#pragma once

// =============================================================================================
// Dlss -- application-side adapter for NVIDIA DLSS (NGX)
// =============================================================================================
//
// One class drives both DLSS feature families:
//
//   * Kind::RR  - DLSS Ray Reconstruction (path-tracer denoiser + upscaler).
//                 Owns an 8-attachment "guide" GBuffer the path tracer writes into.
//
//   * Kind::SR  - DLSS Super Resolution / DLAA (rasterizer AA + upscaling).
//                 Owns a 3-attachment + depth inner GBuffer the rasterizer renders into.
//
// Architecture:
//
//     Application  ->  Dlss (this file)  ->  dlss_wrapper (NgxContext + DlssFeature)  ->  NGX
//
// dlss_wrapper.{hpp,cpp} stays project-independent; this file is the only place that
// understands Resources, GBuffers, ImGui, and the renderer's output image.
//
// Typical frame loop (pseudocode, identical for both Kinds):
//
//     // ---- once at attach ----
//     Dlss dlss(Dlss::Kind::RR);     // or Kind::SR
//     dlss.init(resources);
//     dlss.setBusyWindow(&busy);     // OPTIONAL: show a modal during the 2-5 s NGX init.
//                                    // The Dlss worker thread owns start()/stop(); the renderer
//                                    // doesn't need to manage the modal at all. Skip the call
//                                    // (or pass nullptr) to keep rendering at native during the
//                                    // load instead -- that's what the rasterizer's Dlss does.
//
//     // ---- per frame ----
//     // (1) Drive the state machine. Returns true ONCE on the frame DLSS resources need to be
//     //     (re-)wired (initial NGX-up after async settle, or a UI-driven feature recreate).
//     if (dlss.tick(resources))
//     {
//         dlss.updateSize(cmd, resources, displaySize);
//         dlss.setOutputImage(image, view, format);
//         dlss.setResources();
//     }
//
//     // (2) Per-frame work. isActive() is the strict gate for "DLSS will run this frame";
//     //     it is false during the async-init window AND when the user has DLSS toggled off,
//     //     so the renderer naturally degrades to native rendering in those cases.
//     auto fc = dlss.beginFrame();                 // returns {jitter,frameIndex}; (0,0) when !active
//     // (write fc.jitter into your shader uniforms; render with a jittered viewProj)
//     if (dlss.isActive())
//         dlss.evaluate(cmd, view, projUnjittered);
//
//     // ---- accumulation reset (camera/scene change, OptiX/DLSS toggle) ----
//     dlss.notifyReset();                          // next beginFrame() restarts at index 0
//
//     // ---- viewport (kind-agnostic) ----
//     if (auto guide = dlss.activeGuideImage())
//         tonemap(guide->image, guide->extent);
//
// Wrong-Kind accessors assert in debug builds; release relies on the call-site Kind discipline
// already used by the path tracer / rasterizer (each only ever talks to the side of the API
// matching its own Kind).
//
// See also:
//   * dlss_wrapper.hpp                  - thin NGX/Vulkan wrapper used by this class
//   * shaders/dlss_util.h               - host+shader Halton jitter, motion-vector math
//   * https://github.com/NVIDIA/DLSS    - public DLSS SDK + integration guide
// =============================================================================================

#include "gpu_memory_tracker.hpp"
#include "resources.hpp"

#if defined(USE_DLSS)

#include <array>
#include <atomic>
#include <cassert>
#include <optional>
#include <thread>
#include <vector>

#include <glm/glm.hpp>

#include "nvvk/render_target.hpp"
#include <nvapp/imgui_texture.hpp>
#include "nvutils/parameter_registry.hpp"

#include "dlss_wrapper.hpp"
#include "nvsdk_ngx_helpers_vk.h"

namespace nvgui {
class SettingsHandler;
}

class BusyWindow;


class Dlss
{
public:
  using Kind = DlssFeature::Kind;

  // Public state machine
  enum class State
  {
    eOff,          // User disabled -- or SR Quality::eOff. No DLSS work this frame.
    eLoading,      // Async NGX init in flight, OR brief post-settle window before updateSize()
                   // allocates the inner GBuffer. Renderer should pause + show BusyWindow.
    eActive,       // Ready to consume input this frame.
    eUnsupported,  // Hardware / extensions missing OR async init failed. Sticky for the session.
  };

  // RR-only: how to pick the input resolution NGX recommends for a given output size.
  enum class SizeMode
  {
    eMin,
    eOptimal,
    eMax
  };

  // RR-only: how to populate the DLSS-RR guide buffers for transparent materials.
  enum class TransparencyMode
  {
    eDefault,   // First-hit guides only.
    eImproved,  // Blended guides through transparent layers.
  };

  // SR-only: user-facing DLSS-SR quality ladder.
  enum class Quality
  {
    eOff = 0,           // DLSS disabled; rasterizer renders direct to outputSize.
    eDLAA,              // Native resolution. AI antialiasing only.        ~100% input.
    eQuality,           // NGX MaxQuality.                                 ~67%  input.
    eBalanced,          // NGX Balanced.                                   ~58%  input.
    ePerformance,       // NGX MaxPerf.                                    ~50%  input.
    eUltraPerformance,  // NGX UltraPerformance.                           ~33%  input.
    eCount              // Sentinel, must remain last.
  };

  // Returned by beginFrame(): everything the renderer needs to drive a DLSS-aware frame.
  struct FrameContext
  {
    glm::vec2 jitter{0.0f};   // pixel-space sub-pixel offset (Halton). {0,0} when DLSS isn't running
    uint32_t  frameIndex{0};  // monotonically increasing counter, restarted by notifyReset().
  };

  // Returned by activeGuideImage(): a single bundle the application feeds to the tonemapper.
  struct GuideImage
  {
    VkDescriptorImageInfo image;   // descriptor for the guide image
    VkExtent2D            extent;  // inner GBuffer extent (DLSS render resolution, may be smaller than the display)
    const char*           name;    // human-readable label (matches the thumbnail caption)
  };


  explicit Dlss(Kind kind);
  ~Dlss() { assert(m_linearSampler == VK_NULL_HANDLE && "Dlss::deinit() was not called before destruction"); }

  // ---- Lifecycle -------------------------------------------------------------------------------

  // Setup: capture allocator/sampler/queue handles and allocate the inner GBuffer. NGX itself
  // is initialized lazily on the first tick() that observes state() == eLoading -- the worker
  // runs on a background thread while the renderer keeps drawing at native (or shows
  // the BusyWindow modal if one was wired via setBusyWindow()).
  void init(Resources& resources);
  void deinit(Resources& resources);

  // Per-frame state machine driver; returns true once when DLSS resources need to be (re)initialized due to async init or feature recreate, otherwise false.
  // Idempotent, safe to call every frame.
  bool tick(Resources& resources);

  // Resize: query NGX for an optimal input size, recreate the feature, re-allocate the inner
  // GBuffer. Resources& is consumed only by the SR fallback path (refreshOuterRefs()); RR ignores
  // it but keeps the uniform signature.
  VkExtent2D updateSize(VkCommandBuffer cmd, Resources& resources, VkExtent2D size);

  // Wire the inner-GBuffer-backed inputs (color, motion, depth, RR guides...) into the NGX
  // feature. Called whenever the GBuffer is (re)allocated.
  void setResources();

  // Bind the final display target (outer eImgRendered) as the DLSS output.
  void setOutputImage(VkImage image, VkImageView view, VkFormat format);

  // Advances per-frame state and returns {jitter, frameIndex} for the upcoming frame, safe to call when DLSS is off.
  FrameContext beginFrame();

  // Tells Dlss the renderer just reset its accumulation.
  void notifyReset();

  // Run the NGX feature for this frame. Pulls the jitter snapshotted by the matching
  // beginFrame() and the reset bit armed by notifyReset() from internal state. view + proj
  // must be UNJITTERED (NGX applies the de-jitter itself).
  // PRECONDITION: isActive() == true. The caller is responsible for gating; in debug builds
  // this asserts on misuse, release builds will crash inside NGX (null feature handle).
  void evaluate(VkCommandBuffer cmd, const glm::mat4& view, const glm::mat4& proj);

  // UI panel; returns true when something changed that requires a renderer-side response.
  bool onUi(Resources& resources);

  // Register parameters for the DLSS UI panel or CLI.
  void registerParameters(nvutils::ParameterRegistry* paramReg);
  void setSettingsHandler(nvgui::SettingsHandler* settingsHandler);

  // Optional: wire a BusyWindow modal so the user gets visible feedback during the async init.
  void setBusyWindow(BusyWindow* busyWindow) { m_busyWindow = busyWindow; }

  // ---- Common queries --------------------------------------------------------------------------

  Kind  getKind() const { return m_kind; }                      // RR or SR
  State state() const;                                          // eOff, eLoading, eActive, eUnsupported
  bool  isActive() const { return state() == State::eActive; }  // true when DLSS will run this frame

  // True when the NGX feature must be torn down + recreated (UI preset / size-mode / quality change).
  bool needsRecreate() const;

  // Inner GBuffer extent (== NGX input resolution).
  VkExtent2D getRenderSize() const { return m_innerGBuffer.getSize(); }

  // Guide-buffer visualization: thumbnails shown in onUi(); selection exposes activeGuideImage() for tonemapping integration.
  // To add a new guide buffer, insert it into the kind-specific table in initRr()/initSr(); no enum or call-site changes needed.
  std::optional<GuideImage> activeGuideImage() const;
  void                      clearGuideSelection() { m_selectedGuide = -1; }

  // ---- RR-only accessors (assert m_kind == Kind::RR in debug) --------------------------------

  bool             useDlssTransparency() const;
  TransparencyMode getTransparencyMode() const;

  // Typed access into the RR guide GBuffer; indices match shaderio::OutputImage, hiding m_innerGBuffer implementation details.
  // Allows future GBuffer layout changes without affecting caller code.
  VkDescriptorImageInfo getRrAttachment(shaderio::OutputImage which) const;
  VkImage               getRrImage(shaderio::OutputImage which) const;

  // ---- SR-only API (assert m_kind == Kind::SR in debug) --------------------------------------
  // In normal (HW available) SR mode, color/selection/motion/depth all live in m_innerGBuffer at the DLSS input extent.
  // In fallback mode (HW unavailable), only motion lives in the inner GBuffer; color/selection/depth route to the outer GBuffer (the rasterizer renders direct-to-outer and the post-pass blits are skipped).
  enum class SrSlot
  {
    eColor,
    eSelection,
    eMotion,
    eDepth,
  };

  Quality getQuality() const;
  bool    isFallback() const;  // true when the inner GBuffer holds only motion (HW unavailable)

  // Typed SR attachment access.
  VkImage     getSrImage(SrSlot slot) const;
  VkImageView getSrImageView(SrSlot slot) const;
  VkFormat    getSrFormat(SrSlot slot) const;

  // Layout transitions + 1:1 nearest blit from one of our inner-GBuffer images to an
  // outer-extent target. Used by the rasterizer post-pass to land inner.selection -> outer
  // and inner.color -> outer (when DLSS evaluate did not run). Source/dest both transition
  // GENERAL -> TRANSFER_SRC/DST -> GENERAL.
  void blitInnerToOuter(VkCommandBuffer cmd, VkImage innerSrc, VkImage outerDst, VkExtent2D outerExtent) const;

private:
  // Internal NGX-init lifecycle. Three values that track the asynchronous worker.
  enum class InitStatus
  {
    eNotChecked,   // NGX initialization hasn't been attempted yet (lazy).
    eUnavailable,  // Hardware / extensions missing or NGX initialization failed (sticky).
    eAvailable,    // NGX feature handle is up.
  };

  // Persisted settings (per kind).
  struct Settings
  {
    bool     enableRr    = false;                            // RR only
    SizeMode sizeMode    = SizeMode::eOptimal;               // RR only
    int      qualityMode = static_cast<int>(Quality::eOff);  // SR only (int for SettingsHandler)
  };

  // ---- Internal predicates (callers consume state() / isActive() instead) -------------------
  bool isAvailable() const { return m_state == InitStatus::eAvailable; }
  bool isUserEnabled() const;  // RR: m_settings.enableRr; SR: getQuality() != eOff
  // True between the tick() that started the worker and the tick() that consumed its result; used by onUi() to show the "(Loading...)" inline hint.
  bool isInitializing() const { return m_initInProgress.load(std::memory_order_acquire); }

  // ---- Common helpers -------------------------------------------------------------------------
  // NGX init body for async worker thread; returns status, published to m_pendingInitResult, applied to m_state by main thread.
  InitStatus runNgxInit(Resources& resources);
  void       teardownNgx();
  bool       drawResetHistoryButton();
  // RR-friendly assertion shim -- in release builds (NDEBUG) the inline accessors above just
  // execute the kind-appropriate path without firing.
  void assertKind(Kind expected) const;

  // ---- Kind dispatch --------------------------------------------------------------------------
  void       initRr(Resources& resources);
  void       initSr(Resources& resources);
  void       deinitRr(Resources& resources);
  void       deinitSr(Resources& resources);
  VkExtent2D updateSizeRr(VkCommandBuffer cmd, VkExtent2D size);
  VkExtent2D updateSizeSr(VkCommandBuffer cmd, Resources& resources, VkExtent2D size);
  void       setResourcesRr();
  void       setResourcesSr();
  bool       onUiRr(Resources& resources);
  bool       onUiSr(Resources& resources);

  // Build the per-Kind guide-image table after the inner GBuffer exists.
  void buildGuideEntries();
  // Render the thumbnail grid for the current m_guideEntries; toggles m_selectedGuide on click.
  // Returns true when the selection changed this frame.
  bool drawGuideThumbnails();

  // SR-only helpers carried over verbatim from the old DlssSuperResolution code.
  void destroyImagesSr();
  void createImagesSr(VkCommandBuffer cmd, VkExtent2D size);
  void refreshOuterRefs(Resources& resources);

  // RR-only "release on disable" helpers. For SR, the buffers are always needed.
  void releaseRrInnerGBuffer();
  void reacquireRrInnerGBuffer(Resources& resources);
  void deinitGuideTextures();
  void syncGuideTextures(uint32_t colorCount);

  // SR-only NGX quality-enum mapping (eOff/eDLAA never reach NGX; kept for completeness).
  static NVSDK_NGX_PerfQuality_Value qualityToNgx(Quality q);

  // ---- Common state ---------------------------------------------------------------------------
  Kind       m_kind;  // RR or SR
  InitStatus m_state = InitStatus::eNotChecked;
  bool m_needsRecreate = false;  // true when the NGX feature must be torn down + recreated (UI preset / size-mode / quality change)
  uint32_t m_preset = 0;  // NGX preset hint (RR or SR table; 0 = NGX default)
  Settings m_settings{};  // persisted settings (per kind)


  uint64_t m_globalFrame = 0;  // monotonic counter incremented by every evaluate(); notifyReset() arms a "reset on the next actual evaluate" signal by setting m_forceResetUntilFrame := m_globalFrame + 1.
  uint64_t m_forceResetUntilFrame = 0;  // reset signal is re-armed on every notifyReset() call; evaluate() sees reset == true exactly once and the counter slides past. This is robust to evaluate() being skipped between two notifyReset() calls (deferred presents, debugger pause, async-compile-blocked frames): the signal can never be lost, only re-armed. A simple bool would silently drop the signal if it was set then cleared by something other than evaluate(). uint64_t at 60 fps wraps around in ~9.7 billion years.

  NgxContext  m_ngx{};  // NGX feature handle
  DlssFeature m_dlss;   // NGX feature handle

  // ---- Async lazy-init plumbing --------------------------------------------------------------
  std::thread       m_initThread;             // worker thread
  std::atomic<bool> m_initInProgress{false};  // true when the worker is running
  std::atomic<int> m_pendingInitResult{0};  // 0 == no pending result, 1 == eAvailable just landed, 2 == eUnavailable just landed.
  BusyWindow* m_busyWindow = nullptr;  // optional UI modal owned by the renderer (PathTracer wires it; Rasterizer doesn't). If set: tick() starts it the moment we kick off the worker. the worker thread stops it on completion -- this is what unblocks the render-thread loop, since GltfRenderer::onRender early-returns while the modal is up.

  VkSampler                   m_linearSampler    = VK_NULL_HANDLE;
  VkQueue                     m_graphicsQueue    = VK_NULL_HANDLE;
  nvvkgltf::GpuMemoryTracker* m_appMemoryTracker = nullptr;

  nvvk::RenderTarget                              m_innerGBuffer{};
  static constexpr uint32_t                       kMaxGuideTextures = 8;
  std::array<nvapp::ImTexture, kMaxGuideTextures> m_guideTextures{};
  uint32_t                                        m_guideTextureCount = 0;
  bool                                            m_innerTracked      = false;

  // ---- Per-frame state owned by beginFrame() / consumed by evaluate() ------------------------
  uint32_t  m_frameIndex    = 0;             // starts at 0 and is incremented in beginFrame()
  glm::vec2 m_currentJitter = {0.0f, 0.0f};  // pixel-space sub-pixel offset (Halton). {0,0} when DLSS isn't running

  // ---- Guide-buffer thumbnail table (built per-Kind in init()) -------------------------------
  struct GuideEntry
  {
    const char* label;         // human-readable label (matches the thumbnail caption)
    uint32_t    gbufColorIdx;  // inner GBuffer color attachment index
  };
  std::vector<GuideEntry> m_guideEntries;        // guide-image table
  int                     m_selectedGuide = -1;  // -1 == "show main rendered output"

  // ---- RR-only state --------------------------------------------------------------------------
  TransparencyMode m_transparencyMode = TransparencyMode::eDefault;  // transparency mode

  // ---- SR-only state --------------------------------------------------------------------------
  bool     m_dlssCreated      = false;                // m_dlss has a valid feature handle (cmdInit succeeded)
  bool     m_fallback         = false;                // SR HW/extensions unavailable; inner GBuffer carries motion only
  VkFormat m_innerDepthFormat = VK_FORMAT_UNDEFINED;  // captured from outer at init time (SR only)

  // Inner GBuffer attachment indices when SR HW is available. Must stay in lockstep with the
  // colorFormats list passed to m_innerGBuffer.init() in initSr().
  static constexpr uint32_t kInnerColorIdx     = 0;
  static constexpr uint32_t kInnerSelectionIdx = 1;
  static constexpr uint32_t kInnerMotionIdx    = 2;
  static constexpr uint32_t kInnerColorCount   = 3;
  // In fallback mode the inner GBuffer holds only motion at index 0.
  static constexpr uint32_t kFallbackMotionIdx = 0;

  // SR inner-GBuffer formats: kSrColorFormat must match Resources::eImgRendered, and kSrSelectionFormat must match Resources::eImgSelection.
  // This ensures secondary command buffer format compatibility and avoids conversion during DLSS post-blit.
  static constexpr VkFormat kSrColorFormat     = VK_FORMAT_R32G32B32A32_SFLOAT;
  static constexpr VkFormat kSrSelectionFormat = VK_FORMAT_R32_SFLOAT;
  static constexpr VkFormat kSrMotionFormat    = VK_FORMAT_R16G16_SFLOAT;

  // Snapshot of the outer GBuffer's color/selection/depth, refreshed on every updateSize() in
  // fallback mode. The rasterizer routes through these so it can render directly into the outer
  // surfaces, skipping the post-pass blits entirely.
  struct OuterRefs
  {
    VkImage     colorImage     = VK_NULL_HANDLE;
    VkImageView colorView      = VK_NULL_HANDLE;
    VkImage     selectionImage = VK_NULL_HANDLE;
    VkImageView selectionView  = VK_NULL_HANDLE;
    VkImage     depthImage     = VK_NULL_HANDLE;
    VkImageView depthView      = VK_NULL_HANDLE;
  };
  OuterRefs m_outerRefs{};
};

#endif  // USE_DLSS
