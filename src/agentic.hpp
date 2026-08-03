/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// agentic::Controller -- renderer-side runtime for the optional Agentic bridge.
//
// Owns:
//   - the bridge root path + Comfy embedded-Python detection,
//   - the two filesystem job queues (HDRI + Beautify),
//   - the heartbeat indicator state,
//   - the Vulkan image that holds the beautified output for in-viewport overlay.
//
// Does NOT own:
//   - the renderer's display-buffer selection (it is a member of Resources),
//   - HDR file loading / firefly clamp / path-tracer state (delegated back to
//     the renderer via the applyHdri callback in Dependencies).
//
// Lifecycle:
//   GltfRenderer holds one Controller by value, calls init(Dependencies) once
//   from onAttach, tick() once per UI frame, and deinit() from onDetach.
//
// UI:
//   renderAgenticWindow() is a free function in this namespace and lives in
//   ui_agentic.cpp. It binds ImGui controls directly to the Controller's public
//   fields and calls its action methods (initializeBridge(), queueHdriJob(), etc.).

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>
#include <nvvk/resource_allocator.hpp>

#include "agentic_bridge.hpp"

namespace nvapp {
class Application;
}
struct Resources;

namespace agentic {

// ImGui prompt editors use a fixed char buffer (no CallbackResize).
inline constexpr std::size_t kPromptBufferSize = 2048;

// A named, ready-made prompt shown in the Agentic window's Presets dropdown.
// Loaded from prompts.json (see loadPromptPresets()).
struct PromptPreset
{
  std::string name;
  std::string prompt;
};

// In-flight job book-keeping (one per task kind).
struct PendingJob
{
  bool                    active{false};
  agentic::GenerationTask task{agentic::GenerationTask::eTextToImage};
  std::string             id;
  std::filesystem::path   requestPath;
  std::filesystem::path   responsePath;
  std::filesystem::path   inputPath;
  std::filesystem::path   preferredOutputPath;
};

class Controller
{
public:
  // Renderer subsystems the controller calls into. Wired up once via init().
  // Kept narrow: anything the controller can do without poking at private
  // renderer state belongs here as a function, not as a raw pointer to the
  // owning class.
  struct Dependencies
  {
    nvapp::Application* app{nullptr};
    Resources*          resources{nullptr};
    VkDevice            device{VK_NULL_HANDLE};
    VkCommandPool       transientCmdPool{VK_NULL_HANDLE};

    // Atomic "load + apply this HDR file as the scene's environment" action.
    // The renderer implements this by calling createHDR(), switching
    // envSystem to eHdr, and updating the path-tracer's firefly clamp.
    std::function<void(const std::filesystem::path&)> applyHdri;
    // Tells the renderer to restart accumulation (because the visible image
    // is about to change).
    std::function<void()> resetFrame;
    // Renderer tonemap pass; skipBeautifiedOverlay=true uses the live render instead of the
    // beautified viewport overlay (see saveBeautifyInputImage).
    std::function<void(VkCommandBuffer, bool skipBeautifiedOverlay)> runTonemapPass;
  };

  Controller();
  ~Controller();
  Controller(const Controller&)            = delete;
  Controller& operator=(const Controller&) = delete;

  void init(Dependencies deps);
  void deinit();

  // Called once per UI frame. Runs auto-poll if enabled and refreshes the
  // adapter heartbeat (throttled to once per second).
  void tick();

  // Manual actions (also invoked by the Agentic Window buttons).
  void initializeBridge();
  void pollNow();
  void queueHdriJob();
  void queueBeautifyJob();
  void randomizeGenerationSeed();
  void refreshAdapterStatus(bool force = false);

  // Display-overlay integration. The renderer's viewport-prep code asks the
  // controller whether to override the tonemapped buffer with the beautified
  // image; if so, it reads beautifiedDescriptor().
  bool                         hasBeautifiedImage() const { return m_beautifiedImage.image != VK_NULL_HANDLE; }
  const VkDescriptorImageInfo& beautifiedDescriptor() const { return m_beautifiedImage.descriptor; }
  VkExtent2D                   beautifiedExtent() const { return m_beautifiedExtent; }
  VkDescriptorSet              beautifiedImguiDescriptor() const { return m_beautifiedImguiDescriptor; }
  void                         destroyBeautifiedImage(bool deferred);

  // Override the bridge root (e.g. from the --agenticBridgeRoot CLI flag). Must be
  // called after init(); a no-op for an empty path so it never clobbers the default.
  void setBridgeRoot(const std::filesystem::path& root);

  // Path / configuration helpers (used by the "How to start the adapter" hint).
  std::filesystem::path bridgeRootPath() const;
  std::filesystem::path adapterScriptPath() const;
  std::filesystem::path detectComfyEmbeddedPython() const;

  // Reload the HDRI / Beautify prompt presets from prompts.json. Prefers
  // <bridgeRoot>/prompts.json (per-project override), else the shipped copy next
  // to the adapter script. Best-effort: leaves the lists empty if none is found.
  void                  loadPromptPresets();
  std::filesystem::path promptsFilePath() const;

  // ----- Public UI-bound state (ImGui binds directly to these). ----------------
  bool                      enabled{false};
  bool                      initialized{false};
  bool                      autoPoll{true};
  std::vector<PromptPreset> hdriPresets;      // from prompts.json; drives the HDRI Presets combo
  std::vector<PromptPreset> beautifyPresets;  // from prompts.json; drives the Beautify Presets combo
  char                      bridgeRoot[512]{};
  char                      converterPython[512]{};
  char                      hdriPrompt[kPromptBufferSize]{
      "Professional HDRI environment map, 360 degree equirectangular spherical panorama, "
                           "2:1 aspect ratio, seamless horizontal wrap, full sky and ground, horizon centered, "
                           "eye-level camera position, realistic natural lighting, physically plausible shadows "
                           "and reflections, high dynamic range lighting reference, clean uncluttered environment, "
                           "no main subject, suitable for lighting 3D renders, realistic exposure-bracketed photography, "
                           "natural color balance, detailed environment textures, photorealistic, sharp but not "
                           "oversharpened. Open modern courtyard at sunset, concrete ground, glass buildings, "
                           "warm low sun, soft long shadows, realistic sky, subtle clouds."};
  char beautifyPrompt[kPromptBufferSize]{
      "Photorealistic camera capture of the provided scene. Preserve the original render faithfully: "
      "same composition, camera angle, object identity, shapes, layout, lighting direction, colors, and materials. "
      "Enhance only realism and image quality: physically plausible lighting, natural exposure, "
      "realistic material micro-detail, subtle real-world imperfections, accurate reflections, "
      "soft contact shadows, natural contrast, balanced color grading, slight lens softness, "
      "subtle sensor grain, high-resolution real photograph, full-frame camera, reality."};
  int           generationSteps{20};  // KSampler / Flux2Scheduler steps (HDRI + Beautify)
  std::uint64_t generationSeed{0};    // KSampler.seed / RandomNoise.noise_seed
  // Selects the HDRI upscaler. Both workflows generate at 1024×512 and output
  // 4096×2048; the difference is the 4× upscale. On: hdri_from_prompt_4x.json uses
  // a model-based PixelDiT upscale (needs extra, larger ComfyUI models). Off
  // (default): hdri_from_prompt.json uses a plain bicubic upscale — no extra models.
  bool        hdriUpscale4x{false};
  std::string status{"Bridge disabled"};
  std::string lastMessage;

  // ----- Read-only state for UI display -------------------------------------
  const PendingJob&                          hdriJob() const { return m_hdriJob; }
  const PendingJob&                          beautifyJob() const { return m_beautifyJob; }
  const std::optional<agentic::JobProgress>& hdriProgress() const { return m_hdriProgress; }
  const std::optional<agentic::JobProgress>& beautifyProgress() const { return m_beautifyProgress; }
  const std::filesystem::path&               lastHdrPath() const { return m_lastHdrPath; }
  const std::filesystem::path&               lastBeautifiedPath() const { return m_lastBeautifiedPath; }
  const AdapterStatusInfo&                   adapterStatus() const { return m_adapterStatus; }

private:
  void appendGenerationSamplerParameters(std::map<std::string, std::string>& parameters) const;
  void saveBeautifyInputImage(const std::filesystem::path& path);
  bool loadBeautifiedImage(const std::filesystem::path& path);
  void handleResult(const agentic::GenerationResult& result, agentic::GenerationTask task);
  void refreshJobProgress(const PendingJob& job, std::optional<agentic::JobProgress>& progress);

  Dependencies                          m_deps;
  PendingJob                            m_hdriJob;
  PendingJob                            m_beautifyJob;
  std::optional<agentic::JobProgress>   m_hdriProgress;
  std::optional<agentic::JobProgress>   m_beautifyProgress;
  std::filesystem::path                 m_lastHdrPath;
  std::filesystem::path                 m_lastBeautifiedPath;
  nvvk::Image                           m_beautifiedImage;
  VkExtent2D                            m_beautifiedExtent{};
  VkDescriptorSet                       m_beautifiedImguiDescriptor{VK_NULL_HANDLE};
  AdapterStatusInfo                     m_adapterStatus;
  std::chrono::steady_clock::time_point m_lastAdapterCheck{};
};

// UI rendering -- implementation lives in ui_agentic.cpp. The window is reached
// from the Windows menu / F7; the feature deliberately adds no top-level menu.
void renderAgenticWindow(Controller& ctl, Resources& resources);

}  // namespace agentic
