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

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>
#include <glm/glm.hpp>

#include <nvapp/application.hpp>
#include <nvgui/camera.hpp>
#include <nvshaders_host/tonemapper.hpp>
#include <nvslang/slang.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/hdr_ibl.hpp>
#include <nvvk/ray_picker.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/semaphore.hpp>
#include "gltf_scene.hpp"
#include "gltf_scene_rtx.hpp"
#include "gltf_scene_vk.hpp"
#include <nvvk/profiler_vk.hpp>
#include "nvutils/parameter_parser.hpp"
#include "nvutils/parameter_registry.hpp"
#include "nvutils/parameter_sequencer.hpp"

// Shader Input/Output
#include "shaders/shaderio.h"  // Shared between host and device

#include "benchmarking.hpp"
#include "renderer_pathtracer.hpp"
#include "renderer_rasterizer.hpp"
#include "resources.hpp"
#include "renderer_silhouette.hpp"
#include "hover_picker.hpp"
#include "ui_busy_window.hpp"
#include "ui_scene_browser.hpp"
#include "ui_inspector.hpp"
#include "ui_thumbnail_cache.hpp"
#include "ui_toast.hpp"
#include "scene_selection.hpp"
#include "gizmo_visuals_vk.hpp"
#include "timeline_pipeline.hpp"
#include "undo_redo.hpp"
#ifdef USE_AGENTIC
#include "agentic.hpp"
#endif

class GltfRenderer : public nvapp::IAppElement
{
public:
  GltfRenderer(nvutils::ParameterRegistry* parameterReg, const nvutils::ParameterParser* parameterParser, BenchmarkOptions& benchmarkOptions);
  ~GltfRenderer() override = default;

  /// Emits parseable BENCHMARK_ADV memory block (called from ParameterSequencer post-callback).
  void benchmarkAdvance(const nvutils::ParameterSequencer::State& state);

  void createScene(const std::filesystem::path& sceneFilename);
  // Ensure an editable Scene exists (wired to the UI, no GPU build) so add/import can run from nothing.
  void ensureEmptyScene();
  void createSceneFromDescriptor(const std::filesystem::path& descriptorPath);
  void createHDR(const std::filesystem::path& hdrFilename);
  void onMergeScene(const std::filesystem::path& filename);
  void onReferenceScene(const std::filesystem::path& filename);
  // Shared worker for merge (embed) and reference (glTF 2.1 external asset) imports.
  void addSceneFromFile(const std::filesystem::path& filename, bool asReference);
  /// Full GPU rebuild (textures + geometry + AS). Only uploads GPU data; CPU side must already be
  /// parsed (`mergeScene`, `parseScene`, etc., depending on how the model was changed).
  void                                        rebuildVulkanSceneFull();
  std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() { return m_cameraManip; }
  void                                        registerRecentFilesHandler();
  void                                        setDlssHardwareAvailability(bool rrAvailable, bool srAvailable);
  void                                        setOpacityMicromapAvailable(bool available);
#ifdef USE_AGENTIC
  /// Override the optional Agentic bridge root (from --agenticBridgeRoot). Applied
  /// to the controller once it is initialized in onAttach.
  void setAgenticBridgeRoot(const std::filesystem::path& root);
#endif
  /// Ensures path-tracer accumulation covers the full headless run (--maxFrames >= --frames).
  void alignMaxFramesForHeadless(uint32_t headlessFrames);

private:
  void onAttach(nvapp::Application* app) override;
  void onDetach() override;
  void onFileDrop(const std::filesystem::path& filename) override;
  void onLastHeadlessFrame() override;
  void onRender(VkCommandBuffer cmd) override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;
  void onUIMenu() override;
  void onUIRender() override;

  bool save(const std::filesystem::path& filename, bool selfContained = false);
  bool updateFrameCounter();

  void clearGbuffer(VkCommandBuffer cmd);
  void cleanupScene();  // Helper to cleanup current scene
  void rebuildSceneFromModel();  // Rebuild Vulkan scene after modifying the glTF model in-place (preserves textures); clears undo
  void rebuildSceneGeometry();  // Geometry-only rebuild (preserves textures); does NOT clear undo (used by undoable geometry edits)
  void reconcileGeometryIfNeeded();  // Rebuild geometry when GPU buffers are behind the render-primitive count (e.g. added primitive)
  void applyPendingTextureRebuild();  // Consume DirtyFlags::texturesChanged at frame top: full texture rebuild + cache refresh
  void applyPendingTextureTailSync();  // Consume DirtyFlags::texturesTailChanged at frame top: incremental append/remove of tail textures
  void applyPendingSamplerUpdate();  // Consume DirtyFlags::samplers at frame top: in-place VkSampler update, no image touch
  void refreshCpuSceneGraphFromModel();
  void rebuildVulkanSceneInternal(nvvkgltf::SceneGpu::RebuildMode mode);  // GPU upload + AS; CPU scene must already be parsed
  void compileShaders();
  void createDescriptorSets();
  void createResourceBuffers();
  void createVulkanScene();
  void finalizeSceneSetup(const std::filesystem::path& filename);  // Shared GPU build + UI wiring after a load
  void wireSceneToUi();                                            // Wire current scene into browser/inspector panels
  void buildAccelerationStructures();                              // Helper for BLAS/TLAS building
  void destroyResources();
  void resetFrame();
  void selectSceneNodeFromScript(int nodeIndex);  // Script-driven selection (Inspector/Scene Browser)
  // Script-driven *ray-pick* selection (`pickrendernode`): selects the given glTF node's
  // first render primitive the same way a real 3D-viewport click would - unlike
  // selectSceneNodeFromScript() above, this supplies finite (camera-eye/node-position) selectionPoint/
  // selectionRayOrigin values, so KHR_interactivity event/onSelect's ray-info output sockets can be
  // exercised end-to-end without an actual mouse click.
  void pickSceneNodeFromScript(int nodeIndex);
  void silhouette(VkCommandBuffer cmd);
  void tonemap(VkCommandBuffer cmd);
  void runTonemapPass(VkCommandBuffer cmd, bool skipBeautifiedOverlay);
  void renderVisualHelpers(VkCommandBuffer cmd);
#if defined(USE_DLSS)
  Dlss*       activeDlss();
  const Dlss* activeDlss() const;
#endif

  bool dlssGuideRequired() const;  // True when the path tracer currently needs DLSS/OptiX guide-buffer capture code.
  void updateGizmoAttachment();
  bool updateTextures();
  // Write a contiguous range of scene texture / sampler descriptors (eTextures / eSamplers). updateTextures()
  // writes the whole set; applyPendingTextureTailSync() writes only the newly appended slots.
  bool writeTextureDescriptorRange(uint32_t firstTexture, uint32_t textureCount, uint32_t firstSampler, uint32_t samplerCount);
  void updateHdrImages();

  bool updateSceneChanges(VkCommandBuffer cmd);
  bool updateAnimation(VkCommandBuffer cmd);
  // Shared downstream GPU reconciliation after CPU-side animation-channel evaluation dirtied the
  // model (world matrices, GPU sync, morph/skin compute, BLAS/TLAS update, dirty-flag clear) - the
  // common tail of updateAnimation() (UI-driven single clip) and updateInteractivityGraphs()
  // (KHR_interactivity-driven, possibly several clips at once via animation/start).
  void reconcileAnimationGpuState(VkCommandBuffer cmd);
  // KHR_interactivity: ticks the scene's default behavior graph instance, then applies any
  // animation/start-driven pose changes it computed this tick (via reconcileAnimationGpuState()).
  // pointer/set's glTF-model writes flow through Scene::markNodeDirty/markMaterialDirty and are
  // picked up by the dirty-flags path in updateSceneChanges() right after this call, same as before.
  bool updateInteractivityGraphs(VkCommandBuffer cmd);
  // Polls m_hoverPicker for a completed G-buffer readback and, on a change, notifies the scene's
  // interactivity graph (Scene::notifyNodeHoverChanged) - called once per frame, before
  // updateInteractivityGraphs() so a hover transition this frame feeds this same frame's tick.
  void updateHoverState();
  // Script-driven hover for UI scenario tests: calls the exact same notification path
  // updateHoverState() uses for real cursor input, bypassing the async GPU readback for determinism.
  void hoverSceneNodeFromScript(int nodeIndex);
  // UI-phase capture of cursor-over-viewport state; see the member fields' comments.
  void updateHoverCursorPosition();

  // Polls m_pendingClickResult's frame semaphore and, once signaled, calls m_rayPicker.getResult()
  // and applies it via applyClickPickResult() - called once per frame, next to updateHoverState().
  void updateClickPickState();
  // Applies a completed ray-pick result: selection (and, on a double click, camera recenter) -
  // the exact logic mouseClickedInViewport() used to run inline, now one to two frames removed
  // from the click itself (see m_pendingClickPick's comment for why).
  void applyClickPickResult(const nvvk::RayPicker::PickResult& pickResult, bool isDoubleClick);

  // Headless / scripted benchmark (shared automation paths)
  [[nodiscard]] bool                             isBenchmarkMode() const;
  [[nodiscard]] bool                             isHeadlessMode() const;
  [[nodiscard]] bool                             isAutomatedRun() const;
  BenchmarkController::HeadlessFrameInfo         benchmarkFrameInfo() const;
  std::vector<BenchmarkController::MemorySample> benchmarkMemorySamples() const;
  void                                           saveHeadlessOutputImage();

  // updateSceneChanges phase helpers (keep main function readable)
  void updateSceneChanges_BlasRebuild(const nvvkgltf::Scene::DirtyFlags& df);
  void updateSceneChanges_NodeTransforms(VkCommandBuffer cmd, nvvkgltf::Scene* scene, const nvvkgltf::Scene::DirtyFlags& df);
  uint32_t updateSceneChanges_SyncGpuBuffers(VkCommandBuffer cmd, nvvkgltf::Scene* scene);
  void     updateSceneChanges_TlasUpdate(VkCommandBuffer cmd, nvvkgltf::Scene* scene);
  void     updateSceneChanges_RasterizerInvalidate(bool renderNodeOrNodeDirty);
  void     updateSceneChanges_TangentUpload(VkCommandBuffer cmd, nvvkgltf::Scene* scene, bool& changed);
  void     updateSceneChanges_Finalize(VkCommandBuffer cmd, bool changed, bool stagingFlushed, nvvkgltf::Scene* scene);

  // UI
  void renderUI();
  void renderBenchmarkViewport();  // Minimal fullscreen image (benchmark mode)
  void renderMenu();
  void renderFileMenu(bool                   validScene,
                      bool&                  newScene,
                      bool&                  openFile,
                      bool&                  mergeFile,
                      bool&                  loadHdrFile,
                      bool&                  saveFile,
                      bool&                  saveAsFile,
                      bool&                  saveSelfContainedAsFile,
                      bool&                  saveScreenFile,
                      bool&                  saveImageFile,
                      bool&                  referenceFile,
                      bool&                  closeApp,
                      std::filesystem::path& sceneToLoadFilename,
                      std::filesystem::path& sceneToMergeFilename);
  void renderViewMenu(bool validScene, bool& fitScene, bool& fitObject, bool& toggleVsync);
  void renderWindowsMenu();
  void renderEditMenu(bool validScene);
  void renderCreateMenu();  // "Create" menu: add procedural primitives (enabled whenever a scene exists)
  void renderToolsMenu(bool validScene, bool& reloadShader, bool& compactScene);
  void renderDebugMenu();
  void onUndoRedo();
  void renderMenuToolbarAndGizmos();
  void renderMemoryStatistics();
  void renderEnvironmentWindow();
  void renderTonemapperWindow();
  void renderStatisticsWindow();
  void addToRecentFiles(const std::filesystem::path& filePath, int historySize = 20);
  void removeFromRecentFiles(const std::filesystem::path& filePath);
  void mouseClickedInViewport();
  // True when the current scene has a KHR_interactivity graph and it's actively ticking - the
  // shared gate for every place viewport click behavior changes while a graph is playing (see
  // docs/interactivity.md's design notes): skipping the click-to-deselect toggle
  // (updateSelectionFromPick) and skipping the single/double-click debounce (mouseClickedInViewport).
  bool isInteractivityPlaying() const;
  // `selectionPoint`/`selectionRayOrigin` (global space) are the real ray-pick hit point/origin -
  // default NaN for callers with no ray (e.g. script-driven test commands), matching the
  // spec-sanctioned "no ray info" fallback (see SceneSelection::Event's doc comment).
  void          updateSelectionFromPick(int              renderNodeIdx,
                                        const glm::vec3& selectionPoint = glm::vec3(std::numeric_limits<float>::quiet_NaN()),
                                        const glm::vec3& selectionRayOrigin = glm::vec3(std::numeric_limits<float>::quiet_NaN()));
  nvutils::Bbox getRenderNodeBbox(int renderNodeIndex);
  nvutils::Bbox getRenderNodesBbox(const std::unordered_set<int>& renderNodeIndices);
  void          windowTitle();
  void          applyGltfCamera(int cameraIndex);
  void          setGltfCameraFromView(int cameraIndex);
  void          loadHdrFileDialog();

  // Opens an image file dialog for the inspector's "Load from file" texture action. Returns the chosen
  // path, or an empty path if cancelled.
  std::filesystem::path pickImageFile();

  // Resolve a glTF texture / image index to a bounded ImGui thumbnail (0 if not resident). Handed to
  // the inspector and scene browser so their panels can show thumbnails without knowing about the GPU.
  ImTextureID thumbnailForTexture(int textureIndex);
  ImTextureID thumbnailForImage(int imageIndex);

  // Push a transient notification to the on-screen toast overlay (isError = red). Wired into the
  // inspector / scene browser so their edit actions can surface failures beyond the log.
  void notify(const std::string& message, bool isError);

  // Recent files management
  std::vector<std::filesystem::path> m_recentFiles;

  // Optional Agentic bridge controller (HDRI from prompt, image-to-image
  // beautify). All state, polling, Vulkan upload of the beautified image, and
  // adapter-heartbeat tracking live in agentic::Controller; renderAgenticWindow
  // in ui_agentic.cpp drives the UI. Compiled only when USE_AGENTIC is defined.
#ifdef USE_AGENTIC
  agentic::Controller   m_agentic;
  std::filesystem::path m_agenticBridgeRootOverride;  // from --agenticBridgeRoot; applied after m_agentic.init()
#endif

  // File dialog directories
  std::filesystem::path m_lastSceneDirectory;
  std::filesystem::path m_lastHdrDirectory;
  std::filesystem::path m_lastImageDirectory;
  // Default filename offered by the Save Image / Save Screen Image dialogs: the loaded scene's
  // name (.jpg) until the user saves under a different name, reset on New Scene / new scene load.
  std::filesystem::path m_imageSaveFilename;

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvapp::Application*                         m_app{};               // Application pointer
  VkDevice                                    m_device{};            // Convenient
  nvvk::RayPicker                             m_rayPicker{};         // Ray picker
  nvutils::ProfilerTimeline*                  m_profilerTimeline{};  // Timeline profiler
  nvvk::ProfilerGpuTimer                      m_profilerGpuTimer{};  // GPU profiler
  std::shared_ptr<nvutils::CameraManipulator> m_cameraManip;         // Camera manipulator

  nvutils::PerformanceTimer m_cpuTimer;               // CPU performance timer (path-tracer accumulation window)
  bool                      m_cpuTimePrinted{false};  // Track if CPU time has been printed

#ifndef NDEBUG
  bool m_validateGpuSync{true};
  bool m_skipGpuSyncValidation{false};  // GPU transform path skipped uploadRenderNodes / CPU TLAS sync
#endif

  uint32_t m_maxTextures{100'000U};  // Maximum number of material images (eTextures SAMPLED_IMAGE array)
  uint32_t m_maxSamplers{0U};        // Maximum number of samplers (eSamplers SAMPLER array); set from device limits

  Resources  m_resources;
  PathTracer m_pathTracer;  // Path tracer renderer
  Rasterizer m_rasterizer;  // Rasterizer renderer

  // New Scene Browser system (parallel implementation)
  SceneSelection m_sceneSelection;  // Shared selection state
  UiSceneBrowser m_sceneBrowser;    // New scene browser
  UiInspector    m_inspector;       // New inspector
  ThumbnailCache m_thumbnailCache;  // Bounded ImGui thumbnails for scene textures/images
  UiToasts       m_toasts;          // Transient error/info notifications (e.g. failed image import)
  BusyWindow     m_busy;
  Silhouette     m_silhouette;     // Silhouette renderer
  VisualHelpers  m_visualHelpers;  // Grid + transform gizmo overlay
  HoverPicker    m_hoverPicker;    // KHR_interactivity hover detection (docs/interactivity.md Phase E)

  // Undo/Redo
  UndoStack m_undoStack;

  // Application-level delete confirmation (shared with scene browser via pointers)
  int  m_pendingDeleteNode        = -1;
  bool m_openDeletePopupNextFrame = false;

  // Gizmo TRS local storage (pointers passed to TransformHelperVk)
  glm::vec3 m_gizmoPosition{0.f};
  glm::vec3 m_gizmoRotation{0.f};  // Euler angles in degrees
  glm::vec3 m_gizmoScale{1.f};
  glm::mat4 m_gizmoParentWorldMatrix{1.f};
  int       m_gizmoNodeIndex = -1;

  // Gizmo TRS snapshot for undo (captured at drag start)
  glm::vec3 m_gizmoSnapshotT{0.f};
  glm::quat m_gizmoSnapshotR{1, 0, 0, 0};
  glm::vec3 m_gizmoSnapshotS{1.f};

  // KHR_interactivity hover state: the glTF node currently under the cursor, -1 for none. Driven
  // by m_hoverPicker's async G-buffer readback (see updateHoverState()); also settable directly
  // by the `hovernode`/`clearhover` script commands for deterministic testing.
  int m_hoveredNodeIndex = -1;
  // Cursor position captured during the UI phase (updateHoverCursorPosition(), ui_renderer.cpp)
  // for onRender()'s later use - ImGui hover/cursor queries aren't valid from onRender() itself.
  glm::ivec2 m_hoverCursorPixel{-1, -1};
  bool       m_hoverCursorInViewport = false;

  // Async click ray-pick (avoids the ~300ms stall of the old synchronous submitAndWaitTempCmdBuffer
  // path - see docs/interactivity.md's design note). A click captured during the UI phase
  // (mouseClickedInViewport(), ui_renderer.cpp) becomes a PendingClickPick with everything the pick
  // needs baked in *at click time* (camera state, normalized cursor position, single/double-click
  // classification) - no GPU work yet. onRender(cmd) then records m_rayPicker.run(cmd, ...) into the
  // frame's own command buffer (next to m_hoverPicker.requestReadback - the TLAS is already proven
  // valid there) and moves it into a PendingClickResult carrying just the frame semaphore to poll.
  // updateClickPickState() (called next to updateHoverState()) polls that semaphore non-blocking and,
  // once signaled, reads m_rayPicker.getResult() and applies it via applyClickPickResult(). A newer
  // click simply overwrites whichever of these is pending - nvvk::RayPicker has one internal result
  // buffer, so only the most recent run() before a getResult() call is ever meaningful anyway.
  struct PendingClickPick
  {
    glm::mat4 modelViewInv;
    glm::mat4 perspectiveInv;
    int       isOrthographic;
    glm::vec2 pickPos;
    bool      isDoubleClick;
  };
  std::optional<PendingClickPick> m_pendingClickPick;
  struct PendingClickResult
  {
    nvvk::SemaphoreState semaphoreState;
    bool                 isDoubleClick;
  };
  std::optional<PendingClickResult> m_pendingClickResult;

  // Non-blocking GPU loading pipeline (see timeline_pipeline.hpp for details).
  // Worker threads enqueue command buffers; the main thread calls poll() each frame.
  TimelinePipeline m_loadPipeline;

  glm::mat4 m_prevMVP{1.f};  // Previous MVP matrix for motion vectors

  VkCommandPool m_transientCmdPool{};  // Command pool for transient command buffers

  nvgui::SettingsHandler          m_settingsHandler;    // Settings handler for ImGui.ini
  const nvutils::ParameterParser* m_parameterParser{};  // CLI parameter parser, for INI load filtering (see wasParsed)

  BenchmarkController m_benchmark;
};
