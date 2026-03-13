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

#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>
#include <glm/glm.hpp>

#include <nvapp/application.hpp>
#include <nvgui/camera.hpp>
#include <nvshaders_host/tonemapper.hpp>
#include <nvslang/slang.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/hdr_ibl.hpp>
#include <nvvk/ray_picker.hpp>
#include <nvvk/resource_allocator.hpp>
#include "gltf_scene.hpp"
#include "gltf_scene_rtx.hpp"
#include "gltf_scene_vk.hpp"
#include <nvvk/profiler_vk.hpp>
#include "nvutils/parameter_registry.hpp"

// Shader Input/Output
#include "shaders/shaderio.h"  // Shared between host and device

#include "renderer_pathtracer.hpp"
#include "renderer_rasterizer.hpp"
#include "resources.hpp"
#include "renderer_silhouette.hpp"
#include "ui_busy_window.hpp"
#include "ui_scene_browser.hpp"
#include "ui_inspector.hpp"
#include "scene_selection.hpp"
#include "gizmo_visuals_vk.hpp"
#include "timeline_pipeline.hpp"
#include "undo_redo.hpp"

class GltfRenderer : public nvapp::IAppElement
{
public:
  GltfRenderer(nvutils::ParameterRegistry* parameterReg);
  ~GltfRenderer() override = default;

  void createScene(const std::filesystem::path& sceneFilename);
  void createSceneFromDescriptor(const std::filesystem::path& descriptorPath);
  void createHDR(const std::filesystem::path& hdrFilename);
  void onMergeScene(const std::filesystem::path& filename);
  void rebuildVulkanSceneFull();  // Full GPU rebuild including textures (for merge/compact operations)
  std::shared_ptr<nvutils::CameraManipulator> getCameraManipulator() { return m_cameraManip; }
  void                                        registerRecentFilesHandler();
  void setDlssHardwareAvailability(bool available);  // Set DLSS hardware/extension availability

private:
  void onAttach(nvapp::Application* app) override;
  void onDetach() override;
  void onFileDrop(const std::filesystem::path& filename) override;
  void onLastHeadlessFrame() override;
  void onRender(VkCommandBuffer cmd) override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override;
  void onUIMenu() override;
  void onUIRender() override;

  bool save(const std::filesystem::path& filename);
  bool updateFrameCounter();

  void clearGbuffer(VkCommandBuffer cmd);
  void cleanupScene();           // Helper to cleanup current scene
  void rebuildSceneFromModel();  // Rebuild Vulkan scene after modifying the glTF model in-place (preserves textures)
  void rebuildVulkanSceneInternal(bool rebuildTextures);  // Internal helper for scene rebuilds
  void compileShaders();
  void createDescriptorSets();
  void createResourceBuffers();
  void createVulkanScene();
  void buildAccelerationStructures();  // Helper for BLAS/TLAS building
  void destroyResources();
  void resetFrame();
  void silhouette(VkCommandBuffer cmd);
  void tonemap(VkCommandBuffer cmd);
  void renderVisualHelpers(VkCommandBuffer cmd);
  void updateGizmoAttachment();
  bool updateTextures();
  void updateHdrImages();

  bool updateSceneChanges(VkCommandBuffer cmd);
  bool updateAnimation(VkCommandBuffer cmd);

  // updateSceneChanges phase helpers (keep main function readable)
  void     updateSceneChanges_BlasRebuild(const nvvkgltf::Scene::DirtyFlags& df);
  void     updateSceneChanges_NodeTransforms(nvvkgltf::Scene* scene, const nvvkgltf::Scene::DirtyFlags& df);
  uint32_t updateSceneChanges_SyncGpuBuffers(VkCommandBuffer cmd, nvvkgltf::Scene* scene);
  void     updateSceneChanges_TlasUpdate(VkCommandBuffer cmd, nvvkgltf::Scene* scene);
  void     updateSceneChanges_RasterizerInvalidate(bool renderNodeOrNodeDirty);
  void     updateSceneChanges_TangentUpload(VkCommandBuffer cmd, nvvkgltf::Scene* scene, bool& changed);
  void     updateSceneChanges_Finalize(VkCommandBuffer cmd, bool changed, bool stagingFlushed, nvvkgltf::Scene* scene);

  // UI
  void          renderUI();
  void          renderMenu();
  void          renderFileMenu(bool                   validScene,
                               bool&                  newScene,
                               bool&                  openFile,
                               bool&                  mergeFile,
                               bool&                  loadHdrFile,
                               bool&                  saveFile,
                               bool&                  saveScreenFile,
                               bool&                  saveImageFile,
                               bool&                  closeApp,
                               std::filesystem::path& sceneToLoadFilename,
                               std::filesystem::path& sceneToMergeFilename);
  void          renderViewMenu(bool validScene, bool& fitScene, bool& fitObject, bool& toggleVsync);
  void          renderWindowsMenu();
  void          renderEditMenu(bool validScene);
  void          renderToolsMenu(bool validScene, bool& reloadShader, bool& compactScene);
  void          renderDebugMenu();
  void          onUndoRedo();
  void          renderMenuToolbarAndGizmos();
  void          renderMemoryStatistics();
  void          renderEnvironmentWindow();
  void          renderTonemapperWindow();
  void          renderStatisticsWindow();
  void          addToRecentFiles(const std::filesystem::path& filePath, int historySize = 20);
  void          removeFromRecentFiles(const std::filesystem::path& filePath);
  void          mouseClickedInViewport();
  nvutils::Bbox getRenderNodeBbox(int renderNodeIndex);
  nvutils::Bbox getRenderNodesBbox(const std::unordered_set<int>& renderNodeIndices);
  void          windowTitle();
  void          applyGltfCamera(int cameraIndex);
  void          setGltfCameraFromView(int cameraIndex);
  void          loadHdrFileDialog();

  // Recent files management
  std::vector<std::filesystem::path> m_recentFiles;

  // File dialog directories
  std::filesystem::path m_lastSceneDirectory;
  std::filesystem::path m_lastHdrDirectory;

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvapp::Application*                         m_app{};               // Application pointer
  VkDevice                                    m_device{};            // Convenient
  nvvk::RayPicker                             m_rayPicker{};         // Ray picker
  nvutils::ProfilerTimeline*                  m_profilerTimeline{};  // Timeline profiler
  nvvk::ProfilerGpuTimer                      m_profilerGpuTimer{};  // GPU profiler
  std::shared_ptr<nvutils::CameraManipulator> m_cameraManip;         // Camera manipulator

  nvutils::PerformanceTimer m_cpuTimer;               // CPU performance timer
  bool                      m_cpuTimePrinted{false};  // Track if CPU time has been printed
#ifndef NDEBUG
  bool m_validateGpuSync{true};
#endif

  uint32_t m_maxTextures{100'000U};  // Maximum number of textures supported by the descriptor set

  Resources  m_resources;
  PathTracer m_pathTracer;  // Path tracer renderer
  Rasterizer m_rasterizer;  // Rasterizer renderer

  // New Scene Browser system (parallel implementation)
  SceneSelection m_sceneSelection;  // Shared selection state
  UiSceneBrowser m_sceneBrowser;    // New scene browser
  UiInspector    m_inspector;       // New inspector
  BusyWindow    m_busy;
  Silhouette    m_silhouette;     // Silhouette renderer
  VisualHelpers m_visualHelpers;  // Grid + transform gizmo overlay

  // Undo/Redo
  UndoStack m_undoStack;

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

  // Non-blocking GPU loading pipeline (see timeline_pipeline.hpp for details).
  // Worker threads enqueue command buffers; the main thread calls poll() each frame.
  TimelinePipeline m_loadPipeline;

  glm::mat4 m_prevMVP{1.f};  // Previous MVP matrix for motion vectors

  VkCommandPool m_transientCmdPool{};  // Command pool for transient command buffers

  nvgui::SettingsHandler m_settingsHandler;  // Settings handler for ImGui.ini
};