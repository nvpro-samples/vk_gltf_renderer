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
#include <queue>
#include <mutex>

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
#include "silhouette.hpp"
#include "ui_animation_control.hpp"
#include "ui_busy_window.hpp"
#include "ui_scene_graph.hpp"

class GltfRenderer : public nvapp::IAppElement
{
public:
  GltfRenderer(nvutils::ParameterRegistry* parameterReg);
  ~GltfRenderer() override = default;

  void                                        createScene(const std::filesystem::path& sceneFilename);
  void                                        createHDR(const std::filesystem::path& hdrFilename);
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
  bool processQueuedCommandBuffers();

  void clearGbuffer(VkCommandBuffer cmd);
  void cleanupScene();           // Helper to cleanup current scene
  void rebuildSceneFromModel();  // Rebuild Vulkan scene after modifying the glTF model in-place
  void compileShaders();
  void createDescriptorSets();
  void createResourceBuffers();
  void createVulkanScene();
  void buildAccelerationStructures();  // Helper for BLAS/TLAS building
  void destroyResources();
  void resetFrame();
  void silhouette(VkCommandBuffer cmd);
  void tonemap(VkCommandBuffer cmd);
  bool updateTextures();
  void updateHdrImages();

  bool updateSceneChanges(VkCommandBuffer cmd);
  bool updateAnimation(VkCommandBuffer cmd);

  /////
  /// UI
  void          renderUI();
  void          renderMenu();
  void          renderMemoryStatistics();
  void          renderEnvironmentWindow();
  void          renderTonemapperWindow();
  void          renderStatisticsWindow();
  void          addToRecentFiles(const std::filesystem::path& filePath, int historySize = 20);
  void          removeFromRecentFiles(const std::filesystem::path& filePath);
  void          mouseClickedInViewport();
  nvutils::Bbox getRenderNodeBbox(int nodeID);
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

  uint32_t m_maxTextures{100'000U};  // Maximum number of textures supported by the descriptor set

  Resources  m_resources;
  PathTracer m_pathTracer;  // Path tracer renderer
  Rasterizer m_rasterizer;  // Rasterizer renderer

  UiSceneGraph     m_uiSceneGraph;  // Model UI
  BusyWindow       m_busy;
  AnimationControl m_animControl;  // Animation control (UI)
  Silhouette       m_silhouette;   // Silhouette renderer

  // Command buffer queue for deferred submission
  struct CommandBufferInfo
  {
    VkCommandBuffer cmdBuffer{};
    bool            isBlasBuild{false};  // Indicates if this is a BLAS build command
  };
  std::queue<CommandBufferInfo> m_cmdBufferQueue;
  std::mutex                    m_cmdBufferQueueMutex;

  glm::mat4 m_prevMVP{1.f};  // Previous MVP matrix for motion vectors

  VkCommandPool m_transientCmdPool{};  // Command pool for transient command buffers

  nvgui::SettingsHandler m_settingsHandler;  // Settings handler for ImGui.ini
};