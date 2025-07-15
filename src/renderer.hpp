/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
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
#include <nvvkgltf/scene.hpp>
#include <nvvkgltf/scene_rtx.hpp>
#include <nvvkgltf/scene_vk.hpp>
#include <nvvk/profiler_vk.hpp>
#include "nvutils/parameter_registry.hpp"

// Shader Input/Output
namespace shaderio {
using namespace glm;
#include "shaders/shaderio.h"  // Shared between host and device
}  // namespace shaderio

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

  void createScene(const std::filesystem::path& sceneFilename);
  void createHDR(const std::filesystem::path& hdrFilename);
  void setCameraManipulator(std::shared_ptr<nvutils::CameraManipulator> cameraManip)
  {
    m_resources.cameraManip = cameraManip;
  }
  void registerRecentFilesHandler();

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
  bool updateAnimation(VkCommandBuffer cmd);
  bool updateFrameCounter();
  bool processQueuedCommandBuffers();

  void clearGbuffer(VkCommandBuffer cmd);
  void compileShaders();
  void createDescriptorSets();
  void createResourceBuffers();
  void createVulkanScene();
  void destroyResources();
  void resetFrame();
  void silhouette(VkCommandBuffer cmd);
  void tonemap(VkCommandBuffer cmd);
  void updateNodeToRenderNodeMap();
  void updateTextures();
  void updateHdrImages();

  bool updateSceneChanges(VkCommandBuffer cmd, bool didAnimate);

  /////
  /// UI
  void          renderUI();
  void          renderMenu();
  void          addToRecentFiles(const std::filesystem::path& filePath, int historySize = 20);
  void          mouseClickedInViewport();
  nvutils::Bbox getRenderNodeBbox(int nodeID);
  void          windowTitle();
  void          applyGltfCamera(int cameraIndex);
  void          setGltfCameraFromView(int cameraIndex);

  // Recent files management
  std::vector<std::filesystem::path> m_recentFiles;

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvapp::Application*        m_app{};               // Application pointer
  VkDevice                   m_device{};            // Convenient
  nvvk::RayPicker            m_rayPicker{};         // Ray picker
  nvutils::ProfilerTimeline* m_profilerTimeline{};  // Timeline profiler
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer{};  // GPU profiler

  Resources  m_resources;
  PathTracer m_pathTracer;  // Path tracer renderer
  Rasterizer m_rasterizer;  // Rasterizer renderer

  UiSceneGraph     m_uiSceneGraph;  // Model UI
  BusyWindow       m_busy;
  AnimationControl m_animControl;  // Animation control (UI)
  Silhouette       m_silhouette;   // Silhouette renderer

  std::unordered_map<int, int> m_nodeToRenderNodeMap;  // Maps node IDs to render node indices

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
};