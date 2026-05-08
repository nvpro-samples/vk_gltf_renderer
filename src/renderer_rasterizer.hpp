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
#include <nvapp/application.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/resources.hpp>
#include <nvshaders_host/sky.hpp>

// Shader Input/Output
#include "shaders/shaderio.h"  // Shared between host and device


#include "resources.hpp"
#include "renderer_base.hpp"

class Rasterizer : public BaseRenderer
{
public:
  Rasterizer()          = default;
  virtual ~Rasterizer() = default;

  void onAttach(Resources& resources, nvvk::ProfilerGpuTimer* profiler) override;
  void onDetach(Resources& resources) override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size, Resources& resources) override;
  bool onUIRender(Resources& resources) override;
  void onRender(VkCommandBuffer cmd, Resources& resources) override;
  void onSceneInvalidated(Resources& resources) override;

  void compileShader(Resources& resources, bool fromFile = true) override;
  void createPipeline(Resources& resources) override;
  void freeRecordCommandBuffer(Resources& resources);

  // Register command line parameters
  void registerParameters(nvutils::ParameterRegistry* paramReg);
  void setSettingsHandler(nvgui::SettingsHandler* settingsHandler);

private:
  void renderNodes(VkCommandBuffer cmd, Resources& resources, const std::vector<uint32_t>& nodeIDs);
  void recordRasterScene(Resources& resources);
  void renderRasterScene(VkCommandBuffer cmd, Resources& resources);
  void renderOpaqueOnly(VkCommandBuffer cmd, Resources& resources);
  void renderBlendOnly(VkCommandBuffer cmd, Resources& resources);
  void createRecordCommandBuffer();

  // Allocate / destroy the opaque-pass color capture used for KHR_materials_transmission.
  // Single 2D image, OPAQUE_COLOR_SIZE x OPAQUE_COLOR_SIZE, R16G16B16A16_SFLOAT, mip chain.
  void createOpaqueColorImage(Resources& resources);
  void destroyOpaqueColorImage(Resources& resources);
  // Resolve the rendered opaque image into m_opaqueColorImage and generate the mip chain.
  void captureAndMipOpaqueColor(VkCommandBuffer cmd, Resources& resources);

  // Allocate and bake the sheen directional-albedo LUT (used by the raster IBL composition for
  // both the Charlie BRDF term and the sheen energy-compensation factor). One-time bake at
  // attach; the LUT is independent of the HDR environment.
  void createSheenLut(Resources& resources);
  void destroySheenLut(Resources& resources);

  // Sort blend (transparent) nodes back-to-front against the current view matrix, returning
  // a list whose ordering matches the previous call when the order is stable. Returns true
  // if the order changed since the last call (used to invalidate recorded command buffers).
  bool updateSortedBlendNodes(VkCommandBuffer cmd, Resources& resources);


  VkDevice         m_device{};                 // Vulkan device
  VkCommandBuffer  m_recordedSceneCmd{};       // Command buffer for recording the scene
  VkCommandPool    m_commandPool{};            // Command pool for recording the scene
  VkPipelineLayout m_graphicPipelineLayout{};  // The pipeline layout use with graphics pipeline

  nvvk::GraphicsPipelineState m_dynamicPipeline;  // Graphics pipeline state

  shaderio::RasterPushConstant m_pushConst{};  // Reusing the same push constant structure for now


  VkShaderEXT m_vertexShader{};     // Vertex shader
  VkShaderEXT m_fragmentShader{};   // Fragment shader
  VkShaderEXT m_wireframeShader{};  // Wireframe shader

  nvshaders::SkyPhysical m_skyPhysical;  // Sky physical

  // Transparent (blend) draw ordering. Sorted back-to-front each frame from the camera; we
  // re-record the secondary command buffer when the order changes (camera moves enough).
  // The list persists across frames so the depth sort can exploit temporal coherence: a stable
  // scene typically drifts by a couple of inversions per frame, making insertion sort near-O(N).
  std::vector<uint32_t> m_sortedBlendNodes;
  // Short-circuit memo: when the view matrix, Scene::getShadedNodesRevision(), and
  // Scene::getSceneGraphRevision() are all unchanged since our last sort, m_sortedBlendNodes is
  // still valid and we can skip the whole sort pass. Reset by onSceneInvalidated() on scene
  // swaps so we don't mistake a new Scene's starting revisions for the previous Scene's.
  glm::mat4 m_lastViewMatrix{0.f};
  uint64_t  m_lastShadedNodesRevision = 0;
  uint64_t  m_lastSceneGraphRevision  = 0;

  // Opaque-pass color capture for screen-space transmission (KHR_materials_transmission).
  // Allocated lazily on first use; size matches OPAQUE_COLOR_SIZE in shaderio.h.
  nvvk::Image m_opaqueColorImage{};
  bool        m_sceneHasTransmission = false;  // computed each frame from scene materials

  // Baked sheen directional-albedo LUT (R16G16_SFLOAT, 64x64). Independent of the HDR env, so
  // this is created once at attach time.
  nvvk::Image m_sheenLut{};

  // UI
  bool m_useRecordedCmd = true;   // Use recorded command buffer for rendering
  bool m_lastWireframe  = false;  // Tracks settings.wireframe to invalidate recorded commands
};
