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

#include <nvapp/application.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/resources.hpp>
#include <nvshaders_host/sky.hpp>

// Shader Input/Output
#include "shaders/shaderio.h"  // Shared between host and device


#include "resources.hpp"
#include "renderer_base.hpp"

#if defined(USE_DLSS)
class Dlss;  // forward decl, defined in dlss.hpp; only declared under USE_DLSS so the unique_ptr below has a complete type when needed
#endif

class Rasterizer : public BaseRenderer
{
public:
  Rasterizer();
  ~Rasterizer() override;

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

  void resetHdr() { m_lastHdrDomeView = {}; }

private:
  void renderNodes(VkCommandBuffer cmd, Resources& resources, const std::vector<uint32_t>& nodeIDs);
  void recordRasterScene(Resources& resources, VkExtent2D renderExtent);
  void renderRasterScene(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderExtent);
  void renderOpaqueOnly(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderExtent);
  void renderBlendOnly(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderExtent);
  void createRecordCommandBuffer();

  // Allocate / destroy the opaque-pass color capture used for KHR_materials_transmission.
  void createOpaqueColorImage(Resources& resources);
  void destroyOpaqueColorImage(Resources& resources);
  void captureAndMipOpaqueColor(VkCommandBuffer cmd, Resources& resources);

  // Allocate and bake the sheen directional-albedo LUT
  void createSheenLut(Resources& resources);
  void destroySheenLut(Resources& resources);

  // Sort blend (transparent) nodes back-to-front against the current view matrix. Returns true
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

  std::vector<uint32_t> m_sortedBlendNodes;     // Transparent (blend) draw ordering. Sorted back-to-front
  glm::mat4             m_lastViewMatrix{0.f};  // Last view matrix used to sort blend nodes.
  uint64_t m_lastShadedNodesRevision = 0;  // Last scene graph revision at which we updated m_sortedBlendNodes. Used to detect when we need to re-sort.
  uint64_t m_lastSceneGraphRevision = 0;  // Last scene graph revision we recorded commands against. Used to detect when we need to re-record.

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

  // ---- DLSS-SR (DLAA / Quality / Balanced / Performance / Ultra) ----
#if defined(USE_DLSS)
public:
  Dlss*       getDlss() { return m_dlss.get(); }
  const Dlss* getDlss() const { return m_dlss.get(); }

private:
  std::unique_ptr<Dlss> m_dlss;
#endif
  VkImageView m_lastHdrDomeView = VK_NULL_HANDLE;  // Last image view we wrote into HdrEnvDome::setOutImage
};
