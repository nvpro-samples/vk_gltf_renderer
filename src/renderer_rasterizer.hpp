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
#include <nvapp/application.hpp>
#include <nvvk/graphics_pipeline.hpp>
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

  void pushDescriptorSet(VkCommandBuffer cmd, Resources& resources);
  void compileShader(Resources& resources, bool fromFile = true) override;
  void createPipeline(Resources& resources) override;
  void freeRecordCommandBuffer(Resources& resources);

  // Register command line parameters
  void registerParameters(nvutils::ParameterRegistry* paramReg);

private:
  void renderNodes(VkCommandBuffer cmd, Resources& resources, const std::vector<uint32_t>& nodeIDs);
  void recordRasterScene(Resources& resources);
  void renderRasterScene(VkCommandBuffer cmd, Resources& resources);
  void createRecordCommandBuffer();


  VkDevice         m_device{};                 // Vulkan device
  VkCommandBuffer  m_recordedSceneCmd{};       // Command buffer for recording the scene
  VkCommandPool    m_commandPool{};            // Command pool for recording the scene
  VkPipelineLayout m_graphicPipelineLayout{};  // The pipeline layout use with graphics pipeline

  nvvk::GraphicsPipelineState m_dynamicPipeline;  // Graphics pipeline state
  nvvk::DescriptorBindings    m_descBind;         // Descriptor bindings

  shaderio::RasterPushConstant m_pushConst{};  // Reusing the same push constant structure for now


  VkShaderEXT m_vertexShader{};     // Vertex shader
  VkShaderEXT m_fragmentShader{};   // Fragment shader
  VkShaderEXT m_wireframeShader{};  // Wireframe shader

  nvshaders::SkyPhysical m_skyPhysical;  // Sky physical

  // UI
  bool m_enableWireframe = false;
  bool m_useRecordedCmd  = true;  // Use recorded command buffer for rendering
};
