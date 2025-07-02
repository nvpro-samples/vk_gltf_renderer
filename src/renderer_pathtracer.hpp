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

#include <glm/glm.hpp>

// Shader Input/Output
namespace shaderio {
using namespace glm;
#include "shaders/shaderio.h"  // Shared between host and device
}  // namespace shaderio

#include <nvvk/sbt_generator.hpp>
#include "renderer_base.hpp"

// #DLSS
#if defined(USE_DLSS)
#include "dlss_denoiser.hpp"
#endif


class PathTracer : public BaseRenderer
{
public:
  PathTracer();
  ~PathTracer() override = default;

  enum class RenderTechnique
  {
    Compute,
    RayTracing
  };

  void onAttach(Resources& resources, nvvk::ProfilerGpuTimer* profiler) override;
  void onDetach(Resources& resources) override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size, Resources& resources) override;
  bool onUIRender(Resources& resources) override;
  void onRender(VkCommandBuffer cmd, Resources& resources) override;
  void onUIMenu() override;

  void updateDlssResources(VkCommandBuffer cmd, Resources& resources);
  void pushDescriptorSet(VkCommandBuffer cmd, Resources& resources, VkPipelineBindPoint bindPoint) const;
  void createPipeline(Resources& resources) override;
  void createRtxPipeline(Resources& resources);
  void compileShader(Resources& resources, bool fromFile = true) override;

  // Register command line parameters
  void registerParameters(nvutils::ParameterRegistry* paramReg);

  VkDevice                        m_device{};  // Vulkan device
  VkPipelineLayout                m_pipelineLayout{};
  VkPipeline                      m_pipeline{};   // Ray tracing pipeline
  shaderio::PathtracePushConstant m_pushConst{};  // Information sent to the shader
  VkShaderEXT                     m_shader{};
  float                           m_sceneRadius{1.0f};
  bool                            m_autoFocus{true};  // Enable auto-focus
  VkShaderModule                  m_shaderModule{};   // Shader module for RTX

  // Shader Binding Table (SBT)
  nvvk::Buffer                m_sbtBuffer{};   // Buffer for the Shader Binding Table
  nvvk::SBTGenerator::Regions m_sbtRegions{};  // The SBT regions (raygen, miss, chit, ahit)

  // Ray tracing properties
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtPipelineProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceRayTracingInvocationReorderPropertiesNV m_reorderProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_NV};

  RenderTechnique m_renderTechnique{RenderTechnique::Compute};

  // #DLSS - Implementation of the DLSS denoiser
#if defined(USE_DLSS)
  std::unique_ptr<DlssDenoiser> m_dlss;
#endif
};