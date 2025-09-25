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
#include "shaders/shaderio.h"  // Shared between host and device

#include <nvvk/sbt_generator.hpp>
#include <nvutils/profiler.hpp>
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
  void setProfilerTimeline(nvutils::ProfilerTimeline* timeline) { m_profilerTimeline = timeline; }
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

  // Adaptive sampling for performance optimization
  void                       updateAdaptiveSampling(Resources& resources);
  nvutils::ProfilerTimeline* m_profilerTimeline{nullptr};
  bool                       m_adaptiveSampling{true};
  int                        m_totalSamplesAccumulated{0};  // Track total samples separately
  double                     m_accumulationStartTime{0.0};  // Start time for throughput calculation

  // Adaptive performance targets
  enum class PerformanceTarget
  {
    eInteractive = 0,  // 60 FPS - for real-time interaction
    eBalanced    = 1,  // 30 FPS - good balance of responsiveness and quality
    eQuality     = 2,  // 15 FPS - prioritize quality convergence
    eMaxQuality  = 3   // 10 FPS - maximum GPU utilization for fastest convergence
  };

  PerformanceTarget    m_performanceTarget{PerformanceTarget::eBalanced};  // Default to balanced for path tracing
  static constexpr int MAX_SAMPLES_PER_PIXEL = 100;
  static constexpr int MIN_SAMPLES_PER_PIXEL = 1;

  double getTargetFrameTimeMs() const
  {
    switch(m_performanceTarget)
    {
      case PerformanceTarget::eInteractive:
        return 1000.0 / 60.0;  // 16.67ms
      case PerformanceTarget::eBalanced:
        return 1000.0 / 30.0;  // 33.33ms
      case PerformanceTarget::eQuality:
        return 1000.0 / 15.0;  // 66.67ms
      case PerformanceTarget::eMaxQuality:
        return 1000.0 / 10.0;  // 100ms
      default:
        return 1000.0 / 30.0;
    }
  }

  // #DLSS - Implementation of the DLSS denoiser
#if defined(USE_DLSS)
  std::unique_ptr<DlssDenoiser> m_dlss;
#endif

private:
  float calculateRawSampleThroughput(const Resources& resources) const;
};