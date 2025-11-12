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
#include "utils.hpp"
#include "pipeline_cache_util.hpp"

// #DLSS
#if defined(USE_DLSS)
#include "dlss_denoiser.hpp"
#endif

// #OPTIX
#if defined(USE_OPTIX_DENOISER)
#include "optix_denoiser.hpp"
#endif


class PathTracer : public BaseRenderer
{
public:
  PathTracer();
  ~PathTracer() override = default;

  enum class RenderTechnique
  {
    RayQuery,
    RayTracing
  };

  void onAttach(Resources& resources, nvvk::ProfilerGpuTimer* profiler) override;
  void setProfilerTimeline(nvutils::ProfilerTimeline* timeline) { m_profilerTimeline = timeline; }
  void onDetach(Resources& resources) override;
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size, Resources& resources) override;
  bool onUIRender(Resources& resources) override;
  void onRender(VkCommandBuffer cmd, Resources& resources) override;

  void updateDlssResources(VkCommandBuffer cmd, Resources& resources);
  void updateOptiXResources(VkCommandBuffer cmd, Resources& resources);
  void pushDescriptorSet(VkCommandBuffer cmd, Resources& resources, VkPipelineBindPoint bindPoint) const;
  void createPipeline(Resources& resources) override;
  void createRqPipeline(Resources& resources);
  void createRtxPipeline(Resources& resources);
  void compileShader(Resources& resources, bool fromFile = true) override;


  // Register command line parameters
  void registerParameters(nvutils::ParameterRegistry* paramReg);

  VkDevice                        m_device{};  // Vulkan device
  VkPipelineLayout                m_pipelineLayout{};
  VkPipeline                      m_rtxPipeline{};    // Ray tracing pipeline
  VkPipeline                      m_rqPipeline{};     // Ray tracing pipeline
  shaderio::PathtracePushConstant m_pushConst{};      // Information sent to the shader
  bool                            m_autoFocus{true};  // Enable auto-focus
  VkShaderModule                  m_shaderModule{};   // Shader module for RTX

  nvvk::PipelineCacheManager m_pipelineCache{};  // Pipeline cache for faster creation

  // Shader Binding Table (SBT)
  nvvk::Buffer                m_sbtBuffer{};   // Buffer for the Shader Binding Table
  nvvk::SBTGenerator::Regions m_sbtRegions{};  // The SBT regions (raygen, miss, chit, ahit)

  // Ray tracing properties
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtPipelineProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceRayTracingInvocationReorderPropertiesNV m_reorderProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_NV};

  bool m_supportSER{false};
  bool m_useSER{false};

  // The default rendering technique
  RenderTechnique m_renderTechnique{RenderTechnique::RayTracing};

  // Adaptive sampling for performance optimization
  void                       updateAdaptiveSampling(Resources& resources);
  nvutils::ProfilerTimeline* m_profilerTimeline{nullptr};
  bool                       m_adaptiveSampling{true};
  int                        m_totalSamplesAccumulated{0};  // Track total samples separately

  nvsamples::RollingAverage<float, 100> m_throughputRollingAvg;  // Rolling average of mega-sample-pixels per second (MSPP/s)

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

  bool isDlssEnabled() const
  {
#if defined(USE_DLSS)
    return m_dlss && m_dlss->isEnabled();
#else
    return false;
#endif
  }

  // #DLSS - Implementation of the DLSS denoiser
#if defined(USE_DLSS)
  std::unique_ptr<DlssDenoiser> m_dlss;
  DlssDenoiser*                 getDlssDenoiser() { return m_dlss.get(); }
  const DlssDenoiser*           getDlssDenoiser() const { return m_dlss.get(); }
#endif

  // #OPTIX - Implementation of the OptiX denoiser
#if defined(USE_OPTIX_DENOISER)
  std::unique_ptr<OptiXDenoiser> m_optix;
  OptiXDenoiser*                 getOptiXDenoiser() { return m_optix.get(); }
  const OptiXDenoiser*           getOptiXDenoiser() const { return m_optix.get(); }
#endif


private:
  void updateStatistics(Resources& resources);
  void renderRayQuery(VkCommandBuffer cmd, VkExtent2D renderingSize, Resources& resources);


  void renderRayTrace(VkCommandBuffer cmd, VkExtent2D& renderingSize, Resources& resources);


  void denoiseDlss(VkCommandBuffer cmd, Resources& resources);


  void setupPushConstant(VkCommandBuffer cmd, Resources& resources);

  // Determine if DLSS should actively denoise this frame
  bool getEffectiveDlssEnabled(const Resources& resources) const;

  // Determine if OptiX should actively denoise this frame
  bool getEffectiveOptixEnabled(const Resources& resources) const;
};