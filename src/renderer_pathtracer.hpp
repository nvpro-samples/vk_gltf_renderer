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

#include "settings_registry.hpp"

#include <mutex>
#include <thread>

#include <glm/glm.hpp>

// Shader Input/Output
#include "shaders/shaderio.h"  // Shared between host and device

#include <nvvk/sbt_generator.hpp>
#include <nvutils/profiler.hpp>
#include "renderer_base.hpp"
#include "utils.hpp"
#include "pipeline_cache_util.hpp"
#include "scene_feature_detection.hpp"
#include "ui_busy_window.hpp"

// #DLSS
#if defined(USE_DLSS)
#include "dlss.hpp"
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
  void onSceneInvalidated(Resources& resources) override;
  void notifyDlssContentReset(Resources& resources) override;

  void updateDlssResources(VkCommandBuffer cmd, Resources& resources);
  void updateOptiXResources(VkCommandBuffer cmd, Resources& resources);
  void pushDescriptorSet(VkCommandBuffer cmd, Resources& resources, VkPipelineBindPoint bindPoint) const;
  void createPipeline(Resources& resources) override;
  void createRqPipeline(Resources& resources);
  void createRtxPipeline(Resources& resources);
  bool compileShader(Resources& resources, bool fromFile = true) override;
  // User-initiated hot reload: drop cached variants and recompile from Slang source.
  bool reloadShader(Resources& resources);
  void setBusyWindow(BusyWindow* busy) { m_busyWindow = busy; }

  // Register command line parameters
  /// Declare this subsystem's settings (command line + benchmark + MCP + persistence).
  void registerParameters(SettingsRegistry* settings);

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
  VkPhysicalDeviceRayTracingInvocationReorderPropertiesEXT m_reorderProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_EXT};

  bool m_supportSER{false};      // True when the device supports SER (Shader Execution Reordering).
  bool m_supportRayQuery{true};  // False where the driver cannot build the ray-query pipeline (see onAttach).

  // Path-traced depth when the G-buffer depth format cannot be a storage image (D32_SFLOAT on AMD):
  // the shader writes it into m_depthStaging, and the first frame copies it into the depth buffer.
  bool               m_depthViaStaging{false};
  bool               m_depthCopyable{true};  // The depth format is 32-bit float, so the staged depth can be copied in
  nvvk::RenderTarget m_depthStaging;         // R32_SFLOAT, G-buffer size
  nvvk::Buffer       m_depthCopyBuffer;      // Bridge for the color -> depth copy (no direct image copy between them)
  bool               m_useSER{true};         // Requested SER state; clamped to m_supportSER each frame.
  bool m_shadowTransmission{true};  // Shadow rays pass through transmissive surfaces (biased; see ePtShadowTransmission).
  bool m_pipelineUseSER{false};     // SER value the currently-live pipelines were built with.

  // Variant pipeline cache: avoids slow pipeline (re)compilation by reusing previously built
  // VkShaderModule and pipelines for a given VariantKey. LRU-limited (see kVariantCacheMaxEntries).
  struct VariantKey
  {
    bool wireframe = false;  // True when the shader is the wireframe build.
    bool visualize = false;  // True when the debug-visualization code is compiled in (USE_VISUALIZE).
    bool omm       = false;  // True when the ray-query entry points opt in to opacity micromaps (USE_OMM).
    bool optimal   = false;  // True when the shader is the scene-aware optimized build.
    bool dlss      = false;  // True when DLSS is active (drives USE_DLSS_SHADER: sample-loop gate).
    bool dlssGuide = false;  // True when guide-buffer capture is compiled in (USE_GUIDE_SHADER: DLSS or OptiX).
    nvvkgltf::SceneFeatureSet features{};  // only meaningful when `optimal == true`
    int dlssTransparency = 0;  // USE_DLSS_TRANSP the pipeline was specialized with (Dlss::TransparencyMode).

    // Same SPIR-V: every shader macro matches. dlssTransparency is a pipeline specialization
    // constant, not a macro, so it is left out. dlss and dlssGuide are compared in every mode (they
    // drive USE_DLSS_SHADER / USE_GUIDE_SHADER independently of optimal); the full extension feature
    // set only matters for the optimal build.
    bool sameShader(const VariantKey& o) const
    {
      return wireframe == o.wireframe && visualize == o.visualize && omm == o.omm && optimal == o.optimal
             && dlss == o.dlss && dlssGuide == o.dlssGuide && (optimal ? (features == o.features) : true);
    }
    bool operator==(const VariantKey& o) const { return sameShader(o) && dlssTransparency == o.dlssTransparency; }
  };
  VariantKey m_compiled{};  // The variant the live shader module and pipelines were built as. Guarded by m_compileMutex.

  // Variant cache entry: stores a shader module, RTX/RQ pipelines, and SBT for a given VariantKey.
  struct VariantCacheEntry
  {
    VariantKey                  key;
    VkShaderModule              shaderModule = VK_NULL_HANDLE;
    VkPipeline                  rtxPipeline  = VK_NULL_HANDLE;
    VkPipeline                  rqPipeline   = VK_NULL_HANDLE;
    bool                        useSER       = false;  // SER value the cached pipelines were built with.
    nvvk::Buffer                sbtBuffer{};           // Buffer for the SBT (Shader Binding Table)
    nvvk::SBTGenerator::Regions sbtRegions{};          // The SBT regions (raygen, miss, chit, ahit)
  };
  std::vector<VariantCacheEntry> m_variantCache;  // MRU front, LRU back
  static constexpr size_t        kVariantCacheMaxEntries = 8;

  // Saves active shader/pipeline/SBT to cache by VariantKey; switches to `newKey`.
  // On hit, restores cached handles and returns true; on miss, clears handles for rebuild.
  // LRU-evicted SBTs are freed via Resources.
  bool       swapVariant(Resources& resources, const VariantKey& newKey);
  void       destroyVariantCache(Resources& resources);
  VariantKey makeVariantKey(const Resources& resources) const;  // The variant the current settings ask for.
  // Swaps in the wanted variant if it is cached (cheap; no BusyWindow). Leaves live handles alone on a miss.
  bool tryRestoreCachedVariant(Resources& resources);

  BusyWindow* m_busyWindow{nullptr};  // Modal shown during async shader/pipeline compile.
  std::mutex  m_compileMutex;         // Guards compile metadata and live pipeline handles.
  std::thread m_compileThread;        // Joined in onDetach().

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

  // True when the user has DLSS-RR enabled
  bool isDlssEnabled() const
  {
#if defined(USE_DLSS)
    if(!m_dlss)
      return false;
    const auto s = m_dlss->state();
    return s == Dlss::State::eLoading || s == Dlss::State::eActive;
#else
    return false;
#endif
  }

  // #DLSS - Implementation of the DLSS denoiser (Ray Reconstruction).
#if defined(USE_DLSS)
  std::unique_ptr<Dlss> m_dlss;
  Dlss*                 getDlss() { return m_dlss.get(); }
  const Dlss*           getDlss() const { return m_dlss.get(); }
#endif


  // #OPTIX - Implementation of the OptiX denoiser
#if defined(USE_OPTIX_DENOISER)
  std::unique_ptr<OptiXDenoiser> m_optix;
  OptiXDenoiser*                 getOptiXDenoiser() { return m_optix.get(); }
  const OptiXDenoiser*           getOptiXDenoiser() const { return m_optix.get(); }
#endif


private:
  // What the live shader / pipeline lack for the current settings (see getPendingCompileWork()).
  struct PendingCompileWork
  {
    bool recompile       = false;  // A shader macro changed: needs another shader variant.
    bool stalePipeline   = false;  // Pipelines built with stale specialization constants (SER, DLSS transparency).
    bool missingPipeline = false;  // No pipeline for the selected technique.
    bool any() const { return recompile || stalePipeline || missingPipeline; }
  };
  PendingCompileWork getPendingCompileWork(const Resources& resources);
  bool               prepareFrame(Resources& resources);  // false: skip this frame (async build started)
  void               ensureShadersAndPipelines(Resources& resources);
  void               startAsyncCompile(Resources& resources);
  void               updateStatistics(Resources& resources);
  void               renderRayQuery(VkCommandBuffer cmd, VkExtent2D renderingSize, Resources& resources);
  void               renderRayTrace(VkCommandBuffer cmd, VkExtent2D& renderingSize, Resources& resources);
  void               denoiseDlss(VkCommandBuffer cmd, Resources& resources);
  void               setupPushConstant(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderingSize);
  // Determine if DLSS should actively denoise this frame
  bool getEffectiveDlssEnabled(const Resources& resources) const;
  // Determine if OptiX should actively denoise this frame
  bool getEffectiveOptixEnabled(const Resources& resources) const;
  // Upscale selection ID and depth from render resolution to display resolution (OptiX 2x mode)
  void upscaleSelectionAndDepth(VkCommandBuffer cmd, Resources& resources);
  // Size m_depthStaging / m_depthCopyBuffer to the G-buffer (no-op unless m_depthViaStaging).
  void updateDepthStaging(VkCommandBuffer cmd, Resources& resources);
  // Copy the path-traced depth from m_depthStaging into the G-buffer depth image.
  void cmdCopyStagedDepth(VkCommandBuffer cmd, Resources& resources);
  // Destroy the pipelines for both Ray Query and Ray Tracing
  void destroyPipelinesLocked();
  // True when the scene's BLAS carry opacity micromaps, so the shader must be built with USE_OMM.
  static bool wantOpacityMicromapShader(const Resources& resources);
  bool        m_skipVariantCache{false};  // Set during reloadShader(); bypasses swapVariant lookup.
};
