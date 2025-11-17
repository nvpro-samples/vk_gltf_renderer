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


#include <nvapp/elem_dbgprintf.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/specialization.hpp>
#include <nvutils/parameter_registry.hpp>
#include <nvgui/tooltip.hpp>

#include "renderer_pathtracer.hpp"
#include "utils.hpp"

// Pre-compiled shaders
#include "_autogen/gltf_pathtrace.slang.h"


PathTracer::PathTracer()
{
  // Default parameters for overall material
  m_pushConst.maxDepth              = 5;
  m_pushConst.frameCount            = 0;
  m_pushConst.fireflyClampThreshold = 10.;
  m_pushConst.numSamples            = 1;  // Number of samples per pixel

#if defined(USE_DLSS)
  m_dlss = std::make_unique<DlssDenoiser>();
#endif

#if defined(USE_OPTIX_DENOISER)
  m_optix = std::make_unique<OptiXDenoiser>();
#endif
}

//--------------------------------------------------------------------------------------------------
// Constructor
// Initialize the device, the frame counter and the firefly clamp threshold
void PathTracer::onAttach(Resources& resources, nvvk::ProfilerGpuTimer* profiler)
{
  ::BaseRenderer::onAttach(resources, profiler);
  m_device = resources.allocator.getDevice();

  // Create pipeline cache for faster pipeline creation
  m_pipelineCache.init(m_device, "pipeline_cache.bin");


  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext                  = &m_rtPipelineProperties;
  m_rtPipelineProperties.pNext = &m_reorderProperties;
  vkGetPhysicalDeviceProperties2(resources.allocator.getPhysicalDevice(), &prop2);

  m_supportSER =
      (bool)(m_reorderProperties.rayTracingInvocationReorderReorderingHint & VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_NV) ? 1 : 0;
  m_useSER = m_supportSER;

  // If SER is not supported, force recompiling without SER
  compileShader(resources, (m_supportSER == true) ? false : true);

  // #DLSS - Fast initialization: create GBuffers if hardware available
#if defined(USE_DLSS)
  m_dlss->init(resources);
#endif

  // #OPTIX - Create the OptiX denoiser
#if defined(USE_OPTIX_DENOISER)
  m_optix->init(resources);
#endif
}

//--------------------------------------------------------------------------------------------------
// Register command line parameters for the PathTracer
void PathTracer::registerParameters(nvutils::ParameterRegistry* paramReg)
{
  // PathTracer-specific command line parameters
  paramReg->add({"ptMaxDepth", "PathTracer: Maximum ray depth"}, &m_pushConst.maxDepth);
  paramReg->add({"ptSamples", "PathTracer: Samples per pixel"}, &m_pushConst.numSamples);
  paramReg->add({"ptFireflyClamp", "PathTracer: Firefly clamp threshold"}, &m_pushConst.fireflyClampThreshold);
  paramReg->add({"ptAperture", "PathTracer: Camera aperture"}, &m_pushConst.aperture);
  paramReg->add({"ptFocalDistance", "PathTracer: Focal distance"}, &m_pushConst.focalDistance);
  paramReg->add({"ptAutoFocus", "PathTracer: Enable auto focus"}, &m_autoFocus);
  paramReg->add({"ptTechnique", "PathTracer: Rendering technique [RayQuery:0, RayTracing:1]"}, (int*)&m_renderTechnique);
  paramReg->add({"ptAdaptiveSampling", "PathTracer: Enable adaptive sampling"}, &m_adaptiveSampling);
  paramReg->add({"ptPerformanceTarget", "PathTracer: Performance target [Interactive:0, Balanced:1, Quality:2, MaxQuality:3]"},
                (int*)&m_performanceTarget);
#if defined(USE_DLSS)
  m_dlss->registerParameters(paramReg);
#endif

#if defined(USE_OPTIX_DENOISER)
  m_optix->registerParameters(paramReg);
#endif
}

//--------------------------------------------------------------------------------------------------
// Destroy the resources
void PathTracer::onDetach(Resources& resources)
{
  resources.allocator.destroyBuffer(m_sbtBuffer);

#if defined(USE_DLSS)
  m_dlss->deinit(resources);
#endif

#if defined(USE_OPTIX_DENOISER)
  m_optix->deinit(resources);
#endif
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyShaderModule(m_device, m_shaderModule, nullptr);
  vkDestroyPipeline(m_device, m_rtxPipeline, nullptr);
  vkDestroyPipeline(m_device, m_rqPipeline, nullptr);
  m_pipelineCache.deinit();
}

//--------------------------------------------------------------------------------------------------
// Resize the G-Buffer and the renderers
void PathTracer::onResize(VkCommandBuffer cmd, const VkExtent2D& size, Resources& resources)
{
  updateDlssResources(cmd, resources);
  updateOptiXResources(cmd, resources);
}

void PathTracer::updateDlssResources(VkCommandBuffer cmd, Resources& resources)
{
#if defined(USE_DLSS)
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  VkExtent2D size = resources.gBuffers.getSize();
  m_dlss->updateSize(cmd, size);
  m_dlss->setResources();
  m_dlss->setResource(DlssRayReconstruction::ResourceType::eColorOut, resources.gBuffers.getColorImage(Resources::eImgRendered),
                      resources.gBuffers.getColorImageView(Resources::eImgRendered),
                      resources.gBuffers.getColorFormat(Resources::eImgRendered));
#endif
}

void PathTracer::updateOptiXResources(VkCommandBuffer cmd, Resources& resources)
{
#if defined(USE_OPTIX_DENOISER)
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  VkExtent2D size = resources.gBuffers.getSize();
  m_optix->updateSize(cmd, size);
#endif
}

bool PathTracer::onUIRender(Resources& resources)
{
  // Setting the aperture max slider value, based on the scene size
  float sceneRadius = resources.scene.getSceneBounds().radius();
  float scaleFactor = std::log(sceneRadius);
  scaleFactor       = std::max(scaleFactor, 0.0f);   // Prevent negative values when the scene is small
  float apertureMax = 0.0001f + scaleFactor * 5.0f;  // Minimum max aperture is 0.0001

  namespace PE = nvgui::PropertyEditor;
  bool changed = false;
  if(PE::begin())
  {
    // Rendering technique selector
    const char* techniques[] = {"Ray Query", "Ray Tracing"};
    int         current      = static_cast<int>(m_renderTechnique);
    if(PE::Combo("Rendering Technique", &current, techniques, IM_ARRAYSIZE(techniques)))
    {
      m_renderTechnique = static_cast<RenderTechnique>(current);
      changed           = true;
    }
    nvgui::tooltip(
        "Both Ray Query and Ray Tracing use hardware accelerated ray tracing."
        "Ray Query uses a compute shader interface, while Ray Tracing uses the dedicated RTX pipeline.");

    if(m_supportSER && m_renderTechnique == RenderTechnique::RayTracing)
    {
      bool oldUseSER = m_useSER;
      changed |= PE::Checkbox("Use SER", &m_useSER, "Use shader execution reorder");

      // Recreate RTX pipeline if SER setting changed
      if(oldUseSER != m_useSER)
      {
        vkDeviceWaitIdle(m_device);
        vkDestroyPipeline(m_device, m_rtxPipeline, nullptr);
        m_rtxPipeline = VK_NULL_HANDLE;
      }
    }

    changed |= PE::SliderInt("Max Depth", &m_pushConst.maxDepth, 0, 20, "%d", 0, "Maximum number of bounces");
    changed |= PE::SliderFloat("FireFly Clamp", &m_pushConst.fireflyClampThreshold, 0.0f, 10.0f, "%.2f", 0,
                               "Clamp threshold for fireflies");
    PE::end();
  }

  // Manual sampling controls
  if(PE::begin())
  {
    PE::SliderInt("Max Iterations", &resources.settings.maxFrames, 0, 10000, "%d", 0, "Maximum number of iterations");
    ImGui::BeginDisabled(m_adaptiveSampling || isDlssEnabled());
    PE::SliderInt("Samples", &m_pushConst.numSamples, MIN_SAMPLES_PER_PIXEL, MAX_SAMPLES_PER_PIXEL, "%d", 0,
                  "Number of samples per pixel");
    ImGui::EndDisabled();
    if(isDlssEnabled())
    {
      ImGui::SameLine();
      ImGui::TextDisabled("(DLSS: 1 spp)");
    }

    // Adaptive sampling controls
    ImGui::BeginDisabled(isDlssEnabled());
    PE::Checkbox("Auto SPP", &m_adaptiveSampling, "Automatically adjust samples per pixel based on performance target");
    ImGui::EndDisabled();
    if(isDlssEnabled())
    {
      ImGui::SameLine();
      ImGui::TextDisabled("(DLSS disabled)");
    }
    if(m_adaptiveSampling)
    {
      ImGui::SameLine();
      ImGui::TextDisabled("(Auto: %d spp)", m_pushConst.numSamples);

      // Performance target selection
      const char* targets[] = {"Interactive (60 FPS)", "Balanced (30 FPS)", "Quality (15 FPS)", "Max Quality (10 FPS)"};
      int         currentTarget = static_cast<int>(m_performanceTarget);
      if(PE::Combo("Performance Target", &currentTarget, targets, IM_ARRAYSIZE(targets)))
      {
        m_performanceTarget = static_cast<PerformanceTarget>(currentTarget);
      }
    }
    // Performance info - always visible
    ImGui::TextDisabled("Samples: %d/%d (%.1fx)", m_totalSamplesAccumulated, resources.frameCount + 1,
                        m_totalSamplesAccumulated / float(resources.frameCount + 1));

    ImGui::TextDisabled("Throughput: %.2f MSPP/s", m_throughputRollingAvg.getAverage());
    nvgui::tooltip("Mega-sample-pixels per second (rolling average over last %zu frames)", m_throughputRollingAvg.SAMPLE_COUNT);

    PE::end();
  }

  // Camera controls
  if(PE::begin())
  {
    changed |= PE::SliderFloat("Aperture", &m_pushConst.aperture, 0.0f, apertureMax, "%5.9f",
                               ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat, "Out-of-focus effect");
    changed |= PE::Checkbox("Auto Focus", &m_autoFocus, "Use interest position");
    ImGui::BeginDisabled(m_autoFocus);
    changed |= PE::DragFloat("Focal Distance", &m_pushConst.focalDistance, 100.0f, 0.0f, 1000000.0f, "%5.9f",
                             ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat, "Distance to focal point");
    m_pushConst.focalDistance = std::max(0.000000001f, m_pushConst.focalDistance);
    ImGui::EndDisabled();
    PE::end();
  }

  // Infinite plane
  if(PE::begin())
  {
    changed |= PE::Checkbox("Infinite Plane", (bool*)&resources.settings.useInfinitePlane);
    if(resources.settings.useInfinitePlane)
    {
      const float extentY = resources.scene.valid() ? resources.scene.getSceneBounds().extents().y : 10.0f;
      if(PE::treeNode("Infinite Plane Settings"))
      {
        changed |= PE::SliderFloat("Height", &resources.settings.infinitePlaneDistance, -extentY, extentY, "%5.9f",
                                   ImGuiSliderFlags_NoRoundToFormat, "Distance to infinite plane");
        changed |= PE::ColorEdit3("Color", glm::value_ptr(resources.settings.infinitePlaneBaseColor));
        changed |= PE::SliderFloat("Metallic", &resources.settings.infinitePlaneMetallic, 0.0f, 1.0f);
        changed |= PE::SliderFloat("Roughness", &resources.settings.infinitePlaneRoughness, 0.0f, 1.0f);
        PE::treePop();
      }
    }

    PE::end();
  }

  if(ImGui::CollapsingHeader("Denoisers", ImGuiTreeNodeFlags_DefaultOpen))
  {
// DLSS section
#if defined(USE_DLSS)
    bool oldTransp = m_dlss->useDlssTransparency();
    changed |= m_dlss->onUi(resources);
    if(oldTransp != m_dlss->useDlssTransparency())
    {
      // Need to recompile the shader using the specialization constant
      vkDeviceWaitIdle(m_device);
      vkDestroyPipeline(m_device, m_rtxPipeline, nullptr);
      m_rtxPipeline = VK_NULL_HANDLE;
      vkDestroyPipeline(m_device, m_rqPipeline, nullptr);
      m_rqPipeline = VK_NULL_HANDLE;
    }
#else
    ImGui::TextDisabled("DLSS is not enabled.");
    nvsamples::HelpMarker("Define USE_DLSS in CMake to enable DLSS support.");
#endif

#if defined(USE_OPTIX_DENOISER)
    changed |= m_optix->onUi(resources);
#else
    ImGui::TextDisabled("OptiX Denoiser is not enabled.");
    nvsamples::HelpMarker("Define USE_OPTIX_DENOISER in CMake to enable OptiX denoiser support.");
#endif
  }
  return changed;
}

//--------------------------------------------------------------------------------------------------
// Render the scene
void PathTracer::onRender(VkCommandBuffer cmd, Resources& resources)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

  // Reset display buffer to rendered on first frame
  if((resources.frameCount == 0) && (resources.settings.displayBuffer == DisplayBuffer::eOptixDenoised))
  {
    resources.settings.displayBuffer = DisplayBuffer::eRendered;
  }

  // Handle adaptive sampling (SPP adjustment)
  updateAdaptiveSampling(resources);

  // Setting up the push constant
  setupPushConstant(cmd, resources);

  // Make sure buffer is ready to be used
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

  // Finding the rendering size
  VkExtent2D renderingSize = resources.gBuffers.getSize();
#if USE_DLSS
  // When DLSS is effectively enabled, use DLSS render size
  if(getEffectiveDlssEnabled(resources))
  {
    renderingSize = m_dlss->getRenderSize();
  }
#endif

  // Tracing the rays: Ray Query or Ray Tracing
  if(m_renderTechnique == RenderTechnique::RayQuery)
  {
    renderRayQuery(cmd, renderingSize, resources);
  }
  else if(m_renderTechnique == RenderTechnique::RayTracing)
  {
    renderRayTrace(cmd, renderingSize, resources);
  }

  // Making sure the rendered image is ready to be used by tonemapper
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

#if defined(USE_DLSS)
  // If DLSS is effectively enabled for this frame, perform denoising
  if(getEffectiveDlssEnabled(resources))
  {
    denoiseDlss(cmd, resources);
  }
#endif

#if defined(USE_OPTIX_DENOISER)
  // Update OptiX auto-denoiser (only when effectively enabled)
  if(getEffectiveOptixEnabled(resources))
  {
    m_optix->updateDenoiser(resources);
  }
#endif

  updateStatistics(resources);
}

//--------------------------------------------------------------------------------------------------
// Push the descriptor set
// This is making sure our shader has the latest TLAS, and the latest output images
void PathTracer::pushDescriptorSet(VkCommandBuffer cmd, Resources& resources, VkPipelineBindPoint bindPoint) const
{
  nvvk::WriteSetContainer write{};
  write.append(resources.descriptorBinding[1].getWriteSet(shaderio::BindingPoints::eTlas), resources.sceneRtx.tlas());

  // Normal rendering: basic output images
  std::vector<VkDescriptorImageInfo> outputImages = {
      resources.gBuffers.getDescriptorImageInfo(Resources::eImgRendered),   // eResultImage
      resources.gBuffers.getDescriptorImageInfo(Resources::eImgSelection),  // eSelectImage
  };

#if defined(USE_DLSS)
  if(getEffectiveDlssEnabled(resources))
  {
    // With DLSS active, we have 7 output images
    using namespace shaderio;
    outputImages.resize(7);
    outputImages[eResultImage]         = m_dlss->getGBuffers().getDescriptorImageInfo(eResultImage);
    outputImages[eSelectImage]         = m_dlss->getGBuffers().getDescriptorImageInfo(eSelectImage);
    outputImages[eDlssAlbedo]          = m_dlss->getGBuffers().getDescriptorImageInfo(eDlssAlbedo);
    outputImages[eDlssSpecAlbedo]      = m_dlss->getGBuffers().getDescriptorImageInfo(eDlssSpecAlbedo);
    outputImages[eDlssNormalRoughness] = m_dlss->getGBuffers().getDescriptorImageInfo(eDlssNormalRoughness);
    outputImages[eDlssMotion]          = m_dlss->getGBuffers().getDescriptorImageInfo(eDlssMotion);
    outputImages[eDlssDepth]           = m_dlss->getGBuffers().getDescriptorImageInfo(eDlssDepth);
  }
#endif

#if defined(USE_OPTIX_DENOISER)
  if(getEffectiveOptixEnabled(resources))
  {
    // With OptiX active, add the albedo/normal output image
    using namespace shaderio;
    outputImages.resize(3);  // Adding the extra buffer to store albedo+normal
    outputImages[eOptixAlbedoNormal] = m_optix->getDescriptorImageInfo(OptiXDenoiser::eGBufferAlbedoNormal);
  }
#endif


  VkWriteDescriptorSet allTextures = resources.descriptorBinding[1].getWriteSet(shaderio::BindingPoints::eOutImages);
  allTextures.descriptorCount      = uint32_t(outputImages.size());
  write.append(allTextures, outputImages.data());
  vkCmdPushDescriptorSetKHR(cmd, bindPoint, m_pipelineLayout, 1, write.size(), write.data());
}

//--------------------------------------------------------------------------------------------------
// Create the pipeline
void PathTracer::createPipeline(Resources& resources)
{
  SCOPED_TIMER(__FUNCTION__);
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts{resources.descriptorSetLayout[0], resources.descriptorSetLayout[1],
                                                          resources.hdrIbl.getDescriptorSetLayout()};

  // Creating the pipeline layout
  VkPushConstantRange        pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PathtracePushConstant)};
  VkPipelineLayoutCreateInfo plCreateInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = uint32_t(descriptorSetLayouts.size()),
      .pSetLayouts            = descriptorSetLayouts.data(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstant,
  };
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);
}

//--------------------------------------------------------------------------------------------------
// Create the compute pipeline
void PathTracer::createRqPipeline(Resources& resources)
{
  SCOPED_TIMER(__FUNCTION__);

  nvvk::Specialization specialization;
  specialization.add(0, m_useSER ? 1 : 0);  // USE_SER
#if USE_DLSS
  specialization.add(1, m_dlss->useDlssTransparency() ? 1 : 0);  // USE_DLSS_TRANSP
#endif

  VkPipelineShaderStageCreateInfo shaderStage{
      .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
      .module              = m_shaderModule,
      .pName               = "computeMain",
      .pSpecializationInfo = specialization.getSpecializationInfo(),
  };

  VkComputePipelineCreateInfo cpCreateInfo{
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage  = shaderStage,
      .layout = m_pipelineLayout,
  };

  // NOTE: if the creation is slow, disable the validation layers for faster creation (--vvl 0)
  NVVK_CHECK(vkCreateComputePipelines(m_device, m_pipelineCache.getCache(), 1, &cpCreateInfo, nullptr, &m_rqPipeline));
  NVVK_DBG_NAME(m_rqPipeline);
}


//--------------------------------------------------------------------------------------------------
// Create the RTX pipeline
void PathTracer::createRtxPipeline(Resources& resources)
{
  SCOPED_TIMER(__FUNCTION__);
  // Creating all shaders
  enum ShaderStages
  {
    eRaygen,
    eMiss,
    eShadowMiss,
    eClosestHit,
    eShadowClosestHit,
    eAnyHit,
    eShadowAnyHit,
    eShaderGroupCount
  };

  // RTX Pipeline
  std::array<VkPipelineShaderStageCreateInfo, 7> stages{};
  for(auto& stage : stages)
  {
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.module = m_shaderModule;
  }
  stages[eRaygen].pName = "rgenMain";
  stages[eRaygen].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

  stages[eMiss].pName = "rmissMain";
  stages[eMiss].stage = VK_SHADER_STAGE_MISS_BIT_KHR;

  stages[eClosestHit].pName = "rchitMain";
  stages[eClosestHit].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

  stages[eAnyHit].pName = "rahitMain";
  stages[eAnyHit].stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

  //Shadow
  stages[eShadowMiss].pName = "rmissShadow";
  stages[eShadowMiss].stage = VK_SHADER_STAGE_MISS_BIT_KHR;

  stages[eShadowClosestHit].pName = "rchitShadow";
  stages[eShadowClosestHit].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

  stages[eShadowAnyHit].pName = "rahitShadow";
  stages[eShadowAnyHit].stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{
      .sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
      .generalShader      = VK_SHADER_UNUSED_KHR,
      .closestHitShader   = VK_SHADER_UNUSED_KHR,
      .anyHitShader       = VK_SHADER_UNUSED_KHR,
      .intersectionShader = VK_SHADER_UNUSED_KHR,
  };

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;
  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  shader_groups.push_back(group);

  // Miss-0
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  shader_groups.push_back(group);

  // Shadow Miss-1
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eShadowMiss;
  shader_groups.push_back(group);

  // Hit Group-0
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  group.anyHitShader     = eAnyHit;
  shader_groups.push_back(group);

  // Hit Group-1
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eShadowClosestHit;
  group.anyHitShader     = eShadowAnyHit;
  shader_groups.push_back(group);

  // Shader Execution Reorder (SER)
  nvvk::Specialization specialization;
  specialization.add(0, m_useSER ? 1 : 0);  // USE_SER
#if USE_DLSS
  specialization.add(1, m_dlss->useDlssTransparency() ? 1 : 0);  // USE_DLSS_TRANSP
#endif
  stages[eRaygen].pSpecializationInfo = specialization.getSpecializationInfo();


  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rtPipelineCreateInfo{
      .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .stageCount                   = static_cast<uint32_t>(stages.size()),  // Stages are shaders
      .pStages                      = stages.data(),
      .groupCount                   = static_cast<uint32_t>(shader_groups.size()),
      .pGroups                      = shader_groups.data(),
      .maxPipelineRayRecursionDepth = 2,  // Ray depth
      .layout                       = m_pipelineLayout,
  };
  vkDestroyPipeline(m_device, m_rtxPipeline, nullptr);


  // NOTE: if the creation is slow, disable the validation layers for faster creation (--vvl 0)
  NVVK_CHECK(vkCreateRayTracingPipelinesKHR(m_device, {}, m_pipelineCache.getCache(), 1, &rtPipelineCreateInfo, nullptr, &m_rtxPipeline));
  NVVK_DBG_NAME(m_rtxPipeline);

  // Create the Shading Binding Table
  {
    resources.allocator.destroyBuffer(m_sbtBuffer);

    // Shader Binding Table (SBT) setup
    nvvk::SBTGenerator sbtGenerator;
    sbtGenerator.init(m_device, m_rtPipelineProperties);

    // Prepare SBT data from ray pipeline
    size_t bufferSize = sbtGenerator.calculateSBTBufferSize(m_rtxPipeline, rtPipelineCreateInfo);

    // Create SBT buffer using the size from above
    NVVK_CHECK(resources.allocator.createBuffer(
        m_sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT, sbtGenerator.getBufferAlignment()));
    NVVK_DBG_NAME(m_sbtBuffer.buffer);

    // Pass the manual mapped pointer to fill the SBT data
    NVVK_CHECK(sbtGenerator.populateSBTBuffer(m_sbtBuffer.address, bufferSize, m_sbtBuffer.mapping));

    // Retrieve the regions, which are using addresses based on the m_sbtBuffer.address
    m_sbtRegions = sbtGenerator.getSBTRegions();

    sbtGenerator.deinit();
  }
}


//--------------------------------------------------------------------------------------------------
// Compile the shader
void PathTracer::compileShader(Resources& resources, bool fromFile)
{
  SCOPED_TIMER(__FUNCTION__);

  VkPushConstantRange pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PathtracePushConstant)};

  std::vector<VkDescriptorSetLayout> descriptorSetLayouts{resources.descriptorSetLayout[0], resources.descriptorSetLayout[1],
                                                          resources.hdrIbl.getDescriptorSetLayout()};

  VkShaderCreateInfoEXT shaderInfo{
      .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
      .stage                  = VK_SHADER_STAGE_COMPUTE_BIT,
      .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
      .codeSize               = gltf_pathtrace_slang_sizeInBytes,
      .pCode                  = gltf_pathtrace_slang,
      .pName                  = "computeMain",
      .setLayoutCount         = uint32_t(descriptorSetLayouts.size()),
      .pSetLayouts            = descriptorSetLayouts.data(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstant,
  };

  // Compile from shader file if requested, used when reloading the shader
  if(fromFile)
  {
    SCOPED_TIMER("Slang compile from file");

    resources.slangCompiler.clearMacros();
    std::vector<std::pair<std::string, std::string>> macros = {
        {"AVAILABLE_SER", std::to_string(m_supportSER)},
        // {"USE_DLSS_TRANSP", std::to_string(m_dlss->useDlssTransparency())},
    };
    for(const auto& [k, v] : macros)
      resources.slangCompiler.addMacro({k.c_str(), v.c_str()});

    if(resources.slangCompiler.compileFile("gltf_pathtrace.slang"))
    {
      shaderInfo.codeSize = resources.slangCompiler.getSpirvSize();
      shaderInfo.pCode    = resources.slangCompiler.getSpirv();
    }
    else
    {
      LOGW("Error compiling gltf_pathtrace.slang\n");
    }
  }

  // Create a shader module
  {
    SCOPED_TIMER("Create Shader Module");
    vkDestroyShaderModule(m_device, m_shaderModule, nullptr);

    VkShaderModuleCreateInfo moduleInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shaderInfo.codeSize,
        .pCode    = static_cast<const uint32_t*>(shaderInfo.pCode),
    };
    NVVK_CHECK(vkCreateShaderModule(m_device, &moduleInfo, nullptr, &m_shaderModule));
    NVVK_DBG_NAME(m_shaderModule);
  }

  // Destroy pipeline since there is a new shader
  vkDestroyPipeline(m_device, m_rtxPipeline, nullptr);
  m_rtxPipeline = VK_NULL_HANDLE;
  vkDestroyPipeline(m_device, m_rqPipeline, nullptr);
  m_rqPipeline = VK_NULL_HANDLE;
}


//--------------------------------------------------------------------------------------------------
// Update adaptive sampling based on frame timing
void PathTracer::updateAdaptiveSampling(Resources& resources)
{
  // Don't update adaptive sampling if DLSS is enabled
  if(isDlssEnabled())
    return;

  if(!m_adaptiveSampling || !m_profilerTimeline)
    return;

  // Reset samples when frame count resets to 0 (scene change, etc.)
  if(resources.frameCount == 0)
  {
    m_pushConst.numSamples = MIN_SAMPLES_PER_PIXEL;
    return;
  }

  // Don't adjust on the first few frames to allow for stabilization
  if(resources.frameCount < 5)
    return;

  // Get timing information for the path tracing section
  nvutils::ProfilerTimeline::TimerInfo timerInfo;
  std::string                          apiName;

  // Try both possible timer names based on rendering technique
  const char* timerName = (m_renderTechnique == RenderTechnique::RayQuery) ? "Path Trace (RQ)" : "Path Trace (RTX)";

  if(m_profilerTimeline->getFrameTimerInfo(timerName, timerInfo, apiName))
  {
    // Convert from microseconds to milliseconds
    double currentFrameTimeMs = timerInfo.gpu.last / 1000.0;

    // Adjust samples based on performance target
    double targetTime = getTargetFrameTimeMs();
    if(currentFrameTimeMs < targetTime * 0.8 && m_pushConst.numSamples < MAX_SAMPLES_PER_PIXEL)
    {
      // We have headroom, increase samples
      m_pushConst.numSamples++;
    }
    else if(currentFrameTimeMs > targetTime * 1.1 && m_pushConst.numSamples > MIN_SAMPLES_PER_PIXEL)
    {
      // We're over budget, decrease samples
      m_pushConst.numSamples--;
    }

    // Clamp to valid range
    m_pushConst.numSamples = std::clamp(m_pushConst.numSamples, MIN_SAMPLES_PER_PIXEL, MAX_SAMPLES_PER_PIXEL);
  }
}


void PathTracer::updateStatistics(Resources& resources)
{


  // Update rolling average throughput calculation using wall-clock time
  {
    // Time elapsed for this frame (wall-clock time from user perspective)
    float wallClockFrameTime = ImGui::GetIO().DeltaTime;

    // Total number of pixels in the image
    VkExtent2D imageSize   = resources.gBuffers.getSize();
    uint64_t   totalPixels = static_cast<uint64_t>(imageSize.width) * static_cast<uint64_t>(imageSize.height);

    // Calculate mega-sample-pixels per second of wall-clock time
    // This tells the user how much rendering work is being done per real-world second
    const float MEGA_SCALE_FACTOR = 1000000.0f;  // Convert to mega-sample-pixels
    float       megaSamplePixelsPerSecond =
        (static_cast<float>(m_pushConst.numSamples) * static_cast<float>(totalPixels) / MEGA_SCALE_FACTOR) / wallClockFrameTime;

    // Update rolling average with wall-clock throughput for this frame
    m_throughputRollingAvg.addValue(megaSamplePixelsPerSecond);
  }

  // Track total samples accumulated
  m_totalSamplesAccumulated += m_pushConst.numSamples;
}

void PathTracer::renderRayQuery(VkCommandBuffer cmd, VkExtent2D renderingSize, Resources& resources)
{
  auto timerSection = m_profiler->cmdFrameSection(cmd, "Path Trace (RQ)");

  // Create pipeline if it doesn't exist
  if(m_rqPipeline == VK_NULL_HANDLE)
  {
    createRqPipeline(resources);
  }

  // Bind the shader to use
  VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
  //vkCmdBindShadersEXT(cmd, 1, &stage, &m_shader);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rqPipeline);


  // Bind the descriptor set: TLAS, output image, textures, etc. (Set: 0)
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &resources.descriptorSet, 0, nullptr);

  // Set the Descriptor for HDR (Set: 2)
  VkDescriptorSet hdrDescSet = resources.hdrIbl.getDescriptorSet();
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 2, 1, &hdrDescSet, 0, nullptr);

  pushDescriptorSet(cmd, resources, VK_PIPELINE_BIND_POINT_COMPUTE);

  // Dispatch the compute shader
  VkExtent2D numGroups = nvvk::getGroupCounts(renderingSize, WORKGROUP_SIZE);
  vkCmdDispatch(cmd, numGroups.width, numGroups.height, 1);
}

void PathTracer::renderRayTrace(VkCommandBuffer cmd, VkExtent2D& renderingSize, Resources& resources)
{
  auto timerSection = m_profiler->cmdFrameSection(cmd, "Path Trace (RTX)");

  // Create pipeline if it doesn't exist
  if(m_rtxPipeline == VK_NULL_HANDLE)
  {
    createRtxPipeline(resources);
  }

  // Bind the ray tracing pipeline
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtxPipeline);

  // Bind the descriptor set: TLAS, output image, textures, etc. (Set: 0)
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 0, 1, &resources.descriptorSet, 0, nullptr);

  // Set the Descriptor for HDR (Set: 2)
  VkDescriptorSet hdrDescSet = resources.hdrIbl.getDescriptorSet();
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 2, 1, &hdrDescSet, 0, nullptr);

  pushDescriptorSet(cmd, resources, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR);


  vkCmdTraceRaysKHR(cmd, &m_sbtRegions.raygen, &m_sbtRegions.miss, &m_sbtRegions.hit, &m_sbtRegions.callable,
                    renderingSize.width, renderingSize.height, 1);
}

void PathTracer::denoiseDlss(VkCommandBuffer cmd, Resources& resources)
{
#if defined(USE_DLSS)
  auto timerSection = m_profiler->cmdFrameSection(cmd, "DLSS");

  // #DLSS - Denoising
  const glm::mat4& view   = resources.cameraManip->getViewMatrix();
  const glm::mat4& proj   = resources.cameraManip->getPerspectiveMatrix();
  glm::vec2        jitter = m_pushConst.jitter;

  m_dlss->denoise(cmd, jitter, view, proj, false);

  // Memory barrier to ensure DLSS operations are complete before blit operations
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT,
                         VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

  {
    // Blit the selection image from the DLSS GBuffer (different resolution) to the Renderer GBuffer Selection
    VkOffset3D  minCorner = {0, 0, 0};
    VkImageBlit blitRegions{
        .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
        .srcOffsets = {minCorner, {int(m_dlss->getGBuffers().getSize().width), int(m_dlss->getGBuffers().getSize().height), 1}},
        .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
        .dstOffsets = {minCorner, {int(resources.gBuffers.getSize().width), int(resources.gBuffers.getSize().height), 1}},
    };
    vkCmdBlitImage(cmd, m_dlss->getGBuffers().getColorImage(shaderio::OutputImage::eSelectImage),
                   VK_IMAGE_LAYOUT_GENERAL, resources.gBuffers.getColorImage(Resources::eImgSelection),
                   VK_IMAGE_LAYOUT_GENERAL, 1, &blitRegions, VK_FILTER_LINEAR);

    // Ensure the blit operation completes before any subsequent reads from this image
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
  }
#endif
}

void PathTracer::setupPushConstant(VkCommandBuffer cmd, Resources& resources)
{
  // Current frame count, can be overridden by DLSS
  int frameCount = resources.frameCount;

  // Handle frame reset detection (needed for both adaptive and non-adaptive modes)
  if(resources.frameCount == 0)
  {
    m_totalSamplesAccumulated = 0;  // Reset sample counter when scene/camera changes
  }

  // Adjust focal distance if auto-focus is enabled
  if(m_autoFocus)
  {
    m_pushConst.focalDistance = glm::length(resources.cameraManip->getEye() - resources.cameraManip->getCenter());
  }

#if defined(USE_DLSS)
  // Lazy initialize DLSS NGX on first use (2-5 second delay occurs here, once)
  static uint32_t haltonIndex = 0;

  // Set useDlss based on EFFECTIVE state (false when both enabled and frameCount > 0)
  m_pushConst.useDlss = getEffectiveDlssEnabled(resources);

  if(m_pushConst.useDlss)
  {
    // When DLSS is enabled, force numSamples to 1
    m_pushConst.numSamples = 1;
    frameCount             = ++haltonIndex;  // Override frame count with Halton index

    // Lazy NGX initialization (2-5s, once) OR size update triggers resource setup
    if(m_dlss->tryInitializeNGX(resources) || m_dlss->needsSizeUpdate())
      updateDlssResources(cmd, resources);
  }
  m_pushConst.jitter = shaderio::dlssJitter(frameCount);
#endif

#if defined(USE_OPTIX_DENOISER)
  // Set useOptixDenoiser based on EFFECTIVE state (false when both enabled and frameCount == 0)
  m_pushConst.useOptixDenoiser = getEffectiveOptixEnabled(resources);
#endif
  static int lastRenderedObject = -1;
  m_pushConst.renderSelection   = resources.selectedObject != lastRenderedObject || resources.frameCount == 0;
  lastRenderedObject            = resources.selectedObject;
  m_pushConst.frameCount        = frameCount;
  m_pushConst.totalSamples      = m_totalSamplesAccumulated;
  m_pushConst.frameInfo         = (shaderio::SceneFrameInfo*)resources.bFrameInfo.address;
  m_pushConst.skyParams         = (shaderio::SkyPhysicalParameters*)resources.bSkyParams.address;
  m_pushConst.gltfScene         = (shaderio::GltfScene*)resources.sceneVk.sceneDesc().address;
  m_pushConst.mouseCoord        = nvapp::ElementDbgPrintf::getMouseCoord();  // Use for debugging: printf in shader
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PathtracePushConstant), &m_pushConst);
}

//--------------------------------------------------------------------------------------------------
// Determine if DLSS should actively denoise this frame
// - if DLSS is enabled, return true
// - if Optix is also enabled, only return true on frame 0
bool PathTracer::getEffectiveDlssEnabled(const Resources& resources) const
{
#if defined(USE_DLSS)
  bool dlssEnabled = m_dlss->isEnabled();
#if defined(USE_OPTIX_DENOISER)
  bool optixEnabled = m_optix->isEnabled();

  // When both enabled, DLSS only runs on frame 0
  if(dlssEnabled && optixEnabled)
  {
    return resources.frameCount == 0;
  }
#endif

  // Normal behavior when OptiX is off
  return dlssEnabled;
#else
  return false;
#endif
}

//--------------------------------------------------------------------------------------------------
// Determine if OptiX should actively denoise this frame
bool PathTracer::getEffectiveOptixEnabled(const Resources& resources) const
{
#if defined(USE_OPTIX_DENOISER)
  bool optixEnabled = m_optix->isEnabled();
#if defined(USE_DLSS)
  bool dlssEnabled = m_dlss->isEnabled();

  // When both enabled, OptiX only runs on frame 1+
  if(dlssEnabled && optixEnabled)
  {
    return resources.frameCount > 0;
  }
#endif

  // Normal behavior when DLSS is off
  return optixEnabled;
#else
  return false;
#endif
}
