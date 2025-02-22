/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2024 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


// Purpose: Pathtracer renderer implementation
#include <iostream>

// nvpro-core
#include "nvh/cameramanipulator.hpp"
#include "nvh/timesampler.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/specialization.hpp"
#include "nvvkhl/element_dbgprintf.hpp"
#include "nvvkhl/shaders/dh_comp.h"
#include "nvvkhl/shaders/dh_tonemap.h"

// Shared information between device and host
#include "shaders/dh_bindings.h"

// Local to application
#include "silhouette.hpp"
#include "atrous_denoiser.hpp"
#include "renderer.hpp"

extern std::shared_ptr<nvvkhl::ElementDbgPrintf> g_elemDebugPrintf;

namespace PE = ImGuiH::PropertyEditor;

#include "_autogen/pathtrace.comp.glsl.h"
#include "_autogen/pathtrace.rchit.glsl.h"
#include "_autogen/pathtrace.rgen.glsl.h"
#include "_autogen/pathtrace.rmiss.glsl.h"
#include "_autogen/pathtrace.rahit.glsl.h"
#include "_autogen/shadow.rmiss.glsl.h"
#include "_autogen/shadow.rahit.glsl.h"
#include "_autogen/shadow.rchit.glsl.h"
#include "nvvk/shaders_vk.hpp"
#include "collapsing_header_manager.h"

namespace gltfr {
extern bool g_forceExternalShaders;

// Settings for the pathtracer
PathtraceSettings g_pathtraceSettings;


// This shows path tracing using ray tracing
class RendererPathtracer : public Renderer
{
public:
  RendererPathtracer() = default;
  ~RendererPathtracer() { deinit(); };

  // Implementing the renderer interface
  bool init(Resources& res, Scene& scene) override;
  void deinit(Resources& /*res*/) override { deinit(); }
  void render(VkCommandBuffer cmd, Resources& res, Scene& scene, Settings& settings, nvvk::ProfilerVK& profiler) override;
  bool                  onUI() override;
  void                  handleChange(Resources& res, Scene& scene) override;
  VkDescriptorImageInfo getOutputImage() const override
  {
    return m_gBuffers->getDescriptorImageInfo(GBufferType::eRgbResult);
  }

  bool reloadShaders(Resources& res, Scene& /*scene*/) override;

private:
  void createRtxPipeline(Resources& res, Scene& scene);
  void createIndirectPipeline(Resources& res, Scene& scene);
  bool initShaders(Resources& res, bool reload);
  void createRtxSet();
  void writeRtxSet(Scene& scene);
  void deinit();
  void createGBuffer(Resources& res);

  VkDevice                   m_device{VK_NULL_HANDLE};
  DH::PushConstantPathtracer m_pushConst{};

  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtxSet{};        // Descriptor set
  std::unique_ptr<nvvk::SBTWrapper>             m_sbt{};           // Shading binding table wrapper
  std::unique_ptr<nvvkhl::PipelineContainer>    m_rtxPipe{};       // Raytracing pipeline
  std::unique_ptr<nvvkhl::PipelineContainer>    m_indirectPipe{};  // Raytracing pipeline
  std::unique_ptr<nvvkhl::GBuffer>              m_gBuffers{};      // G-Buffers: RGBA32F
  std::unique_ptr<nvvk::DebugUtil>              m_dutil{};
  std::unique_ptr<Silhouette>                   m_silhouette{};
  std::unique_ptr<AtrousDenoiser>               m_denoiser{};

  nvh::Bbox m_sceneBBox{};

  enum GBufferType
  {
    eRgbLinear,    // Result from path-tracing
    eRgbResult,    // Final result
    eSilhouette,   // Buffer to store object ID for silhouette
    eNormalDepth,  // Denoise - Normal
    eTempResult,   // Denoise - Temporary result
  };

  std::vector<VkFormat> m_gbufferFormats = {
      VK_FORMAT_R32G32B32A32_SFLOAT,  // Accumulated result of path tracing
      VK_FORMAT_R16G16B16A16_SFLOAT,  // Final result - After tonemap
      VK_FORMAT_R8_UNORM,             // For selection, silhouette
      VK_FORMAT_R16G16B16A16_SFLOAT,  // Normal / Depth (Normal in RGB, Depth in A) used by denoiser
      VK_FORMAT_R16G16B16A16_SFLOAT,  // Temp result (Denoiser / Ping-pong for multiple passes)
  };

  // Creating all shaders
  enum ShaderStages
  {
    eRaygen,
    eMiss,
    eShadowMiss,
    eClosestHit,
    eAnyHit,
    eShadowCH,
    eShadowAH,
    eIndirect,
    eShaderGroupCount
  };
  std::vector<shaderc::SpvCompilationResult>    m_spvShader{};
  std::array<VkShaderModule, eShaderGroupCount> m_shaderModules{};

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtPipelineProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceRayTracingInvocationReorderPropertiesNV m_reorderProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_NV};
};

//------------------------------------------------------------------------------
// Initialize the renderer
// - Creating the G-Buffers
// - Creating the ray tracing pipeline
// - Creating the descriptor set for ray tracing
//
bool RendererPathtracer::init(Resources& res, Scene& scene)
{
  m_dutil  = std::make_unique<nvvk::DebugUtil>(res.ctx.device);
  m_device = res.ctx.device;

  if(!initShaders(res, false))
  {
    return false;
  }

  m_silhouette = std::make_unique<Silhouette>(res);
  m_denoiser   = std::make_unique<AtrousDenoiser>(res);

  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext                  = &m_rtPipelineProperties;
  m_rtPipelineProperties.pNext = &m_reorderProperties;
  vkGetPhysicalDeviceProperties2(res.ctx.physicalDevice, &prop2);

  const uint32_t t_queue_index = res.ctx.transfer.familyIndex;

  m_rtxSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);  // Descriptor set for RTX
  m_sbt    = std::make_unique<nvvk::SBTWrapper>();
  m_sbt->setup(m_device, t_queue_index, res.m_allocator.get(), m_rtPipelineProperties);
  m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, res.m_allocator.get());

  createGBuffer(res);
  createRtxSet();
  writeRtxSet(scene);


  m_sceneBBox = scene.m_gltfScene->getSceneBounds();

  return true;
}

//------------------------------------------------------------------------------
// Compile all shaders
//
bool RendererPathtracer::initShaders(Resources& res, bool reload)
{
  nvh::ScopedTimer st(__FUNCTION__);

  if((reload || g_forceExternalShaders) && res.hasGlslCompiler())
  {
    m_spvShader.resize(eShaderGroupCount);
    m_spvShader[eRaygen]     = res.compileGlslShader("pathtrace.rgen.glsl", shaderc_shader_kind::shaderc_raygen_shader);
    m_spvShader[eMiss]       = res.compileGlslShader("pathtrace.rmiss.glsl", shaderc_shader_kind::shaderc_miss_shader);
    m_spvShader[eShadowMiss] = res.compileGlslShader("shadow.rmiss.glsl", shaderc_shader_kind::shaderc_miss_shader);
    m_spvShader[eClosestHit] = res.compileGlslShader("pathtrace.rchit.glsl", shaderc_shader_kind::shaderc_closesthit_shader);
    m_spvShader[eAnyHit]   = res.compileGlslShader("pathtrace.rahit.glsl", shaderc_shader_kind::shaderc_anyhit_shader);
    m_spvShader[eShadowCH] = res.compileGlslShader("shadow.rchit.glsl", shaderc_shader_kind::shaderc_closesthit_shader);
    m_spvShader[eShadowAH] = res.compileGlslShader("shadow.rahit.glsl", shaderc_shader_kind::shaderc_anyhit_shader);
    m_spvShader[eIndirect] = res.compileGlslShader("pathtrace.comp.glsl", shaderc_shader_kind::shaderc_compute_shader);

    for(size_t i = 0; i < m_spvShader.size(); i++)
    {
      auto& s = m_spvShader[i];

      if(s.GetCompilationStatus() != shaderc_compilation_status_success)
      {
        LOGE("Error when loading shaders: %zu\n", i);
        LOGE("Error %s\n", s.GetErrorMessage().c_str());
        return false;
      }
      m_shaderModules[i] = res.createShaderModule(s);
    }
  }
  else
  {
    const auto& rgen_shd  = std::vector<uint32_t>{std::begin(pathtrace_rgen_glsl), std::end(pathtrace_rgen_glsl)};
    const auto& rchit_shd = std::vector<uint32_t>{std::begin(pathtrace_rchit_glsl), std::end(pathtrace_rchit_glsl)};
    const auto& rmiss_shd = std::vector<uint32_t>{std::begin(pathtrace_rmiss_glsl), std::end(pathtrace_rmiss_glsl)};
    const auto& rahit_shd = std::vector<uint32_t>{std::begin(pathtrace_rahit_glsl), std::end(pathtrace_rahit_glsl)};
    const auto& rshadowmiss_shd = std::vector<uint32_t>{std::begin(shadow_rmiss_glsl), std::end(shadow_rmiss_glsl)};
    const auto& rahshadow_shd   = std::vector<uint32_t>{std::begin(shadow_rahit_glsl), std::end(shadow_rahit_glsl)};
    const auto& rchshadow_shd   = std::vector<uint32_t>{std::begin(shadow_rchit_glsl), std::end(shadow_rchit_glsl)};
    const auto& comp_shd        = std::vector<uint32_t>{std::begin(pathtrace_comp_glsl), std::end(pathtrace_comp_glsl)};

    m_shaderModules[eRaygen]     = nvvk::createShaderModule(m_device, rgen_shd);
    m_shaderModules[eMiss]       = nvvk::createShaderModule(m_device, rmiss_shd);
    m_shaderModules[eShadowMiss] = nvvk::createShaderModule(m_device, rshadowmiss_shd);
    m_shaderModules[eClosestHit] = nvvk::createShaderModule(m_device, rchit_shd);
    m_shaderModules[eAnyHit]     = nvvk::createShaderModule(m_device, rahit_shd);
    m_shaderModules[eShadowCH]   = nvvk::createShaderModule(m_device, rchshadow_shd);
    m_shaderModules[eShadowAH]   = nvvk::createShaderModule(m_device, rahshadow_shd);
    m_shaderModules[eIndirect]   = nvvk::createShaderModule(m_device, comp_shd);
  }

  m_dutil->DBG_NAME(m_shaderModules[eRaygen]);
  m_dutil->DBG_NAME(m_shaderModules[eMiss]);
  m_dutil->DBG_NAME(m_shaderModules[eShadowMiss]);
  m_dutil->DBG_NAME(m_shaderModules[eClosestHit]);
  m_dutil->DBG_NAME(m_shaderModules[eAnyHit]);
  m_dutil->DBG_NAME(m_shaderModules[eShadowCH]);
  m_dutil->DBG_NAME(m_shaderModules[eShadowAH]);
  m_dutil->DBG_NAME(m_shaderModules[eIndirect]);

  return true;
}

bool RendererPathtracer::reloadShaders(Resources& res, Scene& /*scene*/)
{
  for(auto& s : m_shaderModules)
  {
    vkDestroyShaderModule(m_device, s, nullptr);
    s = VK_NULL_HANDLE;
  }
  if(!initShaders(res, true))
    return false;
  if(m_sbt)
    m_sbt->destroy();
  if(m_rtxPipe)
    m_rtxPipe->destroy(m_device);
  if(m_indirectPipe)
    m_indirectPipe->destroy(m_device);
  m_rtxPipe.reset();
  m_indirectPipe.reset();
  //m_sbt.reset();
  return true;
}

//------------------------------------------------------------------------------
// Creating the descriptor set for the ray tracing
// - Top level acceleration structure
// - Output image
//
void RendererPathtracer::createRtxSet()
{
  m_rtxSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

  // This descriptor set, holds the top level acceleration structure, the output image and the selection image
  m_rtxSet->addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
  m_rtxSet->addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  m_rtxSet->addBinding(RtxBindings::eNormalDepth, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  m_rtxSet->addBinding(RtxBindings::eSelect, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  m_rtxSet->initLayout();
  m_rtxSet->initPool(1);
  m_dutil->DBG_NAME(m_rtxSet->getLayout());
  m_dutil->DBG_NAME(m_rtxSet->getSet());
}

//------------------------------------------------------------------------------
// Writing the descriptor set for the ray tracing
// - Top level acceleration structure
// - Output image
//
void RendererPathtracer::writeRtxSet(Scene& scene)
{
  if(!scene.isValid())
  {
    return;
  }

  // Write to descriptors
  VkAccelerationStructureKHR                   tlas = scene.m_gltfSceneRtx->tlas();
  VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{
      .sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
      .accelerationStructureCount = 1,
      .pAccelerationStructures    = &tlas,
  };

  const VkDescriptorImageInfo outImage    = m_gBuffers->getDescriptorImageInfo(GBufferType::eRgbLinear);
  const VkDescriptorImageInfo normalDepth = m_gBuffers->getDescriptorImageInfo(GBufferType::eNormalDepth);
  const VkDescriptorImageInfo selectImage = m_gBuffers->getDescriptorImageInfo(GBufferType::eSilhouette);  // for selection

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_rtxSet->makeWrite(0, RtxBindings::eTlas, &desc_as_info));
  writes.emplace_back(m_rtxSet->makeWrite(0, RtxBindings::eOutImage, &outImage));
  writes.emplace_back(m_rtxSet->makeWrite(0, RtxBindings::eNormalDepth, &normalDepth));
  writes.emplace_back(m_rtxSet->makeWrite(0, RtxBindings::eSelect, &selectImage));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//------------------------------------------------------------------------------
// De-initialize the renderer
//
void RendererPathtracer::deinit()
{
  if(m_sbt)
    m_sbt->destroy();
  if(m_rtxPipe)
    m_rtxPipe->destroy(m_device);
  if(m_indirectPipe)
    m_indirectPipe->destroy(m_device);
  m_rtxSet.reset();
  m_rtxPipe.reset();
  m_indirectPipe.reset();
  m_sbt.reset();
  for(auto& s : m_shaderModules)
  {
    vkDestroyShaderModule(m_device, s, nullptr);
    s = VK_NULL_HANDLE;
  }
  m_shaderModules.fill(VK_NULL_HANDLE);
}

//------------------------------------------------------------------------------
// Creating the G-Buffers, only RGBA32F, no depth
//
void RendererPathtracer::createGBuffer(Resources& res)
{
  m_gBuffers->destroy();
  m_gBuffers->create(res.m_finalImage->getSize(), m_gbufferFormats, VK_FORMAT_UNDEFINED);
}

//------------------------------------------------------------------------------
// Rendering the scene using ray tracing
//
void RendererPathtracer::render(VkCommandBuffer cmd, Resources& /*res*/, Scene& scene, Settings& settings, nvvk::ProfilerVK& profiler)
{
  auto scopeDbg = m_dutil->DBG_SCOPE(cmd);
  auto sec      = profiler.timeRecurring("Raytrace", cmd);

  static int lastSelected = -1;

  // Push constant
  m_pushConst.maxDepth      = g_pathtraceSettings.maxDepth;
  m_pushConst.maxSamples    = g_pathtraceSettings.maxSamples;
  m_pushConst.dbgMethod     = g_pathtraceSettings.dbgMethod;
  m_pushConst.maxLuminance  = settings.maxLuminance;
  m_pushConst.useRTDenoiser = m_denoiser->isActivated();
  if(lastSelected != scene.getSelectedRenderNode() || scene.m_sceneFrameInfo.frameCount <= 0)
  {
    lastSelected                   = scene.getSelectedRenderNode();
    m_pushConst.selectedRenderNode = lastSelected;
    if(g_pathtraceSettings.autoFocus)
      g_pathtraceSettings.focalDistance = glm::length(CameraManip.getEye() - CameraManip.getCenter());
  }
  else
    m_pushConst.selectedRenderNode = -1;  // Resetting the selected object, to avoid shooting rays on the same object

  m_pushConst.focalDistance = g_pathtraceSettings.focalDistance;
  m_pushConst.aperture      = g_pathtraceSettings.aperture;
  if(g_elemDebugPrintf)
    m_pushConst.mouseCoord = g_elemDebugPrintf->getMouseCoord();

  // Ray trace
  VkDescriptorSet dsHdr   = scene.m_hdrEnv->getDescriptorSet();
  VkDescriptorSet dsSky   = scene.m_sky->getDescriptorSet();
  VkDescriptorSet dsScene = scene.m_sceneDescriptorSet;

  const VkExtent2D size = m_gBuffers->getSize();

  std::vector<VkDescriptorSet> desc_sets{m_rtxSet->getSet(), dsScene, dsSky, dsHdr};
  if(g_pathtraceSettings.renderMode == 0)
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtxPipe->plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtxPipe->layout, 0,
                            static_cast<uint32_t>(desc_sets.size()), desc_sets.data(), 0, nullptr);
    vkCmdPushConstants(cmd, m_rtxPipe->layout, VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstantPathtracer), &m_pushConst);

    const std::array<VkStridedDeviceAddressRegionKHR, 4>& regions = m_sbt->getRegions();
    vkCmdTraceRaysKHR(cmd, regions.data(), &regions[1], &regions[2], &regions[3], size.width, size.height, 1);
  }
  else
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectPipe->plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_indirectPipe->layout, 0,
                            static_cast<uint32_t>(desc_sets.size()), desc_sets.data(), 0, nullptr);
    vkCmdPushConstants(cmd, m_indirectPipe->layout, VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstantPathtracer), &m_pushConst);

    VkExtent2D groups = getGroupCounts(size);
    vkCmdDispatch(cmd, groups.width, groups.height, 1);
  }

  // Making sure the rendered image is ready to be used
  VkImage outImage = m_gBuffers->getColorImage(GBufferType::eRgbLinear);
  VkImageMemoryBarrier barrier = nvvk::makeImageMemoryBarrier(outImage, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                                              VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                       0, nullptr, 1, &barrier);

  if(m_denoiser->isActivated())
  {
    // Initial buffers
    VkDescriptorImageInfo colorBuffer       = m_gBuffers->getDescriptorImageInfo(GBufferType::eRgbLinear);
    VkDescriptorImageInfo resultBuffer      = m_gBuffers->getDescriptorImageInfo(GBufferType::eRgbResult);
    VkDescriptorImageInfo tmpBuffer         = m_gBuffers->getDescriptorImageInfo(GBufferType::eTempResult);
    VkDescriptorImageInfo normalDepthBuffer = m_gBuffers->getDescriptorImageInfo(GBufferType::eNormalDepth);

    m_denoiser->render(cmd, size, colorBuffer, resultBuffer, normalDepthBuffer, tmpBuffer);
  }
  else
  {
    // Blit the 32-bit color buffer to the 16-bit color buffer
    VkOffset3D  minCorner = {0, 0, 0};
    VkOffset3D  maxCorner = {int(m_gBuffers->getSize().width), int(m_gBuffers->getSize().height), 1};
    VkImageBlit blitRegions{
        .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
        .srcOffsets     = {minCorner, maxCorner},
        .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
        .dstOffsets     = {minCorner, maxCorner},
    };
    vkCmdBlitImage(cmd, m_gBuffers->getColorImage(GBufferType::eRgbLinear), VK_IMAGE_LAYOUT_GENERAL,
                   m_gBuffers->getColorImage(GBufferType::eRgbResult), VK_IMAGE_LAYOUT_GENERAL, 1, &blitRegions, VK_FILTER_LINEAR);
  }

  // Silhouette : adding a contour around the selected object on top of the eRgbResult
  if(m_silhouette->isValid() && scene.getSelectedRenderNode() != -1)
  {
    m_silhouette->updateBinding(SilhoutteImages::eObjectID, m_gBuffers->getColorImageView(GBufferType::eSilhouette),
                                VK_IMAGE_LAYOUT_GENERAL);
    m_silhouette->updateBinding(SilhoutteImages::eRGBAIImage, m_gBuffers->getColorImageView(GBufferType::eRgbResult),
                                VK_IMAGE_LAYOUT_GENERAL);
    m_silhouette->setColor(nvvkhl_shaders::toLinear(settings.silhouetteColor));
    m_silhouette->dispatch(cmd, m_gBuffers->getSize());
  }
}

//------------------------------------------------------------------------------
// User interface for the renderer
//
bool RendererPathtracer::onUI()
{
  auto& headerManager = CollapsingHeaderManager::getInstance();
  bool  changed{false};

  if(headerManager.beginHeader("RendererPathtracer"))
  {
    // Setting the aperture max slidder value, based on the scene size
    float scaleFactor = std::log(m_sceneBBox.radius());
    scaleFactor       = std::max(scaleFactor, 0.0f);   // Prevent negative values when the scene is small
    float apertureMax = 0.0001f + scaleFactor * 5.0f;  // Minimum max aperture is 0.0001


    PE::begin();
    changed |= PE::SliderInt("Max Depth", &g_pathtraceSettings.maxDepth, 1, 100);
    changed |= PE::SliderInt("Max Samples", &g_pathtraceSettings.maxSamples, 1, 100);
    if(PE::treeNode("Depth-of-field"))
    {
      changed |= PE::SliderFloat("Aperture", &g_pathtraceSettings.aperture, 0.0f, apertureMax, "%5.9f",
                                 ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat, "Out-of-focus effect");
      changed |= PE::Checkbox("Auto Focus", &g_pathtraceSettings.autoFocus, "Use interest position");
      ImGui::BeginDisabled(g_pathtraceSettings.autoFocus);
      changed |= PE::DragFloat("Focal Distance", &g_pathtraceSettings.focalDistance, 100.0f, 0.0f, 1000000.0f, "%5.9f",
                               ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat, "Distance to focal point");
      g_pathtraceSettings.focalDistance = std::max(0.000000001f, g_pathtraceSettings.focalDistance);
      ImGui::EndDisabled();
      PE::treePop();
    }
    if(PE::treeNode("Extra"))
    {
      changed |= PE::Combo("Debug Method", reinterpret_cast<int32_t*>(&g_pathtraceSettings.dbgMethod),
                           "None\0Metallic\0Roughness\0Normal\0Tangent\0Bitangent\0BaseColor\0Emissive\0Opacity\0TexCoord0\0TexCoord1\0\0");
      changed |= PE::Combo("Render Mode", reinterpret_cast<int32_t*>(&g_pathtraceSettings.renderMode), "RTX\0Indirect\0\0");
      PE::treePop();
    }

    m_denoiser->onUi();

    PE::end();
  }

  return changed;
}

//------------------------------------------------------------------------------
// When the scene changes, we need to update the descriptor set
//
void RendererPathtracer::handleChange(Resources& res, Scene& scene)
{
  bool writeDescriptor = scene.hasDirtyFlag(Scene::eHdrEnv);
  bool gbufferChanged  = res.hasGBuffersChanged();

  if((g_pathtraceSettings.renderMode == RenderMode::eRTX) && !m_rtxPipe)
    createRtxPipeline(res, scene);
  if((g_pathtraceSettings.renderMode == RenderMode::eIndirect) && !m_indirectPipe)
    createIndirectPipeline(res, scene);


  if(gbufferChanged || writeDescriptor)
  {
    vkDeviceWaitIdle(m_device);
    scene.resetFrameCount();  // Any change in the scene requires a reset of the frame count
  }
  if(gbufferChanged)
  {
    // Need to recreate the output G-Buffers with the new size
    createGBuffer(res);
    writeDescriptor = true;
  }
  if(writeDescriptor)
  {
    // Writing the descriptor set for the ray tracing
    // Which includes the top level acceleration structure and the output images
    writeRtxSet(scene);
  }
}


//------------------------------------------------------------------------------
// Creating the ray tracing pipeline
//
void RendererPathtracer::createRtxPipeline(Resources& res, Scene& scene)
{
  nvh::ScopedTimer st(__FUNCTION__);

  if(m_rtxPipe)
    m_rtxPipe->destroy(m_device);
  m_rtxPipe = std::make_unique<nvvkhl::PipelineContainer>();
  m_rtxPipe->plines.resize(1);


  std::array<VkPipelineShaderStageCreateInfo, 7> stages{};
  for(auto& s : stages)
    s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

  stages[eRaygen].module = m_shaderModules[eRaygen];
  stages[eRaygen].pName  = "main";
  stages[eRaygen].stage  = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  m_dutil->setObjectName(stages[eRaygen].module, "rgen");

  stages[eMiss].module = m_shaderModules[eMiss];
  stages[eMiss].pName  = "main";
  stages[eMiss].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
  m_dutil->setObjectName(stages[eMiss].module, "rmiss");

  stages[eClosestHit].module = m_shaderModules[eClosestHit];
  stages[eClosestHit].pName  = "main";
  stages[eClosestHit].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  m_dutil->setObjectName(stages[eClosestHit].module, "rchit");

  stages[eAnyHit].module = m_shaderModules[eAnyHit];
  stages[eAnyHit].pName  = "main";
  stages[eAnyHit].stage  = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  m_dutil->setObjectName(stages[eAnyHit].module, "rahit");

  //Shadow
  stages[eShadowMiss].module = m_shaderModules[eShadowMiss];
  stages[eShadowMiss].pName  = "main";
  stages[eShadowMiss].stage  = VK_SHADER_STAGE_MISS_BIT_KHR;
  m_dutil->setObjectName(stages[eMiss].module, "shaodw_rmiss");

  stages[eShadowCH].module = m_shaderModules[eShadowCH];
  stages[eShadowCH].pName  = "main";
  stages[eShadowCH].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  m_dutil->setObjectName(stages[eShadowCH].module, "rchit_shadow");

  stages[eShadowAH].module = m_shaderModules[eShadowAH];
  stages[eShadowAH].pName  = "main";
  stages[eShadowAH].stage  = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  m_dutil->setObjectName(stages[eShadowAH].module, "rahit_shadow");


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

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  shader_groups.push_back(group);

  // Shadow Miss
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
  group.closestHitShader = eShadowCH;
  group.anyHitShader     = eShadowAH;
  shader_groups.push_back(group);


  // Shader Execution Reorder (SER)
  int supportSER =
      (m_reorderProperties.rayTracingInvocationReorderReorderingHint & VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_NV) ? 1 : 0;
  nvvk::Specialization specialization;
  specialization.add(0, supportSER);
  stages[eRaygen].pSpecializationInfo = specialization.getSpecialization();

  // Creating of the pipeline layout
  const VkPushConstantRange pushConstantRange{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(DH::PushConstantPathtracer)};
  std::vector<VkDescriptorSetLayout> descSetLayouts = {m_rtxSet->getLayout(), scene.m_sceneDescriptorSetLayout,
                                                       scene.m_sky->getDescriptorSetLayout(),
                                                       scene.m_hdrEnv->getDescriptorSetLayout()};
  VkPipelineLayoutCreateInfo         pipeLayoutCreateInfo{
              .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
              .setLayoutCount         = static_cast<uint32_t>(descSetLayouts.size()),
              .pSetLayouts            = descSetLayouts.data(),
              .pushConstantRangeCount = 1,
              .pPushConstantRanges    = &pushConstantRange,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipeLayoutCreateInfo, nullptr, &m_rtxPipe->layout));
  m_dutil->DBG_NAME(m_rtxPipe->layout);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rtPipelineCreateInfo{
      .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
      .stageCount                   = static_cast<uint32_t>(stages.size()),  // Stages are shaders
      .pStages                      = stages.data(),
      .groupCount                   = static_cast<uint32_t>(shader_groups.size()),
      .pGroups                      = shader_groups.data(),
      .maxPipelineRayRecursionDepth = 2,  // Ray depth
      .layout                       = m_rtxPipe->layout,
  };
  NVVK_CHECK(vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rtPipelineCreateInfo, nullptr, (m_rtxPipe->plines).data()));
  m_dutil->DBG_NAME(m_rtxPipe->plines[0]);

  // Creating the Shading Binding Table
  m_sbt->create(m_rtxPipe->plines[0], rtPipelineCreateInfo);
}


//------------------------------------------------------------------------------
// Creating the ray tracing pipeline
//
void RendererPathtracer::createIndirectPipeline(Resources& res, Scene& scene)
{
  nvh::ScopedTimer st(__FUNCTION__);

  if(m_indirectPipe)
    m_indirectPipe->destroy(m_device);
  m_indirectPipe = std::make_unique<nvvkhl::PipelineContainer>();
  m_indirectPipe->plines.resize(1);

  // Creating of the pipeline layout
  const VkPushConstantRange pushConstantRange{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(DH::PushConstantPathtracer)};
  std::vector<VkDescriptorSetLayout> descSetLayouts = {m_rtxSet->getLayout(), scene.m_sceneDescriptorSetLayout,
                                                       scene.m_sky->getDescriptorSetLayout(),
                                                       scene.m_hdrEnv->getDescriptorSetLayout()};
  VkPipelineLayoutCreateInfo         pipeLayoutCreateInfo{
              .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
              .setLayoutCount         = static_cast<uint32_t>(descSetLayouts.size()),
              .pSetLayouts            = descSetLayouts.data(),
              .pushConstantRangeCount = 1,
              .pPushConstantRanges    = &pushConstantRange,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipeLayoutCreateInfo, nullptr, &m_indirectPipe->layout));
  m_dutil->DBG_NAME(m_indirectPipe->layout);

  VkShaderModule                  module = m_shaderModules[eIndirect];
  VkPipelineShaderStageCreateInfo comp_shd{
      .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = module,
      .pName  = "main",
  };

  VkComputePipelineCreateInfo cpCreateInfo{
      .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage  = comp_shd,
      .layout = m_indirectPipe->layout,
  };

  NVVK_CHECK(vkCreateComputePipelines(m_device, {}, 1, &cpCreateInfo, nullptr, &m_indirectPipe->plines[0]));
  m_dutil->DBG_NAME(m_indirectPipe->plines[0]);
}


//------------------------------------------------------------------------------
// Factory function to create the renderer
//
std::unique_ptr<Renderer> makeRendererPathtracer()
{
  return std::make_unique<RendererPathtracer>();
}

}  // namespace gltfr
