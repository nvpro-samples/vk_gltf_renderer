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

//////////////////////////////////////////////////////////////////////////
/*
    GLTF Rasterizer Implementation

    This rasterizer provides a traditional forward rendering pipeline for GLTF scenes
    with the following key features:
    
    - Forward rendering with PBR (Physically Based Rendering) material support
    - Environment mapping with HDR and procedural sky options
    - Support for transparent and double-sided materials
    - Wireframe rendering mode for debugging
    - Dynamic state management for flexible pipeline configuration
    - Efficient vertex and index buffer handling
    - Support for material variants and animations
    - Integration with the main renderer's G-Buffer system
    
    The implementation uses Vulkan's modern dynamic state features for efficient
    pipeline state management and supports both solid and transparent rendering
    modes with proper depth sorting and blending.
*/
//////////////////////////////////////////////////////////////////////////

#include <algorithm>

#include <nvapp/elem_dbgprintf.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/parameter_registry.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/mipmaps.hpp>
#include <nvvk/descriptors.hpp>

#include <nvshaders_host/pbr_sheen_lut.hpp>

#include "renderer_rasterizer.hpp"

#include "scene_shader_macros.hpp"

// Pre-compiled shaders
#include "_autogen/gltf_raster.slang.h"
#include "_autogen/sky_physical.slang.h"
#include "_autogen/hdr_charlie_brdf_lut.slang.h"

#include "nvvk/default_structs.hpp"

#if defined(USE_DLSS)
#include "dlss.hpp"
#endif


//--------------------------------------------------------------------------------------------------
// Constructor: eagerly creates the DLSS-SR helper so that registerParameters() and
// setSettingsHandler() (both called BEFORE onAttach by GltfRenderer) can register the
// DLSS settings. The helper holds no Vulkan resources yet at this point; init() happens
// later in onAttach() once we have a valid device.
Rasterizer::Rasterizer()
{
#if defined(USE_DLSS)
  m_dlss = std::make_unique<Dlss>(Dlss::Kind::SR);
#endif
}

// Destructor in the cpp so the unique_ptr<Dlss> destructor sees the full type (dlss.hpp is
// included above). Keeps dlss.hpp out of the rasterizer header.
Rasterizer::~Rasterizer() = default;


//--------------------------------------------------------------------------------------------------
// Initialize the rasterizer with required resources and profiler
// This sets up the device, allocator, and initializes the sky physical model
void Rasterizer::onAttach(Resources& resources, nvvk::ProfilerGpuTimer* profiler)
{
  ::BaseRenderer::onAttach(resources, profiler);
  m_device      = resources.allocator.getDevice();
  m_commandPool = resources.commandPool;
  m_skyPhysical.init(&resources.allocator, std::span(sky_physical_slang));
  compileShader(resources, false);  // Compile the shader
  createOpaqueColorImage(resources);
  createSheenLut(resources);

#if defined(USE_DLSS)
  m_dlss->init(resources);
#endif
  // Motion-vector G-buffer is allocated on first onResize where we know the actual viewport size.
}

//--------------------------------------------------------------------------------------------------
// Register command line (CLI) parameters for the Rasterizer
void Rasterizer::registerParameters(nvutils::ParameterRegistry* paramReg)
{
  // Rasterizer-specific command line parameters
  paramReg->add({"rasterUseRecordedCmd", "Rasterizer: Use recorded command buffers"}, &m_useRecordedCmd);
#if defined(USE_DLSS)
  m_dlss->registerParameters(paramReg);
#endif
}

//--------------------------------------------------------------------------------------------------
// Set the settings handler. Persists the rasterizer-only settings to the INI file under
// names that match the command-line `--rasterUseRecordedCmd` flag.
void Rasterizer::setSettingsHandler(nvgui::SettingsHandler* settingsHandler)
{
  if(settingsHandler)
    settingsHandler->setSetting("rasterUseRecordedCmd", &m_useRecordedCmd);
#if defined(USE_DLSS)
  m_dlss->setSettingsHandler(settingsHandler);
#endif
}

//--------------------------------------------------------------------------------------------------
// Clean up rasterizer resources
// Destroys pipeline layout and shaders, and deinitializes the sky physical model
void Rasterizer::onDetach(Resources& resources)
{
#if defined(USE_DLSS)
  // Keep m_dlss alive past onDetach: SettingsHandler stored a raw pointer to its qualityMode
  // for .ini persistence at app shutdown. The unique_ptr will release it via Rasterizer's dtor.
  m_dlss->deinit(resources);
#endif
  destroySheenLut(resources);
  destroyOpaqueColorImage(resources);
  vkDestroyPipelineLayout(m_device, m_graphicPipelineLayout, nullptr);
  vkDestroyShaderEXT(m_device, m_vertexShader, nullptr);
  vkDestroyShaderEXT(m_device, m_fragmentShader, nullptr);
  vkDestroyShaderEXT(m_device, m_wireframeShader, nullptr);

  m_skyPhysical.deinit();
}

//--------------------------------------------------------------------------------------------------
// Handle window resize events
// Updates the renderer's state for the new viewport dimensions
void Rasterizer::onResize(VkCommandBuffer cmd, const VkExtent2D& size, Resources& resources)
{
  freeRecordCommandBuffer(resources);

#if defined(USE_DLSS)
  // DLSS-SR owns its internal color-input AND motion images
  m_dlss->updateSize(cmd, resources, size);

  // Wire current resources into the DLSS feature
  m_dlss->setOutputImage(resources.gBuffers.getColorImage(Resources::eImgRendered),
                         resources.gBuffers.getColorImageView(Resources::eImgRendered),
                         resources.gBuffers.getColorFormat(Resources::eImgRendered));
  // Depth is owned by the inner GBuffer now -- setResources binds it directly.
  m_dlss->setResources();
#endif
}

//--------------------------------------------------------------------------------------------------
// Render the rasterizer's UI controls
// Currently provides a wireframe toggle option
// Returns true if any UI state changed
bool Rasterizer::onUIRender(Resources& resources)
{
  namespace PE = nvgui::PropertyEditor;
  bool changed = false;
  if(PE::begin())
  {
    // Toggling recording on/off must invalidate the cached secondary command buffer; otherwise
    // a recorded buffer that was captured before/after a shader recompile or scene change can
    // be replayed, replaying stale shader bytecode and stale draw order.
    if(PE::Checkbox("Use Recorded Cmd", &m_useRecordedCmd, "Use recorded command buffers for better performance"))
    {
      freeRecordCommandBuffer(resources);
      changed = true;
    }
    PE::end();
  }

#if defined(USE_DLSS)
  // Mirrors PathTracer's "AI Denoisers" header so users get the same UX across the two renderers.
  if(ImGui::CollapsingHeader("DLSS"))
  {
    if(m_dlss->onUi(resources))
    {
      // Quality / preset change
      freeRecordCommandBuffer(resources);
      changed = true;
    }
  }
#endif

  return changed;
}

namespace {

// Where the rasterizer renders this frame: outer GBuffer (DLSS off / build), or the inner
// GBuffer owned by Dlss (DLSS-SR on).
struct RasterTargets
{
  VkImage     colorImage{};
  VkImageView colorView{};
  VkImage     selectionImage{};
  VkImageView selectionView{};
  VkImage     depthImage{};
  VkImageView depthView{};
  VkExtent2D  extent{};

#if defined(USE_DLSS)
  VkImage     motionImage{VK_NULL_HANDLE};
  VkImageView motionView{VK_NULL_HANDLE};
#endif
};

// Outer GBuffer route (USE_DLSS=OFF, or USE_DLSS=ON without an SR helper).
RasterTargets makeOuterTargets(Resources& resources)
{
  RasterTargets t{};
  t.colorImage     = resources.gBuffers.getColorImage(Resources::eImgRendered);
  t.colorView      = resources.gBuffers.getColorImageView(Resources::eImgRendered);
  t.selectionImage = resources.gBuffers.getColorImage(Resources::eImgSelection);
  t.selectionView  = resources.gBuffers.getColorImageView(Resources::eImgSelection);
  t.depthImage     = resources.gBuffers.getDepthImage();
  t.depthView      = resources.gBuffers.getDepthImageView();
  t.extent         = resources.gBuffers.getSize();
  return t;
}

#if defined(USE_DLSS)
// Inner GBuffer route (DLSS-SR alive). Inner extent shrinks per quality mode.
RasterTargets makeInnerTargets(Dlss& sr)
{
  using SrSlot = Dlss::SrSlot;
  RasterTargets t{};
  t.colorImage     = sr.getSrImage(SrSlot::eColor);
  t.colorView      = sr.getSrImageView(SrSlot::eColor);
  t.selectionImage = sr.getSrImage(SrSlot::eSelection);
  t.selectionView  = sr.getSrImageView(SrSlot::eSelection);
  t.depthImage     = sr.getSrImage(SrSlot::eDepth);
  t.depthView      = sr.getSrImageView(SrSlot::eDepth);
  t.motionImage    = sr.getSrImage(SrSlot::eMotion);
  t.motionView     = sr.getSrImageView(SrSlot::eMotion);
  t.extent         = sr.getRenderSize();
  return t;
}
#endif

}  // namespace


//--------------------------------------------------------------------------------------------------
// Main rendering function for the rasterizer
// Handles:
// 1. Environment rendering (sky or HDR)
// 2. Scene geometry rendering with proper material handling
// 3. Wireframe overlay when enabled
// 4. Proper state management for different material types
void Rasterizer::onRender(VkCommandBuffer cmd, Resources& resources)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  auto timerSection = m_profiler->cmdFrameSection(cmd, "Raster");

  if(m_lastWireframe != resources.settings.wireframe)
  {
    freeRecordCommandBuffer(resources);
    m_lastWireframe = resources.settings.wireframe;
  }

  bool dlssActive = false;
#if defined(USE_DLSS)
  // Drive the DLSS state machine. tick() returns true ONCE on initial NGX-up or a UI-driven
  // recreate; that's the cue to populate the GBuffer and rewire NGX inputs.
  if(m_dlss->tick(resources))
  {
    m_dlss->updateSize(cmd, resources, resources.gBuffers.getSize());
    m_dlss->setOutputImage(resources.gBuffers.getColorImage(Resources::eImgRendered),
                           resources.gBuffers.getColorImageView(Resources::eImgRendered),
                           resources.gBuffers.getColorFormat(Resources::eImgRendered));
    m_dlss->setResources();
  }
  dlssActive = m_dlss->isActive();
#endif

  // Depth-sort transparent nodes back-to-front each frame; re-record the secondary
  // command buffer when the order changes (e.g. camera moved through the scene).
  if(updateSortedBlendNodes(cmd, resources))
  {
    freeRecordCommandBuffer(resources);
  }

  // When Dlss (Kind::SR) is present we render into the inner GBuffer; a post-pass either
  // evaluates DLSS (writes eImgRendered) or blits inner.color back. Otherwise direct to outer.
#if defined(USE_DLSS)
  const RasterTargets targets = makeInnerTargets(*m_dlss);
#else
  const RasterTargets targets = makeOuterTargets(resources);
#endif

  // Sky / HDR dome write into the color sink (storage image). Both targets are in GENERAL
  // layout coming in (DLSS color was created in GENERAL; eImgRendered is returned to GENERAL
  // by the previous frame's barrier).
  if(!resources.settings.useSolidBackground)
  {
    glm::mat4 viewMatrix = resources.cameraManip->getViewMatrix();
    glm::mat4 projMatrix = resources.cameraManip->getPerspectiveMatrix();

    if(resources.settings.envSystem == shaderio::EnvSystem::eSky)
    {
      auto                  timerSection = m_profiler->cmdFrameSection(cmd, "Sky Physical");
      VkDescriptorImageInfo skyTarget{};
      skyTarget.imageView   = targets.colorView;
      skyTarget.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      m_skyPhysical.runCompute(cmd, targets.extent, viewMatrix, projMatrix, resources.skyParams, skyTarget);
    }
    else if(resources.settings.envSystem == shaderio::EnvSystem::eHdr)
    {
      auto timerSection = m_profiler->cmdFrameSection(cmd, "HDR Dome");
      // FIXME nvpro_core2: HdrEnvDome's outImage descriptor pool isn't UPDATE_AFTER_BIND, so any
      // view change (DLSS toggle, HDR IBL reload, resize, quality switch) forces a queueWaitIdle
      // before rebinding. Track lazily on view-handle change.
      if(targets.colorView != m_lastHdrDomeView)
      {
        NVVK_CHECK(vkQueueWaitIdle(resources.app->getQueue(0).queue));
        VkDescriptorImageInfo target{};
        target.imageView   = targets.colorView;
        target.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        resources.hdrDome.setOutImage(target);
        m_lastHdrDomeView = targets.colorView;
      }
      resources.hdrDome.draw(cmd, viewMatrix, projMatrix, targets.extent, glm::vec4(resources.settings.hdrEnvIntensity),
                             resources.settings.hdrEnvRotation, resources.settings.hdrBlur);
    }
  }

  // 2 attachments in non-DLSS builds (color, selection); 3 when USE_DLSS=ON (+ motion at SV_TARGET2).
#if defined(USE_DLSS)
  static constexpr size_t kColorAttachmentCount = 3;
#else
  static constexpr size_t kColorAttachmentCount = 2;
#endif
  std::array<VkRenderingAttachmentInfo, kColorAttachmentCount> attachments;
  attachments.fill(DEFAULT_VkRenderingAttachmentInfo);

  attachments[0].imageView  = targets.colorView;
  attachments[0].clearValue = {{{resources.settings.solidBackgroundColor.x, resources.settings.solidBackgroundColor.y,
                                 resources.settings.solidBackgroundColor.z, 0.f}}};
  attachments[0].loadOp = resources.settings.useSolidBackground ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;

  attachments[1].imageView = targets.selectionView;

#if defined(USE_DLSS)
  // CLEAR gives DLSS a clean MV for fragments the rasterizer doesn't cover this frame.
  attachments[2].imageView  = targets.motionView;
  attachments[2].loadOp     = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[2].clearValue = {{{0.f, 0.f, 0.f, 0.f}}};
#endif

  VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
  depthAttachment.imageView                 = targets.depthView;
  depthAttachment.clearValue                = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

  // Barriers: GENERAL -> COLOR_ATTACHMENT_OPTIMAL for all color sinks (+ motion when DLSS),
  // plus depth init. selection / depth route to inner GBuffer when USE_DLSS=ON.
  nvvk::cmdImageMemoryBarrier(cmd, {targets.colorImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
  nvvk::cmdImageMemoryBarrier(cmd, {targets.selectionImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
#if defined(USE_DLSS)
  if(targets.motionImage != VK_NULL_HANDLE)
    nvvk::cmdImageMemoryBarrier(cmd, {targets.motionImage, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
#endif
  nvvk::cmdImageMemoryBarrier(cmd, {targets.depthImage,
                                    VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                    {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});


  // This is for the fragment shader to know if the opaque color is ready.
  m_pushConst.opaqueColorReady = 0;

  // Patch jitter and imageSize into SceneFrameInfo for up-to-date shader access to current render extent and sub-pixel offset.
  const glm::vec2 imageSizePx{static_cast<float>(targets.extent.width), static_cast<float>(targets.extent.height)};
  vkCmdUpdateBuffer(cmd, resources.bFrameInfo.buffer, offsetof(shaderio::SceneFrameInfo, imageSize), sizeof(glm::vec2), &imageSizePx);

  glm::vec2 jitterPx{0.0f};
#if defined(USE_DLSS)
  if(dlssActive)
  {

    // Dlss owns the Halton jitter + frame-index counter.
    jitterPx = m_dlss->beginFrame().jitter;
    vkCmdUpdateBuffer(cmd, resources.bFrameInfo.buffer, offsetof(shaderio::SceneFrameInfo, jitter), sizeof(glm::vec2), &jitterPx);
  }
#endif
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

  // Setting up the push constant.
  m_pushConst.frameInfo  = (shaderio::SceneFrameInfo*)resources.bFrameInfo.address;
  m_pushConst.skyParams  = (shaderio::SkyPhysicalParameters*)resources.bSkyParams.address;
  m_pushConst.gltfScene  = (shaderio::GltfScene*)resources.sceneVk.sceneDesc().address;
  m_pushConst.mouseCoord = nvapp::ElementDbgPrintf::getMouseCoord();  // Use for debugging: printf in shader
  vkCmdPushConstants(cmd, m_graphicPipelineLayout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof(shaderio::RasterPushConstant), &m_pushConst);


  // Create the rendering info. We do not use secondary command buffer recording when running
  // the two-pass transmission flow because the recorded buffer cannot span the mip generation
  // step in between. The recorded path is still used for scenes without transmission.
  const bool twoPass           = m_sceneHasTransmission;
  const bool recordedThisFrame = m_useRecordedCmd && !twoPass;

  VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
  renderingInfo.flags           = recordedThisFrame ? VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT : 0;
  // renderArea matches the active extent (inner when DLSS-SR is wired, outer otherwise). The
  // recorded secondary cmd buffer's viewport/scissor are issued from this same extent.
  renderingInfo.renderArea           = DEFAULT_VkRect2D(targets.extent);
  renderingInfo.colorAttachmentCount = uint32_t(attachments.size());
  renderingInfo.pColorAttachments    = attachments.data();
  renderingInfo.pDepthAttachment     = &depthAttachment;

  if(recordedThisFrame && m_recordedSceneCmd == VK_NULL_HANDLE)
  {
    recordRasterScene(resources, targets.extent);
  }


  if(twoPass)
  {
    // ** PASS 1: opaque + opaque-double-sided -> color sink **
    {
      auto opaqueSection = m_profiler->cmdFrameSection(cmd, "Render Opaque Only");
      vkCmdBeginRendering(cmd, &renderingInfo);
      renderOpaqueOnly(cmd, resources, targets.extent);
      vkCmdEndRendering(cmd);
    }

    // Capture the opaque framebuffer + generate mip chain. Sources the blit from the active
    // color sink so transmission samples reflect the DLSS input image when DLSS is on (the
    // transmission shader uses the unjittered viewProjMatrix so the lookup is geometrically
    // correct even though the source has sub-pixel jitter baked in).
    captureAndMipOpaqueColor(cmd, resources);

    // ** PASS 2: blend bucket (transmission + transparency) reads m_opaqueColorImage **
    for(auto& att : attachments)
      att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAttachment.loadOp          = VK_ATTACHMENT_LOAD_OP_LOAD;
    renderingInfo.pColorAttachments = attachments.data();
    renderingInfo.pDepthAttachment  = &depthAttachment;

    m_pushConst.opaqueColorReady = 1;
    vkCmdPushConstants(cmd, m_graphicPipelineLayout, VK_SHADER_STAGE_ALL_GRAPHICS, 0,
                       sizeof(shaderio::RasterPushConstant), &m_pushConst);

    {
      auto blendSection = m_profiler->cmdFrameSection(cmd, "Render Blend Only");
      vkCmdBeginRendering(cmd, &renderingInfo);
      renderBlendOnly(cmd, resources, targets.extent);
      vkCmdEndRendering(cmd);
    }
  }
  else
  {
    auto timerSection = m_profiler->cmdFrameSection(cmd, "Raster Scene");

    vkCmdBeginRendering(cmd, &renderingInfo);

    if(recordedThisFrame && m_recordedSceneCmd != VK_NULL_HANDLE)
      vkCmdExecuteCommands(cmd, 1, &m_recordedSceneCmd);
    else
      renderRasterScene(cmd, resources, targets.extent);

    vkCmdEndRendering(cmd);
  }

  // Restore layouts to GENERAL on the same images we transitioned in.
  nvvk::cmdImageMemoryBarrier(cmd, {targets.colorImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
  nvvk::cmdImageMemoryBarrier(cmd, {targets.selectionImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
#if defined(USE_DLSS)
  if(targets.motionImage != VK_NULL_HANDLE)
    nvvk::cmdImageMemoryBarrier(cmd, {targets.motionImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
#endif
  nvvk::cmdImageMemoryBarrier(cmd, {targets.depthImage,
                                    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_GENERAL,
                                    {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

#if defined(USE_DLSS)
  // DLSS evaluate: reads inner.color/motion/depth (GENERAL), writes eImgRendered (also GENERAL).
  if(dlssActive)
  {
    auto             dlaaSection = m_profiler->cmdFrameSection(cmd, "DLSS-SR Evaluate");
    const glm::mat4& viewMat     = resources.cameraManip->getViewMatrix();
    const glm::mat4& projMat     = resources.cameraManip->getPerspectiveMatrix();
    m_dlss->evaluate(cmd, viewMat, projMat);
  }

  // Post-raster blits when the rasterizer rendered into the inner GBuffer:
  //   1. inner.selection -> eImgSelection (always; silhouette + picking sample at outer extent).
  //   2. inner.color     -> eImgRendered  (only when DLSS evaluate did not run; in that case
  //                                        updateSize() sized inner == outer so it's 1:1).
  // In SR fallback mode the rasterizer wrote to outer.{color,selection,depth} directly, so the
  // blits are skipped entirely (the inner GBuffer holds only the motion attachment).
  if(!m_dlss->isFallback())
  {
    using SrSlot             = Dlss::SrSlot;
    const VkExtent2D outerSz = resources.gBuffers.getSize();
    m_dlss->blitInnerToOuter(cmd, m_dlss->getSrImage(SrSlot::eSelection),
                             resources.gBuffers.getColorImage(Resources::eImgSelection), outerSz);
    if(!dlssActive)
      m_dlss->blitInnerToOuter(cmd, m_dlss->getSrImage(SrSlot::eColor),
                               resources.gBuffers.getColorImage(Resources::eImgRendered), outerSz);
  }
#endif
}

//--------------------------------------------------------------------------------------------------
// Render a list of GLTF nodes with their associated materials and geometry
// Handles:
// 1. Material and node-specific constant updates
// 2. Vertex and index buffer binding
// 3. Draw calls for each primitive
void Rasterizer::renderNodes(VkCommandBuffer cmd, Resources& resources, const std::vector<uint32_t>& nodeIDs)
{
  NVVK_DBG_SCOPE(cmd);

  nvvkgltf::Scene* scenePtr = resources.getScene();
  if(!scenePtr)
    return;
  nvvkgltf::Scene&   scene   = *scenePtr;
  nvvkgltf::SceneVk& sceneVk = resources.sceneVk;

  const VkDeviceSize                            offsets{0};
  const std::vector<nvvkgltf::RenderNode>&      renderNodes = scene.getRenderNodes();
  const std::vector<nvvkgltf::RenderPrimitive>& subMeshes   = scene.getRenderPrimitives();

  // Structure to hold only the changing parts
  struct NodeSpecificConstants
  {
    int32_t materialID;
    int32_t renderNodeID;
    int32_t renderPrimID;
  };

  // Get the offset of materialID in the RasterPushConstant struct
  // This assumes materialID is the first field that changes in the struct
  uint32_t offset = static_cast<uint32_t>(offsetof(shaderio::RasterPushConstant, materialID));

  for(const uint32_t& nodeID : nodeIDs)
  {
    const nvvkgltf::RenderNode&      renderNode = renderNodes[nodeID];
    const nvvkgltf::RenderPrimitive& subMesh = subMeshes[renderNode.renderPrimID];  // Mesh referred by the draw object

    if(!renderNode.visible)
      continue;

    // Update only the changing fields
    NodeSpecificConstants nodeConstants{.materialID   = renderNode.materialID,
                                        .renderNodeID = static_cast<int>(nodeID),
                                        .renderPrimID = renderNode.renderPrimID};

    // Push only the changing parts
    vkCmdPushConstants(cmd, m_graphicPipelineLayout, VK_SHADER_STAGE_ALL_GRAPHICS, offset, sizeof(NodeSpecificConstants), &nodeConstants);

    // Bind vertex and index buffers and draw the mesh
    vkCmdBindVertexBuffers(cmd, 0, 1, &sceneVk.vertexBuffers()[renderNode.renderPrimID].position.buffer, &offsets);
    vkCmdBindIndexBuffer(cmd, sceneVk.indices()[renderNode.renderPrimID].buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, subMesh.indexCount, 1, 0, 0, 0);
  }
}

//--------------------------------------------------------------------------------------------------
// Create the graphics pipeline for the rasterizer
// Sets up:
// 1. Pipeline layout with descriptor sets and push constants
// 2. Dynamic state configuration
// 3. Color blending settings for transparent objects
void Rasterizer::createPipeline(Resources& resources)
{
  SCOPED_TIMER(__FUNCTION__);
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts{resources.descriptorSetLayout[0]};

  // Push constant is used to pass data to the shader at each frame
  const VkPushConstantRange pushConstantRange{
      .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS, .offset = 0, .size = sizeof(shaderio::RasterPushConstant)};

  // The pipeline layout is used to pass data to the pipeline, anything with "layout" in the shader
  const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
      .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount         = uint32_t(descriptorSetLayouts.size()),
      .pSetLayouts            = descriptorSetLayouts.data(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstantRange,
  };
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_graphicPipelineLayout));
  NVVK_DBG_NAME(m_graphicPipelineLayout);

  // Override default
  //  m_dynamicPipeline.colorBlendEnables[0]                       = VK_TRUE;
  m_dynamicPipeline.colorBlendEquations[0].alphaBlendOp        = VK_BLEND_OP_ADD;
  m_dynamicPipeline.colorBlendEquations[0].colorBlendOp        = VK_BLEND_OP_ADD;
  m_dynamicPipeline.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  m_dynamicPipeline.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  m_dynamicPipeline.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  m_dynamicPipeline.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

  // Add depth bias settings for solid objects
  m_dynamicPipeline.rasterizationState.depthBiasEnable         = VK_TRUE;
  m_dynamicPipeline.rasterizationState.depthBiasConstantFactor = -1.0f;
  m_dynamicPipeline.rasterizationState.depthBiasSlopeFactor    = 1.0f;

  // Attachment #1 - Selection
  m_dynamicPipeline.colorBlendEnables.push_back(false);  // No blending for attachment #1
  m_dynamicPipeline.colorWriteMasks.push_back(
      {VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT});
  m_dynamicPipeline.colorBlendEquations.push_back(VkColorBlendEquationEXT{});

  // Attachment #2 - Motion vectors (R16G16_SFLOAT).
  m_dynamicPipeline.colorBlendEnables.push_back(false);
  m_dynamicPipeline.colorWriteMasks.push_back({VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT});
  m_dynamicPipeline.colorBlendEquations.push_back(VkColorBlendEquationEXT{});
}

//--------------------------------------------------------------------------------------------------
// Compile the rasterizer's shaders
// Creates vertex, fragment, and wireframe shaders from the gltf_raster.slang source
void Rasterizer::compileShader(Resources& resources, bool fromFile)
{
  SCOPED_TIMER(__FUNCTION__);

  // The cached secondary command buffer baked the previous shader's draws and bound the
  // previous VkShaderEXT handles. After we replace the shader handles below, those bindings
  // would dangle. Free the recorded buffer here so the next frame re-records cleanly.
  freeRecordCommandBuffer(resources);

  // Push constant is used to pass data to the shader at each frame
  const VkPushConstantRange pushConstantRange{
      .stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
      .offset     = 0,
      .size       = sizeof(shaderio::RasterPushConstant),
  };

  std::vector<VkDescriptorSetLayout> descriptorSetLayouts{resources.descriptorSetLayout[0]};

  VkShaderCreateInfoEXT shaderInfo{
      .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
      .stage                  = VK_SHADER_STAGE_VERTEX_BIT,
      .nextStage              = VK_SHADER_STAGE_FRAGMENT_BIT,
      .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
      .codeSize               = gltf_raster_slang_sizeInBytes,
      .pCode                  = gltf_raster_slang,
      .pName                  = "vertexMain",
      .setLayoutCount         = uint32_t(descriptorSetLayouts.size()),
      .pSetLayouts            = descriptorSetLayouts.data(),
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushConstantRange,
  };

  if(fromFile)
  {
    resources.slangCompiler.clearMacros();
    nvvkgltf::applyCommonShaderMacros(resources.slangCompiler);

    if(resources.slangCompiler.compileFile("gltf_raster.slang"))
    {
      shaderInfo.codeSize = resources.slangCompiler.getSpirvSize();
      shaderInfo.pCode    = resources.slangCompiler.getSpirv();
    }
    else
    {
      LOGW("Error compiling gltf_raster.slang\n");
    }
  }

  VkDevice device = resources.allocator.getDevice();
  vkDestroyShaderEXT(device, m_vertexShader, nullptr);
  vkDestroyShaderEXT(device, m_fragmentShader, nullptr);
  vkDestroyShaderEXT(device, m_wireframeShader, nullptr);


  NVVK_CHECK(vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &m_vertexShader));
  NVVK_DBG_NAME(m_vertexShader);
  shaderInfo.pName     = "fragmentMain";
  shaderInfo.stage     = VK_SHADER_STAGE_FRAGMENT_BIT;
  shaderInfo.nextStage = 0;
  NVVK_CHECK(vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &m_fragmentShader));
  NVVK_DBG_NAME(m_fragmentShader);
  shaderInfo.pName = "fragmentWireframeMain";
  shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  NVVK_CHECK(vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &m_wireframeShader));
  NVVK_DBG_NAME(m_wireframeShader);
}

//--------------------------------------------------------------------------------------------------
// Recording in a secondary command buffer, the raster rendering of the scene.
//
void Rasterizer::recordRasterScene(Resources& resources, VkExtent2D renderExtent)
{
  SCOPED_TIMER(__FUNCTION__);

  createRecordCommandBuffer();

  // Two or three color attachment formats. USE_DLSS=ON builds always include the DLSS motion image format.
  std::vector<VkFormat> colorFormat = {resources.gBuffers.getColorFormat(Resources::eImgRendered),
                                       resources.gBuffers.getColorFormat(Resources::eImgSelection)};
#if defined(USE_DLSS)
  colorFormat.push_back(m_dlss->getSrFormat(Dlss::SrSlot::eMotion));
#endif

  VkCommandBufferInheritanceRenderingInfo inheritanceRenderingInfo{
      .sType                   = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO,
      .colorAttachmentCount    = uint32_t(colorFormat.size()),
      .pColorAttachmentFormats = colorFormat.data(),
      .depthAttachmentFormat   = resources.gBuffers.getDepthFormat(),
      .rasterizationSamples    = VK_SAMPLE_COUNT_1_BIT,
  };

  VkCommandBufferInheritanceInfo inheritInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
      .pNext = &inheritanceRenderingInfo,
  };

  VkCommandBufferBeginInfo beginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
      .pInheritanceInfo = &inheritInfo,
  };

  NVVK_CHECK(vkBeginCommandBuffer(m_recordedSceneCmd, &beginInfo));
  renderRasterScene(m_recordedSceneCmd, resources, renderExtent);
  NVVK_CHECK(vkEndCommandBuffer(m_recordedSceneCmd));
}

//--------------------------------------------------------------------------------------------------
// Render the entire scene for raster. Splitting the solid and blend-able element and rendering
// on top, the wireframe if active.
// This is done in a recoded command buffer to be replay
// Common dynamic state and bindings used by both single-pass and split-pass paths.
static void cmdSetCommonRasterState(VkCommandBuffer              cmd,
                                    nvvk::GraphicsPipelineState& dyn,
                                    Resources&                   resources,
                                    VkExtent2D                   renderExtent,
                                    VkPipelineLayout             layout,
                                    VkShaderEXT                  vs,
                                    VkShaderEXT                  fs)
{
  dyn.cmdApplyAllStates(cmd);
  dyn.cmdSetViewportAndScissor(cmd, renderExtent);
  dyn.cmdBindShaders(cmd, {.vertex = vs, .fragment = fs});
  vkCmdSetDepthTestEnable(cmd, VK_TRUE);

  const auto& bindingDescription    = std::to_array<VkVertexInputBindingDescription2EXT>({
      {.sType     = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
          .binding   = 0,
          .stride    = sizeof(glm::vec3),
          .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
          .divisor   = 1},
  });
  const auto& attributeDescriptions = std::to_array<VkVertexInputAttributeDescription2EXT>(
      {{.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0}});
  vkCmdSetVertexInputEXT(cmd, uint32_t(bindingDescription.size()), bindingDescription.data(),
                         uint32_t(attributeDescriptions.size()), attributeDescriptions.data());

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &resources.descriptorSet, 0, nullptr);
}

void Rasterizer::renderOpaqueOnly(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderExtent)
{
  // No profiler section here: this function is also recorded into a secondary command buffer
  // by recordRasterScene(). Profiler sections use host-side vkResetQueryPool + a one-time
  // timeline allocation, so baking them into a replayed secondary triggers
  // VUID-vkCmdWriteTimestamp-None-00830 ("query not reset") on every frame after the first.
  // The two-pass primary path opens its own section at the call site in onRender().
  m_pushConst.frameInfo  = (shaderio::SceneFrameInfo*)resources.bFrameInfo.address;
  m_pushConst.skyParams  = (shaderio::SkyPhysicalParameters*)resources.bSkyParams.address;
  m_pushConst.gltfScene  = (shaderio::GltfScene*)resources.sceneVk.sceneDesc().address;
  m_pushConst.mouseCoord = nvapp::ElementDbgPrintf::getMouseCoord();
  vkCmdPushConstants(cmd, m_graphicPipelineLayout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof(shaderio::RasterPushConstant), &m_pushConst);

  cmdSetCommonRasterState(cmd, m_dynamicPipeline, resources, renderExtent, m_graphicPipelineLayout, m_vertexShader, m_fragmentShader);

  vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
  vkCmdSetDepthBias(cmd, -1.0f, 0.0f, 1.0f);
  renderNodes(cmd, resources, resources.getScene()->getShadedNodes(nvvkgltf::Scene::eRasterSolid));

  vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);
  vkCmdSetDepthBias(cmd, 0.0f, 0.0f, 0.0f);
  renderNodes(cmd, resources, resources.getScene()->getShadedNodes(nvvkgltf::Scene::eRasterSolidDoubleSided));
}

void Rasterizer::renderBlendOnly(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderExtent)
{
  // No profiler section here: see comment in renderOpaqueOnly() -- this is also called from
  // a secondary command buffer recorded once and replayed every frame.
  cmdSetCommonRasterState(cmd, m_dynamicPipeline, resources, renderExtent, m_graphicPipelineLayout, m_vertexShader, m_fragmentShader);

  VkBool32 blendEnable  = VK_TRUE;
  VkBool32 blendDisable = VK_FALSE;
  vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);
  vkCmdSetDepthBias(cmd, 0.0f, 0.0f, 0.0f);
  vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blendEnable);
  renderNodes(cmd, resources, m_sortedBlendNodes);

  if(resources.settings.wireframe)
  {
    m_dynamicPipeline.cmdBindShaders(cmd, {.vertex = m_vertexShader, .fragment = m_wireframeShader});
    vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blendDisable);
    vkCmdSetDepthBias(cmd, 0.0f, 0.0f, 0.0f);
    vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_LINE);
    renderNodes(cmd, resources, resources.getScene()->getShadedNodes(nvvkgltf::Scene::eRasterAll));
  }
}

void Rasterizer::renderRasterScene(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderExtent)
{
  // Single-pass legacy path used for scenes without transmission and for the recorded command
  // buffer route. Equivalent to renderOpaqueOnly() + renderBlendOnly() but reuses one push.
  renderOpaqueOnly(cmd, resources, renderExtent);
  renderBlendOnly(cmd, resources, renderExtent);
}

//--------------------------------------------------------------------------------------------------
// Allocate the opaque-pass color capture used by KHR_materials_transmission. We fix the size at
// OPAQUE_COLOR_SIZE x OPAQUE_COLOR_SIZE to match the Khronos reference; mip count is the full
// chain from that base. R16G16B16A16_SFLOAT keeps HDR fidelity for refracted radiance without
// the storage cost of R32_SFLOAT.
void Rasterizer::createOpaqueColorImage(Resources& resources)
{
  if(m_opaqueColorImage.image != VK_NULL_HANDLE)
    return;

  const VkExtent2D extent2D = {OPAQUE_COLOR_SIZE, OPAQUE_COLOR_SIZE};
  const uint32_t   mips     = nvvk::mipLevels(extent2D);

  VkImageCreateInfo imageInfo{
      .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType     = VK_IMAGE_TYPE_2D,
      .format        = VK_FORMAT_R16G16B16A16_SFLOAT,
      .extent        = {extent2D.width, extent2D.height, 1},
      .mipLevels     = mips,
      .arrayLayers   = 1,
      .samples       = VK_SAMPLE_COUNT_1_BIT,
      .tiling        = VK_IMAGE_TILING_OPTIMAL,
      .usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
      .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };

  VkImageViewCreateInfo viewInfo{
      .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format   = imageInfo.format,
      .subresourceRange =
          {
              .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
              .baseMipLevel   = 0,
              .levelCount     = mips,
              .baseArrayLayer = 0,
              .layerCount     = 1,
          },
  };

  NVVK_CHECK(resources.allocator.createImage(m_opaqueColorImage, imageInfo, viewInfo));
  NVVK_DBG_NAME(m_opaqueColorImage.image);

  // Use the existing global linear sampler for the descriptor.
  VkSampler linearSampler{};
  NVVK_CHECK(resources.samplerPool.acquireSampler(linearSampler));
  m_opaqueColorImage.descriptor.sampler = linearSampler;

  // Initial transition: UNDEFINED -> SHADER_READ_ONLY so it's safe to sample even before the
  // first capture (the shader uses opaqueColorReady = 0 to skip the sample anyway).
  VkCommandBuffer cmd{};
  nvvk::beginSingleTimeCommands(cmd, m_device, m_commandPool);
  nvvk::cmdImageMemoryBarrier(cmd, {m_opaqueColorImage.image,
                                    VK_IMAGE_LAYOUT_UNDEFINED,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1}});
  nvvk::endSingleTimeCommands(cmd, m_device, m_commandPool, resources.app->getQueue(0).queue);

  m_opaqueColorImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  // Write into the eTexturesHdr array at HDR_OPAQUE_INDEX. PARTIALLY_BOUND lets us touch only
  // this one slot without affecting the HDR image / BRDF LUT / sheen LUT slots.
  nvvk::WriteSetContainer write{};
  VkWriteDescriptorSet    ws = resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eTexturesHdr,
                                                                          resources.descriptorSet, HDR_OPAQUE_INDEX, 1U);
  write.append(ws, m_opaqueColorImage);
  vkUpdateDescriptorSets(m_device, write.size(), write.data(), 0, nullptr);
}

void Rasterizer::destroyOpaqueColorImage(Resources& resources)
{
  if(m_opaqueColorImage.image == VK_NULL_HANDLE)
    return;
  resources.samplerPool.releaseSampler(m_opaqueColorImage.descriptor.sampler);
  m_opaqueColorImage.descriptor.sampler = VK_NULL_HANDLE;
  resources.allocator.destroyImage(m_opaqueColorImage);
}

//--------------------------------------------------------------------------------------------------
// Bake the Charlie sheen directional-albedo LUT and wire it into the global descriptor set.
// The actual baking (allocate image + dispatch compute shader) lives in
// `nvshaders::bakeCharlieSheenLut` so any sample can share it; we just hand it the embedded
// SPIR-V and attach the resulting image to `texturesHdr[HDR_SHEEN_INDEX]`.
void Rasterizer::createSheenLut(Resources& resources)
{
  if(m_sheenLut.image != VK_NULL_HANDLE)
    return;

  // SPIR-V from the compiled `nvshaders/hdr_charlie_brdf_lut.slang` (embedded via autogen header).
  const std::span<const uint32_t> spirv(hdr_charlie_brdf_lut_slang, hdr_charlie_brdf_lut_slang_sizeInBytes / sizeof(uint32_t));

  m_sheenLut = nvshaders::bakeCharlieSheenLut(resources.allocator, resources.samplerPool, resources.app->getQueue(0), spirv);
  NVVK_DBG_NAME(m_sheenLut.image);

  // Wire the LUT into the eTexturesHdr array at HDR_SHEEN_INDEX.
  nvvk::WriteSetContainer write{};
  VkWriteDescriptorSet    ws = resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eTexturesHdr,
                                                                          resources.descriptorSet, HDR_SHEEN_INDEX, 1U);
  write.append(ws, m_sheenLut);
  vkUpdateDescriptorSets(m_device, write.size(), write.data(), 0, nullptr);
}

void Rasterizer::destroySheenLut(Resources& resources)
{
  if(m_sheenLut.image == VK_NULL_HANDLE)
    return;
  // Release the sampler refcount (see destroyOpaqueColorImage for why).
  resources.samplerPool.releaseSampler(m_sheenLut.descriptor.sampler);
  m_sheenLut.descriptor.sampler = VK_NULL_HANDLE;
  resources.allocator.destroyImage(m_sheenLut);
}

//--------------------------------------------------------------------------------------------------
// Resolve eImgRendered (full render-target resolution) into m_opaqueColorImage mip 0 (1024x1024)
// with linear filtering, then generate the mip chain. After this call the image is left in
// SHADER_READ_ONLY_OPTIMAL layout, ready to be sampled by the transmission pass.
void Rasterizer::captureAndMipOpaqueColor(VkCommandBuffer cmd, Resources& resources)
{
  NVVK_DBG_SCOPE(cmd);
  const uint32_t mips = m_opaqueColorImage.mipLevels;

  // Source image is whichever color sink was active during the opaque pass: the inner
  // GBuffer color (Dlss-owned, Kind::SR) when USE_DLSS=ON, else eImgRendered. Both are
  // R32G32B32A32_SFLOAT and get the same TRANSFER_SRC_OPTIMAL transition.
  VkImage    src     = resources.gBuffers.getColorImage(Resources::eImgRendered);
  VkExtent2D srcSize = resources.gBuffers.getSize();
#if defined(USE_DLSS)
  src     = m_dlss->getSrImage(Dlss::SrSlot::eColor);
  srcSize = m_dlss->getRenderSize();  // inner extent in upscale modes; equals outer in DLAA / Off.
#endif

  // Source image was last written by the opaque rendering pass; expect COLOR_ATTACHMENT_OPTIMAL
  // when the secondary rendering pass ends. Transition source to TRANSFER_SRC_OPTIMAL.
  nvvk::cmdImageMemoryBarrier(cmd, {src, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});

  // Destination: transition the entire mip chain to TRANSFER_DST_OPTIMAL.
  nvvk::cmdImageMemoryBarrier(cmd, {m_opaqueColorImage.image,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1}});

  // Blit the rendered framebuffer to mip 0 of the opaque-color capture (downscale to 1024x1024).
  VkImageBlit2 region{
      .sType          = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffsets     = {{0, 0, 0}, {int32_t(srcSize.width), int32_t(srcSize.height), 1}},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffsets     = {{0, 0, 0}, {int32_t(OPAQUE_COLOR_SIZE), int32_t(OPAQUE_COLOR_SIZE), 1}},
  };
  VkBlitImageInfo2 blitInfo{
      .sType          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
      .srcImage       = src,
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .dstImage       = m_opaqueColorImage.image,
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount    = 1,
      .pRegions       = &region,
      .filter         = VK_FILTER_LINEAR,
  };
  vkCmdBlitImage2(cmd, &blitInfo);

  // Mip 0 is now populated; nvvk::cmdGenerateMipmaps fills levels 1..mips-1. Note that the
  // helper uses its `currentLayout` parameter as BOTH the required input layout for all mips
  // AND the layout it leaves them in on exit, so passing TRANSFER_DST_OPTIMAL here means a
  // subsequent barrier is required to reach SHADER_READ_ONLY_OPTIMAL for sampling.
  if(mips > 1)
  {
    nvvk::cmdGenerateMipmaps(cmd, m_opaqueColorImage.image, {OPAQUE_COLOR_SIZE, OPAQUE_COLOR_SIZE}, mips, 1,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  }
  nvvk::cmdImageMemoryBarrier(cmd, {m_opaqueColorImage.image,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1}});
  m_opaqueColorImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  // Restore the source framebuffer to COLOR_ATTACHMENT_OPTIMAL for the second rendering pass.
  nvvk::cmdImageMemoryBarrier(cmd, {src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
}

//--------------------------------------------------------------------------------------------------
// Sort the blend (transparent) nodes back-to-front against the current view matrix.
// Sorting is done on the AABB centroid taken from the POSITION accessor's min/max attributes,
// which nvvkgltf::Scene caches once per primitive so we don't re-parse tinygltf accessors here.
// Returns true when the order changed compared to the previous frame so callers can invalidate
// any cached secondary command buffer.
//
bool Rasterizer::updateSortedBlendNodes(VkCommandBuffer cmd, Resources& resources)
{
  auto timerSection = m_profiler->cmdFrameSection(cmd, "Sorting blend nodes");

  nvvkgltf::Scene* scenePtr = resources.getScene();
  if(!scenePtr)
  {
    m_sceneHasTransmission = false;
    const bool changed     = !m_sortedBlendNodes.empty();
    m_sortedBlendNodes.clear();
    return changed;
  }

  // Scene-wide query: only materials with non-zero KHR_materials_transmission need the
  // screen-space-refraction capture pass. Plain alpha-blend scenes skip the extra pass.
  m_sceneHasTransmission = scenePtr->hasTransmissionMaterials();

  const std::vector<uint32_t>& blendNodes          = scenePtr->getShadedNodes(nvvkgltf::Scene::eRasterBlend);
  const uint64_t               shadedNodesRevision = scenePtr->getShadedNodesRevision();
  const uint64_t               sceneGraphRevision  = scenePtr->getSceneGraphRevision();

  if(blendNodes.empty())
  {
    const bool changed = !m_sortedBlendNodes.empty();
    m_sortedBlendNodes.clear();
    // Keep the memos fresh so the next non-empty frame reseeds correctly.
    m_lastShadedNodesRevision = shadedNodesRevision;
    m_lastSceneGraphRevision  = sceneGraphRevision;
    return changed;
  }

  const glm::mat4& viewMatrix = resources.cameraManip->getViewMatrix();

  // Short-circuit: camera is stationary, blend-bucket membership is unchanged, and nothing
  // that would move a world matrix (animation step / transform edit / scene reload) happened.
  // This is the common desktop-viewer case. Cross-scene safety (new Scene instances reusing an
  // old one's starting revisions) is handled by GltfRenderer::cleanupScene() calling
  // onSceneInvalidated() before the previous Scene is destroyed.
  if(!m_sortedBlendNodes.empty() && viewMatrix == m_lastViewMatrix && shadedNodesRevision == m_lastShadedNodesRevision
     && sceneGraphRevision == m_lastSceneGraphRevision && m_sortedBlendNodes.size() == blendNodes.size())
  {
    return false;
  }

  // Reseed the persistent order whenever blend-bucket membership may have changed. Triggers:
  //  - Scene reload / setCurrentScene / mergeScene  → sceneGraphRevision bumps (traverseSceneWithVisibility).
  //  - KHR_materials_variants switch that flipped materialIDs  → sceneGraphRevision bumps.
  //  - SceneEditor::setPrimitiveMaterial                       → sceneGraphRevision bumps.
  //  - Any cache-level rebuild (e.g. material bucket key flip) → shadedNodesRevision bumps.
  //  - A size mismatch (defensive).
  if(shadedNodesRevision != m_lastShadedNodesRevision || sceneGraphRevision != m_lastSceneGraphRevision
     || m_sortedBlendNodes.size() != blendNodes.size())
  {
    m_sortedBlendNodes.assign(blendNodes.begin(), blendNodes.end());
  }

  const std::vector<nvvkgltf::RenderNode>& renderNodes = scenePtr->getRenderNodes();
  const std::vector<glm::vec3>&            primCenters = scenePtr->getRenderPrimCenterObj();

  // Row 2 of viewMatrix pulls the view-space Z coordinate out of a vec4 via a single dot product.
  // In glm's column-major layout row r, column c lives at viewMatrix[c][r].
  const glm::vec4 viewRow2{viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2], viewMatrix[3][2]};

  std::vector<float> depths(m_sortedBlendNodes.size());
  for(size_t i = 0; i < m_sortedBlendNodes.size(); ++i)
  {
    const uint32_t nodeID = m_sortedBlendNodes[i];
    if(nodeID >= renderNodes.size())
    {
      depths[i] = 0.f;
      continue;
    }
    const nvvkgltf::RenderNode& rnode = renderNodes[nodeID];
    if(rnode.renderPrimID < 0 || size_t(rnode.renderPrimID) >= primCenters.size())
    {
      depths[i] = 0.f;
      continue;
    }
    const glm::vec4 centerWS = rnode.worldMatrix * glm::vec4(primCenters[rnode.renderPrimID], 1.f);
    // Right-handed view: -Z is forward, so more-negative VS-Z = further. Negate so larger depth
    // = further and the back-to-front order is "descending by depth".
    depths[i] = -glm::dot(viewRow2, centerWS);
  }

  // Insertion sort (descending). Skips all work when the input is already sorted; records a
  // local `changed` flag as soon as any element actually moves, so we avoid the old full-vector
  // comparison against a separate "last frame" copy.
  bool changed = false;
  for(size_t i = 1; i < m_sortedBlendNodes.size(); ++i)
  {
    const float    keyDepth = depths[i];
    const uint32_t keyNode  = m_sortedBlendNodes[i];
    size_t         j        = i;
    while(j > 0 && depths[j - 1] < keyDepth)
    {
      depths[j]             = depths[j - 1];
      m_sortedBlendNodes[j] = m_sortedBlendNodes[j - 1];
      --j;
    }
    if(j != i)
    {
      depths[j]             = keyDepth;
      m_sortedBlendNodes[j] = keyNode;
      changed               = true;
    }
  }

  m_lastViewMatrix          = viewMatrix;
  m_lastShadedNodesRevision = shadedNodesRevision;
  m_lastSceneGraphRevision  = sceneGraphRevision;
  return changed;
}

//--------------------------------------------------------------------------------------------------
// Raster commands are recorded to be replayed, this allocates that command buffer
//
void Rasterizer::createRecordCommandBuffer()
{
  VkCommandBufferAllocateInfo alloc_info{
      .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool        = m_commandPool,
      .level              = VK_COMMAND_BUFFER_LEVEL_SECONDARY,
      .commandBufferCount = 1,
  };
  vkAllocateCommandBuffers(m_device, &alloc_info, &m_recordedSceneCmd);
}

//--------------------------------------------------------------------------------------------------
// Free the raster recorded command buffer without stalling the queue.
//
// The secondary command buffer may still be referenced by an in-flight primary. Instead of
// draining the queue with vkQueueWaitIdle (which used to fire every time the transparent draw
// order flipped, the wireframe toggle changed, or a shader recompile landed), we hand the free
// to Application::submitResourceFree() -- it runs the closure after the current frame-in-flight
// slot comes back around, which is guaranteed to be past any work that referenced this buffer.
//
void Rasterizer::freeRecordCommandBuffer(Resources& resources)
{
  if(m_recordedSceneCmd == VK_NULL_HANDLE)
    return;

  VkDevice        device   = m_device;
  VkCommandPool   pool     = m_commandPool;
  VkCommandBuffer freedCmd = m_recordedSceneCmd;
  resources.app->submitResourceFree([device, pool, freedCmd]() { vkFreeCommandBuffers(device, pool, 1, &freedCmd); });
  m_recordedSceneCmd = VK_NULL_HANDLE;
}

//--------------------------------------------------------------------------------------------------
// Drop all Scene-tied state. Called from GltfRenderer::cleanupScene() before the current
// nvvkgltf::Scene instance is destroyed so the next rendered frame doesn't accidentally reuse
// stale render-node indices or short-circuit memos from the previous Scene. Also frees the
// recorded secondary command buffer since it baked draws from the outgoing Scene.
//
void Rasterizer::onSceneInvalidated(Resources& resources)
{
  freeRecordCommandBuffer(resources);
  m_sortedBlendNodes.clear();
  m_lastViewMatrix          = glm::mat4(0.f);
  m_lastShadedNodesRevision = 0;
  m_lastSceneGraphRevision  = 0;
#if defined(USE_DLSS)
  // Mirror PathTracer::onSceneInvalidated(): drop DLSS-SR/DLAA temporal history so the new
  // scene's first frames don't ghost in pixels from the outgoing scene.
  m_dlss->notifyReset();
#endif
}