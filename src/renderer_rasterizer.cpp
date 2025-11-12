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

#include <nvapp/elem_dbgprintf.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/parameter_registry.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/helpers.hpp>

#include "renderer_rasterizer.hpp"

// Pre-compiled shaders
#include "_autogen/gltf_raster.slang.h"
#include "_autogen/sky_physical.slang.h"

#include "nvvk/default_structs.hpp"


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
}

//--------------------------------------------------------------------------------------------------
// Register command line parameters for the Rasterizer
void Rasterizer::registerParameters(nvutils::ParameterRegistry* paramReg)
{
  // Rasterizer-specific command line parameters
  paramReg->add({"rasterWireframe", "Rasterizer: Enable wireframe mode"}, &m_enableWireframe);
  paramReg->add({"rasterUseRecordedCmd", "Rasterizer: Use recorded command buffers"}, &m_useRecordedCmd);
}

//--------------------------------------------------------------------------------------------------
// Clean up rasterizer resources
// Destroys pipeline layout and shaders, and deinitializes the sky physical model
void Rasterizer::onDetach(Resources& resources)
{
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
  freeRecordCommandBuffer();
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
    if(PE::Checkbox("Wireframe", &m_enableWireframe))
    {
      freeRecordCommandBuffer();
    }
    PE::Checkbox("Use Recorded Cmd", &m_useRecordedCmd, "Use recorded command buffers for better performance");
    PE::end();
  }

  return false;
}

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


  // Rendering the environment
  if(!resources.settings.useSolidBackground)
  {
    glm::mat4 viewMatrix = resources.cameraManip->getViewMatrix();
    glm::mat4 projMatrix = resources.cameraManip->getPerspectiveMatrix();

    // Rendering dome or sky in the background, it is covering the entire screen
    if(resources.settings.envSystem == shaderio::EnvSystem::eSky)
    {
      m_skyPhysical.runCompute(cmd, resources.gBuffers.getSize(), viewMatrix, projMatrix, resources.skyParams,
                               resources.gBuffers.getDescriptorImageInfo(Resources::eImgRendered));
    }
    else if(resources.settings.envSystem == shaderio::EnvSystem::eHdr)
    {
      resources.hdrDome.draw(cmd, viewMatrix, projMatrix, resources.gBuffers.getSize(), glm::vec4(resources.settings.hdrEnvIntensity),
                             resources.settings.hdrEnvRotation, resources.settings.hdrBlur);
    }
  }


  // Two attachments, one for color and one for selection
  std::array<VkRenderingAttachmentInfo, 2> attachments = {{DEFAULT_VkRenderingAttachmentInfo, DEFAULT_VkRenderingAttachmentInfo}};
  // 0 - Color attachment
  attachments[0].imageView  = resources.gBuffers.getColorImageView(Resources::eImgRendered);
  attachments[0].clearValue = {{{resources.settings.solidBackgroundColor.x, resources.settings.solidBackgroundColor.y,
                                 resources.settings.solidBackgroundColor.z, 0.f}}};
  attachments[0].loadOp = resources.settings.useSolidBackground ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
  // 1 - Selection attachment
  attachments[1].imageView = resources.gBuffers.getColorImageView(Resources::eImgSelection);
  // X - Depth
  VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
  depthAttachment.imageView                 = resources.gBuffers.getDepthImageView();
  depthAttachment.clearValue                = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

  nvvk::cmdImageMemoryBarrier(cmd, {resources.gBuffers.getColorImage(Resources::eImgRendered), VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

  // Setting up the push constant
  m_pushConst.frameInfo  = (shaderio::SceneFrameInfo*)resources.bFrameInfo.address;
  m_pushConst.skyParams  = (shaderio::SkyPhysicalParameters*)resources.bSkyParams.address;
  m_pushConst.gltfScene  = (shaderio::GltfScene*)resources.sceneVk.sceneDesc().address;
  m_pushConst.mouseCoord = nvapp::ElementDbgPrintf::getMouseCoord();  // Use for debugging: printf in shader
  vkCmdPushConstants(cmd, m_graphicPipelineLayout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof(shaderio::RasterPushConstant), &m_pushConst);


  // Create the rendering info
  VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
  renderingInfo.flags                = m_useRecordedCmd ? VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT : 0,
  renderingInfo.renderArea           = DEFAULT_VkRect2D(resources.gBuffers.getSize());
  renderingInfo.colorAttachmentCount = uint32_t(attachments.size());
  renderingInfo.pColorAttachments    = attachments.data();
  renderingInfo.pDepthAttachment     = &depthAttachment;

  // Scene is recorded to avoid CPU overhead
  if(m_recordedSceneCmd == VK_NULL_HANDLE && m_useRecordedCmd)
  {
    recordRasterScene(resources);
  }


  // ** BEGIN RENDERING **
  vkCmdBeginRendering(cmd, &renderingInfo);

  if(m_useRecordedCmd && m_recordedSceneCmd != VK_NULL_HANDLE)
  {
    vkCmdExecuteCommands(cmd, 1, &m_recordedSceneCmd);  // Execute the recorded command buffer
  }
  else
  {
    renderRasterScene(cmd, resources);  // Render the scene
  }


  vkCmdEndRendering(cmd);

  nvvk::cmdImageMemoryBarrier(cmd, {resources.gBuffers.getColorImage(Resources::eImgRendered),
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
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

  nvvkgltf::Scene&   scene   = resources.scene;
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
// Push descriptor set updates for the rasterizer
// Currently a placeholder for future descriptor set management
void Rasterizer::pushDescriptorSet(VkCommandBuffer cmd, Resources& resources)
{
  //// Use a compute shader that outputs to the render target for now
  //nvvk::WriteSetContainer write{};
  //write.append(resources.descriptorBinding[1].getWriteSet(0), resources.sceneRtx.tlas());
  //write.append(resources.descriptorBinding[1].getWriteSet(1),
  //             resources.gBuffers.getColorImageView(Resources::eImgRendered), VK_IMAGE_LAYOUT_GENERAL);
  //vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 1, write.size(), write.data());
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
}

//--------------------------------------------------------------------------------------------------
// Compile the rasterizer's shaders
// Creates vertex, fragment, and wireframe shaders from the gltf_raster.slang source
void Rasterizer::compileShader(Resources& resources, bool fromFile)
{
  SCOPED_TIMER(__FUNCTION__);

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
void Rasterizer::recordRasterScene(Resources& resources)
{
  SCOPED_TIMER(__FUNCTION__);

  createRecordCommandBuffer();

  std::vector<VkFormat> colorFormat = {resources.gBuffers.getColorFormat(Resources::eImgRendered),
                                       resources.gBuffers.getColorFormat(Resources::eImgSelection)};

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
  renderRasterScene(m_recordedSceneCmd, resources);
  NVVK_CHECK(vkEndCommandBuffer(m_recordedSceneCmd));
}

//--------------------------------------------------------------------------------------------------
// Render the entire scene for raster. Splitting the solid and blend-able element and rendering
// on top, the wireframe if active.
// This is done in a recoded command buffer to be replay
void Rasterizer::renderRasterScene(VkCommandBuffer cmd, Resources& resources)
{

  // Setting up the push constant
  m_pushConst.frameInfo  = (shaderio::SceneFrameInfo*)resources.bFrameInfo.address;
  m_pushConst.skyParams  = (shaderio::SkyPhysicalParameters*)resources.bSkyParams.address;
  m_pushConst.gltfScene  = (shaderio::GltfScene*)resources.sceneVk.sceneDesc().address;
  m_pushConst.mouseCoord = nvapp::ElementDbgPrintf::getMouseCoord();  // Use for debugging: printf in shader
  vkCmdPushConstants(cmd, m_graphicPipelineLayout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof(shaderio::RasterPushConstant), &m_pushConst);

  // All dynamic states are set here
  m_dynamicPipeline.cmdApplyAllStates(cmd);
  m_dynamicPipeline.cmdSetViewportAndScissor(cmd, resources.gBuffers.getSize());
  m_dynamicPipeline.cmdBindShaders(cmd, {.vertex = m_vertexShader, .fragment = m_fragmentShader});
  vkCmdSetDepthTestEnable(cmd, VK_TRUE);

  // Mesh specific vertex input (can be different for each mesh)
  const auto& bindingDescription = std::to_array<VkVertexInputBindingDescription2EXT>({
      {.sType     = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
       .binding   = 0,  // Position buffer binding
       .stride    = sizeof(glm::vec3),
       .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
       .divisor   = 1},
  });

  const auto& attributeDescriptions = std::to_array<VkVertexInputAttributeDescription2EXT>(
      {{.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0}});

  vkCmdSetVertexInputEXT(cmd, uint32_t(bindingDescription.size()), bindingDescription.data(),
                         uint32_t(attributeDescriptions.size()), attributeDescriptions.data());

  // Bind the descriptor set: textures (Set: 0)
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicPipelineLayout, 0, 1, &resources.descriptorSet, 0, nullptr);


  // Draw the scene
  // Back-face culling with depth bias
  vkCmdSetCullMode(cmd, VK_CULL_MODE_BACK_BIT);
  vkCmdSetDepthBias(cmd, -1.0f, 0.0f, 1.0f);  // Apply depth bias for solid objects
  renderNodes(cmd, resources, resources.scene.getShadedNodes(nvvkgltf::Scene::eRasterSolid));

  // Double sided without depth bias
  vkCmdSetCullMode(cmd, VK_CULL_MODE_NONE);
  vkCmdSetDepthBias(cmd, 0.0f, 0.0f, 0.0f);  // Disable depth bias for double-sided objects
  renderNodes(cmd, resources, resources.scene.getShadedNodes(nvvkgltf::Scene::eRasterSolidDoubleSided));

  // Blendable objects without depth bias
  VkBool32 blendEnable  = VK_TRUE;
  VkBool32 blendDisable = VK_FALSE;
  vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blendEnable);
  renderNodes(cmd, resources, resources.scene.getShadedNodes(nvvkgltf::Scene::eRasterBlend));

  if(m_enableWireframe)
  {
    m_dynamicPipeline.cmdBindShaders(cmd, {.vertex = m_vertexShader, .fragment = m_wireframeShader});
    vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &blendDisable);
    vkCmdSetDepthBias(cmd, 0.0f, 0.0f, 0.0f);  // Disable depth bias for wireframe
    vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_LINE);
    renderNodes(cmd, resources, resources.scene.getShadedNodes(nvvkgltf::Scene::eRasterAll));
  }
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
// Freeing the raster recoded command buffer
//
void Rasterizer::freeRecordCommandBuffer()
{
  vkFreeCommandBuffers(m_device, m_commandPool, 1, &m_recordedSceneCmd);
  m_recordedSceneCmd = VK_NULL_HANDLE;
}