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

//
// 3D transform gizmo rendered with Vulkan. Draws translate/rotate/scale
// handles as GPU meshes, performs ray-based hit testing against gizmo
// geometry for mouse interaction, and computes transform deltas that
// are applied to the selected scene node.
//

#include "gizmo_transform_vk.hpp"

#ifndef NDEBUG
#include <imgui.h>
#endif

#include <nvvk/debug_util.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvutils/primitives.hpp>
#include <nvutils/logger.hpp>

#include <glm/gtx/quaternion.hpp>

#include <cfloat>

#include "_autogen/gizmo_visuals.slang.h"

//-----------------------------------------------------------------------------
// Helper: Upload nvutils primitive mesh to GPU
//-----------------------------------------------------------------------------

static void uploadPrimitiveMesh(const nvutils::PrimitiveMesh&                  primMesh,
                                const glm::mat4&                               transform,
                                TransformHelperVk::GizmoComponent              component,
                                TransformHelperVk::GeometryType                type,
                                VkCommandBuffer                                cmd,
                                nvvk::ResourceAllocator*                       alloc,
                                nvvk::StagingUploader*                         uploader,
                                std::vector<TransformHelperVk::GizmoGeometry>& outGeometry)
{
  std::vector<TransformHelperVk::GizmoVertex> vertices;

  glm::vec3 boundsMin(FLT_MAX);
  glm::vec3 boundsMax(-FLT_MAX);

  glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(transform)));

  vertices.reserve(primMesh.vertices.size());
  for(const auto& v : primMesh.vertices)
  {
    glm::vec3 transformedPos = glm::vec3(transform * glm::vec4(v.pos, 1.0f));
    glm::vec3 transformedNrm = glm::normalize(normalMatrix * v.nrm);
    vertices.push_back({transformedPos, transformedNrm});

    boundsMin = glm::min(boundsMin, transformedPos);
    boundsMax = glm::max(boundsMax, transformedPos);
  }

  static_assert(sizeof(primMesh.triangles[0]) == 3 * sizeof(uint32_t), "Triangle struct must be tightly packed");
  const size_t    indexCount = primMesh.triangles.size() * 3;
  const uint32_t* indexData  = reinterpret_cast<const uint32_t*>(primMesh.triangles.data());

  // Pad thin shaft/cube bounds to make picking easier
  if(component == TransformHelperVk::GizmoComponent::eTranslateX || component == TransformHelperVk::GizmoComponent::eTranslateY
     || component == TransformHelperVk::GizmoComponent::eTranslateZ || component == TransformHelperVk::GizmoComponent::eScaleX
     || component == TransformHelperVk::GizmoComponent::eScaleY || component == TransformHelperVk::GizmoComponent::eScaleZ
     || component == TransformHelperVk::GizmoComponent::eScaleUniform)
  {
    glm::vec3 padding(0.05f);
    boundsMin -= padding;
    boundsMax += padding;
  }

  TransformHelperVk::GizmoGeometry geom;
  geom.component  = component;
  geom.type       = type;
  geom.indexCount = static_cast<uint32_t>(indexCount);
  geom.boundsMin  = boundsMin;
  geom.boundsMax  = boundsMax;

  NVVK_CHECK(alloc->createBuffer(geom.vertexBuffer, vertices.size() * sizeof(TransformHelperVk::GizmoVertex),
                                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
  NVVK_CHECK(alloc->createBuffer(geom.indexBuffer, indexCount * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT));

  uploader->appendBuffer(geom.vertexBuffer, 0, std::span(vertices));
  uploader->appendBuffer(geom.indexBuffer, 0, std::span(indexData, indexCount));
  uploader->cmdUploadAppended(cmd);

  NVVK_DBG_NAME(geom.vertexBuffer.buffer);
  NVVK_DBG_NAME(geom.indexBuffer.buffer);

  outGeometry.push_back(geom);
}

//-----------------------------------------------------------------------------
// Lifecycle
//-----------------------------------------------------------------------------

void TransformHelperVk::init(const Resources& res)
{
  m_app      = res.app;
  m_alloc    = res.alloc;
  m_uploader = res.uploader;
  m_device   = res.device;

  createGizmoGeometry();
  createRasterPipeline();

  LOGI("TransformHelperVk initialized\n");
}

void TransformHelperVk::deinit()
{
  clearAttachment();
  destroyRasterPipeline();
  destroyGizmoGeometry();

  m_app      = nullptr;
  m_alloc    = nullptr;
  m_uploader = nullptr;
  m_device   = VK_NULL_HANDLE;

  LOGI("TransformHelperVk deinitialized\n");
}

//-----------------------------------------------------------------------------
// Transform Attachment API (Entity-Agnostic)
//-----------------------------------------------------------------------------

void TransformHelperVk::attachTransform(glm::vec3* position, glm::vec3* rotation, glm::vec3* scale, uint32_t visibilityFlags)
{
  if(!position || !rotation || !scale)
  {
    clearAttachment();
    return;
  }

  m_attachedPosition = position;
  m_attachedRotation = rotation;
  m_attachedScale    = scale;
  m_visibilityFlags  = visibilityFlags;
  m_isDragging       = false;
  m_hoveredComponent = GizmoComponent::eNone;
  m_draggedComponent = GizmoComponent::eNone;
}

bool TransformHelperVk::isAttached() const
{
  return m_attachedPosition != nullptr && m_attachedRotation != nullptr && m_attachedScale != nullptr;
}

void TransformHelperVk::clearAttachment()
{
  m_attachedPosition  = nullptr;
  m_attachedRotation  = nullptr;
  m_attachedScale     = nullptr;
  m_parentWorldMatrix = nullptr;
  m_visibilityFlags   = ShowAll;
  m_isDragging        = false;
  m_hoveredComponent  = GizmoComponent::eNone;
  m_draggedComponent  = GizmoComponent::eNone;
}

//-----------------------------------------------------------------------------
// Configuration
//-----------------------------------------------------------------------------

void TransformHelperVk::setTransformSpace(TransformSpace space)
{
  if(m_space != space)
  {
    m_space = space;
    if(m_isDragging)
      endDrag();
  }
}

void TransformHelperVk::setSnapValues(float translate, float rotate, float scale)
{
  m_snapTranslate = translate;
  m_snapRotate    = rotate;
  m_snapScale     = scale;
}

//-----------------------------------------------------------------------------
// Interaction
//-----------------------------------------------------------------------------

bool TransformHelperVk::processInput(const glm::vec2& mousePos,
                                     const glm::vec2& mouseDelta,
                                     bool             mouseDown,
                                     bool             mousePressed,
                                     bool             mouseReleased,
                                     const glm::mat4& viewMatrix,
                                     const glm::mat4& projMatrix,
                                     const glm::vec2& viewport)
{
  if(!isAttached())
    return false;

  if(mouseReleased && m_isDragging)
  {
    endDrag();
    return false;
  }

  if(mouseDown && m_isDragging)
  {
    updateDrag(mousePos, mouseDelta, viewMatrix, projMatrix, viewport);
    return true;
  }

  if(mousePressed)
  {
    GizmoComponent component = pickGizmoComponent(mousePos, viewMatrix, projMatrix, viewport);
    if(component != GizmoComponent::eNone)
    {
      startDrag(component, mousePos, viewMatrix, projMatrix, viewport);
      return true;
    }
  }

  if(!mouseDown)
  {
    m_hoveredComponent = pickGizmoComponent(mousePos, viewMatrix, projMatrix, viewport);
  }

  return false;
}

//-----------------------------------------------------------------------------
// Rendering
//-----------------------------------------------------------------------------

void TransformHelperVk::renderRaster(VkCommandBuffer  cmd,
                                     VkDescriptorSet  helperDescriptorSet,
                                     const glm::mat4& viewMatrix,
                                     const glm::mat4& projMatrix,
                                     const glm::vec2& viewportSize,
                                     const glm::vec2& depthBufferSize)
{
  if(!isAttached())
    return;

  if(m_vertexShader == VK_NULL_HANDLE)
    return;

  glm::mat4 gizmoTransform = getGizmoTransform();
  glm::vec3 gizmoPosition  = getDisplayPosition();
  float     scale          = getScreenSpaceScale(gizmoPosition, viewMatrix, projMatrix, viewportSize);
  glm::mat4 scaleMatrix    = glm::scale(glm::mat4(1.0f), glm::vec3(scale));
  glm::mat4 modelMatrix    = gizmoTransform * scaleMatrix;
  glm::mat4 mvp            = projMatrix * viewMatrix * modelMatrix;

  m_dynamicPipeline.cmdApplyAllStates(cmd);
  m_dynamicPipeline.cmdApplyDynamicState(cmd, VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE);
  nvvk::GraphicsPipelineState::cmdBindShaders(cmd, {.vertex = m_vertexShader, .fragment = m_fragmentShader});

  VkVertexInputBindingDescription2EXT vertexBinding{
      .sType     = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
      .binding   = 0,
      .stride    = sizeof(GizmoVertex),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
      .divisor   = 1,
  };
  std::array<VkVertexInputAttributeDescription2EXT, 2> vertexAttrs = {{
      {
          .sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
          .location = 0,
          .binding  = 0,
          .format   = VK_FORMAT_R32G32B32_SFLOAT,
          .offset   = offsetof(GizmoVertex, position),
      },
      {
          .sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
          .location = 1,
          .binding  = 0,
          .format   = VK_FORMAT_R32G32B32_SFLOAT,
          .offset   = offsetof(GizmoVertex, normal),
      },
  }};
  vkCmdSetVertexInputEXT(cmd, 1, &vertexBinding, uint32_t(vertexAttrs.size()), vertexAttrs.data());

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_rasterPipelineLayout, 1, 1, &helperDescriptorSet, 0, nullptr);

  shaderio::visual_helpers::PushConstantVisualHelpers pc{};
  pc.mvp             = mvp;
  pc.viewportSize    = viewportSize;
  pc.depthBufferSize = depthBufferSize;
  pc.mode            = shaderio::visual_helpers::HelperMode::eTransform;
  pc.occlusionDither = m_style.occlusionDither ? 1 : 0;

  for(const auto& geom : m_gizmoGeometry)
  {
    if(!shouldRenderGeometry(geom))
      continue;

    bool      isHovered = (geom.component == m_hoveredComponent);
    glm::vec3 color     = getColorFromComponent(geom.component, isHovered);
    float     alpha     = (geom.type == GeometryType::ePlane) ? m_style.planeAlpha : 1.0f;
    pc.color            = glm::vec4(color, alpha);
    pc.componentID      = static_cast<uint32_t>(geom.component);

    vkCmdPushConstants(cmd, m_rasterPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(shaderio::visual_helpers::PushConstantVisualHelpers), &pc);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &geom.vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmd, geom.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(cmd, geom.indexCount, 1, 0, 0, 0);
  }
}

//-----------------------------------------------------------------------------
// Update Notifications
//-----------------------------------------------------------------------------

void TransformHelperVk::notifyExternalTransformChange()
{
  if(m_isDragging)
    endDrag();
}

//-----------------------------------------------------------------------------
// Geometry Generation
//-----------------------------------------------------------------------------

void TransformHelperVk::createGizmoGeometry()
{
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  generateTranslateGizmo(cmd);
  generateRotateGizmo(cmd);
  generateScaleGizmo(cmd);

  m_app->submitAndWaitTempCmdBuffer(cmd);
  m_uploader->releaseStaging();
}

void TransformHelperVk::destroyGizmoGeometry()
{
  for(auto& geom : m_gizmoGeometry)
  {
    if(geom.vertexBuffer.buffer != VK_NULL_HANDLE)
      m_alloc->destroyBuffer(geom.vertexBuffer);
    if(geom.indexBuffer.buffer != VK_NULL_HANDLE)
      m_alloc->destroyBuffer(geom.indexBuffer);
  }
  m_gizmoGeometry.clear();
}

void TransformHelperVk::generateTranslateGizmo(VkCommandBuffer cmd)
{
  const float shaftRadius = m_style.shaftRadius;
  const float shaftLength = m_style.shaftLength;

  nvutils::PrimitiveMesh shaftMesh = nvutils::createCube(1.0f, 1.0f, 1.0f);

  glm::mat4 xShaftTransform = glm::translate(glm::mat4(1.0f), glm::vec3(shaftLength * 0.5f, 0, 0))
                              * glm::scale(glm::mat4(1.0f), glm::vec3(shaftLength, shaftRadius * 2.0f, shaftRadius * 2.0f));
  uploadPrimitiveMesh(shaftMesh, xShaftTransform, GizmoComponent::eTranslateX, GeometryType::eShaft, cmd, m_alloc,
                      m_uploader, m_gizmoGeometry);

  glm::mat4 yShaftTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, shaftLength * 0.5f, 0))
                              * glm::scale(glm::mat4(1.0f), glm::vec3(shaftRadius * 2.0f, shaftLength, shaftRadius * 2.0f));
  uploadPrimitiveMesh(shaftMesh, yShaftTransform, GizmoComponent::eTranslateY, GeometryType::eShaft, cmd, m_alloc,
                      m_uploader, m_gizmoGeometry);

  glm::mat4 zShaftTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, shaftLength * 0.5f))
                              * glm::scale(glm::mat4(1.0f), glm::vec3(shaftRadius * 2.0f, shaftRadius * 2.0f, shaftLength));
  uploadPrimitiveMesh(shaftMesh, zShaftTransform, GizmoComponent::eTranslateZ, GeometryType::eShaft, cmd, m_alloc,
                      m_uploader, m_gizmoGeometry);

  // Cone arrowheads at shaft tips (createConeMesh default: tip at +Y)
  nvutils::PrimitiveMesh coneMesh = nvutils::createConeMesh(m_style.coneRadius, m_style.coneHeight, m_style.coneSegments);

  // X cone: rotate +Y tip to point along +X
  glm::mat4 xConeTransform = glm::translate(glm::mat4(1.0f), glm::vec3(shaftLength, 0, 0))
                             * glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0, 0, 1));
  uploadPrimitiveMesh(coneMesh, xConeTransform, GizmoComponent::eTranslateX, GeometryType::eCone, cmd, m_alloc,
                      m_uploader, m_gizmoGeometry);

  glm::mat4 yConeTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, shaftLength, 0));
  uploadPrimitiveMesh(coneMesh, yConeTransform, GizmoComponent::eTranslateY, GeometryType::eCone, cmd, m_alloc,
                      m_uploader, m_gizmoGeometry);

  // Z cone: rotate +Y tip to point along +Z
  glm::mat4 zConeTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, shaftLength))
                             * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
  uploadPrimitiveMesh(coneMesh, zConeTransform, GizmoComponent::eTranslateZ, GeometryType::eCone, cmd, m_alloc,
                      m_uploader, m_gizmoGeometry);

  const float            planeSize   = m_style.planeSize;
  const float            planeOffset = m_style.planeOffset;
  nvutils::PrimitiveMesh planeMesh   = nvutils::createPlane(1, planeSize, planeSize);

  // XY plane: rotate 90deg around X to stand upright in XY
  glm::mat4 xyTransform = glm::translate(glm::mat4(1.0f), glm::vec3(planeOffset, planeOffset, 0))
                          * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
  uploadPrimitiveMesh(planeMesh, xyTransform, GizmoComponent::eTranslateXY, GeometryType::ePlane, cmd, m_alloc,
                      m_uploader, m_gizmoGeometry);

  // XZ plane: identity (lies flat in XZ)
  glm::mat4 xzTransform = glm::translate(glm::mat4(1.0f), glm::vec3(planeOffset, 0, planeOffset));
  uploadPrimitiveMesh(planeMesh, xzTransform, GizmoComponent::eTranslateXZ, GeometryType::ePlane, cmd, m_alloc,
                      m_uploader, m_gizmoGeometry);

  // YZ plane: rotate 90deg around Z to stand upright in YZ
  glm::mat4 yzTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, planeOffset, planeOffset))
                          * glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 0, 1));
  uploadPrimitiveMesh(planeMesh, yzTransform, GizmoComponent::eTranslateYZ, GeometryType::ePlane, cmd, m_alloc,
                      m_uploader, m_gizmoGeometry);
}

void TransformHelperVk::generateRotateGizmo(VkCommandBuffer cmd)
{
  nvutils::PrimitiveMesh torusMesh =
      nvutils::createTorusMesh(m_style.ringRadius, m_style.ringTubeRadius, m_style.ringMajorSegs, m_style.ringMinorSegs);

  // Each ring is rotated so its normal aligns with the corresponding axis
  glm::mat4 xTransform = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 0, 1));
  uploadPrimitiveMesh(torusMesh, xTransform, GizmoComponent::eRotateX, GeometryType::eTorus, cmd, m_alloc, m_uploader, m_gizmoGeometry);

  glm::mat4 yTransform = glm::mat4(1.0f);
  uploadPrimitiveMesh(torusMesh, yTransform, GizmoComponent::eRotateY, GeometryType::eTorus, cmd, m_alloc, m_uploader, m_gizmoGeometry);

  glm::mat4 zTransform = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(1, 0, 0));
  uploadPrimitiveMesh(torusMesh, zTransform, GizmoComponent::eRotateZ, GeometryType::eTorus, cmd, m_alloc, m_uploader, m_gizmoGeometry);
}

void TransformHelperVk::generateScaleGizmo(VkCommandBuffer cmd)
{
  const float axisLength = m_style.scaleCubeAxisLength;
  const float cubeSize   = m_style.scaleCubeSize;

  nvutils::PrimitiveMesh cubeMesh = nvutils::createCube(cubeSize, cubeSize, cubeSize);

  glm::mat4 xCubeTransform = glm::translate(glm::mat4(1.0f), glm::vec3(axisLength, 0, 0));
  uploadPrimitiveMesh(cubeMesh, xCubeTransform, GizmoComponent::eScaleX, GeometryType::eBox, cmd, m_alloc, m_uploader, m_gizmoGeometry);

  glm::mat4 yCubeTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, axisLength, 0));
  uploadPrimitiveMesh(cubeMesh, yCubeTransform, GizmoComponent::eScaleY, GeometryType::eBox, cmd, m_alloc, m_uploader, m_gizmoGeometry);

  glm::mat4 zCubeTransform = glm::translate(glm::mat4(1.0f), glm::vec3(0, 0, axisLength));
  uploadPrimitiveMesh(cubeMesh, zCubeTransform, GizmoComponent::eScaleZ, GeometryType::eBox, cmd, m_alloc, m_uploader, m_gizmoGeometry);

  glm::mat4 centerTransform = glm::scale(glm::mat4(1.0f), glm::vec3(m_style.centerCubeScale));
  uploadPrimitiveMesh(cubeMesh, centerTransform, GizmoComponent::eScaleUniform, GeometryType::eCenterBox, cmd, m_alloc,
                      m_uploader, m_gizmoGeometry);
}

//-----------------------------------------------------------------------------
// Rendering Pipelines
//-----------------------------------------------------------------------------

void TransformHelperVk::createRasterPipeline()
{
  // Set 0: empty placeholder (shader uses set 1 for scene depth)
  VkDescriptorSetLayoutCreateInfo emptyLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  VkDescriptorSetLayout           emptyLayout = VK_NULL_HANDLE;
  NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &emptyLayoutInfo, nullptr, &emptyLayout));

  // Set 1: scene depth texture + sampler for occlusion testing
  std::array<VkDescriptorSetLayoutBinding, 2> bindings = {{
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
  }};

  VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  layoutInfo.bindingCount = uint32_t(bindings.size());
  layoutInfo.pBindings    = bindings.data();
  NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descriptorSetLayout));
  NVVK_DBG_NAME(m_descriptorSetLayout);

  std::array<VkDescriptorSetLayout, 2> setLayouts = {emptyLayout, m_descriptorSetLayout};

  VkPushConstantRange pushRange = {
      .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .size       = sizeof(shaderio::visual_helpers::PushConstantVisualHelpers),
  };

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutInfo.setLayoutCount         = uint32_t(setLayouts.size());
  pipelineLayoutInfo.pSetLayouts            = setLayouts.data();
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges    = &pushRange;

  NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_rasterPipelineLayout));
  NVVK_DBG_NAME(m_rasterPipelineLayout);

  vkDestroyDescriptorSetLayout(m_device, emptyLayout, nullptr);

  createShaderObjects();

  // Disable culling so gizmo is visible from all angles
  // (rotation transforms can flip winding order, causing some rings to disappear)
  m_dynamicPipeline.rasterizationState.cullMode        = VK_CULL_MODE_NONE;
  m_dynamicPipeline.rasterizationState.frontFace       = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  m_dynamicPipeline.depthStencilState.depthTestEnable  = VK_TRUE;
  m_dynamicPipeline.depthStencilState.depthWriteEnable = VK_TRUE;
  m_dynamicPipeline.depthStencilState.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;
  m_dynamicPipeline.colorBlendEnables[0]               = VK_FALSE;
}

void TransformHelperVk::destroyRasterPipeline()
{
  if(!m_device)
    return;

  vkDestroyShaderEXT(m_device, m_vertexShader, nullptr);
  m_vertexShader = VK_NULL_HANDLE;
  vkDestroyShaderEXT(m_device, m_fragmentShader, nullptr);
  m_fragmentShader = VK_NULL_HANDLE;

  if(m_rasterPipelineLayout != VK_NULL_HANDLE)
  {
    vkDestroyPipelineLayout(m_device, m_rasterPipelineLayout, nullptr);
    m_rasterPipelineLayout = VK_NULL_HANDLE;
  }

  if(m_descriptorSetLayout != VK_NULL_HANDLE)
  {
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
    m_descriptorSetLayout = VK_NULL_HANDLE;
  }
}

//-----------------------------------------------------------------------------
// Picking and Interaction
//-----------------------------------------------------------------------------

TransformHelperVk::GizmoComponent TransformHelperVk::pickGizmoComponent(const glm::vec2& mousePos,
                                                                        const glm::mat4& viewMatrix,
                                                                        const glm::mat4& projMatrix,
                                                                        const glm::vec2& viewport)
{
  if(!isAttached())
    return GizmoComponent::eNone;

  glm::vec3 rayOrigin, rayDir;
  createRayFromMouse(mousePos, viewMatrix, projMatrix, viewport, rayOrigin, rayDir);

  glm::mat4 gizmoTransform = getGizmoTransform();
  glm::vec3 gizmoPosition  = getDisplayPosition();
  float     scale          = getScreenSpaceScale(gizmoPosition, viewMatrix, projMatrix, viewport);
  glm::mat4 scaleMatrix    = glm::scale(glm::mat4(1.0f), glm::vec3(scale));
  glm::mat4 modelMatrix    = gizmoTransform * scaleMatrix;

  // Pre-transform ray into gizmo-local space (same for all components)
  glm::mat4 invModel       = glm::inverse(modelMatrix);
  glm::vec3 localRayOrigin = glm::vec3(invModel * glm::vec4(rayOrigin, 1.0f));
  glm::vec3 localRayDir    = glm::normalize(glm::vec3(invModel * glm::vec4(rayDir, 0.0f)));

  float          closestDist      = FLT_MAX;
  GizmoComponent closestComponent = GizmoComponent::eNone;

  for(const auto& geom : m_gizmoGeometry)
  {
    if(!shouldRenderGeometry(geom))
      continue;

    if(!rayIntersectsBounds(localRayOrigin, localRayDir, geom.boundsMin, geom.boundsMax))
      continue;

    float dist          = 0.0f;
    float pickThreshold = 0.6f;

    switch(geom.component)
    {
      case GizmoComponent::eTranslateX:
      case GizmoComponent::eTranslateY:
      case GizmoComponent::eTranslateZ:
      case GizmoComponent::eScaleX:
      case GizmoComponent::eScaleY:
      case GizmoComponent::eScaleZ:
      case GizmoComponent::eScaleUniform:
        dist          = rayDistanceToBounds(localRayOrigin, localRayDir, geom.boundsMin, geom.boundsMax);
        pickThreshold = 0.6f;
        break;

      case GizmoComponent::eRotateX:
      case GizmoComponent::eRotateY:
      case GizmoComponent::eRotateZ:
        dist          = rayDistanceToTorus(localRayOrigin, localRayDir, geom.component);
        pickThreshold = 0.2f;
        break;

      case GizmoComponent::eTranslateXY:
      case GizmoComponent::eTranslateXZ:
      case GizmoComponent::eTranslateYZ:
        dist          = rayDistanceToBounds(localRayOrigin, localRayDir, geom.boundsMin, geom.boundsMax);
        pickThreshold = 0.2f;
        break;

      default:
        dist          = rayDistanceToBounds(localRayOrigin, localRayDir, geom.boundsMin, geom.boundsMax);
        pickThreshold = 0.6f;
        break;
    }

    if(dist < pickThreshold && dist < closestDist)
    {
      closestDist      = dist;
      closestComponent = geom.component;
    }
  }

  return closestComponent;
}

void TransformHelperVk::startDrag(GizmoComponent   component,
                                  const glm::vec2& mousePos,
                                  const glm::mat4& viewMatrix,
                                  const glm::mat4& projMatrix,
                                  const glm::vec2& viewport)
{
  m_isDragging        = true;
  m_draggedComponent  = component;
  m_dragStartPosMouse = mousePos;

  if(isAttached())
  {
    m_dragStartPosition      = *m_attachedPosition;
    m_dragStartWorldPosition = getDisplayPosition();
    m_dragStartRotation      = *m_attachedRotation;
    m_dragStartScale         = *m_attachedScale;
  }

  // Compute initial hit point for relative dragging (world space)
  glm::vec3 rayOrigin, rayDir;
  createRayFromMouse(mousePos, viewMatrix, projMatrix, viewport, rayOrigin, rayDir);

  switch(component)
  {
    case GizmoComponent::eTranslateX:
    case GizmoComponent::eTranslateY:
    case GizmoComponent::eTranslateZ: {
      glm::vec3 worldAxis = localAxisToWorld(getAxisFromComponent(component));
      glm::vec3 worldPos  = m_dragStartWorldPosition;

      // Best-view drag plane: perpendicular to the axis, tilted toward the camera
      glm::vec3 viewDir     = glm::normalize(glm::vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]));
      glm::vec3 planeNormal = glm::normalize(glm::cross(worldAxis, glm::cross(viewDir, worldAxis)));

      float t = 0.f;
      if(rayIntersectPlane(rayOrigin, rayDir, worldPos, planeNormal, t))
        m_dragStartHitPoint = rayOrigin + rayDir * t;
      else
        m_dragStartHitPoint = worldPos;
      break;
    }

    case GizmoComponent::eTranslateXY:
    case GizmoComponent::eTranslateXZ:
    case GizmoComponent::eTranslateYZ: {
      glm::vec3 worldNormal = localAxisToWorld(getAxisFromComponent(component));
      glm::vec3 worldPos    = m_dragStartWorldPosition;

      float t = 0.f;
      if(rayIntersectPlane(rayOrigin, rayDir, worldPos, worldNormal, t))
        m_dragStartHitPoint = rayOrigin + rayDir * t;
      else
        m_dragStartHitPoint = worldPos;
      break;
    }

    default:
      m_dragStartHitPoint = m_dragStartWorldPosition;
      break;
  }

  if(m_onTransformBegin)
    m_onTransformBegin();
}

void TransformHelperVk::updateDrag(const glm::vec2& mousePos,
                                   const glm::vec2& mouseDelta,
                                   const glm::mat4& viewMatrix,
                                   const glm::mat4& projMatrix,
                                   const glm::vec2& viewport)
{
  if(!m_isDragging || !isAttached())
    return;

  glm::vec3 rayOrigin, rayDir;
  createRayFromMouse(mousePos, viewMatrix, projMatrix, viewport, rayOrigin, rayDir);

  switch(m_draggedComponent)
  {
    case GizmoComponent::eTranslateX:
    case GizmoComponent::eTranslateY:
    case GizmoComponent::eTranslateZ: {
      glm::vec3 worldAxis = localAxisToWorld(getAxisFromComponent(m_draggedComponent));
      glm::vec3 worldPos  = m_dragStartWorldPosition;

      glm::vec3 viewDir     = glm::normalize(glm::vec3(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2]));
      glm::vec3 planeNormal = glm::normalize(glm::cross(worldAxis, glm::cross(viewDir, worldAxis)));

      float t = 0.f;
      if(rayIntersectPlane(rayOrigin, rayDir, worldPos, planeNormal, t))
      {
        glm::vec3 currentHitPoint = rayOrigin + rayDir * t;
        glm::vec3 worldDelta      = currentHitPoint - m_dragStartHitPoint;
        float     projection      = glm::dot(worldDelta, worldAxis);
        glm::vec3 worldMovement   = worldAxis * projection;

        *m_attachedPosition = m_dragStartPosition + worldDeltaToLocal(worldMovement);

        if(m_onTransformChange)
          m_onTransformChange();
      }
      break;
    }

    case GizmoComponent::eTranslateXY:
    case GizmoComponent::eTranslateXZ:
    case GizmoComponent::eTranslateYZ: {
      glm::vec3 worldNormal = localAxisToWorld(getAxisFromComponent(m_draggedComponent));
      glm::vec3 worldPos    = m_dragStartWorldPosition;

      float t = 0.f;
      if(rayIntersectPlane(rayOrigin, rayDir, worldPos, worldNormal, t))
      {
        glm::vec3 currentHitPoint = rayOrigin + rayDir * t;
        glm::vec3 worldDelta      = currentHitPoint - m_dragStartHitPoint;

        *m_attachedPosition = m_dragStartPosition + worldDeltaToLocal(worldDelta);

        if(m_onTransformChange)
          m_onTransformChange();
      }
      break;
    }

    case GizmoComponent::eRotateX:
    case GizmoComponent::eRotateY:
    case GizmoComponent::eRotateZ: {
      glm::vec3 localAxis = getAxisFromComponent(m_draggedComponent);
      glm::vec3 worldAxis = localAxisToWorld(localAxis);
      glm::vec3 worldPos  = m_dragStartWorldPosition;

      float t = 0.f;
      if(rayIntersectPlane(rayOrigin, rayDir, worldPos, worldAxis, t))
      {
        glm::vec3 hitPoint = rayOrigin + rayDir * t;
        glm::vec3 toHit    = hitPoint - worldPos;

        glm::vec3 startRayOrigin, startRayDir;
        createRayFromMouse(m_dragStartPosMouse, viewMatrix, projMatrix, viewport, startRayOrigin, startRayDir);

        float startT = 0.f;
        if(rayIntersectPlane(startRayOrigin, startRayDir, worldPos, worldAxis, startT))
        {
          glm::vec3 startHitPoint = startRayOrigin + startRayDir * startT;
          glm::vec3 toStartHit    = startHitPoint - worldPos;

          float cosAngle = glm::dot(glm::normalize(toStartHit), glm::normalize(toHit));
          cosAngle       = glm::clamp(cosAngle, -1.0f, 1.0f);
          float angleDeg = glm::degrees(std::acos(cosAngle));

          glm::vec3 cross = glm::cross(toStartHit, toHit);
          if(glm::dot(cross, worldAxis) < 0.0f)
            angleDeg = -angleDeg;

          *m_attachedRotation = m_dragStartRotation + localAxis * angleDeg;

          if(m_onTransformChange)
            m_onTransformChange();
        }
      }
      break;
    }

    case GizmoComponent::eScaleX:
    case GizmoComponent::eScaleY:
    case GizmoComponent::eScaleZ: {
      float scaleFactor = computeScaleFactor(mousePos);

      int axisIndex = (m_draggedComponent == GizmoComponent::eScaleX) ? 0 :
                      (m_draggedComponent == GizmoComponent::eScaleY) ? 1 :
                                                                        2;

      *m_attachedScale              = m_dragStartScale;
      (*m_attachedScale)[axisIndex] = m_dragStartScale[axisIndex] * scaleFactor;

      if(m_onTransformChange)
        m_onTransformChange();
      break;
    }

    case GizmoComponent::eScaleUniform: {
      float scaleFactor = computeScaleFactor(mousePos);

      *m_attachedScale = m_dragStartScale * scaleFactor;

      if(m_onTransformChange)
        m_onTransformChange();
      break;
    }

    default:
      break;
  }
}

void TransformHelperVk::endDrag()
{
  m_isDragging       = false;
  m_draggedComponent = GizmoComponent::eNone;

  if(m_onTransformEnd)
    m_onTransformEnd();
}

float TransformHelperVk::computeScaleFactor(const glm::vec2& mousePos) const
{
  glm::vec2 totalDelta = mousePos - m_dragStartPosMouse;
  float     factor     = 1.0f + (totalDelta.x - totalDelta.y) * 0.005f;
  return glm::max(0.01f, factor);
}

//-----------------------------------------------------------------------------
// Picking Helper Functions
//-----------------------------------------------------------------------------

void TransformHelperVk::createRayFromMouse(const glm::vec2& mousePos,
                                           const glm::mat4& viewMatrix,
                                           const glm::mat4& projMatrix,
                                           const glm::vec2& viewport,
                                           glm::vec3&       rayOrigin,
                                           glm::vec3&       rayDir) const
{
  // Vulkan NDC: X [-1,1] left to right, Y [-1,1] top to bottom, Z [0,1] near to far
  float ndcX = (2.0f * mousePos.x) / viewport.x - 1.0f;
  float ndcY = (2.0f * mousePos.y) / viewport.y - 1.0f;

  glm::mat4 invView = glm::inverse(viewMatrix);
  glm::mat4 invProj = glm::inverse(projMatrix);

  glm::vec4 rayStartNDC(ndcX, ndcY, 0.0f, 1.0f);
  glm::vec4 rayEndNDC(ndcX, ndcY, 1.0f, 1.0f);

  glm::vec4 rayStartView = invProj * rayStartNDC;
  glm::vec4 rayEndView   = invProj * rayEndNDC;

  rayStartView /= rayStartView.w;
  rayEndView /= rayEndView.w;

  glm::vec4 rayStartWorld = invView * rayStartView;
  glm::vec4 rayEndWorld   = invView * rayEndView;

  rayOrigin = glm::vec3(rayStartWorld);
  rayDir    = glm::normalize(glm::vec3(rayEndWorld - rayStartWorld));
}

bool TransformHelperVk::rayIntersectsBounds(const glm::vec3& rayOrigin,
                                            const glm::vec3& rayDir,
                                            const glm::vec3& boundsMin,
                                            const glm::vec3& boundsMax) const
{
  // Slab method for AABB-ray intersection
  glm::vec3 invDir = glm::vec3(1.0f) / rayDir;
  glm::vec3 t0s    = (boundsMin - rayOrigin) * invDir;
  glm::vec3 t1s    = (boundsMax - rayOrigin) * invDir;

  glm::vec3 tsmaller = glm::min(t0s, t1s);
  glm::vec3 tbigger  = glm::max(t0s, t1s);

  float tmin = glm::max(glm::max(tsmaller.x, tsmaller.y), tsmaller.z);
  float tmax = glm::min(glm::min(tbigger.x, tbigger.y), tbigger.z);

  return tmax >= tmin && tmax >= 0.0f;
}

float TransformHelperVk::rayDistanceToBounds(const glm::vec3& rayOrigin,
                                             const glm::vec3& rayDir,
                                             const glm::vec3& boundsMin,
                                             const glm::vec3& boundsMax) const
{
  glm::vec3 center = (boundsMin + boundsMax) * 0.5f;

  glm::vec3 toCenter     = center - rayOrigin;
  float     rayT         = glm::max(0.0f, glm::dot(toCenter, rayDir));
  glm::vec3 closestOnRay = rayOrigin + rayDir * rayT;
  glm::vec3 closestOnBox = glm::clamp(closestOnRay, boundsMin, boundsMax);

  return glm::length(closestOnRay - closestOnBox);
}

float TransformHelperVk::rayDistanceToTorus(const glm::vec3& rayOrigin, const glm::vec3& rayDir, GizmoComponent component) const
{
  glm::vec3 axis   = getAxisFromComponent(component);
  glm::vec3 center = glm::vec3(0.0f);
  float     radius = m_style.ringRadius;

  // Intersect the plane containing the torus, then measure distance to ring radius
  float t = 0.f;
  if(!rayIntersectPlane(rayOrigin, rayDir, center, axis, t))
    return FLT_MAX;

  glm::vec3 hitPoint       = rayOrigin + rayDir * t;
  float     distFromCenter = glm::length(hitPoint - center);

  return glm::abs(distFromCenter - radius);
}

//-----------------------------------------------------------------------------
// Utility Functions
//-----------------------------------------------------------------------------

glm::mat4 TransformHelperVk::getGizmoTransform() const
{
  if(!isAttached())
    return {1.0f};

  glm::vec3 position       = getAttachedPosition();
  glm::quat rotation       = getAttachedRotation();
  glm::mat4 localTransform = glm::translate(glm::mat4(1.0f), position) * glm::mat4_cast(rotation);

  if(m_parentWorldMatrix)
  {
    // In TRS decomposition, T is in parent space and independent of the object's
    // own rotation. Use parent orientation only so gizmo axes match the drag
    // directions computed by localAxisToWorld/worldDeltaToLocal.
    glm::vec3 worldPos = glm::vec3(*m_parentWorldMatrix * glm::vec4(position, 1.0f));
    glm::vec3 axisX    = glm::normalize(glm::vec3((*m_parentWorldMatrix)[0]));
    glm::vec3 axisY    = glm::normalize(glm::vec3((*m_parentWorldMatrix)[1]));
    glm::vec3 axisZ    = glm::normalize(glm::vec3((*m_parentWorldMatrix)[2]));
    return glm::mat4(glm::vec4(axisX, 0), glm::vec4(axisY, 0), glm::vec4(axisZ, 0), glm::vec4(worldPos, 1));
  }

  if(m_space == TransformSpace::eWorld)
    return glm::translate(glm::mat4(1.0f), position);

  return localTransform;
}

glm::vec3 TransformHelperVk::getAttachedPosition() const
{
  if(isAttached())
    return *m_attachedPosition;
  return glm::vec3(0.0f);
}

glm::vec3 TransformHelperVk::getDisplayPosition() const
{
  if(m_parentWorldMatrix)
  {
    glm::vec3 localPos = getAttachedPosition();
    return {*m_parentWorldMatrix * glm::vec4(localPos, 1.0f)};
  }
  return getAttachedPosition();
}

glm::vec3 TransformHelperVk::localAxisToWorld(const glm::vec3& localAxis) const
{
  if(m_parentWorldMatrix)
    return glm::normalize(glm::mat3(*m_parentWorldMatrix) * localAxis);
  return localAxis;
}

glm::vec3 TransformHelperVk::worldDeltaToLocal(const glm::vec3& worldDelta) const
{
  if(m_parentWorldMatrix)
    return {glm::inverse(*m_parentWorldMatrix) * glm::vec4(worldDelta, 0.0f)};
  return worldDelta;
}

glm::quat TransformHelperVk::getAttachedRotation() const
{
  if(!isAttached())
    return {1, 0, 0, 0};

  glm::vec3 radians = glm::radians(*m_attachedRotation);
  return {radians};
}

glm::vec3 TransformHelperVk::getAttachedScale() const
{
  if(isAttached())
    return *m_attachedScale;
  return glm::vec3(1.0f);
}

float TransformHelperVk::getScreenSpaceScale(const glm::vec3& worldPos,
                                             const glm::mat4& viewMatrix,
                                             const glm::mat4& projMatrix,
                                             const glm::vec2& viewport) const
{
  glm::vec4 clipPos = projMatrix * viewMatrix * glm::vec4(worldPos, 1.0f);
  float     depth   = clipPos.w;
  return (m_style.sizePixels / viewport.y) * depth;
}

bool TransformHelperVk::rayIntersectPlane(const glm::vec3& rayOrigin,
                                          const glm::vec3& rayDir,
                                          const glm::vec3& planePoint,
                                          const glm::vec3& planeNormal,
                                          float&           t) const
{
  float denom = glm::dot(rayDir, planeNormal);
  if(glm::abs(denom) < 1e-6f)
    return false;

  t = glm::dot(planePoint - rayOrigin, planeNormal) / denom;
  return t >= 0.0f;
}

glm::vec3 TransformHelperVk::getAxisFromComponent(GizmoComponent component) const
{
  switch(component)
  {
    case GizmoComponent::eTranslateX:
    case GizmoComponent::eRotateX:
    case GizmoComponent::eScaleX:
      return {1, 0, 0};
    case GizmoComponent::eTranslateY:
    case GizmoComponent::eRotateY:
    case GizmoComponent::eScaleY:
      return {0, 1, 0};
    case GizmoComponent::eTranslateZ:
    case GizmoComponent::eRotateZ:
    case GizmoComponent::eScaleZ:
      return {0, 0, 1};
    case GizmoComponent::eTranslateXY:
      return {0, 0, 1};  // Z normal
    case GizmoComponent::eTranslateXZ:
      return {0, 1, 0};  // Y normal
    case GizmoComponent::eTranslateYZ:
      return {1, 0, 0};  // X normal
    default:
      return {0, 0, 0};
  }
}

bool TransformHelperVk::shouldRenderGeometry(const GizmoGeometry& geom) const
{
  if(geom.component == GizmoComponent::eNone || geom.component == GizmoComponent::eRotateScreen)
    return false;

  switch(geom.component)
  {
    case GizmoComponent::eTranslateX:
    case GizmoComponent::eTranslateY:
    case GizmoComponent::eTranslateZ:
    case GizmoComponent::eTranslateXY:
    case GizmoComponent::eTranslateXZ:
    case GizmoComponent::eTranslateYZ:
      return (m_visibilityFlags & ShowTranslation) != 0;

    case GizmoComponent::eRotateX:
    case GizmoComponent::eRotateY:
    case GizmoComponent::eRotateZ:
      return (m_visibilityFlags & ShowRotation) != 0;

    case GizmoComponent::eScaleX:
    case GizmoComponent::eScaleY:
    case GizmoComponent::eScaleZ:
    case GizmoComponent::eScaleUniform:
      return (m_visibilityFlags & ShowScale) != 0;

    default:
      return true;
  }
}

glm::vec3 TransformHelperVk::getColorFromComponent(GizmoComponent component, bool hovered) const
{
  glm::vec3 color;

  switch(component)
  {
    case GizmoComponent::eTranslateX:
    case GizmoComponent::eRotateX:
    case GizmoComponent::eScaleX:
      color = m_style.colorX;
      break;
    case GizmoComponent::eTranslateY:
    case GizmoComponent::eRotateY:
    case GizmoComponent::eScaleY:
      color = m_style.colorY;
      break;
    case GizmoComponent::eTranslateZ:
    case GizmoComponent::eRotateZ:
    case GizmoComponent::eScaleZ:
      color = m_style.colorZ;
      break;
    case GizmoComponent::eTranslateXY:
      color = m_style.colorPlaneXY;
      break;
    case GizmoComponent::eTranslateXZ:
      color = m_style.colorPlaneXZ;
      break;
    case GizmoComponent::eTranslateYZ:
      color = m_style.colorPlaneYZ;
      break;
    case GizmoComponent::eRotateScreen:
      color = m_style.colorScreenRotation;
      break;
    case GizmoComponent::eScaleUniform:
      color = m_style.colorScaleUniform;
      break;
    default:
      color = glm::vec3(0.5f);
      break;
  }

  if(hovered)
    color = glm::mix(color, glm::clamp(color * m_style.hoverBrightness, 0.0f, 1.0f), m_style.hoverMix);

  return color;
}

//-----------------------------------------------------------------------------
// Shader Objects (VK_EXT_shader_object)
//-----------------------------------------------------------------------------

void TransformHelperVk::createShaderObjects()
{
  VkDescriptorSetLayoutCreateInfo emptyLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  VkDescriptorSetLayout           emptyLayout = VK_NULL_HANDLE;
  NVVK_CHECK(vkCreateDescriptorSetLayout(m_device, &emptyLayoutInfo, nullptr, &emptyLayout));

  VkDescriptorSetLayout setLayouts[2] = {emptyLayout, m_descriptorSetLayout};
  VkPushConstantRange   pushRange     = {
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .size       = sizeof(shaderio::visual_helpers::PushConstantVisualHelpers),
  };

  VkShaderCreateInfoEXT shaderInfo{
      .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
      .stage                  = VK_SHADER_STAGE_VERTEX_BIT,
      .nextStage              = VK_SHADER_STAGE_FRAGMENT_BIT,
      .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
      .codeSize               = gizmo_visuals_slang_sizeInBytes,
      .pCode                  = gizmo_visuals_slang,
      .pName                  = "vertmain",
      .setLayoutCount         = 2,
      .pSetLayouts            = setLayouts,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges    = &pushRange,
  };
  NVVK_CHECK(vkCreateShadersEXT(m_device, 1, &shaderInfo, nullptr, &m_vertexShader));
  NVVK_DBG_NAME(m_vertexShader);

  shaderInfo.stage     = VK_SHADER_STAGE_FRAGMENT_BIT;
  shaderInfo.nextStage = 0;
  shaderInfo.pName     = "fragmain";
  NVVK_CHECK(vkCreateShadersEXT(m_device, 1, &shaderInfo, nullptr, &m_fragmentShader));
  NVVK_DBG_NAME(m_fragmentShader);

  vkDestroyDescriptorSetLayout(m_device, emptyLayout, nullptr);
}

void TransformHelperVk::rebuildPipelines()
{
  vkDestroyShaderEXT(m_device, m_vertexShader, nullptr);
  m_vertexShader = VK_NULL_HANDLE;
  vkDestroyShaderEXT(m_device, m_fragmentShader, nullptr);
  m_fragmentShader = VK_NULL_HANDLE;

  createShaderObjects();
}

//-----------------------------------------------------------------------------
// Debug UI (debug builds only)
//-----------------------------------------------------------------------------

#ifndef NDEBUG
bool TransformHelperVk::onDebugUI(GizmoStyle& style)
{
  bool changed = false;

  if(ImGui::CollapsingHeader("Screen Space", ImGuiTreeNodeFlags_DefaultOpen))
  {
    changed |= ImGui::SliderFloat("Size (px)", &style.sizePixels, 10.0f, 200.0f);
  }

  if(ImGui::CollapsingHeader("Colors", ImGuiTreeNodeFlags_DefaultOpen))
  {
    changed |= ImGui::ColorEdit3("Axis X", &style.colorX.x, ImGuiColorEditFlags_Float);
    changed |= ImGui::ColorEdit3("Axis Y", &style.colorY.x, ImGuiColorEditFlags_Float);
    changed |= ImGui::ColorEdit3("Axis Z", &style.colorZ.x, ImGuiColorEditFlags_Float);
    ImGui::Separator();
    changed |= ImGui::ColorEdit3("Plane XY", &style.colorPlaneXY.x, ImGuiColorEditFlags_Float);
    changed |= ImGui::ColorEdit3("Plane XZ", &style.colorPlaneXZ.x, ImGuiColorEditFlags_Float);
    changed |= ImGui::ColorEdit3("Plane YZ", &style.colorPlaneYZ.x, ImGuiColorEditFlags_Float);
    ImGui::Separator();
    ImGui::BeginDisabled(true);
    changed |= ImGui::ColorEdit3("Screen Rotation", &style.colorScreenRotation.x, ImGuiColorEditFlags_Float);
    ImGui::EndDisabled();
    changed |= ImGui::ColorEdit3("Uniform Scale", &style.colorScaleUniform.x, ImGuiColorEditFlags_Float);
  }

  if(ImGui::CollapsingHeader("Hover / Alpha", ImGuiTreeNodeFlags_DefaultOpen))
  {
    changed |= ImGui::SliderFloat("Hover Brightness", &style.hoverBrightness, 1.0f, 3.0f);
    changed |= ImGui::SliderFloat("Hover Mix", &style.hoverMix, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Plane Alpha", &style.planeAlpha, 0.0f, 1.0f);
    changed |= ImGui::Checkbox("Occlusion Dither", &style.occlusionDither);
  }

  if(ImGui::CollapsingHeader("Geometry (requires rebuild)"))
  {
    ImGui::BeginDisabled(true);
    ImGui::SliderFloat("Shaft Radius", &style.shaftRadius, 0.005f, 0.1f);
    ImGui::SliderFloat("Shaft Length", &style.shaftLength, 0.2f, 3.0f);
    ImGui::SliderFloat("Cone Radius", &style.coneRadius, 0.02f, 0.2f);
    ImGui::SliderFloat("Cone Height", &style.coneHeight, 0.05f, 0.5f);
    ImGui::SliderInt("Cone Segments", &style.coneSegments, 4, 64);
    ImGui::Separator();
    ImGui::SliderFloat("Ring Radius", &style.ringRadius, 0.5f, 3.0f);
    ImGui::SliderFloat("Ring Tube Radius", &style.ringTubeRadius, 0.005f, 0.1f);
    ImGui::SliderInt("Ring Major Segs", &style.ringMajorSegs, 16, 128);
    ImGui::SliderInt("Ring Minor Segs", &style.ringMinorSegs, 4, 32);
    ImGui::Separator();
    ImGui::SliderFloat("Scale Cube Size", &style.scaleCubeSize, 0.02f, 0.3f);
    ImGui::SliderFloat("Scale Cube Axis Len", &style.scaleCubeAxisLength, 0.5f, 3.0f);
    ImGui::SliderFloat("Center Cube Scale", &style.centerCubeScale, 0.5f, 3.0f);
    ImGui::Separator();
    ImGui::SliderFloat("Plane Size", &style.planeSize, 0.05f, 0.5f);
    ImGui::SliderFloat("Plane Offset", &style.planeOffset, 0.1f, 1.0f);
    ImGui::EndDisabled();
  }

  return changed;
}
#endif
