/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

//=============================================================================
// GRID HELPER (Vulkan) - How the grid is drawn, step by step
//=============================================================================
//
// 1. HOW LINES ARE CREATED (no vertex buffer)
//    We do NOT upload an array of line positions to the GPU. Instead we use
//    "procedural" drawing: we tell Vulkan "draw N vertices" and the vertex
//    shader receives a counter 0, 1, 2, ... (SV_VertexID). The shader turns
//    that number into a 3D position. So:
//      - First draw: we say "draw 12 * numLines vertices". The shader
//        interprets each pair of vertices as one line (start, end) and
//        computes positions for all grid lines from the vertex ID.
//      - Second draw: we say "draw 6 vertices". The shader interprets
//        vertices 0,1 as the X axis, 2,3 as Y, 4,5 as Z (each pair = one line).
//
// 2. SHADER OBJECTS + DYNAMIC STATE (VK_EXT_shader_object)
//    One vertex + fragment shader object pair; all graphics state is set
//    dynamically via cmdApplyAllStates(). The grid draw uses additive blend
//    (SRC_ALPHA, ONE) and the axis draw overrides to standard alpha-over
//    (SRC_ALPHA, ONE_MINUS_SRC_ALPHA) via vkCmdSetColorBlendEquationEXT.
//    Both use VK_PRIMITIVE_TOPOLOGY_LINE_LIST: every two vertices form one line.
//
// 3. WHY ALPHA BLENDING
//    We blend the grid on top of the already-rendered scene. The blend mode is:
//      new_color = SRC_ALPHA * fragment_color + 1 * framebuffer_color
//    So we add our line color (scaled by alpha) to what's already there. That
//    way overlapping lines (e.g. many grid lines) accumulate and look correct,
//    and the grid "lays on top" of the scene instead of blocking it.
//
// 4. HOW THE IMAGE IS COMPOSITED AT THE END
//    The app first renders the 3D scene into the color (and depth) buffer.
//    Then it runs this grid helper: we draw our lines with blending ON, so
//    our line fragments are blended onto that same color buffer. We also
//    read the scene depth in the fragment shader to discard grid pixels
//    that are behind geometry. Result: the grid appears on the floor and
//    behind objects, and the X/Y/Z axis lines appear on top.
//
//=============================================================================

#include "gizmo_grid_vk.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <glm/gtc/matrix_transform.hpp>

#include <nvvk/debug_util.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvutils/logger.hpp>

#ifndef NDEBUG
#include <imgui.h>
#endif

#include "shaders/gizmo_grid_shaderio.h.slang"
#include "_autogen/gizmo_grid.slang.h"


//-----------------------------------------------------------------------------
// Initialization: create the UBO buffer and shader objects + dynamic pipeline
// state. No vertex/index buffers or VkPipeline objects are created.
//-----------------------------------------------------------------------------

void GridHelperVk::init(const Resources& res)
{
  if(m_initialized)
    deinit();

  m_app                       = res.app;
  m_alloc                     = res.alloc;
  m_device                    = res.device;
  m_helperDescriptorSetLayout = res.helperDescriptorSetLayout;

  createGridUBO();
  createPipeline();
  m_initialized = true;
  LOGI("GridHelperVk initialized\n");
}

void GridHelperVk::deinit()
{
  if(!m_initialized)
    return;
  vkDeviceWaitIdle(m_device);

  if(m_uboBuffer.buffer != VK_NULL_HANDLE)
  {
    m_alloc->destroyBuffer(m_uboBuffer);
    m_uboBuffer = {};
  }
  vkDestroyShaderEXT(m_device, m_vertexShader, nullptr);
  m_vertexShader = VK_NULL_HANDLE;
  vkDestroyShaderEXT(m_device, m_fragmentShader, nullptr);
  m_fragmentShader = VK_NULL_HANDLE;
  if(m_pipelineLayout != VK_NULL_HANDLE)
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    m_pipelineLayout = VK_NULL_HANDLE;
  }
  m_helperDescriptorSetLayout = VK_NULL_HANDLE;
  m_initialized               = false;
}

//-----------------------------------------------------------------------------
// Distance from camera to the floor plane (Blender overlay_grid.hh).
// Perspective: interpolate between (distance along view ray to floor) and
// (height above floor), so grid scale follows view, not origin.
// Ortho: use height so scale is stable.
//-----------------------------------------------------------------------------

float GridHelperVk::distanceToFloor(const glm::vec3& cameraPos, const glm::vec3& viewForward, bool isPerspective) const
{
  if(isPerspective)
  {
    // Y-up: floor is XZ at Y=0. Ray hits floor when cameraPos.y + t*viewForward.y == 0 => t = -cameraPos.y/viewForward.y
    float rayDistToFloor =
        (std::abs(viewForward.y) >= 1e-6f) ? std::abs(cameraPos.y / viewForward.y) : std::abs(cameraPos.y);
    float heightAboveFloor = std::abs(cameraPos.y);
    float blend            = 1.0f - std::min(1.0f, std::abs(viewForward.y));
    return std::max(0.001f, (1.0f - blend) * rayDistToFloor + blend * heightAboveFloor);
  }
  return std::max(0.001f, std::abs(cameraPos.y));
}

//-----------------------------------------------------------------------------
// Level from distance: linear interpolation between step sizes (Blender).
//-----------------------------------------------------------------------------

float GridHelperVk::calculateLevel(float dist) const
{
  float base    = std::max(m_style.baseUnit, 0.001f);
  float logDist = std::log10(std::max(dist / base, 0.001f));
  float level   = std::clamp(logDist, 0.0f, 3.999f);
  return level;
}

//-----------------------------------------------------------------------------
// Grid UBO: a small buffer that holds all the data the shader needs every frame
// (step sizes per level, camera offset, colors, etc.). We update it in
// updateGridUBO() and the shader reads it via buffer device address in push constants.
//-----------------------------------------------------------------------------

void GridHelperVk::createGridUBO()
{
  NVVK_CHECK(m_alloc->createBuffer(m_uboBuffer, sizeof(GridUBO), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT,
                                   VMA_MEMORY_USAGE_CPU_TO_GPU, VMA_ALLOCATION_CREATE_MAPPED_BIT));
  NVVK_DBG_NAME(m_uboBuffer.buffer);
}

void GridHelperVk::updateGridUBO(const glm::vec3& cameraPos, const glm::vec3& viewForward, bool isPerspective)
{
  float dist      = distanceToFloor(cameraPos, viewForward, isPerspective);
  float level     = calculateLevel(dist);
  m_lastGridLevel = level;

  // Grid offset: center on the point on the floor (Y=0) under the view
  glm::vec3 floorPoint = cameraPos - dist * viewForward;
  glm::vec2 gridCenter = isPerspective ? glm::vec2(floorPoint.x, floorPoint.z) : glm::vec2(cameraPos.x, cameraPos.z);

  GridUBO ubo  = {};
  float   base = std::max(m_style.baseUnit, 0.001f);
  for(int i = 0; i < 4; i++)
    ubo.steps[i] = std::pow(10.0f, float(i)) * base;
  ubo.offset         = gridCenter;
  int   coarsestIdx  = std::min(int(std::floor(level)) + 1, 3);
  float coarsestStep = ubo.steps[coarsestIdx];
  float clipExt      = coarsestStep * float(m_style.numLines / 2);
  ubo.clipRect       = glm::vec2(clipExt, clipExt);
  ubo.level          = level;
  ubo.numLines       = m_style.numLines;
  ubo.axisColorX     = m_style.axisColorX;
  ubo.axisColorY     = m_style.axisColorY;
  ubo.axisColorZ     = m_style.axisColorZ;

  if(m_uboBuffer.mapping)
    memcpy(m_uboBuffer.mapping, &ubo, sizeof(ubo));
}

//-----------------------------------------------------------------------------
// Pipeline setup: create shader objects (VK_EXT_shader_object) and configure
// dynamic pipeline state. No VkPipeline objects are created; all graphics
// state is set dynamically at draw time via cmdApplyAllStates().
//-----------------------------------------------------------------------------

void GridHelperVk::createPipeline()
{
  VkPushConstantRange pushRange = {.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   .size       = sizeof(GridPushConstants)};

  VkPipelineLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  layoutInfo.setLayoutCount             = 1;
  layoutInfo.pSetLayouts                = &m_helperDescriptorSetLayout;
  layoutInfo.pushConstantRangeCount     = 1;
  layoutInfo.pPushConstantRanges        = &pushRange;
  NVVK_CHECK(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipelineLayout));
  NVVK_DBG_NAME(m_pipelineLayout);

  VkShaderCreateInfoEXT shaderInfo{
      .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
      .stage                  = VK_SHADER_STAGE_VERTEX_BIT,
      .nextStage              = VK_SHADER_STAGE_FRAGMENT_BIT,
      .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
      .codeSize               = gizmo_grid_slang_sizeInBytes,
      .pCode                  = gizmo_grid_slang,
      .pName                  = "vertmain",
      .setLayoutCount         = 1,
      .pSetLayouts            = &m_helperDescriptorSetLayout,
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

  // Dynamic pipeline state (baseline for grid draw; axis draw overrides blend)
  m_dynamicPipeline.inputAssemblyState.topology                  = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  m_dynamicPipeline.rasterizationState.cullMode                  = VK_CULL_MODE_NONE;
  m_dynamicPipeline.rasterizationState.frontFace                 = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  m_dynamicPipeline.rasterizationState.lineWidth                 = m_style.lineWidth;
  m_dynamicPipeline.rasterizationLineState.lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH;
  m_dynamicPipeline.depthStencilState.depthTestEnable            = VK_FALSE;
  m_dynamicPipeline.depthStencilState.depthWriteEnable           = VK_FALSE;
  m_dynamicPipeline.depthStencilState.depthCompareOp             = VK_COMPARE_OP_LESS_OR_EQUAL;

  // Additive blend: SRC_ALPHA * fragment + ONE * framebuffer
  m_dynamicPipeline.colorBlendEnables[0]                       = VK_TRUE;
  m_dynamicPipeline.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  m_dynamicPipeline.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
  m_dynamicPipeline.colorBlendEquations[0].colorBlendOp        = VK_BLEND_OP_ADD;
  m_dynamicPipeline.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  m_dynamicPipeline.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  m_dynamicPipeline.colorBlendEquations[0].alphaBlendOp        = VK_BLEND_OP_ADD;
}

//-----------------------------------------------------------------------------
// Rendering: called after the main scene is drawn. We update the UBO, then
// issue two draw calls (grid, then axis) with the appropriate pipeline and
// push constants. The framebuffer already contains the scene; we blend our
// lines on top.
//-----------------------------------------------------------------------------

void GridHelperVk::renderRaster(VkCommandBuffer  cmd,
                                VkDescriptorSet  helperDescriptorSet,
                                const glm::mat4& viewMatrix,
                                const glm::mat4& projMatrix,
                                const glm::vec2& viewportSize,
                                const glm::vec2& depthBufferSize)
{
  if(!m_visible || !m_initialized || m_vertexShader == VK_NULL_HANDLE)
    return;

  glm::mat4 invView       = glm::inverse(viewMatrix);
  glm::vec3 cameraPos     = glm::vec3(invView[3]);
  glm::vec3 viewForward   = glm::normalize(glm::vec3(invView[0][2], invView[1][2], invView[2][2]));
  bool      isPerspective = (projMatrix[3][3] < 1e-6f);

  updateGridUBO(cameraPos, viewForward, isPerspective);

  const float kGridNear = 0.1f;
  const float kGridFar  = 100000.0f;
  glm::mat4   viewProj;
  if(isPerspective)
  {
    float     fovY     = 2.0f * std::atan(1.0f / -projMatrix[1][1]);
    float     aspect   = viewportSize.x / std::max(viewportSize.y, 1.0f);
    glm::mat4 gridProj = glm::perspectiveRH_ZO(fovY, aspect, kGridNear, kGridFar);
    gridProj[1][1] *= -1;  // Vulkan NDC Y down
    viewProj = gridProj * viewMatrix;
  }
  else
  {
    viewProj = projMatrix * viewMatrix;
  }

  GridPushConstants pc = {};
  pc.viewProj          = viewProj;
  pc.cameraPos         = cameraPos;
  pc.farClip           = kGridFar;
  pc.viewportSize      = viewportSize;
  pc.depthBufferSize   = depthBufferSize;
  pc.gridFlag          = GRID_FLAG_SHOW_GRID;
  pc.isPerspective     = isPerspective ? 1 : 0;
  pc.minorColor        = m_style.minorColor;
  pc.majorColor        = m_style.majorColor;
  pc.ubo               = (GridUBO*)m_uboBuffer.address;

  m_dynamicPipeline.rasterizationState.lineWidth = m_style.lineWidth;
  m_dynamicPipeline.cmdApplyAllStates(cmd);
  m_dynamicPipeline.cmdApplyDynamicState(cmd, VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE);
  nvvk::GraphicsPipelineState::cmdBindShaders(cmd, {.vertex = m_vertexShader, .fragment = m_fragmentShader});
  vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &helperDescriptorSet, 0, nullptr);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

  // Draw 1: grid lines (additive blend set by cmdApplyAllStates)
  vkCmdDraw(cmd, 12u * m_style.numLines, 1, 0, 0);

  // Draw 2: axis lines (override blend to standard alpha-over)
  pc.gridFlag = (GRID_FLAG_AXIS_X | GRID_FLAG_AXIS_Y | GRID_FLAG_AXIS_Z);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
  const VkColorBlendEquationEXT axisBlend{
      .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      .colorBlendOp        = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
      .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
      .alphaBlendOp        = VK_BLEND_OP_ADD,
  };
  vkCmdSetColorBlendEquationEXT(cmd, 0, 1, &axisBlend);
  vkCmdDraw(cmd, 6, 1, 0, 0);
}

//-----------------------------------------------------------------------------
// Debug UI: ImGui window to tweak GridStyle at runtime (debug builds only)
//-----------------------------------------------------------------------------

#ifndef NDEBUG
bool GridHelperVk::onDebugUI(GridStyle& style)
{
  bool changed = false;

  changed |= ImGui::ColorEdit3("Minor Color", &style.minorColor.x, ImGuiColorEditFlags_Float);
  changed |= ImGui::ColorEdit3("Major Color", &style.majorColor.x, ImGuiColorEditFlags_Float);
  ImGui::Separator();
  changed |= ImGui::ColorEdit3("Axis X Color", &style.axisColorX.x, ImGuiColorEditFlags_Float);
  changed |= ImGui::ColorEdit3("Axis Y Color", &style.axisColorY.x, ImGuiColorEditFlags_Float);
  changed |= ImGui::ColorEdit3("Axis Z Color", &style.axisColorZ.x, ImGuiColorEditFlags_Float);
  ImGui::Separator();
  changed |= ImGui::SliderFloat("Line Width", &style.lineWidth, 0.5f, 5.0f);
  changed |= ImGui::SliderFloat("Base Unit", &style.baseUnit, 0.001f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic);

  int numLines = static_cast<int>(style.numLines);
  if(ImGui::SliderInt("Num Lines", &numLines, 11, 501))
  {
    numLines |= 1;  // keep odd for symmetry
    style.numLines = static_cast<uint32_t>(numLines);
    changed        = true;
  }

  return changed;
}
#endif
