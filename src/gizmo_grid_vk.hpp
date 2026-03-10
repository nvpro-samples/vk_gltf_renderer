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

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "nvvk/resource_allocator.hpp"
#include "nvvk/staging.hpp"
#include "nvvk/graphics_pipeline.hpp"
#include "nvapp/application.hpp"
#include <nvslang/slang.hpp>
#include "shaders/gizmo_grid_shaderio.h.slang"

//-----------------------------------------------------------------------------
// GridHelperVk - Procedural Blender-style grid on the XZ floor plane
//-----------------------------------------------------------------------------
// - No vertex/index buffers: the vertex shader gets only a vertex ID (0, 1, 2, ...)
//   and computes 3D positions from it. See gizmo_grid.slang and gizmo_grid_vk.cpp
//   for detailed comments on how lines are generated and how the pipeline works.
// - Grid: 3 LOD levels, camera-snapped offset, smooth fade; axis lines X=red, Y=green, Z=blue.
// - Dynamic pipeline state (VK_EXT_shader_object); blend overridden per-draw for axes.
// - Additive alpha blending so the grid composites on top of the rendered scene.
//-----------------------------------------------------------------------------

class GridHelperVk
{
public:
  GridHelperVk() = default;
  ~GridHelperVk() { deinit(); }

  GridHelperVk(const GridHelperVk&)            = delete;
  GridHelperVk& operator=(const GridHelperVk&) = delete;

  struct Resources
  {
    nvapp::Application*      app                       = nullptr;
    nvvk::ResourceAllocator* alloc                     = nullptr;
    nvvk::StagingUploader*   uploader                  = nullptr;
    VkDevice                 device                    = VK_NULL_HANDLE;
    nvslang::SlangCompiler*  slangCompiler             = nullptr;
    VkFormat                 colorFormat               = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkFormat                 depthFormat               = VK_FORMAT_D32_SFLOAT;
    VkDescriptorSetLayout    helperDescriptorSetLayout = VK_NULL_HANDLE;  // Set 0: scene depth
  };

  void init(const Resources& res);
  void deinit();

  void setVisible(bool visible) { m_visible = visible; }
  bool isVisible() const { return m_visible; }
  void toggleVisible() { m_visible = !m_visible; }

  void  setLineThickness(float pixels) { m_style.lineWidth = pixels; }
  float getLineThickness() const { return m_style.lineWidth; }

  struct GridStyle
  {
    glm::vec3 minorColor = {0.5f, 0.5f, 0.5f};
    glm::vec3 majorColor = {0.5f, 0.5f, 0.5f};
    // Axis colors: X=red, Y=green, Z=blue (convention)
    glm::vec3 axisColorX = {0.619f, 0.235f, 0.290f};
    glm::vec3 axisColorY = {0.357f, 0.482f, 0.185f};
    glm::vec3 axisColorZ = {0.231f, 0.357f, 0.518f};
    float     lineWidth  = 1.5f;
    float     baseUnit   = 1.0f;  // Step level 0 = baseUnit * 10^0
    uint32_t  numLines   = 151u;  // Lines per level per axis (odd for symmetry)
  };

  GridStyle&       style() { return m_style; }
  const GridStyle& style() const { return m_style; }

#ifndef NDEBUG
  static bool onDebugUI(GridStyle& style);
#endif

  void renderRaster(VkCommandBuffer  cmd,
                    VkDescriptorSet  helperDescriptorSet,
                    const glm::mat4& viewMatrix,
                    const glm::mat4& projMatrix,
                    const glm::vec2& viewportSize,
                    const glm::vec2& depthBufferSize);

private:
  using GridUBO           = shaderio::grid::GridUBO;
  using GridPushConstants = shaderio::grid::GridPushConstants;

  // Distance from camera to floor (view-dependent, Blender-style). Used for level and grid offset.
  float distanceToFloor(const glm::vec3& cameraPos, const glm::vec3& viewForward, bool isPerspective) const;
  // Fractional level from that distance; shader draws 3 adjacent levels.
  float calculateLevel(float dist) const;

  void createPipeline();
  void createGridUBO();
  void updateGridUBO(const glm::vec3& cameraPos, const glm::vec3& viewForward, bool isPerspective);

  nvapp::Application*      m_app    = nullptr;
  nvvk::ResourceAllocator* m_alloc  = nullptr;
  VkDevice                 m_device = VK_NULL_HANDLE;

  VkPipelineLayout      m_pipelineLayout            = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_helperDescriptorSetLayout = VK_NULL_HANDLE;
  VkShaderEXT           m_vertexShader              = VK_NULL_HANDLE;
  VkShaderEXT           m_fragmentShader            = VK_NULL_HANDLE;

  nvvk::GraphicsPipelineState m_dynamicPipeline;

  nvvk::Buffer m_uboBuffer;
  bool         m_visible     = true;
  bool         m_initialized = false;
  GridStyle    m_style;
  float        m_lastGridLevel = 0.0f;
};
