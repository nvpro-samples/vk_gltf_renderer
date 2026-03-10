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

#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvapp/application.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <vector>
#include <functional>

#include "shaders/gizmo_visuals_shaderio.h.slang"

//-----------------------------------------------------------------------------
// Transform Helper for 3D Gizmo Manipulation and Rasterization
// Unified gizmo: translate, rotate, and scale are all visible simultaneously.
// The operation is determined by which component the user clicks.
//-----------------------------------------------------------------------------

class TransformHelperVk
{
public:
  TransformHelperVk() = default;
  ~TransformHelperVk() { deinit(); }

  TransformHelperVk(const TransformHelperVk&)            = delete;
  TransformHelperVk& operator=(const TransformHelperVk&) = delete;

  //-----------------------------------------------------------------------------
  // Enums
  //-----------------------------------------------------------------------------

  enum class TransformSpace
  {
    eWorld,
    // eLocal // Not supported yet
  };

  enum class GizmoComponent
  {
    eNone = 0,
    // Translation
    eTranslateX,
    eTranslateY,
    eTranslateZ,
    eTranslateXY,
    eTranslateXZ,
    eTranslateYZ,
    // Rotation
    eRotateX,
    eRotateY,
    eRotateZ,
    eRotateScreen,
    // Scale
    eScaleX,
    eScaleY,
    eScaleZ,
    eScaleUniform
  };

  enum class GeometryType
  {
    eShaft,     // Elongated axis shaft (cylinder/box)
    eCone,      // Cone tip for translation arrows
    eBox,       // Box at axis end for scale
    eTorus,     // Rotation ring
    ePlane,     // Translation plane handle
    eCenterBox  // Center uniform scale box
  };

  // Flags for controlling which gizmo operations are visible
  enum GizmoVisibilityFlags : uint32_t
  {
    ShowTranslation = 1 << 0,
    ShowRotation    = 1 << 1,
    ShowScale       = 1 << 2,
    ShowAll         = ShowTranslation | ShowRotation | ShowScale
  };

  //-----------------------------------------------------------------------------
  // Visual Style Configuration
  //-----------------------------------------------------------------------------
  // All visual tuning parameters in one place. Change these defaults to
  // customize the gizmo appearance (Blender-like muted palette by default).

  struct GizmoStyle
  {
    float sizePixels = 50.0f;

    // Axis colors (muted Blender-like defaults)
    glm::vec3 colorX = {0.89f, 0.294f, 0.369f};
    glm::vec3 colorY = {0.45f, 0.819f, 0.263f};
    glm::vec3 colorZ = {0.25f, 0.451f, 0.851f};

    // Plane handle colors
    glm::vec3 colorPlaneXY = {0.78f, 0.78f, 0.25f};
    glm::vec3 colorPlaneXZ = {0.78f, 0.25f, 0.78f};
    glm::vec3 colorPlaneYZ = {0.25f, 0.78f, 0.78f};

    // Special component colors
    glm::vec3 colorScreenRotation = {0.75f, 0.75f, 0.75f};
    glm::vec3 colorScaleUniform   = {0.65f, 0.65f, 0.65f};

    // Hover: factor to brighten/saturate on hover (0.0 = no change)
    float hoverBrightness = 2.5f;
    float hoverMix        = 0.9f;

    // Translation geometry
    float shaftRadius  = 0.02f;
    float shaftLength  = 1.0f;
    float coneRadius   = 0.06f;
    float coneHeight   = 0.25f;
    int   coneSegments = 16;

    // Rotation geometry
    float ringRadius     = 1.3f;
    float ringTubeRadius = 0.020f;
    int   ringMajorSegs  = 64;
    int   ringMinorSegs  = 16;

    // Scale geometry
    float scaleCubeSize       = 0.15f;
    float scaleCubeAxisLength = 1.67f;
    float centerCubeScale     = 2.0f;

    // Plane handles
    float planeSize   = 0.22f;
    float planeOffset = 0.5f;
    float planeAlpha  = 0.45f;  // <1.0 enables dithered transparency in shader

    // Occlusion
    bool occlusionDither = false;  // Checkerboard x-ray when gizmo is behind scene geometry
  };

  //-----------------------------------------------------------------------------
  // Structures
  //-----------------------------------------------------------------------------

  struct GizmoVertex
  {
    glm::vec3 position;
    glm::vec3 normal;
  };

  struct GizmoGeometry
  {
    nvvk::Buffer   vertexBuffer;
    nvvk::Buffer   indexBuffer;
    uint32_t       indexCount = 0;
    GizmoComponent component  = GizmoComponent::eNone;
    GeometryType   type       = GeometryType::eShaft;
    glm::vec3      boundsMin  = glm::vec3(0.0f);
    glm::vec3      boundsMax  = glm::vec3(0.0f);
  };

  struct Resources
  {
    nvapp::Application*      app      = nullptr;
    nvvk::ResourceAllocator* alloc    = nullptr;
    nvvk::StagingUploader*   uploader = nullptr;
    VkDevice                 device   = VK_NULL_HANDLE;
  };

  //-----------------------------------------------------------------------------
  // Lifecycle
  //-----------------------------------------------------------------------------

  void init(const Resources& res);
  void deinit();

  //-----------------------------------------------------------------------------
  // Transform Attachment API (Entity-Agnostic)
  //-----------------------------------------------------------------------------

  void attachTransform(glm::vec3* position, glm::vec3* rotation, glm::vec3* scale, uint32_t visibilityFlags = ShowAll);
  void setParentWorldMatrix(const glm::mat4* parentWorld) { m_parentWorldMatrix = parentWorld; }
  bool isAttached() const;
  void clearAttachment();

  //-----------------------------------------------------------------------------
  // Configuration
  //-----------------------------------------------------------------------------

  void setTransformSpace(TransformSpace space);
  void setSnapEnabled(bool enabled) { m_enableSnapping = enabled; }
  void setSnapValues(float translate, float rotate, float scale);
  void setGizmoSize(float sizePixels) { m_style.sizePixels = sizePixels; }

  TransformSpace getTransformSpace() const { return m_space; }
  bool           isSnapEnabled() const { return m_enableSnapping; }

  GizmoStyle&       style() { return m_style; }
  const GizmoStyle& style() const { return m_style; }

#ifndef NDEBUG
  static bool onDebugUI(GizmoStyle& style);
#endif

  //-----------------------------------------------------------------------------
  // Interaction
  //-----------------------------------------------------------------------------

  bool processInput(const glm::vec2& mousePos,
                    const glm::vec2& mouseDelta,
                    bool             mouseDown,
                    bool             mousePressed,
                    bool             mouseReleased,
                    const glm::mat4& viewMatrix,
                    const glm::mat4& projMatrix,
                    const glm::vec2& viewport);

  bool isDragging() const { return m_isDragging; }

  //-----------------------------------------------------------------------------
  // Rendering
  //-----------------------------------------------------------------------------

  void renderRaster(VkCommandBuffer                                   cmd,
                    VkDescriptorSet                                   helperDescriptorSet,
                    /* Set 1: scene depth texture */ const glm::mat4& viewMatrix,
                    const glm::mat4&                                  projMatrix,
                    const glm::vec2&                                  viewportSize,
                    const glm::vec2&                                  depthBufferSize);

  VkDescriptorSetLayout getDescriptorSetLayout() const { return m_descriptorSetLayout; }

  //-----------------------------------------------------------------------------
  // Callbacks
  //-----------------------------------------------------------------------------

  void setOnTransformBegin(std::function<void()> callback) { m_onTransformBegin = callback; }
  void setOnTransformChange(std::function<void()> callback) { m_onTransformChange = callback; }
  void setOnTransformEnd(std::function<void()> callback) { m_onTransformEnd = callback; }

  //-----------------------------------------------------------------------------
  // Update Notifications
  //-----------------------------------------------------------------------------

  void notifyExternalTransformChange();

  //-----------------------------------------------------------------------------
  // Shader Management
  //-----------------------------------------------------------------------------

  void rebuildPipelines();

private:
  //-----------------------------------------------------------------------------
  // Shader Objects
  //-----------------------------------------------------------------------------

  void createShaderObjects();

  //-----------------------------------------------------------------------------
  // Geometry Generation (using nvutils primitives)
  //-----------------------------------------------------------------------------

  void createGizmoGeometry();
  void destroyGizmoGeometry();
  void generateTranslateGizmo(VkCommandBuffer cmd);
  void generateRotateGizmo(VkCommandBuffer cmd);
  void generateScaleGizmo(VkCommandBuffer cmd);

  //-----------------------------------------------------------------------------
  // Rendering
  //-----------------------------------------------------------------------------

  void createRasterPipeline();
  void destroyRasterPipeline();

  //-----------------------------------------------------------------------------
  // Picking and Interaction
  //-----------------------------------------------------------------------------

  GizmoComponent pickGizmoComponent(const glm::vec2& mousePos,
                                    const glm::mat4& viewMatrix,
                                    const glm::mat4& projMatrix,
                                    const glm::vec2& viewport);

  void startDrag(GizmoComponent   component,
                 const glm::vec2& mousePos,
                 const glm::mat4& viewMatrix,
                 const glm::mat4& projMatrix,
                 const glm::vec2& viewport);

  void updateDrag(const glm::vec2& mousePos,
                  const glm::vec2& mouseDelta,
                  const glm::mat4& viewMatrix,
                  const glm::mat4& projMatrix,
                  const glm::vec2& viewport);

  void endDrag();

  float computeScaleFactor(const glm::vec2& mousePos) const;

  //-----------------------------------------------------------------------------
  // Utility
  //-----------------------------------------------------------------------------

  glm::mat4 getGizmoTransform() const;
  glm::vec3 getAttachedPosition() const;
  glm::vec3 getDisplayPosition() const;
  glm::quat getAttachedRotation() const;
  glm::vec3 getAttachedScale() const;

  glm::vec3 localAxisToWorld(const glm::vec3& localAxis) const;
  glm::vec3 worldDeltaToLocal(const glm::vec3& worldDelta) const;

  float getScreenSpaceScale(const glm::vec3& worldPos, const glm::mat4& viewMatrix, const glm::mat4& projMatrix, const glm::vec2& viewport) const;

  bool rayIntersectPlane(const glm::vec3& rayOrigin,
                         const glm::vec3& rayDir,
                         const glm::vec3& planePoint,
                         const glm::vec3& planeNormal,
                         float&           t) const;

  void createRayFromMouse(const glm::vec2& mousePos,
                          const glm::mat4& viewMatrix,
                          const glm::mat4& projMatrix,
                          const glm::vec2& viewport,
                          glm::vec3&       rayOrigin,
                          glm::vec3&       rayDir) const;

  bool rayIntersectsBounds(const glm::vec3& rayOrigin, const glm::vec3& rayDir, const glm::vec3& boundsMin, const glm::vec3& boundsMax) const;

  float rayDistanceToBounds(const glm::vec3& rayOrigin, const glm::vec3& rayDir, const glm::vec3& boundsMin, const glm::vec3& boundsMax) const;

  // Ray-vs-torus picking: intersect the rotation plane, measure distance to ring radius
  float rayDistanceToTorus(const glm::vec3& rayOrigin, const glm::vec3& rayDir, GizmoComponent component) const;

  glm::vec3 getAxisFromComponent(GizmoComponent component) const;
  glm::vec3 getColorFromComponent(GizmoComponent component, bool hovered = false) const;
  bool      shouldRenderGeometry(const GizmoGeometry& geom) const;

  //-----------------------------------------------------------------------------
  // Member Variables
  //-----------------------------------------------------------------------------

  nvapp::Application*      m_app      = nullptr;
  nvvk::ResourceAllocator* m_alloc    = nullptr;
  nvvk::StagingUploader*   m_uploader = nullptr;
  VkDevice                 m_device   = VK_NULL_HANDLE;

  // Attached transform pointers (caller owns the storage)
  glm::vec3*       m_attachedPosition  = nullptr;
  glm::vec3*       m_attachedRotation  = nullptr;  // Euler angles in degrees
  glm::vec3*       m_attachedScale     = nullptr;
  const glm::mat4* m_parentWorldMatrix = nullptr;
  uint32_t         m_visibilityFlags   = ShowAll;

  // Configuration
  TransformSpace m_space          = TransformSpace::eWorld;
  bool           m_enableSnapping = false;
  float          m_snapTranslate  = 0.25f;
  float          m_snapRotate     = 15.0f;
  float          m_snapScale      = 0.1f;
  GizmoStyle     m_style;

  // Interaction state
  bool           m_isDragging             = false;
  GizmoComponent m_hoveredComponent       = GizmoComponent::eNone;
  GizmoComponent m_draggedComponent       = GizmoComponent::eNone;
  glm::vec2      m_dragStartPosMouse      = glm::vec2(0.0f);
  glm::vec3      m_dragStartPosition      = glm::vec3(0.0f);
  glm::vec3      m_dragStartWorldPosition = glm::vec3(0.0f);
  glm::vec3      m_dragStartRotation      = glm::vec3(0.0f);
  glm::vec3      m_dragStartScale         = glm::vec3(1.0f);
  glm::vec3      m_dragStartHitPoint      = glm::vec3(0.0f);

  // Geometry
  std::vector<GizmoGeometry> m_gizmoGeometry;

  // Raster rendering (VK_EXT_shader_object + dynamic state)
  VkPipelineLayout            m_rasterPipelineLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout       m_descriptorSetLayout  = VK_NULL_HANDLE;
  VkShaderEXT                 m_vertexShader         = VK_NULL_HANDLE;
  VkShaderEXT                 m_fragmentShader       = VK_NULL_HANDLE;
  nvvk::GraphicsPipelineState m_dynamicPipeline;

  // Callbacks
  std::function<void()> m_onTransformBegin;
  std::function<void()> m_onTransformChange;
  std::function<void()> m_onTransformEnd;
};
