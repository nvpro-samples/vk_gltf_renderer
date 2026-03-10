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

// Shader interface for procedural Blender-style grid (XZ plane)
// Shared between C++ and gizmo_grid.slang

#pragma once

#ifdef __cplusplus
#include <glm/glm.hpp>
#include "nvshaders/slang_types.h"
namespace shaderio::grid {
#endif

//--------------------------------------------------------------------------------------------------
// Grid flags (gridFlag in push constants)
//--------------------------------------------------------------------------------------------------
#define GRID_FLAG_SHOW_GRID  (1 << 0)
#define GRID_FLAG_AXIS_X     (1 << 1)
#define GRID_FLAG_AXIS_Y     (1 << 2)
#define GRID_FLAG_AXIS_Z     (1 << 3)

//--------------------------------------------------------------------------------------------------
// Grid UBO
//--------------------------------------------------------------------------------------------------
struct GridUBO
{
  float steps[4];      // Per-level step sizes (4 levels for decade spacing)
  float2 offset;        // Camera XZ on grid plane
  float2 clipRect;      // Clip distance per axis
  float  level;        // Fractional level from camera distance
  uint   numLines;      // Lines per level per axis (e.g. 151)
  int    padding0;
  int    padding1;
  float3 axisColorX;    // Axis line colors
  float  padAxis0;
  float3 axisColorY;
  float  padAxis1;
  float3 axisColorZ;
  float  padAxis2;
};

//--------------------------------------------------------------------------------------------------
// Grid push constants
//--------------------------------------------------------------------------------------------------
struct GridPushConstants
{
  float4x4 viewProj;         // Combined view-projection
  float3   cameraPos;        // For steep-angle V and clip fade
  float    farClip;          // Far clip distance
  float2   viewportSize;
  float2   depthBufferSize;
  int      gridFlag;         // GRID_FLAG_SHOW_GRID | AXIS_X | AXIS_Y | AXIS_Z
  int      isPerspective;    // 1 = perspective, 0 = ortho
  float3   minorColor;       // Grid minor line color
  float    padMinorColor;    // std430: 4-byte padding so majorColor is 16-byte aligned
  float3   majorColor;       // Grid major line color
  float    padMajorColor;    // std430: 4-byte padding so ubo (8-byte ptr) is 8-byte aligned
  GridUBO* ubo;              // Buffer device address
};

#ifdef __cplusplus
}  // namespace shaderio::grid
#endif
