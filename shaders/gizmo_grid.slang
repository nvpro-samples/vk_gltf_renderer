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

//=============================================================================
// Procedural infinite grid shader (Blender-style, XZ floor plane)
//
// Two draw calls from C++:
//   1) Grid:  12 * numLines vertices -> 3 LOD levels x 2 axes x numLines lines
//   2) Axes:  6 vertices -> X, Y, Z axis lines (2 endpoints each)
//
// Vertex shader generates all positions procedurally from SV_VertexID.
// Fragment shader composites grid (depth-tested, with edge/angle/distance fade)
// and axis lines (always on top via pipeline depth-test-off).
//=============================================================================

#include "gizmo_grid_shaderio.h.slang"

[[vk::push_constant]]
ConstantBuffer<GridPushConstants> pc;

[[vk::binding(0, 0)]]
Texture2D<float> sceneDepth;
[[vk::binding(1, 0)]]
SamplerState depthSampler;

static const float  kNaN  = asfloat(0x7FC00000u);
static const float3 kNaN3 = float3(kNaN, kNaN, kNaN);
static const float4 kNaN4 = float4(kNaN3, 1.0);


//--------------------------------------------------------------------------------------------------
// Vertex -> Fragment interpolants
//--------------------------------------------------------------------------------------------------
struct GridVertexOutput
{
  float4                 position : SV_Position;
  float3                 worldPos : TEXCOORD0;
  float2                 coord : TEXCOORD1;
  nointerpolation float  alpha : TEXCOORD2;
  nointerpolation float  emphasis : TEXCOORD3;
  nointerpolation float  axisKind : TEXCOORD4;  // 0 = grid, 1/2/3 = X/Y/Z axis
  nointerpolation float3 axisColor : TEXCOORD5;
  nointerpolation float  gridIter : TEXCOORD6;  // LOD level (0/1/2) for additive alpha
};


//--------------------------------------------------------------------------------------------------
// Vertex shader — procedural grid and axis line generation
//--------------------------------------------------------------------------------------------------
[shader("vertex")]
GridVertexOutput vertmain(uint vertexId: SV_VertexID)
{
  GridVertexOutput o;
  o.worldPos  = float3(0.0, 0.0, 0.0);
  o.coord     = float2(0.0, 0.0);
  o.alpha     = 1.0;
  o.emphasis  = 0.0;
  o.axisKind  = 0.0;
  o.axisColor = float3(0.0, 0.0, 0.0);
  o.gridIter  = 0.0;

  const uint  numLines  = pc.ubo.numLines;
  const float levelFrac = frac(pc.ubo.level);
  const int   levelBase = int(floor(pc.ubo.level));

  if((pc.gridFlag & GRID_FLAG_SHOW_GRID) != 0)
  {
    // Decode vertex ID into line topology
    uint lineIndex   = vertexId / 2u;
    uint side        = vertexId % 2u;
    uint block       = lineIndex / numLines;
    uint lineInBlock = lineIndex % numLines;
    uint level       = block / 2u;
    uint axis        = block % 2u;  // 0 = lines along X (fixed Z), 1 = lines along Z (fixed X)

    int    actualLevel = clamp(levelBase + 1 - int(level), 0, 3);
    float  stepSize    = pc.ubo.steps[actualLevel];
    float2 snapOffset  = round(pc.ubo.offset / stepSize) * stepSize;
    float  halfLines   = float(numLines - 1u) * 0.5;
    float  lineP       = float(lineInBlock) - halfLines;
    float  levelClip   = stepSize * halfLines;

    // Unified X/Z: axis selects which world component is the fixed line coordinate vs. the varying extent
    //   axis==0 → fixedIdx=2 (Z), varyIdx=0 (X) ;  axis==1 → fixedIdx=0 (X), varyIdx=2 (Z)
    uint  fixedIdx = 2u * (1u - axis);
    uint  varyIdx  = 2u * axis;
    float fixedPos = snapOffset[1u - axis] + stepSize * lineP;
    float varyPos  = snapOffset[axis] + ((side == 0u) ? -levelClip : levelClip);

    float3 worldPos    = float3(0.0, 0.0, 0.0);
    worldPos[fixedIdx] = fixedPos;
    worldPos[varyIdx]  = varyPos;

    // Cull finer lines that coincide with a coarser LOD level
    if(level >= 1u)
    {
      float nextStep = pc.ubo.steps[min(actualLevel + 1, 3)];
      float j        = abs(fixedPos) / nextStep;
      if(abs(j - round(j)) < 1e-4)
        worldPos = kNaN3;  // Cull by moving to NaN, which fails all depth/screen tests and doesn't require branching
    }

    // Suppress lines on X/Z axes so colored axis lines show through
    float eps    = stepSize * 0.01;
    bool  onAxis = abs(fixedPos) < eps;
    if(onAxis)
      worldPos = kNaN3;

    o.worldPos = worldPos;
    o.coord    = (worldPos.xz - snapOffset) / max(levelClip, 1e-6);
    o.alpha    = saturate(3.0 - float(level) - levelFrac);
    o.emphasis = saturate(2.0 - float(level) - levelFrac);
    o.gridIter = float(level);
    o.position = onAxis ? kNaN4 : mul(float4(worldPos, 1.0), pc.viewProj);
  }
  else
  {
    // --- Axis lines (6 vertices: 2 per axis) ---
    uint side = vertexId & 1u;
    uint axis = vertexId >> 1u;

    // Skip disabled axes
    const uint axisFlags[3] = { GRID_FLAG_AXIS_X, GRID_FLAG_AXIS_Y, GRID_FLAG_AXIS_Z };
    if((pc.gridFlag & axisFlags[axis]) == 0u)
    {
      o.position = kNaN4;
      return o;
    }

    float3 axisColors[3] = { pc.ubo.axisColorX, pc.ubo.axisColorY, pc.ubo.axisColorZ };
    o.axisColor          = axisColors[axis];

    float ext       = (axis == 2u) ? pc.ubo.clipRect.y : pc.ubo.clipRect.x;
    float signedExt = (side == 0u) ? -ext : ext;

    float3 worldPos = float3(0.0, 0.0, 0.0);
    worldPos[axis]  = signedExt;

    float normP = (side == 0u) ? -1.0 : 1.0;

    // For axis lines, use worldPos as interpolant for edge fading in fragment shader, and coord to fade out at the center
    o.worldPos = worldPos;
    o.coord    = (axis == 2u) ? float2(0.0, normP) : float2(normP, 0.0);
    o.axisKind = float(axis + 1u);
    o.position = mul(float4(worldPos, 1.0), pc.viewProj);
  }

  return o;
}


//--------------------------------------------------------------------------------------------------
// Fragment shader
//--------------------------------------------------------------------------------------------------
struct GridFragmentOutput
{
  float4 color : SV_Target0;
};

[shader("fragment")]
GridFragmentOutput fragmain(GridVertexOutput input)
{
  GridFragmentOutput output;
  output.color = float4(0.0, 0.0, 0.0, 0.0);

  float3 rgb;
  float  a    = input.alpha;
  bool   grid = input.axisKind < 0.5;

  if(grid)
  {
    // Depth test against scene geometry
    float2 screenUV = input.position.xy / pc.viewportSize;
    float2 depthUV  = screenUV * (pc.depthBufferSize / pc.viewportSize);
    if(input.position.z > sceneDepth.Sample(depthSampler, depthUV))
      discard;

    // Minor/major color blend with subtle intensity ramp
    rgb = lerp(pc.minorColor, pc.majorColor, input.emphasis);
    rgb *= lerp(0.75, 1.0, input.emphasis);

    // Edge fade: lines dissolve toward grid boundary
    a *= 1.0 - min(1.0, length(input.coord));

    // Steep-angle fade (XZ plane only, not along Y-axis lines)
    if(abs(input.worldPos.y) < 1e-4)
    {
      float3 V = normalize(pc.cameraPos - input.worldPos);
      a *= 1.0 - pow(1.0 - abs(V.y), 3.0);
    }

    // Perspective-only: distance fade + per-level additive alpha to minimize pop-in
    if(pc.isPerspective != 0)
    {
      float dist    = length(pc.cameraPos - input.worldPos);
      float halfFar = pc.farClip * 0.5;
      a *= 1.0 - smoothstep(0.0, halfFar, dist - halfFar);

      static const float kAdditiveAlpha[3] = { 0.4, 0.3, 0.2 };
      a *= kAdditiveAlpha[clamp(int(input.gridIter), 0, 2)];
    }
  }
  else
  {
    rgb = input.axisColor;
  }

  output.color = float4(rgb, a);
  return output;
}
