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

#include "shaderio.h"
#include "gltf_scene_io.h.slang"

#include "nvshaders/tonemap_functions.h.slang"

enum VisualizationResult
{
  eRenderedOutput,    // No override: continue full-quality shading
  eColorOverride,     // Flat color returned in overrideColor; caller should use it directly
  eMaterialOverride,  // pbrMat was mutated in-place; caller should continue shading with it
}

float3
hashToColor(uint id)
{
  uint h = id;
  h      = h * 747796405u + 2891336453u;
  h      = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  h      = (h >> 22u) ^ h;
  return float3(float((h >> 0) & 0xFF), float((h >> 8) & 0xFF), float((h >> 16) & 0xFF)) / 255.0;
}

// Debug visualization override: displays a flat color for the given visualization mode.
VisualizationResult applyVisualization(inout PbrMaterial pbrMat,
                                       HitState          hit,
                                       Visualization     vizMethod,
                                       int               rprimID,
                                       int               primitiveID,
                                       int               ommState,
                                       out float3        overrideColor)
{
  overrideColor = float3(0);

  switch(vizMethod)
  {
    case Visualization::eBaseColor:
      overrideColor = float3(pbrMat.baseColor);
      overrideColor = toSrgb(overrideColor);
      return VisualizationResult::eColorOverride;
    case Visualization::eMetallic:
      overrideColor = float3(pbrMat.metallic);
      overrideColor = toSrgb(overrideColor);
      return VisualizationResult::eColorOverride;
    case Visualization::eRoughness:
      overrideColor = float3(pbrMat.roughness.xyx);
      overrideColor = toSrgb(overrideColor);
      return VisualizationResult::eColorOverride;
    case Visualization::eNormalShading:
      overrideColor = float3(pbrMat.N * .5 + .5);
      overrideColor = toSrgb(overrideColor);
      return VisualizationResult::eColorOverride;
    case Visualization::eNormalGeometric:
      overrideColor = float3(pbrMat.Ng * .5 + .5);
      return VisualizationResult::eColorOverride;
    case Visualization::eTangent:
      overrideColor = pbrMat.T * .5 + .5;
      return VisualizationResult::eColorOverride;
    case Visualization::eBitangent:
      overrideColor = pbrMat.B * .5 + .5;
      return VisualizationResult::eColorOverride;
    case Visualization::eEmissive:
      overrideColor = float3(pbrMat.emissive);
      return VisualizationResult::eColorOverride;
    case Visualization::eOpacity:
      overrideColor = float3(pbrMat.opacity * (1.0 - pbrMat.transmission));
      return VisualizationResult::eColorOverride;
    case Visualization::eTexCoord0:
      overrideColor = float3(fract(hit.uv[0]), 0);
      return VisualizationResult::eColorOverride;
    case Visualization::eTexCoord1:
      overrideColor = float3(fract(hit.uv[1]), 0);
      return VisualizationResult::eColorOverride;
    case Visualization::eClay:
      pbrMat.baseColor = float3(0.8, 0.75, 0.7);
      pbrMat.metallic  = 0.0;
      pbrMat.roughness = float2(0.5 * 0.5);
      pbrMat.emissive  = float3(0);
      return VisualizationResult::eMaterialOverride;
    case Visualization::eTriangleID:
      overrideColor = hashToColor(uint(rprimID) * 65537u + uint(primitiveID));
      return VisualizationResult::eColorOverride;
    case Visualization::eFaceOrientation:
      overrideColor = hit.frontFace ? float3(0, 1, 0) : float3(1, 0, 0);
      return VisualizationResult::eColorOverride;
    // Khronos parity debug views. Each maps to a PbrMaterial field already filled in by
    // evaluateMaterial. Outputs match the conventions used in pbr.frag's debug branches.
    case Visualization::eOcclusion:
      overrideColor = float3(pbrMat.occlusion);
      return VisualizationResult::eColorOverride;
    case Visualization::eClearcoatFactor:
      overrideColor = float3(pbrMat.clearcoat);
      return VisualizationResult::eColorOverride;
    case Visualization::eClearcoatRoughness:
      overrideColor = float3(pbrMat.clearcoatRoughness);
      return VisualizationResult::eColorOverride;
    case Visualization::eClearcoatNormal:
      overrideColor = pbrMat.Nc * 0.5 + 0.5;
      return VisualizationResult::eColorOverride;
    case Visualization::eSheenColor:
      overrideColor = pbrMat.sheenColor;
      return VisualizationResult::eColorOverride;
    case Visualization::eSheenRoughness:
      overrideColor = float3(pbrMat.sheenRoughness);
      return VisualizationResult::eColorOverride;
    case Visualization::eSpecularFactor:
      overrideColor = float3(pbrMat.specular);
      return VisualizationResult::eColorOverride;
    case Visualization::eSpecularColor:
      overrideColor = pbrMat.specularColor;
      return VisualizationResult::eColorOverride;
    case Visualization::eTransmissionFactor:
      overrideColor = float3(pbrMat.transmission);
      return VisualizationResult::eColorOverride;
    case Visualization::eIridescenceFactor:
      overrideColor = float3(pbrMat.iridescence);
      return VisualizationResult::eColorOverride;
    case Visualization::eIridescenceThickness:
      // Khronos pbr.frag normalizes by a fixed 1200; we do the same so screenshots compare.
      overrideColor = float3(pbrMat.iridescenceThickness / 1200.0);
      return VisualizationResult::eColorOverride;
    case Visualization::eAnisotropyStrength:
      // PbrMaterial doesn't store the strength scalar (folded into roughness.xy); recover it
      // from the inflation: strength² = (rx - ry) / (1 - ry).
      {
        float den     = max(1.0 - pbrMat.roughness.y, 1e-5);
        overrideColor = float3(sqrt(saturate((pbrMat.roughness.x - pbrMat.roughness.y) / den)));
      }
      return VisualizationResult::eColorOverride;
    case Visualization::eDiffuseTransmissionFactor:
      overrideColor = float3(pbrMat.diffuseTransmissionFactor);
      return VisualizationResult::eColorOverride;
    case Visualization::eDiffuseTransmissionColor:
      overrideColor = pbrMat.diffuseTransmissionColor;
      return VisualizationResult::eColorOverride;
    case Visualization::eOpacityMicromap:
      if(ommState < 0)
        return VisualizationResult::eRenderedOutput;
      overrideColor = (ommState > 0) ? float3(0.90, 0.80, 0.10)  // unknown: alpha evaluated at runtime by the any-hit
                                       :
                                       float3(0.15, 0.75, 0.15);  // opaque: resolved by the opacity micromap (any-hit skipped)
      return VisualizationResult::eColorOverride;

    default:
      return VisualizationResult::eRenderedOutput;
  }
}

//-----------------------------------------------------------------------
// Check if a flag is set
//-----------------------------------------------------------------------
bool hasFlag(int flags, int bit)
{
  return (flags & bit) != 0;
}
