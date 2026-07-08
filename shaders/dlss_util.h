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

#ifndef DLSS_UTIL_H
#define DLSS_UTIL_H

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#ifdef __cplusplus
#define INLINE inline
#else
#define INLINE
//#define fmod mod
#define floorf floor
#endif


#ifndef __cplusplus
// Specular albedo for DLSS
float3 EnvBRDFApprox2(float3 SpecularColor, float alpha, float NoV)
{
  NoV = abs(NoV);
  // [Ray Tracing Gems, Chapter 32]
  float4 X;
  X.x = 1.f;
  X.y = NoV;
  X.z = NoV * NoV;
  X.w = NoV * X.z;
  float4 Y;
  Y.x            = 1.f;
  Y.y            = alpha;
  Y.z            = alpha * alpha;
  Y.w            = alpha * Y.z;
  float2x2 M1    = float2x2(0.99044f, -1.28514f, 1.29678f, -0.755907f);
  float3x3 M2    = float3x3(1.f, 2.92338f, 59.4188f, 20.3225f, -27.0302f, 222.592f, 121.563f, 626.13f, 316.627f);
  float2x2 M3    = float2x2(0.0365463f, 3.32707, 9.0632f, -9.04756);
  float3x3 M4    = float3x3(1.f, 3.59685f, -1.36772f, 9.04401f, -16.3174f, 9.22949f, 5.56589f, 19.7886f, -20.2123f);
  float    bias  = dot(mul(M1, X.xy), Y.xy) * rcp(dot(mul(M2, X.xyw), Y.xyw));
  float    scale = dot(mul(M3, X.xy), Y.xy) * rcp(dot(mul(M4, X.xzw), Y.xyw));
  // This is a hack for specular reflectance of 0
  bias *= saturate(SpecularColor.g * 50);
  return mad(SpecularColor, max(0, scale), max(0, bias));
}

// Pixel-space motion vector for DLSS / DLAA. The two overloads share one piece of math:
// `MV = (prev_NDC - curr_NDC) * 0.5 * resolution` (equivalent to `(prev_UV - curr_UV) * resolution`).
// NGX expects pixel-space MVs when InMVScaleX/Y = 1.0, which is what both renderers configure.
//
// Overload 1 (raster vertex shader): the vertex shader has already transformed the world position
// through the unjittered viewProj / prevMVP, so the fragment shader just consumes the two clip
// positions interpolated by the rasterizer.
inline float2 calculateMotionVector(float4 currClip, float4 prevClip, float2 resolution)
{
  float2 currNDC = currClip.xy / currClip.w;
  float2 prevNDC = prevClip.xy / prevClip.w;
  return (prevNDC - currNDC) * 0.5f * resolution;
}

// Overload 2 (path tracer): homogeneous current/previous positions reprojected through the two
// cameras. The w component selects the behavior, which is the elegant part:
//   - Surface hits pass w = 1. The previous position is the same surface point under the previous
//     object transform, so the single screen-space difference is the combined camera + object motion
//     (correct under perspective, NOT an additive sum of two MVs). A static surface passes
//     prevWorldPos == currWorldPos and reduces to camera-only motion.
//   - Sky / background pass w = 0: a point at infinity along the (world-space) view direction. Camera
//     translation then cancels out of the projection (only rotation moves the sky), which is the
//     physically correct behavior and is fully projection-independent -- no far-plane distance or
//     magic constants. (Approach from the vk_denoise_dlssrr sample.)
inline float2 calculateMotionVector(float4   currWorldPos,  // Current homogeneous position (w=1 surface, w=0 sky)
                                    float4   prevWorldPos,  // Previous homogeneous position of the same point
                                    float4x4 prevMVP,       // Previous frame's view-projection matrix
                                    float4x4 currentMVP,    // Current frame's view-projection matrix
                                    float2   resolution)      // Render target resolution
{
  float4 currentClipPos = mul(currWorldPos, currentMVP);
  float4 prevClipPos    = mul(prevWorldPos, prevMVP);
  return calculateMotionVector(currentClipPos, prevClipPos, resolution);
}
#endif

// #DLSS - Halton halton low discrepancy sequence, from https://www.shadertoy.com/view/wdXSW8
INLINE float2 halton(int index)
{
  const float2 coprimes = float2(2.0F, 3.0F);
  float2       s        = float2(index, index);
  float4       a        = float4(1, 1, 0, 0);
  while(s.x > 0. && s.y > 0.)
  {
    a.x = a.x / coprimes.x;
    a.y = a.y / coprimes.y;
    a.z += a.x * fmod(s.x, coprimes.x);
    a.w += a.y * fmod(s.y, coprimes.y);
    s.x = floorf(s.x / coprimes.x);
    s.y = floorf(s.y / coprimes.y);
  }
  return float2(a.z, a.w);
}

INLINE uint32_t wangHash(uint32_t seed)
{
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9;
  seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d;
  seed = seed ^ (seed >> 15);
  return seed;
}


INLINE float2 sampleDelta(uint32_t frameIndex)
{
  float2 delta;
  delta.x = wangHash(frameIndex) / float(~0u);
  delta.y = wangHash(frameIndex * 2) / float(~0u);
  return delta - float2(0.5f);
}

INLINE float2 dlssJitter(uint32_t frameIndex)
{
  //return sampleDelta(frameIndex);
  return halton(frameIndex) - float2(0.5, 0.5);
}

NAMESPACE_SHADERIO_END()

#endif  // DLSS_UTIL_H
