/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Rasterizer background pass for authored (OMI_environment_sky) sky types.
//
// The path tracer gets its background by evaluating sky_background.h.slang per escaped ray; the
// rasterizer has no such ray, so it fills the colour target here before geometry is drawn. Both
// call the same evaluator on the same frame data, which is what keeps a raster frame and a path-
// traced frame showing the same sky -- including the sun, which the baked lighting field does not
// contain.
//
// Mirrors nvshaders/sky_physical.slang, the pass this one sits beside for the procedural sky.

#include "nvshaders/functions.h.slang"
#include "shaderio.h"  // provides WORKGROUP_SIZE
#include "camera_ray.h.slang"
#include "sky_background.h.slang"

// clang-format off
[[vk::push_constant]]                                        ConstantBuffer<SkyBackgroundPushConstant> pushConst;
[[vk::binding(SkyBackgroundBindings::eSkyBackgroundOutImage)]] RWTexture2D<float4>                     outImage;
// clang-format on

[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void main(uint3 globalThreadID: SV_DispatchThreadID)
{
  uint2 imageSize;
  outImage.GetDimensions(imageSize.x, imageSize.y);

  if(globalThreadID.x >= imageSize.x || globalThreadID.y >= imageSize.y)
    return;

  // The same construction the path tracer uses, from the same function -- see camera_ray.h.slang
  // for what went wrong when this pass had its own.
  const RayDesc ray =
      getRay(float2(globalThreadID.xy), float2(0.5F), float2(imageSize), pushConst.frameInfo->projInv,
             pushConst.frameInfo->viewInv, (pushConst.frameInfo->flags & SceneFrameInfoFlags::eSceneIsOrthographic) != 0);
  float3 rayDir = ray.Direction;

  // Same rotation the lighting field applies on lookup, so turning the environment turns the
  // visible sky and the light on it together.
  rayDir = quatRotateInverse(pushConst.frameInfo->envRotation, rayDir);

  if(pushConst.mode == SkyBackgroundMode::eSkyBackgroundSunDisk)
  {
    // Additive: the dome has already written the baked sky into this pixel, and all that is
    // missing from it is the disk. Reading back what the dome wrote is what makes this a
    // composite rather than a second, conflicting background.
    const float3 disk = evalSunDiskBackground(rayDir, pushConst.frameInfo) * pushConst.frameInfo->envIntensity;
    if(any(disk > 0.0F))
      outImage[int2(globalThreadID.xy)] += float4(disk, 0.0F);
    return;
  }

  const float3 color = evalAuthoredSkyBackground(rayDir, pushConst.frameInfo) * pushConst.frameInfo->envIntensity;

  // Alpha is the clouds seam the plan reserves for a transmittance term; 1.0 until that ships.
  outImage[int2(globalThreadID.xy)] = float4(color, 1.0F);
}
