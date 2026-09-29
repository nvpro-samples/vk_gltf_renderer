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

#ifndef CAMERA_RAY_H
#define CAMERA_RAY_H 1

//
// The camera ray for a pixel -- one definition, shared.
//
// It lived in the path tracer, and the rasterizer's sky-background pass grew its own. The two
// disagreed: that copy built a world-space *point* on the far plane and normalised it, which is a
// direction from the world origin rather than from the eye. Identical wherever the camera happened
// to sit at the origin, and wrong in proportion to how far away it was -- a half-degree sun disk
// landed about 70 pixels from where the path tracer drew it at 500 units out.
//
// Anything that reconstructs a ray from a pixel should call this rather than re-derive it. The
// two renderers agreeing about where the sky is has to be structural, not a coincidence that
// holds until someone edits one of them.
//

#include "nvshaders/functions.h.slang"

RayDesc getRay(float2 samplePos, float2 offset, float2 imageSize, float4x4 projMatrixI, float4x4 viewMatrixI, bool isOrthographic)
{
  const float2 clipCoords = (samplePos + offset) / imageSize * 2.0 - 1.0;
  float4       viewCoords = mul(float4(clipCoords, -1.0, 1.0), projMatrixI);
  viewCoords /= viewCoords.w;

  RayDesc ray;
  if(isOrthographic)
  {
    ray.Origin    = mul(viewCoords, viewMatrixI).xyz;
    ray.Direction = normalize(mul(float4(0, 0, -1, 0), viewMatrixI).xyz);
  }
  else
  {
    ray.Origin    = viewMatrixI[3].xyz;
    ray.Direction = normalize(mul(viewCoords, viewMatrixI).xyz - ray.Origin);
  }
  ray.TMin = 0.0;
  ray.TMax = INFINITE;
  return ray;
}

#endif  // CAMERA_RAY_H
