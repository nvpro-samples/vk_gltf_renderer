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

// DLSS hand-off: the image shown while the accumulation denoiser (OptiX or OIDN) takes over from
// DLSS-RR, a blend of their two images (see src/denoiser_handoff.hpp).

#include "shaderio.h"

// clang-format off
[[vk::binding(DenoiserHandoffBindings::eHandoffDlss)]]     RWTexture2D<float4> u_dlss;
[[vk::binding(DenoiserHandoffBindings::eHandoffDenoised)]] RWTexture2D<float4> u_denoised;
[[vk::binding(DenoiserHandoffBindings::eHandoffOut)]]      RWTexture2D<float4> u_out;

[[vk::push_constant]] ConstantBuffer<DenoiserHandoffPushConstant> pushConst;
// clang-format on

static const float kHalfMax = 65504.0;  // Largest half float: the output is RGBA16F

[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
  int2 size;
  u_out.GetDimensions(size.x, size.y);
  const int2 pixel = int2(dispatchThreadID.xy);
  if(pixel.x >= size.x || pixel.y >= size.y)
    return;

  // Written as a weighted sum, not a + (b - a) * w: an infinite input would make that NaN. The HDR
  // radiance can exceed the half-float range (a sun disk): clamp rather than overflow to infinity.
  const float  w     = pushConst.weight;
  const float4 color = u_dlss[pixel] * (1.0 - w) + u_denoised[pixel] * w;
  u_out[pixel]       = clamp(color, -kHalfMax, kHalfMax);
}
