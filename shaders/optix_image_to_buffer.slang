/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Compute shader to copy Vulkan images to linear buffers for OptiX denoiser
// This shader reads from tiled images and writes to linear buffers that can be
// exported to CUDA/OptiX. The path tracer accumulates the albedo and the world-space
// normal guides (the same guides as for OIDN); OptiX wants normalized camera-space
// normals, (0,0,0) where nothing was hit, so the normal is converted here.

#include "shaderio.h"

// Bindings
[[vk::binding(OptixBindingPoints::eInRgba)]]
Texture2D<float4> inResultImage;  // RGB rendered image
[[vk::binding(OptixBindingPoints::eInAlbedo)]]
Texture2D<float4> inAlbedoImage;  // Accumulated albedo guide
[[vk::binding(OptixBindingPoints::eInNormal)]]
Texture2D<float4> inNormalImage;  // Accumulated world-space normal guide
[[vk::binding(OptixBindingPoints::eOutRgba, 0)]]
RWStructuredBuffer<float4> outRgbBuffer;  // Output RGB buffer (linear)
[[vk::binding(OptixBindingPoints::eOutAlbedo, 0)]]
RWStructuredBuffer<float4> outAlbedoBuffer;  // Output Albedo buffer (linear)
[[vk::binding(OptixBindingPoints::eOutNormal, 0)]]
RWStructuredBuffer<float4> outNormalBuffer;  // Output Normal buffer (linear)

[[vk::push_constant]]
ConstantBuffer<OptixCopyPushConstant> pushConst;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 threadIdx: SV_DispatchThreadID)
{
  uint2 pixelCoord = threadIdx.xy;

  // Bounds check
  if(pixelCoord.x >= pushConst.imageSize.x || pixelCoord.y >= pushConst.imageSize.y)
    return;

  // Calculate linear buffer index
  uint bufferIndex = pixelCoord.y * pushConst.imageSize.x + pixelCoord.x;

  // Read RGB image
  float4 rgbColor           = inResultImage[pixelCoord];
  outRgbBuffer[bufferIndex] = rgbColor;

  // Write albedo (RGB channels, set alpha to 1.0 for OptiX)
  outAlbedoBuffer[bufferIndex] = float4(inAlbedoImage[pixelCoord].rgb, 1.0);

  // Camera space: right-handed, -Z forward, Y up, X right. The average of the samples' normals is
  // shorter than 1 on edges; it is renormalized, and stays zero where no sample hit anything.
  float3 normal = mul(float3x3(pushConst.frameInfo->viewMatrix), inNormalImage[pixelCoord].xyz);
  float  len    = length(normal);
  normal        = len > 1e-6f ? normal / len : float3(0.0f);

  // Write normal (XYZ in RGB, alpha = 1.0)
  outNormalBuffer[bufferIndex] = float4(normal, 1.0);
}
