/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Compute shader to copy Vulkan images to linear buffers for OptiX denoiser
// This shader reads from tiled images and writes to linear buffers that can be
// exported to CUDA/OptiX. It also decompresses normals that were compressed in
// the path tracer.

#include "nvshaders/normal_compress.h.slang"
#include "shaderio.h"

// Bindings
[[vk::binding(OptixBindingPoints::eInRgba)]]
Texture2D<float4> inResultImage; // RGB rendered image
[[vk::binding(OptixBindingPoints::eInAlbedoNormal)]]
Texture2D<float4> inAlbedoNormalImage; // Albedo (RGB) + compressed normal (A)
[[vk::binding(OptixBindingPoints::eOutRgba, 0)]]
RWStructuredBuffer<float4> outRgbBuffer; // Output RGB buffer (linear)
[[vk::binding(OptixBindingPoints::eOutAlbedo, 0)]]
RWStructuredBuffer<float4> outAlbedoBuffer; // Output Albedo buffer (linear)
[[vk::binding(OptixBindingPoints::eOutNormal, 0)]]
RWStructuredBuffer<float4> outNormalBuffer;  // Output Normal buffer (linear)

struct PushConstant
{
  uint2 imageSize;
};

[[vk::push_constant]]
ConstantBuffer<PushConstant> pushConst;

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

  // Read albedo + compressed normal
  float4 albedoNormal = inAlbedoNormalImage[pixelCoord];

  // Write albedo (RGB channels, set alpha to 1.0 for OptiX)
  outAlbedoBuffer[bufferIndex] = float4(albedoNormal.rgb, 1.0);

  // Decompress normal from alpha channel
  uint   encodedNormal = asuint(albedoNormal.a);
  float3 normal        = decompressUnitVec(encodedNormal);

  // Write normal (XYZ in RGB, alpha = 1.0)
  outNormalBuffer[bufferIndex] = float4(normal, 1.0);
}

