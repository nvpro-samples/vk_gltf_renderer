/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// This shader applies a Sobel filter to detect edges in an R8 image and
// highlights them in the output RGBA image.

#include "shaderio.h"

struct PushConstantSilhouette
{
  float3 color;
};

// clang-format off
[[vk::binding(SilhouetteBindings::eObjectID)]]      Texture2D<float>    u_R8Buffer;
[[vk::binding(SilhouetteBindings::eRGBAIImage)]]    RWTexture2D<float4> u_RgbaImage;

[[vk::push_constant]]   ConstantBuffer<PushConstantSilhouette> pushConst;
// clang-format on

[shader("compute")]
[numthreads(SILHOUETTE_WORKGROUP_SIZE, SILHOUETTE_WORKGROUP_SIZE, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
  uint2 texSize;
  u_RgbaImage.GetDimensions(texSize.x, texSize.y);  // DispatchRaysDimensions();

  const uint2 pixelCoord = uint2(dispatchThreadID.xy);

  // Define the Sobel kernel in the x and y directions
  const float kernelX[9] = float[](-1, 0, 1, -2, 0, 2, -1, 0, 1);
  const float kernelY[9] = float[](-1, -2, -1, 0, 0, 0, 1, 2, 1);

  float sumX = 0.0;
  float sumY = 0.0;

  // Check boundaries
  if(pixelCoord.x > 0 && pixelCoord.x < texSize.x - 1 && pixelCoord.y > 0 && pixelCoord.y < texSize.y - 1)
  {
    [unroll]
    for(int i = -1; i <= 1; i++)
    {
      [unroll]
      for(int j = -1; j <= 1; j++)
      {
        const uint2 offset = uint2(i, j);
        const float val    = u_R8Buffer[pixelCoord + offset].r;
        sumX += val * kernelX[(i + 1) * 3 + (j + 1)];
        sumY += val * kernelY[(i + 1) * 3 + (j + 1)];
      }
    }
  }

  // Compute the magnitude of the gradient
  const float magnitude = (sumX * sumX + sumY * sumY);

  // If the magnitude is above a certain threshold, we consider it an edge
  const float min_threshold = 2.0;
  const float max_threshold = 20.0;

  // Write the result to the output image
  u_RgbaImage[pixelCoord].xyz =
      lerp(u_RgbaImage[pixelCoord].xyz, pushConst.color, smoothstep(min_threshold, max_threshold, magnitude));
}
