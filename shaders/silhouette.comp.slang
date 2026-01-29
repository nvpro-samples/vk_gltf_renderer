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
  int2 texSize;
  u_RgbaImage.GetDimensions(texSize.x, texSize.y);  // DispatchRaysDimensions();

  const int2 pixelCoord = int2(dispatchThreadID.xy);

  // Early exit if pixel is outside image bounds
  if(pixelCoord.x >= texSize.x || pixelCoord.y >= texSize.y)
    return;

  // Define the Sobel kernel in the x and y directions
  const float kernelX[9] = float[](-1, 0, 1, -2, 0, 2, -1, 0, 1);
  const float kernelY[9] = float[](-1, -2, -1, 0, 0, 0, 1, 2, 1);

  float sumX = 0.0;
  float sumY = 0.0;

  // Apply Sobel filter with proper boundary handling
  [unroll]
  for(int i = -1; i <= 1; i++)
  {
    [unroll]
    for(int j = -1; j <= 1; j++)
    {
      const int2 offset = int2(i, j);
      const int2 sampleCoord = pixelCoord + offset;
      
      // Clamp coordinates to image bounds
      const int2 clampedCoord = int2(
        clamp(sampleCoord.x, 0, texSize.x - 1),
        clamp(sampleCoord.y, 0, texSize.y - 1)
      );
      
      const float val = u_R8Buffer[clampedCoord].r;
      sumX += val * kernelX[(i + 1) * 3 + (j + 1)];
      sumY += val * kernelY[(i + 1) * 3 + (j + 1)];
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
