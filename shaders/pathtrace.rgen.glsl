/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_image_load_formatted : enable  // The folowing extension allow to pass images as function parameters
#extension GL_EXT_debug_printf : enable

layout(constant_id = 0) const int USE_SER = 0;  // This will be set to 1 if SER is available

#define USE_FIREFLY_FILTER 1
#define USE_RUSIAN_ROULETTE 1

#include "device_host.h"
#include "dh_bindings.h"
#include "payload.h"


layout(location = 0) rayPayloadEXT HitPayload hitPayload;
layout(location = 1) rayPayloadEXT ShadowPayload shadowPayload;

#include "rt_layout.h"

#include "nvvkhl/shaders/pbr_mat_eval.h"      // Need texturesMap[]
#include "nvvkhl/shaders/hdr_env_sampling.h"  // nedd envSamplingData[]

// Depend on hitPayload and other layouts
#include "rt_rtx.h"
#include "rt_common.h"


/// ---------------------------------------------------------------------------------------
///
///
void main()
{
  vec2 imageSize = gl_LaunchSizeEXT.xy;
  vec2 samplePos = vec2(gl_LaunchIDEXT.xy);

  // Initialize the random number
  uint seed = xxhash32(uvec3(gl_LaunchIDEXT.xy, frameInfo.frameCount));

  // Subpixel jitter: send the ray through a different position inside the
  // pixel each time, to provide antialiasing.
  vec2 subpixelJitter = vec2(0.5f, 0.5f);
  if(frameInfo.frameCount > 0)
    subpixelJitter += ANTIALIASING_STANDARD_DEVIATION * sampleGaussian(vec2(rand(seed), rand(seed)));

  float focalDistance = pc.focalDistance;
  float aperture      = pc.aperture;

  // Sampling n times the pixel

  SampleResult sampleResult = samplePixel(seed, samplePos, subpixelJitter, imageSize, frameInfo.projMatrixI,
                                          frameInfo.viewMatrixI, focalDistance, aperture);
  vec4         pixel_color  = sampleResult.radiance;
  for(int s = 1; s < pc.maxSamples; s++)
  {
    subpixelJitter = vec2(rand(seed), rand(seed));
    sampleResult = samplePixel(seed, samplePos, subpixelJitter, imageSize, frameInfo.projMatrixI, frameInfo.viewMatrixI,
                               focalDistance, aperture);
    pixel_color += sampleResult.radiance;
  }
  pixel_color /= pc.maxSamples;

  if(frameInfo.frameCount == 0)  // first frame
  {
    imageStore(image, ivec2(samplePos.xy), pixel_color);
    if(pc.useRTDenoiser == 1)
    {
      imageStore(normalDepth, ivec2(samplePos.xy), vec4(sampleResult.normal, sampleResult.depth));
    }
  }
  else
  {
    // Do accumulation over time
    float a         = 1.0F / float(frameInfo.frameCount + 1);
    vec4  old_color = imageLoad(image, ivec2(samplePos.xy));
    imageStore(image, ivec2(samplePos.xy), mix(old_color, pixel_color, a));

    if(pc.useRTDenoiser == 1)
    {
      // Normal Depth buffer update
      vec4  oldNormalDepth = imageLoad(normalDepth, ivec2(samplePos.xy));
      float new_depth      = min(oldNormalDepth.w, sampleResult.depth);
      vec3  new_normal     = normalize(mix(oldNormalDepth.xyz, sampleResult.normal, a));

      // Write to the normalDepth buffer
      imageStore(normalDepth, ivec2(samplePos.xy), vec4(new_normal, new_depth));
    }
  }

  selectObject(samplePos, imageSize);
}
