/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#extension GL_EXT_ray_query : require


#define USE_FIREFLY_FILTER 1
#define USE_RUSIAN_ROULETTE 1

#include "device_host.h"
#include "dh_bindings.h"
#include "payload.h"
#include "get_hit.h"

HitPayload hitPayload;  // Global hit payload

layout(local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE) in;
#include "rt_layout.h"

#include "nvvkhl/shaders/pbr_mat_eval.h"      // Need texturesMap[]
#include "nvvkhl/shaders/hdr_env_sampling.h"  // nedd envSamplingData[]


// Depend on hitPayload and other layouts
#include "rt_indirect.h"
#include "rt_common.h"


/// ---------------------------------------------------------------------------------------
///
///
void main()
{
  vec2 imageSize = imageSize(image);
  vec2 samplePos = vec2(gl_GlobalInvocationID.xy);

  if(samplePos == pc.mouseCoord)
  {
    useDebug = true;
  }

  // Debugging, single frame
  if(pc.dbgMethod != eDbgMethod_none)
  {
    if(pc.frame == 0)
    {
      vec3 result = debugRendering(samplePos, imageSize);
      imageStore(image, ivec2(samplePos.xy), vec4(result, 1.0F));
      selectObject(samplePos, imageSize);
    }
    return;
  }

  // Initialize the random number
  uint seed = xxhash32(uvec3(samplePos.xy, pc.frame));

  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  vec2 subpixelJitter = pc.frame == 0 ? vec2(0.5f, 0.5f) : vec2(rand(seed), rand(seed));

  float focalDistance = pc.focalDistance;
  float aperture      = pc.aperture;

  // Sampling n times the pixel

  SampleResult sampleResult = samplePixel(seed, samplePos, subpixelJitter, imageSize, frameInfo.projMatrixI,
                                          frameInfo.viewMatrixI, focalDistance, aperture);
  vec3         pixel_color  = sampleResult.radiance;
  for(int s = 1; s < pc.maxSamples; s++)
  {
    subpixelJitter = vec2(rand(seed), rand(seed));
    sampleResult = samplePixel(seed, samplePos, subpixelJitter, imageSize, frameInfo.projMatrixI, frameInfo.viewMatrixI,
                               focalDistance, aperture);
    pixel_color += sampleResult.radiance;
  }
  pixel_color /= pc.maxSamples;

  if(pc.frame == 0)  // first frame
  {
    imageStore(image, ivec2(samplePos.xy), vec4(pixel_color, 1.0F));
    imageStore(normalDepth, ivec2(samplePos.xy), vec4(sampleResult.normal, sampleResult.depth));
  }
  else
  {
    // Do accumulation over time
    float a         = 1.0F / float(pc.frame + 1);
    vec3  old_color = imageLoad(image, ivec2(samplePos.xy)).xyz;
    imageStore(image, ivec2(samplePos.xy), vec4(mix(old_color, pixel_color, a), 1.0F));

    // Normal Depth buffer update
    vec4  oldNormalDepth = imageLoad(normalDepth, ivec2(samplePos.xy));
    float new_depth      = min(oldNormalDepth.w, sampleResult.depth);
    vec3  new_normal     = normalize(mix(oldNormalDepth.xyz, sampleResult.normal, a));

    // Write to the normalDepth buffer
    imageStore(normalDepth, ivec2(samplePos.xy), vec4(new_normal, new_depth));
  }

  // Adding to the selection buffer the selected object id
  selectObject(samplePos, imageSize);
}
