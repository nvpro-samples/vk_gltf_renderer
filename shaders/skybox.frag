/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
 
#version 450
#extension GL_GOOGLE_include_directive : enable

#include "tonemapping.glsl"
//#include "functions.glsl"

layout(set = 1, binding = 0) uniform sampler2D samplerEnv;

layout(set = 0, location = 0) in vec3 inWorldPosition;

layout(set = 0, location = 0) out vec4 outColor;


layout(set = 0, binding = 0) uniform UBOscene
{
  mat4  projection;
  mat4  model;
  vec4  camPos;
  vec4  lightDir;
  float lightRadiance;
  float exposure;
  float gamma;
  int   materialMode;
  int   tonemap;
}
ubo;

const float ONE_OVER_PI = 0.3183099;

vec2 get_spherical_uv(vec3 v)
{
    float gamma = asin(-v.y);
    float theta = atan(v.z, v.x);

    vec2 uv = vec2(theta * ONE_OVER_PI * 0.5, gamma * ONE_OVER_PI) + 0.5;
    return uv;
}



void main()
{
  vec2 uv    = get_spherical_uv(normalize(inWorldPosition));
  vec3 color = texture(samplerEnv, uv).rgb;

  color    = toneMap(color, ubo.tonemap, ubo.gamma, ubo.exposure);
  outColor = vec4(color, 1.0);
}
