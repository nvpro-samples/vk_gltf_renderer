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
#extension GL_ARB_separate_shader_objects : enable

// Scene UBO
layout(set = 0, binding = 0) uniform UBOscene
{
  mat4  projection;
  mat4  modelView;
  vec4  camPos;
  vec4  lightDir;
  float lightIntensity;
  float exposure;
}
uboScene;


// Input
layout(location = 0) in vec3 inPos;
//layout(location = 1) in vec3 inNormal;
//layout(location = 2) in vec3 inColor;
//layout(location = 3) in vec2 inUV;


// Output
layout(location = 0) out vec3 outWorldPos;
//layout(location = 1) out vec3 outNormal;
//layout(location = 2) out vec3 outColor;
//layout(location = 3) out vec2 outUV;

out gl_PerVertex
{
  vec4 gl_Position;
};


void main()
{
  vec4 pos      = vec4(inPos.xyz, 1.0);
  gl_Position   = uboScene.projection * pos;
  gl_Position.z = 0.5; // always draw
  mat4 m        = inverse(uboScene.modelView);
  m[3][0]       = 0.0;
  m[3][1]       = 0.0;
  m[3][2]       = 0.0;
  outWorldPos   = vec3(m * pos).xyz;
}
