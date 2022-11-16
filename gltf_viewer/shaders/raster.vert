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
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "device_host.h"
#include "dh_bindings.h"
#include "nvvkhl/shaders/dh_scn_desc.h"

// clang-format off
layout(buffer_reference, scalar) buffer  InstancesInfo { InstanceInfo i[]; };

layout(set = 0, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; } ;
// clang-format on

layout(push_constant) uniform RasterPushConstant_
{
  PushConstant pc;
};

layout(location = 0) in vec4 i_pos;
layout(location = 1) in vec4 i_normal;
layout(location = 2) in vec4 i_tangent;

layout(location = 1) out vec2 o_texCoord;
layout(location = 2) out vec3 o_normal;
layout(location = 3) out vec3 o_viewDir;
layout(location = 4) out vec3 o_pos;
layout(location = 5) out vec4 o_tangent;

out gl_PerVertex
{
  vec4 gl_Position;
};


void main()
{
  InstancesInfo instances = InstancesInfo(sceneDesc.instInfoAddress);
  InstanceInfo  instinfo  = instances.i[pc.instanceId];

  vec3 origin   = vec3(frameInfo.viewInv * vec4(0, 0, 0, 1));
  vec3 position = i_pos.xyz;
  vec3 normal   = i_normal.xyz;

  o_pos         = vec3(instinfo.objMatrix * vec4(position, 1.0));
  o_viewDir     = vec3(o_pos - origin);
  o_normal      = vec3(instinfo.objMatrixIT * vec4(normal, 0.0));
  o_texCoord    = vec2(i_pos.w, i_normal.w);
  o_tangent.xyz = (vec3(mat4(instinfo.objMatrix) * vec4(i_tangent.xyz, 0)));
  o_tangent.w   = i_tangent.w;

  gl_Position = frameInfo.proj * frameInfo.view * vec4(o_pos, 1.0);
}
