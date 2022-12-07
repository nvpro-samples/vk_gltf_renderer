/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "device_host.h"
#include "dh_bindings.h"
#include "payload.glsl"
#include "nvvkhl/shaders/dh_scn_desc.h"


hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(buffer_reference, scalar) readonly buffer Vertices { Vertex v[]; };
layout(buffer_reference, scalar) readonly buffer Indices { uvec3 i[]; };
layout(buffer_reference, scalar) readonly buffer PrimMeshInfos { PrimMeshInfo i[]; };

layout(set = 1, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
  // clang-format on


#include "nvvkhl/shaders/func.glsl"
#include "get_hit.glsl"


layout(constant_id = 0) const int USE_SER = 0;


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void main()
{
  // Retrieve the Primitive mesh buffer information
  PrimMeshInfos pInfo_ = PrimMeshInfos(sceneDesc.primInfoAddress);
  PrimMeshInfo  pinfo  = pInfo_.i[gl_InstanceCustomIndexEXT];

  payload.hitT          = gl_HitTEXT;
  payload.instanceIndex = gl_InstanceCustomIndexEXT;
  payload.hit           = getHitState(pinfo.vertexAddress, pinfo.indexAddress);
}
