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

#include "nvvkhl/shaders/constants.glsl"
#include "nvvkhl/shaders/ggx.glsl"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/dh_hdr.h"
#include "nvvkhl/shaders/dh_scn_desc.h"
#include "nvvkhl/shaders/random.glsl"
#include "nvvkhl/shaders/pbr_eval.glsl"


hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(buffer_reference, scalar) readonly buffer Vertices  { Vertex v[]; };
layout(buffer_reference, scalar) readonly buffer Indices   { uvec3 i[]; };
layout(buffer_reference, scalar) readonly buffer PrimMeshInfos { PrimMeshInfo i[]; };
layout(buffer_reference, scalar) readonly buffer Materials { GltfShadeMaterial m[]; };

#include "get_hit.glsl"

layout(set = 0, binding = eTlas ) uniform accelerationStructureEXT topLevelAS;
layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 1, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(set = 1, binding = eTextures)  uniform sampler2D texturesMap[]; // all textures
layout(set = 2, binding = eSkyParam) uniform SkyInfo_ { ProceduralSkyShaderParameters skyInfo; };
layout(set = 3, binding = eImpSamples,  scalar)	buffer _EnvAccel { EnvAccel envSamplingData[]; };
layout(set = 3, binding = eHdr) uniform sampler2D hdrTexture;

layout(push_constant) uniform RtxPushConstant_ { PushConstant pc; };
// clang-format on

// Includes depending on layout description
#include "nvvkhl/shaders/shading.glsl"   // envSamplingData
#include "nvvkhl/shaders/mat_eval.glsl"  // texturesMap

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void main()
{
  // Retrieve the Primitive mesh buffer information
  PrimMeshInfos pInfo_ = PrimMeshInfos(sceneDesc.primInfoAddress);
  PrimMeshInfo  pinfo  = pInfo_.i[gl_InstanceCustomIndexEXT];

  HitState hit = getHitState(pinfo.vertexAddress, pinfo.indexAddress);

  // Scene materials
  uint      matIndex  = max(0, pinfo.materialIndex);  // material of primitive mesh
  Materials materials = Materials(sceneDesc.materialAddress);

  // Material of the object and evaluated material (includes textures)
  GltfShadeMaterial mat    = materials.m[matIndex];
  PbrMaterial       pbrMat = evaluateMaterial(mat, hit.nrm, hit.tangent, hit.bitangent, hit.uv);

  payload.hitT         = gl_HitTEXT;
  ShadingResult result = shading(pbrMat, hit, hdrTexture, frameInfo.useSky == 1);

  payload.weight       = result.weight;
  payload.contrib      = result.radiance;
  payload.rayOrigin    = result.rayOrigin;
  payload.rayDirection = result.rayDirection;

  switch(frameInfo.dbgMethod)
  {
    case eDbgMethod_metallic:
      payload.contrib = vec3(pbrMat.metallic);
      break;
    case eDbgMethod_roughness:
      payload.contrib = vec3(pbrMat.roughness);
      break;
    case eDbgMethod_normal:
      payload.contrib = vec3(pbrMat.normal * .5 + .5);
      break;
    case eDbgMethod_basecolor:
      payload.contrib = vec3(pbrMat.albedo);
      break;
    case eDbgMethod_emissive:
      payload.contrib = vec3(pbrMat.emissive);
      break;
  }


  // -- Debug --
  //  payload.contrib = hit.nrm * .5 + .5;
  //  payload.contrib = matEval.tangent * .5 + .5;
  if(frameInfo.dbgMethod != eDbgMethod_none)
    stopPath();
}
