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
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "device_host.h"
#include "dh_bindings.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/dh_scn_desc.h"
#include "nvvkhl/shaders/pbr_eval.glsl"
#include "nvvkhl/shaders/light_contrib.glsl"


// clang-format off
// Incoming 
layout(location = 1) in vec2 i_texCoord;
layout(location = 2) in vec3 i_normal;
layout(location = 3) in vec3 i_viewDir;
layout(location = 4) in vec3 i_pos;
layout(location = 5) in vec4 i_tangent;

// Outgoing
layout(location = 0) out vec4 outColor;

// Buffers
layout(buffer_reference, scalar) buffer  GltfMaterials { GltfShadeMaterial m[]; };
layout(set = 0, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; } ;
layout(set = 0, binding = eTextures) uniform sampler2D[] texturesMap;

layout(set = 1, binding = 0) uniform sampler2D   u_GGXLUT; // lookup table
layout(set = 1, binding = 1) uniform samplerCube u_LambertianEnvSampler; // 
layout(set = 1, binding = 2) uniform samplerCube u_GGXEnvSampler;  //

layout(set = 2, binding = eSkyParam) uniform SkyInfo_ { ProceduralSkyShaderParameters skyInfo; };

  // clang-format on


#include "nvvkhl/shaders/mat_eval.glsl"

layout(push_constant) uniform RasterPushConstant_
{
  PushConstant pc;
};

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  vec3 pos;
  vec3 nrm;
  vec2 uv;
  vec3 tangent;
  vec3 bitangent;
};

vec3 getDiffuseLight(vec3 n)
{
  vec3 dir = rotate(n, vec3(0, 1, 0), -frameInfo.envRotation);
  return texture(u_LambertianEnvSampler, dir).rgb * frameInfo.envColor.rgb;
}

vec4 getSpecularSample(vec3 reflection, float lod)
{
  vec3 dir = rotate(reflection, vec3(0, 1, 0), -frameInfo.envRotation);
  return textureLod(u_GGXEnvSampler, dir, lod) * frameInfo.envColor;
}

// Calculation of the lighting contribution
vec3 getIBLContribution(vec3 n, vec3 v, float roughness, vec3 diffuseColor, vec3 specularColor)
{
  int   u_MipCount = textureQueryLevels(u_GGXEnvSampler);
  float lod        = (roughness * float(u_MipCount - 1));
  vec3  reflection = normalize(reflect(-v, n));
  float NdotV      = clampedDot(n, v);

  // retrieve a scale and bias to F0. See [1], Figure 3
  vec3 brdf          = (texture(u_GGXLUT, vec2(NdotV, 1.0 - roughness))).rgb;
  vec3 diffuseLight  = getDiffuseLight(n);
  vec3 specularLight = getSpecularSample(reflection, lod).xyz;

  vec3 diffuse  = diffuseLight * diffuseColor;
  vec3 specular = specularLight * (specularColor * brdf.x + brdf.y);

  return diffuse + specular;
}

void main()
{
  // Material of the object
  GltfMaterials     gltfMats = GltfMaterials(sceneDesc.materialAddress);
  GltfShadeMaterial gltfMat  = gltfMats.m[pc.materialId];

  HitState hit;
  hit.pos       = i_pos;
  hit.nrm       = normalize(i_normal);
  hit.uv        = i_texCoord;
  hit.tangent   = normalize(i_tangent.xyz);
  hit.bitangent = cross(hit.nrm, hit.tangent) * i_tangent.w;

  if(gl_FrontFacing == false)
  {
    hit.tangent *= -1.0;
    hit.bitangent *= -1.0;
    hit.nrm *= -1.0;
  }

  PbrMaterial pbrMat = evaluateMaterial(gltfMat, hit.nrm, hit.tangent, hit.bitangent, hit.uv);

  switch(frameInfo.dbgMethod)
  {
    case eDbgMethod_metallic:
      outColor = vec4(vec3(pbrMat.metallic), 1);
      return;
    case eDbgMethod_roughness:
      outColor = vec4(vec3(pbrMat.roughness), 1);
      return;
    case eDbgMethod_normal:
      outColor = vec4(vec3(pbrMat.normal * .5 + .5), 1);
      return;
    case eDbgMethod_basecolor:
      outColor = vec4(vec3(pbrMat.albedo), 1);
      return;
    case eDbgMethod_emissive:
      outColor = vec4(vec3(pbrMat.emissive), 1);
      return;
  }

  if(gltfMat.alphaMode == ALPHA_MASK)
  {
    if(pbrMat.albedo.a < gltfMat.alphaCutoff)
      discard;
  }

  const vec3 eyePos            = vec3(frameInfo.viewInv[3].x, frameInfo.viewInv[3].y, frameInfo.viewInv[3].z);
  const vec3 worldRayDirection = normalize(hit.pos - eyePos);
  const vec3 toEye             = -worldRayDirection;

  // Result
  vec3 result = vec3(0);

  float ambientFactor = 0.3;
  if(frameInfo.useSky != 0)
  {
    vec3 ambientColor = mix(skyInfo.groundColor.rgb, skyInfo.skyColor.rgb, pbrMat.normal.y * 0.5 + 0.5) * ambientFactor;
    result += ambientColor * pbrMat.albedo.rgb;
    result += ambientColor * pbrMat.f0;
  }
  else
  {
    // Calculate lighting contribution from image based lighting source (IBL)
    vec3 diffuseColor  = pbrMat.albedo.rgb * (vec3(1.0) - pbrMat.f0) * (1.0 - pbrMat.metallic);
    vec3 specularColor = mix(pbrMat.f0, pbrMat.albedo.rgb, pbrMat.metallic);

    result += getIBLContribution(pbrMat.normal, toEye, pbrMat.roughness, diffuseColor, specularColor);
  }


  result += pbrMat.emissive;  // emissive

  // All lights
  for(int i = 0; i < frameInfo.nbLights; i++)
  {
    Light        light        = frameInfo.light[i];
    LightContrib lightContrib = singleLightContribution(light, hit.pos, pbrMat.normal, toEye);

    float pdf      = 0;
    vec3  brdf     = pbrEval(pbrMat, toEye, -lightContrib.incidentVector, pdf);
    vec3  radiance = brdf * lightContrib.intensity;
    result += radiance;
  }

  outColor = vec4(result, pbrMat.albedo.a);
}
