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

#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/bsdf_types.h.slang"
#include "nvshaders/gltf_scene_io.h.slang"
#include "nvshaders/gltf_vertex_access.h.slang"
#include "nvshaders/hdr_env_sampling.h.slang"
#include "nvshaders/light_contrib.h.slang"
#include "nvshaders/pbr_ggx_microfacet.h.slang"
#include "nvshaders/pbr_material_eval.h.slang"
#include "nvshaders/ray_utils.h.slang"
#include "nvshaders/sample_blur.h.slang"


#include "shaderio.h"
#include "get_hit.h.slang"
#include "common.h.slang"

// Bindings
// clang-format off
[[vk::push_constant]]                               ConstantBuffer<RasterPushConstant>  pushConst;
[[vk::binding(BindingPoints::eTextures, 0)]]        Sampler2D                           allTextures[];
[[vk::binding(BindingPoints::eTexturesHdr, 0)]]     Sampler2D                           texturesHdr[];
[[vk::binding(BindingPoints::eTexturesCube, 0)]]    SamplerCube                         texturesCube[];

// clang-format on


// Define the vertex attributes that will be provided by the vertex buffer
// Note: In Slang, we use semantic annotations (: POSITION, etc.) to specify
// how these fields map to vertex buffer data
struct VertexInput
{
  [[vk::binding(0)]]
  float3 position : POSITION;  // Vertex position in model space
};

// Define the data that will be interpolated and passed from vertex to fragment shader
// Note: SV_Position is a special semantic in Slang/HLSL for the clip-space position
struct VertexOutput
{
  float4 position : SV_Position;  // Clip space position (required)
  float3 worldPos;
};

// Define the final output of the fragment shader
// Note: SV_TARGET specifies this is the color output to the render target
struct PixelOutput
{
  float4 color : SV_TARGET0;      // First render target output (RGBA)
  float4 selection : SV_TARGET1;  // Second render target output (RGBA)
};


float3 getDiffuseLight(float3 n)
{
  float3 dir = rotate(n, float3(0, 1, 0), -pushConst.frameInfo.envRotation);
  return texturesCube[HDR_DIFFUSE_INDEX].SampleLevel(dir, 0.0f).xyz * pushConst.frameInfo.envIntensity;
}

float4 getSpecularSample(float3 reflection, float lod)
{
  float3 dir = rotate(reflection, float3(0, 1, 0), -pushConst.frameInfo.envRotation);
  return texturesCube[HDR_GLOSSY_INDEX].SampleLevel(dir, lod) * pushConst.frameInfo.envIntensity;
}

float3 getIBLRadianceGGX(float3 n, float3 v, float roughness, float3 F0)
{
  uint  lodLevel = 0;
  uint2 image_size;
  uint  numberOfLevels;
  texturesCube[HDR_GLOSSY_INDEX].GetDimensions(lodLevel, image_size.x, image_size.y, numberOfLevels);


  float  NdotV      = clampedDot(n, v);
  float  lod        = roughness * float(numberOfLevels - 1);
  float3 reflection = normalize(reflect(-v, n));

  float2 brdfSamplePoint = clamp(float2(NdotV, roughness), float2(0.0, 0.0), float2(1.0, 1.0));
  float2 f_ab            = texturesHdr[HDR_LUT_INDEX].Sample(brdfSamplePoint).rg;
  float4 specularSample  = getSpecularSample(reflection, lod);

  float3 specularLight = specularSample.rgb;

  // see https://bruop.github.io/ibl/#single_scattering_results at Single Scattering Results
  // Roughness dependent fresnel, from Fdez-Aguera
  float3 Fr     = max(float3(1.0 - roughness), F0) - F0;
  float3 k_S    = F0 + Fr * pow(1.0 - NdotV, 5.0);
  float3 FssEss = k_S * f_ab.x + f_ab.y;

  return specularLight * FssEss;
}

// specularWeight is introduced with KHR_materials_specular
float3 getIBLRadianceLambertian(float3 n, float3 v, float roughness, float3 diffuseColor, float3 F0)
{
  float  NdotV           = clampedDot(n, v);
  float2 brdfSamplePoint = clamp(float2(NdotV, roughness), float2(0.0, 0.0), float2(1.0, 1.0));
  float2 f_ab            = texturesHdr[HDR_LUT_INDEX].Sample(brdfSamplePoint).rg;

  float3 irradiance = getDiffuseLight(n);

  // see https://bruop.github.io/ibl/#single_scattering_results at Single Scattering Results
  // Roughness dependent fresnel, from Fdez-Aguera

  float3 Fr  = max(float3(1.0 - roughness), F0) - F0;
  float3 k_S = F0 + Fr * pow(1.0 - NdotV, 5.0);
  float3 FssEss = k_S * f_ab.x + f_ab.y;  // <--- GGX / specular light contribution (scale it down if the specularWeight is low)

  // Multiple scattering, from Fdez-Aguera
  float  Ems    = (1.0 - (f_ab.x + f_ab.y));
  float3 F_avg  = (F0 + (1.0 - F0) / 21.0);
  float3 FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
  float3 k_D = diffuseColor * (1.0 - FssEss + FmsEms);  // we use +FmsEms as indicated by the formula in the blog post (might be a typo in the implementation)

  return (FmsEms + k_D) * irradiance;
}

//------------------------------------------------------------------------------
// Vertex Shader
//------------------------------------------------------------------------------
[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
  GltfRenderNode renderNode = pushConst.gltfScene.renderNodes[pushConst.renderNodeID];

  float3 pos = mul(float4(input.position, 1.0), renderNode.objectToWorld).xyz;

  VertexOutput output;
  output.worldPos = pos;
  output.position = mul(float4(pos, 1.0), pushConst.frameInfo.viewProjMatrix);


  return output;
}

//------------------------------------------------------------------------------
// Fragment/Pixel Shader
//------------------------------------------------------------------------------
[shader("fragment")]
PixelOutput fragmentMain(VertexOutput input, uint primitiveID: SV_PrimitiveID, float3 baryWeights: SV_Barycentrics)
{
  PixelOutput output;
  output.color.a = 1.0;

  // Setting up scene info
  GltfShadeMaterial material   = pushConst.gltfScene->materials[pushConst.materialID];      // Buffer of materials
  GltfRenderNode    renderNode = pushConst.gltfScene->renderNodes[pushConst.renderNodeID];  // Buffer of render nodes
  GltfRenderPrimitive renderPrimitive = pushConst.gltfScene->renderPrimitives[pushConst.renderPrimID];  // Buffer of meshes

  float3 worldRayOrigin =
      float3(pushConst.frameInfo.viewInv[3].x, pushConst.frameInfo.viewInv[3].y, pushConst.frameInfo.viewInv[3].z);

  HitState hit = getHitState(renderPrimitive, baryWeights, float4x3(renderNode.worldToObject),
                             float4x3(renderNode.objectToWorld), primitiveID, worldRayOrigin);

  // Evaluate the material at the hit point
  MeshState   mesh   = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, false);
  PbrMaterial pbrMat = evaluateMaterial(material, mesh, allTextures, pushConst.gltfScene->textureInfos);

  output.color.xyz = pbrMat.baseColor;
  output.color.a   = pbrMat.opacity * (1.0 - pbrMat.transmission);

  if(pushConst.renderNodeID == pushConst.frameInfo->selectedRenderNode)
    output.selection = float4(1);
  else
    output.selection = float4(0);


  if(pushConst.frameInfo.debugMethod != DebugMethod::eNone)
  {
    output.color.xyz = debugValue(pbrMat, hit, pushConst.frameInfo.debugMethod);
    output.color.a   = 1.0;
    return output;
  }

  if(material.alphaMode == AlphaMode::eAlphaModeMask)
  {
    if(pbrMat.opacity < material.alphaCutoff)
      discard;
  }

  const float3 worldRayDirection = normalize(hit.pos - worldRayOrigin);
  const float3 toEye             = -worldRayDirection;

  // Result
  float3 contribution = float3(0);

  float3 f0            = lerp(float3(0.04), pbrMat.baseColor, pbrMat.metallic);
  float  ambientFactor = 0.3;
  if(pushConst.frameInfo.environmentType == EnvSystem::eSky)
  {
    float3 ambientColor = lerp(float3(0.4F), float3(0.17F, 0.37F, 0.65F), pbrMat.N.y * 0.5 + 0.5) * ambientFactor;
    contribution += ambientColor * pbrMat.baseColor * f0;

    BsdfEvaluateData evalData;
    evalData.k1 = -worldRayDirection;
    evalData.k2 = pushConst.skyParams.sunDirection;

    bsdfEvaluate(evalData, pbrMat);

    const float3 w = pushConst.skyParams.sunDiskIntensity;
    contribution += w * evalData.bsdf_diffuse;
    contribution += w * evalData.bsdf_glossy;
  }
  else
  {
    // Calculate lighting contribution from image based lighting source (IBL)
    float  perceptualRoughness = lerp(pbrMat.roughness.r, pbrMat.roughness.g, 0.5);  // Ad-hoc anisotropic -> isotropic
    float3 c_diff              = lerp(pbrMat.baseColor, float3(0), pbrMat.metallic);

    float3 f_specular = getIBLRadianceGGX(pbrMat.N, -worldRayDirection, perceptualRoughness, f0);
    float3 f_diffuse  = getIBLRadianceLambertian(pbrMat.N, -worldRayDirection, perceptualRoughness, c_diff, f0);

    contribution += f_specular + f_diffuse;
  }

  contribution += pbrMat.emissive;  // emissive

  // All lights
  for(int i = 0; i < pushConst.gltfScene.numLights; i++)
  {
    GltfLight    light        = pushConst.gltfScene.lights[i];
    LightContrib lightContrib = singleLightContribution(light, hit.pos, pbrMat.N);

    BsdfEvaluateData evalData;
    evalData.k1 = -worldRayDirection;
    evalData.k2 = -lightContrib.incidentVector;

    bsdfEvaluate(evalData, pbrMat);

    const float3 w = lightContrib.intensity;
    contribution += w * evalData.bsdf_diffuse;
    contribution += w * evalData.bsdf_glossy;
  }


  output.color = float4(contribution, pbrMat.opacity);


  return output;
}

//------------------------------------------------------------------------------
// Wireframe Shader
// Return a constant color for wireframe rendering
[shader("fragment")]
PixelOutput fragmentWireframeMain(VertexOutput input)
{
  PixelOutput output;
  output.color = float4(0.4, 0.01, 0.01, 1);
  return output;
}
