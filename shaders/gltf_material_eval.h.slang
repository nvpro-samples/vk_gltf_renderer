/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*-------------------------------------------------------------------------------------------------
glTF material evaluator (local fork, originally derived from nvpro_core2/nvshaders/pbr_material_eval.h.slang).

Samples the textures referenced by GltfShadeMaterial and returns a PbrMaterial struct for
the BSDF functions. The input is glTF-specific (KHR_materials_* extensions, UV transforms,
spec-gloss conversion); the output PbrMaterial is the generic BSDF interface defined in
nvpro_core2.

Design notes for the fork:
 - pbrMat is initialized with defaultPbrMaterial() (instead of being left uninitialized as
   upstream) so gating out an extension block with MAT_EXT_* cannot leave PbrMaterial fields
   undefined when downstream BSDF code reads them.
 - PbrMaterial and the BSDF interface still live in nvpro_core2; we keep including
   nvshaders/pbr_material_types.h.slang.
 - Extension-specific branches are wrapped in #if MAT_EXT_*. The matching field is only
   present on GltfShadeMaterial when the same flag is 1, so host and device must be built
   with consistent flags.
-------------------------------------------------------------------------------------------------*/

#ifndef GLTF_MATERIAL_EVAL_H
#define GLTF_MATERIAL_EVAL_H 1

#include "nvshaders/pbr_material_types.h.slang"
#include "gltf_scene_io.h.slang"

struct MeshState
{
  float3 N;      // Normal
  float3 T;      // Tangent
  float3 B;      // Bitangent
  float3 Ng;     // Geometric normal
  float2 tc[2];  // Texture coordinates
  bool   isInside = false;
  float  texGrad  = 0.0f;  // Ray footprint for SampleGrad; 0 -> Sample() (mip 0 in RT, proper mips in fragment)
};


/*-------------------------------------------------------------------------------------------------
# `NO_TEXTURES` Define
> Define this before including `gltf_material_eval.h` to use a color of `float4(1.0f)`
> for everything instead of reading textures.
-------------------------------------------------------------------------------------------------*/
#ifndef NO_TEXTURES
#define USE_TEXTURES
#endif

/*
texGrad > 0: SampleGrad (path trace ray footprint). texGrad == 0: Sample() — fragment gets real mips;
in ray trace / compute implicit LOD is undefined and effectively samples like mip 0, same as needed there.
*/
float4 getTexture(Sampler2D textures[], in GltfTextureInfo textureInfo, in float2 texCoords[2], float texGrad)
{
#ifdef USE_TEXTURES
  float2 tt = texCoords[textureInfo.texCoord];
#if MAT_EXT_TEXTURE_TRANSFORM
  // KHR_texture_transform
  tt = mul(float3(tt, 1), textureInfo.uvTransform).xy;
#endif
  if(texGrad > 0.0f)
  {
#if MAT_EXT_TEXTURE_TRANSFORM
    float2 ddx_uv = textureInfo.uvTransform[0] * texGrad;
    float2 ddy_uv = textureInfo.uvTransform[1] * texGrad;
#else
    // Identity transform: ddx/ddy are axis-aligned unit vectors scaled by the ray footprint.
    float2 ddx_uv = float2(texGrad, 0.0f);
    float2 ddy_uv = float2(0.0f, texGrad);
#endif
    return textures[textureInfo.index].SampleGrad(tt, ddx_uv, ddy_uv);
  }
  return textures[textureInfo.index].Sample(tt);
#else
  return float4(1.0F);
#endif
}

/* 
Check if a texture is present.
*/
bool isTexturePresent(uint16_t tinfo)
{
  return tinfo > 0;
}

#if MAT_EXT_VOLUME_SCATTER
/*-------------------------------------------------------------------------------------------------
* Convert multi-scatter albedo to single-scatter albedo
* Formula from KHR_materials_volume_scatter specification
* -------------------------------------------------------------------------------------------------*/
float3 multiToSingleScatterAlbedo(float3 rho_ms)
{
  float3 t = 4.09712 + 4.20863 * rho_ms - sqrt(9.59217 + 41.6808 * rho_ms + 17.7126 * rho_ms * rho_ms);
  return 1.0 - t * t;
}
#endif

#if MAT_EXT_SPECULAR_GLOSSINESS
/*-------------------------------------------------------------------------------------------------
* Convert PBR specular glossiness to metallic-roughness
* -------------------------------------------------------------------------------------------------*/
float3 convertSGToMR(float3 diffuseColor, float3 specularColor, float glossiness, out float metallic, out float2 roughness)
{
  // Constants
  const float dielectricSpecular = 0.04f;  // F0 for dielectrics

  // Compute metallic factor
  float specularIntensity = max(specularColor.r, max(specularColor.g, specularColor.b));
  float isMetal           = smoothstep(dielectricSpecular + 0.01f, dielectricSpecular + 0.05f, specularIntensity);
  metallic                = isMetal;

  // Compute base color
  float3 baseColor;
  if(metallic > 0.0f)
  {
    baseColor = specularColor;
  }
  else
  {
    baseColor = diffuseColor / (1.0f - dielectricSpecular * (1.0f - metallic));
    baseColor = clamp(baseColor, 0.0f, 1.0f);
  }

  float r   = 1.0f - glossiness;
  roughness = float2(r * r);

  return baseColor;
}
#endif

/*-------------------------------------------------------------------------------------------------
# `MICROFACET_MIN_ROUGHNESS` Define
> Minimum roughness for microfacet models.
-------------------------------------------------------------------------------------------------*/
#define MICROFACET_MIN_ROUGHNESS 0.0014142f

/*-------------------------------------------------------------------------------------------------
# Function `evaluateMaterial`
> From the incoming `material` and `mesh` info, return a `PbrMaterial` struct for the BSDF system.
-------------------------------------------------------------------------------------------------*/
PbrMaterial evaluateMaterial(in GltfShadeMaterial material, MeshState state, Sampler2D textures[], GltfTextureInfo* texInfos)
{
  // Start from a fully-initialized material so that gating out any extension with MAT_EXT_*
  // cannot leave downstream BSDF reads undefined.
  PbrMaterial pbrMat = defaultPbrMaterial();

  // pbrMetallicRoughness (standard)
#if MAT_EXT_SPECULAR_GLOSSINESS
  if(material.pbrModel == PbrModel::ePbrModelMetallicRoughness)
#endif
  {
    // Base Color/Albedo may be defined from a base texture or a flat color
    float4 baseColor = material.pbrBaseColorFactor;
    if(isTexturePresent(material.pbrBaseColorTexture))
    {
      baseColor *= getTexture(textures, texInfos[material.pbrBaseColorTexture], state.tc, state.texGrad);
    }
    pbrMat.baseColor = baseColor.rgb;
    pbrMat.opacity   = baseColor.a;

    // Metallic-Roughness
    float roughness = material.pbrRoughnessFactor;
    float metallic  = material.pbrMetallicFactor;
    if(isTexturePresent(material.pbrMetallicRoughnessTexture))
    {
      // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
      float4 metallicRoughnessSample =
          getTexture(textures, texInfos[material.pbrMetallicRoughnessTexture], state.tc, state.texGrad);
      roughness *= metallicRoughnessSample.g;
      metallic *= metallicRoughnessSample.b;
    }
    roughness        = max(roughness, MICROFACET_MIN_ROUGHNESS);
    pbrMat.roughness = float2(roughness * roughness);  // Square roughness for the microfacet model
    pbrMat.metallic  = clamp(metallic, 0.0F, 1.0F);
  }
#if MAT_EXT_SPECULAR_GLOSSINESS
  else
  {
    // KHR_materials_pbrSpecularGlossiness: deprecated but still used in many places
    float4 diffuse    = material.pbrDiffuseFactor;
    float  glossiness = material.pbrGlossinessFactor;
    float3 specular   = material.pbrSpecularFactor;

    if(isTexturePresent(material.pbrDiffuseTexture))
    {
      diffuse *= getTexture(textures, texInfos[material.pbrDiffuseTexture], state.tc, state.texGrad);
    }

    if(isTexturePresent(material.pbrSpecularGlossinessTexture))
    {
      float4 specularGlossinessSample =
          getTexture(textures, texInfos[material.pbrSpecularGlossinessTexture], state.tc, state.texGrad);
      specular *= specularGlossinessSample.rgb;
      glossiness *= specularGlossinessSample.a;
    }

    pbrMat.baseColor = convertSGToMR(diffuse.rgb, specular, glossiness, pbrMat.metallic, pbrMat.roughness);
    pbrMat.opacity   = diffuse.a;
  }
#endif


  // Occlusion Map
  pbrMat.occlusion = material.occlusionStrength;
  if(isTexturePresent(material.occlusionTexture))
  {
    float occlusion  = getTexture(textures, texInfos[material.occlusionTexture], state.tc, state.texGrad).r;
    pbrMat.occlusion = 1.0 + pbrMat.occlusion * (occlusion - 1.0);
  }

  // Normal Map
  pbrMat.N                = state.N;
  pbrMat.T                = state.T;
  pbrMat.B                = state.B;
  pbrMat.Ng               = state.Ng;
  bool needsTangentUpdate = false;

  if(isTexturePresent(material.normalTexture))
  {
    float3 normal_vector = getTexture(textures, texInfos[material.normalTexture], state.tc, state.texGrad).xyz;
    normal_vector        = normal_vector * 2.0F - 1.0F;
    normal_vector *= float3(material.normalTextureScale, material.normalTextureScale, 1.0F);
    float3x3 tbn = float3x3(state.T, state.B, state.N);
    pbrMat.N     = normalize(mul(normal_vector, tbn));

    // Mark that we need to update T and B due to normal perturbation
    needsTangentUpdate = true;
  }

  // Emissive term
  pbrMat.emissive = material.emissiveFactor;
  if(isTexturePresent(material.emissiveTexture))
  {
    pbrMat.emissive *= getTexture(textures, texInfos[material.emissiveTexture], state.tc, state.texGrad).rgb;
  }
  pbrMat.emissive = max(float3(0.0F), pbrMat.emissive);

#if MAT_EXT_VOLUME
  // KHR_materials_volume — set before the IOR block so the thin-walled check
  // (`pbrMat.thickness == 0`) drives the inside/outside IOR swap below. When MAT_EXT_VOLUME is
  // gated out, pbrMat.thickness stays at defaultPbrMaterial()'s 0.0 default and the swap is
  // correctly disabled.
  pbrMat.attenuationColor    = material.attenuationColor;
  pbrMat.attenuationDistance = material.attenuationDistance;
  // KHR_materials_volume spec: per-fragment thickness = thicknessFactor * thicknessTexture.g.
  pbrMat.thickness = material.thicknessFactor;
  if(isTexturePresent(material.thicknessTexture))
  {
    pbrMat.thickness *= getTexture(textures, texInfos[material.thicknessTexture], state.tc, state.texGrad).g;
  }
#endif

#if MAT_EXT_SPECULAR
  // KHR_materials_specular
  // https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_specular
  pbrMat.specularColor = material.specularColorFactor;
  if(isTexturePresent(material.specularColorTexture))
  {
    pbrMat.specularColor *= getTexture(textures, texInfos[material.specularColorTexture], state.tc, state.texGrad).rgb;
  }

  pbrMat.specular = material.specularFactor;
  if(isTexturePresent(material.specularTexture))
  {
    pbrMat.specular *= getTexture(textures, texInfos[material.specularTexture], state.tc, state.texGrad).a;
  }
#endif

  // Dielectric Specular — inside/outside IOR swap is enabled only for materials with finite
  // KHR_materials_volume thickness (thickness > 0). Thin-walled (thickness == 0) materials
  // never swap, matching the glTF specification.
  float ior1 = 1.0F;  // IOR of the current medium (e.g., air)
#if MAT_EXT_IOR
  float ior2 = material.ior;  // IOR of the material
#else
  float ior2 = 1.5F;  // glTF default IOR when KHR_materials_ior is gated out
#endif
  if(state.isInside && (pbrMat.thickness > 0.0f))
  {
    ior1 = ior2;
    ior2 = 1.0F;
  }
  pbrMat.ior1 = ior1;
  pbrMat.ior2 = ior2;


#if MAT_EXT_TRANSMISSION
  // KHR_materials_transmission
  pbrMat.transmission = material.transmissionFactor;
  if(isTexturePresent(material.transmissionTexture))
  {
    pbrMat.transmission *= getTexture(textures, texInfos[material.transmissionTexture], state.tc, state.texGrad).r;
  }
#endif

#if MAT_EXT_VOLUME_SCATTER
  // KHR_materials_volume_scatter
  // Compute scattering coefficient from multi-scatter albedo and attenuation
  if(any(material.multiscatterColorFactor > float3(0.0)))
  {
    float3 singleScatterAlbedo = multiToSingleScatterAlbedo(material.multiscatterColorFactor);
    float3 attenuationCoef = -log(max(pbrMat.attenuationColor, float3(0.001))) / max(pbrMat.attenuationDistance, 0.001);
    pbrMat.scatterCoefficient = attenuationCoef * singleScatterAlbedo;
  }
  pbrMat.scatterAnisotropy = material.scatterAnisotropy;
#endif

#if MAT_EXT_CLEARCOAT
  // KHR_materials_clearcoat
  pbrMat.clearcoat          = material.clearcoatFactor;
  pbrMat.clearcoatRoughness = material.clearcoatRoughness;
  pbrMat.Nc                 = pbrMat.N;
  if(isTexturePresent(material.clearcoatTexture))
  {
    pbrMat.clearcoat *= getTexture(textures, texInfos[material.clearcoatTexture], state.tc, state.texGrad).r;
  }
  if(isTexturePresent(material.clearcoatRoughnessTexture))
  {
    pbrMat.clearcoatRoughness *=
        getTexture(textures, texInfos[material.clearcoatRoughnessTexture], state.tc, state.texGrad).g;
  }
  if(isTexturePresent(material.clearcoatNormalTexture))
  {
    float3x3 tbn         = float3x3(pbrMat.T, pbrMat.B, pbrMat.Nc);
    float3 normal_vector = getTexture(textures, texInfos[material.clearcoatNormalTexture], state.tc, state.texGrad).xyz;
    normal_vector        = normal_vector * 2.0F - 1.0F;
    pbrMat.Nc            = normalize(mul(normal_vector, tbn));
  }
  pbrMat.clearcoatRoughness = max(pbrMat.clearcoatRoughness, 0.001F);
#endif

#if MAT_EXT_IRIDESCENCE
  // KHR_materials_iridescence
  float iridescence          = material.iridescenceFactor;
  float iridescenceThickness = material.iridescenceThicknessMaximum;
  pbrMat.iridescenceIor      = material.iridescenceIor;
  if(isTexturePresent(material.iridescenceTexture))
  {
    iridescence *= getTexture(textures, texInfos[material.iridescenceTexture], state.tc, state.texGrad).x;
  }
  if(isTexturePresent(material.iridescenceThicknessTexture))
  {
    const float t = getTexture(textures, texInfos[material.iridescenceThicknessTexture], state.tc, state.texGrad).y;
    iridescenceThickness = lerp(material.iridescenceThicknessMinimum, material.iridescenceThicknessMaximum, t);
  }
  pbrMat.iridescence = (iridescenceThickness > 0.0f) ? iridescence : 0.0f;  // No iridescence when the thickness is zero.
  pbrMat.iridescenceThickness = iridescenceThickness;
#endif

#if MAT_EXT_ANISOTROPY
  // KHR_materials_anisotropy
  float anisotropyStrength = material.anisotropyStrength;
  // If the anisotropyStrength == 0.0f (default), the roughness is isotropic.
  if(anisotropyStrength > 0.0F)
  {
    float2 anisotropyDirection = float2(1.0f, 0.0f);  // By default the anisotropy strength is along the tangent.
    if(isTexturePresent(material.anisotropyTexture))
    {
      const float4 anisotropyTex = getTexture(textures, texInfos[material.anisotropyTexture], state.tc, state.texGrad);

      // .xy encodes the direction in (tangent, bitangent) space. Remap from [0, 1] to [-1, 1].
      anisotropyDirection = normalize(float2(anisotropyTex.xy) * 2.0f - 1.0f);
      // .z encodes the strength in range [0, 1].
      anisotropyStrength *= anisotropyTex.z;
    }

    // Adjust the roughness to account for anisotropy.
    pbrMat.roughness.x = lerp(pbrMat.roughness.y, 1.0f, anisotropyStrength * anisotropyStrength);

    // Rotate the anisotropy direction in the tangent space.
    const float s       = material.anisotropyRotation.x;  // Sin and Cos of the rotation angle.
    const float c       = material.anisotropyRotation.y;
    anisotropyDirection = float2(c * anisotropyDirection.x + s * anisotropyDirection.y,
                                 c * anisotropyDirection.y - s * anisotropyDirection.x);

    // Update the tangent to be along the anisotropy direction in tangent space.
    const float3 T_aniso = pbrMat.T * anisotropyDirection.x + pbrMat.B * anisotropyDirection.y;

    pbrMat.T           = T_aniso;
    needsTangentUpdate = true;
  }
#endif

  // Perform tangent and bitangent updates if necessary
  if(needsTangentUpdate)
  {
    // Ensure T, B, and N are orthonormal
    pbrMat.B            = normalize(cross(pbrMat.N, pbrMat.T));
    float bitangentSign = sign(dot(state.B, pbrMat.B));
    pbrMat.B            = pbrMat.B * bitangentSign;
    pbrMat.T            = normalize(cross(pbrMat.B, pbrMat.N) * bitangentSign);
  }

#if MAT_EXT_SHEEN
  // KHR_materials_sheen
  pbrMat.sheenColor = material.sheenColorFactor;
  if(isTexturePresent(material.sheenColorTexture))
  {
    pbrMat.sheenColor *= float3(getTexture(textures, texInfos[material.sheenColorTexture], state.tc, state.texGrad).xyz);  // sRGB
  }

  pbrMat.sheenRoughness = material.sheenRoughnessFactor;
  if(isTexturePresent(material.sheenRoughnessTexture))
  {
    pbrMat.sheenRoughness *= getTexture(textures, texInfos[material.sheenRoughnessTexture], state.tc, state.texGrad).w;
  }
  pbrMat.sheenRoughness = max(MICROFACET_MIN_ROUGHNESS, pbrMat.sheenRoughness);
#endif

#if MAT_EXT_DISPERSION
  // KHR_materials_dispersion
  pbrMat.dispersion = material.dispersion;
#endif

#if MAT_EXT_DIFFUSE_TRANSMISSION
  // KHR_materials_diffuse_transmission
  pbrMat.diffuseTransmissionFactor = material.diffuseTransmissionFactor;
  if(isTexturePresent(material.diffuseTransmissionTexture))
  {
    pbrMat.diffuseTransmissionFactor *=
        getTexture(textures, texInfos[material.diffuseTransmissionTexture], state.tc, state.texGrad).a;
  }
  pbrMat.diffuseTransmissionColor = material.diffuseTransmissionColor;
  if(isTexturePresent(material.diffuseTransmissionColorTexture))
  {
    pbrMat.diffuseTransmissionColor *=
        getTexture(textures, texInfos[material.diffuseTransmissionColorTexture], state.tc, state.texGrad).rgb;
  }
#endif

  return pbrMat;
}


#endif  // GLTF_MATERIAL_EVAL_H
