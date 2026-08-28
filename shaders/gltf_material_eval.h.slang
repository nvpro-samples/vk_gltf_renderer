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
#include "gltf_eval_config.h"

/*-------------------------------------------------------------------------------------------------
# `MICROFACET_MIN_ROUGHNESS` Define
> Minimum roughness for microfacet models.
-------------------------------------------------------------------------------------------------*/
#define MICROFACET_MIN_ROUGHNESS 0.0014142f


struct MeshState
{
  float3 N;      // Normal
  float3 T;      // Tangent
  float3 B;      // Bitangent
  float3 Ng;     // Geometric normal
  float2 tc[2];  // Texture coordinates
  bool   isInside = false;
  float  texGrad  = 0.0f;  // Ray footprint for SampleGrad; 0 -> mip 0 in path tracer, implicit HW mips in fragment
  float4 baseColorVertexMul = float4(1);  // COLOR_0 multiplier for baseColor or spec-gloss diffuse
};


/*-------------------------------------------------------------------------------------------------
# `NO_TEXTURES` Define
> Define this before including `gltf_material_eval.h` to use a color of `float4(1.0f)`
> for everything instead of reading textures.
-------------------------------------------------------------------------------------------------*/
#ifndef NO_TEXTURES
#define USE_TEXTURES
#endif

// KHR_texture_transform: map a raw texture coordinate into the material's UV space.
// Shared by getTexture() (color path) and the direct texture reads in the alpha/shadow
// paths (pathtrace_functions.h.slang) so both apply the same offset/rotation/scale.
float2 applyTextureTransform(in GltfTextureInfo textureInfo, float2 tt)
{
#if GLTF_USE_TEXTURE_TRANSFORM
  tt = mul(float3(tt, 1), textureInfo.uvTransform).xy;
#endif
  return tt;
}

float4 getTexture(Texture2D images[], SamplerState samplers[], in GltfTextureInfo textureInfo, in float2 texCoords[2], float texGrad)
{
#ifdef USE_TEXTURES
  // Images and samplers are now separate bindless arrays (see GltfTextureInfo.index / samplerIndex).
  Texture2D    tex     = images[textureInfo.index];
  SamplerState sampler = samplers[textureInfo.samplerIndex];
  float2       tt      = applyTextureTransform(textureInfo, texCoords[textureInfo.texCoord]);
  if(texGrad > 0.0f)
  {
#if GLTF_USE_TEXTURE_TRANSFORM
    float2 ddx_uv = textureInfo.uvTransform[0] * texGrad;
    float2 ddy_uv = textureInfo.uvTransform[1] * texGrad;
    return tex.SampleGrad(sampler, tt, ddx_uv, ddy_uv);
#else
    float2 ddx_uv = float2(texGrad, 0.0f);
    float2 ddy_uv = float2(0.0f, texGrad);
    return tex.SampleGrad(sampler, tt, ddx_uv, ddy_uv);
#endif
  }
  // texGrad == 0 means "use the sharpest mip". In a fragment shader (RASTER_PIPELINE) the
  // implicit-LOD Sample() has valid quad derivatives and yields proper hardware mipmapping.
  // In the path tracer (compute Ray Query / raygen Ray Tracing) there are no such derivatives:
  // raygen falls back to mip 0, but a compute dispatch emits quad derivatives that explode at
  // geometry silhouettes/discontinuities, snapping to the coarsest mip and flattening the
  // texture to its average (~base color). Force explicit mip 0 there so both RT paths match.
#ifdef RASTER_PIPELINE
  return tex.Sample(sampler, tt);
#else
  return tex.SampleLevel(sampler, tt, 0.0f);
#endif
#else
  return float4(1.0F);
#endif
}

/*-------------------------------------------------------------------------------------------------
 * Check if a texture is present.
 * -------------------------------------------------------------------------------------------------*/
bool isTexturePresent(uint16_t tinfo)
{
  return tinfo > 0;
}

/*-------------------------------------------------------------------------------------------------
 * Finish decoding a tangent-space normal that has already been mapped from [0,1] to [-1,1].
 *
 * Two-channel formats (BC5, R8G8, EAC_R11G11) are the standard way to ship normal maps: only X and
 * Y are stored and Z is implied. Vulkan samples their missing blue channel as 0, which decodes to
 * Z = -1 and flips the normal into the surface, so it has to be rebuilt from the unit-length
 * assumption. SceneVk marks those textures with eTexInfoTwoChannelSource; three-channel maps carry
 * an authored Z and are returned untouched.
 * -------------------------------------------------------------------------------------------------*/
float3 decodeTangentSpaceNormal(float3 normal_vector, in GltfTextureInfo textureInfo)
{
  if((textureInfo.flags & TextureInfoFlagBits::eTexInfoTwoChannelSource) != 0)
  {
    normal_vector.z = sqrt(saturate(1.0F - dot(normal_vector.xy, normal_vector.xy)));
  }
  return normal_vector;
}

// Volumetric scattering is path-tracer only (see the scatter block in evaluateMaterial), so this
// mapping is not compiled into the raster pipeline.
#if GLTF_USE_SCATTER && !defined(RASTER_PIPELINE)
/*-------------------------------------------------------------------------------------------------
 * Convert multi-scatter albedo to single-scatter albedo (Kulla-Conty mapping).
 * Formula from the KHR_materials_scatter specification. The anisotropy g of the
 * Henyey-Greenstein phase function enters the denominator, so forward/backward
 * scattering media get the correct single-scatter albedo.
 * -------------------------------------------------------------------------------------------------*/
float3 multiToSingleScatterAlbedo(float3 rho_ms, float g)
{
  float3 s = 4.09712 + 4.20863 * rho_ms - sqrt(9.59217 + 41.6808 * rho_ms + 17.7126 * rho_ms * rho_ms);
  // g in (-1,1) and s in [0,1] keep the denominator strictly positive. Clamp the
  // result to [0,1] so scatterCoefficient never exceeds the total extinction.
  return clamp((1.0 - s * s) / (1.0 - g * s * s), float3(0.0), float3(1.0));
}
#endif

#if GLTF_USE_SPECULAR_GLOSSINESS
/*-------------------------------------------------------------------------------------------------
 * Convert PBR specular glossiness to metallic-roughness
 * -------------------------------------------------------------------------------------------------*/
float3 convertSGToMR(float3 diffuseColor, float3 specularColor, float glossiness, out float metallic, out float2 roughness)
{
  const float dielectricSpecular = 0.04f;

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
# Function `evaluateMaterial`
> From the incoming `material` and `mesh` info, return a `PbrMaterial` struct for the BSDF system.
-------------------------------------------------------------------------------------------------*/
PbrMaterial evaluateMaterial(in GltfShadeMaterial material, MeshState state, Texture2D images[], SamplerState samplers[], GltfTextureInfo* texInfos)
{
  // Start from a fully-initialized material so that gating out any extension with MAT_EXT_*
  // cannot leave downstream BSDF reads undefined.
  PbrMaterial pbrMat = defaultPbrMaterial();

#if GLTF_USE_SPECULAR_GLOSSINESS
  // KHR_materials_pbrSpecularGlossiness (deprecated)
  if(material.pbrModel == PbrModel::ePbrModelSpecularGlossiness)
  {
    float4 diffuse    = material.pbrDiffuseFactor * state.baseColorVertexMul;
    float  glossiness = material.pbrGlossinessFactor;
    float3 specular   = material.pbrSpecularFactor;

    if(isTexturePresent(material.pbrDiffuseTexture))
    {
      diffuse *= getTexture(images, samplers, texInfos[material.pbrDiffuseTexture], state.tc, state.texGrad);
    }

    if(isTexturePresent(material.pbrSpecularGlossinessTexture))
    {
      float4 specularGlossinessSample =
          getTexture(images, samplers, texInfos[material.pbrSpecularGlossinessTexture], state.tc, state.texGrad);
      specular *= specularGlossinessSample.rgb;
      glossiness *= specularGlossinessSample.a;
    }

    pbrMat.baseColor = convertSGToMR(diffuse.rgb, specular, glossiness, pbrMat.metallic, pbrMat.roughness);
    pbrMat.opacity   = diffuse.a;
  }
  else
#endif
  // This is the Metallic-Roughness
  {
    float4 baseColor = material.pbrBaseColorFactor * state.baseColorVertexMul;
    if(isTexturePresent(material.pbrBaseColorTexture))
    {
      baseColor *= getTexture(images, samplers, texInfos[material.pbrBaseColorTexture], state.tc, state.texGrad);
    }
    pbrMat.baseColor = baseColor.rgb;
    pbrMat.opacity   = baseColor.a;

    float roughness = material.pbrRoughnessFactor;
    float metallic  = material.pbrMetallicFactor;
    if(isTexturePresent(material.pbrMetallicRoughnessTexture))
    {
      float4 metallicRoughnessSample =
          getTexture(images, samplers, texInfos[material.pbrMetallicRoughnessTexture], state.tc, state.texGrad);
      roughness *= metallicRoughnessSample.g;
      metallic *= metallicRoughnessSample.b;
    }
    roughness        = max(roughness, MICROFACET_MIN_ROUGHNESS);
    pbrMat.roughness = float2(roughness * roughness);
    pbrMat.metallic  = clamp(metallic, 0.0F, 1.0F);
  }

  pbrMat.occlusion = material.occlusionStrength;
  if(isTexturePresent(material.occlusionTexture))
  {
    float occlusion  = getTexture(images, samplers, texInfos[material.occlusionTexture], state.tc, state.texGrad).r;
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
    GltfTextureInfo normalTexInfo  = texInfos[material.normalTexture];
    float3          normal_vector  = getTexture(images, samplers, normalTexInfo, state.tc, state.texGrad).xyz;
    normal_vector                  = normal_vector * 2.0F - 1.0F;
    normal_vector                  = decodeTangentSpaceNormal(normal_vector, normalTexInfo);
    normal_vector *= float3(material.normalTextureScale, material.normalTextureScale, 1.0F);
    float3x3 tbn       = float3x3(state.T, state.B, state.N);
    pbrMat.N           = normalize(mul(normal_vector, tbn));
    needsTangentUpdate = true;  // Mark that we need to update T and B due to normal perturbation
  }

  // Emissive term
  pbrMat.emissive = material.emissiveFactor;
  if(isTexturePresent(material.emissiveTexture))
  {
    pbrMat.emissive *= getTexture(images, samplers, texInfos[material.emissiveTexture], state.tc, state.texGrad).rgb;
  }
  pbrMat.emissive = max(float3(0.0F), pbrMat.emissive);

#if GLTF_USE_VOLUME
  // KHR_materials_volume  set before the IOR block so the thin-walled check
  // (`pbrMat.thickness == 0`) drives the inside/outside IOR swap below. When GLTF_USE_VOLUME is
  // gated out, pbrMat.thickness stays at defaultPbrMaterial()'s 0.0 default and the swap is
  // correctly disabled.
  pbrMat.attenuationColor    = material.attenuationColor;
  pbrMat.attenuationDistance = material.attenuationDistance;
  pbrMat.thickness           = material.thicknessFactor;
  if(isTexturePresent(material.thicknessTexture))
  {
    // KHR_materials_volume spec: per-fragment thickness = thicknessFactor * thicknessTexture.g.
    pbrMat.thickness *= getTexture(images, samplers, texInfos[material.thicknessTexture], state.tc, state.texGrad).g;
  }
#endif

#if GLTF_USE_SPECULAR
  // KHR_materials_specular
  // https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_specular
  pbrMat.specularColor = material.specularColorFactor;
  if(isTexturePresent(material.specularColorTexture))
  {
    pbrMat.specularColor *=
        getTexture(images, samplers, texInfos[material.specularColorTexture], state.tc, state.texGrad).rgb;
  }

  pbrMat.specular = material.specularFactor;
  if(isTexturePresent(material.specularTexture))
  {
    pbrMat.specular *= getTexture(images, samplers, texInfos[material.specularTexture], state.tc, state.texGrad).a;
  }
#endif

  // Dielectric Specular -- inside/outside IOR swap is enabled only for materials with finite
  // KHR_materials_volume thickness (thickness > 0). Thin-walled (thickness == 0) materials
  // never swap, matching the glTF specification.
  float ior1 = 1.0F;  // IOR of the current medium (e.g., air)
#if GLTF_USE_IOR
  float ior2 = material.ior;
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

#if GLTF_USE_TRANSMISSION
  // KHR_materials_transmission
  pbrMat.transmission = material.transmissionFactor;
  if(isTexturePresent(material.transmissionTexture))
  {
    pbrMat.transmission *= getTexture(images, samplers, texInfos[material.transmissionTexture], state.tc, state.texGrad).r;
  }
#endif

  // KHR_materials_scatter is applied after KHR_materials_diffuse_transmission below, because the
  // thin-walled mode repurposes the diffuse-transmission lobes and must not be clobbered by it.

#if GLTF_USE_CLEARCOAT
  // KHR_materials_clearcoat
  pbrMat.clearcoat          = material.clearcoatFactor;
  pbrMat.clearcoatRoughness = material.clearcoatRoughness;
  pbrMat.Nc                 = pbrMat.N;
  if(isTexturePresent(material.clearcoatTexture))
  {
    pbrMat.clearcoat *= getTexture(images, samplers, texInfos[material.clearcoatTexture], state.tc, state.texGrad).r;
  }
  if(isTexturePresent(material.clearcoatRoughnessTexture))
  {
    pbrMat.clearcoatRoughness *=
        getTexture(images, samplers, texInfos[material.clearcoatRoughnessTexture], state.tc, state.texGrad).g;
  }
  if(isTexturePresent(material.clearcoatNormalTexture))
  {
    float3x3        tbn           = float3x3(pbrMat.T, pbrMat.B, pbrMat.Nc);
    GltfTextureInfo normalTexInfo = texInfos[material.clearcoatNormalTexture];
    float3          normal_vector = getTexture(images, samplers, normalTexInfo, state.tc, state.texGrad).xyz;
    normal_vector                 = normal_vector * 2.0F - 1.0F;
    normal_vector                 = decodeTangentSpaceNormal(normal_vector, normalTexInfo);
    pbrMat.Nc                     = normalize(mul(normal_vector, tbn));
  }
  pbrMat.clearcoatRoughness = max(pbrMat.clearcoatRoughness, 0.001F);
#endif

#if GLTF_USE_IRIDESCENCE
  // KHR_materials_iridescence
  float iridescence          = material.iridescenceFactor;
  float iridescenceThickness = material.iridescenceThicknessMaximum;
  pbrMat.iridescenceIor      = material.iridescenceIor;
  if(isTexturePresent(material.iridescenceTexture))
  {
    iridescence *= getTexture(images, samplers, texInfos[material.iridescenceTexture], state.tc, state.texGrad).x;
  }
  if(isTexturePresent(material.iridescenceThicknessTexture))
  {
    const float t =
        getTexture(images, samplers, texInfos[material.iridescenceThicknessTexture], state.tc, state.texGrad).y;
    iridescenceThickness = lerp(material.iridescenceThicknessMinimum, material.iridescenceThicknessMaximum, t);
  }
  pbrMat.iridescence          = (iridescenceThickness > 0.0f) ? iridescence : 0.0f;
  pbrMat.iridescenceThickness = iridescenceThickness;
#endif

#if GLTF_USE_ANISOTROPY
  // KHR_materials_anisotropy
  float anisotropyStrength = material.anisotropyStrength;
  // If the anisotropyStrength == 0.0f (default), the roughness is isotropic.
  if(anisotropyStrength > 0.0F)
  {
    float2 anisotropyDirection = float2(1.0f, 0.0f);  // By default the anisotropy strength is along the tangent.
    if(isTexturePresent(material.anisotropyTexture))
    {
      const float4 anisotropyTex = getTexture(images, samplers, texInfos[material.anisotropyTexture], state.tc, state.texGrad);
      // .xy encodes the direction in (tangent, bitangent) space. Remap from [0, 1] to [-1, 1].
      anisotropyDirection = normalize(float2(anisotropyTex.xy) * 2.0f - 1.0f);
      // .z encodes the strength in range [0, 1].
      anisotropyStrength *= anisotropyTex.z;
    }

    // Adjust the roughness to account for anisotropy.
    pbrMat.roughness.x = lerp(pbrMat.roughness.y, 1.0f, anisotropyStrength * anisotropyStrength);

    // Rotate the anisotropy direction in the tangent space.
    const float s       = material.anisotropyRotation.x;
    const float c       = material.anisotropyRotation.y;
    anisotropyDirection = float2(c * anisotropyDirection.x + s * anisotropyDirection.y,
                                 c * anisotropyDirection.y - s * anisotropyDirection.x);

    // Update the tangent to be along the anisotropy direction in tangent space.
    const float3 T_aniso = pbrMat.T * anisotropyDirection.x + pbrMat.B * anisotropyDirection.y;
    pbrMat.T             = T_aniso;
    needsTangentUpdate   = true;
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

#if GLTF_USE_SHEEN
  // KHR_materials_sheen
  pbrMat.sheenColor = material.sheenColorFactor;
  if(isTexturePresent(material.sheenColorTexture))
  {
    // sRGB texture
    pbrMat.sheenColor *=
        float3(getTexture(images, samplers, texInfos[material.sheenColorTexture], state.tc, state.texGrad).xyz);
  }

  pbrMat.sheenRoughness = material.sheenRoughnessFactor;
  if(isTexturePresent(material.sheenRoughnessTexture))
  {
    pbrMat.sheenRoughness *=
        getTexture(images, samplers, texInfos[material.sheenRoughnessTexture], state.tc, state.texGrad).w;
  }
  pbrMat.sheenRoughness = max(MICROFACET_MIN_ROUGHNESS, pbrMat.sheenRoughness);
#endif

#if GLTF_USE_DISPERSION
  // KHR_materials_dispersion
  pbrMat.dispersion = material.dispersion;
#endif

#if GLTF_USE_DIFFUSE_TRANSMISSION
  // KHR_materials_diffuse_transmission
  pbrMat.diffuseTransmissionFactor = material.diffuseTransmissionFactor;
  if(isTexturePresent(material.diffuseTransmissionTexture))
  {
    pbrMat.diffuseTransmissionFactor *=
        getTexture(images, samplers, texInfos[material.diffuseTransmissionTexture], state.tc, state.texGrad).a;
  }
  pbrMat.diffuseTransmissionColor = material.diffuseTransmissionColor;
  if(isTexturePresent(material.diffuseTransmissionColorTexture))
  {
    pbrMat.diffuseTransmissionColor *=
        getTexture(images, samplers, texInfos[material.diffuseTransmissionColorTexture], state.tc, state.texGrad).rgb;
  }
#endif

#if GLTF_USE_SCATTER
  // KHR_materials_scatter. Applied after diffuse transmission (the thin-walled mode overrides
  // those lobes) and after volume/thickness are resolved (they select the mode).
  // scatterStrength = scatterStrengthFactor * scatterStrengthTexture.a
  float scatterStrength = material.scatterStrengthFactor;
  if(isTexturePresent(material.scatterStrengthTexture))
  {
    scatterStrength *= getTexture(images, samplers, texInfos[material.scatterStrengthTexture], state.tc, state.texGrad).a;
  }

  // The extension is a no-op at zero strength (spec default), so only apply it when there is
  // scattering to do.
  if(scatterStrength > 0.0)
  {
    // multiscatterColor = multiscatterColorFactor * multiscatterColorTexture.rgb
    float3 multiscatterColor = material.multiscatterColorFactor;
    if(isTexturePresent(material.multiscatterColorTexture))
    {
      multiscatterColor *=
          getTexture(images, samplers, texInfos[material.multiscatterColorTexture], state.tc, state.texGrad).rgb;
    }

    float g = material.scatterAnisotropy;

    // Volumetric mode needs to track a path *inside* the medium, which the rasterizer cannot do.
    // The spec allows a renderer without volumetric transport to approximate it with the
    // thin-walled behaviour ("for materials with very short scattering distances relative to
    // their size (dense subsurface), the visual difference is small"), so the raster preview
    // always takes the thin-walled branch and the volume branch is path-tracer only.
#ifndef RASTER_PIPELINE
    if(pbrMat.thickness > 0.0)
    {
      // Volumetric mode (KHR_materials_volume, thickness > 0): split the attenuation coefficient
      // sigma_t into scattering (sigma_s) and absorption via the strength-scaled single-scatter
      // albedo. sigma_s drives the in-volume Henyey-Greenstein scattering in pathtrace_functions.
      // The sigma_t expression must stay identical to volumeExtinctionCoefficient() in
      // nvpro_core2's bsdf_functions.h.slang (not included here), which is what the path tracer
      // uses for the same medium -- if the two drift, sigma_s no longer belongs to sigma_t.
      float3 singleScatterAlbedo = multiToSingleScatterAlbedo(scatterStrength * multiscatterColor, g);
      // attenuationDistance == 0 means "disabled" -- see volumeExtinctionCoefficient().
      float3 attenuationCoef    = (pbrMat.attenuationDistance == 0.0F) ?
                                      float3(0.0F) :
                                      -log(max(pbrMat.attenuationColor, float3(0.001))) / pbrMat.attenuationDistance;
      pbrMat.scatterCoefficient = attenuationCoef * singleScatterAlbedo;
      pbrMat.scatterAnisotropy  = g;  // consumed by the phase function; only meaningful with sigma_s
    }
    else
#endif  // !RASTER_PIPELINE
    {
      // Thin-walled mode (no volume / thickness = 0). The spec replaces the transmissive part of
      // the dielectric base with
      //   mix(specular_btdf * baseColor, scatter_bsdf(multiscatterColor, g), scatterStrength)
      // where scatter_bsdf is a Lambertian BSDF sending (1+g)/2 of its energy through the surface
      // and (1-g)/2 back. There is no dedicated scatter lobe in the BSDF, so it is expressed with
      // the existing diffuse-transmission / diffuse-reflection lobes, whose weights below the
      // specular layer are `transmission` for the BTDF and `1 - transmission` for the diffuse
      // bucket (split by diffuseTransmissionFactor). Moving `scatterWeight = transmission *
      // scatterStrength` from the BTDF into the diffuse bucket reproduces the spec lobe weights
      // exactly, for any transmission and any g. Note both factors: without transmission the
      // extension must have no effect at all.
      float scatterWeight = pbrMat.transmission * scatterStrength;  // energy moved to scatter_bsdf
      if(scatterWeight > 0.0)
      {
        // The diffuse bucket holds four shares: the material's own diffuse reflection and diffuse
        // transmission (untouched by scatter), plus the two halves of scatter_bsdf.
        float forwardFraction = 0.5 * (1.0 + g);
        float baseDiffuse     = 1.0 - pbrMat.transmission;         // base, untouched by scatter
        float baseDtFactor    = pbrMat.diffuseTransmissionFactor;  // 0 unless KHR_materials_diffuse_transmission
        float baseTransWeight = baseDiffuse * baseDtFactor;
        float baseReflWeight  = baseDiffuse * (1.0 - baseDtFactor);
        float forwardWeight   = scatterWeight * forwardFraction;
        float backwardWeight  = scatterWeight - forwardWeight;
        float diffuseWeight   = baseDiffuse + scatterWeight;  // the four shares; >= scatterWeight > 0

        pbrMat.transmission -= scatterWeight;  // = transmission * (1 - scatterStrength)

        // Split the bucket. Note this is the forward share *of the bucket*, not the anisotropy's
        // forward fraction: the bucket also carries the untouched base, so it has to be normalised
        // by the bucket's own weight. (Using the bare fraction is only right at transmission = 1.)
        pbrMat.diffuseTransmissionFactor = (baseTransWeight + forwardWeight) / diffuseWeight;

        // Each of the two diffuse lobes now carries contributions the spec colors differently, so
        // each takes the energy-weighted average of its own two shares. Both shares of a lobe have
        // the same Lambertian shape, so averaging their colors is exact, not an approximation.
        float3 diffuseBaseColor         = pbrMat.baseColor;
        pbrMat.diffuseTransmissionColor = lerp(pbrMat.diffuseTransmissionColor, multiscatterColor,
                                               forwardWeight / max(baseTransWeight + forwardWeight, 1e-6));
        pbrMat.baseColor = lerp(diffuseBaseColor, multiscatterColor, backwardWeight / max(baseReflWeight + backwardWeight, 1e-6));

        // ...but baseColor also tints the residual specular BTDF, which the spec leaves at the
        // material's baseColor. transmissionTint cancels the blend back out for that lobe alone
        // (the BSDF tints it by `baseColor * transmissionTint`), and the product is bounded by
        // baseColor so it can never add energy. It degrades in one corner: at transmission = 1 with
        // a multiscatterColor channel of exactly 0, the blended color is 0 in that channel and a
        // multiplier cannot recover baseColor from it, so the residual BTDF loses that channel.
        pbrMat.transmissionTint =
            (pbrMat.transmission > 0.0) ? diffuseBaseColor / max(pbrMat.baseColor, float3(1e-4)) : float3(1.0);
      }
    }
  }
#endif

#if GLTF_USE_RETROREFLECTION
  // KHR_materials_retroreflection (MRM, Portsmouth et al. 2026)
  pbrMat.retroreflection = material.retroreflectionFactor;
  if(isTexturePresent(material.retroreflectionTexture))
  {
    pbrMat.retroreflection *=
        getTexture(images, samplers, texInfos[material.retroreflectionTexture], state.tc, state.texGrad).r;
  }
#endif

  return pbrMat;
}

PbrMaterial evaluateMaterial(uint materialIndex, MeshState state, Texture2D images[], SamplerState samplers[], GltfTextureInfo* texInfos, GltfShadeMaterial* materials)
{
  return evaluateMaterial(materials[materialIndex], state, images, samplers, texInfos);
}


#endif  // GLTF_MATERIAL_EVAL_H
