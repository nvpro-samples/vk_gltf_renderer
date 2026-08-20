/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PATHTRACE_FUNCTIONS_H
#define PATHTRACE_FUNCTIONS_H 1

#include "gltf_eval_config.h"

//-----------------------------------------------------------------------
// Types and tuning constants
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
// Direct light structure
//-----------------------------------------------------------------------
struct DirectLight
{
  float3 direction;        // Direction to the light
  float3 radianceOverPdf;  // Radiance over pdf
  float  distance;         // Distance to the light
  float  pdf;              // Probability of sampling this light
};

//-----------------------------------------------------------------------
// Lighting technique selection probabilities for MIS
//-----------------------------------------------------------------------
struct LightingTechniqueProbabilities
{
  float lightWeight;     // Normalized probability of the light technique (vs environment)
  float envWeight;       // Normalized probability of the environment technique
  float punctualWeight;  // Within the light technique: probability of a punctual light (punctual + emissive = 1)
  float emissiveWeight;  // Within the light technique: probability of an emissive triangle
};

static const float MIN_TRANSMISSION = 0.01f;  // Minimum shadow transmission factor for transparent materials
static const float ANTIALIASING_STANDARD_DEVIATION = 0.4246609f;  // Standard deviation for antialiasing
static const int   RR_MIN_DEPTH                    = 3;           // Minimum depth for Russian Roulette
static const float VOLUME_MIN_SCATTER              = 0.001f;      // Minimum scatter for volume rendering
static const float VOLUME_RAND_FLOOR               = 1.0e-10f;    // Minimum random floor for volume rendering
static const float VOLUME_RAND_CEIL = 1.0f - 1.0e-7f;  // rand() can return exactly 1.0; keep u in the half-open [0,1) the bucket remap assumes
static const float RR_PCONT_FLOOR = 0.001f;  // Minimum probability for Russian Roulette
static const float RR_PCONT_CAP   = 0.95f;   // Maximum probability for Russian Roulette
// In-volume Russian Roulette. Scattering is bounded by an event budget rather than by the surface
// cap: a dense, high-albedo medium keeps the throughput near 1, so a 0.95 cap culls 5% of the
// *fully energetic* paths at every scatter event and the medium renders far too dark (56% of the
// energy lost at g = 0.95 in a medium that cannot absorb at all). Following the throughput instead
// leaves those paths alone, and the hard stop keeps the cost of a dense medium bounded -- this is
// an interactive path tracer, and an unbounded walk in a non-absorbing medium does not terminate
// at all (measured: device loss, not merely a slow frame).
static const int   VOLUME_FREE_BUDGET  = 64;      // Scatter events before Russian Roulette starts culling a volume path
static const float RR_PCONT_CAP_VOLUME = 0.999f;  // Maximum Russian Roulette probability inside a volume

//-----------------------------------------------------------------------
// Denoiser guide buffers (shared by DLSS and OptiX)
//
// Guide-buffer scratch (GuideOutput / GuideScratch + the helpers further down + the
// per-bounce captures in pathTraceOneBounce) only exists when USE_GUIDE_SHADER is
// defined (set by CMake when USE_DLSS or USE_OPTIX_DENOISER is ON, or by the runtime
// scene-aware variant rebuild). When it's off we drop those scratch fields from
// PathTracerState and SampleResult to shrink per-thread live state.
//
// Two-tier split inside the guide block:
//   USE_GUIDE_SHADER : albedo + normalRoughness (+ motion/depth). Consumed by DLSS and
//                      OptiX, so present whenever any denoiser guide is required.
//   USE_DLSS_SHADER  : specularAlbedo + specularHitDist (+ their per-bounce capture).
//                      DLSS-only; OptiX needs neither, so they are compiled out of the
//                      OptiX/no-denoiser variant to shrink live state further.
//
// Exception: hitPosition is promoted OUT of the guide-gated block because it is
// also consumed by the always-on NDC depth output (used for selection/picking on
// the first frame). Keeping it gated would break picking when DLSS/OptiX are off.
//-----------------------------------------------------------------------
#if USE_GUIDE_SHADER
struct GuideOutput
{
  float16_t4 albedo          = float16_t4(0);
  float16_t4 normalRoughness = float16_t4(0);
#if USE_DLSS_SHADER
  float16_t3 specularAlbedo  = float16_t3(0);
  float      specularHitDist = 0.0f;  // Distance to reflected surface (0=no reflection, 65504=sky reflection)
#endif                                // USE_DLSS_SHADER
};

//-----------------------------------------------------------------------
// DLSS guide temps accumulated during pathTrace (written to SampleResult at end).
//-----------------------------------------------------------------------
struct GuideScratch
{
  float16_t3 albedo          = float16_t3(0);
  float16_t4 normalRoughness = float16_t4(0);
#if USE_DLSS_SHADER
  float16_t3 specularAlbedo     = float16_t3(0);
  float16_t  specularHitDist    = float16_t(0);
  bool       captureSpecularHit = false;
#endif  // USE_DLSS_SHADER
};
#endif  // USE_GUIDE_SHADER

//-----------------------------------------------------------------------
// This is the result of sampling a pixel
//-----------------------------------------------------------------------
struct SampleResult
{
  float4 radiance    = float4(0, 0, 0, 0);
  float3 hitPosition = 1e34f;  // World-space first-hit position; always present (drives NDC depth + DLSS motion).
#if USE_GUIDE_SHADER
  float3      hitPositionPrev = 1e34f;  // Previous-frame world-space position of the first-hit point (instance motion).
  GuideOutput guideOutput     = {};
#endif
};

//-----------------------------------------------------------------------
// Ray-cone footprint (Akenine-Moller et al. 2019).
//-----------------------------------------------------------------------
struct RayCone
{
  float width       = 0.0f;
  float spreadAngle = 0.0f;
};

// VolumeMedium and its helpers only exist when KHR_materials_volume is in use.
// When GLTF_USE_VOLUME=0 (the scene has no volume materials in optimal mode) the medium
// struct, the bounce-loop in-volume tracking, and the corresponding fields on
// PathTracerState are dropped entirely (reduces per-thread live registers).
#if GLTF_USE_VOLUME
// Stored as absorption + scattering rather than extinction + scattering. The two forms are
// equivalent in exact arithmetic, but a dense, high-albedo medium has sigma_a as a tiny
// difference of two large numbers (sigma_t = 2712, sigma_s = 2711.6), and recovering it by
// subtracting stored values destroys it under limited precision. Keeping sigma_a as its own
// field gives every derived quantity -- sigma_t and, more importantly, the single-scatter
// albedo -- full precision, with no cancellation. That matters here: a medium authored with
// multiscatterColor near white has an albedo like 0.9999, and recovering it as sigma_s / sigma_t
// from two lower-precision values can quantize it straight to 1.0 (originally observed with
// float16 storage; kept as float32 headroom against the same failure mode at higher albedos).
struct VolumeMedium
{
  float32_t3 absorption;          // sigma_a
  float32_t3 scatterCoefficient;  // sigma_s
  float32_t  scatterAnisotropy;   // g
};

VolumeMedium makeVolumeMedium(PbrMaterial pbrMat)
{
  // sigma_t is the total extinction (KHR_materials_volume) and sigma_s the scattering part of it
  // (KHR_materials_scatter), so the difference is the absorption. Done in full precision here.
  float3       sigmaT = volumeExtinctionCoefficient(pbrMat);
  float3       sigmaS = min(pbrMat.scatterCoefficient, sigmaT);
  VolumeMedium medium;
  medium.absorption         = float32_t3(sigmaT - sigmaS);
  medium.scatterCoefficient = float32_t3(sigmaS);
  medium.scatterAnisotropy  = float32_t(pbrMat.scatterAnisotropy);
  return medium;
}

float3 volumeExtinction(VolumeMedium medium)
{
  return float3(medium.absorption) + float3(medium.scatterCoefficient);
}

bool hasVolumeMedium(VolumeMedium medium)
{
  float3 extinction = volumeExtinction(medium);
  return max(max(extinction.x, extinction.y), extinction.z) > 0.0F;
}
#endif  // GLTF_USE_VOLUME


//-----------------------------------------------------------------------
// Ray utilities
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
// Self-intersection offset (Wächter & Binder, Ray Tracing Gems 2019).
//-----------------------------------------------------------------------
float3 safeOffsetRay(float3 worldPosition, float3 offsetDirection)
{
  const float scaleValue = 256.0f;
  const int3  scaleInt   = int3(scaleValue * offsetDirection);

  const float3 offsetPosition = float3(  //
      asfloat(asint(worldPosition.x) + ((worldPosition.x < 0) ? -scaleInt.x : scaleInt.x)),
      asfloat(asint(worldPosition.y) + ((worldPosition.y < 0) ? -scaleInt.y : scaleInt.y)),
      asfloat(asint(worldPosition.z) + ((worldPosition.z < 0) ? -scaleInt.z : scaleInt.z)));

  const float origin     = 1.0f / 32.0f;
  const float floatScale = 1.0f / 65536.0f;
  return float3(  //
      abs(worldPosition.x) < origin ? worldPosition.x + floatScale * offsetDirection.x : offsetPosition.x,
      abs(worldPosition.y) < origin ? worldPosition.y + floatScale * offsetDirection.y : offsetPosition.y,
      abs(worldPosition.z) < origin ? worldPosition.z + floatScale * offsetDirection.z : offsetPosition.z);
}


//-----------------------------------------------------------------------
// Ray-cone footprint
//-----------------------------------------------------------------------

float rayConeWorldFootprint(RayCone cone, float hitT, float3 N, float3 V)
{
  float w = cone.width + cone.spreadAngle * hitT;
  return w / max(abs(dot(N, V)), 1e-3);
}


//-----------------------------------------------------------------------
// Alpha testing
//-----------------------------------------------------------------------

//----------------------------------------------------------
// Testing if the hit is opaque or alpha-transparent
// Return 1.0 if it is opaque
//----------------------------------------------------------
float getOpacity(GltfRenderNode renderNode, GltfRenderPrimitive renderPrim, int triangleID, float3 barycentrics)
{
  uint               matIndex  = max(0, renderNode.materialID);
  GltfShadeMaterial* materials = pushConst.gltfScene->materials;
  GltfShadeMaterial  mat       = materials[matIndex];
  GltfTextureInfo*   texInfos  = pushConst.gltfScene->textureInfos;

  if(mat.alphaMode == AlphaMode::eAlphaModeOpaque)
    return 1.0;

  uint3 triangleIndex = getTriangleIndices(renderPrim, triangleID);

  float baseColorAlpha = 1;
#if GLTF_USE_SPECULAR_GLOSSINESS
  if(mat.pbrModel == PbrModel::ePbrModelSpecularGlossiness)
  {
    baseColorAlpha = mat.pbrDiffuseFactor.a;
    if(isTexturePresent(mat.pbrDiffuseTexture))
    {
      GltfTextureInfo texInfo = texInfos[mat.pbrDiffuseTexture];
      float2          uv = getInterpolatedVertexTexCoord(renderPrim, texInfo.texCoord, triangleIndex, barycentrics);
      uv                 = applyTextureTransform(texInfo, uv);
      baseColorAlpha *= allImages[texInfo.index].SampleLevel(allSamplers[texInfo.samplerIndex], uv, 0.0f).a;
    }
  }
  else
#endif
  {
    baseColorAlpha = mat.pbrBaseColorFactor.a;
    if(isTexturePresent(mat.pbrBaseColorTexture))
    {
      GltfTextureInfo texInfo = texInfos[mat.pbrBaseColorTexture];
      float2          uv = getInterpolatedVertexTexCoord(renderPrim, texInfo.texCoord, triangleIndex, barycentrics);
      uv                 = applyTextureTransform(texInfo, uv);

      baseColorAlpha *= allImages[texInfo.index].SampleLevel(allSamplers[texInfo.samplerIndex], uv, 0.0f).a;
    }
  }

  baseColorAlpha *= getInterpolatedVertexColor(renderPrim, triangleIndex, barycentrics).a;

  if(mat.alphaMode == AlphaMode::eAlphaModeMask)
  {
    return baseColorAlpha >= mat.alphaCutoff ? 1.0 : 0.0;
  }

  return baseColorAlpha;
}


//-----------------------------------------------------------------------
// Shadow transmission
//-----------------------------------------------------------------------

//-----------------------------------------------------------------------
// Calculate the transmission of a ray through a surface
//-----------------------------------------------------------------------
float3 getShadowTransmission(GltfRenderNode      renderNode,
                             GltfRenderPrimitive renderPrim,
                             int                 triangleID,
                             float3              barycentrics,
                             float               hitT,
                             float4x3            worldToObject,
                             float3              rayDirection,
                             inout bool          isInside)
{
#if !GLTF_USE_TRANSMISSION
  return float3(0.0);
#else
  uint               matIndex  = max(0, renderNode.materialID);
  GltfShadeMaterial* materials = pushConst.gltfScene->materials;
  GltfShadeMaterial  mat       = materials[matIndex];
  GltfTextureInfo*   texInfos  = pushConst.gltfScene->textureInfos;

  float tFactor = mat.transmissionFactor;

  if(tFactor <= MIN_TRANSMISSION)
  {
    return float3(0.0);
  }

  uint3 indices = getTriangleIndices(renderPrim, triangleID);

  float3 normal;
  {
    float3 v0 = getVertexPosition(renderPrim, indices.x);
    float3 v1 = getVertexPosition(renderPrim, indices.y);
    float3 v2 = getVertexPosition(renderPrim, indices.z);
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    normal    = normalize(cross(e1, e2));
    normal    = normalize(float3(mul(worldToObject, normal).xyz));
  }

  float cosTheta = abs(dot(rayDirection, normal));
#if GLTF_USE_IOR
  float fresnel = schlickFresnel(mat.ior, cosTheta);
#else
  float fresnel = schlickFresnel(1.5F, cosTheta);
#endif

  float3 currentTransmission = tFactor * mat.pbrBaseColorFactor.rgb;
  currentTransmission *= (1.0F - fresnel);

#if GLTF_USE_VOLUME
  if(mat.thicknessFactor > 0.0)
  {
    if(isInside)
    {
      // Beer-Lambert transmittance through the medium interior. -log(c)/d is the TOTAL
      // extinction sigma_t = sigma_a + sigma_s: the out-scattering portion is already
      // included, so the scatter coefficient must not be added on top (that would
      // double-count it and over-darken the ray). See volumeExtinctionCoefficient()
      // in nvpro_core2's bsdf_functions.h.slang. attenuationDistance == 0 means "disabled".
      float3 extinction = (mat.attenuationDistance == 0.0F) ?
                              float3(0.0F) :
                              -log(max(mat.attenuationColor, float3(0.001))) / mat.attenuationDistance;
      currentTransmission *= exp(-hitT * extinction);
    }
    isInside = !isInside;
  }
#endif

  float transmissionAttenuation = 1.0;
  {
    float roughness = mat.pbrRoughnessFactor;
    float metallic  = mat.pbrMetallicFactor;
    if(isTexturePresent(mat.pbrMetallicRoughnessTexture))
    {
      GltfTextureInfo texInfo = texInfos[mat.pbrMetallicRoughnessTexture];
      float2          uv      = getInterpolatedVertexTexCoord(renderPrim, texInfo.texCoord, indices, barycentrics);
      uv                      = applyTextureTransform(texInfo, uv);

      float4 mr_sample = allImages[texInfo.index].SampleLevel(allSamplers[texInfo.samplerIndex], uv, 0.0f);
      roughness *= mr_sample.g;
      metallic *= mr_sample.b;
    }

    transmissionAttenuation *= (1.0 - metallic);

    float roughnessEffect = 1.0 - (roughness * roughness);
    transmissionAttenuation *= lerp(0.65, 1.0, roughnessEffect);
  }
  currentTransmission *= transmissionAttenuation;

  return currentTransmission;
#endif
}


//-----------------------------------------------------------------------
// Raytracer interface
//-----------------------------------------------------------------------

#include "raytracer_interface.h.slang"


//-----------------------------------------------------------------------
// Direct lighting and environment
//-----------------------------------------------------------------------

LightingTechniqueProbabilities getDirectLightingTechniqueProbabilities()
{
  LightingTechniqueProbabilities probs;

  bool  hasPunctual = pushConst.gltfScene.numLights > 0;
  bool  hasEmissive = pushConst.gltfScene.numEmissiveTriangles > 0;
  float lightWeight = (hasPunctual || hasEmissive) ? 0.5f : 0.0f;
  bool  hasEnv      = !hasFlag(pushConst.frameInfo->flags, SceneFrameInfoFlags::eSceneUseNoEnvironment)
                && (!hasFlag(pushConst.frameInfo->flags, SceneFrameInfoFlags::eSceneUseHdrEnvironment)
                    || pushConst.frameInfo->envIntensity.x > 0.0f);
  float envWeight = hasEnv ? 0.5f : 0.0f;

  float totalWeight = lightWeight + envWeight;
  if(totalWeight > 0.0f)
  {
    lightWeight /= totalWeight;
    envWeight /= totalWeight;
  }

  // Split the light technique between punctual lights and emissive triangles so that adding many
  // emissive triangles does not dilute punctual-light sampling (and vice versa).
  float punctualWeight = hasPunctual ? (hasEmissive ? 0.5f : 1.0f) : 0.0f;
  float emissiveWeight = hasEmissive ? (hasPunctual ? 0.5f : 1.0f) : 0.0f;

  probs.lightWeight    = lightWeight;
  probs.envWeight      = envWeight;
  probs.punctualWeight = punctualWeight;
  probs.emissiveWeight = emissiveWeight;
  return probs;
}

//-----------------------------------------------------------------------
// Emissive-mesh area lights (next-event estimation)
//
// Emissive triangles are sampled directly and MIS-combined with BSDF sampling, mirroring the
// environment-light MIS. Selection is POWER-WEIGHTED: triangle i is chosen with probability
// proportional to (world area * luminance), in O(1) via a Vose alias table. The key simplification
// is that the world area cancels in the solid-angle pdf:
//     pdf(i) = P(select i) * dist^2 / (area_i * cos) = luminance_i * dist^2 / (W_total * cos)
// so the MIS pdf for a BSDF ray landing on an emitter is still recovered from local hit geometry
// (its own radiance + W_total) with NO per-triangle lookup table. Triangles are stored as
// references, so vertices/transforms are read live here.
//-----------------------------------------------------------------------
// Defensive luminance: pull the emitter's luminance toward the scene's area-weighted mean so a dim
// emitter is never starved. Uses the shared Rec. 709 luminance() (nvshaders/functions.h.slang); the
// host weight (gltf_scene_vk.cpp uploadEmissiveTriangles) must use the same coefficients.
float emitterDefensiveLuminance(float3 radiance)
{
  float f = EMISSIVE_DEFENSIVE_FRACTION;
  return (1.0f - f) * luminance(radiance) + f * pushConst.gltfScene.emissiveMeanLum;
}

struct EmissiveSample
{
  float3 direction;  // from the shading point toward the sampled point on the emitter
  float  distance;   // shadow-ray TMax (stops just short of the emitter surface)
  float3 radiance;   // emitted radiance of the triangle
  float  pdf;        // solid-angle pdf of the sample; 0 when the sample is invalid
};

// World-space vertices of a referenced emissive triangle.
void getEmissiveTriangleWorld(EmissiveTriangle et, out float3 p0, out float3 p1, out float3 p2)
{
  GltfRenderNode      node = pushConst.gltfScene.renderNodes[et.renderNodeID];
  GltfRenderPrimitive prim = pushConst.gltfScene.renderPrimitives[et.renderPrimID];
  uint3               idx  = getTriangleIndices(prim, int(et.triangleID));
  p0                       = mul(float4(getVertexPosition(prim, idx.x), 1.0), node.objectToWorld).xyz;
  p1                       = mul(float4(getVertexPosition(prim, idx.y), 1.0), node.objectToWorld).xyz;
  p2                       = mul(float4(getVertexPosition(prim, idx.z), 1.0), node.objectToWorld).xyz;
}

// Select an emissive triangle proportional to its selection weight in O(1) via a Vose alias table.
// One uniform picks a bin; its fractional part chooses between that bin and its alias.
int selectEmissiveTriangle(float u01)
{
  int              n      = pushConst.gltfScene.numEmissiveTriangles;
  float            scaled = u01 * float(n);
  int              bin    = min(int(scaled), n - 1);
  float            uf     = scaled - float(bin);  // reuse the fractional part as the second uniform
  EmissiveTriangle et     = pushConst.gltfScene.emissiveTriangles[bin];
  return (uf < et.aliasProb) ? bin : int(et.aliasIndex);
}

// Power-weighted sample of emissive triangle `emissiveIndex` as seen from `shadingPos`.
EmissiveSample sampleEmissiveTriangle(int emissiveIndex, float3 shadingPos, inout uint seed)
{
  EmissiveSample es = {float3(0, 0, 1), 0.0, float3(0), 0.0};

  EmissiveTriangle et = pushConst.gltfScene.emissiveTriangles[emissiveIndex];
  float3           p0, p1, p2;
  getEmissiveTriangleWorld(et, p0, p1, p2);

  // Uniform barycentric sample over the triangle area.
  float2 u        = float2(rand(seed), rand(seed));
  float  su0      = sqrt(u.x);
  float  b1       = 1.0 - su0;
  float  b2       = u.y * su0;
  float3 lightPos = p0 * (1.0 - b1 - b2) + p1 * b1 + p2 * b2;

  float3 toLight = lightPos - shadingPos;
  float  dist2   = dot(toLight, toLight);
  if(dist2 <= 1e-12)
    return es;
  float  dist = sqrt(dist2);
  float3 dir  = toLight / dist;

  float3 ng      = cross(p1 - p0, p2 - p0);
  float  twoArea = length(ng);
  if(twoArea <= 1e-12)
    return es;  // degenerate triangle
  float3 lightN = ng / twoArea;

  float cosLight = dot(lightN, -dir);
  if(et.doubleSided != 0)
    cosLight = abs(cosLight);
  if(cosLight <= 1e-6)
    return es;  // one-sided emitter facing away, or grazing

  float wtot = pushConst.gltfScene.emissiveTotalWeight;
  if(wtot <= 0.0)
    return es;

  es.direction = dir;
  // Stop short of the emitter surface so the shadow ray does not self-intersect it. Use a small
  // absolute offset rather than a dist-relative one: scaling by dist skips a large segment in front
  // of a far emitter (missed occluders -> light leak). Clamp so very near emitters stay positive.
  es.distance = dist - min(dist * 0.5f, 1e-2f);
  es.radiance = et.radiance;
  // Power-weighted solid-angle pdf; the triangle area cancels (see header). Uses the defensive
  // luminance so it matches the host selection weight.
  es.pdf = emitterDefensiveLuminance(et.radiance) * dist2 / (wtot * cosLight);
  return es;
}

// NEE solid-angle pdf (within the light technique) of sampling the hit emissive triangle in the hit
// direction. Power-weighted selection makes the area cancel, so this needs only the emitter's own
// radiance, the geometric normal, and W_total -- no per-triangle lookup. 0 if not a valid emitter dir.
float emitterHitSolidAnglePdf(float3 radiance, float3 geoNormal, float3 rayDir, float hitDist)
{
  float wtot = pushConst.gltfScene.emissiveTotalWeight;
  if(pushConst.gltfScene.numEmissiveTriangles <= 0 || wtot <= 0.0)
    return 0.0;
  // No sidedness gate needed: one-sided emitters cull their back faces (SceneRtx omits the
  // FACING_CULL_DISABLE bit for them, and the path-trace rays use RAY_FLAG_CULL_BACK_FACING_TRIANGLES),
  // so only front faces are hit -- exactly the side the sampler samples. abs() gives the right cosine
  // magnitude for both one-sided (front) and double-sided (either face, sampler also uses abs) emitters.
  float cosLight = abs(dot(geoNormal, rayDir));  // geonrm is pre-flipped toward the ray
  if(cosLight <= 1e-6)
    return 0.0;
  return emitterDefensiveLuminance(radiance) * (hitDist * hitDist) / (wtot * cosLight);
}

// MIS weight for a BSDF ray that landed on an emissive triangle, weighted against the emissive-NEE
// technique. Mirrors computeEnvHitMisWeight; the competing NEE pdf is
// lightWeight * emissiveWeight * (emitter solid-angle pdf).
float computeEmitterHitMisWeight(float lastSamplePdf, float3 radiance, float3 geoNormal, float3 rayDir, float hitDist)
{
  if(lastSamplePdf == DIRAC)
    return 1.0f;
  LightingTechniqueProbabilities probs = getDirectLightingTechniqueProbabilities();
  float emitterNeePdf = probs.lightWeight * probs.emissiveWeight * emitterHitSolidAnglePdf(radiance, geoNormal, rayDir, hitDist);
  if(emitterNeePdf <= 0.0)
    return 1.0f;
  return lastSamplePdf / (lastSamplePdf + emitterNeePdf);
}

// `normal` is used to reject punctual lights that sit below the surface's hemisphere.
// `lightsFromBothSides` disables that rejection for shading points that receive light from the
// back hemisphere as well: volume scattering events, and surfaces with a diffuse transmission
// lobe (KHR_materials_diffuse_transmission, and the thin-walled mode of KHR_materials_scatter --
// a backlit leaf or sheet of paper is exactly the case those describe).
void sampleLights(in float3       pos,
                  float3          normal,
                  in float3       worldRayDirection,
                  inout uint      seed,
                  out DirectLight directLight,
                  in bool         isVolumeSample      = false,
                  in bool         lightsFromBothSides = false)
{
  float3 radiance             = float3(0.0f);
  directLight.pdf             = 0.0f;
  directLight.distance        = INFINITE;
  directLight.radianceOverPdf = float3(0.0f);
  float envPdf                = 0.0f;

  LightingTechniqueProbabilities probs          = getDirectLightingTechniqueProbabilities();
  float                          lightWeight    = probs.lightWeight;
  float                          envWeight      = probs.envWeight;
  float                          emissiveWeight = probs.emissiveWeight;

  if(lightWeight == 0.0f && envWeight == 0.0f)
  {
    return;
  }

  bool sampleLight = (rand(seed) < lightWeight);

  if(sampleLight)
  {
    // Within the light technique, pick punctual vs emissive (punctualWeight + emissiveWeight = 1).
    // Guard the categories explicitly: rand() can return exactly 1.0, so never fall into a branch
    // whose category is empty (e.g. the punctual branch when there are no punctual lights).
    bool pickEmissive;
    if(probs.emissiveWeight <= 0.0f)
      pickEmissive = false;  // no emitters -> punctual
    else if(probs.punctualWeight <= 0.0f)
      pickEmissive = true;  // no punctual lights -> emissive
    else
      pickEmissive = (rand(seed) < emissiveWeight);

    if(!pickEmissive)
    {
      int       numPunctual  = pushConst.gltfScene.numLights;
      float     selectionPdf = probs.punctualWeight / numPunctual;
      int       lightIndex   = min(int(rand(seed) * numPunctual), numPunctual - 1);
      GltfLight light        = pushConst.gltfScene.lights[lightIndex];

      // singleLightContribution() drops any punctual light below the hemisphere of the normal it
      // is given. When the shading point is lit from both sides, hand it a normal pointing at the
      // light so nothing is rejected; the BSDF (or phase function) decides what the light does.
      float3 cullNormal = normal;
      if(isVolumeSample || lightsFromBothSides)
      {
        cullNormal = (light.type == LightType::eLightTypeDirectional) ? -light.direction : normalize(light.position - pos);
      }
      LightContrib contrib = singleLightContribution(light, pos, cullNormal, float2(rand(seed), rand(seed)));

      directLight.direction = -contrib.incidentVector;
      directLight.distance  = contrib.distance;
      radiance              = contrib.intensity / (selectionPdf * lightWeight);

      directLight.pdf = (contrib.pdf == DIRAC) ? DIRAC : selectionPdf * contrib.pdf;
    }
    else
    {
      // Emissive triangle (area light): power-weighted selection, then divide by the solid-angle pdf.
      int            tri    = selectEmissiveTriangle(rand(seed));
      EmissiveSample es     = sampleEmissiveTriangle(tri, pos, seed);
      directLight.direction = es.direction;
      directLight.distance  = es.distance;
      if(es.pdf > 0.0)
      {
        // es.pdf already includes the power-weighted selection; fold in the emissive sub-weight.
        float pdfLight  = emissiveWeight * es.pdf;
        radiance        = es.radiance / (lightWeight * pdfLight);
        directLight.pdf = pdfLight;  // left as 0 otherwise -> caller skips NEE
      }
    }
  }

  if(envWeight > 0 && directLight.pdf != DIRAC)
  {
    if(!hasFlag(pushConst.frameInfo->flags, SceneFrameInfoFlags::eSceneUseHdrEnvironment))
    {
      if(!sampleLight)
      {
        float2            random_sample = float2(rand(seed), rand(seed));
        SkySamplingResult skySample     = samplePhysicalSky(*pushConst.skyParams, random_sample);
        directLight.direction           = skySample.direction;
        envPdf                          = skySample.pdf;
        radiance                        = skySample.radiance / (envPdf * envWeight);
      }
      else
      {
        envPdf = samplePhysicalSkyPDF(*pushConst.skyParams, directLight.direction);
      }
    }
    else
    {
      if(!sampleLight)
      {
        float3 rand_val = float3(rand(seed), rand(seed), rand(seed));
        float4 radiance_pdf = environmentSample(texturesHdr[HDR_IMAGE_INDEX], envSamplingData, rand_val, directLight.direction);
        envPdf                = radiance_pdf.w;
        radiance              = radiance_pdf.xyz * pushConst.frameInfo->envIntensity / (envPdf * envWeight);
        directLight.direction = rotate(directLight.direction, float3(0, 1, 0), pushConst.frameInfo->envRotation);
      }
      else
      {
        float3 dir          = rotate(directLight.direction, float3(0, 1, 0), -pushConst.frameInfo->envRotation);
        float2 uv           = getSphericalUv(dir);
        float4 radiance_pdf = texturesHdr[HDR_IMAGE_INDEX].SampleLevel(uv, 0);
        envPdf              = radiance_pdf.w;
      }
    }
  }

  float misWeight = 1.0f;
  if(directLight.pdf != DIRAC)
  {
    float pdfSum = lightWeight * directLight.pdf + envWeight * envPdf;
    if(pdfSum > 0.0)
      misWeight = (sampleLight ? lightWeight * directLight.pdf : envWeight * envPdf) / pdfSum;

    directLight.pdf = pdfSum;
  }
  radiance *= misWeight;

  directLight.radianceOverPdf = radiance;
}

void sampleEnvironment(float3 direction, SceneFrameInfo* frameInfo, out float3 envColor, out float envPdf)
{
  if(hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneUseNoEnvironment))
  {
    // No environment: contributes no radiance and cannot be importance-sampled.
    envColor = float3(0.0f);
    envPdf   = 0.0f;
    return;
  }

  if(!hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneUseHdrEnvironment))
  {
    envColor = evalPhysicalSky(*pushConst.skyParams, direction);
    envPdf   = samplePhysicalSkyPDF(*pushConst.skyParams, direction);
  }
  else
  {
    float3 dir = rotate(direction, float3(0, 1, 0), -frameInfo->envRotation);
    float2 uv  = getSphericalUv(dir);
    float4 env = texturesHdr[HDR_IMAGE_INDEX].SampleLevel(uv, 0);
    envColor   = env.rgb * frameInfo->envIntensity;
    envPdf     = env.w;
  }
}

float computeEnvHitMisWeight(float lastSamplePdf, float envPdf)
{
  if(lastSamplePdf == DIRAC)
    return 1.0f;

  LightingTechniqueProbabilities probs     = getDirectLightingTechniqueProbabilities();
  float                          envWeight = probs.envWeight;

  return lastSamplePdf / (lastSamplePdf + envWeight * envPdf);
}

//-----------------------------------------------------------------------
// Infinite plane and shadow catcher
//-----------------------------------------------------------------------

// It returns true if the shadow catcher path should continue, false otherwise
bool handleShadowCatcher(IRaytracer      raytracer,
                         HitState        hit,
                         PbrMaterial     pbrMat,
                         inout RayDesc   ray,
                         inout float3    radiance,
                         inout float3    throughput,
                         inout float     lastSamplePdf,
                         inout uint      seed,
                         SceneFrameInfo* frameInfo)
{
  DirectLight directLight;
  sampleLights(hit.pos, pbrMat.N, ray.Direction, seed, directLight);

  float3 shadowFactor = float3(1, 1, 1);
  if(dot(directLight.direction, hit.nrm) > 0.0f && directLight.pdf != 0.0f)
  {
    RayDesc shadowRay;
    shadowRay.Origin    = hit.pos;
    shadowRay.Direction = directLight.direction;
    shadowRay.TMin      = 0.0;
    shadowRay.TMax      = INFINITE;
    shadowFactor        = raytracer.TraceShadow(shadowRay, seed);
  }

  float3 envColor;
  float  envPdf;
  sampleEnvironment(ray.Direction, frameInfo, envColor, envPdf);

  if(all(shadowFactor == float3(1, 1, 1)))
  {
    float misWeight = computeEnvHitMisWeight(lastSamplePdf, envPdf);
    radiance += throughput * misWeight * envColor;
    return false;
  }

  radiance += envColor * shadowFactor;
  radiance -= envColor * (1.0f - shadowFactor) * frameInfo->shadowCatcherDarkenAmount;

  BsdfSampleData sampleData;
  sampleData.k1 = -ray.Direction;
  sampleData.xi = float3(rand(seed), rand(seed), rand(seed));
  bsdfSampleSimple(sampleData, pbrMat);

  if(sampleData.event_type == BSDF_EVENT_ABSORB)
  {
    return false;
  }

  float3 offsetDir = dot(sampleData.k2, hit.geonrm) > 0 ? hit.geonrm : -hit.geonrm;
  ray.Origin       = safeOffsetRay(hit.pos, offsetDir);
  ray.Direction    = sampleData.k2;
  throughput *= sampleData.bsdf_over_pdf;
  lastSamplePdf = sampleData.pdf;

  return true;
}

bool checkInfinitePlaneIntersection(RayDesc ray, inout HitPayload payload, inout HitState hit, SceneFrameInfo* frameInfo)
{
  if(!hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneUseInfinitePlane))
    return false;

  float3 normal      = float3(0, 1, 0);
  float  planeHeight = frameInfo->infinitePlaneDistance;

  if(ray.Origin.y <= planeHeight)
    return false;

  float Dn = dot(ray.Direction, normal);
  if(abs(Dn) <= 1e-6)
    return false;

  float On               = dot(ray.Origin, normal);
  float intersectionDist = (-On + planeHeight) / Dn;
  if(intersectionDist <= 0 || (intersectionDist >= payload.hitT))
    return false;

  payload.hitT  = intersectionDist;
  hit.pos       = ray.Origin + ray.Direction * payload.hitT;
  hit.shadowPos = hit.pos;
  hit.nrm       = normal;
  hit.geonrm    = normal;
  hit.tangent   = float3(1, 0, 0);
  hit.bitangent = float3(0, 0, 1);

  return true;
}


//-----------------------------------------------------------------------
// In-volume step: Beer's-law absorption (always when GLTF_USE_VOLUME) +
// Henyey-Greenstein scatter sampling (only when GLTF_USE_SCATTER).
//-----------------------------------------------------------------------
// IMPORTANT: this function exists whenever GLTF_USE_VOLUME is on, NOT only when
// GLTF_USE_SCATTER is on. KHR_materials_volume by itself (no scatter
// extension) still needs per-segment Beer's-law absorption applied along the
// camera ray, which is what makes a glass with attenuationColor="green" look
// green. Earlier the entire function was gated on GLTF_USE_SCATTER which
// silently dropped that absorption pass in optimal mode for any scene that
// uses volume but not scatter (e.g. CompareVolume.gltf).
//
// Returns true only when a real scatter event happened (GLTF_USE_SCATTER
// must be on for that to be possible). Returns false when only absorption was
// applied. The bounce-loop caller uses the return value to decide whether to
// run scatter-only follow-up work (NEE at the scatter point, scatterBounces++).
#if GLTF_USE_VOLUME
bool handleVolumeScatter(VolumeMedium medium, float hitDistance, inout RayDesc ray, inout float3 throughput, inout float lastSamplePdf, inout uint seed)
{
  float3 extinction = volumeExtinction(medium);

#if GLTF_USE_SCATTER
  float3 scatterCoeff = float3(medium.scatterCoefficient);
  float  maxScatter   = max(scatterCoeff.x, max(scatterCoeff.y, scatterCoeff.z));
  if(maxScatter > VOLUME_MIN_SCATTER)
  {
    // KHR_materials_volume media are homogeneous, so the free-flight distance can be sampled
    // analytically -- no delta tracking and no majorant. The medium is chromatic (sigma_t and
    // sigma_s are per-channel), so one channel is picked to sample the distance and the three
    // per-channel distance pdfs are combined with the balance heuristic (spectral MIS). Every
    // channel then keeps exactly its own transmittance and its own scattering rate.
    //
    // A single-majorant tracker would instead scatter every channel at the majorant's rate:
    // the low-extinction channels would random-walk far too often, and since their extra
    // "collisions" are lossless null collisions, their effective per-collision albedo -- which
    // is what sets the perceived color of a dense medium -- would drift towards 1. That breaks
    // the multi-scatter albedo mapping the extension is built on (a yellow multiscatterColor
    // over a slightly red attenuationColor came out pink).
    float3 channelProb = max(throughput, float3(0.0));
    float  probSum     = channelProb.x + channelProb.y + channelProb.z;
    channelProb        = (probSum > 0.0) ? channelProb / probSum : float3(1.0 / 3.0);

    // Pick the channel whose free-flight distribution is sampled, then draw the distance. The
    // selection consumes only the bucket it lands in, so rescaling within that bucket recovers a
    // fresh uniform for the distance -- one rand() per scatter event instead of two, and in a
    // dense medium there is one of these per event.
    float u             = min(rand(seed), VOLUME_RAND_CEIL);
    float sigmaTChannel = extinction.z;
    if(u < channelProb.x)
    {
      sigmaTChannel = extinction.x;
      u /= channelProb.x;
    }
    else if(u < channelProb.x + channelProb.y)
    {
      sigmaTChannel = extinction.y;
      u             = (u - channelProb.x) / channelProb.y;
    }
    else
    {
      u = (u - channelProb.x - channelProb.y) / channelProb.z;
    }
    float scatterDist = (sigmaTChannel > 0.0) ? -log(max(u, VOLUME_RAND_FLOOR)) / sigmaTChannel : hitDistance;

    if(scatterDist < hitDistance)
    {
      // Scattering event: pdf_i(t) = sigma_t_i * exp(-sigma_t_i * t), combined over the channels.
      float3 transmittance = exp(-extinction * scatterDist);
      float  pdf           = dot(channelProb, extinction * transmittance);
      if(pdf <= 0.0)
      {
        // Every channel's transmittance underflowed: the medium is opaque along this segment.
        throughput = float3(0.0);
        return false;
      }
      throughput *= scatterCoeff * transmittance / pdf;

      ray.Origin        = ray.Origin + ray.Direction * scatterDist;
      float3 wi         = ray.Direction;
      float  anisotropy = float(medium.scatterAnisotropy);
      ray.Direction     = sampleHenyeyGreenstein(float2(rand(seed), rand(seed)), anisotropy, wi);

      lastSamplePdf = henyeyGreensteinPdf(dot(wi, ray.Direction), anisotropy);

      return true;  // scatter event
    }

    // Reached the surface without scattering: the probability of that is exp(-sigma_t_i * d) per
    // channel, again combined with the balance heuristic.
    float3 transmittance = exp(-extinction * hitDistance);
    float  pdf           = dot(channelProb, transmittance);
    throughput *= (pdf > 0.0) ? transmittance / pdf : float3(0.0);
    return false;
  }
#endif  // GLTF_USE_SCATTER

  // Pure-absorption path (no scatter coefficient, or GLTF_USE_SCATTER disabled):
  // straight Beer's law applied over the segment length. This is the case for plain
  // KHR_materials_volume materials and is what produces the colored tint of a glass
  // filled with an absorbing medium.
  throughput *= exp(-hitDistance * extinction);
  return false;
}
#endif  // GLTF_USE_VOLUME

// Scatter-only: next-event estimation at the scatter point. Only meaningful when a
// scatter event can actually happen, so gated on GLTF_USE_SCATTER.
#if GLTF_USE_SCATTER
float3 volumeScatterNEE(IRaytracer raytracer, VolumeMedium medium, float3 scatterPos, float3 wiBeforeScatter, float3 throughput, inout uint seed)
{
  DirectLight directLight;
  sampleLights(scatterPos, /*normal=*/wiBeforeScatter, wiBeforeScatter, seed, directLight, /*isVolumeSample=*/true);

  if(directLight.pdf <= 0.0F)
    return float3(0.0F);

  float cosTheta = dot(wiBeforeScatter, directLight.direction);
  float phasePdf = henyeyGreensteinPdf(cosTheta, float(medium.scatterAnisotropy));

  float misWeight = (directLight.pdf == DIRAC) ? 1.0F : directLight.pdf / (directLight.pdf + phasePdf);

  RayDesc shadowRay;
  shadowRay.Origin    = scatterPos;
  shadowRay.Direction = directLight.direction;
  shadowRay.TMin      = 0.0F;
  shadowRay.TMax      = directLight.distance;
  float3 shadowFactor = raytracer.TraceShadow(shadowRay, seed, /*initialInside=*/true);

  return throughput * directLight.radianceOverPdf * misWeight * phasePdf * shadowFactor;
}
#endif  // GLTF_USE_SCATTER


//-----------------------------------------------------------------------
// DLSS guide buffers
//-----------------------------------------------------------------------

#if USE_DLSS_SHADER
float3 dlssEnforceMinAlbedo(float3 albedo, float3 specularAlbedo)
{
  float sumAlbedo = albedo.x + albedo.y + albedo.z;
  float sumSpec   = specularAlbedo.x + specularAlbedo.y + specularAlbedo.z;
  if(sumAlbedo + sumSpec < 0.05f)
  {
    return max(albedo, float3(0.2f));
  }
  return albedo;
}

bool dlssIsClearGlass(PbrMaterial pbrMat)
{
  return (pbrMat.transmission > 0.9f) && (pbrMat.roughness.x < 0.05f) && (pbrMat.opacity >= 0.99f);
}

float3 dlssGetClearGlassGuideAlbedo(IRaytracer      raytracer,
                                    PbrMaterial     pbrMat,
                                    HitState        hit,
                                    float3          viewDir,
                                    SceneFrameInfo* frameInfo,
                                    inout uint      seed,
                                    bool            isInside,
                                    float3          specularAlbedo)
{
  float3 bgAlbedo = float3(0.5f);
  float3 refrDir  = refract(viewDir, pbrMat.N, (pbrMat.ior1 / pbrMat.ior2));
  if(dot(refrDir, refrDir) > 0.0f)
  {
    RayDesc    guideRay;
    HitPayload guidePayload = {};
    guideRay.Origin         = safeOffsetRay(hit.pos, refrDir);
    guideRay.Direction      = refrDir;
    guideRay.TMin           = 0.0;
    guideRay.TMax           = INFINITE;
    raytracer.Trace(guideRay, guidePayload, seed);

    if(guidePayload.hitT != INFINITE)
    {
      GltfShadeMaterial*   materials       = pushConst.gltfScene->materials;
      GltfRenderNode*      renderNodes     = pushConst.gltfScene->renderNodes;
      GltfRenderPrimitive* renderPrimitive = pushConst.gltfScene->renderPrimitives;
      GltfTextureInfo*     texInfos        = pushConst.gltfScene->textureInfos;
      GltfRenderNode       guideNode       = renderNodes[guidePayload.rnodeID];
      GltfRenderPrimitive  guidePrim       = renderPrimitive[guidePayload.rprimID];
      int                  guideMatIndex   = max(0, guideNode.materialID);
      GltfShadeMaterial    guideMat        = materials[guideMatIndex];

      float3 guideBarys = float3(1.0 - guidePayload.bary.x - guidePayload.bary.y, guidePayload.bary.x, guidePayload.bary.y);

      HitState guideHit = getHitState(guidePrim, guideBarys, guideNode.worldToObject, guideNode.objectToWorld,
                                      guidePayload.primitiveID, guideRay.Direction);

      guideMat.pbrBaseColorFactor *= guideHit.color;
      MeshState guideMesh =
          MeshState(guideHit.nrm, guideHit.tangent, guideHit.bitangent, guideHit.geonrm, guideHit.uv, isInside, 0.0f);
      PbrMaterial guidePbrMat = evaluateMaterial(guideMat, guideMesh, allImages, allSamplers, texInfos);
      bgAlbedo                = guidePbrMat.baseColor.xyz;
    }
  }

  float  ndotv    = saturate(dot(pbrMat.N, -viewDir));
  float  iorRatio = (pbrMat.ior2 > 0.0f) ? (pbrMat.ior2 / pbrMat.ior1) : 1.5f;
  float  fresnel  = schlickFresnel(iorRatio, ndotv);
  float3 albedo   = lerp(bgAlbedo, pbrMat.baseColor.xyz, fresnel);
  return dlssEnforceMinAlbedo(albedo, specularAlbedo);
}
#endif  // USE_DLSS_SHADER


//-----------------------------------------------------------------------
// Wireframe overlay
//-----------------------------------------------------------------------

#if WIREFRAME

float computeMeshWireframe(float4x4 objectToWorld, float3 v0, float3 v1, float3 v2, float2 bary, float hitT, float pixelAngle, float3 N, float3 V)
{
  float3 wp0 = mul(float4(v0, 1.0), objectToWorld).xyz;
  float3 wp1 = mul(float4(v1, 1.0), objectToWorld).xyz;
  float3 wp2 = mul(float4(v2, 1.0), objectToWorld).xyz;

  float3 barys = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

  float3 e1 = wp1 - wp0;
  float3 e2 = wp2 - wp0;

  float worldFoot = hitT * pixelAngle / max(abs(dot(N, V)), 1e-3);

  float  twoArea  = length(cross(e1, e2));
  float3 edgesOpp = float3(length(wp2 - wp1), length(wp0 - wp2), length(wp1 - wp0));
  float3 deltas   = worldFoot * edgesOpp / max(twoArea, 1e-20);

  return processWireframe(WIREFRAME_PRESETS[WIREFRAME_THIN], barys, deltas);
}

#endif  // WIREFRAME


//-----------------------------------------------------------------------
// Camera and selection ray
//-----------------------------------------------------------------------

float2 sampleGaussian(float2 u)
{
  const float r     = sqrt(-2.0f * log(max(1e-38f, u.x)));
  const float theta = 2.0f * M_PI * u.y;
  return r * float2(cos(theta), sin(theta));
}

RayDesc getRay(float2 samplePos, float2 offset, float2 imageSize, float4x4 projMatrixI, float4x4 viewMatrixI, bool isOrthographic)
{
  const float2 clipCoords = (samplePos + offset) / imageSize * 2.0 - 1.0;
  float4       viewCoords = mul(float4(clipCoords, -1.0, 1.0), projMatrixI);
  viewCoords /= viewCoords.w;

  RayDesc ray;
  if(isOrthographic)
  {
    ray.Origin    = mul(viewCoords, viewMatrixI).xyz;
    ray.Direction = normalize(mul(float4(0, 0, -1, 0), viewMatrixI).xyz);
  }
  else
  {
    ray.Origin    = viewMatrixI[3].xyz;
    ray.Direction = normalize(mul(viewCoords, viewMatrixI).xyz - ray.Origin);
  }
  ray.TMin = 0.0;
  ray.TMax = INFINITE;
  return ray;
}

uint traceSelectionRay(IRaytracer raytracer, float2 samplePos, float2 imageSize)
{
  const bool isOrthographic = hasFlag(pushConst.frameInfo->flags, SceneFrameInfoFlags::eSceneIsOrthographic);
  RayDesc ray = getRay(samplePos, float2(0.5f), imageSize, pushConst.frameInfo->projInv, pushConst.frameInfo->viewInv, isOrthographic);
  HitPayload payload = {};
  raytracer.TraceLow(ray, payload);
  return (payload.hitT != INFINITE) ? uint(payload.rnodeID + 1) : 0;
}


//-----------------------------------------------------------------------
// Path trace iteration
//-----------------------------------------------------------------------

// Step results for the path tracing
enum PathStepResult
{
  eOutOfVolume,     // Continue tracing after volume processing
  eVolumeContinue,  // Continue tracing the ray through the volume
  eEarlyContinue,   // Early continue tracing the ray (infinite plane)
  eBreak,           // Break the path tracing
}


// Path tracer state
//
// isInside / volumeMedium / scatterBounces / guide are only carried in builds that
// actually use them. Each field dropped removes that many live registers from the
// per-thread state that survives across every Trace() call in the bounce loop,
// which is what determines warps-per-SM (occupancy) on the megakernel.
//
// isInside is conditional on either GLTF_USE_TRANSMISSION (toggled at refraction)
// or GLTF_USE_VOLUME (used to gate the in-volume scatter check).
//
// firstHitPos is always present (drives the selection/picking NDC depth) even
// when DLSS/OptiX is disabled; it is written on the first hit (or by the
// primary-miss backplate path).
struct PathTracerState
{
  float3  radiance      = float3(0.0F, 0.0F, 0.0F);
  float3  throughput    = float3(1.0F, 1.0F, 1.0F);
  float3  firstHitPos   = float3(1e34f);  // matches SampleResult::hitPosition sentinel
  float   lastSamplePdf = DIRAC;
  float2  maxRoughness  = float2(0.0);
  bool    solid         = true;
  RayCone cone          = {0.0F, pushConst.pixelAngle};
  int     surfaceDepth  = 0;
#if GLTF_USE_TRANSMISSION || GLTF_USE_VOLUME
  bool isInside = false;
#endif
#if GLTF_USE_VOLUME
  VolumeMedium volumeMedium = {};
#endif
#if GLTF_USE_SCATTER
  int scatterBounces = 0;
#endif
#if USE_GUIDE_SHADER
  GuideScratch guide           = {};
  float3       firstHitPosPrev = float3(1e34f);  // Previous-frame world position of the first hit (instance motion).
#endif
#if WIREFRAME
  float firstHitWireframeAlpha = 0.0;
#endif
};

struct BounceScratch
{
  bool   nextEventValid;
  float3 contribution;
  // Keep shadow rays full precision: half packing erases the self-intersection offset.
  float3 shadowRayPos;
  float3 shadowRayDir;
  float  shadowRayDist;
};

// Safe accessor for pt.isInside. The field only exists when GLTF_USE_TRANSMISSION or
// GLTF_USE_VOLUME is compiled in (without those, a ray can never enter a medium so the
// answer is always false). Used by call sites that aren't themselves gated, e.g.
// MeshState construction in the mesh-hit path or the DLSS clear-glass guide.
bool ptIsInside(PathTracerState pt)
{
#if GLTF_USE_TRANSMISSION || GLTF_USE_VOLUME
  return pt.isInside;
#else
  return false;
#endif
}

// Handles the in-volume segment before normal surface shading:
// - Always applies segment transmittance when GLTF_USE_VOLUME is enabled.
// - Handles scatter-only follow-up (cone advance, NEE, RR) when a real scatter happens.
PathStepResult processVolumeSegment(IRaytracer raytracer, float hitDistance, inout RayDesc ray, inout PathTracerState pt, inout uint seed)
{
#if GLTF_USE_VOLUME
  if(pt.isInside && hasVolumeMedium(pt.volumeMedium))
  {
#if GLTF_USE_SCATTER
    // handleVolumeScatter may rotate/translate the ray during a true scatter event.
    // Keep the pre-scatter ray state for HG cosine evaluation and cone travel update.
    const float3 wiBeforeScatter     = ray.Direction;
    const float3 originBeforeScatter = ray.Origin;
#endif

    if(handleVolumeScatter(pt.volumeMedium, hitDistance, ray, pt.throughput, pt.lastSamplePdf, seed))
    {
#if GLTF_USE_SCATTER
      pt.scatterBounces++;

      // Scatter events translate the cone through the medium segment (no slant correction).
      pt.cone.width += pt.cone.spreadAngle * length(ray.Origin - originBeforeScatter);
      pt.radiance += volumeScatterNEE(raytracer, pt.volumeMedium, ray.Origin, wiBeforeScatter, pt.throughput, seed);

      if(pt.scatterBounces >= VOLUME_FREE_BUDGET)
      {
        float rrPcont = min(max(pt.throughput.x, max(pt.throughput.y, pt.throughput.z)) + RR_PCONT_FLOOR, RR_PCONT_CAP_VOLUME);
        if(rand(seed) >= rrPcont)
          return PathStepResult.eBreak;
        pt.throughput /= rrPcont;
      }
      return PathStepResult.eVolumeContinue;  // skips surface-depth increment in the main loop
#endif                                        // GLTF_USE_SCATTER
    }
  }
#endif  // GLTF_USE_VOLUME

  return PathStepResult.eOutOfVolume;
}


// Primary ray missed geometry: DLSS far hit + optional backplate (not part of the lighting integral).
// Returns true when radiance is final and path tracing should stop.
bool tryPrimaryMissBackplate(RayDesc ray, SceneFrameInfo* frameInfo, inout PathTracerState pt)
{
  pt.solid = false;
  // Sky/background: represent it as a point at infinity along the (world-space) view direction.
  // processPixel projects this with w=0 (camera translation cancels). NDC depth for a miss is written
  // as far (1.0) on the !hasSolidHit branch, so this value is consumed only by motion vectors.
  pt.firstHitPos = ray.Direction;
#if USE_GUIDE_SHADER
  // Sky is fixed in world space; the previous-frame view direction is identical (w=0 yields camera-only motion).
  pt.firstHitPosPrev = ray.Direction;
#endif

  if(hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneUseSolidBackground))
  {
    pt.radiance = frameInfo->backgroundColor;
    return true;
  }

  if(hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneUseHdrEnvironment) && frameInfo->envBlur > 0)
  {
    float3 dir  = rotate(ray.Direction, float3(0, 1, 0), -frameInfo->envRotation);
    float2 uv   = getSphericalUv(dir);
    pt.radiance = smoothHDRBlur(texturesHdr[HDR_IMAGE_INDEX], uv, frameInfo->envBlur).xyz * frameInfo->envIntensity;
    return true;
  }

  return false;
}

SampleResult packSampleResult(PathTracerState pt)
{
  SampleResult sampleResult = {};
  sampleResult.radiance     = float4(pt.radiance, pt.solid ? 1 : 0);
  sampleResult.hitPosition  = pt.firstHitPos;
#if USE_GUIDE_SHADER
  sampleResult.hitPositionPrev             = pt.firstHitPosPrev;
  sampleResult.guideOutput.albedo          = float16_t4(pt.guide.albedo, pt.solid ? 1.0h : 0.0h);
  sampleResult.guideOutput.normalRoughness = pt.guide.normalRoughness;
#if USE_DLSS_SHADER
  sampleResult.guideOutput.specularAlbedo  = pt.guide.specularAlbedo;
  sampleResult.guideOutput.specularHitDist = pt.guide.specularHitDist;
#endif  // USE_DLSS_SHADER
#endif  // USE_GUIDE_SHADER
  return sampleResult;
}


#endif  // PATHTRACE_FUNCTIONS_H
