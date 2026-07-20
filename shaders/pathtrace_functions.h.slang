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
  float lightWeight;  // Normalized probability of sampling punctual lights
  float envWeight;    // Normalized probability of sampling environment
};

static const float MIN_TRANSMISSION = 0.01f;  // Minimum shadow transmission factor for transparent materials
static const float ANTIALIASING_STANDARD_DEVIATION = 0.4246609f;  // Standard deviation for antialiasing
static const int   RR_MIN_DEPTH                    = 3;           // Minimum depth for Russian Roulette
static const float VOLUME_MIN_SCATTER              = 0.001f;      // Minimum scatter for volume rendering
static const float VOLUME_RAND_FLOOR               = 1.0e-10f;    // Minimum random floor for volume rendering
static const int   VOLUME_FREE_BUDGET              = 64;          // Maximum free budget for volume rendering
static const float RR_PCONT_FLOOR                  = 0.001f;      // Minimum probability for Russian Roulette
static const float RR_PCONT_CAP                    = 0.95f;       // Maximum probability for Russian Roulette

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
struct VolumeMedium
{
  float16_t3 extinction;
  float16_t3 scatterCoefficient;
  float16_t  scatterAnisotropy;
};

VolumeMedium makeVolumeMedium(PbrMaterial pbrMat)
{
  VolumeMedium medium;
  medium.extinction         = float16_t3(volumeExtinctionCoefficient(pbrMat));
  medium.scatterCoefficient = float16_t3(pbrMat.scatterCoefficient);
  medium.scatterAnisotropy  = float16_t(pbrMat.scatterAnisotropy);
  return medium;
}

bool hasVolumeMedium(VolumeMedium medium)
{
  float3 extinction   = float3(medium.extinction);
  float3 scatterCoeff = float3(medium.scatterCoefficient);
  return max(max(extinction.x, extinction.y), extinction.z) > 0.0F
         || max(max(scatterCoeff.x, scatterCoeff.y), scatterCoeff.z) > 0.0F;
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
      baseColorAlpha *= allTextures[texInfo.index].SampleLevel(uv, 0.0f).a;
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

      baseColorAlpha *= allTextures[texInfo.index].SampleLevel(uv, 0.0f).a;
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
      float3 absCoeff = -log(max(mat.attenuationColor, float3(0.001))) / max(mat.attenuationDistance, 0.001);

#if GLTF_USE_VOLUME_SCATTER
      float3 scatterCoeff = absCoeff * multiToSingleScatterAlbedo(mat.multiscatterColorFactor);
#else
      float3 scatterCoeff = float3(0.0);
#endif

      float3 extinction = absCoeff + scatterCoeff;
      currentTransmission *= exp(-hitT * extinction);

      float maxScatter = max(scatterCoeff.x, max(scatterCoeff.y, scatterCoeff.z));
      if(maxScatter > 0.001f)
      {
        float maxExtinction  = max(extinction.x, max(extinction.y, extinction.z));
        float opticalDepth   = hitT * maxExtinction;
        float scatterPenalty = exp(-opticalDepth);
        currentTransmission *= scatterPenalty;
      }
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

      float4 mr_sample = allTextures[texInfo.index].SampleLevel(uv, 0.0f);
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

  float lightWeight = (pushConst.gltfScene.numLights > 0) ? 0.5f : 0.0f;
  float envWeight   = (!hasFlag(pushConst.frameInfo->flags, SceneFrameInfoFlags::eSceneUseHdrEnvironment)
                     || pushConst.frameInfo->envIntensity.x > 0.0f) ?
                          0.5f :
                          0.0f;

  float totalWeight = lightWeight + envWeight;
  if(totalWeight > 0.0f)
  {
    lightWeight /= totalWeight;
    envWeight /= totalWeight;
  }

  probs.lightWeight = lightWeight;
  probs.envWeight   = envWeight;
  return probs;
}

void sampleLights(in float3 pos, float3 normal, in float3 worldRayDirection, inout uint seed, out DirectLight directLight, in bool isVolumeSample = false)
{
  float3 radiance             = float3(0.0f);
  directLight.pdf             = 0.0f;
  directLight.distance        = INFINITE;
  directLight.radianceOverPdf = float3(0.0f);
  float envPdf                = 0.0f;

  LightingTechniqueProbabilities probs       = getDirectLightingTechniqueProbabilities();
  float                          lightWeight = probs.lightWeight;
  float                          envWeight   = probs.envWeight;

  if(lightWeight == 0.0f && envWeight == 0.0f)
  {
    return;
  }

  bool sampleLight = (rand(seed) < lightWeight);

  if(sampleLight)
  {
    float selectionPdf = 1.0 / pushConst.gltfScene.numLights;

    int       lightIndex = min(int(rand(seed) * pushConst.gltfScene.numLights), pushConst.gltfScene.numLights - 1);
    GltfLight light      = pushConst.gltfScene.lights[lightIndex];

    float3 cullNormal = (isVolumeSample && light.type == LightType::eLightTypeDirectional) ? -light.direction : normal;
    LightContrib contrib = singleLightContribution(light, pos, cullNormal, float2(rand(seed), rand(seed)));

    directLight.direction = -contrib.incidentVector;
    directLight.distance  = contrib.distance;
    radiance              = contrib.intensity / (selectionPdf * lightWeight);

    directLight.pdf = (contrib.pdf == DIRAC) ? DIRAC : selectionPdf * contrib.pdf;
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
// Henyey-Greenstein scatter sampling (only when GLTF_USE_VOLUME_SCATTER).
//-----------------------------------------------------------------------
// IMPORTANT: this function exists whenever GLTF_USE_VOLUME is on, NOT only when
// GLTF_USE_VOLUME_SCATTER is on. KHR_materials_volume by itself (no scatter
// extension) still needs per-segment Beer's-law absorption applied along the
// camera ray, which is what makes a glass with attenuationColor="green" look
// green. Earlier the entire function was gated on GLTF_USE_VOLUME_SCATTER which
// silently dropped that absorption pass in optimal mode for any scene that
// uses volume but not scatter (e.g. CompareVolume.gltf).
//
// Returns true only when a real scatter event happened (GLTF_USE_VOLUME_SCATTER
// must be on for that to be possible). Returns false when only absorption was
// applied. The bounce-loop caller uses the return value to decide whether to
// run scatter-only follow-up work (NEE at the scatter point, scatterBounces++).
#if GLTF_USE_VOLUME
bool handleVolumeScatter(VolumeMedium medium, float hitDistance, inout RayDesc ray, inout float3 throughput, inout float lastSamplePdf, inout uint seed)
{
  float3 extinction = float3(medium.extinction);

#if GLTF_USE_VOLUME_SCATTER
  float3 scatterCoeff = float3(medium.scatterCoefficient);
  float  maxScatter   = max(scatterCoeff.x, max(scatterCoeff.y, scatterCoeff.z));
  if(maxScatter > VOLUME_MIN_SCATTER)
  {
    float maxExtinction = max(extinction.x, max(extinction.y, extinction.z));

    float scatterDist = -log(max(rand(seed), VOLUME_RAND_FLOOR)) / maxExtinction;

    if(scatterDist < hitDistance)
    {
      throughput *= float3(1.0) - (extinction - scatterCoeff) / maxExtinction;

      ray.Origin        = ray.Origin + ray.Direction * scatterDist;
      float3 wi         = ray.Direction;
      float  anisotropy = float(medium.scatterAnisotropy);
      ray.Direction     = sampleHenyeyGreenstein(float2(rand(seed), rand(seed)), anisotropy, wi);

      lastSamplePdf = henyeyGreensteinPdf(dot(wi, ray.Direction), anisotropy);

      return true;  // scatter event
    }

    // No scatter this segment: apply the absorption-only attenuation that the analytical
    // null-collision tracker leaves behind (extinction - scatterCoeff = pure absorption).
    throughput *= exp(hitDistance * (float3(maxExtinction) - extinction));
    return false;
  }
#endif  // GLTF_USE_VOLUME_SCATTER

  // Pure-absorption path (no scatter coefficient, or GLTF_USE_VOLUME_SCATTER disabled):
  // straight Beer's law applied over the segment length. This is the case for plain
  // KHR_materials_volume materials and is what produces the colored tint of a glass
  // filled with an absorbing medium.
  throughput *= exp(-hitDistance * extinction);
  return false;
}
#endif  // GLTF_USE_VOLUME

// Scatter-only: next-event estimation at the scatter point. Only meaningful when a
// scatter event can actually happen, so gated on GLTF_USE_VOLUME_SCATTER.
#if GLTF_USE_VOLUME_SCATTER
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
#endif  // GLTF_USE_VOLUME_SCATTER


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
      PbrMaterial guidePbrMat = evaluateMaterial(guideMat, guideMesh, allTextures, texInfos);
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
#if GLTF_USE_VOLUME_SCATTER
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
#if GLTF_USE_VOLUME_SCATTER
    // handleVolumeScatter may rotate/translate the ray during a true scatter event.
    // Keep the pre-scatter ray state for HG cosine evaluation and cone travel update.
    const float3 wiBeforeScatter     = ray.Direction;
    const float3 originBeforeScatter = ray.Origin;
#endif

    if(handleVolumeScatter(pt.volumeMedium, hitDistance, ray, pt.throughput, pt.lastSamplePdf, seed))
    {
#if GLTF_USE_VOLUME_SCATTER
      pt.scatterBounces++;

      // Scatter events translate the cone through the medium segment (no slant correction).
      pt.cone.width += pt.cone.spreadAngle * length(ray.Origin - originBeforeScatter);
      pt.radiance += volumeScatterNEE(raytracer, pt.volumeMedium, ray.Origin, wiBeforeScatter, pt.throughput, seed);

      if(pt.scatterBounces >= VOLUME_FREE_BUDGET)
      {
        float rrPcont = min(max(pt.throughput.x, max(pt.throughput.y, pt.throughput.z)) + RR_PCONT_FLOOR, RR_PCONT_CAP);
        if(rand(seed) >= rrPcont)
          return PathStepResult.eBreak;
        pt.throughput /= rrPcont;
      }
      return PathStepResult.eVolumeContinue;  // skips surface-depth increment in the main loop
#endif                                        // GLTF_USE_VOLUME_SCATTER
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
