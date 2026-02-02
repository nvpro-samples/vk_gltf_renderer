/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef AVAILABLE_SER
#define AVAILABLE_SER 1
#endif


// NVSHADERS are under: https://github.com/nvpro-samples/nvpro_core2/tree/main/nvshaders
#include "nvshaders/functions.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/sky_functions.h.slang"

#include "nvshaders/bsdf_types.h.slang"
#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/gltf_scene_io.h.slang"
#include "nvshaders/gltf_vertex_access.h.slang"
#include "nvshaders/hdr_env_sampling.h.slang"
#include "nvshaders/light_contrib.h.slang"
#include "nvshaders/normal_compress.h.slang"
#include "nvshaders/pbr_ggx_microfacet.h.slang"
#include "nvshaders/pbr_material_eval.h.slang"
#include "nvshaders/ray_utils.h.slang"
#include "nvshaders/sample_blur.h.slang"

#include "shaderio.h"
#include "get_hit.h.slang"
#include "dlss_util.h"

#include "common.h.slang"
#include "raytracer_interface.h.slang"


// clang-format off

// Bindings
[[vk::push_constant]]                           ConstantBuffer<PathtracePushConstant> pushConst;
[[vk::binding(BindingPoints::eTextures, 0)]]    Sampler2D allTextures[];
[[vk::binding(BindingPoints::eTexturesHdr, 0)]] Sampler2D texturesHdr[];
[[vk::binding(BindingPoints::eTlas, 1)]]        RaytracingAccelerationStructure topLevelAS;
[[vk::binding(BindingPoints::eOutImages, 1)]]   RWTexture2D<float4> outImages[];

// HDR Environment
[[vk::binding(EnvBindings::eImpSamples, 2)]]    StructuredBuffer<EnvAccel> envSamplingData;

// Specialization constants
[[vk::constant_id(0)]]  int USE_SER = 1;
[[vk::constant_id(1)]]  int USE_DLSS_TRANSP = 0; // Primary surface selection mode

// clang-format on

// This is a debug flag to enable/disable debug output
// For example, to use print statements in the shader, set this to true.
// Add printf and enable the Validation layer debug output.
static bool doDebug = false;

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

//-----------------------------------------------------------------------
// Minimum transmission factor to continue tracing
//-----------------------------------------------------------------------
#define MIN_TRANSMISSION 0.01

//-----------------------------------------------------------------------
// Standard deviation of the Gaussian filter used for antialiasing,
// in units of pixels.
// This value of 1 / sqrt(8 ln(2)) makes it so that a Gaussian centered
// on a pixel is at exactly 1/2 its maximum at the midpoints between
// orthogonally adjacent pixels, and 1/4 its maximum at the "corners"
// of pixels. It also empirically looks nice: larger values are
// too blurry, and smaller values make thin lines look jagged.
//-----------------------------------------------------------------------
#define ANTIALIASING_STANDARD_DEVIATION 0.4246609F

//-----------------------------------------------------------------------
// Minimum path depth before Russian Roulette path termination begins.
// Ensures the first N bounces (capturing ~95% of visible energy in typical
// scenes) are always traced before probabilistic termination.
//-----------------------------------------------------------------------
#define RR_MIN_DEPTH 3

//-----------------------------------------------------------------------
// The safeOffsetRay function shifts a point on a surface so that a ray
// bouncing off the surface is no longer treated as intersecting the surface
// from which it originated when tMin = 0.0.
//
// This uses the technique by Carsten Wächter and Nikolaus Binder
// from "A Fast and Robust Method for Avoiding Self-Intersection" in Ray
// Tracing Gems (2019).
// The offset direction is typically the surface normal or its negation,
// depending on whether the ray exits or enters the surface.
float3 safeOffsetRay(float3 worldPosition, float3 offsetDirection)
{
  // Convert the offset direction to an integer offset.
  const float scaleValue = 256.0f;
  const int3  scaleInt   = int3(scaleValue * offsetDirection);

  // Offset each component of worldPosition using its binary representation.
  // Handle the sign bits correctly.
  const float3 offsetPosition = float3(  //
      asfloat(asint(worldPosition.x) + ((worldPosition.x < 0) ? -scaleInt.x : scaleInt.x)),
      asfloat(asint(worldPosition.y) + ((worldPosition.y < 0) ? -scaleInt.y : scaleInt.y)),
      asfloat(asint(worldPosition.z) + ((worldPosition.z < 0) ? -scaleInt.z : scaleInt.z)));

  // Use a floating-point offset instead for points near (0,0,0), the origin.
  const float origin     = 1.0f / 32.0f;
  const float floatScale = 1.0f / 65536.0f;
  return float3(  //
      abs(worldPosition.x) < origin ? worldPosition.x + floatScale * offsetDirection.x : offsetPosition.x,
      abs(worldPosition.y) < origin ? worldPosition.y + floatScale * offsetDirection.y : offsetPosition.y,
      abs(worldPosition.z) < origin ? worldPosition.z + floatScale * offsetDirection.z : offsetPosition.z);
}

//-----------------------------------------------------------------------
// #DLSS - Information for the DLSS
//-----------------------------------------------------------------------
struct DlssOutput
{
  float16_t4 albedo          = float16_t4(0);
  float16_t3 specularAlbedo  = float16_t3(0);
  float16_t4 normalRoughness = float16_t4(0);
  float3     hitPosition     = 1e34f;
};

//-----------------------------------------------------------------------
// This is the result of sampling a pixel
// It returns the radiance and the DLSS output
//-----------------------------------------------------------------------
struct SampleResult
{
  float4     radiance   = float4(0, 0, 0, 0);
  DlssOutput dlssOutput = {};  // DLSS output
};


//-----------------------------------------------------------------------
// Samples a 2D Gaussian distribution with a standard distribution of 1,
// using the Box-Muller algorithm.
// The input must be two random numbers in the range [0,1].
//-----------------------------------------------------------------------
float2 sampleGaussian(float2 u)
{
  const float r     = sqrt(-2.0f * log(max(1e-38f, u.x)));  // Radius
  const float theta = 2.0f * M_PI * u.y;                    // Angle
  return r * float2(cos(theta), sin(theta));
}

//-----------------------------------------------------------------------
// Get a ray from a sample position and offset
// Return the ray description
//-----------------------------------------------------------------------
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
  ray.TMin      = 0.0;
  ray.TMax      = INFINITE;
  return ray;
}


//-----------------------------------------------------------------------
// Computes normalized selection probabilities for direct lighting techniques (MIS).
// Returns probabilities for sampling punctual lights vs environment lighting.
// Ensures consistent MIS weights between Next Event Estimation (NEE) and BSDF sampling.
//
// The function checks:
// - Whether punctual lights exist (numLights > 0)
// - Whether environment lighting is enabled (sky type or envIntensity > 0)
//
// Returns normalized weights that sum to 1.0, or both 0.0 if no lighting is available.
// TODO: adjust based on scene characteristics (light intensity vs environment intensity)
//-----------------------------------------------------------------------
LightingTechniqueProbabilities getDirectLightingTechniqueProbabilities()
{
  LightingTechniqueProbabilities probs;

  // Initial selection probabilities (unnormalized)
  float lightWeight = (pushConst.gltfScene.numLights > 0) ? 0.5f : 0.0f;
  float envWeight =
      ((pushConst.frameInfo->environmentType == EnvSystem::eSky) || pushConst.frameInfo->envIntensity.x > 0.0f) ? 0.5f : 0.0f;

  // Normalize weights so they sum to 1.0
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

//-----------------------------------------------------------------------
// Sample lights using one-sample model MIS between punctual lights and environment.
// See section 9.2.4 of https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf
//
// Key insight: Punctual lights (Point, Spot, Directional) are delta distributions.
// Environment sampling has zero probability of hitting the exact delta direction,
// so MIS weight = 1.0 for both strategies when delta lights are involved.
//-----------------------------------------------------------------------
void sampleLights(in float3 pos, float3 normal, in float3 worldRayDirection, inout uint seed, out DirectLight directLight)
{
  float3 radiance             = float3(0.0f);
  directLight.pdf             = 0.0f;
  directLight.distance        = INFINITE;
  directLight.radianceOverPdf = float3(0.0f);
  float envPdf                = 0.0f;

  // Get normalized selection probabilities for MIS
  LightingTechniqueProbabilities probs       = getDirectLightingTechniqueProbabilities();
  float                          lightWeight = probs.lightWeight;
  float                          envWeight   = probs.envWeight;

  // Early exit if no lighting is available
  if(lightWeight == 0.0f && envWeight == 0.0f)
  {
    return;  // No lights to sample
  }

  // Decide whether to sample the light or the environment.
  bool sampleLight = (rand(seed) < lightWeight);

  // === LIGHT SAMPLING ===
  if(sampleLight)
  {
    // Selection probability for Monte Carlo normalization (1/N lights)
    float selectionPdf = 1.0 / pushConst.gltfScene.numLights;

    int          lightIndex = min(int(rand(seed) * pushConst.gltfScene.numLights), pushConst.gltfScene.numLights - 1);
    GltfLight    light      = pushConst.gltfScene.lights[lightIndex];
    LightContrib contrib    = singleLightContribution(light, pos, normal, float2(rand(seed), rand(seed)));

    directLight.direction = -contrib.incidentVector;
    directLight.distance  = contrib.distance;
    radiance              = contrib.intensity / (selectionPdf * lightWeight);

    // Use PDF from singleLightContribution (DIRAC for punctual, 1/solidAngle for area)
    // Multiply by selectionPdf to account for light selection probability
    directLight.pdf = (contrib.pdf == DIRAC) ? DIRAC : selectionPdf * contrib.pdf;
  }

  // === ENVIRONMENT SAMPLING ===
  // Skip envPdf query if we sampled a punctual light (DIRAC) - it won't be used in MIS calculation
  if(envWeight > 0 && directLight.pdf != DIRAC)
  {
    if(pushConst.frameInfo->environmentType == EnvSystem::eSky)
    {
      if(!sampleLight)  // Use this technique?
      {
        float2            random_sample = float2(rand(seed), rand(seed));
        SkySamplingResult skySample     = samplePhysicalSky(*pushConst.skyParams, random_sample);
        directLight.direction           = skySample.direction;
        envPdf                          = skySample.pdf;
        radiance                        = skySample.radiance / (envPdf * envWeight);
      }
      else
      {
        // Query env PDF at light direction for area light MIS
        envPdf = samplePhysicalSkyPDF(*pushConst.skyParams, directLight.direction);
      }
    }
    else
    {
      if(!sampleLight)  // Use this technique?
      {
        float3 rand_val = float3(rand(seed), rand(seed), rand(seed));
        float4 radiance_pdf = environmentSample(texturesHdr[HDR_IMAGE_INDEX], envSamplingData, rand_val, directLight.direction);
        envPdf                = radiance_pdf.w;
        radiance              = radiance_pdf.xyz * pushConst.frameInfo->envIntensity / (envPdf * envWeight);
        directLight.direction = rotate(directLight.direction, float3(0, 1, 0), pushConst.frameInfo->envRotation);
      }
      else
      {
        // Query env PDF at light direction for area light MIS
        float3 dir          = rotate(directLight.direction, float3(0, 1, 0), -pushConst.frameInfo->envRotation);
        float2 uv           = getSphericalUv(dir);
        float4 radiance_pdf = texturesHdr[HDR_IMAGE_INDEX].SampleLevel(uv, 0);
        envPdf              = radiance_pdf.w;
      }
    }
  }

  // One-sample MIS: randomly pick light or env, weight by balance heuristic (Veach thesis 9.2.4).
  // When sampling a light, we query envPdf so both PDFs contribute to the weight.
  // When sampling the environment, we don't query light PDFs (would require looping over all lights).
  // This means MIS weights won't sum to 1 if lights overlap bright env regions (Veach 9.2.1.1),
  // but that's an acceptable tradeoff for this rare edge case.
  float misWeight = 1.0f;
  if(directLight.pdf != DIRAC)
  {
    float pdfSum = lightWeight * directLight.pdf + envWeight * envPdf;
    if(pdfSum > 0.0)
      misWeight = (sampleLight ? lightWeight * directLight.pdf : envWeight * envPdf) / pdfSum;

    // Output combined PDF for downstream MIS with BSDF in pathTrace()
    directLight.pdf = pdfSum;
  }
  radiance *= misWeight;

  directLight.radianceOverPdf = radiance;
}

//-----------------------------------------------------------------------
// Evaluates the environment (sky or HDR) in a given direction
// Returns the radiance and PDF for Multiple Importance Sampling (MIS)
// This function is used both for environment hits and shadow catcher compositing
//-----------------------------------------------------------------------
void sampleEnvironment(float3 direction, SceneFrameInfo* frameInfo, out float3 envColor, out float envPdf)
{
  if(frameInfo->environmentType == EnvSystem::eSky)
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

//-----------------------------------------------------------------------
// Computes Multiple Importance Sampling (MIS) weight for BSDF samples that hit the environment.
// Balances BSDF sampling (lastSamplePdf) against Next Event Estimation (NEE)
// which could have sampled the same direction via sampleLights().
//
// For delta BSDF events (specular), weight = 1.0 (NEE can't generate exact direction)
// For continuous BSDF, accounts for NEE's selection probability (envWeight)
//-----------------------------------------------------------------------
float computeEnvHitMisWeight(float lastSamplePdf, float envPdf)
{
  if(lastSamplePdf == DIRAC)
    return 1.0f;

  // Get the normalized probability that NEE would sample the environment
  // This ensures consistent MIS weights with sampleLights()
  LightingTechniqueProbabilities probs     = getDirectLightingTechniqueProbabilities();
  float                          envWeight = probs.envWeight;

  return lastSamplePdf / (lastSamplePdf + envWeight * envPdf);
}

//-----------------------------------------------------------------------
// Shadow Catcher: Creates an invisible plane that only shows shadows
// Returns true to continue path tracing (shadowed area), false to break (fully lit)
//-----------------------------------------------------------------------
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
  // Step 1: Sample direct lighting and check for shadows
  DirectLight directLight;
  sampleLights(hit.pos, pbrMat.N, ray.Direction, seed, directLight);

  float3 shadowFactor = float3(1, 1, 1);  // Default: fully lit (no shadow)
  if(dot(directLight.direction, hit.nrm) > 0.0f && directLight.pdf != 0.0f)
  {
    // Trace shadow ray toward the light source
    RayDesc shadowRay;
    shadowRay.Origin    = hit.pos;
    shadowRay.Direction = directLight.direction;
    shadowRay.TMin      = 0.0;
    shadowRay.TMax      = INFINITE;
    shadowFactor        = raytracer.TraceShadow(shadowRay, seed);
  }

  // Step 2: Sample environment color in the view ray direction
  // This is what we'll see where the plane is not in shadow
  float3 envColor;
  float  envPdf;
  sampleEnvironment(ray.Direction, frameInfo, envColor, envPdf);

  // Step 3: Handle fully lit areas (no objects casting shadows)
  // In this case, the plane is "invisible" - just return the environment
  if(all(shadowFactor == float3(1, 1, 1)))
  {
    // Apply MIS weight for BSDF sample hitting environment
    float misWeight = computeEnvHitMisWeight(lastSamplePdf, envPdf);
    radiance += throughput * misWeight * envColor;
    return false;  // Signal to break - plane is invisible here
  }

  // Add direct contribution: environment modulated by shadow (darker in shadow)
  radiance += envColor * shadowFactor;

  // Non-physical control: subtract a fraction of env in shadowed areas.
  // darkenAmount is precomputed on CPU from slider [0..1] with exponential curve.
  // Subtraction can go negative to darken accumulated radiance.
  radiance -= envColor * (1.0f - shadowFactor) * frameInfo->shadowCatcherDarkenAmount;

  // Step 4: Handle shadowed areas - sample BSDF for indirect lighting
  // This bounce back the ray into the scene to gather indirect lighting
  BsdfSampleData sampleData;
  sampleData.k1 = -ray.Direction;
  sampleData.xi = float3(rand(seed), rand(seed), rand(seed));
  bsdfSampleSimple(sampleData, pbrMat);

  // Check for absorption
  if(sampleData.event_type == BSDF_EVENT_ABSORB)
  {
    return false;  // Signal to break - path terminated
  }

  // Continue path tracing to accumulate indirect lighting (sky bounce)
  float3 offsetDir = dot(sampleData.k2, hit.geonrm) > 0 ? hit.geonrm : -hit.geonrm;
  ray.Origin       = safeOffsetRay(hit.pos, offsetDir);
  ray.Direction    = sampleData.k2;
  throughput *= sampleData.bsdf_over_pdf;
  lastSamplePdf = sampleData.pdf;

  return true;  // Signal to continue - keep tracing for indirect lighting
}

//----------------------------------------------------------
// Testing if the hit is opaque or alpha-transparent
// Return 1.0 if it is opaque
//----------------------------------------------------------
float getOpacity(GltfRenderNode renderNode, GltfRenderPrimitive renderPrim, int triangleID, float3 barycentrics)
{
  // Scene materials
  uint               matIndex  = max(0, renderNode.materialID);
  GltfShadeMaterial* materials = pushConst.gltfScene->materials;  // Buffer of materials
  GltfShadeMaterial  mat       = materials[matIndex];
  GltfTextureInfo*   texInfos  = pushConst.gltfScene->textureInfos;

  if(mat.alphaMode == AlphaMode::eAlphaModeOpaque)
    return 1.0;

  // Getting the 3 indices of the triangle (local)
  uint3 triangleIndex = getTriangleIndices(renderPrim, triangleID);

  float baseColorAlpha = 1;
  if(mat.usePbrSpecularGlossiness == 0)
  {
    baseColorAlpha = mat.pbrBaseColorFactor.a;
    if(isTexturePresent(mat.pbrBaseColorTexture))
    {
      // Retrieve the interpolated texture coordinate from the vertex
      float2 uv = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);

      GltfTextureInfo texInfo = texInfos[mat.pbrBaseColorTexture];
      baseColorAlpha *= allTextures[texInfo.index].SampleLevel(uv, 0.0f).a;
    }
  }
  else
  {
    baseColorAlpha = mat.pbrDiffuseFactor.a;
    if(isTexturePresent(mat.pbrDiffuseTexture))
    {
      float2 uv = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);

      GltfTextureInfo texInfo = texInfos[mat.pbrDiffuseTexture];
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
// Calculate the transmission of a ray through a surface
// Return the transmission color
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
  uint               matIndex  = max(0, renderNode.materialID);
  GltfShadeMaterial* materials = pushConst.gltfScene->materials;  // Buffer of materials
  GltfShadeMaterial  mat       = materials[matIndex];
  GltfTextureInfo*   texInfos  = pushConst.gltfScene->textureInfos;

  // If hit a non-transmissive surface, terminate with full shadow
  if(mat.transmissionFactor <= MIN_TRANSMISSION)
  {
    return float3(0.0);
  }

  // Get triangle indices and compute normal
  uint3 indices = getTriangleIndices(renderPrim, triangleID);

  float3 normal;
  {
    // Compute geometric normal
    float3 v0 = getVertexPosition(renderPrim, indices.x);
    float3 v1 = getVertexPosition(renderPrim, indices.y);
    float3 v2 = getVertexPosition(renderPrim, indices.z);
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    normal    = normalize(cross(e1, e2));
    normal    = normalize(float3(mul(worldToObject, normal).xyz));
  }

  // Transmission calculation
  float3 currentTransmission = float3(mat.transmissionFactor);

  // Regular transmission with Fresnel using normal
  float cosTheta = abs(dot(rayDirection, normal));
  float fresnel  = schlickFresnel(mat.ior, cosTheta);
  currentTransmission *= float3((1.0 - fresnel));

  // Apply material color tint to transmission
  currentTransmission *= mat.pbrBaseColorFactor.rgb;

  // Volume attenuation (Beer's law) and scattering (KHR_materials_volume + KHR_materials_volume_scatter)
  if(mat.thicknessFactor > 0.0)
  {
    if(isInside)
    {
      // Absorption coefficient: σ_a
      float3 absCoeff = -log(max(mat.attenuationColor, float3(0.001))) / max(mat.attenuationDistance, 0.001);

      // Scattering coefficient: σ_s
      float3 scatterCoeff = absCoeff * multiToSingleScatterAlbedo(mat.multiscatterColor);

      // Extinction: σ_t = σ_a + σ_s
      float3 extinction = absCoeff + scatterCoeff;
      currentTransmission *= exp(-hitT * extinction);

      // Scattering shadow penalty (heuristic for light redirected away from direct path)
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

  // Attenuation due to roughness and metallic
  float transmissionAttenuation = 1.0;
  {
    float roughness = mat.pbrRoughnessFactor;
    float metallic  = mat.pbrMetallicFactor;
    if(isTexturePresent(mat.pbrMetallicRoughnessTexture))
    {
      float2 tc[2];
      tc[0] = getInterpolatedVertexTexCoord0(renderPrim, indices, barycentrics);
      tc[1] = getInterpolatedVertexTexCoord1(renderPrim, indices, barycentrics);

      float4 mr_sample = allTextures[texInfos[mat.pbrMetallicRoughnessTexture].index].SampleLevel(tc[0], 0.0f);
      roughness *= mr_sample.g;
      metallic *= mr_sample.b;
    }

    // Metallic completely blocks transmission
    transmissionAttenuation *= (1.0 - metallic);

    // Roughness reduces transmission non-linearly
    float roughnessEffect = 1.0 - (roughness * roughness);
    transmissionAttenuation *= lerp(0.65, 1.0, roughnessEffect);
  }
  currentTransmission *= transmissionAttenuation;

  return currentTransmission;
}

//-----------------------------------------------------------------------
// Volume Scattering (KHR_materials_volume_scatter)
//
// Handles volumetric scattering inside transmissive materials using:
// - Delta tracking to sample scatter distance
// - Henyey-Greenstein phase function for scatter direction
//
// Returns true if a scatter event occurred (caller should continue loop).
// Returns false if no scatter (absorption applied, continue to surface).
//-----------------------------------------------------------------------
bool handleVolumeScatter(PbrMaterial pbrMat, float hitDistance, inout RayDesc ray, inout float3 throughput, inout float lastSamplePdf, inout uint seed)
{
  float3 absCoeff     = absorptionCoefficient(pbrMat);
  float3 scatterCoeff = pbrMat.scatterCoefficient;

  // Extinction coefficient: σ_t = σ_a + σ_s
  float3 extinction = absCoeff + scatterCoeff;

  // Check if volume has scattering enabled
  float maxScatter = max(scatterCoeff.x, max(scatterCoeff.y, scatterCoeff.z));
  if(maxScatter > 0.001f)
  {
    float maxExtinction = max(extinction.x, max(extinction.y, extinction.z));

    // Sample distance to next scatter event using delta tracking
    float scatterDist = -log(max(rand(seed), 1e-10f)) / maxExtinction;

    if(scatterDist < hitDistance)
    {
      // Scatter event happens before reaching the surface

      // Apply transmittance (Beer's law) up to scatter point
      throughput *= exp(-scatterDist * extinction);

      // Apply single-scatter albedo: probability of scattering vs absorption
      // albedo = σ_s / σ_t (what fraction of extinction is scattering)
      throughput *= scatterCoeff / max(extinction, float3(0.001f));

      // Move ray to scatter point
      ray.Origin = ray.Origin + ray.Direction * scatterDist;

      // Sample new direction using Henyey-Greenstein phase function
      float3 wi     = ray.Direction;
      ray.Direction = sampleHenyeyGreenstein(float2(rand(seed), rand(seed)), pbrMat.scatterAnisotropy, wi);

      // Update PDF for MIS with environment lighting
      lastSamplePdf = henyeyGreensteinPdf(dot(wi, ray.Direction), pbrMat.scatterAnisotropy);

      return true;  // Scatter occurred - caller should continue loop
    }
  }

  // No scatter event - apply full extinction for path through volume
  throughput *= exp(-hitDistance * extinction);
  return false;  // No scatter - continue to surface interaction
}

//-----------------------------------------------------------------------
// Shoot a ray and store 1 if the ray hits the selected object
//-----------------------------------------------------------------------
void selectObject(float2 samplePos, float2 imageSize)
{
  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  const float2 subpixel_jitter = float2(0.5f, 0.5f);
  const bool isOrthographic = (pushConst.frameInfo->isOrthographic != 0);
  RayDesc    ray            = getRay(samplePos, subpixel_jitter, imageSize, pushConst.frameInfo->projInv,
                                     pushConst.frameInfo->viewInv, isOrthographic);

  RayQuery q;
  q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);

  while(q.Proceed())
  {
    q.CommitNonOpaqueTriangleHit();
  }

  float hitObj = 0.0;
  if(q.CommittedStatus() != COMMITTED_NOTHING)
  {
    int rprimID = q.CommittedInstanceIndex();
    if(rprimID != -1 && rprimID == pushConst.frameInfo->selectedRenderNode)
    {
      hitObj = 1.0f;
    }
  }
  // Store the hit object in the selection image
  outImages[int(OutputImage::eSelectImage)][int2(samplePos)] = float4(hitObj, 0, 0, 0);
}

//-----------------------------------------------------------------------
// Check for infinite plane intersection and update hit state if needed
//-----------------------------------------------------------------------
bool checkInfinitePlaneIntersection(RayDesc ray, inout HitPayload payload, inout HitState hit, SceneFrameInfo* frameInfo)
{
  if(frameInfo->useInfinitePlane == 0)
    return false;

  // Plane definition
  float3 normal      = float3(0, 1, 0);                   // Y-up plane normal
  float  planeHeight = frameInfo->infinitePlaneDistance;  // Height of plane from origin (0, planeHeight, 0)

  // Only report intersection if camera is above the plane
  if(ray.Origin.y <= planeHeight)
    return false;

  // Calculate denominator for ray-plane intersection: dot product of plane normal and ray direction
  // If this is close to zero, the ray is parallel to the plane
  float Dn = dot(ray.Direction, normal);
  if(abs(Dn) <= 1e-6)
    return false;

  float On               = dot(ray.Origin, normal);
  float intersectionDist = (-On + planeHeight) / Dn;
  if(intersectionDist <= 0 || (intersectionDist >= payload.hitT))
    return false;

  // Update hit information
  payload.hitT  = intersectionDist;  // Update hit distance
  hit.pos       = ray.Origin + ray.Direction * payload.hitT;
  hit.shadowPos = hit.pos;
  hit.nrm       = normal;           // Plane normal pointing up
  hit.geonrm    = normal;           // Geometric normal is the same as the plane normal
  hit.tangent   = float3(1, 0, 0);  // Arbitrary tangent
  hit.bitangent = float3(0, 0, 1);  // Arbitrary bitangent

  return true;  // We hit the infinite plane
}

//-----------------------------------------------------------------------
// Path tracing
//
// This function implements unbiased Monte Carlo path tracing:
// 1. Traces rays through the scene, bouncing off surfaces according to their material
//    properties (reflection, refraction, transmission, etc.) up to a maximum depth
// 2. At each intersection, calculates direct lighting via Next Event Estimation (NEE)
//    by sampling lights and evaluating the BSDF
// 3. Samples the BSDF to determine the next ray direction for indirect lighting
// 4. Applies Russian Roulette path termination to improve efficiency
// 5. Handles special features:
//    - Shadow catcher: invisible plane that shows only shadows for compositing
//    - Volumetric effects: transmission, attenuation, and scattering
//    - Alpha transparency: stochastic alpha testing for correct compositing
//
SampleResult pathTrace(IRaytracer raytracer, RayDesc ray, inout uint seed)
{
  float3 radiance     = float3(0.0F, 0.0F, 0.0F);
  float3 throughput   = float3(1.0F, 1.0F, 1.0F);
  bool   isInside     = false;
  float2 maxRoughness = float2(0.0);
  bool   solid        = true;

  float lastSamplePdf = DIRAC;

  // #DLSS - Store data temporarily to avoid writing to sampleResult during loop (reduces live state)
  bool       dlss_hasData         = false;
  float16_t3 dlss_albedo          = float16_t3(0);
  float16_t3 dlss_specularAlbedo  = float16_t3(0);
  float16_t4 dlss_normalRoughness = float16_t4(0);
  float3     dlss_hitPosition     = 1e32f;

  // Path tracing loop, until the ray hits the environment or the maximum depth is reached or the ray is absorbed
  for(int depth = 0; depth < pushConst.maxDepth; depth++)
  {
    // Ensure ray direction is normalized
    ray.Direction = normalize(ray.Direction);

    bool   nextEventValid = false;      // If the next event is valid, then we can sample the BSDF for the light
    float3 contribution   = float3(0);  // Direct lighting contribution

    // Shadow ray data (minimal extraction to reduce live state)
    float16_t3 shadowRayPos;
    float16_t3 shadowRayDir;
    float16_t  shadowRayDist;

    {
      //DirectLight directLight;
      SceneFrameInfo* frameInfo = pushConst.frameInfo;
      HitPayload      payload   = {};

      // Trace the ray through the scene
      raytracer.Trace(ray, payload, seed);

      // Getting the hit information (primitive/mesh that was hit)
      HitState hit = payload.hitState;

      bool firstRay = (depth == 0);

      // Check if we hit the infinite plane
      bool hitInfinitePlane = checkInfinitePlaneIntersection(ray, payload, hit, frameInfo);

      // Hitting the environment, then exit
      if(payload.hitT == INFINITE)
      {
        if(firstRay)  // If we come in here, the first ray didn't hit anything
        {
          solid = false;  // Set it to transparent

          // #DLSS - Set environment hit position for proper motion vectors (store temporarily)
          // sampleResult.dlssOutput.hitPosition = ray.Origin + ray.Direction * 1000000.0f;
          dlss_hitPosition = ray.Origin + ray.Direction * 1000000.0f;

          // Solid color background and blurred HDR environment, aren't part of the
          // lighting equation (backplate), so we can return them directly.
          if(frameInfo->useSolidBackground == 1)
          {
            radiance = frameInfo->backgroundColor;
            break;
          }
          else if(pushConst.frameInfo->environmentType == EnvSystem::eHdr && pushConst.frameInfo->envBlur > 0)
          {
            float3 dir = rotate(ray.Direction, float3(0, 1, 0), -frameInfo->envRotation);
            float2 uv  = getSphericalUv(dir);  // See sampling.glsl
            radiance = smoothHDRBlur(texturesHdr[HDR_IMAGE_INDEX], uv, frameInfo->envBlur).xyz * frameInfo->envIntensity;
            break;
          }
        }

        // Add sky or HDR texture using refactored helper
        float3 envColor;
        float  envPdf;
        sampleEnvironment(ray.Direction, frameInfo, envColor, envPdf);

        // We may hit the environment twice: once via sampleLights() and once when hitting the sky while probing
        // for more indirect hits. This is the counter part of the MIS weighting in sampleLights()
        float misWeight = computeEnvHitMisWeight(lastSamplePdf, envPdf);
        radiance += throughput * misWeight * envColor;

        break;
      }

      GltfShadeMaterial material;  // The orginal glTF material
      PbrMaterial       pbrMat;    // The evalualted PBR material

      // If the ray hits the infinite plane
      if(hitInfinitePlane)
      {
        // Evaluate the material at the hit point
        pbrMat   = defaultPbrMaterial(frameInfo->infinitePlaneBaseColor, frameInfo->infinitePlaneMetallic,
                                      frameInfo->infinitePlaneRoughness, hit.nrm, hit.nrm);
        material = defaultGltfMaterial();

        // Shadow catcher: invisible plane that shows only shadows for compositing
        if(frameInfo->useInfinitePlane == 2)
        {
          bool continuePath = handleShadowCatcher(raytracer, hit, pbrMat, ray, radiance, throughput, lastSamplePdf, seed, frameInfo);
          if(!continuePath)
            break;
          continue;
        }
      }
      else
      {
        // Getting the scene information
        GltfShadeMaterial*   materials       = pushConst.gltfScene->materials;         // Buffer of materials
        GltfRenderNode*      renderNodes     = pushConst.gltfScene->renderNodes;       // Buffer of render nodes
        GltfRenderPrimitive* renderPrimitive = pushConst.gltfScene->renderPrimitives;  // Buffer of meshes
        GltfTextureInfo*     texInfos        = pushConst.gltfScene->textureInfos;      // Buffer of texture infos

        // Setting up the material
        GltfRenderPrimitive renderPrim    = renderPrimitive[payload.rprimID];  // Primitive information
        GltfRenderNode      renderNode    = renderNodes[payload.rnodeID];      // Node information
        int                 materialIndex = max(0, renderNode.materialID);     // Material ID of hit mesh
        material                          = materials[materialIndex];          // Material of the hit object

        material.pbrBaseColorFactor *= hit.color;  // Modulate the base color with the vertex color

        // Evaluate the material at the hit point
        MeshState mesh = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, isInside);
        pbrMat         = evaluateMaterial(material, mesh, allTextures, texInfos);
      }

      // #DLSS - Gather data from first hit (store temporarily, write at end to reduce live state)
      if(firstRay)
      {
        float3 specularAlbedo = EnvBRDFApprox2(pbrMat.specularColor, pbrMat.roughness.x, dot(pbrMat.N, ray.Direction));
        dlss_specularAlbedo   = float16_t3(specularAlbedo);
        // Note: for transparency hack the other information are collected AFTER BSDF sampling to know if surface is transmissive
        if(USE_DLSS_TRANSP == 0 || pushConst.useOptixDenoiser == 1)
        {
          dlss_normalRoughness = float16_t4(float16_t3(pbrMat.N), float16_t(pbrMat.roughness.x));
          dlss_hitPosition     = hit.pos;
          dlss_albedo          = float16_t3(pbrMat.baseColor.xyz);
        }
      }

      // Keep track of the maximum roughness to prevent firefly artifacts
      // by forcing subsequent bounces to be at least as rough
      maxRoughness     = max(pbrMat.roughness, maxRoughness);
      pbrMat.roughness = maxRoughness;

      // Debugging, single frame
      if(frameInfo->debugMethod != DebugMethod::eNone && firstRay)
      {
        radiance.xyz = debugValue(pbrMat, hit, frameInfo->debugMethod);
        break;
      }


      // Adding emissive
      radiance += pbrMat.emissive * throughput;

      // Unlit
      if(material.unlit > 0)
      {
        radiance += pbrMat.baseColor;
        break;
      }

      // Apply volume effects: scattering and/or absorption (KHR_materials_volume + KHR_materials_volume_scatter)
      if(isInside && !pbrMat.isThinWalled)
      {
        if(handleVolumeScatter(pbrMat, payload.hitT, ray, throughput, lastSamplePdf, seed))
        {
          continue;  // Scatter event occurred - skip surface interaction
        }
        // If no scatter, absorption was applied inside handleVolumeScatter
      }

      // Light contribution; can be environment or punctual lights
      DirectLight directLight;
      sampleLights(hit.pos, pbrMat.N, ray.Direction, seed, directLight);

      // Do not next event estimation (but delay the adding of contribution)
      nextEventValid = (dot(directLight.direction, hit.nrm) > 0.0f || pbrMat.diffuseTransmissionFactor > 0.0f)
                       && directLight.pdf != 0.0f;

      // Evaluate BSDF for Light
      if(nextEventValid)
      {
        // Evaluate the BSDF at the hit point
        BsdfEvaluateData evalData;
        evalData.k1 = -ray.Direction;
        evalData.k2 = directLight.direction;
        evalData.xi = float3(rand(seed), rand(seed), rand(seed));
        bsdfEvaluate(evalData, pbrMat);
        // bsdfEvaluateSimple(evalData, pbrMat);

        // If the PDF is greater than 0, then we can sample the BSDF
        if(evalData.pdf > 0.0)
        {
          // Weight for combining light and BSDF sampling strategies (Multiple Importance Sampling)
          const float mis_weight = (directLight.pdf == DIRAC) ? 1.0F : directLight.pdf / (directLight.pdf + evalData.pdf);

          // sample weight
          const float3 w = throughput * directLight.radianceOverPdf * mis_weight;
          contribution += w * evalData.bsdf_diffuse;
          contribution += w * evalData.bsdf_glossy;
        }
      }


      {
        // Sample the BSDF
        BsdfSampleData sampleData;
        sampleData.k1 = -ray.Direction;                              // outgoing direction
        sampleData.xi = float3(rand(seed), rand(seed), rand(seed));  // random number
        bsdfSample(sampleData, pbrMat);
        // bsdfSampleSimple(sampleData, pbrMat);

        // Update the throughput
        throughput *= sampleData.bsdf_over_pdf;
        ray.Direction = sampleData.k2;  // new direction
        lastSamplePdf = sampleData.pdf;

        if(sampleData.event_type != BSDF_EVENT_ABSORB)
        {
          // Continue path
          bool isSpecular     = (sampleData.event_type & BSDF_EVENT_IMPULSE) != 0;
          bool isTransmission = (sampleData.event_type & BSDF_EVENT_TRANSMISSION) != 0;

          float3 offsetDir = dot(ray.Direction, hit.geonrm) > 0 ? hit.geonrm : -hit.geonrm;
          ray.Origin       = safeOffsetRay(hit.pos, offsetDir);

          // Flip the information if we are inside the object, but only if it is a solid object
          // The doubleSided flag is used to know if the object is solid or thin-walled.
          // This is not a glTF specification, but works in many cases.
          if(isTransmission)
          {
            isInside = !isInside;
          }
          // Transparency hack: collect data after knowing if surface is transmissive
          // Skip clear, smooth glass but keep rough/frosted transmissive materials
          if(USE_DLSS_TRANSP == 1 && pushConst.useOptixDenoiser == 0)
          {
            // Only skip glass that's both highly transmissive AND smooth
            bool isClearGlass = (pbrMat.transmission > 0.3f) && (pbrMat.roughness.x < 0.4f);
            if(dlss_hasData == false && !isTransmission && !isClearGlass)
            {
              dlss_hasData         = true;
              dlss_albedo          = float16_t3(pbrMat.baseColor.xyz);
              dlss_normalRoughness = float16_t4(float16_t3(pbrMat.N), float16_t(pbrMat.roughness.x));
              dlss_hitPosition     = hit.pos;
            }
          }
        }
        else if(sampleData.event_type == BSDF_EVENT_ABSORB)  // If the ray is absorbed , then break
        {
          // Exit tracing rays, but still finish this iteration; in particular the visibility test
          // for the light that we may have hit.
          depth = pushConst.maxDepth;
        }

        // End scope to free live states
      }

      // Extract minimal shadow ray data before scope closes (to free hit/directLight structures)
      float3 shadowOffsetDir = dot(directLight.direction, hit.geonrm) > 0 ? hit.geonrm : -hit.geonrm;
      // The shadowPos already contains the correction from pointOffset ("Hacking the shadow terminator"), but offset for planar surfaces
      shadowRayPos  = float16_t3(safeOffsetRay(hit.shadowPos, shadowOffsetDir));
      shadowRayDir  = float16_t3(directLight.direction);
      shadowRayDist = float16_t(directLight.distance);
    }

    // We are adding the contribution to the radiance only if the ray is not occluded by an object.
    if(nextEventValid)
    {
      RayDesc shadowRay;
      shadowRay.Origin    = shadowRayPos;
      shadowRay.Direction = shadowRayDir;
      shadowRay.TMin      = 0.0;
      shadowRay.TMax      = shadowRayDist;
      float3 shadowFactor = raytracer.TraceShadow(shadowRay, seed);
      radiance += contribution * shadowFactor;
    }

    // Russian-Roulette (minimizing live state)
    if(depth >= RR_MIN_DEPTH)
    {
      float rrPcont = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001F, 0.95F);
      if(rand(seed) >= rrPcont)
        break;                // paths with low throughput that won't contribute
      throughput /= rrPcont;  // boost the energy of the non-terminated paths
    }
  }

  // Returning the sample result; radiance + DLSS data
  SampleResult sampleResult = {};
  sampleResult.radiance     = float4(radiance, solid ? 1 : 0);

  sampleResult.dlssOutput.albedo          = float16_t4(dlss_albedo, solid ? 1.0h : 0.0h);
  sampleResult.dlssOutput.specularAlbedo  = dlss_specularAlbedo;
  sampleResult.dlssOutput.normalRoughness = dlss_normalRoughness;
  sampleResult.dlssOutput.hitPosition     = dlss_hitPosition;

  return sampleResult;
}

//-----------------------------------------------------------------------
// Sampling the pixel
// This function samples a pixel (trace a path of rays through the scene)
// and returns the result
//-----------------------------------------------------------------------
SampleResult samplePixel(IRaytracer raytracer,
                         inout uint seed,
                         float2     samplePos,
                         float2     subpixelJitter,
                         float2     imageSize,
                         float4x4   projMatrixI,
                         float4x4   viewMatrixI,
                         float      focalDist,
                         float      aperture)
{
  const bool isOrthographic = (pushConst.frameInfo->isOrthographic != 0);
  RayDesc    ray            = getRay(samplePos, subpixelJitter, imageSize, projMatrixI, viewMatrixI, isOrthographic);

  // Depth-of-Field (disabled for orthographic cameras)
  if(!isOrthographic)
  {
    float3 focalPoint        = focalDist * ray.Direction;
    float  cam_r1            = rand(seed) * M_TWO_PI;
    float  cam_r2            = rand(seed) * aperture;
    float4 cam_right         = mul(viewMatrixI, float4(1, 0, 0, 0));
    float4 cam_up            = mul(viewMatrixI, float4(0, 1, 0, 0));
    float3 randomAperturePos = (cos(cam_r1) * cam_right.xyz + sin(cam_r1) * cam_up.xyz) * sqrt(cam_r2);
    float3 finalRayDir       = normalize(focalPoint - randomAperturePos);

    // Set the new ray origin and direction with depth-of-field
    ray.Origin += randomAperturePos;
    ray.Direction = finalRayDir;
  }

  SampleResult sampleResult = pathTrace(raytracer, ray, seed);

  // Removing fireflies
  float lum = dot(sampleResult.radiance.xyz, float3(1.0F / 3.0F));
  if(lum > pushConst.fireflyClampThreshold)
  {
    sampleResult.radiance *= pushConst.fireflyClampThreshold / lum;
  }

  return sampleResult;
}


//-----------------------------------------------------------------------
// Common function for both compute and ray generation shaders
// This is the main function that is called for each pixel
//-----------------------------------------------------------------------
void processPixel(IRaytracer raytracer, float2 samplePos, float2 imageSize)
{
  // Check if the sample position is within the image bounds
  if(samplePos.x >= imageSize.x || samplePos.y >= imageSize.y)
    return;

  // Check if the sample position is the mouse coordinate
  if(samplePos.x == pushConst.mouseCoord.x && samplePos.y == pushConst.mouseCoord.y)
  {
    // Set the debug flag to true (used for printing debug information)
    doDebug = true;
  }

  // Shoot a ray to find which element is selected, done only when the object selection changed
  // or when rendering is re-starting (camera, object moved, .. )
  if(pushConst.renderSelection == 1 || pushConst.frameCount <= 1)
  {
    selectObject(samplePos, imageSize);
  }

  // Initialize the random number
  uint seed = xxhash32(uint3(uint2(samplePos.xy), pushConst.frameCount));

  // Subpixel jitter: send the ray through a different position inside the
  // pixel each time, to provide antialiasing.
  float2 subpixelJitter = float2(0.5f, 0.5f);
  if(pushConst.frameCount > 0)
  {
    // Add the jitter to the subpixel jitter
    subpixelJitter += ANTIALIASING_STANDARD_DEVIATION * sampleGaussian(float2(rand(seed), rand(seed)));
  }

  // #DLSS - use the DLSS jitter and frame index (not resetting to zero)
  if(pushConst.useDlss == 1)
  {
    subpixelJitter = pushConst.jitter + float2(0.5f, 0.5f);
  }

  // Sampling n times the pixel
  SampleResult sampleResult = samplePixel(raytracer, seed, samplePos, subpixelJitter, imageSize, pushConst.frameInfo->projInv,
                                          pushConst.frameInfo->viewInv, pushConst.focalDistance, pushConst.aperture);
  float4 pixelColor = sampleResult.radiance;
  for(int s = 1; s < pushConst.numSamples; s++)
  {
    subpixelJitter = float2(rand(seed), rand(seed));
    sampleResult   = samplePixel(raytracer, seed, samplePos, subpixelJitter, imageSize, pushConst.frameInfo->projInv,
                                 pushConst.frameInfo->viewInv, pushConst.focalDistance, pushConst.aperture);
    pixelColor += sampleResult.radiance;
  }
  pixelColor /= pushConst.numSamples;

  bool first_frame = (pushConst.frameCount == 0);

  // Saving result
  if(first_frame || (pushConst.useDlss == 1))
  {  // First frame, replace the value in the buffer
    outImages[int(OutputImage::eResultImage)][int2(samplePos)] = pixelColor;
  }
  else
  {
    // Do accumulation over time using uniform weighting
    float  totalSamplesAfter = float(pushConst.totalSamples + pushConst.numSamples);
    float4 old_color         = outImages[0][int2(samplePos)];
    outImages[int(OutputImage::eResultImage)][int2(samplePos)] =
        (old_color * pushConst.totalSamples + pixelColor * pushConst.numSamples) / totalSamplesAfter;
  }

  // #DLSS - Storing the GBuffer for the DLSS denoiser
  if(pushConst.useDlss == 1)
  {
    // Transform world position to clip space and calculate depth
    float4 posScreen = mul(float4(sampleResult.dlssOutput.hitPosition, 1.0), pushConst.frameInfo->viewProjMatrix);
    float  viewZ     = posScreen.z / posScreen.w;  // Depth in NDC space

    // Calculate motion vectors using the hit position (works for both geometry and environment)
    float2 motionVec = calculateMotionVector(sampleResult.dlssOutput.hitPosition, pushConst.frameInfo->prevMVP,
                                             pushConst.frameInfo->viewProjMatrix, imageSize);
    outImages[int(OutputImage::eDlssDepth)][int2(samplePos)]           = float4(abs(viewZ));
    outImages[int(OutputImage::eDlssMotion)][int2(samplePos)]          = float4(motionVec, 0, 0);
    outImages[int(OutputImage::eDlssNormalRoughness)][int2(samplePos)] = sampleResult.dlssOutput.normalRoughness;
    outImages[int(OutputImage::eDlssAlbedo)][int2(samplePos)]          = sampleResult.dlssOutput.albedo;
    outImages[int(OutputImage::eDlssSpecAlbedo)][int2(samplePos)] = float4(sampleResult.dlssOutput.specularAlbedo.xyz, 1.0f);
  }

  if(pushConst.useOptixDenoiser == 1)
  {
    // Transform normal from world space to camera space for OptiX denoiser
    // OptiX requires normals in camera space (right-handed, -Z forward, Y up, X right)
    float3 worldNormal   = sampleResult.dlssOutput.normalRoughness.xyz;
    float3 cameraNormal  = normalize(mul(float3x3(pushConst.frameInfo->viewMatrix), worldNormal));
    uint   normalEncoded = compressUnitVec(cameraNormal);
    outImages[int(OutputImage::eOptixAlbedoNormal)][int2(samplePos)] =
        float4(sampleResult.dlssOutput.albedo.xyz, asfloat(normalEncoded));
  }
}

//-----------------------------------------------------------------------
// RAY GENERATION
//-----------------------------------------------------------------------
[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void computeMain(uint3 threadIdx: SV_DispatchThreadID)
{
  RayQueryRaytracer raytracer;
  float2            samplePos = (float2)threadIdx.xy;
  uint2             imageSize;
  outImages[int(OutputImage::eResultImage)].GetDimensions(imageSize.x, imageSize.y);

  processPixel(raytracer, samplePos, imageSize);
}

//-----------------------------------------------------------------------
// RAY GENERATION
//-----------------------------------------------------------------------
[shader("raygeneration")]
void rgenMain()
{
  TraditionalRaytracer raytracer;
  float2               samplePos = (float2)DispatchRaysIndex().xy;
  float2               imageSize = (float2)DispatchRaysDimensions().xy;

  processPixel(raytracer, samplePos, imageSize);
}

//-----------------------------------------------------------------------
// CLOSEST HIT
//-----------------------------------------------------------------------
[shader("closesthit")]
void rchitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
  float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

  int instanceID   = InstanceIndex();
  int renderPrimID = InstanceID();
  int primitiveID  = PrimitiveIndex();

  // Get the built-in ray tracing variables
  float4x3 worldToObject  = WorldToObject4x3();
  float4x3 objectToWorld  = ObjectToWorld4x3();
  float3   worldRayOrigin = WorldRayOrigin();
  float    hitT           = RayTCurrent();

  // Retrieve the Primitive mesh buffer information
  GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

  HitState hit = getHitState(renderPrim, barycentrics, worldToObject, objectToWorld, primitiveID, WorldRayDirection());

  payload.hitT     = hitT;
  payload.rprimID  = renderPrimID;
  payload.rnodeID  = instanceID;
  payload.hitState = hit;
}

[shader("closesthit")]
void rchitShadow(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
  // When we reach the closest hit for shadow rays, it means we've accepted a hit
  // in rahitShadow due to low transmission. The transmission has already been
  // processed in rahitShadow, so we just ensure it's set to fully occluded.
  payload.totalTransmission = float3(0.0);
}


//-----------------------------------------------------------------------
// MISS
//-----------------------------------------------------------------------
[shader("miss")]
void rmissMain(inout HitPayload payload)
{
  payload.hitT = INFINITE;
}

[shader("miss")]
void rmissShadow(inout ShadowPayload payload)
{
  // When we miss, we've passed through all surfaces along the ray.
  // Keep the accumulated transmission - don't reset it.
  // totalTransmission already contains the accumulated value from rahitShadow
  payload.approxHitT = INFINITE;
}

//-----------------------------------------------------------------------
// INTERSECTION (Any Hit)
//-----------------------------------------------------------------------

[shader("anyhit")]
void rahitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
  float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

  uint instanceID   = InstanceIndex();
  uint renderPrimID = InstanceID();
  uint triangleID   = PrimitiveIndex();

  // Get the built-in ray tracing variables
  float3x4 worldToObject = WorldToObject3x4();
  float3   worldRayDir   = WorldRayDirection();
  float    hitT          = RayTCurrent();

  // Retrieve the Primitive mesh buffer information
  GltfRenderNode      renderNode = pushConst.gltfScene->renderNodes[instanceID];
  GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

  float opacity = getOpacity(renderNode, renderPrim, triangleID, barycentrics);
  if(rand(payload.seed) > opacity)
  {
    IgnoreHit();
  }
}


[shader("anyhit")]
void rahitShadow(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
  float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

  uint instanceID   = InstanceIndex();
  uint renderPrimID = InstanceID();
  uint primitiveID  = PrimitiveIndex();

  // Get the built-in ray tracing variables
  float4x3 worldToObject = WorldToObject4x3();
  float3   worldRayDir   = WorldRayDirection();
  float    hitT          = RayTCurrent();

  // Retrieve the Primitive mesh buffer information
  GltfRenderNode      renderNode = pushConst.gltfScene->renderNodes[instanceID];
  GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

  float opacity = getOpacity(renderNode, renderPrim, primitiveID, barycentrics);
  float r       = rand(payload.seed);
  if(r < opacity)
  {
    payload.approxHitT  = abs(hitT - payload.approxHitT);
    bool   isInside     = payload.isInside;
    float3 transmission = getShadowTransmission(renderNode, renderPrim, primitiveID, barycentrics, payload.approxHitT,
                                                worldToObject, worldRayDir, isInside);

    payload.isInside = isInside;
    payload.totalTransmission *= transmission;

    if(max(max(payload.totalTransmission.r, payload.totalTransmission.g), payload.totalTransmission.b) <= MIN_TRANSMISSION)
    {
      payload.totalTransmission = float3(0.0);
      AcceptHitAndEndSearch();
    }
  }
  // We want all possible intersections
  IgnoreHit();
}
