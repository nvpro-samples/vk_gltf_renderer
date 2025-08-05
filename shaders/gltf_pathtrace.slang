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

#include "nvshaders/functions.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/sky_functions.h.slang"

#include "nvshaders/bsdf_types.h.slang"
#include "nvshaders/bsdf_functions.h.slang"
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
#include "dlss_util.h"

#include "common.h.slang"
#include "raytracer_interface.h.slang"


// Bindings
// clang-format off
[[vk::push_constant]]                               ConstantBuffer<PathtracePushConstant>   pushConst;
[[vk::binding(BindingPoints::eTextures, 0)]]        Sampler2D                               allTextures[];
[[vk::binding(BindingPoints::eTexturesHdr, 0)]]     Sampler2D                               texturesHdr[];
[[vk::binding(BindingPoints::eTlas, 1)]]            RaytracingAccelerationStructure         topLevelAS;
[[vk::binding(BindingPoints::eOutImages, 1)]]       RWTexture2D<float4>                     outImages[];

// HDR Environment
[[vk::binding(EnvBindings::eImpSamples, 2)]]    StructuredBuffer<EnvAccel>  envSamplingData;

[[vk::constant_id(0)]]          int USE_SER;

// clang-format on


static bool doDebug = false;

// Direct light structure
struct DirectLight
{
  float3 direction;        // Direction to the light
  float3 radianceOverPdf;  // Radiance over pdf
  float  distance;         // Distance to the light
  float  pdf;              // Probability of sampling this light
};

static const float MIN_TRANSMISSION = 0.01;  // Minimum transmission factor to continue tracing

// #DLSS - Information for the DLSS
struct DlssOutput
{
  float4 albedo          = float4(0);
  float3 specularAlbedo  = float3(0);
  float4 normalRoughness = float4(0);
  float3 hitPosition     = 1e34f;
};

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
// Standard deviation of the Gaussian filter used for antialiasing,
// in units of pixels.
// This value of 1 / sqrt(8 ln(2)) makes it so that a Gaussian centered
// on a pixel is at exactly 1/2 its maximum at the midpoints between
// orthogonally adjacent pixels, and 1/4 its maximum at the "corners"
// of pixels. It also empirically looks nice: larger values are
// too blurry, and smaller values make thin lines look jagged.
//-----------------------------------------------------------------------
#define ANTIALIASING_STANDARD_DEVIATION 0.4246609F

RayDesc getRay(float2 samplePos, float2 offset, float2 imageSize, float4x4 projMatrixI, float4x4 viewMatrixI)
{
  const float2 clipCoords = (samplePos + offset) / imageSize * 2.0 - 1.0;
  float4       viewCoords = mul(float4(clipCoords, -1.0, 1.0), projMatrixI);
  viewCoords /= viewCoords.w;

  RayDesc ray;
  ray.Origin    = viewMatrixI[3].xyz;
  ray.Direction = normalize(mul(viewCoords, viewMatrixI).xyz - ray.Origin);
  ray.TMin      = 0.0;
  ray.TMax      = INFINITE;
  return ray;
}


//-----------------------------------------------------------------------
// This should sample any lights in the scene, but we only have the sun
void sampleLights(in float3 pos, float3 normal, in float3 worldRayDirection, inout uint seed, out DirectLight directLight)
{

  float3 radiance             = float3(0.);
  directLight.pdf             = 0.0;
  directLight.distance        = INFINITE;
  directLight.radianceOverPdf = float3(0.0);
  float envPdf                = 0.0;

  // We use the one-sample model to perform multiple importance sampling
  // between point lights and environment lights.
  // See section 9.2.4 of https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf .

  // Probability we'll select each sampling scheme (TODO: adjust these based on
  // scene characteristics: light intensity vs environment intensity)
  float lightWeight = (pushConst.gltfScene.numLights > 0) ? 0.5 : 0.0;
  float envWeight =
      ((pushConst.frameInfo->environmentType == EnvSystem::eSky) || pushConst.frameInfo.envIntensity.x > 0.0) ? 0.5 : 0.0;

  // Normalize weights
  float totalWeight = lightWeight + envWeight;
  if(totalWeight == 0.0f)
  {
    return;  // No lights to sample
  }

  lightWeight /= totalWeight;
  envWeight /= totalWeight;

  // Decide whether to sample the light or the environment.
  bool sampleLights = (rand(seed) <= lightWeight);

  // We'll choose a direction from one technique, but for MIS we need the
  // PDFs of each technique in the direction we chose. That's why we always get
  // the light PDF (which is constant), but only sample a light if
  // `sampleLights` is true.

  // Lights
  if(lightWeight > 0)
  {
    directLight.pdf = 1.0 / pushConst.gltfScene.numLights;
    if(sampleLights)  // Use this technique for MIS?
    {
      int lightIndex = min(int(rand(seed) * pushConst.gltfScene.numLights), pushConst.gltfScene.numLights - 1);
      GltfLight light = pushConst.gltfScene.lights[lightIndex];  // RenderLightBuf(sceneDesc.lightAddress)._[lightIndex];
      LightContrib contrib  = singleLightContribution(light, pos, normal, float2(rand(seed), rand(seed)));
      directLight.direction = -contrib.incidentVector;
      radiance              = contrib.intensity / (directLight.pdf * lightWeight);
      directLight.distance  = contrib.distance;
    }
  }

  // Environment
  if(envWeight > 0)
  {
    if(pushConst.frameInfo->environmentType == EnvSystem::eSky)
    {
      if(!sampleLights)  // Use this technique for MIS?
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
      if(!sampleLights)  // Use this technique for MIS?
      {
        float3 rand_val = float3(rand(seed), rand(seed), rand(seed));
        float4 radiance_pdf = environmentSample(texturesHdr[HDR_IMAGE_INDEX], envSamplingData, rand_val, directLight.direction);
        envPdf                = radiance_pdf.w;
        radiance              = radiance_pdf.xyz * pushConst.frameInfo.envIntensity / (envPdf * envWeight);
        directLight.direction = rotate(directLight.direction, float3(0, 1, 0), pushConst.frameInfo.envRotation);
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

  // MIS weight calculation
  float misWeight = (sampleLights ? directLight.pdf : envPdf) / (directLight.pdf + envPdf);
  radiance *= misWeight;
  // Update the total PDF
  directLight.pdf = lightWeight * directLight.pdf + envWeight * envPdf;

  directLight.radianceOverPdf = radiance;  // Radiance over PDF
  return;
}


//----------------------------------------------------------
// Testing if the hit is opaque or alpha-transparent
// Return true is opaque
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

  // Volume attenuation (Beer's law)
  if(mat.thicknessFactor > 0.0)
  {
    if(isInside)
    {
      // Calculate per-channel attenuation with improved color preservation
      float3 absorbance = -log(max(mat.attenuationColor, float3(0.001))) / max(mat.attenuationDistance, 0.001);
      float3 attenuation;
      attenuation.r = exp(-hitT * absorbance.r);
      attenuation.g = exp(-hitT * absorbance.g);
      attenuation.b = exp(-hitT * absorbance.b);

      currentTransmission *= attenuation;
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
// Shoot a ray and store 1 if the ray hits the selected object
void selectObject(float2 samplePos, float2 imageSize)
{
  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  const float2 subpixel_jitter = float2(0.5f, 0.5f);
  const float2 clipCoords      = (samplePos + subpixel_jitter) / imageSize * 2.0 - 1.0;
  float4       viewCoords      = mul(float4(clipCoords, -1.0, 1.0), pushConst.frameInfo->projInv);
  viewCoords /= viewCoords.w;

  RayDesc ray;
  ray.Origin =
      float3(pushConst.frameInfo->viewInv[3].x, pushConst.frameInfo->viewInv[3].y, pushConst.frameInfo->viewInv[3].z);
  ray.Direction = normalize(mul(viewCoords, pushConst.frameInfo->viewInv).xyz - ray.Origin);
  ray.TMin      = 0.0;
  ray.TMax      = INFINITE;

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
  if(frameInfo.useInfinitePlane != 1)
    return false;

  // Plane definition
  float3 normal      = float3(0, 1, 0);                  // Y-up plane normal
  float  planeHeight = frameInfo.infinitePlaneDistance;  // Height of plane from origin (0, planeHeight, 0)

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
  hit.nrm       = normal;           // Plane normal pointing up
  hit.geonrm    = normal;           // Geometric normal is the same as the plane normal
  hit.tangent   = float3(1, 0, 0);  // Arbitrary tangent
  hit.bitangent = float3(0, 0, 1);  // Arbitrary bitangent

  return true;  // We hit the infinite plane
}

//-----------------------------------------------------------------------
// Path tracing
//
// This function:
// 1. Traces rays through a scene, bouncing them off surfaces according to their material
//    properties (like reflection, transmission, etc.) up to a maximum depth
// 2. At each intersection, it calculates direct lighting contribution from light sources
//    (currently only sky/sun) and samples the BSDF to determine the next ray direction
// 3. Accumulates radiance along the path while applying Russian Roulette for optimization,
//    handling both surface and volumetric effects, and returns the final color contribution for that ray path
//
SampleResult pathTrace(IRaytracer raytracer, RayDesc ray, inout uint seed)
{
  SampleResult sampleResult = {};
  float3       radiance     = float3(0.0F, 0.0F, 0.0F);
  float3       throughput   = float3(1.0F, 1.0F, 1.0F);
  bool         isInside     = false;
  float2       maxRoughness = float2(0.0);

  SceneFrameInfo* frameInfo = pushConst.frameInfo;

  HitPayload payload = {};

  float lastSamplePdf = DIRAC;

  // Path tracing loop, until the ray hits the environment or the maximum depth is reached or the ray is absorbed
  for(int depth = 0; depth < pushConst.maxDepth; depth++)
  {
    // Trace the ray through the scene
    raytracer.Trace(ray, payload, seed, depth);

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
        sampleResult.radiance.a = 0.0;  // Set it to transparent

        // #DLSS - Set environment hit position for proper motion vectors
        sampleResult.dlssOutput.hitPosition = ray.Origin + ray.Direction * 1000000.0f;

        // Solid color background and blurred HDR environment, aren't part of the
        // lighting equation (backplate), so we can return them directly.
        if(frameInfo->useSolidBackground == 1)
        {
          sampleResult.radiance.rgb = frameInfo->backgroundColor;
          return sampleResult;
        }
        else if(pushConst.frameInfo->environmentType == EnvSystem::eHdr && pushConst.frameInfo->envBlur > 0)
        {
          float3 dir = rotate(ray.Direction, float3(0, 1, 0), -frameInfo.envRotation);
          float2 uv  = getSphericalUv(dir);  // See sampling.glsl
          sampleResult.radiance.xyz =
              smoothHDRBlur(texturesHdr[HDR_IMAGE_INDEX], uv, frameInfo->envBlur).xyz * frameInfo->envIntensity;
          return sampleResult;
        }
      }

      // Add sky or HDR texture
      float3 envColor;
      float  envPdf;
      if(frameInfo->environmentType == EnvSystem::eSky)
      {
        envColor = evalPhysicalSky(*pushConst.skyParams, ray.Direction);
        envPdf   = samplePhysicalSkyPDF(*pushConst.skyParams, ray.Direction);
      }
      else
      {
        // Adding HDR lookup
        float3 dir = rotate(ray.Direction, float3(0, 1, 0), -frameInfo.envRotation);
        float2 uv  = getSphericalUv(dir);  // See sampling.glsl
        float4 env = texturesHdr[HDR_IMAGE_INDEX].SampleLevel(uv, 0);
        envColor   = env.rgb * frameInfo.envIntensity;
        envPdf     = env.w;
      }

      // We may hit the environment twice: once via sampleLights() and once when hitting the sky while probing
      // for more indirect hits. This is the counter part of the MIS weighting in sampleLights()
      float misWeight = (lastSamplePdf == DIRAC) ? 1.0 : (lastSamplePdf / (lastSamplePdf + envPdf));
      radiance += throughput * misWeight * envColor;

      sampleResult.radiance.xyz = radiance;
      return sampleResult;
    }


    PbrMaterial       pbrMat;
    GltfShadeMaterial material;
    if(hitInfinitePlane)
    {
      // Evaluate the material at the hit point
      pbrMat   = defaultPbrMaterial(frameInfo.infinitePlaneBaseColor, frameInfo.infinitePlaneMetallic,
                                    frameInfo.infinitePlaneRoughness, hit.nrm, hit.nrm);
      material = defaultGltfMaterial();
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

    // #DLSS - Gather data from first hit
    if(firstRay)
    {
      sampleResult.dlssOutput.albedo = float4(pbrMat.baseColor.xyz, 1.0f);
      sampleResult.dlssOutput.specularAlbedo =
          EnvBRDFApprox2(pbrMat.specularColor, pbrMat.roughness.x, dot(pbrMat.N, ray.Direction));
      sampleResult.dlssOutput.normalRoughness = float4(pbrMat.N, pbrMat.roughness.x);
      sampleResult.dlssOutput.hitPosition     = ray.Origin + ray.Direction * payload.hitT;
    }


    // Keep track of the maximum roughness to prevent firefly artifacts
    // by forcing subsequent bounces to be at least as rough
    maxRoughness     = max(pbrMat.roughness, maxRoughness);
    pbrMat.roughness = maxRoughness;

    // Debugging, single frame
    if(frameInfo.debugMethod != DebugMethod::eNone && firstRay)
    {
      sampleResult.radiance.xyz = debugValue(pbrMat, hit, frameInfo.debugMethod);
      sampleResult.radiance.a   = 1.0;
      return sampleResult;
    }


    // Adding emissive
    radiance += pbrMat.emissive * throughput;

    // Unlit
    if(material.unlit > 0)
    {
      radiance += pbrMat.baseColor;
      sampleResult.radiance.xyz = radiance;
      sampleResult.radiance.a   = 1.0;
      return sampleResult;
    }

    // Apply volume attenuation
    if(isInside && !pbrMat.isThinWalled)
    {
      const float3 abs_coeff = absorptionCoefficient(pbrMat);
      throughput *= exp(-payload.hitT * abs_coeff);
    }


    float3 contribution = float3(0);  // Direct lighting contribution

    // Light contribution; can be environment or punctual lights
    DirectLight directLight;
    sampleLights(hit.pos, pbrMat.N, ray.Direction, seed, directLight);

    // Do not next event estimation (but delay the adding of contribution)
    bool nextEventValid = (dot(directLight.direction, hit.geonrm) > 0.0f || pbrMat.diffuseTransmissionFactor > 0.0f)
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

      // Update the throughput
      throughput *= sampleData.bsdf_over_pdf;
      ray.Direction = sampleData.k2;  // new direction
      lastSamplePdf = sampleData.pdf;

      // If the ray is absorbed, then break
      if(sampleData.event_type == BSDF_EVENT_ABSORB)
      {
        // Exit tracing rays, but still finish this iteration; in particular the visibility test
        // for the light that we may have hit.
        depth = pushConst.maxDepth;
      }
      else
      {
        // Continue path
        bool isSpecular     = (sampleData.event_type & BSDF_EVENT_IMPULSE) != 0;
        bool isTransmission = (sampleData.event_type & BSDF_EVENT_TRANSMISSION) != 0;

        float3 offsetDir = dot(ray.Direction, hit.geonrm) > 0 ? hit.geonrm : -hit.geonrm;
        ray.Origin       = offsetRay(hit.pos, offsetDir);

        // Flip the information if we are inside the object, but only if it is a solid object
        // The doubleSided flag is used to know if the object is solid or thin-walled.
        // This is not a glTF specification, but works in many cases.
        if(isTransmission)
        {
          isInside = !isInside;
        }
      }
    }

    // We are adding the contribution to the radiance only if the ray is not occluded by an object.
    if(nextEventValid)
    {
      // shadow origin is the hit position offset by a small amount in the direction of the light
      float3 shadowRayOrigin = offsetRay(hit.pos, (dot(directLight.direction, hit.geonrm) > 0.0f) ? hit.geonrm : -hit.geonrm);
      RayDesc shadowRay    = RayDesc(shadowRayOrigin, 0, directLight.direction, directLight.distance);
      float3  shadowFactor = raytracer.TraceShadow(shadowRay, seed);
      radiance += contribution * shadowFactor;
    }

    // Russian-Roulette (minimizing live state)
    float rrPcont = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001F, 0.95F);
    if(rand(seed) >= rrPcont)
      break;                // paths with low throughput that won't contribute
    throughput /= rrPcont;  // boost the energy of the non-terminated paths
  }

  // Return the radiance
  sampleResult.radiance.xyz = radiance;
  return sampleResult;
}

//-----------------------------------------------------------------------
// Sampling the pixel
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
  RayDesc ray = getRay(samplePos, subpixelJitter, imageSize, projMatrixI, viewMatrixI);

  // Depth-of-Field
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
//-----------------------------------------------------------------------
void processPixel(IRaytracer raytracer, float2 samplePos, float2 imageSize)
{
  if(samplePos.x >= imageSize.x || samplePos.y >= imageSize.y)
    return;

  if(samplePos.x == pushConst.mouseCoord.x && samplePos.y == pushConst.mouseCoord.y)
  {
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
    subpixelJitter += ANTIALIASING_STANDARD_DEVIATION * sampleGaussian(float2(rand(seed), rand(seed)));

  // #DLSS - use the DLSS jitter and frame index (not resetting to zero)
  if(pushConst.useDlss == 1)
  {
    subpixelJitter = pushConst.jitter + float2(0.5f, 0.5f);
  }

  // Sampling n times the pixel
  SampleResult sampleResult = samplePixel(raytracer, seed, samplePos, subpixelJitter, imageSize, pushConst.frameInfo.projInv,
                                          pushConst.frameInfo.viewInv, pushConst.focalDistance, pushConst.aperture);
  float4 pixel_color = sampleResult.radiance;
  for(int s = 1; s < pushConst.numSamples; s++)
  {
    subpixelJitter = float2(rand(seed), rand(seed));
    sampleResult   = samplePixel(raytracer, seed, samplePos, subpixelJitter, imageSize, pushConst.frameInfo.projInv,
                                 pushConst.frameInfo.viewInv, pushConst.focalDistance, pushConst.aperture);
    pixel_color += sampleResult.radiance;
  }
  pixel_color /= pushConst.numSamples;

  bool first_frame = (pushConst.frameCount == 0);

  // Saving result
  if(first_frame || (pushConst.useDlss == 1))
  {  // First frame, replace the value in the buffer
    outImages[int(OutputImage::eResultImage)][int2(samplePos)] = pixel_color;
  }
  else
  {  // Do accumulation over time
    float  a                                                   = 1.0F / float(pushConst.frameCount + 1);
    float4 old_color                                           = outImages[0][int2(samplePos)];
    outImages[int(OutputImage::eResultImage)][int2(samplePos)] = lerp(old_color, pixel_color, a);
  }

  // #DLSS - Storing the GBuffer for the DLSS denoiser
  if(pushConst.useDlss == 1)
  {
    // Transform world position to clip space and calculate depth
    float4 posScreen = mul(float4(sampleResult.dlssOutput.hitPosition, 1.0), pushConst.frameInfo.viewProjMatrix);
    float  viewZ     = posScreen.z / posScreen.w;  // Depth in NDC space

    // Calculate motion vectors using the hit position (works for both geometry and environment)
    float2 motionVec = calculateMotionVector(sampleResult.dlssOutput.hitPosition, pushConst.frameInfo.prevMVP,
                                             pushConst.frameInfo.viewProjMatrix, imageSize);
    outImages[int(OutputImage::eDlssDepth)][int2(samplePos)]           = float4(abs(viewZ));
    outImages[int(OutputImage::eDlssMotion)][int2(samplePos)]          = float4(motionVec, 0, 0);
    outImages[int(OutputImage::eDlssNormalRoughness)][int2(samplePos)] = sampleResult.dlssOutput.normalRoughness;
    outImages[int(OutputImage::eDlssAlbedo)][int2(samplePos)]          = sampleResult.dlssOutput.albedo;
    outImages[int(OutputImage::eDlssSpecAlbedo)][int2(samplePos)] = float4(sampleResult.dlssOutput.specularAlbedo.xyz, 1.0f);
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

  HitState hit = getHitState(renderPrim, barycentrics, worldToObject, objectToWorld, primitiveID, worldRayOrigin);

  payload.hitT     = hitT;
  payload.rprimID  = renderPrimID;
  payload.rnodeID  = instanceID;
  payload.hitState = hit;
}

[shader("closesthit")]
void rchitShadow(inout ShadowPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
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
  payload.hitT              = INFINITE;
  payload.totalTransmission = float3(1.0);
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
    float  currentHitT  = abs(hitT - payload.hitT);
    bool   isInside     = payload.isInside;
    float3 transmission = getShadowTransmission(renderNode, renderPrim, primitiveID, barycentrics, currentHitT,
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
