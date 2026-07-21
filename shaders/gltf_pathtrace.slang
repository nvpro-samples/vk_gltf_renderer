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

#ifndef WIREFRAME
#define WIREFRAME 0
#endif

// This is a debug flag to enable/disable debug output
// For example, to use print statements in the shader, set this to true.
// Add printf and enable the Validation layer debug output.
static bool doDebug = false;

#include "gltf_material_eval.h.slang"
#include "gltf_scene_io.h.slang"
#include "gltf_vertex_access.h.slang"
#include "shaderio.h"
#include "get_hit.h.slang"
#include "dlss_util.h"
#include "common.h.slang"

// NVSHADERS are under: https://github.com/nvpro-samples/nvpro_core2/tree/main/nvshaders
#include "nvshaders/functions.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/sky_functions.h.slang"
#include "nvshaders/bsdf_types.h.slang"
#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/hdr_env_sampling.h.slang"
#include "nvshaders/light_contrib.h.slang"
#include "nvshaders/normal_compress.h.slang"
#include "nvshaders/pbr_ggx_microfacet.h.slang"
#include "nvshaders/ray_utils.h.slang"
#include "nvshaders/sample_blur.h.slang"
#if WIREFRAME
#include "nvshaders/wireframe.h.slang"
#endif


// clang-format off

// Bindings
[[vk::push_constant]]                           ConstantBuffer<PathtracePushConstant> pushConst;
[[vk::binding(BindingPoints::eTextures, 0)]]    Sampler2D allTextures[];
[[vk::binding(BindingPoints::eTexturesHdr, 0)]] Sampler2D texturesHdr[];
[[vk::binding(BindingPoints::eTlas, 1)]]        RaytracingAccelerationStructure topLevelAS;
[[vk::binding(BindingPoints::eOutImages, 1)]]   RWTexture2D<float4> outImages[];
[[vk::binding(BindingPoints::eOutDepth, 1)]]    RWTexture2D<float>  outDepth;

// HDR Environment
[[vk::binding(EnvBindings::eImpSamples, 2)]]    StructuredBuffer<EnvAccel> envSamplingData;

// Specialization constants
[[vk::constant_id(0)]]  int USE_SER = 1;
[[vk::constant_id(1)]]  int USE_DLSS_TRANSP = 0; // Primary surface selection mode

// clang-format on

#include "pathtrace_functions.h.slang"

//-----------------------------------------------------------------------
// PATH TRACE ONE BOUNCE
//
// This function performs a single bounce of path tracing in the renderer. It traces a ray through the scene,
// handles intersection and shading logic for the first or subsequent bounces, and updates path tracing state
// and bounce scratch data accordingly. It manages ray-scene intersections, material retrieval, handling
// of missed rays (e.g., environment hits), and prepares data for direct/indirect lighting and next events.
//-----------------------------------------------------------------------
PathStepResult pathTraceOneBounce(IRaytracer raytracer, inout RayDesc ray, inout uint seed, inout PathTracerState pt, out BounceScratch bounce)
{
  // Initialize the bounce scratch data
  bounce.nextEventValid = false;
  bounce.contribution   = float3(0);
  bounce.shadowRayPos   = float3(0);
  bounce.shadowRayDir   = float3(0);
  bounce.shadowRayDist  = 0.0f;

  {
    //DirectLight directLight;
    SceneFrameInfo* frameInfo = pushConst.frameInfo;
    HitPayload      payload   = {};

    // Trace the ray through the scene
    raytracer.Trace(ray, payload, seed);

    // Getting the scene information
    GltfRenderNode*      renderNodes     = pushConst.gltfScene->renderNodes;
    GltfRenderPrimitive* renderPrimitive = pushConst.gltfScene->renderPrimitives;

    // If we hit something, get the hit state
    HitState            hit        = {};
    GltfRenderNode      renderNode = {};
    GltfRenderPrimitive renderPrim = {};
    if(payload.hitT != INFINITE)
    {
      renderNode   = renderNodes[payload.rnodeID];
      renderPrim   = renderPrimitive[payload.rprimID];
      float3 barys = float3(1.0 - payload.bary.x - payload.bary.y, payload.bary.x, payload.bary.y);
      hit = getHitState(renderPrim, barys, renderNode.worldToObject, renderNode.objectToWorld, payload.primitiveID, ray.Direction);
    }

    // Check if we are on the first ray
    bool firstRay = (pt.surfaceDepth == 0);

    // Check if we hit the infinite plane
    bool hitInfinitePlane = checkInfinitePlaneIntersection(ray, payload, hit, frameInfo);

    //-----------------------------------------------------------------------
    // Hitting the environment
    //-----------------------------------------------------------------------
    if(payload.hitT == INFINITE)
    {
      // Check if we hit the backplate
      if(firstRay && tryPrimaryMissBackplate(ray, frameInfo, pt))
        return PathStepResult.eBreak;

#if USE_DLSS_SHADER
      // #DLSS - If capturing specular hit and we hit environment, use max distance (65504 = fp16 max)
      // This happens when a glossy reflection bounces into the sky.
      if(pt.guide.captureSpecularHit)
      {
        pt.guide.specularHitDist    = float16_t(65504.0f);
        pt.guide.captureSpecularHit = false;
      }
#endif  // USE_DLSS_SHADER

      // Add sky or HDR texture using refactored helper
      float3 envColor;
      float  envPdf;
      sampleEnvironment(ray.Direction, frameInfo, envColor, envPdf);

      // We may hit the environment twice: once via sampleLights() and once when hitting the sky while probing
      // for more indirect hits. This is the counter part of the MIS weighting in sampleLights()
      float misWeight = computeEnvHitMisWeight(pt.lastSamplePdf, envPdf);
      pt.radiance += pt.throughput * misWeight * envColor;

      return PathStepResult.eBreak;
    }

    GltfShadeMaterial* materials     = pushConst.gltfScene->materials;
    GltfTextureInfo*   texInfos      = pushConst.gltfScene->textureInfos;
    int                materialIndex = -1;
    PbrMaterial        pbrMat;

    // Compute the ray-cone footprint at the hit point which is used to compute the texture-LOD gradient
    float worldFoot = rayConeWorldFootprint(pt.cone, payload.hitT, hit.geonrm, -ray.Direction);

    //-----------------------------------------------------------------------
    // Hitting the infinite plane
    //-----------------------------------------------------------------------
    if(hitInfinitePlane)
    {
      // Replaced the material at the hit point with the infinite plane material
      pbrMat = defaultPbrMaterial(frameInfo->infinitePlaneBaseColor, frameInfo->infinitePlaneMetallic,
                                  frameInfo->infinitePlaneRoughness, hit.nrm, hit.nrm);

      // Shadow catcher: invisible plane that shows only shadows for compositing
      if(hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneInfinitePlaneShadowCatcher))
      {
        // Propagate cone through plane bounce before potential early continue from shadow-catcher path.
        pt.cone.width = worldFoot;
        bool continuePath =
            handleShadowCatcher(raytracer, hit, pbrMat, ray, pt.radiance, pt.throughput, pt.lastSamplePdf, seed, frameInfo);
        if(!continuePath)
          return PathStepResult.eBreak;  // Catch the shadow
        return PathStepResult.eEarlyContinue;  // Continue the path tracing from the infinite plane, missed the shadow catcher path
      }
    }
    //-----------------------------------------------------------------------
    // Hitting the mesh
    //-----------------------------------------------------------------------
    else
    {
      // Find material at the hit point and evaluate the material (PBR material evaluation)

      materialIndex = max(0, renderNode.materialID);

      float     texGrad = worldFoot * hit.texelDensity * pushConst.texGradScale;
      MeshState mesh    = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, ptIsInside(pt), texGrad);
      mesh.baseColorVertexMul = hit.color;
      pbrMat                  = evaluateMaterial(materialIndex, mesh, allTextures, texInfos, materials);

#if WIREFRAME
      // Wireframe is only used for the camera-visible surface, so compute it once here on the
      // first hit (instead of every Trace() like the old payload-routed version did). All inputs
      // are already in scope, so this avoids carrying wireframeAlpha + pixelAngle in the payload.
      if(firstRay)
      {
        uint3 triIdx = getTriangleIndices(renderPrim, payload.primitiveID);
        pt.firstHitWireframeAlpha =
            computeMeshWireframe(renderNode.objectToWorld, getVertexPosition(renderPrim, triIdx.x),
                                 getVertexPosition(renderPrim, triIdx.y), getVertexPosition(renderPrim, triIdx.z),
                                 payload.bary, payload.hitT, pushConst.pixelAngle, hit.geonrm, -ray.Direction);
      }
#endif
    }

#if USE_DLSS_SHADER
    // #DLSS - Capture specular hit distance on depth 1 (first reflection bounce)
    if(pt.guide.captureSpecularHit)
    {
      pt.guide.specularHitDist    = float16_t(payload.hitT);
      pt.guide.captureSpecularHit = false;
    }
#endif  // USE_DLSS_SHADER

    //-----------------------------------------------------------------------
    // First-hit captures (NDC-depth always; DLSS/OptiX guide buffers when on).
    //-----------------------------------------------------------------------
    if(firstRay)
    {
      // Always record the first hit position; drives the NDC depth output used by
      // selection/picking, independent of DLSS/OptiX.
      pt.firstHitPos = hit.pos;

#if USE_GUIDE_SHADER
      // #DLSS instance motion - Previous-frame world position of this surface point.
      pt.firstHitPosPrev = hit.pos;
      if(!hitInfinitePlane && pushConst.prevRenderNodeObjectToWorld != nullptr)
      {
        float3 objPos      = mul(float4(hit.pos, 1.0), renderNode.worldToObject).xyz;
        pt.firstHitPosPrev = mul(float4(objPos, 1.0), pushConst.prevRenderNodeObjectToWorld[payload.rnodeID]).xyz;
      }

      // Base-color guide albedo (shared by DLSS and OptiX). Stored temporarily, written at end to
      // reduce live state.
      float3 guideAlbedo = pbrMat.baseColor.xyz;
#if USE_DLSS_SHADER
      // #DLSS specular guide - specular albedo + clear-glass albedo refinement. DLSS-only quality
      // path; OptiX needs neither, so it is compiled out of the OptiX/no-denoiser variant.
      // Apply KHR_materials_specular energy conservation clamping for DLSS guide buffer
      // Spec formula: dielectric_f0 = min(f0_from_ior * specularColor, 1.0) * specular
      float  f0_from_ior          = pow((pbrMat.ior2 - pbrMat.ior1) / (pbrMat.ior2 + pbrMat.ior1), 2.0F);
      float3 specularColorClamped = min(f0_from_ior * pbrMat.specularColor, float3(1.0F));
      float3 specularAlbedo = EnvBRDFApprox2(specularColorClamped, pbrMat.roughness.x, dot(pbrMat.N, ray.Direction));
      if(USE_DLSS_TRANSP == 1 && !hasFlag(pushConst.flags, PathtracerFlags::ePtUseOptixDenoiser) && dlssIsClearGlass(pbrMat))
      {
        guideAlbedo = dlssGetClearGlassGuideAlbedo(raytracer, pbrMat, hit, ray.Direction, frameInfo, seed,
                                                   ptIsInside(pt), specularAlbedo);
      }
      pt.guide.specularAlbedo = float16_t3(specularAlbedo);
#endif  // USE_DLSS_SHADER
      pt.guide.normalRoughness = float16_t4(float16_t3(pbrMat.N), float16_t(pbrMat.roughness.x));
      pt.guide.albedo          = float16_t3(guideAlbedo);
#endif  // USE_GUIDE_SHADER
    }

    // Keep track of the maximum roughness to prevent firefly artifacts by forcing subsequent bounces to be at least as rough
    pt.maxRoughness  = max(pbrMat.roughness, pt.maxRoughness);
    pbrMat.roughness = pt.maxRoughness;

    //-----------------------------------------------------------------------
    // Visualization override (Debug mode)
    // Note: clay visualization mode continues shading; flat-color modes break
    //-----------------------------------------------------------------------
    // Debug-only; compiled out of the normal render path (USE_VISUALIZE, set by the host when the
    // visualization mode is not eRendered).
#if USE_VISUALIZE
    if(frameInfo->visualization != Visualization::eRendered)
    {
      float3              vizColor;
      VisualizationResult vizResult = applyVisualization(pbrMat, hit, frameInfo->visualization, payload.rprimID,
                                                         payload.primitiveID, payload.ommUnknown ? 1 : 0, vizColor);
      if(vizResult == VisualizationResult::eColorOverride && firstRay)
      {
        pt.radiance.xyz = vizColor;
        return PathStepResult.eBreak;
      }
    }
#endif

    //-----------------------------------------------------------------------
    // Adding emissive
    //-----------------------------------------------------------------------
    pt.radiance += pbrMat.emissive * pt.throughput;

    //-----------------------------------------------------------------------
    // Unlit
    //-----------------------------------------------------------------------
#if GLTF_USE_UNLIT
    if(materialIndex >= 0 && materials[materialIndex].unlit > 0)
    {
      pt.radiance += pbrMat.baseColor;
      return PathStepResult.eBreak;
    }
#endif

    PathStepResult volumeStep = processVolumeSegment(raytracer, payload.hitT, ray, pt, seed);
    if(volumeStep != PathStepResult.eOutOfVolume)
      return volumeStep;  // Skip the surface bounce if the ray is still in the volume (ray is scattered)

    // Normal surface bounce: propagate the cone footprint to the next segment. The new ray
    // starts at hit.pos with the slant-corrected worldFoot as its starting width, so the
    // next Trace()'s texture LOD inherits the full accumulated footprint.
    pt.cone.width = worldFoot;

    //-----------------------------------------------------------------------
    // Sampling the lights
    // - Get the light contribution from one light source or the HDR environment
    //-----------------------------------------------------------------------
    DirectLight directLight;
    sampleLights(hit.pos, pbrMat.N, ray.Direction, seed, directLight);

    // This checks if the light could contribute, but doesn't send a next event estimation (NEE) shadow ray yet.
    // The actual test and contribution are delayed to minimize per-bounce state.
    bounce.nextEventValid = (dot(directLight.direction, hit.nrm) > 0.0f || pbrMat.diffuseTransmissionFactor > 0.0f)
                            && directLight.pdf != 0.0f;

    //-----------------------------------------------------------------------
    // Evaluating the BSDF for the light
    //-----------------------------------------------------------------------
    if(bounce.nextEventValid)
    {
      // Evaluating the BSDF at the hit point for the light
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
        const float3 w = pt.throughput * directLight.radianceOverPdf * mis_weight;
        bounce.contribution += w * evalData.bsdf_diffuse;
        bounce.contribution += w * evalData.bsdf_glossy;
      }
    }

    //-----------------------------------------------------------------------
    // Sampling the BSDF :
    // - Finding the next direction and updating the throughput
    //-----------------------------------------------------------------------
    {
      // Sampling the BSDF at the hit point
      BsdfSampleData sampleData;
      sampleData.k1 = -ray.Direction;                              // outgoing direction
      sampleData.xi = float3(rand(seed), rand(seed), rand(seed));  // random number
      bsdfSample(sampleData, pbrMat);
      // bsdfSampleSimple(sampleData, pbrMat);

      // Update the pt.throughput
      pt.throughput *= sampleData.bsdf_over_pdf;
      ray.Direction    = sampleData.k2;  // new direction
      pt.lastSamplePdf = sampleData.pdf;

#if USE_DLSS_SHADER
      // #DLSS - Enable specular hit distance capture on first bounce if this is a glossy reflection
      // Matches reference implementation: only capture for BSDF_EVENT_GLOSSY_REFLECTION
      if(firstRay && (sampleData.event_type == BSDF_EVENT_GLOSSY_REFLECTION || sampleData.event_type == BSDF_EVENT_IMPULSE_REFLECTION))
      {
        pt.guide.captureSpecularHit = true;
      }
#endif  // USE_DLSS_SHADER

      // The ray wasn't absorbed, so we can continue the path
      if(sampleData.event_type != BSDF_EVENT_ABSORB)
      {
        // Continue path (bounce)
        bool isSpecular     = (sampleData.event_type & BSDF_EVENT_IMPULSE) != 0;
        bool isTransmission = (sampleData.event_type & BSDF_EVENT_TRANSMISSION) != 0;

        // Compute the offset position for the next bounce
        float3 offsetDir = dot(ray.Direction, hit.geonrm) > 0 ? hit.geonrm : -hit.geonrm;
        ray.Origin       = safeOffsetRay(hit.pos, offsetDir);

        // Toggle 'pt.isInside' to track whether the ray is currently inside or outside geometry.
        // This update is necessary for correctly handling transmission (refraction) through solid (non-thin-walled) materials.
        // For thin-walled surfaces - like decals or leaves - this state isn't toggled, because there's no real "inside".
        //
        // Gated on GLTF_USE_TRANSMISSION: scenes without transmission never produce
        // BSDF_EVENT_TRANSMISSION, so this branch is dead. makeVolumeMedium only when
        // GLTF_USE_VOLUME is on (medium has nowhere to live otherwise).
#if GLTF_USE_TRANSMISSION
        if(isTransmission)
        {
          pt.isInside = !pt.isInside;
#if GLTF_USE_VOLUME
          if(pt.isInside)
            pt.volumeMedium = makeVolumeMedium(pbrMat);  // Entering volume: capture the medium for scatter/absorption
#endif
        }
#endif
      }
      else if(sampleData.event_type == BSDF_EVENT_ABSORB)  // If the ray is absorbed , then break
      {
        // Exit tracing rays, but still finish this iteration; in particular the visibility test
        // for the light that we may have hit.
        pt.surfaceDepth = pushConst.maxDepth;
      }

      // End scope to free live states
    }

    //-----------------------------------------------------------------------
    // Finding the shadow ray position and direction
    //-----------------------------------------------------------------------
    bool   shadowSideForward = dot(directLight.direction, hit.nrm) > 0.0F;
    float3 shadowOffsetDir   = shadowSideForward ? hit.geonrm : -hit.geonrm;
    float3 shadowOffsetBase  = shadowSideForward ? hit.shadowPos : hit.pos;
    bounce.shadowRayPos      = safeOffsetRay(shadowOffsetBase, shadowOffsetDir);
    bounce.shadowRayDir      = directLight.direction;
    bounce.shadowRayDist     = directLight.distance;
  }

  return PathStepResult.eOutOfVolume;
}

//-----------------------------------------------------------------------
// PATH TRACE
//
// This function performs the main path tracing loop. It initializes the path tracer state,
// traces a ray through the scene, handles intersection and shading logic for the first or subsequent bounces,
// and updates path tracing state and bounce scratch data accordingly. It manages ray-scene intersections,
// material retrieval, handling of missed rays (e.g., environment hits),
// and prepares data for direct/indirect lighting and next events.
//-----------------------------------------------------------------------
SampleResult pathTrace(IRaytracer raytracer, RayDesc ray, inout uint seed)
{
  PathTracerState pt = {};

  while(pt.surfaceDepth < pushConst.maxDepth)
  {
    ray.Direction = normalize(ray.Direction);

    //-----------------------------------------------------------------------
    // Tracing the ray and finding the radiance, the throughput and the next event valid flag
    //-----------------------------------------------------------------------
    BounceScratch  bounce;
    PathStepResult step = pathTraceOneBounce(raytracer, ray, seed, pt, bounce);
    if(step == PathStepResult.eBreak)
      break;
    if(step == PathStepResult.eVolumeContinue || step == PathStepResult.eEarlyContinue)
      continue;

    //-----------------------------------------------------------------------
    // Tracing the shadow ray
    //-----------------------------------------------------------------------
    if(bounce.nextEventValid)
    {
      RayDesc shadowRay;
      shadowRay.Origin    = bounce.shadowRayPos;
      shadowRay.Direction = bounce.shadowRayDir;
      shadowRay.TMin      = 0.0;
      shadowRay.TMax      = bounce.shadowRayDist;
      float3 shadowFactor = raytracer.TraceShadow(shadowRay, seed);
      pt.radiance += bounce.contribution * shadowFactor;
    }

    //-----------------------------------------------------------------------
    // Russian Roulette
    //-----------------------------------------------------------------------
    if(pt.surfaceDepth >= RR_MIN_DEPTH)
    {
      float rrPcont = min(max(pt.throughput.x, max(pt.throughput.y, pt.throughput.z)) + 0.001F, 0.95F);
      if(rand(seed) >= rrPcont)
        break;
      pt.throughput /= rrPcont;
    }

    // Consume one surface bounce
    pt.surfaceDepth++;
  }

#if WIREFRAME
  if(pt.firstHitWireframeAlpha > 0.0)
    pt.radiance = lerp(pt.radiance, float3(0.9, 0.0, 0.0), pt.firstHitWireframeAlpha);
#endif

  return packSampleResult(pt);
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
  const bool isOrthographic = hasFlag(pushConst.frameInfo->flags, SceneFrameInfoFlags::eSceneIsOrthographic);
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
// Per-pixel accumulation and GBuffer output
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

  // Initialize the random number
  uint seed = xxhash32(uint3(uint2(samplePos.xy), pushConst.frameCount));

  bool firstFrame = hasFlag(pushConst.flags, PathtracerFlags::ePtFirstFrame);

  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  // If DLSS is used, the jitter is on the entire frame, not just the pixel.
  float2 subpixelJitter = float2(0.5f, 0.5f);

  // #DLSS - use the DLSS jitter and frame index (not resetting to zero)
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
  {
    subpixelJitter += pushConst.frameInfo.jitter;
  }
  else
  {
    // Add the jitter to the subpixel jitter
    subpixelJitter += ANTIALIASING_STANDARD_DEVIATION * sampleGaussian(float2(rand(seed), rand(seed)));
  }

  // Sampling n times the pixel
  SampleResult sampleResult = samplePixel(raytracer, seed, samplePos, subpixelJitter, imageSize, pushConst.frameInfo->projInv,
                                          pushConst.frameInfo->viewInv, pushConst.focalDistance, pushConst.aperture);
  float4 pixelColor = sampleResult.radiance;

  // DLSS uses only one sample per pixel (it does its own temporal accumulation), so the
  // multi-sample loop is compiled out in the DLSS variant to reduce register pressure.
  // USE_DLSS_SHADER is true only when DLSS is actually active; for the OptiX and no-denoiser
  // paths it is 0, so the loop is compiled in and honors the user's samples-per-pixel setting.
#if !USE_DLSS_SHADER
  for(int s = 1; s < pushConst.numSamples; s++)
  {
    subpixelJitter = float2(rand(seed), rand(seed));
    sampleResult   = samplePixel(raytracer, seed, samplePos, subpixelJitter, imageSize, pushConst.frameInfo->projInv,
                                 pushConst.frameInfo->viewInv, pushConst.focalDistance, pushConst.aperture);
    pixelColor += sampleResult.radiance;
  }
  pixelColor /= pushConst.numSamples;
#endif  // !USE_DLSS_SHADER


  // NDC depth from first-hit world position (Vulkan [0,1] range, 1.0 = far/miss).
  // Written to two targets: outDepth (main GBuffer, full viewport resolution, frame 0 only)
  // and eDlssDepth (DLSS guide buffer, DLSS render resolution, every frame when DLSS active).
  // Both use the same NDC value but the images differ in resolution.
  const bool hasSolidHit = sampleResult.radiance.w > 0.0f;
  float4     clipPos =
      hasSolidHit ? mul(float4(sampleResult.hitPosition, 1.0), pushConst.frameInfo->viewProjMatrix) : float4(0, 0, 1, 1);
  float ndcDepth = hasSolidHit ? (clipPos.z / clipPos.w) : 1.0f;

  // Only on the first frame, trace the selection ray to get the object IDs and save the NDC depth
  if(firstFrame)
  {
    // Trace the selection ray to get the object IDs
    uint objectID                                              = traceSelectionRay(raytracer, samplePos, imageSize);
    outImages[int(OutputImage::eSelectImage)][int2(samplePos)] = float4(asfloat(objectID), 0.0, 0.0, 0.0);
    outDepth[int2(samplePos)]                                  = ndcDepth;
  }

  // Saving result
  if(firstFrame || hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
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

#if USE_DLSS_SHADER
  // #DLSS - Storing the GBuffer for the DLSS denoiser (DLSS-only variant; the specular guide targets
  // do not exist in the OptiX/no-denoiser build).
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
  {
    // Combined camera + instance motion. w selects the projection model: surface hits use w=1 (the
    // previous position carries object motion); sky/background uses w=0, i.e. a point at infinity along
    // the view direction, so camera translation cancels and only rotation moves the sky (no far-plane
    // magic). sampleResult.hitPosition holds a world position for surfaces and a view direction for sky.
    float hitW = hasSolidHit ? 1.0f : 0.0f;
    float2 motionVec = calculateMotionVector(float4(sampleResult.hitPosition, hitW), float4(sampleResult.hitPositionPrev, hitW),
                                             pushConst.frameInfo->prevMVP, pushConst.frameInfo->viewProjMatrix, imageSize);
    outImages[int(OutputImage::eDlssDepth)][int2(samplePos)]           = float4(ndcDepth);
    outImages[int(OutputImage::eDlssMotion)][int2(samplePos)]          = float4(motionVec, 0, 0);
    outImages[int(OutputImage::eDlssNormalRoughness)][int2(samplePos)] = sampleResult.guideOutput.normalRoughness;
    outImages[int(OutputImage::eDlssAlbedo)][int2(samplePos)]          = sampleResult.guideOutput.albedo;
    outImages[int(OutputImage::eDlssSpecAlbedo)][int2(samplePos)] = float4(sampleResult.guideOutput.specularAlbedo.xyz, 1.0f);
    outImages[int(OutputImage::eDlssSpecularHitDist)][int2(samplePos)] = float4(sampleResult.guideOutput.specularHitDist);
  }
#endif  // USE_DLSS_SHADER

#if USE_GUIDE_SHADER
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtUseOptixDenoiser))
  {
    // Transform normal from world space to camera space for OptiX denoiser
    // OptiX requires normals in camera space (right-handed, -Z forward, Y up, X right).
    // On primary misses the guide normal is zero, so fall back to a valid forward-facing
    // normal to avoid normalizing/compressing a degenerate vector.
    float3 cameraNormal = float3(0.0f, 0.0f, 1.0f);
    if(hasSolidHit)
    {
      float3 worldNormal = sampleResult.guideOutput.normalRoughness.xyz;
      cameraNormal       = normalize(mul(float3x3(pushConst.frameInfo->viewMatrix), worldNormal));
    }
    uint normalEncoded = compressUnitVec(cameraNormal);
    outImages[int(OutputImage::eOptixAlbedoNormal)][int2(samplePos)] =
        float4(sampleResult.guideOutput.albedo.xyz, asfloat(normalEncoded));
  }
#endif  // USE_GUIDE_SHADER
}

//-----------------------------------------------------------------------
// RAY-QUERY COMPUTE
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
  payload.hitT        = RayTCurrent();
  payload.rprimID     = InstanceID();
  payload.rnodeID     = InstanceIndex();
  payload.primitiveID = PrimitiveIndex();
  payload.bary        = attr.barycentrics;
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
  // Keep the accumulated transmission -- don't reset it. totalTransmission already contains
  // the accumulated value from rahitShadow.
}

//-----------------------------------------------------------------------
// INTERSECTION (Any Hit)
//-----------------------------------------------------------------------
[shader("anyhit")]
void rahitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
#if USE_VISUALIZE
  // Opacity-micromap debug view: any-hit only runs for "unknown" micro-triangles (OMM-opaque
  // ones are committed by the hardware without any-hit, transparent ones are culled). Flag it and
  // accept the hit so it is colored as unknown; the path breaks at the first hit anyway.
  if(pushConst.frameInfo->visualization == Visualization::eOpacityMicromap)
  {
    payload.ommUnknown = true;
    AcceptHitAndEndSearch();
  }
#endif

  float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

  uint instanceID   = InstanceIndex();
  uint renderPrimID = InstanceID();
  uint triangleID   = PrimitiveIndex();

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
    // Segment length for Beer's law inside getShadowTransmission (only used when isInside == true).
    // abs() guards against any-hit invocations arriving out of spatial order, which Vulkan/DXR
    // do not guarantee. This is an approximation; a fully correct version would chain
    // closest-hit traces along the ray instead of accumulating in any-hit.
    float segmentLength = abs(hitT - payload.prevHitT);

    bool   isInside     = payload.isInside;
    float3 transmission = getShadowTransmission(renderNode, renderPrim, primitiveID, barycentrics, segmentLength,
                                                worldToObject, worldRayDir, isInside);

    payload.isInside = isInside;
    payload.prevHitT = hitT;  // keep the actual previous hit T, not the segment length
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
