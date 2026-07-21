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

#include "get_hit.h.slang"
#include "shaderio.h"

// NVSHADERS are under: https://github.com/nvpro-samples/nvpro_core2/tree/main/nvshaders
#include "nvshaders/constants.h.slang"
#include "nvshaders/random.h.slang"

// Debug-visualization gate. Set to 1 by the host (see PathTracer::compileShader) when a
// visualization mode other than eRendered is selected; 0 in the shipping render path. Gates the
// OMM debug payload field and the visualization branches so they cost nothing in normal rendering.
// Default here so the embedded build (which does not pass the macro) compiles it out cleanly.
#ifndef USE_VISUALIZE
#define USE_VISUALIZE 0
#endif

// Payload for the path tracer
struct HitPayload
{
  uint   seed;
  float  hitT        = 0.0f;
  int    rnodeID     = -1;
  int    rprimID     = -1;
  int    primitiveID = -1;
  float2 bary        = float2(0);  // BuiltInTriangleIntersectionAttributes.barycentrics; third = 1 - x - y
#if USE_VISUALIZE
  bool ommUnknown = false;  // Opacity-micromap debug view: set when any-hit ran (micro-triangle is "unknown")
#endif
};

// Shadow payload for the path tracer
struct ShadowPayload
{
  uint seed;
  float prevHitT = 0.0f;  // The previous accepted any-hit's absolute RayT. rahitShadow takes abs(hitT - prevHitT) as the in-volume segment length for Beer's-law attenuation between consecutive transmissive hits. Spatial ordering of any-hit invocations is not guaranteed; abs() is an approximation that works well for thin/convex volumes.
  bool   isInside          = false;         // If true, the shadow ray is considered to start INSIDE a volume
  float3 totalTransmission = float3(1.0f);  // The total transmission of the shadow ray through the volume
};

// Raytracer interface definition
interface IRaytracer
{
  void   Trace(RayDesc ray, inout HitPayload payload, inout uint seed);
  void   TraceLow(RayDesc ray, inout HitPayload payload);
  float3 TraceShadow(RayDesc ray, inout uint seed, bool initialInside = false);
}

// Implementation using RayQuery
struct RayQueryRaytracer : IRaytracer
{
  void Trace(RayDesc ray, inout HitPayload payload, inout uint seed)
  {
    payload.hitT = INFINITE;  // Default when not hitting anything
#if USE_VISUALIZE
    payload.ommUnknown = false;  // Clear stale debug state; the OMM viz re-sets it below per ray
#endif

    // Back-face culling implements glTF single-sidedness: the ray culls back faces by default,
    // and SceneRtx sets VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR on double-sided
    // (and transmissive/volume) instances to re-enable them (see gltf_scene_rtx.cpp).
    RayQuery rayQuery;
    rayQuery.TraceRayInline(topLevelAS, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, ray);

    while(rayQuery.Proceed())
    {
      int    instanceID   = rayQuery.CandidateInstanceIndex();
      int    renderPrimID = rayQuery.CandidateInstanceID();
      int    triangleID   = rayQuery.CandidatePrimitiveIndex();
      float2 bary         = rayQuery.CandidateTriangleBarycentrics();

      // Barycentric coordinate on the triangle
      const float3 barycentrics = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

#if USE_VISUALIZE
      // Opacity-micromap debug view: reaching a candidate here means the micro-triangle is
      // "unknown" (OMM-opaque micro-triangles are committed without yielding a candidate,
      // transparent ones are culled). Commit it and stop, so it is colored as unknown.
      if(pushConst.frameInfo->visualization == Visualization::eOpacityMicromap)
      {
        payload.ommUnknown = true;
        rayQuery.CommitNonOpaqueTriangleHit();
        break;
      }
#endif

      GltfRenderNode      renderNode = pushConst.gltfScene->renderNodes[instanceID];
      GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

      float opacity = getOpacity(renderNode, renderPrim, triangleID, barycentrics);

      // do alpha blending the stochastically way
      if(rand(seed) <= opacity)
        rayQuery.CommitNonOpaqueTriangleHit();
    }

    if(rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
      payload.hitT        = rayQuery.CommittedRayT();
      payload.rprimID     = rayQuery.CommittedInstanceID();
      payload.rnodeID     = rayQuery.CommittedInstanceIndex();
      payload.primitiveID = rayQuery.CommittedPrimitiveIndex();
      payload.bary        = rayQuery.CommittedTriangleBarycentrics();
    }
  }

  void TraceLow(RayDesc ray, inout HitPayload payload)
  {
    payload.hitT = INFINITE;
    RayQuery rayQuery;
    rayQuery.TraceRayInline(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    while(rayQuery.Proceed())
    {
    }  // All geometry treated as opaque, nothing to process
    if(rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
      payload.hitT    = rayQuery.CommittedRayT();
      payload.rnodeID = rayQuery.CommittedInstanceIndex();
    }
  }

  float3 TraceShadow(RayDesc ray, inout uint seed, bool initialInside = false)
  {
    const float MIN_TRANSMISSION  = 0.01;  // Minimum transmission factor to continue tracing
    float3      totalTransmission = float3(1.0);
    bool        isInside          = initialInside;
    float       prevHitT          = 0;

    RayQuery rayQuery;
    rayQuery.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);

    while(rayQuery.Proceed())
    {
      float    hitT          = rayQuery.CandidateTriangleRayT();
      int      renderPrimID  = rayQuery.CandidateInstanceID();
      int      instanceID    = rayQuery.CandidateInstanceIndex();
      int      triangleID    = rayQuery.CandidatePrimitiveIndex();
      float2   bary          = rayQuery.CandidateTriangleBarycentrics();
      float4x3 worldToObject = rayQuery.CandidateWorldToObject4x3();

      GltfRenderNode      renderNode = pushConst.gltfScene->renderNodes[instanceID];
      GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

      float3 barycentrics = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
      float  opacity      = getOpacity(renderNode, renderPrim, triangleID, barycentrics);

      float r = rand(seed);
      if(r < opacity)
      {
        // RayQuery candidate hits arrive in spatial order: |segment| = hitT - prevHitT.
        float  segmentLength       = max(0.0F, hitT - prevHitT);
        float3 currentTransmission = getShadowTransmission(renderNode, renderPrim, triangleID, barycentrics,
                                                           segmentLength, worldToObject, ray.Direction, isInside);
        prevHitT                   = hitT;
        totalTransmission *= currentTransmission;

        if(max(max(totalTransmission.r, totalTransmission.g), totalTransmission.b) <= MIN_TRANSMISSION)
        {
          return float3(0.0);
        }
      }
    }

    if(rayQuery.CommittedStatus() != COMMITTED_NOTHING)
    {
      return float3(0., 0., 0.);
    }

    return totalTransmission;
  }
}

// Implementation using traditional ray tracing
struct TraditionalRaytracer : IRaytracer
{
  void Trace(RayDesc ray, inout HitPayload payload, inout uint seed)
  {
    payload.seed = seed;
#if USE_VISUALIZE
    payload.ommUnknown = false;  // Clear stale debug state; any-hit re-sets it for the OMM viz
#endif
    if(USE_SER == 1 && AVAILABLE_SER == 1)
    {
      HitObject hitObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 0, 0, ray, payload);

      // Simple default thread reordering, not coherence hinting
      ReorderThread(hitObj);

      // Miss everything, return early
      if(hitObj.IsMiss())
      {
        payload.hitT = INFINITE;
        seed         = payload.seed;
        return;
      }

      // There is a hit: record only the raw triangle hit (HitState is rebuilt in pathTrace).
      BuiltInTriangleIntersectionAttributes attr = hitObj.GetAttributes<BuiltInTriangleIntersectionAttributes>();

      payload.hitT        = hitObj.GetRayDesc().TMax;
      payload.rprimID     = hitObj.GetInstanceID();
      payload.rnodeID     = hitObj.GetInstanceIndex();
      payload.primitiveID = hitObj.GetPrimitiveIndex();
      payload.bary        = attr.barycentrics;
    }
    else
    {
      TraceRay(topLevelAS, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 0, 0, ray, payload);
    }
    seed = payload.seed;
  }

  void TraceLow(RayDesc ray, inout HitPayload payload)
  {
    if(USE_SER == 1 && AVAILABLE_SER == 1)
    {
      HitObject hitObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE, 0xFF, 0, 0, 0, ray, payload);
      if(hitObj.IsMiss())
      {
        payload.hitT = INFINITE;
        return;
      }
      payload.hitT    = hitObj.GetRayDesc().TMax;
      payload.rnodeID = hitObj.GetInstanceIndex();
    }
    else
    {
      // Accepted trade-off: this dispatches rchitMain, but only hitT/rnodeID are used.
      // A dedicated minimal CHS would require an extra SBT entry. The SER path above
      // already avoids that dispatch overhead.
      TraceRay(topLevelAS, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 0, 0, ray, payload);
    }
  }

  float3 TraceShadow(RayDesc ray, inout uint seed, bool initialInside = false)
  {
    ShadowPayload shadowPayload = {};
    shadowPayload.seed          = seed;
    shadowPayload.isInside      = initialInside;

    if(USE_SER == 1 && AVAILABLE_SER == 1)
    {
      HitObject hitObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);

      if(hitObj.IsHit())
      {
        return float3(0.0);
      }
    }
    else
    {
      TraceRay(topLevelAS, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
    }
    seed = shadowPayload.seed;

    return shadowPayload.totalTransmission;
  }
}
