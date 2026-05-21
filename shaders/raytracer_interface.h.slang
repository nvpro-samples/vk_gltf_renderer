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

// Forward declarations
float  getOpacity(GltfRenderNode renderNode, GltfRenderPrimitive renderPrim, int triangleID, float3 barycentrics);
float3 getShadowTransmission(GltfRenderNode      renderNode,
                             GltfRenderPrimitive renderPrim,
                             int                 triangleID,
                             float3              barycentrics,
                             float               hitT,
                             float4x3            worldToObject,
                             float3              rayDirection,
                             inout bool          isInside);

// Payload for the path tracer
struct HitPayload
{
  uint     seed;
  float    hitT        = 0.0f;
  int      rnodeID     = -1;
  int      rprimID     = -1;
  int      primitiveID = -1;
  HitState hitState;
};

// Shadow payload for the path tracer
struct ShadowPayload
{
  uint seed;
  float prevHitT = 0.0f;  // The previous candidate hit's absolute RayT (in-order assumption); rahitShadow / TraceShadow use it to compute the segment length for Beer's-law volume attenuation between consecutive transmissive hits.
  bool   isInside          = false;         // If true, the shadow ray is considered to start INSIDE a volume
  float3 totalTransmission = float3(1.0f);  // The total transmission of the shadow ray through the volume
};

#if WIREFRAME
//-----------------------------------------------------------------------
// Analytical wireframe alpha: closed-form barycentric deltas from the ray-cone footprint.
//   worldFoot = hitT * pixelAngle / |N.V|              (slant-corrected, see rayConeWorldFootprint)
//   d(bary_i) = worldFoot * |edge_opposite_i| / (2A)   (h_i = 2A / |edge_i|)
// Inputs: object-space verts v0..v2 + objectToWorld (float4x4 from GltfRenderNode); world-space
// hit position (barycentrics are recovered here via plane projection so the payload doesn't
// have to carry them); N = world geometric normal; V = -ray.Direction.
//-----------------------------------------------------------------------
float computeMeshWireframe(float4x4 objectToWorld, float3 v0, float3 v1, float3 v2, float3 hitWorldPos, float hitT, float pixelAngle, float3 N, float3 V)
{
  // Triangle in world space (the cone footprint is a world-space quantity).
  float3 wp0 = mul(float4(v0, 1.0), objectToWorld).xyz;
  float3 wp1 = mul(float4(v1, 1.0), objectToWorld).xyz;
  float3 wp2 = mul(float4(v2, 1.0), objectToWorld).xyz;

  // Recover barycentrics from the world-space hit position (Ericson, "Real-Time Collision
  // Detection", section 3.4). Saves carrying attr.barycentrics in the payload just for wireframe.
  float3 e1       = wp1 - wp0;
  float3 e2       = wp2 - wp0;
  float3 ep       = hitWorldPos - wp0;
  float  d11      = dot(e1, e1);
  float  d12      = dot(e1, e2);
  float  d22      = dot(e2, e2);
  float  dp1      = dot(ep, e1);
  float  dp2      = dot(ep, e2);
  float  invDenom = 1.0 / max(d11 * d22 - d12 * d12, 1e-20);
  float  baryV    = (d22 * dp1 - d12 * dp2) * invDenom;
  float  baryW    = (d11 * dp2 - d12 * dp1) * invDenom;
  float3 barys    = float3(1.0 - baryV - baryW, baryV, baryW);

  // Slant-corrected ray-cone footprint on the surface.
  float worldFoot = hitT * pixelAngle / max(abs(dot(N, V)), 1e-3);

  // |grad(bary_i)| in world space is 1/h_i = |edge_opposite_i|/(2A); times worldFoot gives
  // the per-pixel bary delta. Same role as fwidth(bary) in a raster pipeline.
  float  twoArea  = length(cross(e1, e2));
  float3 edgesOpp = float3(length(wp2 - wp1), length(wp0 - wp2), length(wp1 - wp0));
  float3 deltas   = worldFoot * edgesOpp / max(twoArea, 1e-20);

  return processWireframe(WIREFRAME_PRESETS[WIREFRAME_THIN], barys, deltas);
}
#endif

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

      GltfRenderNode      renderNode = pushConst.gltfScene->renderNodes[instanceID];
      GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

      float opacity = getOpacity(renderNode, renderPrim, triangleID, barycentrics);

      // do alpha blending the stochastically way
      if(rand(seed) <= opacity)
        rayQuery.CommitNonOpaqueTriangleHit();
    }

    if(rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
      float2   bary           = rayQuery.CommittedTriangleBarycentrics();
      int      instanceID     = rayQuery.CommittedInstanceIndex();
      int      renderPrimID   = rayQuery.CommittedInstanceID();
      int      triID          = rayQuery.CommittedPrimitiveIndex();
      float4x3 worldToObject  = rayQuery.CommittedWorldToObject4x3();
      float4x3 objectToWorld  = rayQuery.CommittedObjectToWorld4x3();
      float    hitT           = rayQuery.CommittedRayT();
      float3   worldRayOrigin = ray.Origin;

      // Retrieve the Primitive mesh buffer information
      GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

      // Barycentric coordinate on the triangle
      const float3 barycentrics = float3(1.0 - bary.x - bary.y, bary.x, bary.y);

      HitState hit = getHitState(renderPrim, barycentrics, worldToObject, objectToWorld, triID, ray.Direction);

      payload.hitT        = hitT;
      payload.rprimID     = renderPrimID;
      payload.rnodeID     = instanceID;
      payload.primitiveID = triID;
      payload.hitState    = hit;
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

      // There is a hit: filling the payload with the hit information

      // Barycentric coordinate on the triangle
      BuiltInTriangleIntersectionAttributes attr = hitObj.GetAttributes<BuiltInTriangleIntersectionAttributes>();
      float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

      int instanceID   = hitObj.GetInstanceIndex();
      int renderPrimID = hitObj.GetInstanceID();
      int primitiveID  = hitObj.GetPrimitiveIndex();

      // Get the built-in ray tracing variables
      float4x3 worldToObject  = hitObj.GetWorldToObject();
      float4x3 objectToWorld  = hitObj.GetObjectToWorld();
      float3   worldRayOrigin = ray.Origin;
      float    hitT           = hitObj.GetRayDesc().TMax;  // Closest Hit point

      // Retrieve the Primitive mesh buffer information
      GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[renderPrimID];

      HitState hit = getHitState(renderPrim, barycentrics, worldToObject, objectToWorld, primitiveID, ray.Direction);

      payload.hitT        = hitT;
      payload.rprimID     = renderPrimID;
      payload.rnodeID     = instanceID;
      payload.primitiveID = primitiveID;
      payload.hitState    = hit;
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
      // Accepted trade-off: this dispatches rchitMain which does full getHitState(),
      // but only hitT/rnodeID are used. A dedicated minimal CHS would require an
      // extra SBT entry. The SER path above already avoids this overhead.
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
