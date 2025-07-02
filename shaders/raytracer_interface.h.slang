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

#include "get_hit.h.slang"

// Payload for the path tracer
struct HitPayload
{
  uint     seed;
  float    hitT    = 0.0f;
  int      rnodeID = -1;
  int      rprimID = -1;
  HitState hitState;
};

// Shadow payload for the path tracer
struct ShadowPayload
{
  uint   seed;
  float  hitT              = 0.0f;
  bool   isInside          = false;
  float3 totalTransmission = float3(0.0f);
};

// Raytracer interface definition
interface IRaytracer
{
  void   Trace(RayDesc ray, inout HitPayload payload, inout uint seed, int rayDepth = 0);
  float3 TraceShadow(RayDesc ray, inout uint seed);
}

// Implementation using RayQuery
struct RayQueryRaytracer : IRaytracer
{
  void Trace(RayDesc ray, inout HitPayload payload, inout uint seed, int rayDepth)
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

      HitState hit = getHitState(renderPrim, barycentrics, worldToObject, objectToWorld, triID, worldRayOrigin);

      payload.hitT     = hitT;
      payload.rprimID  = renderPrimID;
      payload.rnodeID  = instanceID;
      payload.hitState = hit;
    }
  }

  float3 TraceShadow(RayDesc ray, inout uint seed)
  {
    const float MIN_TRANSMISSION  = 0.01;  // Minimum transmission factor to continue tracing
    float3      totalTransmission = float3(1.0);
    bool        isInside          = false;
    float       approxHitT        = 0;

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
        approxHitT                 = abs(hitT - approxHitT);
        float3 currentTransmission = getShadowTransmission(renderNode, renderPrim, triangleID, barycentrics, approxHitT,
                                                           worldToObject, ray.Direction, isInside);

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
  void Trace(RayDesc ray, inout HitPayload payload, inout uint seed, int rayDepth)
  {
    payload.seed = seed;
    if(USE_SER == 1)
    {
      HitObject hitObj = HitObject::TraceRay(topLevelAS, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 0, 0, ray, payload);
      uint hint = (rayDepth == 1 ? 1 : 0);
      ReorderThread(hitObj, hint, 1);
      HitObject::Invoke(topLevelAS, hitObj, payload);
    }
    else
    {
      TraceRay(topLevelAS, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 0, 0, ray, payload);
    }
    seed = payload.seed;
  }

  float3 TraceShadow(RayDesc ray, inout uint seed)
  {
    ShadowPayload shadowPayload = {};
    shadowPayload.seed          = seed;
    TraceRay(topLevelAS, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
    seed = shadowPayload.seed;

    return shadowPayload.totalTransmission;
  }
}


