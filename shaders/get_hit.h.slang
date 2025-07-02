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


#ifndef GET_HIT_H
#define GET_HIT_H

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  float3 pos;
  float3 nrm;
  float4 color;
  float3 geonrm;
  float2 uv[2];
  float3 tangent;
  float3 bitangent;
};

//-----------------------------------------------------------------------
// Return hit information: position, normal, geonormal, uv, tangent, bitangent
HitState getHitState(GltfRenderPrimitive renderPrim,  //
                     float3              barycentrics,
                     float4x3            worldToObject,
                     float4x3            objectToWorld,
                     int                 triangleID,
                     float3              worldRayOrigin)
{
  HitState hit;

  // Getting the 3 indices of the triangle (local)
  uint3 triangleIndex = getTriangleIndices(renderPrim, triangleID);

  // Position
  const float3 pos0     = getVertexPosition(renderPrim, triangleIndex.x);
  const float3 pos1     = getVertexPosition(renderPrim, triangleIndex.y);
  const float3 pos2     = getVertexPosition(renderPrim, triangleIndex.z);
  const float3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos               = float3(mul(float4(position, 1.0), objectToWorld).xyz);

  // Normal
  const float3 geoNormal      = normalize(cross(pos1 - pos0, pos2 - pos0));
  float3       worldGeoNormal = normalize(float3(mul(worldToObject, geoNormal).xyz));
  float3       normal         = geoNormal;
  float3       worldNormal    = worldGeoNormal;
  if(hasVertexNormal(renderPrim))
  {
    normal      = getInterpolatedVertexNormal(renderPrim, triangleIndex, barycentrics);
    worldNormal = normalize(float3(mul(worldToObject, normal).xyz));
  }
  hit.geonrm = worldGeoNormal;
  hit.nrm    = worldNormal;

  // TexCoord
  hit.uv[0] = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);
  hit.uv[1] = getInterpolatedVertexTexCoord1(renderPrim, triangleIndex, barycentrics);

  // Color
  hit.color = getInterpolatedVertexColor(renderPrim, triangleIndex, barycentrics);

  // Tangent - Bitangent
  float4 tng[3];
  if(hasVertexTangent(renderPrim))
  {
    tng[0] = getVertexTangent(renderPrim, triangleIndex.x);
    tng[1] = getVertexTangent(renderPrim, triangleIndex.y);
    tng[2] = getVertexTangent(renderPrim, triangleIndex.z);
  }
  else
  {
    float4 t = makeFastTangent(normal);
    tng[0]   = t;
    tng[1]   = t;
    tng[2]   = t;
  }
  hit.tangent = normalize(mixBary(tng[0].xyz, tng[1].xyz, tng[2].xyz, barycentrics));
  hit.tangent = float3(mul(float4(hit.tangent, 0.0), objectToWorld));
  hit.tangent   = normalize(hit.tangent - hit.nrm * dot(hit.nrm, hit.tangent));
  hit.bitangent = cross(hit.nrm, hit.tangent) * tng[0].w;

  // Adjusting normal
  const float3 V = (worldRayOrigin - hit.pos);
  if(dot(hit.geonrm, V) < 0)  // Flip if back facing
  {
    hit.geonrm = -hit.geonrm;
  }

  // If backface
  if(dot(hit.geonrm, hit.nrm) < 0)  // Make Normal and GeoNormal on the same side
  {
    hit.nrm       = -hit.nrm;
    hit.tangent   = -hit.tangent;
    hit.bitangent = -hit.bitangent;
  }

  // handle low tessellated meshes with smooth normals
  float3 k2 = reflect(-V, hit.nrm);
  if(dot(hit.geonrm, k2) < 0.0f)
  {
    hit.nrm = hit.geonrm;
  }

  // For low tessalated, avoid internal reflection
  // float3 r = reflect(normalize(worldRayDirection), hit.nrm);
  // if(dot(r, hit.geonrm) < 0)
  // {
  //   hit.nrm = hit.geonrm;
  // }

  return hit;
}


#endif
