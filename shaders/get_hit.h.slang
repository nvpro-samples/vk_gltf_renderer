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


#ifndef GET_HIT_H
#define GET_HIT_H

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  float3 pos;                  // World-space hit position
  float3 nrm;                  // Shading normal (interpolated and transformed)
  float4 color;                // Vertex color
  float3 geonrm;               // Geometric normal (face normal, for back-face detection)
  float3 shadowPos;            // Shadow position
  float2 uv[2];                // UV0 and UV1
  float3 tangent;              // Tangent
  float3 bitangent;            // Bitangent
  float  texelDensity = 0.0;   // UV-to-world area ratio for ray-footprint texture LOD (0 if no UVs)
  bool   frontFace    = true;  // True if the primitive was hit on its front side (CCW from the ray)
};

//-----------------------------------------------------------------------
// Compute the texel density of a triangle: the ratio of UV-space area to world-space area.
// Returns sqrt(uvArea / worldArea). When multiplied by a world-space footprint
// (hitT * pixelAngle), gives the UV-space gradient magnitude for SampleGrad().
float computeTexelDensity(float3 pos0, float3 pos1, float3 pos2, float2 uv0, float2 uv1, float2 uv2, float4x4 objectToWorld)
{
  float3 we1   = mul(float4(pos1 - pos0, 0.0), objectToWorld).xyz;
  float3 we2   = mul(float4(pos2 - pos0, 0.0), objectToWorld).xyz;
  float  wArea = length(cross(we1, we2));

  float2 duv1   = uv1 - uv0;
  float2 duv2   = uv2 - uv0;
  float  uvArea = abs(duv1.x * duv2.y - duv1.y * duv2.x);

  return sqrt(max(uvArea, 1e-20) / max(wArea, 1e-20));
}

//-----------------------------------------------------------------------
// Return hit information: position, normal, geonormal, uv, tangent, bitangent
HitState getHitState(GltfRenderPrimitive renderPrim,  //
                     float3              barycentrics,
                     float4x4            worldToObject,
                     float4x4            objectToWorld,
                     int                 triangleID,
                     float3              worldRayDirection)
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

  // Geometric Normal
  const float3 geoNormal = normalize(cross(pos1 - pos0, pos2 - pos0));
  hit.geonrm             = normalize(mul(worldToObject, float4(geoNormal, 0.0)).xyz);

  // Normal
  float3 nrm0   = geoNormal;
  float3 nrm1   = geoNormal;
  float3 nrm2   = geoNormal;
  float3 normal = geoNormal;
  if(hasVertexNormal(renderPrim))
  {
    nrm0   = getVertexNormal(renderPrim, triangleIndex.x);
    nrm1   = getVertexNormal(renderPrim, triangleIndex.y);
    nrm2   = getVertexNormal(renderPrim, triangleIndex.z);
    normal = nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z;
  }
  hit.nrm = normalize(mul(worldToObject, float4(normal, 0.0)).xyz);

  // Front-face is checked so the shadow-terminator `pointOffset` below can pre-flip
  // the per-vertex normals; without it, back-face hits on doubleSided meshes (e.g. leaf
  // underside) shift `shadowPos` to the wrong side and the shadow ray hits its own surface.
  hit.frontFace        = dot(hit.geonrm, worldRayDirection) < 0.0f;
  const float sideFlip = hit.frontFace ? 1.0f : -1.0f;

  // Hacking the shadow terminator (object-space offset; see Hanika 2021, "Hacking the
  // Shadow Terminator"). `nrm[i] * sideFlip` ensures the offset is on the camera-facing
  // side for back-face hits on doubleSided meshes.
  float3 shadowPos = pointOffset(position, pos0, pos1, pos2,
                                 nrm0 * sideFlip, nrm1 * sideFlip, nrm2 * sideFlip, barycentrics);
  hit.shadowPos    = mul(float4(shadowPos, 1.0), objectToWorld).xyz;

  // TexCoord
  hit.uv[0] = getInterpolatedVertexTexCoord(renderPrim, 0, triangleIndex, barycentrics);
  hit.uv[1] = getInterpolatedVertexTexCoord(renderPrim, 1, triangleIndex, barycentrics);

  // Texel density for ray-footprint texture LOD (geometry-only; used by path tracer)
  if(hasVertexTexCoord(renderPrim, 0))
  {
    float2* texCoords0 = renderPrim.vertexBuffer.texCoords[0];
    float2  uv0        = texCoords0[triangleIndex.x];
    float2  uv1        = texCoords0[triangleIndex.y];
    float2  uv2        = texCoords0[triangleIndex.z];
    hit.texelDensity   = computeTexelDensity(pos0, pos1, pos2, uv0, uv1, uv2, objectToWorld);
  }
  else
  {
    hit.texelDensity = 0.0;
  }

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
  hit.tangent   = normalize(mixBary(tng[0].xyz, tng[1].xyz, tng[2].xyz, barycentrics));
  hit.tangent   = mul(float4(hit.tangent, 0.0), objectToWorld).xyz;
  hit.tangent   = normalize(hit.tangent - hit.nrm * dot(hit.nrm, hit.tangent));
  hit.bitangent = cross(hit.nrm, hit.tangent) * tng[0].w;

  // Flip geometric normal toward the ray origin for back-face hits (front-face already
  // computed above so the shadow-terminator hack can pre-flip the per-vertex normals).
  if(!hit.frontFace)
  {
    hit.geonrm = -hit.geonrm;
  }

  // Ensure shading normal and geometric normal are on the same side
  // This prevents lighting artifacts on back-facing surfaces
  if(dot(hit.geonrm, hit.nrm) < 0)
  {
    hit.nrm       = -hit.nrm;  // Flip shading normal to match geometric normal
    hit.tangent   = -hit.tangent;
    hit.bitangent = -hit.bitangent;
  }

  // For low-tessellated geometry, prevent internal reflection artifacts
  float3 r = reflect(normalize(worldRayDirection), hit.nrm);
  if(dot(r, hit.geonrm) < 0)
  {
    hit.nrm = hit.geonrm;
  }

  return hit;
}


#endif
