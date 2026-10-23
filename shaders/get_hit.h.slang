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
  float  texelDensity   = 0.0;   // UV-to-world area ratio for ray-footprint texture LOD (0 if no UVs)
  float  offsetDistance = 0.0;   // How far a ray leaving `pos` steps off the surface along `geonrm` (safeOffsetRay)
  bool   frontFace      = true;  // True if the primitive was hit on its front side (CCW from the ray)
};

//-----------------------------------------------------------------------
// Compute the texel density of a triangle: the ratio of UV-space area to world-space area.
// Returns sqrt(uvArea / worldArea). When multiplied by a world-space footprint
// (hitT * pixelAngle), gives the UV-space gradient magnitude for SampleGrad().
// `we1` and `we2` are the triangle's world-space edges from vertex 0.
float computeTexelDensity(float3 we1, float3 we2, float2 uv0, float2 uv1, float2 uv2)
{
  float wArea = length(cross(we1, we2));

  float2 duv1   = uv1 - uv0;
  float2 duv2   = uv2 - uv0;
  float  uvArea = abs(duv1.x * duv2.y - duv1.y * duv2.x);

  return sqrt(max(uvArea, 1e-20) / max(wArea, 1e-20));
}

//-----------------------------------------------------------------------
// Rounding-error bounds for the self-intersection offset (HitState::offsetDistance)

static const float FLOAT_UNIT_ROUNDOFF = 5.96046448e-8F;  // u = 2^-24, the relative rounding error of one float operation

// Bound on the relative rounding error of `n` chained float operations, gamma(n) in "Physically Based
// Rendering" (Pharr, Jakob, Humphreys, 4th ed., 6.8.1).
float roundingGamma(int n)
{
  return (n * FLOAT_UNIT_ROUNDOFF) / (1.0F - n * FLOAT_UNIT_ROUNDOFF);
}

// The hardware ray/triangle test's own rounding, in units of u times the triangle's size. Vulkan does
// not specify it, so it is measured (tests/render/ray_offset.py): a floor whose shadow rays run along
// it in object space shadows itself without this term, and is clean from 1. 16 leaves a margin, and
// still lets a ray from a 100 m floor pass under an object 0.5 mm above it (a leak from about 59).
static const float RAY_TEST_ERROR_ULPS = 16.0F;

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

  // World-space edges: the triangle's size, and its area for the texel density below
  const float3 worldEdge1 = mul(float4(pos1 - pos0, 0.0), objectToWorld).xyz;
  const float3 worldEdge2 = mul(float4(pos2 - pos0, 0.0), objectToWorld).xyz;
  const float3 worldEdge3 = worldEdge2 - worldEdge1;

  // Geometric Normal
  const float3 geoNormal = normalize(cross(pos1 - pos0, pos2 - pos0));
  hit.geonrm             = normalize(mul(worldToObject, float4(geoNormal, 0.0)).xyz);

  // How far a ray leaving `pos` must step off the surface not to hit it again, along the geometric
  // normal -- the only direction that matters to a ray leaving the surface. Two errors add up:
  //  - `pos`'s own rounding, per component (PBRT 6.8, "Managing Rounding Error"): that of the
  //    barycentric interpolation, which follows the size of the vertex coordinates, then that of the
  //    instance transform, which also follows its translation. The barycentrics' own error is left
  //    out: it moves the point within the triangle's plane, not off it.
  //  - The rounding of the hardware's test of the next ray against this triangle. PBRT's own
  //    intersector guards against it from inside, rejecting any hit closer than its error; hardware
  //    cannot be told to, so the origin moves past it instead. It follows the size of the
  //    coordinates the test works with: the triangle's extent around the ray origin.
  // Every term is a length of the scene's own, so the step scales with the scene.
  {
    const float3 objectError = roundingGamma(7) * (abs(barycentrics.x * pos0) + abs(barycentrics.y * pos1) + abs(barycentrics.z * pos2));
    // The world error is |objectToWorld| applied to the object error (and to |position|, 1 for the
    // translation); along the normal, dot(|n|, |M| e) = dot(|M|^T |n|, e), so the normal goes through
    // the matrix once instead of each error. Rows 0-2 weigh the object axes, row 3 the translation.
    const float4 normalWeight = mul(abs(objectToWorld), float4(abs(hit.geonrm), 0.0));
    const float  pointError   = (1.0F + roundingGamma(3)) * dot(normalWeight.xyz, objectError)
                             + roundingGamma(3) * (dot(normalWeight.xyz, abs(position)) + normalWeight.w);
    const float  triangleSize = sqrt(max(dot(worldEdge1, worldEdge1), max(dot(worldEdge2, worldEdge2), dot(worldEdge3, worldEdge3))));
    hit.offsetDistance = pointError + RAY_TEST_ERROR_ULPS * FLOAT_UNIT_ROUNDOFF * triangleSize;
  }

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
  float3 shadowPos = pointOffset(position, pos0, pos1, pos2, nrm0 * sideFlip, nrm1 * sideFlip, nrm2 * sideFlip, barycentrics);
  hit.shadowPos = mul(float4(shadowPos, 1.0), objectToWorld).xyz;

  // TexCoord
  hit.uv[0] = getInterpolatedVertexTexCoord(renderPrim, 0, triangleIndex, barycentrics);
  hit.uv[1] = getInterpolatedVertexTexCoord(renderPrim, 1, triangleIndex, barycentrics);

  // Texel density for ray-footprint texture LOD (geometry-only; used by path tracer)
  if(hasVertexTexCoord(renderPrim, 0))
  {
    float2* texCoords0 = renderPrim.vertexBuffer.texCoord0;
    float2  uv0        = texCoords0[triangleIndex.x];
    float2  uv1        = texCoords0[triangleIndex.y];
    float2  uv2        = texCoords0[triangleIndex.z];
    hit.texelDensity   = computeTexelDensity(worldEdge1, worldEdge2, uv0, uv1, uv2);
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
  float3 tangent = mixBary(tng[0].xyz, tng[1].xyz, tng[2].xyz, barycentrics);
  tangent        = mul(float4(tangent, 0.0), objectToWorld).xyz;
  // A zero tangent, or one parallel to the normal (some exporters write these for faces whose UVs are
  // constant, e.g. palette textures), leaves nothing after removing the normal component: normalizing
  // it would give NaN shading. Fall back to an arbitrary tangent perpendicular to the normal.
  // The test is relative to the transformed tangent's own length, so a node's scale cannot trip it.
  const float3 tangentOrtho = tangent - hit.nrm * dot(hit.nrm, tangent);
  const float  orthoLength2 = dot(tangentOrtho, tangentOrtho);
  hit.tangent =
      orthoLength2 > 1e-6 * dot(tangent, tangent) ? tangentOrtho * rsqrt(orthoLength2) : makeFastTangent(hit.nrm).xyz;
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
