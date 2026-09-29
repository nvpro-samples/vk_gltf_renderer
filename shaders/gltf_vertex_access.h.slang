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


// Local fork of nvpro_core2/nvshaders/gltf_vertex_access.h.slang.
// Same include guard (VERTEX_ACCESSORS_H) so any transitive include of the nvpro_core2 version
// is a no-op when this file is included first. Relies on GltfRenderPrimitive from the
// local gltf_scene_io.h.slang.

#ifndef VERTEX_ACCESSORS_H
#define VERTEX_ACCESSORS_H


float4 unpackUnorm4x8(uint packed)
{
  return float4(((packed >> 0) & 0xFF) / 255.0f, ((packed >> 8) & 0xFF) / 255.0f, ((packed >> 16) & 0xFF) / 255.0f,
                ((packed >> 24) & 0xFF) / 255.0f);
}


__generic<let N : int> public vector<float, N> mixBary(vector<float, N> a, vector<float, N> b, vector<float, N> c, float3 bary)
{
  return a * bary.x + b * bary.y + c * bary.z;
}

uint3 getTriangleIndices(GltfRenderPrimitive renderPrim, int primitiveID)
{
  return renderPrim.indices[primitiveID];
}

float3 getVertexPosition(GltfRenderPrimitive renderPrim, int vertexID)
{
  return renderPrim.vertexBuffer.positions[vertexID];
}

float3 getVertexNormal(GltfRenderPrimitive renderPrim, int vertexID)
{
  return renderPrim.vertexBuffer.normals[vertexID];
}

bool hasVertexNormal(GltfRenderPrimitive renderPrim)
{
  return renderPrim.vertexBuffer.normals != nullptr;
}

float3 getInterpolatedVertexNormal(GltfRenderPrimitive renderPrim, uint3 idx, float3 barycentrics)
{
  if(!hasVertexNormal(renderPrim))
    return float3(0.0f, 0.0f, 1.0f);

  float3* normals = renderPrim.vertexBuffer.normals;
  float3  nrm[3];
  nrm[0] = normals[idx.x];
  nrm[1] = normals[idx.y];
  nrm[2] = normals[idx.z];

  return mixBary(nrm[0], nrm[1], nrm[2], barycentrics);
}

bool hasVertexTexCoord(GltfRenderPrimitive renderPrim, int channel)
{
  return renderPrim.vertexBuffer.texCoords[channel] != nullptr;
}

float2 getInterpolatedVertexTexCoord(GltfRenderPrimitive renderPrim, int channel, uint3 idx, float3 barycentrics)
{
  if(!hasVertexTexCoord(renderPrim, channel))
    return float2(0.0f, 0.0f);

  float2* texcoords = renderPrim.vertexBuffer.texCoords[channel];
  float2  uv[3];
  uv[0] = texcoords[idx.x];
  uv[1] = texcoords[idx.y];
  uv[2] = texcoords[idx.z];

  return mixBary(uv[0], uv[1], uv[2], barycentrics);
}


bool hasVertexColor(GltfRenderPrimitive renderPrim)
{
  return renderPrim.vertexBuffer.colors != nullptr;
}

float4 getVertexColor(GltfRenderPrimitive renderPrim, uint idx)
{
  if(!hasVertexColor(renderPrim))
    return float4(1.0f, 1.0f, 1.0f, 1.0f);

  return unpackUnorm4x8(renderPrim.vertexBuffer.colors[idx]);
}

float4 getInterpolatedVertexColor(GltfRenderPrimitive renderPrim, uint3 idx, float3 barycentrics)
{
  if(!hasVertexColor(renderPrim))
    return float4(1.0f, 1.0f, 1.0f, 1.0f);

  uint*  colors = renderPrim.vertexBuffer.colors;
  float4 col[3];
  col[0] = unpackUnorm4x8(colors[idx.x]);
  col[1] = unpackUnorm4x8(colors[idx.y]);
  col[2] = unpackUnorm4x8(colors[idx.z]);
  return col[0] * barycentrics.x + col[1] * barycentrics.y + col[2] * barycentrics.z;
}

bool hasVertexTangent(GltfRenderPrimitive renderPrim)
{
  return renderPrim.vertexBuffer.tangents != nullptr;
}

float4 getVertexTangent(GltfRenderPrimitive renderPrim, uint idx)
{
  if(!hasVertexTangent(renderPrim))
    return float4(1, 0, 0, 1);

  return renderPrim.vertexBuffer.tangents[idx];
}

float4 getInterpolatedVertexTangent(GltfRenderPrimitive renderPrim, uint3 idx, float3 barycentrics)
{
  if(!hasVertexTangent(renderPrim))
    return float4(1, 0, 0, 1);

  float4* tangents = renderPrim.vertexBuffer.tangents;
  float4  tng[3];
  tng[0] = tangents[idx.x];
  tng[1] = tangents[idx.y];
  tng[2] = tangents[idx.z];

  // Interpolate only xyz; w is a discrete handedness (+1 / -1) that must not be blended.
  // Take handedness from vertex 0 — the glTF spec guarantees all three vertices of a triangle
  // share the same sign, so picking any single vertex is correct.
  float3 t = normalize(tng[0].xyz * barycentrics.x + tng[1].xyz * barycentrics.y + tng[2].xyz * barycentrics.z);
  return float4(t, tng[0].w);
}


#endif  // VERTEX_ACCESSORS_H
