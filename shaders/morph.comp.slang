/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Compute shader for morph target blending. Accumulates weighted deltas
// for positions, normals, and tangents from multiple morph targets.

#include "animation_io.h.slang"

[[vk::push_constant]]
ConstantBuffer<MorphPushConstant> pc;

[shader("compute")]
[numthreads(ANIMATION_WORKGROUP_SIZE, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
  uint v = tid.x;
  if(v >= pc.vertexCount)
    return;

  bool hasNormals  = (pc.baseNormals != nullptr);
  bool hasTangents = (pc.baseTangents != nullptr);

  float3 pos = pc.basePositions[v];
  float3 nrm = hasNormals ? pc.baseNormals[v] : float3(0);
  float4 tan = hasTangents ? pc.baseTangents[v] : float4(0);

  for(uint t = 0; t < pc.numTargets; t++)
  {
    float weight = pc.morphWeights[t];
    if(weight == 0.0)
      continue;

    uint offset = t * pc.vertexCount + v;
    pos += weight * pc.positionDeltas[offset];

    if(hasNormals && pc.normalDeltas != nullptr)
      nrm += weight * pc.normalDeltas[offset];

    if(hasTangents && pc.tangentDeltas != nullptr)
      tan = float4(tan.xyz + weight * pc.tangentDeltas[offset], tan.w);
  }

  pc.outPositions[v] = pos;

  if(hasNormals && pc.outNormals != nullptr)
  {
    pc.outNormals[v] = normalize(nrm);
  }

  if(hasTangents && pc.outTangents != nullptr)
  {
    pc.outTangents[v] = float4(normalize(tan.xyz), tan.w);
  }
}
