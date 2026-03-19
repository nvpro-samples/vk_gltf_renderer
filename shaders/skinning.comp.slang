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

// Compute shader for skeletal skinning with up to 4 joint influences per vertex.
// Transforms positions, normals, and tangents using joint matrices.

#include "animation_io.h.slang"

[[vk::push_constant]]
ConstantBuffer<SkinPushConstant> pc;

[shader("compute")]
[numthreads(ANIMATION_WORKGROUP_SIZE, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
  uint v = tid.x;
  if(v >= pc.vertexCount)
    return;

  float4 w = pc.weights[v];
  int4   j = pc.joints[v];

  bool hasNormals  = (pc.baseNormals != nullptr);
  bool hasTangents = (pc.baseTangents != nullptr);

  float3 skinnedPos = float3(0);
  float3 skinnedNrm = float3(0);
  float3 skinnedTan = float3(0);

  [unroll]
  for(int i = 0; i < 4; ++i)
  {
    float jw = w[i];
    if(jw > 0.0 && j[i] >= 0 && uint(j[i]) < pc.numJoints)
    {
      int ji = j[i];
      skinnedPos += jw * mul(float4(pc.basePositions[v], 1.0), pc.jointMatrices[ji]).xyz;

      if(hasNormals)
        skinnedNrm += jw * mul(pc.baseNormals[v], pc.normalMatrices[ji]);

      if(hasTangents)
        skinnedTan += jw * mul(pc.baseTangents[v].xyz, (float3x3)pc.jointMatrices[ji]);
    }
  }

  pc.outPositions[v] = skinnedPos;

  if(hasNormals)
    pc.outNormals[v] = normalize(skinnedNrm);

  if(hasTangents)
    pc.outTangents[v] = float4(normalize(skinnedTan), pc.baseTangents[v].w);
}
