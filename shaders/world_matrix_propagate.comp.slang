/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Phase 1: propagate world matrices level-by-level (BFS order).
 * Matches Scene::updateWorldMatricesParallel
 * math (world = mul(local, parentWorld) for every node in the level — full tree, not subtree-dirty,
 * because the GPU overwrites all world entries from uploaded locals each frame (dirty or full sync).
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "world_matrix_io.h.slang"

[[vk::push_constant]]
ConstantBuffer<PropagateWorldMatricesPushConstant> pc;

[shader("compute")]
[numthreads(WORLD_MATRIX_WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
  uint ti = dtid.x;
  if(ti >= pc.levelCount)
    return;

  int nodeID = pc.topoNodeOrder[pc.levelOffset + ti];
  int parent = pc.parentIndices[nodeID];

  static const float4x4 kIdentity = float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
  float4x4              parentMat = (parent < 0) ? kIdentity : pc.worldMatrices[parent];

  pc.worldMatrices[nodeID] = mul(pc.localMatrices[nodeID], parentMat);
}
