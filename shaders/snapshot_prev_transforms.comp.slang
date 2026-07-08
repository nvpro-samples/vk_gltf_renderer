/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * DLSS instance motion vectors: copy each render node's current objectToWorld into a dedicated
 * "previous transform" buffer. Dispatched once per rendered frame (DLSS only) at frame start,
 * BEFORE the per-frame transform update rewrites the render-node buffer. Because the render-node
 * SSBO still holds last frame's transforms at this point, outPrevObjectToWorld[i] ends up equal to
 * the transform used in the previously rendered frame -- exactly what the path tracer needs to
 * reproject the previous-frame world position of a moving instance.
 *
 * Path-agnostic: reads only the final render-node buffer, so it works whether the transforms were
 * produced by the GPU transform path (update_render_instances.comp) or the CPU upload path.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "world_matrix_io.h.slang"

[[vk::push_constant]]
ConstantBuffer<SnapshotPrevTransformsPushConstant> pc;

[shader("compute")]
[numthreads(WORLD_MATRIX_WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtid: SV_DispatchThreadID)
{
  uint i = dtid.x;
  if(i >= pc.renderNodeCount)
    return;

  pc.outPrevObjectToWorld[i] = pc.inRenderNodes[i].objectToWorld;
}
