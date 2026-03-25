/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Phase 2: For each render node / TLAS slot, compute final world matrix (including GPU instancing),
 * write GltfRenderNode (object/world + inverse + IDs), and update only the 48-byte transform in
 * VkAccelerationStructureInstanceKHR (64-byte stride; remainder preserved for BLAS ref and packed fields).
 *
 * Matrix math (WHY this matches CPU / glm):
 * - Slang compiles to SPIR-V using HLSL-style semantics: `mul(A, B)` is the matrix product A*B
 *   with column-major `float4x4` storage — the same convention as glTF samples and glm on the host.
 * - Do not use GLSL's `*` operator rules here; SPIR-V matrix ops match HLSL-style `mul` / `OpMatrixTimesMatrix`.
 * - Hierarchy on CPU: world = parentWorld * local. Instancing: final = nodeWorld * instanceLocal
 *   (see Scene::updateWorldMatricesParallel / updateRenderNodesFull). Same order: `mul(nodeWorld, instLocal)`.
 *
 * TLAS transform: nvvk::toTransformMatrixKHR transposes the glm mat4 then stores the first 3 rows as
 * row-major 3x4; we use `transpose(worldMat)` then copy three float4 rows into `transform[12]`.
 *
 * Inverse: use SPIR-V GLSL450 `MatrixInverse` so we match a full `glm::inverse` for `worldToObject`.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "world_matrix_io.h.slang"

// SPIR-V path: GLSL450 extended instruction MatrixInverse (same numeric result as a full 4x4 inverse).
// Kept in this .comp.slang only — not in shared .h.slang (C++ includes that header for struct layout).
__generic<T : __BuiltinFloatingPointType, let N : int> [require(spirv)] matrix<T, N, N> inverse(matrix<T, N, N> m)
{
  __target_switch
  {
    case spirv:
      return spirv_asm {
      OpExtInst $$matrix<T, N, N> result glsl450 MatrixInverse $m
    };
  }
}

[[vk::push_constant]]
ConstantBuffer<UpdateRenderInstancesPushConstant> pc;

[shader("compute")]
[numthreads(WORLD_MATRIX_WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtid: SV_DispatchThreadID)
{
  uint i = dtid.x;
  if(i >= pc.renderNodeCount)
    return;

  RenderNodeGpuMapping map = pc.mappings[i];

  float4x4 worldMat  = pc.worldMatrices[map.nodeID];
  float4x4 instLocal = pc.gpuInstLocalMatrices[i];
  worldMat           = mul(worldMat, instLocal);

  // Set the render node (used by the raster)
  GltfRenderNode rn;
  rn.objectToWorld     = worldMat;
  rn.worldToObject     = inverse(worldMat);
  rn.materialID        = map.materialID;
  rn.renderPrimID      = map.renderPrimID;
  pc.outRenderNodes[i] = rn;

  // Set the TLAS transform
  float4x4 tr = transpose(worldMat);

  TlasInstance inst  = pc.outInstances[i];
  inst.transform3x4  = float3x4(tr);
  pc.outInstances[i] = inst;
}
