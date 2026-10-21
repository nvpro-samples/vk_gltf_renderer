/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Phase 2: For each render node / TLAS slot, compute final world matrix (including GPU instancing),
 * write GltfRenderNode (object/world + inverse + IDs), and update the TLAS instance transform.
 *
 * TLAS: we only rewrite transform3x4; accelerationStructureReference is preserved from the loaded
 * instance (last CPU TLAS sync or prior frame). KHR_node_visibility is applied on the CPU via
 * SceneRtx::syncTopLevelAS when DirtyFlags::tlasVisibilityNeedsCpuSync is set — avoids staging
 * multi-megabyte mapping buffers when huge subtrees toggle visibility.
 *
 * Matrix math (WHY the operands look swapped):
 * - The host uploads glm matrices as they are (column-major), but shaders are compiled with
 *   `-matrix-layout-row-major` (nvpro_core2 CompilerSlangShader.cmake), so every float4x4 here is
 *   the transpose of the host matrix, and `mul(A, B)` here is the host's B*A.
 * - Instancing on the host: final = nodeWorld * instanceLocal (see Scene::updateWorldMatricesParallel
 *   / updateRenderNodesFull), so here it is `mul(instLocal, nodeWorld)` -- the same reason
 *   world_matrix_propagate writes `mul(local, parentWorld)` for parentWorld * local.
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
__generic<T : __BuiltinFloatingPointType, let N : int>[require(spirv)] matrix<T, N, N> inverse(matrix<T, N, N> m)
{
  __target_switch
  {
    case spirv:
      return spirv_asm { OpExtInst $$matrix<T, N, N> result glsl450 MatrixInverse $m };
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
  worldMat           = mul(instLocal, worldMat);

  // Set the render node (used by the raster)
  GltfRenderNode rn;
  rn.objectToWorld     = worldMat;
  rn.worldToObject     = inverse(worldMat);
  rn.materialID        = map.materialID;
  rn.renderPrimID      = map.renderPrimID;
  pc.outRenderNodes[i] = rn;

  float4x4 tr = transpose(worldMat);

  TlasInstance inst  = pc.outInstances[i];
  inst.transform3x4  = float3x4(tr);
  pc.outInstances[i] = inst;
}
