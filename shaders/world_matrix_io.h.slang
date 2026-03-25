/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Shared structs for GPU world-matrix propagation (Phase 1) and render-node / TLAS instance
 * writes (Phase 2). Kept in a header so both compute entry points and C++ agree on layout
 * via the same shaderio namespace as other shaders.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef WORLD_MATRIX_IO_H
#define WORLD_MATRIX_IO_H

#include "nvshaders/slang_types.h"
#include "nvshaders/gltf_scene_io.h.slang"

NAMESPACE_SHADERIO_BEGIN()

// Match AnimationVk / skinning workgroup size for occupancy consistency on NVIDIA hardware.
#define WORLD_MATRIX_WORKGROUP_SIZE 256

// Phase 1: one dispatch per topological BFS level. Each thread handles one node in the level.
struct PropagateWorldMatricesPushConstant
{
  float4x4* localMatrices;
  float4x4* worldMatrices;
  int*      parentIndices;
  int*      topoNodeOrder;
  uint      levelOffset;
  uint      levelCount;
};

// One entry per render node (TLAS instance index). Built once at scene load / rebuild.
struct RenderNodeGpuMapping
{
  int nodeID;
  int pad0;
  int materialID;
  int renderPrimID;
};

// 64-byte layout matching VkAccelerationStructureInstanceKHR (transform + packed fields + BLAS address).
struct TlasInstance
{
  float3x4 transform3x4;
  uint32_t instanceCustomIndexAndMask;
  uint32_t instanceShaderBindingTableRecordOffsetAndFlags;
  uint64_t accelerationStructureReference;
};

// Instance buffer uses read/modify/write: reload each instance, replace transform only,
// store back so packed fields and BLAS device address stay valid.
struct UpdateRenderInstancesPushConstant
{
  float4x4*             worldMatrices;
  RenderNodeGpuMapping* mappings;
  float4x4*             gpuInstLocalMatrices;  // one mat4 per render node (identity if unused)
  GltfRenderNode*       outRenderNodes;
  TlasInstance*         outInstances;
  uint                  renderNodeCount;
  uint                  pad0;
};

NAMESPACE_SHADERIO_END()

#endif
