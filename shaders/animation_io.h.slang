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

#ifndef ANIMATION_IO_H
#define ANIMATION_IO_H

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define ANIMATION_WORKGROUP_SIZE 256

struct SkinPushConstant
{
  float3*   basePositions;
  float3*   baseNormals;
  float4*   baseTangents;
  float4*   weights;
  int4*     joints;
  float4x4* jointMatrices;
  float3x3* normalMatrices;
  float3*   outPositions;
  float3*   outNormals;
  float4*   outTangents;
  uint      vertexCount;
  uint      numJoints;
};

struct MorphPushConstant
{
  float3* basePositions;
  float3* baseNormals;
  float4* baseTangents;
  float3* positionDeltas;
  float3* normalDeltas;
  float3* tangentDeltas;
  float*  morphWeights;
  float3* outPositions;
  float3* outNormals;
  float4* outTangents;
  uint    vertexCount;
  uint    numTargets;
};

NAMESPACE_SHADERIO_END()

#endif
