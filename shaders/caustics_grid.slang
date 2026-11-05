/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//-----------------------------------------------------------------------
// The caustics' photon map grid: a counting sort of the frame's photons that landed, by hash cell (docs/caustics.md).
// A shader of its own, not part of the path tracer's: these passes know nothing of the scene, so they are built
// once, instead of with every variant of the path tracer's shader. The host (Caustics::cmdBuildGrid) clears
// cellStarts, then runs, with a barrier between each:
//   causticCountMain       one thread per slot: count the photons of each cell
//   causticScanBlocksMain  one group per 256 cells: the running total inside the block, and the block's total
//   causticScanSumsMain    one group: the running total of the block totals
//   causticAddOffsetsMain  one group per 256 cells: the block's offset added, so each cell holds the end of
//                          its photons in the sorted array
//   causticScatterMain     one thread per slot: each photon takes the position before its cell's end, which
//                          leaves each cell holding the start of its photons
// cellStarts[cell] .. cellStarts[cell + 1] is then the run of the cell's photons (the last entry is the total).
//-----------------------------------------------------------------------

#include "shaderio.h"
#include "nvshaders/random.h.slang"
#include "caustics_grid.h.slang"

struct CausticGridPushConstant
{
  CausticParams* caustics;
};
[[vk::push_constant]] ConstantBuffer<CausticGridPushConstant> pushConst;

[shader("compute")]
[numthreads(64, 1, 1)]
void causticCountMain(uint3 threadIdx: SV_DispatchThreadID)
{
  CausticParams* params = pushConst.caustics;
  if(threadIdx.x >= min(params->photonCount, params->mapSlots))
    return;
  const CausticPhoton photon = params->photons[threadIdx.x];
  if(photon.flux == 0u)
    return;  // An empty slot: the photon did not land
  Atomic<uint>* count = (Atomic<uint>*)(params->cellStarts + causticCellSlot(params, causticCell(params, photon.position)));
  count->add(1u);
}

groupshared uint causticScanShared[256];

// The running total of `value` over the group's threads, each thread's own included
uint causticInclusiveScan(uint local, uint value)
{
  causticScanShared[local] = value;
  GroupMemoryBarrierWithGroupSync();
  for(uint stride = 1u; stride < 256u; stride *= 2u)
  {
    const uint previous = local >= stride ? causticScanShared[local - stride] : 0u;
    GroupMemoryBarrierWithGroupSync();
    causticScanShared[local] += previous;
    GroupMemoryBarrierWithGroupSync();
  }
  return causticScanShared[local];
}

// Where the totals of the blocks of 256 cells are: after the cells and the count
uint* causticBlockSums(CausticParams* params)
{
  return params->cellStarts + params->cellMask + 2u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void causticScanBlocksMain(uint3 group: SV_GroupID, uint local: SV_GroupIndex)
{
  CausticParams* params = pushConst.caustics;
  uint*          starts = params->cellStarts;
  const uint     cell   = group.x * 256u + local;
  const uint     prefix = causticInclusiveScan(local, starts[cell]);
  starts[cell]          = prefix;
  if(local == 255u)
    causticBlockSums(params)[group.x] = prefix;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void causticScanSumsMain(uint local: SV_GroupIndex)
{
  CausticParams* params = pushConst.caustics;
  uint*          sums   = causticBlockSums(params);
  const uint     blocks = (params->cellMask + 1u) / 256u;
  const uint     chunk  = (blocks + 255u) / 256u;  // Block totals each thread owns
  const uint     first  = min(local * chunk, blocks);
  const uint     last   = min(first + chunk, blocks);
  uint           total  = 0u;
  for(uint i = first; i < last; i++)
    total += sums[i];
  const uint prefix = causticInclusiveScan(local, total);
  uint       offset = prefix - total;
  for(uint i = first; i < last; i++)
  {
    const uint blockTotal = sums[i];
    sums[i]               = offset;  // Now where the block starts
    offset += blockTotal;
  }
  if(local == 255u)
    params->cellStarts[params->cellMask + 1u] = prefix;  // All the photons
}

[shader("compute")]
[numthreads(256, 1, 1)]
void causticAddOffsetsMain(uint3 group: SV_GroupID, uint local: SV_GroupIndex)
{
  CausticParams* params = pushConst.caustics;
  params->cellStarts[group.x * 256u + local] += causticBlockSums(params)[group.x];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void causticScatterMain(uint3 threadIdx: SV_DispatchThreadID)
{
  CausticParams* params = pushConst.caustics;
  if(threadIdx.x >= min(params->photonCount, params->mapSlots))
    return;
  const CausticPhoton photon = params->photons[threadIdx.x];
  if(photon.flux == 0u)
    return;
  Atomic<uint>* end = (Atomic<uint>*)(params->cellStarts + causticCellSlot(params, causticCell(params, photon.position)));
  params->sorted[end->sub(1u) - 1u] = photon;
}
