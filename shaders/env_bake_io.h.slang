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

//
// Host/device interface for this renderer's environment bake -- the compute pass
// (shaders/env_bake.slang) that turns an analytic sky into the lat-long image and companion
// importance buffer that importance sampling and the prefiltered cubemaps are built from.
//
// The bindings and push constant are ours: `nvvk::HdrIbl::updateFromGpuImage` takes an image and an
// importance buffer, and says nothing about how a producer is wired. What it does require is the
// texel weighting and the direction mapping, and those live in nvpro_core2 next to the functions
// that read the image back -- `texelSolidAngle` and `latlongToDir` in nvshaders/functions.h.slang.
//

#ifndef ENV_BAKE_IO_H
#define ENV_BAKE_IO_H 1

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

enum EnvBakeBindings
{
  eEnvBakeColor = 0,       // RWTexture2D<float4>       lat-long color output (RGB; alpha
                           //                           reserved for env_write_pdf.slang).
  eEnvBakeImportance = 1,  // RWStructuredBuffer<float> importance out, one value per cell of
                           //                           the sampling grid: the sum over the cell's
                           //                           texels of solidAngle(row) * max(rgb).
  eEnvBakeSkyParams = 2,   // ConstantBuffer<SkyOmiParameters> the authored-sky parameter blocks.
                           //                           Only the one `skyType` names is read, but
                           //                           the binding is always bound -- a declared
                           //                           binding needs a valid descriptor even on a
                           //                           path that does not sample it.
};

// Push constant shared by every dispatch: the output geometry plus the sky selector. The sky
// parameters themselves are not here -- combined they run past the 128-byte guaranteed
// push-constant budget, so they travel in the uniform buffer at `eEnvBakeSkyParams` instead.
struct EnvBakePushConstant
{
  uint2 imageSize;  // Lat-long output resolution
  int   skyType;    // SkyType enum (sky_omi_shaderio.h.slang)
  int   _pad0;

  // PhysicalBruneton only. The Bruneton sky is the only type whose bake depends on where the sun
  // is and where the observer stands; the analytic types are either sun-free or evaluate the sun
  // from their own parameter block.
  float3 sunDirection;      // unit, world space, pointing *at* the sun
  float  observerAltitude;  // km above the planet surface (fixed-observer policy)

  // Where the bake publishes the solar disk it deliberately leaves out of the image: radiance and
  // angular size, for whoever draws the visible sky. Zero when the caller does not want it.
  //
  // An address rather than a binding because the consumers are the renderer's own background
  // passes, not this pipeline -- the value is produced here, where the atmosphere LUTs are, and
  // read somewhere else entirely. A plain address keeps that one-way handoff out of the
  // descriptor layout, and the analytic cases of the bake can ignore the field.
  uint64_t sunDiskAddress;

  // The alias table's sampling grid (kEnvSamplingGrid). The bake sums its importance per cell of
  // this grid rather than writing one value per texel -- see the reduction in env_bake.slang.
  uint2 gridSize;
};

NAMESPACE_SHADERIO_END()

#endif  // ENV_BAKE_IO_H
