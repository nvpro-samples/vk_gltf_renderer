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

#ifndef SKY_OMI_SHADERIO_H
#define SKY_OMI_SHADERIO_H 1

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

/*-------------------------------------------------------------------------------------------------
> Host/device parameter blocks for the authored sky types of the glTF `OMI_environment_sky`
> extension.

These are the *rendering* parameters, not the JSON model: the host deserializes the extension
into its own descriptor, resolves anything the JSON does not carry (notably the sun, which comes
from a `KHR_lights_punctual` directional light rather than from the sky block), and fills these
structs. They are too large for the `env_bake.slang` push constant, so they travel in a uniform
buffer bound at `EnvBakeBindings::eEnvBakeSkyParams`.

Only the member selected by `EnvBakePushConstant::skyType` is read by a given dispatch; the
others are ignored. One buffer for all authored sky types keeps the descriptor set stable as
later phases add types.
-------------------------------------------------------------------------------------------------*/

// Sky types, naming which parameter block below is live. Shared by every stage that has to agree
// on "which sky is this": the lat-long bake, the path tracer's background, and the rasterizer's
// background pass.
//
// Runtime-only: nothing persists a SkyType. The host derives one per frame from its own
// environment type (`bakeSkyType` in the sample's resources.hpp), so these values are free to
// move when a type is added or retired.
enum SkyType
{
  ePlain            = 0,  // OMI_environment_sky `plain`: one solid color.
  eGradient         = 1,  // OMI_environment_sky `gradient`: bottom/horizon/top bands + sun.
  ePhysicalBruneton = 2,  // Bruneton (2008) precomputed atmospheric scattering, evaluated from
                          //   the LUTs SkyBruneton builds.
};

/*-------------------------------------------------------------------------------------------------
# Struct `SkyPlainParameters`
> OMI `plain`: the sky is one solid color, in linear space.

The same color is both the visible background and the light the scene receives, so the lat-long
bake of a plain sky is a constant image and its alias table degenerates to uniform sampling --
which is exactly correct, and why the path tracer needs no special case for it.
-------------------------------------------------------------------------------------------------*/
struct SkyPlainParameters
{
  float3 color SLANG_DEFAULT(float3(0.5F, 0.5F, 0.5F));
  float _pad0  SLANG_DEFAULT(0.0F);
};

/*-------------------------------------------------------------------------------------------------
# Struct `SkyGradientParameters`
> OMI `gradient`: a three-band analytic sky (bottom / horizon / top) with a sun term.

Defaults are the extension's own. The extension names the curve fields but does not give the
formula behind them, and a gradient sky has to round-trip *visually* between tools rather than
just as JSON -- see sky_gradient.h.slang for the evaluation this settles on.

`sunDirection`, `sunColor` and `sunAngularRadius` are not part of the OMI sky block: the sun is a
`KHR_lights_punctual` directional light, and the host resolves it from that light before filling
this struct.
-------------------------------------------------------------------------------------------------*/
struct SkyGradientParameters
{
  float3 bottomColor SLANG_DEFAULT(float3(0.2F, 0.169F, 0.133F));
  float bottomCurve  SLANG_DEFAULT(0.02F);  // Horizon -> bottom falloff exponent, (0..1]

  float3 horizonColor SLANG_DEFAULT(float3(0.646F, 0.656F, 0.67F));
  float topCurve      SLANG_DEFAULT(0.15F);  // Horizon -> top falloff exponent, (0..1]

  float3 topColor   SLANG_DEFAULT(float3(0.385F, 0.454F, 0.55F));
  float sunAngleMax SLANG_DEFAULT(0.5F);  // Angular extent of the sun's glow, radians

  float3 sunColor SLANG_DEFAULT(float3(1.0F, 1.0F, 1.0F));
  float sunCurve  SLANG_DEFAULT(0.15F);  // Disk -> sky falloff exponent across the glow, (0..1]

  float3 sunDirection    SLANG_DEFAULT(float3(0.0F, 0.707106781F, 0.707106781F));  // Toward the sun
  float sunAngularRadius SLANG_DEFAULT(0.0F);  // Radians; 0 gives a point sun with glow only
};

/*-------------------------------------------------------------------------------------------------
# Struct `SkyOmiParameters`
> The uniform buffer `env_bake.slang` binds for authored sky types.

Grows by one member per authored sky type. Members are never reordered or removed: the host
writes the whole struct every commit, so a layout change silently reinterprets in-flight data.
-------------------------------------------------------------------------------------------------*/
struct SkyOmiParameters
{
  SkyPlainParameters    plain;
  SkyGradientParameters gradient;
};

NAMESPACE_SHADERIO_END()

#endif  // SKY_OMI_SHADERIO_H
