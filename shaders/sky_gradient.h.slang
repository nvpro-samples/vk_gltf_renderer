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

#ifndef SKY_GRADIENT_H
#define SKY_GRADIENT_H 1

// Evaluator for the glTF `OMI_environment_sky` `gradient` sky type.
//
// Evaluation only -- no sampling routine and no PDF. Directions and their probabilities come
// from the alias table built over the lat-long bake (see env_bake.slang), which is what lets
// every sky type share one sampling path. A cosine-weighted proxy PDF would be a poor match for
// a sky carrying a sharp sun term and would bias MIS.
//
// The sky is two angular bands -- horizon to top above, horizon to bottom below -- plus a sun
// disk and its glow. The extension fixes those endpoints and says a curve of 1.0 is a linear
// transition, but leaves the shape in between undefined; the reciprocal-exponent form below is
// this renderer's choice, and changing it would change how existing content looks.
//
// The schema gives one `horizonColor` shared by both hemispheres, a single sun, and no intensity
// multiplier -- overall environment intensity is a viewer-side control applied at fetch time.
//
// The curve parameters are exponent reciprocals: `t = 1 - pow(1 - c, 1 / curve)`. Small values
// push the band toward its far color (a hard horizon); 1.0 is a linear ramp. They are clamped in
// the shader rather than trusted: a glTF file is not obliged to respect any editor's range hint,
// and `1 / 0` would poison the whole texel.

#include "sky_omi_shaderio.h.slang"
#include "nvshaders/constants.h.slang"

// Smallest curve we will divide by. Below this the band is visually indistinguishable from a
// hard edge, and the reciprocal starts to lose float precision.
static const float SKY_GRADIENT_MIN_CURVE = 1e-3F;

// Band interpolant: `c` runs 0 at the horizon to 1 at the pole. Returns the blend weight
// from the horizon color toward the pole color.
float skyGradientBandWeight(float c, float curve)
{
  const float safeCurve = max(curve, SKY_GRADIENT_MIN_CURVE);
  return clamp(1.0F - pow(1.0F - clamp(c, 0.0F, 1.0F), 1.0F / safeCurve), 0.0F, 1.0F);
}

// Radiance of the gradient sky along `dir` (unit, world space, +Y up).
//
// `includeSun` selects which of the plan's two environment fields this evaluates:
//   * `false` -- the lighting field baked by env_bake.slang. The sun term is omitted entirely,
//     not just its disk: the sun is a KHR_lights_punctual directional light that next-event
//     estimation already samples, so leaving its glow in the baked environment would light every
//     rough surface with roughly two suns' worth of energy. The failure is silent and looks
//     plausible, which is why it is gated by an explicit test.
//   * `true` -- the background field the renderer evaluates per primary ray (and after a
//     delta/specular bounce), where the sun must be visible and sharp.
float3 evalGradientSky(float3 dir, SkyGradientParameters p, bool includeSun)
{
  // 0 at the zenith, PI/2 at the horizon, PI at the nadir.
  const float vAngle = acos(clamp(dir.y, -1.0F, 1.0F));

  if(vAngle > M_PI_2)
  {
    // Lower hemisphere: horizon -> bottom. The sun term belongs to the sky half only, so there is
    // no sun below the horizon line.
    const float c = (vAngle - M_PI_2) / M_PI_2;
    return lerp(p.horizonColor, p.bottomColor, skyGradientBandWeight(c, p.bottomCurve));
  }

  // Upper hemisphere: horizon -> top.
  const float c   = 1.0F - vAngle / M_PI_2;
  float3      sky = lerp(p.horizonColor, p.topColor, skyGradientBandWeight(c, p.topCurve));

  if(!includeSun)
    return sky;

  // Sun: a flat disk of `sunColor` out to `sunAngularRadius`, then a glow that falls off to the
  // sky color by `sunAngleMax`. Guard the denominator -- a file may set sunAngleMax at or below
  // the disk radius, which collapses the glow rather than dividing by zero.
  const float sunAngle = acos(clamp(dot(p.sunDirection, dir), -1.0F, 1.0F));
  if(sunAngle < p.sunAngularRadius)
    return p.sunColor;

  if(sunAngle < p.sunAngleMax)
  {
    const float glowWidth = p.sunAngleMax - p.sunAngularRadius;
    if(glowWidth > 0.0F)
    {
      const float c2 = (sunAngle - p.sunAngularRadius) / glowWidth;
      sky            = lerp(p.sunColor, sky, skyGradientBandWeight(c2, p.sunCurve));
    }
  }

  return sky;
}

#endif  // SKY_GRADIENT_H
