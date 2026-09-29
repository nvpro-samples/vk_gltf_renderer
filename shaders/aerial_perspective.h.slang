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

#ifndef AERIAL_PERSPECTIVE_H
#define AERIAL_PERSPECTIVE_H 1

// Aerial perspective: the air between the camera and a surface.
//
// The environment answers what a ray sees when it *escapes* the scene. This answers what it sees
// when it stops at something -- the same atmosphere, integrated over a segment instead of a whole
// ray. It is why distant hills go pale and blue while near ones do not, and it is the one thing a
// baked lat-long can never supply, because it depends on how far away the surface is.
//
// Both render paths include this header and call the same function on the same inputs, so the
// rasterizer's haze and the path tracer's cannot drift. Each declares the LUT bindings at its own
// set index and passes them in -- a shared header cannot fix a set number that differs between two
// pipelines, and passing the samplers is what the Bruneton port already does everywhere else.
//
// Physical sky only. The other environments do not model air: a gradient is a painting, a loaded
// HDR already has whatever haze the photographer stood in, and inventing an atmosphere for either
// would be this renderer overruling the author.

#include "shaderio.h"
#include "nvshaders/functions.h.slang"  // luminance, quatRotateInverse
#include "sky_bruneton_functions.h.slang"
#include "sky_bruneton_io.h.slang"

// Cheap test the call sites use to skip the work entirely.
//
// `sunDisk` is non-null exactly for the physical sky -- the same idiom the background pass uses to
// decide whether to draw a solar disk -- so there is no second flag to keep in step with the first.
bool hasAerialPerspective(SceneFrameInfo* frameInfo)
{
  return frameInfo->sunDisk != nullptr && frameInfo->aerialPerspectiveScale > 0.0F;
}

// In-scattered light and transmittance along the segment, by marching it.
//
// **Why a march and not a table lookup.** GetSkyRadianceToPoint answers the same question in two
// texture fetches, by subtracting everything scattered beyond the surface from everything scattered
// beyond the eye. That is elegant and it is what upstream ships, and it fails here in a way that is
// impossible to unsee: the table is stored in two halves, rays that meet the planet and rays that
// escape, and which half to read is decided by where the *extended* ray goes. A view ray tipping
// past the horizon switches halves while the five kilometres of air being shaded does not change,
// and the halves do not agree. Measured on a 10 km mountain, the escaping half spikes in the few
// rows just above the horizon -- where its mu parameterisation is most compressed -- and draws a
// hard line across the rock. Forcing the other half removes the line and is wrong by 20/255 higher
// up, where the ray really does escape.
//
// Marching has no halves, no branch and no subtraction of two large nearly-equal numbers. It is what
// Unreal and Hillaire do for aerial perspective, for the same reason.
//
// Multiple scattering is the one approximation: Bruneton's tables integrate it along whole rays and
// cannot hand back the local source term a march needs, so it comes from the irradiance table as an
// isotropic term. Over the few kilometres this models, it is a small part of a small term -- and it
// is smooth, which the thing it replaces was not.
//
// **The sun's share, and one point on it.** Every point of the march is treated as sunlit, with only
// the planet's shadow in it: the atmosphere knows nothing of the scene. So the sun's single
// scattering is also returned on its own, together with one point on the segment picked in
// proportion to it -- `sunEstimate` is the one-sample estimate of `sunInScatter` taken at that point
// (step chosen with probability luminance / total, `u.y` placing it within the step). A caller that
// can trace scene visibility from that point to the sun (the path tracer) replaces `sunInScatter` by
// `visibility * sunEstimate`, which is unbiased and converges as samples accumulate; one that cannot
// (the rasterizer) keeps the unshadowed total. The multiple-scattering term is light from the whole
// sky, not from the sun, and is left as it is.
struct AerialMarch
{
  float3 inScatter;      // everything the air adds, sun included
  float3 transmittance;  // what is left of the surface behind it
  float3 sunInScatter;   // the sun's single-scattering part of inScatter
  float3 sunEstimate;    // one-sample estimate of sunInScatter at `sunPoint`; zero when the sun adds nothing
  float  sunPoint;       // where that sample stands, 0 at the eye to 1 at the surface
};

AerialMarch marchAerialPerspective(SkyAtmosphereParameters atmosphere,
                                   Sampler2D<float3>       transmittanceTexture,
                                   Sampler2D<float3>       irradianceTexture,
                                   float3                  observer,
                                   float3                  displacement,
                                   float3                  sunDirection,
                                   float2                  u)
{
  // Sixteen steps. The segment is short and the air along it varies slowly, so the integrand is
  // nearly linear; the per-step integration below is analytic rather than a Riemann sum, which is
  // what lets the count stay this low.
  const int kSteps = 16;

  const float  d   = length(displacement);
  const float3 dir = displacement / d;
  const float  nu  = dot(dir, sunDirection);

  // Angular terms are constant along a straight segment, so they leave the loop.
  const float rayleighPhase = RayleighPhaseFunction(nu);
  const float miePhase      = MiePhaseFunction(atmosphere.miePhaseFunctionG, nu);
  const float stepLength    = d / float(kSteps);

  float3 inScatter  = float3(0.0F);
  float3 throughput = float3(1.0F);

  // The sun's share, and the streaming pick over it: step i replaces the current choice with
  // probability luminance_i / (running total), one uniform rescaled after each decision.
  float3 sunInScatter = float3(0.0F);
  float  sunTotal     = 0.0F;
  float3 sunPicked    = float3(0.0F);
  float  pickedLum    = 0.0F;
  int    pickedStep   = 0;
  float  pick         = u.x;

  for(int i = 0; i < kSteps; ++i)
  {
    const float3 p        = observer + dir * (float(i) + 0.5F) * stepLength;
    const float  r        = max(length(p), atmosphere.bottomRadius);
    const float  altitude = r - atmosphere.bottomRadius;

    const float rayleighDensity   = GetProfileDensity(atmosphere.rayleighDensity, altitude);
    const float mieDensity        = GetProfileDensity(atmosphere.mieDensity, altitude);
    const float absorptionDensity = GetProfileDensity(atmosphere.absorptionDensity, altitude);

    const float3 rayleighScatter = atmosphere.rayleighScattering * rayleighDensity;
    const float3 mieScatter      = atmosphere.mieScattering * mieDensity;
    // Extinction, not scattering: the aerosol absorbs as well as redirects, and the ozone only
    // absorbs. A segment that used scattering here would never go dark, only bluer.
    const float3 extinction =
        max(rayleighScatter + atmosphere.mieExtinction * mieDensity + atmosphere.absorptionExtinction * absorptionDensity,
            float3(1e-9F));

    // How much sun reaches this point, with the planet's own shadow already in it.
    const float  muSun            = dot(p, sunDirection) / r;
    const float3 sunTransmittance = GetTransmittanceToSun(atmosphere, transmittanceTexture, r, muSun);

    const float3 sunSource =
        atmosphere.solarIrradiance * sunTransmittance * (rayleighScatter * rayleighPhase + mieScatter * miePhase);
    const float3 skySource =
        (rayleighScatter + mieScatter) * GetIrradiance(atmosphere, irradianceTexture, r, muSun) * (1.0F / (4.0F * M_PI));

    // Integrated over the step in closed form, assuming the source is constant across it: the
    // energy-conserving form, which does not lose light as the step count drops.
    const float3 stepTransmittance = exp(-extinction * stepLength);
    const float3 stepFraction      = throughput * (float3(1.0F) - stepTransmittance) / extinction;
    const float3 sunStep           = sunSource * stepFraction;
    inScatter += sunStep + skySource * stepFraction;
    sunInScatter += sunStep;
    throughput *= stepTransmittance;

    const float weight = luminance(sunStep);
    if(weight > 0.0F)
    {
      sunTotal += weight;
      const float keep = weight / sunTotal;
      if(pick < keep)
      {
        sunPicked  = sunStep;
        pickedLum  = weight;
        pickedStep = i;
        pick       = pick / keep;
      }
      else
      {
        pick = (pick - keep) / (1.0F - keep);
      }
    }
  }

  AerialMarch march;
  march.inScatter     = inScatter;
  march.transmittance = throughput;
  march.sunInScatter  = sunInScatter;
  march.sunEstimate   = (pickedLum > 0.0F) ? sunPicked * (sunTotal / pickedLum) : float3(0.0F);
  march.sunPoint      = (float(pickedStep) + u.y) / float(kSteps);
  return march;
}

// What the camera sees of a surface, once the air in between has had its say.
//
//     seen = surfaceRadiance * transmittance + inScatter
//
// `worldPos` is the surface, `cameraPos` the eye, both in world units. The atmosphere works in
// kilometres from the planet's centre, and the two spaces are bridged by the same fixed-observer
// policy the bake uses: the eye sits at the observer altitude above the surface, wherever it is in
// the scene, and the surface sits where the scene puts it *relative* to the eye.
//
// That is the only mapping that can be right here. A scene is a few metres or a few hundred wide
// and stands nowhere in particular on a planet 6371 km in radius; what aerial perspective needs to
// know is not where the scene is but how much air is inside it, and that is the relative
// displacement. It also means the haze does not lurch when the camera moves, which a mapping
// anchored to the scene's origin would do.
//
// The conversion assumes **one scene unit is one metre**, which is what glTF specifies, and
// `aerialPerspectiveScale` multiplies it for the scenes that do not obey -- or for an artist who
// wants more air than the model gives. It is a distance scale rather than an opacity: doubling it
// says "this scene is twice as big", which stays inside the model instead of tinting its output.
//
// The segment is marched in the environment's frame, as the bake is: up is the rotated sky's up, so
// a tilted sky tilts its haze with it.
//
// Applied to the whole radiance the surface sends back, indirect bounces included. That is exact
// for the segment between the eye and the first surface, which is what this models, and it says
// nothing about the air *inside* the scene between later bounces -- correct at the scale a glTF
// asset actually occupies, and the place to look first if someone ever renders a valley.
AerialMarch evalAerialPerspective(float3                  worldPos,
                                  float3                  cameraPos,
                                  SceneFrameInfo*         frameInfo,
                                  SkyAtmosphereParameters atmosphere,
                                  Sampler2D<float3>       transmittanceTexture,
                                  Sampler2D<float3>       irradianceTexture,
                                  float2                  u)
{
  const float  kmPerSceneUnit = 0.001F * frameInfo->aerialPerspectiveScale;
  const float3 displacement   = quatRotateInverse(frameInfo->envRotation, (worldPos - cameraPos) * kmPerSceneUnit);

  // A surface the eye is standing on has no air in front of it, and normalize() of a zero vector
  // is a NaN that would spread across the whole pixel.
  if(dot(displacement, displacement) < 1e-16F)
  {
    AerialMarch none;
    none.inScatter     = float3(0.0F);
    none.transmittance = float3(1.0F);
    none.sunInScatter  = float3(0.0F);
    none.sunEstimate   = float3(0.0F);
    none.sunPoint      = 0.0F;
    return none;
  }

  // Up is +Y in the environment's frame, as it is in the bake: the model only reaches the world
  // through dot products against this vector.
  const float3 observer = float3(0.0F, atmosphere.bottomRadius + frameInfo->observerAltitudeKm, 0.0F);
  const float3 sunDir   = quatRotateInverse(frameInfo->envRotation, frameInfo->sunDirection);
  AerialMarch  march =
      marchAerialPerspective(atmosphere, transmittanceTexture, irradianceTexture, observer, displacement, sunDir, u);

  // The transmittance is dimensionless and needs no conversion; the in-scatter is spectral radiance
  // at three wavelengths and does, exactly as the bake converts what GetSkyRadiance hands it. Miss
  // this and the two halves disagree by orders of magnitude: distant surfaces dim correctly and
  // gain no haze to replace what they lost, so a mountain goes black instead of blue.
  const float3 toScene = SKY_RADIANCE_TO_LUMINANCE * frameInfo->envIntensity;
  march.inScatter *= toScene;
  march.sunInScatter *= toScene;
  march.sunEstimate *= toScene;
  return march;
}

// The rasterizer's form: no scene visibility to trace, so the sun lights the whole segment.
//
//     seen = surfaceRadiance * transmittance + inScatter
float3 applyAerialPerspective(float3                  surfaceRadiance,
                              float3                  worldPos,
                              float3                  cameraPos,
                              SceneFrameInfo*         frameInfo,
                              SkyAtmosphereParameters atmosphere,
                              Sampler2D<float3>       transmittanceTexture,
                              Sampler2D<float3>       irradianceTexture)
{
  const AerialMarch march = evalAerialPerspective(worldPos, cameraPos, frameInfo, atmosphere, transmittanceTexture,
                                                  irradianceTexture, float2(0.5F));
  return surfaceRadiance * march.transmittance + march.inScatter;
}

#endif  // AERIAL_PERSPECTIVE_H
