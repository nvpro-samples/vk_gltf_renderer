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
// The physical sky's radiance toward a direction, sun excluded -- one function for the two places
// that need it: the bake (env_bake.slang), which writes it into the lighting environment, and the
// path tracer, which evaluates it per ray for what the camera and mirrors see
// (evalPhysicalSkyBackground in gltf_pathtrace.slang). Sharing it is what keeps the lit scene and
// the visible sky the same sky; a second copy would drift silently.
//
// The observer is fixed: a point on the +Y axis, `observerRadius` km from the planet's centre,
// wherever the scene camera is. That is what lets the sky be baked at all -- an observer that moved
// with the camera would invalidate the image every frame -- and a per-ray caller has to stand where
// the bake stood, or its horizon would not meet the baked one. The bake publishes the radius it
// used in SkySunDisk::observerRadius for exactly that. Up is +Y because the rest of the renderer is
// Y-up; the model only ever uses the observer through dot products, so the axis is ours to pick.
//

#ifndef SKY_BRUNETON_RADIANCE_H
#define SKY_BRUNETON_RADIANCE_H

#include "nvshaders/constants.h.slang"  // M_PI
#include "sky_bruneton_functions.h.slang"
#include "sky_bruneton_io.h.slang"

// `dir` and `sunDirection` are unit vectors in the environment's frame. Returns luminance, in the
// same units as the baked image.
float3 evalBrunetonSkyRadiance(SkyAtmosphereParameters atmosphere,
                               Sampler2D<float3>       transmittanceTexture,
                               Sampler3D<float4>       scatteringTexture,
                               Sampler3D<float4>       singleMieScatteringTexture,
                               Sampler2D<float3>       irradianceTexture,
                               float                   observerRadius,
                               float3                  dir,
                               float3                  sunDirection)
{
  const float3 observer = float3(0.0F, observerRadius, 0.0F);

  // Spectral radiance at three wavelengths is not a colour until the CIE integral is applied.
  // Skipping it leaves the sky both far too dark and too blue, since the three factors differ.
  float3       transmittance;
  const float3 skyRadiance = GetSkyRadiance(atmosphere, transmittanceTexture, scatteringTexture,
                                            singleMieScatteringTexture, observer, dir, sunDirection, transmittance);
  float3       color       = skyRadiance * SKY_RADIANCE_TO_LUMINANCE;

  // Below the horizon, add the ground.
  //
  // GetSkyRadiance already returns the right thing for a downward ray -- the scattering LUT is
  // parameterised by whether the ray meets the ground, so what comes back is the light scattered
  // into the segment between the eye and the ground, and nothing beyond it. From an observer a
  // metre up that segment is short, which is why this half of the sphere was almost black: the
  // model was never asked what the ground itself reflects.
  //
  // So that is the only term missing. A Lambertian surface lit by sun and sky, attenuated by the
  // air between it and the eye -- and because the in-scatter above is already integrated over the
  // same segment, the ground fades into the horizon haze instead of ending at a seam. It is the
  // same aerial perspective the renderers apply per ray, for the one case this function can answer
  // on its own: a fixed observer looking at its own planet.
  const float mu = dir.y;  // the observer sits on +Y, so its zenith is +Y and mu is just dir.y
  if(RayIntersectsGround(atmosphere, observerRadius, mu))
  {
    const float  d      = DistanceToBottomAtmosphereBoundary(atmosphere, observerRadius, mu);
    const float3 point  = observer + dir * d;
    const float3 normal = normalize(point);

    float3       skyIrradiance;
    const float3 sunIrradiance = GetSunAndSkyIrradiance(atmosphere, transmittanceTexture, irradianceTexture, point,
                                                        normal, sunDirection, skyIrradiance);

    // Lambertian: albedo/pi times the irradiance it receives. Each irradiance is converted with
    // the factors for its own spectrum, as upstream's GetSunAndSkyIlluminance does: sunlight with
    // the sun's, skylight with the sky's.
    const float3 groundIlluminance = sunIrradiance * SUN_RADIANCE_TO_LUMINANCE + skyIrradiance * SKY_RADIANCE_TO_LUMINANCE;
    const float3 groundColor = atmosphere.groundAlbedo * (1.0F / M_PI) * groundIlluminance;

    color += groundColor * GetTransmittance(atmosphere, transmittanceTexture, observerRadius, mu, d, true);
  }

  return color;
}

#endif  // SKY_BRUNETON_RADIANCE_H
