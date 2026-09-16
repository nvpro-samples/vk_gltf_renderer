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

/*-------------------------------------------------------------------------------------------------
EXT_lights_ies: reshapes a punctual light's angular falloff using a photometric (IES) profile.

The profile is a normalized (peak = 1), azimuthally-averaged 1D candela distribution over the
vertical angle from the light's photometric (forward) axis -- see src/ies_profile.hpp for the
host-side parse and its "Simplification" note on azimuthal averaging. KHR_lights_punctual's own
`intensity` still carries the light's absolute physical scale; this only multiplies it by a
[0,1] shape factor, applied at the call site on top of `singleLightContribution()` (upstream
nvpro_core2/nvshaders/light_contrib.h.slang) rather than forking that file.
-------------------------------------------------------------------------------------------------*/

#ifndef GLTF_LIGHT_IES_H
#define GLTF_LIGHT_IES_H 1

#include "gltf_scene_io.h.slang"

// Linear lookup into GltfScene.iesProfiles at the given vertical angle (degrees, [0,180]).
float sampleIesProfile(GltfScene* scenePtr, int profileIndex, float thetaDeg)
{
  int   sampleCount = scenePtr->iesProfileSampleCount;
  float t           = clamp(thetaDeg, 0.0f, 180.0f) / 180.0f * float(sampleCount - 1);
  int   lo          = int(t);
  int   hi          = min(lo + 1, sampleCount - 1);
  float frac        = t - float(lo);
  int   base        = profileIndex * sampleCount;
  return lerp(scenePtr->iesProfiles[base + lo], scenePtr->iesProfiles[base + hi], frac);
}

// Multiplicative attenuation factor in [0,1] to fold into a punctual light's contribution.
// Returns 1.0 (no-op) for directional lights, or when the light has no IES profile attached.
float evalIesFactor(GltfScene* scenePtr, GltfLight light, float3 surfacePos)
{
  if(light.iesProfile < 0 || light.type == LightType::eLightTypeDirectional)
    return 1.0f;

  float3 toSurface = normalize(surfacePos - light.position);
  float  cosTheta  = clamp(dot(toSurface, light.direction), -1.0f, 1.0f);
  float  thetaDeg  = degrees(acos(cosTheta));
  return sampleIesProfile(scenePtr, light.iesProfile, thetaDeg);
}

#endif
