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

#ifndef SKY_BACKGROUND_H
#define SKY_BACKGROUND_H 1

// The background field for authored (OMI_environment_sky) sky types -- the sky you *see*, as
// opposed to the sky that lights the scene.
//
// The two are separate on purpose. The lat-long bake that feeds importance sampling and the
// prefiltered cubemaps is the *lighting* field, and it has no sun in it: the sun is a
// KHR_lights_punctual directional light that next-event estimation samples directly, so baking it
// there too would light every rough surface with roughly two suns. That leaves the visible sun to
// this file, which evaluates it analytically per ray -- which also keeps the disk sharp, something
// a 1024x512 bake could never do for a half-degree sun.
//
// Both render paths include this header so the two can never drift: the path tracer calls it on
// camera rays and after a delta bounce, and the rasterizer's background pass calls it per pixel.
//
// Who calls this, and when, is the part that has to be exactly right:
//
//   * camera ray                  -> background (you must see the sun)
//   * escaped ray, last bounce was delta/specular
//                                 -> background (a mirror must reflect a sharp sun; NEE cannot
//                                    reach a delta lobe, so nothing double-counts)
//   * escaped ray, any other bounce
//                                 -> lighting field, plus evalSkySunDisk() MIS-weighted against
//                                    the sun's NEE sample. Unweighted, the disk counts the sun
//                                    twice; left out, a lobe narrower than the disk -- smooth glass,
//                                    a polished mirror -- never shows the sun, since NEE's cone
//                                    samples almost never land in it.
//
// Getting that wrong is silent and off by 2x, which is why it is stated here rather than left to
// the call sites.

#include "sky_gradient.h.slang"
#include "sky_omi_shaderio.h.slang"
#include "shaderio.h"
#include "nvshaders/functions.h.slang"  // quatRotateInverse

// The solar disk for the physical sky, or black outside it.
//
// The physical sky is the one type whose background is not evaluated from scratch here: its baked
// image already carries the full scattered sky, aureole included, because that bake is a real
// atmosphere simulation rather than a gradient. The single thing a 1024x512 lat-long cannot hold is
// the half-degree disk, so that is the only thing this adds -- on top of the lat-long lookup, not
// instead of it.
//
// `dir` must be in the same space as that lookup (brought into the environment's frame: quatRotateInverse(envRotation, worldDir)). `sunDirection` is world
// space -- it is also the light -- so it is brought into that frame here, the same way the host
// hands it to the bake, and the disk lands on the aureole around it.
//
// Same call rule as evalAuthoredSkyBackground: unweighted on camera rays and post-delta bounces;
// anywhere else only through evalSkySunDisk() and its MIS weight.
float3 evalSunDiskBackground(float3 dir, SceneFrameInfo* frameInfo)
{
  if(frameInfo->sunDisk == nullptr)
    return float3(0.0F);  // not the physical sky, or the bake has not published a disk yet

  const SkySunDisk disk = *frameInfo->sunDisk;

  // Below the horizon the planet is in the way. The baked image already stops there -- it switches
  // from sky to ground -- so without this the disk is composited over the ground and the sun
  // never sets, it just slides down the picture. The cut is hard because the horizon is: at a
  // metre above the surface there is nothing between you and it.
  if(dir.y <= disk.horizonMu)
    return float3(0.0F);

  const float3 sunDir = quatRotateInverse(frameInfo->envRotation, frameInfo->sunDirection);
  return (dot(dir, sunDir) >= disk.cosAngularRadius) ? disk.radiance : float3(0.0F);
}

// The sky's own sun, as a light.
//
// The sun belongs to the sky, not to the scene. A glTF directional light is a delta light with no
// angular extent -- that is what KHR_lights_punctual specifies -- and a scene may hold several of
// them; none of that makes any one of them a sun. So the sky carries its own, and the scene's
// lights are left exactly as authored, hard shadows included.
//
// Both skies with a sun can describe one, but they arrive at its illuminance by different routes,
// because only one of them measured anything:
//
//   * physical -- the bake reports the disk's radiance in the renderer's own units, so the
//     illuminance is that radiance times the solid angle the disk subtends.
//   * gradient -- `sunColor` is the colour the disk is *drawn* in, not a radiance. Putting it
//     through the same product gives a sun of 7e-5 lux (a half-degree disk subtends 6.8e-5 sr)
//     under a dome delivering about 1.5: a sun you can see and cannot light by. The host scales it
//     against that dome instead -- see GltfRenderer::gradientSunIlluminance().
GltfLight makeSkySunLight(SceneFrameInfo* frameInfo)
{
  float3 illuminance   = float3(0.0F);
  float  angularRadius = 0.0F;

  if(frameInfo->sunDisk != nullptr)
  {
    const SkySunDisk disk = *frameInfo->sunDisk;
    angularRadius         = disk.angularRadius;
    illuminance           = disk.radiance * (M_TWO_PI * (1.0F - cos(angularRadius)));
  }
  else
  {
    // Below the horizon there is no sun: evalGradientSky stops drawing one there, and a light that
    // keeps shining after its disk has gone is a sun that never sets -- it would light the scene
    // from underneath.
    illuminance   = (frameInfo->skyOmi.gradient.sunDirection.y > 0.0F) ? frameInfo->skySunIlluminance : float3(0.0F);
    angularRadius = frameInfo->skyOmi.gradient.sunAngularRadius;
  }

  GltfLight light;
  light.type                  = LightType::eLightTypeDirectional;
  light.direction             = -frameInfo->sunDirection;  // the way the light travels
  light.position              = float3(0.0F);
  light.radius                = 0.0F;
  light.angularSizeOrInvRange = 2.0F * angularRadius;  // the field is the full angle
  light.innerAngle            = 0.0F;
  light.outerAngle            = 0.0F;
  light.iesProfile            = -1;
  // The illuminance is carried in `color` so the sun's colour survives; `intensity` is the scalar
  // the two would otherwise be squeezed into.
  light.color     = illuminance * frameInfo->envIntensity;
  light.intensity = 1.0F;
  return light;
}

// Drive the scene's marked sun light from the sky.
//
// `sunLightIndex` names the one light the scene marks as the sky's sun. That light is the sky's to
// define -- same radiance, same solid angle, same angular size as the disk drawn in the
// background -- so what you see and what casts the shadows are one object. Every other light in
// the scene, directional or not, is used exactly as authored.
//
// The alternative, honouring the file's intensity, would double-attenuate: what a save writes is
// the sun *above* the atmosphere, precisely so a reader that has the atmosphere can apply it. We
// have the atmosphere, so we apply it here rather than trusting a number meant for someone else.
void applySkySunLight(inout GltfLight light, int lightIndex, SceneFrameInfo* frameInfo)
{
  if(lightIndex != frameInfo->sunLightIndex)
    return;

  const GltfLight sun         = makeSkySunLight(frameInfo);
  light.direction             = sun.direction;
  light.color                 = sun.color;
  light.intensity             = sun.intensity;
  light.angularSizeOrInvRange = sun.angularSizeOrInvRange;
}

// True when `envSkyType` names a sky this header can evaluate.
bool hasAuthoredSkyBackground(int envSkyType)
{
  return envSkyType == SkyType::ePlain || envSkyType == SkyType::eGradient;
}

// Whether `dir` lies on the gradient sky's sun disk, and the disk's radiance there.
//
// The radiance of the light NEE samples rather than `sunColor`, which is only a display colour
// (see makeSkySunLight). Otherwise a mirror would show a different sun than a surface of roughness
// 0.01 reflects.
bool gradientSunDisk(float3 dir, SceneFrameInfo* frameInfo, out float3 radiance)
{
  radiance                               = float3(0.0F);
  const SkyGradientParameters g          = frameInfo->skyOmi.gradient;
  const float                 solidAngle = M_TWO_PI * (1.0F - cos(g.sunAngularRadius));
  if(solidAngle <= 0.0F || dir.y <= 0.0F || dot(dir, g.sunDirection) < cos(g.sunAngularRadius))
    return false;
  if(g.sunDirection.y > 0.0F)
    radiance = frameInfo->skySunIlluminance / solidAngle;
  return true;
}

// Solid angle of the sky's sun disk, or 0 for a sky without one. The sun's two MIS weights -- NEE
// in sampleLights(), the escaped ray in computeSkySunHitMisWeight() -- sum to one only if both
// derive its pdf from this one number.
float skySunSolidAngle(SceneFrameInfo* frameInfo)
{
  if(frameInfo->sunDisk != nullptr)
    return M_TWO_PI * (1.0F - frameInfo->sunDisk->cosAngularRadius);
  if(frameInfo->envSkyType == SkyType::eGradient)
    return M_TWO_PI * (1.0F - cos(frameInfo->skyOmi.gradient.sunAngularRadius));
  return 0.0F;
}

// The sky's sun disk along `dir`, before `envIntensity`: the same light makeSkySunLight() hands
// NEE, seen instead of sampled. Black off the disk, and for every sky without a sun. `dir` is in
// the environment frame, as for evalAuthoredSkyBackground.
float3 evalSkySunDisk(float3 dir, SceneFrameInfo* frameInfo)
{
  if(frameInfo->sunDisk != nullptr)
    return evalSunDiskBackground(dir, frameInfo);
  float3 radiance = float3(0.0F);
  if(frameInfo->envSkyType == SkyType::eGradient)
    gradientSunDisk(dir, frameInfo, radiance);
  return radiance;
}

// Visible radiance along `dir` (unit) for an authored sky, sun included, before `envIntensity`.
//
// `dir` is the view direction brought into the environment's frame (quatRotateInverse(envRotation, .)) by the caller, exactly as for a lat-long
// lookup, so the background and the lighting field stay aligned.
float3 evalAuthoredSkyBackground(float3 dir, SceneFrameInfo* frameInfo)
{
  const SkyOmiParameters sky = frameInfo->skyOmi;
  switch(frameInfo->envSkyType)
  {
    case SkyType::ePlain:
      // Plain has no sun term, so its background and lighting fields are the same radiance. It is
      // still evaluated here rather than sampled from the bake, so that every authored type takes
      // one code path and picks up fetch-time scaling identically.
      return sky.plain.color;
    case SkyType::eGradient: {
      // The disk replaces the glow under it; the glow keeps `sunColor`, as it is not part of the light.
      float3 disk;
      if(gradientSunDisk(dir, frameInfo, disk))
        return disk;
      return evalGradientSky(dir, sky.gradient, /*includeSun*/ true);
    }
    default:
      return float3(0.0F);
  }
}

#endif  // SKY_BACKGROUND_H
