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


// Producer for the unified environment-sampling pipeline: the analytic skies.
//
// Bakes a sky evaluator into a lat-long RGBA float image AND writes per-texel importance
// (= solidAngle(row) * max(color.rgb)) into a storage buffer, in one compute pass. The
// two outputs are consumed by `nvvk::HdrIbl::updateFromGpuImage` -- CPU Vose from the
// importance buffer, `env_write_pdf.slang` for the PDF write into image alpha.
//
// The alpha channel is deliberately left at 0 here; env_write_pdf overwrites it after the
// integral is known on the CPU. Keeping the two responsibilities in separate passes lets
// the alias table (built on the CPU from importance) and the PDF (written by GPU from the
// same integral) stay bit-comparable with the file-load path in `createEnvironmentAccel`.
//
// The importance metric matches `createEnvironmentAccel`: `area(y) * max(color.rgb)`.
// Both producers feed the same `buildAliasmap` on the CPU so the alias table is identical
// whether the source was a loaded HDR file or a GPU-baked sky.
//
// Sky-type dispatch is a single switch below, covering every sky this renderer bakes. Each
// non-trivial evaluator lives in its own header (sky_gradient.h.slang, sky_bruneton_functions.h.
// slang) so the switch stays a thin dispatcher.
//
// One switch rather than one shader per sky: the outputs are a contract -- the same lat-long
// mapping, the same importance metric, the same alpha left for env_write_pdf -- and a second
// entry point would have to restate all three and keep restating them. The cost is that set 1
// (Bruneton's LUTs) is bound on every bake, including the analytic ones that never read it.
// SkyBruneton precomputes at startup, so that set is always valid and binding it is free.
//
// What nvpro_core2 owns of all this is the part its own code has to agree with: `latlongToDir`
// and `texelSolidAngle` (nvshaders/functions.h.slang), which live next to the functions that read
// the image back. The wiring -- bindings and push constant -- is ours, in env_bake_io.h.slang,
// and so is every sky in the switch.
//
// What this pass bakes is Field L -- the *lighting* environment. The visible sun disk is
// deliberately absent: the sun is a KHR_lights_punctual directional light that next-event
// estimation samples directly, so baking it here would light every rough surface twice. The
// disk-bearing Field B is evaluated analytically per primary ray by the renderer instead.

#include "nvshaders/hdr_io.h.slang"
#include "nvshaders/constants.h.slang"
#include "nvshaders/functions.h.slang"  // latlongToDir, texelSolidAngle
#include "env_bake_io.h.slang"
#include "shaderio.h"  // SkySunDisk
#include "sky_omi_shaderio.h.slang"
#include "sky_gradient.h.slang"
#include "sky_bruneton_functions.h.slang"
#include "sky_bruneton_io.h.slang"
#include "sky_bruneton_radiance.h.slang"

// clang-format off
[[vk::binding(EnvBakeBindings::eEnvBakeColor, 0)]]      RWTexture2D<float4>                   g_colorImage;
[[vk::binding(EnvBakeBindings::eEnvBakeImportance, 0)]] RWStructuredBuffer<float>             g_importance;
[[vk::binding(EnvBakeBindings::eEnvBakeSkyParams, 0)]]  ConstantBuffer<SkyOmiParameters>      g_skyParams;

// Set 1 is SkyBruneton's runtime set: its four LUTs and the atmosphere they were built from. Read
// by the PhysicalBruneton case alone; bound unconditionally, since one pipeline now serves all.
[[vk::binding(0, 1)]] Sampler2D<float3>                       g_transmittance;
[[vk::binding(1, 1)]] Sampler3D<float4>                       g_scattering;
[[vk::binding(2, 1)]] Sampler3D<float4>                       g_singleMie;
[[vk::binding(3, 1)]] Sampler2D<float3>                       g_irradiance;
[[vk::binding(4, 1)]] ConstantBuffer<SkyAtmosphereParameters> g_atmosphere;

[[vk::push_constant]]                                    ConstantBuffer<EnvBakePushConstant>  pc;
// clang-format on


// -- Sky evaluators ---------------------------------------------------------------------------

// OMI_environment_sky `plain`: one solid color in every direction. Baking a constant is not
// wasted work -- it is what lets the plain sky reuse the alias table, the prefiltered cubemaps
// and the MIS weights unchanged, instead of earning a branch in every sampling path.
float3 evalSkyPlain()
{
  return g_skyParams.plain.color;
}

// OMI_environment_sky `gradient`. `includeSun` is false here: this pass bakes the lighting
// field, and the sun is sampled as a directional light -- see the note at the top of the file.
float3 evalSkyGradient(float3 dir)
{
  return evalGradientSky(dir, g_skyParams.gradient, false);
}


// OMI `physical`: Bruneton (2008) precomputed atmospheric scattering, evaluated from SkyBruneton's
// LUTs -- the only evaluator here that reads textures, which is why set 1 exists. The body is
// shared with the path tracer's per-ray background; see sky_bruneton_radiance.h.slang.
float3 evalSkyBruneton(float3 dir)
{
  return evalBrunetonSkyRadiance(g_atmosphere, g_transmittance, g_scattering, g_singleMie, g_irradiance,
                                 g_atmosphere.bottomRadius + pc.observerAltitude, dir, pc.sunDirection);
}

// Publish the solar disk the physical sky's image deliberately does not contain.
//
// It belongs in the shader rather than on the host because the disk's colour is the solar spectrum
// times the transmittance along the sun's own path through the atmosphere, and that transmittance
// lives in a LUT the host never reads back. And it belongs in the bake rather than in the background
// passes, which bind the same LUTs, because it is one value per sun and atmosphere: the bake already
// runs whenever either changes, so one thread computes it once instead of every pixel every frame.
void publishSunDisk()
{
  SkySunDisk* out = (SkySunDisk*)pc.sunDiskAddress;

  // The observer is on the +Y axis (see sky_bruneton_radiance.h.slang), so its zenith direction is
  // simply +Y and the sun's zenith cosine is the Y component of the sun direction.
  const float  r     = g_atmosphere.bottomRadius + pc.observerAltitude;
  const float  muS   = pc.sunDirection.y;
  const float3 toSun = GetTransmittanceToSun(g_atmosphere, g_transmittance, r, muS);

  // Radiance of the disk itself: irradiance spread over the solid angle it subtends. Upstream's
  // GetSolarRadiance, converted with the *sun's* factors -- same calibration as the sky, so the
  // two stay in one another's units, but a different spectral weighting, because the sun's
  // spectrum is not the sky's. See SUN_SPECTRAL_TO_LUMINANCE.
  const float3 solarRadiance = g_atmosphere.solarIrradiance / (M_PI * g_atmosphere.sunAngularRadius * g_atmosphere.sunAngularRadius);

  out->radiance         = solarRadiance * toSun * SUN_RADIANCE_TO_LUMINANCE;
  out->cosAngularRadius = cos(g_atmosphere.sunAngularRadius);
  out->angularRadius    = g_atmosphere.sunAngularRadius;

  // Where the ground begins, from this observer. Below it the planet is in the way, so whoever
  // draws the disk can stop at the same line the baked image does -- which is what lets a sun
  // set behind the horizon instead of sitting on top of it.
  const float sinHorizon = g_atmosphere.bottomRadius / r;
  out->horizonMu         = -sqrt(max(1.0F - sinHorizon * sinHorizon, 0.0F));

  // Where the observer stood, so a per-ray evaluation of the sky stands there too and its horizon
  // meets the one baked into the lighting environment.
  out->observerRadius = r;
}


// -- Entry point ------------------------------------------------------------------------------

// One importance value per texel of this workgroup, for the per-cell sum below.
groupshared float s_importance[HDR_WORKGROUP_SIZE * HDR_WORKGROUP_SIZE];

[shader("compute")]
[numthreads(HDR_WORKGROUP_SIZE, HDR_WORKGROUP_SIZE, 1)]
void main(uint3 threadIdx: SV_DispatchThreadID, uint3 localIdx: SV_GroupThreadID)
{
  // No early return: every thread has to reach the barrier below. One outside the image (none
  // is, at kEnvBakeSize) contributes nothing.
  const bool inside = all(threadIdx.xy < pc.imageSize);

  float importance = 0.0F;
  if(inside)
  {
    const float2 pixelCenter = float2(threadIdx.xy) + float2(0.5F);
    const float2 uv          = pixelCenter / float2(pc.imageSize);
    const float3 dir         = latlongToDir(uv);

    float3 color;
    switch(pc.skyType)
    {
      case SkyType::ePlain:
        color = evalSkyPlain();
        break;
      case SkyType::eGradient:
        color = evalSkyGradient(dir);
        break;
      case SkyType::ePhysicalBruneton:
        color = evalSkyBruneton(dir);
        break;
      default:
        color = float3(0.0F);
        break;
    }

    g_colorImage[threadIdx.xy] = float4(color, 0.0F);  // alpha overwritten by env_write_pdf

    // Only the physical sky has a disk to publish; the gradient evaluates its own, and plain has
    // none. See the note on publishSunDisk for why one thread does this here.
    if(pc.skyType == SkyType::ePhysicalBruneton && pc.sunDiskAddress != 0 && all(threadIdx.xy == uint2(0, 0)))
    {
      publishSunDisk();
    }

    const float area    = texelSolidAngle(threadIdx.y, pc.imageSize.x, pc.imageSize.y);
    const float maxChan = max(color.r, max(color.g, color.b));
    importance          = area * maxChan;
  }

  // Sum the importance per cell of the sampling grid, which is what the alias table is built over.
  // A cell's texels all sit in one workgroup -- the block divides the workgroup, which EnvBaker
  // checks at compile time -- so the sum stays in shared memory: one thread per cell adds its
  // block and writes the cell. No atomics, so no float-atomic extension, and a deterministic sum.
  s_importance[localIdx.y * HDR_WORKGROUP_SIZE + localIdx.x] = importance;
  GroupMemoryBarrierWithGroupSync();

  const uint2 block = pc.imageSize / pc.gridSize;  // texels per cell
  if(inside && all(localIdx.xy % block == 0))
  {
    float sum = 0.0F;
    for(uint y = 0; y < block.y; ++y)
      for(uint x = 0; x < block.x; ++x)
        sum += s_importance[(localIdx.y + y) * HDR_WORKGROUP_SIZE + localIdx.x + x];

    const uint2 cell                              = threadIdx.xy / block;
    g_importance[cell.y * pc.gridSize.x + cell.x] = sum;
  }
}
