/*
 * Copyright (c) 2017 Eric Bruneton
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holders nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Ported to Slang/Vulkan for vk_gltf_renderer.
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef SKY_BRUNETON_IO_H
#define SKY_BRUNETON_IO_H 1

//
// Host/device parameter block for the Bruneton (2008) precomputed atmospheric scattering model.
//
// Ported from `atmosphere/definitions.glsl` in
// github.com/ebruneton/precomputed_atmospheric_scattering. Upstream declares these structs in
// GLSL behind dimensional-analysis `#define`s (`Length`, `Number`, `InverseLength`, ...) that let
// the same file compile as C++ for offline validation. That scaffolding is not ported: this
// project compiles Slang only, so the aliases are spelled out as `float` / `float3` and the
// structs are shared with the host verbatim.
//
// The other half of that change is that upstream has no parameter *buffer* at all -- `Model::Init`
// synthesizes GLSL source at runtime with every value `std::to_string`'d into a module-scope
// `const AtmosphereParameters ATMOSPHERE = ...`. Slang here is compiled at CMake time, so the
// values arrive in this uniform block instead. The evaluation functions are unaffected: upstream
// already passes `atmosphere` as an argument to every one of them.
//
// Units are the paper's: lengths in **kilometres**, angles in radians, and the "spectrum" triples
// are RGB (the spectral path is not ported). Keep them that way -- the LUT parameterisation is
// built around the planet radii being small numbers.
//

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

// LUT dimensions, from `atmosphere/constants.h`. These are baked into the LUT parameterisation in
// sky_bruneton_functions.h.slang, so they are compile-time constants on both sides rather than
// parameters -- changing one means rebuilding every LUT.
#define TRANSMITTANCE_TEXTURE_WIDTH 256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64

#define SCATTERING_TEXTURE_R_SIZE 32
#define SCATTERING_TEXTURE_MU_SIZE 128
#define SCATTERING_TEXTURE_MU_S_SIZE 32
#define SCATTERING_TEXTURE_NU_SIZE 8

// The 4D scattering function is stored as a 3D texture, with nu and mu_s folded into width.
#define SCATTERING_TEXTURE_WIDTH (SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE)
#define SCATTERING_TEXTURE_HEIGHT SCATTERING_TEXTURE_MU_SIZE
#define SCATTERING_TEXTURE_DEPTH SCATTERING_TEXTURE_R_SIZE

#define IRRADIANCE_TEXTURE_WIDTH 64
#define IRRADIANCE_TEXTURE_HEIGHT 16

// Workgroup size for every precompute dispatch. 8x8 covers the 2D LUTs; the 3D passes dispatch
// one slice per z and use the same tile.
#define SKY_BRUNETON_WORKGROUP_SIZE 8

// Bindings of the transient descriptor set used only while the LUTs are being built. The runtime
// set -- the four finished LUTs that evaluation samples -- is declared separately, because
// rendering binds only that one; the intermediates stay alive for the next rebuild.
// One set serves all six passes. Every binding is written for every dispatch even when the pass
// does not use it -- a declared binding must have a valid descriptor -- which is cheaper to reason
// about than six layouts that differ by two entries each.
//
// Read and write views of the same LUT are separate bindings because they are different descriptor
// types: a pass that samples with interpolation needs a combined image sampler, one that writes
// needs a storage image, and several passes do both to different LUTs in the same dispatch.
enum SkyBrunetonPrecomputeBindings
{
  eBrunetonParams = 0,  // ConstantBuffer<SkyAtmosphereParameters>

  // Storage images, written by whichever pass owns them.
  eOutTransmittance     = 1,
  eOutDeltaIrradiance   = 2,
  eOutIrradiance        = 3,  // accumulated across orders
  eOutDeltaRayleigh     = 4,
  eOutDeltaMie          = 5,
  eOutScattering        = 6,  // accumulated across orders
  eOutSingleMie         = 7,
  eOutScatteringDensity = 8,
  eOutDeltaMultiple     = 9,

  // Sampled views, read by later passes.
  eInTransmittance     = 10,
  eInDeltaIrradiance   = 11,
  eInDeltaRayleigh     = 12,
  eInDeltaMie          = 13,
  eInDeltaMultiple     = 14,
  eInScatteringDensity = 15,
};

// Per-dispatch state. The 3D passes dispatch one slice at a time, matching upstream's
// layer-by-layer rendering, so `layer` selects the slice and `scatteringOrder` the bounce count.
struct SkyBrunetonPushConstant
{
  int layer;
  int scatteringOrder;
  // Zenith samples in the scattering-density integral (azimuth is twice this). Runtime rather than
  // a constant because its cost is quadratic and it is the only dial that moves the precomputation
  // between "interactive" and "reference" -- see SkyBruneton::Quality.
  int densitySamples;
  int _pad;
};

// Radiance -> luminance, per channel, divided by MAX_LUMINOUS_EFFICACY (683).
//
// GetSkyRadiance returns spectral radiance at 680/550/440 nm. Turning that into something a
// renderer can display needs the CIE colour-matching integral upstream performs in
// ComputeSpectralRadianceToLuminanceFactors: without it the sky is not merely dim, it is the wrong
// colour, because the three factors differ by a wide margin -- red is boosted 61% relative to
// green.
//
// Computed offline from upstream's own CIE table, XYZ_TO_SRGB matrix and solar spectrum, following
// that function exactly, rather than vendoring 380 entries of colour-matching data for a
// calculation that runs once. The raw factors are 114975 / 71306 / 65311, divided here by
// MAX_LUMINOUS_EFFICACY (683) -- this port's own normalisation, so the calibration below starts
// from factors near 100 rather than 100000.
//
// This part is physics. The ratios between the channels are what fix the sky's colour, and
// nothing below touches them.
#define SKY_SPECTRAL_TO_LUMINANCE float3(168.3387F, 104.4009F, 95.6237F)

// The same conversion for the *sun*, which is a different set of numbers and not an oversight.
//
// Upstream computes the factors twice, with different spectral weightings:
//
//     ComputeSpectralRadianceToLuminanceFactors(..., -3 /* lambda_power */, &sky_k_r, ...);
//     ComputeSpectralRadianceToLuminanceFactors(...,  0 /* lambda_power */, &sun_k_r, ...);
//
// The -3 exists because the sky's radiance is blue-biased -- Rayleigh scattering goes as a steep
// inverse power of wavelength -- so reconstructing luminance from three samples of *that* spectrum
// needs red weighted up to compensate. The sun's spectrum has no such bias, so it needs the flat
// weighting. Using the sky's set for the sun over-boosts red: it put the sun above the atmosphere
// at ~5500 K where the real one is ~5800 K, and the error compounded through the air mass, which
// is why a sunlit scene came out amber.
//
// Both sets carry the same units, which is what made the mistake easy: only the *ratios* differ,
// and the ratios are the colour. Sky is 1.612 : 1 : 0.916, sun is 1.404 : 1 : 0.950 -- about 15%
// more red in the sky's set.
//
// Upstream computes both at runtime from its own CIE and solar-spectrum tables rather than
// publishing constants, and this port vendors neither. utils/sky_luminance_factors.py reads those
// tables from upstream's source and runs the same function; it reproduces both lines exactly.
#define SUN_SPECTRAL_TO_LUMINANCE float3(143.8401F, 102.4223F, 97.3280F)

// And this part is not: it is the unit the renderer works in.
//
// Absolute radiometric scale is a convention, only ratios are physical, and this renderer's
// convention is set by the HDR files people load -- they are authored to look right at exposure
// 1.0 and their environment integrals land between 8 and 12 (std_env 8.1, env3 9.6, daytime 12.3;
// the integral is what HdrIbl reports and what the firefly clamp keys off). An uncalibrated
// Bruneton sky integrates to 90.6 at the default sun elevation, so it arrives roughly nine times
// hot and every scene needs its exposure pulled down to 0.1 before it looks like daylight.
//
// Nine is therefore the number: it puts the default sky at an integral of 10, in the middle of
// that band, and the whole sky scales together -- ground, aureole, solar disk, and the sun light
// derived from it -- so every ratio the model computes survives.
//
// The previous MDL sky did exactly this with two constants of its own (rgbUnitConversion 1/80000
// times multiplier 0.1). This is the same decision, stated once and explained.
//
// For physical output: 683.0F gives luminance in cd/m^2 and illuminance in lux, at which point
// exposure has to carry the remaining five orders of magnitude. 1.0F gives upstream's
// precomputed-illuminance units.
#define SKY_LUMINANCE_CALIBRATION (1.0F / 9.0F)

#define SKY_RADIANCE_TO_LUMINANCE (SKY_SPECTRAL_TO_LUMINANCE * SKY_LUMINANCE_CALIBRATION)
// Same calibration, so the disk and the sky it sits in stay in one another's units.
#define SUN_RADIANCE_TO_LUMINANCE (SUN_SPECTRAL_TO_LUMINANCE * SKY_LUMINANCE_CALIBRATION)

// One layer of an atmosphere density profile. Density at altitude h is
//   expTerm * exp(expScale * h) + linearTerm * h + constantTerm
// clamped to [0,1]. A profile is two layers; the first applies below `width`, the second above.
struct DensityProfileLayer
{
  float width        SLANG_DEFAULT(0.0F);  // km; 0 means "this layer runs up to the next one"
  float expTerm      SLANG_DEFAULT(0.0F);  // dimensionless
  float expScale     SLANG_DEFAULT(0.0F);  // 1/km
  float linearTerm   SLANG_DEFAULT(0.0F);  // 1/km
  float constantTerm SLANG_DEFAULT(0.0F);  // dimensionless
  float _pad0        SLANG_DEFAULT(0.0F);  // std140: keep what follows 16-byte aligned on both sides
  float _pad1        SLANG_DEFAULT(0.0F);
  float _pad2        SLANG_DEFAULT(0.0F);
};

struct DensityProfile
{
  DensityProfileLayer layers[2];
};

// Everything the model needs to describe one atmosphere.
//
// Field order follows upstream so the two can be read side by side; padding is inserted only where
// std140 requires it. `float3` members are followed by a scalar on purpose -- that fills the vec4
// slot the layout rules reserve anyway, rather than wasting it.
//
// Defaults are Earth, derived from upstream's own spectral data rather than copied from a
// secondary source: `demo/demo.cc`'s 48-sample solar and ozone tables sampled at Model::kLambdaR/G/B
// (680/550/440 nm, all exact grid points so no interpolation is involved), Rayleigh as
// 1.24062e-6 * lambda^-4, Mie as 5.328e-3/1200 with a single-scattering albedo of 0.9, and every
// coefficient converted from 1/m to 1/km. They reproduce the RGB constants published for
// Frostbite's atmosphere to every digit shown, which is the cross-check that the derivation is
// right.
struct SkyAtmosphereParameters
{
  // Solar irradiance at the top of the atmosphere, W/m^2, per RGB channel.
  float3 solarIrradiance SLANG_DEFAULT(float3(1.474F, 1.8504F, 1.91198F));
  // Angular radius of the sun disk, radians. Earth's is 0.00935/2.
  float sunAngularRadius SLANG_DEFAULT(0.004675F);

  // Rayleigh scattering coefficient at sea level, 1/km, per channel.
  float3 rayleighScattering SLANG_DEFAULT(float3(0.00580234F, 0.0135578F, 0.0331F));
  float bottomRadius        SLANG_DEFAULT(6360.0F);  // planet radius, km

  // Mie scattering and extinction at sea level, 1/km. Extinction >= scattering; the difference is
  // absorption.
  // Raised from upstream's 0.003996 -- see Settings::atmoMieScattering. Kept in step with the
  // Earth preset and the settings default so all three describe the same planet.
  float3 mieScattering SLANG_DEFAULT(float3(0.06F));
  float topRadius      SLANG_DEFAULT(6420.0F);  // atmosphere outer radius, km

  float3 mieExtinction    SLANG_DEFAULT(float3(0.0666667F));  // mieScattering / 0.9 single-scattering albedo
  float miePhaseFunctionG SLANG_DEFAULT(0.8F);                // Cornette-Shanks asymmetry, dimensionless

  // Extinction of the absorbing layer -- ozone on Earth, at 300 Dobson units, 1/km.
  float3 absorptionExtinction SLANG_DEFAULT(float3(0.000649717F, 0.0018809F, 8.50167e-05F));
  // Cosine of the largest sun-zenith angle the LUTs cover (120 degrees). Past that the sun is far
  // enough below the horizon that the sky is black, and storing it would only waste resolution.
  float muSMin SLANG_DEFAULT(-0.5F);

  // Average ground reflectance, which the multiple-scattering orders bounce off.
  float3 groundAlbedo SLANG_DEFAULT(float3(0.1F));
  float _pad0         SLANG_DEFAULT(0.0F);

  // These three carry no in-struct defaults -- an array of structs cannot express one that both
  // sides agree on -- so a default-constructed parameter block has *empty* atmosphere profiles and
  // renders black. `skyBrunetonEarthDefaults()` in src/sky_bruneton.hpp fills them; call it rather
  // than value-initialising this struct and hoping.
  //
  // Upstream pads a single-layer profile by inserting an all-zero layer *before* it, so the real
  // layer is always the second one. Matching that exactly matters: the shader picks layer 0 below
  // `layers[0].width` and layer 1 above, and a zero-width first layer makes the second cover
  // everything.
  DensityProfile rayleighDensity;
  DensityProfile mieDensity;
  DensityProfile absorptionDensity;
};

NAMESPACE_SHADERIO_END()

#endif  // SKY_BRUNETON_IO_H
