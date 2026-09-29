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

#ifndef SKY_BRUNETON_FUNCTIONS_H
#define SKY_BRUNETON_FUNCTIONS_H 1

//
// Bruneton (2008) precomputed atmospheric scattering -- the maths.
//
// A direct port of `atmosphere/functions.glsl` from
// github.com/ebruneton/precomputed_atmospheric_scattering. **The algorithm is copied, not
// redesigned**: same formulae, same sample counts, same LUT parameterisation, same function names,
// in the same order as upstream so the two can be read side by side. Only the syntax layer moves:
//
//   * `Length` / `Number` / `InverseLength` / `Area` and friends -> `float`; the dimensional
//     analysis exists upstream so the file can also compile as C++ for offline validation, which
//     is not a path this project has.
//   * `DimensionlessSpectrum` / `IrradianceSpectrum` / `RadianceSpectrum` / ... -> `float3` (RGB;
//     the spectral path is not ported).
//   * `vec2`/`vec3` -> `float2`/`float3`, `IN(x)`/`OUT(x)` -> `in`/`out`.
//   * `TransmittanceTexture` -> `Sampler2D<float3>` and `ScatteringTexture` -> `Sampler3D<float4>`,
//     with `texture(t, uv)` -> `t.SampleLevel(uv, 0)`. Slang's combined texture-sampler types keep
//     these signatures as close to upstream's as the language allows.
//   * `assert(...)` dropped -- upstream's own macro compiles it out in GLSL too.
//   * The `ATMOSPHERE` module-scope constant does not exist here; every function already took
//     `atmosphere` as a parameter upstream, and now it comes from a uniform block.
//
// Units are kilometres and radians, so upstream's `* m` / `/ rad` unit tags simply vanish.
//
// Where a comment explains *why* something is done, it is upstream's reasoning preserved. Where it
// explains a porting decision, it says so.
//

#include "nvshaders/constants.h.slang"  // M_PI, used by the phase functions
#include "sky_bruneton_io.h.slang"

//-------------------------------------------------------------------------------------------------
// Small helpers (upstream: "Physical units" / utility section)

float ClampCosine(float mu)
{
  return clamp(mu, -1.0F, 1.0F);
}

float ClampDistance(float d)
{
  return max(d, 0.0F);
}

float ClampRadius(SkyAtmosphereParameters atmosphere, float r)
{
  return clamp(r, atmosphere.bottomRadius, atmosphere.topRadius);
}

// sqrt of a quantity that is non-negative in exact arithmetic but can go slightly negative through
// rounding -- which would otherwise produce NaNs along the horizon.
float SafeSqrt(float a)
{
  return sqrt(max(a, 0.0F));
}

//-------------------------------------------------------------------------------------------------
// Distances and intersections

// Distance from a point at radius r, looking along a direction with cosine mu to the zenith, to the
// top of the atmosphere.
float DistanceToTopAtmosphereBoundary(SkyAtmosphereParameters atmosphere, float r, float mu)
{
  const float discriminant = r * r * (mu * mu - 1.0F) + atmosphere.topRadius * atmosphere.topRadius;
  return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

// Same, to the ground. Assumes the ray does intersect it -- ask RayIntersectsGround first.
float DistanceToBottomAtmosphereBoundary(SkyAtmosphereParameters atmosphere, float r, float mu)
{
  const float discriminant = r * r * (mu * mu - 1.0F) + atmosphere.bottomRadius * atmosphere.bottomRadius;
  return ClampDistance(-r * mu - SafeSqrt(discriminant));
}

bool RayIntersectsGround(SkyAtmosphereParameters atmosphere, float r, float mu)
{
  return mu < 0.0F && r * r * (mu * mu - 1.0F) + atmosphere.bottomRadius * atmosphere.bottomRadius >= 0.0F;
}

//-------------------------------------------------------------------------------------------------
// Density profiles

float GetLayerDensity(DensityProfileLayer layer, float altitude)
{
  const float density = layer.expTerm * exp(layer.expScale * altitude) + layer.linearTerm * altitude + layer.constantTerm;
  return clamp(density, 0.0F, 1.0F);
}

float GetProfileDensity(DensityProfile profile, float altitude)
{
  return altitude < profile.layers[0].width ? GetLayerDensity(profile.layers[0], altitude) :
                                              GetLayerDensity(profile.layers[1], altitude);
}

// Integral of the density profile along the ray, by the trapezoidal rule. The sample count is
// upstream's; it is only paid during precomputation.
float ComputeOpticalLengthToTopAtmosphereBoundary(SkyAtmosphereParameters atmosphere, DensityProfile profile, float r, float mu)
{
  const int   SAMPLE_COUNT = 500;
  const float dx           = DistanceToTopAtmosphereBoundary(atmosphere, r, mu) / float(SAMPLE_COUNT);

  float result = 0.0F;
  for(int i = 0; i <= SAMPLE_COUNT; ++i)
  {
    const float d_i = float(i) * dx;
    // Distance between the planet centre and the sample point.
    const float r_i = sqrt(d_i * d_i + 2.0F * r * mu * d_i + r * r);
    // Density at that sample, and its weight in the trapezoidal rule.
    const float y_i      = GetProfileDensity(profile, r_i - atmosphere.bottomRadius);
    const float weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5F : 1.0F;
    result += y_i * weight_i * dx;
  }
  return result;
}

float3 ComputeTransmittanceToTopAtmosphereBoundary(SkyAtmosphereParameters atmosphere, float r, float mu)
{
  return exp(-(atmosphere.rayleighScattering * ComputeOpticalLengthToTopAtmosphereBoundary(atmosphere, atmosphere.rayleighDensity, r, mu)
               + atmosphere.mieExtinction * ComputeOpticalLengthToTopAtmosphereBoundary(atmosphere, atmosphere.mieDensity, r, mu)
               + atmosphere.absorptionExtinction
                     * ComputeOpticalLengthToTopAtmosphereBoundary(atmosphere, atmosphere.absorptionDensity, r, mu)));
}

//-------------------------------------------------------------------------------------------------
// Texture coordinate mapping
//
// Texture samples sit at texel centres, so storing f(0) and f(1) at the centres of the first and
// last texels -- rather than at the texture edges, where they would only ever be extrapolated --
// needs this half-texel inset. Every LUT in the model uses it.

float GetTextureCoordFromUnitRange(float x, int texture_size)
{
  return 0.5F / float(texture_size) + x * (1.0F - 1.0F / float(texture_size));
}

float GetUnitRangeFromTextureCoord(float u, int texture_size)
{
  return (u - 0.5F / float(texture_size)) / (1.0F - 1.0F / float(texture_size));
}

//-------------------------------------------------------------------------------------------------
// Transmittance LUT

float2 GetTransmittanceTextureUvFromRMu(SkyAtmosphereParameters atmosphere, float r, float mu)
{
  // Distance to the top boundary for a horizontal ray at the ground.
  const float H = sqrt(atmosphere.topRadius * atmosphere.topRadius - atmosphere.bottomRadius * atmosphere.bottomRadius);
  // Distance to the horizon.
  const float rho = SafeSqrt(r * r - atmosphere.bottomRadius * atmosphere.bottomRadius);
  // Distance to the top boundary for the ray (r, mu), and its min/max over all mu.
  const float d     = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
  const float d_min = atmosphere.topRadius - r;
  const float d_max = rho + H;
  const float x_mu  = (d - d_min) / (d_max - d_min);
  const float x_r   = rho / H;
  return float2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
                GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
}

void GetRMuFromTransmittanceTextureUv(SkyAtmosphereParameters atmosphere, float2 uv, out float r, out float mu)
{
  const float x_mu = GetUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
  const float x_r  = GetUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);

  const float H = sqrt(atmosphere.topRadius * atmosphere.topRadius - atmosphere.bottomRadius * atmosphere.bottomRadius);
  const float rho = H * x_r;
  r               = sqrt(rho * rho + atmosphere.bottomRadius * atmosphere.bottomRadius);

  const float d_min = atmosphere.topRadius - r;
  const float d_max = rho + H;
  const float d     = d_min + x_mu * (d_max - d_min);
  mu                = (d == 0.0F) ? 1.0F : (H * H - rho * rho - d * d) / (2.0F * r * d);
  mu                = ClampCosine(mu);
}

// One texel of the transmittance LUT. `fragCoord` is the texel centre, i.e. pixel index + 0.5.
float3 ComputeTransmittanceToTopAtmosphereBoundaryTexture(SkyAtmosphereParameters atmosphere, float2 fragCoord)
{
  const float2 TRANSMITTANCE_TEXTURE_SIZE = float2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
  float        r;
  float        mu;
  GetRMuFromTransmittanceTextureUv(atmosphere, fragCoord / TRANSMITTANCE_TEXTURE_SIZE, r, mu);
  return ComputeTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu);
}

//-------------------------------------------------------------------------------------------------
// Transmittance lookups

float3 GetTransmittanceToTopAtmosphereBoundary(SkyAtmosphereParameters atmosphere, Sampler2D<float3> transmittanceTexture, float r, float mu)
{
  const float2 uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu);
  return transmittanceTexture.SampleLevel(uv, 0.0F);
}

// Transmittance between two points inside the atmosphere, as the ratio of their transmittances to
// the top boundary. Two lookups rather than an integration.
float3 GetTransmittance(SkyAtmosphereParameters atmosphere, Sampler2D<float3> transmittanceTexture, float r, float mu, float d, bool rayRMuIntersectsGround)
{
  const float r_d  = ClampRadius(atmosphere, sqrt(d * d + 2.0F * r * mu * d + r * r));
  const float mu_d = ClampCosine((r * mu + d) / r_d);

  if(rayRMuIntersectsGround)
  {
    return min(GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittanceTexture, r_d, -mu_d)
                   / GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittanceTexture, r, -mu),
               float3(1.0F));
  }
  return min(GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittanceTexture, r, mu)
                 / GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittanceTexture, r_d, mu_d),
             float3(1.0F));
}

// Transmittance to the sun, softened across the solar disk: the sun is not a point, so near the
// horizon part of it is occluded by the planet while part is not.
float3 GetTransmittanceToSun(SkyAtmosphereParameters atmosphere, Sampler2D<float3> transmittanceTexture, float r, float mu_s)
{
  const float sin_theta_h = atmosphere.bottomRadius / r;
  const float cos_theta_h = -sqrt(max(1.0F - sin_theta_h * sin_theta_h, 0.0F));
  return GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittanceTexture, r, mu_s)
         * smoothstep(-sin_theta_h * atmosphere.sunAngularRadius, sin_theta_h * atmosphere.sunAngularRadius, mu_s - cos_theta_h);
}

//-------------------------------------------------------------------------------------------------
// Single scattering
//
// The solar irradiance, the scattering coefficients and the phase functions are deliberately left
// out of the integrand and applied afterwards -- the phase functions only at render time, which is
// what buys the angular precision a 4D LUT could not store.

void ComputeSingleScatteringIntegrand(SkyAtmosphereParameters atmosphere,
                                      Sampler2D<float3>       transmittanceTexture,
                                      float                   r,
                                      float                   mu,
                                      float                   mu_s,
                                      float                   nu,
                                      float                   d,
                                      bool                    rayRMuIntersectsGround,
                                      out float3              rayleigh,
                                      out float3              mie)
{
  const float  r_d           = ClampRadius(atmosphere, sqrt(d * d + 2.0F * r * mu * d + r * r));
  const float  mu_s_d        = ClampCosine((r * mu_s + d * nu) / r_d);
  const float3 transmittance = GetTransmittance(atmosphere, transmittanceTexture, r, mu, d, rayRMuIntersectsGround)
                               * GetTransmittanceToSun(atmosphere, transmittanceTexture, r_d, mu_s_d);
  rayleigh = transmittance * GetProfileDensity(atmosphere.rayleighDensity, r_d - atmosphere.bottomRadius);
  mie      = transmittance * GetProfileDensity(atmosphere.mieDensity, r_d - atmosphere.bottomRadius);
}

float DistanceToNearestAtmosphereBoundary(SkyAtmosphereParameters atmosphere, float r, float mu, bool rayRMuIntersectsGround)
{
  return rayRMuIntersectsGround ? DistanceToBottomAtmosphereBoundary(atmosphere, r, mu) :
                                  DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
}

void ComputeSingleScattering(SkyAtmosphereParameters atmosphere,
                             Sampler2D<float3>       transmittanceTexture,
                             float                   r,
                             float                   mu,
                             float                   mu_s,
                             float                   nu,
                             bool                    rayRMuIntersectsGround,
                             out float3              rayleigh,
                             out float3              mie)
{
  const int   SAMPLE_COUNT = 50;
  const float dx = DistanceToNearestAtmosphereBoundary(atmosphere, r, mu, rayRMuIntersectsGround) / float(SAMPLE_COUNT);

  float3 rayleigh_sum = float3(0.0F);
  float3 mie_sum      = float3(0.0F);
  for(int i = 0; i <= SAMPLE_COUNT; ++i)
  {
    const float d_i = float(i) * dx;
    float3      rayleigh_i;
    float3      mie_i;
    ComputeSingleScatteringIntegrand(atmosphere, transmittanceTexture, r, mu, mu_s, nu, d_i, rayRMuIntersectsGround,
                                     rayleigh_i, mie_i);
    const float weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5F : 1.0F;
    rayleigh_sum += rayleigh_i * weight_i;
    mie_sum += mie_i * weight_i;
  }
  rayleigh = rayleigh_sum * dx * atmosphere.solarIrradiance * atmosphere.rayleighScattering;
  mie      = mie_sum * dx * atmosphere.solarIrradiance * atmosphere.mieScattering;
}

float RayleighPhaseFunction(float nu)
{
  const float k = 3.0F / (16.0F * M_PI);
  return k * (1.0F + nu * nu);
}

float MiePhaseFunction(float g, float nu)
{
  const float k = 3.0F / (8.0F * M_PI) * (1.0F - g * g) / (2.0F + g * g);
  return k * (1.0F + nu * nu) / pow(1.0F + g * g - 2.0F * g * nu, 1.5F);
}

//-------------------------------------------------------------------------------------------------
// The 4D scattering LUT, stored as 3D
//
// Scattering depends on four parameters (r, mu, mu_s, nu). There are no 4D textures, so nu and
// mu_s share the width: the texture is NU_SIZE slabs of MU_S_SIZE columns, and a lookup samples two
// neighbouring slabs and blends them, which is quadrilinear interpolation by hand.

float4 GetScatteringTextureUvwzFromRMuMuSNu(SkyAtmosphereParameters atmosphere, float r, float mu, float mu_s, float nu, bool rayRMuIntersectsGround)
{
  // Distance to the top boundary for a horizontal ray at ground level, and to the horizon.
  const float H = sqrt(atmosphere.topRadius * atmosphere.topRadius - atmosphere.bottomRadius * atmosphere.bottomRadius);
  const float rho = SafeSqrt(r * r - atmosphere.bottomRadius * atmosphere.bottomRadius);
  const float u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);

  // Discriminant of the ray-ground intersection, as in RayIntersectsGround.
  const float r_mu         = r * mu;
  const float discriminant = r_mu * r_mu - r * r + atmosphere.bottomRadius * atmosphere.bottomRadius;
  float       u_mu;
  if(rayRMuIntersectsGround)
  {
    // Distance to the ground, against its range over all mu -- from (r,-1) and (r,mu_horizon).
    const float d     = -r_mu - SafeSqrt(discriminant);
    const float d_min = r - atmosphere.bottomRadius;
    const float d_max = rho;
    u_mu = 0.5F - 0.5F * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0F : (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
  }
  else
  {
    // Same, to the top boundary -- from (r,1) and (r,mu_horizon).
    const float d     = -r_mu + SafeSqrt(discriminant + H * H);
    const float d_min = atmosphere.topRadius - r;
    const float d_max = rho + H;
    u_mu = 0.5F + 0.5F * GetTextureCoordFromUnitRange((d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
  }

  const float d     = DistanceToTopAtmosphereBoundary(atmosphere, atmosphere.bottomRadius, mu_s);
  const float d_min = atmosphere.topRadius - atmosphere.bottomRadius;
  const float d_max = H;
  const float a     = (d - d_min) / (d_max - d_min);
  const float D     = DistanceToTopAtmosphereBoundary(atmosphere, atmosphere.bottomRadius, atmosphere.muSMin);
  const float A     = (D - d_min) / (d_max - d_min);
  // Ad-hoc, and upstream says so: 0 at mu_s = muSMin, 1 at mu_s = 1, with a steep slope around
  // mu_s = 0 so that texels concentrate near the horizon where the sky changes fastest.
  const float u_mu_s = GetTextureCoordFromUnitRange(max(1.0F - a / A, 0.0F) / (1.0F + a), SCATTERING_TEXTURE_MU_S_SIZE);

  const float u_nu = (nu + 1.0F) / 2.0F;
  return float4(u_nu, u_mu_s, u_mu, u_r);
}

void GetRMuMuSNuFromScatteringTextureUvwz(SkyAtmosphereParameters atmosphere,
                                          float4                  uvwz,
                                          out float               r,
                                          out float               mu,
                                          out float               mu_s,
                                          out float               nu,
                                          out bool                rayRMuIntersectsGround)
{
  const float H = sqrt(atmosphere.topRadius * atmosphere.topRadius - atmosphere.bottomRadius * atmosphere.bottomRadius);
  const float rho = H * GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);
  r               = sqrt(rho * rho + atmosphere.bottomRadius * atmosphere.bottomRadius);

  // The lower half of the mu axis holds ground-intersecting rays, the upper half the rest. That
  // split is what keeps the horizon sharp: the two families are interpolated separately.
  if(uvwz.z < 0.5F)
  {
    const float d_min = r - atmosphere.bottomRadius;
    const float d_max = rho;
    const float d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(1.0F - 2.0F * uvwz.z, SCATTERING_TEXTURE_MU_SIZE / 2);
    mu                     = (d == 0.0F) ? -1.0F : ClampCosine(-(rho * rho + d * d) / (2.0F * r * d));
    rayRMuIntersectsGround = true;
  }
  else
  {
    const float d_min = atmosphere.topRadius - r;
    const float d_max = rho + H;
    const float d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(2.0F * uvwz.z - 1.0F, SCATTERING_TEXTURE_MU_SIZE / 2);
    mu                     = (d == 0.0F) ? 1.0F : ClampCosine((H * H - rho * rho - d * d) / (2.0F * r * d));
    rayRMuIntersectsGround = false;
  }

  const float x_mu_s = GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);
  const float d_min  = atmosphere.topRadius - atmosphere.bottomRadius;
  const float d_max  = H;
  const float D      = DistanceToTopAtmosphereBoundary(atmosphere, atmosphere.bottomRadius, atmosphere.muSMin);
  const float A      = (D - d_min) / (d_max - d_min);
  const float a      = (A - x_mu_s * A) / (1.0F + x_mu_s * A);
  const float d      = d_min + min(a, A) * (d_max - d_min);
  mu_s               = (d == 0.0F) ? 1.0F : ClampCosine((H * H - d * d) / (2.0F * atmosphere.bottomRadius * d));

  nu = ClampCosine(uvwz.x * 2.0F - 1.0F);
}

void GetRMuMuSNuFromScatteringTextureFragCoord(SkyAtmosphereParameters atmosphere,
                                               float3                  fragCoord,
                                               out float               r,
                                               out float               mu,
                                               out float               mu_s,
                                               out float               nu,
                                               out bool                rayRMuIntersectsGround)
{
  const float4 SCATTERING_TEXTURE_SIZE = float4(SCATTERING_TEXTURE_NU_SIZE - 1, SCATTERING_TEXTURE_MU_S_SIZE,
                                                SCATTERING_TEXTURE_MU_SIZE, SCATTERING_TEXTURE_R_SIZE);
  // Unpack nu and mu_s, which share the x axis.
  const float  fragCoordNu  = floor(fragCoord.x / float(SCATTERING_TEXTURE_MU_S_SIZE));
  const float  fragCoordMuS = fmod(fragCoord.x, float(SCATTERING_TEXTURE_MU_S_SIZE));
  const float4 uvwz         = float4(fragCoordNu, fragCoordMuS, fragCoord.y, fragCoord.z) / SCATTERING_TEXTURE_SIZE;
  GetRMuMuSNuFromScatteringTextureUvwz(atmosphere, uvwz, r, mu, mu_s, nu, rayRMuIntersectsGround);

  // nu is not independent: its range follows from mu and mu_s, and the functions above assume the
  // constraint holds. Texels outside it exist in the LUT but describe impossible geometry.
  nu = clamp(nu, mu * mu_s - sqrt((1.0F - mu * mu) * (1.0F - mu_s * mu_s)),
             mu * mu_s + sqrt((1.0F - mu * mu) * (1.0F - mu_s * mu_s)));
}

void ComputeSingleScatteringTexture(SkyAtmosphereParameters atmosphere,
                                    Sampler2D<float3>       transmittanceTexture,
                                    float3                  fragCoord,
                                    out float3              rayleigh,
                                    out float3              mie)
{
  float r;
  float mu;
  float mu_s;
  float nu;
  bool  rayRMuIntersectsGround;
  GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, fragCoord, r, mu, mu_s, nu, rayRMuIntersectsGround);
  ComputeSingleScattering(atmosphere, transmittanceTexture, r, mu, mu_s, nu, rayRMuIntersectsGround, rayleigh, mie);
}

// Two 3D lookups blended along nu, emulating one 4D lookup with quadrilinear interpolation.
//
// Upstream declares this once behind TEMPLATE(AbstractSpectrum) so the same body serves the
// three-channel LUTs and the four-channel combined one. Here every LUT is allocated RGBA32F, so one
// float4 signature covers both cases and callers take `.rgb` or `.a` as they need -- no generic,
// which is also the version Slang actually accepts.
//
// Split in two because only the first half is expensive, and only its x depends on nu: a caller
// that sweeps nu at fixed (r, mu, mu_s) maps once and patches `uvwz.x = (nu + 1) / 2` per sample.
float4 SampleScatteringUvwz(Sampler3D<float4> scatteringTexture, float4 uvwz)
{
  const float  texCoordX = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
  const float  texX      = floor(texCoordX);
  const float  lerpT     = texCoordX - texX;
  const float3 uvw0      = float3((texX + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
  const float3 uvw1      = float3((texX + 1.0F + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
  return scatteringTexture.SampleLevel(uvw0, 0.0F) * (1.0F - lerpT) + scatteringTexture.SampleLevel(uvw1, 0.0F) * lerpT;
}

float4 GetScatteringTexel(SkyAtmosphereParameters atmosphere, Sampler3D<float4> scatteringTexture, float r, float mu, float mu_s, float nu, bool rayRMuIntersectsGround)
{
  return SampleScatteringUvwz(scatteringTexture,
                              GetScatteringTextureUvwzFromRMuMuSNu(atmosphere, r, mu, mu_s, nu, rayRMuIntersectsGround));
}

//-------------------------------------------------------------------------------------------------
// Multiple scattering
//
// Order n is built from order n-1 in two steps rather than one: first the *density* of light
// scattered at every point and direction after n-1 bounces, then a single integration of that along
// the view ray. Doing it directly would nest an integral inside an integral and recompute order n-1
// for every point, which is the cost the paper's two-step scheme exists to avoid.

// Forward declaration: ComputeScatteringDensity needs the ground irradiance, whose lookup is
// defined further down with the rest of the irradiance section, exactly as upstream orders it.
float3 GetIrradiance(SkyAtmosphereParameters atmosphere, Sampler2D<float3> irradianceTexture, float r, float mu_s);

// Scattering of a given order at an already-mapped `uvwz` (see SampleScatteringUvwz), with the
// phase functions applied for order 1 only -- higher orders already have them folded in when their
// LUT was written.
float3 GetScatteringOrder(SkyAtmosphereParameters atmosphere,
                          Sampler3D<float4>       singleRayleighScatteringTexture,
                          Sampler3D<float4>       singleMieScatteringTexture,
                          Sampler3D<float4>       multipleScatteringTexture,
                          float4                  uvwz,
                          float                   nu,
                          int                     scatteringOrder)
{
  if(scatteringOrder == 1)
  {
    const float3 rayleigh = SampleScatteringUvwz(singleRayleighScatteringTexture, uvwz).rgb;
    const float3 mie      = SampleScatteringUvwz(singleMieScatteringTexture, uvwz).rgb;
    return rayleigh * RayleighPhaseFunction(nu) + mie * MiePhaseFunction(atmosphere.miePhaseFunctionG, nu);
  }
  return SampleScatteringUvwz(multipleScatteringTexture, uvwz).rgb;
}

float3 ComputeScatteringDensity(SkyAtmosphereParameters atmosphere,
                                Sampler2D<float3>       transmittanceTexture,
                                Sampler3D<float4>       singleRayleighScatteringTexture,
                                Sampler3D<float4>       singleMieScatteringTexture,
                                Sampler3D<float4>       multipleScatteringTexture,
                                Sampler2D<float3>       irradianceTexture,
                                float                   r,
                                float                   mu,
                                float                   mu_s,
                                float                   nu,
                                int                     scatteringOrder,
                                int                     sampleCount)
{
  // Unit vectors for the zenith, the view direction and the sun, chosen so that the three cosines
  // come out as mu, mu_s and nu. Only their relative geometry matters.
  const float3 zenithDirection = float3(0.0F, 0.0F, 1.0F);
  const float3 omega           = float3(sqrt(1.0F - mu * mu), 0.0F, mu);
  const float  sunDirX         = (omega.x == 0.0F) ? 0.0F : (nu - mu * mu_s) / omega.x;
  const float  sunDirY         = sqrt(max(1.0F - sunDirX * sunDirX - mu_s * mu_s, 0.0F));
  const float3 omega_s         = float3(sunDirX, sunDirY, mu_s);

  const int   SAMPLE_COUNT = sampleCount;
  const float dphi         = M_PI / float(SAMPLE_COUNT);
  const float dtheta       = M_PI / float(SAMPLE_COUNT);
  float3      rayleighMie  = float3(0.0F);

  // How much this altitude scatters does not depend on which direction the light arrives from, so
  // the two density evaluations -- an exp() apiece -- are lifted out of a loop that runs 2N^2 times
  // per texel. This is the most expensive pass in the precomputation, and the loop body is where
  // all of it is.
  const float altitude = r - atmosphere.bottomRadius;
  const float3 rayleighAtAltitude = atmosphere.rayleighScattering * GetProfileDensity(atmosphere.rayleighDensity, altitude);
  const float3 mieAtAltitude = atmosphere.mieScattering * GetProfileDensity(atmosphere.mieDensity, altitude);

  // Integral over every incident direction omega_i.
  for(int l = 0; l < SAMPLE_COUNT; ++l)
  {
    const float theta                     = (float(l) + 0.5F) * dtheta;
    const float cosTheta                  = cos(theta);
    const float sinTheta                  = sin(theta);
    const bool  rayRThetaIntersectsGround = RayIntersectsGround(atmosphere, r, cosTheta);

    // Distance and transmittance to the ground depend on theta alone, so they are hoisted out of
    // the inner loop. So is the solid angle, which varies only with theta.
    const float domega_i = dtheta * dphi * sinTheta;

    float  distanceToGround      = 0.0F;
    float3 transmittanceToGround = float3(0.0F);
    if(rayRThetaIntersectsGround)
    {
      distanceToGround      = DistanceToBottomAtmosphereBoundary(atmosphere, r, cosTheta);
      transmittanceToGround = GetTransmittance(atmosphere, transmittanceTexture, r, cosTheta, distanceToGround, true);
    }

    // Every lookup of the inner loop is at (r, cosTheta, mu_s, nu1), and only nu1 moves -- so every
    // one of them lands on the same NU_SIZE texels, the nu slabs of one (mu_s, mu, r) row, and
    // differs only in how it blends two neighbours (SampleScatteringUvwz). Fetching the row once
    // here and blending in registers returns the same values as 2N pairs of filtered 3D fetches,
    // which is what bounds this pass. Order 1 needs its Rayleigh and Mie rows kept apart, as their
    // phase functions are applied after the blend.
    const bool firstOrder = (scatteringOrder - 1) == 1;
    const float4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(atmosphere, r, cosTheta, mu_s, 0.0F, rayRThetaIntersectsGround);
    float3 rowA[SCATTERING_TEXTURE_NU_SIZE];  // Rayleigh for order 1, else the multiple-scattering order
    float3 rowB[SCATTERING_TEXTURE_NU_SIZE];  // Mie for order 1, else unused
    [ForceUnroll] for(int k = 0; k < SCATTERING_TEXTURE_NU_SIZE; ++k)
    {
      const float3 uvw = float3((float(k) + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
      if(firstOrder)
      {
        rowA[k] = singleRayleighScatteringTexture.SampleLevel(uvw, 0.0F).rgb;
        rowB[k] = singleMieScatteringTexture.SampleLevel(uvw, 0.0F).rgb;
      }
      else
      {
        rowA[k] = multipleScatteringTexture.SampleLevel(uvw, 0.0F).rgb;
        rowB[k] = float3(0.0F);
      }
    }

    for(int m = 0; m < 2 * SAMPLE_COUNT; ++m)
    {
      const float  phi     = (float(m) + 0.5F) * dphi;
      const float3 omega_i = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

      // Radiance arriving from omega_i after n-1 bounces: the precomputed (n-1)-th order...
      const float nu1       = dot(omega_s, omega_i);
      const float texCoordX = (nu1 + 1.0F) / 2.0F * float(SCATTERING_TEXTURE_NU_SIZE - 1);
      float3      a         = float3(0.0F);
      float3      b         = float3(0.0F);
      [ForceUnroll] for(int k = 0; k < SCATTERING_TEXTURE_NU_SIZE; ++k)
      {
        const float w = max(1.0F - abs(texCoordX - float(k)), 0.0F);  // the linear blend, as a tent
        a += rowA[k] * w;
        b += rowB[k] * w;
      }
      float3 incidentRadiance =
          firstOrder ? a * RayleighPhaseFunction(nu1) + b * MiePhaseFunction(atmosphere.miePhaseFunctionG, nu1) : a;

      // ...plus the paths whose last bounce was the ground: transmittance there, times the albedo,
      // times the Lambertian BRDF, times the irradiance the ground received after n-2 bounces.
      //
      // Only when this theta actually sees the ground. Otherwise the transmittance and the albedo
      // are both exactly zero and the whole term vanishes -- but the irradiance fetch still had to
      // happen to reach that conclusion, a third of this loop's texture traffic spent on a result
      // multiplied away. The test depends on theta alone, so the branch is uniform across the
      // inner loop rather than divergent within it.
      if(rayRThetaIntersectsGround)
      {
        const float3 groundNormal = normalize(zenithDirection * r + omega_i * distanceToGround);
        const float3 groundIrradiance =
            GetIrradiance(atmosphere, irradianceTexture, atmosphere.bottomRadius, dot(groundNormal, omega_s));
        incidentRadiance += transmittanceToGround * atmosphere.groundAlbedo * (1.0F / M_PI) * groundIrradiance;
      }

      // What that incident radiance scatters towards -omega, summed over both particle types.
      const float nu2 = dot(omega, omega_i);
      rayleighMie += incidentRadiance
                     * (rayleighAtAltitude * RayleighPhaseFunction(nu2)
                        + mieAtAltitude * MiePhaseFunction(atmosphere.miePhaseFunctionG, nu2))
                     * domega_i;
    }
  }
  return rayleighMie;
}

float3 ComputeMultipleScattering(SkyAtmosphereParameters atmosphere,
                                 Sampler2D<float3>       transmittanceTexture,
                                 Sampler3D<float4>       scatteringDensityTexture,
                                 float                   r,
                                 float                   mu,
                                 float                   mu_s,
                                 float                   nu,
                                 bool                    rayRMuIntersectsGround)
{
  const int   SAMPLE_COUNT = 50;
  const float dx = DistanceToNearestAtmosphereBoundary(atmosphere, r, mu, rayRMuIntersectsGround) / float(SAMPLE_COUNT);

  float3 rayleighMieSum = float3(0.0F);
  for(int i = 0; i <= SAMPLE_COUNT; ++i)
  {
    const float d_i = float(i) * dx;

    // The parameters at this sample point, as in the single-scattering section.
    const float r_i    = ClampRadius(atmosphere, sqrt(d_i * d_i + 2.0F * r * mu * d_i + r * r));
    const float mu_i   = ClampCosine((r * mu + d_i) / r_i);
    const float mu_s_i = ClampCosine((r * mu_s + d_i * nu) / r_i);

    const float3 rayleighMie_i =
        GetScatteringTexel(atmosphere, scatteringDensityTexture, r_i, mu_i, mu_s_i, nu, rayRMuIntersectsGround).rgb
        * GetTransmittance(atmosphere, transmittanceTexture, r, mu, d_i, rayRMuIntersectsGround) * dx;
    const float weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5F : 1.0F;
    rayleighMieSum += rayleighMie_i * weight_i;
  }
  return rayleighMieSum;
}

float3 ComputeScatteringDensityTexture(SkyAtmosphereParameters atmosphere,
                                       Sampler2D<float3>       transmittanceTexture,
                                       Sampler3D<float4>       singleRayleighScatteringTexture,
                                       Sampler3D<float4>       singleMieScatteringTexture,
                                       Sampler3D<float4>       multipleScatteringTexture,
                                       Sampler2D<float3>       irradianceTexture,
                                       float3                  fragCoord,
                                       int                     scatteringOrder,
                                       int                     sampleCount)
{
  float r;
  float mu;
  float mu_s;
  float nu;
  bool  rayRMuIntersectsGround;
  GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, fragCoord, r, mu, mu_s, nu, rayRMuIntersectsGround);
  return ComputeScatteringDensity(atmosphere, transmittanceTexture, singleRayleighScatteringTexture, singleMieScatteringTexture,
                                  multipleScatteringTexture, irradianceTexture, r, mu, mu_s, nu, scatteringOrder, sampleCount);
}

float3 ComputeMultipleScatteringTexture(SkyAtmosphereParameters atmosphere,
                                        Sampler2D<float3>       transmittanceTexture,
                                        Sampler3D<float4>       scatteringDensityTexture,
                                        float3                  fragCoord,
                                        out float               nu)
{
  float r;
  float mu;
  float mu_s;
  bool  rayRMuIntersectsGround;
  GetRMuMuSNuFromScatteringTextureFragCoord(atmosphere, fragCoord, r, mu, mu_s, nu, rayRMuIntersectsGround);
  return ComputeMultipleScattering(atmosphere, transmittanceTexture, scatteringDensityTexture, r, mu, mu_s, nu, rayRMuIntersectsGround);
}

//-------------------------------------------------------------------------------------------------
// Irradiance
//
// What reaches the ground: the sun directly, and the sky after n bounces. The multiple-scattering
// passes need it because light that bounced off the ground is part of what they integrate.

float3 ComputeDirectIrradiance(SkyAtmosphereParameters atmosphere, Sampler2D<float3> transmittanceTexture, float r, float mu_s)
{
  const float alpha_s = atmosphere.sunAngularRadius;
  // Average of the cosine factor over the visible fraction of the sun's disk. Not a point light:
  // near the horizon part of the disk is below it, and this is the cheap approximation upstream
  // uses rather than integrating over the disk.
  const float averageCosineFactor =
      (mu_s < -alpha_s) ? 0.0F : ((mu_s > alpha_s) ? mu_s : (mu_s + alpha_s) * (mu_s + alpha_s) / (4.0F * alpha_s));

  return atmosphere.solarIrradiance * GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittanceTexture, r, mu_s) * averageCosineFactor;
}

float3 ComputeIndirectIrradiance(SkyAtmosphereParameters atmosphere,
                                 Sampler3D<float4>       singleRayleighScatteringTexture,
                                 Sampler3D<float4>       singleMieScatteringTexture,
                                 Sampler3D<float4>       multipleScatteringTexture,
                                 float                   r,
                                 float                   mu_s,
                                 int                     scatteringOrder)
{
  const int   SAMPLE_COUNT = 32;
  const float dphi         = M_PI / float(SAMPLE_COUNT);
  const float dtheta       = M_PI / float(SAMPLE_COUNT);

  float3       result  = float3(0.0F);
  const float3 omega_s = float3(sqrt(1.0F - mu_s * mu_s), 0.0F, mu_s);
  // Upper hemisphere only -- this is the irradiance arriving on an upward-facing surface.
  for(int j = 0; j < SAMPLE_COUNT / 2; ++j)
  {
    const float theta = (float(j) + 0.5F) * dtheta;
    // As in ComputeScatteringDensity: only nu varies with phi, so the mapping is done once.
    float4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(atmosphere, r, cos(theta), mu_s, 0.0F, false /* rayRThetaIntersectsGround */);
    for(int i = 0; i < 2 * SAMPLE_COUNT; ++i)
    {
      const float  phi    = (float(i) + 0.5F) * dphi;
      const float3 omega  = float3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
      const float  domega = dtheta * dphi * sin(theta);

      const float nu = dot(omega, omega_s);
      uvwz.x         = (nu + 1.0F) / 2.0F;
      result += GetScatteringOrder(atmosphere, singleRayleighScatteringTexture, singleMieScatteringTexture,
                                   multipleScatteringTexture, uvwz, nu, scatteringOrder)
                * omega.z * domega;
    }
  }
  return result;
}

// Irradiance depends on only two parameters, so its LUT is a small 2D table with a plain linear
// mapping -- none of the horizon-concentrating trickery the scattering LUT needs.
float2 GetIrradianceTextureUvFromRMuS(SkyAtmosphereParameters atmosphere, float r, float mu_s)
{
  const float x_r    = (r - atmosphere.bottomRadius) / (atmosphere.topRadius - atmosphere.bottomRadius);
  const float x_mu_s = mu_s * 0.5F + 0.5F;
  return float2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),
                GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));
}

void GetRMuSFromIrradianceTextureUv(SkyAtmosphereParameters atmosphere, float2 uv, out float r, out float mu_s)
{
  const float x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
  const float x_r    = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);
  r                  = atmosphere.bottomRadius + x_r * (atmosphere.topRadius - atmosphere.bottomRadius);
  mu_s               = ClampCosine(2.0F * x_mu_s - 1.0F);
}

static const float2 IRRADIANCE_TEXTURE_SIZE = float2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);

float3 ComputeDirectIrradianceTexture(SkyAtmosphereParameters atmosphere, Sampler2D<float3> transmittanceTexture, float2 fragCoord)
{
  float r;
  float mu_s;
  GetRMuSFromIrradianceTextureUv(atmosphere, fragCoord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);
  return ComputeDirectIrradiance(atmosphere, transmittanceTexture, r, mu_s);
}

float3 ComputeIndirectIrradianceTexture(SkyAtmosphereParameters atmosphere,
                                        Sampler3D<float4>       singleRayleighScatteringTexture,
                                        Sampler3D<float4>       singleMieScatteringTexture,
                                        Sampler3D<float4>       multipleScatteringTexture,
                                        float2                  fragCoord,
                                        int                     scatteringOrder)
{
  float r;
  float mu_s;
  GetRMuSFromIrradianceTextureUv(atmosphere, fragCoord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);
  return ComputeIndirectIrradiance(atmosphere, singleRayleighScatteringTexture, singleMieScatteringTexture,
                                   multipleScatteringTexture, r, mu_s, scatteringOrder);
}

// Definition of the lookup forward-declared above the multiple-scattering section.
float3 GetIrradiance(SkyAtmosphereParameters atmosphere, Sampler2D<float3> irradianceTexture, float r, float mu_s)
{
  const float2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);
  return irradianceTexture.SampleLevel(uv, 0.0F);
}

//-------------------------------------------------------------------------------------------------
// Runtime evaluation
//
// What rendering actually calls: the radiance of a ray that escapes the scene. Everything above
// exists to make these cheap.
//
// Rayleigh and Mie live in separate LUTs, so upstream's GetExtrapolatedSingleMieScattering (which
// recovers Mie from a packed texture's alpha) is not ported; the precompute writes alpha 1.

// Rayleigh (plus higher orders) and single Mie, from the two separate LUTs.
float3 GetCombinedScattering(SkyAtmosphereParameters atmosphere,
                             Sampler3D<float4>       scatteringTexture,
                             Sampler3D<float4>       singleMieScatteringTexture,
                             float                   r,
                             float                   mu,
                             float                   mu_s,
                             float                   nu,
                             bool                    rayRMuIntersectsGround,
                             out float3              singleMieScattering)
{
  const float4 uvwz   = GetScatteringTextureUvwzFromRMuMuSNu(atmosphere, r, mu, mu_s, nu, rayRMuIntersectsGround);
  singleMieScattering = SampleScatteringUvwz(singleMieScatteringTexture, uvwz).rgb;
  return SampleScatteringUvwz(scatteringTexture, uvwz).rgb;
}

// Radiance of the sky along `viewRay` from `camera`, and the transmittance through the atmosphere
// along it -- what a ray that escapes the scene sees.
//
// `camera` is a position in planet space, in km from the planet centre. Under the fixed-observer
// policy the host builds it from the observer altitude rather than from the scene camera, so the
// sky does not shift as the viewer walks around a glTF asset.
//
// The `shadowLength` parameter upstream uses for light shafts is dropped: nothing produces one
// here, and carrying an always-zero argument through every call site invites someone to assume it
// works. Its branch in upstream is a strict addition, easy to restore if light shafts arrive.
float3 GetSkyRadiance(SkyAtmosphereParameters atmosphere,
                      Sampler2D<float3>       transmittanceTexture,
                      Sampler3D<float4>       scatteringTexture,
                      Sampler3D<float4>       singleMieScatteringTexture,
                      float3                  camera,
                      float3                  viewRay,
                      float3                  sunDirection,
                      out float3              transmittance)
{
  // Distance to the top of the atmosphere, assuming the viewer might be outside it.
  float r   = length(camera);
  float rmu = dot(camera, viewRay);
  const float distanceToTopAtmosphereBoundary = -rmu - sqrt(rmu * rmu - r * r + atmosphere.topRadius * atmosphere.topRadius);

  if(distanceToTopAtmosphereBoundary > 0.0F)
  {
    // Viewer in space with the ray entering the atmosphere: move them to the boundary.
    camera = camera + viewRay * distanceToTopAtmosphereBoundary;
    r      = atmosphere.topRadius;
    rmu += distanceToTopAtmosphereBoundary;
  }
  else if(r > atmosphere.topRadius)
  {
    // Ray misses the atmosphere entirely: empty space, fully transparent.
    transmittance = float3(1.0F);
    return float3(0.0F);
  }

  const float mu                     = rmu / r;
  const float mu_s                   = dot(camera, sunDirection) / r;
  const float nu                     = dot(viewRay, sunDirection);
  const bool  rayRMuIntersectsGround = RayIntersectsGround(atmosphere, r, mu);

  transmittance = rayRMuIntersectsGround ? float3(0.0F) :
                                           GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittanceTexture, r, mu);

  float3       singleMieScattering;
  const float3 scattering = GetCombinedScattering(atmosphere, scatteringTexture, singleMieScatteringTexture, r, mu,
                                                  mu_s, nu, rayRMuIntersectsGround, singleMieScattering);

  // The phase functions are applied here rather than baked into the LUT, which is what preserves
  // angular detail the table's nu resolution could not hold -- the sharp forward Mie lobe above all.
  return scattering * RayleighPhaseFunction(nu) + singleMieScattering * MiePhaseFunction(atmosphere.miePhaseFunctionG, nu);
}

// GetSkyRadianceToPoint -- upstream's aerial-perspective entry point -- is deliberately not here.
//
// It was ported, used, and removed again, which is worth a sentence so nobody ports it a second
// time. It answers "what is in the air between here and that surface" by subtracting everything
// scattered beyond the surface from everything scattered beyond the eye: two texture fetches, no
// integration, very elegant. The table is stored in two halves though -- rays that meet the planet
// and rays that escape -- and which half to read is decided by where the *extended* ray goes. A view
// ray tipping past the horizon switches halves while the air being shaded has not changed, and the
// halves do not agree; the escaping half spikes in the degenerate rows just above the horizon. On a
// 10 km mountain that draws a hard line across the rock.
//
// shaders/aerial_perspective.h.slang marches the segment instead. See it for the rest of the
// reasoning and for what the march approximates in exchange.

// Sun and sky irradiance arriving on a surface at `point` with `normal`. The return value is the
// direct sun term; the sky term comes back through `skyIrradiance`.
float3 GetSunAndSkyIrradiance(SkyAtmosphereParameters atmosphere,
                              Sampler2D<float3>       transmittanceTexture,
                              Sampler2D<float3>       irradianceTexture,
                              float3                  point,
                              float3                  normal,
                              float3                  sunDirection,
                              out float3              skyIrradiance)
{
  const float r    = length(point);
  const float mu_s = dot(point, sunDirection) / r;

  // Indirect, approximated for a surface that is not horizontal.
  skyIrradiance = GetIrradiance(atmosphere, irradianceTexture, r, mu_s) * (1.0F + dot(normal, point) / r) * 0.5F;

  // Direct.
  return atmosphere.solarIrradiance * GetTransmittanceToSun(atmosphere, transmittanceTexture, r, mu_s)
         * max(dot(normal, sunDirection), 0.0F);
}

#endif  // SKY_BRUNETON_FUNCTIONS_H
