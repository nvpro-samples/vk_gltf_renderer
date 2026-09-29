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
// Bruneton precompute: all six passes, as six entrypoints in one module.
//
// Upstream runs each pass as a fragment shader over a full-screen quad into an FBO, with additive
// blending where an order accumulates; these are the same maths dispatched as compute into storage
// images, with the accumulation done as an explicit read-modify-write.
//
// One file because the six shared everything but their ten-line bodies: the same includes and the
// same twenty-line binding block, copied six times, which is six places to edit when a binding
// moves. Slang compiles every [shader("compute")] entrypoint in a module, so the host picks a pass
// by entrypoint name at pipeline creation -- see SkyBruneton::createPrecomputePipelines.
//
// The order below is the order they run, and it is a dependency chain: transmittance depends only
// on the atmosphere parameters, and each later pass reads what the earlier ones wrote. Passes 4-6
// repeat per scattering order.
//
// `luminance_from_radiance` is dropped throughout: it is identity in the RGB-only configuration
// this build uses, and a matrix multiply by identity in every pass is noise.
//

#include "sky_bruneton_functions.h.slang"
#include "sky_bruneton_io.h.slang"

// clang-format off
[[vk::binding(SkyBrunetonPrecomputeBindings::eBrunetonParams, 0)]]       ConstantBuffer<SkyAtmosphereParameters> atmosphere;

[[vk::binding(SkyBrunetonPrecomputeBindings::eOutTransmittance, 0)]]     RWTexture2D<float4> outTransmittance;
[[vk::binding(SkyBrunetonPrecomputeBindings::eOutDeltaIrradiance, 0)]]   RWTexture2D<float4> outDeltaIrradiance;
[[vk::binding(SkyBrunetonPrecomputeBindings::eOutIrradiance, 0)]]        RWTexture2D<float4> outIrradiance;
[[vk::binding(SkyBrunetonPrecomputeBindings::eOutDeltaRayleigh, 0)]]     RWTexture3D<float4> outDeltaRayleigh;
[[vk::binding(SkyBrunetonPrecomputeBindings::eOutDeltaMie, 0)]]          RWTexture3D<float4> outDeltaMie;
[[vk::binding(SkyBrunetonPrecomputeBindings::eOutScattering, 0)]]        RWTexture3D<float4> outScattering;
[[vk::binding(SkyBrunetonPrecomputeBindings::eOutSingleMie, 0)]]         RWTexture3D<float4> outSingleMie;
[[vk::binding(SkyBrunetonPrecomputeBindings::eOutScatteringDensity, 0)]] RWTexture3D<float4> outScatteringDensity;
[[vk::binding(SkyBrunetonPrecomputeBindings::eOutDeltaMultiple, 0)]]     RWTexture3D<float4> outDeltaMultiple;

[[vk::binding(SkyBrunetonPrecomputeBindings::eInTransmittance, 0)]]      Sampler2D<float3> inTransmittance;
[[vk::binding(SkyBrunetonPrecomputeBindings::eInDeltaIrradiance, 0)]]    Sampler2D<float3> inDeltaIrradiance;
[[vk::binding(SkyBrunetonPrecomputeBindings::eInDeltaRayleigh, 0)]]      Sampler3D<float4> inDeltaRayleigh;
[[vk::binding(SkyBrunetonPrecomputeBindings::eInDeltaMie, 0)]]           Sampler3D<float4> inDeltaMie;
[[vk::binding(SkyBrunetonPrecomputeBindings::eInDeltaMultiple, 0)]]      Sampler3D<float4> inDeltaMultiple;
[[vk::binding(SkyBrunetonPrecomputeBindings::eInScatteringDensity, 0)]]  Sampler3D<float4> inScatteringDensity;

[[vk::push_constant]] ConstantBuffer<SkyBrunetonPushConstant> pushConst;
// clang-format on


[shader("compute")]
[numthreads(SKY_BRUNETON_WORKGROUP_SIZE, SKY_BRUNETON_WORKGROUP_SIZE, 1)]
void Transmittance(uint3 threadId: SV_DispatchThreadID)
{
  const int2 texel = int2(threadId.xy);
  if(texel.x >= TRANSMITTANCE_TEXTURE_WIDTH || texel.y >= TRANSMITTANCE_TEXTURE_HEIGHT)
    return;

  // Upstream passes gl_FragCoord, which is the texel centre. The +0.5 is not cosmetic: the LUT
  // mapping stores f(0) and f(1) at the centres of the first and last texels, so an integer
  // coordinate here would shift the whole table by half a texel.
  const float2 fragCoord = float2(texel) + 0.5F;

  const float3 transmittance = ComputeTransmittanceToTopAtmosphereBoundaryTexture(atmosphere, fragCoord);
  outTransmittance[texel]    = float4(transmittance, 1.0F);
}

//--------------------------------------------------------------------------------------------------
[shader("compute")]
[numthreads(SKY_BRUNETON_WORKGROUP_SIZE, SKY_BRUNETON_WORKGROUP_SIZE, 1)]
void DirectIrradiance(uint3 threadId: SV_DispatchThreadID)
{
  const int2 texel = int2(threadId.xy);
  if(texel.x >= IRRADIANCE_TEXTURE_WIDTH || texel.y >= IRRADIANCE_TEXTURE_HEIGHT)
    return;

  const float3 irradiance   = ComputeDirectIrradianceTexture(atmosphere, inTransmittance, float2(texel) + 0.5F);
  outDeltaIrradiance[texel] = float4(irradiance, 1.0F);

  // Upstream writes zero to the accumulated irradiance here rather than the direct term: the
  // ground irradiance the later orders read is the *indirect* one, and the direct part is added
  // back at render time. Starting it at zero is also what makes the accumulation below correct.
  outIrradiance[texel] = float4(0.0F, 0.0F, 0.0F, 1.0F);
}

//--------------------------------------------------------------------------------------------------
[shader("compute")]
[numthreads(SKY_BRUNETON_WORKGROUP_SIZE, SKY_BRUNETON_WORKGROUP_SIZE, 1)]
void SingleScattering(uint3 threadId: SV_DispatchThreadID)
{
  const int2 texel = int2(threadId.xy);
  if(texel.x >= SCATTERING_TEXTURE_WIDTH || texel.y >= SCATTERING_TEXTURE_HEIGHT)
    return;
  const int3 coord = int3(texel, pushConst.layer);

  float3 rayleigh;
  float3 mie;
  ComputeSingleScatteringTexture(atmosphere, inTransmittance,
                                 float3(float2(texel) + 0.5F, float(pushConst.layer) + 0.5F), rayleigh, mie);

  // The deltas are the input to the next order; the finals are what evaluation samples.
  outDeltaRayleigh[coord] = float4(rayleigh, 1.0F);
  outDeltaMie[coord]      = float4(mie, 1.0F);
  // First write of the accumulators, so assignment rather than accumulation.
  outScattering[coord] = float4(rayleigh, 1.0F);
  outSingleMie[coord]  = float4(mie, 1.0F);
}

//--------------------------------------------------------------------------------------------------
[shader("compute")]
[numthreads(SKY_BRUNETON_WORKGROUP_SIZE, SKY_BRUNETON_WORKGROUP_SIZE, 1)]
void ScatteringDensity(uint3 threadId: SV_DispatchThreadID)
{
  const int2 texel = int2(threadId.xy);
  if(texel.x >= SCATTERING_TEXTURE_WIDTH || texel.y >= SCATTERING_TEXTURE_HEIGHT)
    return;

  // The most expensive pass by a wide margin: an NxN integral over incident directions, per texel
  // of a 256x128x32 LUT, repeated for every order after the first. Its cost is quadratic in
  // `densitySamples`, which is what lets a drag preview the atmosphere at all -- see
  // SkyBruneton::Quality.
  const float3 density =
      ComputeScatteringDensityTexture(atmosphere, inTransmittance, inDeltaRayleigh, inDeltaMie, inDeltaMultiple,
                                      inDeltaIrradiance, float3(float2(texel) + 0.5F, float(pushConst.layer) + 0.5F),
                                      pushConst.scatteringOrder, pushConst.densitySamples);

  outScatteringDensity[int3(texel, pushConst.layer)] = float4(density, 1.0F);
}

//--------------------------------------------------------------------------------------------------
[shader("compute")]
[numthreads(SKY_BRUNETON_WORKGROUP_SIZE, SKY_BRUNETON_WORKGROUP_SIZE, 1)]
void IndirectIrradiance(uint3 threadId: SV_DispatchThreadID)
{
  const int2 texel = int2(threadId.xy);
  if(texel.x >= IRRADIANCE_TEXTURE_WIDTH || texel.y >= IRRADIANCE_TEXTURE_HEIGHT)
    return;

  const float3 irradiance = ComputeIndirectIrradianceTexture(atmosphere, inDeltaRayleigh, inDeltaMie, inDeltaMultiple,
                                                             float2(texel) + 0.5F, pushConst.scatteringOrder);

  // Replaces the delta (it is this order's contribution) but accumulates into the final, which is
  // the sum over all orders. Upstream gets the same effect from additive blending on attachment 1.
  outDeltaIrradiance[texel] = float4(irradiance, 1.0F);
  outIrradiance[texel] += float4(irradiance, 0.0F);
}

//--------------------------------------------------------------------------------------------------
[shader("compute")]
[numthreads(SKY_BRUNETON_WORKGROUP_SIZE, SKY_BRUNETON_WORKGROUP_SIZE, 1)]
void MultipleScattering(uint3 threadId: SV_DispatchThreadID)
{
  const int2 texel = int2(threadId.xy);
  if(texel.x >= SCATTERING_TEXTURE_WIDTH || texel.y >= SCATTERING_TEXTURE_HEIGHT)
    return;
  const int3 coord = int3(texel, pushConst.layer);

  float        nu;
  const float3 multiple = ComputeMultipleScatteringTexture(atmosphere, inTransmittance, inScatteringDensity,
                                                           float3(float2(texel) + 0.5F, float(pushConst.layer) + 0.5F), nu);

  outDeltaMultiple[coord] = float4(multiple, 1.0F);
  // Divided by the Rayleigh phase function because evaluation multiplies the whole accumulated
  // scattering by it again -- orders above the first have no single phase function of their own,
  // so this pre-division cancels the one applied at render time.
  outScattering[coord] += float4(multiple / RayleighPhaseFunction(nu), 0.0F);
}
