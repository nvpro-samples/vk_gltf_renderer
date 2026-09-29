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

// CPU-only unit test for nvvk::buildEnvAliasmap (Vose's method for environment importance
// sampling). Exercises the exact function backing both the file-load path
// (createEnvironmentAccel) and the GPU-image path (HdrIbl::updateFromGpuImage), so any
// regression here surfaces in both producer pipelines.
//
// The test does not stand in for the Monte-Carlo end-to-end validation the
// `--benchmark-env-pipeline` mode runs on the GPU; it pins the *algorithm*, not its use.

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

// Pull in the standalone alias-map header rather than <nvvk/hdr_ibl.hpp>. The latter drags the
// whole Vulkan/VMA descriptor stack into the test binary (via nvvk::ResourceAllocator), which
// then fails to link because the test target does not compile a VMA implementation.
#include <nvvk/env_aliasmap.hpp>

namespace {

// Sum of a `size` array of ones and one large spike -- the classic sunset case where a single
// texel dominates the integral. Vose's method must correctly assign many low-energy texels as
// aliases of the spike.
std::vector<float> makeSunset(size_t size, float spike)
{
  std::vector<float> data(size, 1.0F);
  data[size / 2] = spike;
  return data;
}

// Uniform importance across all texels. Every accel.q must round to 1.0 and each texel
// remains its own alias.
std::vector<float> makeUniform(size_t size)
{
  return std::vector<float>(size, 3.14F);
}

// Random non-negative importance in [0..1). Sanity fixture for the general case.
std::vector<float> makeRandom(size_t size, uint32_t seed)
{
  std::mt19937                          rng(seed);
  std::uniform_real_distribution<float> dist(0.0F, 1.0F);
  std::vector<float>                    data(size);
  for(auto& v : data)
    v = dist(rng);
  return data;
}

// Direct evaluation of the Vose sampling equation used by hdr_env_sampling.h.slang:
//   P(pick j) = (1/N) * ( q[j] + sum_{i : alias[i] == j and q[i] < 1} (1 - q[i]) )
// This is what actually matters for the PT: samples produced by (i, u) draws must land on j
// with probability proportional to data[j]. Any bug in Vose partitioning will show up as a
// distribution that no longer matches data / sum(data).
std::vector<double> computeSamplingDistribution(const std::vector<shaderio::EnvAccel>& accel)
{
  const size_t        n = accel.size();
  std::vector<double> p(n, 0.0);
  for(size_t i = 0; i < n; ++i)
  {
    // Direct-hit contribution: uniform pick of index i, then u < q[i].
    p[i] += std::min<double>(accel[i].q, 1.0);
    // Alias contribution: uniform pick of some other index k with q[k] < 1 that aliases to i.
    if(accel[i].q < 1.0F)
      p[accel[i].alias] += 1.0 - static_cast<double>(accel[i].q);
  }
  for(auto& v : p)
    v /= static_cast<double>(n);
  return p;
}

}  // namespace


TEST(AliasTable, UniformImportanceKeepsIdentityAliases)
{
  constexpr size_t kSize = 128;
  const auto       data  = makeUniform(kSize);

  std::vector<shaderio::EnvAccel> accel(kSize);
  const float                     integral = nvvk::buildEnvAliasmap(data, accel);

  // std::accumulate over 128 floats does not produce the same rounding as a single multiply,
  // so we compare against the accumulated ground truth rather than 3.14F * kSize directly.
  const double expectedIntegral = std::accumulate(data.begin(), data.end(), 0.0);
  EXPECT_NEAR(integral, static_cast<float>(expectedIntegral), 1e-3F) << "Integral must equal sum(importance)";

  for(size_t i = 0; i < kSize; ++i)
  {
    // All q values should be exactly 1 (or extremely close, within float rounding). The alias
    // partition never triggers when q>=1 for every texel, so every texel keeps its own index.
    EXPECT_NEAR(accel[i].q, 1.0F, 1e-5F) << "Uniform input: accel[" << i << "].q must be ~1";
    EXPECT_EQ(accel[i].alias, static_cast<uint32_t>(i)) << "Uniform input: accel[" << i << "].alias must be self";
  }
}


TEST(AliasTable, SunsetSpikeAliasesLowEnergyTexelsToTheSpike)
{
  constexpr size_t kSize  = 256;
  constexpr float  kSpike = 200.0F;
  const auto       data   = makeSunset(kSize, kSpike);

  std::vector<shaderio::EnvAccel> accel(kSize);
  const float                     integral = nvvk::buildEnvAliasmap(data, accel);

  // Integral = (kSize - 1) * 1.0 + kSpike.
  EXPECT_FLOAT_EQ(integral, static_cast<float>(kSize - 1) + kSpike);

  // Every dim texel (q < 1) must alias to the spike, which is the only high-energy entry.
  const uint32_t spikeIdx       = kSize / 2;
  size_t         aliasedToSpike = 0;
  for(size_t i = 0; i < kSize; ++i)
  {
    if(i == spikeIdx)
      continue;
    // The low-energy texels have q < 1 and must alias to the spike (no other high-energy
    // texel exists in this fixture).
    EXPECT_LT(accel[i].q, 1.0F) << "Dim texel " << i << " should have q < 1";
    if(accel[i].alias == spikeIdx)
      ++aliasedToSpike;
  }
  EXPECT_EQ(aliasedToSpike, kSize - 1) << "All dim texels must alias to the single spike";
}


TEST(AliasTable, SamplingDistributionMatchesInputForRandomImportance)
{
  // The load-bearing invariant: after alias construction, the effective sampling probability
  // for each index j -- computed the same way the sampling shader does it -- must match the
  // input importance normalized to a PDF. This is what the PT relies on for unbiased NEE.
  constexpr size_t         kSize  = 512;
  const std::vector<float> data   = makeRandom(kSize, 0x1234u);
  const double             refSum = std::accumulate(data.begin(), data.end(), 0.0);

  std::vector<shaderio::EnvAccel> accel(kSize);
  const float                     integral = nvvk::buildEnvAliasmap(data, accel);
  EXPECT_NEAR(integral, static_cast<float>(refSum), 1e-3F);

  const std::vector<double> p      = computeSamplingDistribution(accel);
  double                    totalP = 0.0;
  for(size_t j = 0; j < kSize; ++j)
  {
    const double expected = static_cast<double>(data[j]) / refSum;
    // The tolerance is generous because Vose is done in single-precision floats.
    EXPECT_NEAR(p[j], expected, 1e-4) << "P(sample " << j << ") diverges from input importance";
    totalP += p[j];
  }
  EXPECT_NEAR(totalP, 1.0, 1e-4) << "Sampling distribution must integrate to 1";
}


TEST(AliasTable, AliasIndicesAreInBounds)
{
  constexpr size_t kSize = 1024;
  const auto       data  = makeRandom(kSize, 0xC0FFEEu);

  std::vector<shaderio::EnvAccel> accel(kSize);
  nvvk::buildEnvAliasmap(data, accel);

  for(size_t i = 0; i < kSize; ++i)
  {
    EXPECT_LT(accel[i].alias, static_cast<uint32_t>(kSize)) << "alias[" << i << "] out of range";
    EXPECT_GE(accel[i].q, 0.0F) << "q[" << i << "] must be non-negative";
  }
}


TEST(AliasTable, ZeroImportanceProducesFallbackIntegral)
{
  // All zeros -> integral would be 0; buildEnvAliasmap must return 1 (the sentinel) to avoid
  // downstream division by zero in HdrIbl.
  const std::vector<float>        data(64, 0.0F);
  std::vector<shaderio::EnvAccel> accel(64);
  const float                     integral = nvvk::buildEnvAliasmap(data, accel);

  EXPECT_FLOAT_EQ(integral, 1.0F);
  for(size_t i = 0; i < accel.size(); ++i)
  {
    EXPECT_FLOAT_EQ(accel[i].q, 0.0F);
    EXPECT_EQ(accel[i].alias, static_cast<uint32_t>(i));
  }
}
