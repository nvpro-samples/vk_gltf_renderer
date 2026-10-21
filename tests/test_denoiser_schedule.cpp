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

// AutoDenoiseSchedule: when the OptiX and OIDN denoisers run on their own, in samples per pixel.
// handoffWeight: the share of the denoised image during the DLSS hand-off.

#include <vector>

#include <gtest/gtest.h>

#include "denoiser_schedule.hpp"

namespace {

// Rendered frames of `samplesPerFrame` up to `maxSamples` (the last one traces only what is missing),
// then idle frames. Returns the spp of each denoise, the way OIDN schedules: the image read includes
// the frame's samples, and the frame that reaches the cap is final. `handoffEnd`: see isDue().
std::vector<int> runAccumulation(AutoDenoiseSchedule& schedule, int samplesPerFrame, int maxSamples, int interval, int handoffEnd = 0)
{
  std::vector<int> denoised;
  for(int samples = 0; samples < maxSamples;)
  {
    samples = std::min(samples + samplesPerFrame, maxSamples);
    if(schedule.isDue(samples, interval, samples >= maxSamples, handoffEnd))
    {
      schedule.markDenoised(samples);
      denoised.push_back(samples);
    }
  }
  for(int idle = 0; idle < 3; ++idle)
    if(schedule.isFinalDue(maxSamples))
    {
      schedule.markDenoised(maxSamples);
      denoised.push_back(maxSamples);
    }
  return denoised;
}

}  // namespace

TEST(AutoDenoiseSchedule, DenoisesAtMultiplesAndTheFinalImage)
{
  AutoDenoiseSchedule schedule;
  // 3 spp per frame: 3, 6, 9, 10. Multiples of 4 fall inside frames 6 and 9; 10 is the cap.
  EXPECT_EQ(runAccumulation(schedule, 3, 10, 4), (std::vector<int>{6, 9, 10}));
}

TEST(AutoDenoiseSchedule, FinalOnAMultipleIsDenoisedOnce)
{
  AutoDenoiseSchedule schedule;
  EXPECT_EQ(runAccumulation(schedule, 1, 8, 4), (std::vector<int>{4, 8}));
}

TEST(AutoDenoiseSchedule, SingleFrameAccumulationsDenoiseEachRestart)
{
  // Every accumulation is one frame reaching the cap (e.g. a camera moving each frame). The count
  // does not grow from one to the next, which is a restart, so each is denoised.
  AutoDenoiseSchedule schedule;
  for(int i = 0; i < 3; ++i)
    EXPECT_EQ(runAccumulation(schedule, 4, 4, 100), (std::vector<int>{4}));
}

TEST(AutoDenoiseSchedule, IdleFrameDenoisesAFinalImageNotYetDenoised)
{
  // One frame behind (OptiX): the frame reaching the cap is never seen by isDue(), the idle frame
  // denoises it, and only once.
  AutoDenoiseSchedule schedule;
  for(int samples : {0, 3, 6, 9})  // the image read, before each rendered frame's samples
    if(schedule.isDue(samples, 4))
      schedule.markDenoised(samples);
  EXPECT_TRUE(schedule.isFinalDue(10));
  schedule.markDenoised(10);
  EXPECT_FALSE(schedule.isFinalDue(10));
}

TEST(AutoDenoiseSchedule, NewNetworkDenoisesTheFinalImageAgain)
{
  // The render is done and its final image denoised; a new network (quality, guides) makes it due again.
  AutoDenoiseSchedule schedule;
  EXPECT_EQ(runAccumulation(schedule, 4, 16, 8), (std::vector<int>{8, 16}));
  EXPECT_FALSE(schedule.isFinalDue(16));
  schedule.forgetDenoised();
  EXPECT_TRUE(schedule.isFinalDue(16));
}

TEST(AutoDenoiseSchedule, NextSamplesStopsAtTheCap)
{
  EXPECT_EQ(AutoDenoiseSchedule::nextSamples(5, 4, 10), 8);
  EXPECT_EQ(AutoDenoiseSchedule::nextSamples(9, 4, 10), 10);
  EXPECT_EQ(AutoDenoiseSchedule::nextSamples(9, 0, 10), 0);
}

TEST(AutoDenoiseSchedule, HandoffDenoisesWhenTheSamplesDouble)
{
  // A DLSS hand-off ending at 64 spp: each doubling, 64 itself, then the multiples of the interval
  AutoDenoiseSchedule schedule;
  EXPECT_EQ(runAccumulation(schedule, 1, 200, 50, 64), (std::vector<int>{1, 2, 4, 8, 16, 32, 64, 100, 150, 200}));
}

TEST(AutoDenoiseSchedule, HandoffWithSeveralSamplesPerFrame)
{
  // 3 spp per frame: the first frame whose count doubled denoises; 64 falls inside the frame to 66
  AutoDenoiseSchedule schedule;
  EXPECT_EQ(runAccumulation(schedule, 3, 120, 50, 64), (std::vector<int>{3, 6, 12, 24, 48, 66, 102, 120}));
}

TEST(AutoDenoiseSchedule, HandoffRestartsWithTheAccumulation)
{
  AutoDenoiseSchedule schedule;
  EXPECT_EQ(runAccumulation(schedule, 1, 10, 50, 8), (std::vector<int>{1, 2, 4, 8, 10}));
  EXPECT_EQ(runAccumulation(schedule, 1, 10, 50, 8), (std::vector<int>{1, 2, 4, 8, 10}));
}

TEST(HandoffWeight, SmoothstepOverTheHandoff)
{
  EXPECT_EQ(handoffWeight(0, 64), 0.0f);
  EXPECT_FLOAT_EQ(handoffWeight(32, 64), 0.5f);
  EXPECT_FLOAT_EQ(handoffWeight(16, 64), 0.15625f);
  EXPECT_EQ(handoffWeight(64, 64), 1.0f);
  EXPECT_EQ(handoffWeight(1000, 64), 1.0f);
  EXPECT_EQ(handoffWeight(5, 0), 1.0f);  // no hand-off
}

TEST(GuideAccumulation, CountsSamplesSinceTheGuidesRestarted)
{
  GuideAccumulation guides;
  EXPECT_EQ(guides.update(true, 0), 0);  // first frame of the accumulation: replace
  EXPECT_EQ(guides.update(true, 4), 4);
  EXPECT_EQ(guides.update(true, 8), 8);
  EXPECT_TRUE(guides.isCurrent());
}

TEST(GuideAccumulation, RestartsWhenTurnedOnMidAccumulation)
{
  GuideAccumulation guides;
  EXPECT_EQ(guides.update(false, 0), 0);   // denoiser off: guides not written
  EXPECT_EQ(guides.update(false, 16), 0);  // still off
  EXPECT_FALSE(guides.isCurrent());
  EXPECT_EQ(guides.update(true, 32), 0);  // turned on: replace, while the color keeps its 32 samples
  EXPECT_EQ(guides.update(true, 36), 4);  // weighted by its own 4 samples, not the color's 36
}

TEST(GuideAccumulation, RestartsWithTheAccumulationAndAfterReallocation)
{
  GuideAccumulation guides;
  guides.update(true, 0);
  guides.update(true, 10);
  EXPECT_EQ(guides.update(true, 0), 0);  // the accumulation restarted
  EXPECT_EQ(guides.update(true, 1), 1);
  guides.restart();  // images reallocated
  EXPECT_FALSE(guides.isCurrent());
  EXPECT_EQ(guides.update(true, 2), 0);
}
