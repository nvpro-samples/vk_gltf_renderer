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

#pragma once

//-------------------------------------------------------------------------------------------------------
// How many samples per pixel the path tracer traces in a frame.
//
// frameSamples():         the frame ends exactly on the next stop -- the samples cap, or a count
//                         --outputSamples saves -- so a stop is never stepped over, whatever the samples
//                         per frame.
// FrameRateControl:       Auto SPP's frame-rate targets: climbs to the target frame time, then holds
//                         the count while the view is still.
// ThroughputSearch:       Auto SPP's Throughput target, the samples per frame with the lowest frame
//                         time per sample.
//-------------------------------------------------------------------------------------------------------

#include <algorithm>
#include <span>

namespace sample_schedule {

// Samples per pixel to trace this frame: `perFrame`, or fewer when the accumulation would step over
// `maxSamples` or the next of the sorted `stops` above `accumulated`. Always at least one.
inline int frameSamples(int accumulated, int perFrame, int maxSamples, std::span<const int> stops)
{
  int count = std::min(perFrame, maxSamples - accumulated);
  if(const auto next = std::upper_bound(stops.begin(), stops.end(), accumulated); next != stops.end())
    count = std::min(count, *next - accumulated);
  return std::max(count, 1);
}

// Auto SPP's frame-rate targets, over one accumulation (restart() on every reset). From one sample per
// frame it adds one each frame until the path-trace GPU time reaches the target, steps back down while
// it is over the target, and once it has reached the target it never climbs again until the next reset:
// the view is still, so the cost of a sample does not change, and climbing again on a timing wobble is
// what made the count fluctuate. The GPU time lags the count by a few frames, which a step of one keeps
// small.
class FrameRateControl
{
public:
  static constexpr double kLow  = 0.8;  // Under this fraction of the target the count climbs
  static constexpr double kHigh = 1.1;  // Over this fraction of the target it steps down

  void restart() { *this = {}; }
  bool held() const { return m_held; }

  // The latest path-trace GPU time `gpuMs`, with `samples` the count returned last time. Returns the
  // count to trace next.
  int update(int samples, double gpuMs, double targetMs, int minSamples, int maxSamples)
  {
    if(gpuMs <= 0.0)
      return samples;
    if(gpuMs > targetMs * kHigh)
      return std::max(samples - 1, minSamples);
    if(gpuMs >= targetMs * kLow)
      m_held = true;
    if(!m_held && samples < maxSamples)
      return samples + 1;
    return samples;
  }

private:
  bool m_held{false};  // The target was reached: no more climbing in this accumulation
};

// Auto SPP's Throughput target: the samples per frame with the lowest frame time per sample. Several
// samples per frame amortize the per-frame work, but on some scenes they cost more per sample than one
// (measured on complex scenes), so the count is searched, not derived from a frame time. The frame time
// is the whole frame's (wall time), since the per-frame work is what the samples share. From one, it
// tries counts kStep apart until a frame would pass `maxFrameMs`, then settles on the one with the
// lowest time per sample; a larger count must beat a smaller one by more than the timing noise. All are
// tried: the gain of one step can be small (a few percent) while the larger counts gain more. Each is
// measured over a span of time rather than a number of frames, since short frames are the noisiest.
class ThroughputSearch
{
public:
  static constexpr int    kStep         = 4;      // Each count tried is kStep times the previous one
  static constexpr int    kMinReadings  = 4;      // Frame times averaged per count, at least...
  static constexpr double kMinMeasureMs = 250.0;  // ...and over this much time
  static constexpr double kMinGain      = 0.01;   // A smaller gain per sample is noise

  // The time `frameMs` of a frame of `samples` per pixel (the count returned last time). Returns the
  // count to trace next.
  int update(int samples, double frameMs, double maxFrameMs, int maxSamples)
  {
    if(m_settled)
      return m_best;
    if(samples != m_measuring)  // a new count: start averaging
    {
      m_measuring = samples;
      m_sumMs     = 0.0;
      m_readings  = 0;
    }
    if(frameMs <= 0.0)
      return samples;
    m_sumMs += frameMs;
    if(++m_readings < kMinReadings || m_sumMs < kMinMeasureMs)
      return samples;

    const double perSample = m_sumMs / m_readings / samples;
    if(m_best == 0 || perSample < m_bestPerSample * (1.0 - kMinGain))
    {
      m_best          = samples;
      m_bestPerSample = perSample;
    }
    const int next = samples * kStep;
    if(next > maxSamples || perSample * next > maxFrameMs)  // the next frame would be too long
      return settle();
    return next;
  }

  bool settled() const { return m_settled; }
  int  best() const { return m_best; }

private:
  int settle()
  {
    m_settled = true;
    return m_best;
  }

  int    m_measuring{0};  // Count whose times are being averaged
  double m_sumMs{0.0};
  int    m_readings{0};
  int    m_best{0};  // Count with the lowest time per sample so far
  double m_bestPerSample{0.0};
  bool   m_settled{false};
};

}  // namespace sample_schedule
