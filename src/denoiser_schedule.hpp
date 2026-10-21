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

#include <algorithm>

//-------------------------------------------------------------------------------------------------------
// AutoDenoiseSchedule
//
// When the OptiX and OIDN denoisers run on their own: each time the accumulated image reaches the next
// multiple of `interval` samples per pixel (interval, 2*interval, ...). Counting samples, not frames,
// makes the schedule independent of the samples per frame: with several per frame a multiple can fall
// inside a frame, and the image of that frame is denoised. An interval of 1 denoises every frame.
// The final image of an accumulation (it reached the samples cap) is denoised too, off the interval:
// it is the one shown from then on.
//
// During a hand-off from DLSS (DenoiserHandoff), the first denoises come sooner: one each time the
// samples double (1, 2, 4, ...), and one when they reach `handoffEnd`, then the interval as usual.
// Each doubling halves the noise variance, so each of these denoises is a visible step.
//
//   if(schedule.isDue(samples, interval, final, handoffEnd) && denoise())   // a rendered frame
//     schedule.markDenoised(samples);
//   if(schedule.isFinalDue(samples) && denoise())                           // an idle frame, accumulation complete
//     schedule.markDenoised(samples);
//
// `samples` is that of the image the denoiser reads. It grows on every rendered frame, so a count that
// does not grow means the accumulation restarted.
//-------------------------------------------------------------------------------------------------------
class AutoDenoiseSchedule
{
public:
  // A rendered frame: true when `samples` reached a multiple of `interval` the last denoise had not,
  // or is the final image of the accumulation (`final`). Before `handoffEnd` (0: no hand-off), true
  // when `samples` doubled since the last denoise or reached `handoffEnd`.
  bool isDue(int samples, int interval, bool final = false, int handoffEnd = 0)
  {
    if(samples <= m_lastSamples)  // the accumulation restarted
      m_lastSamples = 0;
    if(samples <= 0)
      return false;
    if(final)
      return true;
    if(m_lastSamples < handoffEnd)
      return samples >= std::min(2 * m_lastSamples, handoffEnd);
    if(interval <= 0)
      return false;
    return samples / interval > m_lastSamples / interval;
  }

  // An idle frame, after the accumulation completed: true when its final image, of `samples`, is not
  // the one last denoised. The count no longer grows there, so an equal count is not a restart.
  bool isFinalDue(int samples) const { return samples > 0 && samples != m_lastSamples; }

  void markDenoised(int samples) { m_lastSamples = samples; }

  // The network changed (quality, guides): no denoise so far is current, the next frame denoises
  void forgetDenoised() { m_lastSamples = 0; }

  // The samples per pixel at which isDue() accepts next, for display: the next multiple, or the cap
  static int nextSamples(int samples, int interval, int maxSamples)
  {
    return interval > 0 ? std::min((samples / interval + 1) * interval, maxSamples) : 0;
  }

private:
  int m_lastSamples = 0;  // Samples per pixel of the last denoise
};

//-------------------------------------------------------------------------------------------------------
// handoffWeight
//
// The DLSS hand-off (DenoiserHandoff) shows lerp(dlss, denoised, handoffWeight(samples, handoffEnd)).
// A smoothstep over the samples per pixel: the denoised image weighs little while it comes from few
// samples and is still blotchy, then takes over, and is exactly 1 from `handoffEnd` on, so the image
// left on screen is the denoised one alone. 0 or less for `handoffEnd` means no hand-off: 1.
//-------------------------------------------------------------------------------------------------------
inline float handoffWeight(int samples, int handoffEnd)
{
  if(handoffEnd <= 0)
    return 1.0f;
  const float t = std::clamp(float(samples) / float(handoffEnd), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

//-------------------------------------------------------------------------------------------------------
// GuideAccumulation
//
// The albedo and normal guides of the OptiX and OIDN denoisers are accumulated by the path tracer like
// the color, but with a sample count of their own: they restart when they were not written on the
// previous frame (the denoiser was just turned on, or its images were reallocated) while the color
// accumulation goes on. Each denoiser keeps one for its guide images.
//
//   pushConst.guideSamples = guides.update(writtenThisFrame, totalSamples);  // before each rendered frame
//
// update() returns the samples already in the guides, which weight this frame's: 0 means replace them.
//-------------------------------------------------------------------------------------------------------
class GuideAccumulation
{
public:
  // `totalSamples` is the color's sample count before this frame; a smaller one than at the restart
  // means the accumulation itself restarted
  int update(bool writtenThisFrame, int totalSamples)
  {
    if(!m_current || totalSamples < m_start)
      m_start = totalSamples;
    m_current = writtenThisFrame;
    return totalSamples - m_start;
  }

  // The guides were written by the last rendered frame
  bool isCurrent() const { return m_current; }

  // The guide images were reallocated: their content is gone
  void restart() { m_current = false; }

private:
  bool m_current = false;  // Written by the last rendered frame
  int  m_start   = 0;      // Color sample count when the guides restarted
};
