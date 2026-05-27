/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Benchmark controller. Bridges the nvutils parameter registry (driven by the
// external benchmark script) with the renderer's camera/scene/screenshot
// callbacks, and produces machine-readable progress, summary and memory
// telemetry for headless runs. Output is emitted on stdout as both a
// human-readable line and a "BENCHMARK_JSON ..." line consumed by the
// post-processing tools in utils/benchmark/.
//

#include "benchmarking.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

#include <nvutils/logger.hpp>
#include <tinygltf/json.hpp>

namespace {

using json = nlohmann::json;

//--------------------------------------------------------------------------------------------------
// Print a single JSON record on its own line, prefixed with a stable sentinel
// ("BENCHMARK_JSON") so the benchmark scripts can grep it out of the regular
// log stream. A schema version is stamped to keep the consumer side forward
// compatible if the payload evolves.
void emitJsonLine(json record)
{
  record["schema"] = 1;
  std::cout << "BENCHMARK_JSON " << record.dump() << std::endl;
}

//--------------------------------------------------------------------------------------------------
// Round to a fixed number of decimal places (scale=1000 => 3 decimals) to keep
// the emitted JSON values compact and stable across runs.
double roundTo(double value, double scale)
{
  return std::round(value * scale) / scale;
}

//--------------------------------------------------------------------------------------------------
// Legacy plain-text memory line kept for backward compatibility with older
// benchmark log parsers. The same data is also emitted in JSON form below.
void printLegacyMemoryLine(const BenchmarkController::MemorySample& sample)
{
  std::cout << " Memory " << sample.category << "; Host used \t" << sample.hostUsed << "; Device Used \t"
            << sample.deviceUsed << "; Device Allocated \t" << sample.deviceAllocated << "; (bytes)" << std::endl;
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// Holds a reference to the shared BenchmarkOptions so that command-line / script
// driven values stay in sync with whatever the rest of the application reads.
BenchmarkController::BenchmarkController(BenchmarkOptions& options)
    : m_options(options)
{
}

//--------------------------------------------------------------------------------------------------
// Register the parameters that the external benchmark script can drive through
// the nvutils parameter registry (parsed from the .cfg files in utils/benchmark/).
//
// Each parameter is wired to one of the renderer callbacks supplied by the
// application so that the controller stays decoupled from the renderer's
// internals. Most parameters also implicitly reset path-tracer accumulation
// because the script typically changes the view/scene between captures.
void BenchmarkController::registerParameters(nvutils::ParameterRegistry* parameterRegistry, Callbacks callbacks)
{
  m_callbacks = std::move(callbacks);

  parameterRegistry->add({.name = "gltfCamera",
                          .help = "Apply glTF camera by index (benchmark script). Resets path-tracer accumulation.",
                          .callbackSuccess =
                              [this](const nvutils::ParameterBase* const) {
                                if(m_callbacks.applyGltfCamera)
                                {
                                  m_callbacks.applyGltfCamera(m_options.gltfCameraIndex);
                                }
                                if(m_callbacks.resetFrame)
                                {
                                  m_callbacks.resetFrame();
                                }
                              }},
                         &m_options.gltfCameraIndex);

  parameterRegistry->add({.name = "fitScene",
                          .help = "Fit camera to scene bounds (benchmark script). Resets path-tracer accumulation.",
                          .callbackSuccess =
                              [this](const nvutils::ParameterBase* const) {
                                if(m_callbacks.fitScene)
                                {
                                  m_callbacks.fitScene();
                                }
                                if(m_callbacks.resetFrame)
                                {
                                  m_callbacks.resetFrame();
                                }
                              }},
                         &m_options.fitSceneTrigger, true);

  parameterRegistry->add({.name = "resetFrame",
                          .help = "Reset path-tracer frame accumulation (benchmark script).",
                          .callbackSuccess =
                              [this](const nvutils::ParameterBase* const) {
                                if(m_callbacks.resetFrame)
                                {
                                  m_callbacks.resetFrame();
                                }
                              }},
                         &m_options.resetFrameTrigger, true);

  parameterRegistry->add({.name = "updateData",
                          .help = "Alias for resetFrame after settings change (benchmark script).",
                          .callbackSuccess =
                              [this](const nvutils::ParameterBase* const) {
                                if(m_callbacks.resetFrame)
                                {
                                  m_callbacks.resetFrame();
                                }
                              }},
                         &m_options.updateDataTrigger, true);

  parameterRegistry->add({.name = "screenshot",
                          .help = "Save tonemapped render to file (benchmark script).",
                          .callbackSuccess =
                              [this](const nvutils::ParameterBase* const) {
                                if(m_callbacks.saveScreenshot && !m_options.screenshotFilename.empty())
                                {
                                  m_callbacks.saveScreenshot(m_options.screenshotFilename);
                                }
                              }},
                         {".png", ".jpg", ".jpeg"}, &m_options.screenshotFilename);
}

//--------------------------------------------------------------------------------------------------
// Ensure the path-tracer accumulation budget (maxFrames) is at least as large
// as the number of frames the headless run will render. Without this, the
// accumulator would stop refining before the captured screenshot is taken and
// the result would be noisier than expected.
void BenchmarkController::alignMaxFramesForHeadless(int& maxFrames, uint32_t headlessFrames)
{
  const int minMaxFrames = static_cast<int>(headlessFrames);
  if(maxFrames < minMaxFrames)
  {
    LOGW("maxFrames (%d) is less than headless --frames (%u); setting maxFrames to %u\n", maxFrames, headlessFrames, headlessFrames);
    maxFrames = minMaxFrames;
  }
}

//--------------------------------------------------------------------------------------------------
// Start wall-clock timing for a headless run. The first call (when not already
// active) latches the timer and emits a HEADLESS_START record so consumers can
// correlate the subsequent progress/summary records with this run.
void BenchmarkController::beginHeadlessTimingIfNeeded(bool isHeadless, const HeadlessFrameInfo& info)
{
  if(!isHeadless || m_headlessTimingActive)
  {
    return;
  }

  m_headlessWallTimer.reset();
  m_headlessMeasuredTimer.reset();
  m_headlessTimingActive         = true;
  m_headlessMeasuredTimingActive = false;
  m_headlessFramesDone           = 0;
  m_headlessMeasuredStartFrame   = 0;
  m_headlessLastProgressLogMs    = 0.0;

  LOGI("HEADLESS_START frames=%u maxFrames=%d ptSamples=%d\n", info.totalFrames, info.maxFrames, info.ptSamples);
  emitJsonLine({{"type", "headless_start"}, {"frames", info.totalFrames}, {"maxFrames", info.maxFrames}, {"ptSamples", info.ptSamples}});
}

//--------------------------------------------------------------------------------------------------
// Called once per rendered frame during a headless run. Emits a progress
// record only on selected frames to avoid drowning the log: the first frame,
// the last frame, every Nth frame, or whenever at least kHeadlessLogMinIntervalMs
// has passed since the previous report (whichever happens first).
void BenchmarkController::updateHeadlessProgressIfNeeded(const HeadlessFrameInfo& info)
{
  if(!m_headlessTimingActive)
  {
    return;
  }

  ++m_headlessFramesDone;
  const double elapsedMs   = m_headlessWallTimer.getMilliseconds();
  const bool   onInterval  = (m_headlessFramesDone % kHeadlessLogEveryNFrames) == 0;
  const bool   onTime      = (elapsedMs - m_headlessLastProgressLogMs) >= kHeadlessLogMinIntervalMs;
  const bool   firstOrLast = m_headlessFramesDone == 1 || m_headlessFramesDone >= info.totalFrames;

  if(!m_headlessMeasuredTimingActive && m_headlessFramesDone >= kHeadlessWarmupFrames)
  {
    m_headlessMeasuredTimer.reset();
    m_headlessMeasuredTimingActive = true;
    m_headlessMeasuredStartFrame   = m_headlessFramesDone;
  }

  if(!firstOrLast && !onInterval && !onTime)
  {
    return;
  }

  const float pct =
      info.totalFrames > 0 ? 100.0F * static_cast<float>(m_headlessFramesDone) / static_cast<float>(info.totalFrames) : 0.0F;
  const double msPerFrame = m_headlessFramesDone > 0 ? elapsedMs / static_cast<double>(m_headlessFramesDone) : 0.0;

  LOGI("HEADLESS_PROGRESS app_frame %u/%u (%.0f%%) elapsed_ms=%.1f ms_per_frame=%.2f\n", m_headlessFramesDone,
       info.totalFrames, pct, elapsedMs, msPerFrame);
  emitJsonLine({{"type", "headless_progress"},
                {"app_frame", m_headlessFramesDone},
                {"frames", info.totalFrames},
                {"percent", roundTo(pct, 1000.0)},
                {"elapsed_ms", roundTo(elapsedMs, 1000.0)},
                {"ms_per_frame", roundTo(msPerFrame, 1000.0)}});
  m_headlessLastProgressLogMs = elapsedMs;
}

//--------------------------------------------------------------------------------------------------
// Emit the final headless summary: wall time, ms/frame, and the equivalent
// path-tracing throughput in megasamples per second.
//
// effective_spp accounts for both the number of accumulated frames (capped to
// maxFrames so we don't count frames that no longer contribute) and the
// per-frame sample count, giving a single figure of merit comparable across
// runs at different resolutions.
void BenchmarkController::logHeadlessSummary(const HeadlessFrameInfo& info)
{
  if(!m_headlessTimingActive)
  {
    return;
  }

  const double totalWallMs     = m_headlessWallTimer.getMilliseconds();
  const double totalMsPerFrame = info.totalFrames > 0 ? totalWallMs / static_cast<double>(info.totalFrames) : 0.0;

  const uint32_t completedFrames = std::min(m_headlessFramesDone, info.totalFrames);
  uint32_t       warmupFrames    = 0;
  uint32_t       measuredFrames  = completedFrames;
  double         measuredWallMs  = totalWallMs;
  if(m_headlessMeasuredTimingActive && completedFrames >= m_headlessMeasuredStartFrame)
  {
    warmupFrames   = m_headlessMeasuredStartFrame;
    measuredFrames = completedFrames - m_headlessMeasuredStartFrame;
    measuredWallMs = measuredFrames > 0 ? m_headlessMeasuredTimer.getMilliseconds() : 0.0;
  }

  const double measuredWallSec    = measuredWallMs / 1000.0;
  const double measuredMsPerFrame = measuredFrames > 0 ? measuredWallMs / static_cast<double>(measuredFrames) : 0.0;

  const int accumFrames = std::min(static_cast<int>(info.totalFrames), std::max(info.maxFrames, 0));
  const int measuredAccumFrames = std::clamp(accumFrames - static_cast<int>(warmupFrames), 0, static_cast<int>(measuredFrames));
  const int      effectiveSpp         = accumFrames * std::max(info.ptSamples, 1);
  const int      measuredEffectiveSpp = measuredAccumFrames * std::max(info.ptSamples, 1);
  const uint64_t pixels = static_cast<uint64_t>(info.imageSize.width) * static_cast<uint64_t>(info.imageSize.height);
  const double   measuredSamples = static_cast<double>(pixels) * static_cast<double>(measuredEffectiveSpp);
  const double   throughputMSps  = measuredWallSec > 0.0 ? measuredSamples / measuredWallSec / 1e6 : 0.0;
  const double   sppPerSec = measuredWallSec > 0.0 ? static_cast<double>(measuredEffectiveSpp) / measuredWallSec : 0.0;

  LOGI(
      "HEADLESS_SUMMARY frames=%u maxFrames=%d ptSamples=%d effective_spp=%d measured_effective_spp=%d "
      "resolution=%ux%u wall_ms=%.3f ms_per_frame=%.3f total_wall_ms=%.3f total_ms_per_frame=%.3f "
      "warmup_frames=%u measured_frames=%u throughput_MSps=%.3f spp_per_sec=%.2f\n",
      info.totalFrames, info.maxFrames, info.ptSamples, effectiveSpp, measuredEffectiveSpp, info.imageSize.width,
      info.imageSize.height, measuredWallMs, measuredMsPerFrame, totalWallMs, totalMsPerFrame, warmupFrames,
      measuredFrames, throughputMSps, sppPerSec);
  emitJsonLine({{"type", "headless_summary"},
                {"frames", info.totalFrames},
                {"maxFrames", info.maxFrames},
                {"ptSamples", info.ptSamples},
                {"effective_spp", effectiveSpp},
                {"measured_effective_spp", measuredEffectiveSpp},
                {"resolution_w", info.imageSize.width},
                {"resolution_h", info.imageSize.height},
                {"wall_ms", roundTo(measuredWallMs, 1000.0)},
                {"ms_per_frame", roundTo(measuredMsPerFrame, 1000.0)},
                {"total_wall_ms", roundTo(totalWallMs, 1000.0)},
                {"total_ms_per_frame", roundTo(totalMsPerFrame, 1000.0)},
                {"warmup_frames", warmupFrames},
                {"measured_frames", measuredFrames},
                {"throughput_MSps", roundTo(throughputMSps, 1000.0)},
                {"spp_per_sec", roundTo(sppPerSec, 100.0)}});
}

//--------------------------------------------------------------------------------------------------
// Stop the headless timer. Subsequent progress/summary calls become no-ops
// until beginHeadlessTimingIfNeeded() is called again.
void BenchmarkController::finishHeadlessTiming()
{
  m_headlessTimingActive         = false;
  m_headlessMeasuredTimingActive = false;
}

//--------------------------------------------------------------------------------------------------
// Emit a memory snapshot for one benchmark sequence (one entry in the .cfg
// matrix). Two outputs are produced: the legacy "BENCHMARK_ADV { ... }" block
// for older parsers, and a structured JSON record for the current tooling.
// The sequence id is auto-incremented so consumers can join memory records
// with the matching timing/screenshot records.
void BenchmarkController::emitSequenceMemory(const std::vector<MemorySample>& samples)
{
  std::cout << "BENCHMARK_ADV " << m_sequenceId << " {" << std::endl;
  for(const MemorySample& sample : samples)
  {
    printLegacyMemoryLine(sample);
  }
  std::cout << "}" << std::endl;

  json memory = json::array();
  for(const MemorySample& sample : samples)
  {
    memory.push_back({{"category", sample.category},
                      {"host_used", sample.hostUsed},
                      {"device_used", sample.deviceUsed},
                      {"device_allocated", sample.deviceAllocated}});
  }
  emitJsonLine({{"type", "sequence_memory"}, {"id", m_sequenceId}, {"memory", memory}});

  ++m_sequenceId;
}
