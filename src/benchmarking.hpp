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

#pragma once

//
// Benchmark controller. Wires the external benchmark script (see
// utils/benchmark/) into the renderer through the nvutils parameter
// registry, and emits human-readable + JSON telemetry for headless runs
// (start/progress/summary) and per-sequence memory snapshots.
//
// The controller owns no renderer state; the application supplies
// Callbacks that perform the actual camera/scene/screenshot mutations.
//

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include <vulkan/vulkan_core.h>

#include <nvutils/parameter_registry.hpp>
#include <nvutils/timers.hpp>

// Options driven by the command line and the benchmark script. A single
// instance is owned by the application and shared (by reference) with the
// BenchmarkController so script-driven changes are observed everywhere.
struct BenchmarkOptions
{
  bool                  enabled{false};            // True when running under the benchmark harness
  int                   gltfCameraIndex{0};        // Index of the glTF camera to activate
  bool                  fitSceneTrigger{false};    // Pulse: fit camera to scene bounds
  bool                  resetFrameTrigger{false};  // Pulse: reset path-tracer accumulation
  bool                  updateDataTrigger{false};  // Pulse: alias of resetFrame after settings change
  std::filesystem::path screenshotFilename;        // Output path for the next screenshot capture
};

//--------------------------------------------------------------------------------------------------
// BenchmarkController - parameter registration, headless timing, and telemetry
//
// Lifetime: created once at startup, kept alive for the whole session.
// registerParameters() must be called before the parameter registry parses
// any script so the benchmark commands are recognized.
//--------------------------------------------------------------------------------------------------
class BenchmarkController
{
public:
  // Renderer-side hooks invoked from the script-driven parameter callbacks.
  // Any callback may be left empty; the controller will silently skip it.
  struct Callbacks
  {
    std::function<void(int)>                          applyGltfCamera;  // Apply glTF camera by index
    std::function<void()>                             fitScene;         // Fit camera to scene bounds
    std::function<void()>                             resetFrame;       // Reset path-tracer accumulation
    std::function<void(const std::filesystem::path&)> saveScreenshot;   // Save tonemapped render
  };

  // Snapshot of the headless run configuration, passed to every timing call.
  // Cheap to copy and intentionally free of renderer types.
  struct HeadlessFrameInfo
  {
    uint32_t   totalFrames{0};  // Frames the headless run will render
    int        maxFrames{0};    // Path-tracer accumulation budget
    int        ptSamples{1};    // Samples per pixel per frame
    VkExtent2D imageSize{};     // Render target resolution
  };

  // One row of memory usage at a benchmark sequence boundary. Bytes are
  // raw counts as reported by the GPU memory tracker; the consumer is
  // responsible for any unit conversion.
  struct MemorySample
  {
    std::string category;            // Human-readable category (e.g. "Geometry", "Images")
    uint64_t    hostUsed{0};         // Host-visible bytes currently in use
    uint64_t    deviceUsed{0};       // Device-local bytes currently in use
    uint64_t    deviceAllocated{0};  // Device-local bytes reserved (>= used)
  };

  explicit BenchmarkController(BenchmarkOptions& options);

  // Register all benchmark-script-driven parameters with the registry and
  // store the renderer callbacks they will invoke.
  void registerParameters(nvutils::ParameterRegistry* parameterRegistry, Callbacks callbacks);

  // True when the application is running under the benchmark harness.
  [[nodiscard]] bool isBenchmarkMode() const { return m_options.enabled; }

  // Raise maxFrames to at least headlessFrames so path-tracer accumulation
  // keeps refining for the whole capture. Logs a warning when it changes.
  static void alignMaxFramesForHeadless(int& maxFrames, uint32_t headlessFrames);

  // Begin / update / finalize wall-clock timing for a headless run.
  // beginHeadlessTimingIfNeeded() is idempotent within a single run, and
  // update/summary become no-ops once finishHeadlessTiming() is called.
  void beginHeadlessTimingIfNeeded(bool isHeadless, const HeadlessFrameInfo& info);
  void updateHeadlessProgressIfNeeded(const HeadlessFrameInfo& info);
  void logHeadlessSummary(const HeadlessFrameInfo& info);
  void finishHeadlessTiming();

  // Emit a memory snapshot for the current benchmark sequence. The internal
  // sequence id is incremented on each call so downstream tools can join
  // memory records with the corresponding timing/screenshot records.
  void emitSequenceMemory(const std::vector<MemorySample>& samples);

private:
  // Throttling for headless progress logs: emit at most every N frames or
  // every kHeadlessLogMinIntervalMs of wall time, whichever comes first.
  static constexpr uint32_t kHeadlessLogEveryNFrames  = 50;
  static constexpr double   kHeadlessLogMinIntervalMs = 5000.0;

  BenchmarkOptions&         m_options;                         // Shared with the application
  Callbacks                 m_callbacks;                       // Renderer hooks (may be empty)
  nvutils::PerformanceTimer m_headlessWallTimer;               // Wall-clock for the active headless run
  bool                      m_headlessTimingActive{false};     // Guard for begin/update/summary calls
  uint32_t                  m_headlessFramesDone{0};           // Frames completed in the active run
  double                    m_headlessLastProgressLogMs{0.0};  // Wall time of the last progress log
  int                       m_sequenceId{0};                   // Auto-incremented per emitSequenceMemory() call
};
