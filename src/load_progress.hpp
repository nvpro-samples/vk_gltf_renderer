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

/*
 * Stage-by-stage progress of a scene load, reported by the loader and read by the busy window.
 *
 * A load is a sequence of named stages ("Reading glTF", "Decoding textures", ...). A stage may have
 * a known item count that workers advance in parallel, and a detail line (the file just finished).
 * Finished stages keep their duration, so the dialog shows where the time went.
 *
 * Every call is thread-safe, and every call but begin() is a no-op while no load is active. That
 * lets shared code (GPU rebuilds, external-asset placement) report unconditionally: outside a load
 * it costs nothing and shows nothing.
 *
 * Usage:
 *   progress.begin("sponza.gltf");          // worker thread, start of the load
 *   progress.stage("Decoding textures", n);
 *   parallel_for(... { decode(i); progress.advance(1, name); });
 *   progress.end();                          // when the last GPU work has completed
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <nvutils/logger.hpp>

class LoadProgress
{
public:
  struct Stage
  {
    std::string name{};
    uint32_t    total   = 0;  // item count, 0 when the stage is not counted
    double      seconds = 0.0;
  };

  // What the UI draws: a consistent copy taken under the lock.
  struct Snapshot
  {
    bool               active = false;
    std::string        title{};     // what is being loaded
    std::vector<Stage> finished{};  // in order
    Stage              current{};   // seconds = time spent in it so far
    uint32_t           done = 0;    // items of `current` finished
    std::string        detail{};    // last item finished in `current`
    double             totalSeconds = 0.0;
  };

  // Start a load, discarding the previous one.
  void begin(std::string title)
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_title = std::move(title);
    m_finished.clear();
    m_current = {};
    m_detail.clear();
    m_done   = 0;
    m_start  = Clock::now();
    m_active = true;
  }

  // Close the current stage (if any) and open the next one.
  void stage(std::string name, uint32_t total = 0)
  {
    if(!m_active)
      return;
    std::lock_guard<std::mutex> lock(m_mutex);
    closeStage();
    m_current = {std::move(name), total};
    m_detail.clear();
    m_done       = 0;
    m_stageStart = Clock::now();
  }

  // `count` items of the current stage finished; `detail` names the last one.
  void advance(uint32_t count = 1, const std::string& detail = {})
  {
    if(!m_active)
      return;
    m_done.fetch_add(count, std::memory_order_relaxed);
    if(!detail.empty())
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_detail = detail;
    }
  }

  // Close the last stage, log where the time went, and stop reporting.
  void end()
  {
    if(!m_active)
      return;
    std::lock_guard<std::mutex> lock(m_mutex);
    closeStage();
    m_active = false;

    std::string stages;
    for(size_t i = 0; i < m_finished.size(); i++)
    {
      const Stage&      stage = m_finished[i];
      const std::string name = stage.total > 0 ? fmt::format("{} ({})", stage.name, stage.total) : stage.name;
      stages += fmt::format("\n  {:<60} {:7.2f} s", name, stage.seconds);
    }
    LOGI("%s: %.2f s%s\n", m_title.c_str(), seconds(m_start, Clock::now()), stages.c_str());
  }

  bool active() const { return m_active; }

  Snapshot snapshot() const
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    const Clock::time_point     now = Clock::now();
    Snapshot                    snapshot;
    snapshot.active          = m_active;
    snapshot.title           = m_title;
    snapshot.finished        = m_finished;
    snapshot.current         = m_current;
    snapshot.current.seconds = m_current.name.empty() ? 0.0 : seconds(m_stageStart, now);
    snapshot.done            = m_done.load(std::memory_order_relaxed);
    snapshot.detail          = m_detail;
    snapshot.totalSeconds    = seconds(m_start, now);
    return snapshot;
  }

private:
  using Clock = std::chrono::steady_clock;

  static double seconds(Clock::time_point from, Clock::time_point to)
  {
    return std::chrono::duration<double>(to - from).count();
  }

  // Requires m_mutex.
  void closeStage()
  {
    if(m_current.name.empty())
      return;
    m_current.seconds = seconds(m_stageStart, Clock::now());
    m_finished.push_back(std::move(m_current));
    m_current = {};
  }

  mutable std::mutex    m_mutex{};
  std::atomic<bool>     m_active{false};
  std::atomic<uint32_t> m_done{0};
  std::string           m_title{};
  std::vector<Stage>    m_finished{};
  Stage                 m_current{};
  std::string           m_detail{};
  Clock::time_point     m_start{};
  Clock::time_point     m_stageStart{};
};
