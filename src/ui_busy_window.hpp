/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * This class implements a modal popup window that:
 * - Centers itself on the screen
 * - Blocks all user input
 * - Displays a progress indicator
 * - Shows a custom status message
 * 
 * It is designed to provide visual feedback during long-running operations
 * such as asset loading or background processing.
 *
 * A scene load also reports into progress() (see load_progress.hpp): the window then lists the
 * finished stages with their durations and the current stage with its item count. A load stays on
 * screen after stop(), through the GPU upload and acceleration-structure builds, until the loader
 * ends the progress.
 *
 * Usage example:
 * ```
 * BusyWindow busy;
 * 
 * // In your main thread or UI thread:
 * void renderUI() {
 *     busy.show();  // Call this in your ImGui render loop
 * }
 *
 * // In your worker thread:
 * void loadAssets() {
 *     busy.start("Loading assets...");
 *     // ... do your long operation ...
 *     busy.setReason("Finalizing...");   // update message mid-flight
 *     // ... more work ...
 *     busy.stop();
 * }
 *
 * // Optional: Check if operation is done
 * if (busy.isDone()) {
 *     busy.consumeDone();
 *     // Handle completion
 * }
 * ```
 */

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>

#include <fmt/format.h>
#include <imgui.h>
#include <nvgui/fonts.hpp>

#include "load_progress.hpp"

class BusyWindow
{
public:
  void start(const std::string& reason)
  {
    {
      std::lock_guard<std::mutex> lock(m_reasonMutex);
      m_reason = reason;
    }
    m_busy = true;
  }

  // Update the progress message while the busy window is running. Safe to call from
  // the worker thread; the UI thread picks up the new text on its next show() call.
  void setReason(const std::string& reason)
  {
    std::lock_guard<std::mutex> lock(m_reasonMutex);
    m_reason = reason;
  }

  void stop()
  {
    {
      std::lock_guard<std::mutex> lock(m_reasonMutex);
      m_reason.clear();  // a load's GPU tail keeps the window up; don't let a stale reason linger
    }
    m_busy = false;
    m_done = true;
  }
  void consumeDone() { m_done = false; }
  bool isBusy() const { return m_busy; }
  bool isDone() const { return m_done; }
  // The window is drawn while a worker is busy or a load is still reporting (its GPU tail).
  bool isVisible() const { return m_busy || m_progress.active(); }

  LoadProgress&       progress() { return m_progress; }
  const LoadProgress& progress() const { return m_progress; }

  // Display a modal window when loading assets or other long operation on separated thread
  inline void show()
  {
    const LoadProgress::Snapshot load = m_progress.snapshot();

    // Hold the lock across the whole draw so the worker thread cannot mutate m_reason
    std::lock_guard<std::mutex> lock(m_reasonMutex);
    if(m_reason.empty() && !load.active)
      return;

    ImGui::OpenPopup("Busy Info");

    // Position in the center of the main window when appearing. A load is wider: it lists its stages.
    const float  em = ImGui::GetFontSize();
    const ImVec2 win_size(load.active ? 30.0f * em : 300.0f, 0.0f);
    ImGui::SetNextWindowSize(win_size);
    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5F, 0.5F));

    // Window without any decoration
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 15.0);
    if(ImGui::BeginPopupModal("Busy Info", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration))
    {
      // A load's stages say more than the reason line, so they replace it.
      if(load.active)
        showLoad(load);
      else
      {
        ImGui::TextDisabled("Please wait ...");
        ImGui::NewLine();
        ImGui::ProgressBar(-.20f * float(ImGui::GetTime()), ImVec2(-1.0f, 0.0f), m_reason.c_str());
      }
      ImGui::EndPopup();
    }
    ImGui::PopStyleVar();
  }

private:
  // Title and total time, then one row per stage: finished ones with their duration, the current one
  // with its count and a bar (a marquee when it is not counted), and the last item it finished.
  static void showLoad(const LoadProgress::Snapshot& load)
  {
    const float timeColumn = ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("000.00 s").x;
    auto        timeCell   = [timeColumn](double seconds) {
      ImGui::SameLine(timeColumn);
      ImGui::TextDisabled("%6.2f s", seconds);
    };

    ImGui::TextUnformatted(load.title.c_str());
    timeCell(load.totalSeconds);
    ImGui::Separator();

    for(size_t i = 0; i < load.finished.size(); i++)
    {
      const LoadProgress::Stage& stage = load.finished[i];
      ImGui::TextDisabled(ICON_MS_CHECK);
      ImGui::SameLine();
      if(stage.total > 0)
        ImGui::Text("%s (%u)", stage.name.c_str(), stage.total);
      else
        ImGui::TextUnformatted(stage.name.c_str());
      timeCell(stage.seconds);
    }

    if(load.current.name.empty())
      return;

    ImGui::TextUnformatted(ICON_MS_PENDING);
    ImGui::SameLine();
    ImGui::TextUnformatted(load.current.name.c_str());
    timeCell(load.current.seconds);

    if(load.current.total > 0)
    {
      const uint32_t    done    = std::min(load.done, load.current.total);
      const std::string overlay = fmt::format("{} / {}", done, load.current.total);
      ImGui::ProgressBar(float(done) / float(load.current.total), ImVec2(-1.0f, 0.0f), overlay.c_str());
    }
    else
      ImGui::ProgressBar(-.20f * float(ImGui::GetTime()), ImVec2(-1.0f, 0.0f), "");

    // Reserve the detail line even when empty so the window does not jump between items.
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", load.detail.empty() ? " " : load.detail.c_str());
    ImGui::PopTextWrapPos();
  }

  std::atomic<bool> m_busy{false};
  std::atomic<bool> m_done{false};
  std::mutex        m_reasonMutex;
  std::string       m_reason;
  LoadProgress      m_progress;
};
