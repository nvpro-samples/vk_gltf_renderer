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

#include <string>
#include <atomic>

#include "imgui.h"

class BusyWindow
{
public:
  void start(const std::string& reason)
  {
    m_busy   = true;
    m_reason = reason;
  }
  void stop()
  {
    m_busy = false;
    m_done = true;
  }
  void consumeDone() { m_done = false; }
  bool isBusy() const { return m_busy; }
  bool isDone() const { return m_done; }

  // Display a modal window when loading assets or other long operation on separated thread
  inline void show()
  {
    if(m_reason.empty())
      return;

    // Display a modal window when loading assets or other long operation on separated thread
    ImGui::OpenPopup("Busy Info");

    // Position in the center of the main window when appearing
    const ImVec2 win_size(300, 100);
    ImGui::SetNextWindowSize(win_size);
    const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));

    // Window without any decoration
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 15.0);
    if(ImGui::BeginPopupModal("Busy Info", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration))
    {
      // Center text in window
      ImGui::TextDisabled("Please wait ...");
      ImGui::NewLine();
      ImGui::ProgressBar(-.20f * float(ImGui::GetTime()), ImVec2(-1.0f, 0.0f), m_reason.c_str());
      ImGui::EndPopup();
    }
    ImGui::PopStyleVar();
  }

private:
  std::atomic<bool> m_busy{false};
  std::atomic<bool> m_done{false};
  std::string       m_reason;
};
