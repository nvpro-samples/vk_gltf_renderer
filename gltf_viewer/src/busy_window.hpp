/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include "imgui.h"


//--------------------------------------------------------------------------------------------------
// This function place a Popup window in the middle of the screen, blocking all inputs and is
// use to show the application is busy doing something
//
inline void showBusyWindow(const std::string& busyReasonText)
{
  if(busyReasonText.empty())
    return;

  // Display a modal window when loading assets or other long operation on separated thread
  ImGui::OpenPopup("Busy Info");


  // Position in the center of the main window when appearing
  const ImVec2 win_size(300, 75);
  ImGui::SetNextWindowSize(win_size);
  const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));

  // Window without any decoration
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 15.0);
  if(ImGui::BeginPopupModal("Busy Info", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration))
  {
    // Center text in window
    const ImVec2 available = ImGui::GetContentRegionAvail();
    const ImVec2 text_size = ImGui::CalcTextSize(busyReasonText.c_str(), nullptr, false, available.x);

    ImVec2       pos;
    pos.x = (available.x - text_size.x) * 0.5F;
    pos.y = (available.y - text_size.y) * 0.2F;

    ImGui::SetCursorPosX(pos.x);
    ImGui::Text("%s", busyReasonText.c_str());

    // Add animation \ | / -
    ImGui::SetCursorPosX(available.x * 0.5F);
    ImGui::Text("%c", "|/-\\"[static_cast<int>(ImGui::GetTime() / 0.25F) & 3]);
    ImGui::EndPopup();
  }
  ImGui::PopStyleVar();
}
