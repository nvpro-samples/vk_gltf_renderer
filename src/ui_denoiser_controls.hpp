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

#include <imgui/imgui.h>
#include <nvgui/fonts.hpp>

namespace nvsamples::denoiserui {

inline ImVec4 readyColor()
{
  return ImVec4(0.42f, 0.78f, 0.42f, 1.0f);
}

inline ImVec4 workingColor()
{
  return ImVec4(0.90f, 0.70f, 0.24f, 1.0f);
}

inline ImVec4 unavailableColor()
{
  return ImVec4(0.86f, 0.31f, 0.31f, 1.0f);
}

inline ImVec4 mutedColor()
{
  return ImVec4(0.60f, 0.60f, 0.60f, 1.0f);
}

inline void tooltip(const char* desc)
{
  if(desc && desc[0] && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("%s", desc);
}

inline void statusDot(const ImVec4& color)
{
  const ImVec2 p      = ImGui::GetCursorScreenPos();
  const float  lineH  = ImGui::GetTextLineHeight();
  const float  radius = lineH * 0.28f;
  ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.x + radius, p.y + lineH * 0.5f), radius,
                                              ImGui::ColorConvertFloat4ToU32(color));
  ImGui::Dummy(ImVec2(radius * 2.0f, lineH));
}

inline void statusText(const char* state, const ImVec4& color)
{
  statusDot(color);
  ImGui::SameLine(0.0f, 4.0f);
  ImGui::TextColored(color, "%s", state);
}

inline void statusRow(const char* label, const char* state, const ImVec4& color, const char* detail = nullptr)
{
  ImGui::AlignTextToFramePadding();
  if(label && label[0])
  {
    ImGui::TextDisabled("%s", label);
    ImGui::SameLine();
  }
  statusText(state, color);
  if(detail && detail[0])
  {
    ImGui::SameLine();
    ImGui::TextDisabled("%s", detail);
  }
}

inline bool activeButton(const char* label, bool active, const ImVec2& size = ImVec2(0, 0))
{
  if(active)
  {
    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
  }
  const bool pressed = ImGui::Button(label, size);
  if(active)
    ImGui::PopStyleColor(2);
  return pressed;
}

inline bool featureRow(const char*   id,
                       const char*   label,
                       bool*         enabled,
                       bool          enableAvailable,
                       const char*   state,
                       const ImVec4& stateColor,
                       bool*         settingsOpen,
                       const char*   enableTooltip,
                       const char*   settingsTooltip)
{
  ImGui::PushID(id);

  bool        changed    = false;
  const float buttonSize = ImGui::GetFrameHeight();
  float       statusW    = ImGui::CalcTextSize(state).x + ImGui::GetTextLineHeight() + 14.0f;
  if(statusW < 56.0f)
    statusW = 56.0f;

  if(!ImGui::BeginTable("##feature_row", 4, ImGuiTableFlags_SizingStretchProp))
  {
    ImGui::PopID();
    return false;
  }

  ImGui::TableSetupColumn("enable", ImGuiTableColumnFlags_WidthFixed, buttonSize);
  ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthStretch);
  ImGui::TableSetupColumn("status", ImGuiTableColumnFlags_WidthFixed, statusW);
  ImGui::TableSetupColumn("settings", ImGuiTableColumnFlags_WidthFixed, buttonSize);
  ImGui::TableNextRow();
  ImGui::TableSetColumnIndex(0);
  if(!enableAvailable)
    ImGui::BeginDisabled();
  if(ImGui::Checkbox("##enable", enabled))
    changed = true;
  tooltip(enableTooltip);
  if(!enableAvailable)
    ImGui::EndDisabled();

  ImGui::TableSetColumnIndex(1);
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(label);
  ImGui::TableSetColumnIndex(2);
  ImGui::AlignTextToFramePadding();
  statusText(state, stateColor);

  ImGui::TableSetColumnIndex(3);
  if(activeButton(ICON_MS_SETTINGS "##settings", settingsOpen && *settingsOpen, ImVec2(buttonSize, buttonSize)) && settingsOpen)
    *settingsOpen = !*settingsOpen;
  tooltip(settingsTooltip);

  ImGui::EndTable();
  ImGui::PopID();
  return changed;
}

}  // namespace nvsamples::denoiserui
