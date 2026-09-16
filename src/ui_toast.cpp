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

//
// Minimal transient notification overlay for surfacing UI actions (mostly errors) that would
// otherwise only reach the log. push() a message; render() draws the unexpired ones stacked in the
// lower-right of the main viewport and drops them after a few seconds, timed off ImGui::GetTime()
// (no separate ticker needed).
//

#include <algorithm>
#include <cstdio>

#include <imgui.h>
#include <nvgui/IconsMaterialSymbols.h>

#include "ui_toast.hpp"

// How long a toast stays on screen. Errors linger a little longer than info so they are not missed.
static constexpr double kInfoSeconds  = 3.0;
static constexpr double kErrorSeconds = 6.0;

void UiToasts::push(std::string message, Level level)
{
  const double life = (level == Level::Error) ? kErrorSeconds : kInfoSeconds;
  m_toasts.push_back({m_nextId++, std::move(message), level, ImGui::GetTime() + life});
}

void UiToasts::render()
{
  if(m_toasts.empty())
    return;

  const double now = ImGui::GetTime();
  std::erase_if(m_toasts, [now](const Toast& t) { return now >= t.expireTime; });

  const ImGuiViewport* vp  = ImGui::GetMainViewport();
  const float          pad = 10.0f;
  float                y   = vp->WorkPos.y + vp->WorkSize.y - pad;  // stack upward from the bottom-right

  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize
                                 | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing
                                 | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoMove;

  // Newest toast nearest the bottom edge.
  for(auto it = m_toasts.rbegin(); it != m_toasts.rend(); ++it)
  {
    ImGui::SetNextWindowBgAlpha(0.88f);
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x - pad, y), ImGuiCond_Always, ImVec2(1.0f, 1.0f));

    char windowId[32];
    snprintf(windowId, sizeof(windowId), "##toast_%llu", static_cast<unsigned long long>(it->id));
    ImGui::Begin(windowId, nullptr, flags);

    const bool   isError = it->level == Level::Error;
    const ImVec4 color   = isError ? ImVec4(1.0f, 0.45f, 0.45f, 1.0f) : ImVec4(0.55f, 0.85f, 1.0f, 1.0f);
    ImGui::TextColored(color, "%s", isError ? ICON_MS_ERROR : ICON_MS_INFO);
    ImGui::SameLine();
    ImGui::TextUnformatted(it->message.c_str());

    y -= ImGui::GetWindowSize().y + 4.0f;
    ImGui::End();
  }
}
