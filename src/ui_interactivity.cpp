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

//
// KHR_interactivity Graphs panel (Phase F), plus a viewport toolbar indicator: InteractivityControl
// (play/pause/reset state for the default graph's runtime instance, no ImGui in the struct itself),
// the standalone "Interactivity" window (transport controls, live graph/node/variable stats, a
// custom event sender, a debug/log history surface), and a small toolbar Play/Pause button shown
// only when the loaded scene has a graph. State owner: Resources::interactivityControl.
//

#include "ui_interactivity.hpp"

#include <string>

#include <imgui.h>
#include <nvgui/fonts.hpp>

#include "gltf_interactivity_eval.hpp"
#include "gltf_interactivity_graph.hpp"
#include "gltf_interactivity_instance.hpp"
#include "gltf_scene.hpp"

namespace ui::interactivity {

bool hasGraph(nvvkgltf::Scene* scene)
{
  return scene && !scene->getInteractivityGraphs().empty();
}

void renderToolbarIndicator(nvvkgltf::Scene* scene, InteractivityControl& ctrl)
{
  if(!hasGraph(scene))
    return;

  const ImVec4 activeColor = ImGui::GetStyle().Colors[ImGuiCol_ButtonActive];
  const ImVec4 idleColor   = ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
  ImGui::PushStyleColor(ImGuiCol_Button, ctrl.play ? activeColor : idleColor);
  const float buttonSize = ImGui::GetFrameHeight();
  if(ImGui::Button(ctrl.play ? ICON_MS_PAUSE : ICON_MS_PLAY_ARROW, ImVec2(buttonSize, buttonSize)))
    ctrl.togglePlay();
  ImGui::PopStyleColor();
  if(ImGui::IsItemHovered())
    ImGui::SetTooltip(ctrl.play ? "KHR_interactivity: Playing (click to pause)" : "KHR_interactivity: Paused (click to play)");
}

void renderWindow(nvvkgltf::Scene* scene, InteractivityControl& ctrl, bool* open)
{
  if(!open || !*open || !hasGraph(scene))
    return;

  if(!ImGui::Begin("Interactivity", open))
  {
    ImGui::End();
    return;
  }

  const std::vector<nvvkgltf::InteractivityGraph>& graphs = scene->getInteractivityGraphs();

  const int                             defaultGraphIndex = scene->getDefaultInteractivityGraph();
  nvvkgltf::InteractivityGraphInstance* instance =
      defaultGraphIndex >= 0 ? scene->getInteractivityInstance(defaultGraphIndex) : nullptr;
  if(defaultGraphIndex < 0 || !instance)
  {
    ImGui::TextDisabled("KHR_interactivity: no valid default graph.");
    ImGui::End();
    return;
  }
  const nvvkgltf::InteractivityGraph& graph = graphs[defaultGraphIndex];

  ImGui::PushID("Interactivity");

  // Transport controls
  if(ImGui::Button(ctrl.play ? "Pause" : "Play"))
    ctrl.togglePlay();
  ImGui::SameLine();
  if(ImGui::Button("Reset"))
    ctrl.requestReset();
  ImGui::SameLine();
  ImGui::TextDisabled("%s", graph.name().empty() ? "(unnamed graph)" : graph.name().c_str());
  if(graphs.size() > 1)
  {
    ImGui::SameLine();
    ImGui::TextDisabled("[graph %d/%d]", defaultGraphIndex + 1, static_cast<int>(graphs.size()));
  }

  // Live stats
  ImGui::Separator();
  ImGui::Text("Nodes: %d   Variables: %d   Events: %d", static_cast<int>(graph.nodes().size()),
              static_cast<int>(graph.variables().size()), static_cast<int>(graph.events().size()));
  ImGui::Text("Started: %s   Ticked: %s   Time: %.2fs", instance->started() ? "yes" : "no",
              instance->ticked() ? "yes" : "no", instance->timeSinceStart());

  // Variables (spec: anonymous, addressed only by index - no name to show)
  if(!graph.variables().empty() && ImGui::TreeNode("Variables"))
  {
    for(int i = 0; i < static_cast<int>(graph.variables().size()); ++i)
      ImGui::BulletText("[%d] %s", i, nvvkgltf::stringifyInteractivityValue(instance->variable(i)).c_str());
    ImGui::TreePop();
  }

  // Custom event sender (spec event/send + event/receive) - lets an author trigger a graph's
  // event/receive handlers without needing a hover/select/pointer-set to do it indirectly.
  if(!graph.events().empty() && ImGui::TreeNode("Send Event"))
  {
    if(ctrl.selectedEventIndex < 0 || ctrl.selectedEventIndex >= static_cast<int>(graph.events().size()))
      ctrl.selectedEventIndex = 0;

    auto eventLabel = [&](int i) {
      const nvvkgltf::InteractivityEventInfo& e = graph.events()[i];
      return e.id.empty() ? ("event " + std::to_string(i)) : e.id;
    };

    if(ImGui::BeginCombo("Event", eventLabel(ctrl.selectedEventIndex).c_str()))
    {
      for(int i = 0; i < static_cast<int>(graph.events().size()); ++i)
      {
        const bool selected = i == ctrl.selectedEventIndex;
        if(ImGui::Selectable(eventLabel(i).c_str(), selected))
          ctrl.selectedEventIndex = i;
        if(selected)
          ImGui::SetItemDefaultFocus();
      }
      ImGui::EndCombo();
    }
    ImGui::SameLine();
    if(ImGui::Button("Send"))
      instance->sendEvent(ctrl.selectedEventIndex);
    ImGui::TreePop();
  }

  // Error/log surface: debug/log's captured history, newest last.
  if(ImGui::TreeNode("Log"))
  {
    ImGui::SameLine();
    if(ImGui::SmallButton("Clear"))
      instance->clearLogEntries();

    ImGui::BeginChild("InteractivityLogScroll", ImVec2(0.0f, 150.0f), true);
    const auto& entries = instance->logEntries();
    if(entries.empty())
      ImGui::TextDisabled("(no log entries)");
    else
      for(const auto& entry : entries)
        ImGui::TextWrapped("%s", entry.message.c_str());
    if(ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f)
      ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();
    ImGui::TreePop();
  }

  ImGui::PopID();
  ImGui::End();
}

}  // namespace ui::interactivity
