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

#include "ui_dock_layout.hpp"

#include <imgui_internal.h>

// Must match the flags nvapp::Application::setupImguiDock() passes to DockSpaceOverViewport(), so a
// rebuilt node behaves like the one a fresh run creates (transparent central node, nothing dockable
// on top of the viewport).
static constexpr ImGuiDockNodeFlags kDockFlags = ImGuiDockNodeFlags_PassthruCentralNode | ImGuiDockNodeFlags_NoDockingInCentralNode;

void ui::buildDefaultDockLayout(ImGuiID viewportID)
{
  // Left side panel container
  ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.25F, nullptr, &viewportID);
  ImGui::DockBuilderDockWindow("Camera", settingID);
  ImGui::DockBuilderDockWindow("Settings", settingID);

  // Under Setting
  ImGuiID tonemapID = ImGui::DockBuilderSplitNode(settingID, ImGuiDir_Down, 0.35F, nullptr, &settingID);
  ImGui::DockBuilderDockWindow("Tonemapper", tonemapID);
  ImGui::DockBuilderDockWindow("Environment", tonemapID);

  // Right side: Scene Browser, Inspector (bottom)
  ImGuiID sceneBrowserID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.25F, nullptr, &viewportID);
  ImGui::DockBuilderDockWindow("Scene Browser", sceneBrowserID);
  ImGuiID inspectorID = ImGui::DockBuilderSplitNode(sceneBrowserID, ImGuiDir_Down, 0.35F, nullptr, &sceneBrowserID);
  ImGui::DockBuilderDockWindow("Inspector", inspectorID);

  // bottom panel container
  ImGuiID logID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.35F, nullptr, &viewportID);
  ImGui::DockBuilderDockWindow("Log", logID);
  ImGuiID monitorID = ImGui::DockBuilderSplitNode(logID, ImGuiDir_Right, 0.35F, nullptr, &logID);
  ImGui::DockBuilderDockWindow("NVML Monitor", monitorID);
  ImGuiID profilerID = ImGui::DockBuilderSplitNode(logID, ImGuiDir_Right, 0.33F, nullptr, &logID);
  ImGui::DockBuilderDockWindow("Profiler", profilerID);
  ImGuiID memStatsID = ImGui::DockBuilderSplitNode(logID, ImGuiDir_Right, 0.33F, nullptr, &logID);
  ImGui::DockBuilderDockWindow("Memory Statistics", memStatsID);
  ImGui::DockBuilderDockWindow("Statistics", memStatsID);
}

void ui::resetDockLayout()
{
  // DockSpaceOverViewport() hosts its dockspace in a window named after the viewport and derives the
  // node id from that window, so recompute the same id rather than storing one nvapp owns.
  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  char                 hostWindowName[32];
  ImFormatString(hostWindowName, IM_ARRAYSIZE(hostWindowName), "WindowOverViewport_%08X", viewport->ID);
  ImGuiWindow* hostWindow = ImGui::FindWindowByName(hostWindowName);
  if(hostWindow == nullptr)
    return;  // no dockspace yet (first frame): a fresh layout is about to be built anyway
  const ImGuiID dockID = hostWindow->GetID("DockSpace");

  // Drop the whole tree (children included; docked windows are released, not closed) and start over
  // from an empty dockspace node the size of the viewport.
  ImGui::DockBuilderRemoveNode(dockID);
  ImGui::DockBuilderAddNode(dockID, kDockFlags | ImGuiDockNodeFlags_DockSpace);
  ImGui::DockBuilderSetNodeSize(dockID, viewport->WorkSize);

  // Mark the fresh node central and tab-bar-less, the way nvapp::Application::setupImguiDock() ends
  // up doing for a first run, so the viewport keeps its bare, undockable look.
  //
  // It has to be set on the node itself rather than through DockBuilderGetCentralNode(): that
  // returns the root's cached central-node pointer, which ImGui only fills in while processing the
  // dockspace, so it is still null this early and the flag would be dropped. Both flags are in
  // ImGuiDockNodeFlags_LocalFlagsTransferMask_, so they follow the viewport into whichever child
  // node inherits it as the panels are split off below.
  ImGuiDockNode* rootNode = ImGui::DockBuilderGetNode(dockID);
  if(rootNode == nullptr)
    return;
  rootNode->SetLocalFlags(rootNode->LocalFlags | ImGuiDockNodeFlags_CentralNode | ImGuiDockNodeFlags_NoTabBar);

  // Same order as nvapp::Application::setupImguiDock(): the viewport takes the central node, then
  // the panels are split off around it.
  ImGui::DockBuilderDockWindow("Viewport", dockID);
  buildDefaultDockLayout(dockID);
  ImGui::DockBuilderFinish(dockID);
}
