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

// The application's default docking arrangement, declared once.
//
// It is needed in two places -- at start-up, where nvapp::Application builds it only when the
// ImGui.ini has no layout yet (ApplicationCreateInfo::dockSetup, see main.cpp), and at runtime for
// Windows > Reset UI Layout. Keeping both on the same function is what stops the menu item from
// restoring a layout that no longer matches the one a fresh run gets.

#include <imgui.h>

namespace ui {

// Split `dockID` (the central dockspace node, already holding "Viewport") into the default panel
// arrangement. Matches nvapp::ApplicationCreateInfo::dockSetup's signature so it can be used as-is.
void buildDefaultDockLayout(ImGuiID dockID);

// Discard the current docking arrangement and rebuild buildDefaultDockLayout() in its place.
// Windows that the user had floated or moved are re-docked where a fresh run would put them.
// Call between frames' window submissions (top of the UI pass), not from inside a docked window.
void resetDockLayout();

}  // namespace ui
