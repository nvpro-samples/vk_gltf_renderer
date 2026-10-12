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

// The Help menu and the windows it opens.
//
//  - Keyboard & Mouse: a reference of every binding. The Camera tab lays the mouse out as a
//    button-by-modifier table for the camera's current navigation mode and lights up the cell being
//    used, so the table doubles as a way to learn the controls by trying them. The Shortcuts tab is
//    a searchable list of the keyboard shortcuts, grouped as the menus are.
//  - About: version, GPU and driver, and the project links.
//
// The Shortcuts tab is generated from the command registry (ui_commands.hpp), the same one the menus,
// the toolbar and the key handling use. The Camera tab's mouse table and keys are generated from the
// camera binding table (camera_controls.hpp), the same one that drives the camera.

#include <volk.h>

namespace nvutils {
class CameraManipulator;
}
namespace ui {
class CommandRegistry;
}

namespace ui::help {

struct HelpState
{
  bool showShortcuts{false};
  bool showAbout{false};
};

// The "Help" menu, to be called inside the main menu bar. Also handles its global shortcut ('?').
void renderMenu(HelpState& state, const ui::CommandRegistry& commands);

// The Help windows; call once per frame outside of any other window.
void renderWindows(HelpState& state, const nvutils::CameraManipulator& camera, const ui::CommandRegistry& commands, VkPhysicalDevice physicalDevice);

}  // namespace ui::help
