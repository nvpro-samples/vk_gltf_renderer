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

// The application's commands, each declared once. A command is what an action does and how it is
// presented (label, icon, enabled, checked); its key binding is held apart from it, so bindings can
// later become user-settable without touching the commands. The menus (menuItem()), the toolbar,
// the key handling (dispatch()) and Help > Keyboard & Mouse are all views of the registry.
//
// Keys go through Dear ImGui's shortcut routing (ImGui::Shortcut()), so a widget that is using the
// keyboard keeps it: while a text field is being edited, letters, Delete, Home and Esc go to the
// field, and its own Ctrl+Z / Ctrl+Y win over the application's.

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <imgui.h>

namespace ui {

// Where a binding is live, which decides when and with what routing it is polled.
enum class KeyScope : uint8_t
{
  eGlobal,        // Anywhere in the application; polled by the menu bar, so not while it is hidden
  eAlways,        // Anywhere, including while the menu bar is hidden (presentation mode)
  eViewport,      // While the cursor is over the viewport or the viewport has focus
  ePresentation,  // Only in presentation mode; polled by GltfRenderer::updatePresentationMode() then
};

struct KeyBinding
{
  ImGuiKeyChord chord{ImGuiKey_None};
  KeyScope      scope{KeyScope::eGlobal};
  const char*   text{nullptr};  // Shown after the chord's modifiers instead of a key: a key matched as a
                                // typed character ("?"), or a gesture ("Drop file"). Matched by the
                                // code that handles it, never polled by dispatch()

  bool isBound() const { return chord != ImGuiKey_None || text != nullptr; }
};

struct Command
{
  const char* id{nullptr};       // "file.save"
  const char* label{nullptr};    // As the menu shows it
  const char* icon{nullptr};     // ICON_MS_*, drawn before the label
  const char* group{nullptr};    // The menu it belongs to; Help groups by it
  const char* tooltip{nullptr};  // On the menu item, and in Help
  KeyBinding  binding{};

  std::function<void()> run;      // Empty: the key is handled by other code (nvapp); listed only
  std::function<bool()> enabled;  // Empty: always enabled
  std::function<bool()> checked;  // Empty: not a toggle
};

class CommandRegistry
{
public:
  void add(Command command);

  const Command&           get(std::string_view id) const;  // The id must exist
  std::span<const Command> all() const { return m_commands; }

  // ImGui::MenuItem() for a command: icon and label (label replaced by `label` when given), its
  // shortcut text, enabled and checked state, and its tooltip. Runs it when clicked.
  bool menuItem(std::string_view id, const char* label = nullptr) const;

  // Runs every command of `scope` whose binding was pressed this frame. Call it every frame, from the
  // window the scope belongs to: eViewport from inside the viewport window.
  void dispatch(KeyScope scope) const;

private:
  std::vector<Command> m_commands;
};

// A binding as text, "Ctrl+Shift+S" or "Shift+Drop file". Modifiers in the order Ctrl, Alt, Shift.
std::string shortcutText(const KeyBinding& binding);
std::string chordText(ImGuiKeyChord chord);

// The name of a key as the shortcut text spells it: "Del" and "Esc" rather than ImGui's names.
const char* keyName(ImGuiKey key);

}  // namespace ui
