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

#include "ui_commands.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace ui {
namespace {

// The routing a binding is polled with.
// - Global: the lowest priority, so a window or a widget that claims the same chord keeps it.
// - Viewport: the viewport window calls dispatch() itself. Hovered, it takes the key over the
//   focused window, so a key works under the cursor without clicking the viewport first;
//   otherwise only while the viewport is in the focus route. Neither: the route scores nothing.
ImGuiInputFlags routeFlags(KeyScope scope)
{
  if(scope == KeyScope::eViewport)
    return ImGui::IsWindowHovered(ImGuiHoveredFlags_RootWindow) ? ImGuiInputFlags_RouteGlobal | ImGuiInputFlags_RouteOverFocused :
                                                                  ImGuiInputFlags_RouteFocused;
  return ImGuiInputFlags_RouteGlobal;
}

}  // namespace

void CommandRegistry::add(Command command)
{
  assert(std::none_of(m_commands.begin(), m_commands.end(),
                      [&](const Command& c) { return std::strcmp(c.id, command.id) == 0; })
         && "Command id registered twice");
  m_commands.push_back(std::move(command));
}

const Command& CommandRegistry::get(std::string_view id) const
{
  for(size_t i = 0; i < m_commands.size(); ++i)
  {
    if(id == m_commands[i].id)
      return m_commands[i];
  }
  assert(false && "Unknown command id");
  return m_commands.front();
}

bool CommandRegistry::menuItem(std::string_view id, const char* label) const
{
  const Command&    cmd  = get(id);
  const std::string text = std::string(cmd.icon != nullptr ? cmd.icon : "") + (cmd.icon != nullptr ? " " : "")
                           + (label != nullptr ? label : cmd.label);
  const std::string shortcut = shortcutText(cmd.binding);
  const bool        enabled  = !cmd.enabled || cmd.enabled();
  const bool        checked  = cmd.checked && cmd.checked();
  const bool clicked = ImGui::MenuItem(text.c_str(), shortcut.empty() ? nullptr : shortcut.c_str(), checked, enabled);
  if(cmd.tooltip != nullptr)
    ImGui::SetItemTooltip("%s", cmd.tooltip);
  if(clicked && cmd.run)
    cmd.run();
  return clicked;
}

void CommandRegistry::dispatch(KeyScope scope) const
{
  const ImGuiInputFlags route = routeFlags(scope);
  for(const Command& cmd : m_commands)
  {
    if(cmd.binding.scope != scope || cmd.binding.chord == ImGuiKey_None || cmd.binding.text != nullptr || !cmd.run)
      continue;
    // Shortcut() also registers the route for the next frame, so it is called every frame, before
    // the enabled test.
    if(ImGui::Shortcut(cmd.binding.chord, route) && (!cmd.enabled || cmd.enabled()))
      cmd.run();
  }
}

const char* keyName(ImGuiKey key)
{
  switch(key)
  {
    case ImGuiKey_Delete:
      return "Del";
    case ImGuiKey_Escape:
      return "Esc";
    case ImGuiKey_LeftArrow:
      return "Left";
    case ImGuiKey_RightArrow:
      return "Right";
    case ImGuiKey_UpArrow:
      return "Up";
    case ImGuiKey_DownArrow:
      return "Down";
    default:
      return ImGui::GetKeyName(key);
  }
}

std::string chordText(ImGuiKeyChord chord)
{
  std::string text;
  if(chord & ImGuiMod_Ctrl)
    text += "Ctrl+";
  if(chord & ImGuiMod_Alt)
    text += "Alt+";
  if(chord & ImGuiMod_Shift)
    text += "Shift+";
  const ImGuiKey key = ImGuiKey(chord & ~ImGuiMod_Mask_);
  if(key != ImGuiKey_None)
    text += keyName(key);
  return text;
}

std::string shortcutText(const KeyBinding& binding)
{
  if(binding.text != nullptr)
    return chordText(binding.chord) + binding.text;
  return binding.chord != ImGuiKey_None ? chordText(binding.chord) : std::string();
}

}  // namespace ui
