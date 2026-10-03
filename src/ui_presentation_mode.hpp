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

// Presentation mode (F11): the viewport alone, borderless, covering the monitor the window is on.
//
// Entering hides every registered panel and the main menu bar -- ImGui then hides the emptied dock
// nodes, so the central "Viewport" node fills the window -- and turns the window into an
// undecorated one sized to its monitor. That is "windowed full screen" rather than an exclusive
// glfwSetWindowMonitor() mode, so Alt-Tab to another application (e.g. slides on a second screen)
// is instant and does not minimize or re-mode the display.
//
// Leaving restores exactly what entering changed: each panel's visibility, the menu bar, and the
// window's position, size, decoration and maximized state. While active, ImGui.ini is not written,
// so the hidden layout can never become the one the next session starts with.

#include <functional>
#include <vector>

#include <glm/glm.hpp>

namespace nvapp {
class Application;
}

class PresentationMode
{
public:
  // A window whose visibility presentation mode hides and later restores. The raw-pointer form
  // covers the renderer's own Settings::show* flags; the callback form covers panels owned by other
  // application elements (profiler, logger, NVML monitor), wired in main.cpp.
  void addWindow(bool* visible);
  void addWindow(std::function<bool()> isVisible, std::function<void(bool)> setVisible);

  [[nodiscard]] bool isActive() const { return m_active; }

  // Enter or leave. A no-op when already in the requested state, and when there is no window to
  // work with (headless). Application thread only; call outside of any ImGui window submission.
  void setActive(nvapp::Application* app, bool active);

private:
  void enter(nvapp::Application* app);
  void leave(nvapp::Application* app);

  struct Window
  {
    std::function<bool()>     isVisible;
    std::function<void(bool)> setVisible;
    bool                      savedVisible{false};
  };
  std::vector<Window> m_windows;

  bool m_active{false};

  // State saved on enter, restored on leave.
  bool        m_savedUseMenubar{true};
  const char* m_savedIniFilename{nullptr};
  glm::ivec2  m_savedPos{0};
  glm::ivec2  m_savedSize{0};
  bool        m_savedDecorated{true};
  bool        m_savedMaximized{false};
};
