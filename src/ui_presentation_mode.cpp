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

#include "ui_presentation_mode.hpp"

#include <algorithm>
#include <utility>

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <nvapp/application.hpp>
#include <nvutils/logger.hpp>

namespace {

// The monitor the window is on: the one it overlaps most, which is also the one Windows picks the
// window's DPI from -- so covering it does not trigger a DPI-change resize behind our back. Falls
// back to the primary monitor when the window is entirely off-screen.
GLFWmonitor* monitorOfWindow(const glm::ivec2& winPos, const glm::ivec2& winSize)
{
  int           count    = 0;
  GLFWmonitor** monitors = glfwGetMonitors(&count);
  GLFWmonitor*  best     = nullptr;
  int64_t       bestArea = 0;
  for(int i = 0; i < count; i++)
  {
    const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
    if(mode == nullptr)
      continue;
    glm::ivec2 monPos{};
    glfwGetMonitorPos(monitors[i], &monPos.x, &monPos.y);
    const glm::ivec2 lo   = glm::max(winPos, monPos);
    const glm::ivec2 hi   = glm::min(winPos + winSize, monPos + glm::ivec2(mode->width, mode->height));
    const int64_t    area = int64_t(std::max(0, hi.x - lo.x)) * int64_t(std::max(0, hi.y - lo.y));
    if(area > bestArea)
    {
      bestArea = area;
      best     = monitors[i];
    }
  }
  return best != nullptr ? best : glfwGetPrimaryMonitor();
}

}  // namespace

void PresentationMode::addWindow(bool* visible)
{
  addWindow([visible] { return *visible; }, [visible](bool v) { *visible = v; });
}

void PresentationMode::addWindow(std::function<bool()> isVisible, std::function<void(bool)> setVisible)
{
  m_windows.push_back({.isVisible = std::move(isVisible), .setVisible = std::move(setVisible)});
}

void PresentationMode::setActive(nvapp::Application* app, bool active)
{
  if(active == m_active || app == nullptr || app->isHeadless() || app->getWindowHandle() == nullptr)
    return;
  if(active)
    enter(app);
  else
    leave(app);
}

void PresentationMode::enter(nvapp::Application* app)
{
  GLFWwindow* window = app->getWindowHandle();

  // An exclusive full-screen window is not something this mode created, so it has no windowed
  // geometry to save and restore. The application never makes one; refuse rather than guess.
  if(glfwGetWindowMonitor(window) != nullptr)
  {
    LOGW("Presentation mode: the window is in exclusive full screen; ignored\n");
    return;
  }

  // Panels and menu bar. The ini is switched off first so ImGui's periodic auto-save cannot write
  // the hidden state; the pointer is restored as-is on leave (it is already null in scripted runs).
  ImGuiIO& io        = ImGui::GetIO();
  m_savedIniFilename = io.IniFilename;
  io.IniFilename     = nullptr;
  for(Window& w : m_windows)
  {
    w.savedVisible = w.isVisible();
    w.setVisible(false);
  }
  m_savedUseMenubar = app->getUseMenubar();
  app->setUseMenubar(false);

  // Window. Pick the monitor before un-maximizing: that is where the user sees the window now.
  if(glfwGetWindowAttrib(window, GLFW_ICONIFIED) == GLFW_TRUE)
    glfwRestoreWindow(window);
  glm::ivec2 pos{}, size{};
  glfwGetWindowPos(window, &pos.x, &pos.y);
  glfwGetWindowSize(window, &size.x, &size.y);
  GLFWmonitor* monitor = monitorOfWindow(pos, size);

  // Save the normal (restored) geometry, so leaving can re-maximize on top of it.
  m_savedMaximized = glfwGetWindowAttrib(window, GLFW_MAXIMIZED) == GLFW_TRUE;
  if(m_savedMaximized)
    glfwRestoreWindow(window);
  glfwGetWindowPos(window, &m_savedPos.x, &m_savedPos.y);
  glfwGetWindowSize(window, &m_savedSize.x, &m_savedSize.y);
  m_savedDecorated = glfwGetWindowAttrib(window, GLFW_DECORATED) == GLFW_TRUE;

  // Borderless, covering the whole monitor (not just its work area: the taskbar goes too). With a
  // null monitor, glfwSetWindowMonitor only moves and resizes a windowed window, in one call.
  const GLFWvidmode* mode = monitor != nullptr ? glfwGetVideoMode(monitor) : nullptr;
  if(mode != nullptr)
  {
    glm::ivec2 monPos{};
    glfwGetMonitorPos(monitor, &monPos.x, &monPos.y);
    glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_FALSE);
    glfwSetWindowMonitor(window, nullptr, monPos.x, monPos.y, mode->width, mode->height, GLFW_DONT_CARE);
  }
  else
  {
    LOGW("Presentation mode: no monitor found; panels hidden but the window is left as is\n");
  }

  m_active = true;
  LOGI("Presentation mode on (F11 or Esc to leave)\n");
  // The swapchain follows on its own: Application::run() polls the framebuffer size every frame
  // and rebuilds on change, and the "Viewport" size change resizes the G-buffers.
}

void PresentationMode::leave(nvapp::Application* app)
{
  GLFWwindow* window = app->getWindowHandle();

  // Window first, so the panels come back into a window of the right size.
  if(glfwGetWindowAttrib(window, GLFW_ICONIFIED) == GLFW_TRUE)
    glfwRestoreWindow(window);
  glfwSetWindowAttrib(window, GLFW_DECORATED, m_savedDecorated ? GLFW_TRUE : GLFW_FALSE);
  glfwSetWindowMonitor(window, nullptr, m_savedPos.x, m_savedPos.y, m_savedSize.x, m_savedSize.y, GLFW_DONT_CARE);
  if(m_savedMaximized)
    glfwMaximizeWindow(window);

  for(Window& w : m_windows)
    w.setVisible(w.savedVisible);
  app->setUseMenubar(m_savedUseMenubar);
  ImGui::GetIO().IniFilename = m_savedIniFilename;

  m_active = false;
  LOGI("Presentation mode off\n");
}
