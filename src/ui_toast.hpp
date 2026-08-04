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

#include <cstdint>
#include <string>
#include <vector>

// Minimal transient notification overlay for surfacing UI actions (mostly errors) that would otherwise
// only reach the log. push() a message; render() draws the unexpired ones stacked in the lower-right of
// the main viewport and drops them after a few seconds. Timing uses ImGui::GetTime() (the frame clock),
// so there is no separate ticker to drive.
class UiToasts
{
public:
  enum class Level
  {
    Info,
    Error
  };

  void push(std::string message, Level level = Level::Info);
  void render();  // Call once per frame from the top-level UI.

private:
  struct Toast
  {
    uint64_t    id;  // stable per-toast ImGui window key
    std::string message;
    Level       level;
    double      expireTime;  // ImGui::GetTime() value after which the toast is dropped
  };

  std::vector<Toast> m_toasts;
  uint64_t           m_nextId = 0;
};
