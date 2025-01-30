/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
A state machine for detecting mouse interactions: single clicks, double clicks, and drag operations.
To be used if there need to differentiate between these interactions. For example, a single click might
select an element, a double click might open it, but double click must not select the element.

This helper class tracks mouse events to differentiate between various click patterns. When a user interacts with an element, the class:

1. Detects when a click is released
2. Waits for a brief period to check if a second click occurs
3. If no second click occurs, registers it as a single click
4. If a second click occurs within the timeout period, registers it as a double click
5. If mouse movement is detected while holding the click, registers it as a drag operation

Note: Every double click is preceded by a single click, but this class helps determine the user's final intended action.

 State machine:
    IDLE -> DRAG (if dragging detected)
    IDLE -> SINGLE_PENDING (on release)
    SINGLE_PENDING -> DOUBLE (if double clicked)
    SINGLE_PENDING -> IDLE (if double click time expires)
    DRAG/DOUBLE -> IDLE (when mouse released)
*/
struct ClickStateMachine
{
  bool isMouseClicked(ImGuiMouseButton button) const { return (singleClicked[button] || doubleClicked[button]); }
  bool isMouseSingleClicked(ImGuiMouseButton button) const { return singleClicked[button]; }
  bool isMouseDoubleClicked(ImGuiMouseButton button) const { return doubleClicked[button]; }
  bool isMouseDragging(ImGuiMouseButton button) const { return state[button] == State::DRAG; }

  void update()
  {
    ImGuiContext& g  = *GImGui;
    ImGuiIO&      io = g.IO;

    for(int i = 0; i < IM_ARRAYSIZE(io.MouseDown); i++)
    {
      // Reset outputs at start of frame
      singleClicked[i] = false;
      doubleClicked[i] = false;

      switch(state[i])
      {
        case State::IDLE:
          if(ImGui::IsMouseDragging(i))
          {
            state[i] = State::DRAG;
          }
          else if(ImGui::IsMouseReleased(i))
          {
            state[i] = State::SINGLE_PENDING;
          }
          break;

        case State::SINGLE_PENDING:
          if(ImGui::IsMouseDoubleClicked(i))
          {
            state[i]         = State::DOUBLE;
            doubleClicked[i] = true;
          }
          else if((float)(g.Time - io.MouseClickedTime[i]) > io.MouseDoubleClickTime)
          {
            // Mouse released without double click
            state[i]         = State::IDLE;
            singleClicked[i] = true;
          }
          break;

        case State::DOUBLE:
        case State::DRAG:
          if(!ImGui::IsMouseDown(i))
          {
            // Reset after drag or double click is complete
            state[i] = State::IDLE;
          }
          break;
      }
    }
  }

private:
  enum class State
  {
    IDLE,            // No click in progress
    SINGLE_PENDING,  // First click detected, waiting for potential double
    DOUBLE,          // Double click detected
    DRAG
  };

  std::array<State, 5> state         = {};
  std::array<bool, 5>  singleClicked = {};  // Output: Single click confirmed
  std::array<bool, 5>  doubleClicked = {};  // Output: Double click detected
};