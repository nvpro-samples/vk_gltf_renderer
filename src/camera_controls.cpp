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

#include "camera_controls.hpp"

#include <array>

#include <imgui_internal.h>
#include <nvapp/elem_camera.hpp>
#include <nvgui/window.hpp>

namespace camera_controls {
namespace {

constexpr Projection kAny   = Projection::eAny;
constexpr Projection kPersp = Projection::ePerspective;
constexpr Projection kOrtho = Projection::eOrthographic;

// Movement keys are plain keys: Shift makes them faster, and Ctrl or Alt turns them into application
// shortcuts (Ctrl+S, Ctrl+D, ...), so the camera leaves them alone then.
constexpr float kFasterFactor = 5.0F;

// clang-format off
constexpr auto kBindings = std::to_array<Binding>({
    // Left drag
    {Input::eLeftDrag, ImGuiKey_None, ImGuiMod_Alt,   ImGuiMod_Shift | ImGuiMod_Ctrl, eExamine,     kAny, Action::eLookAround,  "Look around", "Turn the camera in place."},
    {Input::eLeftDrag, ImGuiKey_None, ImGuiMod_Alt,   ImGuiMod_Shift | ImGuiMod_Ctrl, eFly | eWalk, kAny, Action::eOrbit,       "Orbit", "Orbit around the point of interest."},
    {Input::eLeftDrag, ImGuiKey_None, ImGuiMod_Ctrl,  ImGuiMod_Shift,            eAllModes,    kAny, Action::ePan,         "Pan", nullptr},
    {Input::eLeftDrag, ImGuiKey_None, kModNone,  kModAny,              eExamine,     kAny, Action::eOrbit,       "Orbit", "Orbit around the point of interest."},
    {Input::eLeftDrag, ImGuiKey_None, kModNone,  kModAny,              eFly | eWalk, kAny, Action::eLookAround,  "Look around", "Turn the camera in place."},
    // Middle drag
    {Input::eMiddleDrag, ImGuiKey_None, kModNone,  kModAny,  eAllModes, kAny, Action::ePan,       "Pan", nullptr},
    // Right drag
    {Input::eRightDrag, ImGuiKey_None, kModNone,  kModAny,  eAllModes, kPersp, Action::eDolly,       "Dolly", nullptr},
    {Input::eRightDrag, ImGuiKey_None, kModNone,  kModAny,  eAllModes, kOrtho, Action::eDolly,       "Zoom", nullptr},
    // Wheel
    {Input::eWheel, ImGuiKey_None, ImGuiMod_Alt,   ImGuiMod_Shift | ImGuiMod_Ctrl, eAllModes,    kPersp, Action::eWheelZoom,  "Field of view", "Narrow or widen the field of view."},
    {Input::eWheel, ImGuiKey_None, ImGuiMod_Alt,   ImGuiMod_Shift | ImGuiMod_Ctrl, eAllModes,    kOrtho, Action::eWheelZoom,  "Zoom", nullptr},
    {Input::eWheel, ImGuiKey_None, ImGuiMod_Ctrl,  ImGuiMod_Shift,            eFly | eWalk, kPersp, Action::eWheelDollyKeepCenter, "Dolly, keep target",
     "Dolly without moving the point of interest along."},
    {Input::eWheel, ImGuiKey_None, kModNone,  kModAny,              eAllModes,    kPersp, Action::eWheelDolly, "Dolly", nullptr},
    {Input::eWheel, ImGuiKey_None, kModNone,  kModAny,              eAllModes,    kOrtho, Action::eWheelDolly, "Zoom", nullptr},
    // Clicks
    {Input::eLeftClick,       ImGuiKey_None, kModNone, kModAny, eAllModes, kAny, Action::eSelect,             "Select what is under the cursor", nullptr},
    {Input::eLeftDoubleClick, ImGuiKey_None, kModNone, kModAny, eAllModes, kAny, Action::eSetPointOfInterest, "Set the point of interest on that surface", nullptr},
    // Keys
    {Input::eKey, ImGuiKey_W,          kModNone,  ImGuiMod_Shift, eAllModes, kAny, Action::eMoveForward, "Move forward / back", nullptr},
    {Input::eKey, ImGuiKey_S,          kModNone,  ImGuiMod_Shift, eAllModes, kAny, Action::eMoveBack,    "Move forward / back", nullptr},
    {Input::eKey, ImGuiKey_A,          kModNone,  ImGuiMod_Shift, eAllModes, kAny, Action::eMoveLeft,    "Move left / right", nullptr},
    {Input::eKey, ImGuiKey_D,          kModNone,  ImGuiMod_Shift, eAllModes, kAny, Action::eMoveRight,   "Move left / right", nullptr},
    {Input::eKey, ImGuiKey_LeftArrow,  kModNone,  ImGuiMod_Shift, eAllModes, kAny, Action::eMoveLeft,    "Move left / right / up / down", nullptr},
    {Input::eKey, ImGuiKey_RightArrow, kModNone,  ImGuiMod_Shift, eAllModes, kAny, Action::eMoveRight,   "Move left / right / up / down", nullptr},
    {Input::eKey, ImGuiKey_UpArrow,    kModNone,  ImGuiMod_Shift, eAllModes, kAny, Action::eMoveUp,      "Move left / right / up / down", nullptr},
    {Input::eKey, ImGuiKey_DownArrow,  kModNone,  ImGuiMod_Shift, eAllModes, kAny, Action::eMoveDown,    "Move left / right / up / down", nullptr},
    {Input::eKey, ImGuiKey_None,       ImGuiMod_Shift, kModNone,  eAllModes, kAny, Action::eMoveFaster,  "Move faster", "Hold with a movement key."},
});
// clang-format on

Manipulator::Actions motionOf(Action action)
{
  switch(action)
  {
    case Action::eOrbit:
      return Manipulator::Actions::Orbit;
    case Action::ePan:
      return Manipulator::Actions::Pan;
    case Action::eDolly:
      return Manipulator::Actions::Dolly;
    case Action::eLookAround:
      return Manipulator::Actions::LookAround;
    default:
      return Manipulator::Actions::NoAction;
  }
}

// The drag a set of held buttons makes, the left button first.
std::optional<Input> dragInput(const Manipulator::Inputs& inputs)
{
  if(inputs.lmb)
    return Input::eLeftDrag;
  if(inputs.mmb)
    return Input::eMiddleDrag;
  if(inputs.rmb)
    return Input::eRightDrag;
  return std::nullopt;
}

}  // namespace

std::span<const Binding> bindings()
{
  return kBindings;
}

bool matches(const Binding& binding, Mods held, Manipulator::Modes mode, Manipulator::ProjectionType projection)
{
  const bool projectionMatches = binding.projection == Projection::eAny
                                 || (binding.projection == Projection::eOrthographic) == (projection == Manipulator::Orthographic);
  return (held & ~binding.optional) == binding.mods && (binding.modes & (1 << mode)) != 0 && projectionMatches;
}

const Binding* find(Input input, Mods held, Manipulator::Modes mode, Manipulator::ProjectionType projection)
{
  for(size_t i = 0; i < kBindings.size(); ++i)
  {
    if(kBindings[i].input == input && matches(kBindings[i], held, mode, projection))
      return &kBindings[i];
  }
  return nullptr;
}

const Binding* findAction(Action action)
{
  for(size_t i = 0; i < kBindings.size(); ++i)
  {
    if(kBindings[i].action == action)
      return &kBindings[i];
  }
  return nullptr;
}

Mods heldMods()
{
  return Mods(ImGui::GetIO().KeyMods & kModAny);
}

void update(Manipulator& camera, ImGuiWindow* viewportWindow)
{
  if(!nvgui::isWindowHovered(viewportWindow))
    return;

  const Mods                        held       = heldMods();
  const Manipulator::Modes          mode       = camera.getMode();
  const Manipulator::ProjectionType projection = camera.getProjectionType();

  // Movement keys, unless a text field takes the keys.
  const Binding* faster = nullptr;
  for(size_t i = 0; i < kBindings.size(); ++i)
  {
    if(kBindings[i].action == Action::eMoveFaster && matches(kBindings[i], held, mode, projection))
      faster = &kBindings[i];
  }
  const float step = ImGui::GetIO().DeltaTime * (faster != nullptr ? kFasterFactor : 1.0F);
  for(size_t i = 0; i < kBindings.size() && !ImGui::GetIO().WantTextInput; ++i)
  {
    const Binding& binding = kBindings[i];
    if(binding.input != Input::eKey || binding.key == ImGuiKey_None || !ImGui::IsKeyDown(binding.key)
       || !matches(binding, held, mode, projection))
      continue;
    switch(binding.action)
    {
      case Action::eMoveForward:
        camera.keyMotion({step, 0}, Manipulator::Actions::Dolly);
        break;
      case Action::eMoveBack:
        camera.keyMotion({-step, 0}, Manipulator::Actions::Dolly);
        break;
      case Action::eMoveRight:
        camera.keyMotion({step, 0}, Manipulator::Actions::Pan);
        break;
      case Action::eMoveLeft:
        camera.keyMotion({-step, 0}, Manipulator::Actions::Pan);
        break;
      case Action::eMoveUp:
        camera.keyMotion({0, step}, Manipulator::Actions::Pan);
        break;
      case Action::eMoveDown:
        camera.keyMotion({0, -step}, Manipulator::Actions::Pan);
        break;
      default:
        break;
    }
  }

  if(ImGui::IsMouseDragging(ImGuiMouseButton_Left, 1.0F) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 1.0F)
     || ImGui::IsMouseDragging(ImGuiMouseButton_Right, 1.0F))
  {
    Manipulator::Inputs inputs;
    inputs.lmb = ImGui::IsMouseDown(ImGuiMouseButton_Left);
    inputs.mmb = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
    inputs.rmb = ImGui::IsMouseDown(ImGuiMouseButton_Right);

    // Start from the previous cursor position so the motion follows the per-frame mouse delta. This
    // avoids a jump when a drag begins right after interacting with another window: on such a press
    // frame the viewport may not be seen as hovered, so the press position would be stale.
    const ImVec2 mousePos = ImGui::GetMousePos();
    const ImVec2 delta    = ImGui::GetIO().MouseDelta;
    camera.setMousePosition({mousePos.x - delta.x, mousePos.y - delta.y});

    const std::optional<Input> input   = dragInput(inputs);
    const Binding*             binding = input ? find(*input, held, mode, projection) : nullptr;
    const Manipulator::Actions action = binding != nullptr ? motionOf(binding->action) : Manipulator::Actions::NoAction;
    if(action != Manipulator::Actions::NoAction)
      camera.motion({mousePos.x, mousePos.y}, action);
  }

  if(ImGui::GetIO().MouseWheel != 0.0F)
  {
    // The base wheel() picks its motion from Inputs: Shift zooms (field of view, or orthographic size)
    // and Ctrl dollies without moving the center of interest.
    const Binding* binding = find(Input::eWheel, held, mode, projection);
    const double   value   = ImGui::GetIO().MouseWheel * nvapp::ElementCamera::wheelSpeed;
    if(binding != nullptr)
    {
      switch(binding->action)
      {
        case Action::eWheelZoom:
          camera.wheel(value, {.shift = true});
          break;
        case Action::eWheelDollyKeepCenter:
          camera.wheel(value, {.ctrl = true});
          break;
        default:
          camera.wheel(value, {});
          break;
      }
    }
  }
}

}  // namespace camera_controls
