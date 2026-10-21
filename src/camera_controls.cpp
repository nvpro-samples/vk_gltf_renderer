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

constexpr const char* kAnchorTooltip =
    "Pivots on the surface under the cursor, picked as the button goes down, instead of the point of "
    "interest. The farther the surface, the faster the motion. Over the environment it falls back to "
    "the point of interest.";

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
    {Input::eLeftDrag, ImGuiKey_None, ImGuiMod_Shift, kModNone,             eAllModes,    kAny, Action::eOrbitAnchor, "Orbit the surface", kAnchorTooltip},
    {Input::eLeftDrag, ImGuiKey_None, ImGuiMod_Ctrl,  ImGuiMod_Shift,            eAllModes,    kAny, Action::ePan,         "Pan", nullptr},
    {Input::eLeftDrag, ImGuiKey_None, kModNone,  kModAny,              eExamine,     kAny, Action::eOrbit,       "Orbit", "Orbit around the point of interest."},
    {Input::eLeftDrag, ImGuiKey_None, kModNone,  kModAny,              eFly | eWalk, kAny, Action::eLookAround,  "Look around", "Turn the camera in place."},
    // Middle drag
    {Input::eMiddleDrag, ImGuiKey_None, ImGuiMod_Shift, kModNone, eAllModes, kAny, Action::ePanAnchor, "Pan, surface follows", kAnchorTooltip},
    {Input::eMiddleDrag, ImGuiKey_None, kModNone,  kModAny,  eAllModes, kAny, Action::ePan,       "Pan", nullptr},
    // Right drag
    {Input::eRightDrag, ImGuiKey_None, ImGuiMod_Shift, kModNone, eAllModes, kPersp, Action::eDollyAnchor, "Dolly to the surface", kAnchorTooltip},
    {Input::eRightDrag, ImGuiKey_None, ImGuiMod_Shift, kModNone, eAllModes, kOrtho, Action::eDollyAnchor, "Zoom",
     "Zoom, and move the point of interest to the depth of the surface under the cursor."},
    {Input::eRightDrag, ImGuiKey_None, kModNone,  kModAny,  eAllModes, kPersp, Action::eDolly,       "Dolly", nullptr},
    {Input::eRightDrag, ImGuiKey_None, kModNone,  kModAny,  eAllModes, kOrtho, Action::eDolly,       "Zoom", nullptr},
    // Wheel
    {Input::eWheel, ImGuiKey_None, ImGuiMod_Alt,   ImGuiMod_Shift | ImGuiMod_Ctrl, eAllModes,    kPersp, Action::eWheelZoom,  "Field of view", "Narrow or widen the field of view."},
    {Input::eWheel, ImGuiKey_None, ImGuiMod_Alt,   ImGuiMod_Shift | ImGuiMod_Ctrl, eAllModes,    kOrtho, Action::eWheelZoom,  "Zoom", nullptr},
    {Input::eWheel, ImGuiKey_None, ImGuiMod_Shift, kModNone,             eAllModes,    kPersp, Action::eWheelDollyAnchor, "Dolly to the surface",
     "Dollies toward the surface under the cursor, which stays under it. Over the environment it falls back to the point of interest."},
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
    case Action::eOrbitAnchor:
      return Manipulator::Actions::Orbit;
    case Action::ePan:
    case Action::ePanAnchor:
      return Manipulator::Actions::Pan;
    case Action::eDolly:
    case Action::eDollyAnchor:
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

Mods modsOf(const Manipulator::Inputs& inputs)
{
  return Mods((inputs.shift ? ImGuiMod_Shift : 0) | (inputs.ctrl ? ImGuiMod_Ctrl : 0) | (inputs.alt ? ImGuiMod_Alt : 0));
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

void update(AnchorCameraManipulator& camera, ImGuiWindow* viewportWindow)
{
  // While no button is held, make sure no anchor is left over, whether or not the viewport is hovered.
  // The press frame can be missed, since this returns early when the viewport is not seen as hovered,
  // and a stale anchor would otherwise silently become the pivot for a drag that starts outside and
  // enters the viewport. A drag released before its pick answered must not get the anchor afterwards.
  if(!ImGui::IsMouseDown(ImGuiMouseButton_Left) && !ImGui::IsMouseDown(ImGuiMouseButton_Middle)
     && !ImGui::IsMouseDown(ImGuiMouseButton_Right))
  {
    camera.releaseAnchor();
  }

  if(!nvgui::isWindowHovered(viewportWindow))
    return;

  const Mods          held = heldMods();
  Manipulator::Inputs inputs;
  inputs.lmb                                   = ImGui::IsMouseDown(ImGuiMouseButton_Left);
  inputs.rmb                                   = ImGui::IsMouseDown(ImGuiMouseButton_Right);
  inputs.mmb                                   = ImGui::IsMouseDown(ImGuiMouseButton_Middle);
  inputs.ctrl                                  = (held & ImGuiMod_Ctrl) != 0;
  inputs.shift                                 = (held & ImGuiMod_Shift) != 0;
  inputs.alt                                   = (held & ImGuiMod_Alt) != 0;
  const ImVec2                      mousePos   = ImGui::GetMousePos();
  const Manipulator::Modes          mode       = camera.getMode();
  const Manipulator::ProjectionType projection = camera.getProjectionType();

  // Movement keys, unless a text field takes the keys. A key that moves the camera keeps its modifiers
  // to itself, so Shift held for speed does not also turn a drag or the wheel into an anchored one.
  const Binding* faster = nullptr;
  for(size_t i = 0; i < kBindings.size(); ++i)
  {
    if(kBindings[i].action == Action::eMoveFaster && matches(kBindings[i], held, mode, projection))
      faster = &kBindings[i];
  }
  const float step  = ImGui::GetIO().DeltaTime * (faster != nullptr ? kFasterFactor : 1.0F);
  bool        moved = false;
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
        continue;
    }
    moved = true;
  }
  if(moved)
    inputs.shift = inputs.ctrl = false;

  // Begin an interaction on mouse down, which is where an anchored binding picks its point. After the
  // movement keys, so Shift held for speed in the same frame does not latch an anchored drag.
  if(ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseClicked(ImGuiMouseButton_Middle)
     || ImGui::IsMouseClicked(ImGuiMouseButton_Right))
  {
    camera.beginInteraction({mousePos.x, mousePos.y}, inputs);
  }

  if(ImGui::IsMouseDragging(ImGuiMouseButton_Left, 1.0F) || ImGui::IsMouseDragging(ImGuiMouseButton_Middle, 1.0F)
     || ImGui::IsMouseDragging(ImGuiMouseButton_Right, 1.0F))
  {
    // Start from the previous cursor position so the motion follows the per-frame mouse delta. This
    // avoids a jump when a drag begins right after interacting with another window: on such a press
    // frame the viewport may not be seen as hovered, so beginInteraction() would be skipped and the
    // first motion computed against a stale position.
    const ImVec2 delta = ImGui::GetIO().MouseDelta;
    camera.setMousePosition({mousePos.x - delta.x, mousePos.y - delta.y});
    camera.mouseMove({mousePos.x, mousePos.y}, inputs);
  }

  if(ImGui::GetIO().MouseWheel != 0.0F)
    camera.wheel(ImGui::GetIO().MouseWheel * nvapp::ElementCamera::wheelSpeed, inputs);
}

}  // namespace camera_controls

//--------------------------------------------------------------------------------------------------
// AnchorCameraManipulator

using namespace camera_controls;

const Binding* AnchorCameraManipulator::findBinding(Input input, Mods held) const
{
  return find(input, held, getMode(), getProjectionType());
}

void AnchorCameraManipulator::resolveAnchor(const std::optional<glm::dvec3>& hit)
{
  const Pending pending = std::exchange(m_pending, Pending::None);
  if(pending == Pending::Drag && hit)
  {
    setAnchor(*hit);
  }
  else if(pending == Pending::Wheel)
  {
    m_wheelAnchor = hit;
    wheelToward(hit, std::exchange(m_wheelPending, {}));
  }
}

void AnchorCameraManipulator::releaseAnchor()
{
  clearAnchor();
  if(m_pending == Pending::Drag)
    m_pending = Pending::None;
}

void AnchorCameraManipulator::cancelAnchor()
{
  clearAnchor();
  m_pending = Pending::None;
  m_wheelPending.clear();
  m_wheelAnchor.reset();
}

void AnchorCameraManipulator::beginInteraction(const glm::dvec2& pos, const Inputs& inputs)
{
  CameraManipulator::beginInteraction(pos, inputs);
  m_wheelAnchor.reset();
  m_wheelPending.clear();

  const std::optional<Input> input   = dragInput(inputs);
  const Binding*             binding = input ? findBinding(*input, modsOf(inputs)) : nullptr;
  m_pressMods                        = modsOf(inputs);
  m_anchored                         = binding != nullptr && isAnchored(binding->action) && m_anchorRequest;
  m_pending                          = (m_anchored && m_anchorRequest(pos)) ? Pending::Drag : Pending::None;
}

AnchorCameraManipulator::Actions AnchorCameraManipulator::mouseMove(glm::dvec2 screenPos, const Inputs& inputs)
{
  const std::optional<Input> input = dragInput(inputs);
  if(!input)
  {
    setMousePosition(screenPos);
    return Actions::NoAction;
  }

  if(m_anchored && m_pending == Pending::Drag)
  {
    setMousePosition(screenPos);
    return Actions::NoAction;
  }

  // An anchored drag keeps the modifiers it began with, so letting go of Shift does not drop the
  // anchor halfway. Any other drag follows the modifiers as they change; an anchored binding reached
  // that way has no anchor and moves about the center of interest.
  const Binding* binding = findBinding(*input, m_anchored ? m_pressMods : modsOf(inputs));
  const Actions  action  = binding != nullptr ? motionOf(binding->action) : Actions::NoAction;
  if(action != Actions::NoAction)
    motion(screenPos, action);
  return action;
}

void AnchorCameraManipulator::wheel(double value, const Inputs& inputs)
{
  const Binding* binding = findBinding(Input::eWheel, modsOf(inputs));
  if(binding == nullptr)
    return;

  // The base wheel() picks its motion from Inputs: Shift zooms (field of view, or orthographic size)
  // and Ctrl dollies without moving the center of interest.
  switch(binding->action)
  {
    case Action::eWheelZoom:
      CameraManipulator::wheel(value, {.shift = true});
      return;
    case Action::eWheelDollyKeepCenter:
      CameraManipulator::wheel(value, {.ctrl = true});
      return;
    case Action::eWheelDollyAnchor:
      break;
    default:
      CameraManipulator::wheel(value, {});
      return;
  }

  // During a drag the drag's anchor, if any, stays the pivot.
  if(hasAnchor() || m_pending == Pending::Drag || !m_anchorRequest)
  {
    CameraManipulator::wheel(value, {});
    return;
  }

  // Dollying moves the eye along the ray through the anchor, so the anchor stays under the cursor and
  // later steps can reuse it until the cursor or the camera moves some other way (any change of the
  // view or projection, including a turn in place that keeps the eye).
  const ImVec2     mouse = ImGui::GetMousePos();
  const glm::dvec2 cursor{mouse.x, mouse.y};
  if(m_wheelAnchor && cursor == m_wheelCursor && getViewMatrix() == m_wheelView && getPerspectiveMatrix() == m_wheelProj)
  {
    wheelToward(m_wheelAnchor, {value});
    return;
  }

  m_wheelPending.push_back(value);
  if(m_pending == Pending::Wheel)
    return;
  m_wheelCursor = cursor;
  if(m_anchorRequest(cursor))
    m_pending = Pending::Wheel;
  else
    wheelToward(std::nullopt, std::exchange(m_wheelPending, {}));
}

void AnchorCameraManipulator::wheelToward(const std::optional<glm::dvec3>& anchor, const std::vector<double>& steps)
{
  if(anchor)
    setAnchor(*anchor);
  for(double step : steps)
    CameraManipulator::wheel(step, {});
  clearAnchor();
  m_wheelView = getViewMatrix();
  m_wheelProj = getPerspectiveMatrix();
}
