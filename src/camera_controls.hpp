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

// The viewport's mouse and keyboard controls for the camera and picking, owned by the application.
//
// One table, kBindings in camera_controls.cpp, maps every input to an action. Its readers share it,
// so they cannot disagree:
//  - AnchorCameraManipulator's mouseMove() and wheel(), which replace nvutils::CameraManipulator's
//    built-in mapping,
//  - camera_controls::update(), which feeds the manipulator from ImGui every frame in place of
//    nvapp::ElementCamera::updateCamera(), and handles the movement keys,
//  - GltfRenderer::mouseClickedInViewport(), which takes the click that selects and the double-click
//    that sets the point of interest from it,
//  - Help > Keyboard & Mouse (ui_help.cpp), which draws its Camera tab from it.
//
// The manipulator still does the motion itself; the table only decides which motion an input asks
// for: drags through the public motion(), keys through keyMotion(), and the wheel through the base
// wheel() with Inputs made up to select its dolly, dolly-keeping-the-center or zoom.

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <imgui.h>
#include <nvutils/camera_manipulator.hpp>

struct ImGuiWindow;

namespace camera_controls {

using Manipulator = nvutils::CameraManipulator;

enum class Input : uint8_t
{
  eLeftDrag,
  eMiddleDrag,
  eRightDrag,
  eWheel,
  eLeftClick,        // Press and release without dragging
  eLeftDoubleClick,  //
  eKey,
};

// Is the input a drag or the wheel, the ones the manipulator handles?
constexpr bool isMotion(Input input)
{
  return input == Input::eLeftDrag || input == Input::eMiddleDrag || input == Input::eRightDrag || input == Input::eWheel;
}

// The button of a click.
constexpr ImGuiMouseButton clickButton(Input input)
{
  return (input == Input::eLeftClick || input == Input::eLeftDoubleClick) ? ImGuiMouseButton_Left : -1;
}

// Modifier keys, as ImGuiMod_* bits: the same ones the command registry's key chords use.
using Mods              = ImGuiKeyChord;
constexpr Mods kModNone = 0;
constexpr Mods kModAny  = ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiMod_Alt;

// Navigation modes, as a mask of 1 << nvutils::CameraManipulator::Modes.
enum ModeMask : uint8_t
{
  eExamine  = 1 << Manipulator::Examine,
  eFly      = 1 << Manipulator::Fly,
  eWalk     = 1 << Manipulator::Walk,
  eAllModes = eExamine | eFly | eWalk,
};

enum class Projection : uint8_t
{
  eAny,
  ePerspective,
  eOrthographic,
};

enum class Action : uint8_t
{
  // Drags
  eOrbit,        // About the center of interest
  ePan,          //
  eDolly,        // Toward the center of interest; orthographic: zoom
  eLookAround,   // Turn in place
  eOrbitAnchor,  // About the surface under the cursor
  ePanAnchor,    // Keeping the surface under the cursor
  eDollyAnchor,  // Toward the surface under the cursor
  // Wheel
  eWheelDolly,            // Toward the center of interest; orthographic: zoom
  eWheelDollyKeepCenter,  // Fly / Walk: leave the center of interest in place
  eWheelDollyAnchor,      // Toward the surface under the cursor
  eWheelZoom,             // Field of view; orthographic: zoom
  // Keys
  eMoveForward,
  eMoveBack,
  eMoveLeft,
  eMoveRight,
  eMoveUp,
  eMoveDown,
  eMoveFaster,  // Held with the movement keys
  // Clicks, handled by GltfRenderer::mouseClickedInViewport()
  eSelect,              // Select what is under the cursor
  eSetPointOfInterest,  // Move the center of interest to the surface under the cursor
};

// Picks a surface point when the button goes down (or the wheel turns), see AnchorCameraManipulator.
constexpr bool isAnchored(Action action)
{
  return action == Action::eOrbitAnchor || action == Action::ePanAnchor || action == Action::eDollyAnchor
         || action == Action::eWheelDollyAnchor;
}

struct Binding
{
  Input       input;
  ImGuiKey    key;       // Input::eKey only; ImGuiKey_None for a binding made of modifiers alone
  Mods        mods;      // Modifiers that must be held
  Mods        optional;  // Modifiers that may be held too; any other held modifier rules the binding out
  uint8_t     modes;     // ModeMask
  Projection  projection;
  Action      action;
  const char* label;
  const char* tooltip;  // nullptr: none
};

// Every binding. For a given input, the first that matches wins.
std::span<const Binding> bindings();

// The binding an input triggers, or nullptr.
const Binding* find(Input input, Mods held, Manipulator::Modes mode, Manipulator::ProjectionType projection);

// Does the binding apply, given the held modifiers and the camera's mode and projection?
bool matches(const Binding& binding, Mods held, Manipulator::Modes mode, Manipulator::ProjectionType projection);

// The first binding of an action, in any mode, or nullptr.
const Binding* findAction(Action action);

// The modifiers ImGui sees held.
Mods heldMods();

}  // namespace camera_controls

// Camera manipulator driven by the camera_controls table, whose motion can also pivot on the point
// under the cursor instead of the center of interest (nvutils::CameraManipulator's anchor).
//
// The anchored bindings (Shift with a button or the wheel) pick that point. The application answers
// the pick asynchronously through resolveAnchor() (GltfRenderer records it into the next frame and
// polls that frame's semaphore). Until then a drag holds still and wheel steps accumulate, instead of
// moving around the center of interest. A miss falls back to the center of interest.
class AnchorCameraManipulator : public nvutils::CameraManipulator
{
public:
  // Starts a pick at an absolute screen position. Returns false if no answer will come.
  using AnchorRequest = std::function<bool(const glm::dvec2& screenPos)>;

  void setAnchorRequest(AnchorRequest request) { m_anchorRequest = std::move(request); }

  // Answers the pending request; nullopt when nothing was hit or the pick was dropped.
  void resolveAnchor(const std::optional<glm::dvec3>& hit);

  // No mouse button is held: clears the anchor and drops a drag's pending request, so an answer
  // arriving after the release does not set an anchor (setAnchor() also moves the center of interest).
  void releaseAnchor();

  // Drops any pending request and queued wheel steps without moving the camera, e.g. on scene teardown.
  void cancelAnchor();

  void    beginInteraction(const glm::dvec2& pos, const Inputs& inputs) override;
  Actions mouseMove(glm::dvec2 screenPos, const Inputs& inputs) override;
  void    wheel(double value, const Inputs& inputs) override;

private:
  enum class Pending
  {
    None,
    Drag,   // The current anchored drag waits for its anchor
    Wheel,  // Wheel steps accumulate in m_wheelPending
  };

  const camera_controls::Binding* findBinding(camera_controls::Input input, camera_controls::Mods held) const;

  // The inherited wheel dolly, toward the anchor when there is one. Steps are applied one at a time
  // because the inherited response to a step is not linear in its value.
  void wheelToward(const std::optional<glm::dvec3>& anchor, const std::vector<double>& steps);

  AnchorRequest             m_anchorRequest = {};
  bool                      m_anchored      = false;  // The current drag began on an anchored binding
  camera_controls::Mods     m_pressMods     = camera_controls::kModNone;  // Its modifiers, which it keeps
  Pending                   m_pending       = Pending::None;
  std::vector<double>       m_wheelPending  = {};            // Wheel steps waiting for the pick
  std::optional<glm::dvec3> m_wheelAnchor   = std::nullopt;  // Last wheel pick, valid while the view below holds
  glm::dvec2                m_wheelCursor   = {};
  glm::dmat4                m_wheelView     = {};  // Any change of view or projection moves the cursor ray
  glm::dmat4                m_wheelProj     = {};
};

namespace camera_controls {

// Feeds the camera from ImGui's mouse and keyboard, as the table maps them, while the cursor is over
// viewportWindow. Call once per frame inside that window; it does not advance the camera animation.
void update(AnchorCameraManipulator& camera, ImGuiWindow* viewportWindow);

}  // namespace camera_controls
