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

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

//--------------------------------------------------------------------------------------------------
// Who the sun is, and where it points.
//
// OMI_environment_sky's model is that the sky is a medium and the sun is a light, which is why a
// saved sky carries a directional light beside it. That leaves the sun with two possible owners --
// the renderer's own `Resources::sunDirection`, or a light in the scene that the document marks as
// the sun -- and exactly one invariant worth having a class for: *which one is in charge*, and
// therefore what moving the sun costs. With no marked light it is a float; with one it is a scene
// edit and belongs on the undo stack like any other.
//
// Every path that moves the sun -- the panel's sliders, the Time of Day widget, the viewport's
// Ctrl+Shift+L drag, a preset load -- goes through aim(). That is the point: the "who owns it"
// question is answered once rather than at each call site.
//
// An unmarked directional light is never adopted. A scene may hold any number of them and none is
// a sun by virtue of being directional; guessing at one is the mistake the marker exists to retire.
//--------------------------------------------------------------------------------------------------

struct Resources;
class UndoStack;

class SkySun
{
public:
  // `resetFrame` restarts accumulation; `environmentPreview` reports that the sky needs re-baking,
  // which only matters for sky types whose bake depends on the sun.
  struct Actions
  {
    std::function<void()> resetFrame;
    std::function<void()> environmentPreview;
  };

  void init(Resources* resources, UndoStack* undoStack, Actions actions);

  // Re-read the scene's marker and adopt whatever light it names, or fall back to the renderer's
  // own sun. Runs at the frame top, so moving that light -- gizmo, inspector, animation -- carries
  // the sky with it.
  void syncFromMarkedLight();

  // Point the sun along `toSun`, through whatever owns it, and report that it moved.
  void aim(const glm::vec3& toSun);

  // Drive the sun from azimuth/elevation in degrees, and the inverse: mirror the direction back
  // into the reported angles so reading sunAzimuth/sunElevation tells the truth after a UI move.
  void setAngles(float azimuthDegrees, float elevationDegrees);
  void syncAngles();

  // Recompute from the Time of Day settings (place, date, clock, north offset) and aim. No-op if
  // the configured date does not parse.
  void applyTimeOfDay();

  // Directional lights the scene offers as candidate suns: node index and display name, in scene
  // order. The sun is chosen rather than guessed, so this is what the choice is made from.
  [[nodiscard]] std::vector<std::pair<int, std::string>> candidateLights() const;

  // Make `nodeIndex` the sky's sun, or -1 to hand the sun back to the renderer. Undoable: the
  // marker lives in the scene, so setting it is a scene edit.
  void setSunNode(int nodeIndex);

  // Scene light index of the sun, or -1 when the renderer supplies its own. Read by the frame
  // push-constants, which need to tell the shaders whether to add a sun of their own.
  [[nodiscard]] int lightIndex() const { return m_lightIndex; }
  // Node index of the marked light, for the panel to edit through undo_redo.
  [[nodiscard]] int lightNode() const { return m_lightNode; }

private:
  // Point the scene's marked sun light along `toSun`, as an undoable edit. False when there is no
  // marked light to move, which is how aim() decides whether it is editing a scene or a float.
  bool setMarkedDirection(const glm::vec3& toSun);

  Resources* m_resources{nullptr};
  UndoStack* m_undoStack{nullptr};
  Actions    m_actions;

  int m_lightIndex{-1};
  int m_lightNode{-1};
};
