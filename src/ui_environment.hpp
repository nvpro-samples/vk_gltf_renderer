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
#include <glm/gtc/quaternion.hpp>

//--------------------------------------------------------------------------------------------------
// The Environment panel: which environment lights the scene, and everything that describes it.
//
// A panel class like UiSceneBrowser / UiInspector / AnimationControl, and for the same reason: what
// it needs from the renderer arrives through the setters below rather than through membership. An
// earlier version hung all twenty-six of these sections off GltfRenderer because they edit renderer
// state and hold none of their own -- which was true, and still put a quarter of the renderer's
// public surface in a file about ImGui rows.
//
// The panel writes Settings directly, through the Resources pointer. That is deliberate and matches
// the other panels: a setting *is* the shared state, and routing every slider through an accessor
// would buy indirection rather than encapsulation. What goes through EnvironmentPanelActions is
// everything that is not a value -- the consequences an edit has for the renderer.
//--------------------------------------------------------------------------------------------------

struct Resources;
class SceneSelection;

// One day at one place: when each named moment falls, in local clock hours. NaN means the moment
// does not happen (a polar day has no sunrise). The slider ticks and the Jump To presets read from
// the same solve so they can never disagree about where the horizon is.
struct TimeOfDayEvents
{
  bool  valid{false};  // the date parsed; every hour below is meaningless otherwise
  float sunrise{};
  float goldenHourAM{};
  float solarNoon{};
  float goldenHourPM{};
  float sunset{};
  float blueHour{};
  float solarMidnight{};
};

// What an edit in this panel means to the renderer.
//
// One std::function per consequence rather than a renderer back-pointer, so the panel states its
// dependencies instead of inheriting all of them -- the same shape UiInspector uses.
//
// All of them are required. setActions() asserts that, rather than each of the ~20 call sites
// testing a function that is never legitimately empty: a missing action is a wiring mistake, and
// it should fail at attach where the mistake is, not on the first frame a user happens to open the
// fold that needs it.
struct EnvironmentPanelActions
{
  std::function<void()> environmentChanged;  // the environment needs rebuilding from scratch
  std::function<void()> requestPreview;      // a drag is in progress: preview now, commit on settle
  std::function<void()> resetFrame;          // restart accumulation
  std::function<void()> applyTimeOfDay;      // place/date/clock moved: recompute the sun and aim it
  std::function<void()> environmentRotated;  // Settings::envRotation was edited: north moved, re-place the sun

  std::function<void(float azimuthDeg, float elevationDeg)> setSunAngles;
  std::function<void(const glm::vec3& toSun)>               aimSun;

  // Which light is the sun. -1 hands it back to the renderer; the setter is undoable because the
  // marker lives in the scene.
  std::function<int()>                                      sunLightNode;
  std::function<void(int nodeIndex)>                        setSkySunNode;
  std::function<std::vector<std::pair<int, std::string>>()> directionalLightNodes;

  // File dialogs. The panel owns the button; the renderer owns what a chosen path means -- a load
  // touches scene and Vulkan state, which is the application thread's business, not ImGui's.
  std::function<void()> loadHdrFileDialog;
  std::function<void()> loadSkyPresetDialog;
  std::function<void()> saveSkyPresetDialog;
};

class UiEnvironment
{
public:
  void setResources(Resources* resources) { m_resources = resources; }       // required
  void setSelection(SceneSelection* selection) { m_selection = selection; }  // required
  void setActions(EnvironmentPanelActions actions);

  // The whole panel, one ImGui window.
  void render();

  // Restore the selected environment type's settings to their defaults, leaving every other type --
  // and the loaded HDR file -- untouched. Public because the `envResetDefaults` setting action
  // reaches it too, not only the button in the panel.
  void resetDefaults();

private:
  // One small function per row/fold, so render() and sunAndTimeOfDay() are short call sequences
  // whose order is shuffled by moving a single line. Each returns true when the user edited
  // something this frame.
  bool envResetButton();
  bool envTypePicker();
  bool environmentRotation();
  bool hdrSection();
  bool plainSkySection();
  bool gradientSkySection();
  bool physicalSkySection();
  bool authoredSkyBakeSection();

  // Sky types that carry a sun get two tabs (Sun & Time / look) because the two groups edit
  // independent things and one long scroll made users step over the group they didn't want.
  bool skyWithSunTabs();

  // Physical Sky sub-sections, one per category the Unreal sky component uses. `atmoChanged`
  // (out-param) tracks edits that need a scattering-table re-bake (requestPreview); the return
  // value covers edits that only need to restart accumulation -- Aerial Perspective's scene scale
  // is the one where the two differ, which is why it does not take the out-param.
  bool atmospherePresetPicker(bool& atmoChanged);
  bool atmosphereRayleigh(bool& atmoChanged);
  bool atmosphereMie(bool& atmoChanged);
  bool atmosphereOzone(bool& atmoChanged);
  bool atmosphereAerialPerspective();
  bool atmospherePlanet(bool& atmoChanged);
  bool atmosphereSun(bool& atmoChanged);

  // Sun & Time of Day sub-sections. Callers solve the day once and pass the events in.
  bool sunAndTimeOfDay();
  bool sunTimeSlider(const TimeOfDayEvents& events);
  bool sunJumpToPresets(const TimeOfDayEvents& events);
  bool sunAngleSliders();
  bool sunAdvancedFold(const TimeOfDayEvents& events);
  bool sunGizmoButton();
  bool sunLocationAndDate(const TimeOfDayEvents& events);
  void sunSourcePicker();

  Resources*              m_resources{nullptr};
  SceneSelection*         m_selection{nullptr};
  EnvironmentPanelActions m_actions;

  // environmentRotation()'s angles, and the quaternion they were last derived from or written as.
  struct
  {
    glm::vec3 degrees{0.0F};  // heading (Y, the slider), then the kept tilt about X and Z
    glm::quat quat{1.0F, 0.0F, 0.0F, 0.0F};
  } m_rotationEuler;
};
