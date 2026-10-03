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

//--------------------------------------------------------------------------------------------------
// The Environment panel: which environment lights the scene, and everything that describes it.
//
// One panel per file, as with ui_inspector / ui_scene_browser / ui_animation; ui_renderer.cpp keeps
// the viewport and the menus. This grew large enough to be worth its own file when the physical
// sky's atmosphere arrived -- an HDR file, four authored sky types and a full atmosphere is most of
// what a user can say about lighting, and none of it is about the renderer window.
//
// A panel class, like the others -- see ui_environment.hpp for why what it needs from the
// renderer arrives through setters rather than through membership.
//

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <unordered_map>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <imgui.h>

#include <nvgui/azimuth_sliders.hpp>
#include <nvgui/file_dialog.hpp>
#include <nvgui/fonts.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/sky.hpp>
#include <nvgui/tooltip.hpp>
#include <nvutils/file_operations.hpp>

#include "env_image_loader.hpp"
#include "gltf_environment_sky.hpp"
#include "renderer.hpp"
#include "ui_environment.hpp"
#include "sky_bruneton.hpp"
#include "sun_position.hpp"
#include "ui_linear_color.hpp"

namespace PE = nvgui::PropertyEditor;

//--------------------------------------------------------------------------------------------------
// Time of Day: the day's shape, solved rather than tabulated.
//
// "Sunset at 19:30" is a fixed clock time and is wrong for most of the year at every latitude but
// one. Each named moment here is instead an *altitude* the sun passes through -- 0 for the horizon,
// +6 for the golden hour, -4 for the blue one -- solved for the configured place and date from the
// same function that drives the slider. Which also means a moment that does not happen (a polar day
// has no sunrise) comes back as a NaN and can be shown as unavailable rather than as a lie.
//
namespace {

// Named moments, in the order they appear in the day and in the menu.
enum class SunEvent
{
  eSunrise,
  eGoldenHourAM,
  eSolarNoon,
  eGoldenHourPM,
  eSunset,
  eBlueHour,
  eSolarMidnight,
};

struct TimeOfDayPreset
{
  const char* name;
  SunEvent    event;
};

constexpr TimeOfDayPreset kTimeOfDayPresets[] = {
    {"Sunrise", SunEvent::eSunrise},        {"Golden Hour AM", SunEvent::eGoldenHourAM},
    {"Noon", SunEvent::eSolarNoon},         {"Golden Hour PM", SunEvent::eGoldenHourPM},
    {"Sunset", SunEvent::eSunset},          {"Blue Hour", SunEvent::eBlueHour},
    {"Midnight", SunEvent::eSolarMidnight},
};

// The clock hour for a named event, out of an already-solved TimeOfDayEvents. NaN when the event
// never happens on that day. Free function rather than a member of TimeOfDayEvents so the struct
// stays plain data in the header, out of every translation unit that includes renderer.hpp.
[[nodiscard]] inline float hourOfEvent(const TimeOfDayEvents& ev, SunEvent e)
{
  switch(e)
  {
    case SunEvent::eSunrise:
      return ev.sunrise;
    case SunEvent::eGoldenHourAM:
      return ev.goldenHourAM;
    case SunEvent::eSolarNoon:
      return ev.solarNoon;
    case SunEvent::eGoldenHourPM:
      return ev.goldenHourPM;
    case SunEvent::eSunset:
      return ev.sunset;
    case SunEvent::eBlueHour:
      return ev.blueHour;
    case SunEvent::eSolarMidnight:
      return ev.solarMidnight;
  }
  return std::numeric_limits<float>::quiet_NaN();
}

// The sun's altitude at a reading of the local clock. The one place the clock-to-UTC conversion
// happens on this side of the panel; UiEnvironment::applyTimeOfDay does the same on the other.
float elevationAtLocalHour(const Settings& st, int year, int month, int day, float localHour)
{
  return sun_position::computeSunPosition({st.todLatitude, st.todLongitude}, {year, month, day, localHour - st.todUtcOffset})
      .elevationDeg;
}

// The clock hour at which the sun passes `targetElevation`, going up if `rising`. NaN when it never
// does on that day.
//
// A coarse scan for the bracket and a bisection inside it, rather than an analytic inversion. The
// inversion exists but has to special-case the polar cases separately anyway, and this reuses the
// exact function the slider drives -- so the tick, the preset and the rendered sun cannot disagree
// about where the horizon is.
float solveCrossing(const Settings& st, int year, int month, int day, float targetElevation, bool rising)
{
  constexpr int kScanSteps = 288;  // five-minute resolution: finer than any crossing is sharp
  float         prevHour   = 0.0F;
  float         prevElev   = elevationAtLocalHour(st, year, month, day, 0.0F);

  for(int i = 1; i <= kScanSteps; ++i)
  {
    const float hour = 24.0F * static_cast<float>(i) / static_cast<float>(kScanSteps);
    const float elev = elevationAtLocalHour(st, year, month, day, hour);

    const bool goingUp  = elev > prevElev;
    const bool brackets = (prevElev - targetElevation) * (elev - targetElevation) <= 0.0F;
    if(brackets && goingUp == rising)
    {
      float lo = prevHour, hi = hour;
      for(int iter = 0; iter < 40; ++iter)
      {
        const float mid     = 0.5F * (lo + hi);
        const float midElev = elevationAtLocalHour(st, year, month, day, mid);
        // Below the target on a rising crossing means the answer is still ahead, and vice versa.
        if((midElev < targetElevation) == rising)
          lo = mid;
        else
          hi = mid;
      }
      return 0.5F * (lo + hi);
    }
    prevHour = hour;
    prevElev = elev;
  }
  return std::numeric_limits<float>::quiet_NaN();
}

// The clock hour of the day's highest or lowest sun. Always exists, unlike the crossings.
float solveExtremum(const Settings& st, int year, int month, int day, bool highest)
{
  constexpr int kScanSteps = 288;
  float         bestHour   = 0.0F;
  float         bestElev   = elevationAtLocalHour(st, year, month, day, 0.0F);
  for(int i = 1; i <= kScanSteps; ++i)
  {
    const float hour = 24.0F * static_cast<float>(i) / static_cast<float>(kScanSteps);
    const float elev = elevationAtLocalHour(st, year, month, day, hour);
    if(highest ? (elev > bestElev) : (elev < bestElev))
    {
      bestElev = elev;
      bestHour = hour;
    }
  }
  // Refine inside the bracket the scan leaves: a ternary search, since the extremum is smooth.
  float lo = bestHour - 24.0F / kScanSteps;
  float hi = bestHour + 24.0F / kScanSteps;
  for(int iter = 0; iter < 30; ++iter)
  {
    const float a = lo + (hi - lo) / 3.0F;
    const float b = hi - (hi - lo) / 3.0F;
    if((elevationAtLocalHour(st, year, month, day, a) < elevationAtLocalHour(st, year, month, day, b)) == highest)
      lo = a;
    else
      hi = b;
  }
  return std::fmod(0.5F * (lo + hi) + 24.0F, 24.0F);
}

// Everything the widget needs to know about today. Recomputed each frame the panel is open: it is a
// few hundred evaluations of ~50 floating-point operations, which is far below the cost of caching
// it correctly against five settings that can each change from the UI, the ini or the command line.
TimeOfDayEvents solveDay(const Settings& st)
{
  TimeOfDayEvents ev;
  int             year = 0, month = 0, day = 0;
  if(!sun_position::parseIsoDate(st.todDate, year, month, day))
    return ev;

  ev.valid         = true;
  ev.sunrise       = solveCrossing(st, year, month, day, 0.0F, true);
  ev.goldenHourAM  = solveCrossing(st, year, month, day, 6.0F, true);
  ev.goldenHourPM  = solveCrossing(st, year, month, day, 6.0F, false);
  ev.sunset        = solveCrossing(st, year, month, day, 0.0F, false);
  ev.blueHour      = solveCrossing(st, year, month, day, -4.0F, false);
  ev.solarNoon     = solveExtremum(st, year, month, day, true);
  ev.solarMidnight = solveExtremum(st, year, month, day, false);
  return ev;
}

// "07:42" for 7.7. Also the slider's display format: ImGui prints a format string holding no
// conversion specifier verbatim, which is the cheapest way to put a clock face on a float.
void formatClock(char* out, size_t size, float hour)
{
  const float wrapped = std::fmod(std::fmod(hour, 24.0F) + 24.0F, 24.0F);
  int         hours   = static_cast<int>(wrapped);
  int         minutes = static_cast<int>((wrapped - static_cast<float>(hours)) * 60.0F + 0.5F);
  if(minutes == 60)  // 23:59.7 must read 00:00, not 23:60
  {
    minutes = 0;
    hours   = (hours + 1) % 24;
  }
  snprintf(out, size, "%02d:%02d", hours, minutes);
}

// Four marks under the time slider: sunrise, noon, sunset, midnight for the configured place and
// date. Call immediately after the slider -- it reads the rectangle ImGui just laid out.
//
// They are what makes the slider legible: 06:00 means nothing on its own, and "just left of the
// sunrise tick" means everything. Marks for events that do not happen are simply absent.
void drawDayEventTicks(const TimeOfDayEvents& ev)
{
  if(!ev.valid)
    return;

  const ImVec2 lo = ImGui::GetItemRectMin();
  const ImVec2 hi = ImGui::GetItemRectMax();
  ImDrawList*  dl = ImGui::GetWindowDrawList();

  const struct
  {
    float hour;
    ImU32 color;
  } marks[] = {
      {ev.solarMidnight, IM_COL32(110, 140, 210, 220)},
      {ev.sunrise, IM_COL32(255, 170, 90, 230)},
      {ev.solarNoon, IM_COL32(255, 240, 170, 230)},
      {ev.sunset, IM_COL32(255, 130, 70, 230)},
  };

  for(const auto& mark : marks)
  {
    if(!std::isfinite(mark.hour))
      continue;
    const float x = lo.x + (hi.x - lo.x) * std::clamp(mark.hour / 24.0F, 0.0F, 1.0F);
    dl->AddLine(ImVec2(x, hi.y - 4.0F), ImVec2(x, hi.y), mark.color, 2.0F);
  }
}

// "strength (/km) + tint" for one atmospheric component, both derived from and written back into
// the stored per-km RGB vector -- so the two display halves and the file layout can't drift.
// Called inside each component's tree; short child labels ("Scattering" / "Tint") work because the
// tree header supplies the component name ("Rayleigh > Scattering" reads left to right).
//
// The pair is remembered between frames and only re-derived when the stored vector changes
// underneath it (a preset, a loaded scene, the command line). Deriving it every frame normalised the
// tint to a peak of 1, so any edit that lowered the brightest channel -- a greyer or darker tint --
// was folded straight back into the strength, and the swatch snapped back: to exactly white for
// Mie's grey default. Remembered, the tint is what the user set, and a strength of 0 no longer
// forgets it either.
[[nodiscard]] bool scatteringRow(const char* strengthLabel, glm::vec3& perKm, float maxPerKm, const std::string& strengthTooltip)
{
  struct Split
  {
    glm::vec3 tint{1.0F};
    float     strength{0.0F};
  };
  static std::unordered_map<ImGuiID, Split> s_splits;  // per row; the tree scope makes the ID unique

  Split& split = s_splits[ImGui::GetID(strengthLabel)];
  if(split.tint * split.strength != perKm)  // exact: it is the same product this row writes back
  {
    const float peak = std::max({perKm.x, perKm.y, perKm.z});
    split            = {(peak > 0.0F) ? perKm / peak : glm::vec3(1.0F), peak};
  }

  bool hit = PE::SliderFloat(strengthLabel, &split.strength, 0.0F, maxPerKm, "%.5f /km", ImGuiSliderFlags_Logarithmic, strengthTooltip);
  hit |= uicolor::colorEdit3Linear("Tint", glm::value_ptr(split.tint),
                                   "Which wavelengths this affects. On Earth: Rayleigh runs blue (air scatters short "
                                   "wavelengths hardest), Mie stays near white, ozone tints yellow-green. The stored "
                                   "per-channel coefficient is Strength x Tint.");
  if(hit)
    perKm = split.tint * split.strength;
  return hit;
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// Every action is required; a missing one is a wiring mistake in GltfRenderer::onAttach.
//
// Checked here, once, rather than at each of the ~20 call sites. A guard at the call site would
// turn that mistake into a row that silently does nothing, which is the harder bug: the panel looks
// wired, and only the one fold nobody opened during testing is dead.
//
void UiEnvironment::setActions(EnvironmentPanelActions actions)
{
  assert(actions.environmentChanged && actions.requestPreview && actions.resetFrame && actions.applyTimeOfDay);
  assert(actions.environmentRotated);
  assert(actions.setSunAngles && actions.aimSun);
  assert(actions.sunLightNode && actions.setSkySunNode && actions.directionalLightNodes);
  assert(actions.loadHdrFileDialog && actions.loadSkyPresetDialog && actions.saveSkyPresetDialog);
  m_actions = std::move(actions);
}

//--------------------------------------------------------------------------------------------------
// Environment window - orchestrates the panel by calling one helper per section.
//
// Each section helper below is a small function that renders its own rows and returns whether the
// user touched anything this frame. Reordering the panel is a one-line move here; hiding a section
// behind a condition is a one-line change. The complexity that used to live inline moved into
// atmosphereQuickKnobs() / atmosphereAdvancedFold() where it can be moved between tiers freely.
//
void UiEnvironment::render()
{
  if(!m_resources->settings.showEnvironmentWindow)
    return;

  if(!ImGui::Begin("Environment", &m_resources->settings.showEnvironmentWindow))
  {
    ImGui::End();
    return;
  }
  nvgui::tooltip("Press F5 to toggle this window");

  bool changed = false;
  changed |= envResetButton();
  changed |= envTypePicker();
  changed |= environmentRotation();

  // The per-type section, in the order the type picker offers them. Sky and Gradient each carry
  // two orthogonal groups of settings -- what the sky looks like, and where the sun is -- so those
  // two split into tabs below; the other types have only one group and render inline.
  switch(m_resources->settings.envSystem)
  {
    case shaderio::EnvSystem::eHdr:
      changed |= hdrSection();
      break;
    case shaderio::EnvSystem::ePlain:
      changed |= plainSkySection();
      break;
    case shaderio::EnvSystem::eSky:
    case shaderio::EnvSystem::eGradient:
      changed |= skyWithSunTabs();
      break;
    case shaderio::EnvSystem::eNone:
      break;  // nothing to configure
  }

  // The save toggle and presets apply to every authored sky type.
  if(isBakedEnvironment(m_resources->settings.envSystem))
    changed |= authoredSkyBakeSection();

  if(changed)
    m_actions.resetFrame();

  ImGui::End();
}

//--------------------------------------------------------------------------------------------------
// Reset button: acts on whichever type is selected, so it sits above the type picker rather than
// inside a fold that would suggest it only reaches that section. Disabled for None, which has
// nothing to restore.
//
bool UiEnvironment::envResetButton()
{
  const bool resettable = m_resources->settings.envSystem != shaderio::EnvSystem::eNone;
  bool       changed    = false;
  ImGui::BeginDisabled(!resettable);
  if(ImGui::SmallButton(ICON_MS_RESTART_ALT " Reset"))
  {
    resetDefaults();
    changed = true;
  }
  ImGui::EndDisabled();
  if(ImGui::IsItemHovered())
    ImGui::SetTooltip(resettable ? "Restore the selected environment type's settings to their defaults.\n"
                                   "Other types, the loaded HDR file, and the background color are left alone." :
                                   "Nothing to reset: \"None\" has no settings.");
  return changed;
}

//--------------------------------------------------------------------------------------------------
// Environment Type combo + Solid Color toggle + background color.
//
bool UiEnvironment::envTypePicker()
{
  bool changed = false;
  if(!PE::begin())
    return changed;

  // Explicit label/value table rather than a positional combo string: display order is chosen
  // for readability (authored sky types first) while the underlying integers are an on-disk
  // format that must stay Sky=0, HDR=1, None=2, Plain=3, Gradient=4. A positional string would
  // weld the two together forever.
  struct EnvTypeEntry
  {
    const char*         label;
    shaderio::EnvSystem value;
    const char*         tooltip;
  };
  static constexpr EnvTypeEntry kEnvTypes[] = {
      {"Plain", shaderio::EnvSystem::ePlain,
       "Authored solid-color sky (OMI_environment_sky). A plain sky at [0,0,0] renders identically to None but "
       "preserves the sky in the glTF and still runs environment sampling -- use None for the fastest no-sky mode."},
      {"Gradient", shaderio::EnvSystem::eGradient, "Authored bottom/horizon/top gradient with a sun term (OMI_environment_sky)."},
      {"Sky", shaderio::EnvSystem::eSky, "Procedural physical sky."},
      {"HDR", shaderio::EnvSystem::eHdr, "Lat-long HDR image loaded from a file."},
      {"None", shaderio::EnvSystem::eNone, "No environment lighting or background. Cheapest mode: environment sampling is disabled entirely."},
  };

  const auto currentEnvLabel = [&]() -> const char* {
    for(const EnvTypeEntry& e : kEnvTypes)
      if(e.value == m_resources->settings.envSystem)
        return e.label;
    return "?";
  };

  if(PE::entry(
         "Environment Type",
         [&] {
           bool picked = false;
           if(ImGui::BeginCombo("##EnvType", currentEnvLabel()))
           {
             for(const EnvTypeEntry& e : kEnvTypes)
             {
               const bool selected = (e.value == m_resources->settings.envSystem);
               if(ImGui::Selectable(e.label, selected))
               {
                 m_resources->settings.envSystem = e.value;
                 picked                          = true;
               }
               if(ImGui::IsItemHovered())
                 ImGui::SetTooltip("%s", e.tooltip);
               if(selected)
                 ImGui::SetItemDefaultFocus();
             }
             ImGui::EndCombo();
           }
           return picked;
         },
         "Which environment lights the scene and shows behind it"))
  {
    // The new mode may need a bake, or may need the HDR file rebound; onEnvironmentChanged
    // defers both to the top of the next frame, where a queue submit is safe. The firefly clamp is
    // recalibrated there too, once the new environment's integral exists.
    m_actions.environmentChanged();
    changed = true;
  }
  changed |= PE::Checkbox("Solid Color", &m_resources->settings.useSolidBackground);
  if(m_resources->settings.useSolidBackground)
  {
    changed |= uicolor::colorEdit3Linear("Background Color", glm::value_ptr(m_resources->settings.solidBackgroundColor),
                                         "Solid background color (shown/edited in linear; swatch/wheel perceptual).");
  }
  PE::end();
  return changed;
}

//--------------------------------------------------------------------------------------------------
// The environment's orientation: one rotation for the HDR, every sky, and the Time of Day compass.
//
// Stored as OMI_environment_sky's quaternion, edited as one angle about +Y. The quaternion is split
// Y-X-Z into that heading and whatever tilt remains; the slider replaces only the heading, so a tilt
// a file arrived with (or --envRotation set) survives the edit and the next save rather than being
// silently flattened. The angles are cached and only re-derived when the quaternion changes
// underneath them (a scene load, the command line, MCP), so a drag is not fed back through a
// decomposition that would snap it around -- the Inspector's rotation does the same.
//
bool UiEnvironment::environmentRotation()
{
  Settings& settings = m_resources->settings;
  if(!environmentTurns(settings.envSystem))
    return false;
  if(!PE::begin("Orientation"))
    return false;

  const glm::quat current = settings.envRotationQuat();
  if(std::abs(glm::dot(current, m_rotationEuler.quat)) < 1.0F - 1e-6F)
  {
    float heading = 0.0F, pitch = 0.0F, roll = 0.0F;
    glm::extractEulerAngleYXZ(glm::mat4_cast(current), heading, pitch, roll);
    m_rotationEuler.degrees = glm::degrees(glm::vec3(heading, pitch, roll));
    m_rotationEuler.quat    = current;
  }

  glm::vec3& deg    = m_rotationEuler.degrees;
  const bool edited = PE::SliderFloat("Rotation", &deg.x, -180.0F, 180.0F, "%.0f deg", 0,
                                      "Turns the environment about the Y axis: the HDR image, the sky, and the "
                                      "sun with it. At 0 the compass the Time of Day places the sun with has north "
                                      "at -Z, east +X, south +Z, west -X -- north is the way a glTF camera faces.");
  PE::end();

  if(!edited)
    return false;

  const glm::vec3 rad     = glm::radians(deg);
  const glm::quat rotated = glm::quat_cast(glm::eulerAngleYXZ(rad.x, rad.y, rad.z));
  settings.envRotation    = {rotated.x, rotated.y, rotated.z, rotated.w};
  m_rotationEuler.quat    = rotated;
  m_actions.environmentRotated();
  return true;
}

//--------------------------------------------------------------------------------------------------
// HDR image: load / intensity / blur. Its rotation is the environment's, above the per-type section.
//
bool UiEnvironment::hdrSection()
{
  bool changed = false;
  if(!PE::begin("HDR"))
    return changed;

  if(PE::entry("", [&] { return ImGui::SmallButton("load"); }, "Load HDR Image"))
  {
    m_actions.loadHdrFileDialog();
    changed = true;
  }
  changed |= PE::SliderFloat("Intensity", &m_resources->settings.hdrEnvIntensity, 0, 100, "%.3f",
                             ImGuiSliderFlags_Logarithmic, "HDR intensity");
  changed |= PE::SliderFloat("Blur", &m_resources->settings.hdrBlur, 0, 1, "%.3f", 0, "Blur the environment");
  PE::end();
  return changed;
}

//--------------------------------------------------------------------------------------------------
// Plain (authored solid-color) sky: one color, which both lights and backgrounds the scene.
//
bool UiEnvironment::plainSkySection()
{
  bool changed = false;
  if(!PE::begin("Plain Sky"))
    return changed;

  if(uicolor::colorEdit3Linear("Color", glm::value_ptr(m_resources->settings.plainColor),
                               "Solid sky color; lights the scene and shows as the background."))
  {
    m_actions.requestPreview();
    changed = true;
  }
  PE::end();
  return changed;
}

//--------------------------------------------------------------------------------------------------
// Gradient sky: bottom/horizon/top colors plus curves, and a drawn sun disk + glow.
//
bool UiEnvironment::gradientSkySection()
{
  if(!PE::begin("Gradient Sky"))
    return false;

  bool skyChanged = false;
  skyChanged |= uicolor::colorEdit3Linear("Top", glm::value_ptr(m_resources->settings.gradientTopColor), "Zenith color.");
  skyChanged |= uicolor::colorEdit3Linear("Horizon", glm::value_ptr(m_resources->settings.gradientHorizonColor),
                                          "Horizon color, shared by the sky and ground halves.");
  skyChanged |= uicolor::colorEdit3Linear("Bottom", glm::value_ptr(m_resources->settings.gradientBottomColor), "Nadir color.");
  skyChanged |= PE::SliderFloat("Top Curve", &m_resources->settings.gradientTopCurve, 0.0F, 1.0F, "%.3f", 0,
                                "Horizon-to-zenith falloff. Small values push the zenith color down toward the "
                                "horizon; 1.0 is a linear ramp.");
  skyChanged |= PE::SliderFloat("Bottom Curve", &m_resources->settings.gradientBottomCurve, 0.0F, 1.0F, "%.3f", 0,
                                "Horizon-to-nadir falloff. Same shape as Top Curve, mirrored below the horizon.");
  // Where the sun is lives in the shared section below; these two describe what its glow looks
  // like in this particular painted sky.
  skyChanged |= uicolor::colorEdit3Linear("Sun Color", glm::value_ptr(m_resources->settings.gradientSunColor),
                                          "Color of the sun disk and its glow, and the color and brightness of "
                                          "the light it casts -- this sky's sun is drawn rather than measured, so "
                                          "its illuminance is scaled from this against the sky around it.");
  skyChanged |= PE::SliderFloat("Sun Angle Max", &m_resources->settings.gradientSunAngleMax, 0.0F, 3.1416F, "%.3f rad",
                                0, "Angular extent of the sun glow.");
  skyChanged |= PE::SliderFloat("Sun Curve", &m_resources->settings.gradientSunCurve, 0.0F, 1.0F, "%.3f", 0,
                                "Falloff from the sun color to the sky color across the glow.");
  if(skyChanged)
    m_actions.requestPreview();
  PE::end();
  return skyChanged;
}

//--------------------------------------------------------------------------------------------------
// Physical Sky: preset picker + one collapsible group per physical concept, following the Unreal
// Engine sky component's category-per-component layout. Each atmospheric component (Rayleigh, Mie,
// ozone) owns its scattering strength, tint, and profile shape in one place -- the scale height
// is not more "advanced" than the strength, it just answers a different question about the same
// substance. Planet and Sun sit as their own groups because they describe the world and the star
// rather than the air. Aerial Perspective is one flat row: no component, no group.
//
// atmoChanged accumulates edits that need a scattering-table re-bake; the return value covers all
// edits (Aerial Perspective only restarts accumulation, which is why it's a distinct flag).
//
bool UiEnvironment::physicalSkySection()
{
  if(!PE::begin("Physical Sky"))
    return false;

  bool atmoChanged = false;
  bool anyChanged  = false;

  // Reorder freely: each helper is self-contained and owns its own tree/rows. Order below is:
  // preset (whole atmosphere in one click), the three atmospheric components in the order light
  // meets them (Rayleigh, Mie, ozone), the scene-scale bridge (Aerial Perspective), then the
  // world-shape facts (Planet, Sun).
  anyChanged |= atmospherePresetPicker(atmoChanged);
  anyChanged |= atmosphereRayleigh(atmoChanged);
  anyChanged |= atmosphereMie(atmoChanged);
  anyChanged |= atmosphereOzone(atmoChanged);
  anyChanged |= atmosphereAerialPerspective();
  anyChanged |= atmospherePlanet(atmoChanged);
  anyChanged |= atmosphereSun(atmoChanged);

  if(atmoChanged)
  {
    // Ground albedo, sun angular radius and observer altitude are bake inputs and show
    // immediately; everything else needs the scattering tables, which the drag rebuilds at
    // preview quality and the commit at full.
    m_actions.requestPreview();
    anyChanged = true;
  }
  PE::end();
  return anyChanged;
}

//--------------------------------------------------------------------------------------------------
// Preset combo: applies a whole atmosphere at once. The combo reads its answer back out of the
// values rather than remembering what was last picked, so editing any slider below drops it to
// Custom, and a scene that happens to carry Earth's atmosphere shows as Earth.
//
bool UiEnvironment::atmospherePresetPicker(bool& atmoChanged)
{
  static const char* kPresetNames[] = {"Earth", "Mars", "Alien", "Custom"};
  Settings&          st             = m_resources->settings;
  const int          match          = matchingAtmospherePreset(st);
  int                shown          = (match < 0) ? eAtmospherePresetCount : match;
  const bool         picked         = PE::Combo("Preset", &shown, kPresetNames, IM_ARRAYSIZE(kPresetNames), -1,
                                                "A whole atmosphere at once. Mars and Alien are plausible, not authoritative -- "
                                                                "somewhere to start, not numbers to cite.")
                      && shown < eAtmospherePresetCount;
  if(picked)
  {
    applyAtmospherePreset(st, shown);
    atmoChanged = true;
  }
  return picked;
}

//--------------------------------------------------------------------------------------------------
// Common flags for atmosphere group headers. Closed by default so opening the tab shows a short
// menu of physical concepts (Rayleigh / Mie / Ozone / Planet / Sun) rather than every knob at
// once -- users open only the group they came to edit. ImGui remembers each tree's open state per
// session and writes it to imgui.ini at shutdown, so the second visit reopens what was open.
//
static constexpr ImGuiTreeNodeFlags kAtmoGroupFlags = ImGuiTreeNodeFlags_SpanFullWidth;

//--------------------------------------------------------------------------------------------------
// Rayleigh: scattering by the air molecules themselves. What makes a clear sky blue.
//
bool UiEnvironment::atmosphereRayleigh(bool& atmoChanged)
{
  if(!PE::treeNode("Rayleigh", kAtmoGroupFlags))
    return false;

  Settings& st  = m_resources->settings;
  bool      hit = false;
  hit |= scatteringRow("Scattering", st.atmoRayleighScattering, 0.5F,
                       "Scattering by the air molecules themselves, at surface altitude. This is what makes a "
                       "clear sky blue and a low sun red.");
  hit |= PE::SliderFloat("Scale Height", &st.atmoRayleighScaleHeight, 0.1F, 100.0F, "%.2f km", ImGuiSliderFlags_Logarithmic,
                         "Altitude at which the air's density falls to 1/e (~37%) of its surface value. Earth's "
                         "is 8 km.");
  PE::treePop();
  if(hit)
    atmoChanged = true;
  return hit;
}

//--------------------------------------------------------------------------------------------------
// Mie: scattering by aerosols -- haze, dust, smoke. What makes the day hazy and puts the glow
// around the sun. Anisotropy shapes the forward glow; albedo is how much the aerosol scatters vs
// absorbs.
//
bool UiEnvironment::atmosphereMie(bool& atmoChanged)
{
  if(!PE::treeNode("Mie", kAtmoGroupFlags))
    return false;

  Settings& st  = m_resources->settings;
  bool      hit = false;
  hit |= scatteringRow("Scattering", st.atmoMieScattering, 0.5F,
                       "Scattering by aerosols -- haze, dust, smoke -- at surface altitude. Raise it for a hazier "
                       "day; largely colourless on Earth, which is why heavy haze washes the sky towards white.");
  hit |= PE::SliderFloat("Anisotropy", &st.atmoMieAnisotropy, -0.99F, 0.99F, "%.3f", 0,
                         "How strongly aerosols scatter forward. Earth's 0.8 is what puts the bright halo around "
                         "the sun; 0 scatters evenly in every direction, negative back toward the light.");
  hit |= PE::SliderFloat("Albedo", &st.atmoMieAlbedo, 0.05F, 1.0F, "%.3f", 0,
                         "Single-scattering albedo -- the fraction of light the aerosol removes that gets "
                         "scattered rather than absorbed. 1 is clean haze; lower is sootier and darkens the sky.");
  hit |= PE::SliderFloat("Scale Height", &st.atmoMieScaleHeight, 0.1F, 100.0F, "%.2f km", ImGuiSliderFlags_Logarithmic,
                         "Altitude at which the aerosol density falls to 1/e (~37%) of its surface value. Earth's "
                         "1.2 km says haze hugs the ground far more closely than air does.");
  PE::treePop();
  if(hit)
    atmoChanged = true;
  return hit;
}

//--------------------------------------------------------------------------------------------------
// Ozone: a high layer that absorbs without scattering. Center and thickness describe the tent
// distribution -- where the layer peaks and how far it spreads.
//
bool UiEnvironment::atmosphereOzone(bool& atmoChanged)
{
  if(!PE::treeNode("Ozone", kAtmoGroupFlags))
    return false;

  Settings& st  = m_resources->settings;
  bool      hit = false;
  hit |= scatteringRow("Absorption", st.atmoOzoneExtinction, 0.1F,
                       "A high layer that absorbs without scattering -- ozone, on Earth -- and what keeps the "
                       "zenith blue at low sun instead of turning it grey. The /km value is the coefficient at "
                       "the peak of the layer; Center and Thickness shape the tent around it.");
  hit |= PE::SliderFloat("Center", &st.atmoOzoneCenter, 0.0F, 200.0F, "%.1f km", 0,
                         "Altitude of the tent's peak -- where the layer is densest. Earth's ozone peaks near "
                         "25 km.");
  hit |= PE::SliderFloat("Thickness", &st.atmoOzoneWidth, 0.1F, 400.0F, "%.1f km", ImGuiSliderFlags_Logarithmic,
                         "Base width of the tent. Density rises linearly from zero at the lower edge to the peak "
                         "at Center, then falls linearly back to zero at the upper edge. Earth's ozone spans "
                         "about 30 km (10 to 40 km).");
  PE::treePop();
  if(hit)
    atmoChanged = true;
  return hit;
}

//--------------------------------------------------------------------------------------------------
// Aerial Perspective: the one knob that operates on the finished scattering tables rather than on
// the atmosphere they describe. Its scale is read straight out of SceneFrameInfo by both shading
// paths, so it needs no re-bake and no LUT rebuild -- restarts accumulation only. That is why it
// returns anyChanged rather than setting atmoChanged.
//
bool UiEnvironment::atmosphereAerialPerspective()
{
  Settings& st = m_resources->settings;
  return PE::SliderFloat("Aerial Perspective", &st.atmoAerialPerspectiveScale, 0.0F, 1000.0F, "%.1f m/unit", ImGuiSliderFlags_Logarithmic,
                         "How many metres of air one scene unit is worth. 1.0 is glTF's own answer -- the format "
                         "says distances are metres -- and 0 switches the haze off.\n\n"
                         "It is a distance, not an opacity: raising it says the scene is bigger, which is the "
                         "honest way to exaggerate. Push it far enough and the scene sinks below the observer "
                         "altitude in Planet, where there is no air left to model.\n\n"
                         "Only a physical Sky has air. A gradient is a painting and an HDR already has whatever "
                         "haze the photographer stood in.");
}

//--------------------------------------------------------------------------------------------------
// Planet: how big the world is, what the ground reflects, how thick the air is around it, how
// high the observer stands. Ground Albedo lives here rather than as a peer of Rayleigh/Mie/Ozone
// because it is a property of the surface, not of a scattering layer.
//
bool UiEnvironment::atmospherePlanet(bool& atmoChanged)
{
  if(!PE::treeNode("Planet", kAtmoGroupFlags))
    return false;

  Settings& st  = m_resources->settings;
  bool      hit = false;
  hit |= PE::SliderFloat("Ground Radius", &st.atmoBottomRadius, 100.0F, 20000.0F, "%.0f km", ImGuiSliderFlags_Logarithmic,
                         "Distance from the planet centre to its ground level. Sets how fast the horizon curves "
                         "away, and with it how much air a grazing ray crosses. Earth's is 6360 km.");
  hit |= uicolor::colorEdit3Linear("Ground Albedo", glm::value_ptr(st.atmoGroundAlbedo),
                                   "What the ground reflects. Lights the lower half of the sky, and bounces "
                                   "back into the atmosphere through multiple scattering.");
  hit |= PE::SliderFloat("Atmosphere Thickness", &st.atmoThickness, 1.0F, 500.0F, "%.0f km", ImGuiSliderFlags_Logarithmic,
                         "Depth of the atmosphere above the surface. Everything above it is space.");
  hit |= PE::SliderFloat("Observer Altitude", &st.atmoObserverAltitude, 0.0F, 20000.0F, "%.0f m", ImGuiSliderFlags_Logarithmic,
                         "Height above the ground the sky is baked from. Fixed rather than following the camera "
                         "-- that is what makes a single baked image valid for the whole scene. Raise it and the "
                         "horizon haze thins out.");
  PE::treePop();
  if(hit)
    atmoChanged = true;
  return hit;
}

//--------------------------------------------------------------------------------------------------
// Sun: the star's own properties, not its position. Position lives in the Sun & Time tab because
// it's the same sun and only one place should aim it. These two knobs describe what the star
// *is*: the size of its disk in the sky and how much energy it delivers.
//
bool UiEnvironment::atmosphereSun(bool& atmoChanged)
{
  if(!PE::treeNode("Sun", kAtmoGroupFlags))
    return false;

  Settings& st  = m_resources->settings;
  bool      hit = false;
  hit |= PE::SliderFloat("Angular Radius", &st.atmoSunAngularRadius, 0.0F, 0.05F, "%.5f rad", 0,
                         "Angular radius of the sun disk. Earth's is 0.004675 rad (about half a degree); larger "
                         "values soften shadows the same way a bigger sun would.");
  hit |= PE::DragFloat3("Irradiance", glm::value_ptr(st.atmoSolarIrradiance), 0.01F, 0.0F, 100.0F, "%.3f W/m2", 0,
                        "What reaches the top of the atmosphere, per channel. Falls with the square of the "
                        "distance to the star, and its colour is the star's.");
  PE::treePop();
  if(hit)
    atmoChanged = true;
  return hit;
}

//--------------------------------------------------------------------------------------------------
// The "Save with scene" toggle and sky presets: apply to every authored sky type.
//
bool UiEnvironment::authoredSkyBakeSection()
{
  bool changed = false;
  if(!PE::begin("Authored Sky"))
    return changed;

  changed |= PE::Checkbox("Save with scene", &m_resources->settings.envSaveToGltf,
                          "Write OMI_environment_sky into the glTF when the scene is saved.");

  // Presets: the same sky, as a file of its own. One sky per scene stays the rule -- this is how
  // one gets to another scene, or to somebody else.
  PE::entry(
      "Preset",
      [&] {
        if(ImGui::SmallButton(ICON_MS_SAVE " Save..."))
          m_actions.saveSkyPresetDialog();
        ImGui::SameLine();
        if(ImGui::SmallButton(ICON_MS_FOLDER_OPEN " Load..."))
          m_actions.loadSkyPresetDialog();
        return false;
      },
      "Save this sky as a .sky.json, or apply one.\n\n"
      "The file is one entry of the OMI_environment_sky extension -- the same thing a scene carries, "
      "on its own -- plus the sun's angle, which the extension has no field for and which is half of "
      "any look worth keeping.\n\n"
      "A preset can also be dropped straight onto the viewport.");

  PE::end();
  return changed;
}

//--------------------------------------------------------------------------------------------------
// Split the two orthogonal groups of settings a sky-with-sun exposes -- the sun's position over
// time vs. what the sky itself looks like -- into two tabs. They share Resources::sunDirection but
// nothing else, so putting them side by side kept forcing users to scroll past the group they
// weren't editing. The type-specific label ("Physical Sky" / "Gradient Sky") reminds the user which
// look tab they're editing without a duplicate header inside.
//
bool UiEnvironment::skyWithSunTabs()
{
  bool changed = false;
  const char* lookTab = (m_resources->settings.envSystem == shaderio::EnvSystem::eSky) ? "Physical Sky" : "Gradient Sky";
  if(ImGui::BeginTabBar("EnvSkyTabs"))
  {
    if(ImGui::BeginTabItem("Sun & Time"))
    {
      changed |= sunAndTimeOfDay();
      ImGui::EndTabItem();
    }
    if(ImGui::BeginTabItem(lookTab))
    {
      if(m_resources->settings.envSystem == shaderio::EnvSystem::eSky)
        changed |= physicalSkySection();
      else
        changed |= gradientSkySection();
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
  return changed;
}

//--------------------------------------------------------------------------------------------------
// "Sun & Time of Day": one sun, and the ways to aim it, in the order most users reach for.
//
// Shared by the gradient and the physical sky rather than duplicated inside each, because there is
// only one sun -- Resources::sunDirection, or the scene's marked light when it has one -- and a
// second copy of these rows was a second chance for them to disagree.
//
// Simple first: the clock (which is what a user changes to "set the mood"), then Jump To as a
// shortcut *to* Time. Where north is has no row here: it is the environment's Rotation, above the
// tabs, because the compass turns with the sky rather than apart from it. The astronomy that
// explains those knobs -- which light is the sun, the exact angles, the gizmo handle
// for that light, the place and date the clock is measured against -- lives under Advanced.
//
bool UiEnvironment::sunAndTimeOfDay()
{
  if(!PE::begin("Sun & Time of Day"))
    return false;

  // Solve the day once and pass the result to every helper that needs it, so the clock ticks and
  // the preset menu read from the same solution -- see the note above solveDay().
  const TimeOfDayEvents events = solveDay(m_resources->settings);

  bool changed = false;
  // Time first: it is the value the rest of this section describes. Jump To sits below because it
  // *sets* Time from a named moment.
  changed |= sunTimeSlider(events);
  changed |= sunJumpToPresets(events);
  changed |= sunLocationAndDate(events);
  changed |= sunAdvancedFold(events);

  PE::end();
  return changed;
}

//--------------------------------------------------------------------------------------------------
// The clock. Dragging it is cheap for every sky type: the authored skies keep the sun out of their
// baked lighting field entirely, and the physical one re-bakes at preview quality until the mouse
// comes up. The four ticks under the slider are today's solar midnight, sunrise, noon and sunset,
// so a reading of 06:00 has something to mean against.
//
bool UiEnvironment::sunTimeSlider(const TimeOfDayEvents& events)
{
  Settings& st        = m_resources->settings;
  bool      changed   = false;
  char      clock[16] = {};
  formatClock(clock, sizeof(clock), st.todHour);
  if(PE::SliderFloat("Time", &st.todHour, 0.0F, 24.0F, clock, 0,
                     "Local clock time at the UTC offset under Advanced. The sun follows it through the NOAA "
                     "solar position algorithm, so this is where the sun really was."))
  {
    m_actions.applyTimeOfDay();
    changed = true;
  }
  drawDayEventTicks(events);
  return changed;
}

//--------------------------------------------------------------------------------------------------
// Named moments. Solved, not tabulated -- see the note above solveDay(). An event that does not
// happen that day is offered greyed out rather than silently snapping the sun somewhere plausible.
//
bool UiEnvironment::sunJumpToPresets(const TimeOfDayEvents& events)
{
  Settings&  st     = m_resources->settings;
  const bool picked = PE::entry(
      "Jump To",
      [&] {
        bool hit = false;
        if(ImGui::BeginCombo("##TodPreset", "Choose a moment"))
        {
          for(const TimeOfDayPreset& preset : kTimeOfDayPresets)
          {
            const float hour = events.valid ? hourOfEvent(events, preset.event) : std::numeric_limits<float>::quiet_NaN();
            const bool real = std::isfinite(hour);

            char label[64] = {};
            if(real)
            {
              char clock[16] = {};
              formatClock(clock, sizeof(clock), hour);
              snprintf(label, sizeof(label), "%s  %s", preset.name, clock);
            }
            else
            {
              snprintf(label, sizeof(label), "%s  (not today, here)", preset.name);
            }

            ImGui::BeginDisabled(!real);
            if(ImGui::Selectable(label) && real)
            {
              st.todHour = hour;
              hit        = true;
            }
            ImGui::EndDisabled();
          }
          ImGui::EndCombo();
        }
        return hit;
      },
      "Named moments, solved for the place and date under Advanced rather than fixed clock times -- sunset is "
      "not at 19:30 for most of the year at most latitudes. Greyed out where the sun never reaches that "
      "altitude on that day: inside the polar circles, most of them do not happen.");
  if(picked)
    m_actions.applyTimeOfDay();
  return picked;
}

//--------------------------------------------------------------------------------------------------
// Azimuth and elevation, degrees. Folded under Advanced because Time and Location cover the common
// case now; angles are the escape hatch for typing an exact number, and they are what the command
// line carries and what a marked light stores.
//
bool UiEnvironment::sunAngleSliders()
{
  glm::vec3 editedSun = m_resources->sunDirection;
  if(nvgui::azimuthElevationSliders(editedSun, false, m_resources->sunYIsUp))
  {
    m_actions.aimSun(editedSun);
    return true;
  }
  return false;
}


//--------------------------------------------------------------------------------------------------
// Advanced: the widgets most sessions do not need. Sun Source (which light is the sun) and Location
// & Date (where and when) are workflow choices, not per-frame edits; Azimuth/Elevation are the
// escape hatch for typing an exact angle when Time and Location are not enough; the Gizmo button is
// an action that only reaches a marked light.
//
bool UiEnvironment::sunAdvancedFold(const TimeOfDayEvents& events)
{
  if(!PE::treeNode("Advanced"))
    return false;

  bool changed = false;

  // Which light is the sun, and the chance to say so. Without this the only sun that can exist is
  // one a save materialised: a directional light the user placed could never become the sky's, and
  // nothing explained why it was being ignored.
  sunSourcePicker();

  changed |= sunAngleSliders();
  changed |= sunGizmoButton();

  PE::treePop();
  return changed;
}

//--------------------------------------------------------------------------------------------------
// The gizmo is the scene's own transform gizmo, on the light that is the sun -- not a second one.
// With no marked light there is no node to attach it to, which is exactly what the disabled state
// says.
//
bool UiEnvironment::sunGizmoButton()
{
  Settings&  st         = m_resources->settings;
  const bool hasSunNode = m_actions.sunLightNode() >= 0;
  bool       changed    = false;
  ImGui::BeginDisabled(!hasSunNode);
  if(PE::entry(
         "", [&] { return ImGui::SmallButton(ICON_MS_3D_ROTATION " Gizmo on sun light"); },
         hasSunNode ? "Select the sun's light and switch the transform gizmo on, so it can be aimed from the "
                      "viewport." :
                      "No light in this scene is the sun, so there is no node for a gizmo to hold. Pick one in Sun "
                      "Source above."))
  {
    m_selection->selectNode(m_actions.sunLightNode());
    st.showGizmo = true;
    changed      = true;
  }
  ImGui::EndDisabled();
  return changed;
}

//--------------------------------------------------------------------------------------------------
// Where and when the sky is being calculated for: city / lat / long / date / UTC offset, plus a
// "Now" button that fills the date, clock and offset from this machine. The city list is *derived*
// from the coordinates rather than remembered, so dragging either slider drops it to Custom and a
// scene set up at Tokyo's latitude reads as Tokyo -- the same rule the atmosphere presets follow.
//
bool UiEnvironment::sunLocationAndDate(const TimeOfDayEvents& events)
{
  if(!PE::treeNode("Location & Date"))
    return false;

  Settings& st    = m_resources->settings;
  bool      moved = false;

  // "Now" first: it fills the date, clock and UTC offset in one click, so it belongs at the top as
  // the "set this whole section from my machine" action rather than sitting mid-column between
  // fields it might or might not be seen to touch. The place is left alone -- it is the one thing
  // the computer cannot tell us.
  if(PE::entry(
         "", [&] { return ImGui::SmallButton(ICON_MS_SCHEDULE " Now"); },
         "Set the date, the clock and the UTC offset from this machine. The place is left alone -- it is the "
         "one thing the computer cannot tell us."))
  {
    if(const sun_position::LocalClock now = sun_position::systemLocalClock(); now.valid)
    {
      st.todDate      = now.date;
      st.todHour      = now.hour;
      st.todUtcOffset = now.utcOffsetHours;
      moved           = true;
    }
  }

  {
    const int current = sun_position::cityAt(st.todLatitude, st.todLongitude);
    if(PE::entry(
           "City",
           [&] {
             bool picked = false;
             if(ImGui::BeginCombo("##TodCity", current >= 0 ? sun_position::kCities[current].name : "Custom"))
             {
               for(int i = 0; i < sun_position::kCityCount; ++i)
               {
                 const sun_position::City& city      = sun_position::kCities[i];
                 char                      label[64] = {};
                 snprintf(label, sizeof(label), "%-16s %5.1f%c", city.name, std::fabs(city.latitudeDeg),
                          city.latitudeDeg >= 0.0F ? 'N' : 'S');
                 if(ImGui::Selectable(label, i == current))
                 {
                   st.todLatitude  = city.latitudeDeg;
                   st.todLongitude = city.longitudeDeg;
                   st.todUtcOffset = city.utcOffsetHours;
                   picked          = true;
                 }
               }
               ImGui::EndCombo();
             }
             return picked;
           },
           "A few places, north to south -- the order the sun cares about. Each sets the latitude, the "
           "longitude and the UTC offset below.\n\n"
           "The offset is standard time: no renderer here carries a time-zone database, so a summer date at "
           "most of these is an hour out until you nudge UTC Offset."))
    {
      moved = true;
    }
  }

  // Latitude/Longitude flat, not nested. Two sliders don't earn another fold -- and if you have
  // opened Location & Date you almost always want to see them.
  moved |= PE::SliderFloat("Latitude", &st.todLatitude, -90.0F, 90.0F, "%.2f deg", 0,
                           "Degrees north of the equator, negative south. This is what sets how high the sun "
                           "gets and how steeply it crosses the horizon.");
  moved |= PE::SliderFloat("Longitude", &st.todLongitude, -180.0F, 180.0F, "%.2f deg", 0,
                           "Degrees east of Greenwich. With the UTC offset it decides how far the clock runs "
                           "from the sun: a city at the edge of its time zone can be most of an hour out.");

  char dateBuffer[16] = {};
  snprintf(dateBuffer, sizeof(dateBuffer), "%s", st.todDate.c_str());
  if(PE::InputText("Date", dateBuffer, sizeof(dateBuffer), ImGuiInputTextFlags_CharsNoBlank,
                   "Date as yyyy-mm-dd. The season is the other half of the sun's height: the same clock time "
                   "in June and in December is two very different skies."))
  {
    st.todDate = dateBuffer;
    moved      = true;
  }
  if(!events.valid)
  {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0F, 0.6F, 0.2F, 1.0F), ICON_MS_WARNING);
    nvgui::tooltip("Not a yyyy-mm-dd date, so the sun is staying where it is.");
  }

  moved |= PE::SliderFloat("UTC Offset", &st.todUtcOffset, -12.0F, 14.0F, "%+.1f h", 0,
                           "Hours the local clock runs ahead of UTC, daylight saving included. A raw number "
                           "rather than a named time zone: no zone database ships with this renderer, and "
                           "nothing here needs one.");

  PE::treePop();

  if(moved)
    m_actions.applyTimeOfDay();
  return moved;
}

//--------------------------------------------------------------------------------------------------
// "Sun source" row: which light the sky's sun is, and how to change it.
//
// The sun is chosen, never guessed. A scene may hold several directional lights and the
// specification does not say any of them is a sun, so the renderer supplies its own until someone
// points at one -- and this is where they point. Picking one marks it (undoably); the marker is
// what a save writes out and a reload reads back.
//
void UiEnvironment::sunSourcePicker()
{
  namespace PE = nvgui::PropertyEditor;

  const std::vector<std::pair<int, std::string>> candidates = m_actions.directionalLightNodes();

  PE::entry(
      "Sun Source",
      [&] {
        bool        changed = false;
        const char* current = "Renderer";
        for(const auto& [nodeIdx, name] : candidates)
        {
          if(nodeIdx == m_actions.sunLightNode())
            current = name.c_str();
        }

        if(ImGui::BeginCombo("##SunSource", current))
        {
          if(ImGui::Selectable("Renderer", m_actions.sunLightNode() < 0))
          {
            m_actions.setSkySunNode(-1);
            changed = true;
          }
          for(const auto& [nodeIdx, name] : candidates)
          {
            if(ImGui::Selectable(name.c_str(), nodeIdx == m_actions.sunLightNode()))
            {
              m_actions.setSkySunNode(nodeIdx);
              changed = true;
            }
          }
          ImGui::EndCombo();
        }
        return changed;
      },
      "Which light is this sky's sun.\n\n"
      "\"Renderer\" means the sky carries its own: nothing is added to your scene, and saving the "
      "sky writes one out so the file is complete for other renderers.\n\n"
      "Choosing one of the scene's directional lights makes that light the sun -- it drives the "
      "sky, takes the atmosphere's brightness and angular size, and travels with the file. Every "
      "other light is left exactly as authored.");

  // One directional light and nobody has said whether it is the sun: offer, rather than guess.
  //
  // Guessing is what "the first directional light is the sun" does, and it is unstable -- add a
  // fill light and the sky can swing to it. But saying nothing is its own failure: the light sits
  // there being an ordinary light while the sky carries a sun of its own, and from the outside
  // that reads as the renderer ignoring it. An offer is stable *and* visible, and one click
  // records the answer in the file.
  //
  // Only with exactly one candidate. With several there is a real question, and the combo above
  // is where it gets asked.
  if(candidates.size() == 1 && m_actions.sunLightNode() < 0)
  {
    const auto& [nodeIdx, name] = candidates.front();
    const std::string label     = "Use '" + name + "' as the sun";
    if(PE::entry(
           "", [&] { return ImGui::SmallButton(label.c_str()); },
           "This scene has one directional light, and nothing yet says whether it is this sky's "
           "sun.\n\n"
           "Until you choose, it stays an ordinary light and the sky lights the scene with a sun of its "
           "own -- so the two are independent, and saving the sky would write a second light beside "
           "yours.\n\n"
           "Adopting it makes that light the sun: it aims the sky, takes the atmosphere's brightness and "
           "angular size, and is the light a save updates."))
    {
      m_actions.setSkySunNode(nodeIdx);
    }
  }

  if(candidates.empty())
  {
    ImGui::SameLine();
    nvgui::tooltip("The scene has no directional light to choose from.");
  }
}

//--------------------------------------------------------------------------------------------------
// Restore the selected environment type's settings to their defaults.
//
// Scoped to the active type on purpose. "Reset All to Default" already exists for the whole
// application; what is missing is a way to abandon an experiment with one sky without losing the
// rest of the session -- the HDR you loaded, the background color.
//
// Defaults come from a default-constructed Settings, which is the same
// in-class initializer SettingsRegistry captured at declaration time. Reading them from the type
// rather than listing values here keeps this from becoming a second copy of the defaults that
// drifts the moment someone edits the struct.
//
void UiEnvironment::resetDefaults()
{
  const Settings defaults{};
  Settings&      settings = m_resources->settings;

  // The orientation is shared by every type that shows it, so it resets with any of them. First, so
  // that a sun reset below lands on its default angles instead of being turned after them.
  if(environmentTurns(settings.envSystem))
  {
    settings.envRotation = defaults.envRotation;
    m_actions.environmentRotated();
  }

  switch(settings.envSystem)
  {
    case shaderio::EnvSystem::ePlain:
      settings.plainColor = defaults.plainColor;
      break;

    case shaderio::EnvSystem::eGradient:
      settings.gradientBottomColor  = defaults.gradientBottomColor;
      settings.gradientHorizonColor = defaults.gradientHorizonColor;
      settings.gradientTopColor     = defaults.gradientTopColor;
      settings.gradientSunColor     = defaults.gradientSunColor;
      settings.gradientBottomCurve  = defaults.gradientBottomCurve;
      settings.gradientTopCurve     = defaults.gradientTopCurve;
      settings.gradientSunAngleMax  = defaults.gradientSunAngleMax;
      settings.gradientSunCurve     = defaults.gradientSunCurve;
      // The gradient panel aims the sun, so reset puts it back too. One sun is shared by every
      // sky type, so this moves the physical sky's sun as well -- which is the same coupling the
      // sliders already have, not a new one.
      settings.sunAzimuth   = defaults.sunAzimuth;
      settings.sunElevation = defaults.sunElevation;
      m_actions.setSunAngles(settings.sunAzimuth, settings.sunElevation);
      break;

    case shaderio::EnvSystem::eSky:
      settings.sunAzimuth   = defaults.sunAzimuth;
      settings.sunElevation = defaults.sunElevation;
      m_actions.setSunAngles(settings.sunAzimuth, settings.sunElevation);
      // The whole atmosphere goes back to Earth, including anything a loaded scene authored:
      // reset means "the default sky", and leaving a scene's atmosphere behind would make the
      // button lie. Applying the preset rather than copying fields one by one is what keeps this
      // from drifting the next time the atmosphere grows a parameter.
      applyAtmospherePreset(settings, eAtmosphereEarth);
      // Not part of any preset -- where you stand is a viewer preference, not a property of the
      // world you are standing on.
      settings.atmoObserverAltitude = defaults.atmoObserverAltitude;
      break;

    case shaderio::EnvSystem::eHdr:
      // Intensity and blur, and the rotation above. The loaded file is not a setting to restore:
      // dropping the user's environment image on a button labelled "reset" would be a surprise.
      settings.hdrEnvIntensity = defaults.hdrEnvIntensity;
      settings.hdrBlur         = defaults.hdrBlur;
      break;

    case shaderio::EnvSystem::eNone:
      break;  // nothing to restore; the button is disabled for this case
  }

  // Authored skies are baked, so the new values have to reach the lat-long image and the alias
  // table. onEnvironmentChanged() defers that to the top of the next frame, where a submit is safe.
  m_actions.environmentChanged();
}
