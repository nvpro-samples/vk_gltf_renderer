/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//////////////////////////////////////////////////////////////////////////
/*
    GLTF Renderer with Ray Tracing and Rasterization Support

    This renderer demonstrates advanced real-time rendering of GLTF scenes 
    using both ray tracing and rasterization pipelines. Key features include:
    
    - Dual rendering modes: path tracing and traditional rasterization
    - PBR (Physically Based Rendering) material system
    - HDR environment mapping with prefiltered importance sampling
    - Procedural sky simulation
    - Animation support with skeletal and keyframe animations
    - Progressive rendering for path tracing
    - GLTF 2.0 specification compliance with extensions
    - Interactive ray picking for scene manipulation
    - UI-driven scene editing capabilities
    
    The implementation uses Vulkan with ray tracing extensions and
    employs a modular architecture to handle the full rendering pipeline
    from scene loading to final display, with careful memory management
    and asynchronous command processing for optimal performance.
*/
//////////////////////////////////////////////////////////////////////////

#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }
#define IMGUI_DEFINE_MATH_OPERATORS

#include <glm/gtx/quaternion.hpp>  // glm::rotation
#include <cmath>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <fstream>
#include <iterator>
#include <span>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vulkan/vulkan_core.h>
#include <stb/stb_image.h>

#include "GLFW/glfw3.h"
#undef APIENTRY

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// Shader Input/Output
#include "shaders/shaderio.h"  // Shared between host and device

// Pre-compiled shaders
#include "_autogen/tonemapper.slang.h"
#include "_autogen/hdr_dome.slang.h"
#include "_autogen/hdr_integrate_brdf.slang.h"
#include "_autogen/hdr_prefilter_diffuse.slang.h"
#include "_autogen/hdr_prefilter_glossy.slang.h"
#include "_autogen/env_write_pdf.slang.h"
#include "_autogen/sky_bruneton_precompute.slang.h"

//
#include <backends/imgui_impl_vulkan.h>
#include <nvaftermath/aftermath.hpp>
#include <nvutils/profiler.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/mipmaps.hpp>
#include "gltf_camera_utils.hpp"
#include "gltf_scene_editor.hpp"
#include "gltf_scene_animation.hpp"
#include "gltf_scene_transform_vk.hpp"

#include <glm/gtc/quaternion.hpp>

#include "renderer.hpp"
#include "env_image_loader.hpp"
#include "gltf_environment_sky.hpp"
#include "sun_position.hpp"
#include "sky_preset.hpp"
#include "scene_descriptor.hpp"
#include "staging_config.hpp"
#include "tinygltf_utils.hpp"
#include "utils.hpp"
#include "tinyobjloader/tiny_obj_loader.h"
#include "tinygltf_converter.hpp"

extern nvutils::ProfilerManager g_profilerManager;  // #PROFILER

namespace {
// Background clear color used when no scene is loaded or to show DLSS render resolution borders
constexpr VkClearColorValue kBackgroundClearColor = {{0.17f, 0.21f, 0.25f, 1.f}};

// Angular radius of the disk the gradient sky draws. The extension has no field for it, so this is
// the sun's true radius as seen from Earth (~0.27 deg) -- enough for the disk to read as a sun.
// Authored decoration rather than a measurement, unlike the physical sky's, which comes from the
// atmosphere. One constant because three places need the same number: the drawn disk, the light's
// angular size, and what a save writes.
constexpr float kGradientSunAngularRadius = 0.00465f;

// How much brighter the gradient sky's sun is than the dome it sits under, as illuminance on a
// surface facing each.
//
// A number this file has to invent, because the gradient sky never measured anything: its colours
// are authored in arbitrary linear units where the dome sits near 1, and nothing in them says how
// bright a sun should be. So it is fitted to the real thing instead, against the clear-day
// direct:diffuse band a measured clear day gives -- which makes it checkable rather than merely
// plausible.
//
// Fitted across the whole band rather than at one sun angle, because this sky cannot follow its
// shape. A real sky's diffuse half dims as the sun drops, so direct:diffuse climbs faster than the
// sun's own cosine; the gradient's dome does not move with the sun at all (its bake excludes the
// sun entirely), so the modelled ratio is exactly proportional to sin(elevation). That leaves one
// free scale, and only 5.2-8.0 keeps every reference elevation from 15 to 80 degrees inside the
// band. 6.5 sits in the middle of that window: measured 1.8 / 3.4 / 4.8 / 5.8 / 6.6 against
// 1.0-2.2 / 2.2-4.0 / 3.5-6.5 / 4.5-8.0 / 5.0-9.0.
constexpr float kGradientSunToSkyRatio = 6.5f;

// Host-side twin of skyGradientBandWeight() in shaders/sky_gradient.h.slang. The two must agree:
// the sun's illuminance is scaled against the dome's, and the dome is this curve.
float gradientBandWeight(float c, float curve)
{
  const float safeCurve = std::max(curve, 1e-3f);
  return glm::clamp(1.0f - std::pow(1.0f - glm::clamp(c, 0.0f, 1.0f), 1.0f / safeCurve), 0.0f, 1.0f);
}

}  // namespace

namespace {

uint64_t sumTrackerCurrentBytes(const nvvkgltf::GpuMemoryTracker& tracker)
{
  return tracker.getTotalStats().currentBytes;
}

uint64_t sumTrackerPeakBytes(const nvvkgltf::GpuMemoryTracker& tracker)
{
  return tracker.getTotalStats().peakBytes;
}

}  // namespace

// The constructor registers the parameters that can be set from the command line
GltfRenderer::GltfRenderer(nvutils::ParameterRegistry* paramReg, const nvutils::ParameterParser* paramParser, BenchmarkOptions& benchmarkOptions)
    : m_parameterParser(paramParser)
    , m_benchmark(benchmarkOptions)
    , m_settings(paramReg)
{
  // Every user-settable value is declared once here (see settings_registry.hpp): the command line,
  // benchmark sequences, MCP, and ImGui.ini all follow from this one call. Persist says whether the
  // value is remembered between runs.
  m_settings.add({.name            = "envSystem",
                  .help            = "Environment: [Sky:0, HDR:1, None:2, Plain:3, Gradient:4]",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { onEnvironmentChanged(); }},
                 (int*)&m_resources.settings.envSystem, Persist::eYes, 0, 4);
  m_settings.add({.name            = "renderSystem",
                  .help            = "Renderer [Path tracer:0, Rasterizer:1]",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { onRenderSystemChanged(); }},
                 (int*)&m_resources.settings.renderSystem, Persist::eYes, 0, 1);
  m_settings.add({"uiShowAxis", "Show Axis"}, &m_resources.settings.showAxis, Persist::eYes);
  m_settings.add({"uiShowMemStats", "Show Memory Statistics"}, &m_resources.settings.showMemStats, Persist::eYes);
  m_settings.add({"hdrIntensity", "HDR Environment Intensity"}, &m_resources.settings.hdrEnvIntensity, Persist::eYes, 0.0F, 100.0F);
  // One orientation for every environment -- OMI_environment_sky's `rotation`, in its own form. It
  // is also where north is, so like the Time of Day settings, an edit from outside the UI re-places
  // the sun after the ini restore (see m_todDrivesSun).
  m_settings.addVector({.name = "envRotation",
                        .help = "Environment rotation: unit quaternion x y z w (glTF order); turns the HDR, "
                                "every sky, and the compass the sun is placed with",
                        .callbackSuccess =
                            [this](const nvutils::ParameterBase* const) {
                              if(skyHasSun(m_resources.settings.envSystem))
                                m_todDrivesSun = true;
                              onEnvironmentRotated();
                            }},
                       &m_resources.settings.envRotation, Persist::eYes);
  m_settings.add({"hdrBlur", "HDR Environment Blur"}, &m_resources.settings.hdrBlur, Persist::eYes, 0.0F, 1.0F);
  m_settings.addVector({"silhouetteColor", "Color of the silhouette"}, &m_resources.settings.silhouetteColor, Persist::eYes);
  m_settings.add({"dbgVisualization", "Visualization Mode"}, (int*)&m_resources.settings.visualization, Persist::eYes);
  m_settings.add({"dbgWireframe", "Enable wireframe overlay"}, &m_resources.settings.wireframe, Persist::eYes);
  m_settings.add({"ptOptimalShader",
                  "Compile gltf_pathtrace.slang with GLTF_USE_* gates specialized per scene "
                  "(no runtime MAT_EXT_* changes; triggers shader recompile on scene/material change). Default off."},
                 &m_resources.settings.optimalShader, Persist::eYes);
  m_settings.add({"useSolidBackground", "Use solid color background"}, &m_resources.settings.useSolidBackground, Persist::eYes, true);
  m_settings.addVector({"solidBackgroundColor", "Solid Background Color"}, &m_resources.settings.solidBackgroundColor, Persist::eYes);
  m_settings.add({"ptMaxFrames", "Maximum number of iterations"}, &m_resources.settings.maxFrames, Persist::eYes);
  // Not persisted: a scripted or headless run must not change what the next interactive session plays.
  m_settings.add(
      {.name = "animPlay",
       .help = "Play animations [Off:0, On:1]. Default: on, off in headless runs. A scene with a behavior graph never autoplays",
       .callbackSuccess = [this](const nvutils::ParameterBase* const) { m_resources.animationControl.play = m_animPlay; }},
      &m_animPlay, Persist::eNo);
  // Headless-only output path: consumed at start-up, so remembering it would be misleading.
  m_settings.add({"output", "Output image file path for headless mode"}, &m_resources.headlessOutputPath, Persist::eNo);

  // Tonemapper. Ranges match the UI sliders in nvgui/tonemapper.cpp, so a script or an agent is
  // bounded the same way the user is. tmBrightness and tmVignette were once tmGamma and
  // tmWhitePoint -- names for fields TonemapperData does not have.
  m_settings.add({"tmMethod", "Tonemapper: Method [Filmic:0, Uncharted:1, Clip:2, ACES:3, AgX:4, KhronosPBR:5]"},
                 &m_resources.tonemapperData.method, Persist::eYes, 0, 5);
  m_settings.add({"tmActive", "Tonemapper: Enable tone mapping [Off:0, On:1]"}, &m_resources.tonemapperData.isActive,
                 Persist::eYes, 0, 1);
  // Floor below the UI slider's 0.1, which is not enough headroom for a bright environment with
  // auto-exposure off -- what a reproducible capture wants. Out-of-range values are clamped rather
  // than refused, so too high a floor is silent: the run succeeds and the image is simply not the
  // one that was asked for.
  m_settings.add({"tmExposure", "Tonemapper: Exposure multiplier"}, &m_resources.tonemapperData.exposure, Persist::eYes,
                 0.01F, 200.0F);
  m_settings.add({"tmContrast", "Tonemapper: Contrast"}, &m_resources.tonemapperData.contrast, Persist::eYes, 0.0F, 2.0F);
  m_settings.add({"tmBrightness", "Tonemapper: Brightness"}, &m_resources.tonemapperData.brightness, Persist::eYes, 0.0F, 2.0F);
  m_settings.add({"tmSaturation", "Tonemapper: Saturation"}, &m_resources.tonemapperData.saturation, Persist::eYes, 0.0F, 2.0F);
  m_settings.add({"tmVignette", "Tonemapper: Vignette (0 = none)"}, &m_resources.tonemapperData.vignette, Persist::eYes, -1.0F, 1.0F);
  m_settings.add({"tmDither", "Tonemapper: Dither [Off:0, On:1]"}, &m_resources.tonemapperData.dither, Persist::eYes, 0, 1);
  // White balance
  m_settings.add({"tmTemperature", "Tonemapper: White balance temperature (Kelvin)"},
                 &m_resources.tonemapperData.temperature, Persist::eYes, 2000.0F, 15000.0F);
  m_settings.add({"tmTint", "Tonemapper: White balance tint (ANSI C78.377 Duv)"}, &m_resources.tonemapperData.tint,
                 Persist::eYes, -0.03F, 0.03F);
  // Colour grading
  m_settings.add({"tmVibrance", "Tonemapper: Vibrance (boosts muted colors only)"},
                 &m_resources.tonemapperData.vibrance, Persist::eYes, -1.0F, 1.0F);
  m_settings.add({"tmShadowBias", "Tonemapper: Shadow tone bias"}, &m_resources.tonemapperData.shadowBias,
                 Persist::eYes, -1.0F, 1.0F);
  m_settings.add({"tmMidtoneBias", "Tonemapper: Midtone brightness bias"}, &m_resources.tonemapperData.midtoneBias,
                 Persist::eYes, -1.0F, 1.0F);
  m_settings.add({"tmHighlightBias", "Tonemapper: Highlight tone bias"}, &m_resources.tonemapperData.highlightBias,
                 Persist::eYes, -1.0F, 1.0F);
  m_settings.addVector({"tmCoolColor", "Tonemapper: Split-toning tint for shadows"},
                       &m_resources.tonemapperData.coolColor, Persist::eYes);
  m_settings.addVector({"tmWarmColor", "Tonemapper: Split-toning tint for highlights"},
                       &m_resources.tonemapperData.warmColor, Persist::eYes);
  m_settings.add({"tmSplitBalance", "Tonemapper: Split-toning cool/warm balance"},
                 &m_resources.tonemapperData.splitBalance, Persist::eYes, -0.5F, 0.5F);
  // Auto exposure (turn off for reproducible headless captures)
  m_settings.add({"tmAutoExposure", "Tonemapper: Auto-exposure [Off:0, On:1]"},
                 &m_resources.tonemapperData.autoExposure, Persist::eYes, 0, 1);
  m_settings.add({"tmAutoExposureSpeed", "Tonemapper: Auto-exposure adaptation speed"},
                 &m_resources.tonemapperData.autoExposureSpeed, Persist::eYes, 0.0F, 100.0F);
  m_settings.add({"tmEvMin", "Tonemapper: Auto-exposure minimum (EV100)"}, &m_resources.tonemapperData.evMinValue,
                 Persist::eYes, -24.0F, 24.0F);
  m_settings.add({"tmEvMax", "Tonemapper: Auto-exposure maximum (EV100)"}, &m_resources.tonemapperData.evMaxValue,
                 Persist::eYes, -24.0F, 24.0F);

  // Panel visibility: persisted before, but not settable. Exposed now so a scripted capture can
  // hide the chrome (see docs/benchmarking.md).
  m_settings.add({"uiShowCamera", "Show the Camera window"}, &m_resources.settings.showCameraWindow, Persist::eYes);
  m_settings.add({"uiShowSettings", "Show the Settings window"}, &m_resources.settings.showSettingsWindow, Persist::eYes);
  m_settings.add({"uiShowEnvironment", "Show the Environment window"}, &m_resources.settings.showEnvironmentWindow, Persist::eYes);
  m_settings.add({"uiShowTonemapper", "Show the Tonemapper window"}, &m_resources.settings.showTonemapperWindow, Persist::eYes);
  m_settings.add({"uiShowStatistics", "Show the Statistics window"}, &m_resources.settings.showStatisticsWindow, Persist::eYes);
  m_settings.add({"uiShowSceneBrowser", "Show the Scene Browser window"}, &m_resources.settings.showSceneBrowserWindow,
                 Persist::eYes);
  m_settings.add({"uiShowInspector", "Show the Inspector window"}, &m_resources.settings.showInspectorWindow, Persist::eYes);
  m_settings.add({"uiShowInteractivity", "Show the KHR_interactivity Graphs window"},
                 &m_resources.settings.showInteractivityWindow, Persist::eYes);
  m_settings.add({"uiShowAgentic", "Show the Agentic bridge window"}, &m_resources.settings.showAgenticWindow, Persist::eYes);
  m_settings.add({"uiShowGridSettings", "Show the Grid & Snap settings window"},
                 &m_resources.settings.showGridSettingsWindow, Persist::eYes);

  // Presentation mode (F11): viewport only, borderless on the window's monitor. Not persisted --
  // a talk should not start the next session in it -- and ignored in headless runs. The value is a
  // request that updatePresentationMode() applies at the top of the next UI pass.
  m_settings.add({"presentationMode",
                  "Presentation mode: hide all panels and the menu bar, borderless full screen on "
                  "the window's monitor [Off:0, On:1] (F11)"},
                 &m_presentationModeRequested, Persist::eNo);
  // Every panel presentation mode hides (main.cpp adds the profiler, logger and NVML monitor).
  for(bool* visible : {&m_resources.settings.showCameraWindow, &m_resources.settings.showSettingsWindow,
                       &m_resources.settings.showEnvironmentWindow, &m_resources.settings.showTonemapperWindow,
                       &m_resources.settings.showStatisticsWindow, &m_resources.settings.showSceneBrowserWindow,
                       &m_resources.settings.showInspectorWindow, &m_resources.settings.showInteractivityWindow,
                       &m_resources.settings.showAgenticWindow, &m_resources.settings.showGridSettingsWindow,
                       &m_resources.settings.showMemStats, &m_resources.animationControl.showStrip})
    m_presentation.addWindow(visible);
#ifndef NDEBUG
  m_presentation.addWindow(&m_resources.settings.showGridStyleWindow);
  m_presentation.addWindow(&m_resources.settings.showGizmoStyleWindow);
#endif

  // The sun. One direction is shared by every sky type that has a sun, which is why these are not
  // named sky*: Resources::sunDirection is the single source of truth and the angles drive it
  // through the callback. Everything else below is direct.
  //
  // SkySun is initialized here, not in onAttach: the command line is parsed between construction
  // and onAttach, and these callbacks fire during that parse.
  m_skySun.init(&m_resources, &m_undoStack,
                {
                    .resetFrame         = [this] { resetFrame(); },
                    .environmentPreview = [this] { requestEnvironmentPreview(); },
                });
  // SkySun::aim() reports the consequences: a re-bake for the physical sky (the only one whose image
  // depends on the sun) and an accumulation reset.
  const auto applySunAngles = [this](const nvutils::ParameterBase* const) {
    m_skySun.setAngles(m_resources.settings.sunAzimuth, m_resources.settings.sunElevation);
  };
  m_settings.add({.name = "sunAzimuth", .help = "Sun azimuth (degrees); shared by every sky type", .callbackSuccess = applySunAngles},
                 &m_resources.settings.sunAzimuth, Persist::eYes, -180.0F, 180.0F);
  m_settings.add({.name = "sunElevation", .help = "Sun elevation (degrees); shared by every sky type", .callbackSuccess = applySunAngles},
                 &m_resources.settings.sunElevation, Persist::eYes, -90.0F, 90.0F);
  // ImGui.ini restore writes sunAzimuth/Elevation directly and never runs applySunAngles, so
  // Resources::sunDirection -- the value rendering and the sky UI actually use -- would stay at
  // its default and the remembered sun position would silently not be restored. Re-run the
  // conversion once, after Application::run() has reloaded the ini (see onUIRender's one-shot).
  //
  // CLI precedence: the loadFilter installed in onAttach already skips ini writes for any key
  // ParameterParser::wasParsed() marked, so a CLI-set azimuth or elevation survives the ini
  // reload. This hook then recomputes sunDirection from the resulting per-key mix (CLI value
  // where the user overrode it, ini value otherwise), which is exactly the intended
  // CLI-wins-per-key behavior.
  m_settings.addPostRestoreHook([this, applySunAngles]() { applySunAngles(nullptr); });

  // Time of Day: the same sun, said differently. These do not add a second source of truth -- each
  // one recomputes sunAzimuth/sunElevation through m_skySun.applyTimeOfDay(), so the sun still has exactly
  // one owner and the two angles remain what is persisted, scripted and written to a marked light.
  //
  // Persisted so the widget reopens where it was left; the callback is what makes a command-line
  // --todHour actually move the sun. ini restore deliberately does not fire it: the sun's direction
  // is restored from its own persisted angles, and re-deriving it here would overwrite a sun that
  // had since been moved by hand.
  // Open on today and on this machine's clock, rather than on an arbitrary epoch somebody has to
  // correct before the widget says anything true. Read here, before the declarations below, so it
  // is the value SettingsRegistry captures as the default -- and therefore what "Reset All to
  // Default" comes back to.
  if(const sun_position::LocalClock now = sun_position::systemLocalClock(); now.valid)
  {
    m_resources.settings.todDate      = now.date;
    m_resources.settings.todHour      = now.hour;
    m_resources.settings.todUtcOffset = now.utcOffsetHours;
  }
  const auto applyTod = [this](const nvutils::ParameterBase* const) {
    m_todDrivesSun = true;
    m_skySun.applyTimeOfDay();
  };
  m_settings.add({.name = "todHour", .help = "Time of day, in local clock hours (0..24) at --todUtcOffset", .callbackSuccess = applyTod},
                 &m_resources.settings.todHour, Persist::eYes, 0.0F, 24.0F);
  m_settings.add({.name = "todDate", .help = "Date the sun is computed for, yyyy-mm-dd", .callbackSuccess = applyTod},
                 &m_resources.settings.todDate, Persist::eYes);
  m_settings.add({.name = "todLatitude", .help = "Observer latitude in degrees, positive north", .callbackSuccess = applyTod},
                 &m_resources.settings.todLatitude, Persist::eYes, -90.0F, 90.0F);
  m_settings.add({.name = "todLongitude", .help = "Observer longitude in degrees, positive east", .callbackSuccess = applyTod},
                 &m_resources.settings.todLongitude, Persist::eYes, -180.0F, 180.0F);
  m_settings.add({.name = "todUtcOffset", .help = "Hours the local clock runs ahead of UTC", .callbackSuccess = applyTod},
                 &m_resources.settings.todUtcOffset, Persist::eYes, -12.0F, 14.0F);

  // A city is an action, not a mode: it writes latitude, longitude and the offset, and nothing
  // reads it back -- the same shape as atmoPreset, and not persisted for the same reason.
  //
  // Named rather than indexed, unlike atmoPreset. Twenty-odd entries make an index unreadable in a
  // script, and worse, inserting a city would silently move every script and saved sequence that
  // used one. A name survives the list growing.
  m_settings.add({.name = "todCity",
                  .help = "Set latitude, longitude and UTC offset from a named city (see the Environment panel "
                          "for the list); standard time, no daylight saving",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        const int index = sun_position::findCity(m_resources.settings.todCity);
                        if(index < 0)
                        {
                          LOGW("--todCity: '%s' is not one of the listed cities; the location is unchanged.\n",
                               m_resources.settings.todCity.c_str());
                          return;
                        }
                        const sun_position::City& city    = sun_position::kCities[index];
                        m_resources.settings.todLatitude  = city.latitudeDeg;
                        m_resources.settings.todLongitude = city.longitudeDeg;
                        m_resources.settings.todUtcOffset = city.utcOffsetHours;
                        m_todDrivesSun                    = true;
                        m_skySun.applyTimeOfDay();
                      }},
                 &m_resources.settings.todCity, Persist::eNo);

  // Precedence, and the one place it has to be stated.
  //
  // The command line is parsed before the ini is read, so `--todHour 20` computes a sun and then
  // the sunAzimuth/sunElevation hook above puts the *remembered* sun back over it -- those two keys
  // were not on the command line, so the ini is allowed to supply them, and the hook that re-derives
  // the direction from them cannot tell it is undoing an explicit instruction. Silently: the run
  // simply renders yesterday's sun, which looks exactly like the widget not working.
  //
  // So if anything asked for a time of day from outside the UI, it asks again here, after the
  // restore. Registered after that hook, and hooks run in order, so it wins. With no such request
  // the flag is false and the remembered angles stand, which is what reopening the app should do.
  m_settings.addPostRestoreHook([this]() {
    if(m_todDrivesSun)
      m_skySun.applyTimeOfDay();
  });
  // Authored sky types (glTF OMI_environment_sky). Every one of these invalidates the baked
  // lighting environment, so they share one callback that marks it for a re-bake; the bake itself
  // runs on the application thread at the top of the next frame, never inside a parameter write.
  const auto invalidateBake = [this](const nvutils::ParameterBase* const) { onEnvironmentChanged(); };
  m_settings.add({.name = "envSaveToGltf", .help = "Write OMI_environment_sky into the glTF when saving the scene"},
                 &m_resources.settings.envSaveToGltf, Persist::eYes);
  m_settings.addVector({.name = "plainColor", .help = "Plain sky: solid color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.plainColor, Persist::eYes);
  m_settings.addVector({.name = "gradientBottomColor", .help = "Gradient sky: nadir color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.gradientBottomColor, Persist::eYes);
  m_settings.addVector({.name = "gradientHorizonColor", .help = "Gradient sky: horizon color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.gradientHorizonColor, Persist::eYes);
  m_settings.addVector({.name = "gradientTopColor", .help = "Gradient sky: zenith color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.gradientTopColor, Persist::eYes);
  m_settings.addVector({.name = "gradientSunColor", .help = "Gradient sky: sun color (linear)", .callbackSuccess = invalidateBake},
                       &m_resources.settings.gradientSunColor, Persist::eYes);
  m_settings.add({.name = "gradientBottomCurve", .help = "Gradient sky: horizon-to-nadir falloff", .callbackSuccess = invalidateBake},
                 &m_resources.settings.gradientBottomCurve, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "gradientTopCurve", .help = "Gradient sky: horizon-to-zenith falloff", .callbackSuccess = invalidateBake},
                 &m_resources.settings.gradientTopCurve, Persist::eYes, 0.0F, 1.0F);
  m_settings.add({.name = "gradientSunAngleMax", .help = "Gradient sky: angular extent of the sun glow (radians)", .callbackSuccess = invalidateBake},
                 &m_resources.settings.gradientSunAngleMax, Persist::eYes, 0.0F, 3.1416F);
  m_settings.add({.name = "gradientSunCurve", .help = "Gradient sky: sun-to-sky falloff across the glow", .callbackSuccess = invalidateBake},
                 &m_resources.settings.gradientSunCurve, Persist::eYes, 0.0F, 1.0F);

  // The physical sky's atmosphere.
  //
  // Every one of these goes through the same preview-then-settle path a sky colour does. Most of
  // them change nothing but the scattering tables, so the drag rebuilds those at preview quality
  // and the commit rebuilds them at full -- see SkyBruneton::Quality. Ground albedo, sun angular
  // radius and altitude are the exceptions that the bake also reads directly, so they move even
  // before the tables catch up.
  const auto invalidateAtmosphere = [this](const nvutils::ParameterBase* const) { requestEnvironmentPreview(); };
  m_settings.add({.name = "atmoSunAngularRadius", .help = "Physical sky: angular radius of the sun disk (radians)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoSunAngularRadius, Persist::eYes, 0.0F, 0.2F);
  m_settings.addVector({.name = "atmoGroundAlbedo", .help = "Physical sky: ground reflectance, linear RGB", .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoGroundAlbedo, Persist::eYes);
  // Altitude changes no LUT -- it is a bake input only -- so it is the cheap one of the three.
  m_settings.add({.name            = "atmoObserverAltitude",
                  .help            = "Physical sky: observer height above the ground (metres)",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoObserverAltitude, Persist::eYes, 0.0F, 60000.0F);

  // Aerial perspective. Both shading paths read it straight out of SceneFrameInfo, so unlike every
  // other atmosphere setting it needs no re-bake and no LUT rebuild: it changes what a frame does
  // with the tables, not the tables. Hence resetFrame() alone.
  m_settings.add({.name            = "atmoAerialPerspectiveScale",
                  .help            = "Aerial perspective: metres of air per scene unit (0 disables)",
                  .callbackSuccess = [this](const nvutils::ParameterBase* const) { resetFrame(); }},
                 &m_resources.settings.atmoAerialPerspectiveScale, Persist::eYes, 0.0F, 1000.0F);

  m_settings.addVector({.name            = "atmoSolarIrradiance",
                        .help            = "Physical sky: solar irradiance at the top of the atmosphere (W/m^2)",
                        .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoSolarIrradiance, Persist::eYes);

  // Rayleigh -- the air itself.
  m_settings.addVector({.name            = "atmoRayleighScattering",
                        .help            = "Physical sky: Rayleigh scattering at the surface (1/km, linear RGB)",
                        .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoRayleighScattering, Persist::eYes);
  m_settings.add({.name            = "atmoRayleighScaleHeight",
                  .help            = "Physical sky: height over which the air thins by 1/e (km)",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoRayleighScaleHeight, Persist::eYes, 0.1F, 100.0F);

  // Mie -- the aerosol.
  m_settings.addVector({.name            = "atmoMieScattering",
                        .help            = "Physical sky: Mie scattering at the surface (1/km, linear RGB)",
                        .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoMieScattering, Persist::eYes);
  m_settings.add({.name            = "atmoMieScaleHeight",
                  .help            = "Physical sky: height over which the aerosol thins by 1/e (km)",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoMieScaleHeight, Persist::eYes, 0.1F, 100.0F);
  m_settings.add({.name            = "atmoMieAnisotropy",
                  .help            = "Physical sky: Cornette-Shanks asymmetry; > 0 scatters forward",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoMieAnisotropy, Persist::eYes, -0.99F, 0.99F);
  m_settings.add({.name            = "atmoMieAlbedo",
                  .help            = "Physical sky: aerosol single-scattering albedo; 1 scatters all it removes",
                  .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoMieAlbedo, Persist::eYes, 0.05F, 1.0F);

  // Ozone -- an absorbing layer well above the ground, and what keeps a clear zenith blue rather
  // than grey at low sun.
  m_settings.addVector({.name            = "atmoOzoneExtinction",
                        .help            = "Physical sky: absorption at the peak of the ozone layer (1/km, linear RGB)",
                        .callbackSuccess = invalidateAtmosphere},
                       &m_resources.settings.atmoOzoneExtinction, Persist::eYes);
  m_settings.add({.name = "atmoOzoneCenter", .help = "Physical sky: altitude of the ozone peak (km)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoOzoneCenter, Persist::eYes, 0.0F, 200.0F);
  m_settings.add({.name = "atmoOzoneWidth", .help = "Physical sky: full thickness of the ozone layer (km)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoOzoneWidth, Persist::eYes, 0.1F, 400.0F);

  // The planet.
  m_settings.add({.name = "atmoBottomRadius", .help = "Physical sky: planet radius (km)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoBottomRadius, Persist::eYes, 100.0F, 20000.0F);
  m_settings.add({.name = "atmoThickness", .help = "Physical sky: atmosphere depth above the surface (km)", .callbackSuccess = invalidateAtmosphere},
                 &m_resources.settings.atmoThickness, Persist::eYes, 1.0F, 500.0F);

  // A preset is an action, not a mode: its callback overwrites every field above and nothing reads
  // it back. That is also why it does not persist -- restoring it after the individual values
  // would undo them.
  m_settings.add({.name = "atmoPreset",
                  .help = "Physical sky: apply an atmosphere preset [Earth:0, Mars:1, Alien:2]",
                  .callbackSuccess =
                      [this](const nvutils::ParameterBase* const) {
                        applyAtmospherePreset(m_resources.settings, m_resources.settings.atmoPreset);
                        requestEnvironmentPreview();
                      }},
                 &m_resources.settings.atmoPreset, Persist::eNo, 0, eAtmospherePresetCount - 1);

  // The MDL sky's eleven knobs (skyMultiplier, skyHaze, skyRedBlueShift, skySaturation,
  // skyHorizonHeight, skyHorizonBlur, skyGroundColor, skyNightColor, skySunDiskScale,
  // skySunDiskIntensity, skySunGlowIntensity) were registered here until that sky was deleted.
  // They are deliberately not re-registered under new names: the Bruneton atmosphere is not a
  // renaming of those parameters, it is a different model. A persisted .ini still carrying them
  // is handled by SettingsRegistry, which ignores keys it does not know.

  // Gizmo, grid and snap: previously persisted but unreachable from the command line or MCP.
  m_settings.add({"uiShowGrid", "Show the infinite grid"}, &m_resources.settings.showGrid, Persist::eYes);
  m_settings.add({"uiShowGizmo", "Show the transform gizmo on the selected node"}, &m_resources.settings.showGizmo, Persist::eYes);
  m_settings.add({"uiGridUnit", "Grid base unit (world units)"}, &m_resources.settings.gridUnit, Persist::eYes);
  m_settings.add({"uiSnapEnabled", "Snap gizmo transforms to grid increments"}, &m_resources.settings.snapEnabled, Persist::eYes);
  m_settings.add({"uiSnapRotation", "Rotation snap increment (degrees)"}, &m_resources.settings.snapRotation, Persist::eYes);
  m_settings.add({"uiSnapScale", "Scale snap increment"}, &m_resources.settings.snapScale, Persist::eYes);

  // The two Windows-menu resets, also reachable from a script, a benchmark sequence and MCP. Both
  // only raise the pending flag that applyPendingResets() consumes at the top of the next UI pass,
  // so they are safe to fire from any of those paths. onUIRender() runs in headless and benchmark
  // runs too, so the settings reset lands there as well; the layout reset is the part that
  // self-skips, since those runs never build a dockspace. "Reset all" implies the layout too: it
  // is the no-ini start-up state.
  m_settings.addAction({"resetUiLayout", "Restore the default docking layout (Windows > Reset UI Layout)"},
                       [this]() { m_pendingResetLayout = true; });
  m_settings.addAction({"resetAllToDefault", "Restore every setting and the layout to their defaults (Windows > Reset All to Default)"},
                       [this]() {
                         m_pendingResetSettings = true;
                         m_pendingResetLayout   = true;
                       });
  // Narrower sibling of the above, and the same action the Environment panel's Reset button runs.
  // Registered rather than left UI-only so a benchmark sequence or an agent can put one sky type
  // back to a known state without resetting the whole application.
  m_settings.addAction({"envResetDefaults", "Restore the selected environment type's settings to their defaults"}, [this]() {
    m_uiEnvironment.resetDefaults();
    resetFrame();
  });

  // Register PathTracer-specific command line parameters
  m_pathTracer.registerParameters(&m_settings);
  m_rasterizer.registerParameters(&m_settings);

  m_benchmark.registerParameters(
      paramReg, {
                    .applyGltfCamera = [this](int cameraIndex) { applyGltfCamera(cameraIndex); },
                    .fitScene =
                        [this]() {
                          if(m_resources.getScene() && m_resources.getScene()->valid())
                          {
                            const nvutils::Bbox bbox = m_resources.getScene()->getSceneBounds();
                            m_cameraManip->fit(bbox.min(), bbox.max(), false, true, m_cameraManip->getAspectRatio());
                            resetFrame();
                          }
                        },
                    .resetFrame = [this]() { resetFrame(); },
                    .saveScreenshot =
                        [this](const std::filesystem::path& filename) {
                          if(m_app)
                          {
                            m_app->saveImageToFile(m_resources.gBuffers.getColorImage(Resources::eImgTonemapped),
                                                   m_resources.gBuffers.getSize(), filename);
                          }
                        },
                    .saveUiScreenshot =
                        [this](const std::filesystem::path& filename) {
                          // Capture the full composited window (ImGui panels + viewport). This reads the swapchain
                          // PRESENT image, which only exists in a windowed run; headless has no swapchain.
                          if(m_app && !m_app->isHeadless())
                          {
                            m_app->requestScreenShot(filename, 100);
                          }
                          else
                          {
                            LOGW("--uiScreenshot ignored: requires a windowed run (no swapchain in headless mode)\n");
                          }
                        },
                    .selectSceneNode = [this](int index) { selectSceneNodeFromScript(index); },
                });

  // Initialize camera manipulator
  m_cameraManip           = std::make_shared<nvutils::CameraManipulator>();
  m_resources.cameraManip = m_cameraManip;  // Share with resources
}

//--------------------------------------------------------------------------------------------------
// The onAttach method is called when the application is attached to the renderer
void GltfRenderer::onAttach(nvapp::Application* app)
{
  SCOPED_TIMER("GltfRenderer::onAttach");

  m_app                = app;
  m_device             = app->getDevice();
  m_resources.instance = app->getInstance();
  m_resources.app      = app;

  // ===== Settings Handler (ImGui persistent) =====
  // The list lives with the declarations in the constructor; this replays the persisted ones now
  // that the handler exists. Nothing to keep in sync by hand.
  m_settingsHandler.setHandlerName("GltfRenderer");
  m_settings.applyPersistence(m_settingsHandler);
  m_settingsHandler.setLoadFilter([this](const std::string& key) {
    // Skip loading settings that were explicitly set via the command line
    return !(m_parameterParser && m_parameterParser->wasParsed(key));
  });
  m_settingsHandler.addImGuiHandler();

  // Customize ImGui style for better visibility
  ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] = (ImVec4)ImColor::HSV(0.3F, 0.5F, 0.5F);

  // ===== Memory Allocation & Buffer Management =====
  m_resources.allocator.init({
      .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice   = app->getPhysicalDevice(),
      .device           = app->getDevice(),
      .instance         = app->getInstance(),
      .vulkanApiVersion = VK_API_VERSION_1_4,
  });  // Allocator

  // If there is a leak (nvvkAllocID -> ID)
  // m_resources.allocator.setLeakID(155);


  m_transientCmdPool = nvvk::createTransientCommandPool(m_device, app->getQueue(0).familyIndex);
  NVVK_DBG_NAME(m_transientCmdPool);

  m_loadPipeline.init(m_device, app->getQueue(0).queue, m_transientCmdPool);

  // FrameUploader records copies into the caller's command buffer and retires staging
  // once the frame timeline semaphore (or a load-pipeline wait) has signaled.
  // blockSize is the max size of a single append (circular allocator constraint) and
  // is larger than the library default so scene-create can stage large meshes/textures.
  NVVK_CHECK(m_resources.staging.init({
      .allocator = &m_resources.allocator,
      .blockSize = kFrameUploaderBlockSize,  // shared with SceneVk::createImage row-strip chunking
      .debugName = "frameUploads",
  }));

  m_resources.commandPool = app->getCommandPool();


  // ===== Texture & Image Resources =====
  m_resources.samplerPool.init(m_device);
  VkSampler linearSampler{};
  NVVK_CHECK(m_resources.samplerPool.acquireSampler(linearSampler));
  NVVK_DBG_NAME(linearSampler);

  // IBL environment map
  m_resources.hdrIbl.init(&m_resources.allocator, &m_resources.samplerPool);
  // Required before HdrIbl::updateFromGpuImage: the file-load path writes the per-texel PDF into
  // the image alpha on the CPU, but a GPU-baked environment is never read back, so the PDF write
  // is a compute pass that has to be created up front. Without it EnvBaker::commit dispatches a
  // null pipeline -- caught only by an assert, so a release build simply crashes.
  m_resources.hdrIbl.initGpuWriter(std::span(env_write_pdf_slang));
  m_resources.hdrDome.init(&m_resources.allocator, &m_resources.samplerPool, m_app->getQueue(0));

  // Application-level memory tracker (frame targets, DLSS, OptiX images)
  m_resources.appMemoryTracker.init(&m_resources.allocator);

  // G-Buffer
  NVVK_CHECK(m_resources.gBuffers.init({.device = m_device,
                                        .alloc  = &m_resources.allocator,
                                        .colorFormats =
                                            {
                                                VK_FORMAT_R8G8B8A8_UNORM,       // Tonemapped (eImgTonemapped)
                                                VK_FORMAT_R32G32B32A32_SFLOAT,  // Rendered image (eImgRendered)
                                                VK_FORMAT_R32_SFLOAT,  // ObjectID for selection/silhouette (eImgSelection), .r = render node ID+1
                                            },
                                        .depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice()),
                                        .debugName   = "GBuffers"}));
  m_resources.linearSampler = linearSampler;
  {
    VkCommandBuffer cmd{};
    nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
    NVVK_CHECK(m_resources.gBuffers.update(cmd, {100, 100}));
    m_resources.gBuffers.cmdClear(cmd);
    nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_app->getQueue(0).queue);
    m_resources.appMemoryTracker.track("GBuffers", m_resources.gBuffers, Resources::eImgCount);
  }
  m_resources.tonemappedUi.init(m_resources.gBuffers.getUiImageView(Resources::eImgTonemapped));

  // ===== Rendering Utilities =====

  // Ray picker
  m_rayPicker.init(&m_resources.allocator);

  // Tonemapper
  m_resources.tonemapper.init(&m_resources.allocator, tonemapper_slang);

  // Silhouette renderer
  m_silhouette.init(m_resources);

  // Async G-buffer readback for KHR_interactivity hover detection (docs/interactivity.md Phase E)
  m_hoverPicker.init(m_resources);

  // Producer for analytic skies. Its lat-long image feeds the same HdrIbl / HdrEnvDome pair the
  // HDR file path uses, so both render paths consume one environment regardless of sky type.
  // Allocated here but only dispatched once a baked sky type is selected.

  // Bruneton atmosphere LUTs. Built once here from the Earth defaults; they depend only on the
  // atmosphere, never on the sun or the camera, so moving the sun later costs nothing.
  m_skyBruneton.init(m_resources, SkyBruneton::Shaders{.precompute = std::span(sky_bruneton_precompute_slang)});
  // Seed from the settings, not from the Earth defaults init() installs. Otherwise the first
  // refreshBakedEnvironment() finds the parameters changed and rebuilds the whole set immediately
  // -- two full precomputes, every startup, even when nothing was customised.
  m_skyBruneton.setParameters(buildAtmosphereParameters());
  m_skyBruneton.precompute(m_resources);

  // After SkyBruneton: EnvBaker takes its descriptor-set layout to build a pipeline layout that
  // covers both producers.
  m_envBaker.init(m_resources, m_skyBruneton.getDescriptorSetLayout(), m_skyBruneton.getDescriptorSet());
  m_envBaker.setResolution(m_resources, kEnvBakeSize);

  // Publish the same set to the two renderers, which read the tables directly for aerial
  // perspective. Here rather than in each of them: they build their pipeline layouts from
  // Resources, and neither needs to know which class owns the LUTs.
  m_resources.skyLutDescriptorSetLayout = m_skyBruneton.getDescriptorSetLayout();
  m_resources.skyLutDescriptorSet       = m_skyBruneton.getDescriptorSet();

  // ===== Scene & Acceleration Structure =====
  m_resources.sceneGpu.init(&m_resources.allocator, &m_resources.samplerPool, m_app->getQueue(0).queue,
                            [app](std::function<void()>&& fn) { app->submitResourceFree(std::move(fn)); });
  m_resources.transformCompute.init(&m_resources.allocator);
  m_resources.transformCompute.setGraphicsQueue(m_app->getQueue(0).queue);
  m_resources.transformCompute.setDeferredFree(
      [app](std::function<void()>&& fn) { app->submitResourceFree(std::move(fn)); });

  // ===== Profiling & Performance =====
  {
    SCOPED_TIMER("Profiler");
    m_profilerTimeline = g_profilerManager.createTimeline({.name = "Primary Timeline"});
    m_profilerGpuTimer.init(m_profilerTimeline, m_app->getDevice(), m_app->getPhysicalDevice(),
                            int32_t(m_app->getQueue(0).familyIndex), false);
  }


  // ===== Shader Compilation =====
  {
    SCOPED_TIMER("Shader Slang");
    using namespace slang;
    m_resources.slangCompiler.addSearchPaths(nvsamples::getShaderDirs());
    m_resources.slangCompiler.defaultTarget();
    m_resources.slangCompiler.defaultOptions();

    // Specific options for this sample
    m_resources.slangCompiler.addOption(
        {CompilerOptionName::DebugInformation, {CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_STANDARD}});
    m_resources.slangCompiler.addOption(
        {CompilerOptionName::Optimization, {CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_NONE}});

    // Enable specific capabilities for better performance and features.
    m_resources.slangCompiler.addCapability("spvShaderInvocationReorderEXT");  // SER via VK_EXT_ray_tracing_invocation_reorder
    m_resources.slangCompiler.addCapability("spvInt64Atomics");                // # 64-bit atomic operations
    m_resources.slangCompiler.addCapability("spvShaderClockKHR");              // # Shader clock for profiling
    m_resources.slangCompiler.addCapability("spvRayTracingMotionBlurNV");  // # Motion blur for ray tracing
    m_resources.slangCompiler.addCapability("spvRayQueryKHR");             // # Ray query operations
    m_resources.slangCompiler.addCapability("spvGroupNonUniformBallot");  // # Ballot operations for subgroup functionality
    m_resources.slangCompiler.addCapability("spvGroupNonUniformArithmetic");  // # Arithmetic operations across subgroups

#if defined(USE_DLSS)
    m_resources.slangCompiler.addMacro({"HAS_DLSS_MOTION", "1"});
#endif

#if defined(AFTERMATH_AVAILABLE)
    // This aftermath callback is used to report the shader hash (Spirv) to the Aftermath library.
    m_resources.slangCompiler.setCompileCallback([&](const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize) {
      std::span<const uint32_t> data(spirvCode, spirvSize / sizeof(uint32_t));
      AftermathCrashTracker::getInstance().addShaderBinary(data);
    });
#endif
  }

  // ===== Renderer Initialization =====

  // Create resources
  createDescriptorSets();
  createHDR("");  // Dummy HDR
  createResourceBuffers();

  // Initialize the renderers
  m_pathTracer.onAttach(m_resources, &m_profilerGpuTimer);
  m_pathTracer.setProfilerTimeline(m_profilerTimeline);
  m_pathTracer.setBusyWindow(&m_busy);  // Show BusyWindow during async shader/pipeline compiles
  m_rasterizer.onAttach(m_resources, &m_profilerGpuTimer);

  m_pathTracer.createPipeline(m_resources);
  m_rasterizer.createPipeline(m_resources);

  // Wire the create-catalog hook up front so the menu-bar "Create" works with nothing loaded:
  // it stands up an empty, UI-wired scene on demand (wireSceneToUi re-sets this same hook later).
  m_sceneBrowser.setBeforeCreateCallback([this] { ensureEmptyScene(); });

  // The Environment panel. It edits Settings through m_resources directly; everything below is a
  // consequence an edit has for the renderer, which the panel has no business knowing how to do.
  m_uiEnvironment.setResources(&m_resources);
  m_uiEnvironment.setSelection(&m_sceneSelection);
  m_uiEnvironment.setActions({
      .environmentChanged    = [this] { onEnvironmentChanged(); },
      .requestPreview        = [this] { requestEnvironmentPreview(); },
      .resetFrame            = [this] { resetFrame(); },
      .applyTimeOfDay        = [this] { m_skySun.applyTimeOfDay(); },
      .environmentRotated    = [this] { onEnvironmentRotated(); },
      .setSunAngles          = [this](float az, float el) { m_skySun.setAngles(az, el); },
      .aimSun                = [this](const glm::vec3& toSun) { m_skySun.aim(toSun); },
      .sunLightNode          = [this] { return m_skySun.lightNode(); },
      .setSkySunNode         = [this](int node) { m_skySun.setSunNode(node); },
      .directionalLightNodes = [this] { return m_skySun.candidateLights(); },
      .loadHdrFileDialog     = [this] { loadHdrFileDialog(); },
      .loadSkyPresetDialog   = [this] { loadSkyPresetDialog(); },
      .saveSkyPresetDialog   = [this] { saveSkyPresetDialog(); },
  });

  // ===== Visual Helpers (Grid + Transform Gizmo) =====
  {
    VkFormat depthFormat = m_resources.gBuffers.getDepthFormat();
    VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;  // Matches eImgTonemapped

    VisualHelpers::Resources helperRes{
        .app           = m_app,
        .alloc         = &m_resources.allocator,
        .uploader      = &m_resources.staging,
        .device        = m_device,
        .sampler       = linearSampler,
        .slangCompiler = &m_resources.slangCompiler,
        .colorFormat   = colorFormat,
        .depthFormat   = depthFormat,
    };
    m_visualHelpers.init(helperRes);

    m_visualHelpers.transform.setOnTransformBegin([this]() {
      if(m_gizmoNodeIndex >= 0 && m_resources.getScene())
      {
        const auto& node = m_resources.getScene()->editor().getNode(m_gizmoNodeIndex);
        tinygltf::utils::getNodeTRS(node, m_gizmoSnapshotT, m_gizmoSnapshotR, m_gizmoSnapshotS);
      }
    });

    m_visualHelpers.transform.setOnTransformChange([this]() {
      if(m_gizmoNodeIndex >= 0 && m_resources.getScene())
      {
        glm::quat rotation = glm::quat(glm::radians(m_gizmoRotation));
        m_resources.getScene()->editor().setNodeTRS(m_gizmoNodeIndex, m_gizmoPosition, rotation, m_gizmoScale);
        resetFrame();
      }
    });

    m_visualHelpers.transform.setOnTransformEnd([this]() {
      if(m_gizmoNodeIndex >= 0 && m_resources.getScene())
      {
        glm::vec3   newT, newS;
        glm::quat   newR;
        const auto& node = m_resources.getScene()->editor().getNode(m_gizmoNodeIndex);
        tinygltf::utils::getNodeTRS(node, newT, newR, newS);
        auto cmd = std::make_unique<SetTransformCommand>(*m_resources.getScene(), m_gizmoNodeIndex, m_gizmoSnapshotT,
                                                         m_gizmoSnapshotR, m_gizmoSnapshotS, newT, newR, newS);
        m_undoStack.pushExecuted(std::move(cmd));
      }
    });
  }

#ifdef USE_AGENTIC
  // ===== Agentic bridge controller =====
  // Done last so all the resources / queues / path-tracer state the callbacks
  // reach into are fully initialized.
  m_agentic.init({
      .app              = m_app,
      .resources        = &m_resources,
      .device           = m_device,
      .transientCmdPool = m_transientCmdPool,
      // Atomic "load + apply this HDR" action used when an HDRI job comes back.
      .applyHdri =
          [this](const std::filesystem::path& path) {
            createHDR(path);
            m_resources.settings.envSystem = shaderio::EnvSystem::eHdr;
            calibrateFireflyClamp();
          },
      .resetFrame     = [this]() { resetFrame(); },
      .runTonemapPass = [this](VkCommandBuffer cmd,
                               bool            skipBeautifiedOverlay) { runTonemapPass(cmd, skipBeautifiedOverlay); },
  });

  // Apply the optional --agenticBridgeRoot override now that the controller has
  // set up its default root; a no-op when the flag was not passed.
  m_agentic.setBridgeRoot(m_agenticBridgeRootOverride);
#endif  // USE_AGENTIC
}

#ifdef USE_AGENTIC
void GltfRenderer::setAgenticBridgeRoot(const std::filesystem::path& root)
{
  m_agenticBridgeRootOverride = root;
}
#endif

//--------------------------------------------------------------------------------------------------
// Detach the renderers and destroy the resources
void GltfRenderer::onDetach()
{
  // SYNC NOTE: Full device wait during shutdown is the standard Vulkan teardown pattern.
  vkDeviceWaitIdle(m_device);
  m_thumbnailCache.clear();  // release ImGui thumbnail descriptor sets while idle
  m_visualHelpers.deinit();
  m_pathTracer.onDetach(m_resources);
  m_rasterizer.onDetach(m_resources);
  destroyResources();
}

//--------------------------------------------------------------------------------------------------
// Resize the G-Buffer and the renderers
void GltfRenderer::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  m_resources.appMemoryTracker.untrack("GBuffers", m_resources.gBuffers, Resources::eImgCount);
  NVVK_CHECK(m_resources.gBuffers.update(cmd, size));
  m_resources.gBuffers.cmdClear(cmd);
  m_resources.appMemoryTracker.track("GBuffers", m_resources.gBuffers, Resources::eImgCount);
  m_pathTracer.onResize(cmd, size, m_resources);
  m_rasterizer.onResize(cmd, size, m_resources);
  m_resources.hdrDome.setOutImage(m_resources.gBuffers.getColorStorageImageInfo(Resources::eImgRendered));
  m_resources.tonemappedUi.update(m_resources.gBuffers.getUiImageView(Resources::eImgTonemapped));

  // Resize visual helpers (depth buffer + scene depth descriptor set)
  m_visualHelpers.onResize(cmd, size, m_resources.gBuffers.getDepthImage(), m_resources.gBuffers.getDepthImageView(),
                           m_resources.linearSampler);

  // Camera (was handled by ElementCamera, now owned by the renderer)
  m_cameraManip->setWindowSize({size.width, size.height});
  m_cameraManip->adjustOrthographicAspect();

  resetFrame();  // Reset frame to restart the rendering
}

//--------------------------------------------------------------------------------------------------
// Render the UI elements and handle UI-driven scene interactions
// This method is responsible for:
// 1. Rendering the settings panel with renderer selection, environment options, and debug controls
// 2. Displaying the scene graph hierarchy and handling object selection
// 3. Managing variant and animation controls when available in the loaded scene
// 4. Showing scene statistics and performance metrics
// 5. Rendering the viewport with the tonemapped image and optional 3D axis overlay
// 6. Processing changes from UI interactions and triggering re-rendering when needed
// 7. Displaying the busy indicator during asynchronous operations
// The UI layout is organized hierarchically with collapsible sections for better usability
bool GltfRenderer::isBenchmarkMode() const
{
  return m_benchmark.isBenchmarkMode();
}

bool GltfRenderer::isHeadlessMode() const
{
  return m_app != nullptr && m_app->isHeadless();
}

bool GltfRenderer::isAutomatedRun() const
{
  return isHeadlessMode() || isBenchmarkMode();
}

void GltfRenderer::alignMaxFramesForHeadless(uint32_t headlessFrames)
{
  BenchmarkController::alignMaxFramesForHeadless(m_resources.settings.maxFrames, headlessFrames);
}

void GltfRenderer::applyHeadlessDefaults()
{
  if(!(m_parameterParser && m_parameterParser->wasParsed("animPlay")))
    m_animPlay = false;
  m_resources.animationControl.play = m_animPlay;
}

BenchmarkController::HeadlessFrameInfo GltfRenderer::benchmarkFrameInfo() const
{
  return {.totalFrames = m_app ? m_app->getHeadlessFrameCount() : 0,
          .maxFrames   = m_resources.settings.maxFrames,
          .ptSamples   = m_pathTracer.m_pushConst.numSamples,
          .imageSize   = m_resources.gBuffers.getSize()};
}

std::vector<BenchmarkController::MemorySample> GltfRenderer::benchmarkMemorySamples() const
{
  std::vector<BenchmarkController::MemorySample> samples;
  if(m_resources.getScene() && m_resources.getScene()->valid())
  {
    const uint64_t sceneUsed = sumTrackerCurrentBytes(m_resources.sceneVk.getMemoryTracker())
                               + sumTrackerCurrentBytes(m_resources.sceneRtx.getMemoryTracker())
                               + sumTrackerCurrentBytes(m_resources.transformCompute.getMemoryTracker())
                               + sumTrackerCurrentBytes(m_resources.animationVk.getMemoryTracker());
    const uint64_t scenePeak = sumTrackerPeakBytes(m_resources.sceneVk.getMemoryTracker())
                               + sumTrackerPeakBytes(m_resources.sceneRtx.getMemoryTracker())
                               + sumTrackerPeakBytes(m_resources.transformCompute.getMemoryTracker())
                               + sumTrackerPeakBytes(m_resources.animationVk.getMemoryTracker());
    samples.push_back({.category = "Scene", .deviceUsed = sceneUsed, .deviceAllocated = scenePeak});
  }
  else
  {
    samples.push_back({.category = "Scene"});
  }

  const auto& appStats = m_resources.appMemoryTracker.getTotalStats();
  samples.push_back({.category = (m_resources.settings.renderSystem == RenderingMode::ePathtracer) ? "PathTracer" : "Rasterizer",
                     .deviceUsed      = appStats.currentBytes,
                     .deviceAllocated = appStats.peakBytes});
  return samples;
}

void GltfRenderer::saveHeadlessOutputImage()
{
  std::string                 outputPath = m_resources.headlessOutputPath.empty() ?
                                               nvutils::getExecutablePath().replace_extension(".jpg").string() :
                                               m_resources.headlessOutputPath.string();
  const std::filesystem::path parentDir  = std::filesystem::path(outputPath).parent_path();
  if(!parentDir.empty() && !std::filesystem::exists(parentDir))
  {
    LOGW("GltfRenderer::saveHeadlessOutputImage(): output directory does not exist for path %s\n", outputPath.c_str());
  }
  if(!m_app)
  {
    LOGE("GltfRenderer::saveHeadlessOutputImage(): application is not set; cannot save image to %s\n", outputPath.c_str());
    return;
  }
  m_app->saveImageToFile(m_resources.gBuffers.getColorImage(Resources::eImgTonemapped), m_resources.gBuffers.getSize(), outputPath);
}

// Consume a pending image/texture-set change (import/replace/remove/reload/sampler/undo) at the very
// start of the frame, BEFORE any ImGui::Image is recorded this frame.
//
// Why here and nowhere else: rebuildVulkanSceneFull() frees the ImGui thumbnail descriptor sets
// (ThumbnailCache::clear) and destroys the scene image views they reference. If it ran after the panels
// have already recorded ImGui::Image draw commands, ImGui_ImplVulkan_RenderDrawData would later replay
// them with freed descriptors (VUID-vkCmdBindDescriptorSets-pDescriptorSets-parameter: "Invalid
// VkDescriptorSet Object"). rebuildVulkanSceneFull() waits on the queue first, so the prior frame that
// used the now-freed thumbnails/views has finished; the current frame re-acquires fresh thumbnails in
// renderUI(). The prior frame's material copy-struct write-backs (spec/gloss) have also flushed, so
// sRGB detection sees the final material usage.
//
// Two other sites deliberately defer to this method (grep DirtyFlags::texturesChanged):
//   - reconcileGeometryIfNeeded(): must NOT rebuild here (it runs mid-frame via onUndoRedo).
//   - updateSceneChanges(): skips its per-frame sync while the flag is pending, so the material buffer
//     never references a texture index that eTextures[] does not contain yet.
void GltfRenderer::applyPendingTextureRebuild()
{
  nvvkgltf::Scene* scene = m_resources.getScene();
  if(!scene || !scene->getDirtyFlags().texturesChanged)
    return;

  scene->getDirtyFlags().texturesChanged = false;
  rebuildVulkanSceneFull();
  resetFrame();
  m_sceneBrowser.markCachesDirty();
  m_inspector.refreshTextureNames();
}

// Consume a pending TAIL-ONLY texture change (import / undo / redo of an imported texture) at frame top,
// alongside applyPendingTextureRebuild(). Unlike the full rebuild, this reconciles only the appended or
// removed tail (SceneVk::syncTextureTail): it neither stalls the GPU nor frees existing image views, so
// thumbnails and in-flight frames stay valid and no acceleration-structure work is needed. The per-frame
// material sync (which carries the new texture index) stays gated until the flag clears -- see the
// texturesTailChanged guard in updateSceneChanges().
void GltfRenderer::applyPendingTextureTailSync()
{
  nvvkgltf::Scene* scene = m_resources.getScene();
  if(!scene || !scene->getDirtyFlags().texturesTailChanged)
    return;

  const uint32_t firstTexture = m_resources.sceneVk.textureCount();
  const uint32_t firstSampler = m_resources.sceneVk.samplerCount();

  // Load/create new tail images (or deferred-free removed ones) on a transient command buffer, uploaded
  // asynchronously via the load pipeline -- the same pattern as the full rebuild, minus the queue wait.
  VkCommandBuffer cmd{};
  nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
  m_resources.sceneVk.syncTextureTail(cmd, m_resources.staging, *scene);
  NVVK_CHECK(vkEndCommandBuffer(cmd));
  m_loadPipeline.enqueue(cmd);

  // Write only the newly appended descriptors; removed tail slots become unused (PARTIALLY_BOUND) and
  // need no write. The image views exist immediately (created above); their contents land once the
  // enqueued upload completes, before any frame samples the new texture.
  const uint32_t newTexture = m_resources.sceneVk.textureCount();
  const uint32_t newSampler = m_resources.sceneVk.samplerCount();
  if(newTexture > firstTexture || newSampler > firstSampler)
  {
    if(!writeTextureDescriptorRange(firstTexture, newTexture > firstTexture ? newTexture - firstTexture : 0,  //
                                    firstSampler, newSampler > firstSampler ? newSampler - firstSampler : 0))
      return;  // capacity overflow (logged FATAL): leave texturesTailChanged set so updateSceneChanges keeps
               // the material sync gated -- the material must not reference a texture whose descriptor is unwritten.
  }

  // Descriptors are now consistent with the model; lift the per-frame material-sync gate.
  scene->getDirtyFlags().texturesTailChanged = false;

  // A tail shrink (undo) deferred-freed the removed image views. Park any thumbnail descriptors keyed on
  // them so a later view-handle reuse cannot hit a stale descriptor (a grow frees nothing, so it needs no
  // eviction, and existing thumbnails stay valid).
  if(newTexture < firstTexture)
    m_thumbnailCache.clearDeferred();

  resetFrame();
  m_sceneBrowser.markCachesDirty();
  m_inspector.refreshTextureNames();
}

// Consume pending sampler edits (DirtyFlags::samplers) at frame top: an Inspector wrap/filter edit
// changes model.samplers[i] in place. Unlike applyPendingTextureRebuild(), this never touches m_images
// or m_textures -- only the affected VkSampler is recreated and its single eSamplers descriptor slot
// rewritten, so image data is never re-read from disk for a sampler-only change.
void GltfRenderer::applyPendingSamplerUpdate()
{
  nvvkgltf::Scene* scene = m_resources.getScene();
  if(!scene || scene->getDirtyFlags().samplers.empty())
    return;

  const tinygltf::Model& model = scene->getModel();
  for(int samplerIndex : scene->getDirtyFlags().samplers)
  {
    m_resources.sceneVk.updateSampler(model, samplerIndex);
    if(!writeTextureDescriptorRange(0, 0, static_cast<uint32_t>(samplerIndex) + 1, 1))
      return;  // capacity overflow (logged FATAL): leave DirtyFlags::samplers set so the stale eSamplers
               // slot -- whose old VkSampler is already queued for deferred release -- is not forgotten.
  }
  scene->getDirtyFlags().samplers.clear();

  resetFrame();
}

void GltfRenderer::onUIRender()
{
  // Run any post-restore hooks now that Application::run() has reloaded the ini (which happens
  // between onAttach and the first frame, in both windowed and headless paths). ImGui writes
  // restored values straight into storage without firing the parameter's callbackSuccess, so
  // derived state (e.g. Resources::sunDirection from the restored sunAzimuth/Elevation) would
  // otherwise stay stale until the user touched a UI slider.
  if(m_pendingRestoreCallbacks)
  {
    m_pendingRestoreCallbacks = false;
    m_settings.runPostRestoreHooks();
  }

  applyPendingAuthoredSky();         // frame-top: adopt a loaded scene's OMI sky, after the ini restore above
  applyPendingSkyPresets();          // frame-top: start-up presets, after the scene's sky so they win over it
  applyPendingEnvironmentRefresh();  // frame-top: re-bake / rebind the environment after a sky change (see method)
  applyPendingTextureRebuild();   // frame-top: full GPU texture rebuild for a prior-frame structural edit (see method)
  applyPendingTextureTailSync();  // frame-top: incremental append/remove for a prior-frame import/undo/redo (see method)
  applyPendingSamplerUpdate();    // frame-top: in-place VkSampler update for a prior-frame sampler wrap/filter edit

  // Advance the thumbnail cache's deferred-free ring before any panel acquires thumbnails this frame.
  m_thumbnailCache.beginFrame(m_app->getFrameCycleSize());

  // Windows > Reset UI Layout / Reset All to Default, before any panel is submitted. This sits
  // ahead of the benchmark branch because the two resets are also scriptable actions: a benchmark
  // sequence that issues --resetAllToDefault expects the settings to be back at their defaults for
  // the measurements that follow, and benchmark mode never reaches renderUI().
  applyPendingResets();
  updatePresentationMode();

  if(isBenchmarkMode())
  {
    renderBenchmarkViewport();
    return;
  }
  renderUI();
  m_toasts.render();  // transient notifications overlay (drawn on top of the panels)
}

//--------------------------------------------------------------------------------------------------
// Resolve a glTF texture / image index to a bounded ImGui thumbnail. Returns 0 when the index is out
// of range or the image is not resident on the GPU (e.g. an unused image that was never uploaded).
ImTextureID GltfRenderer::thumbnailForTexture(int textureIndex)
{
  if(textureIndex < 0)
    return 0;
  return m_thumbnailCache.acquire(m_resources.sceneVk.textureView(static_cast<uint32_t>(textureIndex)));
}

ImTextureID GltfRenderer::thumbnailForImage(int imageIndex)
{
  if(imageIndex < 0)
    return 0;
  return m_thumbnailCache.acquire(m_resources.sceneVk.imageView(static_cast<uint32_t>(imageIndex)));
}


//--------------------------------------------------------------------------------------------------
// Render the scene
void GltfRenderer::onRender(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  m_profilerTimeline->frameAdvance();
  // Before anything reads the sun: if the scene marks a light as the sky's sun, that light owns
  // its direction.
  m_skySun.syncFromMarkedLight();
  // Don't do anything if the busy window is open
  if(m_busy.isBusy())
  {
    return;
  }

  // Consume the done signal from the busy state, this will remove the Progress Bar from the UI.
  if(m_busy.isDone())
  {
    m_busy.consumeDone();
    resetFrame();
  }

  // Loading pipeline: submit queued work, poll completion, run callbacks
  if(isAutomatedRun())
    m_loadPipeline.drain();
  else if(m_loadPipeline.poll())
    return;  // Still loading -- give control back to the UI

  // Preview re-bake for a control that is being dragged. Unlike the commit path this records into
  // the frame's own command buffer and waits on nothing, so it belongs here rather than in
  // onUIRender: it is per-frame work, not a frame-boundary rebuild.
  if(m_envPreviewPending)
  {
    m_envPreviewPending = false;
    if(isBakedEnvironment(m_resources.settings.envSystem))
    {
      // Only the rasterizer reads the prefiltered cubemaps, so the path tracer skips the prefilter
      // and a drag costs it little more than the bake itself.
      const bool refreshDome = m_resources.settings.renderSystem == RenderingMode::eRasterizer;

      // Nothing waits here, so order explicitly: the LUTs, the lat-long image and the sun-disk
      // buffer are still being read by frames in flight, and are read again by this frame's
      // ray-tracing and graphics passes after the rewrite.
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

      // Most of the atmosphere controls -- Rayleigh, Mie, ozone, the planet radii -- change
      // nothing except the LUTs, so without a preview rebuild they look like dead sliders until
      // the mouse comes up. Preview quality is what makes that affordable: a sixteenth of the
      // directions and two bounces instead of four (see SkyBruneton::Quality), and recorded into this
      // frame rather than submitted and waited on. The commit that follows restores full quality.
      if(m_resources.settings.envSystem == shaderio::EnvSystem::eSky)
      {
        const shaderio::SkyAtmosphereParameters wanted = buildAtmosphereParameters();
        if(!m_skyBruneton.lutsSatisfy(wanted, SkyBruneton::kQualityPreview))
        {
          m_skyBruneton.setParameters(wanted);
          m_skyBruneton.recordPrecompute(cmd, SkyBruneton::kQualityPreview);
        }
      }

      // The baker holds the parameters from the last commit; refresh them or the preview re-bakes
      // the image it already has. Invisible for the authored skies -- their background is drawn
      // per-ray from settings, so only the lighting lagged -- but the Bruneton sky *is* its baked
      // image, so without this a drag showed nothing until the mouse came up.
      updateBakerParameters();
      m_envBaker.preview(cmd, m_resources, refreshDome);
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
      resetFrame();
    }
  }
  else if(m_envCommitOwed && !ImGui::IsAnyItemActive())
  {
    // First frame since the edits stopped: pay for the full rebuild the previews skipped.
    //
    // "Stopped" also means the control has been let go. A slider reports a change only on frames
    // where the mouse moved, so a drag held still for one frame used to commit mid-drag -- a device
    // wait and a full-quality rebuild -- and the next movement dropped back to preview quality. The
    // physical sky flickered between the two, worst under heavy Mie, whose haze is mostly the
    // multiple scattering a preview trims. Drivers that are not ImGui controls (the gizmo, an
    // animated sun) still settle as before: when they stop requesting previews.
    m_envCommitOwed = false;
    onEnvironmentChanged();
  }

  // Recycle staging from completed frames, then tag new uploads with this frame's signal.
  m_resources.staging.releaseCompletedAllocations();
  m_resources.staging.updateFrameSemaphoreState(nvvk::SemaphoreState::makeFixed(m_app->getFrameSignalSemaphore()));

  // Empty scene, clear the G-Buffer
  if(!m_resources.getScene() || !m_resources.getScene()->valid())
  {
    clearGbuffer(cmd);
    return;
  }

  // Keep runtime denoiser guide usage synchronized with the optimal-shader feature set.
  // PathTracer::ensureShadersAndPipelines() watches currentFeatureSet and recompiles when this flips.
  m_resources.currentFeatureSet.set(nvvkgltf::SceneFeatureSet::eDlssGuide, dlssGuideRequired());

  m_benchmark.beginHeadlessTimingIfNeeded(isHeadlessMode(), benchmarkFrameInfo());

  // Start the profiler section for the GPU timer
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

  // KHR_interactivity: tick the graph (pointer/set writes, via Scene::markNodeDirty) BEFORE the DLSS
  // instance-motion snapshot decision below - that decision reads DirtyFlags::nodes to know whether
  // anything moved *this* frame, and a graph-driven move (e.g. a per-tick pointer/set slerp, as in the
  // official PlaceOnClickPuzzle.glb) only populates that flag here. Running this after the snapshot
  // decision (as it used to) meant a moving piece's dirty flag wasn't visible yet when the check ran,
  // so dlssInstanceMotionActive stayed false and the shader fell back to camera-only motion vectors
  // for that instance every single tick of the move - not just the first frame, since the same
  // one-frame-late pattern repeats on every subsequent tick too. The result was a persistent
  // trailing/ghosting artifact on the moving piece for its entire flight, visible only with DLSS on
  // (this is instance-motion-vector territory - see docs/denoising.md). Hover/click are ticked in the
  // same block since they already had to run before the graph tick (a same-frame hover/select feeds
  // this tick), and moving them doesn't change their own correctness.
  updateHoverState();      // KHR_interactivity: poll HoverPicker, notify onHoverIn/onHoverOut before this tick
  updateClickPickState();  // Poll the previous click's async ray-pick, apply it once ready (see m_pendingClickPick)
  // KHR_interactivity: pointer/set writes flow into `changed` via the dirty-flags path below
  // (Scene::markNodeDirty etc.), but an animation/start-driven pose (applied synchronously inside
  // this call via reconcileAnimationGpuState()) clears those same dirty flags before that path ever
  // sees them - so it must feed `changed` directly via this return value instead, or the path
  // tracer's progressive accumulation never restarts and the animated frames blend together.
  const bool interactivityAnimationApplied = updateInteractivityGraphs(cmd);

  // #DLSS instance motion vectors: snapshot the previous-frame render-node transforms BEFORE any
  // animation / gizmo edit rewrites them this frame. The previous transforms feed per-instance motion
  // vectors in the path tracer (camera + object motion combined). This is only meaningful when node
  // transforms are actually changing this frame -- animation advancing, or a pending gizmo/editor/
  // interactivity edit (markNodeDirty populates DirtyFlags::nodes during UI or the graph tick above,
  // both before this check). When nothing moves, prev == curr, so we skip both the snapshot pass and
  // the per-instance reprojection and let the path tracer produce exact camera-only motion
  // (prevRenderNodeObjectToWorld stays unbound). The snapshot reads the live render-node buffer each
  // frame independently, so skipping static frames is safe with no trailing-frame artifact. Gated on
  // DLSS specifically (not dlssGuideRequired(), which also covers OptiX) since only the DLSS path
  // consumes these vectors. See gltf_pathtrace.slang.
  m_resources.dlssInstanceMotionActive = false;
  if(m_resources.settings.renderSystem == RenderingMode::ePathtracer && m_pathTracer.isDlssEnabled())
  {
    nvvkgltf::Scene*                   scn = m_resources.getScene();
    const nvvkgltf::Scene::DirtyFlags& df  = scn->getDirtyFlags();
    const bool animActive = ui::animation::hasPlayableAnimation(scn) && m_resources.animationControl.doAnimation();
    // reconcileAnimationGpuState() (called from updateInteractivityGraphs() above whenever it applied
    // an animation/start-driven pose this tick) already cleared DirtyFlags::nodes as its own tail, so
    // `df.nodes`/`allRenderNodesDirty` can't see that motion here - interactivityAnimationApplied
    // carries it through explicitly, the same way it's OR'd into `changed` below.
    // dirtyNodesReachGeometry, not merely a dirty node: a moved light or camera moves no instance.
    const bool nodesDirty                = scn->dirtyNodesReachGeometry() || interactivityAnimationApplied;
    m_resources.dlssInstanceMotionActive = animActive || nodesDirty;

    if(m_resources.dlssInstanceMotionActive)
    {
      auto snapSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Snapshot prev transforms");
      m_resources.transformCompute.cmdSnapshotPrevObjectToWorld(cmd, m_resources.sceneVk, scn->getRenderNodes().size());
    }
  }

  // Check for changes
  bool changed{false};
  changed |= interactivityAnimationApplied;
  changed |= updateAnimation(cmd);  // Update the animation
  changed |= updateSceneChanges(cmd);
  if(changed)
  {
    resetFrame();
  }
  bool frameChanged = updateFrameCounter();  // Check if the frame counter has changed

  if(changed || frameChanged)
  {
    if(m_resources.frameCount == 0)
    {
      m_cpuTimer.reset();
      m_cpuTimePrinted = false;  // Reset print flag when rendering starts
    }

    // Update the scene frame information uniform buffer
    const glm::mat4          viewProj = m_cameraManip->getPerspectiveMatrix() * m_cameraManip->getViewMatrix();
    const VkExtent2D         gbufSize = m_resources.gBuffers.getSize();
    const glm::quat          envRot   = m_resources.settings.envRotationQuat();  // normalised
    shaderio::SceneFrameInfo finfo{
        .viewMatrix     = m_cameraManip->getViewMatrix(),
        .projInv        = glm::inverse(m_cameraManip->getPerspectiveMatrix()),
        .viewInv        = glm::inverse(m_cameraManip->getViewMatrix()),
        .viewProjMatrix = viewProj,
        .prevMVP        = m_prevMVP,
        .jitter         = {0.0f, 0.0f},
        .imageSize      = {float(gbufSize.width), float(gbufSize.height)},
        .flags =
            ((m_cameraManip->getProjectionType() == nvutils::CameraManipulator::Orthographic) ? shaderio::eSceneIsOrthographic : 0)
            | (m_resources.settings.useSolidBackground ? shaderio::eSceneUseSolidBackground : 0)
            // A baked analytic sky IS an environment image by the time the shaders see it:
            // same lat-long texture, same alias table, same prefiltered cubes. Flagging it as
            // such is what keeps sampling, MIS and raster IBL free of per-sky-type branches.
            | ((m_resources.settings.envSystem == shaderio::EnvSystem::eHdr || isBakedEnvironment(m_resources.settings.envSystem)) ?
                   shaderio::eSceneUseHdrEnvironment :
                   0)
            | ((m_resources.settings.envSystem == shaderio::EnvSystem::eNone) ? shaderio::eSceneUseNoEnvironment : 0)
            | (m_resources.settings.useInfinitePlane ? shaderio::eSceneUseInfinitePlane : 0)
            | ((m_resources.settings.useInfinitePlane && m_resources.settings.isShadowCatcher) ? shaderio::eSceneInfinitePlaneShadowCatcher : 0),
        .envRotation               = {envRot.x, envRot.y, envRot.z, envRot.w},
        .envBlur                   = m_resources.settings.hdrBlur,
        .envIntensity              = m_resources.settings.hdrEnvIntensity,
        .backgroundColor           = m_resources.settings.solidBackgroundColor,
        .visualization             = m_resources.settings.visualization,
        .infinitePlaneDistance     = m_resources.settings.infinitePlaneDistance,
        .infinitePlaneBaseColor    = m_resources.settings.infinitePlaneBaseColor,
        .infinitePlaneMetallic     = m_resources.settings.infinitePlaneMetallic,
        .infinitePlaneRoughness    = m_resources.settings.infinitePlaneRoughness,
        .shadowCatcherDarkenAmount = std::max(m_resources.settings.shadowCatcherDarkness, 0.0f),
        // Only the physical sky has a disk to draw; a null pointer is how every other environment
        // tells the background passes there is nothing to add. Declaration order matters here --
        // these sit before envSkyType in SceneFrameInfo.
        .sunDisk      = (m_resources.settings.envSystem == shaderio::EnvSystem::eSky) ?
                            reinterpret_cast<shaderio::SkySunDisk*>(m_envBaker.sunDiskAddress()) :
                            nullptr,
        .sunDirection = m_resources.sunDirection,
        // The sky's own sun, supplied only while the scene has no marked one. Selecting an
        // environment must not edit the user's scene, so nothing is added to the graph until a
        // save asks for it; from then on the marked light is the sun and this stays off.
        .rendererSunLight = (skyHasSun(m_resources.settings.envSystem) && m_skySun.lightIndex() < 0) ? 1 : 0,
        .sunLightIndex    = skyHasSun(m_resources.settings.envSystem) ? m_skySun.lightIndex() : -1,
        .envSkyType       = authoredSkyType(m_resources.settings.envSystem),
        .skyOmi           = buildSkyOmiParameters(),
        // Aerial perspective. Physical sky only (the sunDisk gate above says which), and the
        // altitude is the same fixed observer the bake uses -- metres in the settings because that
        // is what a person types, kilometres here because that is what the atmosphere works in.
        .aerialPerspectiveScale = m_resources.settings.atmoAerialPerspectiveScale,
        .observerAltitudeKm     = aerialPerspectiveEyeAltitudeKm(),
        // The gradient sky's sun. Zero for every other environment: the physical sky's sun comes
        // from `sunDisk` above, and no other sky has one.
        .skySunIlluminance = (m_resources.settings.envSystem == shaderio::EnvSystem::eGradient) ? gradientSunIlluminance() :
                                                                                                  glm::vec3(0.0F),
        .envSamplingGridWidth  = m_resources.hdrIbl.getSamplingGrid().width,
        .envSamplingGridHeight = m_resources.hdrIbl.getSamplingGrid().height,
    };
    // Update the camera information
    m_prevMVP = finfo.viewProjMatrix;

    vkCmdUpdateBuffer(cmd, m_resources.bFrameInfo.buffer, 0, sizeof(shaderio::SceneFrameInfo), &finfo);
    // Update the sky
    m_resources.sunYIsUp = m_cameraManip->getUp().y > 0.5f;
    // Make sure buffer is ready to be used
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Switch between renderers based on the current mode
    switch(m_resources.settings.renderSystem)
    {
      case RenderingMode::ePathtracer:
        m_pathTracer.onRender(cmd, m_resources);
        break;
      case RenderingMode::eRasterizer:
        m_rasterizer.onRender(cmd, m_resources);
        break;
    }
  }
  else
  {
    // Print CPU time only once after render completes
    if(!m_cpuTimePrinted)
    {
      LOGI("Rendering finished: %f ms\n", m_cpuTimer.getMilliseconds());
      m_cpuTimePrinted = true;
    }
  }

  // Apply the post-processing effects
  const bool nrAllowed = tonemap(cmd);
#if defined(USE_DLSS) && defined(USE_DLSSNR)
  // DLSS-NR: enhance eImgTonemapped (LDR) after tonemapping; NR expects display-range input.
  // Skip on guide buffers, visualization debug modes, and bypassed-tonemapper paths.
  if(nrAllowed)
    if(Dlss* dlss = activeDlss())
      if(dlss->isNrActive())
      {
        auto nrSection = m_profilerGpuTimer.cmdFrameSection(cmd, "DLSS-NR");
        dlss->evaluateNr(cmd);
      }
#endif
  silhouette(cmd);
  // KHR_interactivity hover detection: record this frame's cursor readback after silhouette's own
  // read of eImgSelection (see HoverPicker::requestReadback's comment for why that ordering avoids
  // any extra synchronization). Skipped entirely when the cursor isn't over the viewport.
  if(m_hoverCursorInViewport)
    m_hoverPicker.requestReadback(cmd, m_resources, m_hoverCursorPixel);
  // Async click ray-pick: record the pick dispatch on this frame's own command buffer - the TLAS is
  // already proven valid here (the path tracer/rasterizer just used it above) - then hand off a
  // frame-semaphore to poll instead of blocking. See m_pendingClickPick's comment (renderer.hpp).
  if(m_pendingClickPick)
  {
    m_rayPicker.run(cmd, {.modelViewInv   = m_pendingClickPick->modelViewInv,
                          .perspectiveInv = m_pendingClickPick->perspectiveInv,
                          .isOrthographic = m_pendingClickPick->isOrthographic,
                          .pickPos        = m_pendingClickPick->pickPos,
                          .tlas           = m_resources.sceneRtx.topLevelAS()});
    m_pendingClickResult =
        PendingClickResult{.semaphoreState = nvvk::SemaphoreState::makeFixed(m_app->getFrameSignalSemaphore()),
                           .isDoubleClick  = m_pendingClickPick->isDoubleClick};
    m_pendingClickPick.reset();
  }
  if(!isAutomatedRun())
  {
    renderVisualHelpers(cmd);
  }

  m_benchmark.updateHeadlessProgressIfNeeded(benchmarkFrameInfo());
}

//--------------------------------------------------------------------------------------------------
// Per-sequence benchmark memory report (parsed by utils/benchmark/benchmark.py)
void GltfRenderer::benchmarkAdvance(const nvutils::ParameterSequencer::State& state)
{
  (void)state;
  m_benchmark.emitSequenceMemory(benchmarkMemorySamples());
}


//--------------------------------------------------------------------------------------------------
// Render the UI menu: File, Tools, Renderer
void GltfRenderer::onUIMenu()
{
  renderMenu();
}

//--------------------------------------------------------------------------------------------------
// Called with headless rendering, to save the final image
void GltfRenderer::onLastHeadlessFrame()
{
  m_benchmark.logHeadlessSummary(benchmarkFrameInfo());
  saveHeadlessOutputImage();
  m_benchmark.finishHeadlessTiming();
}

//--------------------------------------------------------------------------------------------------
// Add another glTF into the current scene, either embedded (merge) or linked (glTF 2.1 external
// asset reference). Shared worker for onMergeScene()/onReferenceScene(): the two modes differ only
// in the Scene call and the status text, so the threading + post-import rebuild live here.
//
void GltfRenderer::addSceneFromFile(const std::filesystem::path& filename, bool asReference)
{
  // Bootstrap a scene when nothing is loaded (or the current one is invalid). This covers BOTH the
  // merge and reference paths below -- they call getScene()->mergeScene()/referenceScene(), so this
  // guarantees a non-null scene for either. (Both mergeScene() and referenceScene() also ensure a
  // scene container on an empty model, so the imported content has a root.)
  if(!m_resources.getScene() || !m_resources.getScene()->valid())
  {
    auto scn = std::make_unique<nvvkgltf::Scene>();
    scn->supportedExtensions().insert(EXT_TEXTURE_WEBP_EXTENSION_NAME);
    m_resources.scene = std::move(scn);
  }

  if(m_busy.isBusy())
    return;

  // Ensure no in-flight frame still references GPU resources (incl. per-frame staging buffers) that
  // the upcoming rebuild will destroy/replace. The drag-drop path idles in onFileDrop, but the File
  // menu entry points reach here directly, so idle here to cover both. (main thread; see VUID-00922)
  vkQueueWaitIdle(m_app->getQueue(0).queue);

  // Set busy BEFORE starting the worker thread to prevent UI access during scene modification
  m_busy.start(asReference ? "Referencing Scene" : "Merging Scene");

  std::thread([=, this]() {
    nvvkgltf::Scene* scene = m_resources.getScene();
    // Capture BEFORE the import mutates the model: merge/reference is a pure tail-append, so when the base
    // already has images and textures we preserve them on the GPU and load only the new tail
    // (eMergeAppend). An empty / texture-less base carries 1x1 GPU dummy defaults whose sizes don't match
    // the model, so it still needs a full rebuild.
    const bool wasTextured = scene && !scene->getModel().images.empty() && !scene->getModel().textures.empty();

    const int         nodeIdx = asReference ? scene->referenceScene(filename) :
                                              scene->mergeScene(filename, static_cast<uint32_t>(m_maxTextures));
    const std::string name    = nvutils::utf8FromPath(filename.filename());
    if(nodeIdx >= 0)
    {
      m_undoStack.clear();
      rebuildVulkanSceneInternal(wasTextured ? nvvkgltf::SceneGpu::RebuildMode::eMergeAppend : nvvkgltf::SceneGpu::RebuildMode::eFull);
      // Imported glTF may bring extensions the previous scene didn't use; recompute so optimal-mode
      // rebuilds the shader if the feature set widened.
      m_resources.recomputeSceneFeatures(dlssGuideRequired());
      resetFrame();
      m_sceneSelection.selectNode(nodeIdx);
      m_sceneBrowser.focusOnSelection();
      LOGI("Scene %s successfully: %s\n", asReference ? "referenced" : "merged", name.c_str());
    }
    else
    {
      LOGE("Failed to %s scene: %s\n", asReference ? "reference" : "merge", name.c_str());
    }
    m_busy.stop();
  }).detach();
}

//--------------------------------------------------------------------------------------------------
// Merge (embed) a glTF scene into the current one. Called from File menu or Shift+drag-drop.
//
void GltfRenderer::onMergeScene(const std::filesystem::path& filename)
{
  addSceneFromFile(filename, /*asReference=*/false);
}

//--------------------------------------------------------------------------------------------------
// glTF 2.1: add a scene as a referenced external asset (read-only, re-externalized on save) instead
// of embedding it. Called from File>Reference Scene... or Ctrl+Shift+drag-drop.
//
void GltfRenderer::onReferenceScene(const std::filesystem::path& filename)
{
  addSceneFromFile(filename, /*asReference=*/true);
}

//--------------------------------------------------------------------------------------------------
// Load a glTF scene or an HDR file (called from both Load Scene and Load HDR Environment menu items)
// Shift+drag-drop merges the file into the current scene instead of replacing it.
//
void GltfRenderer::onFileDrop(const std::filesystem::path& filename)
{
  // Sky presets first: a `.sky.json` is also a `.json`, and the descriptor test below is a suffix
  // match. Cheap and self-contained -- no scene teardown, so no queue wait (a panorama preset goes
  // through createHDR, which waits for itself) -- but still refused while a worker load is in
  // flight, since that load will apply its own sky when it lands.
  if(sky_preset::isPresetPath(filename))
  {
    if(!m_busy.isBusy())
      loadSkyPreset(filename);
    return;
  }

  // SYNC NOTE: User-initiated file load/merge — wait ensures GPU is idle before scene teardown/rebuild.
  vkQueueWaitIdle(m_app->getQueue(0).queue);

  bool isDescriptor = filename.string().ends_with(".scene.json");
  if(isDescriptor)
  {
    if(m_busy.isBusy())
      return;

    m_loadPipeline.clear();
    cleanupScene();  // also frees rasterizer record cmd + clears sort state via onSceneInvalidated()

    m_busy.start("Loading Descriptor");
    m_imageSaveFilename = std::filesystem::path(filename.stem()).replace_extension(".jpg");
    std::thread([=, this]() {
      m_lastSceneDirectory = filename.parent_path();
      createSceneFromDescriptor(filename);
      m_busy.stop();
    }).detach();
  }
  else if(nvutils::extensionMatches(filename, ".gltf") || nvutils::extensionMatches(filename, ".glb")
          || nvutils::extensionMatches(filename, ".obj"))
  {
    if(m_busy.isBusy())
      return;

    // Shift state: when dropping from another app, our window often doesn't have focus so
    // ImGui/GLFW don't see the key. On Windows use GetKeyState (state when the drop message
    // was generated) so it works for Explorer, Everything, and other DnD sources.
    bool shiftHeld = false;
    bool ctrlHeld  = false;
#if defined(_WIN32)
    shiftHeld = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
    ctrlHeld  = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
#else
    shiftHeld = ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift);
    ctrlHeld  = ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl);
#endif

    if(shiftHeld && ctrlHeld)
    {
      // glTF 2.1: Ctrl+Shift+drop references the file as an external asset (read-only) instead of embedding.
      onReferenceScene(filename);
    }
    else if(shiftHeld)
    {
      onMergeScene(filename);
    }
    else
    {
      m_loadPipeline.clear();
      cleanupScene();  // also frees rasterizer record cmd + clears sort state via onSceneInvalidated()

      // Set busy BEFORE starting the worker thread to prevent re-entrant drops
      m_busy.start("Loading");
      m_imageSaveFilename = std::filesystem::path(filename.stem()).replace_extension(".jpg");

      std::thread([=, this]() {
        m_lastSceneDirectory = filename.parent_path();
        createScene(filename);
        m_busy.stop();
      }).detach();
    }
  }
  else if(isEnvImageExtension(filename))
  {
    m_lastHdrDirectory = filename.parent_path();
    createHDR(filename);
    m_resources.settings.envSystem = shaderio::EnvSystem::eHdr;
    calibrateFireflyClamp();
  }

  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Save the scene
bool GltfRenderer::save(const std::filesystem::path& filename, bool selfContained)
{
  if(m_resources.getScene() && m_resources.getScene()->valid() && !filename.empty())
  {
    std::vector<nvvkgltf::RenderCamera> cameras = nvvkgltf::getCamerasFromWidget();

    // Replace the first camera with the current view
    if(!cameras.empty())
    {
      nvvkgltf::RenderCamera& camera = cameras[0];
      m_cameraManip->getLookat(camera.eye, camera.center, camera.up);
      camera.znear = m_cameraManip->getClipPlanes().x;
      camera.zfar  = m_cameraManip->getClipPlanes().y;

      if(m_cameraManip->getProjectionType() == nvutils::CameraManipulator::Orthographic)
      {
        camera.type = nvvkgltf::RenderCamera::CameraType::eOrthographic;
        camera.xmag = static_cast<double>(m_cameraManip->getOrthographicXmag());
        camera.ymag = static_cast<double>(m_cameraManip->getOrthographicYmag());
      }
      else
      {
        camera.type = nvvkgltf::RenderCamera::CameraType::ePerspective;
        camera.yfov = glm::radians(m_cameraManip->getFov());
      }

      // Set all cameras
      m_resources.getScene()->setSceneCameras(cameras);
    }

    // Install the environment-sky writer for this save. Done here rather than at scene creation
    // so that no load path can forget it -- there is exactly one place a scene is saved.
    m_resources.getScene()->setPreSaveHook([this](tinygltf::Model& outModel, const std::filesystem::path& destination) {
      writeEnvironmentSky(outModel, destination);
    });

    // Saving the scene
    return m_resources.getScene()->save(filename, selfContained);
  }
  return false;
}

#if defined(USE_DLSS)
//--------------------------------------------------------------------------------------------------
// Returns the Dlss instance owned by whichever renderer is currently active.
Dlss* GltfRenderer::activeDlss()
{
  return (m_resources.settings.renderSystem == RenderingMode::ePathtracer) ? m_pathTracer.getDlss() : m_rasterizer.getDlss();
}

const Dlss* GltfRenderer::activeDlss() const
{
  return (m_resources.settings.renderSystem == RenderingMode::ePathtracer) ? m_pathTracer.getDlss() : m_rasterizer.getDlss();
}
#endif

//--------------------------------------------------------------------------------------------------
// Runtime gate for path-tracer guide-buffer shader code.
bool GltfRenderer::dlssGuideRequired() const
{
  if(m_resources.settings.renderSystem != RenderingMode::ePathtracer)
    return false;

#if defined(USE_DLSS)
  if(m_pathTracer.isDlssEnabled())
    return true;
#endif

#if defined(USE_OPTIX_DENOISER)
  if(const OptiXDenoiser* optix = m_pathTracer.getOptiXDenoiser(); optix && optix->isEnabled())
    return true;
#endif

  return false;
}

//--------------------------------------------------------------------------------------------------
// Adopt the sky authored by the scene that was just loaded.
//
// Precedence is glTF over ini, and only for the sky itself: a file that carries
// OMI_environment_sky is stating what its environment is, so it overwrites the corresponding
// settings and they persist to the ini as usual (reopening the same file lands where the user
// left off). A file without the extension changes nothing.
//
// Only plain and gradient are adopted as the active mode. A panorama or physical sky is parsed --
// and, importantly, kept verbatim so a later save does not drop it -- but the renderer stays on
// whatever environment it was showing, because neither type is produced by the baker yet.
//
void GltfRenderer::applyEnvironmentSkyFromScene()
{
  nvvkgltf::Scene* scene = m_resources.getScene();
  if(scene == nullptr || !scene->valid())
    return;

  const tinygltf::Model& model      = scene->getModel();
  const int              sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;

  // Parse now, apply at the next frame top. The settings cannot be written here: on start-up the
  // ini is restored *after* the command-line scene load, and the restore writes envSystem
  // straight into storage -- so anything set here is silently overwritten and the file's sky
  // loses to the ini, which is backwards. applyPendingAuthoredSky() runs after the restore hooks
  // and re-asserts it.
  m_environmentSky        = gltf_environment_sky::parse(model, sceneIndex);
  m_environmentSkyPending = m_environmentSky.has_value();
}

//--------------------------------------------------------------------------------------------------
// Push the authored sky into the live settings. Runs at the frame top, after the ini restore, so
// that "glTF wins over ini" holds regardless of which order the two landed in.
//
void GltfRenderer::applyPendingAuthoredSky()
{
  if(!m_environmentSkyPending || !m_environmentSky.has_value())
    return;
  m_environmentSkyPending      = false;
  const nvvkgltf::Scene* scene = m_resources.getScene();
  applySkyDescriptor(*m_environmentSky, SkySource::eScene, scene ? scene->getFilename().parent_path() : std::filesystem::path{});
}

//--------------------------------------------------------------------------------------------------
// Push an authored sky into the live settings. See the declaration for what `source` decides.
//
void GltfRenderer::applySkyDescriptor(const SkyDescriptor& sky, SkySource source, const std::filesystem::path& uriBase)
{
  Settings& settings = m_resources.settings;

  // An explicit `--envSystem` is a statement about this run, and it outranks the file: the same
  // rule the .ini already follows, extended to the place it should have reached first. Without
  // this a script that forces an environment silently gets whatever the scene authored instead,
  // and the measurement it was taking is of something else.
  //
  // Only the *selection* is suppressed. The sky's parameters are still applied, so forcing the
  // mode a scene authored still picks up its colours, and `m_environmentSky` keeps the descriptor
  // either way so a later save preserves it.
  //
  // A preset is the exception, and obviously so: loading one by hand at runtime *is* the statement,
  // and a flag from start-up has no business overruling it.
  // OMI's `rotation` is the renderer's own, quaternion for quaternion, tilt included.
  //
  // Only applied when the file actually states one. The rotation is viewer state -- the user, the
  // command line or MCP may have set it -- and loadScene() leaves viewer state alone on purpose; a
  // scene that says nothing about orientation must not silently reset it. A scene that does state
  // one is making a statement about its own sky, and that wins, like every other authored sky value
  // here. The sun is left where it is: a file's sun is its marked light and a preset's is its
  // sunRotation, both already in world space.
  if(sky.rotationAuthored)
    settings.envRotation = {sky.rotation.x, sky.rotation.y, sky.rotation.z, sky.rotation.w};

  const bool forcedOnCommandLine = source == SkySource::eScene && m_parameterParser && m_parameterParser->wasParsed("envSystem");
  const auto selectEnvironment = [&](shaderio::EnvSystem env) {
    if(forcedOnCommandLine)
    {
      if(settings.envSystem != env)
      {
        LOGI("OMI_environment_sky: scene authors a different environment; --envSystem wins for this run.\n");
      }
      return;
    }
    settings.envSystem = env;
  };

  switch(sky.type)
  {
    case SkyDescriptor::Type::ePlain:
      settings.plainColor = sky.plain.color;
      selectEnvironment(shaderio::EnvSystem::ePlain);
      break;
    case SkyDescriptor::Type::eGradient:
      settings.gradientBottomColor  = sky.gradient.bottomColor;
      settings.gradientHorizonColor = sky.gradient.horizonColor;
      settings.gradientTopColor     = sky.gradient.topColor;
      settings.gradientBottomCurve  = sky.gradient.bottomCurve;
      settings.gradientTopCurve     = sky.gradient.topCurve;
      settings.gradientSunAngleMax  = sky.gradient.sunAngleMax;
      settings.gradientSunCurve     = sky.gradient.sunCurve;
      selectEnvironment(shaderio::EnvSystem::eGradient);
      break;
    case SkyDescriptor::Type::ePanorama: {
      // The image is an external asset, so the URI resolves against the file that named it -- the
      // glTF or the preset, the same rule every other external asset follows
      // (docs/external_assets.md), and NOT against the resources directories, which would silently
      // load someone else's environment when two scenes name the same file.
      if(sky.panorama.uri.empty())
      {
        LOGW("OMI_environment_sky: panorama sky has no NV_environment_sky_panorama.uri%s; keeping the current environment.\n",
             sky.panorama.equirectangular >= 0 ? " (its `equirectangular` texture index is not supported)" : "");
        break;
      }
      const std::filesystem::path resolved =
          uriBase.empty() ? std::filesystem::path(sky.panorama.uri) : uriBase / std::filesystem::path(sky.panorama.uri);
      if(!std::filesystem::exists(resolved))
      {
        LOGW("OMI_environment_sky: panorama image not found: %s\n", nvutils::utf8FromPath(resolved).c_str());
        break;
      }
      createHDR(resolved);
      selectEnvironment(shaderio::EnvSystem::eHdr);
      break;
    }
    case SkyDescriptor::Type::ePhysical:
      // The inverse of the split on save: colour times coefficient, m^-1 back to km^-1.
      settings.atmoRayleighScattering = sky.physical.rayleighColor * (sky.physical.rayleighCoefficient * 1000.0F);
      settings.atmoMieScattering      = sky.physical.mieColor * (sky.physical.mieCoefficient * 1000.0F);
      settings.atmoMieAnisotropy      = sky.physical.mieAnisotropy;
      settings.atmoGroundAlbedo       = sky.physical.groundColor;

      // Then the NV overlay, for everything OMI cannot say. Only when the file carried one: a
      // scene with just the OMI block has not said what its planet is, so leaving the current
      // values alone is more honest than forcing Earth on it.
      if(sky.atmosphere.present)
      {
        const SkyDescriptor::Atmosphere& a = sky.atmosphere;
        settings.atmoSolarIrradiance       = a.solarIrradiance;
        settings.atmoSunAngularRadius      = a.sunAngularRadius;
        settings.atmoMieAlbedo             = a.mieAlbedo;
        // Metres in the file, kilometres in the model. Divided rather than multiplied by 0.001:
        // the reciprocal is not representable in binary, so scaling by it loses a few ulps on
        // every load, and a scene saved and reopened repeatedly would creep. Dividing by an exact
        // power-of-ten float keeps radii like 3389.5 km landing back on themselves.
        settings.atmoOzoneExtinction     = a.ozoneExtinction * 1000.0F;
        settings.atmoRayleighScaleHeight = a.rayleighScaleHeight / 1000.0F;
        settings.atmoMieScaleHeight      = a.mieScaleHeight / 1000.0F;
        settings.atmoOzoneCenter         = a.ozoneCenter / 1000.0F;
        settings.atmoOzoneWidth          = a.ozoneWidth / 1000.0F;
        settings.atmoBottomRadius        = a.planetRadius / 1000.0F;
        settings.atmoThickness           = a.atmosphereThickness / 1000.0F;
      }
      selectEnvironment(shaderio::EnvSystem::eSky);
      break;
    default:
      assert(false && "unhandled SkyDescriptor::Type");
      break;
  }

  // No calibrateFireflyClamp() here: hdrIbl still describes the previous environment until the
  // refresh at the next frame top, which calibrates when the environment kind changed.
  onEnvironmentChanged();
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// The URI to record for an environment image, relative to the glTF being written when that is
// possible and absolute when it is not.
//
// Relative is what makes a scene and its environment movable as a pair, which is the common case.
// It stops being the right answer once the image lives outside the glTF's tree -- on another
// drive, or in a shared asset library -- where a relative path would be a long chain of "..\"
// that breaks the moment either end moves. Absolute is honest there.
//
std::string GltfRenderer::relativeUriForSave(const std::filesystem::path& asset, const std::filesystem::path& destination) const
{
  const std::filesystem::path base = destination.parent_path();
  if(base.empty())
    return nvutils::utf8FromPath(asset);

  std::error_code             ec;
  const std::filesystem::path rel = std::filesystem::relative(asset, base, ec);
  if(ec || rel.empty() || *rel.begin() == "..")
    return nvutils::utf8FromPath(asset);

  // glTF URIs use forward slashes regardless of platform.
  std::string uri = nvutils::utf8FromPath(rel);
  std::replace(uri.begin(), uri.end(), '\\', '/');
  return uri;
}

//--------------------------------------------------------------------------------------------------
// Write the current sky into the model being serialized. Runs from Scene::save, on the copy that
// is actually written to disk.
//
// With the toggle off we strip rather than skip: a scene loaded with a sky and saved with the
// toggle off should not keep an extension that no longer matches what the user is looking at.
//
// Reconcile the scene's marked sun light with the sky being written.
//
// OMI_environment_sky describes a medium and says to add suns with KHR_lights_punctual, so a sky
// with a sun is only half-written without one: another renderer would build the atmosphere, find
// nothing to light it, and show an unlit sky. That failure is invisible here, because this
// renderer supplies its own sun when the scene has none -- which is exactly why it has to be
// fixed at the point of writing.
//
// The light is marked (gltf_environment_sky.hpp) so a reload knows which light is the sky's
// rather than guessing, and so this function knows which light is its own to manage. Any other
// light in the file is untouched.
//
// Intensity is the solar illuminance *above* the atmosphere, not the attenuated value this
// renderer shows. The atmosphere travels in the same file; a reader that applies it to an already
// attenuated sun would attenuate twice.
//
void GltfRenderer::writeSkySunLight(tinygltf::Model& outModel) const
{
  const int  existingNode = gltf_environment_sky::findSkySunNode(outModel);
  const bool wantSun      = m_resources.settings.envSaveToGltf && skyHasSun(m_resources.settings.envSystem);

  if(!wantSun)
  {
    // The sky being written has no sun, so nothing should still claim to be one.
    if(existingNode >= 0)
    {
      tinygltf::Node& node = outModel.nodes[existingNode];
      const bool      ours = gltf_environment_sky::isRendererOwnedSun(node);

      tinygltf::Value::Object extras = node.extras.Get<tinygltf::Value::Object>();
      extras.erase(gltf_environment_sky::kSkySunMarkerKey);
      node.extras = tinygltf::Value(std::move(extras));

      if(ours)
      {
        // A sun this renderer invented for an earlier save. Withdrawing it is what keeps the file
        // describing what is on screen.
        //
        // The node is emptied rather than erased: deleting it would renumber every node index in
        // the file. The light entry is left too, for the same reason -- compaction collects both
        // if the user asks for it.
        LOGI("OMI_environment_sky: withdrawing the sun light this renderer added; the sky being saved has none.\n");
        node.light = -1;
        node.extensions.erase("KHR_lights_punctual");
      }
      else
      {
        // Someone else's light that was chosen as the sun. It stops being the sun and stays
        // exactly as it was -- unpicking it is not the same as deleting it.
        LOGI("OMI_environment_sky: '%s' is no longer the sky's sun; the light itself is unchanged.\n", node.name.c_str());
      }
    }
    return;
  }

  // Colour and illuminance of the sun this sky implies, in the units the light carries.
  const shaderio::SkyAtmosphereParameters atmosphere = buildAtmosphereParameters();
  glm::vec3                               illuminance(0.0F);
  float                                   angularRadius = atmosphere.sunAngularRadius;
  if(m_resources.settings.envSystem == shaderio::EnvSystem::eSky)
  {
    // Above the atmosphere on purpose -- see the note below on double attenuation. Shared with the
    // firefly clamp, which wants the same number for its own reasons.
    illuminance = skySunIlluminanceAboveAtmosphere();
  }
  else
  {
    // The gradient sky's sun is drawn rather than measured, so there is no disk radiance to
    // convert; the illuminance is scaled against the dome instead. Same number the renderer lights
    // with, so the file describes what is on screen.
    angularRadius = kGradientSunAngularRadius;
    illuminance   = gradientSunIlluminance() * m_resources.settings.hdrEnvIntensity;
  }

  const float     peak  = std::max({illuminance.x, illuminance.y, illuminance.z});
  const glm::vec3 color = (peak > 0.0F) ? illuminance / peak : glm::vec3(1.0F);

  // Rotation taking +Z to the sun: glTF aims a light down -Z, and the renderer's sunDirection
  // points at it. A world rotation; turned into the node's local one below.
  const glm::vec3 toSun = glm::normalize(m_resources.sunDirection);
  const glm::quat rot   = glm::rotation(glm::vec3(0.0F, 0.0F, 1.0F), toSun);

  int nodeIdx = existingNode;
  if(nodeIdx < 0)
  {
    tinygltf::Node node;
    node.name = "Sun";
    gltf_environment_sky::setSkySunMarker(node, gltf_environment_sky::kSkySunOwnerRenderer);
    outModel.nodes.push_back(node);
    nodeIdx = static_cast<int>(outModel.nodes.size()) - 1;

    const int sceneIndex = outModel.defaultScene >= 0 ? outModel.defaultScene : 0;
    if(sceneIndex < static_cast<int>(outModel.scenes.size()))
      outModel.scenes[sceneIndex].nodes.push_back(nodeIdx);
  }

  // A sun the user picked may sit under a rotated parent; the one created above is a root.
  const glm::quat local = gltf_environment_sky::localRotationForWorld(outModel, nodeIdx, rot);
  tinygltf::Node& node  = outModel.nodes[nodeIdx];
  if(!node.matrix.empty())
  {
    // A matrix would win over the TRS set below, so keep its translation and scale as TRS.
    glm::vec3 t, s;
    glm::quat r;
    tinygltf::utils::getNodeTRS(node, t, r, s);
    node.translation = {t.x, t.y, t.z};
    node.scale       = {s.x, s.y, s.z};
    node.matrix.clear();
  }
  node.rotation = {local.x, local.y, local.z, local.w};

  // Reuse the light this node already points at, so a re-save does not add one per save.
  //
  // `node.light` rather than the extensions map: that is the canonical reference in this codebase
  // -- tinygltf mirrors the extension into it on load and serialises from it on save, which is why
  // SceneEditor::addLightNode writes only this field.
  int        lightIdx    = node.light;
  const bool createdHere = (lightIdx < 0 || lightIdx >= static_cast<int>(outModel.lights.size()));
  if(createdHere)
  {
    outModel.lights.push_back({});
    lightIdx = static_cast<int>(outModel.lights.size()) - 1;
  }
  node.light = lightIdx;

  tinygltf::Light& light = outModel.lights[lightIdx];
  // Name only what we create. Adopting someone's "Directional Light" and calling it "Sun" is
  // their scene being edited by a viewer, which is the thing this whole design avoids.
  if(createdHere)
    light.name = "Sun";
  light.type      = "directional";
  light.color     = {color.x, color.y, color.z};
  light.intensity = peak;

  tinygltf::utils::syncExtensionsUsed(outModel);
}

void GltfRenderer::writeEnvironmentSky(tinygltf::Model& outModel, const std::filesystem::path& destination) const
{
  if(!m_resources.settings.envSaveToGltf)
  {
    gltf_environment_sky::strip(outModel);
    writeSkySunLight(outModel);  // drops the sun too: the environment is not part of this file
    return;
  }

  const EnvironmentState state = skyDescriptorFromSettings(destination);
  if(!state.has_value())
  {
    gltf_environment_sky::strip(outModel);
    writeSkySunLight(outModel);  // no sky, so no sun either: withdraw the marker it would leave behind
    return;
  }

  const int sceneIndex = outModel.defaultScene >= 0 ? outModel.defaultScene : 0;
  gltf_environment_sky::write(outModel, sceneIndex, *state);
  writeSkySunLight(outModel);
}

//--------------------------------------------------------------------------------------------------
// Capture the current sky as a `.sky.json`, and apply one.
//
// The sun angle travels with it, because a sky and the angle of its sun are one look. On the way
// out it is read from the direction the renderer is actually using, whichever of the four things
// set it; on the way in it is applied through m_skySun.aim(), so a scene whose sun is a marked light
// records the move on its undo stack exactly as a slider drag would.
//
// Note what is *not* undoable: everything else a preset changes is viewer state -- the atmosphere,
// the ambient terms, the environment type -- and no environment control in this renderer is on the
// undo stack. Putting presets there alone would mean a settings-snapshot command unlike anything
// else in undo_redo, for a consistency the panel does not otherwise have.
//
bool GltfRenderer::saveSkyPreset(const std::filesystem::path& path) const
{
  const EnvironmentState sky = skyDescriptorFromSettings(path);
  if(!sky.has_value())
  {
    LOGW("Sky preset: the current environment is not a sky that can be written (%s).\n", nvutils::utf8FromPath(path).c_str());
    return false;
  }

  sky_preset::Preset preset{.sky = *sky};
  if(skyHasSun(m_resources.settings.envSystem))
  {
    // glTF aims a light down -Z, so the rotation that takes +Z to the sun is what a node would
    // store -- the same convention SkySun::setMarkedDirection writes.
    preset.sunRotation = glm::rotation(glm::vec3(0.0F, 0.0F, 1.0F), glm::normalize(m_resources.sunDirection));
  }
  return sky_preset::save(path, preset);
}

bool GltfRenderer::loadSkyPreset(const std::filesystem::path& path)
{
  const std::optional<sky_preset::Preset> preset = sky_preset::load(path);
  if(!preset.has_value())
    return false;

  // A preset loaded after a scene supersedes that scene's sky, even one still waiting for the frame
  // top -- otherwise the scene's would land a frame later and silently undo this.
  m_environmentSkyPending = false;
  applySkyDescriptor(preset->sky, SkySource::ePreset, path.parent_path());

  // Absent means "leave the sun alone", which is why this is not an else.
  if(preset->sunRotation.has_value())
  {
    m_skySun.aim(*preset->sunRotation * glm::vec3(0.0F, 0.0F, 1.0F));
    m_skySun.syncAngles();
  }

  // The preset is now what a save would write, so a scene saved after loading one carries it.
  m_environmentSky = preset->sky;
  return true;
}

void GltfRenderer::queueStartupSkyPresets(const std::filesystem::path& load, const std::filesystem::path& save)
{
  m_pendingLoadSkyPreset = load;
  m_pendingSaveSkyPreset = save;
}

void GltfRenderer::applyPendingSkyPresets()
{
  if(!m_pendingLoadSkyPreset.empty())
    loadSkyPreset(std::exchange(m_pendingLoadSkyPreset, {}));
  if(!m_pendingSaveSkyPreset.empty())
    saveSkyPreset(std::exchange(m_pendingSaveSkyPreset, {}));
}

//--------------------------------------------------------------------------------------------------
// The live settings as an authored sky. See the declaration for what nullopt means.
//
EnvironmentState GltfRenderer::skyDescriptorFromSettings(const std::filesystem::path& destination) const
{
  const Settings& settings = m_resources.settings;

  // Start from the descriptor the scene was loaded with, so unmodelled keys and any sky type this
  // build does not render survive the round trip. A scene that authored nothing starts fresh.
  SkyDescriptor sky = m_environmentSky.value_or(SkyDescriptor{});

  // The environment's orientation is the sky's, the same quaternion, so it is written exactly.
  sky.rotation         = settings.envRotationQuat();
  sky.rotationAuthored = true;

  switch(settings.envSystem)
  {
    case shaderio::EnvSystem::ePlain:
      // A sky type this build does not recognize loads as this plain fallback (see
      // SkyDescriptor::unknownType). While the fallback is still on screen untouched, the file's
      // own sky is the truer thing to save; every branch that authors a sky of its own drops it.
      if(settings.plainColor != sky.plain.color)
        sky.unknownType.clear();
      sky.type        = SkyDescriptor::Type::ePlain;
      sky.plain.color = settings.plainColor;
      break;
    case shaderio::EnvSystem::eGradient:
      sky.unknownType.clear();
      sky.type                  = SkyDescriptor::Type::eGradient;
      sky.gradient.bottomColor  = settings.gradientBottomColor;
      sky.gradient.horizonColor = settings.gradientHorizonColor;
      sky.gradient.topColor     = settings.gradientTopColor;
      sky.gradient.bottomCurve  = settings.gradientBottomCurve;
      sky.gradient.topCurve     = settings.gradientTopCurve;
      sky.gradient.sunAngleMax  = settings.gradientSunAngleMax;
      sky.gradient.sunCurve     = settings.gradientSunCurve;
      break;
    case shaderio::EnvSystem::eSky: {
      // OMI describes the atmosphere as a magnitude plus a colour, and in m^-1; the model keeps a
      // per-channel coefficient in km^-1. Splitting on the largest channel is what makes the pair
      // reproduce the extension's own defaults in shape -- its rayleighColor is (0.3, 0.5, 1.0),
      // blue-dominant and normalised to 1 -- and it is exactly invertible, which is the property
      // the round trip depends on.
      const auto split = [](const glm::vec3& perChannelPerKm, float& outCoefficient, glm::vec3& outColor) {
        const float peak = std::max({perChannelPerKm.x, perChannelPerKm.y, perChannelPerKm.z});
        // Divided, not scaled by 0.001: the reciprocal is not representable in binary, so scaling
        // by it costs a few ulps and the value no longer lands back on itself when the file is
        // reopened and saved again.
        outCoefficient = peak / 1000.0F;  // km^-1 -> m^-1, as the extension specifies
        outColor       = (peak > 0.0F) ? perChannelPerKm / peak : glm::vec3(1.0F);
      };

      sky.unknownType.clear();
      sky.type = SkyDescriptor::Type::ePhysical;
      split(settings.atmoRayleighScattering, sky.physical.rayleighCoefficient, sky.physical.rayleighColor);
      split(settings.atmoMieScattering, sky.physical.mieCoefficient, sky.physical.mieColor);
      sky.physical.mieAnisotropy = settings.atmoMieAnisotropy;
      sky.physical.groundColor   = settings.atmoGroundAlbedo;

      // Everything OMI has no field for, in the sibling NV block. Kilometres in the model, metres
      // in the file.
      sky.atmosphere.present             = true;
      sky.atmosphere.solarIrradiance     = settings.atmoSolarIrradiance;
      sky.atmosphere.sunAngularRadius    = settings.atmoSunAngularRadius;
      sky.atmosphere.mieAlbedo           = settings.atmoMieAlbedo;
      sky.atmosphere.ozoneExtinction     = settings.atmoOzoneExtinction / 1000.0F;
      sky.atmosphere.rayleighScaleHeight = settings.atmoRayleighScaleHeight * 1000.0F;
      sky.atmosphere.mieScaleHeight      = settings.atmoMieScaleHeight * 1000.0F;
      sky.atmosphere.ozoneCenter         = settings.atmoOzoneCenter * 1000.0F;
      sky.atmosphere.ozoneWidth          = settings.atmoOzoneWidth * 1000.0F;
      sky.atmosphere.planetRadius        = settings.atmoBottomRadius * 1000.0F;
      sky.atmosphere.atmosphereThickness = settings.atmoThickness * 1000.0F;
      break;
    }
    case shaderio::EnvSystem::eHdr: {
      // eHdr *is* OMI's `panorama`; the enumerator kept its name and value because both the .ini
      // and the glTF store it as an integer.
      if(m_lastHdrFile.empty())
      {
        // Nothing loaded (the start-up dummy). Writing `panorama` with no source would describe an
        // environment that does not exist, so fall through to the preserve-or-strip rule below.
        if(!m_environmentSky.has_value())
          return std::nullopt;
        break;
      }
      sky.unknownType.clear();
      sky.type = SkyDescriptor::Type::ePanorama;
      // Relative to the glTF being written, so the pair moves together. An image outside that
      // tree keeps an absolute path rather than a chain of "..\" that only works from here.
      sky.panorama.uri = relativeUriForSave(m_lastHdrFile, destination);
      break;
    }
    default:
      // The active mode is not an authored OMI type (None). If the scene came with a sky,
      // re-emit it unchanged rather than inventing one or silently discarding the author's data;
      // if it did not, there is nothing to write.
      if(!m_environmentSky.has_value())
        return std::nullopt;
      break;
  }

  return sky;
}

//--------------------------------------------------------------------------------------------------
// Frame-top hook for a pending environment change, paired with onEnvironmentChanged().
//
// Runs from onUIRender rather than onRender for the same reason applyPendingTextureRebuild does:
// rebuilding the environment waits on the device and rewrites descriptors that in-flight frames
// are reading, which is not safe in the middle of recording a frame's command buffer. onUIRender
// is the frame-top point this renderer already uses for deferred GPU rebuilds, and it runs in the
// headless and benchmark paths too.
//
void GltfRenderer::applyPendingEnvironmentRefresh()
{
  if(!m_envBakeDirty)
    return;

  if(isBakedEnvironment(m_resources.settings.envSystem))
  {
    refreshBakedEnvironment();
    return;
  }

  m_envBakeDirty = false;

  if(m_envOwnedByBaker && m_resources.settings.envSystem == shaderio::EnvSystem::eHdr)
  {
    // Leaving a baked sky for the HDR path: hdrIbl still points at EnvBaker's image and holds an
    // alias table built over it, so the file has to be reloaded to take ownership back.
    createHDR(m_lastHdrFile);
  }
  // eNone samples neither hdrIbl nor the prefiltered cubes, so a stale binding is harmless -- and
  // re-selecting a baked sky re-commits before anything reads it.

  // Same rule as refreshBakedEnvironment: calibrate on the first refresh after the kind changes.
  if(m_fireflyClampCalibratedFor != int(m_resources.settings.envSystem))
    calibrateFireflyClamp();
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Assemble the authored sky parameters the shaders read.
//
// Built in one place because two consumers need the same numbers for different purposes: the bake
// evaluates them with the sun excluded (that image is the lighting field, and the sun is a
// directional light NEE samples), while the per-ray background evaluates them with the sun in.
// Splitting the fill would let the visible sky and the sky that lights the scene drift apart.
//
shaderio::SkyOmiParameters GltfRenderer::buildSkyOmiParameters() const
{
  const Settings&            settings = m_resources.settings;
  shaderio::SkyOmiParameters params{};

  params.plain.color = settings.plainColor;

  params.gradient.bottomColor  = settings.gradientBottomColor;
  params.gradient.horizonColor = settings.gradientHorizonColor;
  params.gradient.topColor     = settings.gradientTopColor;
  params.gradient.sunColor     = settings.gradientSunColor;
  params.gradient.bottomCurve  = settings.gradientBottomCurve;
  params.gradient.topCurve     = settings.gradientTopCurve;
  params.gradient.sunAngleMax  = settings.gradientSunAngleMax;
  params.gradient.sunCurve     = settings.gradientSunCurve;

  // Resources::sunDirection is the one sun the whole renderer shares -- the sky UI, the physical
  // sky and next-event estimation all read it -- so the visible disk lands exactly where the light
  // that casts the shadows is. The gradient is evaluated in the environment's frame, so the sun
  // is handed over in that frame too.
  params.gradient.sunDirection = sunInEnvironmentFrame();

  params.gradient.sunAngularRadius = kGradientSunAngularRadius;

  return params;
}

glm::vec3 GltfRenderer::sunInEnvironmentFrame() const
{
  // The same conversion the shaders apply before an environment lookup (quatRotateInverse).
  return glm::inverse(m_resources.settings.envRotationQuat()) * m_resources.sunDirection;
}

//--------------------------------------------------------------------------------------------------
// Settings::envRotation was edited. The rotation is where north is, so a sky's sun is placed again
// from Time of Day on the turned compass -- what North Offset used to do. That goes through aim(),
// which also re-bakes the physical sky and reaches a marked sun light. An HDR has no sun to place.
//
void GltfRenderer::onEnvironmentRotated()
{
  if(skyHasSun(m_resources.settings.envSystem))
    m_skySun.applyTimeOfDay();
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Illuminance of the gradient sky's sun, before `envIntensity`.
//
// The physical sky measures its sun: the bake reports a disk radiance, and radiance times the solid
// angle the disk subtends is an illuminance. The gradient sky measures nothing. `sunColor` is the
// colour the disk is *drawn* in, and running it through that same product is what left this sun
// visible but dark -- a half-degree disk subtends 6.8e-5 sr, so a `sunColor` of 1 became 7e-5 lux
// under a dome delivering about 1.5. Forty thousand times too dim, and silently so: the disk was
// still there to look at.
//
// So the number is scaled against the sky instead of derived from the disk. The dome's own
// illuminance is the one quantity the gradient sky does define -- it is those authored colours over
// a hemisphere -- and `kGradientSunToSkyRatio` puts the sun where a clear day puts it relative to
// that. `sunColor` keeps its job as the sun's colour *and* its relative brightness, so brightening
// it in the UI now brightens the light as well as the disk.
//
// The sky half only: the ground half of the gradient lights nothing that faces up, and including it
// would make the sun follow the ground colour.
//
glm::vec3 GltfRenderer::gradientSunIlluminance() const
{
  const Settings& settings = m_resources.settings;

  // Illuminance the dome alone puts on an up-facing surface:
  //   E = 2*PI * integral(0..PI/2) L(v) cos(v) sin(v) dv
  // with L the same horizon -> top band the shader evaluates, and v measured from the zenith.
  // Numeric because the band shape is a free exponent; 256 steps settle this to well under a
  // percent, and it is evaluated once per frame on the CPU.
  constexpr int kSteps = 256;
  glm::vec3     dome(0.0f);
  for(int i = 0; i < kSteps; ++i)
  {
    const float     v = (static_cast<float>(i) + 0.5f) / static_cast<float>(kSteps) * glm::half_pi<float>();
    const float     c = 1.0f - v / glm::half_pi<float>();
    const glm::vec3 L =
        glm::mix(settings.gradientHorizonColor, settings.gradientTopColor, gradientBandWeight(c, settings.gradientTopCurve));
    dome += L * std::cos(v) * std::sin(v);
  }
  dome *= glm::two_pi<float>() * glm::half_pi<float>() / static_cast<float>(kSteps);

  // One scalar for the dome: the ratio is a brightness, and letting it run per channel would tint
  // the sun by the inverse of the sky it stands against.
  const float domeIlluminance = (dome.x + dome.y + dome.z) / 3.0f;
  return settings.gradientSunColor * (kGradientSunToSkyRatio * domeIlluminance);
}

//--------------------------------------------------------------------------------------------------
// The atmosphere the physical sky is built from: Earth, with the panel's settings laid over it.
//
shaderio::SkyAtmosphereParameters GltfRenderer::buildAtmosphereParameters() const
{
  const Settings& s = m_resources.settings;

  // Starts from the Earth defaults for the two fields no setting covers: muSMin, which is LUT
  // coverage rather than physics, and the zero padding layer every density profile needs.
  shaderio::SkyAtmosphereParameters p = skyBrunetonEarthDefaults();

  p.solarIrradiance    = s.atmoSolarIrradiance;
  p.sunAngularRadius   = s.atmoSunAngularRadius;
  p.groundAlbedo       = s.atmoGroundAlbedo;
  p.rayleighScattering = s.atmoRayleighScattering;
  p.mieScattering      = s.atmoMieScattering;
  p.miePhaseFunctionG  = s.atmoMieAnisotropy;

  // Extinction is what the aerosol removes; scattering is the part it gives back. The albedo is
  // the ratio, and it is what the setting exposes -- OMI has no absorption field, so a scene that
  // sets mieScattering keeps the ratio rather than inheriting an unrelated extinction.
  p.mieExtinction = s.atmoMieScattering / std::max(s.atmoMieAlbedo, 1e-3F);

  p.absorptionExtinction = s.atmoOzoneExtinction;
  p.bottomRadius         = s.atmoBottomRadius;
  p.topRadius            = s.atmoBottomRadius + s.atmoThickness;

  // Both gases fall off exponentially, so a scale height is the whole profile. Layer 0 stays the
  // zero padding upstream inserts -- see skyBrunetonEarthDefaults().
  p.rayleighDensity.layers[1].expTerm  = 1.0F;
  p.rayleighDensity.layers[1].expScale = -1.0F / std::max(s.atmoRayleighScaleHeight, 1e-3F);
  p.mieDensity.layers[1].expTerm       = 1.0F;
  p.mieDensity.layers[1].expScale      = -1.0F / std::max(s.atmoMieScaleHeight, 1e-3F);

  // Ozone is a tent: zero at centre-width/2, one at the centre, zero again at centre+width/2.
  // Layer 0 covers the rise and layer 1 the fall, so the split is at the centre.
  const float halfWidth                      = std::max(s.atmoOzoneWidth, 1e-3F) * 0.5F;
  const float slope                          = 1.0F / halfWidth;
  p.absorptionDensity.layers[0].width        = s.atmoOzoneCenter;
  p.absorptionDensity.layers[0].expTerm      = 0.0F;
  p.absorptionDensity.layers[0].expScale     = 0.0F;
  p.absorptionDensity.layers[0].linearTerm   = slope;
  p.absorptionDensity.layers[0].constantTerm = 1.0F - s.atmoOzoneCenter * slope;
  p.absorptionDensity.layers[1].expTerm      = 0.0F;
  p.absorptionDensity.layers[1].expScale     = 0.0F;
  p.absorptionDensity.layers[1].linearTerm   = -slope;
  p.absorptionDensity.layers[1].constantTerm = 1.0F + s.atmoOzoneCenter * slope;

  return p;
}

//--------------------------------------------------------------------------------------------------
// Push the current settings into EnvBaker -- the parameters both the preview and the commit bake
// from.
//
// Separate from refreshBakedEnvironment because the preview path needs it too, and needs only
// this: these are member assignments, uploaded by the bake through the command buffer, so they
// are safe in the middle of recording a frame. setResolution is deliberately *not* here -- it
// reallocates the image, which is a frame-boundary operation and belongs to the commit alone.
//
// The atmosphere block is not here either: the bake reads it for ground albedo and the sun's
// angular size, but any change to it also fails SkyBruneton::lutsSatisfy, and the rebuild that
// follows uploads it.
//
void GltfRenderer::updateBakerParameters()
{
  // Metres in the UI, kilometres in the model.
  m_envBaker.setObserverAltitude(m_resources.settings.atmoObserverAltitude * 0.001F);

  m_envBaker.skyParams() = buildSkyOmiParameters();
  m_envBaker.setSkyType(shaderio::SkyType(bakeSkyType(m_resources.settings.envSystem)));
  m_envBaker.setSunDirection(sunInEnvironmentFrame());
}

//--------------------------------------------------------------------------------------------------
// Rebuild the baked lighting environment from the current settings.
//
// This is the single point where an analytic sky becomes the environment every renderer path
// samples. After it runs, hdrIbl and hdrDome hold a lat-long image, an alias table and prefiltered
// cubemaps built from EnvBaker's output instead of from an HDR file -- which is why the path
// tracer and the rasterizer need no per-sky-type branch to light a scene.
//
// Application thread only: EnvBaker::commit submits and waits.
//
void GltfRenderer::refreshBakedEnvironment()
{
  m_envBakeDirty = false;
  // A full commit supersedes any preview raised before it, and the commit that preview would owe.
  m_envPreviewPending = false;
  m_envCommitOwed     = false;
  if(!isBakedEnvironment(m_resources.settings.envSystem))
    return;

  // Both the bake and the descriptor rewrite below touch resources that in-flight frames are
  // still reading: the bake overwrites the lat-long image, and updateHdrImages() rewrites set 0
  // while a previously submitted frame may still be sampling it. createHDR() opens with
  // the same wait for the same reason -- without it this crashes, not merely trips validation.
  NVVK_CHECK(vkDeviceWaitIdle(m_device));

  const Settings& settings = m_resources.settings;

  // Rebuild the LUTs unless they already hold this atmosphere at full quality. Compared by value
  // rather than tracked with a dirty flag so that an .ini restore, a benchmark script and an MCP
  // write are all covered without each having to remember to raise one -- and so that a settle
  // that changed nothing does not pay for a full precompute.
  //
  // Quality is part of the test, not just the parameters: a drag leaves the LUTs holding the right
  // atmosphere at preview quality, and this is the moment that owes the upgrade.
  if(settings.envSystem == shaderio::EnvSystem::eSky)
  {
    const shaderio::SkyAtmosphereParameters wanted = buildAtmosphereParameters();
    if(!m_skyBruneton.lutsSatisfy(wanted, SkyBruneton::kQualityFinal))
    {
      m_skyBruneton.setParameters(wanted);
      m_skyBruneton.precompute(m_resources);
    }
  }

  updateBakerParameters();

  // Only the rasterizer reads the prefiltered cubemaps; skipping them on the path-tracing path
  // saves ~28 ms of prefilter work per commit for an image nothing would sample.
  const bool refreshDome = m_resources.settings.renderSystem == RenderingMode::eRasterizer;
  m_envBaker.commit(m_resources, refreshDome);
  m_envOwnedByBaker = true;

  // Rebind the environment into the renderer's own descriptor set. HdrIbl::updateFromGpuImage
  // repoints HdrIbl's internal set, which is what the path tracer binds for the alias table --
  // but `texturesHdr[HDR_IMAGE_INDEX]` lives in set 0 here and still names whichever image was
  // bound last (at start-up, the dummy from createHDR("")). Without this the scene is lit and
  // backed by that stale image and the authored sky is silently ignored.
  //
  // Only that one element is rewritten. The BRDF LUT and the prefiltered cubes in the same
  // binding are owned by HdrEnvDome and keep their identity across an update -- rewriting them
  // per commit is the descriptor churn the design explicitly warns against.
  {
    nvvk::WriteSetContainer    write{};
    const VkWriteDescriptorSet hdrTexture =
        m_resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eTexturesHdr, m_resources.descriptorSet,
                                                     HDR_IMAGE_INDEX, 1U);
    write.append(hdrTexture, m_resources.hdrIbl.getHdrImage());
    vkUpdateDescriptorSets(m_device, write.size(), write.data(), 0, nullptr);
  }

  // A baked environment has its own luminance integral, exactly like a loaded HDR -- but this runs
  // on every commit, and a commit happens every time the sun moves or an atmosphere slider settles.
  //
  // So calibrate once, on the first commit after the environment changes kind, and then leave it
  // alone. Two reasons, and the second is the important one:
  //
  //   - The value would otherwise swing with the sun, because the sky's integral does.
  //   - It is a *starting point*, not a derived quantity. The clamp has to accommodate every light
  //     in the scene -- emissive materials, punctual lights, a bright HDR backplate -- and the
  //     environment integral knows about none of them. Once the user has adjusted it, recomputing
  //     from the environment alone throws that away, and does so on a gesture as ordinary as
  //     dragging the sun.
  //
  // This is where the calibration belongs rather than at the switch itself: hdrIbl's integral does
  // not describe the new environment until the bake above has run.
  if(m_fireflyClampCalibratedFor != int(settings.envSystem))
  {
    calibrateFireflyClamp();
  }
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Returns the scene-appropriate firefly clamp threshold:
//   - HDR or baked analytic environment → use its luminance integral (already calibrated to the
//     environment's range; a baked sky populates the same integral through updateFromGpuImage)
//   - Otherwise → a fixed baseline high enough to preserve legitimately bright highlights
void GltfRenderer::calibrateFireflyClamp()
{
  m_fireflyClampCalibratedFor                    = int(m_resources.settings.envSystem);
  m_pathTracer.m_pushConst.fireflyClampThreshold = defaultFireflyClamp();
}

//--------------------------------------------------------------------------------------------------
// Where aerial perspective stands the eye. See the declaration for why this is not the sky's
// observer altitude.
//
float GltfRenderer::aerialPerspectiveEyeAltitudeKm() const
{
  const float liftKm = m_resources.settings.atmoObserverAltitude * 0.001F;

  const nvvkgltf::Scene* scene = m_resources.getScene();
  if(scene == nullptr || !scene->valid() || m_cameraManip == nullptr)
    return liftKm;

  const nvutils::Bbox bounds = scene->getSceneBounds();
  if(!bounds.isEmpty())
  {
    // The scene's floor goes `liftKm` above the planet's surface, so nothing in it is ever
    // underground however the camera moves; the eye then stands wherever it stands above that
    // floor. A camera below the floor -- inside the terrain, or under a ground plane -- would
    // otherwise push itself under the surface instead, so the lift is the floor for that too.
    const float kmPerSceneUnit = 0.001F * m_resources.settings.atmoAerialPerspectiveScale;
    // The camera manipulator works in doubles; the difference is a scene-scale number, so it fits a
    // float long before it is turned into kilometres.
    const float aboveFloor = static_cast<float>(m_cameraManip->getEye().y - bounds.min().y) * kmPerSceneUnit;
    return liftKm + std::max(aboveFloor, 0.0F);
  }
  return liftKm;
}

glm::vec3 GltfRenderer::skySunIlluminanceAboveAtmosphere() const
{
  // The constant is written in shader types; float3 is an alias that lives in `shaderio`.
  using shaderio::float3;
  // The sun's factor set, not the sky's -- this is the sun. See SUN_SPECTRAL_TO_LUMINANCE.
  const glm::vec3 radianceToLuminance = SUN_RADIANCE_TO_LUMINANCE;
  return buildAtmosphereParameters().solarIrradiance * radianceToLuminance * m_resources.settings.hdrEnvIntensity;
}

float GltfRenderer::defaultFireflyClamp() const
{
  const shaderio::EnvSystem env = m_resources.settings.envSystem;
  if(env != shaderio::EnvSystem::eHdr && !isBakedEnvironment(env))
    return 30.0f;

  // The environment's own integral: the sum of max(R,G,B) over solid angle.
  float clamp = m_resources.hdrIbl.getIntegral();

  // The physical sky keeps its sun *out* of that image -- the sun is a directional light so that
  // its disk stays sharp and next-event estimation can sample it -- so the integral measures the
  // sky alone, and for a daylit scene the sky is the smaller half by a wide margin. An .hdr file
  // has its sun baked in and needs no correction, which is why this only surfaced once the
  // analytic skies grew a sun of their own.
  //
  // Added in the integral's own units: a disk of radiance L covering solid angle W contributes
  // max(L) * W to that sum, and radiance times solid angle is illuminance. So the term is the
  // largest channel of the sun's illuminance.
  //
  // Which sun, though -- the one above the atmosphere, not the one currently shining. The
  // attenuated sun is the physically honest answer to "how much light is there right now", and it
  // is the wrong answer here, because it goes to zero at dusk and takes the clamp with it: at 2
  // degrees below the horizon this threshold was 0.34, at 5 degrees 0.03, and at 10 degrees it
  // reached exactly 0 -- which the shader reads as "disabled", so the clamp flipped from crushing
  // every sample to not clamping at all. A clamp is a safety net against outliers, and a safety
  // net that scales itself to nothing at night is worse than a fixed one.
  //
  // The unattenuated value is a property of the star rather than of the hour, so it is stable
  // across a whole time-of-day scrub, and it still follows the atmosphere: Mars' sun is dimmer
  // than Earth's, and editing the solar irradiance moves it.
  //
  // The gradient sky keeps its sun out of the bake for the same reason and needs the same
  // correction; there is just no atmosphere to ask, so the term is the sun this sky implies.
  glm::vec3 sun(0.0f);
  if(env == shaderio::EnvSystem::eSky)
    sun = skySunIlluminanceAboveAtmosphere();
  else if(env == shaderio::EnvSystem::eGradient)
    sun = gradientSunIlluminance() * m_resources.settings.hdrEnvIntensity;
  clamp += std::max({sun.x, sun.y, sun.z});
  return clamp;
}

//--------------------------------------------------------------------------------------------------
// Apply the tonemapper on the rendered image
bool GltfRenderer::tonemap(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);
  return runTonemapPass(cmd, /*skipBeautifiedOverlay=*/false);
}

bool GltfRenderer::runTonemapPass(VkCommandBuffer cmd, bool skipBeautifiedOverlay)
{
  // Select which buffer to tonemap based on user selection
  VkDescriptorImageInfo inputBuffer =
      m_resources.gBuffers.getColorSampleDescriptorImageInfo(Resources::eImgRendered, m_resources.linearSampler);
  VkExtent2D gbufSize         = m_resources.gBuffers.getSize();
  bool       usingGuideBuffer = false;
  bool       bypassTonemapper = false;

#ifdef USE_AGENTIC
  if(!skipBeautifiedOverlay && m_resources.settings.displayBuffer == DisplayBuffer::eAgenticBeautified
     && m_agentic.hasBeautifiedImage())
  {
    // Using beautified image as input buffer
    inputBuffer      = m_agentic.beautifiedDescriptor();
    bypassTonemapper = true;
  }
#endif

  // OptiX denoised output (path-tracer only). Routed via the global DisplayBuffer toggle the
  // OptiXDenoiser writes when the user clicks its thumbnail; behavior is unchanged.
#if defined(USE_OPTIX_DENOISER)
  if(m_resources.settings.renderSystem == RenderingMode::ePathtracer && m_resources.settings.displayBuffer == DisplayBuffer::eOptixDenoised)
  {
    const OptiXDenoiser* optix = m_pathTracer.getOptiXDenoiser();
    if(optix && optix->hasValidDenoisedOutput())
    {
      inputBuffer         = optix->getDescriptorImageInfo(OptiXDenoiser::eGBufferDenoised);
      inputBuffer.sampler = m_resources.linearSampler;
    }
  }
#endif

#if defined(USE_DLSS)
  // Displaying or not DLSS guide buffer.
  if(Dlss* dlss = activeDlss())
  {
    if(auto guide = dlss->activeGuideImage())
    {
      inputBuffer         = guide->image;
      inputBuffer.sampler = m_resources.linearSampler;
      gbufSize            = guide->extent;
      usingGuideBuffer    = true;

      // Clear the output image to a distinct color so the DLSS render-resolution borders are
      // visible when the guide buffer is smaller than the display.
      VkImageSubresourceRange range = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
      vkCmdClearColorImage(cmd, m_resources.gBuffers.getColorImage(Resources::eImgTonemapped), VK_IMAGE_LAYOUT_GENERAL,
                           &kBackgroundClearColor, 1, &range);
      // Barrier: clear must complete before tonemapper compute shader runs
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
    }
  }
#endif

  // Disable tonemapping for debug buffers or guide buffers (display raw values)
  shaderio::TonemapperData tonemapperData = m_resources.tonemapperData;
  const bool               vizNormal      = (m_resources.settings.visualization == shaderio::Visualization::eRendered
                          || m_resources.settings.visualization == shaderio::Visualization::eClay);
  if(!vizNormal || usingGuideBuffer)
    tonemapperData.isActive = 0;
  if(bypassTonemapper)
    tonemapperData.isActive = 0;

  m_resources.tonemapper.runCompute(cmd, gbufSize, tonemapperData, inputBuffer,
                                    m_resources.gBuffers.getColorStorageImageInfo(Resources::eImgTonemapped));

  // Barrier: tonemapper compute writes must be visible to both the DLSS-NR compute pass (which
  // reads eImgTonemapped immediately after tonemap()) and the silhouette fragment shader.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

  // NR post-process is only valid on the main rendered path (not guide buffers, viz modes, or bypassed tonemapper).
  return vizNormal && !usingGuideBuffer && !bypassTonemapper;
}

//--------------------------------------------------------------------------------------------------
// Render the silhouette of the selected object(s)
//
// Selection visualization uses an ObjectID buffer (R32_UINT, filled on frame 0 by the path tracer
// with the first-hit render node ID per pixel) and a selection bitmask (one bit per
// render node). The silhouette compute shader reads ObjectID per pixel, checks the bitmask, and
// applies a Sobel edge filter where the bit is set, then composites the outline onto the image.
void GltfRenderer::silhouette(VkCommandBuffer cmd)
{
  // Sync with UI: when selection was cleared in the scene browser (e.g. toggle off), no event
  // is emitted, so we clear Resources here so the silhouette does not keep drawing.
  if(!m_sceneSelection.hasSelection())
    m_resources.selectedRenderNodes.clear();

  // Run the silhouette pass only when something is selected (one primitive or a node and its branch).
  // We rebuild the CPU bitmask from selectedRenderNodes, upload it, then dispatch the silhouette compute.
  if(m_sceneSelection.hasSelection())
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

    // Rebuild selection bitmask and ensure GPU buffer exists and is up to date.
    int numRenderNodes = m_resources.getScene() && m_resources.getScene()->valid() ?
                             static_cast<int>(m_resources.getScene()->getRenderNodes().size()) :
                             0;
    m_resources.updateSelectionBitMask(numRenderNodes);
    const VkDeviceSize bitmaskBytes = m_resources.selectionBitMask.size() * sizeof(uint32_t);
    if(bitmaskBytes > 0)
    {
      // Build or update the selection bitmask buffer
      if(m_resources.bSelectionBitMask.buffer == VK_NULL_HANDLE || m_resources.bSelectionBitMask.bufferSize < bitmaskBytes)
      {
        if(m_resources.bSelectionBitMask.buffer != VK_NULL_HANDLE)
        {
          // SYNC NOTE: Wait required before destroying buffer that may be in-flight on the GPU.
          // This only triggers when node count grows past the current allocation (infrequent).
          vkQueueWaitIdle(m_app->getQueue(0).queue);
          m_resources.allocator.destroyBuffer(m_resources.bSelectionBitMask);
        }
        NVVK_CHECK(m_resources.allocator.createBuffer(m_resources.bSelectionBitMask, bitmaskBytes,
                                                      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                                          | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                                      VMA_MEMORY_USAGE_CPU_TO_GPU));
        NVVK_DBG_NAME(m_resources.bSelectionBitMask.buffer);
      }

      // In case the bitmask is too large (> 524,288 render nodes), we need to update it in chunks
      constexpr VkDeviceSize kMaxCmdUpdateSize = 65536;
      for(VkDeviceSize offset = 0; offset < bitmaskBytes; offset += kMaxCmdUpdateSize)
      {
        const VkDeviceSize chunkSize = std::min(kMaxCmdUpdateSize, bitmaskBytes - offset);
        vkCmdUpdateBuffer(cmd, m_resources.bSelectionBitMask.buffer, offset, chunkSize,
                          reinterpret_cast<const uint8_t*>(m_resources.selectionBitMask.data()) + offset);
      }
      nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    }

    std::vector<VkDescriptorImageInfo> imageInfos = {
        m_resources.gBuffers.getColorStorageImageInfo(Resources::eImgSelection),
        m_resources.gBuffers.getColorStorageImageInfo(Resources::eImgTonemapped),
    };
    VkDescriptorBufferInfo bitmaskBufferInfo = {m_resources.bSelectionBitMask.buffer, 0, bitmaskBytes};
    m_silhouette.dispatch(cmd, m_resources.gBuffers.getSize(), imageInfos, bitmaskBufferInfo,
                          static_cast<uint32_t>(m_resources.selectionBitMask.size()));
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
  }
}

//--------------------------------------------------------------------------------------------------
// Render visual helpers (grid + transform gizmo) onto the tonemapped image
void GltfRenderer::renderVisualHelpers(VkCommandBuffer cmd)
{
  // Sync grid and snap settings to visual helpers
  m_visualHelpers.grid.setVisible(m_resources.settings.showGrid);
  m_visualHelpers.grid.style().baseUnit = m_resources.settings.gridUnit;
  m_visualHelpers.transform.setSnapEnabled(m_resources.settings.snapEnabled);
  m_visualHelpers.transform.setSnapValues(m_resources.settings.gridUnit, m_resources.settings.snapRotation,
                                          m_resources.settings.snapScale);

  // Sync gizmo attachment from selection state
  updateGizmoAttachment();

  if(!m_visualHelpers.shouldRender())
    return;

  NVVK_DBG_SCOPE(cmd);

  const VkExtent2D size = m_resources.gBuffers.getSize();
  glm::vec2        viewportSize(static_cast<float>(size.width), static_cast<float>(size.height));
  glm::vec2        depthBufferSize = viewportSize;
#if defined(USE_DLSS)
  // Align the depth-buffer extent the gizmo grid samples to whichever DLSS instance is active.
  // isActive() is strict (false during async init), so we never pick up a 0x0 inner extent.
  if(const Dlss* dlss = activeDlss(); dlss && dlss->isActive())
  {
    const VkExtent2D rs = dlss->getRenderSize();
    depthBufferSize     = {static_cast<float>(rs.width), static_cast<float>(rs.height)};
  }
#endif

  m_visualHelpers.render(cmd, m_resources.gBuffers.getColorImage(Resources::eImgTonemapped),
                         m_resources.gBuffers.getColorAttachmentView(Resources::eImgTonemapped), m_resources.descriptorSet,
                         m_cameraManip->getViewMatrix(), m_cameraManip->getPerspectiveMatrix(), viewportSize, depthBufferSize);
}

//--------------------------------------------------------------------------------------------------
// Update gizmo attachment based on current node selection
void GltfRenderer::updateGizmoAttachment()
{
  if(!m_resources.settings.showGizmo || !m_resources.getScene() || !m_resources.getScene()->valid())
  {
    if(m_gizmoNodeIndex >= 0)
    {
      m_visualHelpers.transform.clearAttachment();
      m_gizmoNodeIndex = -1;
    }
    return;
  }

  int nodeIdx = -1;
  if(m_sceneSelection.hasSelection())
  {
    auto sel = m_sceneSelection.getSelection();
    if(sel.type == SceneSelection::SelectionType::eNode || sel.type == SceneSelection::SelectionType::ePrimitive)
    {
      nodeIdx = sel.nodeIndex;
    }
  }

  if(nodeIdx < 0)
  {
    if(m_gizmoNodeIndex >= 0)
    {
      m_visualHelpers.transform.clearAttachment();
      m_gizmoNodeIndex = -1;
    }
    return;
  }

  // Attach or update if selection changed
  if(nodeIdx != m_gizmoNodeIndex)
  {
    m_gizmoNodeIndex = nodeIdx;

    glm::quat rotation;
    tinygltf::utils::getNodeTRS(m_resources.getScene()->editor().getNode(nodeIdx), m_gizmoPosition, rotation, m_gizmoScale);
    m_gizmoRotation = glm::degrees(glm::eulerAngles(rotation));

    int parentIdx = m_resources.getScene()->editor().getNodeParent(nodeIdx);
    m_gizmoParentWorldMatrix = (parentIdx >= 0) ? m_resources.getScene()->computeNodeWorldMatrix(parentIdx) : glm::mat4(1.f);

    m_visualHelpers.transform.attachTransform(&m_gizmoPosition, &m_gizmoRotation, &m_gizmoScale);
    m_visualHelpers.transform.setParentWorldMatrix(&m_gizmoParentWorldMatrix);
  }
  else if(!m_visualHelpers.transform.isDragging())
  {
    // Re-read TRS from the scene node so inspector edits are reflected immediately
    glm::quat rotation;
    tinygltf::utils::getNodeTRS(m_resources.getScene()->editor().getNode(nodeIdx), m_gizmoPosition, rotation, m_gizmoScale);
    m_gizmoRotation = glm::degrees(glm::eulerAngles(rotation));
  }
}

//--------------------------------------------------------------------------------------------------
// Set DLSS hardware/extension availability per NGX feature (Ray Reconstruction / Super Resolution).
// This should be called early, before any DLSS initialization occurs.
void GltfRenderer::setDlssHardwareAvailability(bool rrAvailable, bool srAvailable)
{
  m_resources.settings.dlssRrHardwareAvailable = rrAvailable;
  m_resources.settings.dlssSrHardwareAvailable = srAvailable;
}

//--------------------------------------------------------------------------------------------------
// Set Opacity Micromap (VK_EXT_opacity_micromap) availability. When unavailable, the
// EXT_mesh_opacity_micromap glTF extension is ignored. Call early, before scene creation.
void GltfRenderer::setOpacityMicromapAvailable(bool available)
{
  m_resources.settings.opacityMicromapSupported = available;
}

//--------------------------------------------------------------------------------------------------
// Load the scene
bool GltfRenderer::createScene(const std::filesystem::path& sceneFilename)
{
  nvutils::ScopedTimer st(__FUNCTION__);
  m_sceneSelection.clearSelection();  // Clear selection in new UI system
  m_resources.selectedRenderNodes.clear();

  if(sceneFilename.empty())
  {
    return false;
  }

  std::filesystem::path filename = nvutils::findFile(sceneFilename, nvsamples::getResourcesDirs(), false);
  if(!filename.has_filename())
  {
    LOGW("Cannot find file: %s\n", nvutils::utf8FromPath(sceneFilename).c_str());
    removeFromRecentFiles(filename);
    return false;
  }

  // Convert OBJ to glTF
  if(nvutils::extensionMatches(sceneFilename, ".obj"))
  {
    tinyobj::ObjReaderConfig readerConfig;
    readerConfig.mtl_search_path = std::filesystem::path(filename).parent_path().string();
    tinyobj::ObjReader reader;

    bool        result = reader.ParseFromFile(nvutils::utf8FromPath(filename), readerConfig);
    std::string warn   = reader.Warning();
    std::string error  = reader.Error();

    if(result)
    {
      TinyConverter   converter;
      tinygltf::Model model;
      converter.convert(model, reader);
      auto scn = std::make_unique<nvvkgltf::Scene>();
      scn->takeModel(std::move(model));
      m_resources.scene = std::move(scn);
    }
    else
    {
      LOGW("Error loading OBJ: %s\n", error.c_str());
      LOGW("Warning: %s\n", warn.c_str());
      removeFromRecentFiles(filename);
      return false;
    }
  }
  else
  {
    LOGI("Loading scene: %s\n", nvutils::utf8FromPath(filename).c_str());
    auto scn = std::make_unique<nvvkgltf::Scene>();
    scn->supportedExtensions().insert(EXT_TEXTURE_WEBP_EXTENSION_NAME);  // Register support for WebP images in glTF (local to this project)
    if(!scn->load(filename))
    {
      LOGW("Error loading scene: %s\n", nvutils::utf8FromPath(filename).c_str());
      removeFromRecentFiles(filename);
      return false;
    }
    m_resources.scene = std::move(scn);
  }

  // Scene object is ready; wire up GPU resources and UI (shared with createEmptyScene()).
  finalizeSceneSetup(filename);
  return true;
}

//--------------------------------------------------------------------------------------------------
// Shared tail for creating/loading a scene: builds the Vulkan scene, feature set, UI panels, camera
// and textures. `filename` is empty for a new/empty scene (no camera list, no recent-files entry).
//
void GltfRenderer::finalizeSceneSetup(const std::filesystem::path& filename)
{
  // Scene object is set, we can create the Vulkan scene
  createVulkanScene();
  if(ui::animation::hasPlayableAnimation(m_resources.getScene()))
    m_resources.animationControl.showStrip = true;
  // Spec: "When a glTF asset contains a behavior graph, all glTF animations are assumed to be
  // controlled by the graph so they MUST NOT play automatically." Only gates the default - manual
  // scrubbing via the Animation Strip is still available.
  if(!m_resources.getScene()->getInteractivityGraphs().empty())
    m_resources.animationControl.play = false;

  // Detect which KHR_materials_* the loaded scene actually uses so the path tracer
  // can specialize its shader when settings.optimalShader is on. Safe to call always:
  // when optimalShader is off, the path tracer ignores currentFeatureSet.
  m_resources.recomputeSceneFeatures(dlssGuideRequired());

  // Calibrate the firefly clamp to the scene: retroreflective materials can produce legitimately
  // high peak radiance that the default clamp of 10 would clip, visibly suppressing the effect.
  calibrateFireflyClamp();

  wireSceneToUi();  // Scene Browser + Inspector pointers, callbacks, bounds

  nvvkgltf::Scene* scene = m_resources.getScene();
  m_resources.settings.infinitePlaneDistance = scene->getSceneBounds().min().y;  // Set the infinite plane distance to the bottom of the scene

  // Set camera from scene
  nvvkgltf::addSceneCamerasToWidget(m_cameraManip, filename, scene->getRenderCameras(), scene->getSceneBounds());

  // Viewer-side environment controls (hdrEnvIntensity, envRotation, the backplate override)
  // are deliberately left alone by a scene load -- resetting them used to discard whatever the
  // user, the command line, or MCP had set. The *authored* sky is different: a glTF carrying
  // OMI_environment_sky is stating what the scene's sky is, and that wins over the ini.
  //
  // The one place the two meet is `rotation`, which is both. It is applied only when the file
  // states it -- see applySkyDescriptor -- so an authored orientation wins while a scene that is
  // silent about it leaves the user's rotation where they left it.
  applyEnvironmentSkyFromScene();

  // Need to update (push) all textures
  if(!updateTextures())
  {
    LOGE("Failed to update textures - cannot safely render scene");

    // Clean up the scene we just loaded - it's unsafe to render
    vkDeviceWaitIdle(m_device);
    cleanupScene();

    if(!filename.empty())
      removeFromRecentFiles(filename);
    return;
  }

  if(!filename.empty())
    addToRecentFiles(filename);
}

//--------------------------------------------------------------------------------------------------
// Wire the current Scene into the UI panels (browser + inspector): pointers, callbacks and bounds.
// Shared by finalizeSceneSetup() (after a load) and ensureEmptyScene() (a wired empty scene).
//
void GltfRenderer::wireSceneToUi()
{
  nvvkgltf::Scene* scene = m_resources.getScene();

  // Host services shared by both panels (file dialog, texture thumbnails, toasts). Built once here and
  // handed to each, instead of wiring the same three hooks separately. IMAGE/TEXTURE edits made through
  // these set DirtyFlags::texturesChanged and are consumed at frame top by applyPendingTextureRebuild().
  UiHostServices host;
  host.pickImageFile    = [this] { return pickImageFile(); };
  host.textureThumbnail = [this](int i) { return thumbnailForTexture(i); };
  host.notify           = [this](const std::string& msg, bool err) { notify(msg, err); };

  m_sceneBrowser.setScene(scene);
  m_sceneBrowser.setSelection(&m_sceneSelection);
  m_sceneBrowser.setUndoStack(&m_undoStack);
  m_sceneBrowser.setBbox(scene->getSceneBounds());
  m_sceneBrowser.setPendingDelete(&m_pendingDeleteNode, &m_openDeletePopupNextFrame);
  m_sceneBrowser.setHostServices(host);
  // GEOMETRY edits fire this callback (immediate reconcile), unlike image/texture edits which defer.
  m_sceneBrowser.setGeometryChangedCallback([this] { reconcileGeometryIfNeeded(); });
  m_sceneBrowser.setBeforeCreateCallback([this] { ensureEmptyScene(); });
  m_sceneBrowser.setImageThumbnailCallback([this](int i) { return thumbnailForImage(i); });

  m_inspector.setScene(scene);
  m_inspector.setSelection(&m_sceneSelection);
  m_inspector.setUndoStack(&m_undoStack);
  m_inspector.setBbox(scene->getSceneBounds());
  m_inspector.setHostServices(host);
  m_inspector.setViewImageCallback([this](int imageIndex) { m_sceneBrowser.openImageViewer(imageIndex); });
  m_inspector.setImageThumbnailCallback([this](int i) { return thumbnailForImage(i); });
}

// Push a transient notification (isError = red) to the on-screen toast overlay.
void GltfRenderer::notify(const std::string& message, bool isError)
{
  m_toasts.push(message, isError ? UiToasts::Level::Error : UiToasts::Level::Info);
}

//--------------------------------------------------------------------------------------------------
// Ensure an editable Scene exists so the normal add path (popup + AddPrimitiveCommand) can run even
// with nothing loaded. Creates a fresh, empty Scene and wires it to the UI, but does NOT build GPU
// resources: an empty scene is not valid() so onRender() just clears the G-Buffer (see the guard in
// onRender). The first primitive's AddPrimitiveCommand then triggers the (full) GPU build via
// reconcileGeometryIfNeeded(). No-op if a scene already exists.
//
void GltfRenderer::ensureEmptyScene()
{
  if(m_resources.getScene())
    return;

  auto scn = std::make_unique<nvvkgltf::Scene>();
  scn->supportedExtensions().insert(EXT_TEXTURE_WEBP_EXTENSION_NAME);
  // One empty scene container so the browser shows a scene root (matches a loaded scene minus nodes).
  // Still not valid() until a node is added, so onRender() clears the G-Buffer rather than drawing.
  scn->getModel().scenes.emplace_back().name = "Scene";
  m_resources.scene                          = std::move(scn);
  wireSceneToUi();
}

//--------------------------------------------------------------------------------------------------
// Load a legacy .scene.json descriptor and bridge it to glTF 2.1 external assets: each model
// instance is added as a *reference* (not embedded), so the composed scene round-trips as external
// assets on save. This turns the old descriptor format into a one-way importer onto the modern path.
//
void GltfRenderer::createSceneFromDescriptor(const std::filesystem::path& descriptorPath)
{
  nvutils::ScopedTimer st(__FUNCTION__);
  m_sceneSelection.clearSelection();
  m_resources.selectedRenderNodes.clear();

  SceneDescriptor desc;
  if(!loadSceneDescriptor(descriptorPath, desc) || desc.models.empty())
    return;

  // Build the scene locally so it's not visible to the UI thread during construction
  auto scn = std::make_unique<nvvkgltf::Scene>();
  scn->supportedExtensions().insert(EXT_TEXTURE_WEBP_EXTENSION_NAME);
  nvvkgltf::Scene* scene = scn.get();

  auto instancesByModel = desc.getInstancesByModel();

  for(const auto& [modelIdx, instances] : instancesByModel)
  {
    const auto& modelEntry = desc.models[modelIdx];
    // referenceScene() loads the file on first use and, for repeats of the same model, duplicates the
    // instance so geometry (meshes/BLAS) is shared. Each call returns the editable instance node that
    // carries the externalAsset link; we only set its transform/name.
    for(const auto& instPtr : instances)
    {
      const auto& inst            = *instPtr;
      int         instanceNodeIdx = scene->referenceScene(modelEntry.resolvedPath);
      if(instanceNodeIdx < 0)
      {
        LOGE("Failed to reference model: %s\n", nvutils::utf8FromPath(modelEntry.resolvedPath).c_str());
        break;  // the file won't load for the remaining instances either
      }
      scene->editor().setNodeTRS(instanceNodeIdx, inst.translation, inst.rotation, inst.scale);
      if(!inst.name.empty())
        scene->getModel().nodes[instanceNodeIdx].name = inst.name;
    }
  }

  if(!scene->valid())
  {
    LOGW("Scene descriptor produced no valid scene\n");
    return;
  }

  scene->setCurrentScene(scene->getCurrentScene());  // Final reparse: world matrices + render nodes

  // Publish to resources only when fully constructed (UI thread can now see it)
  m_resources.scene = std::move(scn);
  createVulkanScene();
  if(ui::animation::hasPlayableAnimation(scene))
    m_resources.animationControl.showStrip = true;
  // Spec: KHR_interactivity disables animation autoplay (see finalizeSceneSetup()'s matching comment).
  if(!scene->getInteractivityGraphs().empty())
    m_resources.animationControl.play = false;

  // Detect which KHR_materials_* the loaded scene actually uses so the path tracer
  // can specialize its shader when settings.optimalShader is on.
  m_resources.recomputeSceneFeatures(dlssGuideRequired());

  calibrateFireflyClamp();

  wireSceneToUi();

  m_resources.settings.infinitePlaneDistance = scene->getSceneBounds().min().y;

  nvvkgltf::addSceneCamerasToWidget(m_cameraManip, descriptorPath, scene->getRenderCameras(), scene->getSceneBounds());

  if(!updateTextures())
  {
    LOGE("Failed to update textures from descriptor scene\n");
    vkDeviceWaitIdle(m_device);
    cleanupScene();
    return;
  }

  addToRecentFiles(descriptorPath);
}

//--------------------------------------------------------------------------------------------------
// Helper function to cleanup the current scene
//
void GltfRenderer::cleanupScene()
{
  m_undoStack.clear();
  // Drop any renderer-side state tied to the outgoing Scene BEFORE the unique_ptr is reset.
  // The heap allocator is free to hand the same address back to the next Scene instance, so
  // any Scene-pointer-based invalidation inside the render loop would be unreliable -- this is
  // the authoritative invalidation point.
  m_pathTracer.onSceneInvalidated(m_resources);
  m_rasterizer.onSceneInvalidated(m_resources);
  if(m_resources.getScene())
    m_resources.transformCompute.destroyGpuBuffers();
  m_resources.scene.reset();
  m_thumbnailCache.clear();
  m_resources.sceneGpu.destroy();
  m_sceneBrowser.setScene(nullptr);
  m_inspector.setScene(nullptr);
  m_sceneSelection.clearSelection();  // Clear selection in new UI system
  m_resources.selectedRenderNodes.clear();
  m_resources.animationControl      = AnimationControl{};
  m_resources.animationControl.play = m_animPlay;
  m_resources.interactivityControl  = InteractivityControl{};

  // Drop async pick/hover state tied to the outgoing scene - a pending click ray-pick or a
  // cached hover node index would otherwise be consumed by the newly loaded scene once its
  // semaphore/readback settles, applying a pick result to the wrong node indices.
  m_pendingClickPick.reset();
  m_pendingClickResult.reset();
  m_hoveredNodeIndex = -1;

  // Reset memory statistics for the new scene
  // Keeps lifetime allocation/deallocation counts but resets current and peak values
  m_resources.sceneVk.getMemoryTracker().reset();
  m_resources.sceneRtx.getMemoryTracker().reset();
  m_resources.transformCompute.getMemoryTracker().reset();
  m_resources.animationVk.getMemoryTracker().reset();
}

//--------------------------------------------------------------------------------------------------
void GltfRenderer::refreshCpuSceneGraphFromModel()
{
  nvvkgltf::Scene* scene = m_resources.getScene();
  if(scene)
    scene->setCurrentScene(scene->getCurrentScene());
}

//--------------------------------------------------------------------------------------------------
// Unified scene rebuild with optional texture update
// Internal helper that consolidates the common rebuild logic between geometry-only and full rebuilds
//
// Does not call parseScene — callers that modified the glTF without mergeScene/parseScene must call
// refreshCpuSceneGraphFromModel() first (e.g. compact, rebuildSceneFromModel).
//
void GltfRenderer::rebuildVulkanSceneInternal(nvvkgltf::SceneGpu::RebuildMode mode)
{
  using RebuildMode = nvvkgltf::SceneGpu::RebuildMode;
  // Modes that add or reload images (eFull re-reads all; eMergeAppend loads only the new tail). Both
  // need a descriptor rewrite; eGeometryOnly leaves textures untouched.
  const bool touchesTextures = (mode == RebuildMode::eFull || mode == RebuildMode::eMergeAppend);

  // SYNC NOTE: Full scene rebuild (merge/compact/geometry change) -- wait ensures GPU is idle.
  NVVK_CHECK(vkQueueWaitIdle(m_app->getQueue(0).queue));

  // Only eFull destroys the existing texture/image views; release the thumbnails referencing them now
  // (GPU is idle) so panels re-acquire against the new views. eMergeAppend keeps existing views, so the
  // thumbnails stay valid and must NOT be cleared.
  if(mode == RebuildMode::eFull)
    m_thumbnailCache.clear();

  nvvkgltf::Scene* scene = m_resources.getScene();

  {
    if(scene)
      m_resources.transformCompute.destroyGpuBuffers();  // Before scene RTX rebuild
    VkCommandBuffer cmd{};
    nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
    m_resources.sceneGpu.rebuild(cmd, *scene, mode);
    NVVK_CHECK(vkEndCommandBuffer(cmd));
    m_loadPipeline.enqueue(cmd);
  }

  buildAccelerationStructures();

  // Update UI system
  scene = m_resources.getScene();
  if(scene)
  {
    m_sceneBrowser.setScene(scene);
    m_sceneBrowser.setBbox(scene->getSceneBounds());
    m_inspector.setScene(scene);
    m_inspector.setBbox(scene->getSceneBounds());

    // After merge: select the merged file's first animation clip (appended after base clips).
    // Otherwise currentAnimation often stays on a base-scene clip and merged motion appears "stuck".
    AnimationControl& animCtrl = m_resources.animationControl;
    if(int prefer = scene->takeMergePreferredAnimationIndex(); prefer >= 0)
    {
      const int nAnim = scene->animation().getNumAnimations();
      if(nAnim > 0 && prefer < nAnim)
        animCtrl.currentAnimation = prefer;
    }
    else if(scene->animation().hasAnimation())
    {
      const int nAnim = scene->animation().getNumAnimations();
      if(nAnim > 0 && (animCtrl.currentAnimation < 0 || animCtrl.currentAnimation >= nAnim))
        animCtrl.currentAnimation = 0;
    }
    if(ui::animation::hasPlayableAnimation(scene))
      animCtrl.showStrip = true;
    // Spec: KHR_interactivity disables animation autoplay (see finalizeSceneSetup()'s matching comment).
    if(!scene->getInteractivityGraphs().empty())
      animCtrl.play = false;
  }

  // Update textures if images were (re)loaded. eFull rewrites the whole array; eMergeAppend rewrites it
  // too (cheap, no disk) so the new tail textures become visible. eGeometryOnly preserves textures.
  if(touchesTextures)
  {
    if(!updateTextures())
    {
      LOGE("Failed to update textures - scene may not render correctly\n");
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Rebuild the Vulkan scene after modifying the glTF model in-place.
// Use this when you've modified model geometry (vertices, indices, accessors) and need to
// recreate GPU resources. The model data itself is preserved.
//
// Example use cases:
// - After MikkTSpace tangent generation with vertex splitting
// - After mesh optimization that changes vertex/index counts
// - After any operation that modifies buffer data or accessor indices
//
// Note: This preserves textures since they don't change during geometry modifications.
//
void GltfRenderer::rebuildSceneFromModel()
{
  m_undoStack.clear();  // structural model edit outside the command system invalidates history
  rebuildSceneGeometry();
}

//--------------------------------------------------------------------------------------------------
// Geometry-only rebuild that preserves the undo stack. Use from undoable geometry edits (e.g. adding
// a procedural primitive) where the mutation IS tracked by a command and undo history must survive.
//
void GltfRenderer::rebuildSceneGeometry()
{
  refreshCpuSceneGraphFromModel();
  rebuildVulkanSceneInternal(nvvkgltf::SceneGpu::RebuildMode::eGeometryOnly);  // Geometry only, preserve textures
  resetFrame();  // geometry changed -> restart path-tracer accumulation (the rebuild clears the
                 // dirty flags before updateSceneChanges runs, so nothing else would reset it)
}

//--------------------------------------------------------------------------------------------------
// Reconcile GPU resources with the model after a create/undo/redo edit. Two cases:
//
// 1. First build: the scene just became valid() (e.g. the first object - primitive, light or empty
//    node - added to an empty scene from the "Create" menu). No textures/materials/render-node
//    buffers exist yet, so onRender()/updateSceneChanges() would touch unbuilt GPU resources. Detect
//    this via the material buffer (null until the first full build) and do a full build.
// 2. Primitive topology changed (primitivesChanged): syncFromScene() never (re)allocates the
//    per-primitive vertex/index buffers, so they must be recreated to match the current render
//    primitives before the per-frame acceleration-structure build references them. A count check is
//    not sufficient: add -> undo -> add-a-different-primitive can leave counts equal while the GPU
//    buffers still hold the old geometry, so key off primitivesChanged. Light / empty-node / transform
//    edits on an already-built scene need nothing here - updateSceneChanges() handles them per frame.
//
// Runs in UI context (after a command / undo-redo), matching the safe rebuild pattern used by tangent
// regen and merge.
//
void GltfRenderer::reconcileGeometryIfNeeded()
{
  nvvkgltf::Scene* scene = m_resources.getScene();
  if(!scene || !scene->valid())
    return;

  const bool firstBuild = (m_resources.sceneVk.material().buffer == VK_NULL_HANDLE);
  if(firstBuild)
  {
    rebuildVulkanSceneFull();  // textures + materials + geometry + AS
    resetFrame();              // restart path-tracer accumulation (full rebuild doesn't itself reset)
  }
  else if(scene->getDirtyFlags().primitivesChanged)
  {
    rebuildSceneGeometry();  // geometry-only (recreates vertex/index buffers + BLAS/TLAS, resets frame)
  }
  // NOTE: DirtyFlags::texturesChanged is intentionally NOT handled here. A full texture rebuild frees
  // the ImGui thumbnail descriptor sets, so it must run at the START of the frame, not mid-frame from
  // here (this runs during onUIRender, e.g. via onUndoRedo). See applyPendingTextureRebuild().
}

//--------------------------------------------------------------------------------------------------
// Full GPU resource rebuild including textures. Used after operations that modify the model
// structure: merging scenes, compacting resources, etc.
// Destroys all GPU resources and recreates from the current model.
//
void GltfRenderer::rebuildVulkanSceneFull()
{
  rebuildVulkanSceneInternal(nvvkgltf::SceneGpu::RebuildMode::eFull);
}

//--------------------------------------------------------------------------------------------------
// This function creates the Vulkan scene from the glTF model
// It builds the bottom-level and top-level acceleration structure
// The function is called when the scene is loaded
void GltfRenderer::createVulkanScene()
{
  {
    // Enable opacity micromap (EXT_mesh_opacity_micromap) build when the device supports it
    m_resources.sceneVk.setOpacityMicromapEnabled(m_resources.settings.opacityMicromapSupported);

    // Create and queue command buffer for scene data upload (vertices, indices, materials, etc.)
    // This work happens asynchronously via the command buffer queue
    VkCommandBuffer cmd{};
    nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
    m_resources.sceneGpu.create(cmd, *m_resources.getScene(), true);
    NVVK_CHECK(vkEndCommandBuffer(cmd));
    m_loadPipeline.enqueue(cmd);
  }

  buildAccelerationStructures();
}

//--------------------------------------------------------------------------------------------------
// Build BLAS and TLAS acceleration structures for ray tracing
// Used by both createVulkanScene() and rebuildSceneFromModel()
//
void GltfRenderer::buildAccelerationStructures()
{
  // Skip BLAS compaction for animated scenes to avoid freezing mesh positions; only static scenes are compacted.
  const bool isAnimated = m_resources.getScene() && m_resources.getScene()->animation().hasAnimation();
  VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  flags |= isAnimated ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR : VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;

  // Create the bottom-level acceleration structure descriptors (no building yet)
  m_resources.sceneRtx.createBottomLevelAccelerationStructure(*m_resources.getScene(), m_resources.sceneVk, flags);

  // Build the bottom-level acceleration structure
  // Memory-conscious approach: build within a fixed memory budget using multiple command buffers if needed
  // Each build command is queued separately and (for non-animated scenes) followed by compaction to optimize memory usage
  {
    bool finished = false;

    // Building BLAS within a memory budget, which could involve multiple calls to cmdBuildBottomLevelAccelerationStructure
    do
    {
      VkCommandBuffer cmd{};
      nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
      constexpr VkDeviceSize kBlasBuildMemoryBudget = 512ULL * 1024 * 1024;  // 512 MB per build pass
      finished = m_resources.sceneRtx.cmdBuildBottomLevelAccelerationStructure(cmd, kBlasBuildMemoryBudget);
      NVVK_CHECK(vkEndCommandBuffer(cmd));
      if(isAnimated)
      {
        m_loadPipeline.enqueue(cmd);
      }
      else
      {
        m_loadPipeline.enqueue(cmd, [this] {
          VkCommandBuffer compactCmd{};
          nvvk::beginSingleTimeCommands(compactCmd, m_device, m_transientCmdPool);
          m_resources.sceneRtx.cmdCompactBlas(compactCmd);
          NVVK_CHECK(vkEndCommandBuffer(compactCmd));
          m_loadPipeline.enqueue(compactCmd);
        });
      }

    } while(!finished);

    // Track all BLAS allocations now that they're all built
    m_resources.sceneRtx.trackBlasMemory();

    // Queue TLAS building for after all BLAS work completes
    // TLAS is the top-level structure referencing all bottom-level acceleration structures
    {
      VkCommandBuffer cmd{};
      nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
      m_resources.sceneRtx.cmdCreateBuildTopLevelAccelerationStructure(cmd, m_resources.staging, *m_resources.getScene());
      m_resources.staging.cmdUploadAppended(cmd);
      NVVK_CHECK(vkEndCommandBuffer(cmd));
      m_loadPipeline.enqueue(cmd, [this] { m_resources.staging.releaseCompletedAllocations(true); });
    }
  }

  // Avoid double-build: whoever called us (createVulkanScene, rebuildVulkanSceneInternal, or updateSceneChanges) just queued BLAS+TLAS.
  if(m_resources.getScene())
    m_resources.getScene()->getDirtyFlags().primitivesChanged = false;

  // GPU transform SSBOs (hierarchy, matrices, RenderNodeGpuMapping). Queue after TLAS instance buffer exists.
  if(m_resources.getScene())
  {
    m_resources.transformCompute.createGpuBuffers(m_resources.staging, *m_resources.getScene());
    VkCommandBuffer upCmd{};
    nvvk::beginSingleTimeCommands(upCmd, m_device, m_transientCmdPool);
    m_resources.staging.cmdUploadAppended(upCmd);
    NVVK_CHECK(vkEndCommandBuffer(upCmd));
    m_loadPipeline.enqueue(upCmd);
  }
}

//--------------------------------------------------------------------------------------------------
// Clear the G-Buffer
void GltfRenderer::clearGbuffer(VkCommandBuffer cmd)
{
  VkImageSubresourceRange range = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
  vkCmdClearColorImage(cmd, m_resources.gBuffers.getColorImage(Resources::eImgTonemapped), VK_IMAGE_LAYOUT_GENERAL,
                       &kBackgroundClearColor, 1, &range);

  // Ensure the clear operation completes before any subsequent reads from this image
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
}

//--------------------------------------------------------------------------------------------------
// Create the uniform buffer for frame-specific data: per-frame camera matrices, environment
// settings and debug information, updated each frame.
//
void GltfRenderer::createResourceBuffers()
{
  // Create the buffer of the current camera transformation, changing at each frame
  NVVK_CHECK(m_resources.allocator.createBuffer(m_resources.bFrameInfo, sizeof(shaderio::SceneFrameInfo),
                                                VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                                VMA_MEMORY_USAGE_CPU_TO_GPU));
  NVVK_DBG_NAME(m_resources.bFrameInfo.buffer);
}

//--------------------------------------------------------------------------------------------------
// Create the descriptor set and the pipelines
// There are two descriptor: one for the textures (set) and one (push) for the top level acceleration structure and the default output image
// There are two pipelines: one for the PathTracer and one for the Rasterizer
// The descriptor set is shared between the two pipelines
void GltfRenderer::createDescriptorSets()
{
  // Reserve texture descriptors (m_maxTextures from renderer.hpp, clamped to device limits)
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(m_app->getPhysicalDevice(), &deviceProperties);
  m_maxTextures = std::min(m_maxTextures, deviceProperties.limits.maxDescriptorSetSampledImages - 1);  // Set limits of sample textures (defaut: 100 000)
  m_maxSamplers = std::min(m_maxTextures, deviceProperties.limits.maxDescriptorSetSamplers - 1);

  // 0: Descriptor SET: scene material images (SAMPLED_IMAGE) and samplers (SAMPLER) are now separate
  // bindless arrays instead of one combined-image-sampler array (see GltfTextureInfo.index/samplerIndex).
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eTextures, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                                              m_maxTextures, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eSamplers, VK_DESCRIPTOR_TYPE_SAMPLER,
                                              m_maxSamplers, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  // 2D IBL/transmission textures (see shaderio.h HDR_* indices):
  //   [0] HDR_IMAGE_INDEX   lat-long HDR environment image
  //   [1] HDR_LUT_INDEX     GGX split-sum BRDF LUT
  //   [2] HDR_SHEEN_INDEX   Charlie sheen directional-albedo LUT (raster only)
  //   [3] HDR_OPAQUE_INDEX  opaque-pass color capture for screen-space transmission (raster only)
  // PARTIALLY_BOUND lets the path tracer leave slots [2]/[3] unbound.
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eTexturesHdr, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              HDR_TEXTURE_COUNT, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  // Prefiltered HDR cubemaps: [0] diffuse Lambertian, [1] GGX glossy.
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eTexturesCube,
                                              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  NVVK_CHECK(m_resources.descriptorBinding[0].createDescriptorSetLayout(
      m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT, &m_resources.descriptorSetLayout[0]));
  NVVK_DBG_NAME(m_resources.descriptorSetLayout[0]);

  // Pool sizes are derived from descriptorBinding[0] (now SAMPLED_IMAGE + SAMPLER for the scene, plus
  // COMBINED_IMAGE_SAMPLER for the HDR/cube arrays). Each nvapp::ImTexture (viewport: 1, DLSS guides: 8,
  // OptiX denoised: 1 + margin) additionally uses one VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE descriptor.
  constexpr uint32_t                kImTextureMaxSets = 15;
  std::vector<VkDescriptorPoolSize> poolSize          = m_resources.descriptorBinding[0].calculatePoolSizes();
  poolSize.push_back({VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kImTextureMaxSets});
  VkDescriptorPoolCreateInfo dpoolInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |  // allows descriptor sets to be updated after they have been bound to a command buffer
               VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,  // individual descriptor sets can be freed from the descriptor pool
      .maxSets       = kImTextureMaxSets + 1,                      // ImTexture sets + the scene texture set
      .poolSizeCount = uint32_t(poolSize.size()),
      .pPoolSizes    = poolSize.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(m_device, &dpoolInfo, nullptr, &m_resources.descriptorPool));
  NVVK_DBG_NAME(m_resources.descriptorPool);

  VkDescriptorSetAllocateInfo allocInfo = {
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_resources.descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_resources.descriptorSetLayout[0],
  };
  NVVK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_resources.descriptorSet));
  NVVK_DBG_NAME(m_resources.descriptorSet);


  // 1: Descriptor PUSH: top level acceleration structure and the output image
  m_resources.descriptorBinding[1].addBinding(shaderio::BindingPoints::eTlas,
                                              VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
  m_resources.descriptorBinding[1].addBinding(shaderio::BindingPoints::eOutImages, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10,
                                              VK_SHADER_STAGE_ALL);
  m_resources.descriptorBinding[1].addBinding(shaderio::BindingPoints::eOutDepth, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                              VK_SHADER_STAGE_ALL);

  NVVK_CHECK(m_resources.descriptorBinding[1].createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                                                        &m_resources.descriptorSetLayout[1]));
  NVVK_DBG_NAME(m_resources.descriptorSetLayout[1]);
}

//--------------------------------------------------------------------------------------------------
// Load an environment map / scene by path.
//
// These back the --hdrfile / --scenefile parameters, which carry a callbackSuccess so that setting
// them does the load -- from the command line, a benchmark sequence, or nvpro_set_parameters alike
// (nvmcp invokes the callback on the application thread). Before that, writing either was accepted
// and did nothing, because both were only read once during start-up.
//
// Path resolution is delegated to createHDR / createScene: both call nvutils::findFile against
// nvsamples::getResourcesDirs() so bare resource-relative inputs (e.g. `--hdrfile studio.hdr`
// where studio.hdr ships under the sample's resources dir) work exactly like a cwd-relative or
// absolute path. Reporting success/failure is likewise the loaders' job: HdrIbl::isValid() flips
// off with a warning log when its input cannot be loaded, and Scene creation leaves getScene()
// null when it bails -- so we no longer need a pre-flight std::filesystem::exists() gate here
// that would reject resource-relative inputs the loader would otherwise have found.
bool GltfRenderer::loadHdrEnvironment(const std::filesystem::path& filename)
{
  // Not attached yet: this is the start-up command-line parse, and main() performs that load
  // itself once the device exists.
  if(!m_app)
    return false;

  createHDR(filename);
  if(!m_resources.hdrIbl.isValid())
    return false;

  // Loading an environment the renderer is not told to sample would be another silent no-op.
  m_resources.settings.envSystem = shaderio::EnvSystem::eHdr;
  // A new HDR changes hdrIbl.getIntegral(), which defaultFireflyClamp() uses when envSystem is
  // eHdr. Refresh the path-tracer clamp in lock-step -- matching the pattern in onFileDrop, the
  // Agentic applyHdri callback, and the HDR-picker UI -- so a --hdrfile change from a benchmark
  // sequence or MCP write doesn't keep clamping against the previous HDR's integral.
  calibrateFireflyClamp();
  resetFrame();
  return true;
}

bool GltfRenderer::loadSceneFile(const std::filesystem::path& filename)
{
  if(!m_app)
    return false;

  // Programmatic full-scene replacement (CLI/benchmark/MCP). Mirror the teardown that the
  // onFileDrop replace-path runs before `createScene` -- without it, we leak/keep-stale the
  // previous scene's derived state: sceneRtx BLAS/TLAS, sceneGpu buffers, transformCompute GPU
  // buffers, undo stack, thumbnails, hover/pick, animation & interactivity controls, and the
  // rasterizer's recorded secondary command buffer (see cleanupScene()'s comment naming itself
  // "the authoritative invalidation point"). createHDR self-quiesces so loadHdrEnvironment does
  // not need a parallel wrapper-level teardown; createScene does not, so we do it here.
  //
  // Synchronous (unlike onFileDrop, which threads the load behind a busy indicator) because a
  // benchmark/MCP client needs the scene fully installed on the GPU before this call returns --
  // otherwise the next sequenced parameter change would race the upload.
  //
  // Consequence: a failed load leaves an empty scene rather than the previous one. That is the
  // right failure mode for programmatic clients (an obviously empty capture beats a silently
  // wrong one attributed to the requested filename); the caller in main.cpp also emits a LOGW
  // naming the offending file.
  vkQueueWaitIdle(m_app->getQueue(0).queue);
  m_loadPipeline.clear();
  cleanupScene();

  // Return createScene's own outcome, not `getScene() != nullptr`: even after cleanupScene()
  // above, we want the return to describe what the *new* attempt did, not accidentally report
  // success just because some scene pointer happens to be non-null.
  return createScene(filename);
}

//--------------------------------------------------------------------------------------------------
// Recompile the active renderer's shaders and show the result.
// SYNC NOTE: the recompile destroys live pipelines, so the queue must be drained first.
bool GltfRenderer::reloadShaders()
{
  vkQueueWaitIdle(m_app->getQueue(0).queue);
  const bool compiled = compileShaders();
  resetFrame();
  return compiled;
}

//--------------------------------------------------------------------------------------------------
// Recompile the shaders of the current renderer. See onUIMenu() for the key binding
bool GltfRenderer::compileShaders()
{
  nvutils::ScopedTimer st(__FUNCTION__);
  if(m_resources.settings.renderSystem == RenderingMode::ePathtracer)
  {
    return m_pathTracer.reloadShader(m_resources);
  }
  return m_rasterizer.compileShader(m_resources, true);
}

//--------------------------------------------------------------------------------------------------
// Update the textures: this is called when the scene is loaded
// Textures are updated in the descriptor set (0)
bool GltfRenderer::updateTextures()
{
  const uint32_t imageCount = m_resources.sceneVk.textureCount();
  if(imageCount == 0)
    return true;
  return writeTextureDescriptorRange(0, imageCount, 0, m_resources.sceneVk.samplerCount());
}

//--------------------------------------------------------------------------------------------------
// Write a contiguous range of the scene texture (eTextures) and sampler (eSamplers) descriptor arrays.
// updateTextures() writes the whole set (elements 0..count); applyPendingTextureTailSync() writes only
// the appended tail. The arrays are bindless (UPDATE_AFTER_BIND + PARTIALLY_BOUND), so a partial write
// leaves the untouched slots -- and any in-flight frames referencing them -- valid.
bool GltfRenderer::writeTextureDescriptorRange(uint32_t firstTexture, uint32_t textureCount, uint32_t firstSampler, uint32_t samplerCount)
{
  // CRITICAL: materials index eTextures[] / eSamplers[] directly; exceeding capacity would read
  // uninitialized descriptors (undefined behavior). Fail loudly instead.
  if(firstTexture + textureCount > m_maxTextures)
  {
    LOGE("FATAL: Scene needs %u material images but the descriptor set only supports %u!", firstTexture + textureCount, m_maxTextures);
    LOGE("       Increase m_maxTextures in renderer.hpp, or reduce/deduplicate scene textures.");
    return false;
  }
  if(firstSampler + samplerCount > m_maxSamplers)
  {
    LOGE("FATAL: Scene needs %u samplers but the descriptor set only supports %u!", firstSampler + samplerCount, m_maxSamplers);
    return false;
  }

  nvvk::WriteSetContainer write{};

  // eTextures: SAMPLED_IMAGE array. The nvvk::Image descriptors supply imageView + layout; their
  // sampler field is ignored by Vulkan for this descriptor type.
  if(textureCount > 0)
  {
    VkWriteDescriptorSet images =
        m_resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eTextures, m_resources.descriptorSet,
                                                     firstTexture, textureCount);
    write.append(images, m_resources.sceneVk.textures().data() + firstTexture);
  }

  // eSamplers: SAMPLER array. Only the sampler field is used; imageView stays null. samplerInfos must
  // outlive vkUpdateDescriptorSets below, so it lives at function scope.
  std::vector<VkDescriptorImageInfo> samplerInfos;
  if(samplerCount > 0)
  {
    samplerInfos.resize(samplerCount);
    const std::vector<VkSampler>& samplers = m_resources.sceneVk.samplers();
    for(uint32_t i = 0; i < samplerCount; ++i)
      samplerInfos[i] = VkDescriptorImageInfo{
          .sampler = samplers[firstSampler + i], .imageView = VK_NULL_HANDLE, .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkWriteDescriptorSet allSamplers =
        m_resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eSamplers, m_resources.descriptorSet,
                                                     firstSampler, samplerCount);
    write.append(allSamplers, samplerInfos.data());
  }

  if(write.size() > 0)
    vkUpdateDescriptorSets(m_device, write.size(), write.data(), 0, nullptr);
  return true;
}

//--------------------------------------------------------------------------------------------------
// Update the HDR images : add the 2D images to allTextures and the cube images to allTexturesCube
//
void GltfRenderer::updateHdrImages()
{
  const std::vector<nvvk::Image>& hdrPreconvolutedTextures = m_resources.hdrDome.getTextures();
  nvvk::WriteSetContainer         write{};
  VkWriteDescriptorSet hdrTextures = m_resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eTexturesHdr,
                                                                                  m_resources.descriptorSet, HDR_IMAGE_INDEX, 1U);
  // Adding the HDR image (RGBA32F)
  write.append(hdrTextures, m_resources.hdrIbl.getHdrImage());
  // Add pre-integrated LUT BRDF
  hdrTextures.dstArrayElement = HDR_LUT_INDEX;
  write.append(hdrTextures, hdrPreconvolutedTextures[2]);

  // Adding cube images: diffuse, glossy
  VkWriteDescriptorSet hdrTexturesCube =
      m_resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eTexturesCube, m_resources.descriptorSet, 0, 2U);
  write.append(hdrTexturesCube, m_resources.hdrDome.getTextures().data());

  vkUpdateDescriptorSets(m_device, write.size(), write.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Reset the frame counter
void GltfRenderer::resetFrame()
{
  m_resources.frameCount = -1;
}

// Select a scene-graph node by index from a script parameter (--selectNode). The index is a glTF node
// index, exactly as shown in the Scene Browser tree. This routes through the shared SceneSelection so the
// Scene Browser highlight, the Inspector contents, and the render-node selection set all update precisely
// as they would from a click in the tree (selectNode emits NodeSelected, which the UI callback turns into
// the selectedRenderNodes set). A negative or out-of-range index clears the selection. Used by the
// windowed scripted-run harness so UI/UX changes to the scene list and inspector can be exercised and
// captured with --uiScreenshot.
void GltfRenderer::selectSceneNodeFromScript(int nodeIndex)
{
  nvvkgltf::Scene* scene = m_resources.getScene();
  const int        count = (scene && scene->valid()) ? static_cast<int>(scene->getModel().nodes.size()) : 0;
  if(nodeIndex >= 0 && nodeIndex < count)
  {
    m_sceneSelection.selectNode(nodeIndex);
  }
  else
  {
    if(nodeIndex >= 0)
    {
      LOGW("--selectNode %d ignored: scene has %d node(s)\n", nodeIndex, count);
    }
    // clearSelection() does not emit an event, so drop the render-node set explicitly.
    m_sceneSelection.clearSelection();
    m_resources.selectedRenderNodes.clear();
  }
  resetFrame();
}

void GltfRenderer::pickSceneNodeFromScript(int nodeIndex)
{
  nvvkgltf::Scene* scene = m_resources.getScene();
  if(!scene || !scene->valid())
    return;

  const int nodeCount = static_cast<int>(scene->getModel().nodes.size());
  if(nodeIndex < 0 || nodeIndex >= nodeCount)
  {
    updateSelectionFromPick(-1);
    resetFrame();
    return;
  }

  // Find a render node for this glTF node (first primitive) - same "which render node represents
  // this glTF node" question a real click's TLAS instance ID answers implicitly.
  const auto& renderNodes   = scene->getRenderNodes();
  int         renderNodeIdx = -1;
  for(size_t i = 0; i < renderNodes.size(); ++i)
  {
    if(renderNodes[i].refNodeID == nodeIndex)
    {
      renderNodeIdx = static_cast<int>(i);
      break;
    }
  }
  if(renderNodeIdx < 0)
  {
    LOGW("pickrendernode %d ignored: glTF node has no render primitive (not a mesh node?)\n", nodeIndex);
    updateSelectionFromPick(-1);
    resetFrame();
    return;
  }

  // Synthetic but finite ray data (no real cursor/GPU raycast in a scripted run): the current camera
  // eye as the ray origin, the node's own world-space position as the hit point - enough to exercise
  // KHR_interactivity event/onSelect's selectionPoint/selectionRayOrigin end-to-end.
  glm::dvec3 eye, center, up;
  m_cameraManip->getLookat(eye, center, up);
  const auto&     worldMatrices = scene->getNodesWorldMatrices();
  const glm::vec3 worldPos =
      nodeIndex < static_cast<int>(worldMatrices.size()) ? glm::vec3(worldMatrices[nodeIndex][3]) : glm::vec3(0);
  updateSelectionFromPick(renderNodeIdx, worldPos, glm::vec3(eye));
  resetFrame();
}

void GltfRenderer::onUndoRedo()
{
  // Undo/redo may re-introduce geometry (e.g. redo of an added primitive) that has no GPU buffers yet;
  // rebuild before the next frame's acceleration-structure build tries to reference them.
  reconcileGeometryIfNeeded();
  resetFrame();
  m_sceneBrowser.markCachesDirty();
  m_inspector.refreshTextureNames();  // undo/redo of a texture import changes the texture set
}

//--------------------------------------------------------------------------------------------------
// Update the frame counter
// This is called every frame to update the frame counter or to reset it if the camera has changed
// The frame counter is used to limit the number of frames rendered
// If the frame counter is greater than the maximum number of frames, the rendering stops
// Returns true if the frame counter is less than the maximum number of frames
bool GltfRenderer::updateFrameCounter()
{
  ++m_resources.renderPassCount;  // monotonic - never reset by resetFrame(), see its doc comment

  static nvutils::CameraManipulator::Camera ref_camera{};

  const auto currentCamera = m_cameraManip->getCamera();

  if(ref_camera != currentCamera)
  {
    resetFrame();
    ref_camera = currentCamera;
  }

  if(m_resources.frameCount >= m_resources.settings.maxFrames)
  {
    return false;
  }
  m_resources.frameCount++;
  return true;
}

//--------------------------------------------------------------------------------------------------
// Create or load the HDR environment map
// If the filename is empty, a default environment map (empty) is created, which allow the descriptor set to be updated
void GltfRenderer::createHDR(const std::filesystem::path& hdrFilename)
{
  // Agentic HDRI reload can happen mid-frame (pollNow from renderUI). Wait until
  // in-flight work finishes before tearing down env images the rasterizer/path
  // tracer may still be sampling.
  NVVK_CHECK(vkDeviceWaitIdle(m_device));

  VkCommandBuffer cmd{};
  nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
  nvvk::FrameUploader uploader;
  NVVK_CHECK(uploader.init({
      .allocator = &m_resources.allocator,
      .blockSize = 1024ull * 1024 * 1024,  // 8K RGBA32F HDR is 1 GiB
      .debugName = "hdrUpload",
  }));

  // Load an HDR and create the important sampling acceleration structure
  std::filesystem::path filename;
  if(!hdrFilename.empty())
  {
    if(std::filesystem::exists(hdrFilename))
      filename = hdrFilename;
    else
      filename = nvutils::findFile(hdrFilename, nvsamples::getResourcesDirs(), false);
  }
  // Decode here rather than through HdrIbl's path overload, so .hdr and .exr take one route --
  // see env_image_loader.hpp for why EXR cannot live inside nvvk. An empty or failed decode hands
  // over an empty span, which is what produces the dummy environment at start-up.
  EnvImage envImage = loadEnvImage(filename);

  // Say which file this is, or that there is none. A request that does not resolve, or an image
  // that does not decode, silently becomes the dummy environment above -- and a dummy is easy to
  // mistake for a real one, because plenty of studio HDRs genuinely are flat and achromatic.
  // Without a line here the only way to tell them apart is to know what the file looks like.
  if(hdrFilename.empty())
  {
    LOGI("HDR environment: none (dummy)\n");
  }
  else if(envImage.pixels.empty())
  {
    LOGW("HDR environment: %s could not be %s; using the dummy environment\n",
         nvutils::utf8FromPath(hdrFilename).c_str(), filename.empty() ? "found" : "decoded");
  }
  else
  {
    LOGI("HDR environment: %s (%ux%u)\n", nvutils::utf8FromPath(filename).c_str(), envImage.size.width, envImage.size.height);
  }

  m_resources.hdrIbl.destroyEnvironment();
  m_resources.hdrIbl.loadEnvironment(cmd, uploader, std::span<float>(envImage.pixels), envImage.size, true);
  // Remembered so that switching to a baked sky and back to eHdr can rebind hdrIbl to this file:
  // EnvBaker::commit repoints hdrIbl at its own image, and only a reload undoes that. The resolved
  // path, not the request: a bare name found through the resources directories would otherwise be
  // saved as a panorama URI that resolves nowhere. Empty when nothing decoded -- the dummy is not a
  // panorama, and saving it as one would describe an environment that does not exist.
  m_lastHdrFile     = envImage.pixels.empty() ? std::filesystem::path{} : filename;
  m_envOwnedByBaker = false;

  uploader.cmdUploadAppended(cmd);

  // Generate mipmaps for the HDR image
  VkExtent2D hdrSize = m_resources.hdrIbl.getHdrImageSize();
  if(hdrSize.width > 1 && hdrSize.height > 1)
  {
    nvvk::cmdGenerateMipmaps(cmd, m_resources.hdrIbl.getHdrImage().image, hdrSize, nvvk::mipLevels(hdrSize));
  }

  nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_app->getQueue(0).queue);
  uploader.deinit();

  // Create the diffuse and glossy cube maps for the HDR image (raster)
  m_resources.hdrDome.create(m_resources.hdrIbl.getDescriptorSet(), m_resources.hdrIbl.getDescriptorSetLayout(),
                             std::span(hdr_prefilter_diffuse_slang), std::span(hdr_prefilter_glossy_slang),
                             std::span(hdr_integrate_brdf_slang), std::span(hdr_dome_slang));

  updateHdrImages();
  m_resources.hdrDome.setOutImage(m_resources.gBuffers.getColorStorageImageInfo(Resources::eImgRendered));
  m_rasterizer.resetHdr();
  // addToRecentFiles(hdrFilename);
}

//--------------------------------------------------------------------------------------------------
// Destroy the resources
// Resource cleanup follows a specific order to prevent validation errors:
// 1. First flush any pending command buffers to ensure GPU work is complete
// 2. Then destroy higher-level objects before their dependencies
// 3. Finally clean up allocator after all resources using it are destroyed
// This ensures proper synchronization and prevents use-after-free errors
void GltfRenderer::destroyResources()
{
#ifdef USE_AGENTIC
  // Free the beautified image + its ImGui texture first, while the allocator and
  // ImGui backend are still alive and before the descriptor pool / layouts go.
  m_agentic.deinit();
#endif

  m_loadPipeline.destroy();

  m_resources.allocator.destroyBuffer(m_resources.bFrameInfo);
  if(m_resources.bSelectionBitMask.buffer != VK_NULL_HANDLE)
    m_resources.allocator.destroyBuffer(m_resources.bSelectionBitMask);

  vkDestroyDescriptorSetLayout(m_device, m_resources.descriptorSetLayout[0], nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_resources.descriptorSetLayout[1], nullptr);
  vkDestroyDescriptorPool(m_device, m_resources.descriptorPool, nullptr);
  vkDestroyCommandPool(m_device, m_transientCmdPool, nullptr);

  m_profilerGpuTimer.deinit();
  {
#ifdef USE_NVMCP
    // Automation (MCP) reads the timeline from a worker thread under this same lock, so take it
    // here: the release only happens once no read is in flight, and every later read sees nullptr.
    const std::lock_guard<std::mutex> lock(m_profilerTimelineMutex);
#endif
    g_profilerManager.destroyTimeline(m_profilerTimeline);
    m_profilerTimeline = nullptr;
  }
  m_silhouette.deinit(m_resources);
  m_hoverPicker.deinit(m_resources);
  m_skyBruneton.deinit(m_resources);
  m_envBaker.deinit(m_resources);

  m_resources.tonemapper.deinit();
  m_resources.appMemoryTracker.untrack("GBuffers", m_resources.gBuffers, Resources::eImgCount);
  m_resources.tonemappedUi.deinit();
  m_resources.gBuffers.deinit();
  m_resources.transformCompute.deinit();
  m_resources.sceneGpu.deinit();
  m_resources.hdrIbl.deinit();
  m_resources.hdrDome.deinit();
  m_resources.samplerPool.deinit();
  m_resources.staging.deinit();
  m_rayPicker.deinit();
  m_resources.allocator.deinit();
}


//--------------------------------------------------------------------------------------------------
// Update the scene animation
// - If there is an animation in the scene, and animation is enabled, update the animation
// - Update the relevant buffers and acceleration structures
// - Reset the frame counter to restart progressive rendering
//
bool GltfRenderer::updateAnimation(VkCommandBuffer cmd)
{
  nvvkgltf::Scene* scnPtr = m_resources.getScene();
  if(!scnPtr)
    return false;
  nvvkgltf::Scene&  scn      = *scnPtr;
  AnimationControl& animCtrl = m_resources.animationControl;


  if(ui::animation::hasPlayableAnimation(scnPtr) && animCtrl.doAnimation())
  {
    const int nAnim = scn.animation().getNumAnimations();
    if(nAnim <= 0)
      return false;
    if(animCtrl.currentAnimation < 0 || animCtrl.currentAnimation >= nAnim)
      animCtrl.currentAnimation = 0;

    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Update animation");
    NVVK_DBG_SCOPE(cmd);

    float                    deltaTime = animCtrl.deltaTime();
    nvvkgltf::AnimationInfo& animInfo  = scn.animation().getAnimationInfo(animCtrl.currentAnimation);
    if(animCtrl.isReset())
      animInfo.reset();
    else
      animInfo.incrementTime(deltaTime);

    // Evaluate animation channels (marks Scene nodes dirty internally; also marks
    // render nodes for skins whose joints moved, and materials/lights for pointer channels)
    {
      auto t = m_profilerGpuTimer.cmdFrameSection(cmd, "Eval channels");
      if(!scn.animation().updateAnimation(animCtrl.currentAnimation))
        return false;
    }

    animCtrl.clearStates();
    reconcileAnimationGpuState(cmd);
    return true;
  }

  return false;
}

void GltfRenderer::reconcileAnimationGpuState(VkCommandBuffer cmd)
{
  nvvkgltf::Scene&    scn    = *m_resources.getScene();
  nvvkgltf::SceneVk&  scnVk  = m_resources.sceneVk;
  nvvkgltf::SceneRtx& scnRtx = m_resources.sceneRtx;

  // Recompute world matrices for dirty nodes and expand dirty flags to all affected
  // render nodes (including descendants needed for transform-only animated nodes).
  {
    auto t = m_profilerGpuTimer.cmdFrameSection(cmd, "World matrices + dirty");
    scn.updateNodeWorldMatrices();
  }

  scnRtx.updateInstanceFlagsCache(scn);

  const bool gpuTransform = m_resources.sceneGpu.shouldUseGpuTransform(scn);

  {
    auto t = m_profilerGpuTimer.cmdFrameSection(cmd, "Sync to GPU");
    if(gpuTransform)
    {
      // Animated emitters are handled upstream: updateNodeWorldMatrices() (called above) refreshes
      // the render-node world matrices and latches DirtyFlags::emissiveDirty when an emissive node
      // moves, so syncFromScene rebuilds the emitter list even though this restricted mask omits
      // render-node sync.
      (void)scnVk.syncFromScene(m_resources.staging, scn, nvvkgltf::SceneVk::eSyncMaterials | nvvkgltf::SceneVk::eSyncLights);
      (void)scnVk.flushSceneDescIfDirty(m_resources.staging, scn);
    }
    else
    {
      // A prior GPU transform frame may have moved nodes on-device only; reconcile just those before
      // the CPU sync sources render-node / TLAS transforms from the CPU mirror.
      if(scn.mergeGpuStaleNodesIntoDirty())
        scn.updateNodeWorldMatrices();
      markGpuTransformStaleAfterCpuSync(scn);
      (void)scnVk.syncFromScene(m_resources.staging, scn);
    }
  }

  bool hasMorphOrSkin = scn.animation().hasMorphTargets() || scn.animation().hasSkinning();
  if(hasMorphOrSkin)
  {
    auto timerSectionMorph = m_profilerGpuTimer.cmdFrameSection(cmd, "Morph or Skin");
    m_resources.sceneGpu.applyAnimation(cmd, scn);
  }

  {
    auto t = m_profilerGpuTimer.cmdFrameSection(cmd, "Staging flush");
    m_resources.staging.cmdUploadAppended(cmd);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                           VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT
                               | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
  }

  {
    auto timerSectionAS = m_profilerGpuTimer.cmdFrameSection(cmd, "AS update");
    if(hasMorphOrSkin)
      scnRtx.updateBottomLevelAS(cmd, scn);
    if(gpuTransform)
    {
      m_resources.transformCompute.dispatchTransformUpdate(cmd, m_resources.staging, scn, scnVk, scnRtx);
    }
    else
    {
      (void)scnRtx.syncTopLevelAS(cmd, m_resources.staging, scn);
    }
  }

  // Preserve the deferred texture/sampler signals across this per-frame flag clear: all three are
  // consumed at frame top (applyPendingTextureRebuild() / applyPendingTextureTailSync() /
  // applyPendingSamplerUpdate()), not here, so a texture/image/sampler edit or an import/undo made
  // while an animation plays must not be wiped before then.
  // materials/lights are also preserved: an animation channel that targets a material/light property
  // (KHR_animation_pointer, or an interactivity animation/start clip driving one) marks them dirty
  // inside scn.animation().updateAnimation() above, but this function's own GPU sync already
  // consumed that write - clearing them here (instead of after updateSceneChanges() sees them) would
  // silently skip notifyDlssContentReset()'s appearance-discontinuity check for an animation-driven
  // material/light change, the same DLSS temporal-history bug already fixed for pointer/set writes.
  const bool              pendingTextures    = scn.getDirtyFlags().texturesChanged;
  const bool              pendingTextureTail = scn.getDirtyFlags().texturesTailChanged;
  std::unordered_set<int> pendingSamplers    = std::move(scn.getDirtyFlags().samplers);
  std::unordered_set<int> pendingMaterials   = std::move(scn.getDirtyFlags().materials);
  std::unordered_set<int> pendingLights      = std::move(scn.getDirtyFlags().lights);
  scn.clearDirtyFlags();
  scn.getDirtyFlags().texturesChanged     = pendingTextures;
  scn.getDirtyFlags().texturesTailChanged = pendingTextureTail;
  scn.getDirtyFlags().samplers            = std::move(pendingSamplers);
  scn.getDirtyFlags().materials           = std::move(pendingMaterials);
  scn.getDirtyFlags().lights              = std::move(pendingLights);
}

//--------------------------------------------------------------------------------------------------
// KHR_interactivity: ticks the scene's default behavior graph instance once per frame. Returns true
// only when an animation/start-driven pose was applied this tick (see below) - a graph merely
// *existing* (or ticking with no writes) never resets progressive accumulation. pointer/set-driven
// glTF-model writes (Phase C) also reset it, but via the normal Scene::markNodeDirty/
// markMaterialDirty -> updateSceneChanges() dirty-flags path right after this call, not through this
// function's own return value - reconcileAnimationGpuState() below clears those same dirty flags
// before updateSceneChanges() ever sees them, so an animation/start pose has no other way to reach
// `changed` in onRender() and must be signaled explicitly. `cmd` is used directly for the animation
// GPU-reconcile path below, unlike the rest of this function.
//--------------------------------------------------------------------------------------------------
bool GltfRenderer::updateInteractivityGraphs(VkCommandBuffer cmd)
{
  nvvkgltf::Scene* scnPtr = m_resources.getScene();
  if(!scnPtr || scnPtr->getInteractivityGraphs().empty())
    return false;

  InteractivityControl& ctrl = m_resources.interactivityControl;
  nvvkgltf::InteractivityGraphInstance* instance = scnPtr->getInteractivityInstance(scnPtr->getDefaultInteractivityGraph());
  if(ctrl.resetRequested)
  {
    if(instance)
      instance->reset();
    ctrl.clearResetRequest();
  }
  if(!ctrl.play)
    return false;

  // Ticking already applies any animation/start-driven poses synchronously, CPU-side, via
  // InteractivityAnimationResolver::applyPose() (called from InteractivityGraphInstance::
  // advanceAnimations() - see gltf_interactivity_animation.hpp for why that has to happen
  // before `tick()` returns, not after: the spec applies the pose to the asset before firing
  // `done`, so a pointer/get reached from that same `done` activation must already see it).
  scnPtr->tickInteractivityGraphs(ImGui::GetIO().DeltaTime);

  // Only the heavier GPU-side reconciliation (world matrices, GPU sync, BLAS update) still needs
  // to happen here, once, after however many animation/start entries applyPose() touched this tick.
  const bool animationApplied = instance && !instance->pendingAnimationApplies().empty();
  if(animationApplied)
    reconcileAnimationGpuState(cmd);

  return animationApplied;
}

//--------------------------------------------------------------------------------------------------
// KHR_interactivity hover detection (docs/interactivity.md Phase E). Called once per frame, before
// updateInteractivityGraphs() so a transition detected this frame feeds this same frame's tick.
//
// If the cursor isn't over the viewport, clears hover synchronously (no GPU round-trip needed - we
// already know nothing is hovered). Otherwise polls m_hoverPicker non-blocking; on a genuinely new
// result, maps the render-node index to a glTF node (same renderNode.refNodeID mapping click-
// selection uses) and redirects to the nearest KHR_node_hoverability-hoverable ancestor, mirroring
// how click-selection redirects for KHR_node_selectability - see docs/interactivity.md for why this
// is a documented approximation of the spec's "skip non-hoverable geometry" ray-termination
// semantics, which a single G-buffer sample can't express.
//--------------------------------------------------------------------------------------------------
void GltfRenderer::updateHoverState()
{
  nvvkgltf::Scene* scene = m_resources.getScene();
  if(!scene || !scene->valid())
    return;

  if(!m_hoverCursorInViewport)
  {
    m_hoverPicker.pollResult(m_device);  // drain any in-flight readback so its slot isn't leaked
    if(m_hoveredNodeIndex != -1)
    {
      scene->notifyNodeHoverChanged(m_hoveredNodeIndex, -1);
      m_hoveredNodeIndex = -1;
    }
    return;
  }

  std::optional<int32_t> renderNodeResult = m_hoverPicker.pollResult(m_device);
  if(!renderNodeResult.has_value())
    return;  // nothing new completed since the last poll

  int           newGltfNode   = -1;
  const int32_t renderNodeIdx = *renderNodeResult;
  const auto&   renderNodes   = scene->getRenderNodes();
  if(renderNodeIdx >= 0 && renderNodeIdx < static_cast<int>(renderNodes.size()))
    newGltfNode = scene->nearestHoverableAncestor(renderNodes[renderNodeIdx].refNodeID);

  if(newGltfNode == m_hoveredNodeIndex)
    return;

  scene->notifyNodeHoverChanged(m_hoveredNodeIndex, newGltfNode);
  m_hoveredNodeIndex = newGltfNode;
}

//--------------------------------------------------------------------------------------------------
// Non-blocking poll of the async click ray-pick (see m_pendingClickPick's comment in renderer.hpp).
// Once the frame that recorded the pick has signaled, nvvk::RayPicker::getResult() is a plain
// host-mapped-buffer read (confirmed in nvvk/ray_picker.cpp - no internal wait of its own), so this
// is a cheap poll-then-memcpy, exactly like m_hoverPicker.pollResult()'s pattern.
//--------------------------------------------------------------------------------------------------
void GltfRenderer::updateClickPickState()
{
  if(!m_pendingClickResult || !m_pendingClickResult->semaphoreState.testSignaled(m_device))
    return;

  const bool isDoubleClick = m_pendingClickResult->isDoubleClick;
  m_pendingClickResult.reset();
  applyClickPickResult(m_rayPicker.getResult(), isDoubleClick);
}

// Script-driven hover for UI scenario tests (`hovernode <id>` / `clearhover` -> nodeIndex -1) -
// calls the exact same notification path updateHoverState() uses for real cursor input.
void GltfRenderer::hoverSceneNodeFromScript(int nodeIndex)
{
  nvvkgltf::Scene* scene = m_resources.getScene();
  if(!scene || !scene->valid() || nodeIndex == m_hoveredNodeIndex)
    return;
  if(nodeIndex != -1 && (nodeIndex < 0 || nodeIndex >= static_cast<int>(scene->getModel().nodes.size())))
    return;
  scene->notifyNodeHoverChanged(m_hoveredNodeIndex, nodeIndex);
  m_hoveredNodeIndex = nodeIndex;
}

//--------------------------------------------------------------------------------------------------
// updateSceneChanges
//
// Syncs CPU scene state (dirty flags) to GPU: SceneVk buffers (materials, lights, render nodes,
// vertices) and SceneRtx acceleration structures (BLAS/TLAS). Called once per frame before draw.
//
//--------------------------------------------------------------------------------------------------
// updateSceneChanges phase helpers
//--------------------------------------------------------------------------------------------------
void GltfRenderer::updateSceneChanges_BlasRebuild(const nvvkgltf::Scene::DirtyFlags& df)
{
  if(df.primitivesChanged)
    buildAccelerationStructures();
}

void GltfRenderer::updateSceneChanges_NodeTransforms(VkCommandBuffer cmd, nvvkgltf::Scene* scene, const nvvkgltf::Scene::DirtyFlags& df)
{
  if(df.nodes.empty())
    return;

  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "UpdateNodeWorldMatrices");
  scene->updateNodeWorldMatrices();
}

uint32_t GltfRenderer::updateSceneChanges_SyncGpuBuffers(VkCommandBuffer cmd, nvvkgltf::Scene* scene)
{
  uint32_t synced = m_resources.sceneVk.syncFromScene(m_resources.staging, *scene);

  if(m_resources.sceneVk.flushSceneDescIfDirty(m_resources.staging, *scene))
    synced |= nvvkgltf::SceneVk::eSyncRenderNodes;

  if(synced != nvvkgltf::SceneVk::eSyncNone)
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "SyncGpuBuffers");
    // A prior staging flush earlier this frame (updateAnimation's "Staging flush") may have written
    // some of these same buffers (e.g. m_bSceneDesc when a structural change re-dirties it). Those
    // writes were exposed to compute / AS-build / vertex-input consumers, so this second transfer
    // write must be ordered after both the prior transfer write (WAW) and those readers (WAR).
    nvvk::cmdMemoryBarrier(cmd,
                           VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
                               | VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT,
                           VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
                               | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT);
    m_resources.staging.cmdUploadAppended(cmd);
  }
  return synced;
}

//--------------------------------------------------------------------------------------------------
// After a CPU transform sync, tell the GPU transform path what it no longer holds. Call once the CPU
// world-matrix walk has run, so the render-node dirty sets say whether geometry moved.
//
// A frame that moved only mesh-less nodes -- the sky's sun on every frame of a Time of Day drag --
// changed those nodes' locals and nothing else, so only they are re-uploaded on the next GPU frame.
// Anything else keeps the full re-upload, the conservative choice this path always made.
//
void GltfRenderer::markGpuTransformStaleAfterCpuSync(const nvvkgltf::Scene& scene)
{
  const auto& df = scene.getDirtyFlags();
  if(df.isEmpty())
    return;
  if(!df.nodes.empty() && !scene.dirtyNodesReachGeometry())
    m_resources.transformCompute.markLocalsStale(df.nodes);
  else
    m_resources.transformCompute.markGpuStale();
}

void GltfRenderer::updateSceneChanges_TlasUpdate(VkCommandBuffer cmd, nvvkgltf::Scene* scene)
{
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "SyncTopLevelAS");
  (void)m_resources.sceneRtx.syncTopLevelAS(cmd, m_resources.staging, *scene);
}

void GltfRenderer::updateSceneChanges_RasterizerInvalidate(bool renderNodeOrNodeDirty)
{
  if(renderNodeOrNodeDirty)
    m_rasterizer.freeRecordCommandBuffer(m_resources);
}

void GltfRenderer::updateSceneChanges_TangentUpload(VkCommandBuffer cmd, nvvkgltf::Scene* scene, bool& changed)
{
  (void)cmd;
  if(m_resources.dirtyFlags.test(DirtyFlags::eDirtyTangents))
  {
    m_resources.sceneVk.uploadVertexBuffers(m_resources.staging, *scene);
    m_resources.dirtyFlags.reset(DirtyFlags::eDirtyTangents);
    changed = true;
  }
}

void GltfRenderer::updateSceneChanges_Finalize(VkCommandBuffer cmd, bool changed, bool stagingFlushed, nvvkgltf::Scene* scene)
{
  if(changed && !stagingFlushed)
    m_resources.staging.cmdUploadAppended(cmd);

  // Preserve a pending sampler edit across this clear: it is only consumed at the next frame's top
  // (applyPendingSamplerUpdate()), not here, so an Inspector edit made this frame must survive until then.
  if(nvvkgltf::Scene* s = m_resources.getScene())
  {
    std::unordered_set<int> pendingSamplers = std::move(s->getDirtyFlags().samplers);
    s->clearDirtyFlags();
    s->getDirtyFlags().samplers = std::move(pendingSamplers);
  }

#ifndef NDEBUG
  if(changed && m_validateGpuSync && scene && !m_skipGpuSyncValidation)
  {
    auto mismatches = m_resources.sceneVk.validateGpuSync(*scene, m_resources.sceneRtx.getTlasInstances());
    for(const auto& m : mismatches)
      LOGE("GPU sync mismatch: %s\n", m.description.c_str());
    assert(mismatches.empty() && "GPU sync validation failed -- see log for details");
  }
#endif
}

//--------------------------------------------------------------------------------------------------
// Sync scene dirty state to GPU (materials, lights, render nodes, BLAS/TLAS, rasterizer state).
// Returns true if any change was applied (caller may reset frame counter for progressive rendering).
// All dirty state lives in Scene; this function clears dirty flags at the end.
//--------------------------------------------------------------------------------------------------
bool GltfRenderer::updateSceneChanges(VkCommandBuffer cmd)
{
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

  nvvkgltf::Scene* scene = m_resources.getScene();
  if(!scene)
    return false;

#ifndef NDEBUG
  m_skipGpuSyncValidation = false;
#endif

  // A texture-set change is pending (structural via texturesChanged, or a tail import/undo/redo via
  // texturesTailChanged). The GPU material buffer must not be updated to reference a new texture index
  // before eTextures[] contains it -- that write runs at the next frame top (applyPendingTextureRebuild()
  // / applyPendingTextureTailSync()). Skip this frame's sync and keep rendering the previous,
  // self-consistent GPU state; the dirty flags persist until then.
  if(scene->getDirtyFlags().texturesChanged || scene->getDirtyFlags().texturesTailChanged)
    return false;

  const auto& df             = scene->getDirtyFlags();
  bool        changed        = !df.isEmpty();
  bool        stagingFlushed = false;

  // A mesh-less node (a light, a camera) moving changes nothing the recorded draws depend on.
  bool renderNodeOrNodeDirty = df.allRenderNodesDirty || !df.renderNodesVk.empty() || scene->dirtyNodesReachGeometry();

  // Material edit may have added or removed a KHR_materials_* extension; refresh the
  // scene feature set so optimal-mode shader rebuild picks it up. Cheap check (walk
  // a few maps per material); only runs when materials are actually dirty.
  if(!df.materials.empty())
    m_resources.recomputeSceneFeatures(dlssGuideRequired());

  // A material/light property changed (e.g. a KHR_interactivity pointer/set writing a
  // texture-transform offset, or a live Inspector edit) - this is an appearance discontinuity
  // motion vectors can't describe (nothing moved; the surface's content just changed), so DLSS's
  // temporal history would blend old and new content at the same screen location. Drop it via the
  // active renderer's own DLSS instance, same as a resize/quality-change discontinuity. Node
  // transform changes alone don't need this - they're already correctly handled by instance
  // motion vectors (see the ordering comment in onRender() above / docs/denoising.md).
  if(!df.materials.empty() || !df.lights.empty())
  {
    switch(m_resources.settings.renderSystem)
    {
      case RenderingMode::ePathtracer:
        m_pathTracer.notifyDlssContentReset(m_resources);
        break;
      case RenderingMode::eRasterizer:
        m_rasterizer.notifyDlssContentReset(m_resources);
        break;
    }
  }

  updateSceneChanges_BlasRebuild(df);
  m_resources.sceneRtx.updateInstanceFlagsCache(*scene);

  const bool gpuTransform = m_resources.sceneGpu.shouldUseGpuTransform(*scene);

  if(gpuTransform)
  {
    // GPU handles world-matrix propagation, render-node updates, and TLAS rebuild.
    // CPU only refreshes local matrices (for staging upload) and light world matrices.
    if(!df.nodes.empty())
    {
      auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "UpdateLocalMatrices");
      scene->updateLocalMatricesAndLights();
    }

    uint32_t synced = m_resources.sceneVk.syncFromScene(m_resources.staging, *scene,
                                                        nvvkgltf::SceneVk::eSyncMaterials | nvvkgltf::SceneVk::eSyncLights);
    if(m_resources.sceneVk.flushSceneDescIfDirty(m_resources.staging, *scene))
      synced |= nvvkgltf::SceneVk::eSyncRenderNodes;

    if(synced != nvvkgltf::SceneVk::eSyncNone)
    {
      auto timerSectionSync = m_profilerGpuTimer.cmdFrameSection(cmd, "SyncGpuBuffers");
      m_resources.staging.cmdUploadAppended(cmd);
    }
    stagingFlushed = (synced != nvvkgltf::SceneVk::eSyncNone);

    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    {
      auto timerSectionGpu = m_profilerGpuTimer.cmdFrameSection(cmd, "GPU transform + TLAS update");
      m_resources.transformCompute.dispatchTransformUpdate(cmd, m_resources.staging, *scene, m_resources.sceneVk,
                                                           m_resources.sceneRtx);
    }
#ifndef NDEBUG
    m_skipGpuSyncValidation = true;
#endif
  }
  else
  {
    // If the GPU transform path moved nodes on-device, the CPU world-matrix mirror is stale for just
    // those nodes. Only reconcile when this CPU frame actually syncs (df not empty): the upcoming
    // syncFromScene / syncTopLevelAS source transforms from the mirror, so the moved nodes must be
    // current (otherwise a moved object snaps to its stale CPU transform on a full TLAS rebuild). On
    // idle frames the GPU-side transforms stay authoritative, so we leave the stale set pending and skip
    // the work — this is what keeps large scenes from hitching after a gizmo release.
    if(!df.isEmpty() && scene->mergeGpuStaleNodesIntoDirty())
      renderNodeOrNodeDirty = true;  // moved-node transforms changed -> invalidate rasterizer accumulation

    updateSceneChanges_NodeTransforms(cmd, scene, df);
    markGpuTransformStaleAfterCpuSync(*scene);

    uint32_t synced = updateSceneChanges_SyncGpuBuffers(cmd, scene);
    stagingFlushed  = (synced != nvvkgltf::SceneVk::eSyncNone);

    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    updateSceneChanges_TlasUpdate(cmd, scene);
  }

  updateSceneChanges_RasterizerInvalidate(renderNodeOrNodeDirty);
  updateSceneChanges_TangentUpload(cmd, scene, changed);
  updateSceneChanges_Finalize(cmd, changed, stagingFlushed, scene);

  return changed;
}
