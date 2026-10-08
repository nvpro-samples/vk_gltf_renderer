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

// Core resource management header for the Vulkan glTF renderer.
//
// Defines the main resource structures and settings for a Vulkan-based 3D renderer
// that supports both path tracing and rasterization. Manages Vulkan resources, glTF scene data,
// environment maps, and rendering settings.

#pragma once
#include <algorithm>
#include <bitset>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/ext/scalar_constants.hpp>

#include "shaders/shaderio.h"  // Shared between host and device

#include <nvshaders_host/hdr_env_dome.hpp>
#include <nvshaders_host/tonemapper.hpp>
#include <nvslang/slang.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/hdr_ibl.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/frame_uploader.hpp>
#include "gltf_scene.hpp"
#include "gltf_scene_gpu.hpp"
#include "ui_animation.hpp"
#include "ui_interactivity.hpp"
#include "gltf_scene_transform_vk.hpp"
#include "gpu_memory_tracker.hpp"
#include "scene_feature_detection.hpp"
#include <nvapp/application.hpp>
#include <nvapp/imgui_texture.hpp>


enum class RenderingMode
{
  ePathtracer,
  eRasterizer
};

// Which special view the user wants in the viewport. The DLSS guide-buffer entries that used
// to live here are gone -- DLSS guide selection is now encapsulated inside the Dlss class
// (see Dlss::activeGuideImage() in dlss.hpp). What remains are only the two app-level overrides
// of the rendered output: the regular renderer output, or the OptiX denoised image.
enum class DisplayBuffer
{
  eRendered,           // Final rendered image (default)
  eOptixDenoised,      // OptiX Denoised output (handled via OptiXDenoiser::getDescriptorImageInfo)
  eAgenticBeautified,  // External image-to-image result from the agentic bridge
};

enum DirtyFlags
{
  eDirtyTangents,  // When tangents need to be pushed to GPU

  eNumDirtyFlags  // Keep last - Number of dirty flags
};

// Questions about an environment type, asked in several places and answered once. Free functions
// rather than members: they are properties of the enum, and both the renderer and the Environment
// panel need them.

// True when `env` has an analytic background field the renderer can evaluate per ray -- the
// sun-bearing counterpart of the baked lighting image. Mirrors hasAuthoredSkyBackground() in
// shaders/sky_background.h.slang, which is the shader-side authority.
inline bool hasAuthoredSkyBackground(shaderio::EnvSystem env)
{
  return env == shaderio::EnvSystem::ePlain || env == shaderio::EnvSystem::eGradient;
}

// True when `env` is an analytic sky EnvBaker produces a lat-long image for, rather than one that
// arrives from a file.
inline bool isBakedEnvironment(shaderio::EnvSystem env)
{
  return env == shaderio::EnvSystem::ePlain || env == shaderio::EnvSystem::eGradient || env == shaderio::EnvSystem::eSky;
}

// The SkyType the bake dispatches for an environment, or -1 when `env` is not a sky the bake
// produces at all (HDR, or none).
//
// Two of them because the questions differ: authoredSkyType is "which authored OMI type is this",
// which the per-ray background pass asks; bakeSkyType additionally maps the physical sky onto its
// Bruneton evaluator, which only the bake needs to know.
inline int authoredSkyType(shaderio::EnvSystem env)
{
  switch(env)
  {
    case shaderio::EnvSystem::ePlain:
      return int(shaderio::SkyType::ePlain);
    case shaderio::EnvSystem::eGradient:
      return int(shaderio::SkyType::eGradient);
    default:
      return -1;  // Not an authored sky: the HDR / physical / none background paths apply.
  }
}

inline int bakeSkyType(shaderio::EnvSystem env)
{
  if(env == shaderio::EnvSystem::eSky)
    return int(shaderio::SkyType::ePhysicalBruneton);
  return authoredSkyType(env);
}

// True when `env` is a sky that has a sun at all -- the two analytic types that draw one. Plain has
// none, and a panorama's sun is already in its image.
inline bool skyHasSun(shaderio::EnvSystem env)
{
  return env == shaderio::EnvSystem::eSky || env == shaderio::EnvSystem::eGradient;
}

// True when turning `env` (Settings::envRotation) changes anything: a panorama, and the two skies
// with a sun. A plain sky looks the same from every direction, and None has nothing to turn.
inline bool environmentTurns(shaderio::EnvSystem env)
{
  return env == shaderio::EnvSystem::eHdr || skyHasSun(env);
}

struct Settings
{
  RenderingMode           renderSystem           = RenderingMode::ePathtracer;          // Renderer to use
  shaderio::Visualization visualization          = shaderio::Visualization::eRendered;  // Visualization mode
  bool                    wireframe              = false;                      // Wireframe overlay on rendered meshes
  shaderio::EnvSystem     envSystem              = shaderio::EnvSystem::eSky;  // Environment system: Sky or HDR
  bool                    showAxis               = true;                       // Show the axis (bottom left)
  bool                    showGrid               = false;                      // Show infinite grid
  bool                    showGizmo              = false;                      // Show transform gizmo on selected node
  bool                    snapEnabled            = false;  // Snap gizmo transforms to grid increments
  float                   gridUnit               = 1.0f;   // Grid base unit (world units)
  float                   snapRotation           = 45.0f;  // Rotation snap increment (degrees)
  float                   snapScale              = 0.1f;   // Scale snap increment
  bool                    showMemStats           = false;  // Show memory statistics window
  bool                    showCameraWindow       = true;   // Show Camera window
  bool                    showSettingsWindow     = true;   // Show Settings window
  bool                    showEnvironmentWindow  = true;   // Show Environment window
  bool                    showTonemapperWindow   = true;   // Show Tonemapper window
  bool                    showStatisticsWindow   = false;  // Show Statistics window
  bool                    showSceneBrowserWindow = true;   // Show Scene Browser window
  bool                    showInspectorWindow    = true;   // Show Inspector window
  bool  showInteractivityWindow = false;  // Show KHR_interactivity Graphs window (opt-in, unlike the above)
  bool  showAgenticWindow       = false;  // Show Agentic bridge window
  bool  showGridSettingsWindow  = false;  // Show Grid & Snap settings window
  float hdrEnvIntensity         = 1.0f;   // Intensity of the environment (HDR)
  // The environment's orientation: a unit quaternion taking its own frame to world, stored x, y, z, w
  // -- OMI_environment_sky's `rotation`, verbatim, and the only orientation any environment has. It
  // turns the HDR image, every analytic sky, and the compass Time of Day places the sun with.
  // Stored as given (the command line may not normalise it); read through envRotationQuat().
  glm::vec4 envRotation          = {0.0f, 0.0f, 0.0f, 1.0f};
  float     hdrBlur              = 0.0f;                      // Blur of the environment (HDR)
  glm::vec3 silhouetteColor      = {0.933f, 0.580f, 0.180f};  // Color of the silhouette
  bool      useSolidBackground   = false;                     // Use solid background color
  glm::vec3 solidBackgroundColor = {0.0f, 0.0f, 0.0f};        // Solid background color
  int       maxFrames            = {500};                     // Maximum number of frames to render
  // Sun position as the UI and the command line express it. Resources::sunDirection stays the
  // single source of truth; these are kept in sync with it both ways (see syncSunAngles).
  float sunAzimuth   = 90.0f;  // degrees
  float sunElevation = 45.0f;  // degrees

  // Time of Day: where and when, from which the two angles above are computed (src/sun_position.*).
  //
  // Convenience state, and ini-only on purpose. OMI_environment_sky has no field for a place or a
  // date -- it describes a medium, not a moment -- so writing them would fork the schema for
  // something the sun's direction already captures exactly. What travels with the scene is the
  // marked sun light's rotation; these reconstruct the widget on the same machine.
  //
  // todDate and todUtcOffset are seeded from the system clock at start-up, so a fresh install
  // opens on today rather than on an arbitrary epoch.
  float       todHour      = 12.0f;   // local clock hours past midnight, at todUtcOffset
  float       todLatitude  = 47.37f;  // Zurich
  float       todLongitude = 8.54f;
  float       todUtcOffset = 1.0f;          // hours east of UTC, as a clock offset (not a time zone)
  std::string todDate      = "2026-01-01";  // yyyy-mm-dd
  // Write-only: picking a city overwrites the three above and nothing reads it back, which is why
  // it does not persist -- restoring it after them would undo whatever was edited since. The panel
  // derives the name it shows from the coordinates instead (sun_position::cityAt).
  std::string todCity;

  // Physical sky (Bruneton). Kept here, in the same place as every other user-settable value,
  // rather than inside SkyBruneton -- that class owns GPU resources, not user intent, and
  // SettingsRegistry needs a stable address.
  //
  // Altitude is in metres because that is what a person types; SkyBruneton works in kilometres.
  float atmoSunAngularRadius = 0.004675f;  // radians; Earth's sun is 0.00935/2
  float atmoObserverAltitude = 300.0f;     // metres above the planet surface
  // Aerial perspective: how much air the scene contains between the eye and a surface. A distance
  // multiplier on glTF's metre, so 1.0 means "this scene is in metres, as the format says" and 2.0
  // means "treat it as twice the size"; 0 switches the term off. A viewer setting rather than scene
  // data -- it describes how big the model is meant to be, which no extension has a field for.
  float     atmoAerialPerspectiveScale = 1.0f;
  glm::vec3 atmoGroundAlbedo           = {0.1f, 0.1f, 0.1f};  // linear; what the ground reflects

  // The rest of the atmosphere: what the planet is made of, and how big it is.
  //
  // These describe a *world*, and all of them are scene data: they arrive from the glTF and go
  // back out to it -- the first six through OMI_environment_sky's `physical` type, the rest
  // through the NV_environment_sky_atmosphere block beside it.
  //
  // Persisted all the same, because the scene cannot lose: a load overwrites them outright, and it
  // happens after the .ini is restored. So persistence only decides what you get when *no* scene
  // carries an atmosphere, and there the answer should be the one you last chose.
  //
  // Units are km and km^-1 throughout, matching SkyBruneton. OMI states m^-1; that conversion
  // happens at the file boundary, not here.
  glm::vec3 atmoRayleighScattering = {0.00580234f, 0.0135578f, 0.0331f};
  // Haze. Upstream's Bruneton demo uses 0.003996, which is an aerosol optical depth of about 0.005 --
  // cleaner than the cleanest place on Earth, and it shows: it left the sun outrunning the sky by
  // 2-3x at every elevation. Measured direct-to-diffuse illuminance on a horizontal surface against
  // a real clear day:
  //
  //     elevation      60     45     30     15
  //     upstream     13.4   11.1    8.0    4.0
  //     here          6.0    4.9    3.5    1.7
  //     measured    ~6-7     ~5     ~3    ~1.3
  //
  // 0.06 /km is an optical depth of ~0.08, which is a clear continental day rather than a mountain
  // observatory. It barely changes how much light reaches the scene (total illuminance moves under
  // 1%); what it changes is the *balance*, which is what was wrong.
  glm::vec3 atmoMieScattering       = {0.06f, 0.06f, 0.06f};
  float     atmoMieAnisotropy       = 0.8f;                         // Cornette-Shanks g; forward-scattering at g > 0
  glm::vec3 atmoSolarIrradiance     = {1.474f, 1.8504f, 1.91198f};  // W/m^2 at the top of the atmosphere
  float     atmoRayleighScaleHeight = 8.0f;                         // km; the air thins by 1/e over this height
  float     atmoMieScaleHeight      = 1.2f;                         // km; aerosols hug the ground far more closely
  // Single-scattering albedo of the aerosol: what fraction of what it removes it scatters rather
  // than absorbs. Extinction is derived from it, because that is the ratio people reason about --
  // "how sooty is the haze" -- and because OMI carries no absorption field at all.
  float     atmoMieAlbedo       = 0.9f;
  glm::vec3 atmoOzoneExtinction = {0.000649717f, 0.0018809f, 8.50167e-05f};  // 1/km
  float     atmoOzoneCenter     = 25.0f;                                     // km; peak of the ozone tent
  float     atmoOzoneWidth      = 30.0f;                                     // km; full width of the tent, zero to zero
  float     atmoBottomRadius    = 6360.0f;                                   // km; planet radius
  float     atmoThickness       = 60.0f;                                     // km; atmosphere depth above the surface
  // Applying a preset overwrites every field above. It is an action rather than a mode: nothing
  // reads it back, and the UI works out which preset (if any) the current values match.
  int atmoPreset = 0;  // AtmospherePresetIndex

  // Authored sky parameters (glTF OMI_environment_sky). Defaults match the extension's own. When a
  // loaded scene carries the extension it overwrites these; they are the fallback for every scene
  // that does not, and what the user edits in the Environment panel.
  glm::vec3 plainColor           = {0.5f, 0.5f, 0.5f};
  glm::vec3 gradientBottomColor  = {0.2f, 0.169f, 0.133f};
  glm::vec3 gradientHorizonColor = {0.646f, 0.656f, 0.67f};
  glm::vec3 gradientTopColor     = {0.385f, 0.454f, 0.55f};
  glm::vec3 gradientSunColor     = {1.0f, 1.0f, 1.0f};
  float     gradientBottomCurve  = 0.02f;  // Horizon -> bottom falloff
  float     gradientTopCurve     = 0.15f;  // Horizon -> top falloff
  float     gradientSunAngleMax  = 1.74f;  // Angular extent of the sun glow (radians)
  float     gradientSunCurve     = 0.05f;  // Disk -> sky falloff across the glow
  // Write OMI_environment_sky when saving the scene. Off keeps the viewer non-authoring.
  bool      envSaveToGltf           = false;
  bool      useInfinitePlane        = false;                     // Use infinite plane
  bool      isShadowCatcher         = true;                      // Infinite place only catch shadow
  float     infinitePlaneDistance   = 0;                         // Distance/height of the infinite plane
  glm::vec3 infinitePlaneBaseColor  = glm::vec3(0.5, 0.5, 0.5);  // Default gray color
  float     infinitePlaneMetallic   = 0.0;                       // Default non-metallic
  float     infinitePlaneRoughness  = 0.5;                       // Default medium roughness
  float     shadowCatcherDarkness   = 0.0f;                      // Non-physical shadow darkening
  bool      dlssRrHardwareAvailable = false;  // DLSS Ray Reconstruction hardware/extensions available (set at startup)
  bool dlssSrHardwareAvailable = false;  // DLSS Super Resolution / DLAA hardware/extensions available (set at startup)
  bool serHardwareAvailable = false;  // VK_EXT_ray_tracing_invocation_reorder enabled on the device (set at startup)
  bool opacityMicromapSupported = false;  // VK_KHR_opacity_micromap available (set at startup); gates EXT_mesh_opacity_micromap
  bool opacityMicromapMissing = false;  // --useOpacityMicromap is on but the device lacks VK_KHR_opacity_micromap
  DisplayBuffer displayBuffer = DisplayBuffer::eRendered;  // Which buffer to display in viewport

  // Enable scene-based shader optimization. When true, only features used by the scene are
  // enabled, reducing shader size and register usage at the cost of a one-time recompile per scene change.
  bool optimalShader = false;

#ifndef NDEBUG
  bool showGridStyleWindow  = false;  // Show Grid Style debug window
  bool showGizmoStyleWindow = false;  // Show Gizmo Style debug window
#endif

  // `envRotation` as a quaternion, normalised. The stored value may not be unit -- the command line
  // and MCP write it verbatim -- and a zero-length one is no rotation at all, so it reads as identity.
  [[nodiscard]] glm::quat envRotationQuat() const
  {
    const glm::quat q(envRotation.w, envRotation.x, envRotation.y, envRotation.z);
    const float     len = glm::length(q);
    return (len > 1e-6f) ? q / len : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  }
};


struct Resources
{
  enum ImageType
  {
    eImgTonemapped,
    eImgRendered,
    eImgSelection,
    eImgCount,  // Sentinel: number of frame-target color attachments. KEEP LAST.
  };

  nvapp::Application* app{nullptr};

  VkInstance              instance{};
  nvvk::ResourceAllocator allocator{};  // Vulkan Memory Allocator
  nvvk::FrameUploader     staging;      // Per-frame + load-path staging copies (records into caller cmd)

  nvvk::SamplerPool      samplerPool{};    // Texture Sampler Pool
  VkCommandPool          commandPool{};    // Command pool for secondary command buffer
  nvslang::SlangCompiler slangCompiler{};  // Slang compiler

  std::unique_ptr<nvvkgltf::Scene> scene;
  nvvkgltf::SceneVk                sceneVk;
  nvvkgltf::SceneRtx               sceneRtx;
  nvvkgltf::AnimationVk            animationVk;
  nvvkgltf::TransformComputeVk     transformCompute{};
  nvvkgltf::SceneGpu sceneGpu{sceneVk, animationVk, sceneRtx, transformCompute, staging};  // Must be declared after its reference dependencies

  nvvkgltf::Scene*       getScene() { return scene.get(); }
  const nvvkgltf::Scene* getScene() const { return scene.get(); }

  // Resources
  nvvk::HdrIbl hdrIbl;  // HDR environment map

  // The physical sky's scattering LUTs, as the shading passes see them.
  //
  // Both renderers build their pipeline layouts from Resources and neither should have to know
  // which class owns the tables, so SkyBruneton publishes its runtime set here once at start-up --
  // the same shape `hdrIbl` above already has, minus the ownership. Valid for the whole session:
  // the images are allocated at init and only their *contents* change when the atmosphere does.
  //
  // Read by aerial perspective (shaders/aerial_perspective.h.slang). The bake reaches the same set
  // directly, because EnvBaker is handed it at init.
  VkDescriptorSetLayout skyLutDescriptorSetLayout{};
  VkDescriptorSet       skyLutDescriptorSet{};
  nvshaders::HdrEnvDome hdrDome;
  // Main frame target (tonemapped + rendered + selection + depth). Accessor cheat sheet:
  //   raster attachment  -> getColorAttachmentView() / getDepthImageView()
  //   compute write      -> getColorStorageImageInfo()
  //   sampled read       -> getColorSampleDescriptorImageInfo(..., linearSampler)
  //   ImGui              -> getUiImageView() + tonemappedUi
  nvvk::RenderTarget gBuffers;
  nvapp::ImTexture   tonemappedUi{};   // Viewport display (eImgTonemapped only)
  VkSampler          linearSampler{};  // Linear sampler (visual helpers, etc.)
  nvvk::Buffer       bFrameInfo;       // Scene/Frame information
  // The sky's sun. Every sky type with a sun reads it, and the light the renderer supplies for
  // that sun aims along it. Driven by sunAzimuth/sunElevation -- themselves driven by the Time of
  // Day widget -- unless the scene marks one of its directional lights as the sun, in which case
  // that light owns the direction and this follows it (see GltfRenderer::syncSunFromMarkedLight).
  // It used to live inside the MDL sky's parameter block, which is why it survived that sky's
  // deletion while nothing else in it did.
  //
  // `sunYIsUp` is not a constant: it tracks the camera's up axis, because a Z-up scene has to
  // interpret the same azimuth/elevation pair differently.
  glm::vec3                                   sunDirection{-1.23413404e-08F, 0.707106829F, 0.707106709F};
  bool                                        sunYIsUp{true};
  nvshaders::Tonemapper                       tonemapper{};                       // Tonemapper
  shaderio::TonemapperData                    tonemapperData{.autoExposure = 1};  // Tonemapper data
  std::shared_ptr<nvutils::CameraManipulator> cameraManip;         // Camera manipulator (owned by GltfRenderer)
  std::filesystem::path                       headlessOutputPath;  // --output: override for headless image save path

  // Pipeline
  std::array<nvvk::DescriptorBindings, 2> descriptorBinding{};    // Descriptor bindings: 0: textures, 1: tlas
  std::array<VkDescriptorSetLayout, 2>    descriptorSetLayout{};  // Descriptor set layout
  VkDescriptorSet                         descriptorSet{};        // Descriptor set for the textures
  VkDescriptorPool                        descriptorPool{};

  // Animation playback state
  AnimationControl animationControl{};

  // KHR_interactivity Graphs panel playback state
  InteractivityControl interactivityControl{};

  int frameCount{0};

  // Monotonic count of completed onRender() calls since the app started - unlike frameCount above
  // (which resets to -1 on every dirty-flag/camera-move reset, by design, to avoid ghosting), this
  // never resets. A KHR_interactivity graph that writes a pointer every tick (e.g. a permanent idle
  // animation via event/onTick) resets frameCount every single frame, so it can never reach >= 1 -
  // callers that just need "has at least one frame actually rendered" (e.g. a UI scenario script's
  // ready gate) should use this instead.
  uint64_t renderPassCount{0};

  // #DLSS: True if any node transforms changed this frame (for DLSS per-instance motion).
  bool dlssInstanceMotionActive{false};

  // Selection: set of render node indices (TLAS order). One primitive = set of size 1; node + branch = many.
  std::unordered_set<int> selectedRenderNodes;

  // Selection bitmask for silhouette: one bit per render node (GPU buffer + CPU mirror)
  std::vector<uint32_t> selectionBitMask;
  nvvk::Buffer          bSelectionBitMask;
  bool                  selectionDirty = true;

  // Build CPU-side selection bitmask from selectedRenderNodes. Cleared and refilled each call.
  void updateSelectionBitMask(int numRenderNodes)
  {
    const size_t numWords = numRenderNodes > 0 ? (static_cast<size_t>(numRenderNodes) + 31u) / 32u : 1u;
    selectionBitMask.resize(numWords, 0u);
    std::fill(selectionBitMask.begin(), selectionBitMask.end(), 0u);
    for(int rnIdx : selectedRenderNodes)
    {
      if(rnIdx >= 0 && static_cast<size_t>(rnIdx) < numRenderNodes)
      {
        const size_t word = static_cast<size_t>(rnIdx) / 32u;
        const size_t bit  = static_cast<size_t>(rnIdx) % 32u;
        selectionBitMask[word] |= (1u << bit);
      }
    }
  }

  nvvkgltf::GpuMemoryTracker appMemoryTracker;  // Application-level GPU memory tracking (frame targets, denoisers, etc.)

  Settings settings;

  std::bitset<32> dirtyFlags;

  // Scene feature set (KHR_materials_* and denoiser guide buffer);
  // updated via recomputeSceneFeatures() on scene/material/denoiser changes.
  nvvkgltf::SceneFeatureSet currentFeatureSet{};

  // Update currentFeatureSet from the scene. Returns true if changed.
  bool recomputeSceneFeatures(bool dlssGuideActive)
  {
    nvvkgltf::SceneFeatureSet newSet{};
    if(scene)
      newSet = nvvkgltf::detectSceneFeatures(scene->getModel());
    newSet.set(nvvkgltf::SceneFeatureSet::eDlssGuide, dlssGuideActive);
    const bool changed = (newSet != currentFeatureSet);
    currentFeatureSet  = newSet;
    return changed;
  }
};
