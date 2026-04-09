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

#include <cmath>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vulkan/vulkan_core.h>
#include <webp/decode.h>

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

//
#include <nvaftermath/aftermath.hpp>
#include <nvutils/profiler.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/mipmaps.hpp>
#include "gltf_camera_utils.hpp"
#include "gltf_scene_editor.hpp"
#include "gltf_scene_animation.hpp"
#include "gltf_scene_transform_vk.hpp"

#include <glm/gtc/quaternion.hpp>

#include "renderer.hpp"
#include "scene_descriptor.hpp"
#include "tinygltf_utils.hpp"
#include "utils.hpp"
#include "tinyobjloader/tiny_obj_loader.h"
#include "tinygltf_converter.hpp"

extern nvutils::ProfilerManager g_profilerManager;  // #PROFILER

namespace {
// Background clear color used when no scene is loaded or to show DLSS render resolution borders
constexpr VkClearColorValue kBackgroundClearColor = {{0.17f, 0.21f, 0.25f, 1.f}};

// WebP callback for glTF image loading. Decodes an image into a SceneImage
// object, returning `true` on success.
bool webPLoadCallback(nvvkgltf::SceneVk::SceneImage& image, const void* data, size_t byteLength)
{
  const uint8_t* dataU8 = reinterpret_cast<const uint8_t*>(data);

  int width = 0, height = 0;
  if(!WebPGetInfo(dataU8, byteLength, &width, &height) || width <= 0 || height <= 0 || width > INT_MAX / 4)
  {
    return false;
  }

  std::vector<char> decompressed(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
  if(!WebPDecodeRGBAInto(dataU8, byteLength,                                                    //
                         reinterpret_cast<uint8_t*>(decompressed.data()), decompressed.size(),  //
                         width * 4))
  {
    LOGW("Failed to decode WebP image '%s'.\n", image.imgName.c_str());
    return false;
  }

  image.format  = VK_FORMAT_R8G8B8A8_UNORM;
  image.size    = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  image.mipData = {std::move(decompressed)};
  return true;
}
}  // namespace

// The constructor registers the parameters that can be set from the command line
GltfRenderer::GltfRenderer(nvutils::ParameterRegistry* paramReg)
{
  // All parameters that can be set from the command line
  paramReg->add({"envSystem", "Environment: [Sky:0, HDR:1]"}, (int*)&m_resources.settings.envSystem);
  paramReg->add({"renderSystem", "Renderer [Path tracer:0, Rasterizer:1]"}, (int*)&m_resources.settings.renderSystem);
  paramReg->add({"showAxis", "Show Axis"}, &m_resources.settings.showAxis);
  paramReg->add({"showMemStats", "Show Memory Statistics"}, &m_resources.settings.showMemStats);
  paramReg->add({"hdrEnvIntensity", "HDR Environment Intensity"}, &m_resources.settings.hdrEnvIntensity);
  paramReg->add({"hdrEnvRotation", "HDR Environment Rotation"}, &m_resources.settings.hdrEnvRotation);
  paramReg->add({"hdrBlur", "HDR Environment Blur"}, &m_resources.settings.hdrBlur);
  paramReg->addVector({"silhouetteColor", "Color of the silhouette"}, &m_resources.settings.silhouetteColor);
  paramReg->add({"debugMethod", "Debug Method"}, (int*)&m_resources.settings.debugMethod);
  paramReg->add({"useSolidBackground", "Use solid color background"}, &m_resources.settings.useSolidBackground, true);
  paramReg->addVector({"solidBackgroundColor", "Solid Background Color"}, &m_resources.settings.solidBackgroundColor);
  paramReg->add({"maxFrames", "Maximum number of iterations"}, &m_resources.settings.maxFrames);

  paramReg->add({"tmMethod", "Tonemapper method: [Filmic:0, Uncharted:1, Clip:2, ACES:3, AgX:4, KhronosPBR:5]"},
                &m_resources.tonemapperData.method);
  paramReg->add({"tmExposure", "Tonemapper exposure"}, &m_resources.tonemapperData.exposure);
  paramReg->add({"tmGamma", "Tonemapper brightness"}, &m_resources.tonemapperData.brightness);
  paramReg->add({"tmContrast", "Tonemapper contrast"}, &m_resources.tonemapperData.contrast);
  paramReg->add({"tmSaturation", "Tonemapper saturation"}, &m_resources.tonemapperData.saturation);
  paramReg->add({"tmWhitePoint", "Tonemapper vignette"}, &m_resources.tonemapperData.vignette);

  // Register PathTracer-specific command line parameters
  m_pathTracer.registerParameters(paramReg);
  m_rasterizer.registerParameters(paramReg);

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

  // ===== Settings Handler (ImGui persistant) =====
  if(!m_app->isHeadless())
  {
    // Read/store the information in the settings file, only if not headless
    m_settingsHandler.setHandlerName("GltfRenderer");
    m_settingsHandler.setSetting("maxFrames", &m_resources.settings.maxFrames);
    m_settingsHandler.setSetting("showAxis", &m_resources.settings.showAxis);
    m_settingsHandler.setSetting("showGrid", &m_resources.settings.showGrid);
    m_settingsHandler.setSetting("showGizmo", &m_resources.settings.showGizmo);
    m_settingsHandler.setSetting("snapEnabled", &m_resources.settings.snapEnabled);
    m_settingsHandler.setSetting("gridUnit", &m_resources.settings.gridUnit);
    m_settingsHandler.setSetting("snapRotation", &m_resources.settings.snapRotation);
    m_settingsHandler.setSetting("snapScale", &m_resources.settings.snapScale);
    m_settingsHandler.setSetting("showGridSettingsWindow", &m_resources.settings.showGridSettingsWindow);
    m_settingsHandler.setSetting("showMemStats", &m_resources.settings.showMemStats);
    m_settingsHandler.setSetting("showCameraWindow", &m_resources.settings.showCameraWindow);
    m_settingsHandler.setSetting("showSettingsWindow", &m_resources.settings.showSettingsWindow);
    m_settingsHandler.setSetting("showEnvironmentWindow", &m_resources.settings.showEnvironmentWindow);
    m_settingsHandler.setSetting("showTonemapperWindow", &m_resources.settings.showTonemapperWindow);
    m_settingsHandler.setSetting("showStatisticsWindow", &m_resources.settings.showStatisticsWindow);
    m_settingsHandler.setSetting("showSceneBrowserWindow", &m_resources.settings.showSceneBrowserWindow);
    m_settingsHandler.setSetting("showInspectorWindow", &m_resources.settings.showInspectorWindow);
    m_settingsHandler.setSetting("envSystem", (int*)&m_resources.settings.envSystem);
    m_settingsHandler.setSetting("renderSystem", (int*)&m_resources.settings.renderSystem);
    m_settingsHandler.setSetting("useSolidBackground", &m_resources.settings.useSolidBackground);
    m_settingsHandler.setSetting("solidBackgroundColor", &m_resources.settings.solidBackgroundColor);
    m_pathTracer.setSettingsHandler(&m_settingsHandler);
    m_rasterizer.setSettingsHandler(&m_settingsHandler);
    m_settingsHandler.addImGuiHandler();
  }

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
  // m_resources.allocator.setLeakID(32914);


  m_transientCmdPool = nvvk::createTransientCommandPool(m_device, app->getQueue(0).familyIndex);
  NVVK_DBG_NAME(m_transientCmdPool);

  m_loadPipeline.init(m_device, app->getQueue(0).queue, m_transientCmdPool);

  // Staging buffer uploader
  m_resources.staging.init(&m_resources.allocator, true);

  m_resources.commandPool = app->getCommandPool();


  // ===== Texture & Image Resources =====
  m_resources.samplerPool.init(m_device);
  VkSampler linearSampler{};
  NVVK_CHECK(m_resources.samplerPool.acquireSampler(linearSampler));
  NVVK_DBG_NAME(linearSampler);

  // IBL environment map
  m_resources.hdrIbl.init(&m_resources.allocator, &m_resources.samplerPool);
  m_resources.hdrDome.init(&m_resources.allocator, &m_resources.samplerPool, m_app->getQueue(0));

  // Application-level memory tracker (GBuffers, DLSS, OptiX images)
  m_resources.appMemoryTracker.init(&m_resources.allocator);

  // G-Buffer
  m_resources.gBuffers.init({.allocator = &m_resources.allocator,
                             .colorFormats =
                                 {
                                     VK_FORMAT_R8G8B8A8_UNORM,       // Tonemapped (eImgTonemapped)
                                     VK_FORMAT_R32G32B32A32_SFLOAT,  // Rendered image (eImgRendered)
                                     VK_FORMAT_R32_SFLOAT,  // ObjectID for selection/silhouette (eImgSelection), .r = render node ID+1
                                 },
                             .depthFormat    = nvvk::findDepthFormat(app->getPhysicalDevice()),
                             .imageSampler   = linearSampler,
                             .descriptorPool = m_app->getTextureDescriptorPool()});
  {
    VkCommandBuffer cmd{};
    nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
    m_resources.gBuffers.update(cmd, {100, 100});
    nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_app->getQueue(0).queue);
    m_resources.appMemoryTracker.track("GBuffers", m_resources.gBuffers, 3);
  }

  // ===== Rendering Utilities =====

  // Ray picker
  m_rayPicker.init(&m_resources.allocator);

  // Tonemapper
  m_resources.tonemapper.init(&m_resources.allocator, tonemapper_slang);

  // Silhouette renderer
  m_silhouette.init(m_resources);

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

    // Enable specific capabilities for better performance and features
    m_resources.slangCompiler.addCapability("spvShaderInvocationReorderNV");  // Enable the shader invocation reorder capability for better performance on NVIDIA hardware
    m_resources.slangCompiler.addCapability("spvInt64Atomics");            // # 64-bit atomic operations
    m_resources.slangCompiler.addCapability("spvShaderClockKHR");          // # Shader clock for profiling
    m_resources.slangCompiler.addCapability("spvRayTracingMotionBlurNV");  // # Motion blur for ray tracing
    m_resources.slangCompiler.addCapability("spvRayQueryKHR");             // # Ray query operations
    m_resources.slangCompiler.addCapability("spvGroupNonUniformBallot");  // # Ballot operations for subgroup functionality
    m_resources.slangCompiler.addCapability("spvGroupNonUniformArithmetic");  // # Arithmetic operations across subgroups

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
  m_rasterizer.onAttach(m_resources, &m_profilerGpuTimer);

  m_pathTracer.createPipeline(m_resources);
  m_rasterizer.createPipeline(m_resources);

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
}

//--------------------------------------------------------------------------------------------------
// Detach the renderers and destroy the resources
void GltfRenderer::onDetach()
{
  // SYNC NOTE: Full device wait during shutdown is the standard Vulkan teardown pattern.
  vkDeviceWaitIdle(m_device);
  m_visualHelpers.deinit();
  m_pathTracer.onDetach(m_resources);
  m_rasterizer.onDetach(m_resources);
  destroyResources();
}

//--------------------------------------------------------------------------------------------------
// Resize the G-Buffer and the renderers
void GltfRenderer::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  m_resources.appMemoryTracker.untrack("GBuffers", m_resources.gBuffers, 3);
  m_resources.gBuffers.update(cmd, size);
  m_resources.appMemoryTracker.track("GBuffers", m_resources.gBuffers, 3);
  m_pathTracer.onResize(cmd, size, m_resources);
  m_rasterizer.onResize(cmd, size, m_resources);
  m_resources.hdrDome.setOutImage(m_resources.gBuffers.getDescriptorImageInfo(Resources::eImgRendered));

  // Resize visual helpers (depth buffer + scene depth descriptor set)
  VkSampler sampler = m_resources.gBuffers.getDescriptorImageInfo(Resources::eImgTonemapped).sampler;
  m_visualHelpers.onResize(cmd, size, m_resources.gBuffers.getDepthImage(), m_resources.gBuffers.getDepthImageView(), sampler);

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
void GltfRenderer::onUIRender()
{
  renderUI();
}


//--------------------------------------------------------------------------------------------------
// Render the scene
void GltfRenderer::onRender(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  m_profilerTimeline->frameAdvance();
  // Don't do anything if the busy window is open
  if(m_busy.isBusy())
  {
    return;
  }

  // Consume the done signal from the busy state, this will remove the Progress Bar from the UI
  if(m_busy.isDone())
  {
    m_busy.consumeDone();
  }

  // Loading pipeline: submit queued work, poll completion, run callbacks
  if(m_app->isHeadless())
    m_loadPipeline.drain();
  else if(m_loadPipeline.poll())
    return;  // Still loading -- give control back to the UI

  // Begin the frame for the staging uploader, using the semaphore from the current frame to clear and synchronize
  m_resources.staging.beginFrame(m_app->getFrameSignalSemaphore());

  // Empty scene, clear the G-Buffer
  if(!m_resources.getScene() || !m_resources.getScene()->valid())
  {
    clearGbuffer(cmd);
    return;
  }

  // Start the profiler section for the GPU timer
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

  // Check for changes
  bool changed{false};
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
    shaderio::SceneFrameInfo finfo{
        .viewMatrix     = m_cameraManip->getViewMatrix(),
        .projInv        = glm::inverse(m_cameraManip->getPerspectiveMatrix()),
        .viewInv        = glm::inverse(m_cameraManip->getViewMatrix()),
        .viewProjMatrix = m_cameraManip->getPerspectiveMatrix() * m_cameraManip->getViewMatrix(),
        .prevMVP        = m_prevMVP,
        .flags = ((m_cameraManip->getProjectionType() == nvutils::CameraManipulator::Orthographic) ? shaderio::eSceneIsOrthographic : 0)
                 | (m_resources.settings.useSolidBackground ? shaderio::eSceneUseSolidBackground : 0)
                 | ((m_resources.settings.envSystem == shaderio::EnvSystem::eHdr) ? shaderio::eSceneUseHdrEnvironment : 0)
                 | (m_resources.settings.useInfinitePlane ? shaderio::eSceneUseInfinitePlane : 0)
                 | ((m_resources.settings.useInfinitePlane && m_resources.settings.isShadowCatcher) ? shaderio::eSceneInfinitePlaneShadowCatcher :
                                                                                                      0),
        .envRotation               = m_resources.settings.hdrEnvRotation,
        .envBlur                   = m_resources.settings.hdrBlur,
        .envIntensity              = m_resources.settings.hdrEnvIntensity,
        .backgroundColor           = m_resources.settings.solidBackgroundColor,
        .debugMethod               = m_resources.settings.debugMethod,
        .infinitePlaneDistance     = m_resources.settings.infinitePlaneDistance,
        .infinitePlaneBaseColor    = m_resources.settings.infinitePlaneBaseColor,
        .infinitePlaneMetallic     = m_resources.settings.infinitePlaneMetallic,
        .infinitePlaneRoughness    = m_resources.settings.infinitePlaneRoughness,
        .shadowCatcherDarkenAmount = 1.0f - exp2f(-std::max(m_resources.settings.shadowCatcherDarkness, 0.0f)),
    };
    // Update the camera information
    m_prevMVP = finfo.viewProjMatrix;

    vkCmdUpdateBuffer(cmd, m_resources.bFrameInfo.buffer, 0, sizeof(shaderio::SceneFrameInfo), &finfo);
    // Update the sky
    m_resources.skyParams.yIsUp = m_cameraManip->getUp().y > 0.5f;
    vkCmdUpdateBuffer(cmd, m_resources.bSkyParams.buffer, 0, sizeof(shaderio::SkyPhysicalParameters), &m_resources.skyParams);
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
  tonemap(cmd);
  silhouette(cmd);
  renderVisualHelpers(cmd);
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
  m_app->saveImageToFile(m_resources.gBuffers.getColorImage(Resources::eImgTonemapped), m_resources.gBuffers.getSize(),
                         nvutils::getExecutablePath().replace_extension(".jpg").string());
}

//--------------------------------------------------------------------------------------------------
// Merge a glTF scene into the current one. Called from File menu or Shift+drag-drop.
//
void GltfRenderer::onMergeScene(const std::filesystem::path& filename)
{
  if(!m_resources.getScene() || !m_resources.getScene()->valid())
  {
    auto scn = std::make_unique<nvvkgltf::Scene>();
    scn->supportedExtensions().insert(EXT_TEXTURE_WEBP_EXTENSION_NAME);
    m_resources.scene = std::move(scn);
  }

  if(m_busy.isBusy())
    return;

  // Set busy BEFORE starting the worker thread to prevent UI access during scene modification
  m_busy.start("Merging Scene");

  std::thread([=, this]() {
    int wrapperNodeIdx = m_resources.getScene()->mergeScene(filename, static_cast<uint32_t>(m_maxTextures));
    if(wrapperNodeIdx >= 0)
    {
      m_undoStack.clear();
      rebuildVulkanSceneFull();
      resetFrame();
      m_sceneSelection.selectNode(wrapperNodeIdx);
      m_sceneBrowser.focusOnSelection();
      LOGI("Scene merged successfully: %s\n", nvutils::utf8FromPath(filename.filename()).c_str());
    }
    else
    {
      LOGE("Failed to merge scene: %s\n", nvutils::utf8FromPath(filename.filename()).c_str());
    }
    m_busy.stop();
  }).detach();
}

//--------------------------------------------------------------------------------------------------
// Load a glTF scene or an HDR file (called from both Load Scene and Load HDR Environment menu items)
// Shift+drag-drop merges the file into the current scene instead of replacing it.
//
void GltfRenderer::onFileDrop(const std::filesystem::path& filename)
{
  // SYNC NOTE: User-initiated file load/merge — wait ensures GPU is idle before scene teardown/rebuild.
  vkQueueWaitIdle(m_app->getQueue(0).queue);

  bool isDescriptor = filename.string().ends_with(".scene.json") || nvutils::extensionMatches(filename, ".glxf");
  if(isDescriptor)
  {
    if(m_busy.isBusy())
      return;

    m_loadPipeline.clear();
    cleanupScene();
    m_rasterizer.freeRecordCommandBuffer(m_resources);

    m_busy.start("Loading Descriptor");
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
#if defined(_WIN32)
    shiftHeld = (GetKeyState(VK_SHIFT) & 0x8000) != 0;
#else
    shiftHeld = ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift);
#endif

    if(shiftHeld)
    {
      onMergeScene(filename);
    }
    else
    {
      m_loadPipeline.clear();
      cleanupScene();  // Cleanup current scene
      m_rasterizer.freeRecordCommandBuffer(m_resources);

      // Set busy BEFORE starting the worker thread to prevent re-entrant drops
      m_busy.start("Loading");

      std::thread([=, this]() {
        m_lastSceneDirectory = filename.parent_path();
        createScene(filename);
        m_busy.stop();
      }).detach();
    }
  }
  else if(nvutils::extensionMatches(filename, ".hdr"))
  {
    m_lastHdrDirectory = filename.parent_path();
    createHDR(filename);
    m_resources.settings.envSystem                 = shaderio::EnvSystem::eHdr;
    m_pathTracer.m_pushConst.fireflyClampThreshold = m_resources.hdrIbl.getIntegral();
  }

  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Save the scene
bool GltfRenderer::save(const std::filesystem::path& filename)
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

    // Saving the scene
    return m_resources.getScene()->save(filename);
  }
  return false;
}

//--------------------------------------------------------------------------------------------------
// Apply the tonemapper on the rendered image
void GltfRenderer::tonemap(VkCommandBuffer cmd)
{
  NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

  // Select which buffer to tonemap based on user selection
  VkDescriptorImageInfo inputBuffer      = m_resources.gBuffers.getDescriptorImageInfo(Resources::eImgRendered);
  VkExtent2D            gbufSize         = m_resources.gBuffers.getSize();
  bool                  usingGuideBuffer = false;

  // Check if we want to display a DLSS guide buffer or OptiX denoised output (only for pathtracer)
  if(m_resources.settings.renderSystem == RenderingMode::ePathtracer && m_resources.settings.displayBuffer != DisplayBuffer::eRendered)
  {
    // Handle OptiX denoised output
#if defined(USE_OPTIX_DENOISER)
    if(m_resources.settings.displayBuffer == DisplayBuffer::eOptixDenoised)
    {
      const OptiXDenoiser* optix = m_pathTracer.getOptiXDenoiser();
      if(optix && optix->hasValidDenoisedOutput())
      {
        inputBuffer      = optix->getDescriptorImageInfo(OptiXDenoiser::eGBufferDenoised);
        usingGuideBuffer = false;  // We want to tonemap the denoised output, not the guide buffer
      }
    }
    else
#endif
    {
      // Handle DLSS guide buffers
#if defined(USE_DLSS)
      const DlssDenoiser* dlss = m_pathTracer.getDlssDenoiser();
      if(dlss && dlss->isEnabled())
      {
        shaderio::OutputImage dlssBuffer = displayBufferToOutputImage(m_resources.settings.displayBuffer);
        inputBuffer                      = dlss->getDescriptorImageInfo(dlssBuffer);
        usingGuideBuffer                 = true;
        gbufSize                         = dlss->getRenderSize();

        // Clear output image since guide buffer may be smaller than display size
        // Use distinct color to visually show the DLSS render resolution vs display resolution
        VkImageSubresourceRange range = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
        vkCmdClearColorImage(cmd, m_resources.gBuffers.getColorImage(Resources::eImgTonemapped),
                             VK_IMAGE_LAYOUT_GENERAL, &kBackgroundClearColor, 1, &range);
        // Barrier: clear must complete before tonemapper compute shader runs
        nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                               VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_WRITE_BIT);
      }
#endif
    }
  }

  // Disable tonemapping for debug buffers or guide buffers (display raw values)
  shaderio::TonemapperData tonemapperData = m_resources.tonemapperData;
  if(m_resources.settings.debugMethod != shaderio::DebugMethod::eNone || usingGuideBuffer)
  {
    tonemapperData.isActive = 0;
  }

  m_resources.tonemapper.runCompute(cmd, gbufSize, tonemapperData, inputBuffer,
                                    m_resources.gBuffers.getDescriptorImageInfo(Resources::eImgTonemapped));

  // Memory barrier to ensure compute shader writes are complete before fragment shader reads
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
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
        m_resources.gBuffers.getDescriptorImageInfo(Resources::eImgSelection),
        m_resources.gBuffers.getDescriptorImageInfo(Resources::eImgTonemapped),
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
  const DlssDenoiser* dlss = m_pathTracer.getDlssDenoiser();
  if(dlss && dlss->isEnabled())
  {
    const VkExtent2D rs = dlss->getRenderSize();
    depthBufferSize     = {static_cast<float>(rs.width), static_cast<float>(rs.height)};
  }
#endif

  m_visualHelpers.render(cmd, m_resources.gBuffers.getColorImage(Resources::eImgTonemapped),
                         m_resources.gBuffers.getColorImageView(Resources::eImgTonemapped), m_resources.descriptorSet,
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
// Set DLSS hardware/extension availability
// This should be called early, before any DLSS initialization occurs
void GltfRenderer::setDlssHardwareAvailability(bool available)
{
  m_resources.settings.dlssHardwareAvailable = available;
}

//--------------------------------------------------------------------------------------------------
// Load the scene
void GltfRenderer::createScene(const std::filesystem::path& sceneFilename)
{
  nvutils::ScopedTimer st(__FUNCTION__);
  m_sceneSelection.clearSelection();  // Clear selection in new UI system
  m_resources.selectedRenderNodes.clear();

  if(sceneFilename.empty())
  {
    return;
  }

  std::filesystem::path filename = nvutils::findFile(sceneFilename, nvsamples::getResourcesDirs(), false);
  if(!filename.has_filename())
  {
    LOGW("Cannot find file: %s\n", nvutils::utf8FromPath(sceneFilename).c_str());
    removeFromRecentFiles(filename);
    return;
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
      return;
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
      return;
    }
    m_resources.scene = std::move(scn);
  }

  // Scene is loaded, we can create the Vulkan scene
  createVulkanScene();

  nvvkgltf::Scene* scene = m_resources.getScene();
  // Scene Browser system
  m_sceneBrowser.setScene(scene);
  m_sceneBrowser.setSelection(&m_sceneSelection);
  m_sceneBrowser.setUndoStack(&m_undoStack);
  m_sceneBrowser.setBbox(scene->getSceneBounds());
  m_sceneBrowser.setPendingDelete(&m_pendingDeleteNode, &m_openDeletePopupNextFrame);

  m_inspector.setScene(scene);
  m_inspector.setSelection(&m_sceneSelection);
  m_inspector.setUndoStack(&m_undoStack);
  m_inspector.setBbox(scene->getSceneBounds());

  m_resources.settings.infinitePlaneDistance = scene->getSceneBounds().min().y;  // Set the infinite plane distance to the bottom of the scene

  // Set camera from scene
  nvvkgltf::addSceneCamerasToWidget(m_cameraManip, filename, scene->getRenderCameras(), scene->getSceneBounds());

  // Default sky parameters
  m_resources.skyParams = {};

  // Need to update (push) all textures
  if(!updateTextures())
  {
    LOGE("Failed to update textures - cannot safely render scene");

    // Clean up the scene we just loaded - it's unsafe to render
    vkDeviceWaitIdle(m_device);
    cleanupScene();

    removeFromRecentFiles(filename);
    return;
  }

  addToRecentFiles(filename);
}

//--------------------------------------------------------------------------------------------------
// Load a .scene.json / .glXf descriptor: merge each model into a single Scene, duplicate for instances.
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
    const auto& modelEntry     = desc.models[modelIdx];
    int         wrapperNodeIdx = scene->mergeScene(modelEntry.resolvedPath, static_cast<uint32_t>(m_maxTextures));
    if(wrapperNodeIdx < 0)
    {
      LOGE("Failed to merge model: %s\n", nvutils::utf8FromPath(modelEntry.resolvedPath).c_str());
      continue;
    }
    const auto& firstInst = *instances[0];
    scene->editor().setNodeTRS(wrapperNodeIdx, firstInst.translation, firstInst.rotation, firstInst.scale);
    if(!firstInst.name.empty())
      scene->getModel().nodes[wrapperNodeIdx].name = firstInst.name;

    for(size_t i = 1; i < instances.size(); ++i)
    {
      const auto& inst        = *instances[i];
      int         copyNodeIdx = scene->editor().duplicateNode(wrapperNodeIdx, false);
      if(copyNodeIdx < 0)
      {
        LOGE("Failed to duplicate node for instance: %s\n", nvutils::utf8FromPath(modelEntry.resolvedPath).c_str());
        continue;
      }
      scene->editor().setNodeTRS(copyNodeIdx, inst.translation, inst.rotation, inst.scale);
      if(!inst.name.empty())
        scene->getModel().nodes[copyNodeIdx].name = inst.name;
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

  m_sceneBrowser.setScene(scene);
  m_sceneBrowser.setSelection(&m_sceneSelection);
  m_sceneBrowser.setUndoStack(&m_undoStack);
  m_sceneBrowser.setBbox(scene->getSceneBounds());
  m_sceneBrowser.setPendingDelete(&m_pendingDeleteNode, &m_openDeletePopupNextFrame);

  m_inspector.setScene(scene);
  m_inspector.setSelection(&m_sceneSelection);
  m_inspector.setUndoStack(&m_undoStack);
  m_inspector.setBbox(scene->getSceneBounds());

  m_resources.settings.infinitePlaneDistance = scene->getSceneBounds().min().y;

  nvvkgltf::addSceneCamerasToWidget(m_cameraManip, descriptorPath, scene->getRenderCameras(), scene->getSceneBounds());

  m_resources.skyParams = {};

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
  if(m_resources.getScene())
    m_resources.transformCompute.destroyGpuBuffers();
  m_resources.scene.reset();
  m_resources.sceneGpu.destroy();
  m_sceneBrowser.setScene(nullptr);
  m_inspector.setScene(nullptr);
  m_sceneSelection.clearSelection();  // Clear selection in new UI system
  m_resources.selectedRenderNodes.clear();

  // Reset animation control to avoid out-of-bounds access when loading a scene with fewer animations
  m_sceneBrowser.getAnimationControl().currentAnimation = 0;

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
void GltfRenderer::rebuildVulkanSceneInternal(bool rebuildTextures)
{
  // SYNC NOTE: Full scene rebuild (merge/compact/geometry change) -- wait ensures GPU is idle.
  NVVK_CHECK(vkQueueWaitIdle(m_app->getQueue(0).queue));

  nvvkgltf::Scene* scene = m_resources.getScene();

  {
    if(scene)
      m_resources.transformCompute.destroyGpuBuffers();  // Before scene RTX rebuild

    // Add WebP loading support to SceneVk (only needed for full rebuild with textures)
    if(rebuildTextures)
      m_resources.sceneVk.setImageLoadCallback(webPLoadCallback);

    VkCommandBuffer cmd{};
    nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
    m_resources.sceneGpu.rebuild(cmd, *scene, rebuildTextures);
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
    AnimationControl& animCtrl = m_sceneBrowser.getAnimationControl();
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
  }

  // Update textures if requested
  if(rebuildTextures)
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
  m_undoStack.clear();
  refreshCpuSceneGraphFromModel();
  rebuildVulkanSceneInternal(false);  // Geometry only, preserve textures
}

//--------------------------------------------------------------------------------------------------
// Full GPU resource rebuild including textures. Used after operations that modify the model
// structure: merging scenes, compacting resources, etc.
// Destroys all GPU resources and recreates from the current model.
//
void GltfRenderer::rebuildVulkanSceneFull()
{
  rebuildVulkanSceneInternal(true);
}

//--------------------------------------------------------------------------------------------------
// This function creates the Vulkan scene from the glTF model
// It builds the bottom-level and top-level acceleration structure
// The function is called when the scene is loaded
void GltfRenderer::createVulkanScene()
{
  {
    // Add WebP loading support to SceneVk
    m_resources.sceneVk.setImageLoadCallback(webPLoadCallback);

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
  VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                               | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
  if(m_resources.getScene() && m_resources.getScene()->animation().hasAnimation())
    flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

  // Create the bottom-level acceleration structure descriptors (no building yet)
  m_resources.sceneRtx.createBottomLevelAccelerationStructure(*m_resources.getScene(), m_resources.sceneVk, flags);

  // Build the bottom-level acceleration structure
  // Memory-conscious approach: build within a fixed memory budget using multiple command buffers if needed
  // Each build command is queued separately and followed by compaction to optimize memory usage
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
      m_loadPipeline.enqueue(cmd, [this] {
        VkCommandBuffer compactCmd{};
        nvvk::beginSingleTimeCommands(compactCmd, m_device, m_transientCmdPool);
        m_resources.sceneRtx.cmdCompactBlas(compactCmd);
        NVVK_CHECK(vkEndCommandBuffer(compactCmd));
        m_loadPipeline.enqueue(compactCmd);
      });

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
      m_loadPipeline.enqueue(cmd, [this] { m_resources.staging.releaseStaging(true); });
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
// Create the uniform buffers for frame-specific data
// This function initializes two key uniform buffers:
// 1. bFrameInfo - Contains per-frame camera matrices, environment settings, and debug information
//    Updated each frame with current view/projection matrices and rendering settings
// 2. bSkyParams - Contains physical parameters for the procedural sky simulation
//    Used when environment type is set to Sky instead of HDR
//
void GltfRenderer::createResourceBuffers()
{
  // Create the buffer of the current camera transformation, changing at each frame
  NVVK_CHECK(m_resources.allocator.createBuffer(m_resources.bFrameInfo, sizeof(shaderio::SceneFrameInfo),
                                                VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                                VMA_MEMORY_USAGE_CPU_TO_GPU));
  NVVK_DBG_NAME(m_resources.bFrameInfo.buffer);
  // Create the buffer of sky parameters, updated at each frame
  NVVK_CHECK(m_resources.allocator.createBuffer(m_resources.bSkyParams, sizeof(shaderio::SkyPhysicalParameters),
                                                VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                                VMA_MEMORY_USAGE_CPU_TO_GPU));
  NVVK_DBG_NAME(m_resources.bSkyParams.buffer);
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

  // 0: Descriptor SET: all textures of the scene
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              m_maxTextures, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  // The 2 textures are for the HDR environment map: one is the pre-integrated BRDF LUT, the other is the HDR image
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eTexturesHdr,
                                              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  // The 2 other HDR textures with cube maps: pre-convoluted diffuse and glossy maps
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eTexturesCube,
                                              VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2, VK_SHADER_STAGE_ALL, nullptr,
                                              VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                  | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
  NVVK_CHECK(m_resources.descriptorBinding[0].createDescriptorSetLayout(
      m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT, &m_resources.descriptorSetLayout[0]));
  NVVK_DBG_NAME(m_resources.descriptorSetLayout[0]);

  std::vector<VkDescriptorPoolSize> poolSize  = m_resources.descriptorBinding[0].calculatePoolSizes();
  VkDescriptorPoolCreateInfo        dpoolInfo = {
             .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
             .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |  // allows descriptor sets to be updated after they have been bound to a command buffer
               VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,  // individual descriptor sets can be freed from the descriptor pool
             .maxSets       = 15,  // For all GBuffer images (main: 3, DLSS: 8, OptiX: 2) + margin
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
// Recompile the shaders of the current renderer. See onUIMenu() for the key binding
void GltfRenderer::compileShaders()
{
  nvutils::ScopedTimer st(__FUNCTION__);
  if(m_resources.settings.renderSystem == RenderingMode::ePathtracer)
  {
    m_pathTracer.compileShader(m_resources);
  }
  else
  {
    m_rasterizer.compileShader(m_resources);
  }
}

//--------------------------------------------------------------------------------------------------
// Update the textures: this is called when the scene is loaded
// Textures are updated in the descriptor set (0)
bool GltfRenderer::updateTextures()
{
  // Now do the textures
  nvvk::WriteSetContainer write{};
  VkWriteDescriptorSet allTextures = m_resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eTextures);
  allTextures.dstSet               = m_resources.descriptorSet;

  uint32_t sceneTextureCount = m_resources.sceneVk.textureCount();

  if(sceneTextureCount == 0)
    return true;

  // CRITICAL: Materials directly index into allTextures[] - if scene exceeds capacity,
  // materials will access uninitialized descriptors causing crashes or corruption
  if(sceneTextureCount > m_maxTextures)
  {
    LOGE("FATAL: Scene has %u textures but descriptor set only supports %u!", sceneTextureCount, m_maxTextures);
    LOGE("       Materials would access invalid texture descriptors (undefined behavior).");
    LOGE("       Solutions:");
    LOGE("         1. Increase m_maxTextures in renderer.hpp (currently %u)", m_maxTextures);
    LOGE("         2. Reduce scene texture count (optimize/deduplicate textures)");
    return false;
  }

  allTextures.descriptorCount = sceneTextureCount;

  write.append(allTextures, m_resources.sceneVk.textures().data());
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

void GltfRenderer::onUndoRedo()
{
  resetFrame();
  m_sceneBrowser.markCachesDirty();
}

//--------------------------------------------------------------------------------------------------
// Update the frame counter
// This is called every frame to update the frame counter or to reset it if the camera has changed
// The frame counter is used to limit the number of frames rendered
// If the frame counter is greater than the maximum number of frames, the rendering stops
// Returns true if the frame counter is less than the maximum number of frames
bool GltfRenderer::updateFrameCounter()
{
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
  VkCommandBuffer cmd{};
  nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
  nvvk::StagingUploader uploader;
  uploader.init(&m_resources.allocator, true);

  // Load an HDR and create the important sampling acceleration structure
  std::filesystem::path filename;
  if(!hdrFilename.empty())
    filename = nvutils::findFile(hdrFilename, nvsamples::getResourcesDirs(), false);
  m_resources.hdrIbl.destroyEnvironment();
  m_resources.hdrIbl.loadEnvironment(cmd, uploader, filename, true);

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
  m_resources.hdrDome.setOutImage(m_resources.gBuffers.getDescriptorImageInfo(Resources::eImgRendered));
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
  m_loadPipeline.destroy();

  m_resources.allocator.destroyBuffer(m_resources.bFrameInfo);
  m_resources.allocator.destroyBuffer(m_resources.bSkyParams);
  if(m_resources.bSelectionBitMask.buffer != VK_NULL_HANDLE)
    m_resources.allocator.destroyBuffer(m_resources.bSelectionBitMask);

  vkDestroyDescriptorSetLayout(m_device, m_resources.descriptorSetLayout[0], nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_resources.descriptorSetLayout[1], nullptr);
  vkDestroyDescriptorPool(m_device, m_resources.descriptorPool, nullptr);
  vkDestroyCommandPool(m_device, m_transientCmdPool, nullptr);

  m_profilerGpuTimer.deinit();
  g_profilerManager.destroyTimeline(m_profilerTimeline);
  m_silhouette.deinit(m_resources);

  m_resources.tonemapper.deinit();
  m_resources.appMemoryTracker.untrack("GBuffers", m_resources.gBuffers, 3);
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
  AnimationControl& animCtrl = m_sceneBrowser.getAnimationControl();


  if(scn.animation().hasAnimation() && animCtrl.doAnimation())
  {
    const int nAnim = scn.animation().getNumAnimations();
    if(nAnim <= 0)
      return false;
    if(animCtrl.currentAnimation < 0 || animCtrl.currentAnimation >= nAnim)
      animCtrl.currentAnimation = 0;

    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Update animation");

    NVVK_DBG_SCOPE(cmd);
    nvvkgltf::SceneVk&  scnVk  = m_resources.sceneVk;
    nvvkgltf::SceneRtx& scnRtx = m_resources.sceneRtx;

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
        (void)scnVk.syncFromScene(m_resources.staging, scn, nvvkgltf::SceneVk::eSyncMaterials | nvvkgltf::SceneVk::eSyncLights);
        (void)scnVk.flushSceneDescIfDirty(m_resources.staging, scn);
      }
      else
      {
        m_resources.transformCompute.markGpuStale();
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

    scn.clearDirtyFlags();
    return true;
  }

  return false;
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
    m_resources.staging.cmdUploadAppended(cmd);
  }
  return synced;
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

  if(m_resources.getScene())
    m_resources.getScene()->clearDirtyFlags();

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

  const auto& df             = scene->getDirtyFlags();
  bool        changed        = !df.isEmpty();
  bool        stagingFlushed = false;

  const bool renderNodeOrNodeDirty = df.allRenderNodesDirty || !df.renderNodesVk.empty() || !df.nodes.empty();

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
    updateSceneChanges_NodeTransforms(cmd, scene, df);

    if(!df.isEmpty())
      m_resources.transformCompute.markGpuStale();

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
