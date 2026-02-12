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

#include "renderer.hpp"
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
  paramReg->add({"showMemStats", "Show Axis"}, &m_resources.settings.showMemStats);
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
    m_settingsHandler.setSetting("showMemStats", &m_resources.settings.showMemStats);
    m_settingsHandler.setSetting("showCameraWindow", &m_resources.settings.showCameraWindow);
    m_settingsHandler.setSetting("showSceneGraphWindow", &m_resources.settings.showSceneGraphWindow);
    m_settingsHandler.setSetting("showSettingsWindow", &m_resources.settings.showSettingsWindow);
    m_settingsHandler.setSetting("showPropertiesWindow", &m_resources.settings.showPropertiesWindow);
    m_settingsHandler.setSetting("showEnvironmentWindow", &m_resources.settings.showEnvironmentWindow);
    m_settingsHandler.setSetting("showTonemapperWindow", &m_resources.settings.showTonemapperWindow);
    m_settingsHandler.setSetting("showStatisticsWindow", &m_resources.settings.showStatisticsWindow);
    m_settingsHandler.setSetting("envSystem", (int*)&m_resources.settings.envSystem);
    m_settingsHandler.setSetting("renderSystem", (int*)&m_resources.settings.renderSystem);
    m_settingsHandler.setSetting("useSolidBackground", &m_resources.settings.useSolidBackground);
    m_settingsHandler.setSetting("solidBackgroundColor", &m_resources.settings.solidBackgroundColor);
    m_pathTracer.setSettingsHandler(&m_settingsHandler);
    m_rasterizer.setSettingsHandler(&m_settingsHandler);
    m_settingsHandler.addImGuiHandler();
  }

  // ===== Memory Allocation & Buffer Management =====
  m_resources.allocator.init({
      .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
      .physicalDevice   = app->getPhysicalDevice(),
      .device           = app->getDevice(),
      .instance         = app->getInstance(),
      .vulkanApiVersion = VK_API_VERSION_1_4,
  });  // Allocator

  m_transientCmdPool = nvvk::createTransientCommandPool(m_device, app->getQueue(0).familyIndex);
  NVVK_DBG_NAME(m_transientCmdPool);

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

  // G-Buffer
  m_resources.gBuffers.init({.allocator = &m_resources.allocator,
                             .colorFormats =
                                 {
                                     VK_FORMAT_R8G8B8A8_UNORM,       // Tonemapped (eImgTonemapped)
                                     VK_FORMAT_R32G32B32A32_SFLOAT,  // Rendered image (eImgRendered)
                                     VK_FORMAT_R8_UNORM,             // Selection/Silhouette (eImgSelection)
                                 },
                             .depthFormat    = nvvk::findDepthFormat(app->getPhysicalDevice()),
                             .imageSampler   = linearSampler,
                             .descriptorPool = m_app->getTextureDescriptorPool()});
  {
    VkCommandBuffer cmd{};
    nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
    m_resources.gBuffers.update(cmd, {100, 100});
    nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_app->getQueue(0).queue);
  }

  // ===== Rendering Utilities =====

  // Ray picker
  m_rayPicker.init(&m_resources.allocator);

  // Tonemapper
  m_resources.tonemapper.init(&m_resources.allocator, tonemapper_slang);

  // Silhouette renderer
  m_silhouette.init(m_resources);

  // ===== Scene & Acceleration Structure =====
  m_resources.sceneVk.init(&m_resources.allocator, &m_resources.samplerPool);
  m_resources.sceneRtx.init(&m_resources.allocator);

  m_resources.scene.supportedExtensions().insert(EXT_TEXTURE_WEBP_EXTENSION_NAME);
  m_resources.sceneVk.setImageLoadCallback(webPLoadCallback);

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
    m_resources.slangCompiler.addOption(
        {CompilerOptionName::DebugInformation, {CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_MAXIMAL}});
    m_resources.slangCompiler.addOption(
        {CompilerOptionName::Optimization, {CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_DEFAULT}});

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
}

//--------------------------------------------------------------------------------------------------
// Detach the renderers and destroy the resources
void GltfRenderer::onDetach()
{
  vkDeviceWaitIdle(m_device);
  m_pathTracer.onDetach(m_resources);
  m_rasterizer.onDetach(m_resources);
  destroyResources();
}

//--------------------------------------------------------------------------------------------------
// Resize the G-Buffer and the renderers
void GltfRenderer::onResize(VkCommandBuffer cmd, const VkExtent2D& size)
{
  m_resources.gBuffers.update(cmd, size);
  m_pathTracer.onResize(cmd, size, m_resources);
  m_rasterizer.onResize(cmd, size, m_resources);
  m_resources.hdrDome.setOutImage(m_resources.gBuffers.getDescriptorImageInfo(Resources::eImgRendered));

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

  // Process queued command buffers in FIFO order
  while(processQueuedCommandBuffers())
  {
    // In headless, process all command buffers, don't give back control to the UI
    // so everything is ready for the first frame
    if(!m_app->isHeadless())
      return;  // Give back control to the UI if not headless
  }

  // Empty scene, clear the G-Buffer
  if(!m_resources.scene.valid())
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
        .viewMatrix         = m_cameraManip->getViewMatrix(),
        .projInv            = glm::inverse(m_cameraManip->getPerspectiveMatrix()),
        .viewInv            = glm::inverse(m_cameraManip->getViewMatrix()),
        .viewProjMatrix     = m_cameraManip->getPerspectiveMatrix() * m_cameraManip->getViewMatrix(),
        .prevMVP            = m_prevMVP,
        .isOrthographic     = (m_cameraManip->getProjectionType() == nvutils::CameraManipulator::Orthographic) ? 1 : 0,
        .envRotation        = m_resources.settings.hdrEnvRotation,
        .envBlur            = m_resources.settings.hdrBlur,
        .envIntensity       = m_resources.settings.hdrEnvIntensity,
        .useSolidBackground = m_resources.settings.useSolidBackground ? 1 : 0,
        .backgroundColor    = m_resources.settings.solidBackgroundColor,
        .environmentType    = (int)m_resources.settings.envSystem,
        .selectedRenderNode = m_resources.selectedRenderNode,
        .debugMethod        = m_resources.settings.debugMethod,
        .useInfinitePlane = m_resources.settings.useInfinitePlane ? (m_resources.settings.isShadowCatcher ? 2 : 1) : 0,
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
// Load a glTF scene or an HDR file (called from both Load Scene and Load HDR Environment menu items)
void GltfRenderer::onFileDrop(const std::filesystem::path& filename)
{
  vkQueueWaitIdle(m_app->getQueue(0).queue);

  if(nvutils::extensionMatches(filename, ".gltf") || nvutils::extensionMatches(filename, ".glb")
     || nvutils::extensionMatches(filename, ".obj"))
  {
    if(m_busy.isBusy())
      return;

    m_cmdBufferQueue = {};  // Clear the command buffer queue
    cleanupScene();         // Cleanup current scene
    m_rasterizer.freeRecordCommandBuffer(m_resources);

    std::thread([=, this]() {
      m_busy.start("Loading");
      m_lastSceneDirectory = filename.parent_path();
      createScene(filename);
      m_busy.stop();
    }).detach();
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
  if(m_resources.scene.valid() && !filename.empty())
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
      m_resources.scene.setSceneCameras(cameras);
    }

    // Saving the scene
    return m_resources.scene.save(filename);
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
// Render the silhouette of the selected object
void GltfRenderer::silhouette(VkCommandBuffer cmd)
{
  // Adding the silhouette pass after all rendering passes
  if(m_resources.selectedRenderNode > -1)
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

    std::vector<VkDescriptorImageInfo> imageInfos = {
        m_resources.gBuffers.getDescriptorImageInfo(Resources::eImgSelection),
        m_resources.gBuffers.getDescriptorImageInfo(Resources::eImgTonemapped),
    };
    m_silhouette.dispatch(cmd, m_resources.gBuffers.getSize(), imageInfos);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
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
  m_uiSceneGraph.setModel(nullptr);

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
      m_resources.scene.takeModel(std::move(model));
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
    if(!m_resources.scene.load(filename))  // Loading the scene
    {
      LOGW("Error loading scene: %s\n", nvutils::utf8FromPath(filename).c_str());
      removeFromRecentFiles(filename);
      return;
    }
  }

  // Scene is loaded, we can create the Vulkan scene
  createVulkanScene();

  // UI needs to be updated
  m_uiSceneGraph.setModel(&m_resources.scene.getModel());
  m_uiSceneGraph.setBbox(m_resources.scene.getSceneBounds());
  m_resources.settings.infinitePlaneDistance = m_resources.scene.getSceneBounds().min().y;  // Set the infinite plane distance to the bottom of the scene

  // Set camera from scene
  nvvkgltf::addSceneCamerasToWidget(m_cameraManip, filename, m_resources.scene.getRenderCameras(),
                                    m_resources.scene.getSceneBounds());

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
// Helper function to cleanup the current scene
//
void GltfRenderer::cleanupScene()
{
  m_resources.scene.destroy();
  m_resources.sceneVk.destroy();
  m_resources.sceneRtx.destroy();
  m_uiSceneGraph.setModel(nullptr);
  m_resources.selectedRenderNode = -1;

  // Reset animation control to avoid out-of-bounds access when loading a scene with fewer animations
  m_animControl.currentAnimation = 0;

  // Reset memory statistics for the new scene
  // Keeps lifetime allocation/deallocation counts but resets current and peak values
  m_resources.sceneVk.getMemoryTracker().reset();
  m_resources.sceneRtx.getMemoryTracker().reset();
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
  vkDeviceWaitIdle(m_device);

  // Destroy only geometry resources (preserve textures - they didn't change)
  m_resources.sceneRtx.destroy();
  m_resources.sceneVk.destroyGeometry();

  // Re-parse the scene to update RenderPrimitives with new accessor counts
  m_resources.scene.setCurrentScene(m_resources.scene.getCurrentScene());

  // Recreate only geometry resources
  {
    VkCommandBuffer cmd{};
    nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
    m_resources.sceneVk.createGeometry(cmd, m_resources.staging, m_resources.scene);
    m_resources.staging.cmdUploadAppended(cmd);
    {
      std::lock_guard<std::mutex> lock(m_cmdBufferQueueMutex);
      m_cmdBufferQueue.push({cmd, false});
    }
  }

  // Rebuild acceleration structures
  buildAccelerationStructures();

  // Update UI with the modified model
  m_uiSceneGraph.setModel(&m_resources.scene.getModel());
  m_uiSceneGraph.setBbox(m_resources.scene.getSceneBounds());

  // Note: No updateTextures() needed - textures were preserved
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

    m_resources.sceneVk.create(cmd, m_resources.staging, m_resources.scene, false);  // Creating the scene in Vulkan buffers
    m_resources.staging.cmdUploadAppended(cmd);
    {
      std::lock_guard<std::mutex> lock(m_cmdBufferQueueMutex);
      m_cmdBufferQueue.push({cmd, false});  // Not a BLAS build command
    }
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
  if(m_resources.scene.hasAnimation())
    flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

  // Create the bottom-level acceleration structure descriptors (no building yet)
  m_resources.sceneRtx.createBottomLevelAccelerationStructure(m_resources.scene, m_resources.sceneVk, flags);

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
      // This won't compact the BLAS, but will create the acceleration structure
      finished = m_resources.sceneRtx.cmdBuildBottomLevelAccelerationStructure(cmd, 512'000'000);
      {
        std::lock_guard<std::mutex> lock(m_cmdBufferQueueMutex);
        m_cmdBufferQueue.push({cmd, true});  // Mark as BLAS build command for immediate compaction
      }

    } while(!finished);

    // Track all BLAS allocations now that they're all built
    m_resources.sceneRtx.trackBlasMemory();

    // Queue TLAS building for after all BLAS work completes
    // TLAS is the top-level structure referencing all bottom-level acceleration structures
    {
      VkCommandBuffer cmd{};
      nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
      m_resources.sceneRtx.cmdCreateBuildTopLevelAccelerationStructure(cmd, m_resources.staging, m_resources.scene);
      m_resources.staging.cmdUploadAppended(cmd);
      {
        std::lock_guard<std::mutex> lock(m_cmdBufferQueueMutex);
        m_cmdBufferQueue.push({cmd, false});  // Not a BLAS build command
      }
    }
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
  // Reserve 2050 textures (2000 for scene textures + 50 for other purposes like the environment)
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

  uint32_t sceneTextureCount = m_resources.sceneVk.nbTextures();

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
  // Process any remaining command buffers in the queue
  {
    std::lock_guard<std::mutex> lock(m_cmdBufferQueueMutex);
    while(!m_cmdBufferQueue.empty())
    {
      CommandBufferInfo cmdInfo = m_cmdBufferQueue.front();
      m_cmdBufferQueue.pop();
      nvvk::endSingleTimeCommands(cmdInfo.cmdBuffer, m_device, m_transientCmdPool, m_app->getQueue(0).queue);
    }
  }

  m_resources.allocator.destroyBuffer(m_resources.bFrameInfo);
  m_resources.allocator.destroyBuffer(m_resources.bSkyParams);

  vkDestroyDescriptorSetLayout(m_device, m_resources.descriptorSetLayout[0], nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_resources.descriptorSetLayout[1], nullptr);
  vkDestroyDescriptorPool(m_device, m_resources.descriptorPool, nullptr);
  vkDestroyCommandPool(m_device, m_transientCmdPool, nullptr);

  m_profilerGpuTimer.deinit();
  g_profilerManager.destroyTimeline(m_profilerTimeline);
  m_silhouette.deinit(m_resources);

  m_resources.tonemapper.deinit();
  m_resources.gBuffers.deinit();
  m_resources.sceneVk.deinit();
  m_resources.sceneRtx.deinit();
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
  nvvkgltf::Scene& scn = m_resources.scene;


  if(scn.hasAnimation() && m_animControl.doAnimation())
  {
    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Update animation");

    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
    nvvkgltf::SceneVk&  scnVk  = m_resources.sceneVk;
    nvvkgltf::SceneRtx& scnRtx = m_resources.sceneRtx;

    bool hasMorphOrSkin = !scn.getMorphPrimitives().empty() || !scn.getSkinNodes().empty();

    // Find the current animation and update its time
    float                    deltaTime = m_animControl.deltaTime();
    nvvkgltf::AnimationInfo& animInfo  = scn.getAnimationInfo(m_animControl.currentAnimation);
    if(m_animControl.isReset())
    {
      animInfo.reset();
    }
    else
    {
      animInfo.incrementTime(deltaTime);
    }

    // Update the element values: transformation, weights
    std::unordered_set<int> dirtyNodeIds = scn.updateAnimation(m_animControl.currentAnimation);

    // KHR_animation_pointer: Check if any materials/lights were animated and need GPU update
    auto& animPointer = scn.getAnimationPointer();
    if(animPointer.hasDirty())
    {
      // Materials were animated - surgical update only dirty materials
      if(!animPointer.getDirtyMaterials().empty())
      {
        scnVk.updateMaterialBuffer(m_resources.staging, scn, animPointer.getDirtyMaterials());
      }

      // Lights were animated - surgical update only dirty lights
      if(!animPointer.getDirtyLights().empty())
      {
        scnVk.updateRenderLightsBuffer(m_resources.staging, scn, animPointer.getDirtyLights());
      }

      // Animated visibility changes - update TLAS if needed
      if(!animPointer.getDirtyNodes().empty())
      {
        // Currently only visibility is supported for animation pointer, which is why we directly update TLAS here and not matrices.
        std::unordered_set<int> dirtyRenderNodes;
        bool updateAllRenderNodes = scn.collectRenderNodeIndices(animPointer.getDirtyNodes(), dirtyRenderNodes, true, 0.5f);
        if(updateAllRenderNodes)
          dirtyRenderNodes.clear();  // empty = full update
        m_resources.sceneRtx.updateTopLevelAS(cmd, m_resources.staging, m_resources.scene, dirtyRenderNodes);
      }

      // Clear dirty flags after upload
      animPointer.clearDirty();
    }

    m_animControl.clearStates();

    // Update the world matrices of the scene nodes
    scn.updateNodeWorldMatrices(dirtyNodeIds);

    // Surgical update: only update dirty renderNodes
    std::unordered_set<int> dirtyRenderNodes;
    bool updateAllRenderNodes = scn.collectRenderNodeIndices(dirtyNodeIds, dirtyRenderNodes, true, 0.5f);
    if(updateAllRenderNodes)
      dirtyRenderNodes.clear();  // empty = full update

    // Update to the GPU the matrices of the rendernodes that changed
    scnVk.updateRenderNodesBuffer(m_resources.staging, scn, dirtyRenderNodes);


    // Update the morph and skinning related buffers
    if(hasMorphOrSkin)
    {
      auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "Morph or Skin");
      scnVk.updateRenderPrimitivesBuffer(cmd, m_resources.staging, scn);
    }

    // Make sure the staging buffers are uploaded before the acceleration structures are updated
    m_resources.staging.cmdUploadAppended(cmd);

    // Ensure all buffer copy operations complete before acceleration structure build begins
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);

    // Update the bottom-level acceleration structures if morphing or skinning is used
    {
      auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, "AS update");
      if(hasMorphOrSkin)
        scnRtx.updateBottomLevelAS(cmd, scn);

      // Update the top-level acceleration structure
      scnRtx.updateTopLevelAS(cmd, m_resources.staging, scn, dirtyRenderNodes);
    }

    return true;
  }

  return false;
}

//--------------------------------------------------------------------------------------------------
// Update the scene based on changes from UI or animation
// This is a critical synchronization point for changes to scene data, ensuring that:
// 1. UI modifications to materials, lights, and transformations are propagated to GPU buffers
// 2. Animation changes are reflected in acceleration structures
// 3. Vulkan buffers and acceleration structures remain in sync with scene state
// 4. Frame counter is reset when needed to restart progressive rendering
// Returns true if any changes were made that require re-rendering
bool GltfRenderer::updateSceneChanges(VkCommandBuffer cmd)
{
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

  bool changed             = false;
  changed                  = m_uiSceneGraph.hasAnyChanges();  // Will update command buffer for any changes
  bool stagingUploadIssued = false;

  // Update the materials
  if(m_uiSceneGraph.hasMaterialChanged())
  {
    m_resources.sceneVk.updateMaterialBuffer(m_resources.staging, m_resources.scene, m_uiSceneGraph.getDirtyMaterials());
  }

  // When alpha or double side change, the TLAS `VK_GEOMETRY_INSTANCE_*` flag change
  if(m_uiSceneGraph.hasMaterialInstanceFlagChanges())
  {
    const auto& dirtyRenderNodes = m_resources.scene.getMaterialRenderNodes(m_uiSceneGraph.getMaterialInstanceFlagsChanged());
    m_resources.sceneRtx.updateTopLevelAS(cmd, m_resources.staging, m_resources.scene, dirtyRenderNodes);
  }

  // Update the lights
  if(m_uiSceneGraph.hasLightChanged())
  {
    m_resources.sceneVk.updateRenderLightsBuffer(m_resources.staging, m_resources.scene, m_uiSceneGraph.getDirtyLights());
  }

  // Update the render nodes for the material variants
  if(m_resources.dirtyMaterialVariants.size() > 0)
  {
    m_resources.sceneVk.updateRenderNodesBuffer(m_resources.staging, m_resources.scene, m_resources.dirtyMaterialVariants);
    m_resources.dirtyMaterialVariants.clear();
    m_rasterizer.freeRecordCommandBuffer(m_resources);
    changed = true;
  }

  // Recursive visibility update
  if(m_uiSceneGraph.hasVisibilityChanged())
  {
    const auto& dirtyNodes = m_uiSceneGraph.getDirtyVisibilityNodes();
    for(auto& dirtyNode : dirtyNodes)
    {
      m_resources.scene.updateVisibility(dirtyNode);
    }

    // Update for visibility TLAS
    std::unordered_set<int> dirtyRenderNodes;
    bool updateAllRenderNodes = m_resources.scene.collectRenderNodeIndices(dirtyNodes, dirtyRenderNodes, true, 0.5f);
    if(updateAllRenderNodes)
      dirtyRenderNodes.clear();  // empty = full update
    m_resources.sceneRtx.updateTopLevelAS(cmd, m_resources.staging, m_resources.scene, dirtyRenderNodes);
  }

  // Update the transforms
  if(m_uiSceneGraph.hasTransformChanged())
  {
    SCOPED_TIMER("hasTransformChanged");

    // Surgical update: only update dirty renderNodes (empty set during animation = update all)
    const auto&             dirtyNodes = m_uiSceneGraph.getDirtyNodes();
    std::unordered_set<int> dirtyRenderNodes;
    bool                    updateAllRenderNodes = false;

    // Update the world matrices of the scene nodes
    m_resources.scene.updateNodeWorldMatrices(dirtyNodes);

    // Find which render nodes need to be updated
    updateAllRenderNodes = m_resources.scene.collectRenderNodeIndices(dirtyNodes, dirtyRenderNodes, true, 0.5f);
    if(updateAllRenderNodes)
      dirtyRenderNodes.clear();  // empty = full update
    m_resources.sceneVk.updateRenderNodesBuffer(m_resources.staging, m_resources.scene, dirtyRenderNodes);
    m_resources.sceneVk.updateRenderLightsBuffer(m_resources.staging, m_resources.scene);  // Empty set = update all

    // Make sure the staging buffers are uploaded before the acceleration structures are updated
    m_resources.staging.cmdUploadAppended(cmd);
    stagingUploadIssued = true;

    // Ensure all buffer copy operations complete before acceleration structure build begins
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    m_resources.sceneRtx.updateTopLevelAS(cmd, m_resources.staging, m_resources.scene, dirtyRenderNodes);
  }

  // Re-pushing the tangents if they were recomputed
  if(m_resources.dirtyFlags.test(DirtyFlags::eDirtyTangents))
  {
    m_resources.sceneVk.updateVertexBuffers(m_resources.staging, m_resources.scene);
    m_resources.dirtyFlags.reset(DirtyFlags::eDirtyTangents);
    changed = true;
  }

  // Update changes if needed
  if(changed && !stagingUploadIssued)
  {
    m_resources.staging.cmdUploadAppended(cmd);
  }
  m_uiSceneGraph.resetChanges();

  return changed;
}


//--------------------------------------------------------------------------------------------------
// Process queued command buffers in FIFO order
// Those command buffers are created in worker threads while loading or processing a scene
// It will process one command buffer at a time, then give back control to the UI
// Command buffers can be of two types:
// 1. Regular command buffers (isBlasBuild=false): These execute scene creation, texture uploads, etc.
// 2. BLAS build command buffers (isBlasBuild=true): These build bottom-level acceleration structures
//    and are immediately followed by BLAS compaction to optimize memory usage
//
bool GltfRenderer::processQueuedCommandBuffers()
{
  std::lock_guard<std::mutex> lock(m_cmdBufferQueueMutex);
  if(!m_cmdBufferQueue.empty())
  {
    SCOPED_TIMER("Processing queued command buffer\n");

    // Get the command buffer information from the queue
    CommandBufferInfo cmdInfo = m_cmdBufferQueue.front();
    m_cmdBufferQueue.pop();

    // Execute the command buffer
    nvvk::endSingleTimeCommands(cmdInfo.cmdBuffer, m_device, m_transientCmdPool, m_app->getQueue(0).queue);

    // If this was a BLAS build command, immediately compact after it
    if(cmdInfo.isBlasBuild)
    {
      // Create a command buffer for compaction
      VkCommandBuffer cmd{};
      nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool);
      m_resources.sceneRtx.cmdCompactBlas(cmd);
      // Submit the compaction command buffer immediately
      nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, m_app->getQueue(0).queue);
    }
    if(m_cmdBufferQueue.empty())
      m_resources.staging.releaseStaging(true);
    return true;  // Command buffer was processed
  }
  return false;  // No command buffer was processed
}
