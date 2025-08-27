/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <thread>
#include <vulkan/vulkan_core.h>
#include <glm/glm.hpp>
#include <fmt/format.h>

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
#include <nvapp/elem_dbgprintf.hpp>
#include <nvgui/axis.hpp>
#include <nvgui/file_dialog.hpp>
#include <nvgui/tonemapper.hpp>
#include <nvutils/profiler.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/mipmaps.hpp>
#include <nvvkgltf/camera_utils.hpp>

#include "create_tangent.hpp"
#include "renderer.hpp"
#include "ui_collapsing_header_manager.h"
#include "ui_mouse_state.hpp"
#include "utils.hpp"
#include "tinyobjloader/tiny_obj_loader.h"
#include "nvvkgltf/converter.hpp"

extern nvutils::ProfilerManager g_profilerManager;  // #PROFILER

// The constructor registers the parameters that can be set from the command line
GltfRenderer::GltfRenderer(nvutils::ParameterRegistry* paramReg)
{
  // All parameters that can be set from the command line
  paramReg->add({"envSystem", "Environment: [Sky:0, HDR:1]"}, (int*)&m_resources.settings.envSystem);
  paramReg->add({"renderSystem", "Renderer [Path tracer:0, Rasterizer:1]"}, (int*)&m_resources.settings.renderSystem);
  paramReg->add({"showAxis", "Show Axis"}, &m_resources.settings.showAxis);
  paramReg->add({"hdrEnvIntensity", "HDR Environment Intensity"}, &m_resources.settings.hdrEnvIntensity);
  paramReg->add({"hdrEnvRotation", "HDR Environment Rotation"}, &m_resources.settings.hdrEnvRotation);
  paramReg->add({"hdrBlur", "HDR Environment Blur"}, &m_resources.settings.hdrBlur);
  paramReg->addVector({"silhouetteColor", "Color of the silhouette"}, &m_resources.settings.silhouetteColor);
  paramReg->add({"debugMethod", "Debug Method"}, (int*)&m_resources.settings.debugMethod);
  paramReg->add({"useSolidBackground", "Use solid color background"}, &m_resources.settings.useSolidBackground, true);
  paramReg->addVector({"solidBackgroundColor", "Solid Background Color"}, &m_resources.settings.solidBackgroundColor);
  paramReg->add({"maxFrames", "Maximum number of iterations"}, &m_resources.settings.maxFrames);

  paramReg->add({"tmMethod", "Tonemapper method: [Filmic:0, Uncharted:1, Clip:2, ACES:3, Agx:4, KhronosPBR:5]"},
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
                                     VK_FORMAT_R8G8B8A8_UNORM,       // Tonemapped
                                     VK_FORMAT_R32G32B32A32_SFLOAT,  // Rendered image
                                     VK_FORMAT_R8_UNORM,             // Selection (Silhouette)
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

  // ===== Profiling & Performance =====
  {
    SCOPED_TIMER("Profiler");
    m_profilerTimeline = g_profilerManager.createTimeline({.name = "Primary Timeline"});
    m_profilerGpuTimer.init(m_profilerTimeline, m_app->getDevice(), m_app->getPhysicalDevice(), m_app->getQueue(0).familyIndex, false);
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
  if(processQueuedCommandBuffers())
  {
    return;  // Give back control to the UI
  }

  // Empty scene, clear the G-Buffer
  if(!m_resources.scene.valid())
  {
    clearGbuffer(cmd);
    return;
  }

  // Start the profiler section for the GPU timer
  auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

  // Update the animation
  bool didAnimate = updateAnimation(cmd);

  // Check for changes
  bool changed      = updateSceneChanges(cmd, didAnimate);
  bool frameChanged = updateFrameCounter();  // Check if the frame counter has changed

  if(changed || frameChanged)
  {

    // Update the scene frame information uniform buffer
    shaderio::SceneFrameInfo finfo{
        .viewMatrix             = m_cameraManip->getViewMatrix(),
        .projInv                = glm::inverse(m_cameraManip->getPerspectiveMatrix()),
        .viewInv                = glm::inverse(m_cameraManip->getViewMatrix()),
        .viewProjMatrix         = m_cameraManip->getPerspectiveMatrix() * m_cameraManip->getViewMatrix(),
        .prevMVP                = m_prevMVP,
        .envRotation            = m_resources.settings.hdrEnvRotation,
        .envBlur                = m_resources.settings.hdrBlur,
        .envIntensity           = m_resources.settings.hdrEnvIntensity,
        .useSolidBackground     = m_resources.settings.useSolidBackground ? 1 : 0,
        .backgroundColor        = m_resources.settings.solidBackgroundColor,
        .environmentType        = (int)m_resources.settings.envSystem,
        .selectedRenderNode     = m_resources.selectedObject,
        .debugMethod            = m_resources.settings.debugMethod,
        .useInfinitePlane       = m_resources.settings.useInfinitePlane ? 1 : 0,
        .infinitePlaneDistance  = m_resources.settings.infinitePlaneDistance,
        .infinitePlaneBaseColor = m_resources.settings.infinitePlaneBaseColor,
        .infinitePlaneMetallic  = m_resources.settings.infinitePlaneMetallic,
        .infinitePlaneRoughness = m_resources.settings.infinitePlaneRoughness,
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

    m_cmdBufferQueue = {};             // Clear the command buffer queue
    m_resources.scene.destroy();       // Destroy the current scene
    m_resources.selectedObject = -1;   // Reset the selected object
    m_uiSceneGraph.setModel(nullptr);  // Reset the UI model
    m_rasterizer.freeRecordCommandBuffer();

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
    // First, copy the camera
    nvvkgltf::RenderCamera camera;
    m_cameraManip->getLookat(camera.eye, camera.center, camera.up);
    camera.yfov  = glm::radians(m_cameraManip->getFov());
    camera.znear = m_cameraManip->getClipPlanes().x;
    camera.zfar  = m_cameraManip->getClipPlanes().y;
    m_resources.scene.setSceneCamera(camera);

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

  // When debug method is not none, the tonemapper should do nothing to visualize the data
  shaderio::TonemapperData tonemapperData = m_resources.tonemapperData;
  if(m_resources.settings.debugMethod != shaderio::DebugMethod::eNone)
  {
    tonemapperData.isActive = 0;
  }
  m_resources.tonemapper.runCompute(cmd, m_resources.gBuffers.getSize(), tonemapperData,
                                    m_resources.gBuffers.getDescriptorImageInfo(Resources::eImgRendered),
                                    m_resources.gBuffers.getDescriptorImageInfo(Resources::eImgTonemapped));

  // Memory barrier to ensure compute shader writes are complete before fragment shader reads
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
}

//--------------------------------------------------------------------------------------------------
// Render the silhouette of the selected object
void GltfRenderer::silhouette(VkCommandBuffer cmd)
{
  // Adding the silhouette pass after all rendering passes
  if(m_resources.selectedObject > -1)
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
  updateTextures();

  addToRecentFiles(filename);
}

//--------------------------------------------------------------------------------------------------
// This function creates the Vulkan scene from the glTF model
// It builds the bottom-level and top-level acceleration structure
// The function is called when the scene is loaded
void GltfRenderer::createVulkanScene()
{
  VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                               | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
  if(m_resources.scene.hasAnimation())
  {
    flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;  // Allow update
  }

  {
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

  // Build mapping for faster node lookups
  updateNodeToRenderNodeMap();
}

//--------------------------------------------------------------------------------------------------
// Clear the G-Buffer
void GltfRenderer::clearGbuffer(VkCommandBuffer cmd)
{
  const VkClearColorValue clearValue = {{0.17f, 0.21f, 0.25f, 1.f}};
  VkImageSubresourceRange range      = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
  vkCmdClearColorImage(cmd, m_resources.gBuffers.getColorImage(Resources::eImgTonemapped), VK_IMAGE_LAYOUT_GENERAL,
                       &clearValue, 1, &range);

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
  uint32_t maxTextures = std::min(10000U, deviceProperties.limits.maxDescriptorSetSampledImages - 1);

  // 0: Descriptor SET: all textures of the scene
  m_resources.descriptorBinding[0].addBinding(shaderio::BindingPoints::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              maxTextures, VK_SHADER_STAGE_ALL, nullptr,
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
             .maxSets       = 10,                                         // For all DLSS images
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
void GltfRenderer::updateTextures()
{
  // Now do the textures
  nvvk::WriteSetContainer write{};
  VkWriteDescriptorSet allTextures = m_resources.descriptorBinding[0].getWriteSet(shaderio::BindingPoints::eTextures);
  allTextures.dstSet               = m_resources.descriptorSet;
  allTextures.descriptorCount      = m_resources.sceneVk.nbTextures();
  if(allTextures.descriptorCount == 0)
    return;
  write.append(allTextures, m_resources.sceneVk.textures().data());
  vkUpdateDescriptorSets(m_device, write.size(), write.data(), 0, nullptr);
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
  static float     ref_fov{0};
  static glm::mat4 ref_cam_matrix;

  const auto& m   = m_cameraManip->getViewMatrix();
  const auto  fov = m_cameraManip->getFov();

  if(ref_cam_matrix != m || ref_fov != fov)
  {
    resetFrame();
    ref_cam_matrix = m;
    ref_fov        = fov;
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
// Update the animation
bool GltfRenderer::updateAnimation(VkCommandBuffer cmd)
{
  if(m_resources.scene.hasAnimation() && m_animControl.doAnimation())
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight
    float                    deltaTime = m_animControl.deltaTime();
    nvvkgltf::AnimationInfo& animInfo  = m_resources.scene.getAnimationInfo(m_animControl.currentAnimation);
    if(m_animControl.isReset())
    {
      animInfo.reset();
    }
    else
    {
      animInfo.incrementTime(deltaTime);
    }

    m_resources.scene.updateAnimation(m_animControl.currentAnimation);
    m_resources.scene.updateRenderNodes();

    m_animControl.clearStates();

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
bool GltfRenderer::updateSceneChanges(VkCommandBuffer cmd, bool didAnimate)
{
  bool changed = m_uiSceneGraph.hasAnyChanges();
  if(m_uiSceneGraph.hasMaterialChanged())
  {
    m_resources.sceneVk.updateMaterialBuffer(cmd, m_resources.staging, m_resources.scene);
  }
  if(m_uiSceneGraph.hasLightChanged())
  {
    m_resources.sceneVk.updateRenderLightsBuffer(cmd, m_resources.staging, m_resources.scene);
  }
  if(m_resources.dirtyFlags.test(DirtyFlags::eVulkanScene))
  {
    m_resources.scene.updateRenderNodes();
    m_resources.sceneVk.updateRenderNodesBuffer(cmd, m_resources.staging, m_resources.scene);
    m_resources.sceneVk.updateRenderPrimitivesBuffer(cmd, m_resources.staging, m_resources.scene);
    m_resources.sceneVk.updateRenderLightsBuffer(cmd, m_resources.staging, m_resources.scene);
    m_resources.dirtyFlags.reset(DirtyFlags::eVulkanScene);
    changed = true;
  }
  if(m_uiSceneGraph.hasTransformChanged() || didAnimate)
  {
    m_resources.scene.updateRenderNodes();
    m_resources.sceneVk.updateRenderNodesBuffer(cmd, m_resources.staging, m_resources.scene);
    m_resources.sceneVk.updateRenderPrimitivesBuffer(cmd, m_resources.staging, m_resources.scene);
    m_resources.sceneVk.updateRenderLightsBuffer(cmd, m_resources.staging, m_resources.scene);
    // Make sure the staging buffers are uploaded before the acceleration structures are updated
    m_resources.staging.cmdUploadAppended(cmd);
    // Ensure all buffer copy operations complete before acceleration structure build begins
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    m_resources.sceneRtx.updateBottomLevelAS(cmd, m_resources.scene);
    m_resources.sceneRtx.updateTopLevelAS(cmd, m_resources.staging, m_resources.scene);
  }
  if(m_uiSceneGraph.hasMaterialFlagChanges() || m_uiSceneGraph.hasVisibilityChanged())
  {
    m_resources.scene.updateRenderNodes();
    m_resources.sceneRtx.updateTopLevelAS(cmd, m_resources.staging, m_resources.scene);
  }
  if(changed || didAnimate)
  {
    m_resources.staging.cmdUploadAppended(cmd);
    resetFrame();
  }
  m_uiSceneGraph.resetChanges();

  return changed || didAnimate;
}


//--------------------------------------------------------------------------------------------------
// Create a mapping from node ID to render node index for faster lookups
// This is a critical optimization that enables O(1) lookups from scene graph nodes to renderer nodes,
// enabling quick interaction between UI selections and the actual render objects.
// Called during scene creation and whenever the scene structure changes.
void GltfRenderer::updateNodeToRenderNodeMap()
{
  m_nodeToRenderNodeMap.clear();
  auto& renderNodes = m_resources.scene.getRenderNodes();
  for(size_t i = 0; i < renderNodes.size(); i++)
  {
    m_nodeToRenderNodeMap[renderNodes[i].refNodeID] = static_cast<int>(i);
  }
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
