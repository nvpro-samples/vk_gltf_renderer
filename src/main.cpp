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

// #define USE_NSIGHT_AFTERMATH

// #define USE_DBG_PRINTF


#include <fmt/format.h>
#include <stb/stb_image.h>
#include <GLFW/glfw3.h>
#include <unordered_set>
#include <string>
#undef APIENTRY

#include <nvaftermath/aftermath.hpp>
#include <nvapp/application.hpp>
#include <nvapp/elem_dbgprintf.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvapp/elem_sequencer.hpp>
#include <nvgpu_monitor/elem_gpu_monitor.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/parameter_sequencer.hpp>
#include <nvvk/context.hpp>
#include <nvvk/validation_settings.hpp>

#include "renderer.hpp"
#include "docs/app_icon_png.h"
#include "version.hpp"

nvutils::ProfilerManager g_profilerManager;  // #PROFILER

//////////////////////////////////////////////////////////////////////////
// Create and set the window icon
static void setWindowIcon(GLFWwindow* window)
{
  GLFWimage icon{};
  int       channels = 0;
  icon.pixels        = stbi_load_from_memory(app_icon_png, app_icon_png_len, &icon.width, &icon.height, &channels, 4);
  if(icon.pixels)
  {
    glfwSetWindowIcon(window, 1, &icon);  // Set icon to window
    glfwPollEvents();                     // Force icon to be immediately shown
    stbi_image_free(icon.pixels);
  }
}

//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char** argv) -> int
{
  nvapp::ApplicationCreateInfo appInfo;
  nvvk::ContextInitInfo        vkSetup;
  nvutils::Logger&             logger   = nvutils::Logger::getInstance();
  nvutils::Logger::LogLevel    logLevel = nvutils::Logger::LogLevel::eINFO;
  nvutils::Logger::ShowFlags   logShow  = nvutils::Logger::ShowBits::eSHOW_TIME;
#if NDEBUG
  logger.breakOnError(false);
#endif

  BenchmarkOptions benchmarkOptions;

  nvutils::ParameterRegistry            parameterRegistry;
  nvutils::ParameterParser              cli(nvutils::getExecutablePath().stem().string(), {".txt"});
  nvutils::ParameterSequencer::InitInfo sequencerInfo{
      .parameterParser   = &cli,
      .parameterRegistry = &parameterRegistry,
      .profilerManager   = &g_profilerManager,
  };

  // Global variables
  std::filesystem::path sceneFilename{};             // "shader_ball.gltf"};  // Default scene
  std::filesystem::path hdrFilename{"std_env.hdr"};  // Default HDR

  // Application defaults overrides
  appInfo.preferredVsyncOffMode = VK_PRESENT_MODE_MAILBOX_KHR;

  // Command line parameters registration
  parameterRegistry.add({"scenefile", "Input scene filename"}, {".gltf"}, &sceneFilename);
  parameterRegistry.add({"hdrfile", "Input HDR filename"}, {".hdr"}, &hdrFilename);
  parameterRegistry.addVector({"size", "Size of the window to be created", "s"}, &appInfo.windowSize);
  parameterRegistry.add({"headless"}, &appInfo.headless, true);
  parameterRegistry.add({"frames", "Number of frames to run in headless mode"}, &appInfo.headlessFrameCount);
  parameterRegistry.add({"vsync"}, &appInfo.vSync);
  parameterRegistry.add({"benchmark", "Enable benchmarking: scripted sequences, no vsync, minimal UI"},
                        &benchmarkOptions.enabled);
  parameterRegistry.add({"vvl", "Activate Vulkan Validation Layer"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"logLevel", "Log level: [Stats:1, Info:3, Warning:4, Error:5]"}, reinterpret_cast<int*>(&logLevel));
  parameterRegistry.add({"logShow", "Show extra log info (bitset): [0:None, 1:Time, 2:Level]"}, reinterpret_cast<int*>(&logShow));
  parameterRegistry.add({"device", "force a vulkan device via index into the device list"}, &vkSetup.forceGPU);
  parameterRegistry.add({"vsyncOffMode", "Preferred VSync Off mode: [0:Immediate, 1:Mailbox, 2:FIFO, 3:FIFO Relax]"},
                        reinterpret_cast<int*>(&appInfo.preferredVsyncOffMode));
  parameterRegistry.add({"floatingWindows", "Allow dock windows to be separate windows"}, &appInfo.hasUndockableViewport, true);

  // Don't show the profiler by default
  auto profilerSettings  = std::make_shared<nvapp::ElementProfiler::ViewSettings>();
  profilerSettings->show = false;

  // Create renderer early so it can register CLI/benchmark parameters
  auto elemGltfRenderer = std::make_shared<GltfRenderer>(&parameterRegistry, &cli, benchmarkOptions);

  sequencerInfo.registerScriptParameters(parameterRegistry, cli);
  sequencerInfo.postCallbacks.emplace_back(
      [&](const nvutils::ParameterSequencer::State& state) { elemGltfRenderer->benchmarkAdvance(state); });

  // Adding the parameter registry to the command line parser and parsing arguments
  cli.add(parameterRegistry);
  cli.parse(argc, argv);
  cli.setVerbose(benchmarkOptions.enabled);

  if(appInfo.headless)
  {
    elemGltfRenderer->alignMaxFramesForHeadless(appInfo.headlessFrameCount);
  }

  if(benchmarkOptions.enabled)
  {
    logLevel = nvutils::Logger::LogLevel::eSTATS;
    if(!sequencerInfo.hasScript())
    {
      LOGE("Benchmark mode requires --sequencefile or --sequencestring\n");
      return -1;
    }
  }

  // Using the command line parameters
  logger.setMinimumLogLevel(logLevel);
  logger.setShowFlags(logShow);

  std::shared_ptr<nvapp::ElementSequencer> elemSequencer;
  if(sequencerInfo.hasScript())
  {
    elemSequencer = std::make_shared<nvapp::ElementSequencer>(sequencerInfo);
  }

  auto elemGpuMonitor = std::make_shared<nvgpu_monitor::ElementGpuMonitor>();
  auto elemProfiler   = std::make_shared<nvapp::ElementProfiler>(&g_profilerManager, profilerSettings);
  auto elemLogger     = std::make_shared<nvapp::ElementLogger>(false);

#ifdef USE_DBG_PRINTF
  auto elemDbgPrintf = std::make_shared<nvapp::ElementDbgPrintf>();
#endif

  if(!benchmarkOptions.enabled)
  {
    elemLogger->setLevelFilter(nvapp::ElementLogger::eBitERROR | nvapp::ElementLogger::eBitWARNING | nvapp::ElementLogger::eBitINFO);

    nvutils::Logger::getInstance().setLogCallback([elemLogger](nvutils::Logger::LogLevel logLevelCb, const std::string& str) {
      elemLogger->addLog(logLevelCb, "%s", str.c_str());
    });
  }

  // Extension feature needed.
  // clang-format off
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR computeDerivativesFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
  VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR baryFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  VkPhysicalDeviceNestedCommandBufferFeaturesEXT nestedCmdFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_FEATURES_EXT};
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV reorderFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV};
  // clang-format on

  // Requesting the extensions and features needed
  vkSetup.instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
  vkSetup.deviceExtensions   = {
      {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
      {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
      {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature},
      {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature},
      {VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature},
      {VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, &computeDerivativesFeature},
      {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures},
      {VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &baryFeatures},
      {VK_EXT_NESTED_COMMAND_BUFFER_EXTENSION_NAME, &nestedCmdFeature},
      {VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, &reorderFeature, false},
  };

  // Only request the graphics queue
  vkSetup.queues = {{VK_QUEUE_GRAPHICS_BIT}};

  // If not headless, add the surface extensions for both instance and device (i.e swapchain)
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions, &vkSetup.deviceExtensions);
  }

#ifdef USE_OPTIX_DENOISER
  // Instance extensions
  vkSetup.instanceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  vkSetup.instanceExtensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

  // Device extensions
  vkSetup.deviceExtensions.emplace_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#ifdef WIN32
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif  // WIN32
#endif  // USE_OPTIX_DENOISER


#if defined(USE_NSIGHT_AFTERMATH)
  // Adding the Aftermath extension to the device and initialize the Aftermath
  auto& aftermath = AftermathCrashTracker::getInstance();
  aftermath.initialize();
  aftermath.addExtensions(vkSetup.deviceExtensions);
  // The callback function is called when a validation error is triggered. This will wait to give time to dump the GPU crash.
  nvvk::CheckError::getInstance().setCallbackFunction([&](VkResult result) { aftermath.errorCallback(result); });
#endif

  nvvk::ValidationSettings validation{};
#ifdef USE_DBG_PRINTF
  validation.setPreset(nvvk::ValidationSettings::LayerPresets::eDebugPrintf);
  validation.printf_to_stdout = VK_FALSE;  // Different from default to capture output

#else
  validation.setPreset(nvvk::ValidationSettings::LayerPresets::eStandard);
#endif

  // Optimize VVL for fast pipeline creation while keeping critical validation
  if(vkSetup.enableValidationLayers)
  {
    // Disable expensive shader validation during pipeline creation
    validation.check_shaders         = VK_FALSE;
    validation.check_shaders_caching = VK_FALSE;
  }

  vkSetup.instanceCreateInfoExt = validation.buildPNextChain();


#if defined(USE_DLSS)
  // Probe BOTH DLSS Ray Reconstruction (RR) and DLSS Super Resolution (SR) extension
  // requirements, union by extension name (RR and SR share most extensions) so we ask for each
  // extension only once when creating the Vulkan instance + device. Track per-feature query
  // success so we can split availability into dlssRrHardwareAvailable / dlssSrHardwareAvailable.
  static std::vector<VkExtensionProperties> extraInstanceExtRr;
  static std::vector<VkExtensionProperties> extraInstanceExtSr;
  NgxContext::getRequiredInstanceExtensions(NVSDK_NGX_Feature_RayReconstruction, {}, extraInstanceExtRr);
  NgxContext::getRequiredInstanceExtensions(NVSDK_NGX_Feature_SuperSampling, {}, extraInstanceExtSr);
  {
    std::unordered_set<std::string> seen;
    auto                            appendOnce = [&seen, &vkSetup](const VkExtensionProperties& ext) {
      if(seen.insert(ext.extensionName).second)
        vkSetup.instanceExtensions.push_back(ext.extensionName);
    };
    for(auto& ext : extraInstanceExtRr)
      appendOnce(ext);
    for(auto& ext : extraInstanceExtSr)
      appendOnce(ext);
  }

  // After selecting the device, we also request extensions DLSS needs using
  // nvvk::Context's callback. NGX may report that one or both features are not
  // available on this GPU; we keep the device alive either way and just track
  // per-feature query success.
  static bool                               dlssRrQueryOk = false;
  static bool                               dlssSrQueryOk = false;
  static std::vector<VkExtensionProperties> extraDeviceExtRr;
  static std::vector<VkExtensionProperties> extraDeviceExtSr;
  vkSetup.postSelectPhysicalDeviceCallback = [](VkInstance instance, VkPhysicalDevice physicalDevice, nvvk::ContextInitInfo& vkSetup) {
    dlssRrQueryOk = (NVSDK_NGX_Result_Success
                     == NgxContext::getRequiredDeviceExtensions(NVSDK_NGX_Feature_RayReconstruction, {}, instance,
                                                                physicalDevice, extraDeviceExtRr));
    dlssSrQueryOk = (NVSDK_NGX_Result_Success
                     == NgxContext::getRequiredDeviceExtensions(NVSDK_NGX_Feature_SuperSampling, {}, instance,
                                                                physicalDevice, extraDeviceExtSr));

    std::unordered_set<std::string> seen;
    auto                            appendOnce = [&seen, &vkSetup](const VkExtensionProperties& ext) {
      if(seen.insert(ext.extensionName).second)
        vkSetup.deviceExtensions.push_back({.extensionName = ext.extensionName, .required = false, .specVersion = ext.specVersion});
    };
    if(dlssRrQueryOk)
      for(auto& ext : extraDeviceExtRr)
        appendOnce(ext);
    if(dlssSrQueryOk)
      for(auto& ext : extraDeviceExtSr)
        appendOnce(ext);

    return true;  // Continue with this device (even if DLSS is not available)
  };
#endif


  // Create the Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Failed to initialize Vulkan context!");
    return -1;
  }

  // Check per-feature extension enablement after vkContext.init: the NGX query may have
  // succeeded but the loader could still have dropped optional extensions on the floor.
  bool dlssRrHardwareAvailable = false;
  bool dlssSrHardwareAvailable = false;
#if defined(USE_DLSS)
  dlssRrHardwareAvailable = dlssRrQueryOk;
  for(auto& ext : extraDeviceExtRr)
    dlssRrHardwareAvailable &= vkContext.hasExtensionEnabled(ext.extensionName);

  dlssSrHardwareAvailable = dlssSrQueryOk;
  for(auto& ext : extraDeviceExtSr)
    dlssSrHardwareAvailable &= vkContext.hasExtensionEnabled(ext.extensionName);

  if(!dlssRrHardwareAvailable)
    LOGW("DLSS-RR: Required Vulkan extensions not available - Ray Reconstruction will be disabled\n");
  if(!dlssSrHardwareAvailable)
    LOGW("DLSS-SR: Required Vulkan extensions not available - DLSS-SR will be disabled\n");

  elemGltfRenderer->setDlssHardwareAvailability(dlssRrHardwareAvailable, dlssSrHardwareAvailable);
#endif


  // Application information
  appInfo.name = fmt::format("{} {} ({})", nvutils::getExecutablePath().stem().string(), APP_VERSION_STRING, "Slang");
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();
  appInfo.useMenu        = !benchmarkOptions.enabled;

  // Setting up the layout of the application
  appInfo.dockSetup = [](ImGuiID viewportID) {
    // Left side panel container
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.25F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Camera", settingID);
    ImGui::DockBuilderDockWindow("Settings", settingID);

    // Under Setting
    ImGuiID tonemapID = ImGui::DockBuilderSplitNode(settingID, ImGuiDir_Down, 0.35F, nullptr, &settingID);
    ImGui::DockBuilderDockWindow("Tonemapper", tonemapID);
    ImGui::DockBuilderDockWindow("Environment", tonemapID);

    // Right side: Scene Browser, Inspector (bottom)
    ImGuiID sceneBrowserID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.25F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Scene Browser", sceneBrowserID);
    ImGuiID inspectorID = ImGui::DockBuilderSplitNode(sceneBrowserID, ImGuiDir_Down, 0.35F, nullptr, &sceneBrowserID);
    ImGui::DockBuilderDockWindow("Inspector", inspectorID);

    // bottom panel container
    ImGuiID logID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.35F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Log", logID);
    ImGuiID monitorID = ImGui::DockBuilderSplitNode(logID, ImGuiDir_Right, 0.35F, nullptr, &logID);
    ImGui::DockBuilderDockWindow("NVML Monitor", monitorID);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(logID, ImGuiDir_Right, 0.33F, nullptr, &logID);
    ImGui::DockBuilderDockWindow("Profiler", profilerID);
    ImGuiID memStatsID = ImGui::DockBuilderSplitNode(logID, ImGuiDir_Right, 0.33F, nullptr, &logID);
    ImGui::DockBuilderDockWindow("Memory Statistics", memStatsID);
    ImGui::DockBuilderDockWindow("Statistics", memStatsID);
  };

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // Set the window icon
  if(!appInfo.headless)
  {
    setWindowIcon(app.getWindowHandle());
  }

  elemGltfRenderer->registerRecentFilesHandler();

  if(elemSequencer)
  {
    app.addElement(elemSequencer);
  }
  app.addElement(elemGltfRenderer);
  if(!benchmarkOptions.enabled)
  {
    app.addElement(elemLogger);
    app.addElement(elemGpuMonitor);
    app.addElement(elemProfiler);
#ifdef USE_DBG_PRINTF
    app.addElement(elemDbgPrintf);
#endif
  }

  if(benchmarkOptions.enabled)
  {
    app.setVsync(false);
  }

  // Loading the scene and the HDR
#ifdef USE_DEFAULT_SCENE
  // If USE_DEFAULT_SCENE is enabled and no scene file is specified, load the default scene
  if(sceneFilename.empty())
  {
    sceneFilename = "shader_ball.gltf";
  }
#endif

  // Load a scene if specified, otherwise the application starts empty
  if(!sceneFilename.empty())
  {
    elemGltfRenderer->createScene(sceneFilename);
  }
  // Load an HDR if specified
  if(!hdrFilename.empty())
  {
    elemGltfRenderer->createHDR(hdrFilename);
  }

  // In headless mode, prevent ImGui from writing the .ini back to disk:
  if(appInfo.headless)
  {
    ImGui::GetIO().IniFilename = nullptr;
  }

  app.run();
  app.deinit();

  // Clear callbacks before scope ends to avoid dangling references
  nvutils::Logger::getInstance().setLogCallback(nullptr);
#if defined(USE_NSIGHT_AFTERMATH)
  nvvk::CheckError::getInstance().setCallbackFunction(nullptr);
#endif

  // Deinit Vulkan context
  vkContext.deinit();
}
