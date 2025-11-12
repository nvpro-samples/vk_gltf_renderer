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

// #define USE_NSIGHT_AFTERMATH

#include <fmt/format.h>
#include <stb/stb_image.h>
#include <GLFW/glfw3.h>
#undef APIENTRY

#include <nvaftermath/aftermath.hpp>
#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvgpu_monitor/elem_gpu_monitor.hpp>
#include <nvgui/camera.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/validation_settings.hpp>

#include "renderer.hpp"
#include "doc/app_icon_png.h"

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
  nvutils::Logger::ShowFlags   logShow  = nvutils::Logger::ShowBits::eSHOW_NONE;

  // Global variables
  std::filesystem::path sceneFilename{};  // "shader_ball.gltf"};  // Default scene
  std::filesystem::path hdrFilename{};    // "env3.hdr"};         // Default HDR

  // Application defaults overrides
  appInfo.preferredVsyncOffMode = VK_PRESENT_MODE_MAILBOX_KHR;

  // Command line parameters registration
  nvutils::ParameterRegistry parameterRegistry;
  parameterRegistry.add({"scenefile", "Input scene filename"}, {".gltf"}, &sceneFilename);
  parameterRegistry.add({"hdrfile", "Input HDR filename"}, {".hdr"}, &hdrFilename);
  parameterRegistry.addVector({"size", "Size of the window to be created", "s"}, &appInfo.windowSize);
  parameterRegistry.add({"headless"}, &appInfo.headless, true);
  parameterRegistry.add({"frames", "Number of frames to run in headless mode"}, &appInfo.headlessFrameCount);
  parameterRegistry.add({"vsync"}, &appInfo.vSync);
  parameterRegistry.add({"vvl", "Activate Vulkan Validation Layer"}, &vkSetup.enableValidationLayers);
  parameterRegistry.add({"logLevel", "Log level: [Info:0, Warning:1, Error:2]"}, reinterpret_cast<int*>(&logLevel));
  parameterRegistry.add({"logShow", "Show extra log info (bitset): [0:None, 1:Time, 2:Level]"}, reinterpret_cast<int*>(&logShow));
  parameterRegistry.add({"device", "force a vulkan device via index into the device list"}, &vkSetup.forceGPU);
  parameterRegistry.add({"vsyncOffMode", "Preferred VSync Off mode: [0:Immediate, 1:Mailbox, 2:FIFO, 3:FIFO Relax]"},
                        reinterpret_cast<int*>(&appInfo.preferredVsyncOffMode));
  parameterRegistry.add({"floatingWindows", "Allow dock windows to be separate windows"}, &appInfo.hasUndockableViewport, true);


  // Don't show the profiler by default
  auto profilerSettings  = std::make_shared<nvapp::ElementProfiler::ViewSettings>();
  profilerSettings->show = false;


  // Create all application elements
  auto elemCamera       = std::make_shared<nvapp::ElementCamera>();
  auto elemGltfRenderer = std::make_shared<GltfRenderer>(&parameterRegistry);
  auto elemGpuMonitor   = std::make_shared<nvgpu_monitor::ElementGpuMonitor>();
  auto elemProfiler     = std::make_shared<nvapp::ElementProfiler>(&g_profilerManager, profilerSettings);
  auto elemLogger       = std::make_shared<nvapp::ElementLogger>(false);


  // Adding an element logger (UI), where all log will be redirected to
  elemLogger->setLevelFilter(nvapp::ElementLogger::eBitERROR | nvapp::ElementLogger::eBitWARNING | nvapp::ElementLogger::eBitINFO);

  // The logger will redirect the log to the Element Logger, to be displayed in the UI
  nvutils::Logger::getInstance().setLogCallback([elemLogger](nvutils::Logger::LogLevel logLevel, const std::string& str) {
    elemLogger->addLog(logLevel, "%s", str.c_str());
  });

  // Adding the parameter registry to the command line parser
  nvutils::ParameterParser cli(nvutils::getExecutablePath().stem().string());
  cli.add(parameterRegistry);
  cli.parse(argc, argv);

  // Using the command line parameters
  logger.setMinimumLogLevel(logLevel);
  logger.setShowFlags(logShow);


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
  validation.setPreset(nvvk::ValidationSettings::LayerPresets::eStandard);
  validation.printf_to_stdout = VK_TRUE;

  // Optimize VVL for fast pipeline creation while keeping critical validation
  if(vkSetup.enableValidationLayers)
  {
    // Disable expensive shader validation during pipeline creation
    validation.check_shaders         = VK_FALSE;
    validation.check_shaders_caching = VK_FALSE;
  }

  vkSetup.instanceCreateInfoExt = validation.buildPNextChain();


#if USE_DLSS
  // Adding the DLSS extensions to the instance
  static std::vector<VkExtensionProperties> extraInstanceExtensions;
  DlssRayReconstruction::getRequiredInstanceExtensions({}, extraInstanceExtensions);
  for(auto& ext : extraInstanceExtensions)
  {
    vkSetup.instanceExtensions.emplace_back(ext.extensionName);
  }

  // After selecting the device, we also request extensions DLSS needs using
  // nvvk::Context's callback.
  // Note at this stage NGX can report that DLSS is not available, so we need
  // to handle that.
  static bool                               dlssQueryExtensionsOk = false;
  static std::vector<VkExtensionProperties> extraDeviceExtensions;
  vkSetup.postSelectPhysicalDeviceCallback = [](VkInstance instance, VkPhysicalDevice physicalDevice, nvvk::ContextInitInfo& vkSetup) {
    const NVSDK_NGX_Result result =
        DlssRayReconstruction::getRequiredDeviceExtensions({}, instance, physicalDevice, extraDeviceExtensions);
    if(NVSDK_NGX_Result_Success == result)
    {
      dlssQueryExtensionsOk = true;
      for(auto& ext : extraDeviceExtensions)
      {
        vkSetup.deviceExtensions.push_back({.extensionName = ext.extensionName, .required = false, .specVersion = ext.specVersion});
      }
    }

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

  // Check that DLSS extensions are enabled
  bool dlssHardwareAvailable = false;  // Default: DLSS not available
#if USE_DLSS
  dlssHardwareAvailable = dlssQueryExtensionsOk;
  for(auto& dlssExt : extraDeviceExtensions)
  {
    dlssHardwareAvailable &= vkContext.hasExtensionEnabled(dlssExt.extensionName);
  }

  if(!dlssHardwareAvailable)
  {
    LOGW("DLSS: Required Vulkan extensions not available - DLSS will be disabled\n");
  }

  // Set DLSS hardware availability based on extension check
  elemGltfRenderer->setDlssHardwareAvailability(dlssHardwareAvailable);
#endif


  // Application information
  appInfo.name           = fmt::format("{} ({})", nvutils::getExecutablePath().stem().string(), "Slang");
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();


  // Setting up the layout of the application
  appInfo.dockSetup = [](ImGuiID viewportID) {
    // right side panel container
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.25F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);
    ImGui::DockBuilderDockWindow("Scene Graph", settingID);
    ImGui::DockBuilderDockWindow("Camera", settingID);

    ImGuiID propID = ImGui::DockBuilderSplitNode(settingID, ImGuiDir_Down, 0.35F, nullptr, &settingID);
    ImGui::DockBuilderDockWindow("Properties", propID);

    // bottom panel container
    ImGuiID logID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Down, 0.35F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Log", logID);
    ImGuiID monitorID = ImGui::DockBuilderSplitNode(logID, ImGuiDir_Right, 0.35F, nullptr, &logID);
    ImGui::DockBuilderDockWindow("NVML Monitor", monitorID);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(logID, ImGuiDir_Right, 0.33F, nullptr, &logID);
    ImGui::DockBuilderDockWindow("Profiler", profilerID);
  };

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // Set the window icon
  if(!appInfo.headless)
  {
    setWindowIcon(app.getWindowHandle());
  }

  // Set the camera manipulator to elements that need it.
  auto cameraManip = elemGltfRenderer->getCameraManipulator();
  elemCamera->setCameraManipulator(cameraManip);
  elemGltfRenderer->registerRecentFilesHandler();

  app.addElement(elemCamera);
  app.addElement(elemGltfRenderer);
  app.addElement(elemLogger);
  app.addElement(elemGpuMonitor);
  app.addElement(elemProfiler);

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