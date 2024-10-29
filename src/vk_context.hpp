/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2024 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include <csignal>

#include <vulkan/vulkan_core.h>
#include <vulkan/vk_enum_string_helper.h>

#include <stdexcept>
#include <vector>
#include <cstring>
#include <unordered_set>

#include "nvh/nvprint.hpp"
#include "nvh/timesampler.hpp"
#include "nvvk/error_vk.hpp"
#include "nvvkhl/application.hpp"  // For QueueInfo

//--------------------------------------------------------------------------------------------------
// CATCHING VULKAN ERRORS
//--------------------------------------------------------------------------------------------------
static VKAPI_ATTR VkBool32 VKAPI_CALL VkContextDebugReport(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                                           VkDebugUtilsMessageTypeFlagsEXT             messageType,
                                                           const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                                           void*                                       userData)
{
  int level = messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT   ? LOGLEVEL_ERROR :
              messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ? LOGLEVEL_WARNING :
                                                                                  LOGLEVEL_INFO;
  nvprintfLevel(level, "%s\n", callbackData->pMessage);
  for(uint32_t count = 0; count < callbackData->objectCount; count++)
  {
    LOGI("Object[%d] \n\t- Type %s\n\t- Value %p\n\t- Name %s\n", count,
         string_VkObjectType(callbackData->pObjects[count].objectType), callbackData->pObjects[count].objectHandle,
         callbackData->pObjects[count].pObjectName);
  }
  if(messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
  {
#if defined(_MSVC_LANG)
    __debugbreak();  // If you break here, there is a Vulkan error that needs to be fixed
                     // To ignore specific message, insert it to ValidationLayerInfo.message_id_filter
                     // ex: "MessageID = 0x30b6e267"  ->  ValidationLayerInfo.message_id_filter.insert(0x30b6e267);
#elif defined(LINUX)
    raise(SIGTRAP);
#endif
  }

  return VK_FALSE;
}

// Vulkan Queue information
//struct QueueInfo
//{
//  uint32_t familyIndex = ~0U;
//  uint32_t queueIndex  = ~0U;
//  VkQueue  queue       = VK_NULL_HANDLE;
//};
using QueueInfo = nvvkhl::ApplicationQueue;

// Struct to hold an extension and its corresponding feature
struct ExtensionFeaturePair
{
  const char* extensionName    = nullptr;
  void*       feature          = nullptr;  // [optional] Pointer to the feature structure for the extension
  bool        required         = true;     // If the extension is required
  uint32_t    specVersion      = 0;        // [optional] Spec version of the extension, this version or higher
  bool        exactSpecVersion = false;    // [optional] If true, the spec version must match exactly
};


// Forward declarations
std::vector<VkExtensionProperties> getDeviceExtensions(VkPhysicalDevice physicalDevice);
std::string                        getVersionString(uint32_t version);
std::string                        getVendorName(uint32_t vendorID);
std::string                        getDeviceType(uint32_t deviceType);
void                               printVulkanVersion();
void                               printInstanceLayers();
void                               printInstanceExtensions(const std::vector<const char*> ext = {});
void printDeviceExtensions(VkPhysicalDevice physicalDevice, const std::vector<ExtensionFeaturePair> ext = {});
void printPhysicalDeviceProperties(const VkPhysicalDeviceProperties& properties);
void printGpus(VkInstance instance, VkPhysicalDevice usedGpu);

// Specific to the creation of Vulkan context
struct VkContextSettings
{
  std::vector<const char*> instanceExtensions = {};  // Instance extensions: VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
  std::vector<ExtensionFeaturePair> deviceExtensions = {};  // Device extensions: {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}, {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature}, {OTHER}}
  std::vector<VkQueueFlags> queues = {VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT};  // All desired queues, first is always GTC
  void*       instanceCreateInfoExt = nullptr;      // Instance create info extension (ex: VkLayerSettingsCreateInfoEXT)
  const char* applicationName       = "No Engine";  // Application name
  uint32_t    apiVersion            = VK_API_VERSION_1_3;  // Vulkan API version
  VkAllocationCallbacks* alloc      = nullptr;             // Allocation callbacks
  bool    enableAllFeatures         = true;  // If true, pull all capability of `features` from the physical device
  int32_t forceGPU                  = -1;    // If != -1, use GPU index.
#if NDEBUG
  bool enableValidationLayers = false;  // Disable validation layers in release
  bool verbose                = false;
#else
  bool enableValidationLayers = true;  // Enable validation layers
  bool verbose                = true;
#endif
};


//--------------------------------------------------------------------------------------------------
// Simple class to handle the Vulkan context creation
class VulkanContext
{
public:
  VulkanContext() = default;
  VulkanContext(const VkContextSettings& settings) { init(settings); }
  ~VulkanContext() { deinit(); }

  VkInstance             getInstance() const { return m_instance; }
  VkDevice               getDevice() const { return m_device; }
  VkPhysicalDevice       getPhysicalDevice() const { return m_physicalDevice; }
  QueueInfo              getQueueInfo(uint32_t index) const { return m_queueInfos[index]; }
  std::vector<QueueInfo> getQueueInfos() const { return m_queueInfos; }
  bool                   isValid() const
  {
    return m_instance != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE && m_physicalDevice != VK_NULL_HANDLE;
  }

  void init(const VkContextSettings& settings = VkContextSettings())
  {
    m_settings = settings;
    {
      nvh::ScopedTimer st("Creating Vulkan Context");
      createInstance();
      selectPhysicalDevice();
      createDevice();
    }
    if(m_settings.verbose)
    {
      printVulkanVersion();
      printInstanceLayers();
      printInstanceExtensions(m_settings.instanceExtensions);
      printDeviceExtensions(m_physicalDevice, m_settings.deviceExtensions);
      printGpus(m_instance, m_physicalDevice);
      LOGI("_________________________________________________\n");
    }
  }

  void deinit()
  {
    if(m_dbgMessenger)
    {
      auto vkDestroyDebugUtilsMessengerEXT =
          (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
      assert(vkDestroyDebugUtilsMessengerEXT != nullptr);
      vkDestroyDebugUtilsMessengerEXT(m_instance, m_dbgMessenger, m_settings.alloc);
      m_dbgMessenger = nullptr;
    }
    vkDestroyDevice(m_device, m_settings.alloc);
    vkDestroyInstance(m_instance, m_settings.alloc);
    m_device   = VK_NULL_HANDLE;
    m_instance = VK_NULL_HANDLE;
  }


private:
  VkContextSettings m_settings{};  // What was used to create the information

  VkInstance       m_instance{};
  VkDevice         m_device{};
  VkPhysicalDevice m_physicalDevice{};

  // For device creation
  VkPhysicalDeviceFeatures2        m_deviceFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  VkPhysicalDeviceVulkan11Features features11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};

  // For Queue creation
  std::vector<VkQueueFlags>            m_desiredQueues{};
  std::vector<VkDeviceQueueCreateInfo> m_queueCreateInfos{};
  std::vector<QueueInfo>               m_queueInfos{};
  std::vector<std::vector<float>>      m_queuePriorities{};  // Store priorities here

  // Callback for debug messages
  VkDebugUtilsMessengerEXT m_dbgMessenger = VK_NULL_HANDLE;

  void createInstance()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    VkApplicationInfo appInfo{
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName   = m_settings.applicationName,
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName        = "My Engine",
        .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion         = m_settings.apiVersion,
    };

    std::vector<const char*> layers;
    if(m_settings.enableValidationLayers)
    {
      layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    VkInstanceCreateInfo createInfo{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext                   = m_settings.instanceCreateInfoExt,
        .pApplicationInfo        = &appInfo,
        .enabledLayerCount       = uint32_t(layers.size()),
        .ppEnabledLayerNames     = layers.data(),
        .enabledExtensionCount   = uint32_t(m_settings.instanceExtensions.size()),
        .ppEnabledExtensionNames = m_settings.instanceExtensions.data(),
    };

    VkResult result = vkCreateInstance(&createInfo, m_settings.alloc, &m_instance);
    if(result != VK_SUCCESS)
    {
      assert(!"failed to create instance!");
      return;
    }

    if(m_settings.enableValidationLayers)
    {
      auto vkCreateDebugUtilsMessengerEXT =
          (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
      if(vkCreateDebugUtilsMessengerEXT)
      {
        assert(vkCreateDebugUtilsMessengerEXT != nullptr);
        VkDebugUtilsMessengerCreateInfoEXT dbg_messenger_create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        dbg_messenger_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT   // GPU info, bug
                                                    | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;  // Invalid usage
        dbg_messenger_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT      // Violation of spec
                                                | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;  // Non-optimal use
        dbg_messenger_create_info.pfnUserCallback = VkContextDebugReport;
        NVVK_CHECK(vkCreateDebugUtilsMessengerEXT(m_instance, &dbg_messenger_create_info, nullptr, &m_dbgMessenger));
      }
      else
      {
        LOGW("\nMissing VK_EXT_DEBUG_UTILS extension, cannot use vkCreateDebugUtilsMessengerEXT for validation layers.\n");
      }
    }
  }

  void selectPhysicalDevice()
  {
    if(m_instance == VK_NULL_HANDLE)
      return;

    // nvh::ScopedTimer st(std::string(__FUNCTION__) + "\n");
    uint32_t deviceCount = 0;
    NVVK_CHECK(vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr));
    if(deviceCount == 0)
      assert(!"Failed to find GPUs with Vulkan support!");
    std::vector<VkPhysicalDevice> gpus(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, gpus.data());

    // Find the discrete GPU if present, or use first one available.
    if((m_settings.forceGPU == -1) || (m_settings.forceGPU >= int(deviceCount)))
    {
      m_physicalDevice = gpus[0];
      for(VkPhysicalDevice& device : gpus)
      {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        if(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
          m_physicalDevice = device;
        }
      }
    }
    else
    {
      // Using specified GPU
      m_physicalDevice = gpus[m_settings.forceGPU];
    }

    {  // Check for available Vulkan version
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(m_physicalDevice, &properties);
      uint32_t apiVersion = properties.apiVersion;
      if((VK_VERSION_MAJOR(apiVersion) < VK_VERSION_MAJOR(m_settings.apiVersion))
         || (VK_VERSION_MINOR(apiVersion) < VK_VERSION_MINOR(m_settings.apiVersion)))
      {
        LOGE("Requested Vulkan version (%d.%d) is higher than available version (%d.%d).\n", VK_VERSION_MAJOR(m_settings.apiVersion),
             VK_VERSION_MINOR(m_settings.apiVersion), VK_VERSION_MAJOR(apiVersion), VK_VERSION_MINOR(apiVersion));
        m_physicalDevice = {};
        return;
      }
    }

    // Query the physical device features
    m_deviceFeatures.pNext = &features11;
    if(m_settings.apiVersion >= VK_API_VERSION_1_2)
      features11.pNext = &features12;
    if(m_settings.apiVersion >= VK_API_VERSION_1_3)
      features12.pNext = &features13;
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &m_deviceFeatures);

    // Find the queues that we need
    m_desiredQueues = m_settings.queues;
    if(!findQueueFamilies())
    {
      m_physicalDevice = {};
      return;
    }

    // Filter the available extensions otherwise the device creation will fail
    std::vector<ExtensionFeaturePair> filteredExtensions;
    bool allFound = filterAvailableExtensions(getDeviceExtensions(m_physicalDevice), m_settings.deviceExtensions, filteredExtensions);
    if(!allFound)
    {
      m_physicalDevice = {};
      return;
    }
    m_settings.deviceExtensions = filteredExtensions;
  }

  void createDevice()
  {
    if(m_physicalDevice == VK_NULL_HANDLE)
      return;

    // nvh::ScopedTimer st(__FUNCTION__);
    // Chain all custom features to the pNext chain of m_deviceFeatures
    for(const auto& extension : m_settings.deviceExtensions)
    {
      if(extension.feature)
        prependFeatures(reinterpret_cast<VkBaseOutStructure*>(&m_deviceFeatures),
                        reinterpret_cast<VkBaseOutStructure*>(extension.feature));
    }
    // Activate features on request
    if(m_settings.enableAllFeatures)
      vkGetPhysicalDeviceFeatures2(m_physicalDevice, &m_deviceFeatures);

    // List of extensions to enable
    std::vector<const char*> enabledExtensions;
    for(const auto& ext : m_settings.deviceExtensions)
      enabledExtensions.push_back(ext.extensionName);
    VkDeviceCreateInfo createInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &m_deviceFeatures,
        .queueCreateInfoCount    = uint32_t(m_queueCreateInfos.size()),
        .pQueueCreateInfos       = m_queueCreateInfos.data(),
        .enabledExtensionCount   = static_cast<uint32_t>(enabledExtensions.size()),
        .ppEnabledExtensionNames = enabledExtensions.data(),
    };

    if(vkCreateDevice(m_physicalDevice, &createInfo, m_settings.alloc, &m_device) != VK_SUCCESS)
      assert(!"failed to create logical device!");

    for(auto& queue : m_queueInfos)
      vkGetDeviceQueue(m_device, queue.familyIndex, queue.queueIndex, &queue.queue);
  }

  void prependFeatures(VkBaseOutStructure* baseStruct, VkBaseOutStructure* prependStruct)
  {
    // Traverse the pNext chain of prependStruct to find the last element
    VkBaseOutStructure* lastPrepend = prependStruct;
    while(lastPrepend->pNext)
    {
      lastPrepend = reinterpret_cast<VkBaseOutStructure*>(lastPrepend->pNext);
    }
    lastPrepend->pNext = baseStruct->pNext;  // Append the original pNext chain of baseStruct to the end of prependStruct's chain
    baseStruct->pNext = prependStruct;  // Prepend the prependStruct to the baseStruct
  }

  bool findQueueFamilies()
  {
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, queueFamilies.data());

    std::unordered_map<uint32_t, uint32_t> queueFamilyUsage;
    for(uint32_t i = 0; i < queueFamilyCount; ++i)
    {
      queueFamilyUsage[i] = 0;
    }

    for(size_t i = 0; i < m_desiredQueues.size(); ++i)
    {
      bool found = false;
      for(uint32_t j = 0; j < queueFamilyCount; ++j)
      {
        // Check for an exact match and unused queue family
        // Avoid queue family with VK_QUEUE_GRAPHICS_BIT if not needed
        if((queueFamilies[j].queueFlags & m_desiredQueues[i]) == m_desiredQueues[i] && queueFamilyUsage[j] == 0
           && ((m_desiredQueues[i] & VK_QUEUE_GRAPHICS_BIT) || !(queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
        {
          m_queueInfos.push_back({j, queueFamilyUsage[j]});
          queueFamilyUsage[j]++;
          found = true;
          break;
        }
      }

      if(!found)
      {
        for(uint32_t j = 0; j < queueFamilyCount; ++j)
        {
          // Check for an exact match and allow reuse if queue count not exceeded
          // Avoid queue family with VK_QUEUE_GRAPHICS_BIT if not needed
          if((queueFamilies[j].queueFlags & m_desiredQueues[i]) == m_desiredQueues[i]
             && queueFamilyUsage[j] < queueFamilies[j].queueCount
             && ((m_desiredQueues[i] & VK_QUEUE_GRAPHICS_BIT) || !(queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
          {
            m_queueInfos.push_back({j, queueFamilyUsage[j]});
            queueFamilyUsage[j]++;
            found = true;
            break;
          }
        }
      }

      if(!found)
      {
        for(uint32_t j = 0; j < queueFamilyCount; ++j)
        {
          // Check for a partial match and allow reuse if queue count not exceeded
          // Avoid queue family with VK_QUEUE_GRAPHICS_BIT if not needed
          if((queueFamilies[j].queueFlags & m_desiredQueues[i]) && queueFamilyUsage[j] < queueFamilies[j].queueCount
             && ((m_desiredQueues[i] & VK_QUEUE_GRAPHICS_BIT) || !(queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
          {
            m_queueInfos.push_back({j, queueFamilyUsage[j]});
            queueFamilyUsage[j]++;
            found = true;
            break;
          }
        }
      }

      if(!found)
      {
        for(uint32_t j = 0; j < queueFamilyCount; ++j)
        {
          // Check for a partial match and allow reuse if queue count not exceeded
          if((queueFamilies[j].queueFlags & m_desiredQueues[i]) && queueFamilyUsage[j] < queueFamilies[j].queueCount)
          {
            m_queueInfos.push_back({j, queueFamilyUsage[j]});
            queueFamilyUsage[j]++;
            found = true;
            break;
          }
        }
      }

      if(!found)
      {
        // If no suitable queue family is found, assert a failure
        LOGE("Failed to find a suitable queue family!\n");
        return false;
      }
    }

    for(const auto& usage : queueFamilyUsage)
    {
      if(usage.second > 0)
      {
        m_queuePriorities.emplace_back(usage.second, 1.0f);  // Same priority for all queues in a family
        m_queueCreateInfos.push_back({VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, usage.first, usage.second,
                                      m_queuePriorities.back().data()});
      }
    }
    return true;
  }

  // Filters available Vulkan extensions based on desired extensions and their specifications.
  bool filterAvailableExtensions(const std::vector<VkExtensionProperties>& availableExtensions,
                                 const std::vector<ExtensionFeaturePair>&  desiredExtensions,
                                 std::vector<ExtensionFeaturePair>&        filteredExtensions)
  {
    bool allFound = true;

    // Create a map for quick lookup of available extensions and their versions
    std::unordered_map<std::string, uint32_t> availableExtensionsMap;
    for(const auto& ext : availableExtensions)
    {
      availableExtensionsMap[ext.extensionName] = ext.specVersion;
    }

    // Iterate through all desired extensions
    for(const auto& desiredExtension : desiredExtensions)
    {
      auto     it           = availableExtensionsMap.find(desiredExtension.extensionName);
      bool     found        = it != availableExtensionsMap.end();
      uint32_t specVersion  = found ? it->second : 0;
      bool     validVersion = desiredExtension.exactSpecVersion ? desiredExtension.specVersion == specVersion :
                                                                  desiredExtension.specVersion <= specVersion;
      if(found && validVersion)
      {
        filteredExtensions.push_back(desiredExtension);
      }
      else
      {
        std::string versionInfo;
        if(desiredExtension.specVersion != 0 || desiredExtension.exactSpecVersion)
          versionInfo = fmt::format(" (v.{} {} v.{})", specVersion, specVersion ? "==" : ">=", desiredExtension.specVersion);
        if(desiredExtension.required)
          allFound = false;
        nvprintfLevel(desiredExtension.required ? LOGLEVEL_ERROR : LOGLEVEL_WARNING, "Extension not available: %s %s\n",
                      desiredExtension.extensionName, versionInfo.c_str());
      }
    }

    return allFound;
  }
};

inline void printVulkanVersion()
{
  uint32_t version;
  vkEnumerateInstanceVersion(&version);
  LOGI("\n_________________________________________________\n");
  LOGI("Vulkan Version:  %d.%d.%d\n", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version), VK_VERSION_PATCH(version));
}

inline std::vector<VkExtensionProperties> getDeviceExtensions(VkPhysicalDevice physicalDevice)
{
  uint32_t                           count;
  std::vector<VkExtensionProperties> extensionProperties;
  NVVK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr));
  extensionProperties.resize(count);
  NVVK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensionProperties.data()));
  extensionProperties.resize(std::min(extensionProperties.size(), size_t(count)));
  return extensionProperties;
}

inline std::string getVersionString(uint32_t version)
{
  return fmt::format("{}.{}.{}", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version), VK_VERSION_PATCH(version));
}

inline void printInstanceLayers()
{
  uint32_t                       count;
  std::vector<VkLayerProperties> layerProperties;
  NVVK_CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
  layerProperties.resize(count);
  NVVK_CHECK(vkEnumerateInstanceLayerProperties(&count, layerProperties.data()));
  LOGI("_________________________________________________\n");
  LOGI("Available Instance Layers :\n");
  for(auto& it : layerProperties)
  {
    LOGI("%s (v. %d.%d.%d %x) : %s\n", it.layerName, VK_VERSION_MAJOR(it.specVersion), VK_VERSION_MINOR(it.specVersion),
         VK_VERSION_PATCH(it.specVersion), it.implementationVersion, it.description);
  }
}

inline void printInstanceExtensions(const std::vector<const char*> ext)
{
  std::unordered_set<std::string> exist;
  for(auto& e : ext)
    exist.insert(e);

  uint32_t                           count;
  std::vector<VkExtensionProperties> extensionProperties;
  NVVK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
  extensionProperties.resize(count);
  NVVK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &count, extensionProperties.data()));
  LOGI("_________________________________________________\n");
  LOGI("Available Instance Extensions :\n");
  for(const VkExtensionProperties& it : extensionProperties)
  {
    LOGI("[%s] ", (exist.find(it.extensionName) != exist.end()) ? "x" : " ");
    LOGI("%s (v. %d)\n", it.extensionName, it.specVersion);
  }
}

inline void printDeviceExtensions(VkPhysicalDevice physicalDevice, const std::vector<ExtensionFeaturePair> ext)
{
  if(physicalDevice == VK_NULL_HANDLE)
    return;

  std::unordered_set<std::string> exist;
  for(auto& e : ext)
    exist.insert(e.extensionName);

  uint32_t                           count;
  std::vector<VkExtensionProperties> extensionProperties;
  NVVK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr));
  extensionProperties.resize(count);
  NVVK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensionProperties.data()));
  LOGI("_________________________________________________\n");
  LOGI("Available Device Extensions :\n");
  for(const VkExtensionProperties& it : extensionProperties)
  {
    LOGI("[%s] ", (exist.find(it.extensionName) != exist.end()) ? "x" : " ");
    LOGI("%s (v. %d)\n", it.extensionName, it.specVersion);
  }
}

inline std::string getVendorName(uint32_t vendorID)
{
  static const std::unordered_map<uint32_t, std::string> vendorMap = {{0x1002, "AMD"},      {0x1010, "ImgTec"},
                                                                      {0x10DE, "NVIDIA"},   {0x13B5, "ARM"},
                                                                      {0x5143, "Qualcomm"}, {0x8086, "INTEL"}};

  auto it = vendorMap.find(vendorID);
  return it != vendorMap.end() ? it->second : "Unknown Vendor";
}

inline std::string getDeviceType(uint32_t deviceType)
{
  static const std::unordered_map<uint32_t, std::string> deviceTypeMap = {{VK_PHYSICAL_DEVICE_TYPE_OTHER, "Other"},
                                                                          {VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, "Integrated GPU"},
                                                                          {VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, "Discrete GPU"},
                                                                          {VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU, "Virtual GPU"},
                                                                          {VK_PHYSICAL_DEVICE_TYPE_CPU, "CPU"}};

  auto it = deviceTypeMap.find(deviceType);
  return it != deviceTypeMap.end() ? it->second : "Unknown";
}

inline void printGpus(VkInstance instance, VkPhysicalDevice usedGpu)
{
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
  std::vector<VkPhysicalDevice> gpus(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, gpus.data());
  LOGI("_________________________________________________\n");
  LOGI("Available GPUS: %d\n", deviceCount);

  VkPhysicalDeviceProperties properties;
  for(uint32_t d = 0; d < deviceCount; d++)
  {
    vkGetPhysicalDeviceProperties(gpus[d], &properties);
    LOGI(" - %d) %s\n", d, properties.deviceName);
  }
  if(usedGpu == VK_NULL_HANDLE)
  {
    LOGE("No compatible GPU\n");
    return;
  }

  LOGI("Using GPU:\n");
  vkGetPhysicalDeviceProperties(usedGpu, &properties);
  printPhysicalDeviceProperties(properties);
}

inline void printPhysicalDeviceProperties(const VkPhysicalDeviceProperties& properties)
{
  LOGI(" - Device Name    : %s\n", properties.deviceName);
  LOGI(" - Vendor         : %s\n", getVendorName(properties.vendorID).c_str());
  LOGI(" - Driver Version : %s\n", getVersionString(properties.driverVersion).c_str());
  LOGI(" - API Version    : %s\n", getVersionString(properties.apiVersion).c_str());
  LOGI(" - Device Type    : %s\n", getDeviceType(properties.deviceType).c_str());
}


// https://vulkan.lunarg.com/doc/sdk/1.3.296.0/windows/khronos_validation_layer.html
struct ValidationSettings
{
  VkBool32                 fine_grained_locking{true};
  VkBool32                 validate_core{true};
  VkBool32                 check_image_layout{true};
  VkBool32                 check_command_buffer{true};
  VkBool32                 check_object_in_use{true};
  VkBool32                 check_query{true};
  VkBool32                 check_shaders{true};
  VkBool32                 check_shaders_caching{true};
  VkBool32                 debug_disable_spirv_val{false};
  VkBool32                 unique_handles{true};
  VkBool32                 object_lifetime{true};
  VkBool32                 stateless_param{true};
  VkBool32                 thread_safety{true};
  VkBool32                 validate_sync{false};
  VkBool32                 syncval_submit_time_validation{true};
  VkBool32                 syncval_shader_accesses_heuristic{false};
  std::vector<const char*> validate_gpu_based{"GPU_BASED_NONE"};  // "GPU_BASED_DEBUG_PRINTF", "GPU_BASED_GPU_ASSISTED"
  VkBool32                 printf_to_stdout{true};
  VkBool32                 printf_verbose{false};
  int32_t                  printf_buffer_size{1024};
  VkBool32                 gpuav_shader_instrumentation{true};
  VkBool32                 gpuav_descriptor_checks{true};
  VkBool32                 gpuav_warn_on_robust_oob{true};
  VkBool32                 gpuav_buffer_address_oob{true};
  int32_t                  gpuav_max_buffer_device_addresses{10000};
  VkBool32                 gpuav_validate_ray_query{true};
  VkBool32                 gpuav_cache_instrumented_shaders{true};
  VkBool32                 gpuav_select_instrumented_shaders{false};
  VkBool32                 gpuav_buffers_validation{true};
  VkBool32                 gpuav_indirect_draws_buffers{false};
  VkBool32                 gpuav_indirect_dispatches_buffers{false};
  VkBool32                 gpuav_indirect_trace_rays_buffers{false};
  VkBool32                 gpuav_buffer_copies{true};
  VkBool32                 gpuav_reserve_binding_slot{true};
  VkBool32                 gpuav_vma_linear_output{true};
  VkBool32                 gpuav_debug_validate_instrumented_shaders{false};
  VkBool32                 gpuav_debug_dump_instrumented_shaders{false};
  int32_t                  gpuav_debug_max_instrumented_count{0};
  VkBool32                 gpuav_debug_print_instrumentation_info{false};
  VkBool32                 validate_best_practices{false};
  VkBool32                 validate_best_practices_arm{false};
  VkBool32                 validate_best_practices_amd{false};
  VkBool32                 validate_best_practices_img{false};
  VkBool32                 validate_best_practices_nvidia{false};
  std::vector<const char*> debug_action{"VK_DBG_LAYER_ACTION_LOG_MSG"};  // "VK_DBG_LAYER_ACTION_DEBUG_OUTPUT", "VK_DBG_LAYER_ACTION_BREAK"
  std::vector<const char*> report_flags{"error"};                        // "info", "warn", "perf"
  VkBool32                 enable_message_limit{true};
  int32_t                  duplicate_message_limit{3};  // Was 10
  std::vector<uint32_t>    message_id_filter{};         // "MessageID = 0x30b6e267"
  VkBool32                 message_format_display_application_name{false};


  // Vulkan interface
  VkBaseInStructure* buildPNextChain()
  {
    updateSettings();
    return reinterpret_cast<VkBaseInStructure*>(&m_layerSettingsCreateInfo);
  }

  void updateSettings()
  {
    // clang-format off
    m_settings = {
        {m_layer_name, "fine_grained_locking", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &fine_grained_locking},
        {m_layer_name, "validate_core", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_core},
        {m_layer_name, "check_image_layout", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &check_image_layout},
        {m_layer_name, "check_command_buffer", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &check_command_buffer},
        {m_layer_name, "check_object_in_use", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &check_object_in_use},
        {m_layer_name, "check_query", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &check_query},
        {m_layer_name, "check_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &check_shaders},
        {m_layer_name, "check_shaders_caching", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &check_shaders_caching},
        {m_layer_name, "debug_disable_spirv_val", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &debug_disable_spirv_val},
        {m_layer_name, "unique_handles", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &unique_handles},
        {m_layer_name, "object_lifetime", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &object_lifetime},
        {m_layer_name, "stateless_param", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &stateless_param},
        {m_layer_name, "thread_safety", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &thread_safety},
        {m_layer_name, "validate_sync", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_sync},
        {m_layer_name, "syncval_submit_time_validation", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &syncval_submit_time_validation},
        {m_layer_name, "syncval_shader_accesses_heuristic", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &syncval_shader_accesses_heuristic},
        {m_layer_name, "validate_gpu_based", VK_LAYER_SETTING_TYPE_STRING_EXT, uint32_t(validate_gpu_based.size()), validate_gpu_based.data()},
        {m_layer_name, "printf_to_stdout", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &printf_to_stdout},
        {m_layer_name, "printf_verbose", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &printf_verbose},
        {m_layer_name, "printf_buffer_size", VK_LAYER_SETTING_TYPE_INT32_EXT, 1, &printf_buffer_size},
        {m_layer_name, "gpuav_shader_instrumentation", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_shader_instrumentation},
        {m_layer_name, "gpuav_descriptor_checks", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_descriptor_checks},
        {m_layer_name, "gpuav_warn_on_robust_oob", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_warn_on_robust_oob},
        {m_layer_name, "gpuav_buffer_address_oob", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_buffer_address_oob},
        {m_layer_name, "gpuav_max_buffer_device_addresses", VK_LAYER_SETTING_TYPE_INT32_EXT, 1, &gpuav_max_buffer_device_addresses},
        {m_layer_name, "gpuav_validate_ray_query", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_validate_ray_query},
        {m_layer_name, "gpuav_cache_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_cache_instrumented_shaders},
        {m_layer_name, "gpuav_select_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_select_instrumented_shaders},
        {m_layer_name, "gpuav_buffers_validation", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_buffers_validation},
        {m_layer_name, "gpuav_indirect_draws_buffers", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_indirect_draws_buffers},
        {m_layer_name, "gpuav_indirect_dispatches_buffers", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_indirect_dispatches_buffers},
        {m_layer_name, "gpuav_indirect_trace_rays_buffers", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_indirect_trace_rays_buffers},
        {m_layer_name, "gpuav_buffer_copies", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_buffer_copies},
        {m_layer_name, "gpuav_reserve_binding_slot", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_reserve_binding_slot},
        {m_layer_name, "gpuav_vma_linear_output", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_vma_linear_output},
        {m_layer_name, "gpuav_debug_validate_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_debug_validate_instrumented_shaders},
        {m_layer_name, "gpuav_debug_dump_instrumented_shaders", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_debug_dump_instrumented_shaders},
        {m_layer_name, "gpuav_debug_max_instrumented_count", VK_LAYER_SETTING_TYPE_INT32_EXT, 1, &gpuav_debug_max_instrumented_count},
        {m_layer_name, "gpuav_debug_print_instrumentation_info", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &gpuav_debug_print_instrumentation_info},
        {m_layer_name, "validate_best_practices", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices},
        {m_layer_name, "validate_best_practices_arm", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices_arm},
        {m_layer_name, "validate_best_practices_amd", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices_amd},
        {m_layer_name, "validate_best_practices_img", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices_img},
        {m_layer_name, "validate_best_practices_nvidia", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &validate_best_practices_nvidia},
        {m_layer_name, "debug_action", VK_LAYER_SETTING_TYPE_STRING_EXT, uint32_t(debug_action.size()), debug_action.data()},
        {m_layer_name, "enable_message_limit", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &enable_message_limit},
        {m_layer_name, "duplicate_message_limit", VK_LAYER_SETTING_TYPE_INT32_EXT, 1, &duplicate_message_limit},
        {m_layer_name, "report_flags", VK_LAYER_SETTING_TYPE_STRING_EXT, uint32_t(report_flags.size()), report_flags.data()},
        {m_layer_name, "message_id_filter", VK_LAYER_SETTING_TYPE_UINT32_EXT, uint32_t(message_id_filter.size()), message_id_filter.data()},
        {m_layer_name, "message_format_display_application_name", VK_LAYER_SETTING_TYPE_BOOL32_EXT, 1, &message_format_display_application_name},
    };
    // clang-format on

    m_layerSettingsCreateInfo.sType        = VK_STRUCTURE_TYPE_LAYER_SETTINGS_CREATE_INFO_EXT;
    m_layerSettingsCreateInfo.settingCount = static_cast<uint32_t>(m_settings.size());
    m_layerSettingsCreateInfo.pSettings    = m_settings.data();
  }


  VkLayerSettingsCreateInfoEXT   m_layerSettingsCreateInfo;
  std::vector<VkLayerSettingEXT> m_settings;
  static constexpr const char*   m_layer_name{"VK_LAYER_KHRONOS_validation"};
};