/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

//
// NVIDIA DLSS (Deep Learning Super Sampling) wrapper. Initializes the
// NGX SDK, creates and configures DLSS-SR and DLSS-RR features, handles
// quality-mode selection and jitter offsets, and evaluates the DLSS
// network each frame to produce upscaled or denoised output.
//

#define _CRT_SECURE_NO_WARNINGS
#include <volk.h>
#include <unordered_set>

#include "dlss_wrapper.hpp"


#define NGX_ENABLE_DEPRECATED_GET_PARAMETERS 1

#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>
#include <nvsdk_ngx_defs_dlssd.h>
#include <nvsdk_ngx_helpers_dlssd.h>

#include <glm/ext.hpp>

#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <cstring>  // memset
#include <mutex>
#include "nvutils/logger.hpp"


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// kNgxApplicationID is declared in dlss_wrapper.hpp and overridable via the CMake define
// DLSS_APPLICATION_ID. Production deployments should request a per-application ID from
// NVIDIA DevRel rather than ship the placeholder magic number.


// Function taken from stack overflow https://stackoverflow.com/questions/4804298/how-to-convert-wstring-into-string
static std::string wstringToString(const std::wstring& wstr)
{
  std::mbstate_t state{};
  const wchar_t* src = wstr.c_str();
  std::size_t    len = std::wcsrtombs(nullptr, &src, 0, &state);
  if(len == static_cast<std::size_t>(-1))
  {
    throw std::runtime_error("Conversion failed");
  }

  std::string str(len, '\0');
  std::wcsrtombs(&str[0], &src, len, &state);
  return str;
}

static std::wstring stringToWstring(const std::string& str)
{
  size_t len = std::mbstowcs(nullptr, str.c_str(), 0);
  if(len == static_cast<size_t>(-1))
  {
    return L"";  // Conversion failed
  }
  std::wstring wstr(len, L'\0');
  std::mbstowcs(&wstr[0], str.c_str(), len);
  return wstr;
}


static NVSDK_NGX_Result checkNgxResult(NVSDK_NGX_Result result, const char* func, int line)
{
  if(NVSDK_NGX_FAILED(result))
  {
    std::ostringstream str;
    str << "NGX Error: " << wstringToString(GetNGXResultAsString(result)) << " at " << func << ":" << line;

    LOGW("%s\n", str.str().c_str());
  }

  return result;
}

#define CALL_NGX(x)                                                                                                    \
  {                                                                                                                    \
    NVSDK_NGX_Result r = checkNgxResult((x), __func__, __LINE__);                                                      \
    if(NVSDK_NGX_FAILED(r))                                                                                            \
      return r;                                                                                                        \
  }

void NVSDK_CONV NGX_AppLogCallback(const char* message, NVSDK_NGX_Logging_Level loggingLevel, NVSDK_NGX_Feature sourceComponent)
{
  LOGI("%s", message);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Class code
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// NGX library state is process-global. DLSS-RR (path tracer) and DLSS-SR (rasterizer) each
// own their own NgxContext; refcount so Init/Shutdown1 happen exactly once across both.
// Dlss::tick() launches a worker std::thread per Kind, so the RR and SR workers can race
// into init()/deinit() concurrently; the mutex serializes the check-then-act around the
// refcount and the NGX Init/Shutdown1 entry points.
static std::mutex s_ngxMutex;  // Static needed because it guards the process-global NGX ref-count + Init/Shutdown1 across all NgxContext instances
static int s_ngxRefCount = 0;  // Static needed because NGX library state is process-global; ref-counted across RR + SR NgxContext instances

NVSDK_NGX_Result NgxContext::init(const NgxContext::InitInfo& initInfo)
{
  std::wstring   exeString  = stringToWstring(initInfo.appInfo.applicationPath);
  const wchar_t* exeWString = exeString.c_str();

  m_initInfo = initInfo;

  NVSDK_NGX_FeatureCommonInfo info     = {};
  info.LoggingInfo.MinimumLoggingLevel = initInfo.loggingLevel;

  // Hold the lock across the check-then-act on s_ngxRefCount AND the NGX_VULKAN_Init call so a
  // concurrent worker can't observe refcount==0 in parallel and double-initialize the library.
  std::lock_guard<std::mutex> lock(s_ngxMutex);

  if(s_ngxRefCount == 0)
  {
    CALL_NGX(NVSDK_NGX_VULKAN_Init(kNgxApplicationID, exeWString, initInfo.instance, initInfo.physicalDevice,
                                   initInfo.device, vkGetInstanceProcAddr, vkGetDeviceProcAddr, &info));
  }
  ++s_ngxRefCount;

  CALL_NGX(NVSDK_NGX_VULKAN_GetCapabilityParameters(&m_ngxParams));

  if(m_ngxParams == nullptr)
    return NVSDK_NGX_Result_Fail;

  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::deinit()
{
  if(m_ngxParams)
  {
    // Symmetric with init(): serialize the decrement and the conditional Shutdown1 so two
    // concurrent deinits can't both observe refcount==0 (or both miss it).
    std::lock_guard<std::mutex> lock(s_ngxMutex);

    CALL_NGX(NVSDK_NGX_VULKAN_DestroyParameters(m_ngxParams));
    m_ngxParams = {};
    if(--s_ngxRefCount == 0)
    {
      CALL_NGX(NVSDK_NGX_VULKAN_Shutdown1(m_initInfo.device));
    }
  }
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::isFeatureAvailable(NVSDK_NGX_Feature feature)
{
  if(m_ngxParams == nullptr)
  {
    LOGW("DLSS: NGX parameters not initialized\n");
    return NVSDK_NGX_Result_Fail;
  }

  // Query NGX for the requested DLSS feature support (RR or SR, etc.)
  NVSDK_NGX_FeatureCommonInfo    commonInfo = {};
  NVSDK_NGX_FeatureDiscoveryInfo info       = {};
  info.SDKVersion                           = NVSDK_NGX_Version_API;
  info.FeatureID                            = feature;
  info.Identifier.IdentifierType            = NVSDK_NGX_Application_Identifier_Type_Application_Id;
  info.Identifier.v.ApplicationId           = kNgxApplicationID;
  info.ApplicationDataPath                  = L" ";  // Non-empty string required by NGX API
  info.FeatureInfo                          = &commonInfo;

  NVSDK_NGX_FeatureRequirement requirement = {};

  NVSDK_NGX_Result result =
      NVSDK_NGX_VULKAN_GetFeatureRequirements(m_initInfo.instance, m_initInfo.physicalDevice, &info, &requirement);
  if(NVSDK_NGX_FAILED(result))
  {
    LOGW("DLSS: Failed to query feature requirements from NGX (feature: %d, error: %d)\n", int(feature), result);
    return result;
  }

  if(requirement.FeatureSupported != NVSDK_NGX_FeatureSupportResult_Supported)
  {
    LOGW("DLSS: Feature %d not supported by GPU/driver\n", int(feature));
    return NVSDK_NGX_Result_FAIL_Denied;
  }

  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::getRequiredInstanceExtensions(NVSDK_NGX_Feature                   feature,
                                                           const ApplicationInfo&              appInfo,
                                                           std::vector<VkExtensionProperties>& extensions)
{
  std::wstring appPath = stringToWstring(appInfo.applicationPath);
  extensions.clear();
  NVSDK_NGX_FeatureCommonInfo commonInfo = {};

  NVSDK_NGX_FeatureDiscoveryInfo info{};
  info.SDKVersion                             = NVSDK_NGX_Version_API;
  info.FeatureID                              = feature;
  info.Identifier.IdentifierType              = NVSDK_NGX_Application_Identifier_Type_Project_Id;
  info.Identifier.v.ProjectDesc.EngineType    = appInfo.engineType;
  info.Identifier.v.ProjectDesc.EngineVersion = appInfo.engineVersion.c_str();
  info.Identifier.v.ProjectDesc.ProjectId     = appInfo.projectId.c_str();

  info.ApplicationDataPath = appPath.c_str();
  info.FeatureInfo         = &commonInfo;

  uint32_t               numExtensions = 0;
  VkExtensionProperties* props{};

  CALL_NGX(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&info, &numExtensions, &props));

  extensions = std::vector<VkExtensionProperties>(props, props + numExtensions);
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result NgxContext::getRequiredDeviceExtensions(NVSDK_NGX_Feature                   feature,
                                                         const ApplicationInfo&              appInfo,
                                                         const VkInstance&                   instance,
                                                         const VkPhysicalDevice&             physicalDevice,
                                                         std::vector<VkExtensionProperties>& extensions)
{
  std::wstring appPath = stringToWstring(appInfo.applicationPath);
  extensions.clear();

  NVSDK_NGX_FeatureCommonInfo    commonInfo{};
  NVSDK_NGX_FeatureDiscoveryInfo info{};
  info.SDKVersion                             = NVSDK_NGX_Version_API;
  info.FeatureID                              = feature;
  info.Identifier.IdentifierType              = NVSDK_NGX_Application_Identifier_Type_Project_Id;
  info.Identifier.v.ProjectDesc.EngineType    = appInfo.engineType;
  info.Identifier.v.ProjectDesc.EngineVersion = appInfo.engineVersion.c_str();
  info.Identifier.v.ProjectDesc.ProjectId     = appInfo.projectId.c_str();
  info.ApplicationDataPath                    = appPath.c_str();
  info.FeatureInfo                            = &commonInfo;


  uint32_t               numExtensions = 0;
  VkExtensionProperties* props{};

  CALL_NGX(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, physicalDevice, &info, &numExtensions, &props));

  extensions = std::vector<VkExtensionProperties>(props, props + numExtensions);

  return NVSDK_NGX_Result_Success;
}

bool DlssFeature::querySupport(const NgxContext& context, Kind kind)
{
  int32_t  supported             = 0;
  int32_t  needsUpdatedDriver    = 1;
  uint32_t minDriverVersionMajor = ~0u;
  uint32_t minDriverVersionMinor = ~0u;

  // RR uses the SuperSamplingDenoising_* parameter family; SR uses SuperSampling_*. Both query
  // the same five sub-keys, so the control flow is identical - only the parameter strings differ.
  const bool  useRR             = (kind == Kind::RR);
  const char* tag               = useRR ? "DLSS_RR" : "DLSS_SR";
  const char* paramNeedsUpdate  = useRR ? NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver :
                                          NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver;
  const char* paramVersionMajor = useRR ? NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMajor :
                                          NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor;
  const char* paramVersionMinor = useRR ? NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMinor :
                                          NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor;
  const char* paramAvailable = useRR ? NVSDK_NGX_Parameter_SuperSamplingDenoising_Available : NVSDK_NGX_Parameter_SuperSampling_Available;
  const char* paramFeatureInitRes = useRR ? NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult :
                                            NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult;

  NVSDK_NGX_Result resUpdatedDriver = context.getNgxParams()->Get(paramNeedsUpdate, &needsUpdatedDriver);
  NVSDK_NGX_Result resVersionMajor  = context.getNgxParams()->Get(paramVersionMajor, &minDriverVersionMajor);
  NVSDK_NGX_Result resVersionMinor  = context.getNgxParams()->Get(paramVersionMinor, &minDriverVersionMinor);

  if(NVSDK_NGX_SUCCEED(resUpdatedDriver) && needsUpdatedDriver)
  {
    if(NVSDK_NGX_SUCCEED(resVersionMajor) && NVSDK_NGX_SUCCEED(resVersionMinor))
    {
      LOGW("%s: Driver update required - minimum version: %d.%d\n", tag, minDriverVersionMajor, minDriverVersionMinor);
    }
    else
    {
      LOGW("%s: Driver update required (version information unavailable)\n", tag);
    }
    return false;
  }

  NVSDK_NGX_Result resDlssSupported = context.getNgxParams()->Get(paramAvailable, &supported);
  if(NVSDK_NGX_FAILED(resDlssSupported) || !supported)
  {
    LOGW("%s: Not available on this GPU/platform\n", tag);
    return false;
  }

  resDlssSupported = context.getNgxParams()->Get(paramFeatureInitRes, &supported);
  if(NVSDK_NGX_FAILED(resDlssSupported) || !supported)
  {
    LOGW("%s: Feature initialization denied for this application\n", tag);
    return false;
  }

  return true;
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


NVSDK_NGX_Result DlssFeature::cmdInit(VkCommandBuffer cmd, NgxContext& context, const InitInfo& info)
{
  m_initInfo = info;

  if(m_kind == Kind::RR)
  {
    // ---- DLSS Ray Reconstruction (denoiser + upscaler) ----
    NVSDK_NGX_DLSSD_Create_Params dlssdParams = {};

    dlssdParams.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;

    if(info.packedNormalRoughness)
      dlssdParams.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;  // we pack roughness into the normal's w channel
    else
      dlssdParams.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Unpacked;

    if(info.hardwareDepth)
      dlssdParams.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_HW;  // we're providing hardware (raster) depth
    else
      dlssdParams.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;

    dlssdParams.InWidth        = info.inputSize.width;
    dlssdParams.InHeight       = info.inputSize.height;
    dlssdParams.InTargetWidth  = info.outputSize.width;
    dlssdParams.InTargetHeight = info.outputSize.height;

    // Though marked as 'optional', these are absolutely needed
    dlssdParams.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;

    // NGX bakes the preset hint at feature creation. Pin all four perf qualities to the same
    // preset so the user-visible toggle between Quality/Balanced/Performance does not silently
    // downgrade the model. Preset_Default (0) lets NGX pick the current SDK default.
    const auto rrPreset = static_cast<NVSDK_NGX_RayReconstruction_Hint_Render_Preset>(info.preset);
    context.getNgxParams()->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality, rrPreset);
    context.getNgxParams()->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced, rrPreset);
    context.getNgxParams()->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance, rrPreset);
    context.getNgxParams()->Set(NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance, rrPreset);

    CALL_NGX(NGX_VULKAN_CREATE_DLSSD_EXT1(context.getDevice(), cmd, info.creationNodeMask, info.visibilityNodeMask,
                                          &m_handle, context.getNgxParams(), &dlssdParams));
  }
  else
  {
    // ---- DLSS Super Resolution (DLAA / upscaling) ----
    NVSDK_NGX_DLSS_Create_Params dlssParams = {};
    dlssParams.Feature.InWidth              = info.inputSize.width;
    dlssParams.Feature.InHeight             = info.inputSize.height;
    dlssParams.Feature.InTargetWidth        = info.outputSize.width;
    dlssParams.Feature.InTargetHeight       = info.outputSize.height;
    dlssParams.Feature.InPerfQualityValue   = info.quality;

    // HDR linear color, render-resolution motion vectors, internal exposure estimation.
    dlssParams.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR       //
                                      | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes  //
                                      | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;

    // Same preset-pinning story as RR above; SR uses the DLSS_Hint_Render_Preset enum which
    // shares the letter -> integer mapping with the RR enum, so the integer survives the cast.
    const auto srPreset = static_cast<NVSDK_NGX_DLSS_Hint_Render_Preset>(info.preset);
    context.getNgxParams()->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, srPreset);
    context.getNgxParams()->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, srPreset);
    context.getNgxParams()->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, srPreset);
    context.getNgxParams()->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, srPreset);
    context.getNgxParams()->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, srPreset);

    CALL_NGX(NGX_VULKAN_CREATE_DLSS_EXT1(context.getDevice(), cmd, info.creationNodeMask, info.visibilityNodeMask,
                                         &m_handle, context.getNgxParams(), &dlssParams));
  }

  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result DlssFeature::deinit()
{
  if(m_handle)
  {
    CALL_NGX(NVSDK_NGX_VULKAN_ReleaseFeature(m_handle));
    m_handle = nullptr;
  }
  return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result DlssFeature::querySupportedInputSizes(NgxContext& context, const SupportedSizeInfo& info, SupportedSizes* sizes) const
{
  if(!sizes)
    return NVSDK_NGX_Result_FAIL_InvalidParameter;

  // NGX_DLSSD_* (RR) and NGX_DLSS_* (SR) helpers have the same signature and the same out-params;
  // only the NGX-side capability tables they read differ.
  float sharpness = 0.f;
  if(m_kind == Kind::RR)
  {
    CALL_NGX(NGX_DLSSD_GET_OPTIMAL_SETTINGS(context.getNgxParams(), info.outputSize.width, info.outputSize.height, info.perfQualityValue,
                                            &sizes->optimalSize.width, &sizes->optimalSize.height, &sizes->maxSize.width,
                                            &sizes->maxSize.height, &sizes->minSize.width, &sizes->minSize.height, &sharpness));
  }
  else
  {
    CALL_NGX(NGX_DLSS_GET_OPTIMAL_SETTINGS(context.getNgxParams(), info.outputSize.width, info.outputSize.height, info.perfQualityValue,
                                           &sizes->optimalSize.width, &sizes->optimalSize.height, &sizes->maxSize.width,
                                           &sizes->maxSize.height, &sizes->minSize.width, &sizes->minSize.height, &sharpness));
  }

  return NVSDK_NGX_Result_Success;
}

void DlssFeature::setResource(const Resource& resource)
{
  VkExtent2D size = (resource.type == ResourceType::eColorOut) ? m_initInfo.outputSize : m_initInfo.inputSize;

  bool isReadWrite = (resource.type == ResourceType::eColorOut);

  NVSDK_NGX_Resource_VK r = NVSDK_NGX_Create_ImageView_Resource_VK(resource.imageView, resource.image, resource.range,
                                                                   resource.format, size.width, size.height, isReadWrite);

  m_resources[int(resource.type)] = r;
}

NVSDK_NGX_Result DlssFeature::cmdEvaluate(VkCommandBuffer cmd, NgxContext& context, const EvaluateInfo& info)
{
  // Shared eval parameters between RR and SR (jitter, MV scale, subrect, reset). The jitter
  // negation here is REQUIRED: callers pass info.jitter = actual sub-pixel sample offset (positive
  // = "sample to the right of pixel center"), but NGX wants the offset that DE-jitters the input,
  // i.e. the negative.
  const float jitterX = -info.jitter.x;
  const float jitterY = -info.jitter.y;

  if(m_kind == Kind::RR)
  {
    NVSDK_NGX_VK_DLSSD_Eval_Params evalParams = {};
    evalParams.pInDiffuseAlbedo               = &m_resources[int(ResourceType::eDiffuseAlbedo)];
    evalParams.pInDiffuseHitDistance          = nullptr;
    evalParams.pInSpecularAlbedo              = &m_resources[int(ResourceType::eSpecularAlbedo)];
    // Specular-hit-distance is optional; NGX expects nullptr (not an empty struct) when absent.
    evalParams.pInSpecularHitDistance = m_resources[int(ResourceType::eSpecularHitDistance)].Resource.ImageViewInfo.ImageView ?
                                            &m_resources[int(ResourceType::eSpecularHitDistance)] :
                                            nullptr;

    evalParams.pInNormals = &m_resources[int(ResourceType::eNormalRoughness)];
    // In Packed roughness mode NGX still wants pInRoughness set (pointed at the same texture as
    // pInNormals). Leaving it null with Packed produces "missing input" failures at evaluate.
    evalParams.pInRoughness = m_initInfo.packedNormalRoughness ? &m_resources[int(ResourceType::eNormalRoughness)] :
                                                                 &m_resources[int(ResourceType::eRoughness)];

    evalParams.pInColor         = &m_resources[int(ResourceType::eColorIn)];
    evalParams.pInOutput        = &m_resources[int(ResourceType::eColorOut)];
    evalParams.pInDepth         = &m_resources[int(ResourceType::eDepth)];
    evalParams.pInMotionVectors = &m_resources[int(ResourceType::eMotionVector)];

    evalParams.InJitterOffsetX = jitterX;
    evalParams.InJitterOffsetY = jitterY;
    evalParams.InMVScaleX      = 1.0f;
    evalParams.InMVScaleY      = 1.0f;

    evalParams.InRenderSubrectDimensions.Width  = m_initInfo.inputSize.width;
    evalParams.InRenderSubrectDimensions.Height = m_initInfo.inputSize.height;

    // glm is column-major; NGX wants row-major. Transpose then hand off via float-pointer.
    glm::mat4 modelViewRowMajor     = glm::transpose(info.modelView);
    glm::mat4 projectionRowMajor    = glm::transpose(info.projection);
    evalParams.pInWorldToViewMatrix = glm::value_ptr(modelViewRowMajor);
    evalParams.pInViewToClipMatrix  = glm::value_ptr(projectionRowMajor);

    evalParams.InReset = info.reset;

    NVSDK_NGX_Result evalResult = NGX_VULKAN_EVALUATE_DLSSD_EXT(cmd, m_handle, context.getNgxParams(), &evalParams);
    if(NVSDK_NGX_FAILED(evalResult))
    {
      // Loud failure logging: NGX evaluate misconfiguration (bad resource binding, stale feature
      // handle, mismatched extents, ...) is almost always a bug in the host code, not a transient
      // condition the user can ignore. CALL_NGX would only LOGW.
      LOGE("DLSS-RR: NGX evaluate failed (%s) at %s:%d\n", wstringToString(GetNGXResultAsString(evalResult)).c_str(),
           __func__, __LINE__);
      return evalResult;
    }
  }
  else
  {
    NVSDK_NGX_VK_DLSS_Eval_Params evalParams = {};
    evalParams.Feature.pInColor              = &m_resources[int(ResourceType::eColorIn)];
    evalParams.Feature.pInOutput             = &m_resources[int(ResourceType::eColorOut)];
    evalParams.pInDepth                      = &m_resources[int(ResourceType::eDepth)];
    evalParams.pInMotionVectors              = &m_resources[int(ResourceType::eMotionVector)];

    evalParams.InJitterOffsetX = jitterX;
    evalParams.InJitterOffsetY = jitterY;
    evalParams.InMVScaleX      = 1.0f;
    evalParams.InMVScaleY      = 1.0f;

    evalParams.InRenderSubrectDimensions.Width  = m_initInfo.inputSize.width;
    evalParams.InRenderSubrectDimensions.Height = m_initInfo.inputSize.height;

    evalParams.InReset = info.reset ? 1 : 0;

    NVSDK_NGX_Result evalResult = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, m_handle, context.getNgxParams(), &evalParams);
    if(NVSDK_NGX_FAILED(evalResult))
    {
      LOGE("DLSS-SR: NGX evaluate failed (%s) at %s:%d\n", wstringToString(GetNGXResultAsString(evalResult)).c_str(),
           __func__, __LINE__);
      return evalResult;
    }
  }

  return NVSDK_NGX_Result_Success;
}
