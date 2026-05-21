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
#pragma once
#include <cstdint>
#include <vector>
#include <string>

#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>

#include "nvvk/resource_allocator.hpp"
#include "nvsdk_ngx_vk.h"


// NGX application identifier passed to NVSDK_NGX_VULKAN_Init. The 0x12345678ABCDEF01 magic
// number is a placeholder — production builds should request a real per-application ID from
// NVIDIA DevRel and define DLSS_APPLICATION_ID via CMake to override this default.
#ifndef DLSS_APPLICATION_ID
#define DLSS_APPLICATION_ID 0x12345678ABCDEF01ULL
#endif
inline constexpr uint64_t kNgxApplicationID = DLSS_APPLICATION_ID;


class NgxContext
{
public:
  struct ApplicationInfo
  {
    std::string engineVersion   = "0.0";
    std::string projectId       = "nvpro-samples";
    std::string applicationPath = "";
    // WAR: custom type prevents creating the NGX API context, using Omniverse instead
    NVSDK_NGX_EngineType engineType = NVSDK_NGX_ENGINE_TYPE_OMNIVERSE;
  };

  struct InitInfo
  {
    VkInstance              instance       = VK_NULL_HANDLE;
    VkPhysicalDevice        physicalDevice = VK_NULL_HANDLE;
    VkDevice                device         = VK_NULL_HANDLE;
    NVSDK_NGX_Logging_Level loggingLevel   = NVSDK_NGX_LOGGING_LEVEL_OFF;
    ApplicationInfo         appInfo        = {};
  };

  NVSDK_NGX_Result init(const InitInfo& info);
  NVSDK_NGX_Result deinit();

  NVSDK_NGX_EngineType       getEngineType() const { return m_initInfo.appInfo.engineType; }
  const std::string&         getEngineVersion() const { return m_initInfo.appInfo.engineVersion; }
  const std::string&         getProjectId() const { return m_initInfo.appInfo.projectId; }
  const NVSDK_NGX_Parameter* getNgxParams() const { return m_ngxParams; }
  NVSDK_NGX_Parameter*       getNgxParams() { return m_ngxParams; }

  // Generic NGX feature query, valid after init().
  NVSDK_NGX_Result isFeatureAvailable(NVSDK_NGX_Feature feature);

  // Feature-agnostic extension queries. Called once per NGX feature (RR and SR) at startup; the
  // unioned set of extensions is what the app must enable at instance/device creation.
  static NVSDK_NGX_Result getRequiredInstanceExtensions(NVSDK_NGX_Feature                   feature,
                                                        const ApplicationInfo&              appInfo,
                                                        std::vector<VkExtensionProperties>& extensions);
  static NVSDK_NGX_Result getRequiredDeviceExtensions(NVSDK_NGX_Feature                   feature,
                                                      const ApplicationInfo&              appInfo,
                                                      const VkInstance&                   instance,
                                                      const VkPhysicalDevice&             physicalDevice,
                                                      std::vector<VkExtensionProperties>& extensions);

  const std::string& getApplicationPath() const { return m_initInfo.appInfo.applicationPath; }

  VkDevice         getDevice() const { return m_initInfo.device; }
  VkInstance       getInstance() const { return m_initInfo.instance; }
  VkPhysicalDevice getPhysicalDevice() const { return m_initInfo.physicalDevice; }

private:
  InitInfo             m_initInfo  = {};
  NVSDK_NGX_Parameter* m_ngxParams = nullptr;
};


// Single NGX DLSS feature instance — covers both DLSS Ray Reconstruction (DLSS-RR, the AI
// denoiser+upscaler used by the path tracer) and DLSS Super Resolution (DLSS-SR, used at native
// resolution = DLAA, or smaller-than-display = SR upscaling, used by the rasterizer). The two
// NGX features share the same plumbing (init / eval / resource binding) and the same per-frame
// eval inputs; only the underlying NGX entry points and a few create flags differ. The "kind"
// (RR vs SR) is fixed at construction because the NGX handle is feature-specific. Each instance
// owns exactly one NGX handle.
//
// Caller pattern:
//   DlssFeature dlss(DlssFeature::Kind::SR);  // or default-constructed for RR
//   dlss.cmdInit(cmd, ngx, info);             // creates exactly one NGX handle (RR or SR)
//   dlss.setResource(...);                    // bind inputs/outputs (SR ignores RR-only entries)
//   dlss.cmdEvaluate(cmd, ngx, info);         // wraps NVSDK_NGX_VULKAN_EvaluateFeature[_DLSSD]
//
// ResourceType is the union of both features' inputs; SR uses only eColorIn/eColorOut/
// eMotionVector/eDepth and ignores the RR-specific entries (diffuse/specular albedo, normals,
// roughness, specular hit distance).
class DlssFeature
{
public:
  enum class Kind
  {
    SR,  // DLSS Super Resolution (DLAA + upscaling)
    RR,  // DLSS Ray Reconstruction (denoiser + upscaler)
  };

  enum class ResourceType
  {
    eColorIn,
    eColorOut,
    eDiffuseAlbedo,        // RR only
    eSpecularAlbedo,       // RR only
    eSpecularHitDistance,  // RR only (optional)
    eNormalRoughness,      // RR only
    eRoughness,            // RR only (used when packedNormalRoughness == false)
    eMotionVector,
    eDepth,
    eResourceTypeCount
  };

  // Default-construct for RR (the original behavior); pass Kind::SR for the rasterizer side.
  DlssFeature() = default;
  explicit DlssFeature(Kind kind)
      : m_kind(kind)
  {
  }


  struct SupportedSizes
  {
    VkExtent2D minSize;
    VkExtent2D maxSize;
    VkExtent2D optimalSize;
  };


  struct SupportedSizeInfo
  {
    VkExtent2D outputSize{};
    // RR typically uses MaxQuality; SR uses one of DLAA / MaxQuality / Balanced / MaxPerf /
    // UltraPerformance. Caller picks the right value for the active Kind.
    NVSDK_NGX_PerfQuality_Value perfQualityValue = NVSDK_NGX_PerfQuality_Value_MaxQuality;
  };

  // Static availability check (driver / hardware / per-app entitlement). Picks the RR or SR
  // parameter family based on `kind`.
  static bool querySupport(const NgxContext& context, Kind kind = Kind::RR);

  // Branches on m_kind for the right NGX optimal-settings helper (RR vs SR).
  NVSDK_NGX_Result querySupportedInputSizes(NgxContext& context, const SupportedSizeInfo& info, SupportedSizes* sizes) const;

  struct InitInfo
  {
    // RR-only fields (ignored when kind == SR):
    bool packedNormalRoughness = true;
    bool hardwareDepth         = true;
    // SR-only field (ignored when kind == RR); typical values: MaxQuality (upscale) or DLAA
    // (native-resolution antialias). For RR, the perf-quality preset is set per-creation via
    // RayReconstruction_Hint_Render_Preset NGX parameters in cmdInit.
    NVSDK_NGX_PerfQuality_Value quality = NVSDK_NGX_PerfQuality_Value_MaxQuality;
    // Network/model preset hint. Reinterpreted at cmdInit time as either
    // NVSDK_NGX_RayReconstruction_Hint_Render_Preset (RR) or NVSDK_NGX_DLSS_Hint_Render_Preset
    // (SR). Both NGX enums share the same letter -> integer mapping, so a single integer works.
    // 0 (Default) lets NGX pick whatever model is current for the SDK build.
    uint32_t   preset = 0;
    VkExtent2D inputSize{};
    VkExtent2D outputSize{};
    // GPU node mask; change only on multi-GPU systems.
    uint32_t creationNodeMask   = 0x1;
    uint32_t visibilityNodeMask = 0x1;
  };

  NVSDK_NGX_Result cmdInit(VkCommandBuffer cmd, NgxContext& context, const InitInfo& info);
  NVSDK_NGX_Result deinit();


  struct Resource
  {
    ResourceType            type{};
    VkImage                 image{};
    VkImageView             imageView{};
    VkFormat                format{};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  };

  void setResource(const Resource& resource);

  struct EvaluateInfo
  {
    glm::vec2 jitter;
    glm::mat4 modelView;
    glm::mat4 projection;
    bool      reset = false;
  };

  // Wraps NVSDK_NGX_VULKAN_EvaluateFeature_DLSSD (Kind::RR -- denoise + upscale) or
  // NVSDK_NGX_VULKAN_EvaluateFeature (Kind::SR -- DLAA / super-resolution upscale). The kind-
  // dispatch lives in the .cpp; from the caller's perspective this is "run NGX for one frame".
  NVSDK_NGX_Result cmdEvaluate(VkCommandBuffer cmd, NgxContext& context, const EvaluateInfo& info);

  Kind getKind() const { return m_kind; }

private:
  Kind                                                                     m_kind = Kind::RR;
  InitInfo                                                                 m_initInfo{};
  NVSDK_NGX_Handle*                                                        m_handle{};
  std::array<NVSDK_NGX_Resource_VK, int(ResourceType::eResourceTypeCount)> m_resources{};
};
