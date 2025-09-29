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

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#include "nvshaders/slang_types.h"
#include "nvshaders/sky_io.h.slang"
#include "nvshaders/gltf_scene_io.h.slang"
#include "nvshaders/hdr_io.h.slang"

NAMESPACE_SHADERIO_BEGIN()

#define WORKGROUP_SIZE 16
#define SILHOUETTE_WORKGROUP_SIZE 16


#define HDR_DIFFUSE_INDEX 0
#define HDR_GLOSSY_INDEX 1
#define HDR_IMAGE_INDEX 0
#define HDR_LUT_INDEX 1

// Environment types
enum class EnvSystem
{
  eSky,
  eHdr,
};

// Output image types
enum OutputImage
{
  eResultImage = 0,      // Output image (RGBA32)
  eSelectImage,          // Selection image (R8)
  eDlssAlbedo,           // Diffuse albedo (RGBA8)
  eDlssSpecAlbedo,       // Specular albedo (RGBA32)
  eDlssNormalRoughness,  // Normal and roughness (RGBA32)
  eDlssMotion,           // Motion (RGBA32)
  eDlssDepth,            // Depth (R32)
};

// Binding points for descriptors
enum BindingPoints
{
  eTlas = 0,      // Top level acceleration structure
  eOutImages,     // Output image (RGBA32)
  eTextures,      // Textures (array of textures)
  eTexturesCube,  // Textures (array of textures)
  eTexturesHdr,   // HDR textures (array of textures)
  eTexturesStorage,
};

// Binding points for descriptors
enum SilhouetteBindings
{
  eObjectID,    // In: the object ID image
  eRGBAIImage,  // Out: the output image
};

enum DebugMethod
{
  eNone,
  eBaseColor,
  eMetallic,
  eRoughness,
  eNormal,
  eTangent,
  eBitangent,
  eEmissive,
  eOpacity,
  eTexCoord0,
  eTexCoord1,
};


// Camera info
struct SceneFrameInfo
{
  float4x4    viewMatrix;                   // View matrix
  float4x4    projInv;                      // Inverse projection matrix
  float4x4    viewInv;                      // Inverse view matrix
  float4x4    viewProjMatrix;               // View-projection matrix (P*V)
  float4x4    prevMVP;                      // Previous view-projection matrix
  float       envRotation;                  // Environment rotation (used for the HDR)
  float       envBlur;                      // Level of blur for the environment map (0.0: no blur, 1.0: full blur)
  float       envIntensity = 1.f;           // Environment intensity
  int         useSolidBackground;           // Use solid background color (0==false, 1==true)
  float3      backgroundColor;              // Background color when using solid background
  int         environmentType        = 0;   // Environment type; 0: sky, 1: environment map
  int         selectedRenderNode     = -1;  // Selected render node
  DebugMethod debugMethod            = DebugMethod::eNone;  // Debug method
  int         useInfinitePlane       = 0;
  float       infinitePlaneDistance  = 0;
  float3      infinitePlaneBaseColor = float3(0.5, 0.5, 0.5);  // Default gray color
  float       infinitePlaneMetallic  = 0.0;                    // Default non-metallic
  float       infinitePlaneRoughness = 0.5;                    // Default medium roughness
};

// Push constant
struct PathtracePushConstant
{
  int   maxDepth              = 5;     // Maximum depth of the ray
  int   frameCount            = 0;     // Frame number
  float fireflyClampThreshold = 10.f;  // Firefly clamp threshold
  int   numSamples            = 1;     // Number of samples per pixel per frame
  int   totalSamples          = 0;     // Total samples accumulated so far
  float focalDistance         = 0.0f;  // Focal distance for depth of field
  float aperture              = 0.0f;  // Aperture for depth of field
  int   useDlss               = 0;     // Use DLSS (0: no, 1: yes)
  int   renderSelection       = 1;     // Padding to align the structure
  /// Infinite plane
  float2                 jitter;               // Jitter for the DLSS
  float2                 mouseCoord = {0, 0};  // Mouse coordinates (use for debug)
  SceneFrameInfo*        frameInfo;            // Camera info
  SkyPhysicalParameters* skyParams;            // Sky physical parameters
  GltfScene*             gltfScene;            // GLTF sceneF
};

// Push constant
struct RasterPushConstant
{
  int                    materialID   = 0;       // Material used by the rendering instance
  int                    renderNodeID = 0;       // Node used by the rendering instance
  int                    renderPrimID = 0;       // Primitive used by the rendering instance
  float2                 mouseCoord   = {0, 0};  // Mouse coordinates (use for debug)
  SceneFrameInfo*        frameInfo;              // Camera info
  SkyPhysicalParameters* skyParams;              // Sky physical parameters
  GltfScene*             gltfScene;              // GLTF sceneF
};


struct SilhouettePushConstant
{
  float3 color;
};

NAMESPACE_SHADERIO_END()

#endif  // HOST_DEVICE_H
