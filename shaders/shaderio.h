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

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#include "nvshaders/slang_types.h"
#include "nvshaders/sky_io.h.slang"
#include "gltf_scene_io.h.slang"
#include "nvshaders/hdr_io.h.slang"

NAMESPACE_SHADERIO_BEGIN()

#define WORKGROUP_SIZE 16
#define SILHOUETTE_WORKGROUP_SIZE 16


// Indices into `texturesCube[]` (eTexturesCube binding).
#define HDR_DIFFUSE_INDEX 0  // Lambert-prefiltered HDR environment cubemap
#define HDR_GLOSSY_INDEX 1   // GGX-prefiltered HDR environment cubemap

// Indices into `texturesHdr[]` (eTexturesHdr binding). Grouped here because they are all 2D
// Sampler2D slots consumed together at shade time for IBL + KHR_materials_transmission.
#define HDR_IMAGE_INDEX 0    // Lat-long HDR environment image
#define HDR_LUT_INDEX 1      // GGX split-sum BRDF LUT (hdr_integrate_brdf)
#define HDR_SHEEN_INDEX 2    // Charlie sheen directional-albedo LUT (hdr_charlie_brdf_lut)
#define HDR_OPAQUE_INDEX 3   // Raster-only opaque-pass color capture (screen-space transmission)
#define HDR_TEXTURE_COUNT 4  // Count for descriptor binding

// Environment types
enum class EnvSystem
{
  eSky,
  eHdr,
};

// Output image types
enum OutputImage
{
  eResultImage = 0,        // Output image (RGBA32)
  eSelectImage,            // Selection image (R32)
  eDlssAlbedo = 2,         // Diffuse albedo (RGBA8)
  eDlssSpecAlbedo,         // Specular albedo (RGBA32)
  eDlssNormalRoughness,    // Normal and roughness (RGBA32)
  eDlssMotion,             // Motion (RGBA32)
  eDlssDepth,              // Depth (R32)
  eDlssSpecularHitDist,    // Specular hit distance (R16F)
  eOptixAlbedoNormal = 2,  // Albedo/encoded normal (RGBA32)
};

// Binding points for descriptors
enum BindingPoints
{
  eTlas = 0,      // Top level acceleration structure
  eOutImages,     // Output image (RGBA32); eSelectImage slot = ObjectID in .r (R32_SFLOAT)
  eOutDepth,      // Scene depth output (D32_SFLOAT as r32f storage image)
  eTextures,      // glTF material textures (bindless array)
  eTexturesCube,  // Prefiltered HDR env cubemaps (HDR_DIFFUSE_INDEX, HDR_GLOSSY_INDEX)
  eTexturesHdr,   // 2D IBL/transmission textures (HDR_{IMAGE,LUT,SHEEN,OPAQUE}_INDEX)
};

// Fixed resolution of the opaque-pass color capture used for screen-space transmission.
// Matches the Khronos reference (1024x1024). Square so the LOD = log2(size) is unambiguous.
#define OPAQUE_COLOR_SIZE 1024

enum OptixBindingPoints
{
  eInRgba = 0,      // Incoming image (RGBA32)
  eInAlbedoNormal,  // Incoming albedo/normal (RGBA32)
  eOutRgba,         // Outgoing image in buffer
  eOutAlbedo,       // Outgoing albedo in buffer
  eOutNormal,       // Outgoing normal in buffer
};


// Binding points for descriptors
enum SilhouetteBindings
{
  eObjectID,          // In: the object ID image (R32_UINT)
  eRGBAIImage,        // Out: the output image
  eSelectionBitMask,  // In: storage buffer of uint32_t (one bit per render node)
};

enum Visualization
{
  eRendered,
  eBaseColor,
  eMetallic,
  eRoughness,
  eNormalShading,
  eNormalGeometric,
  eTangent,
  eBitangent,
  eEmissive,
  eOpacity,
  eTexCoord0,
  eTexCoord1,
  eClay,
  eTriangleID,
  eFaceOrientation,
  // Khronos glTF-Sample-Renderer DEBUG_* parity (subset — only the modes that map to fields
  // already populated in PbrMaterial; otherwise we render the base color as a fallback).
  eOcclusion,
  eClearcoatFactor,
  eClearcoatRoughness,
  eClearcoatNormal,
  eSheenColor,
  eSheenRoughness,
  eSpecularFactor,
  eSpecularColor,
  eTransmissionFactor,
  eIridescenceFactor,
  eIridescenceThickness,
  eAnisotropyStrength,
  eDiffuseTransmissionFactor,
  eDiffuseTransmissionColor,
};


// Bit flags for SceneFrameInfo::flags
enum SceneFrameInfoFlags
{
  eSceneIsOrthographic             = 1 << 0,
  eSceneUseSolidBackground         = 1 << 1,
  eSceneUseHdrEnvironment          = 1 << 2,
  eSceneUseInfinitePlane           = 1 << 3,
  eSceneInfinitePlaneShadowCatcher = 1 << 4,
};

// Camera info
struct SceneFrameInfo
{
  float4x4      viewMatrix;          // View matrix
  float4x4      projInv;             // Inverse projection matrix
  float4x4      viewInv;             // Inverse view matrix
  float4x4      viewProjMatrix;      // View-projection matrix (P*V)
  float4x4      prevMVP;             // Previous view-projection matrix
  float2        jitter;              // DLSS sub-pixel jitter in pixel units, range [-0.5, +0.5]
  float2        imageSize;           // Render extent in pixels (W,H)
  int           flags = 0;           // Bit flags: see SceneFrameInfoFlags
  float         envRotation;         // Environment rotation (used for the HDR)
  float         envBlur;             // Level of blur for the environment map (0.0: no blur, 1.0: full blur)
  float         envIntensity = 1.f;  // Environment intensity
  float3        backgroundColor;     // Background color when using solid background
  Visualization visualization             = Visualization::eRendered;  // Visualization mode
  float         infinitePlaneDistance     = 0;
  float3        infinitePlaneBaseColor    = float3(0.5, 0.5, 0.5);  // Default gray color
  float         infinitePlaneMetallic     = 0.0;                    // Default non-metallic
  float         infinitePlaneRoughness    = 0.5;                    // Default medium roughness
  float         shadowCatcherDarkenAmount = 0.0;  // Non-physical shadow darkening (precomputed from darkness slider)
};

enum PathtracerFlags
{
  ePtUseDlss          = 1 << 0,
  ePtUseOptixDenoiser = 1 << 1,
  ePtFirstFrame       = 1 << 2,
};


// Push constant
struct PathtracePushConstant
{
  int                    maxDepth              = 5;     // Maximum depth of the ray
  int                    frameCount            = 0;     // Frame number
  float                  fireflyClampThreshold = 10.f;  // Firefly clamp threshold
  float                  texGradScale          = 1.f;   // Ray-footprint gradient scale
  int                    numSamples            = 1;     // Number of samples per pixel per frame
  int                    totalSamples          = 0;     // Total samples accumulated so far
  float                  focalDistance         = 0.0f;  // Focal distance for depth of field
  float                  aperture              = 0.0f;  // Aperture for depth of field
  int                    flags                 = 0;     // Bit flags: ePtUseDlss | ePtUseOptixDenoiser | ePtFirstFrame
  float                  pixelAngle = 0.0f;    // Angular size of one pixel (radians) for ray-cone footprint LOD
  float2                 mouseCoord = {0, 0};  // Mouse coordinates (use for debug)
  SceneFrameInfo*        frameInfo;            // Camera info (incl. SceneFrameInfo::jitter when DLSS is active)
  SkyPhysicalParameters* skyParams;            // Sky physical parameters
  GltfScene*             gltfScene;            // GLTF scene
};

// Push constant
struct RasterPushConstant
{
  int                    materialID       = 0;       // Material used by the rendering instance
  int                    renderNodeID     = 0;       // Node used by the rendering instance
  int                    renderPrimID     = 0;       // Primitive used by the rendering instance
  int                    opaqueColorReady = 0;       // 1 = transmission framebuffer ready (mip chain valid)
  float2                 mouseCoord       = {0, 0};  // Mouse coordinates (use for debug)
  SceneFrameInfo*        frameInfo;                  // Camera info (incl. viewProjMatrix, prevMVP, jitter)
  SkyPhysicalParameters* skyParams;                  // Sky physical parameters
  GltfScene*             gltfScene;                  // GLTF scene
};


struct SilhouettePushConstant
{
  float3 color;
  uint   selectionBitMaskWordCount;  // Number of uint32 words in selection bitmask (for bounds check)
};

NAMESPACE_SHADERIO_END()

#endif  // HOST_DEVICE_H
