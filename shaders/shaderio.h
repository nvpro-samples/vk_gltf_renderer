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
#include "sky_omi_shaderio.h.slang"
// Bruneton atmosphere parameters. Its own header rather than part of this bundle: only the sky
// subsystem needs it, and it carries an upstream BSD-3 notice that should stay with its content.
#include "sky_bruneton_io.h.slang"
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

// Environment types.
//
// Values are explicit and appended, never reordered: `envSystem` is persisted as a raw integer in
// the .ini and is part of the command-line and MCP surface, so inserting a value would silently
// remap every existing user's setting. The UI shows them in a different, more readable order
// through an explicit label/value table rather than a positional combo string.
enum class EnvSystem
{
  eSky      = 0,  // Procedural physical sky
  eHdr      = 1,  // Lat-long HDR image
  eNone     = 2,  // No environment: no sky/HDR lighting or background (black backdrop)
  ePlain    = 3,  // OMI_environment_sky `plain`: one authored solid color
  eGradient = 4,  // OMI_environment_sky `gradient`: authored bottom/horizon/top bands + sun
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
  eNrMask,                 // NR per-material control mask (RGBA16F)
  eOptixAlbedoNormal = 2,  // Albedo/encoded normal (RGBA32)
};

// Binding points for descriptors
enum BindingPoints
{
  eTlas = 0,      // Top level acceleration structure
  eOutImages,     // Output image (RGBA32); eSelectImage slot = ObjectID in .r (R32_SFLOAT)
  eOutDepth,      // Scene depth output (D32_SFLOAT as r32f storage image)
  eTextures,      // glTF material images (bindless SAMPLED_IMAGE array; indexed by GltfTextureInfo.index)
  eTexturesCube,  // Prefiltered HDR env cubemaps (HDR_DIFFUSE_INDEX, HDR_GLOSSY_INDEX)
  eTexturesHdr,   // 2D IBL/transmission textures (HDR_{IMAGE,LUT,SHEEN,OPAQUE}_INDEX)
  eSamplers,      // glTF samplers (bindless SAMPLER array; indexed by GltfTextureInfo.samplerIndex, slot 0 = default)
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
  eOpacityMicromap,
};


// Bit flags for SceneFrameInfo::flags
enum SceneFrameInfoFlags
{
  eSceneIsOrthographic             = 1 << 0,
  eSceneUseSolidBackground         = 1 << 1,
  eSceneUseHdrEnvironment          = 1 << 2,
  eSceneUseInfinitePlane           = 1 << 3,
  eSceneInfinitePlaneShadowCatcher = 1 << 4,
  eSceneUseNoEnvironment           = 1 << 5,  // No sky/HDR environment lighting or background
};

// Camera info
// The solar disk the Bruneton bake leaves out of its image, written by that bake and read by the
// background passes. Kept out of SceneFrameInfo's own storage because the value is produced on the
// GPU -- the transmittance toward the sun comes from the atmosphere LUTs -- and the host never
// learns it. See publishSunDisk() in shaders/env_bake.slang.
struct SkySunDisk
{
  float3 radiance;          // already in the baked image's units
  float  cosAngularRadius;  // cos of the sun's angular radius; the disk test is a dot product
  float  angularRadius;     // radians. Stored rather than derived so no one pays an acos per ray
  // Cosine of the zenith angle at which the ground starts, for the observer the sky was baked
  // from. Rays below it end on the planet, so the sun is behind it and must not be drawn.
  float horizonMu;
  // Distance of that observer from the planet's centre, in km. A per-ray evaluation of the sky has
  // to stand exactly where the bake stood -- see sky_bruneton_radiance.h.slang.
  float observerRadius;
  float _pad;
};

struct SceneFrameInfo
{
  float4x4 viewMatrix;      // View matrix
  float4x4 projInv;         // Inverse projection matrix
  float4x4 viewInv;         // Inverse view matrix
  float4x4 viewProjMatrix;  // View-projection matrix (P*V)
  float4x4 prevMVP;         // Previous view-projection matrix
  float2   jitter;          // DLSS sub-pixel jitter in pixel units, range [-0.5, +0.5]
  float2   imageSize;       // Render extent in pixels (W,H)
  int      flags = 0;       // Bit flags: see SceneFrameInfoFlags
  float4 envRotation;  // Environment orientation: unit quaternion (x, y, z, w), environment frame -> world. Lookups use quatRotateInverse(envRotation, worldDir)
  float         envBlur;             // Level of blur for the environment map (0.0: no blur, 1.0: full blur)
  float         envIntensity = 1.f;  // Environment intensity
  float3        backgroundColor;     // Background color when using solid background
  Visualization visualization             = Visualization::eRendered;  // Visualization mode
  float         infinitePlaneDistance     = 0;
  float3        infinitePlaneBaseColor    = float3(0.5, 0.5, 0.5);  // Default gray color
  float         infinitePlaneMetallic     = 0.0;                    // Default non-metallic
  float         infinitePlaneRoughness    = 0.5;                    // Default medium roughness
  float         shadowCatcherDarkenAmount = 0.0;  // Non-physical shadow darkening (precomputed from darkness slider)

  // Background field for the authored sky types. This is what a camera ray -- and a ray leaving a
  // mirror -- sees, and it is evaluated analytically per ray rather than read from the baked
  // lat-long: the bake is the *lighting* field and deliberately has no sun in it. Ignored unless
  // `envSkyType` names an authored type.
  //
  // The solar disk for the physical sky, or null for every other environment -- which is also how
  // a background pass knows whether to draw one. `sunDirection` is the world-space direction the
  // bake used, so the disk lands on the aureole the bake drew around it.
  SkySunDisk* sunDisk      = nullptr;
  float3      sunDirection = {0.0F, 1.0F, 0.0F};
  // 1 when the renderer supplies the sun as a light of its own: the sky has one and the scene
  // carries no marked sun light. It is not a glTF node -- nothing to save, nothing to undo, and no
  // scene edit as a side effect of picking an environment.
  int rendererSunLight = 0;
  // Index into the scene's punctual lights of the light the sky's sun *is*, or -1. Set only for a
  // light the scene marks as the sky's sun (gltf_environment_sky.hpp): that one is driven by the
  // sky, every other light is used exactly as authored.
  int sunLightIndex = -1;

  // `envSkyType` is a SkyType, or -1 when the environment is not an authored sky (a loaded HDR,
  // the procedural sky, or none) and the existing background paths apply instead.
  int              envSkyType = -1;
  SkyOmiParameters skyOmi;

  // Aerial perspective -- the air between the camera and a surface. Physical sky only; the gate
  // is `sunDisk != nullptr`, the same one the solar disk uses, so there is no second flag to
  // keep in step with the first. See shaders/aerial_perspective.h.slang.
  //
  // The scale is a *distance* multiplier on top of glTF's metre rather than an opacity: 2.0 says
  // the scene is twice as big, which stays inside the model. 0 switches the term off.
  //
  // **Appended, not inserted.** These two first went in beside `envSkyType`, where the host and
  // Slang disagreed about the resulting layout and every field read back as garbage -- silently,
  // because a wrong float is still a float. Nothing in this struct may be inserted into; new
  // fields go on the end.
  float aerialPerspectiveScale = 1.0F;
  // Aerial perspective's eye, in km above the planet surface. Not the bake's observer: that one is
  // fixed, while this follows the camera -- see GltfRenderer::aerialPerspectiveEyeAltitudeKm(). The
  // bake's is SkySunDisk::observerRadius.
  float observerAltitudeKm = 0.3F;

  // Illuminance of the gradient sky's sun, before `envIntensity`. Unused by every other
  // environment: the physical sky's sun is measured by the bake and read from `sunDisk`, and no
  // other sky has one.
  //
  // It is computed on the host rather than in makeSkySunLight() because the gradient sky's sun has
  // no radiance to convert -- `sunColor` says what the disk is drawn in, not how bright it is --
  // so the number comes from integrating the dome it sits under. See
  // GltfRenderer::gradientSunIlluminance(). (Appended, per the note above.)
  float3 skySunIlluminance = {0.0F, 0.0F, 0.0F};

  // Resolution of the environment's alias table -- HdrIbl::getSamplingGrid() -- which a baked sky
  // builds coarser than its image. environmentSample() needs it to turn an alias entry back into a
  // direction; the image's own size would be wrong. Two scalars rather than a uint2, so the host and
  // Slang cannot disagree about alignment after the float3 above. (Appended, per the note above.)
  uint envSamplingGridWidth  = 1;
  uint envSamplingGridHeight = 1;
};

enum PathtracerFlags
{
  ePtUseDlss            = 1 << 0,
  ePtUseOptixDenoiser   = 1 << 1,
  ePtFirstFrame         = 1 << 2,
  ePtShadowTransmission = 1 << 3,  // Biased: shadow rays pass straight through transmissive surfaces
};


// Push constant
struct PathtracePushConstant
{
  int             maxDepth              = 5;       // Maximum depth of the ray
  int             frameCount            = 0;       // Frame number
  float           fireflyClampThreshold = 10.f;    // Firefly clamp threshold
  float           texGradScale          = 0.f;     // Ray-footprint gradient scale
  int             numSamples            = 1;       // Number of samples per pixel per frame
  int             totalSamples          = 0;       // Total samples accumulated so far
  float           focalDistance         = 0.0f;    // Focal distance for depth of field
  float           aperture              = 0.0f;    // Aperture for depth of field
  int             flags                 = 0;       // Bit flags: see PathtracerFlags
  float           pixelAngle            = 0.0f;    // Angular size of one pixel (radians) for ray-cone footprint LOD
  float2          mouseCoord            = {0, 0};  // Mouse coordinates (use for debug)
  uint2           renderSize = {0, 0};          // Traced region: smaller than the output images under OptiX 2x upscale
  SceneFrameInfo* frameInfo;                    // Camera info (incl. SceneFrameInfo::jitter when DLSS is active)
  GltfScene*      gltfScene;                    // GLTF scene
  float4x4*       prevRenderNodeObjectToWorld;  // #DLSS instance motion: previous-frame objectToWorld per render node
};

// Push constant
struct RasterPushConstant
{
  int             materialID       = 0;       // Material used by the rendering instance
  int             renderNodeID     = 0;       // Node used by the rendering instance
  int             renderPrimID     = 0;       // Primitive used by the rendering instance
  int             opaqueColorReady = 0;       // 1 = transmission framebuffer ready (mip chain valid)
  float2          mouseCoord       = {0, 0};  // Mouse coordinates (use for debug)
  SceneFrameInfo* frameInfo;                  // Camera info (incl. viewProjMatrix, prevMVP, jitter)
  GltfScene*      gltfScene;                  // GLTF scene
};


// Background pass for authored skies (rasterizer). The sky parameters ride in SceneFrameInfo
// rather than in the push constant: SkyOmiParameters plus a 4x4 matrix would exceed the 128-byte
// push-constant size Vulkan guarantees, and the frame info already carries them for the path
// tracer, so both paths read the same numbers from the same place.
enum SkyBackgroundBindings
{
  eSkyBackgroundOutImage = 0,
};

// What the rasterizer's background pass should do with the pixel it owns.
enum SkyBackgroundMode
{
  eSkyBackgroundAuthored = 0,  // Write the authored sky's radiance (plain, gradient)
  eSkyBackgroundSunDisk  = 1,  // Add the solar disk on top of a backdrop already drawn
};

struct SkyBackgroundPushConstant
{
  SceneFrameInfo* frameInfo;  // Sky type + parameters + intensity/rotation, and the camera
  int             mode;       // SkyBackgroundMode
  int             _pad[3];
};

struct SilhouettePushConstant
{
  float3 color;
  uint   selectionBitMaskWordCount;  // Number of uint32 words in selection bitmask (for bounds check)
};

NAMESPACE_SHADERIO_END()

#endif  // HOST_DEVICE_H
