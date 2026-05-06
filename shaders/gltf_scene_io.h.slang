/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*-------------------------------------------------------------------------------------------------
Local fork of nvpro_core2/nvshaders/gltf_scene_io.h.slang.

Keeps the include guard DH_SCN_DESC_H identical to the upstream file so any transitive
include of the nvpro_core2 version (e.g. from nvshaders/light_contrib.h.slang) is a no-op
as long as this file is #included first in the translation unit.

Adds MAT_EXT_* preprocessor gates around every glTF extension's fields so the
GltfShadeMaterial struct can be shrunk for scenes that don't use those extensions.
With all flags = 1 (the default) the struct layout is byte-identical to upstream.
-------------------------------------------------------------------------------------------------*/

#ifndef DH_SCN_DESC_H
#define DH_SCN_DESC_H 1

#include "nvshaders/slang_types.h"
#include "gltf_material_config.h"

NAMESPACE_SHADERIO_BEGIN()

// This is the GLTF Node structure, but flattened
struct GltfRenderNode
{
  float4x4 objectToWorld;
  float4x4 worldToObject;
  int      materialID;
  int      renderPrimID;
};

// This is all the information about a vertex buffer
struct VertexBuffers
{
  float3* positions;
  float3* normals;
  uint*   colors;  // packed as 4 uint8_t
  float4* tangents;
  float2* texCoords[2];  // texCoords[0] = TEXCOORD_0, texCoords[1] = TEXCOORD_1
};

// This is the GLTF Primitive structure
struct GltfRenderPrimitive
{
  uint3*        indices;
  VertexBuffers vertexBuffer;
};


/*-------------------------------------------------------------------------------------------------
Common structures used for lights other than environment lighting.
-------------------------------------------------------------------------------------------------*/

// Light type
enum LightType
{
  eLightTypeNone,
  eLightTypeDirectional,
  eLightTypeSpot,
  eLightTypePoint,
};


// Light
// Defaults produce a dark directional light pointing roughly down and away; match what
// defaultLight() used to return. normalize()/radians() resolve to glm on host and to
// Slang builtins on device (via `using namespace glm` in slang_types.h).
struct GltfLight
{
  float3 direction = normalize(float3(0.0F, -0.7F, -0.7F));  // Direction of the light
  int    type      = LightType::eLightTypeDirectional;       // Type of the light

  float3 position = float3(5.0F, 5.0F, 5.0F);  // Position of the light
  float  radius   = 1.0F;                      // Radius of the light

  float3 color = float3(1.0F);  // Color of the light
  float intensity = 0.0F;  // illuminance (lm/m2) for directional lights, luminous intensity (lm/sr) for positional lights

  float angularSizeOrInvRange = radians(0.53F);  // angular size for directional lights, 1/range for spot and point lights
  float innerAngle = radians(10.0F);             // Inner angle of the light
  float outerAngle = radians(30.0F);             // Outer angle of the light
  int   _pad       = 0;                          // Padding
};


// alphaMode
enum AlphaMode
{
  eAlphaModeOpaque,
  eAlphaModeMask,
  eAlphaModeBlend,
};

// PBR shading model selector. Defaults to metallic-roughness; set to specular-glossiness
// when KHR_materials_pbrSpecularGlossiness is present (and MAT_EXT_SPECULAR_GLOSSINESS is on).
enum PbrModel
{
  ePbrModelMetallicRoughness,
  ePbrModelSpecularGlossiness,
};

// Texture info
// Total 32 bytes with MAT_EXT_TEXTURE_TRANSFORM = 1, 8 bytes when off.
struct GltfTextureInfo
{
#if MAT_EXT_TEXTURE_TRANSFORM
  float3x2 uvTransform = float3x2(1);  // KHR_texture_transform (24 bytes, 3x2 matrix)
#endif
  int index    = -1;  // 4 bytes
  int texCoord = 0;   // 4 bytes
};

/*-------------------------------------------------------------------------------------------------
GltfShadeMaterial

Field layout is organized as:
  1. Base (always-on) float/int block at the top.
  2. Extension float/int blocks, each wrapped in #if MAT_EXT_*.
  3. Base uint16_t texture slots.
  4. Extension uint16_t texture slots, each wrapped in #if MAT_EXT_*.
  5. Trailing uint64_t _pad to keep sizeof a multiple of 8 and lift struct alignment to 8
     for every MAT_EXT_* combination (enables LDG.64 for aligned field pairs in the BDA array).

This ensures the base block's offsets are stable regardless of which MAT_EXT_* flags are
enabled — e.g. occlusionStrength, doubleSided, and the base texture slots always sit at
the same position. Extension blocks are collapsed out of the struct when gated, shrinking
the total size. The layout is NOT identical to upstream nvpro_core2; that's a deliberate
trade-off for forkability (see docs/developer.md > Material System).
-------------------------------------------------------------------------------------------------*/
struct GltfShadeMaterial
{
  float4 pbrBaseColorFactor = float4(1);
  float3 emissiveFactor     = float3(0);
  float  normalTextureScale = 1;

  float pbrRoughnessFactor = 1;
  float pbrMetallicFactor  = 1;
  int   alphaMode          = AlphaMode::eAlphaModeOpaque;
  float alphaCutoff        = 0.5;

  float occlusionStrength = 1;
  int   doubleSided       = 0;

#if MAT_EXT_VOLUME
  float3 attenuationColor = float3(1);
#endif

#if MAT_EXT_IOR
  float ior = 1.5;
#endif

#if MAT_EXT_TRANSMISSION
  float transmissionFactor = 0;
#endif

#if MAT_EXT_VOLUME
  float thicknessFactor     = 0;  // KHR_materials_volume
  float attenuationDistance = 0;
#endif

#if MAT_EXT_CLEARCOAT
  float clearcoatFactor = 0;
#endif

#if MAT_EXT_SPECULAR
  float3 specularColorFactor = float3(1);
#endif

#if MAT_EXT_CLEARCOAT
  float clearcoatRoughness = 0;
#endif

#if MAT_EXT_SPECULAR
  float specularFactor = 0;
#endif

#if MAT_EXT_UNLIT
  int unlit = 0;
#endif

#if MAT_EXT_IRIDESCENCE
  float iridescenceFactor           = 0;
  float iridescenceThicknessMinimum = 100;
  float iridescenceThicknessMaximum = 400;
  float iridescenceIor              = 1.3f;
#endif

#if MAT_EXT_ANISOTROPY
  float2 anisotropyRotation = float2(0);
#endif

#if MAT_EXT_SHEEN
  float3 sheenColorFactor = float3(0);
#endif

#if MAT_EXT_ANISOTROPY
  float anisotropyStrength = 0;
#endif

#if MAT_EXT_SHEEN
  float sheenRoughnessFactor = 0;
#endif

#if MAT_EXT_DISPERSION
  float dispersion = 0;
#endif

#if MAT_EXT_SPECULAR_GLOSSINESS
  int pbrModel = PbrModel::ePbrModelMetallicRoughness;

  float4 pbrDiffuseFactor    = float4(1);
  float3 pbrSpecularFactor   = float3(1);
  float  pbrGlossinessFactor = 1;
#endif

#if MAT_EXT_DIFFUSE_TRANSMISSION
  float3 diffuseTransmissionColor  = float3(1);
  float  diffuseTransmissionFactor = 0;
#endif

#if MAT_EXT_VOLUME_SCATTER
  float3 multiscatterColor = float3(0);
  float  scatterAnisotropy = 0;
#endif

  // Texture infos (uint16_t, 2 bytes each). Index 0 is the sentinel "no texture" slot
  // (textureInfos[0] is reserved as invalid on the host), so all texture members default to 0.
  // Base (always-on) texture slots come first so their offsets within the texture section
  // are stable across MAT_EXT_* flag combinations.
  uint16_t pbrBaseColorTexture         = 0;
  uint16_t normalTexture               = 0;
  uint16_t pbrMetallicRoughnessTexture = 0;
  uint16_t emissiveTexture             = 0;
  uint16_t occlusionTexture            = 0;

#if MAT_EXT_TRANSMISSION
  uint16_t transmissionTexture = 0;
#endif

#if MAT_EXT_VOLUME
  uint16_t thicknessTexture = 0;
#endif

#if MAT_EXT_CLEARCOAT
  uint16_t clearcoatTexture          = 0;
  uint16_t clearcoatRoughnessTexture = 0;
  uint16_t clearcoatNormalTexture    = 0;
#endif

#if MAT_EXT_SPECULAR
  uint16_t specularTexture      = 0;
  uint16_t specularColorTexture = 0;
#endif
#if MAT_EXT_IRIDESCENCE
  uint16_t iridescenceTexture          = 0;
  uint16_t iridescenceThicknessTexture = 0;
#endif

#if MAT_EXT_ANISOTROPY
  uint16_t anisotropyTexture = 0;
#endif

#if MAT_EXT_SHEEN
  uint16_t sheenColorTexture     = 0;
  uint16_t sheenRoughnessTexture = 0;
#endif

#if MAT_EXT_SPECULAR_GLOSSINESS
  uint16_t pbrDiffuseTexture            = 0;
  uint16_t pbrSpecularGlossinessTexture = 0;
#endif

#if MAT_EXT_DIFFUSE_TRANSMISSION
  uint16_t diffuseTransmissionTexture      = 0;
  uint16_t diffuseTransmissionColorTexture = 0;
#endif

  // Trailing 8-byte explicit pad. Two reasons for uint64_t (rather than just uint32_t):
  //  1. It guarantees sizeof(GltfShadeMaterial) is a multiple of 8 for every MAT_EXT_*
  //     combination (struct alignment follows max member alignment, which uint64_t makes 8).
  //  2. With the base BDA pointer already 8-aligned (Vulkan guarantees this for storage
  //     buffers), an 8-multiple stride keeps every `materials[i]` 8-aligned, letting the
  //     driver emit LDG.64 for adjacent 4-byte field pairs across all elements.
  uint64_t _pad = 0;
};

// The scene description is a pointer to the material, render node and render primitive
// The buffers are all arrays of the above structures
struct GltfScene
{
  GltfShadeMaterial*   materials;
  GltfTextureInfo*     textureInfos;
  GltfRenderNode*      renderNodes;
  GltfRenderPrimitive* renderPrimitives;
  GltfLight*           lights;
  int                  numLights;  // number of punctual lights
};


NAMESPACE_SHADERIO_END()

#endif
