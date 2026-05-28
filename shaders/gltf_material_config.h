/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*-------------------------------------------------------------------------------------------------
MAT_EXT_* preprocessor gates for the local GltfShadeMaterial fork.

Each flag controls whether the corresponding glTF extension is baked into the material
struct and evaluated by evaluateMaterial(). All default to 1 (enabled) which reproduces
the original nvpro_core2 behavior byte-for-byte. Flip any flag to 0 (via -D or by
predefining before including this header) to shrink the struct and skip that extension's
work in the shader.

Flags must match between host (C++) and device (Slang) compilation of the same build,
otherwise the material buffer layout will disagree.
-------------------------------------------------------------------------------------------------*/

#ifndef GLTF_MATERIAL_CONFIG_H
#define GLTF_MATERIAL_CONFIG_H

// default for all MAT_EXT_* flags if not overridden by -D or predefinition0
#ifndef MAT_EXT_VAL
#define MAT_EXT_VAL 1
#endif

#ifndef MAT_EXT_SPECULAR_GLOSSINESS
#define MAT_EXT_SPECULAR_GLOSSINESS MAT_EXT_VAL  // KHR_materials_pbrSpecularGlossiness
#endif
#ifndef MAT_EXT_IOR
#define MAT_EXT_IOR MAT_EXT_VAL  // KHR_materials_ior
#endif
#ifndef MAT_EXT_TRANSMISSION
#define MAT_EXT_TRANSMISSION MAT_EXT_VAL  // KHR_materials_transmission
#endif
#ifndef MAT_EXT_VOLUME
#define MAT_EXT_VOLUME MAT_EXT_VAL  // KHR_materials_volume (attenuation + thickness)
#endif
#ifndef MAT_EXT_VOLUME_SCATTER
#define MAT_EXT_VOLUME_SCATTER MAT_EXT_VAL  // KHR_materials_volume_scatter
#endif
#ifndef MAT_EXT_CLEARCOAT
#define MAT_EXT_CLEARCOAT MAT_EXT_VAL  // KHR_materials_clearcoat
#endif
#ifndef MAT_EXT_IRIDESCENCE
#define MAT_EXT_IRIDESCENCE MAT_EXT_VAL  // KHR_materials_iridescence
#endif
#ifndef MAT_EXT_ANISOTROPY
#define MAT_EXT_ANISOTROPY MAT_EXT_VAL  // KHR_materials_anisotropy
#endif
#ifndef MAT_EXT_SHEEN
#define MAT_EXT_SHEEN MAT_EXT_VAL  // KHR_materials_sheen
#endif
#ifndef MAT_EXT_DISPERSION
#define MAT_EXT_DISPERSION MAT_EXT_VAL  // KHR_materials_dispersion
#endif
#ifndef MAT_EXT_DIFFUSE_TRANSMISSION
#define MAT_EXT_DIFFUSE_TRANSMISSION MAT_EXT_VAL  // KHR_materials_diffuse_transmission
#endif
#ifndef MAT_EXT_UNLIT
#define MAT_EXT_UNLIT MAT_EXT_VAL  // KHR_materials_unlit
#endif
#ifndef MAT_EXT_SPECULAR
#define MAT_EXT_SPECULAR MAT_EXT_VAL  // KHR_materials_specular
#endif
#ifndef MAT_EXT_TEXTURE_TRANSFORM
#define MAT_EXT_TEXTURE_TRANSFORM                                                                                      \
  MAT_EXT_VAL  // KHR_texture_transform (adds float3x2 to every GltfTextureInfo, ~24B/slot)
#endif


#endif  // GLTF_MATERIAL_CONFIG_H
