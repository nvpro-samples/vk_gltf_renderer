/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*-------------------------------------------------------------------------------------------------
GLTF_USE_* behavior gates for glTF material evaluation and related shader paths.

These gates control whether extension work runs at shader compile time / constant-fold time.
They do NOT change GltfShadeMaterial or GltfTextureInfo layout; MAT_EXT_* remains the sole
layout authority and must match between host and device builds.

Default (non-optimal) compiles: each GLTF_USE_X defaults to MAT_EXT_X (typically all 1).

Optimal path-tracer recompile: host sets every GLTF_USE_X explicitly to 0 or 1 from
SceneFeatureSet. All shader behavior (material eval, path-tracer state, raster) uses
#if GLTF_USE_X only. MAT_EXT_* is for gltf_scene_io.h.slang struct layout (host/device
ABI), not behavioral #if guards.

Illegal: GLTF_USE_X enabled while MAT_EXT_X is disabled at build time (behavior on,
layout off). Each pair has a compile-time guard below.
-------------------------------------------------------------------------------------------------*/

#ifndef GLTF_EVAL_CONFIG_H
#define GLTF_EVAL_CONFIG_H

// Get the MAT_EXT_ values from the material config (Struct layout)
#include "gltf_material_config.h"

// Make by default the GLTF_USE_X values the same as the MAT_EXT_X values
// We split them, so that we can remove a functionality at compile time
// without changing the struct layout (GltfShadeMaterial).

#ifndef GLTF_USE_SPECULAR_GLOSSINESS
#define GLTF_USE_SPECULAR_GLOSSINESS MAT_EXT_SPECULAR_GLOSSINESS
#endif
#ifndef GLTF_USE_IOR
#define GLTF_USE_IOR MAT_EXT_IOR
#endif
#ifndef GLTF_USE_TRANSMISSION
#define GLTF_USE_TRANSMISSION MAT_EXT_TRANSMISSION
#endif
#ifndef GLTF_USE_VOLUME
#define GLTF_USE_VOLUME MAT_EXT_VOLUME
#endif
#ifndef GLTF_USE_VOLUME_SCATTER
#define GLTF_USE_VOLUME_SCATTER MAT_EXT_VOLUME_SCATTER
#endif
#ifndef GLTF_USE_CLEARCOAT
#define GLTF_USE_CLEARCOAT MAT_EXT_CLEARCOAT
#endif
#ifndef GLTF_USE_IRIDESCENCE
#define GLTF_USE_IRIDESCENCE MAT_EXT_IRIDESCENCE
#endif
#ifndef GLTF_USE_ANISOTROPY
#define GLTF_USE_ANISOTROPY MAT_EXT_ANISOTROPY
#endif
#ifndef GLTF_USE_SHEEN
#define GLTF_USE_SHEEN MAT_EXT_SHEEN
#endif
#ifndef GLTF_USE_DISPERSION
#define GLTF_USE_DISPERSION MAT_EXT_DISPERSION
#endif
#ifndef GLTF_USE_DIFFUSE_TRANSMISSION
#define GLTF_USE_DIFFUSE_TRANSMISSION MAT_EXT_DIFFUSE_TRANSMISSION
#endif
#ifndef GLTF_USE_RETROREFLECTION
#define GLTF_USE_RETROREFLECTION MAT_EXT_RETROREFLECTION
#endif
#ifndef GLTF_USE_UNLIT
#define GLTF_USE_UNLIT MAT_EXT_UNLIT
#endif
#ifndef GLTF_USE_SPECULAR
#define GLTF_USE_SPECULAR MAT_EXT_SPECULAR
#endif
#ifndef GLTF_USE_TEXTURE_TRANSFORM
#define GLTF_USE_TEXTURE_TRANSFORM MAT_EXT_TEXTURE_TRANSFORM
#endif

// Compile-time guards: behavior-on while layout-off is illegal.
// We cannot turn on a GLTF_USE_X if the MAT_EXT_X is disabled.
#if GLTF_USE_SPECULAR_GLOSSINESS && !MAT_EXT_SPECULAR_GLOSSINESS
#error GLTF_USE_SPECULAR_GLOSSINESS requires MAT_EXT_SPECULAR_GLOSSINESS
#endif
#if GLTF_USE_IOR && !MAT_EXT_IOR
#error GLTF_USE_IOR requires MAT_EXT_IOR
#endif
#if GLTF_USE_TRANSMISSION && !MAT_EXT_TRANSMISSION
#error GLTF_USE_TRANSMISSION requires MAT_EXT_TRANSMISSION
#endif
#if GLTF_USE_VOLUME && !MAT_EXT_VOLUME
#error GLTF_USE_VOLUME requires MAT_EXT_VOLUME
#endif
#if GLTF_USE_VOLUME_SCATTER && !MAT_EXT_VOLUME_SCATTER
#error GLTF_USE_VOLUME_SCATTER requires MAT_EXT_VOLUME_SCATTER
#endif
#if GLTF_USE_CLEARCOAT && !MAT_EXT_CLEARCOAT
#error GLTF_USE_CLEARCOAT requires MAT_EXT_CLEARCOAT
#endif
#if GLTF_USE_IRIDESCENCE && !MAT_EXT_IRIDESCENCE
#error GLTF_USE_IRIDESCENCE requires MAT_EXT_IRIDESCENCE
#endif
#if GLTF_USE_ANISOTROPY && !MAT_EXT_ANISOTROPY
#error GLTF_USE_ANISOTROPY requires MAT_EXT_ANISOTROPY
#endif
#if GLTF_USE_SHEEN && !MAT_EXT_SHEEN
#error GLTF_USE_SHEEN requires MAT_EXT_SHEEN
#endif
#if GLTF_USE_DISPERSION && !MAT_EXT_DISPERSION
#error GLTF_USE_DISPERSION requires MAT_EXT_DISPERSION
#endif
#if GLTF_USE_DIFFUSE_TRANSMISSION && !MAT_EXT_DIFFUSE_TRANSMISSION
#error GLTF_USE_DIFFUSE_TRANSMISSION requires MAT_EXT_DIFFUSE_TRANSMISSION
#endif
#if GLTF_USE_RETROREFLECTION && !MAT_EXT_RETROREFLECTION
#error GLTF_USE_RETROREFLECTION requires MAT_EXT_RETROREFLECTION
#endif
#if GLTF_USE_UNLIT && !MAT_EXT_UNLIT
#error GLTF_USE_UNLIT requires MAT_EXT_UNLIT
#endif
#if GLTF_USE_SPECULAR && !MAT_EXT_SPECULAR
#error GLTF_USE_SPECULAR requires MAT_EXT_SPECULAR
#endif
#if GLTF_USE_TEXTURE_TRANSFORM && !MAT_EXT_TEXTURE_TRANSFORM
#error GLTF_USE_TEXTURE_TRANSFORM requires MAT_EXT_TEXTURE_TRANSFORM
#endif

#endif  // GLTF_EVAL_CONFIG_H
