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

#pragma once

#include <string>
#include <utility>
#include <vector>

#include <nvslang/slang.hpp>

#include "scene_feature_detection.hpp"

namespace nvvkgltf {

/*-------------------------------------------------------------------------------------------------
# function applyCommonShaderMacros

> Reapply preprocessor macros that must exist for every file-based Slang compile on the shared
> `nvslang::SlangCompiler` instance (for example `HAS_DLSS_MOTION` when DLSS is enabled).
>
> Call after `clearMacros()` and before pass-specific or scene-optimal macros so a prior path
> tracer recompile cannot leave stale defines on the compiler used by the rasterizer.
-------------------------------------------------------------------------------------------------*/
void applyCommonShaderMacros(nvslang::SlangCompiler& compiler);

/*-------------------------------------------------------------------------------------------------
# function appendPathTracerOptimalMacros

> Append scene-aware optimal-mode macros for path tracer recompiles.
>
> When "Optimize shader for current scene" is on, every `GLTF_USE_*` gate is set explicitly to
> `"0"` or `"1"` from `SceneFeatureSet`, plus `USE_DLSS_SHADER`. Shader behavior uses
> `#if GLTF_USE_*` only; `MAT_EXT_*` remains the host/layout authority and must never be
> forced to `0` at runtime (that would change `GltfShadeMaterial` layout while the host
> uploads the all-on struct).
>
> Future backend option: once gates are constant branches inside the shader, behavior toggles
> can move to Vulkan specialization constants (see `gltf_pathtrace.slang` `USE_SER` /
> `USE_DLSS_TRANSP`) or Slang link-time constants to avoid per-scene Slang recompiles.
-------------------------------------------------------------------------------------------------*/
void appendPathTracerOptimalMacros(std::vector<std::pair<std::string, std::string>>& macros, const SceneFeatureSet& features);

}  // namespace nvvkgltf
