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

//
// Builds Slang `-D` macro lists for shared and scene-optimal shader compiles.
// Consumes `SceneFeatureSet` from scene_feature_detection and is called from the path
// tracer and rasterizer before `SlangCompiler::compileFile()`.
//

#include "scene_shader_macros.hpp"

namespace nvvkgltf {

namespace {

// Push `name=0` or `name=1` so optimal recompiles do not rely on header defaults alone.
void pushExplicit(std::vector<std::pair<std::string, std::string>>& macros, const char* name, bool used)
{
  macros.push_back({name, used ? "1" : "0"});
}

}  // namespace

void applyCommonShaderMacros(nvslang::SlangCompiler& compiler)
{
#if defined(USE_DLSS)
  compiler.addMacro({"HAS_DLSS_MOTION", "1"});
#endif
}

void appendPathTracerOptimalMacros(std::vector<std::pair<std::string, std::string>>& macros, const SceneFeatureSet& features)
{
  pushExplicit(macros, "GLTF_USE_TRANSMISSION", features.transmission);
  pushExplicit(macros, "GLTF_USE_VOLUME", features.volume);
  pushExplicit(macros, "GLTF_USE_VOLUME_SCATTER", features.volumeScatter);
  pushExplicit(macros, "GLTF_USE_CLEARCOAT", features.clearcoat);
  pushExplicit(macros, "GLTF_USE_IRIDESCENCE", features.iridescence);
  pushExplicit(macros, "GLTF_USE_ANISOTROPY", features.anisotropy);
  pushExplicit(macros, "GLTF_USE_SHEEN", features.sheen);
  pushExplicit(macros, "GLTF_USE_DISPERSION", features.dispersion);
  pushExplicit(macros, "GLTF_USE_DIFFUSE_TRANSMISSION", features.diffuseTransmission);
  pushExplicit(macros, "GLTF_USE_UNLIT", features.unlit);
  pushExplicit(macros, "GLTF_USE_SPECULAR", features.specular);
  pushExplicit(macros, "GLTF_USE_IOR", features.ior);
  pushExplicit(macros, "GLTF_USE_SPECULAR_GLOSSINESS", features.specularGlossiness);
  pushExplicit(macros, "GLTF_USE_TEXTURE_TRANSFORM", features.textureTransform);

  macros.push_back({"USE_DLSS_SHADER", features.dlssGuide ? "1" : "0"});
}

}  // namespace nvvkgltf
