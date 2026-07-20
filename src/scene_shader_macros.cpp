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

#include <array>

namespace nvvkgltf {

namespace {

// Push `name=0` or `name=1` so optimal recompiles do not rely on header defaults alone.
void pushExplicit(std::vector<std::pair<std::string, std::string>>& macros, const char* name, bool used)
{
  macros.push_back({name, used ? "1" : "0"});
}

struct FeatureMacro
{
  SceneFeatureSet::Feature feature;
  const char*              macro;
};

constexpr std::array<FeatureMacro, SceneFeatureSet::kExtensionFeatureCount> kPathTracerFeatureMacros{{
    {SceneFeatureSet::eTransmission, "GLTF_USE_TRANSMISSION"},
    {SceneFeatureSet::eVolume, "GLTF_USE_VOLUME"},
    {SceneFeatureSet::eVolumeScatter, "GLTF_USE_VOLUME_SCATTER"},
    {SceneFeatureSet::eClearcoat, "GLTF_USE_CLEARCOAT"},
    {SceneFeatureSet::eIridescence, "GLTF_USE_IRIDESCENCE"},
    {SceneFeatureSet::eAnisotropy, "GLTF_USE_ANISOTROPY"},
    {SceneFeatureSet::eSheen, "GLTF_USE_SHEEN"},
    {SceneFeatureSet::eDispersion, "GLTF_USE_DISPERSION"},
    {SceneFeatureSet::eDiffuseTransmission, "GLTF_USE_DIFFUSE_TRANSMISSION"},
    {SceneFeatureSet::eRetroreflection, "GLTF_USE_RETROREFLECTION"},
    {SceneFeatureSet::eUnlit, "GLTF_USE_UNLIT"},
    {SceneFeatureSet::eSpecular, "GLTF_USE_SPECULAR"},
    {SceneFeatureSet::eIor, "GLTF_USE_IOR"},
    {SceneFeatureSet::eSpecularGlossiness, "GLTF_USE_SPECULAR_GLOSSINESS"},
    {SceneFeatureSet::eTextureTransform, "GLTF_USE_TEXTURE_TRANSFORM"},
}};

}  // namespace

void applyCommonShaderMacros(nvslang::SlangCompiler& compiler)
{
#if defined(USE_DLSS)
  compiler.addMacro({"HAS_DLSS_MOTION", "1"});
#endif
}

void appendPathTracerOptimalMacros(std::vector<std::pair<std::string, std::string>>& macros, const SceneFeatureSet& features)
{
  for(const FeatureMacro& entry : kPathTracerFeatureMacros)
  {
    pushExplicit(macros, entry.macro, features.has(entry.feature));
  }
}

void appendPathTracerDlssShaderMacro(std::vector<std::pair<std::string, std::string>>& macros, const SceneFeatureSet& features, bool dlssActive)
{
  // Two independent, runtime-driven compile gates (both refreshed every frame, independent of
  // optimal mode):
  //   USE_GUIDE_SHADER : guide-buffer capture (albedo/normal/etc.). Needed by DLSS *and* OptiX,
  //                      so it follows eDlssGuide == dlssGuideRequired() (DLSS or OptiX active).
  //   USE_DLSS_SHADER  : DLSS-specific path. True only when DLSS is actually active. Drives the
  //                      per-frame multisample-loop gate (loop compiled out for DLSS to cut
  //                      register pressure; compiled in for OptiX / no-denoiser so samples work).
#if defined(USE_DLSS) || defined(USE_OPTIX_DENOISER)
  const bool useGuideShader = features.has(SceneFeatureSet::eDlssGuide);
#else
  const bool useGuideShader = false;
#endif
#if defined(USE_DLSS)
  const bool useDlssShader = dlssActive;
#else
  const bool useDlssShader = false;
#endif
  pushExplicit(macros, "USE_GUIDE_SHADER", useGuideShader);
  pushExplicit(macros, "USE_DLSS_SHADER", useDlssShader);
}

}  // namespace nvvkgltf
