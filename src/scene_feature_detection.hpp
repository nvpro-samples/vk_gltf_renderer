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

#include <bitset>
#include <cstddef>
#include <string>
#include <vector>

namespace tinygltf {
class Model;
struct Material;
}  // namespace tinygltf

namespace nvvkgltf {

/*-------------------------------------------------------------------------------------------------
# struct nvvkgltf::SceneFeatureSet

> Bitmask of which `KHR_materials_*` (+ `KHR_texture_transform`) extensions are used
> by the loaded glTF scene, plus other shader-level toggles that scene-aware variant
> compilation can flip (currently: whether DLSS/OptiX guide buffers need to exist).
>
> Used by the path tracer's optimal-mode shader rebuild: when "Optimize shader for
> current scene" is enabled, the renderer detects which extensions actually appear
> in the scene's materials, then recompiles `gltf_pathtrace.slang` with `-DGLTF_USE_X=0`
> or `1` for every extension gate (explicit emission; defaults resolve to `MAT_EXT_*`), plus
> `-DUSE_DLSS_SHADER=0/1` when guide
> buffers are not needed. This shrinks the megakernel and material-eval paths without
> changing `GltfShadeMaterial` buffer layout.
>
> NOTE: this does NOT change the host-side `GltfShadeMaterial` struct layout. Host
> `MAT_EXT_*` flags stay at build time (default all-on). Runtime optimal mode must
> never pass `MAT_EXT_X=0` to Slang recompiles.
--------------------------------------------------------------------------------------------------*/
struct SceneFeatureSet
{
  enum Feature : std::size_t
  {
    // KHR_materials_* usage bits. Names mirror MAT_EXT_*; runtime optimal mode drives
    // GLTF_USE_* gates (not MAT_EXT_*) from these flags.
    eTransmission,
    eVolume,
    eVolumeScatter,
    eClearcoat,
    eIridescence,
    eAnisotropy,
    eSheen,
    eDispersion,
    eDiffuseTransmission,
    eRetroreflection,
    eUnlit,
    eSpecular,
    eIor,
    eSpecularGlossiness,
    eTextureTransform,

    // Shader-side guide-buffer code (GuideScratch / dlssGetClearGlassGuideAlbedo / etc.).
    // True when the active denoiser (DLSS or OptiX) wants the path tracer to populate
    // first-hit guide buffers. The build-time CMake gate also controls this via
    // USE_GUIDE_SHADER, but runtime variant rebuild can be more aggressive (e.g. turn
    // off when DLSS is compiled in but the user has disabled the denoiser).
    eDlssGuide,

    eCount
  };

  static constexpr std::size_t kFeatureCount          = static_cast<std::size_t>(eCount);
  static constexpr std::size_t kExtensionFeatureCount = static_cast<std::size_t>(eDlssGuide);

  void enable(Feature feature) { m_bits.set(static_cast<std::size_t>(feature)); }
  void set(Feature feature, bool enabled) { m_bits.set(static_cast<std::size_t>(feature), enabled); }
  bool has(Feature feature) const { return m_bits.test(static_cast<std::size_t>(feature)); }

  // True when `this` set already covers every feature `other` needs. Used to skip
  // unnecessary recompiles when a smaller scene is merged into the current one.
  bool isSupersetOf(const SceneFeatureSet& other) const;

  bool operator==(const SceneFeatureSet& o) const;
  bool operator!=(const SceneFeatureSet& o) const { return !(*this == o); }

  // Comma-separated string for logs / UI feedback (e.g. "transmission,volume,clearcoat").
  // Empty string when no extensions are used.
  std::string toString() const;

  // Number of material/texture extension gates that are currently unused.
  // Excludes non-extension shader gates such as dlssGuide, which are optimized separately.
  // 0 means every counted extension gate is used, not that optimal mode has no remaining shader gates.
  int unusedExtensionCount() const;

private:
  std::bitset<kFeatureCount> m_bits{};
};

// Walk a glTF material list and detect which gated extensions appear. A material is
// flagged as using an extension if its `extensions` map contains the corresponding
// KHR_materials_* key. KHR_texture_transform is checked on core material texture
// infos and texture-info objects nested inside material extension payloads.
//
// `model` is retained for signature compatibility and may be nullptr.
SceneFeatureSet detectSceneFeatures(const std::vector<tinygltf::Material>& materials, const tinygltf::Model* model = nullptr);

// Convenience: detect features from a whole model.
SceneFeatureSet detectSceneFeatures(const tinygltf::Model& model);

}  // namespace nvvkgltf
