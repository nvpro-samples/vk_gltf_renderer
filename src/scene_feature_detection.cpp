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
// Detects which glTF material extensions are present in the current scene and
// folds them into a compact SceneFeatureSet used by path-tracer shader variant
// selection ("optimize shader for current scene" mode).
//

#include "scene_feature_detection.hpp"

#include <tinygltf/tiny_gltf.h>

#include "tinygltf_utils.hpp"

namespace nvvkgltf {

namespace {

// Probe a tinygltf extensions map for a given KHR extension name. We use
// tinygltf::utils::hasElementName so we follow the same key-lookup convention
// as the rest of the codebase (see src/gltf_material_cache.cpp).
inline bool hasExt(const tinygltf::ExtensionMap& exts, const char* name)
{
  return tinygltf::utils::hasElementName(exts, name);
}

// KHR_texture_transform can live on any tinygltf::TextureInfo or
// NormalTextureInfo or OcclusionTextureInfo; they all carry .extensions.
template <typename TexInfo>
inline bool textureInfoHasTransform(const TexInfo& ti)
{
  return ti.index >= 0 && hasExt(ti.extensions, KHR_TEXTURE_TRANSFORM_EXTENSION_NAME);
}

inline bool textureInfoValueHasTransform(const tinygltf::Value& v)
{
  if(!v.IsObject() || !v.Has("index") || !v.Get("index").IsNumber() || !v.Has("extensions") || !v.Get("extensions").IsObject())
    return false;

  tinygltf::TextureInfo ti;
  ti.index      = v.Get("index").GetNumberAsInt();
  ti.extensions = v.Get("extensions").Get<tinygltf::Value::Object>();
  return textureInfoHasTransform(ti);
}

bool valueTreeHasTextureTransform(const tinygltf::Value& v)
{
  if(v.IsObject())
  {
    if(textureInfoValueHasTransform(v))
      return true;

    for(const auto& entry : v.Get<tinygltf::Value::Object>())
    {
      if(valueTreeHasTextureTransform(entry.second))
        return true;
    }
  }
  else if(v.IsArray())
  {
    for(const auto& child : v.Get<tinygltf::Value::Array>())
    {
      if(valueTreeHasTextureTransform(child))
        return true;
    }
  }
  return false;
}

// Returns true if any texture slot inside a material uses KHR_texture_transform.
inline bool materialHasTextureTransform(const tinygltf::Material& m)
{
  // PBR metallic-roughness slots
  if(textureInfoHasTransform(m.pbrMetallicRoughness.baseColorTexture))
    return true;
  if(textureInfoHasTransform(m.pbrMetallicRoughness.metallicRoughnessTexture))
    return true;

  // Common slots (normal/occlusion/emissive have different texture-info types
  // but all expose .extensions in the same way)
  if(textureInfoHasTransform(m.normalTexture))
    return true;
  if(textureInfoHasTransform(m.occlusionTexture))
    return true;
  if(textureInfoHasTransform(m.emissiveTexture))
    return true;

  for(const auto& ext : m.extensions)
  {
    if(valueTreeHasTextureTransform(ext.second))
      return true;
  }

  return false;
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// Stable hash key for scene-feature variant caching
uint64_t SceneFeatureSet::hash() const
{
  // FNV-1a-style mix over packed bits; cheap and order-stable.
  uint64_t bits = 0;
  bits |= (uint64_t)transmission << 0;
  bits |= (uint64_t)volume << 1;
  bits |= (uint64_t)volumeScatter << 2;
  bits |= (uint64_t)clearcoat << 3;
  bits |= (uint64_t)iridescence << 4;
  bits |= (uint64_t)anisotropy << 5;
  bits |= (uint64_t)sheen << 6;
  bits |= (uint64_t)dispersion << 7;
  bits |= (uint64_t)diffuseTransmission << 8;
  bits |= (uint64_t)unlit << 9;
  bits |= (uint64_t)specular << 10;
  bits |= (uint64_t)ior << 11;
  bits |= (uint64_t)specularGlossiness << 12;
  bits |= (uint64_t)textureTransform << 13;
  bits |= (uint64_t)dlssGuide << 14;

  // Simple splittable64 mix so neighboring sets hash to distant values.
  uint64_t h = bits + 0x9E3779B97F4A7C15ULL;
  h          = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
  h          = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
  h          = h ^ (h >> 31);
  return h;
}

//--------------------------------------------------------------------------------------------------
// True when this feature set covers every feature in the other set
bool SceneFeatureSet::isSupersetOf(const SceneFeatureSet& o) const
{
  auto covers = [](bool mine, bool theirs) { return !theirs || mine; };
  return covers(transmission, o.transmission) && covers(volume, o.volume) && covers(volumeScatter, o.volumeScatter)
         && covers(clearcoat, o.clearcoat) && covers(iridescence, o.iridescence) && covers(anisotropy, o.anisotropy)
         && covers(sheen, o.sheen) && covers(dispersion, o.dispersion)
         && covers(diffuseTransmission, o.diffuseTransmission) && covers(unlit, o.unlit) && covers(specular, o.specular)
         && covers(ior, o.ior) && covers(specularGlossiness, o.specularGlossiness)
         && covers(textureTransform, o.textureTransform) && covers(dlssGuide, o.dlssGuide);
}

//--------------------------------------------------------------------------------------------------
// Exact equality on all scene-feature toggles
bool SceneFeatureSet::operator==(const SceneFeatureSet& o) const
{
  return transmission == o.transmission && volume == o.volume && volumeScatter == o.volumeScatter && clearcoat == o.clearcoat
         && iridescence == o.iridescence && anisotropy == o.anisotropy && sheen == o.sheen && dispersion == o.dispersion
         && diffuseTransmission == o.diffuseTransmission && unlit == o.unlit && specular == o.specular && ior == o.ior
         && specularGlossiness == o.specularGlossiness && textureTransform == o.textureTransform && dlssGuide == o.dlssGuide;
}

//--------------------------------------------------------------------------------------------------
// Comma-separated list of enabled features for logs/UI
std::string SceneFeatureSet::toString() const
{
  std::string out;
  auto        add = [&](const char* name, bool v) {
    if(!v)
      return;
    if(!out.empty())
      out += ",";
    out += name;
  };
  add("transmission", transmission);
  add("volume", volume);
  add("volumeScatter", volumeScatter);
  add("clearcoat", clearcoat);
  add("iridescence", iridescence);
  add("anisotropy", anisotropy);
  add("sheen", sheen);
  add("dispersion", dispersion);
  add("diffuseTransmission", diffuseTransmission);
  add("unlit", unlit);
  add("specular", specular);
  add("ior", ior);
  add("specularGlossiness", specularGlossiness);
  add("textureTransform", textureTransform);
  add("dlssGuide", dlssGuide);
  return out;
}

//--------------------------------------------------------------------------------------------------
// Count currently-disabled extension gates
int SceneFeatureSet::unusedExtensionCount() const
{
  // Count of MAT_EXT_* style flags that are currently false. The dlssGuide flag is
  // intentionally not counted here because it is not a MAT_EXT_* (it's its own gate).
  int unused = 0;
  unused += transmission ? 0 : 1;
  unused += volume ? 0 : 1;
  unused += volumeScatter ? 0 : 1;
  unused += clearcoat ? 0 : 1;
  unused += iridescence ? 0 : 1;
  unused += anisotropy ? 0 : 1;
  unused += sheen ? 0 : 1;
  unused += dispersion ? 0 : 1;
  unused += diffuseTransmission ? 0 : 1;
  unused += unlit ? 0 : 1;
  unused += specular ? 0 : 1;
  unused += ior ? 0 : 1;
  unused += specularGlossiness ? 0 : 1;
  unused += textureTransform ? 0 : 1;
  return unused;
}

//--------------------------------------------------------------------------------------------------
// Scan materials/model and collect the scene's active feature toggles
SceneFeatureSet detectSceneFeatures(const std::vector<tinygltf::Material>& materials, const tinygltf::Model* /*model*/)
{
  SceneFeatureSet set{};

  for(const auto& m : materials)
  {
    const auto& ext = m.extensions;
    if(hasExt(ext, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME))
      set.transmission = true;
    if(hasExt(ext, KHR_MATERIALS_VOLUME_EXTENSION_NAME))
      set.volume = true;
    if(hasExt(ext, KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME))
      set.volumeScatter = true;
    if(hasExt(ext, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME))
      set.clearcoat = true;
    if(hasExt(ext, KHR_MATERIALS_IRIDESCENCE_EXTENSION_NAME))
      set.iridescence = true;
    if(hasExt(ext, KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME))
      set.anisotropy = true;
    if(hasExt(ext, KHR_MATERIALS_SHEEN_EXTENSION_NAME))
      set.sheen = true;
    if(hasExt(ext, KHR_MATERIALS_DISPERSION_EXTENSION_NAME))
      set.dispersion = true;
    if(hasExt(ext, KHR_MATERIALS_DIFFUSE_TRANSMISSION_EXTENSION_NAME))
      set.diffuseTransmission = true;
    if(hasExt(ext, KHR_MATERIALS_UNLIT_EXTENSION_NAME))
      set.unlit = true;
    if(hasExt(ext, KHR_MATERIALS_SPECULAR_EXTENSION_NAME))
      set.specular = true;
    if(hasExt(ext, KHR_MATERIALS_IOR_EXTENSION_NAME))
      set.ior = true;
    if(hasExt(ext, KHR_MATERIALS_PBR_SPECULAR_GLOSSINESS_EXTENSION_NAME))
      set.specularGlossiness = true;

    if(materialHasTextureTransform(m))
      set.textureTransform = true;
  }

  // Feature dependency promotion. The path tracer's bounce-loop code paths form a
  // strict containment chain, and consumers of this set should not have to re-derive
  // it. We do it once here so SceneFeatureSet::hash() / operator== / toString() /
  // unusedExtensionCount() all see the fully-promoted truth.
  //
  //   scatter  =>  volume   (scatter samples Henyey-Greenstein inside a medium - the
  //                          medium has to exist, makeVolumeMedium is gated on volume)
  //   volume   =>  transmission   (the bounce loop only enters a volume via the
  //                                transmission isInside toggle; without transmission
  //                                pt.isInside never flips so the volume code is dead)
  if(set.volumeScatter)
    set.volume = true;
  if(set.volume)
    set.transmission = true;

  return set;
}

//--------------------------------------------------------------------------------------------------
// Convenience overload for full-model feature detection
SceneFeatureSet detectSceneFeatures(const tinygltf::Model& model)
{
  return detectSceneFeatures(model.materials, &model);
}

}  // namespace nvvkgltf
