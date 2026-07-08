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

#include <array>
#include <tinygltf/tiny_gltf.h>

#include "tinygltf_utils.hpp"

namespace nvvkgltf {

namespace {

struct FeatureName
{
  SceneFeatureSet::Feature feature;
  const char*              name;
};

constexpr std::array<FeatureName, SceneFeatureSet::kFeatureCount> kFeatureNames{{
    {SceneFeatureSet::eTransmission, "transmission"},
    {SceneFeatureSet::eVolume, "volume"},
    {SceneFeatureSet::eVolumeScatter, "volumeScatter"},
    {SceneFeatureSet::eClearcoat, "clearcoat"},
    {SceneFeatureSet::eIridescence, "iridescence"},
    {SceneFeatureSet::eAnisotropy, "anisotropy"},
    {SceneFeatureSet::eSheen, "sheen"},
    {SceneFeatureSet::eDispersion, "dispersion"},
    {SceneFeatureSet::eDiffuseTransmission, "diffuseTransmission"},
    {SceneFeatureSet::eRetroreflection, "retroreflection"},
    {SceneFeatureSet::eUnlit, "unlit"},
    {SceneFeatureSet::eSpecular, "specular"},
    {SceneFeatureSet::eIor, "ior"},
    {SceneFeatureSet::eSpecularGlossiness, "specularGlossiness"},
    {SceneFeatureSet::eTextureTransform, "textureTransform"},
    {SceneFeatureSet::eDlssGuide, "dlssGuide"},
}};

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
// True when this feature set covers every feature in the other set
bool SceneFeatureSet::isSupersetOf(const SceneFeatureSet& o) const
{
  return (m_bits & o.m_bits) == o.m_bits;
}

//--------------------------------------------------------------------------------------------------
// Exact equality on all scene-feature toggles
bool SceneFeatureSet::operator==(const SceneFeatureSet& o) const
{
  return m_bits == o.m_bits;
}

//--------------------------------------------------------------------------------------------------
// Comma-separated list of enabled features for logs/UI
std::string SceneFeatureSet::toString() const
{
  std::string out;
  for(const FeatureName& featureName : kFeatureNames)
  {
    if(!has(featureName.feature))
      continue;
    if(!out.empty())
      out += ",";
    out += featureName.name;
  }
  return out;
}

//--------------------------------------------------------------------------------------------------
// Count currently-disabled extension gates
int SceneFeatureSet::unusedExtensionCount() const
{
  // Count of MAT_EXT_* style flags that are currently false. The dlssGuide flag is
  // intentionally not counted here because it is not a MAT_EXT_* (it's its own gate).
  std::bitset<kFeatureCount> extensionBits = m_bits;
  extensionBits.reset(static_cast<std::size_t>(eDlssGuide));
  return static_cast<int>(kExtensionFeatureCount - extensionBits.count());
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
      set.enable(SceneFeatureSet::eTransmission);
    if(hasExt(ext, KHR_MATERIALS_VOLUME_EXTENSION_NAME))
      set.enable(SceneFeatureSet::eVolume);
    if(hasExt(ext, KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME))
      set.enable(SceneFeatureSet::eVolumeScatter);
    if(hasExt(ext, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME))
      set.enable(SceneFeatureSet::eClearcoat);
    if(hasExt(ext, KHR_MATERIALS_IRIDESCENCE_EXTENSION_NAME))
      set.enable(SceneFeatureSet::eIridescence);
    if(hasExt(ext, KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME))
      set.enable(SceneFeatureSet::eAnisotropy);
    if(hasExt(ext, KHR_MATERIALS_SHEEN_EXTENSION_NAME))
      set.enable(SceneFeatureSet::eSheen);
    if(hasExt(ext, KHR_MATERIALS_DISPERSION_EXTENSION_NAME))
      set.enable(SceneFeatureSet::eDispersion);
    if(hasExt(ext, KHR_MATERIALS_DIFFUSE_TRANSMISSION_EXTENSION_NAME))
      set.enable(SceneFeatureSet::eDiffuseTransmission);
    if(hasExt(ext, EXT_materials_retroreflection_EXTENSION_NAME))
      set.enable(SceneFeatureSet::eRetroreflection);
    if(hasExt(ext, KHR_MATERIALS_UNLIT_EXTENSION_NAME))
      set.enable(SceneFeatureSet::eUnlit);
    if(hasExt(ext, KHR_MATERIALS_SPECULAR_EXTENSION_NAME))
      set.enable(SceneFeatureSet::eSpecular);
    if(hasExt(ext, KHR_MATERIALS_IOR_EXTENSION_NAME))
      set.enable(SceneFeatureSet::eIor);
    if(hasExt(ext, KHR_MATERIALS_PBR_SPECULAR_GLOSSINESS_EXTENSION_NAME))
      set.enable(SceneFeatureSet::eSpecularGlossiness);

    if(materialHasTextureTransform(m))
      set.enable(SceneFeatureSet::eTextureTransform);
  }

  // Feature dependency promotion. The path tracer's bounce-loop code paths form a
  // strict containment chain, and consumers of this set should not have to re-derive
  // it. We do it once here so operator== / toString() / unusedExtensionCount()
  // all see the fully-promoted truth.
  //
  //   scatter  =>  volume   (scatter samples Henyey-Greenstein inside a medium - the
  //                          medium has to exist, makeVolumeMedium is gated on volume)
  //   volume   =>  transmission   (the bounce loop only enters a volume via the
  //                                transmission isInside toggle; without transmission
  //                                pt.isInside never flips so the volume code is dead)
  if(set.has(SceneFeatureSet::eVolumeScatter))
    set.enable(SceneFeatureSet::eVolume);
  if(set.has(SceneFeatureSet::eVolume))
    set.enable(SceneFeatureSet::eTransmission);

  return set;
}

//--------------------------------------------------------------------------------------------------
// Convenience overload for full-model feature detection
SceneFeatureSet detectSceneFeatures(const tinygltf::Model& model)
{
  return detectSceneFeatures(model.materials, &model);
}

}  // namespace nvvkgltf
