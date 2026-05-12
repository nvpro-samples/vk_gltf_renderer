/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use it except in compliance with the License.
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

//
// Extracts glTF material properties (PBR metallic-roughness, specular-glossiness,
// KHR extensions, etc.) from tinygltf and packs them into GPU-friendly
// GltfShadeMaterial structures for use in shaders. Caches the converted
// material array so it can be rebuilt only when materials change.
//

#include "gltf_material_cache.hpp"

#include <cstddef>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "nvutils/logger.hpp"
#include "tinygltf_utils.hpp"

// Layout guardrails for the GPU material struct. These must hold regardless of which
// MAT_EXT_* flags are enabled so the material buffer array stride stays consistent and
// base (always-on) field offsets do not drift.
//
// The buffer is bound via Vulkan buffer_reference with scalar block layout
// (-force-glsl-scalar-layout in compile_slang). A trailing uint64_t _pad forces the
// struct's max member alignment to 8 so every materials[i] is 8-aligned on the GPU;
// the driver can then emit LDG.64 for adjacent 4-byte field pairs (roughness/metallic,
// alphaMode/alphaCutoff, ...) across every element of the array, not just element 0.
static_assert(alignof(shaderio::GltfShadeMaterial) >= 8,
              "GltfShadeMaterial must be at least 8-byte aligned (trailing uint64_t _pad enforces this)");
static_assert(sizeof(shaderio::GltfShadeMaterial) % 8 == 0,
              "GltfShadeMaterial size must be a multiple of 8 for LDG.64-friendly BDA array stride");
static_assert(offsetof(shaderio::GltfShadeMaterial, pbrBaseColorFactor) == 0, "pbrBaseColorFactor must stay at offset 0");
static_assert(offsetof(shaderio::GltfShadeMaterial, pbrRoughnessFactor) == 32,
              "pbrRoughnessFactor must stay at offset 32 (anchors base block)");
static_assert(offsetof(shaderio::GltfShadeMaterial, alphaMode) == 40, "alphaMode must stay at offset 40 (anchors base block)");
static_assert(offsetof(shaderio::GltfShadeMaterial, occlusionStrength) == 48,
              "occlusionStrength must stay at offset 48 (promoted to base block)");
static_assert(offsetof(shaderio::GltfShadeMaterial, doubleSided) == 52, "doubleSided must stay at offset 52 (promoted to base block)");

namespace nvvkgltf {

namespace {

template <typename T>
shaderio::GltfTextureInfo getTextureInfoImpl(const T& tinfo)
{
  shaderio::GltfTextureInfo ti{};
  ti.index = tinfo.index;

  // The renderer only uploads TEXCOORD_0 and TEXCOORD_1 (see VertexBuffers::texCoords[2]
  // in shaders/gltf_scene_io.h.slang). glTF allows TEXCOORD_N for higher N but we clamp
  // to 1 and warn once per occurrence so the user notices missing UV sets instead of
  // silently rendering with the wrong coordinates.
  if(tinfo.texCoord > 1)
  {
    LOGW(
        "GltfTextureInfo: texture references TEXCOORD_%d but the renderer only supports "
        "TEXCOORD_0 and TEXCOORD_1; clamping to TEXCOORD_1.\n",
        tinfo.texCoord);
  }
  ti.texCoord = std::min(tinfo.texCoord, 1);

#if MAT_EXT_TEXTURE_TRANSFORM
  const KHR_texture_transform& transform = tinygltf::utils::getTextureTransform(tinfo);
  ti.uvTransform = shaderio::float3x2(transform.uvTransform[0][0], transform.uvTransform[1][0], transform.uvTransform[0][1],
                                      transform.uvTransform[1][1], transform.uvTransform[0][2], transform.uvTransform[1][2]);
#endif
  return ti;
}

template <typename T>
uint16_t addTextureInfoImpl(const T& tinfo, std::vector<shaderio::GltfTextureInfo>& textureInfos)
{
  shaderio::GltfTextureInfo ti = getTextureInfoImpl(tinfo);
  if(ti.index != -1)
  {
    uint16_t idx = static_cast<uint16_t>(textureInfos.size());
    textureInfos.push_back(ti);
    return idx;
  }
  return 0;
}

template <typename TextureHandler>
void populateShaderMaterial(shaderio::GltfShadeMaterial& dstMat, const tinygltf::Material& srcMat, TextureHandler handleTexture)
{
  dstMat.alphaMode          = srcMat.alphaMode == "OPAQUE" ? 0 : (srcMat.alphaMode == "MASK" ? 1 : 2);
  dstMat.alphaCutoff        = static_cast<float>(srcMat.alphaCutoff);
  dstMat.doubleSided        = srcMat.doubleSided ? 1 : 0;
  dstMat.pbrBaseColorFactor = glm::make_vec4<double>(srcMat.pbrMetallicRoughness.baseColorFactor.data());
  dstMat.pbrMetallicFactor  = static_cast<float>(srcMat.pbrMetallicRoughness.metallicFactor);
  dstMat.pbrRoughnessFactor = static_cast<float>(srcMat.pbrMetallicRoughness.roughnessFactor);
  dstMat.normalTextureScale = static_cast<float>(srcMat.normalTexture.scale);
  dstMat.occlusionStrength  = static_cast<float>(srcMat.occlusionTexture.strength);

  if(!srcMat.emissiveFactor.empty())
    dstMat.emissiveFactor = glm::make_vec3<double>(srcMat.emissiveFactor.data());

  handleTexture(dstMat.emissiveTexture, srcMat.emissiveTexture);
  handleTexture(dstMat.normalTexture, srcMat.normalTexture);
  handleTexture(dstMat.pbrBaseColorTexture, srcMat.pbrMetallicRoughness.baseColorTexture);
  handleTexture(dstMat.pbrMetallicRoughnessTexture, srcMat.pbrMetallicRoughness.metallicRoughnessTexture);
  handleTexture(dstMat.occlusionTexture, srcMat.occlusionTexture);

#if MAT_EXT_TRANSMISSION
  KHR_materials_transmission transmission = tinygltf::utils::getTransmission(srcMat);
  dstMat.transmissionFactor               = transmission.factor;
  handleTexture(dstMat.transmissionTexture, transmission.texture);
#endif

#if MAT_EXT_IOR
  KHR_materials_ior ior = tinygltf::utils::getIor(srcMat);
  dstMat.ior            = ior.ior;
#endif

#if MAT_EXT_VOLUME
  KHR_materials_volume volume = tinygltf::utils::getVolume(srcMat);
  dstMat.attenuationColor     = volume.attenuationColor;
  dstMat.thicknessFactor      = volume.thicknessFactor;
  dstMat.attenuationDistance  = volume.attenuationDistance;
  handleTexture(dstMat.thicknessTexture, volume.thicknessTexture);
#endif

#if MAT_EXT_CLEARCOAT
  KHR_materials_clearcoat clearcoat = tinygltf::utils::getClearcoat(srcMat);
  dstMat.clearcoatFactor            = clearcoat.factor;
  dstMat.clearcoatRoughness         = clearcoat.roughnessFactor;
  handleTexture(dstMat.clearcoatRoughnessTexture, clearcoat.roughnessTexture);
  handleTexture(dstMat.clearcoatTexture, clearcoat.texture);
  handleTexture(dstMat.clearcoatNormalTexture, clearcoat.normalTexture);
#endif

#if MAT_EXT_SPECULAR
  KHR_materials_specular specular = tinygltf::utils::getSpecular(srcMat);
  dstMat.specularFactor           = specular.specularFactor;
  dstMat.specularColorFactor      = specular.specularColorFactor;
  handleTexture(dstMat.specularTexture, specular.specularTexture);
  handleTexture(dstMat.specularColorTexture, specular.specularColorTexture);
#endif

  // KHR_materials_emissive_strength is always applied: it only scales emissiveFactor, no
  // struct field of its own. Default strength is 1.0 when the extension is absent.
  KHR_materials_emissive_strength emissiveStrength = tinygltf::utils::getEmissiveStrength(srcMat);
  dstMat.emissiveFactor *= emissiveStrength.emissiveStrength;

#if MAT_EXT_UNLIT
  KHR_materials_unlit unlit = tinygltf::utils::getUnlit(srcMat);
  dstMat.unlit              = unlit.active ? 1 : 0;
#endif

#if MAT_EXT_IRIDESCENCE
  KHR_materials_iridescence iridescence = tinygltf::utils::getIridescence(srcMat);
  dstMat.iridescenceFactor              = iridescence.iridescenceFactor;
  dstMat.iridescenceIor                 = iridescence.iridescenceIor;
  dstMat.iridescenceThicknessMinimum    = iridescence.iridescenceThicknessMinimum;
  dstMat.iridescenceThicknessMaximum    = iridescence.iridescenceThicknessMaximum;
  handleTexture(dstMat.iridescenceTexture, iridescence.iridescenceTexture);
  handleTexture(dstMat.iridescenceThicknessTexture, iridescence.iridescenceThicknessTexture);
#endif

#if MAT_EXT_ANISOTROPY
  KHR_materials_anisotropy anisotropy = tinygltf::utils::getAnisotropy(srcMat);
  dstMat.anisotropyRotation = glm::vec2(glm::sin(anisotropy.anisotropyRotation), glm::cos(anisotropy.anisotropyRotation));
  dstMat.anisotropyStrength = anisotropy.anisotropyStrength;
  handleTexture(dstMat.anisotropyTexture, anisotropy.anisotropyTexture);
#endif

#if MAT_EXT_SHEEN
  KHR_materials_sheen sheen   = tinygltf::utils::getSheen(srcMat);
  dstMat.sheenColorFactor     = sheen.sheenColorFactor;
  dstMat.sheenRoughnessFactor = sheen.sheenRoughnessFactor;
  handleTexture(dstMat.sheenColorTexture, sheen.sheenColorTexture);
  handleTexture(dstMat.sheenRoughnessTexture, sheen.sheenRoughnessTexture);
#endif

#if MAT_EXT_DISPERSION
  KHR_materials_dispersion dispersion = tinygltf::utils::getDispersion(srcMat);
  dstMat.dispersion                   = dispersion.dispersion;
#endif

#if MAT_EXT_SPECULAR_GLOSSINESS
  // Default pbrModel is ePbrModelMetallicRoughness (struct member default); switch to
  // ePbrModelSpecularGlossiness only when the extension is actually present on this material.
  KHR_materials_pbrSpecularGlossiness pbr = tinygltf::utils::getPbrSpecularGlossiness(srcMat);
  if(tinygltf::utils::hasElementName(srcMat.extensions, KHR_MATERIALS_PBR_SPECULAR_GLOSSINESS_EXTENSION_NAME))
  {
    dstMat.pbrModel            = shaderio::PbrModel::ePbrModelSpecularGlossiness;
    dstMat.pbrDiffuseFactor    = pbr.diffuseFactor;
    dstMat.pbrSpecularFactor   = pbr.specularFactor;
    dstMat.pbrGlossinessFactor = pbr.glossinessFactor;
  }
  handleTexture(dstMat.pbrDiffuseTexture, pbr.diffuseTexture);
  handleTexture(dstMat.pbrSpecularGlossinessTexture, pbr.specularGlossinessTexture);
#endif

#if MAT_EXT_DIFFUSE_TRANSMISSION
  KHR_materials_diffuse_transmission diffuseTransmission = tinygltf::utils::getDiffuseTransmission(srcMat);
  dstMat.diffuseTransmissionFactor                       = diffuseTransmission.diffuseTransmissionFactor;
  dstMat.diffuseTransmissionColor                        = diffuseTransmission.diffuseTransmissionColor;
  handleTexture(dstMat.diffuseTransmissionTexture, diffuseTransmission.diffuseTransmissionTexture);
  handleTexture(dstMat.diffuseTransmissionColorTexture, diffuseTransmission.diffuseTransmissionColorTexture);
#endif

#if MAT_EXT_VOLUME_SCATTER
  KHR_materials_volume_scatter volumeScatter = tinygltf::utils::getVolumeScatter(srcMat);
  dstMat.multiscatterColorFactor             = volumeScatter.multiscatterColorFactor;
  dstMat.scatterAnisotropy                   = volumeScatter.scatterAnisotropy;
#endif
}

}  // namespace

void MaterialCache::buildFromMaterials(const std::vector<tinygltf::Material>& materials)
{
  m_textureInfos.clear();
  m_textureInfos.push_back({});
  m_shadeMaterials.clear();
  m_shadeMaterials.reserve(materials.size());
  for(const auto& srcMat : materials)
  {
    shaderio::GltfShadeMaterial dstMat = {};
    populateShaderMaterial(dstMat, srcMat, [this](uint16_t& texIndex, const auto& srcTexInfo) {
      texIndex = addTextureInfoImpl(srcTexInfo, m_textureInfos);
    });
    m_shadeMaterials.emplace_back(dstMat);
  }
}

MaterialUpdateResult MaterialCache::updateMaterial(int index, const tinygltf::Material& srcMat)
{
  MaterialUpdateResult result{};
  if(index < 0 || index >= static_cast<int>(m_shadeMaterials.size()))
    return result;

  shaderio::GltfShadeMaterial& dstMat = m_shadeMaterials[index];
  shaderio::GltfTextureInfo*   infos  = m_textureInfos.data();

  populateShaderMaterial(dstMat, srcMat, [&](uint16_t& texIndex, const auto& srcTexInfo) {
    const bool hasSrcTexture = srcTexInfo.index != -1;

    if(texIndex == 0)
    {
      if(hasSrcTexture)
        result.topologyChanged = true;
      return;
    }
    if(!hasSrcTexture)
    {
      result.topologyChanged = true;
      return;
    }

    infos[texIndex]    = getTextureInfoImpl(srcTexInfo);
    result.span.minIdx = std::min(result.span.minIdx, texIndex);
    result.span.maxIdx = std::max(result.span.maxIdx, texIndex);
    result.span.count++;
  });
  return result;
}

void MaterialCache::clear()
{
  m_shadeMaterials.clear();
  m_textureInfos.clear();
}

}  // namespace nvvkgltf
