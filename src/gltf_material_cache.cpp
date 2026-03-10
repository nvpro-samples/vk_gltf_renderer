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

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "tinygltf_utils.hpp"

namespace nvvkgltf {

namespace {

template <typename T>
shaderio::GltfTextureInfo getTextureInfoImpl(const T& tinfo)
{
  const KHR_texture_transform& transform = tinygltf::utils::getTextureTransform(tinfo);
  const int                    texCoord  = std::min(tinfo.texCoord, 1);

  return {
      .uvTransform = shaderio::float3x2(transform.uvTransform[0][0], transform.uvTransform[1][0], transform.uvTransform[0][1],
                                        transform.uvTransform[1][1], transform.uvTransform[0][2], transform.uvTransform[1][2]),
      .index    = tinfo.index,
      .texCoord = texCoord,
  };
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

  KHR_materials_transmission transmission = tinygltf::utils::getTransmission(srcMat);
  dstMat.transmissionFactor               = transmission.factor;
  handleTexture(dstMat.transmissionTexture, transmission.texture);

  KHR_materials_ior ior = tinygltf::utils::getIor(srcMat);
  dstMat.ior            = ior.ior;

  KHR_materials_volume volume = tinygltf::utils::getVolume(srcMat);
  dstMat.attenuationColor     = volume.attenuationColor;
  dstMat.thicknessFactor      = volume.thicknessFactor;
  dstMat.attenuationDistance  = volume.attenuationDistance;
  handleTexture(dstMat.thicknessTexture, volume.thicknessTexture);

  KHR_materials_clearcoat clearcoat = tinygltf::utils::getClearcoat(srcMat);
  dstMat.clearcoatFactor            = clearcoat.factor;
  dstMat.clearcoatRoughness         = clearcoat.roughnessFactor;
  handleTexture(dstMat.clearcoatRoughnessTexture, clearcoat.roughnessTexture);
  handleTexture(dstMat.clearcoatTexture, clearcoat.texture);
  handleTexture(dstMat.clearcoatNormalTexture, clearcoat.normalTexture);

  KHR_materials_specular specular = tinygltf::utils::getSpecular(srcMat);
  dstMat.specularFactor           = specular.specularFactor;
  dstMat.specularColorFactor      = specular.specularColorFactor;
  handleTexture(dstMat.specularTexture, specular.specularTexture);
  handleTexture(dstMat.specularColorTexture, specular.specularColorTexture);

  KHR_materials_emissive_strength emissiveStrength = tinygltf::utils::getEmissiveStrength(srcMat);
  dstMat.emissiveFactor *= emissiveStrength.emissiveStrength;

  KHR_materials_unlit unlit = tinygltf::utils::getUnlit(srcMat);
  dstMat.unlit              = unlit.active ? 1 : 0;

  KHR_materials_iridescence iridescence = tinygltf::utils::getIridescence(srcMat);
  dstMat.iridescenceFactor              = iridescence.iridescenceFactor;
  dstMat.iridescenceIor                 = iridescence.iridescenceIor;
  dstMat.iridescenceThicknessMaximum    = iridescence.iridescenceThicknessMaximum;
  dstMat.iridescenceThicknessMinimum    = iridescence.iridescenceThicknessMinimum;
  handleTexture(dstMat.iridescenceTexture, iridescence.iridescenceTexture);
  handleTexture(dstMat.iridescenceThicknessTexture, iridescence.iridescenceThicknessTexture);

  KHR_materials_anisotropy anisotropy = tinygltf::utils::getAnisotropy(srcMat);
  dstMat.anisotropyRotation = glm::vec2(glm::sin(anisotropy.anisotropyRotation), glm::cos(anisotropy.anisotropyRotation));
  dstMat.anisotropyStrength = anisotropy.anisotropyStrength;
  handleTexture(dstMat.anisotropyTexture, anisotropy.anisotropyTexture);

  KHR_materials_sheen sheen   = tinygltf::utils::getSheen(srcMat);
  dstMat.sheenColorFactor     = sheen.sheenColorFactor;
  dstMat.sheenRoughnessFactor = sheen.sheenRoughnessFactor;
  handleTexture(dstMat.sheenColorTexture, sheen.sheenColorTexture);
  handleTexture(dstMat.sheenRoughnessTexture, sheen.sheenRoughnessTexture);

  KHR_materials_dispersion dispersion = tinygltf::utils::getDispersion(srcMat);
  dstMat.dispersion                   = dispersion.dispersion;

  KHR_materials_pbrSpecularGlossiness pbr = tinygltf::utils::getPbrSpecularGlossiness(srcMat);
  dstMat.usePbrSpecularGlossiness =
      tinygltf::utils::hasElementName(srcMat.extensions, KHR_MATERIALS_PBR_SPECULAR_GLOSSINESS_EXTENSION_NAME);
  if(dstMat.usePbrSpecularGlossiness)
  {
    dstMat.pbrDiffuseFactor    = pbr.diffuseFactor;
    dstMat.pbrSpecularFactor   = pbr.specularFactor;
    dstMat.pbrGlossinessFactor = pbr.glossinessFactor;
  }
  handleTexture(dstMat.pbrDiffuseTexture, pbr.diffuseTexture);
  handleTexture(dstMat.pbrSpecularGlossinessTexture, pbr.specularGlossinessTexture);

  KHR_materials_diffuse_transmission diffuseTransmission = tinygltf::utils::getDiffuseTransmission(srcMat);
  dstMat.diffuseTransmissionFactor                       = diffuseTransmission.diffuseTransmissionFactor;
  dstMat.diffuseTransmissionColor                        = diffuseTransmission.diffuseTransmissionColor;
  handleTexture(dstMat.diffuseTransmissionTexture, diffuseTransmission.diffuseTransmissionTexture);
  handleTexture(dstMat.diffuseTransmissionColorTexture, diffuseTransmission.diffuseTransmissionColorTexture);

  KHR_materials_volume_scatter volumeScatter = tinygltf::utils::getVolumeScatter(srcMat);
  dstMat.multiscatterColor                   = volumeScatter.multiscatterColor;
  dstMat.scatterAnisotropy                   = volumeScatter.scatterAnisotropy;
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
    shaderio::GltfShadeMaterial dstMat = shaderio::defaultGltfMaterial();
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
