
/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Utility functions for working with tinygltf models. Provides helpers
// for extension parsing (KHR_materials_*), morph target processing,
// mesh attribute access, normal/tangent recomputation, and other
// common operations on glTF model data.
//

#include "tinygltf_utils.hpp"

#include <unordered_set>

#include <glm/gtx/norm.hpp>
#include "nvutils/logger.hpp"
#include "nvutils/parallel_work.hpp"

#include "nvshaders/functions.h.slang"

namespace {

// Texture extensions that store the backing image index in `source` (see `getTextureImageIndex`).
const char* const kTextureImageSourceExtensionNames[] = {
    EXT_TEXTURE_WEBP_EXTENSION_NAME,
    MSFT_TEXTURE_DDS_NAME,
    KHR_TEXTURE_BASISU_EXTENSION_NAME,
};

// Recursively records extension names nested inside an extension's raw JSON value. This catches
// extensions living on extension-owned TextureInfo slots (e.g. KHR_texture_transform on a
// clearcoat/sheen/transmission texture), which are stored as opaque JSON inside mat.extensions[...]
// and are therefore invisible to a walk of the typed ExtensionMap members alone.
void collectNestedExtensionNames(const tinygltf::Value&           value,
                                 std::vector<std::string>&        discovered,
                                 std::unordered_set<std::string>& discoveredSet)
{
  if(value.IsObject())
  {
    for(const std::string& key : value.Keys())
    {
      const tinygltf::Value& child = value.Get(key);
      if(key == "extensions" && child.IsObject())
      {
        for(const std::string& extName : child.Keys())
        {
          if(discoveredSet.insert(extName).second)
            discovered.push_back(extName);
        }
      }
      collectNestedExtensionNames(child, discovered, discoveredSet);
    }
  }
  else if(value.IsArray())
  {
    for(size_t i = 0; i < value.ArrayLen(); i++)
      collectNestedExtensionNames(value.Get(i), discovered, discoveredSet);
  }
}

}  // namespace

KHR_materials_displacement tinygltf::utils::getDisplacement(const tinygltf::Material& tmat)
{
  KHR_materials_displacement gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_DISPLACEMENT_EXTENSION_NAME))
  {
    tinygltf::utils::getValue(*ext, "displacementGeometryTexture", gmat.displacementGeometryTexture);
    tinygltf::utils::getValue(*ext, "displacementGeometryFactor", gmat.displacementGeometryFactor);
    tinygltf::utils::getValue(*ext, "displacementGeometryOffset", gmat.displacementGeometryOffset);
  }
  return gmat;
}

void tinygltf::utils::setDisplacement(tinygltf::Material& tmat, const KHR_materials_displacement& displacement)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_DISPLACEMENT_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "displacementGeometryTexture", displacement.displacementGeometryTexture);
  tinygltf::utils::setValue(ext, "displacementGeometryFactor", displacement.displacementGeometryFactor);
  tinygltf::utils::setValue(ext, "displacementGeometryOffset", displacement.displacementGeometryOffset);
}

KHR_materials_emissive_strength tinygltf::utils::getEmissiveStrength(const tinygltf::Material& tmat)
{
  KHR_materials_emissive_strength gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_EMISSIVE_STRENGTH_EXTENSION_NAME))
  {
    tinygltf::utils::getValue(*ext, "emissiveStrength", gmat.emissiveStrength);
  }
  return gmat;
}

void tinygltf::utils::setEmissiveStrength(tinygltf::Material& tmat, const KHR_materials_emissive_strength& emissiveStrength)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_EMISSIVE_STRENGTH_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "emissiveStrength", emissiveStrength.emissiveStrength);
}


KHR_materials_volume tinygltf::utils::getVolume(const tinygltf::Material& tmat)
{
  KHR_materials_volume gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_VOLUME_EXTENSION_NAME))
  {
    tinygltf::utils::getValue(*ext, "thicknessFactor", gmat.thicknessFactor);
    tinygltf::utils::getValue(*ext, "thicknessTexture", gmat.thicknessTexture);
    tinygltf::utils::getValue(*ext, "attenuationDistance", gmat.attenuationDistance);
    tinygltf::utils::getArrayValue(*ext, "attenuationColor", gmat.attenuationColor);
  }
  return gmat;
}

void tinygltf::utils::setVolume(tinygltf::Material& tmat, const KHR_materials_volume& volume)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_VOLUME_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "thicknessFactor", volume.thicknessFactor);
  tinygltf::utils::setValue(ext, "thicknessTexture", volume.thicknessTexture);
  tinygltf::utils::setValue(ext, "attenuationDistance", volume.attenuationDistance);
  tinygltf::utils::setArrayValue(ext, "attenuationColor", 3, glm::value_ptr(volume.attenuationColor));
}

KHR_materials_volume_scatter tinygltf::utils::getVolumeScatter(const tinygltf::Material& tmat)
{
  KHR_materials_volume_scatter gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME))
  {
    tinygltf::utils::getArrayValue(*ext, "multiscatterColorFactor", gmat.multiscatterColorFactor);
    tinygltf::utils::getValue(*ext, "scatterAnisotropy", gmat.scatterAnisotropy);
    gmat.scatterAnisotropy = std::clamp(gmat.scatterAnisotropy, -0.999f, 0.999f);

    // If multiscatterColor is present (old version), set multiscatterColorFactor to it
    if(ext->Has("multiscatterColor"))
    {
      tinygltf::utils::getArrayValue(*ext, "multiscatterColor", gmat.multiscatterColorFactor);
    }
  }
  return gmat;
}

void tinygltf::utils::setVolumeScatter(tinygltf::Material& tmat, const KHR_materials_volume_scatter& scatter)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME);
  tinygltf::utils::setArrayValue(ext, "multiscatterColorFactor", 3, glm::value_ptr(scatter.multiscatterColorFactor));
  tinygltf::utils::setValue(ext, "scatterAnisotropy", scatter.scatterAnisotropy);
}

KHR_materials_unlit tinygltf::utils::getUnlit(const tinygltf::Material& tmat)
{
  KHR_materials_unlit gmat;
  if(tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_UNLIT_EXTENSION_NAME))
  {
    gmat.active = 1;
  }
  return gmat;
}

void tinygltf::utils::setUnlit(tinygltf::Material& tmat, const KHR_materials_unlit& unlit)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_UNLIT_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "unlit", true);
}


KHR_materials_specular tinygltf::utils::getSpecular(const tinygltf::Material& tmat)
{
  KHR_materials_specular gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_SPECULAR_EXTENSION_NAME))
  {
    tinygltf::utils::getValue(*ext, "specularFactor", gmat.specularFactor);
    tinygltf::utils::getValue(*ext, "specularTexture", gmat.specularTexture);
    tinygltf::utils::getArrayValue(*ext, "specularColorFactor", gmat.specularColorFactor);
    tinygltf::utils::getValue(*ext, "specularColorTexture", gmat.specularColorTexture);
  }
  return gmat;
}

void tinygltf::utils::setSpecular(tinygltf::Material& tmat, const KHR_materials_specular& specular)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_SPECULAR_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "specularFactor", specular.specularFactor);
  tinygltf::utils::setValue(ext, "specularTexture", specular.specularTexture);
  tinygltf::utils::setValue(ext, "specularColorTexture", specular.specularColorTexture);
  tinygltf::utils::setArrayValue(ext, "specularColorFactor", 3, glm::value_ptr(specular.specularColorFactor));
}


KHR_materials_clearcoat tinygltf::utils::getClearcoat(const tinygltf::Material& tmat)
{
  KHR_materials_clearcoat gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME))
  {
    tinygltf::utils::getValue(*ext, "clearcoatFactor", gmat.factor);
    tinygltf::utils::getValue(*ext, "clearcoatTexture", gmat.texture);
    tinygltf::utils::getValue(*ext, "clearcoatRoughnessFactor", gmat.roughnessFactor);
    tinygltf::utils::getValue(*ext, "clearcoatRoughnessTexture", gmat.roughnessTexture);
    tinygltf::utils::getValue(*ext, "clearcoatNormalTexture", gmat.normalTexture);
  }
  return gmat;
}

void tinygltf::utils::setClearcoat(tinygltf::Material& tmat, const KHR_materials_clearcoat& clearcoat)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "clearcoatFactor", clearcoat.factor);
  tinygltf::utils::setValue(ext, "clearcoatRoughnessFactor", clearcoat.roughnessFactor);
  tinygltf::utils::setValue(ext, "clearcoatTexture", clearcoat.texture);
  tinygltf::utils::setValue(ext, "clearcoatRoughnessTexture", clearcoat.roughnessTexture);
  tinygltf::utils::setValue(ext, "clearcoatNormalTexture", clearcoat.normalTexture);
}

KHR_materials_sheen tinygltf::utils::getSheen(const tinygltf::Material& tmat)
{
  KHR_materials_sheen gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_SHEEN_EXTENSION_NAME))
  {
    tinygltf::utils::getArrayValue(*ext, "sheenColorFactor", gmat.sheenColorFactor);
    tinygltf::utils::getValue(*ext, "sheenColorTexture", gmat.sheenColorTexture);
    tinygltf::utils::getValue(*ext, "sheenRoughnessFactor", gmat.sheenRoughnessFactor);
    tinygltf::utils::getValue(*ext, "sheenRoughnessTexture", gmat.sheenRoughnessTexture);
  }
  return gmat;
}

void tinygltf::utils::setSheen(tinygltf::Material& tmat, const KHR_materials_sheen& sheen)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_SHEEN_EXTENSION_NAME);
  tinygltf::utils::setArrayValue(ext, "sheenColorFactor", 3, glm::value_ptr(sheen.sheenColorFactor));
  tinygltf::utils::setValue(ext, "sheenColorTexture", sheen.sheenColorTexture);
  tinygltf::utils::setValue(ext, "sheenRoughnessFactor", sheen.sheenRoughnessFactor);
  tinygltf::utils::setValue(ext, "sheenRoughnessTexture", sheen.sheenRoughnessTexture);
}


KHR_materials_transmission tinygltf::utils::getTransmission(const tinygltf::Material& tmat)
{
  KHR_materials_transmission gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME))
  {
    tinygltf::utils::getValue(*ext, "transmissionFactor", gmat.factor);
    tinygltf::utils::getValue(*ext, "transmissionTexture", gmat.texture);
  }
  return gmat;
}

void tinygltf::utils::setTransmission(tinygltf::Material& tmat, const KHR_materials_transmission& transmission)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "transmissionFactor", transmission.factor);
  tinygltf::utils::setValue(ext, "transmissionTexture", transmission.texture);
}

KHR_materials_anisotropy tinygltf::utils::getAnisotropy(const tinygltf::Material& tmat)
{
  KHR_materials_anisotropy gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME))
  {
    tinygltf::utils::getValue(*ext, "anisotropyStrength", gmat.anisotropyStrength);
    tinygltf::utils::getValue(*ext, "anisotropyRotation", gmat.anisotropyRotation);
    tinygltf::utils::getValue(*ext, "anisotropyTexture", gmat.anisotropyTexture);
  }
  return gmat;
}

void tinygltf::utils::setAnisotropy(tinygltf::Material& tmat, const KHR_materials_anisotropy& anisotropy)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "anisotropyStrength", anisotropy.anisotropyStrength);
  tinygltf::utils::setValue(ext, "anisotropyRotation", anisotropy.anisotropyRotation);
  tinygltf::utils::setValue(ext, "anisotropyTexture", anisotropy.anisotropyTexture);
}

KHR_materials_ior tinygltf::utils::getIor(const tinygltf::Material& tmat)
{
  KHR_materials_ior gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_IOR_EXTENSION_NAME))
  {
    tinygltf::utils::getValue(*ext, "ior", gmat.ior);
  }
  return gmat;
}

void tinygltf::utils::setIor(tinygltf::Material& tmat, const KHR_materials_ior& ior)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_IOR_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "ior", ior.ior);
}


KHR_materials_iridescence tinygltf::utils::getIridescence(const tinygltf::Material& tmat)
{
  KHR_materials_iridescence gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_IRIDESCENCE_EXTENSION_NAME))
  {
    tinygltf::utils::getValue(*ext, "iridescenceFactor", gmat.iridescenceFactor);
    tinygltf::utils::getValue(*ext, "iridescenceTexture", gmat.iridescenceTexture);
    tinygltf::utils::getValue(*ext, "iridescenceIor", gmat.iridescenceIor);
    tinygltf::utils::getValue(*ext, "iridescenceThicknessMinimum", gmat.iridescenceThicknessMinimum);
    tinygltf::utils::getValue(*ext, "iridescenceThicknessMaximum", gmat.iridescenceThicknessMaximum);
    tinygltf::utils::getValue(*ext, "iridescenceThicknessTexture", gmat.iridescenceThicknessTexture);
  }
  return gmat;
}

void tinygltf::utils::setIridescence(tinygltf::Material& tmat, const KHR_materials_iridescence& iridescence)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_IRIDESCENCE_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "iridescenceFactor", iridescence.iridescenceFactor);
  tinygltf::utils::setValue(ext, "iridescenceTexture", iridescence.iridescenceTexture);
  tinygltf::utils::setValue(ext, "iridescenceIor", iridescence.iridescenceIor);
  tinygltf::utils::setValue(ext, "iridescenceThicknessMinimum", iridescence.iridescenceThicknessMinimum);
  tinygltf::utils::setValue(ext, "iridescenceThicknessMaximum", iridescence.iridescenceThicknessMaximum);
  tinygltf::utils::setValue(ext, "iridescenceThicknessTexture", iridescence.iridescenceThicknessTexture);
}


KHR_materials_dispersion tinygltf::utils::getDispersion(const tinygltf::Material& tmat)
{
  KHR_materials_dispersion gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_DISPERSION_EXTENSION_NAME))
  {
    tinygltf::utils::getValue(*ext, "dispersion", gmat.dispersion);
  }
  return gmat;
}

void tinygltf::utils::setDispersion(tinygltf::Material& tmat, const KHR_materials_dispersion& dispersion)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_DISPERSION_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "dispersion", dispersion.dispersion);
}

KHR_materials_retroreflection tinygltf::utils::getRetroreflection(const tinygltf::Material& tmat)
{
  KHR_materials_retroreflection gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_RETROREFLECTION_EXTENSION_NAME))
  {
    tinygltf::utils::getValue(*ext, "retroreflectionFactor", gmat.retroreflectionFactor);
    tinygltf::utils::getValue(*ext, "retroreflectionTexture", gmat.retroreflectionTexture);
  }
  return gmat;
}

void tinygltf::utils::setRetroreflection(tinygltf::Material& tmat, const KHR_materials_retroreflection& retro)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_RETROREFLECTION_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "retroreflectionFactor", retro.retroreflectionFactor);
  tinygltf::utils::setValue(ext, "retroreflectionTexture", retro.retroreflectionTexture);
}

KHR_node_visibility tinygltf::utils::getNodeVisibility(const tinygltf::Node& node)
{
  KHR_node_visibility gnode;
  if(const auto* ext = tinygltf::utils::findExtension(node.extensions, KHR_NODE_VISIBILITY_EXTENSION_NAME))
  {
    tinygltf::utils::getValue(*ext, "visible", gnode.visible);
  }
  return gnode;
}

void tinygltf::utils::setNodeVisibility(tinygltf::Node& node, const KHR_node_visibility& visibility)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(node.extensions, KHR_NODE_VISIBILITY_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "visible", visibility.visible);
}

KHR_materials_pbrSpecularGlossiness tinygltf::utils::getPbrSpecularGlossiness(const tinygltf::Material& tmat)
{
  KHR_materials_pbrSpecularGlossiness gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_PBR_SPECULAR_GLOSSINESS_EXTENSION_NAME))
  {
    tinygltf::utils::getArrayValue(*ext, "diffuseFactor", gmat.diffuseFactor);
    tinygltf::utils::getValue(*ext, "diffuseTexture", gmat.diffuseTexture);
    tinygltf::utils::getArrayValue(*ext, "specularFactor", gmat.specularFactor);
    tinygltf::utils::getValue(*ext, "glossinessFactor", gmat.glossinessFactor);
    tinygltf::utils::getValue(*ext, "specularGlossinessTexture", gmat.specularGlossinessTexture);
  }
  return gmat;
}

void tinygltf::utils::setPbrSpecularGlossiness(tinygltf::Material& tmat, const KHR_materials_pbrSpecularGlossiness& pbr)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_PBR_SPECULAR_GLOSSINESS_EXTENSION_NAME);
  tinygltf::utils::setArrayValue(ext, "diffuseFactor", 4, glm::value_ptr(pbr.diffuseFactor));
  tinygltf::utils::setArrayValue(ext, "specularFactor", 3, glm::value_ptr(pbr.specularFactor));
  tinygltf::utils::setValue(ext, "glossinessFactor", pbr.glossinessFactor);
  tinygltf::utils::setValue(ext, "diffuseTexture", pbr.diffuseTexture);
  tinygltf::utils::setValue(ext, "specularGlossinessTexture", pbr.specularGlossinessTexture);
}

//-------------------------------------------------------------------------------------------------
//
//


KHR_materials_diffuse_transmission tinygltf::utils::getDiffuseTransmission(const tinygltf::Material& tmat)
{
  KHR_materials_diffuse_transmission gmat;
  if(const auto* ext = tinygltf::utils::findExtension(tmat.extensions, KHR_MATERIALS_DIFFUSE_TRANSMISSION_EXTENSION_NAME))
  {
    tinygltf::utils::getValue(*ext, "diffuseTransmissionFactor", gmat.diffuseTransmissionFactor);
    tinygltf::utils::getValue(*ext, "diffuseTransmissionTexture", gmat.diffuseTransmissionTexture);
    tinygltf::utils::getArrayValue(*ext, "diffuseTransmissionColor", gmat.diffuseTransmissionColor);
    tinygltf::utils::getValue(*ext, "diffuseTransmissionColorTexture", gmat.diffuseTransmissionColorTexture);
  }
  return gmat;
}

void tinygltf::utils::setDiffuseTransmission(tinygltf::Material& tmat, const KHR_materials_diffuse_transmission& diffuseTransmission)
{
  tinygltf::Value& ext = tinygltf::utils::ensureExtension(tmat.extensions, KHR_MATERIALS_DIFFUSE_TRANSMISSION_EXTENSION_NAME);
  tinygltf::utils::setValue(ext, "diffuseTransmissionFactor", diffuseTransmission.diffuseTransmissionFactor);
  tinygltf::utils::setValue(ext, "diffuseTransmissionTexture", diffuseTransmission.diffuseTransmissionTexture);
  tinygltf::utils::setArrayValue(ext, "diffuseTransmissionColor", 3, glm::value_ptr(diffuseTransmission.diffuseTransmissionColor));
  tinygltf::utils::setValue(ext, "diffuseTransmissionColorTexture", diffuseTransmission.diffuseTransmissionColorTexture);
}

//-------------------------------------------------------------------------------------------------
//
//

bool tinygltf::utils::getMeshoptCompression(const tinygltf::BufferView& bview, KHR_meshopt_compression& mcomp)
{
  mcomp = {};

  // Support both KHR and EXT variants of meshopt compression (identical data format)
  const char* extName = nullptr;
  if(tinygltf::utils::hasElementName(bview.extensions, KHR_MESHOPT_COMPRESSION_EXTENSION_NAME))
  {
    extName = KHR_MESHOPT_COMPRESSION_EXTENSION_NAME;
  }
  else if(tinygltf::utils::hasElementName(bview.extensions, EXT_MESHOPT_COMPRESSION_EXTENSION_NAME))
  {
    extName = EXT_MESHOPT_COMPRESSION_EXTENSION_NAME;
  }

  if(extName)
  {
    const tinygltf::Value& ext = tinygltf::utils::getElementValue(bview.extensions, extName);
    tinygltf::utils::getValue(ext, "buffer", mcomp.buffer);
    int64_t byteOffset{0};
    int64_t byteLength{0};
    int64_t byteStride{0};
    int64_t count{0};
    tinygltf::utils::getValue(ext, "byteOffset", byteOffset);
    tinygltf::utils::getValue(ext, "byteLength", byteLength);
    tinygltf::utils::getValue(ext, "byteStride", byteStride);
    tinygltf::utils::getValue(ext, "count", count);
    mcomp.byteOffset = size_t(byteOffset);
    mcomp.byteLength = size_t(byteLength);
    mcomp.byteStride = size_t(byteStride);
    mcomp.count      = size_t(count);

    std::string filter;
    tinygltf::utils::getValue(ext, "filter", filter);
    if(filter == "NONE")
    {
      mcomp.compressionFilter = KHR_meshopt_compression::MESHOPT_COMPRESSION_FILTER_NONE;
    }
    else if(filter == "OCTAHEDRAL")
    {
      mcomp.compressionFilter = KHR_meshopt_compression::MESHOPT_COMPRESSION_FILTER_OCTAHEDRAL;
    }
    else if(filter == "QUATERNION")
    {
      mcomp.compressionFilter = KHR_meshopt_compression::MESHOPT_COMPRESSION_FILTER_QUATERNION;
    }
    else if(filter == "EXPONENTIAL")
    {
      mcomp.compressionFilter = KHR_meshopt_compression::MESHOPT_COMPRESSION_FILTER_EXPONENTIAL;
    }


    std::string mode;
    tinygltf::utils::getValue(ext, "mode", mode);
    if(mode == "ATTRIBUTES")
    {
      mcomp.compressionMode = KHR_meshopt_compression::MESHOPT_COMPRESSION_MODE_ATTRIBUTES;
    }
    else if(mode == "TRIANGLES")
    {
      mcomp.compressionMode = KHR_meshopt_compression::MESHOPT_COMPRESSION_MODE_TRIANGLES;
    }
    else if(mode == "INDICES")
    {
      mcomp.compressionMode = KHR_meshopt_compression::MESHOPT_COMPRESSION_MODE_INDICES;
    }
    return true;
  }
  return false;
}

tinygltf::Value tinygltf::utils::convertToTinygltfValue(int numElements, const float* elements)
{
  tinygltf::Value::Array result;
  result.reserve(numElements);

  for(int i = 0; i < numElements; ++i)
  {
    result.emplace_back(static_cast<double>(elements[i]));
  }

  return tinygltf::Value(std::move(result));
}

tinygltf::Value tinygltf::utils::convertToTinygltfValue(int numElements, const double* elements)
{
  tinygltf::Value::Array result;
  result.reserve(numElements);

  for(int i = 0; i < numElements; ++i)
  {
    result.emplace_back(elements[i]);
  }

  return tinygltf::Value(std::move(result));
}

void tinygltf::utils::getNodeTRS(const tinygltf::Node& node, glm::vec3& translation, glm::quat& rotation, glm::vec3& scale)
{
  // Initialize translation, rotation, and scale to default values
  translation = glm::vec3(0.0f, 0.0f, 0.0f);
  rotation    = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  scale       = glm::vec3(1.0f, 1.0f, 1.0f);

  // Check if the node has a matrix defined
  if(node.matrix.size() == 16)
  {
    glm::mat4 matrix = glm::make_mat4(node.matrix.data());
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(matrix, scale, rotation, translation, skew, perspective);
    return;
  }

  // Retrieve translation if available
  if(node.translation.size() == 3)
  {
    translation = glm::make_vec3(node.translation.data());
  }

  // Retrieve rotation if available
  if(node.rotation.size() == 4)
  {
    rotation.x = float(node.rotation[0]);
    rotation.y = float(node.rotation[1]);
    rotation.z = float(node.rotation[2]);
    rotation.w = float(node.rotation[3]);
  }

  // Retrieve scale if available
  if(node.scale.size() == 3)
  {
    scale = glm::make_vec3(node.scale.data());
  }
}

// -------------------------------------------------------------------------------------------------
// Set the TRS components of a node
// If a component is equal to the default value, it is cleared from the node
void tinygltf::utils::setNodeTRS(tinygltf::Node& node, const glm::vec3& translation, const glm::quat& rotation, const glm::vec3& scale)
{
  if(translation != glm::vec3(0.0f, 0.0f, 0.0f))
  {
    node.translation = {translation.x, translation.y, translation.z};
  }
  else
  {
    node.translation.clear();
  }

  if(rotation != glm::quat(1.0f, 0.0f, 0.0f, 0.0f))
  {
    node.rotation = {rotation.x, rotation.y, rotation.z, rotation.w};
  }
  else
  {
    node.rotation.clear();
  }

  if(scale != glm::vec3(1.0f, 1.0f, 1.0f))
  {
    node.scale = {scale.x, scale.y, scale.z};
  }
  else
  {
    node.scale.clear();
  }
}

glm::mat4 tinygltf::utils::getNodeMatrix(const tinygltf::Node& node)
{
  if(node.matrix.size() == 16)
  {
    return glm::make_mat4(node.matrix.data());
  }

  glm::vec3 translation;
  glm::quat rotation;
  glm::vec3 scale;
  getNodeTRS(node, translation, rotation, scale);

  return glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
}

std::string tinygltf::utils::generatePrimitiveKey(const tinygltf::Primitive& primitive)
{
  std::stringstream o;
  for(const auto& kv : primitive.attributes)
  {
    o << kv.first << ":" << kv.second << " ";
  }
  o << "indices:" << primitive.indices;
  return o.str();
}

void tinygltf::utils::traverseSceneGraph(const tinygltf::Model&                            model,
                                         int                                               nodeID,
                                         const glm::mat4&                                  parentMat,
                                         const std::function<bool(int, const glm::mat4&)>& fctCam /*= nullptr*/,
                                         const std::function<bool(int, const glm::mat4&)>& fctLight /*= nullptr*/,
                                         const std::function<bool(int, const glm::mat4&)>& fctMesh /*= nullptr*/,
                                         const std::function<bool(int, const glm::mat4&)>& anyNode)
{
  const auto& node     = model.nodes[nodeID];
  glm::mat4   worldMat = parentMat * tinygltf::utils::getNodeMatrix(node);

  if(node.camera > -1 && fctCam && fctCam(nodeID, worldMat))
  {
    return;
  }
  if(node.light > -1 && fctLight && fctLight(nodeID, worldMat))
  {
    return;
  }
  if(node.mesh > -1 && fctMesh && fctMesh(nodeID, worldMat))
  {
    return;
  }
  if(anyNode)
  {
    anyNode(nodeID, worldMat);
  }

  for(const auto& child : node.children)
  {
    traverseSceneGraph(model, child, worldMat, fctCam, fctLight, fctMesh, anyNode);
  }
}

size_t tinygltf::utils::getVertexCount(const tinygltf::Model& model, const tinygltf::Primitive& primitive)
{
  const tinygltf::Accessor& vertexAccessor = model.accessors.at(primitive.attributes.at("POSITION"));
  return vertexAccessor.count;
}

size_t tinygltf::utils::getIndexCount(const tinygltf::Model& model, const tinygltf::Primitive& primitive)
{
  if(primitive.indices > -1)
  {
    const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
    return indexAccessor.count;
  }
  // Return the vertex count when no indices are present
  return getVertexCount(model, primitive);
}

int tinygltf::utils::getTextureImageIndex(const tinygltf::Texture& texture)
{
  int source_image = texture.source;

  for(const char* extName : kTextureImageSourceExtensionNames)
  {
    if(hasElementName(texture.extensions, extName))
    {
      const tinygltf::Value& ext = getElementValue(texture.extensions, extName);
      getValue(ext, "source", source_image);
    }
  }

  return source_image;
}

void tinygltf::utils::remapTextureExtensionImageSources(tinygltf::Texture& texture, const std::vector<int>& imageRemap)
{
  auto safeRemap = [&imageRemap](int oldIdx) -> int {
    return (oldIdx >= 0 && oldIdx < static_cast<int>(imageRemap.size())) ? imageRemap[oldIdx] : oldIdx;
  };

  for(const char* extName : kTextureImageSourceExtensionNames)
  {
    auto it = texture.extensions.find(extName);
    if(it == texture.extensions.end())
      continue;
    tinygltf::Value& ext = it->second;
    if(!ext.Has("source"))
      continue;
    const int idx = ext.Get("source").GetNumberAsInt();
    if(idx >= 0)
      ext.Get<tinygltf::Value::Object>()["source"] = tinygltf::Value(safeRemap(idx));
  }
}

void tinygltf::utils::offsetTextureExtensionImageSources(tinygltf::Texture& texture, int imageIndexDelta)
{
  if(imageIndexDelta == 0)
    return;

  for(const char* extName : kTextureImageSourceExtensionNames)
  {
    auto it = texture.extensions.find(extName);
    if(it == texture.extensions.end())
      continue;
    tinygltf::Value& ext = it->second;
    if(!ext.Has("source"))
      continue;
    const int idx = ext.Get("source").GetNumberAsInt();
    if(idx >= 0)
      ext.Get<tinygltf::Value::Object>()["source"] = tinygltf::Value(idx + imageIndexDelta);
  }
}

std::string tinygltf::utils::getTextureUiLabel(const tinygltf::Model& model, int textureIndex)
{
  if(textureIndex < 0 || textureIndex >= static_cast<int>(model.textures.size()))
    return "None";

  const tinygltf::Texture& texture = model.textures[textureIndex];
  if(!texture.name.empty())
    return texture.name + " (tex " + std::to_string(textureIndex) + ")";

  const int imageIndex = getTextureImageIndex(texture);
  if(imageIndex >= 0 && imageIndex < static_cast<int>(model.images.size()))
  {
    const tinygltf::Image& image = model.images[imageIndex];
    if(!image.name.empty())
      return image.name + " (tex " + std::to_string(textureIndex) + ")";
  }

  return "Texture " + std::to_string(textureIndex);
}

// Creating missing tangent attribute
// This is to be set when a material has normalmap, but no tangents.
void tinygltf::utils::createTangentAttribute(tinygltf::Model& model, tinygltf::Primitive& primitive)
{
  // Already have tangents
  if(primitive.attributes.find("TANGENT") != primitive.attributes.end())
  {
    return;
  }

  // Create a new TANGENT attribute
  tinygltf::Accessor tangentAccessor{};
  tangentAccessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
  tangentAccessor.type          = TINYGLTF_TYPE_VEC4;
  tangentAccessor.count         = tinygltf::utils::getVertexCount(model, primitive);
  tangentAccessor.sparse        = {};

  tinygltf::BufferView tangentBufferView{};
  tangentBufferView.buffer     = 0;  // Assume using the first buffer
  tangentBufferView.byteOffset = model.buffers[0].data.size();
  tangentBufferView.byteLength = tangentAccessor.count * 4 * sizeof(float);

  model.buffers[0].data.resize(tangentBufferView.byteOffset + tangentBufferView.byteLength, 0);

  tangentAccessor.bufferView = static_cast<int32_t>(model.bufferViews.size());
  model.bufferViews.emplace_back(tangentBufferView);

  primitive.attributes["TANGENT"] = static_cast<int32_t>(model.accessors.size());
  model.accessors.emplace_back(tangentAccessor);
}


// Current implementation
// http://foundationsofgameenginedev.com/FGED2-sample.pdf
void tinygltf::utils::simpleCreateTangents(tinygltf::Model& model, tinygltf::Primitive& primitive)
{
  const size_t  indexCount  = tinygltf::utils::getIndexCount(model, primitive);
  const size_t  numVertices = tinygltf::utils::getVertexCount(model, primitive);
  const int32_t numFaces    = static_cast<int32_t>(indexCount / 3);

  std::vector<uint32_t>      indexStorage;
  std::vector<glm::vec3>     positionStorage;
  std::vector<glm::vec3>     normalStorage;
  std::vector<glm::vec2>     uvStorage;
  std::span<const uint32_t>  indices   = getAccessorData(model, model.accessors[primitive.indices], &indexStorage);
  std::span<const glm::vec3> positions = getAttributeData3(model, primitive, "POSITION", &positionStorage);
  std::span<const glm::vec3> normals   = getAttributeData3(model, primitive, "NORMAL", &normalStorage);
  std::span<const glm::vec2> uvs       = getAttributeData3(model, primitive, "TEXCOORD_0", &uvStorage);
  // Must not be complex, since we write to it
  std::span<glm::vec4> tangents = getAttributeData3<glm::vec4>(model, primitive, "TANGENT", nullptr);

  if(tangents.data() == nullptr || positions.data() == nullptr)
  {
    LOGE("Attributes are missing\n");
    return;
  }

  bool hasUV     = uvs.data() != nullptr;
  bool hasNormal = normals.data() != nullptr;

  // In case the normal is missing, we will compute it
  std::vector<glm::vec3> geoNormal;
  if(!hasNormal)
  {
    geoNormal.resize(numVertices);
  }


  for(int32_t i = 0; i < numFaces; i++)
  {
    // local index
    uint32_t i0 = indices[i * 3 + 0];
    uint32_t i1 = indices[i * 3 + 1];
    uint32_t i2 = indices[i * 3 + 2];

    const glm::vec3& p0 = positions[i0];
    const glm::vec3& p1 = positions[i1];
    const glm::vec3& p2 = positions[i2];
    glm::vec4&       t0 = tangents[i0];
    glm::vec4&       t1 = tangents[i1];
    glm::vec4&       t2 = tangents[i2];

    // Find the normal or create it
    glm::vec3 n0;
    if(hasNormal)
    {
      n0 = normals[i0];
    }
    else
    {
      geoNormal[i0] = geoNormal[i1] = geoNormal[i2] = glm::normalize(glm::cross(p1 - p0, p2 - p0));
      n0                                            = geoNormal[i0];
    }

    if(hasUV)
    {
      const glm::vec2& uv0 = uvs[i0];
      const glm::vec2& uv1 = uvs[i1];
      const glm::vec2& uv2 = uvs[i2];

      glm::vec3 edge1 = p1 - p0;
      glm::vec3 edge2 = p2 - p0;

      glm::vec2 deltaUV1 = uv1 - uv0;
      glm::vec2 deltaUV2 = uv2 - uv0;

      float f = 1.0F;
      float a = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
      if(fabs(a) > 0.0F)  // Catch degenerated UV
      {
        f = 1.0f / a;
      }

      glm::vec3 tangent   = f * (deltaUV2.y * edge1 - deltaUV1.y * edge2);
      glm::vec3 bitangent = f * (deltaUV2.x * edge1 - deltaUV1.x * edge2);

      // Handedness calculation:
      float handedness = (glm::dot(glm::cross(tangent, bitangent), n0) > 0.0F) ? 1.0F : -1.0F;

      t0 = glm::vec4(tangent + glm::vec3(t0), handedness);
      t1 = glm::vec4(tangent + glm::vec3(t1), handedness);
      t2 = glm::vec4(tangent + glm::vec3(t2), handedness);
    }
    else
    {
      // No UVs, we will use the geometric normal to calculate the tangent
      t0 = t1 = t2 = shaderio::makeFastTangent(n0);
    }
  }


  // Ortho-normalize each tangent and apply the handedness.
  nvutils::parallel_batches(numVertices, [&](uint64_t i) {
    const uint32_t vertex = static_cast<uint32_t>(i);
    glm::vec4&     t0     = tangents[vertex];
    glm::vec3      n0;
    if(hasNormal)
    {
      n0 = normals[vertex];
    }
    else
    {
      n0 = geoNormal[i];
    }

    // Gram-Schmidt orthogonalize
    glm::vec3 ot0 = glm::normalize(glm::vec3(t0) - (glm::dot(n0, glm::vec3(t0)) * n0));

    // In case the tangent is invalid
    if(glm::length2(ot0) < 0.1F || glm::any(glm::isnan(ot0)))
      ot0 = glm::vec3(shaderio::makeFastTangent(n0));

    float handedness = t0.w;
    t0               = glm::vec4(ot0, handedness);
  });
}

void tinygltf::utils::syncExtensionsUsed(tinygltf::Model& model)
{
  // Extension names discovered by walking every object's `extensions` map.
  // Kept in traversal order (deduplicated) so newly added names have a stable position.
  std::vector<std::string>        discovered;
  std::unordered_set<std::string> discoveredSet;

  auto collect = [&discovered, &discoveredSet](const tinygltf::ExtensionMap& ext) {
    for(const auto& kv : ext)
    {
      if(discoveredSet.insert(kv.first).second)
        discovered.push_back(kv.first);
      // Also scan the extension's JSON for extensions nested on extension-owned texture infos.
      collectNestedExtensionNames(kv.second, discovered, discoveredSet);
    }
  };

  // Top-level and asset-level extensions (e.g. KHR_lights_punctual, KHR_materials_variants, XMP).
  collect(model.extensions);
  collect(model.asset.extensions);

  for(size_t i = 0; i < model.scenes.size(); i++)
    collect(model.scenes[i].extensions);

  for(size_t i = 0; i < model.nodes.size(); i++)
    collect(model.nodes[i].extensions);

  for(size_t i = 0; i < model.meshes.size(); i++)
  {
    const tinygltf::Mesh& mesh = model.meshes[i];
    collect(mesh.extensions);
    for(size_t p = 0; p < mesh.primitives.size(); p++)
      collect(mesh.primitives[p].extensions);  // KHR_draco_mesh_compression, KHR_materials_variants
  }

  // Materials carry object-level extensions (KHR_materials_*) and texture-info extensions
  // (KHR_texture_transform) on the typed PBR TextureInfo members below. Extensions nested on
  // extension-owned texture infos (e.g. a transform on a clearcoat texture) live as opaque JSON
  // inside mat.extensions and are picked up by collectNestedExtensionNames() via collect().
  for(size_t i = 0; i < model.materials.size(); i++)
  {
    const tinygltf::Material& mat = model.materials[i];
    collect(mat.extensions);
    collect(mat.pbrMetallicRoughness.extensions);
    collect(mat.pbrMetallicRoughness.baseColorTexture.extensions);
    collect(mat.pbrMetallicRoughness.metallicRoughnessTexture.extensions);
    collect(mat.normalTexture.extensions);
    collect(mat.occlusionTexture.extensions);
    collect(mat.emissiveTexture.extensions);
  }

  for(size_t i = 0; i < model.textures.size(); i++)
    collect(model.textures[i].extensions);  // EXT_texture_webp, MSFT_texture_dds, KHR_texture_basisu
  for(size_t i = 0; i < model.images.size(); i++)
    collect(model.images[i].extensions);
  for(size_t i = 0; i < model.samplers.size(); i++)
    collect(model.samplers[i].extensions);

  for(size_t i = 0; i < model.skins.size(); i++)
    collect(model.skins[i].extensions);

  for(size_t i = 0; i < model.animations.size(); i++)
  {
    const tinygltf::Animation& anim = model.animations[i];
    collect(anim.extensions);
    for(size_t c = 0; c < anim.channels.size(); c++)
    {
      collect(anim.channels[c].extensions);
      collect(anim.channels[c].target_extensions);  // KHR_animation_pointer lives on the channel target
    }
    for(size_t s = 0; s < anim.samplers.size(); s++)
      collect(anim.samplers[s].extensions);
  }

  for(size_t i = 0; i < model.accessors.size(); i++)
  {
    const tinygltf::Accessor& acc = model.accessors[i];
    collect(acc.extensions);
    if(acc.sparse.isSparse)
    {
      collect(acc.sparse.extensions);
      collect(acc.sparse.indices.extensions);
      collect(acc.sparse.values.extensions);
    }
  }

  for(size_t i = 0; i < model.bufferViews.size(); i++)
    collect(model.bufferViews[i].extensions);  // EXT/KHR_meshopt_compression
  for(size_t i = 0; i < model.buffers.size(); i++)
    collect(model.buffers[i].extensions);

  for(size_t i = 0; i < model.cameras.size(); i++)
  {
    const tinygltf::Camera& cam = model.cameras[i];
    collect(cam.extensions);
    collect(cam.perspective.extensions);
    collect(cam.orthographic.extensions);
  }

  for(size_t i = 0; i < model.lights.size(); i++)
    collect(model.lights[i].extensions);

  for(size_t i = 0; i < model.audioEmitters.size(); i++)
  {
    const tinygltf::AudioEmitter& emitter = model.audioEmitters[i];
    collect(emitter.extensions);
    collect(emitter.positional.extensions);
  }
  for(size_t i = 0; i < model.audioSources.size(); i++)
    collect(model.audioSources[i].extensions);

  // Recompute extensionsUsed:
  //  1. keep previously listed names that are still used (preserve original order),
  //  2. append newly discovered names (traversal order),
  //  3. force in any author-declared required name we could not detect. This keeps the
  //     spec invariant (extensionsRequired is a subset of extensionsUsed) and avoids losing
  //     footprint-less required extensions such as KHR_mesh_quantization, which has no
  //     `extensions` entry anywhere and is signaled only by quantized accessor types.
  std::vector<std::string>        newUsed;
  std::unordered_set<std::string> newUsedSet;
  auto                            addUsed = [&newUsed, &newUsedSet](const std::string& name) {
    if(newUsedSet.insert(name).second)
      newUsed.push_back(name);
  };

  for(size_t i = 0; i < model.extensionsUsed.size(); i++)
    if(discoveredSet.count(model.extensionsUsed[i]))
      addUsed(model.extensionsUsed[i]);
  for(size_t i = 0; i < discovered.size(); i++)
    addUsed(discovered[i]);
  for(size_t i = 0; i < model.extensionsRequired.size(); i++)
    addUsed(model.extensionsRequired[i]);

  // extensionsRequired is preserved as authored (deduplicated, order kept). We never add to it;
  // "required" cannot be inferred from the data. The loop above guarantees every entry is also
  // in extensionsUsed, so we simply keep the authored set.
  std::vector<std::string>        newRequired;
  std::unordered_set<std::string> newRequiredSet;
  for(size_t i = 0; i < model.extensionsRequired.size(); i++)
    if(newUsedSet.count(model.extensionsRequired[i]) && newRequiredSet.insert(model.extensionsRequired[i]).second)
      newRequired.push_back(model.extensionsRequired[i]);

  model.extensionsUsed     = std::move(newUsed);
  model.extensionsRequired = std::move(newRequired);
}
