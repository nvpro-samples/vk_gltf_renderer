/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <mutex>
#include <sstream>
#include <span>


#include <glm/glm.hpp>

#include "nvshaders/gltf_scene_io.h.slang"  // Shared between host and device

#include <stb/stb_image.h>
#include "nvimageformats/nv_dds.h"
#include "nvimageformats/nv_ktx.h"
#include "nvimageformats/texture_formats.h"
#include "nvutils/file_mapping.hpp"
#include "nvutils/file_operations.hpp"
#include "nvutils/logger.hpp"
#include "nvutils/timers.hpp"
#include "nvvk/debug_util.hpp"
#include "nvvk/mipmaps.hpp"
#include "nvvk/check_error.hpp"
#include "nvvk/default_structs.hpp"
#include "nvvk/mipmaps.hpp"

#include "gltf_scene_vk.hpp"
#include "nvutils/parallel_work.hpp"
#include "nvvk/helpers.hpp"

// GPU memory category names for scene resources
namespace {
constexpr std::string_view kMemCategoryGeometry  = "Geometry";
constexpr std::string_view kMemCategorySceneData = "SceneData";
constexpr std::string_view kMemCategoryImages    = "Images";

// Convert KTX swizzle to Vulkan component swizzle
VkComponentSwizzle ktxSwizzleToVk(nv_ktx::KTX_SWIZZLE swizzle)
{
  switch(swizzle)
  {
    case nv_ktx::KTX_SWIZZLE::ZERO:
      return VK_COMPONENT_SWIZZLE_ZERO;
    case nv_ktx::KTX_SWIZZLE::ONE:
      return VK_COMPONENT_SWIZZLE_ONE;
    case nv_ktx::KTX_SWIZZLE::R:
      return VK_COMPONENT_SWIZZLE_R;
    case nv_ktx::KTX_SWIZZLE::G:
      return VK_COMPONENT_SWIZZLE_G;
    case nv_ktx::KTX_SWIZZLE::B:
      return VK_COMPONENT_SWIZZLE_B;
    case nv_ktx::KTX_SWIZZLE::A:
      return VK_COMPONENT_SWIZZLE_A;
    default:
      return VK_COMPONENT_SWIZZLE_IDENTITY;
  }
}

// Convert KTX swizzle array to VkComponentMapping
VkComponentMapping ktxSwizzleToVkComponentMapping(const std::array<nv_ktx::KTX_SWIZZLE, 4>& swizzle)
{
  return {ktxSwizzleToVk(swizzle[0]), ktxSwizzleToVk(swizzle[1]), ktxSwizzleToVk(swizzle[2]), ktxSwizzleToVk(swizzle[3])};
}

// Gets the friendly name of a TinyGLTF image for logs and UIs.
std::string getImageName(const tinygltf::Image& img, size_t index)
{
  if(!img.uri.empty())
  {
    return img.uri;
  }

  if(!img.name.empty())
  {
    return img.name;
  }

  return "Embedded image " + std::to_string(index);
}

// Resolves disk path of a tinygltf::Image from its URI; returns an empty path
// if the image is embedded instead.
std::filesystem::path resolveImagePath(const std::filesystem::path& basedir, const tinygltf::Image& img)
{
  if(img.uri.empty())
  {
    return {};
  }

  std::string uriDecoded;  // This is UTF-8, but TinyGlTF uses `char` instead of `char8_t` for it
  tinygltf::URIDecode(img.uri, &uriDecoded, nullptr);  // ex. whitespace may be represented as %20
  return basedir / nvutils::pathFromUtf8(uriDecoded);
}

// Gets the size in bytes of the compressed data of a tinygltf::Image.
size_t getImageByteSize(const tinygltf::Model& model, const tinygltf::Image& img, const std::filesystem::path& diskPath)
{
  // This needs to match the order of preference in loadImage, in case of ambiguity.
  if(img.bufferView >= 0)
  {
    return model.bufferViews[img.bufferView].byteLength;
  }

  if(!img.image.empty())
  {
    return img.image.size();
  }

  std::error_code ec;
  auto            size = std::filesystem::file_size(diskPath, ec);
  return ec ? 0 : size;
}


}  // namespace

//--------------------------------------------------------------------------------------------------
// Forward declaration
std::vector<shaderio::GltfLight> getShaderLights(const std::vector<nvvkgltf::RenderLight>& rlights,
                                                 const std::vector<tinygltf::Light>&       gltfLights);

//-------------------------------------------------------------------------------------------------
//
//

void nvvkgltf::SceneVk::init(nvvk::ResourceAllocator* alloc, nvvk::SamplerPool* samplerPool)
{
  assert(!m_alloc);

  m_device         = alloc->getDevice();
  m_physicalDevice = alloc->getPhysicalDevice();
  m_alloc          = alloc;
  m_samplerPool    = samplerPool;
  m_memoryTracker.init(alloc);
}

void nvvkgltf::SceneVk::deinit()
{
  if(!m_alloc)
  {
    return;
  }

  destroy();

  m_alloc          = nullptr;
  m_samplerPool    = nullptr;
  m_physicalDevice = VK_NULL_HANDLE;
  m_device         = VK_NULL_HANDLE;
}

//--------------------------------------------------------------------------------------------------
// Create all Vulkan resources to hold a nvvkgltf::Scene
//
void nvvkgltf::SceneVk::create(VkCommandBuffer        cmd,
                               nvvk::StagingUploader& staging,
                               const nvvkgltf::Scene& scn,
                               bool                   generateMipmaps /*= true*/,
                               bool                   enableRayTracing /*= true*/)
{
  nvutils::ScopedTimer st(__FUNCTION__);
  destroy();  // Make sure not to leave allocated buffers

  m_generateMipmaps   = generateMipmaps;
  m_rayTracingEnabled = enableRayTracing;

  namespace fs     = std::filesystem;
  fs::path basedir = fs::path(scn.getFilename()).parent_path();
  updateMaterialBuffer(staging, scn);
  updateRenderNodesBuffer(staging, scn);
  createVertexBuffers(cmd, staging, scn);
  createTextureImages(cmd, staging, scn.getModel(), basedir);
  updateRenderLightsBuffer(staging, scn);

  // Update the buffers for morph and skinning
  updateRenderPrimitivesBuffer(cmd, staging, scn);

  updateSceneDescBuffer(staging, scn);
}

void nvvkgltf::SceneVk::update(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn)
{
  updateMaterialBuffer(staging, scn);
  updateRenderNodesBuffer(staging, scn);
  updateRenderPrimitivesBuffer(cmd, staging, scn);
}

//--------------------------------------------------------------------------------------------------
// Destroy only geometry resources (vertex/index buffers, render primitives)
// Preserves textures and materials - useful for geometry-only rebuilds like tangent generation
//
void nvvkgltf::SceneVk::destroyGeometry()
{
  for(auto& vertexBuffer : m_vertexBuffers)
  {
    if(vertexBuffer.position.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, vertexBuffer.position.allocation);
      m_alloc->destroyBuffer(vertexBuffer.position);
    }
    if(vertexBuffer.normal.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, vertexBuffer.normal.allocation);
      m_alloc->destroyBuffer(vertexBuffer.normal);
    }
    if(vertexBuffer.tangent.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, vertexBuffer.tangent.allocation);
      m_alloc->destroyBuffer(vertexBuffer.tangent);
    }
    if(vertexBuffer.texCoord0.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, vertexBuffer.texCoord0.allocation);
      m_alloc->destroyBuffer(vertexBuffer.texCoord0);
    }
    if(vertexBuffer.texCoord1.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, vertexBuffer.texCoord1.allocation);
      m_alloc->destroyBuffer(vertexBuffer.texCoord1);
    }
    if(vertexBuffer.color.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, vertexBuffer.color.allocation);
      m_alloc->destroyBuffer(vertexBuffer.color);
    }
  }
  m_vertexBuffers.clear();

  for(auto& indicesBuffer : m_bIndices)
  {
    if(indicesBuffer.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryGeometry, indicesBuffer.allocation);
      m_alloc->destroyBuffer(indicesBuffer);
    }
  }
  m_bIndices.clear();

  if(m_bRenderPrim.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bRenderPrim.allocation);
    m_alloc->destroyBuffer(m_bRenderPrim);
  }

  if(m_bSceneDesc.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bSceneDesc.allocation);
    m_alloc->destroyBuffer(m_bSceneDesc);
  }
}

//--------------------------------------------------------------------------------------------------
// Recreate only geometry resources (vertex/index buffers, render primitives)
// Call after destroyGeometry() - preserves existing textures
//
void nvvkgltf::SceneVk::createGeometry(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn)
{
  createVertexBuffers(cmd, staging, scn);
  updateRenderPrimitivesBuffer(cmd, staging, scn);

  updateSceneDescBuffer(staging, scn);
}

void nvvkgltf::SceneVk::updateSceneDescBuffer(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn)
{
  // Buffer references
  shaderio::GltfScene scene_desc{};
  scene_desc.materials        = (shaderio::GltfShadeMaterial*)m_bMaterial.address;
  scene_desc.textureInfos     = (shaderio::GltfTextureInfo*)m_bTextureInfos.address;
  scene_desc.renderPrimitives = (shaderio::GltfRenderPrimitive*)m_bRenderPrim.address;
  scene_desc.renderNodes      = (shaderio::GltfRenderNode*)m_bRenderNode.address;
  scene_desc.lights           = (shaderio::GltfLight*)m_bLights.address;
  scene_desc.numLights        = static_cast<int>(scn.getRenderLights().size());

  if(m_bSceneDesc.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(m_bSceneDesc, std::span(&scene_desc, 1).size_bytes(),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
  }
  NVVK_CHECK(staging.appendBuffer(m_bSceneDesc, 0, std::span(&scene_desc, 1)));
  NVVK_DBG_NAME(m_bSceneDesc.buffer);
  m_memoryTracker.track(kMemCategorySceneData, m_bSceneDesc.allocation);
}

template <typename T>
inline shaderio::GltfTextureInfo getTextureInfo(const T& tinfo)
{
  const KHR_texture_transform& transform = tinygltf::utils::getTextureTransform(tinfo);
  const int                    texCoord  = std::min(tinfo.texCoord, 1);  // Only 2 texture coordinates

  // This is the texture info that will be used in the shader
  return {
      .uvTransform = shaderio::float3x2(transform.uvTransform[0][0], transform.uvTransform[1][0],   //
                                        transform.uvTransform[0][1], transform.uvTransform[1][1],   //
                                        transform.uvTransform[0][2], transform.uvTransform[1][2]),  //
      .index       = tinfo.index,
      .texCoord    = texCoord,
  };
}

// Helper to handle texture info and update textureInfos vector
template <typename T>
uint16_t addTextureInfo(const T& tinfo, std::vector<shaderio::GltfTextureInfo>& textureInfos)
{
  shaderio::GltfTextureInfo ti = getTextureInfo(tinfo);
  if(ti.index != -1)
  {
    uint16_t idx = static_cast<uint16_t>(textureInfos.size());
    textureInfos.push_back(ti);
    return idx;
  }
  return 0;  // No texture
}

//--------------------------------------------------------------------------------------------------
// Shared material population logic. TextureHandler should have the signature:
//   void(uint16_t& texIndex, const tinygltf::TextureInfo& srcTexInfo)
// On creation: assign texIndex = addTextureInfo(srcTexInfo, textureInfos)
// On update: if texIndex != 0, update cachedTextureInfos[texIndex] in-place
//
template <typename TextureHandler>
static void populateShaderMaterial(shaderio::GltfShadeMaterial& dstMat, const tinygltf::Material& srcMat, TextureHandler handleTexture)
{
  // Core PBR properties
  dstMat.alphaMode          = srcMat.alphaMode == "OPAQUE" ? 0 : (srcMat.alphaMode == "MASK" ? 1 : 2 /*BLEND*/);
  dstMat.alphaCutoff        = static_cast<float>(srcMat.alphaCutoff);
  dstMat.doubleSided        = srcMat.doubleSided ? 1 : 0;
  dstMat.pbrBaseColorFactor = glm::make_vec4<double>(srcMat.pbrMetallicRoughness.baseColorFactor.data());
  dstMat.pbrMetallicFactor  = static_cast<float>(srcMat.pbrMetallicRoughness.metallicFactor);
  dstMat.pbrRoughnessFactor = static_cast<float>(srcMat.pbrMetallicRoughness.roughnessFactor);
  dstMat.normalTextureScale = static_cast<float>(srcMat.normalTexture.scale);
  dstMat.occlusionStrength  = static_cast<float>(srcMat.occlusionTexture.strength);

  if(!srcMat.emissiveFactor.empty())
    dstMat.emissiveFactor = glm::make_vec3<double>(srcMat.emissiveFactor.data());

  // Core textures
  handleTexture(dstMat.emissiveTexture, srcMat.emissiveTexture);
  handleTexture(dstMat.normalTexture, srcMat.normalTexture);
  handleTexture(dstMat.pbrBaseColorTexture, srcMat.pbrMetallicRoughness.baseColorTexture);
  handleTexture(dstMat.pbrMetallicRoughnessTexture, srcMat.pbrMetallicRoughness.metallicRoughnessTexture);
  handleTexture(dstMat.occlusionTexture, srcMat.occlusionTexture);

  // Extensions
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

//--------------------------------------------------------------------------------------------------
// Create a new shader material, appending texture infos to the vector
//
static void getShaderMaterial(const tinygltf::Material&                 srcMat,
                              std::vector<shaderio::GltfShadeMaterial>& shadeMaterials,
                              std::vector<shaderio::GltfTextureInfo>&   textureInfos)
{
  shaderio::GltfShadeMaterial dstMat = shaderio::defaultGltfMaterial();
  populateShaderMaterial(dstMat, srcMat, [&](uint16_t& texIndex, const auto& srcTexInfo) {
    texIndex = addTextureInfo(srcTexInfo, textureInfos);
  });
  shadeMaterials.emplace_back(dstMat);
}

//--------------------------------------------------------------------------------------------------
// Information about a contiguous span of texture infos used by a material
//  Used to upload the texture infos for a material in a single call
struct TextureInfoSpan
{
  uint16_t minIdx = std::numeric_limits<uint16_t>::max();
  uint16_t maxIdx = 0;
  uint16_t count  = 0;

  bool   hasAny() const { return count > 0; }
  size_t spanSize() const { return static_cast<size_t>(maxIdx - minIdx + 1); }
};

struct MaterialUpdateResult
{
  TextureInfoSpan span{};
  bool            topologyChanged = false;
};

// Update an existing cached material, preserving texture indices, and updating texture info.
// Returns the contiguous texture-info span used by this material and whether texture slot topology changed.
static MaterialUpdateResult updateCachedMaterial(shaderio::GltfShadeMaterial& dstMat,
                                                 const tinygltf::Material&    srcMat,
                                                 shaderio::GltfTextureInfo*   cachedTextureInfos)
{
  MaterialUpdateResult result{};
  populateShaderMaterial(dstMat, srcMat, [&](uint16_t& texIndex, const auto& srcTexInfo) {
    const bool hasSrcTexture = srcTexInfo.index != -1;

    // Check if a texture slot was added or removed
    {
      // The cached material has no texture slots, check if the source material has any texture slots
      if(texIndex == 0)
      {
        if(hasSrcTexture)
          result.topologyChanged = true;  // New texture slots, we need to rebuild the cache
        return;
      }

      // The cached material has texture slots, so we need to check if the source material has any texture slots
      if(!hasSrcTexture)
      {
        result.topologyChanged = true;
        return;
      }
    }

    // Update the cached texture info and the span of texture infos
    cachedTextureInfos[texIndex] = getTextureInfo(srcTexInfo);
    result.span.minIdx           = std::min(result.span.minIdx, texIndex);  // Minimum txt ID in this material
    result.span.maxIdx = std::max(result.span.maxIdx, texIndex);  //Max to check later if the texture infos are contiguous
    result.span.count++;                                          // Count the number of texture infos for this material
  });
  return result;
}

//--------------------------------------------------------------------------------------------------
// Create a buffer of all materials, with only the elements we need
//
void nvvkgltf::SceneVk::updateMaterialBuffer(nvvk::StagingUploader&         staging,
                                             const nvvkgltf::Scene&         scn,
                                             const std::unordered_set<int>& dirtyIndices)
{
  // nvutils::ScopedTimer st(__FUNCTION__);

  using namespace tinygltf;
  const std::vector<tinygltf::Material>& materials = scn.getModel().materials;

  // Rebuild the cached materials and texture infos
  auto rebuildCaches = [&]() {
    m_cachedTextureInfos.clear();
    m_cachedTextureInfos.push_back({});  // 0 is reserved for no texture
    m_cachedShadeMaterials.clear();
    m_cachedShadeMaterials.reserve(materials.size());
    for(const auto& srcMat : materials)
    {
      getShaderMaterial(srcMat, m_cachedShadeMaterials, m_cachedTextureInfos);
    }
  };

  // Ensure that the buffer has the required capacity to avoid resizing the buffer
  // If the buffer is already large enough, return false
  // If the buffer is not large enough, destroy the buffer and create a new one
  // Return true if the buffer was resized
  auto ensureBufferCapacity = [&](nvvk::Buffer& buffer, VkDeviceSize requiredBytes) {
    if(buffer.buffer != VK_NULL_HANDLE && buffer.bufferSize >= requiredBytes)
      return false;

    if(buffer.buffer != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategorySceneData, buffer.allocation);
      m_alloc->destroyBuffer(buffer);
    }

    NVVK_CHECK(m_alloc->createBuffer(buffer, requiredBytes,
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_DBG_NAME(buffer.buffer);
    m_memoryTracker.track(kMemCategorySceneData, buffer.allocation);
    return true;
  };

  // Ensure that the material and texture buffers have the required capacity to avoid resizing the buffers
  // If the buffers are already large enough, return false
  // If the buffers are not large enough, destroy the buffers and create new ones
  // Return true if the buffers were resized
  auto ensureMaterialBuffers = [&]() {
    const VkDeviceSize materialBytes = std::span(m_cachedShadeMaterials).size_bytes();
    const VkDeviceSize textureBytes  = std::span(m_cachedTextureInfos).size_bytes();

    bool resized = false;
    resized |= ensureBufferCapacity(m_bMaterial, materialBytes);
    resized |= ensureBufferCapacity(m_bTextureInfos, textureBytes);
    return resized;
  };

  // If more than half of materials are dirty, a full update is faster (fewer staging calls)
  const bool doFullUpdate = dirtyIndices.empty() || dirtyIndices.size() > materials.size() / 2;

  // Rebuild all materials and texture infos into cache
  if(doFullUpdate)
  {
    rebuildCaches();
  }

  // If the buffer changed, update the scene descriptor buffer (contain the buffer addresses)
  const bool buffersResized = ensureMaterialBuffers();
  if(buffersResized && m_bSceneDesc.buffer != VK_NULL_HANDLE)
  {
    updateSceneDescBuffer(staging, scn);
  }

  // Full update: upload all materials and texture infos (faster when many materials changed)
  if(doFullUpdate || buffersResized)
  {
    staging.appendBuffer(m_bMaterial, 0, std::span(m_cachedShadeMaterials));
    staging.appendBuffer(m_bTextureInfos, 0, std::span(m_cachedTextureInfos));
    return;
  }

  //--------------------------------------------------------------------------------------------------
  // From here, we are doing a surgical update: only process dirty materials,
  // but fall back to full rebuild if texture slots change

  struct PendingUpload
  {
    int             idx = -1;
    TextureInfoSpan span{};
  };

  // We will upload the materials and texture infos for the dirty materials after the process
  std::vector<PendingUpload> pendingUploads;
  pendingUploads.reserve(dirtyIndices.size());

  // We will track if the texture slots changed
  bool topologyChanged = false;

  // Go through the dirty materials indices and update the materials and texture infos
  for(int idx : dirtyIndices)
  {
    if(idx < 0 || idx >= static_cast<int>(materials.size()) || idx >= static_cast<int>(m_cachedShadeMaterials.size()))
      continue;

    shaderio::GltfShadeMaterial& cachedMat = m_cachedShadeMaterials[idx];

    // Update material properties AND texture infos in one pass (parses extensions once)
    MaterialUpdateResult update = updateCachedMaterial(cachedMat, materials[idx], m_cachedTextureInfos.data());
    if(update.topologyChanged)
    {
      topologyChanged = true;  // New texture slots, we need to rebuild the entirecache
      break;
    }

    // Add the material and texture infos to the pending uploads
    pendingUploads.push_back({idx, update.span});
  }

  // If the texture slots changed, we need to rebuild the cache and upload
  // the new materials and texture infos
  if(topologyChanged)
  {
    rebuildCaches();
    const bool resized = ensureMaterialBuffers();
    if(resized)
    {
      // Making sure the scene descriptor buffer is updated with the new material and texture info buffer addresses
      updateSceneDescBuffer(staging, scn);
    }
    // Upload all materials and texture infos
    staging.appendBuffer(m_bMaterial, 0, std::span(m_cachedShadeMaterials));
    staging.appendBuffer(m_bTextureInfos, 0, std::span(m_cachedTextureInfos));
    return;
  }

  // Upload the materials and texture infos for the dirty materials
  for(const PendingUpload& upload : pendingUploads)
  {
    shaderio::GltfShadeMaterial& cachedMat = m_cachedShadeMaterials[upload.idx];

    // Upload material `idx` (only one material per upload)
    size_t matOffset = upload.idx * sizeof(shaderio::GltfShadeMaterial);
    staging.appendBuffer(m_bMaterial, matOffset, sizeof(shaderio::GltfShadeMaterial), &cachedMat);

    // Batch-upload the contiguous texture info range for this material (ex. texID 9, 10, 11)
    if(upload.span.hasAny())
    {
      const size_t spanSize = upload.span.spanSize();
      assert(spanSize == upload.span.count && "Texture infos for a material are expected to be contiguous");

      size_t texOffset = upload.span.minIdx * sizeof(shaderio::GltfTextureInfo);
      staging.appendBuffer(m_bTextureInfos, texOffset, spanSize * sizeof(shaderio::GltfTextureInfo),
                           &m_cachedTextureInfos[upload.span.minIdx]);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Function to blend positions of a primitive with morph targets
static std::vector<glm::vec3> getBlendedPositions(const tinygltf::Accessor&  baseAccessor,
                                                  const glm::vec3*           basePositionData,
                                                  const tinygltf::Primitive& primitive,
                                                  const tinygltf::Mesh&      mesh,
                                                  const tinygltf::Model&     model)
{
  // Prepare for blending positions
  std::vector<glm::vec3> blendedPositions(baseAccessor.count);
  std::copy(basePositionData, basePositionData + baseAccessor.count, blendedPositions.begin());

  // Blend the positions with the morph targets
  for(size_t targetIndex = 0; targetIndex < primitive.targets.size(); ++targetIndex)
  {
    // Retrieve the weight for the current morph target
    float weight = float(mesh.weights[targetIndex]);
    if(weight == 0.0f)
      continue;  // Skip this morph target if its weight is zero

    // Get the morph target attribute (e.g., POSITION)
    const auto& findResult = primitive.targets[targetIndex].find("POSITION");
    if(findResult != primitive.targets[targetIndex].end())
    {
      const tinygltf::Accessor& morphAccessor = model.accessors[findResult->second];
      std::vector<glm::vec3>    tempStorage;
      const std::span<const glm::vec3> morphTargetData = tinygltf::utils::getAccessorData(model, morphAccessor, &tempStorage);

      // Apply the morph target offset in parallel, scaled by the corresponding weight
      nvutils::parallel_batches(blendedPositions.size(),
                                [&](uint64_t v) { blendedPositions[v] += weight * morphTargetData[v]; });
    }
  }

  return blendedPositions;
}

// Unified skinning function that transforms positions, normals, and tangents in a single pass.
//
// Normals are transformed by the inverse-transpose of the joint matrix (correct for non-uniform scaling)
// Tangents are transformed by the upper 3x3 of the joint matrix, preserving the w (handedness) component
//
// Returns spans into workspace buffers - valid until next applySkinning call with same workspace
struct SkinningResult
{
  std::span<const glm::vec3> positions;
  std::span<const glm::vec3> normals;   // Empty if input had no normals
  std::span<const glm::vec4> tangents;  // Empty if input had no tangents
};

static SkinningResult applySkinning(nvvkgltf::SkinningWorkspace&       workspace,
                                    const std::span<const glm::vec3>&  basePositions,
                                    const std::span<const glm::vec3>&  baseNormals,   // Can be empty
                                    const std::span<const glm::vec4>&  baseTangents,  // Can be empty
                                    const std::span<const glm::vec4>&  weights,
                                    const std::span<const glm::ivec4>& joints,
                                    const std::vector<glm::mat4>&      jointMatrices)
{
  const size_t vertexCount = weights.size();
  const bool   hasNormals  = !baseNormals.empty();
  const bool   hasTangents = !baseTangents.empty();
  const size_t numJoints   = jointMatrices.size();

  // Reserve workspace (only allocates if current buffers are too small)
  workspace.reserve(vertexCount, numJoints, hasNormals, hasTangents);

  // Pre-compute normal matrices (inverse-transpose of upper 3x3) once per joint
  for(size_t i = 0; i < numJoints; ++i)
  {
    glm::mat3 upperLeft3x3      = glm::mat3(jointMatrices[i]);
    workspace.normalMatrices[i] = glm::transpose(glm::inverse(upperLeft3x3));
  }

  // Apply skinning to all attributes in a single parallel pass
  nvutils::parallel_batches<2048>(vertexCount, [&](uint64_t v) {
    const glm::vec4&  w = weights[v];
    const glm::ivec4& j = joints[v];

    glm::vec3 skinnedPos(0.0f);
    glm::vec3 skinnedNrm(0.0f);
    glm::vec3 skinnedTan(0.0f);

    // Process all 4 joint influences in one loop
    for(int i = 0; i < 4; ++i)
    {
      const float jointWeight = w[i];
      if(jointWeight > 0.0f)
      {
        const int jointIndex = j[i];

        // Position: transform as point (w=1)
        skinnedPos += jointWeight * glm::vec3(jointMatrices[jointIndex] * glm::vec4(basePositions[v], 1.0f));

        // Normal: transform with inverse-transpose matrix
        if(hasNormals)
          skinnedNrm += jointWeight * (workspace.normalMatrices[jointIndex] * baseNormals[v]);

        // Tangent: transform with upper 3x3 matrix
        if(hasTangents)
          skinnedTan += jointWeight * (glm::mat3(jointMatrices[jointIndex]) * glm::vec3(baseTangents[v]));
      }
    }

    // Store results (normalize direction vectors)
    workspace.positions[v] = skinnedPos;
    if(hasNormals)
      workspace.normals[v] = glm::normalize(skinnedNrm);
    if(hasTangents)
      workspace.tangents[v] = glm::vec4(glm::normalize(skinnedTan), baseTangents[v].w);  // Preserve handedness
  });

  // Return spans into workspace buffers (valid until next applySkinning call with same workspace)
  SkinningResult result;
  result.positions = std::span(workspace.positions.data(), vertexCount);
  result.normals   = hasNormals ? std::span(workspace.normals.data(), vertexCount) : std::span<const glm::vec3>{};
  result.tangents  = hasTangents ? std::span(workspace.tangents.data(), vertexCount) : std::span<const glm::vec4>{};
  return result;
}

//--------------------------------------------------------------------------------------------------
// Array of instance information
// - Use by the vertex shader to retrieve the position of the instance
// - dirtyIndices contains renderNode indices that changed (empty means update all)
void nvvkgltf::SceneVk::updateRenderNodesBuffer(nvvk::StagingUploader&         staging,
                                                const nvvkgltf::Scene&         scn,
                                                const std::unordered_set<int>& dirtyIndices)
{
  // nvutils::ScopedTimer st(__FUNCTION__);

  const std::vector<nvvkgltf::RenderNode>& renderNodes       = scn.getRenderNodes();
  const auto                               buildInstanceInfo = [](const nvvkgltf::RenderNode& renderNode) {
    shaderio::GltfRenderNode info{};
    info.objectToWorld = renderNode.worldMatrix;
    info.worldToObject = glm::inverse(renderNode.worldMatrix);
    info.materialID    = renderNode.materialID;
    info.renderPrimID  = renderNode.renderPrimID;
    return info;
  };

  const bool wasNullBuffer = (m_bRenderNode.buffer == VK_NULL_HANDLE);
  if(wasNullBuffer)
  {
    // Create the buffer early (size is known), fill below
    NVVK_CHECK(m_alloc->createBuffer(m_bRenderNode, renderNodes.size() * sizeof(shaderio::GltfRenderNode),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_DBG_NAME(m_bRenderNode.buffer);
    m_memoryTracker.track(kMemCategorySceneData, m_bRenderNode.allocation);
  }

  if(wasNullBuffer || dirtyIndices.empty())
  {
    // First time or empty dirty set means update all
    std::vector<shaderio::GltfRenderNode> instanceInfo;
    instanceInfo.reserve(renderNodes.size());
    for(const nvvkgltf::RenderNode& renderNode : renderNodes)
    {
      instanceInfo.emplace_back(buildInstanceInfo(renderNode));
    }
    staging.appendBuffer(m_bRenderNode, 0, std::span(instanceInfo));
  }
  else
  {
    // Surgical update: dirtyIndices are renderNode indices
    const size_t renderNodeCount = renderNodes.size();
    for(int renderNodeIdx : dirtyIndices)
    {
      if(renderNodeIdx < 0 || static_cast<size_t>(renderNodeIdx) >= renderNodeCount)
        continue;
      const shaderio::GltfRenderNode info   = buildInstanceInfo(renderNodes[renderNodeIdx]);
      const size_t                   offset = static_cast<size_t>(renderNodeIdx) * sizeof(shaderio::GltfRenderNode);
      staging.appendBuffer(m_bRenderNode, offset, sizeof(shaderio::GltfRenderNode), &info);
    }
  }
}


//--------------------------------------------------------------------------------------------------
// Update the buffer of all lights
// - If the light data was changes, the buffer needs to be updated
// - dirtyIndices contains glTF light indices that changed (empty means update all)
void nvvkgltf::SceneVk::updateRenderLightsBuffer(nvvk::StagingUploader&         staging,
                                                 const nvvkgltf::Scene&         scn,
                                                 const std::unordered_set<int>& dirtyIndices)
{
  const std::vector<nvvkgltf::RenderLight>& rlights = scn.getRenderLights();
  if(rlights.empty())
    return;

  std::vector<shaderio::GltfLight> shaderLights = getShaderLights(rlights, scn.getModel().lights);

  if(m_bLights.buffer == VK_NULL_HANDLE)
  {
    // First time: create buffer and upload all
    NVVK_CHECK(m_alloc->createBuffer(m_bLights, std::span(shaderLights).size_bytes(),
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(staging.appendBuffer(m_bLights, 0, std::span(shaderLights)));
    NVVK_DBG_NAME(m_bLights.buffer);
    m_memoryTracker.track(kMemCategorySceneData, m_bLights.allocation);
  }
  else if(dirtyIndices.empty())
  {
    // Empty dirty set means update all (backward compatibility / full re-parse)
    staging.appendBuffer(m_bLights, 0, std::span(shaderLights));
  }
  else
  {
    // Surgical update: find RenderLights that reference dirty glTF lights
    for(size_t renderLightIdx = 0; renderLightIdx < rlights.size(); ++renderLightIdx)
    {
      if(dirtyIndices.contains(rlights[renderLightIdx].light))
      {
        size_t offset = renderLightIdx * sizeof(shaderio::GltfLight);
        staging.appendBuffer(m_bLights, offset, sizeof(shaderio::GltfLight), &shaderLights[renderLightIdx]);
      }
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Update the buffer of all primitives that have morph targets
//
void nvvkgltf::SceneVk::updateRenderPrimitivesBuffer(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn)
{
  // SCOPED_TIMER(__FUNCTION__);
  const tinygltf::Model& model = scn.getModel();

  // ** Morph **
  for(uint32_t renderPrimID : scn.getMorphPrimitives())
  {
    const nvvkgltf::RenderPrimitive& renderPrimitive  = scn.getRenderPrimitive(renderPrimID);
    const tinygltf::Primitive&       primitive        = *renderPrimitive.pPrimitive;
    const tinygltf::Mesh&            mesh             = model.meshes[renderPrimitive.meshID];
    const tinygltf::Accessor&        positionAccessor = model.accessors[primitive.attributes.at("POSITION")];
    std::vector<glm::vec3>           tempStorage;
    const std::span<const glm::vec3> positionData = tinygltf::utils::getAccessorData(model, positionAccessor, &tempStorage);

    // Get blended position
    std::vector<glm::vec3> blendedPositions = getBlendedPositions(positionAccessor, positionData.data(), primitive, mesh, model);

    // Flush any pending buffer operations and add synchronization before updating morph/skinning buffers
    staging.cmdUploadAppended(cmd);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // Update buffer
    VertexBuffers& vertexBuffers = m_vertexBuffers[renderPrimID];
    staging.appendBuffer(vertexBuffers.position, 0, std::span(blendedPositions));
  }

  // ** Skin **
  const std::vector<nvvkgltf::RenderNode>& renderNodes = scn.getRenderNodes();
  for(uint32_t skinNodeID : scn.getSkinNodes())
  {
    const nvvkgltf::RenderNode& skinNode  = renderNodes[skinNodeID];
    const tinygltf::Skin&       skin      = model.skins[skinNode.skinID];
    const tinygltf::Primitive&  primitive = *scn.getRenderPrimitive(skinNode.renderPrimID).pPrimitive;

    int32_t                numJoints = int32_t(skin.joints.size());
    std::vector<glm::mat4> inverseBindMatrices(numJoints, glm::mat4(1));
    std::vector<glm::mat4> jointMatrices(numJoints, glm::mat4(1));

    if(skin.inverseBindMatrices > -1)
    {
      std::vector<glm::mat4> storage;
      std::span<const glm::mat4> ibm = tinygltf::utils::getAccessorData(model, model.accessors[skin.inverseBindMatrices], &storage);
      for(int i = 0; i < numJoints; i++)
      {
        inverseBindMatrices[i] = ibm[i];
      }
    }

    // Calculate joint matrices
    const std::vector<glm::mat4>& nodeMatrices = scn.getNodesWorldMatrices();
    glm::mat4 invNode = glm::inverse(nodeMatrices[skinNode.refNodeID]);  // Removing current node transform as it will be applied by the shaders
    for(int i = 0; i < numJoints; ++i)
    {
      int jointNodeID = skin.joints[i];
      jointMatrices[i] = invNode * nodeMatrices[jointNodeID] * inverseBindMatrices[i];  // World matrix of the joint's node
    }

    // Get skinning weights and joint indices
    std::vector<glm::vec4> tempWeightStorage;
    std::span<const glm::vec4> weights = tinygltf::utils::getAttributeData3(model, primitive, "WEIGHTS_0", &tempWeightStorage);

    std::vector<glm::ivec4> tempJointStorage;
    std::span<const glm::ivec4> joints = tinygltf::utils::getAttributeData3(model, primitive, "JOINTS_0", &tempJointStorage);

    // Get base vertex attributes
    std::vector<glm::vec3>           tempPosStorage;
    const std::span<const glm::vec3> basePositions =
        tinygltf::utils::getAttributeData3(model, primitive, "POSITION", &tempPosStorage);

    std::vector<glm::vec3> tempNrmStorage;
    const std::span<const glm::vec3> baseNormals = tinygltf::utils::getAttributeData3(model, primitive, "NORMAL", &tempNrmStorage);

    std::vector<glm::vec4> tempTanStorage;
    const std::span<const glm::vec4> baseTangents = tinygltf::utils::getAttributeData3(model, primitive, "TANGENT", &tempTanStorage);

    // Apply skinning to all attributes in a single pass
    SkinningResult skinned =
        applySkinning(m_skinningWorkspace, basePositions, baseNormals, baseTangents, weights, joints, jointMatrices);

    // Flush any pending buffer operations and add synchronization before updating skinning buffers
    staging.cmdUploadAppended(cmd);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // Update GPU buffers
    VertexBuffers& vertexBuffers = m_vertexBuffers[skinNode.renderPrimID];
    staging.appendBuffer(vertexBuffers.position, 0, skinned.positions);

    // Sanity check: skinned results and GPU buffers should be consistent (both derive from primitive attributes)
    assert(skinned.normals.empty() == (vertexBuffers.normal.buffer == VK_NULL_HANDLE));
    assert(skinned.tangents.empty() == (vertexBuffers.tangent.buffer == VK_NULL_HANDLE));

    if(!skinned.normals.empty() && vertexBuffers.normal.buffer != VK_NULL_HANDLE)
    {
      staging.appendBuffer(vertexBuffers.normal, 0, skinned.normals);
    }

    if(!skinned.tangents.empty() && vertexBuffers.tangent.buffer != VK_NULL_HANDLE)
    {
      staging.appendBuffer(vertexBuffers.tangent, 0, skinned.tangents);
    }
  }
}

// Function to create attribute buffers in Vulkan only if the attribute is present
// Return true if a buffer was created, false if the buffer was updated
template <typename T>
bool nvvkgltf::SceneVk::updateAttributeBuffer(const std::string&         attributeName,
                                              const tinygltf::Model&     model,
                                              const tinygltf::Primitive& primitive,
                                              nvvk::ResourceAllocator*   alloc,
                                              nvvk::StagingUploader*     staging,
                                              nvvk::Buffer&              attributeBuffer)
{
  const auto& findResult = primitive.attributes.find(attributeName);
  if(findResult != primitive.attributes.end())
  {
    const tinygltf::Accessor& accessor = model.accessors[findResult->second];
    std::vector<T>            tempStorage;
    const std::span<const T>  data = tinygltf::utils::getAccessorData(model, accessor, &tempStorage);
    if(data.empty())
    {
      return false;  // The data was invalid
    }

    if(attributeBuffer.buffer == VK_NULL_HANDLE)
    {
      // We add VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT so it can be bound to
      // a vertex input binding:
      VkBufferUsageFlags2 bufferUsageFlag = getBufferUsageFlags() | VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT;
      NVVK_CHECK(alloc->createBuffer(attributeBuffer, std::span(data).size_bytes(), bufferUsageFlag));
      NVVK_CHECK(staging->appendBuffer(attributeBuffer, 0, std::span(data)));
      m_memoryTracker.track(kMemCategoryGeometry, attributeBuffer.allocation);
      return true;
    }
    else
    {
      staging->appendBuffer(attributeBuffer, 0, std::span(data));
    }
  }
  return false;
}

//--------------------------------------------------------------------------------------------------
// Returns the common usage flags used for all buffers.
VkBufferUsageFlags2 nvvkgltf::SceneVk::getBufferUsageFlags() const
{
  VkBufferUsageFlags2 bufferUsageFlag =
      VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT           // Buffer read/write access within shaders, without size limitation
      | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT  // The buffer can be referred to using its address instead of a binding
      | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT           // Buffer can be copied into
      | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;          // Buffer can be copied from (e.g. for inspection)

  if(m_rayTracingEnabled)
  {
    // Usage as a data source for acceleration structure builds
    bufferUsageFlag |= VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  }

  return bufferUsageFlag;
}

//--------------------------------------------------------------------------------------------------
// Creating information per primitive
// - Create a buffer of Vertex and Index for each primitive
// - Each primInfo has a reference to the vertex and index buffer, and which material id it uses
//
void nvvkgltf::SceneVk::createVertexBuffers(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn)
{
  nvutils::ScopedTimer st(__FUNCTION__);

  const auto& model = scn.getModel();

  std::vector<shaderio::GltfRenderPrimitive> renderPrim;  // The array of all primitive information

  size_t numUniquePrimitive = scn.getNumRenderPrimitives();
  m_bIndices.resize(numUniquePrimitive);
  m_vertexBuffers.resize(numUniquePrimitive);
  renderPrim.resize(numUniquePrimitive);

  for(size_t primID = 0; primID < scn.getNumRenderPrimitives(); primID++)
  {
    const tinygltf::Primitive& primitive     = *scn.getRenderPrimitive(primID).pPrimitive;
    const tinygltf::Mesh&      mesh          = model.meshes[scn.getRenderPrimitive(primID).meshID];
    VertexBuffers&             vertexBuffers = m_vertexBuffers[primID];

    updateAttributeBuffer<glm::vec3>("POSITION", model, primitive, m_alloc, &staging, vertexBuffers.position);
    updateAttributeBuffer<glm::vec3>("NORMAL", model, primitive, m_alloc, &staging, vertexBuffers.normal);
    updateAttributeBuffer<glm::vec2>("TEXCOORD_0", model, primitive, m_alloc, &staging, vertexBuffers.texCoord0);
    updateAttributeBuffer<glm::vec2>("TEXCOORD_1", model, primitive, m_alloc, &staging, vertexBuffers.texCoord1);
    updateAttributeBuffer<glm::vec4>("TANGENT", model, primitive, m_alloc, &staging, vertexBuffers.tangent);

    if(tinygltf::utils::hasElementName(primitive.attributes, "COLOR_0"))
    {
      // For color, we need to pack it into a single int
      const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("COLOR_0")];
      std::vector<uint32_t>     tempIntData(accessor.count);
      if(accessor.type == TINYGLTF_TYPE_VEC3)
      {
        std::vector<glm::vec3>     tempStorage;
        std::span<const glm::vec3> colors = tinygltf::utils::getAccessorData(model, accessor, &tempStorage);
        for(size_t i = 0; i < accessor.count; i++)
        {
          tempIntData[i] = glm::packUnorm4x8(glm::vec4(colors[i], 1));
        }
      }
      else if(accessor.type == TINYGLTF_TYPE_VEC4)
      {
        std::vector<glm::vec4>     tempStorage;
        std::span<const glm::vec4> colors = tinygltf::utils::getAccessorData(model, accessor, &tempStorage);
        for(size_t i = 0; i < accessor.count; i++)
        {
          tempIntData[i] = glm::packUnorm4x8(colors[i]);
        }
      }
      else
      {
        assert(!"Unknown color type");
      }

      NVVK_CHECK(m_alloc->createBuffer(vertexBuffers.color, std::span(tempIntData).size_bytes(),
                                       getBufferUsageFlags() | VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT));
      NVVK_CHECK(staging.appendBuffer(vertexBuffers.color, 0, std::span(tempIntData)));
      m_memoryTracker.track(kMemCategoryGeometry, vertexBuffers.color.allocation);
    }

    // Debug name
    if(vertexBuffers.position.buffer != VK_NULL_HANDLE)
      NVVK_DBG_NAME(vertexBuffers.position.buffer);
    if(vertexBuffers.normal.buffer != VK_NULL_HANDLE)
      NVVK_DBG_NAME(vertexBuffers.normal.buffer);
    if(vertexBuffers.texCoord0.buffer != VK_NULL_HANDLE)
      NVVK_DBG_NAME(vertexBuffers.texCoord0.buffer);
    if(vertexBuffers.texCoord1.buffer != VK_NULL_HANDLE)
      NVVK_DBG_NAME(vertexBuffers.texCoord1.buffer);
    if(vertexBuffers.tangent.buffer != VK_NULL_HANDLE)
      NVVK_DBG_NAME(vertexBuffers.tangent.buffer);
    if(vertexBuffers.color.buffer != VK_NULL_HANDLE)
      NVVK_DBG_NAME(vertexBuffers.color.buffer);


    // Buffer of indices
    std::vector<uint32_t> indexBuffer;
    if(primitive.indices > -1)
    {
      const tinygltf::Accessor& accessor = model.accessors[primitive.indices];
      bool                      ok       = tinygltf::utils::copyAccessorData(model, accessor, indexBuffer);
      assert(ok);
    }
    else
    {  // Primitive without indices, creating them
      const tinygltf::Accessor& accessor = model.accessors[primitive.attributes.at("POSITION")];

      indexBuffer.resize(accessor.count);
      for(auto i = 0; i < accessor.count; i++)
        indexBuffer[i] = i;
    }

    // Creating the buffer for the indices
    nvvk::Buffer& i_buffer = m_bIndices[primID];
    NVVK_CHECK(m_alloc->createBuffer(i_buffer, std::span(indexBuffer).size_bytes(),
                                     getBufferUsageFlags() | VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT));
    NVVK_CHECK(staging.appendBuffer(i_buffer, 0, std::span(indexBuffer)));
    NVVK_DBG_NAME(i_buffer.buffer);
    m_memoryTracker.track(kMemCategoryGeometry, i_buffer.allocation);

    // Filling the primitive information
    renderPrim[primID].indices = (glm::uvec3*)i_buffer.address;

    shaderio::VertexBuffers vBuf = {};
    vBuf.positions               = (glm::vec3*)vertexBuffers.position.address;
    vBuf.normals                 = (glm::vec3*)vertexBuffers.normal.address;
    vBuf.tangents                = (glm::vec4*)vertexBuffers.tangent.address;
    vBuf.texCoords0              = (glm::vec2*)vertexBuffers.texCoord0.address;
    vBuf.texCoords1              = (glm::vec2*)vertexBuffers.texCoord1.address;
    vBuf.colors                  = (glm::uint*)vertexBuffers.color.address;

    renderPrim[primID].vertexBuffer = vBuf;
  }

  // Creating the buffer of all primitive information
  NVVK_CHECK(m_alloc->createBuffer(m_bRenderPrim, std::span(renderPrim).size_bytes(), getBufferUsageFlags()));
  NVVK_CHECK(staging.appendBuffer(m_bRenderPrim, 0, std::span(renderPrim)));
  NVVK_DBG_NAME(m_bRenderPrim.buffer);
  m_memoryTracker.track(kMemCategorySceneData, m_bRenderPrim.allocation);

  // Barrier to make sure the data is in the GPU
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  if(m_rayTracingEnabled)
  {
    barrier.dstAccessMask |= VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
  }
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0,
                       nullptr, 0, nullptr);
}

// This version updates all the vertex buffers
void nvvkgltf::SceneVk::updateVertexBuffers(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scene)
{
  const auto& model = scene.getModel();

  for(size_t primID = 0; primID < scene.getNumRenderPrimitives(); primID++)
  {
    const tinygltf::Primitive& primitive     = *scene.getRenderPrimitive(primID).pPrimitive;
    VertexBuffers&             vertexBuffers = m_vertexBuffers[primID];
    bool                       newBuffer     = false;
    updateAttributeBuffer<glm::vec3>("POSITION", model, primitive, m_alloc, &staging, vertexBuffers.position);
    newBuffer |= updateAttributeBuffer<glm::vec3>("NORMAL", model, primitive, m_alloc, &staging, vertexBuffers.normal);
    newBuffer |= updateAttributeBuffer<glm::vec2>("TEXCOORD_0", model, primitive, m_alloc, &staging, vertexBuffers.texCoord0);
    newBuffer |= updateAttributeBuffer<glm::vec2>("TEXCOORD_1", model, primitive, m_alloc, &staging, vertexBuffers.texCoord1);
    newBuffer |= updateAttributeBuffer<glm::vec4>("TANGENT", model, primitive, m_alloc, &staging, vertexBuffers.tangent);

    // A buffer was created (most likely tangent buffer), we need to update the RenderPrimitive buffer
    if(newBuffer)
    {
      shaderio::GltfRenderPrimitive renderPrim{};  // The array of all primitive information
      renderPrim.indices                 = (glm::uvec3*)m_bIndices[primID].address;
      renderPrim.vertexBuffer.positions  = (glm::vec3*)vertexBuffers.position.address;
      renderPrim.vertexBuffer.normals    = (glm::vec3*)vertexBuffers.normal.address;
      renderPrim.vertexBuffer.tangents   = (glm::vec4*)vertexBuffers.tangent.address;
      renderPrim.vertexBuffer.texCoords0 = (glm::vec2*)vertexBuffers.texCoord0.address;
      renderPrim.vertexBuffer.texCoords1 = (glm::vec2*)vertexBuffers.texCoord1.address;
      renderPrim.vertexBuffer.colors     = (glm::uint*)vertexBuffers.color.address;
      staging.appendBuffer(m_bRenderPrim, sizeof(shaderio::GltfRenderPrimitive) * primID, std::span(&renderPrim, 1));
    }
  }
}


//--------------------------------------------------------------------------------------------------------------
// Returning the Vulkan sampler information from the information in the tinygltf
//
static VkSamplerCreateInfo getSampler(const tinygltf::Model& model, int index)
{
  VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerInfo.minFilter  = VK_FILTER_LINEAR;
  samplerInfo.magFilter  = VK_FILTER_LINEAR;
  samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerInfo.maxLod     = VK_LOD_CLAMP_NONE;

  if(index < 0)
    return samplerInfo;

  const auto& sampler = model.samplers[index];

  const std::map<int, VkFilter> filters = {{9728, VK_FILTER_NEAREST}, {9729, VK_FILTER_LINEAR},
                                           {9984, VK_FILTER_NEAREST}, {9985, VK_FILTER_LINEAR},
                                           {9986, VK_FILTER_NEAREST}, {9987, VK_FILTER_LINEAR}};

  const std::map<int, VkSamplerMipmapMode> mipmapModes = {
      {9728, VK_SAMPLER_MIPMAP_MODE_NEAREST}, {9729, VK_SAMPLER_MIPMAP_MODE_LINEAR},
      {9984, VK_SAMPLER_MIPMAP_MODE_NEAREST}, {9985, VK_SAMPLER_MIPMAP_MODE_LINEAR},
      {9986, VK_SAMPLER_MIPMAP_MODE_NEAREST}, {9987, VK_SAMPLER_MIPMAP_MODE_LINEAR}};

  const std::map<int, VkSamplerAddressMode> wrapModes = {
      {TINYGLTF_TEXTURE_WRAP_REPEAT, VK_SAMPLER_ADDRESS_MODE_REPEAT},
      {TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE},
      {TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT}};

  if(sampler.minFilter > -1)
    samplerInfo.minFilter = filters.at(sampler.minFilter);
  if(sampler.magFilter > -1)
  {
    samplerInfo.magFilter  = filters.at(sampler.magFilter);
    samplerInfo.mipmapMode = mipmapModes.at(sampler.magFilter);
  }
  samplerInfo.addressModeU = wrapModes.at(sampler.wrapS);
  samplerInfo.addressModeV = wrapModes.at(sampler.wrapT);

  return samplerInfo;
}

//--------------------------------------------------------------------------------------------------------------
// This is creating all images stored in textures
//
void nvvkgltf::SceneVk::createTextureImages(VkCommandBuffer              cmd,
                                            nvvk::StagingUploader&       staging,
                                            const tinygltf::Model&       model,
                                            const std::filesystem::path& basedir)
{
  nvutils::ScopedTimer st(std::string(__FUNCTION__) + "\n");

  VkSamplerCreateInfo default_sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  default_sampler.minFilter  = VK_FILTER_LINEAR;
  default_sampler.magFilter  = VK_FILTER_LINEAR;
  default_sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  default_sampler.maxLod     = VK_LOD_CLAMP_NONE;

  // Find and all textures/images that should be sRgb encoded.
  findSrgbImages(model);

  // Make dummy image(1,1), needed as we cannot have an empty array
  auto addDefaultImage = [&](uint32_t idx, const std::array<uint8_t, 4>& color) {
    VkImageCreateInfo image_create_info = DEFAULT_VkImageCreateInfo;
    image_create_info.extent            = {1, 1, 1};
    image_create_info.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    nvvk::Image image;
    //
    NVVK_CHECK(m_alloc->createImage(image, image_create_info, DEFAULT_VkImageViewCreateInfo));
    NVVK_CHECK(staging.appendImage(image, std::span<const uint8_t>(color.data(), 4), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    NVVK_DBG_NAME(image.image);
    // assert(idx < m_images.size());
    m_images[idx] = SceneImage{.imageTexture = image};
    nvvk::DebugUtil::getInstance().setObjectName(m_images[idx].imageTexture.image, "Dummy");
  };

  // Adds a texture that points to image 0, so that every texture points to some image.
  auto addDefaultTexture = [&]() {
    assert(!m_images.empty());
    nvvk::Image tex = m_images[0].imageTexture;
    NVVK_CHECK(m_samplerPool->acquireSampler(tex.descriptor.sampler));
    NVVK_DBG_NAME(tex.descriptor.sampler);
    m_textures.push_back(tex);
  };

  // Collect images that are in use by textures
  // If an image is not used, it will not be loaded. Instead, a dummy image will be created to avoid modifying the texture image source index.
  std::set<int> usedImages;
  for(const auto& texture : model.textures)
  {
    int source_image = tinygltf::utils::getTextureImageIndex(texture);
    usedImages.insert(source_image);
  }

  // Load images in parallel, sorting by their size so larger images come first
  // for better multi-thread utilization. While we do this we also resolve
  // file paths and image names.
  m_images.resize(model.images.size());
  struct ImageLoadItem
  {
    std::filesystem::path diskPath{};
    size_t                numBytes{};
    uint64_t              imageId{};
  };
  std::vector<ImageLoadItem> imageLoadItems;
  const std::string          indent = st.indent();
  for(size_t i = 0; i < model.images.size(); i++)
  {
    if(usedImages.find(static_cast<int>(i)) == usedImages.end())
      continue;  // Skip unused images

    const auto& gltfImage = model.images[i];

    ImageLoadItem item{.imageId = i};
    item.diskPath       = resolveImagePath(basedir, gltfImage);
    item.numBytes       = getImageByteSize(model, gltfImage, item.diskPath);
    m_images[i].imgName = getImageName(gltfImage, i);

    imageLoadItems.push_back(std::move(item));
    LOGI("%s(%" PRIu64 ") %s \n", indent.c_str(), i, m_images[i].imgName.c_str());
  }

  std::sort(imageLoadItems.begin(), imageLoadItems.end(),
            [](const ImageLoadItem& a, const ImageLoadItem& b) { return a.numBytes > b.numBytes; });

  nvutils::parallel_batches<1>(  // Not batching
      imageLoadItems.size(), [&](uint64_t i) {
        const ImageLoadItem& item = imageLoadItems[i];
        loadImage(item.diskPath, model, item.imageId);
      });

  // Create Vulkan images
  for(size_t i = 0; i < m_images.size(); i++)
  {
    if(!createImage(cmd, staging, m_images[i]))
    {
      addDefaultImage((uint32_t)i, {255, 0, 255, 255});  // Image not present or incorrectly loaded (image.empty)
    }
  }

  // Add default image if nothing was loaded
  if(model.images.empty())
  {
    m_images.resize(1);
    addDefaultImage(0, {255, 255, 255, 255});
  }

  // Creating the textures using the above images
  m_textures.reserve(model.textures.size());
  for(size_t i = 0; i < model.textures.size(); i++)
  {
    const auto& texture      = model.textures[i];
    int         source_image = tinygltf::utils::getTextureImageIndex(texture);

    if(source_image >= model.images.size() || source_image < 0)
    {
      addDefaultTexture();  // Incorrect source image
      continue;
    }

    VkSamplerCreateInfo sampler = getSampler(model, texture.sampler);

    SceneImage& sceneImage = m_images[source_image];

    nvvk::Image tex = sceneImage.imageTexture;
    NVVK_CHECK(m_samplerPool->acquireSampler(tex.descriptor.sampler, sampler));
    NVVK_DBG_NAME(tex.descriptor.sampler);
    m_textures.push_back(tex);
  }

  // Add a default texture, cannot work with empty descriptor set
  if(model.textures.empty())
  {
    addDefaultTexture();
  }
}

//-------------------------------------------------------------------------------------------------
// Some images must be sRgb encoded, we find them and will be uploaded with the _SRGB format.
//
void nvvkgltf::SceneVk::findSrgbImages(const tinygltf::Model& model)
{
  // Lambda helper functions
  auto addImage = [&](int texID) {
    if(texID > -1)
    {
      const tinygltf::Texture& texture = model.textures[texID];
      m_sRgbImages.insert(tinygltf::utils::getTextureImageIndex(texture));
    }
  };

  // For images in extensions
  auto addImageFromExtension = [&](const tinygltf::Material& mat, const std::string extName, const std::string name) {
    const auto& ext = mat.extensions.find(extName);
    if(ext != mat.extensions.end())
    {
      if(ext->second.Has(name))
        addImage(ext->second.Get(name).Get<int>());
    }
  };

  // Loop over all materials and find the sRgb textures
  for(tinygltf::Material const& mat : model.materials)
  {
    // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#metallic-roughness-material
    addImage(mat.pbrMetallicRoughness.baseColorTexture.index);
    addImage(mat.emissiveTexture.index);

    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_specular/README.md#extending-materials
    addImageFromExtension(mat, "KHR_materials_specular", "specularColorTexture");

    // https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_sheen/README.md#sheen
    addImageFromExtension(mat, "KHR_materials_sheen", "sheenColorTexture");

    // **Deprecated** but still used with some scenes
    // https://kcoley.github.io/glTF/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness
    addImageFromExtension(mat, "KHR_materials_pbrSpecularGlossiness", "diffuseTexture");
    addImageFromExtension(mat, "KHR_materials_pbrSpecularGlossiness", "specularGlossinessTexture");
  }

  // Special, if the 'extra' in the texture has a gamma defined greater than 1.0, it is sRGB
  for(tinygltf::Texture const& texture : model.textures)
  {
    if(texture.extras.Has("gamma") && texture.extras.Get("gamma").GetNumberAsDouble() > 1.0)
    {
      m_sRgbImages.insert(tinygltf::utils::getTextureImageIndex(texture));
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Loads glTF image `imageID` into m_images[imageID].
//
void nvvkgltf::SceneVk::loadImage(const std::filesystem::path& diskPath, const tinygltf::Model& model, uint64_t imageID)
{
  const tinygltf::Image& gltfImage = model.images[imageID];
  SceneImage&            outImage  = m_images[imageID];

  // Is this an embedded image?
  const int bufferViewIndex = gltfImage.bufferView;
  if(bufferViewIndex >= 0)
  {
    // Get the buffer data; make sure it's in range.
    // Images use buffer views, so we can't use nvvkgltf's Accessor utilities
    // and must load it manually.
    if(static_cast<size_t>(bufferViewIndex) >= model.bufferViews.size())
    {
      LOGW("The buffer view index (%i) for image %" PRIu64 " was out of range.\n", bufferViewIndex, imageID);
      return;
    }

    const tinygltf::BufferView& bufferView  = model.bufferViews[bufferViewIndex];
    const int                   bufferIndex = bufferView.buffer;
    if(bufferIndex < 0 || static_cast<size_t>(bufferIndex) >= model.buffers.size())
    {
      LOGW("The buffer index (%i) from the buffer view (%i) for image %" PRIu64 " was out of range.\n", bufferIndex,
           bufferViewIndex, imageID);
      return;
    }

    const tinygltf::Buffer& buffer = model.buffers[bufferIndex];
    // Make sure the data's in-bounds. TinyGLTF doesn't seem to verify this (!)
    const size_t byteOffset = bufferView.byteOffset;
    const size_t byteLength = bufferView.byteLength;
    if(byteOffset > buffer.data.size() || byteLength > buffer.data.size() - byteOffset)
    {
      LOGW("The buffer offset (%zu) and length (%zu) were out-of-range for buffer %i, which has length %zu, for image %" PRIu64 ".\n",
           byteOffset, byteLength, bufferIndex, buffer.data.size(), imageID);
      return;
    }

    loadImageFromMemory(imageID, &buffer.data[byteOffset], byteLength);
  }
  else if(!gltfImage.image.empty())
  {
    // Image data was stored by our callback (e.g., from a data URI)
    loadImageFromMemory(imageID, gltfImage.image.data(), gltfImage.image.size());
  }
  else if(!diskPath.empty())
  {
    // Image from disk
    nvutils::FileReadMapping fileMapping;
    if(!fileMapping.open(diskPath))
    {
      LOGW("The file for image %" PRIu64 " (%s) could not be opened.\n", imageID, nvutils::utf8FromPath(diskPath).c_str());
      return;
    }

    loadImageFromMemory(imageID, fileMapping.data(), fileMapping.size());
  }
  else
  {
    LOGW("Image %" PRIu64 " has no data source (no bufferView, no stored data, and no URI).\n", imageID);
  }
}

//-------------------------------------------------------------------------------------------------
// Loads data, extent, and swizzle for an image loaded or mapped to a range of
// memory into m_images[imageID] without changing the other fields.
//
void nvvkgltf::SceneVk::loadImageFromMemory(uint64_t imageID, const void* data, size_t byteLength)
{
  SceneImage& image = m_images[imageID];
  image.srgb        = m_sRgbImages.find(static_cast<int>(imageID)) != m_sRgbImages.end();

  // Try the custom image load callback first.
  if(m_imageLoadCallback && m_imageLoadCallback(image, data, byteLength))
  {
    return;  // Successfully loaded
  }

  // Look at the first few bytes to determine the type of the image and choose
  // between our other loaders.
  const char    ddsIdentifier[4] = {'D', 'D', 'S', ' '};
  const uint8_t ktxIdentifier[5] = {0xAB, 0x4B, 0x54, 0x58, 0x20};  // Common for KTX1 + KTX2

  if(byteLength >= sizeof(ddsIdentifier) && memcmp(data, ddsIdentifier, sizeof(ddsIdentifier)) == 0)
  {
    nv_dds::Image        ddsImage{};
    nv_dds::ReadSettings settings{};
    const nv_dds::ErrorWithText readResult = ddsImage.readFromMemory(reinterpret_cast<const char*>(data), byteLength, settings);
    if(readResult.has_value())
    {
      LOGW("Failed to read image %" PRIu64 " using nv_dds: %s\n", imageID, readResult.value().c_str());
      return;
    }

    image.size.width  = ddsImage.getWidth(0);
    image.size.height = ddsImage.getHeight(0);
    if(ddsImage.getDepth(0) > 1)
    {
      LOGW("This DDS image had a depth of %u, but loadImage() cannot handle volume textures.\n", ddsImage.getDepth(0));
      return;
    }
    if(ddsImage.getNumFaces() > 1)
    {
      LOGW("This DDS image had %u faces, but loadImage() cannot handle cubemaps.\n", ddsImage.getNumFaces());
      return;
    }
    if(ddsImage.getNumLayers() > 1)
    {
      LOGW("This DDS image had %u array elements, but loadImage() cannot handle array textures.\n", ddsImage.getNumLayers());
      return;
    }
    image.format = texture_formats::dxgiToVulkan(ddsImage.dxgiFormat);
    image.format = texture_formats::tryForceVkFormatTransferFunction(image.format, image.srgb);
    if(VK_FORMAT_UNDEFINED == image.format)
    {
      LOGW("Could not determine a VkFormat for DXGI format %u (%s).\n", ddsImage.dxgiFormat,
           texture_formats::getDXGIFormatName(ddsImage.dxgiFormat));
      return;
    }

    // Add all mip-levels. We don't need the ddsImage after this so we can move instead of copy.
    for(uint32_t i = 0; i < ddsImage.getNumMips(); i++)
    {
      std::vector<char>& mip = ddsImage.subresource(i, 0, 0).data;
      image.mipData.push_back(std::move(mip));
    }
  }
  else if(byteLength >= sizeof(ktxIdentifier) && memcmp(data, ktxIdentifier, sizeof(ktxIdentifier)) == 0)
  {
    nv_ktx::KTXImage            ktxImage;
    const nv_ktx::ReadSettings  ktxReadSettings;
    const nv_ktx::ErrorWithText maybeError =
        ktxImage.readFromMemory(reinterpret_cast<const char*>(data), byteLength, ktxReadSettings);
    if(maybeError.has_value())
    {
      LOGW("Failed to read image %" PRIu64 " using nv_ktx: %s\n", imageID, maybeError->c_str());
      return;
    }

    image.size.width  = ktxImage.mip_0_width;
    image.size.height = ktxImage.mip_0_height;
    if(ktxImage.mip_0_depth > 1)
    {
      LOGW("KTX image %" PRIu64 " had a depth of %u, but loadImage() cannot handle volume textures.\n", imageID,
           ktxImage.mip_0_depth);
      return;
    }
    if(ktxImage.num_faces > 1)
    {
      LOGW("KTX image %" PRIu64 " had %u faces, but loadImage() cannot handle cubemaps.\n", imageID, ktxImage.num_faces);
      return;
    }
    if(ktxImage.num_layers_possibly_0 > 1)
    {
      LOGW("KTX image %" PRIu64 " had %u array elements, but loadImage() cannot handle array textures.\n", imageID,
           ktxImage.num_layers_possibly_0);
      return;
    }
    image.format           = texture_formats::tryForceVkFormatTransferFunction(ktxImage.format, image.srgb);
    image.componentMapping = ktxSwizzleToVkComponentMapping(ktxImage.swizzle);

    // Add all mip-levels. We don't need the ktxImage after this so we can move instead of copy.
    for(uint32_t i = 0; i < ktxImage.num_mips; i++)
    {
      std::vector<char>& mip = ktxImage.subresource(i, 0, 0);
      image.mipData.push_back(std::move(mip));
    }
  }
  else
  {
    // Try to load the image using stb_image.
    if(byteLength > std::numeric_limits<int>::max())
    {
      LOGW("File for image %" PRIu64 " was too large (%zu bytes) for stb_image to read.\n", imageID, byteLength);
      return;
    }

    // stb_image wants a buffer of type stbi_uc* and length of type int:
    const stbi_uc* dataStb   = reinterpret_cast<const stbi_uc*>(data);
    const int      lengthStb = static_cast<int>(byteLength);

    // Read the header once to check how many channels it has. We can't trivially use RGB/VK_FORMAT_R8G8B8_UNORM and
    // need to set requiredComponents=4 in such cases.
    int w = 0, h = 0, comp = 0;
    if(!stbi_info_from_memory(dataStb, lengthStb, &w, &h, &comp))
    {
      LOGW("Failed to get info using stb_image for image %" PRIu64 "\n", imageID);
      return;
    }

    // Read the header again to check if it has 16 bit data, e.g. for a heightmap.
    const bool is16Bit = stbi_is_16_bit_from_memory(dataStb, lengthStb);

    // Load the image
    stbi_uc* decompressed = nullptr;
    size_t   bytesPerPixel{0};
    int      requiredComponents = comp == 1 ? 1 : 4;
    if(is16Bit)
    {
      stbi_us* decompressed16 = stbi_load_16_from_memory(dataStb, lengthStb, &w, &h, &comp, requiredComponents);
      bytesPerPixel           = sizeof(*decompressed16) * requiredComponents;
      decompressed            = (stbi_uc*)(decompressed16);
    }
    else
    {
      decompressed  = stbi_load_from_memory(dataStb, lengthStb, &w, &h, &comp, requiredComponents);
      bytesPerPixel = sizeof(*decompressed) * requiredComponents;
    }

    switch(requiredComponents)
    {
      case 1:
        image.format = is16Bit ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
        // For 1-component textures, expand the single channel to RGB for proper grayscale display
        image.componentMapping = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE};
        break;
      case 4:
        image.format = is16Bit    ? VK_FORMAT_R16G16B16A16_UNORM :
                       image.srgb ? VK_FORMAT_R8G8B8A8_SRGB :
                                    VK_FORMAT_R8G8B8A8_UNORM;

        break;
    }

    // Make a copy of the image data to be uploaded to Vulkan later
    if(decompressed && w > 0 && h > 0 && image.format != VK_FORMAT_UNDEFINED)
    {
      VkDeviceSize bufferSize = static_cast<VkDeviceSize>(w) * h * bytesPerPixel;
      image.size              = VkExtent2D{(uint32_t)w, (uint32_t)h};
      image.mipData           = {{decompressed, decompressed + bufferSize}};
    }

    stbi_image_free(decompressed);
  }
}

bool nvvkgltf::SceneVk::createImage(const VkCommandBuffer& cmd, nvvk::StagingUploader& staging, SceneImage& image)
{
  if(image.size.width == 0 || image.size.height == 0)
    return false;

  VkFormat   format  = image.format;
  VkExtent2D imgSize = image.size;

  // Check if we can generate mipmap with the the incoming image
  bool               canGenerateMipmaps = false;
  VkFormatProperties formatProperties;
  vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &formatProperties);
  if((formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) == VK_FORMAT_FEATURE_BLIT_DST_BIT)
  {
    canGenerateMipmaps = true;
  }
  VkImageCreateInfo imageCreateInfo = DEFAULT_VkImageCreateInfo;
  imageCreateInfo.extent            = VkExtent3D{imgSize.width, imgSize.height, 1};
  imageCreateInfo.format            = format;
  imageCreateInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  // Mip-mapping images were defined (.ktx, .dds), use the number of levels defined
  if(image.mipData.size() > 1)
  {
    imageCreateInfo.mipLevels = static_cast<uint32_t>(image.mipData.size());
  }
  else if(canGenerateMipmaps && m_generateMipmaps)
  {
    // Compute the number of mipmaps levels
    imageCreateInfo.mipLevels = nvvk::mipLevels(imgSize);
  }

  // Use custom view info with component mapping (e.g. for grayscale textures)
  VkImageViewCreateInfo imageViewCreateInfo = DEFAULT_VkImageViewCreateInfo;
  imageViewCreateInfo.components            = image.componentMapping;

  nvvk::Image resultImage;
  NVVK_CHECK(m_alloc->createImage(resultImage, imageCreateInfo, imageViewCreateInfo));
  NVVK_DBG_NAME(resultImage.image);
  NVVK_DBG_NAME(resultImage.descriptor.imageView);

  // Track the image allocation
  m_memoryTracker.track(kMemCategoryImages, resultImage.allocation);

  // Set the initial layout to TRANSFER_DST_OPTIMAL
  resultImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;  // Setting this, tells the appendImage that the image is in this layout (no need to transfer)
  nvvk::cmdImageMemoryBarrier(cmd, {resultImage.image, VK_IMAGE_LAYOUT_UNDEFINED, resultImage.descriptor.imageLayout});
  NVVK_CHECK(staging.appendImage(resultImage, std::span(image.mipData[0]), resultImage.descriptor.imageLayout));
  staging.cmdUploadAppended(cmd);  // Upload the first mip level

  // The image require to generate the mipmaps
  if(image.mipData.size() == 1 && (canGenerateMipmaps && m_generateMipmaps))
  {
    nvvk::cmdGenerateMipmaps(cmd, resultImage.image, imgSize, imageCreateInfo.mipLevels, 1, resultImage.descriptor.imageLayout);
  }
  else
  {
    for(uint32_t mip = 1; mip < (uint32_t)imageCreateInfo.mipLevels; mip++)
    {
      imageCreateInfo.extent.width  = std::max(1u, image.size.width >> mip);
      imageCreateInfo.extent.height = std::max(1u, image.size.height >> mip);

      VkOffset3D               offset{};
      VkImageSubresourceLayers subresource{};
      subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      subresource.layerCount = 1;
      subresource.mipLevel   = mip;

      if(imageCreateInfo.extent.width > 0 && imageCreateInfo.extent.height > 0)
      {
        staging.appendImageSub(resultImage, offset, imageCreateInfo.extent, subresource, std::span(image.mipData[mip]));
      }
    }
    // Upload all the mip levels
    staging.cmdUploadAppended(cmd);
  }
  // Barrier to change the layout to SHADER_READ_ONLY_OPTIMAL
  resultImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  nvvk::cmdImageMemoryBarrier(cmd, {resultImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, resultImage.descriptor.imageLayout});

  if(!image.imgName.empty())
  {
    nvvk::DebugUtil::getInstance().setObjectName(resultImage.image, image.imgName);
  }
  else
  {
    NVVK_DBG_NAME(resultImage.image);
  }

  // Clear image.mipData as it is no longer needed
  // image.srgb and image.imgName are preserved
  image.imageTexture = resultImage;
  image.mipData.clear();

  return true;
}

std::vector<shaderio::GltfLight> getShaderLights(const std::vector<nvvkgltf::RenderLight>& renderlights,
                                                 const std::vector<tinygltf::Light>&       gltfLights)
{
  std::vector<shaderio::GltfLight> lightsInfo;
  lightsInfo.reserve(renderlights.size());
  for(auto& l : renderlights)
  {
    const auto& gltfLight = gltfLights[l.light];

    shaderio::GltfLight info{};
    info.position   = l.worldMatrix[3];
    info.direction  = -l.worldMatrix[2];  // glm::vec3(l.worldMatrix * glm::vec4(0, 0, -1, 0));
    info.innerAngle = static_cast<float>(gltfLight.spot.innerConeAngle);
    info.outerAngle = static_cast<float>(gltfLight.spot.outerConeAngle);
    if(gltfLight.color.size() == 3)
      info.color = glm::vec3(gltfLight.color[0], gltfLight.color[1], gltfLight.color[2]);
    else
      info.color = glm::vec3(1, 1, 1);  // default color (white)
    info.intensity = static_cast<float>(gltfLight.intensity);
    info.type      = gltfLight.type == "point" ? shaderio::eLightTypePoint :
                     gltfLight.type == "spot"  ? shaderio::eLightTypeSpot :
                                                 shaderio::eLightTypeDirectional;

    info.radius = gltfLight.extras.Has("radius") ? float(gltfLight.extras.Get("radius").GetNumberAsDouble()) : 0.0f;

    if(info.type == shaderio::eLightTypeDirectional)
    {
      const double sun_distance     = 149597870.0;  // km
      double       angular_size_rad = 2.0 * std::atan(info.radius / sun_distance);
      info.angularSizeOrInvRange    = static_cast<float>(angular_size_rad);
    }
    else
    {
      info.angularSizeOrInvRange = (gltfLight.range > 0.0) ? 1.0f / static_cast<float>(gltfLight.range) : 0.0f;
    }

    lightsInfo.emplace_back(info);
  }
  return lightsInfo;
}

void nvvkgltf::SceneVk::destroy()
{
  // Destroy geometry (vertex/index buffers, render primitives, scene descriptor)
  destroyGeometry();

  // Destroy remaining scene data buffers
  if(m_bMaterial.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bMaterial.allocation);
    m_alloc->destroyBuffer(m_bMaterial);
  }
  if(m_bTextureInfos.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bTextureInfos.allocation);
    m_alloc->destroyBuffer(m_bTextureInfos);
  }
  if(m_bLights.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bLights.allocation);
    m_alloc->destroyBuffer(m_bLights);
  }
  if(m_bRenderNode.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bRenderNode.allocation);
    m_alloc->destroyBuffer(m_bRenderNode);
  }

  // Destroy textures and images
  for(auto& texture : m_textures)
  {
    m_samplerPool->releaseSampler(texture.descriptor.sampler);
  }
  for(auto& image : m_images)
  {
    if(image.imageTexture.image != VK_NULL_HANDLE)
    {
      m_memoryTracker.untrack(kMemCategoryImages, image.imageTexture.allocation);
      m_alloc->destroyImage(image.imageTexture);
    }
  }
  m_images.clear();
  m_textures.clear();

  m_sRgbImages.clear();

  // Release CPU skinning workspace memory
  m_skinningWorkspace.clear();
  m_cachedShadeMaterials.clear();
  m_cachedTextureInfos.clear();
}

//////////////////////////////////////////////////////////////////////////
///
/// SkinningWorkspace
///
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Growing the workspace buffers if needed
void nvvkgltf::SkinningWorkspace::reserve(size_t vertexCount, size_t jointCount, bool needNormals, bool needTangents)
{
  if(normalMatrices.size() < jointCount)
    normalMatrices.resize(jointCount);
  if(positions.size() < vertexCount)
    positions.resize(vertexCount);
  if(needNormals && normals.size() < vertexCount)
    normals.resize(vertexCount);
  if(needTangents && tangents.size() < vertexCount)
    tangents.resize(vertexCount);
}

//--------------------------------------------------------------------------------------------------
// Clearing the workspace buffers
void nvvkgltf::SkinningWorkspace::clear()
{
  normalMatrices = {};
  positions      = {};
  normals        = {};
  tangents       = {};
}
