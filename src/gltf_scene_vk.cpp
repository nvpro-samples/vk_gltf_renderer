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

//
// Creates and manages Vulkan GPU resources for rendering a glTF scene.
// Uploads geometry buffers (vertices, indices), loads and creates texture
// images, builds scene-data buffers (materials, instances, render nodes),
// and maintains descriptor sets. Handles resource updates for animations.
//

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <span>


#include <glm/glm.hpp>

#include "nvshaders/gltf_scene_io.h.slang"  // Shared between host and device

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
#include "gltf_scene_animation.hpp"
#include "gltf_image_loader.hpp"
#include "nvutils/parallel_work.hpp"
#include "nvvk/helpers.hpp"

namespace nvvkgltf {
namespace {

// After images are decoded for upload, mirror width/height into tinygltf::Image for UI/tools (same
// path as loadFromMemory: DDS, KTX, WebP callback, stb, etc.).
void syncTinyGltfImageDimensionsFromLoadedImages(tinygltf::Model& model, const std::vector<SceneVk::SceneImage>& images)
{
  if(model.images.size() != images.size())
    return;
  for(size_t i = 0; i < model.images.size(); ++i)
  {
    const VkExtent2D& ext = images[i].size;
    if(ext.width > 0 && ext.height > 0)
    {
      model.images[i].width  = static_cast<int>(ext.width);
      model.images[i].height = static_cast<int>(ext.height);
    }
  }
}

}  // namespace
}  // namespace nvvkgltf

// GPU memory category names for scene resources
namespace {
constexpr std::string_view kMemCategoryGeometry  = "Geometry";
constexpr std::string_view kMemCategorySceneData = "SceneData";
constexpr std::string_view kMemCategoryImages    = "Images";

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

//--------------------------------------------------------------------------------------------------
// SceneVk implementation grouped by work area:
//   1. Init / Deinit  - allocator setup and teardown
//   2. Create         - full or partial resource creation (create, createGeometry, buffers, textures)
//   3. Sync / Upload  - push scene data to GPU (syncFromScene, upload*)
//   4. Destroy        - release geometry or all resources (destroyGeometry, destroy)
//--------------------------------------------------------------------------------------------------

//========== Init / Deinit ==========

//--------------------------------------------------------------------------------------------------
// Initialize the Vulkan scene with allocator and sampler pool. Must be called before create().
void nvvkgltf::SceneVk::init(nvvk::ResourceAllocator* alloc, nvvk::SamplerPool* samplerPool)
{
  assert(!m_alloc);

  m_device         = alloc->getDevice();
  m_physicalDevice = alloc->getPhysicalDevice();
  m_alloc          = alloc;
  m_samplerPool    = samplerPool;
  m_memoryTracker.init(alloc);
}

//--------------------------------------------------------------------------------------------------
// Tear down allocator reference and release all resources. Idempotent if already deinit.
void nvvkgltf::SceneVk::destroyBufferDeferred(nvvk::Buffer& buf)
{
  if(buf.buffer == VK_NULL_HANDLE)
    return;

  nvvk::Buffer             oldBuf  = buf;
  nvvk::ResourceAllocator* alloc   = m_alloc;
  GpuMemoryTracker*        tracker = &m_memoryTracker;
  auto                     cleanup = [=]() mutable {
    tracker->untrack(kMemCategorySceneData, oldBuf.allocation);
    alloc->destroyBuffer(oldBuf);
  };
  buf = {};

  if(m_deferredFree)
  {
    m_deferredFree(std::move(cleanup));
  }
  else
  {
    if(m_graphicsQueue)
      vkQueueWaitIdle(m_graphicsQueue);
    cleanup();
  }
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

//========== Create ==========

//--------------------------------------------------------------------------------------------------
// Create all Vulkan resources for the given scene: materials, render nodes, vertex/index buffers,
// textures, lights, and scene descriptor. Calls destroy() first to ensure a clean state.
// Note: does NOT apply morph/skinning deformation to vertex buffers. The caller is responsible
// for running the initial animation pass (GPU compute or CPU fallback) after creation.
void nvvkgltf::SceneVk::create(VkCommandBuffer        cmd,
                               nvvk::StagingUploader& staging,
                               nvvkgltf::Scene&       scn,
                               bool                   generateMipmaps /*= true*/,
                               bool                   enableRayTracing /*= true*/)
{
  nvutils::ScopedTimer st(__FUNCTION__);
  destroy();  // Make sure not to leave allocated buffers

  m_generateMipmaps   = generateMipmaps;
  m_rayTracingEnabled = enableRayTracing;

  std::vector<std::filesystem::path> imageSearchPaths = scn.getImageSearchPaths();
  if(imageSearchPaths.empty())
  {
    std::error_code       ec;
    std::filesystem::path baseDir = std::filesystem::absolute(scn.getFilename().parent_path(), ec);
    if(!ec)
      imageSearchPaths.push_back(baseDir);
  }

  uploadMaterials(staging, scn);
  uploadRenderNodes(staging, scn);
  createVertexBuffers(cmd, staging, scn);
  createTextureImages(cmd, staging, scn, imageSearchPaths);
  uploadLights(staging, scn);

  (void)flushSceneDescIfDirty(staging, scn);
}

//========== Sync / Upload ==========

//--------------------------------------------------------------------------------------------------
// Sync GPU buffers from Scene dirty flags. Uploads materials, lights, and/or render nodes according
// to mask; clears the corresponding dirty sets. Returns a bitmask of which categories were updated.
uint32_t nvvkgltf::SceneVk::syncFromScene(nvvk::StagingUploader& staging, nvvkgltf::Scene& scn, uint32_t mask)
{
  uint32_t result = eSyncNone;
  auto&    df     = scn.getDirtyFlags();

  constexpr float fullUpdateRatio = nvvkgltf::kFullUpdateRatio;

  if(mask & eSyncMaterials)
  {
    const auto&                            dirty     = df.materials;
    const std::vector<tinygltf::Material>& materials = scn.getModel().materials;
    const VkDeviceSize requiredMaterialBytes         = materials.size() * sizeof(shaderio::GltfShadeMaterial);
    if(!dirty.empty() || m_bMaterial.buffer == VK_NULL_HANDLE || m_bMaterial.bufferSize < requiredMaterialBytes)
    {
      uploadMaterials(staging, scn, dirty);
      df.materials.clear();
      result |= eSyncMaterials;
    }
  }

  if(mask & eSyncLights)
  {
    const auto&                     dirty         = df.lights;
    const std::vector<RenderLight>& rlights       = scn.getRenderLights();
    const VkDeviceSize              requiredBytes = rlights.size() * sizeof(shaderio::GltfLight);
    if(!rlights.empty() && (dirty.empty() == false || m_bLights.buffer == VK_NULL_HANDLE || m_bLights.bufferSize != requiredBytes))
    {
      uploadLights(staging, scn, dirty);
      df.lights.clear();
      result |= eSyncLights;
    }
  }

  if(mask & eSyncRenderNodes)
  {
    const auto&        dirty         = df.renderNodesVk;
    const auto&        renderNodes   = scn.getRenderNodes();
    const VkDeviceSize requiredBytes = renderNodes.size() * sizeof(shaderio::GltfRenderNode);
    const bool         needsUpdate = df.allRenderNodesDirty || !dirty.empty() || m_bRenderNode.buffer == VK_NULL_HANDLE
                             || m_bRenderNode.bufferSize != requiredBytes;
    if(needsUpdate)
    {
      const bool useFullUpdate = df.allRenderNodesDirty || renderNodes.empty()
                                 || float(dirty.size()) / float(renderNodes.size()) >= fullUpdateRatio;
      uploadRenderNodes(staging, scn, useFullUpdate ? std::unordered_set<int>{} : dirty);
      df.renderNodesVk.clear();
      df.allRenderNodesDirty = false;
      result |= eSyncRenderNodes;
    }
  }

  if(flushSceneDescIfDirty(staging, scn))
    result |= eSyncRenderNodes;  // Staging was appended; ensure caller flushes.

  return result;
}

//--------------------------------------------------------------------------------------------------
// Recreate only geometry (vertex/index buffers). Call after destroyGeometry(); preserves textures
// and materials. The caller is responsible for running the initial animation pass after this.
void nvvkgltf::SceneVk::createGeometry(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn)
{
  m_sceneDescDirty = true;  // Geometry buffers may be recreated below.
  createVertexBuffers(cmd, staging, scn);
  (void)flushSceneDescIfDirty(staging, scn);
}

//--------------------------------------------------------------------------------------------------
// Update or create the scene descriptor buffer (GPU pointer to materials, textures, primitives,
// render nodes, lights). Called when buffer addresses change (e.g. after create or buffer resize).
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
    NVVK_DBG_NAME(m_bSceneDesc.buffer);
    m_memoryTracker.track(kMemCategorySceneData, m_bSceneDesc.allocation);  // Only track when creating!
  }
  NVVK_CHECK(staging.appendBuffer(m_bSceneDesc, 0, std::span(&scene_desc, 1)));
}

//--------------------------------------------------------------------------------------------------
// Flush scene descriptor buffer once per sync cycle if any buffer address changed.
bool nvvkgltf::SceneVk::flushSceneDescIfDirty(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn)
{
  if(!m_sceneDescDirty)
    return false;
  m_sceneDescDirty = false;
  updateSceneDescBuffer(staging, scn);
  return true;
}

//--------------------------------------------------------------------------------------------------
// Upload material and texture-info data to GPU. dirtyIndices = material indices to update;
// empty = full upload. Supports surgical update when few materials change.
void nvvkgltf::SceneVk::uploadMaterials(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn, const std::unordered_set<int>& dirtyIndices)
{
  // nvutils::ScopedTimer st(__FUNCTION__);

  using namespace tinygltf;
  const std::vector<tinygltf::Material>& materials = scn.getModel().materials;

  const std::vector<shaderio::GltfShadeMaterial>& shadeMaterials = m_materialCache.getShadeMaterials();
  const std::vector<shaderio::GltfTextureInfo>&   textureInfos   = m_materialCache.getTextureInfos();

  // Ensure that the buffer has the required capacity to avoid resizing the buffer
  // If the buffer is already large enough, return false
  // If the buffer is not large enough, destroy the buffer and create a new one
  // Return true if the buffer was resized
  auto ensureBufferCapacity = [&](nvvk::Buffer& buffer, VkDeviceSize requiredBytes) {
    if(buffer.buffer != VK_NULL_HANDLE && buffer.bufferSize >= requiredBytes)
      return false;

    if(buffer.buffer != VK_NULL_HANDLE)
    {
      destroyBufferDeferred(buffer);
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
    const VkDeviceSize materialBytes = std::span(shadeMaterials).size_bytes();
    const VkDeviceSize textureBytes  = std::span(textureInfos).size_bytes();

    bool resized = false;
    resized |= ensureBufferCapacity(m_bMaterial, materialBytes);
    resized |= ensureBufferCapacity(m_bTextureInfos, textureBytes);
    return resized;
  };

  const bool doFullUpdate = dirtyIndices.empty() || float(dirtyIndices.size()) / float(materials.size()) >= nvvkgltf::kFullUpdateRatio;

  // Rebuild all materials and texture infos into cache
  if(doFullUpdate)
  {
    m_materialCache.buildFromMaterials(materials);
  }

  const bool buffersResized = ensureMaterialBuffers();
  if(buffersResized && m_bSceneDesc.buffer != VK_NULL_HANDLE)
    m_sceneDescDirty = true;

  // Full update: upload all materials and texture infos (faster when many materials changed)
  if(doFullUpdate || buffersResized)
  {
    staging.appendBuffer(m_bMaterial, 0, std::span(shadeMaterials));
    staging.appendBuffer(m_bTextureInfos, 0, std::span(textureInfos));
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

  std::vector<PendingUpload> pendingUploads;
  pendingUploads.reserve(dirtyIndices.size());
  bool topologyChanged = false;

  for(int idx : dirtyIndices)
  {
    if(idx < 0 || idx >= static_cast<int>(materials.size()) || idx >= static_cast<int>(shadeMaterials.size()))
      continue;

    MaterialUpdateResult update = m_materialCache.updateMaterial(idx, materials[idx]);
    if(update.topologyChanged)
    {
      topologyChanged = true;
      break;
    }
    pendingUploads.push_back({idx, update.span});
  }

  if(topologyChanged)
  {
    m_materialCache.buildFromMaterials(materials);
    const bool resized = ensureMaterialBuffers();
    if(resized)
      m_sceneDescDirty = true;
    const std::vector<shaderio::GltfShadeMaterial>& shadeMaterialsRebuilt = m_materialCache.getShadeMaterials();
    const std::vector<shaderio::GltfTextureInfo>&   textureInfosRebuilt   = m_materialCache.getTextureInfos();
    staging.appendBuffer(m_bMaterial, 0, std::span(shadeMaterialsRebuilt));
    staging.appendBuffer(m_bTextureInfos, 0, std::span(textureInfosRebuilt));
    return;
  }

  for(const PendingUpload& upload : pendingUploads)
  {
    const shaderio::GltfShadeMaterial& cachedMat = m_materialCache.getShadeMaterials()[upload.idx];

    size_t matOffset = upload.idx * sizeof(shaderio::GltfShadeMaterial);
    staging.appendBuffer(m_bMaterial, matOffset, sizeof(shaderio::GltfShadeMaterial), &cachedMat);

    if(upload.span.hasAny())
    {
      const size_t spanSize = upload.span.spanSize();
      assert(spanSize == upload.span.count && "Texture infos for a material are expected to be contiguous");

      size_t texOffset = upload.span.minIdx * sizeof(shaderio::GltfTextureInfo);
      staging.appendBuffer(m_bTextureInfos, texOffset, spanSize * sizeof(shaderio::GltfTextureInfo),
                           &m_materialCache.getTextureInfos()[upload.span.minIdx]);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Upload render node (instance) data to GPU SSBO. dirtyIndices = render node indices to update;
// empty = full upload. Resizes buffer if node count changed.
// Convert CPU render node to GPU format.
static shaderio::GltfRenderNode buildRenderNodeInfo(const nvvkgltf::RenderNode& renderNode)
{
  shaderio::GltfRenderNode info{};
  info.objectToWorld = renderNode.worldMatrix;
  info.worldToObject = glm::inverse(renderNode.worldMatrix);
  info.materialID    = renderNode.materialID;
  info.renderPrimID  = renderNode.renderPrimID;
  return info;
}

//--------------------------------------------------------------------------------------------------
// Ensure the render node GPU buffer matches the required size. Destroys and nulls the buffer if
// the current size doesn't match, so the caller can detect wasNullBuffer and recreate.
void nvvkgltf::SceneVk::ensureRenderNodeBuffer(nvvk::StagingUploader& staging, size_t renderNodeCount)
{
  const size_t requiredBytes = renderNodeCount * sizeof(shaderio::GltfRenderNode);
  if(m_bRenderNode.bufferSize != requiredBytes)
  {
    if(m_bRenderNode.buffer != VK_NULL_HANDLE)
    {
      destroyBufferDeferred(m_bRenderNode);
    }
    m_bRenderNode = {};
  }

  if(renderNodeCount == 0)
    return;

  if(m_bRenderNode.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(m_bRenderNode, requiredBytes,
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_DBG_NAME(m_bRenderNode.buffer);
    m_memoryTracker.track(kMemCategorySceneData, m_bRenderNode.allocation);
    m_sceneDescDirty = true;
  }
}

void nvvkgltf::SceneVk::uploadRenderNodes(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn, const std::unordered_set<int>& dirtyIndices)
{
  const std::vector<nvvkgltf::RenderNode>& renderNodes = scn.getRenderNodes();

  const VkDeviceSize prevSize = m_bRenderNode.bufferSize;
  ensureRenderNodeBuffer(staging, renderNodes.size());
  if(renderNodes.empty())
    return;

  const bool bufferRecreated = (m_bRenderNode.bufferSize != prevSize) || (prevSize == 0);
  if(bufferRecreated || dirtyIndices.empty())
  {
    std::vector<shaderio::GltfRenderNode> instanceInfo;
    instanceInfo.reserve(renderNodes.size());
    for(const nvvkgltf::RenderNode& rn : renderNodes)
      instanceInfo.emplace_back(buildRenderNodeInfo(rn));
    staging.appendBuffer(m_bRenderNode, 0, std::span(instanceInfo));
  }
  else
  {
    for(int renderNodeIdx : dirtyIndices)
    {
      if(renderNodeIdx < 0 || static_cast<size_t>(renderNodeIdx) >= renderNodes.size())
        continue;
      const shaderio::GltfRenderNode info   = buildRenderNodeInfo(renderNodes[renderNodeIdx]);
      const size_t                   offset = static_cast<size_t>(renderNodeIdx) * sizeof(shaderio::GltfRenderNode);
      staging.appendBuffer(m_bRenderNode, offset, sizeof(shaderio::GltfRenderNode), &info);
    }
  }

#ifndef NDEBUG
  debugUpdateShadowCopy(scn);
#endif
}

//--------------------------------------------------------------------------------------------------
// Upload light data to GPU. dirtyIndices = glTF light indices that changed; empty = full upload.
void nvvkgltf::SceneVk::uploadLights(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn, const std::unordered_set<int>& dirtyIndices)
{
  const std::vector<nvvkgltf::RenderLight>& rlights = scn.getRenderLights();
  if(rlights.empty())
    return;

  std::vector<shaderio::GltfLight> shaderLights = getShaderLights(rlights, scn.getModel().lights);

  const VkDeviceSize requiredBytes = std::span(shaderLights).size_bytes();
  const VkDeviceSize prevSize      = m_bLights.bufferSize;

  if(m_bLights.bufferSize != requiredBytes)
  {
    if(m_bLights.buffer != VK_NULL_HANDLE)
      destroyBufferDeferred(m_bLights);
    m_bLights = {};
  }

  if(m_bLights.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(m_alloc->createBuffer(m_bLights, requiredBytes,
                                     VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_DBG_NAME(m_bLights.buffer);
    m_memoryTracker.track(kMemCategorySceneData, m_bLights.allocation);
    m_sceneDescDirty = true;
  }

  const bool bufferRecreated = (m_bLights.bufferSize != prevSize) || (prevSize == 0);
  if(bufferRecreated || dirtyIndices.empty())
  {
    staging.appendBuffer(m_bLights, 0, std::span(shaderLights));
  }
  else
  {
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
// Upload render primitive info (and morph/skin vertex data) to GPU. Used for morph targets and skinning.
void nvvkgltf::SceneVk::uploadPrimitives(VkCommandBuffer cmd, nvvk::StagingUploader& staging, nvvkgltf::Scene& scn)
{
  scn.animation().computeMorphTargets();
  scn.animation().computeSkinning();

  // ** Morph **
  for(size_t i = 0; i < scn.animation().getMorphPrimitives().size(); i++)
  {
    const auto& morph = scn.animation().getMorphResult(i);
    if(morph.renderPrimID < 0)
      continue;

    staging.cmdUploadAppended(cmd);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VertexBuffers& vertexBuffers = m_vertexBuffers[morph.renderPrimID];
    staging.appendBuffer(vertexBuffers.position, 0, std::span(morph.blendedPositions));

    if(morph.blendedNormals.empty() != (vertexBuffers.normal.buffer == VK_NULL_HANDLE))
      LOGW("Morph prim %d: normal buffer/data mismatch\n", morph.renderPrimID);
    if(morph.blendedTangents.empty() != (vertexBuffers.tangent.buffer == VK_NULL_HANDLE))
      LOGW("Morph prim %d: tangent buffer/data mismatch\n", morph.renderPrimID);

    if(!morph.blendedNormals.empty() && vertexBuffers.normal.buffer != VK_NULL_HANDLE)
      staging.appendBuffer(vertexBuffers.normal, 0, std::span(morph.blendedNormals));

    if(!morph.blendedTangents.empty() && vertexBuffers.tangent.buffer != VK_NULL_HANDLE)
      staging.appendBuffer(vertexBuffers.tangent, 0, std::span(morph.blendedTangents));
  }

  // ** Skin **
  for(size_t i = 0; i < scn.animation().getSkinTasks().size(); i++)
  {
    const auto& task    = scn.animation().getSkinTasks()[i];
    const auto& skinned = scn.animation().getSkinningResult(i);

    staging.cmdUploadAppended(cmd);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VertexBuffers& vertexBuffers = m_vertexBuffers[task.renderPrimID];
    staging.appendBuffer(vertexBuffers.position, 0, std::span(skinned.positions));

    if(skinned.normals.empty() != (vertexBuffers.normal.buffer == VK_NULL_HANDLE))
      LOGW("Skin prim %d: normal buffer/data mismatch\n", task.renderPrimID);
    if(skinned.tangents.empty() != (vertexBuffers.tangent.buffer == VK_NULL_HANDLE))
      LOGW("Skin prim %d: tangent buffer/data mismatch\n", task.renderPrimID);

    if(!skinned.normals.empty() && vertexBuffers.normal.buffer != VK_NULL_HANDLE)
      staging.appendBuffer(vertexBuffers.normal, 0, std::span(skinned.normals));

    if(!skinned.tangents.empty() && vertexBuffers.tangent.buffer != VK_NULL_HANDLE)
      staging.appendBuffer(vertexBuffers.tangent, 0, std::span(skinned.tangents));
  }
}

//--------------------------------------------------------------------------------------------------
// Create or update a vertex attribute buffer for a primitive. Returns true if a new buffer was created.
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
    const int accessorIdx = findResult->second;
    if(accessorIdx < 0 || accessorIdx >= static_cast<int>(model.accessors.size()))
      return false;
    const tinygltf::Accessor& accessor = model.accessors[accessorIdx];
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
// Common buffer usage flags for scene buffers (storage, device address, transfer). Adds ray tracing bit if enabled.
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
// Create vertex/index buffers and render-primitive info for all primitives. One vertex buffer set
// per primitive; primitive buffer references vertex/index buffers and material id.
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

//--------------------------------------------------------------------------------------------------
// Re-upload all vertex buffers (e.g. after tangent recompute). Updates primitives that changed.
void nvvkgltf::SceneVk::uploadVertexBuffers(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scene)
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

//--------------------------------------------------------------------------------------------------
// Create GPU images for all textures referenced by the scene. Loads from disk or embedded data.
void nvvkgltf::SceneVk::createTextureImages(VkCommandBuffer                           cmd,
                                            nvvk::StagingUploader&                    staging,
                                            nvvkgltf::Scene&                          scn,
                                            const std::vector<std::filesystem::path>& imageSearchPaths)
{
  nvutils::ScopedTimer   st(std::string(__FUNCTION__) + "\n");
  const tinygltf::Model& model = scn.getModel();

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

    const auto&           gltfImage = model.images[i];
    std::filesystem::path diskPath;
    if(!gltfImage.uri.empty() && gltfImage.bufferView < 0 && (gltfImage.uri.size() < 5 || gltfImage.uri.compare(0, 5, "data:") != 0))
    {
      std::string uriDecoded;
      tinygltf::URIDecode(gltfImage.uri, &uriDecoded, nullptr);
      diskPath = nvutils::findFile(nvutils::pathFromUtf8(uriDecoded), imageSearchPaths, false);
    }
    ImageLoadItem item{.imageId = i};
    item.diskPath       = diskPath;
    item.numBytes       = getImageByteSize(model, gltfImage, item.diskPath);
    m_images[i].imgName = getImageName(gltfImage, i);

    imageLoadItems.push_back(std::move(item));
    LOGI("%s(%" PRIu64 ") %s \n", indent.c_str(), i, m_images[i].imgName.c_str());
  }

  std::sort(imageLoadItems.begin(), imageLoadItems.end(),
            [](const ImageLoadItem& a, const ImageLoadItem& b) { return a.numBytes > b.numBytes; });

  std::atomic<uint32_t> failedImageCount{0};
  nvutils::parallel_batches<1>(  // Not batching
      imageLoadItems.size(), [&](uint64_t i) {
        const ImageLoadItem& item = imageLoadItems[i];
        if(!loadImage(item.diskPath, model, item.imageId))
        {
          ++failedImageCount;
        }
      });
  if(failedImageCount > 0)
  {
    LOGW("%" PRIu32 " of %zu image(s) failed to load.\n", failedImageCount.load(), imageLoadItems.size());
  }

  syncTinyGltfImageDimensionsFromLoadedImages(scn.getModel(), m_images);

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

//--------------------------------------------------------------------------------------------------
// Identify images that must use sRGB format (e.g. base color). Stored in m_sRgbImages for createImage.
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
      {
        const auto& texInfo = ext->second.Get(name);
        if(texInfo.Has("index"))
          addImage(texInfo.Get("index").GetNumberAsInt());
      }
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
// Load glTF image by ID from disk or embedded buffer; populate m_images[imageID] (size, format, mipData).
bool nvvkgltf::SceneVk::loadImage(const std::filesystem::path& diskPath, const tinygltf::Model& model, uint64_t imageID)
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
      return false;
    }

    const tinygltf::BufferView& bufferView  = model.bufferViews[bufferViewIndex];
    const int                   bufferIndex = bufferView.buffer;
    if(bufferIndex < 0 || static_cast<size_t>(bufferIndex) >= model.buffers.size())
    {
      LOGW("The buffer index (%i) from the buffer view (%i) for image %" PRIu64 " was out of range.\n", bufferIndex,
           bufferViewIndex, imageID);
      return false;
    }

    const tinygltf::Buffer& buffer = model.buffers[bufferIndex];
    // Make sure the data's in-bounds. TinyGLTF doesn't seem to verify this (!)
    const size_t byteOffset = bufferView.byteOffset;
    const size_t byteLength = bufferView.byteLength;
    if(byteOffset >= buffer.data.size() || byteLength > buffer.data.size() - byteOffset)
    {
      LOGW("The buffer offset (%zu) and length (%zu) were out-of-range for buffer %i, which has length %zu, for image %" PRIu64 ".\n",
           byteOffset, byteLength, bufferIndex, buffer.data.size(), imageID);
      return false;
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
      return false;
    }

    loadImageFromMemory(imageID, fileMapping.data(), fileMapping.size());
  }
  else
  {
    LOGW("Image %" PRIu64 " has no data source (no bufferView, no stored data, and no URI).\n", imageID);
    return false;
  }

  return true;
}

//--------------------------------------------------------------------------------------------------
// Load image from memory (e.g. embedded glTF buffer). Fills m_images[imageID] size, format, mipData.
// Custom callback is tried first; then ImageLoader decodes DDS, KTX, or stb_image formats.
void nvvkgltf::SceneVk::loadImageFromMemory(uint64_t imageID, const void* data, size_t byteLength)
{
  SceneImage& image = m_images[imageID];
  image.srgb        = m_sRgbImages.find(static_cast<int>(imageID)) != m_sRgbImages.end();

  if(m_imageLoadCallback && m_imageLoadCallback(image, data, byteLength))
    return;

  LoadedImageData loaded;
  if(!nvvkgltf::loadFromMemory(loaded, data, byteLength, image.srgb, imageID))
    return;

  image.format           = loaded.format;
  image.size             = loaded.size;
  image.mipData          = std::move(loaded.mipData);
  image.componentMapping = loaded.componentMapping;
}

//--------------------------------------------------------------------------------------------------
// Create a Vulkan image from a populated SceneImage (size, format, mipData). Returns false if invalid.
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

//========== Destroy ==========

//--------------------------------------------------------------------------------------------------
// Release only geometry resources (vertex/index buffers, render primitives, scene descriptor).
// Preserves textures and materials. Use for geometry-only rebuilds (e.g. tangent generation).
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
    m_bRenderPrim = {};  // Reset to ensure clean state
  }

  if(m_bSceneDesc.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bSceneDesc.allocation);
    m_alloc->destroyBuffer(m_bSceneDesc);
    m_bSceneDesc = {};  // Reset to ensure clean state
  }
}

//--------------------------------------------------------------------------------------------------
// Release all Vulkan resources (geometry, scene data buffers, textures, images). Idempotent.
void nvvkgltf::SceneVk::destroy()
{
  destroyGeometry();

  // Destroy remaining scene data buffers
  if(m_bMaterial.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bMaterial.allocation);
    m_alloc->destroyBuffer(m_bMaterial);
    m_bMaterial = {};  // Reset to ensure clean state
  }
  if(m_bTextureInfos.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bTextureInfos.allocation);
    m_alloc->destroyBuffer(m_bTextureInfos);
    m_bTextureInfos = {};  // Reset to ensure clean state
  }
  if(m_bLights.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bLights.allocation);
    m_alloc->destroyBuffer(m_bLights);
    m_bLights = {};  // Reset to ensure clean state
  }
  if(m_bRenderNode.buffer != VK_NULL_HANDLE)
  {
    m_memoryTracker.untrack(kMemCategorySceneData, m_bRenderNode.allocation);
    m_alloc->destroyBuffer(m_bRenderNode);
    m_bRenderNode = {};  // Reset to ensure clean state
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

  m_materialCache.clear();
}

//========== Debug Validation (debug builds only) ==========

#ifndef NDEBUG

void nvvkgltf::SceneVk::debugUpdateShadowCopy(const nvvkgltf::Scene& scn)
{
  const auto& rn = scn.getRenderNodes();
  m_debugLastUploadedRN.resize(rn.size());
  for(size_t i = 0; i < rn.size(); i++)
  {
    m_debugLastUploadedRN[i].materialID   = rn[i].materialID;
    m_debugLastUploadedRN[i].renderPrimID = rn[i].renderPrimID;
  }
}

std::vector<nvvkgltf::SceneVk::GpuSyncMismatch> nvvkgltf::SceneVk::validateGpuSync(const nvvkgltf::Scene& scene,
                                                                                   const std::vector<VkAccelerationStructureInstanceKHR>& tlasInstances) const
{
  std::vector<GpuSyncMismatch> mismatches;
  const auto&                  renderNodes = scene.getRenderNodes();
  char                         buf[256];

  size_t gpuRNCount = m_bRenderNode.bufferSize / sizeof(shaderio::GltfRenderNode);
  if(gpuRNCount != renderNodes.size())
  {
    snprintf(buf, sizeof(buf), "RenderNode count: GPU buffer holds %zu but CPU has %zu", gpuRNCount, renderNodes.size());
    mismatches.push_back({buf});
  }

  if(m_debugLastUploadedRN.size() != renderNodes.size())
  {
    snprintf(buf, sizeof(buf), "RenderNode shadow: uploaded %zu but CPU has %zu", m_debugLastUploadedRN.size(),
             renderNodes.size());
    mismatches.push_back({buf});
  }
  else
  {
    for(size_t i = 0; i < renderNodes.size(); i++)
    {
      if(m_debugLastUploadedRN[i].materialID != renderNodes[i].materialID
         || m_debugLastUploadedRN[i].renderPrimID != renderNodes[i].renderPrimID)
      {
        snprintf(buf, sizeof(buf), "RenderNode[%zu]: GPU mat=%d prim=%d, CPU mat=%d prim=%d", i,
                 m_debugLastUploadedRN[i].materialID, m_debugLastUploadedRN[i].renderPrimID, renderNodes[i].materialID,
                 renderNodes[i].renderPrimID);
        mismatches.push_back({buf});
      }
    }
  }

  if(tlasInstances.size() != renderNodes.size())
  {
    snprintf(buf, sizeof(buf), "TLAS instance count: %zu but CPU has %zu render nodes", tlasInstances.size(),
             renderNodes.size());
    mismatches.push_back({buf});
  }
  else
  {
    for(size_t i = 0; i < renderNodes.size(); i++)
    {
      if(tlasInstances[i].instanceCustomIndex != static_cast<uint32_t>(renderNodes[i].renderPrimID))
      {
        snprintf(buf, sizeof(buf), "TLAS[%zu].instanceCustomIndex=%u but CPU renderPrimID=%d", i,
                 tlasInstances[i].instanceCustomIndex, renderNodes[i].renderPrimID);
        mismatches.push_back({buf});
      }
    }
  }

  size_t gpuMatCount = m_bMaterial.bufferSize / sizeof(shaderio::GltfShadeMaterial);
  if(gpuMatCount < scene.getModel().materials.size())
  {
    snprintf(buf, sizeof(buf), "Material buffer: GPU holds %zu but CPU has %zu", gpuMatCount, scene.getModel().materials.size());
    mismatches.push_back({buf});
  }

  return mismatches;
}

#endif  // NDEBUG
