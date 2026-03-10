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

#pragma once

#include <filesystem>
#include <functional>
#include <set>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>
#include <nvvk/resource_allocator.hpp>

#include "gltf_scene.hpp"
#include "gltf_material_cache.hpp"
#include "nvvk/sampler_pool.hpp"
#include "nvvk/staging.hpp"
#include "gpu_memory_tracker.hpp"
#include "nvshaders/gltf_scene_io.h.slang"


/*-------------------------------------------------------------------------------------------------
# class nvvkgltf::SceneVk

>  This class is responsible for the Vulkan version of the scene. 

It is using `nvvkgltf::Scene` to create the Vulkan buffers and images.

-------------------------------------------------------------------------------------------------*/

namespace nvvkgltf {

// Create the Vulkan version of the Scene
// Allocate the buffers, etc.
class SceneVk
{
public:
  // Those are potential buffers that can be created for vertices
  struct VertexBuffers
  {
    nvvk::Buffer position;
    nvvk::Buffer normal;
    nvvk::Buffer tangent;
    nvvk::Buffer texCoord0;
    nvvk::Buffer texCoord1;
    nvvk::Buffer color;
  };

  SceneVk() = default;
  virtual ~SceneVk() { assert(!m_alloc); }  // Missing deinit call

  void init(nvvk::ResourceAllocator* alloc, nvvk::SamplerPool* samplerPool);
  void deinit();

  // Optional: set the graphics queue used for scene buffers. When set, the fallback buffer
  // reallocation path waits on this queue instead of the entire device.
  void setGraphicsQueue(VkQueue queue) { m_graphicsQueue = queue; }

  // Optional: set a callback for deferred GPU resource destruction. When set, buffer
  // reallocation schedules destruction via this callback instead of stalling the GPU.
  // The callback receives a void() function to execute later (after the GPU is done).
  // Typical wiring: setDeferredFree([app](auto&& fn){ app->submitResourceFree(std::move(fn)); });
  using DeferredFreeFunc = std::function<void(std::function<void()>&&)>;
  void setDeferredFree(DeferredFreeFunc func) { m_deferredFree = std::move(func); }

  virtual void create(VkCommandBuffer        cmd,
                      nvvk::StagingUploader& staging,
                      nvvkgltf::Scene&       scn,
                      bool                   generateMipmaps  = true,
                      bool                   enableRayTracing = true);

  // --- Tier 1: sync API (reads + clears Scene dirty flags) ---
  enum SyncFlags : uint32_t
  {
    eSyncNone        = 0,
    eSyncRenderNodes = 1 << 0,
    eSyncMaterials   = 1 << 1,
    eSyncLights      = 1 << 2,
  };
  [[nodiscard]] uint32_t syncFromScene(nvvk::StagingUploader& staging, nvvkgltf::Scene& scn, uint32_t mask = ~0u);

  // --- Tier 2: explicit upload (const Scene, caller owns dirty tracking) ---
  void uploadRenderNodes(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn, const std::unordered_set<int>& dirtyIndices = {});
  void uploadMaterials(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn, const std::unordered_set<int>& dirtyIndices = {});
  void uploadLights(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn, const std::unordered_set<int>& dirtyIndices = {});
  void uploadPrimitives(VkCommandBuffer cmd, nvvk::StagingUploader& staging, nvvkgltf::Scene& scn);
  void uploadVertexBuffers(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);

  // Call once per sync cycle after all buffer uploads. Updates scene descriptor if any buffer address changed.
  // Returns true if an update was performed (staging was appended).
  [[nodiscard]] bool flushSceneDescIfDirty(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);

  virtual void destroy();

  // Geometry-only recreation (preserves textures) - useful after tangent generation or mesh optimization
  void destroyGeometry();
  void createGeometry(VkCommandBuffer cmd, nvvk::StagingUploader& staging, nvvkgltf::Scene& scn);

  // Getters
  const nvvk::Buffer&               material() const { return m_bMaterial; }
  const nvvk::Buffer&               primitiveBuffer() const { return m_bRenderPrim; }
  const nvvk::Buffer&               renderNodeBuffer() const { return m_bRenderNode; }
  const nvvk::Buffer&               sceneDesc() const { return m_bSceneDesc; }
  const std::vector<VertexBuffers>& vertexBuffers() const { return m_vertexBuffers; }
  const std::vector<nvvk::Buffer>&  indices() const { return m_bIndices; }
  const std::vector<nvvk::Image>&   textures() const { return m_textures; }
  [[nodiscard]] uint32_t            textureCount() const { return static_cast<uint32_t>(m_textures.size()); }
  const GpuMemoryTracker&           getMemoryTracker() const { return m_memoryTracker; }
  GpuMemoryTracker&                 getMemoryTracker() { return m_memoryTracker; }

  // An image to be loaded and created.
  struct SceneImage
  {
    // GPU image.
    nvvk::Image imageTexture{};

    // Loaded information.
    std::string imgName{};
    bool        srgb{false};
    // Custom image loaders must set these:
    VkFormat                       format{VK_FORMAT_UNDEFINED};
    VkExtent2D                     size{0, 0};
    std::vector<std::vector<char>> mipData{};
    // And optionally set the component swizzle for image view (e.g. grayscale expansion):
    VkComponentMapping componentMapping{};
  };

  // A custom callback for loading images that will be called before
  // SceneVK's built-in image loaders.
  // This must fill SceneImage::{size, format, mipData}, optionally fill
  // `SceneImage::componentMapping`, and return whether the image was
  // successfully loaded. The rest can be left unchanged.
  // For an example, see `webPLoadCallback()` in vk_gltf_renderer.
  using ImageLoadCallback = std::function<bool(SceneImage& outImage, const void* data, size_t byteLength)>;
  void setImageLoadCallback(ImageLoadCallback callback) { m_imageLoadCallback = callback; }

protected:
  VkBufferUsageFlags2 getBufferUsageFlags() const;
  virtual void createVertexBuffers(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);
  template <typename T>
  bool updateAttributeBuffer(const std::string&         attributeName,
                             const tinygltf::Model&     model,
                             const tinygltf::Primitive& primitive,
                             nvvk::ResourceAllocator*   alloc,
                             nvvk::StagingUploader*     staging,
                             nvvk::Buffer&              attributeBuffer);
  // imageSearchPaths: directories to search for image files (base first, then imports). Empty or missing files yield default image.
  virtual void createTextureImages(VkCommandBuffer                           cmd,
                                   nvvk::StagingUploader&                    staging,
                                   const nvvkgltf::Scene&                    scn,
                                   const std::vector<std::filesystem::path>& imageSearchPaths);

  void findSrgbImages(const tinygltf::Model& model);

  // Rebuild scene descriptor buffer (buffer addresses + numLights). Called internally when buffers change.
  void updateSceneDescBuffer(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);

  // Ensure render node buffer matches required size; recreates if needed. Marks scene descriptor dirty on buffer creation.
  void ensureRenderNodeBuffer(nvvk::StagingUploader& staging, size_t renderNodeCount);

  virtual bool loadImage(const std::filesystem::path& basedir, const tinygltf::Model& model, uint64_t imageID);
  virtual void loadImageFromMemory(uint64_t imageID, const void* data, size_t byteLength);
  virtual bool createImage(const VkCommandBuffer& cmd, nvvk::StagingUploader& staging, SceneImage& image);

  //--
  VkDevice         m_device{VK_NULL_HANDLE};
  VkPhysicalDevice m_physicalDevice{VK_NULL_HANDLE};
  VkQueue m_graphicsQueue{VK_NULL_HANDLE};  // Optional; when set, buffer realloc uses queue wait instead of device wait

  nvvk::ResourceAllocator* m_alloc       = nullptr;
  nvvk::SamplerPool*       m_samplerPool = nullptr;

  nvvk::Buffer               m_bMaterial;
  nvvk::Buffer               m_bTextureInfos;
  nvvk::Buffer               m_bLights;
  nvvk::Buffer               m_bRenderPrim;
  nvvk::Buffer               m_bRenderNode;
  nvvk::Buffer               m_bSceneDesc;
  std::vector<nvvk::Buffer>  m_bIndices;
  std::vector<VertexBuffers> m_vertexBuffers;
  std::vector<SceneImage>    m_images;
  std::vector<nvvk::Image>   m_textures;  // Vector of all textures of the scene

  // All images the glTF specification implies should be forced to use the sRGB
  // transfer function. This is used to fix cases where an image is loaded as
  // e.g. VK_FORMAT_R8G8B8A8_UNORM, but should be read as VK_FORMAT_R8G8B8A8_SRGB.
  std::set<int>     m_sRgbImages;
  ImageLoadCallback m_imageLoadCallback = {};

  // Cached material data for updates.
  MaterialCache m_materialCache;

  bool m_sceneDescDirty    = false;  // Set when any scene buffer address changes; cleared by flushSceneDescIfDirty.
  bool m_generateMipmaps   = {};
  bool m_rayTracingEnabled = {};

  DeferredFreeFunc m_deferredFree;                            // Optional: schedules deferred GPU resource destruction
  void             destroyBufferDeferred(nvvk::Buffer& buf);  // Destroy via m_deferredFree or fallback to queue wait

  GpuMemoryTracker m_memoryTracker;  // GPU memory tracking

#ifndef NDEBUG
public:
  struct GpuSyncMismatch
  {
    std::string description;
  };
  std::vector<GpuSyncMismatch> validateGpuSync(const nvvkgltf::Scene& scene,
                                               const std::vector<VkAccelerationStructureInstanceKHR>& tlasInstances) const;

  struct DebugRenderNodeEntry
  {
    int materialID;
    int renderPrimID;
  };
  std::vector<DebugRenderNodeEntry> m_debugLastUploadedRN;
  void                              debugUpdateShadowCopy(const nvvkgltf::Scene& scn);
#endif
};

}  // namespace nvvkgltf
