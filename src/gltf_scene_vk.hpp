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

#include <glm/glm.hpp>
#include <nvvk/resource_allocator.hpp>

#include "gltf_scene.hpp"
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

// Reusable workspace for CPU skinning operations - avoids per-frame allocations
// Buffers grow as needed and are released when the scene is destroyed
struct SkinningWorkspace
{
  // Per-joint normal matrices: inverse-transpose of upper 3x3 (reused across primitives)
  std::vector<glm::mat3> normalMatrices;

  // Output buffers (reused across frames)
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec4> tangents;

  // Ensure buffers are large enough, only grows (never shrinks during scene lifetime)
  void reserve(size_t vertexCount, size_t jointCount, bool needNormals, bool needTangents);

  // Release all memory
  void clear();
};

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

  virtual void create(VkCommandBuffer        cmd,
                      nvvk::StagingUploader& staging,
                      const nvvkgltf::Scene& scn,
                      bool                   generateMipmaps  = true,
                      bool                   enableRayTracing = true);

  void update(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);

  // Update render node transforms and material/primitive IDs.
  // @param dirtyIndices  renderNode indices that changed; empty = update all (full refresh)
  void updateRenderNodesBuffer(nvvk::StagingUploader&         staging,
                               const nvvkgltf::Scene&         scn,
                               const std::unordered_set<int>& dirtyIndices = {});

  void updateRenderPrimitivesBuffer(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);

  // Update light positions and properties.
  // @param dirtyIndices  glTF light indices (model.extensions["KHR_lights_punctual"]) that changed; empty = update all
  void updateRenderLightsBuffer(nvvk::StagingUploader&         staging,
                                const nvvkgltf::Scene&         scn,
                                const std::unordered_set<int>& dirtyIndices = {});

  // Update material properties and texture bindings.
  // @param dirtyIndices  glTF material indices (model.materials[]) that changed; empty = update all
  void updateMaterialBuffer(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn, const std::unordered_set<int>& dirtyIndices = {});

  void         updateVertexBuffers(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scene);
  virtual void destroy();

  // Geometry-only recreation (preserves textures) - useful after tangent generation or mesh optimization
  void destroyGeometry();
  void createGeometry(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);

  // Getters
  const nvvk::Buffer&               material() const { return m_bMaterial; }
  const nvvk::Buffer&               primInfo() const { return m_bRenderPrim; }
  const nvvk::Buffer&               instances() const { return m_bRenderNode; }
  const nvvk::Buffer&               sceneDesc() const { return m_bSceneDesc; }
  const std::vector<VertexBuffers>& vertexBuffers() const { return m_vertexBuffers; }
  const std::vector<nvvk::Buffer>&  indices() const { return m_bIndices; }
  const std::vector<nvvk::Image>&   textures() const { return m_textures; }
  uint32_t                          nbTextures() const { return static_cast<uint32_t>(m_textures.size()); }
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
  bool         updateAttributeBuffer(const std::string&         attributeName,
                                     const tinygltf::Model&     model,
                                     const tinygltf::Primitive& primitive,
                                     nvvk::ResourceAllocator*   alloc,
                                     nvvk::StagingUploader*     staging,
                                     nvvk::Buffer&              attributeBuffer);
  virtual void createTextureImages(VkCommandBuffer              cmd,
                                   nvvk::StagingUploader&       staging,
                                   const tinygltf::Model&       model,
                                   const std::filesystem::path& basedir);

  void findSrgbImages(const tinygltf::Model& model);

  // Rebuild scene descriptor buffer (used when buffer addresses change)
  void updateSceneDescBuffer(nvvk::StagingUploader& staging, const nvvkgltf::Scene& scn);

  virtual void loadImage(const std::filesystem::path& basedir, const tinygltf::Model& model, uint64_t imageID);
  virtual void loadImageFromMemory(uint64_t imageID, const void* data, size_t byteLength);
  virtual bool createImage(const VkCommandBuffer& cmd, nvvk::StagingUploader& staging, SceneImage& image);

  //--
  VkDevice         m_device{VK_NULL_HANDLE};
  VkPhysicalDevice m_physicalDevice{VK_NULL_HANDLE};

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
  std::vector<shaderio::GltfShadeMaterial> m_cachedShadeMaterials;
  std::vector<shaderio::GltfTextureInfo>   m_cachedTextureInfos;

  bool m_generateMipmaps   = {};
  bool m_rayTracingEnabled = {};

  GpuMemoryTracker  m_memoryTracker;      // GPU memory tracking
  SkinningWorkspace m_skinningWorkspace;  // Reusable workspace for CPU skinning (avoids per-frame allocations)
};

}  // namespace nvvkgltf
