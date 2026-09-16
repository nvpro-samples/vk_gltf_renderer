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

#include <array>
#include <filesystem>
#include <functional>
#include <set>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>
#include <nvvk/resource_allocator.hpp>

#include "gltf_scene.hpp"
#include "gltf_scene_omm.hpp"
#include "gltf_material_cache.hpp"
#include "nvvk/sampler_pool.hpp"
#include <nvvk/uploader_interface.hpp>
#include "gpu_memory_tracker.hpp"
#include "shaders/gltf_scene_io.h.slang"


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

  virtual void create(VkCommandBuffer cmd,

                      nvvk::CmdUploaderInterface& staging,
                      nvvkgltf::Scene&            scn,

                      bool generateMipmaps = true,

                      bool enableRayTracing = true);


  // --- Tier 1: sync API (reads + clears Scene dirty flags) ---
  enum SyncFlags : uint32_t
  {
    eSyncNone        = 0,
    eSyncRenderNodes = 1 << 0,
    eSyncMaterials   = 1 << 1,
    eSyncLights      = 1 << 2,
  };
  [[nodiscard]] uint32_t syncFromScene(nvvk::CmdUploaderInterface& staging, nvvkgltf::Scene& scn, uint32_t mask = ~0u);

  // --- Tier 2: explicit upload (const Scene, caller owns dirty tracking) ---
  void uploadRenderNodes(nvvk::CmdUploaderInterface& staging,

                         const nvvkgltf::Scene& scn,

                         const std::unordered_set<int>& dirtyIndices = {});

  void uploadMaterials(nvvk::CmdUploaderInterface& staging, const nvvkgltf::Scene& scn, const std::unordered_set<int>& dirtyIndices = {});
  void uploadLights(nvvk::CmdUploaderInterface& staging, const nvvkgltf::Scene& scn, const std::unordered_set<int>& dirtyIndices = {});
  void uploadEmissiveTriangles(nvvk::CmdUploaderInterface& staging, const nvvkgltf::Scene& scn);
  void uploadPrimitives(VkCommandBuffer cmd, nvvk::CmdUploaderInterface& staging, nvvkgltf::Scene& scn);
  void uploadVertexBuffers(nvvk::CmdUploaderInterface& staging, const nvvkgltf::Scene& scn);

  // Call once per sync cycle after all buffer uploads. Updates scene descriptor if any buffer address changed.
  // Returns true if an update was performed (staging was appended).
  [[nodiscard]] bool flushSceneDescIfDirty(nvvk::CmdUploaderInterface& staging, const nvvkgltf::Scene& scn);

  virtual void destroy();

  // Geometry-only recreation (preserves textures) - useful after tangent generation or mesh optimization
  void destroyGeometry();
  void createGeometry(VkCommandBuffer cmd, nvvk::CmdUploaderInterface& staging, const nvvkgltf::Scene& scn);

  // Incremental texture/image reconcile for a TAIL-ONLY change (DirtyFlags::texturesTailChanged): brings
  // GPU residency in line with the model when the only edits were appends to, or removals from, the end of
  // model.images / model.textures -- every lower index is left untouched. Growth loads + creates the new
  // tail images and appends their texture views; shrink deferred-frees the removed tail images. This is the
  // fast path behind importing (and undoing/redoing) a texture, avoiding a full create()/destroy() cycle
  // that would re-read every image from disk. The caller writes only the new descriptor slots afterwards.
  void syncTextureTail(VkCommandBuffer cmd, nvvk::CmdUploaderInterface& staging, nvvkgltf::Scene& scn);

  // In-place update of one sampler's wrap/filter (DirtyFlags::samplers): recreates only the VkSampler
  // at model.samplers[samplerIndex]'s slot via the sampler pool. Images and texture views are untouched;
  // the caller only needs to rewrite that one eSamplers descriptor slot afterwards.
  void updateSampler(const tinygltf::Model& model, int samplerIndex);

  // Rebuild for a merge/reference append: geometry, render nodes, materials and lights are re-derived
  // from the (grown) model, but existing GPU images/textures/samplers are kept and only the new tail
  // images are loaded (via syncTextureTail). This is create() without the destroy()/full image re-read,
  // so a merge no longer re-reads every image from disk. Requires a tail-only, non-empty-base texture set
  // (existing GPU arrays must match the pre-merge model sizes); the merge/reference paths guarantee this.
  void recreatePreservingTextures(VkCommandBuffer cmd, nvvk::CmdUploaderInterface& staging, nvvkgltf::Scene& scn);

  // Getters
  const nvvk::Buffer&               material() const { return m_bMaterial; }
  const nvvk::Buffer&               primitiveBuffer() const { return m_bRenderPrim; }
  const nvvk::Buffer&               renderNodeBuffer() const { return m_bRenderNode; }
  const nvvk::Buffer&               sceneDesc() const { return m_bSceneDesc; }
  const std::vector<VertexBuffers>& vertexBuffers() const { return m_vertexBuffers; }
  const std::vector<nvvk::Buffer>&  indices() const { return m_bIndices; }
  const SceneOmm&                   opacityMicromap() const { return m_sceneOmm; }

  // Enable building opacity micromaps (EXT_mesh_opacity_micromap). Driven from
  // VK_EXT_opacity_micromap availability. Set before create().
  void                            setOpacityMicromapEnabled(bool enabled) { m_sceneOmm.setEnabled(enabled); }
  const std::vector<nvvk::Image>& textures() const { return m_textures; }
  [[nodiscard]] uint32_t          textureCount() const { return static_cast<uint32_t>(m_textures.size()); }

  // Deduplicated glTF samplers, bound as a separate SAMPLER array (eSamplers). Slot 0 is the default
  // sampler; slots 1..N mirror model.samplers. GltfTextureInfo.samplerIndex indexes into this.
  const std::vector<VkSampler>& samplers() const { return m_samplers; }
  [[nodiscard]] uint32_t        samplerCount() const { return static_cast<uint32_t>(m_samplers.size()); }
  // Per-texture data baked into GltfTextureInfo (sampler slot, source-format flags). Fed to MaterialCache.
  const TextureSlotTable& textureSlots() const { return m_textureSlots; }

  // GPU image view for a glTF texture / image index, or VK_NULL_HANDLE when out of range or not
  // resident (e.g. an unused image that was not uploaded). Used by the UI to build thumbnails.
  [[nodiscard]] VkImageView textureView(uint32_t textureIndex) const
  {
    return textureIndex < m_textures.size() ? m_textures[textureIndex].descriptor.imageView : VK_NULL_HANDLE;
  }
  [[nodiscard]] VkImageView imageView(uint32_t imageIndex) const
  {
    return imageIndex < m_images.size() ? m_images[imageIndex].imageTexture.descriptor.imageView : VK_NULL_HANDLE;
  }
  const GpuMemoryTracker& getMemoryTracker() const { return m_memoryTracker; }
  GpuMemoryTracker&       getMemoryTracker() { return m_memoryTracker; }

  // Number of emissive area-light triangles in the NEE emitter list (0 when no material emits).
  [[nodiscard]] uint32_t numEmissiveTriangles() const { return m_numEmissiveTriangles; }

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
  virtual void createVertexBuffers(VkCommandBuffer cmd, nvvk::CmdUploaderInterface& staging, const nvvkgltf::Scene& scn);
  template <typename T>
  bool updateAttributeBuffer(const std::string& attributeName,

                             const tinygltf::Model& model,

                             const tinygltf::Primitive& primitive,

                             nvvk::ResourceAllocator* alloc,

                             nvvk::CmdUploaderInterface* staging,

                             nvvk::Buffer& attributeBuffer);

  // imageSearchPaths: directories to search for image files (base first, then imports). Empty or missing files yield default image.
  virtual void createTextureImages(VkCommandBuffer             cmd,
                                   nvvk::CmdUploaderInterface& staging,

                                   nvvkgltf::Scene&                          scn,
                                   const std::vector<std::filesystem::path>& imageSearchPaths);

  // --- Shared image/texture/sampler building blocks (used by both createTextureImages and syncTextureTail) ---

  // Directories searched for image URIs: the scene's search paths, or the glTF's own folder as a fallback.
  std::vector<std::filesystem::path> resolveImageSearchPaths(const nvvkgltf::Scene& scn) const;
  // Resolve model.images[imageId]'s on-disk path from its URI (empty for embedded / data-URI / missing).
  std::filesystem::path resolveImageDiskPath(const tinygltf::Model&                    model,
                                             const std::vector<std::filesystem::path>& imageSearchPaths,
                                             size_t                                    imageId) const;
  // Replace m_images[idx] with a 1x1 solid-color image (magenta = load failure, white = empty scene).
  void createDefaultImage(nvvk::CmdUploaderInterface& staging, uint32_t idx, const std::array<uint8_t, 4>& color);
  // Create the GPU image for m_images[imageId] (already loaded), substituting the magenta default on failure.
  void materializeImage(VkCommandBuffer cmd, nvvk::CmdUploaderInterface& staging, size_t imageId);
  // Append one texture view to m_textures, resolving model.textures[textureIndex]'s source image (default on bad source).
  void appendTextureView(const tinygltf::Model& model, size_t textureIndex);
  // Append the fallback texture view (image 0) so every texture entry references some image view.
  void pushDefaultTextureView();
  // Ensure m_samplers holds slot 0 (default) + one per model.samplers; acquires only the missing tail slots.
  void ensureSamplers(const tinygltf::Model& model);
  // Destroy a GPU image via the deferred-free callback (or a queue wait fallback). Mirrors destroyBufferDeferred.
  void destroyImageDeferred(nvvk::Image& image);
  // Release a VkSampler via the deferred-free callback (or a queue wait fallback). Mirrors destroyImageDeferred.
  void releaseSamplerDeferred(VkSampler sampler);

  // Fill m_textureSlots.samplerSlots (glTF texture index -> sampler slot) from the model. Pure model
  // data, so it must run before uploadMaterials(), which bakes the slots into GltfTextureInfo.samplerIndex.
  void buildTextureSamplerSlots(const tinygltf::Model& model);

  // Fill m_textureSlots.twoChannelSource from the decoded image formats (see TextureSlotTable). Needs
  // m_images populated, so it must run after createTextureImages() / syncTextureTail() and before
  // uploadMaterials(), which bakes the flag into GltfTextureInfo.flags.
  void buildTextureFormatFlags(const tinygltf::Model& model);

  void findSrgbImages(const tinygltf::Model& model);

  // Rebuild scene descriptor buffer (buffer addresses + numLights). Called internally when buffers change.
  void updateSceneDescBuffer(nvvk::CmdUploaderInterface& staging, const nvvkgltf::Scene& scn);

  // EXT_lights_ies (load-only, not re-run on edits): resolves and parses every profile in
  // extensions.EXT_lights_ies.lights[] (uri relative to the glTF's search paths, or an embedded
  // bufferView) via ies_profile.hpp, and uploads the flattened table to m_bIesProfiles. A profile
  // that fails to load/parse is left as a flat (all-1.0) table so referencing lights just fall
  // back to their unmodified KHR_lights_punctual distribution instead of erroring out.
  void loadIesProfiles(nvvk::CmdUploaderInterface& staging, const nvvkgltf::Scene& scn);

  // Ensure render node buffer matches required size; recreates if needed. Marks scene descriptor dirty on buffer creation.
  void ensureRenderNodeBuffer(nvvk::CmdUploaderInterface& staging, size_t renderNodeCount);

  virtual bool loadImage(const std::filesystem::path& basedir, const tinygltf::Model& model, uint64_t imageID);
  virtual void loadImageFromMemory(uint64_t imageID, const void* data, size_t byteLength);
  virtual bool createImage(const VkCommandBuffer& cmd, nvvk::CmdUploaderInterface& staging, SceneImage& image);

  //--
  VkDevice         m_device{VK_NULL_HANDLE};
  VkPhysicalDevice m_physicalDevice{VK_NULL_HANDLE};
  VkQueue m_graphicsQueue{VK_NULL_HANDLE};  // Optional; when set, buffer realloc uses queue wait instead of device wait

  nvvk::ResourceAllocator* m_alloc       = nullptr;
  nvvk::SamplerPool*       m_samplerPool = nullptr;

  nvvk::Buffer m_bMaterial;
  nvvk::Buffer m_bTextureInfos;
  nvvk::Buffer m_bLights;
  nvvk::Buffer m_bIesProfiles;        // EXT_lights_ies: flattened [profile][sample] table, see loadIesProfiles()
  nvvk::Buffer m_bEmissiveTriangles;  // Referenced emissive triangles sampled as area lights
  uint32_t     m_numEmissiveTriangles = 0;
  float        m_emissiveTotalWeight  = 0.0f;  // Sum of per-triangle selection weights (area * defensive luminance)
  float        m_emissiveMeanLum      = 0.0f;  // Area-weighted mean emitter luminance (defensive sampling)
  nvvk::Buffer m_bRenderPrim;
  nvvk::Buffer m_bRenderNode;
  nvvk::Buffer m_bSceneDesc;
  std::vector<nvvk::Buffer>  m_bIndices;
  std::vector<VertexBuffers> m_vertexBuffers;
  std::vector<SceneImage>    m_images;
  std::vector<nvvk::Image>   m_textures;  // One per glTF texture; bound as a SAMPLED_IMAGE array (image views only)
  std::vector<VkSampler> m_samplers;  // Deduplicated samplers; slot 0 = default, 1..N = model.samplers (SAMPLER array)
  TextureSlotTable       m_textureSlots;  // Per-glTF-texture sampler slot + source-format flags (fed to MaterialCache)

  // All images the glTF specification implies should be forced to use the sRGB
  // transfer function. This is used to fix cases where an image is loaded as
  // e.g. VK_FORMAT_R8G8B8A8_UNORM, but should be read as VK_FORMAT_R8G8B8A8_SRGB.
  std::set<int>     m_sRgbImages;
  ImageLoadCallback m_imageLoadCallback = {};

  // Cached material data for updates.
  MaterialCache m_materialCache;

  // Opacity micromaps (EXT_mesh_opacity_micromap), built alongside geometry and consumed by SceneRtx.
  SceneOmm m_sceneOmm;

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
