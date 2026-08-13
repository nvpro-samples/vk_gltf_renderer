/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*
 * ThumbnailCache - bounded, virtualization-friendly ImGui thumbnails for scene textures/images.
 *
 * Turns a VkImageView (a scene texture/image already resident on the GPU) into an ImGui-displayable
 * ImTextureID (a descriptor set allocated from the ImGui-Vulkan backend pool). Designed for scenes
 * with thousands of images: callers request thumbnails only for on-screen rows (drive the caller with
 * ImGuiListClipper), and the cache keeps at most `capacity` live descriptor sets, evicting the
 * least-recently-used ones.
 *
 * Descriptor-set lifetime is frame-safe: an evicted descriptor is not freed immediately (the GPU may
 * still reference it from an in-flight frame) but parked in a ring of trash bins and released once
 * `framesInFlight` frames have elapsed. Call beginFrame() once per frame before acquiring.
 *
 * clear() releases everything immediately and MUST be called while the GPU is idle (e.g. right before
 * the image views backing these descriptors are destroyed on a full scene rebuild).
 */

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan_core.h>
#include <imgui.h>

#include <nvapp/imgui_texture.hpp>

class ThumbnailCache
{
public:
  ThumbnailCache() = default;
  ~ThumbnailCache();

  ThumbnailCache(const ThumbnailCache&)            = delete;
  ThumbnailCache& operator=(const ThumbnailCache&) = delete;

  // Maximum number of live descriptor sets. Keep well below the ImGui descriptor pool size
  // (nvapp::ApplicationCreateInfo::texturePoolSize). Default is conservative.
  void setCapacity(size_t capacity) { m_capacity = capacity; }

  // Advance the deferred-free ring and release descriptors parked long enough to be GPU-safe.
  // framesInFlight is the app's frame-cycle size (nvapp::Application::getFrameCycleSize()).
  void beginFrame(uint32_t framesInFlight);

  // Displayable ImGui id for `view` (assumed SHADER_READ_ONLY_OPTIMAL), created lazily and marked used
  // this frame. Returns 0 when `view` is null. The returned id is valid for the current frame's draw.
  ImTextureID acquire(VkImageView view);

  // Release all descriptor sets immediately. MUST be called while the GPU is idle.
  void clear();

  // Release all descriptor sets via the deferred-free ring instead of immediately, so it is safe to call
  // WITHOUT a GPU idle. Used when a scene image view is destroyed through deferred-free (e.g. undoing an
  // imported texture): parking the descriptors keyed on that view prevents a later handle-reuse hit from
  // returning a stale descriptor. Entries are re-acquired lazily afterwards.
  void clearDeferred();

private:
  struct Entry
  {
    nvapp::ImTexture tex;  // owns the ImGui descriptor set for one image view
    uint64_t         lastUse{0};
  };

  void evictIfNeeded();

  std::unordered_map<VkImageView, std::unique_ptr<Entry>> m_entries;
  std::vector<std::vector<std::unique_ptr<Entry>>>        m_trash;  // ring of pending-free bins
  uint32_t                                                m_trashHead{0};
  uint64_t                                                m_clock{0};
  size_t                                                  m_capacity{512};
};
