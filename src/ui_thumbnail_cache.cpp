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

//
// ThumbnailCache -- bounded, virtualization-friendly ImGui thumbnails for scene textures/images.
// Turns a VkImageView already resident on the GPU into an ImGui-displayable ImTextureID (a
// descriptor set from the ImGui-Vulkan backend pool). Designed for scenes with thousands of
// images: callers request thumbnails only for on-screen rows, and the cache keeps at most
// `capacity` live descriptor sets, evicting least-recently-used ones into a frames-in-flight-deep
// trash ring so an evicted descriptor outlives any frame still referencing it.
//

#include <cassert>
#include <utility>

#include "ui_thumbnail_cache.hpp"

ThumbnailCache::~ThumbnailCache()
{
  // clear() must have run while the GPU was idle; nothing should remain to free here.
  assert(m_entries.empty() && "ThumbnailCache::clear() not called before destruction");
}

void ThumbnailCache::beginFrame(uint32_t framesInFlight)
{
  // Size the ring to framesInFlight + 1: an evicted descriptor parked this frame is released once we
  // wrap back to its bin, i.e. after framesInFlight frames - long enough for any in-flight frame
  // referencing it to have completed.
  const size_t ringSize = static_cast<size_t>(framesInFlight) + 1;
  if(m_trash.size() != ringSize)
  {
    // On resize, flush everything currently parked (safe: called between frames, and the only other
    // resize trigger is startup). Deinit each parked descriptor before dropping it.
    for(auto& bin : m_trash)
      for(auto& e : bin)
        e->tex.deinit();
    m_trash.clear();
    m_trash.resize(ringSize);  // default-construct empty bins (inner vector is move-only)
    m_trashHead = 0;
  }

  // Advance to the next bin and release whatever was parked a full ring ago.
  m_trashHead    = (m_trashHead + 1) % static_cast<uint32_t>(m_trash.size());
  auto& reusable = m_trash[m_trashHead];
  for(auto& e : reusable)
    e->tex.deinit();
  reusable.clear();
}

ImTextureID ThumbnailCache::acquire(VkImageView view)
{
  if(view == VK_NULL_HANDLE)
    return 0;

  auto it = m_entries.find(view);
  if(it == m_entries.end())
  {
    auto entry = std::make_unique<Entry>();
    entry->tex.init(view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    it = m_entries.emplace(view, std::move(entry)).first;
  }
  it->second->lastUse = ++m_clock;

  const ImTextureID id = ImTextureRef(it->second->tex).GetTexID();
  evictIfNeeded();
  return id;
}

void ThumbnailCache::evictIfNeeded()
{
  // Evict least-recently-used entries down to capacity. Entries used this frame have the highest
  // lastUse values, so as long as capacity >= the number of thumbnails shown in a single frame, only
  // entries from previous frames are evicted - and those are safe to park for deferred release.
  if(m_trash.empty())
    return;  // beginFrame() not called yet; skip eviction until the ring exists

  while(m_entries.size() > m_capacity)
  {
    auto lru = m_entries.begin();
    for(auto it = std::next(m_entries.begin()); it != m_entries.end(); ++it)
    {
      if(it->second->lastUse < lru->second->lastUse)
        lru = it;
    }
    m_trash[m_trashHead].push_back(std::move(lru->second));  // park; freed after a full ring
    m_entries.erase(lru);
  }
}

void ThumbnailCache::clear()
{
  for(auto& [view, entry] : m_entries)
    entry->tex.deinit();
  m_entries.clear();

  for(auto& bin : m_trash)
  {
    for(auto& e : bin)
      e->tex.deinit();
    bin.clear();
  }
}

void ThumbnailCache::clearDeferred()
{
  if(m_trash.empty())
  {
    clear();  // No ring yet (beginFrame never ran): nothing was drawn, so immediate release is safe.
    return;
  }
  // Park every live entry into the current bin; it is released once a full ring has elapsed (see
  // beginFrame), i.e. after any in-flight frame referencing it has completed. m_entries is emptied so
  // acquire() rebuilds fresh descriptors, so a view handle reused after its image is freed cannot hit a
  // stale descriptor.
  auto& bin = m_trash[m_trashHead];
  for(auto& [view, entry] : m_entries)
    bin.push_back(std::move(entry));
  m_entries.clear();
}
