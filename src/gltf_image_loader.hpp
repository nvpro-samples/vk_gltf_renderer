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

#include <cstdint>
#include <vector>

#include <vulkan/vulkan_core.h>

namespace nvvkgltf {

// CPU-side result of decoding an image from raw bytes (disk, buffer, or embedded).
// Used by ImageLoader; SceneVk copies this into SceneImage and then creates the Vulkan image.
struct LoadedImageData
{
  VkFormat                       format{VK_FORMAT_UNDEFINED};
  VkExtent2D                     size{0, 0};
  std::vector<std::vector<char>> mipData{};
  VkComponentMapping             componentMapping{};
};

// Decodes raw image bytes into LoadedImageData. Dispatches by magic bytes:
// DDS, KTX (1/2), or falls back to stb_image (PNG, JPEG, etc.).
// srgb: when true, format may be forced to an sRGB variant where applicable.
// imageIDForLog: used only for log messages (e.g. "image 3").
// Returns true if decoding succeeded and out is filled; false on failure or unsupported format.
[[nodiscard]] bool loadFromMemory(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog = 0);

}  // namespace nvvkgltf
