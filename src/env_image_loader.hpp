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

//
// Decoding a lat-long environment image to float RGBA.
//
// This is the one place that decides which file formats can be an environment, which is a sample
// concern rather than a library one. `nvvk::HdrIbl` offers a path overload that decodes Radiance
// `.hdr` itself, and this renderer deliberately does not use it: OpenEXR's decoder is a compiled
// library, so putting it behind that overload would drag basis_universal into every consumer of
// nvvk. Decoding here instead keeps nvvk's dependencies untouched and -- the reason that matters
// day to day -- keeps both formats on one code path, so there is a single place to get the
// validity checks, the warnings and the upload right.
//
// See gltf_image_loader.{cpp,hpp} for the sibling that does this for *scene* textures; that one
// handles the glTF-facing formats (DDS, KTX, WebP, stb) and has nothing to say about lat-longs.
//

#include <filesystem>
#include <vector>

#include <vulkan/vulkan_core.h>

// A decoded lat-long image, ready for nvvk::HdrIbl::loadEnvironment's pixels overload.
struct EnvImage
{
  // Tightly packed RGBA32F, `size.width * size.height * 4` floats, row-major from the top-left.
  // Note that HdrIbl *writes* to this while building the alias table (it stores each texel's
  // sampling PDF in alpha), so treat it as consumed once handed over.
  std::vector<float> pixels;
  VkExtent2D         size{0, 0};

  bool valid() const { return !pixels.empty() && size.width > 0 && size.height > 0; }
};

// Decode `file` as a lat-long environment. Supports Radiance `.hdr` and, where basis_universal's
// tinyexr is available, OpenEXR `.exr`.
//
// Dispatch is on content rather than on the file extension: both formats have a cheap, reliable
// magic-number probe, and an environment that renders correctly only when someone spelled the
// suffix the way we expected is a bad trade for two `if`s.
//
// Returns an invalid EnvImage on any failure, having logged why. Callers can hand the result
// straight to HdrIbl, which turns an empty span into its dummy environment.
EnvImage loadEnvImage(const std::filesystem::path& file);

// Extensions this loader accepts, for file dialogs and CLI help -- e.g. "hdr,exr". Kept next to
// the loader so a new format cannot be added without the UI that offers it following along.
const char* envImageExtensions();

// True when `file` has one of those extensions. Used by the drop handler, which has to decide
// whether a dropped file is an environment before opening it.
bool isEnvImageExtension(const std::filesystem::path& file);
