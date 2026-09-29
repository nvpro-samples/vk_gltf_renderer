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
// Environment image decoding -- see env_image_loader.hpp for what this covers and why the EXR
// decoder lives in the sample rather than in nvvk.
//
// Both formats end at the same place: a tightly packed RGBA32F buffer, which is exactly what
// nvvk::HdrIbl's pixels overload consumes. Only the middle differs -- stb_image for Radiance
// .hdr, tinyexr for OpenEXR -- and each is probed by magic number rather than by file extension,
// so an environment does not depend on someone having spelled the suffix as expected.
//
// Every failure path returns an invalid EnvImage after logging why. The caller can hand that
// straight on: an empty span is what makes HdrIbl produce its dummy environment, so there is no
// separate error branch to keep in sync.
//

#include <cstring>
#include <limits>

#include <stb/stb_image.h>

#ifdef NVP_SUPPORTS_BASISU
// tinyexr comes from basis_universal's vendored copy, reached through the include directory that
// target exports. Deliberately not vendored a second time: two copies of a decoder is two sets of
// CVEs to track. NVP_SUPPORTS_BASISU arrives transitively from nvimageformats, which links basisu
// PUBLIC -- so an EXR-less configuration still compiles, it just loses the format.
#include <3rdparty/tinyexr.h>
#endif

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>

#include "env_image_loader.hpp"

const char* envImageExtensions()
{
#ifdef NVP_SUPPORTS_BASISU
  return "hdr,exr";
#else
  return "hdr";
#endif
}

bool isEnvImageExtension(const std::filesystem::path& file)
{
  if(nvutils::extensionMatches(file, ".hdr"))
    return true;
#ifdef NVP_SUPPORTS_BASISU
  if(nvutils::extensionMatches(file, ".exr"))
    return true;
#endif
  return false;
}

EnvImage loadEnvImage(const std::filesystem::path& file)
{
  nvutils::ScopedTimer st(__FUNCTION__);
  EnvImage             out;

  if(file.empty())
    return out;

  // Read once into memory: it sidesteps the text-encoding question in the stbi filename API, and
  // both decoders below want a buffer anyway.
  const std::string contents = nvutils::loadFile(file);
  if(contents.empty())
  {
    LOGW("Environment image does not exist or is empty: %s\n", nvutils::utf8FromPath(file).c_str());
    return out;
  }
  if(contents.size() > std::numeric_limits<int>::max())
  {
    LOGW("Environment image is too large to decode: %s\n", nvutils::utf8FromPath(file).c_str());
    return out;
  }

  const unsigned char* data = reinterpret_cast<const unsigned char*>(contents.data());
  const int            size = static_cast<int>(contents.size());

#ifdef NVP_SUPPORTS_BASISU
  if(IsEXRFromMemory(data, contents.size()) == TINYEXR_SUCCESS)
  {
    float*      rgba  = nullptr;
    int         width = 0, height = 0;
    const char* error = nullptr;
    // LoadEXRFromMemory always produces RGBA float, which is exactly the layout HdrIbl wants; a
    // single-channel or RGB EXR is expanded for us.
    if(LoadEXRFromMemory(&rgba, &width, &height, data, contents.size(), &error) != TINYEXR_SUCCESS)
    {
      LOGW("EXR decode failed for %s: %s\n", nvutils::utf8FromPath(file).c_str(), error ? error : "unknown error");
      FreeEXRErrorMessage(error);
      return out;
    }
    out.size = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    out.pixels.assign(rgba, rgba + size_t(width) * size_t(height) * 4);
    free(rgba);  // tinyexr allocates with malloc
    return out;
  }
#endif

  if(!stbi_is_hdr_from_memory(data, size))
  {
    LOGW("Not a supported environment image (expected %s): %s\n", envImageExtensions(), nvutils::utf8FromPath(file).c_str());
    return out;
  }

  int    width = 0, height = 0, components = 0;
  float* rgba = stbi_loadf_from_memory(data, size, &width, &height, &components, STBI_rgb_alpha);
  if(!rgba)
  {
    LOGW("HDR decode failed for %s: %s\n", nvutils::utf8FromPath(file).c_str(), stbi_failure_reason());
    return out;
  }
  out.size = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  out.pixels.assign(rgba, rgba + size_t(width) * size_t(height) * 4);
  stbi_image_free(rgba);
  return out;
}
