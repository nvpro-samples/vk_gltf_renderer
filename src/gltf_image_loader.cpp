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
// Loads glTF image data from various container formats (KTX2, DDS, WebP, OpenEXR,
// and standard formats via stb_image) into Vulkan-compatible image
// descriptors. Handles format detection, swizzle mapping, mip chains,
// and conversion to VkFormat-based ImageData structures.
//

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <limits>

#include <glm/gtc/packing.hpp>
#include <stb/stb_image.h>
#include <webp/decode.h>
#include "nvimageformats/nv_dds.h"
#include "nvimageformats/nv_ktx.h"
#include "nvimageformats/texture_formats.h"
#include "nvutils/logger.hpp"
#include "gltf_image_loader.hpp"

#ifdef NVP_SUPPORTS_BASISU
// Same tinyexr the environment loader uses (see env_image_loader.cpp): basis_universal's vendored
// copy, not a second one. NVP_SUPPORTS_BASISU arrives transitively from nvimageformats.
#include <3rdparty/tinyexr.h>
#endif

namespace nvvkgltf {

namespace {

VkComponentSwizzle ktxSwizzleToVk(nv_ktx::Swizzle swizzle)
{
  switch(swizzle)
  {
    case nv_ktx::Swizzle::ZERO:
      return VK_COMPONENT_SWIZZLE_ZERO;
    case nv_ktx::Swizzle::ONE:
      return VK_COMPONENT_SWIZZLE_ONE;
    case nv_ktx::Swizzle::R:
      return VK_COMPONENT_SWIZZLE_R;
    case nv_ktx::Swizzle::G:
      return VK_COMPONENT_SWIZZLE_G;
    case nv_ktx::Swizzle::B:
      return VK_COMPONENT_SWIZZLE_B;
    case nv_ktx::Swizzle::A:
      return VK_COMPONENT_SWIZZLE_A;
    default:
      return VK_COMPONENT_SWIZZLE_IDENTITY;
  }
}

VkComponentMapping ktxSwizzleToVkComponentMapping(const std::array<nv_ktx::Swizzle, 4>& swizzle)
{
  return {ktxSwizzleToVk(swizzle[0]), ktxSwizzleToVk(swizzle[1]), ktxSwizzleToVk(swizzle[2]), ktxSwizzleToVk(swizzle[3])};
}

bool loadDds(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog)
{
  const char ddsIdentifier[4] = {'D', 'D', 'S', ' '};
  if(byteLength < sizeof(ddsIdentifier) || memcmp(data, ddsIdentifier, sizeof(ddsIdentifier)) != 0)
    return false;

  nv_dds::Image        ddsImage{};
  nv_dds::ReadSettings settings{};
  const nv_dds::ErrorWithText readResult = ddsImage.readFromMemory(reinterpret_cast<const char*>(data), byteLength, settings);
  if(readResult.has_value())
  {
    LOGW("Failed to read image %" PRIu64 " using nv_dds: %s\n", imageIDForLog, readResult.value().c_str());
    return false;
  }

  out.size.width  = ddsImage.getWidth(0);
  out.size.height = ddsImage.getHeight(0);
  if(ddsImage.getDepth(0) > 1)
  {
    LOGW("This DDS image had a depth of %u, but loadFromMemory() cannot handle volume textures.\n", ddsImage.getDepth(0));
    return false;
  }
  if(ddsImage.getNumFaces() > 1)
  {
    LOGW("This DDS image had %u faces, but loadFromMemory() cannot handle cubemaps.\n", ddsImage.getNumFaces());
    return false;
  }
  if(ddsImage.getNumLayers() > 1)
  {
    LOGW("This DDS image had %u array elements, but loadFromMemory() cannot handle array textures.\n", ddsImage.getNumLayers());
    return false;
  }
  out.format = texture_formats::dxgiToVulkan(ddsImage.dxgiFormat);
  out.format = texture_formats::tryForceVkFormatTransferFunction(out.format, srgb);
  if(out.format == VK_FORMAT_UNDEFINED)
  {
    LOGW("Could not determine a VkFormat for DXGI format %u (%s).\n", ddsImage.dxgiFormat,
         texture_formats::getDXGIFormatName(ddsImage.dxgiFormat));
    return false;
  }

  for(uint32_t i = 0; i < ddsImage.getNumMips(); i++)
  {
    std::vector<char>& mip = ddsImage.subresource(i, 0, 0).data;
    out.mipData.push_back(std::move(mip));
  }
  return true;
}

bool loadKtx(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog)
{
  const uint8_t ktxIdentifier[5] = {0xAB, 0x4B, 0x54, 0x58, 0x20};  // Common for KTX1 + KTX2
  if(byteLength < sizeof(ktxIdentifier) || memcmp(data, ktxIdentifier, sizeof(ktxIdentifier)) != 0)
    return false;

  nv_ktx::Image              ktxImage;
  const nv_ktx::ReadSettings ktxReadSettings;
  const nv_ktx::ErrorWithText maybeError = ktxImage.readFromMemory(reinterpret_cast<const char*>(data), byteLength, ktxReadSettings);
  if(maybeError.has_value())
  {
    LOGW("Failed to read image %" PRIu64 " using nv_ktx: %s\n", imageIDForLog, maybeError->c_str());
    return false;
  }

  out.size.width  = ktxImage.mip0Width;
  out.size.height = ktxImage.mip0Height;
  if(ktxImage.mip0Depth > 1)
  {
    LOGW("KTX image %" PRIu64 " had a depth of %u, but loadFromMemory() cannot handle volume textures.\n",
         imageIDForLog, ktxImage.mip0Depth);
    return false;
  }
  if(ktxImage.numFaces > 1)
  {
    LOGW("KTX image %" PRIu64 " had %u faces, but loadFromMemory() cannot handle cubemaps.\n", imageIDForLog, ktxImage.numFaces);
    return false;
  }
  if(ktxImage.numLayersPossibly0 > 1)
  {
    LOGW("KTX image %" PRIu64 " had %u array elements, but loadFromMemory() cannot handle array textures.\n",
         imageIDForLog, ktxImage.numLayersPossibly0);
    return false;
  }
  out.format           = texture_formats::tryForceVkFormatTransferFunction(ktxImage.format, srgb);
  out.componentMapping = ktxSwizzleToVkComponentMapping(ktxImage.swizzle);

  for(uint32_t i = 0; i < ktxImage.numMips; i++)
  {
    std::vector<char>& mip = ktxImage.subresource(i, 0, 0);
    out.mipData.push_back(std::move(mip));
  }
  return true;
}

bool loadWebp(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog)
{
  // VP8/VP8L bitstreams store each dimension in 14 bits. Only the VP8X canvas can declare more, and
  // WebPGetFeatures returns it unchecked for animated or truncated files, so a tiny file could otherwise
  // request a multi-GiB allocation.
  constexpr int kMaxWebpDimension = 1 << 14;

  const uint8_t* dataU8 = reinterpret_cast<const uint8_t*>(data);

  WebPBitstreamFeatures features{};
  if(WebPGetFeatures(dataU8, byteLength, &features) != VP8_STATUS_OK)
    return false;
  if(features.has_animation)
  {
    LOGW("WebP image %" PRIu64 " is animated, which is not supported.\n", imageIDForLog);
    return false;
  }
  const int width  = features.width;
  const int height = features.height;
  if(width <= 0 || height <= 0 || width > kMaxWebpDimension || height > kMaxWebpDimension)
  {
    LOGW("WebP image %" PRIu64 " has invalid dimensions %dx%d.\n", imageIDForLog, width, height);
    return false;
  }

  std::vector<char> decompressed(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
  if(!WebPDecodeRGBAInto(dataU8, byteLength, reinterpret_cast<uint8_t*>(decompressed.data()), decompressed.size(), width * 4))
  {
    LOGW("Failed to decode WebP image %" PRIu64 "\n", imageIDForLog);
    return false;
  }

  out.format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
  out.size   = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  out.mipData.clear();
  out.mipData.push_back(std::move(decompressed));
  return true;
}

// PROVISIONAL(OpenEXR textures): not part of glTF 2.0 / 2.1. A glTF image may point at an OpenEXR
// file (uri ending in .exr, mimeType image/x-exr by convention); the decoder dispatches on the file's
// magic number, so the suffix and mimeType are not consulted. See docs/gltf_2_1.md.
//
// EXR is scene-linear data, so the result is always a float format: the sRGB flag is ignored (an
// sRGB transfer function applied to linear floats would be wrong) and the values are kept as authored,
// above 1.0 included. Half is the working precision, as in the file itself: one channel stays a
// single-channel R16F (sampled as RRR1, like a grayscale 8-bit image), anything else becomes
// RGBA16F with alpha 1 when the file has none. Mips are generated on the GPU like for any
// single-level image.
bool isExr(const void* data, size_t byteLength)
{
#ifdef NVP_SUPPORTS_BASISU
  return IsEXRFromMemory(static_cast<const unsigned char*>(data), byteLength) == TINYEXR_SUCCESS;
#else
  const uint8_t magic[4] = {0x76, 0x2f, 0x31, 0x01};
  return byteLength >= sizeof(magic) && memcmp(data, magic, sizeof(magic)) == 0;
#endif
}

#ifdef NVP_SUPPORTS_BASISU

// glm's conversion does not saturate: a float above half range would become infinity, which a filtered
// sample can turn into NaN. Clamp to the finite range; NaN becomes 0.
uint16_t floatToHalf(float v)
{
  constexpr float kHalfMax = 65504.0F;
  return glm::packHalf1x16(v != v ? 0.0F : std::clamp(v, -kHalfMax, kHalfMax));
}

bool loadExr(LoadedImageData& out, const void* data, size_t byteLength, uint64_t imageIDForLog)
{
  const unsigned char* bytes = static_cast<const unsigned char*>(data);

  // The channel count decides the Vulkan format; LoadEXRFromMemory below always returns RGBA.
  int numChannels = 0;
  {
    EXRVersion version{};
    EXRHeader  header{};
    InitEXRHeader(&header);
    const char* error = nullptr;
    if(ParseEXRVersionFromMemory(&version, bytes, byteLength) != TINYEXR_SUCCESS
       || ParseEXRHeaderFromMemory(&header, &version, bytes, byteLength, &error) != TINYEXR_SUCCESS)
    {
      LOGW("Failed to read the EXR header of image %" PRIu64 ": %s\n", imageIDForLog, error ? error : "unknown error");
      FreeEXRErrorMessage(error);
      return false;
    }
    numChannels = header.num_channels;
    FreeEXRHeader(&header);
  }

  float*      rgba   = nullptr;
  int         width  = 0;
  int         height = 0;
  const char* error  = nullptr;
  if(LoadEXRFromMemory(&rgba, &width, &height, bytes, byteLength, &error) != TINYEXR_SUCCESS)
  {
    LOGW("Failed to decode EXR image %" PRIu64 ": %s\n", imageIDForLog, error ? error : "unknown error");
    FreeEXRErrorMessage(error);
    return false;
  }

  const size_t texels   = static_cast<size_t>(width) * static_cast<size_t>(height);
  const bool   gray     = numChannels == 1;
  const size_t channels = gray ? 1 : 4;

  std::vector<char> mip(texels * channels * sizeof(uint16_t));
  uint16_t*         dst = reinterpret_cast<uint16_t*>(mip.data());
  if(gray)
  {
    for(size_t i = 0; i < texels; ++i)
      dst[i] = floatToHalf(rgba[4 * i]);
    out.format           = VK_FORMAT_R16_SFLOAT;
    out.componentMapping = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE};
  }
  else
  {
    for(size_t i = 0; i < texels * 4; ++i)
      dst[i] = floatToHalf(rgba[i]);
    out.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  }
  free(rgba);  // tinyexr allocates with malloc

  out.size = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  out.mipData.clear();
  out.mipData.push_back(std::move(mip));
  return true;
}

#else

bool loadExr(LoadedImageData&, const void*, size_t, uint64_t imageIDForLog)
{
  LOGW("Image %" PRIu64 " is an OpenEXR file, which this build cannot decode (basis_universal is not available).\n", imageIDForLog);
  return false;
}

#endif

bool loadStb(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog)
{
  if(byteLength > static_cast<size_t>(std::numeric_limits<int>::max()))
  {
    LOGW("File for image %" PRIu64 " was too large (%zu bytes) for stb_image to read.\n", imageIDForLog, byteLength);
    return false;
  }

  const stbi_uc* dataStb   = reinterpret_cast<const stbi_uc*>(data);
  const int      lengthStb = static_cast<int>(byteLength);

  int w = 0, h = 0, comp = 0;
  if(!stbi_info_from_memory(dataStb, lengthStb, &w, &h, &comp))
  {
    LOGW("Failed to get info using stb_image for image %" PRIu64 "\n", imageIDForLog);
    return false;
  }

  const bool is16Bit = stbi_is_16_bit_from_memory(dataStb, lengthStb);

  stbi_uc* decompressed = nullptr;
  size_t   bytesPerPixel{0};
  int      requiredComponents = comp == 1 ? 1 : 4;
  if(is16Bit)
  {
    stbi_us* decompressed16 = stbi_load_16_from_memory(dataStb, lengthStb, &w, &h, &comp, requiredComponents);
    bytesPerPixel           = sizeof(*decompressed16) * requiredComponents;
    decompressed            = reinterpret_cast<stbi_uc*>(decompressed16);
  }
  else
  {
    decompressed  = stbi_load_from_memory(dataStb, lengthStb, &w, &h, &comp, requiredComponents);
    bytesPerPixel = sizeof(*decompressed) * requiredComponents;
  }

  switch(requiredComponents)
  {
    case 1:
      out.format = is16Bit ? VK_FORMAT_R16_UNORM : VK_FORMAT_R8_UNORM;
      out.componentMapping = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_ONE};
      break;
    case 4:
      out.format = is16Bit ? VK_FORMAT_R16G16B16A16_UNORM : srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
      break;
    default:
      stbi_image_free(decompressed);
      return false;
  }

  if(decompressed && w > 0 && h > 0 && out.format != VK_FORMAT_UNDEFINED)
  {
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(w) * h * bytesPerPixel;
    out.size                = VkExtent2D{static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
    out.mipData.clear();
    out.mipData.emplace_back(decompressed, decompressed + bufferSize);
  }

  stbi_image_free(decompressed);
  return out.size.width > 0 && out.size.height > 0;
}

}  // namespace

bool loadFromMemory(LoadedImageData& out, const void* data, size_t byteLength, bool srgb, uint64_t imageIDForLog)
{
  out = LoadedImageData{};

  if(data == nullptr || byteLength == 0)
    return false;

  if(loadDds(out, data, byteLength, srgb, imageIDForLog))
    return true;
  if(loadKtx(out, data, byteLength, srgb, imageIDForLog))
    return true;
  if(loadWebp(out, data, byteLength, srgb, imageIDForLog))
    return true;
  if(isExr(data, byteLength))  // claimed by magic: a corrupt EXR logs its own reason, not stb's
    return loadExr(out, data, byteLength, imageIDForLog);
  if(loadStb(out, data, byteLength, srgb, imageIDForLog))
    return true;

  return false;
}

}  // namespace nvvkgltf
