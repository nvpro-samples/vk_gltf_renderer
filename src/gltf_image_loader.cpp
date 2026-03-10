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
// Loads glTF image data from various container formats (KTX2, DDS,
// and standard formats via stb_image) into Vulkan-compatible image
// descriptors. Handles format detection, swizzle mapping, mip chains,
// and conversion to VkFormat-based ImageData structures.
//

#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <limits>

#include <stb/stb_image.h>
#include "nvimageformats/nv_dds.h"
#include "nvimageformats/nv_ktx.h"
#include "nvimageformats/texture_formats.h"
#include "nvutils/logger.hpp"
#include "gltf_image_loader.hpp"

namespace nvvkgltf {

namespace {

VkComponentSwizzle ktxSwizzleToVk(nv_ktx::KTX_SWIZZLE swizzle)
{
  switch(swizzle)
  {
    case nv_ktx::KTX_SWIZZLE::ZERO:
      return VK_COMPONENT_SWIZZLE_ZERO;
    case nv_ktx::KTX_SWIZZLE::ONE:
      return VK_COMPONENT_SWIZZLE_ONE;
    case nv_ktx::KTX_SWIZZLE::R:
      return VK_COMPONENT_SWIZZLE_R;
    case nv_ktx::KTX_SWIZZLE::G:
      return VK_COMPONENT_SWIZZLE_G;
    case nv_ktx::KTX_SWIZZLE::B:
      return VK_COMPONENT_SWIZZLE_B;
    case nv_ktx::KTX_SWIZZLE::A:
      return VK_COMPONENT_SWIZZLE_A;
    default:
      return VK_COMPONENT_SWIZZLE_IDENTITY;
  }
}

VkComponentMapping ktxSwizzleToVkComponentMapping(const std::array<nv_ktx::KTX_SWIZZLE, 4>& swizzle)
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

  nv_ktx::KTXImage           ktxImage;
  const nv_ktx::ReadSettings ktxReadSettings;
  const nv_ktx::ErrorWithText maybeError = ktxImage.readFromMemory(reinterpret_cast<const char*>(data), byteLength, ktxReadSettings);
  if(maybeError.has_value())
  {
    LOGW("Failed to read image %" PRIu64 " using nv_ktx: %s\n", imageIDForLog, maybeError->c_str());
    return false;
  }

  out.size.width  = ktxImage.mip_0_width;
  out.size.height = ktxImage.mip_0_height;
  if(ktxImage.mip_0_depth > 1)
  {
    LOGW("KTX image %" PRIu64 " had a depth of %u, but loadFromMemory() cannot handle volume textures.\n",
         imageIDForLog, ktxImage.mip_0_depth);
    return false;
  }
  if(ktxImage.num_faces > 1)
  {
    LOGW("KTX image %" PRIu64 " had %u faces, but loadFromMemory() cannot handle cubemaps.\n", imageIDForLog, ktxImage.num_faces);
    return false;
  }
  if(ktxImage.num_layers_possibly_0 > 1)
  {
    LOGW("KTX image %" PRIu64 " had %u array elements, but loadFromMemory() cannot handle array textures.\n",
         imageIDForLog, ktxImage.num_layers_possibly_0);
    return false;
  }
  out.format           = texture_formats::tryForceVkFormatTransferFunction(ktxImage.format, srgb);
  out.componentMapping = ktxSwizzleToVkComponentMapping(ktxImage.swizzle);

  for(uint32_t i = 0; i < ktxImage.num_mips; i++)
  {
    std::vector<char>& mip = ktxImage.subresource(i, 0, 0);
    out.mipData.push_back(std::move(mip));
  }
  return true;
}

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
    out.mipData             = {{decompressed, decompressed + bufferSize}};
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
  if(loadStb(out, data, byteLength, srgb, imageIDForLog))
    return true;

  return false;
}

}  // namespace nvvkgltf
