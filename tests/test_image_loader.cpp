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

// CPU-only unit test for nvvkgltf::loadFromMemory's WebP path: the decoder shared by glTF loading
// and the material inspector's image import.

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>
#include <webp/encode.h>

#include "gltf_image_loader.hpp"

namespace {

constexpr int kWidth  = 3;
constexpr int kHeight = 2;

// Distinct RGBA texel values so a row/channel swap would be caught.
std::vector<uint8_t> makePixels()
{
  std::vector<uint8_t> rgba(kWidth * kHeight * 4);
  for(size_t i = 0; i < rgba.size(); ++i)
    rgba[i] = static_cast<uint8_t>(i * 11 + 7);
  return rgba;
}

// Lossless, so the decoded bytes must match the source exactly.
std::vector<uint8_t> encodeLossless(const std::vector<uint8_t>& rgba)
{
  uint8_t*             out  = nullptr;
  const size_t         size = WebPEncodeLosslessRGBA(rgba.data(), kWidth, kHeight, kWidth * 4, &out);
  std::vector<uint8_t> file(out, out + size);
  WebPFree(out);
  return file;
}

// A RIFF/WEBP file holding only a VP8X chunk that declares a 65536 x 65535 canvas. libwebp reports
// that canvas without a bitstream to check it against, so it must be rejected before allocating.
std::vector<uint8_t> makeHugeCanvasHeader(bool animated)
{
  std::vector<uint8_t> file = {'R', 'I', 'F', 'F', 22, 0, 0, 0, 'W', 'E', 'B', 'P', 'V', 'P', '8', 'X', 10, 0, 0, 0};
  const uint8_t vp8x[10] = {static_cast<uint8_t>(animated ? 0x02 : 0x00), 0, 0, 0, 0xFF, 0xFF, 0x00, 0xFE, 0xFF, 0x00};
  file.insert(file.end(), vp8x, vp8x + sizeof(vp8x));
  return file;
}

}  // namespace

TEST(ImageLoader, WebpDecodesLosslessPixels)
{
  const std::vector<uint8_t> pixels = makePixels();
  const std::vector<uint8_t> file   = encodeLossless(pixels);
  ASSERT_FALSE(file.empty());

  nvvkgltf::LoadedImageData out;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ false));
  EXPECT_EQ(out.size.width, static_cast<uint32_t>(kWidth));
  EXPECT_EQ(out.size.height, static_cast<uint32_t>(kHeight));
  ASSERT_EQ(out.mipData.size(), 1u);
  ASSERT_EQ(out.mipData[0].size(), pixels.size());
  EXPECT_EQ(std::memcmp(out.mipData[0].data(), pixels.data(), pixels.size()), 0);
}

TEST(ImageLoader, WebpFormatFollowsSrgbFlag)
{
  const std::vector<uint8_t> file = encodeLossless(makePixels());

  nvvkgltf::LoadedImageData linear;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(linear, file.data(), file.size(), /*srgb*/ false));
  EXPECT_EQ(linear.format, VK_FORMAT_R8G8B8A8_UNORM);

  nvvkgltf::LoadedImageData srgb;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(srgb, file.data(), file.size(), /*srgb*/ true));
  EXPECT_EQ(srgb.format, VK_FORMAT_R8G8B8A8_SRGB);
}

TEST(ImageLoader, WebpRejectsUnbackedHugeCanvas)
{
  for(bool animated : {true, false})
  {
    const std::vector<uint8_t> file = makeHugeCanvasHeader(animated);
    nvvkgltf::LoadedImageData  out;
    EXPECT_FALSE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ false)) << "animated=" << animated;
    EXPECT_TRUE(out.mipData.empty());
  }
}
