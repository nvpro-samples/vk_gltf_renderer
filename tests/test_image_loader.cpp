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

// CPU-only unit test for nvvkgltf::loadFromMemory's WebP and OpenEXR paths: the decoder shared by
// glTF loading and the material inspector's image import.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <glm/gtc/packing.hpp>
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

// An uncompressed scanline OpenEXR with HALF channels, written by hand so a decoder bug cannot cancel
// out against a shared encoder. `channelNames` must already be in the alphabetical order EXR stores;
// `values` is width * height * channelNames.size() floats, interleaved per texel in that order.
std::vector<uint8_t> makeExr(int width, int height, const std::vector<std::string>& channelNames, const std::vector<float>& values)
{
  std::vector<uint8_t> out;
  auto put = [&](const void* p, size_t n) { out.insert(out.end(), static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + n); };
  auto putI32 = [&](int32_t v) { put(&v, 4); };
  auto putStr = [&](const std::string& str) { put(str.c_str(), str.size() + 1); };
  auto putAttr = [&](const std::string& name, const std::string& type, const std::vector<uint8_t>& payload) {
    putStr(name);
    putStr(type);
    putI32(static_cast<int32_t>(payload.size()));
    put(payload.data(), payload.size());
  };
  auto bytesOf = [](std::initializer_list<int32_t> ints) {
    std::vector<uint8_t> b(ints.size() * 4);
    std::memcpy(b.data(), ints.begin(), b.size());
    return b;
  };

  putI32(0x01312F76);  // magic
  putI32(2);           // version 2, scanline, single part

  std::vector<uint8_t> chlist;
  for(const std::string& name : channelNames)
  {
    chlist.insert(chlist.end(), name.begin(), name.end());
    chlist.push_back(0);
    const int32_t fields[4] = {1 /*HALF*/, 0 /*pLinear + reserved*/, 1, 1 /*x, y sampling*/};
    const auto*   f         = reinterpret_cast<const uint8_t*>(fields);
    chlist.insert(chlist.end(), f, f + 16);
  }
  chlist.push_back(0);
  putAttr("channels", "chlist", chlist);
  putAttr("compression", "compression", {0});  // NO_COMPRESSION
  putAttr("dataWindow", "box2i", bytesOf({0, 0, width - 1, height - 1}));
  putAttr("displayWindow", "box2i", bytesOf({0, 0, width - 1, height - 1}));
  putAttr("lineOrder", "lineOrder", {0});
  putAttr("pixelAspectRatio", "float", {0, 0, 0x80, 0x3f});
  putAttr("screenWindowCenter", "v2f", {0, 0, 0, 0, 0, 0, 0, 0});
  putAttr("screenWindowWidth", "float", {0, 0, 0x80, 0x3f});
  out.push_back(0);  // end of header

  const size_t   numCh    = channelNames.size();
  const size_t   rowBytes = size_t(width) * numCh * 2;
  const uint64_t first    = out.size() + 8u * height;
  for(int y = 0; y < height; ++y)
  {
    const uint64_t offset = first + uint64_t(y) * (8 + rowBytes);
    put(&offset, 8);
  }
  for(int y = 0; y < height; ++y)
  {
    putI32(y);
    putI32(static_cast<int32_t>(rowBytes));
    for(size_t c = 0; c < numCh; ++c)  // planar within a scanline: all of a channel's texels, then the next
      for(int x = 0; x < width; ++x)
      {
        const uint16_t h = glm::packHalf1x16(values[(size_t(y) * width + x) * numCh + c]);
        put(&h, 2);
      }
  }
  return out;
}

uint16_t halfAt(const nvvkgltf::LoadedImageData& image, size_t index)
{
  uint16_t h = 0;
  std::memcpy(&h, image.mipData[0].data() + index * 2, 2);
  return h;
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

// Values are powers of two and halves, exactly representable in half, so equality is exact. The 3.5 is
// above 1: an EXR is HDR data and must not be clamped to an 8-bit range.
TEST(ImageLoader, ExrRgbaDecodesToHalfFloat)
{
  const std::vector<float> texels = {0.25F, 0.5F, 0.75F, 1.0F, 3.5F, 0.0F, 2.0F, 0.5F,  //
                                     1.0F,  1.5F, 2.5F,  0.0F, 0.125F, 8.0F, 4.0F, 0.25F};
  const std::vector<uint8_t> file = makeExr(2, 2, {"A", "B", "G", "R"}, [&] {
    // The writer takes texels in channel-storage order (A, B, G, R); the expectation below is RGBA.
    std::vector<float> abgr;
    for(size_t t = 0; t < 4; ++t)
      abgr.insert(abgr.end(), {texels[t * 4 + 3], texels[t * 4 + 2], texels[t * 4 + 1], texels[t * 4 + 0]});
    return abgr;
  }());

  nvvkgltf::LoadedImageData out;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ false));
  EXPECT_EQ(out.format, VK_FORMAT_R16G16B16A16_SFLOAT);
  EXPECT_EQ(out.size.width, 2u);
  EXPECT_EQ(out.size.height, 2u);
  ASSERT_EQ(out.mipData.size(), 1u);
  ASSERT_EQ(out.mipData[0].size(), texels.size() * 2);
  for(size_t i = 0; i < texels.size(); ++i)
    EXPECT_EQ(halfAt(out, i), glm::packHalf1x16(texels[i])) << "component " << i;
}

TEST(ImageLoader, ExrIsLinearWhateverTheSrgbFlag)
{
  const std::vector<uint8_t> file = makeExr(1, 1, {"B", "G", "R"}, {0.5F, 0.5F, 0.5F});

  nvvkgltf::LoadedImageData out;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ true));
  EXPECT_EQ(out.format, VK_FORMAT_R16G16B16A16_SFLOAT);
  EXPECT_EQ(halfAt(out, 3), glm::packHalf1x16(1.0F)) << "a file without alpha is opaque";
}

TEST(ImageLoader, ExrSingleChannelStaysSingleChannel)
{
  const std::vector<uint8_t> file = makeExr(2, 1, {"Y"}, {0.25F, 0.75F});

  nvvkgltf::LoadedImageData out;
  ASSERT_TRUE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ false));
  EXPECT_EQ(out.format, VK_FORMAT_R16_SFLOAT);
  ASSERT_EQ(out.mipData[0].size(), 4u);
  EXPECT_EQ(halfAt(out, 0), glm::packHalf1x16(0.25F));
  EXPECT_EQ(halfAt(out, 1), glm::packHalf1x16(0.75F));
  // Sampled as RRR1, like a one-channel 8-bit image.
  EXPECT_EQ(out.componentMapping.g, VK_COMPONENT_SWIZZLE_R);
  EXPECT_EQ(out.componentMapping.b, VK_COMPONENT_SWIZZLE_R);
  EXPECT_EQ(out.componentMapping.a, VK_COMPONENT_SWIZZLE_ONE);
}

TEST(ImageLoader, ExrCorruptFileFailsWithoutImage)
{
  std::vector<uint8_t> file = makeExr(8, 8, {"B", "G", "R"}, std::vector<float>(8 * 8 * 3, 0.5F));
  file.resize(file.size() / 2);  // the offset table now points past the end of the data

  nvvkgltf::LoadedImageData out;
  EXPECT_FALSE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ false));
  EXPECT_TRUE(out.mipData.empty());

  file.resize(12);  // the magic number alone, no header
  EXPECT_FALSE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ false));
}

// Documented limit: only R, G, B, A (or a lone channel) are read, so a two-channel file is refused
// with a log line rather than guessed at.
TEST(ImageLoader, ExrTwoChannelFileIsRefused)
{
  const std::vector<uint8_t> file = makeExr(2, 2, {"G", "R"}, std::vector<float>(2 * 2 * 2, 0.5F));

  nvvkgltf::LoadedImageData out;
  EXPECT_FALSE(nvvkgltf::loadFromMemory(out, file.data(), file.size(), /*srgb*/ false));
  EXPECT_TRUE(out.mipData.empty());
}
