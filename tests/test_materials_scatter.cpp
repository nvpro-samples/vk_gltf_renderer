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

// Host-side contract of KHR_materials_scatter: property defaults, clamping, the legacy
// KHR_materials_volume_scatter alias, and the read/write round-trip used by the Inspector.
// Specification: https://github.com/KhronosGroup/glTF/pull/2579

#include <gtest/gtest.h>

#include "tinygltf_utils.hpp"

namespace {

tinygltf::Value textureInfoValue(int index)
{
  tinygltf::Value::Object t;
  t["index"]    = tinygltf::Value(index);
  t["texCoord"] = tinygltf::Value(0);
  return tinygltf::Value(std::move(t));
}

}  // namespace

// A material without the extension reports the specification defaults, and in particular a zero
// strength, so the extension is a no-op.
TEST(MaterialsScatter, DefaultsWhenExtensionAbsent)
{
  tinygltf::Material          mat;
  const KHR_materials_scatter scatter = tinygltf::utils::getScatter(mat);

  EXPECT_FLOAT_EQ(scatter.scatterStrengthFactor, 0.0f);
  EXPECT_FLOAT_EQ(scatter.scatterAnisotropy, 0.0f);
  EXPECT_EQ(scatter.multiscatterColorFactor, glm::vec3(1.0f));
  EXPECT_EQ(scatter.scatterStrengthTexture.index, -1);
  EXPECT_EQ(scatter.multiscatterColorTexture.index, -1);
}

// An empty KHR_materials_scatter object also yields the specification defaults.
TEST(MaterialsScatter, DefaultsWhenExtensionEmpty)
{
  tinygltf::Material mat;
  mat.extensions[KHR_MATERIALS_SCATTER_EXTENSION_NAME] = tinygltf::Value(tinygltf::Value::Object{});

  const KHR_materials_scatter scatter = tinygltf::utils::getScatter(mat);
  EXPECT_FLOAT_EQ(scatter.scatterStrengthFactor, 0.0f);
  EXPECT_EQ(scatter.multiscatterColorFactor, glm::vec3(1.0f));
}

// The earlier KHR_materials_volume_scatter draft had no scatterStrengthFactor: scattering was
// always fully on, so legacy assets must keep scattering rather than silently switching off.
TEST(MaterialsScatter, LegacyExtensionNameDefaultsToFullStrength)
{
  tinygltf::Material      mat;
  tinygltf::Value::Object ext;
  ext["multiscatterColor"] =
      tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(0.25), tinygltf::Value(0.5), tinygltf::Value(0.75)});
  mat.extensions[KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME] = tinygltf::Value(std::move(ext));

  const KHR_materials_scatter scatter = tinygltf::utils::getScatter(mat);
  EXPECT_FLOAT_EQ(scatter.scatterStrengthFactor, 1.0f);
  // The legacy field name feeds the current multiscatterColorFactor.
  EXPECT_FLOAT_EQ(scatter.multiscatterColorFactor.x, 0.25f);
  EXPECT_FLOAT_EQ(scatter.multiscatterColorFactor.y, 0.5f);
  EXPECT_FLOAT_EQ(scatter.multiscatterColorFactor.z, 0.75f);
}

// Out-of-range authored values are clamped to the ranges the specification defines.
TEST(MaterialsScatter, ClampsToSpecifiedRanges)
{
  tinygltf::Material      mat;
  tinygltf::Value::Object ext;
  ext["scatterStrengthFactor"]                         = tinygltf::Value(4.0);
  ext["scatterAnisotropy"]                             = tinygltf::Value(-2.0);
  mat.extensions[KHR_MATERIALS_SCATTER_EXTENSION_NAME] = tinygltf::Value(std::move(ext));

  const KHR_materials_scatter scatter = tinygltf::utils::getScatter(mat);
  EXPECT_FLOAT_EQ(scatter.scatterStrengthFactor, 1.0f);
  EXPECT_GT(scatter.scatterAnisotropy, -1.0f);  // open range: never exactly -1
  EXPECT_LT(scatter.scatterAnisotropy, 0.0f);
}

// Every property survives a set/get round-trip, textures included.
TEST(MaterialsScatter, RoundTripsAllProperties)
{
  KHR_materials_scatter src;
  src.scatterStrengthFactor          = 0.5f;
  src.multiscatterColorFactor        = glm::vec3(0.8f, 0.8f, 0.1f);
  src.scatterAnisotropy              = 0.25f;
  src.scatterStrengthTexture.index   = 3;
  src.multiscatterColorTexture.index = 4;

  tinygltf::Material mat;
  tinygltf::utils::setScatter(mat, src);

  const KHR_materials_scatter dst = tinygltf::utils::getScatter(mat);
  EXPECT_FLOAT_EQ(dst.scatterStrengthFactor, src.scatterStrengthFactor);
  EXPECT_FLOAT_EQ(dst.scatterAnisotropy, src.scatterAnisotropy);
  EXPECT_EQ(dst.multiscatterColorFactor, src.multiscatterColorFactor);
  EXPECT_EQ(dst.scatterStrengthTexture.index, 3);
  EXPECT_EQ(dst.multiscatterColorTexture.index, 4);
}

// Clearing a texture slot in the Inspector must remove the property, never write "index": -1.
TEST(MaterialsScatter, ClearingATextureRemovesTheProperty)
{
  tinygltf::Material      mat;
  tinygltf::Value::Object ext;
  ext["scatterStrengthFactor"]                         = tinygltf::Value(1.0);
  ext["scatterStrengthTexture"]                        = textureInfoValue(3);
  ext["multiscatterColorTexture"]                      = textureInfoValue(4);
  mat.extensions[KHR_MATERIALS_SCATTER_EXTENSION_NAME] = tinygltf::Value(std::move(ext));

  KHR_materials_scatter scatter    = tinygltf::utils::getScatter(mat);
  scatter.scatterStrengthTexture   = {};
  scatter.multiscatterColorTexture = {};
  tinygltf::utils::setScatter(mat, scatter);

  const tinygltf::Value& written = mat.extensions[KHR_MATERIALS_SCATTER_EXTENSION_NAME];
  EXPECT_FALSE(written.Has("scatterStrengthTexture"));
  EXPECT_FALSE(written.Has("multiscatterColorTexture"));
  EXPECT_EQ(tinygltf::utils::getScatter(mat).scatterStrengthTexture.index, -1);
}
