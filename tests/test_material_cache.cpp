/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include "gltf_material_cache.hpp"

// M7d: MaterialCache tests — conversion, incremental updates, topology change detection

TEST(MaterialCache, BuildFromEmptyMaterials)
{
  nvvkgltf::MaterialCache         cache;
  std::vector<tinygltf::Material> materials;
  cache.buildFromMaterials(materials);

  EXPECT_TRUE(cache.getShadeMaterials().empty());
  EXPECT_EQ(cache.getTextureInfos().size(), 1u);  // Sentinel entry at index 0
}

TEST(MaterialCache, BuildFromSingleOpaqueMaterial)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material mat;
  mat.alphaMode                            = "OPAQUE";
  mat.doubleSided                          = false;
  mat.pbrMetallicRoughness.baseColorFactor = {1.0, 0.0, 0.0, 1.0};
  mat.pbrMetallicRoughness.metallicFactor  = 0.5;
  mat.pbrMetallicRoughness.roughnessFactor = 0.8;

  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials);

  ASSERT_EQ(cache.getShadeMaterials().size(), 1u);

  const auto& shade = cache.getShadeMaterials()[0];
  EXPECT_EQ(shade.alphaMode, 0);  // OPAQUE
  EXPECT_EQ(shade.doubleSided, 0);
  EXPECT_FLOAT_EQ(shade.pbrBaseColorFactor[0], 1.0f);
  EXPECT_FLOAT_EQ(shade.pbrBaseColorFactor[1], 0.0f);
  EXPECT_FLOAT_EQ(shade.pbrMetallicFactor, 0.5f);
  EXPECT_FLOAT_EQ(shade.pbrRoughnessFactor, 0.8f);
}

TEST(MaterialCache, BuildFromMaskAndBlendAlphaModes)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material maskMat;
  maskMat.alphaMode   = "MASK";
  maskMat.alphaCutoff = 0.3;

  tinygltf::Material blendMat;
  blendMat.alphaMode = "BLEND";

  std::vector<tinygltf::Material> materials = {maskMat, blendMat};
  cache.buildFromMaterials(materials);

  ASSERT_EQ(cache.getShadeMaterials().size(), 2u);
  EXPECT_EQ(cache.getShadeMaterials()[0].alphaMode, 1);  // MASK
  EXPECT_FLOAT_EQ(cache.getShadeMaterials()[0].alphaCutoff, 0.3f);
  EXPECT_EQ(cache.getShadeMaterials()[1].alphaMode, 2);  // BLEND
}

TEST(MaterialCache, BuildWithBaseColorTexture)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material mat;
  mat.pbrMetallicRoughness.baseColorTexture.index    = 0;
  mat.pbrMetallicRoughness.baseColorTexture.texCoord = 0;

  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials);

  ASSERT_EQ(cache.getShadeMaterials().size(), 1u);
  // Texture index > 0 means a texture info was added
  EXPECT_GT(cache.getShadeMaterials()[0].pbrBaseColorTexture, 0u);
  // Sentinel + at least one real texture info
  EXPECT_GE(cache.getTextureInfos().size(), 2u);
}

TEST(MaterialCache, UpdateMaterialInPlace)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material mat;
  mat.alphaMode                                   = "OPAQUE";
  mat.pbrMetallicRoughness.roughnessFactor        = 0.5;
  mat.pbrMetallicRoughness.baseColorTexture.index = 0;

  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials);

  // Modify roughness
  mat.pbrMetallicRoughness.roughnessFactor = 0.9;
  auto result                              = cache.updateMaterial(0, mat);

  EXPECT_FALSE(result.topologyChanged);
  EXPECT_FLOAT_EQ(cache.getShadeMaterials()[0].pbrRoughnessFactor, 0.9f);
}

TEST(MaterialCache, UpdateDetectsTopologyChangeWhenTextureAdded)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material mat;
  mat.alphaMode = "OPAQUE";

  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials);

  // Now add a base color texture — topology change
  mat.pbrMetallicRoughness.baseColorTexture.index = 0;
  auto result                                     = cache.updateMaterial(0, mat);

  EXPECT_TRUE(result.topologyChanged);
}

TEST(MaterialCache, UpdateDetectsTopologyChangeWhenTextureRemoved)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material mat;
  mat.pbrMetallicRoughness.baseColorTexture.index = 0;

  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials);

  // Remove the texture — topology change
  mat.pbrMetallicRoughness.baseColorTexture.index = -1;
  auto result                                     = cache.updateMaterial(0, mat);

  EXPECT_TRUE(result.topologyChanged);
}

TEST(MaterialCache, UpdateOutOfRangeReturnsEmpty)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material              mat;
  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials);

  auto result = cache.updateMaterial(5, mat);
  EXPECT_FALSE(result.topologyChanged);
  EXPECT_FALSE(result.span.hasAny());
}

TEST(MaterialCache, ClearResetsAll)
{
  nvvkgltf::MaterialCache cache;

  tinygltf::Material              mat;
  std::vector<tinygltf::Material> materials = {mat};
  cache.buildFromMaterials(materials);

  cache.clear();
  EXPECT_TRUE(cache.getShadeMaterials().empty());
  EXPECT_TRUE(cache.getTextureInfos().empty());
}
