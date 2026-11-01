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

// Scene feature detection: the scene set is the promoted union of the per-material sets, so a live
// material edit can re-detect one material instead of the whole list.

#include <gtest/gtest.h>

#include <tinygltf/tiny_gltf.h>

#include "scene_feature_detection.hpp"

namespace {

tinygltf::Material withExtension(const char* name)
{
  tinygltf::Material m;
  m.extensions[name] = tinygltf::Value(tinygltf::Value::Object{});
  return m;
}

}  // namespace

TEST(SceneFeatures, UnionOfMaterialsMatchesSceneDetection)
{
  const std::vector<tinygltf::Material> materials = {withExtension("KHR_materials_clearcoat"), tinygltf::Material{},
                                                     withExtension("KHR_materials_scatter"), withExtension("KHR_materials_sheen")};
  nvvkgltf::SceneFeatureSet             unionSet;
  for(const tinygltf::Material& m : materials)
    unionSet.merge(nvvkgltf::detectMaterialFeatures(m));
  unionSet.promote();
  EXPECT_EQ(unionSet, nvvkgltf::detectSceneFeatures(materials));

  // Promotion runs on the union only: scatter brings volume and transmission with it.
  EXPECT_TRUE(unionSet.has(nvvkgltf::SceneFeatureSet::eTransmission));
  EXPECT_FALSE(nvvkgltf::detectMaterialFeatures(materials[2]).has(nvvkgltf::SceneFeatureSet::eTransmission));
  EXPECT_FALSE(nvvkgltf::detectMaterialFeatures(materials[1]).has(nvvkgltf::SceneFeatureSet::eClearcoat));
}
