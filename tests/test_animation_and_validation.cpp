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
#include "common/test_utils.hpp"
#include "gltf_scene.hpp"
#include "gltf_scene_animation.hpp"
#include "gltf_scene_validator.hpp"

using namespace gltf_test;

// Compact on scene without valid parse should return false and not crash
TEST(AnimationAndValidation, CompactWithoutValidSceneReturnsFalse)
{
  nvvkgltf::Scene scene;
  EXPECT_FALSE(scene.compactModel()) << "compactModel() should return false when no valid scene is loaded";
}

// Validator on empty/minimal model (no scene loaded)
TEST(AnimationAndValidation, ValidatorOnEmptyScene)
{
  nvvkgltf::Scene scene;
  auto            result = scene.validator().validateModel();
  (void)result;
}

// Load a valid scene, run validator, then compact (no-op when nothing to remove)
TEST(AnimationAndValidation, ValidatorAndCompactOnLoadedScene)
{
  nvvkgltf::Scene scene;
  try
  {
    auto path   = TestResources::getResourcePath("Box.glb");
    bool loaded = scene.load(path);
    ASSERT_TRUE(loaded);

    auto result = scene.validator().validateModel();
    EXPECT_TRUE(result.valid) << "Box.glb should pass validation; got: " << (result.errors.empty() ? "" : result.errors[0]);

    bool compacted = scene.compactModel();
    EXPECT_FALSE(compacted) << "Box with no orphans should report no compaction";
    EXPECT_TRUE(scene.valid());
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }
}

// Animation API is reachable after load; validator and compact are exercised in other tests.
// (Empty-sampler fix is in processAnimationChannel; updateAnimation is used in render loop.)
TEST(AnimationAndValidation, AnimationApiReachable)
{
  nvvkgltf::Scene scene;
  try
  {
    auto path   = TestResources::getResourcePath("Box.glb");
    bool loaded = scene.load(path);
    ASSERT_TRUE(loaded);
    EXPECT_GE(scene.animation().getNumAnimations(), 0);
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }
}
