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

using namespace gltf_test;

// M7e: AnimationSystem tests — animation update, time advancement

TEST(AnimationUpdate, UpdateAnimationOnSceneWithAnimations)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  // AnimatedCube has a rotation animation
  auto            path = assetsPath / "Models/AnimatedCube/glTF/AnimatedCube.gltf";
  nvvkgltf::Scene scene;
  if(!scene.load(path))
    GTEST_SKIP() << "AnimatedCube not found";

  auto& anim = scene.animation();
  ASSERT_GT(anim.getNumAnimations(), 0) << "Expected at least one animation";

  // Capture initial node transforms
  const auto&                      model = scene.getModel();
  std::vector<std::vector<double>> initialTranslations;
  for(const auto& node : model.nodes)
    initialTranslations.push_back(node.translation);

  // Advance animation time
  auto& info       = anim.getAnimationInfo(0);
  info.currentTime = info.start + (info.end - info.start) * 0.5f;

  // Update should succeed without crashing
  bool updated = anim.updateAnimation(0);
  EXPECT_TRUE(updated);

  // Node transforms should be reachable (no crash)
  scene.updateNodeWorldMatrices();
  const auto& matrices = scene.getNodesWorldMatrices();
  EXPECT_FALSE(matrices.empty());
}

TEST(AnimationUpdate, UpdateAnimationWithTimeWrapping)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  auto            path = assetsPath / "Models/AnimatedCube/glTF/AnimatedCube.gltf";
  nvvkgltf::Scene scene;
  if(!scene.load(path))
    GTEST_SKIP() << "AnimatedCube not found";

  auto& anim = scene.animation();
  ASSERT_GT(anim.getNumAnimations(), 0);

  auto& info     = anim.getAnimationInfo(0);
  float duration = info.end - info.start;
  ASSERT_GT(duration, 0.0f);

  // Advance past the end — should wrap
  info.incrementTime(duration * 2.5f, true);
  EXPECT_GE(info.currentTime, info.start);
  EXPECT_LE(info.currentTime, info.end);
}

TEST(AnimationUpdate, AnimationInfoIncrementTimeNoLoop)
{
  nvvkgltf::AnimationInfo info;
  info.start       = 0.0f;
  info.end         = 2.0f;
  info.currentTime = 0.0f;

  info.incrementTime(3.0f, false);
  EXPECT_FLOAT_EQ(info.currentTime, 2.0f) << "Should clamp to end when loop=false";
}

TEST(AnimationUpdate, AnimationInfoIncrementTimeLoop)
{
  nvvkgltf::AnimationInfo info;
  info.start       = 1.0f;
  info.end         = 3.0f;
  info.currentTime = 1.0f;

  info.incrementTime(2.5f, true);
  EXPECT_GE(info.currentTime, 1.0f);
  EXPECT_LE(info.currentTime, 3.0f);
  EXPECT_FLOAT_EQ(info.currentTime, 1.5f);  // 1.0 + 2.5 = 3.5, wrapped to 1.5
}

TEST(AnimationUpdate, AnimationInfoReset)
{
  nvvkgltf::AnimationInfo info;
  info.start       = 1.0f;
  info.end         = 5.0f;
  info.currentTime = 3.0f;

  float t = info.reset();
  EXPECT_FLOAT_EQ(t, 1.0f);
  EXPECT_FLOAT_EQ(info.currentTime, 1.0f);
}

TEST(AnimationUpdate, MultipleAnimationUpdatesStable)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  auto            path = assetsPath / "Models/AnimatedCube/glTF/AnimatedCube.gltf";
  nvvkgltf::Scene scene;
  if(!scene.load(path))
    GTEST_SKIP() << "AnimatedCube not found";

  auto& anim = scene.animation();
  if(anim.getNumAnimations() == 0)
    GTEST_SKIP() << "No animations in scene";

  auto& info = anim.getAnimationInfo(0);

  // Simulate 100 frames of animation updates — should never crash
  for(int frame = 0; frame < 100; ++frame)
  {
    info.incrementTime(1.0f / 60.0f, true);
    bool ok = anim.updateAnimation(0);
    ASSERT_TRUE(ok) << "updateAnimation failed at frame " << frame;
    scene.updateNodeWorldMatrices();
  }
}
