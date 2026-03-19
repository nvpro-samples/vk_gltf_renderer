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
#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>
#include "common/test_utils.hpp"
#include "gltf_scene.hpp"
#include "gltf_scene_animation.hpp"

using namespace gltf_test;

// Helper: load a model from glTF-Sample-Assets, skip if unavailable
static bool loadSampleModel(nvvkgltf::Scene& scene, const std::string& modelPath)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    return false;
  auto path = assetsPath / modelPath;
  return scene.load(path);
}

//==========================================================================
// Deferred CPU Allocation Tests
//
// Verify the branch contract: after parseAnimations(), CPU output vectors
// are NOT pre-allocated. They are only sized on demand by ensureCpuOutput().
//==========================================================================

TEST(ComputeAnimation, SkinTaskOutputNotAllocatedAfterParse)
{
  nvvkgltf::Scene scene;
  if(!loadSampleModel(scene, "Models/SimpleSkin/glTF/SimpleSkin.gltf"))
    GTEST_SKIP() << "SimpleSkin not found";

  auto& anim = scene.animation();
  ASSERT_TRUE(anim.hasSkinning()) << "SimpleSkin should have skin tasks";

  const auto& skinTasks = anim.getSkinTasks();
  for(const auto& task : skinTasks)
  {
    EXPECT_TRUE(task.result.positions.empty()) << "Positions should not be pre-allocated after parse";
    EXPECT_TRUE(task.result.normals.empty()) << "Normals should not be pre-allocated after parse";
    EXPECT_TRUE(task.result.tangents.empty()) << "Tangents should not be pre-allocated after parse";
  }
}

TEST(ComputeAnimation, MorphOutputNotAllocatedAfterParse)
{
  nvvkgltf::Scene scene;
  if(!loadSampleModel(scene, "Models/SimpleMorph/glTF/SimpleMorph.gltf"))
    GTEST_SKIP() << "SimpleMorph not found";

  auto& anim = scene.animation();
  ASSERT_TRUE(anim.hasMorphTargets()) << "SimpleMorph should have morph targets";

  const auto& morphPrims = anim.getMorphPrimitives();
  for(size_t i = 0; i < morphPrims.size(); i++)
  {
    const auto& mr = anim.getMorphResult(i);
    EXPECT_TRUE(mr.blendedPositions.empty()) << "Blended positions should not be pre-allocated after parse";
    EXPECT_TRUE(mr.blendedNormals.empty()) << "Blended normals should not be pre-allocated after parse";
    EXPECT_TRUE(mr.blendedTangents.empty()) << "Blended tangents should not be pre-allocated after parse";
  }
}

//==========================================================================
// ensureCpuOutput() Unit Tests
//
// Verify correct sizing and idempotency of the deferred allocation methods.
//==========================================================================

TEST(ComputeAnimation, SkinTaskEnsureCpuOutputSizesCorrectly)
{
  nvvkgltf::SkinTask task{};
  task.weights.resize(100, glm::vec4(0.25f));
  task.basePositions.resize(100, glm::vec3(1.0f));
  task.baseNormals.resize(100, glm::vec3(0, 1, 0));
  task.baseTangents.resize(100, glm::vec4(1, 0, 0, 1));

  task.ensureCpuOutput();

  EXPECT_EQ(task.result.positions.size(), 100u);
  EXPECT_EQ(task.result.normals.size(), 100u);
  EXPECT_EQ(task.result.tangents.size(), 100u);
}

TEST(ComputeAnimation, SkinTaskEnsureCpuOutputIdempotent)
{
  nvvkgltf::SkinTask task{};
  task.weights.resize(50, glm::vec4(0.5f));
  task.basePositions.resize(50, glm::vec3(1.0f));

  task.ensureCpuOutput();
  ASSERT_EQ(task.result.positions.size(), 50u);

  task.result.positions[0] = glm::vec3(42.0f);
  task.ensureCpuOutput();

  EXPECT_EQ(task.result.positions.size(), 50u);
  EXPECT_EQ(task.result.positions[0], glm::vec3(42.0f)) << "Second ensureCpuOutput should not reallocate";
}

TEST(ComputeAnimation, SkinTaskEnsureCpuOutputSkipsAbsentAttributes)
{
  nvvkgltf::SkinTask task{};
  task.weights.resize(30, glm::vec4(1.0f));
  task.basePositions.resize(30, glm::vec3(0.0f));
  // baseNormals and baseTangents intentionally left empty

  task.ensureCpuOutput();

  EXPECT_EQ(task.result.positions.size(), 30u);
  EXPECT_TRUE(task.result.normals.empty()) << "Normals should remain empty when baseNormals is empty";
  EXPECT_TRUE(task.result.tangents.empty()) << "Tangents should remain empty when baseTangents is empty";
}

TEST(ComputeAnimation, MorphResultEnsureCpuOutputSizesCorrectly)
{
  nvvkgltf::MorphResult mr{};
  mr.basePositions.resize(80, glm::vec3(1.0f));
  mr.baseNormals.resize(80, glm::vec3(0, 1, 0));
  mr.baseTangents.resize(80, glm::vec4(1, 0, 0, 1));

  mr.ensureCpuOutput();

  EXPECT_EQ(mr.blendedPositions.size(), 80u);
  EXPECT_EQ(mr.blendedNormals.size(), 80u);
  EXPECT_EQ(mr.blendedTangents.size(), 80u);
}

TEST(ComputeAnimation, MorphResultEnsureCpuOutputIdempotent)
{
  nvvkgltf::MorphResult mr{};
  mr.basePositions.resize(20, glm::vec3(1.0f));

  mr.ensureCpuOutput();
  ASSERT_EQ(mr.blendedPositions.size(), 20u);

  mr.blendedPositions[0] = glm::vec3(99.0f);
  mr.ensureCpuOutput();

  EXPECT_EQ(mr.blendedPositions.size(), 20u);
  EXPECT_EQ(mr.blendedPositions[0], glm::vec3(99.0f)) << "Second ensureCpuOutput should not reallocate";
}

TEST(ComputeAnimation, MorphResultEnsureCpuOutputSkipsAbsentAttributes)
{
  nvvkgltf::MorphResult mr{};
  mr.basePositions.resize(10, glm::vec3(0.0f));
  // baseNormals and baseTangents intentionally left empty

  mr.ensureCpuOutput();

  EXPECT_EQ(mr.blendedPositions.size(), 10u);
  EXPECT_TRUE(mr.blendedNormals.empty());
  EXPECT_TRUE(mr.blendedTangents.empty());
}

//==========================================================================
// Plain Model Tests
//
// Models without skin or morph should have no tasks after parse.
//==========================================================================

TEST(ComputeAnimation, NoSkinOrMorphOnAnimatedTriangle)
{
  nvvkgltf::Scene scene;
  if(!loadSampleModel(scene, "Models/AnimatedTriangle/glTF/AnimatedTriangle.gltf"))
    GTEST_SKIP() << "AnimatedTriangle not found";

  auto& anim = scene.animation();
  EXPECT_GT(anim.getNumAnimations(), 0) << "Should have rotation animation";
  EXPECT_FALSE(anim.hasSkinning()) << "AnimatedTriangle has no skin";
  EXPECT_FALSE(anim.hasMorphTargets()) << "AnimatedTriangle has no morph targets";
}

TEST(ComputeAnimation, NoSkinOrMorphOnAnimatedCube)
{
  nvvkgltf::Scene scene;
  if(!loadSampleModel(scene, "Models/AnimatedCube/glTF/AnimatedCube.gltf"))
    GTEST_SKIP() << "AnimatedCube not found";

  auto& anim = scene.animation();
  EXPECT_GT(anim.getNumAnimations(), 0);
  EXPECT_FALSE(anim.hasSkinning());
  EXPECT_FALSE(anim.hasMorphTargets());
}

//==========================================================================
// CPU Skinning Tests (SimpleSkin)
//
// SimpleSkin has 2 joints, 10 vertices, position-only (no normals/tangents).
// Animating the joint node and running computeSkinning should produce
// transformed positions that differ from the base geometry.
//==========================================================================

TEST(ComputeAnimation, CpuSkinningProducesTransformedPositions)
{
  nvvkgltf::Scene scene;
  if(!loadSampleModel(scene, "Models/SimpleSkin/glTF/SimpleSkin.gltf"))
    GTEST_SKIP() << "SimpleSkin not found";

  auto& anim = scene.animation();
  ASSERT_TRUE(anim.hasSkinning());
  ASSERT_GT(anim.getNumAnimations(), 0);

  // Advance animation to ~halfway (joint node rotates)
  auto& info       = anim.getAnimationInfo(0);
  info.currentTime = info.start + (info.end - info.start) * 0.5f;
  anim.updateAnimation(0);
  scene.updateNodeWorldMatrices();

  // Run CPU skinning
  anim.computeSkinning();

  const auto& skinTasks = anim.getSkinTasks();
  ASSERT_FALSE(skinTasks.empty());

  const auto& result = anim.getSkinningResult(0);
  EXPECT_FALSE(result.positions.empty()) << "computeSkinning should allocate and fill positions";

  // At least one vertex should have moved from the base position
  const auto& basePos = skinTasks[0].basePositions;
  ASSERT_EQ(result.positions.size(), basePos.size());

  bool anyMoved = false;
  for(size_t i = 0; i < basePos.size(); i++)
  {
    if(glm::any(glm::epsilonNotEqual(result.positions[i], basePos[i], 1e-6f)))
    {
      anyMoved = true;
      break;
    }
  }
  EXPECT_TRUE(anyMoved) << "At mid-animation, skinned vertices should differ from base positions";
}

TEST(ComputeAnimation, CpuSkinningOutputMatchesSizeOfBaseGeometry)
{
  nvvkgltf::Scene scene;
  if(!loadSampleModel(scene, "Models/SimpleSkin/glTF/SimpleSkin.gltf"))
    GTEST_SKIP() << "SimpleSkin not found";

  auto& anim = scene.animation();
  ASSERT_TRUE(anim.hasSkinning());

  auto& info       = anim.getAnimationInfo(0);
  info.currentTime = info.start;
  anim.updateAnimation(0);
  scene.updateNodeWorldMatrices();
  anim.computeSkinning();

  const auto& tasks = anim.getSkinTasks();
  for(size_t i = 0; i < tasks.size(); i++)
  {
    const auto& task   = tasks[i];
    const auto& result = anim.getSkinningResult(i);
    EXPECT_EQ(result.positions.size(), task.weights.size());

    // SimpleSkin has no normals or tangents in its mesh
    EXPECT_TRUE(result.normals.empty()) << "SimpleSkin has no normals; result should be empty";
    EXPECT_TRUE(result.tangents.empty()) << "SimpleSkin has no tangents; result should be empty";
  }
}

TEST(ComputeAnimation, CpuSkinningStableOverMultipleFrames)
{
  nvvkgltf::Scene scene;
  if(!loadSampleModel(scene, "Models/SimpleSkin/glTF/SimpleSkin.gltf"))
    GTEST_SKIP() << "SimpleSkin not found";

  auto& anim = scene.animation();
  ASSERT_TRUE(anim.hasSkinning());

  auto& info = anim.getAnimationInfo(0);
  for(int frame = 0; frame < 30; frame++)
  {
    info.incrementTime(1.0f / 30.0f, true);
    anim.updateAnimation(0);
    scene.updateNodeWorldMatrices();
    anim.computeSkinning();

    const auto& result = anim.getSkinningResult(0);
    ASSERT_FALSE(result.positions.empty()) << "Frame " << frame << ": positions should be filled";

    for(const auto& pos : result.positions)
    {
      ASSERT_FALSE(std::isnan(pos.x) || std::isnan(pos.y) || std::isnan(pos.z)) << "Frame " << frame << ": NaN in skinned position";
      ASSERT_FALSE(std::isinf(pos.x) || std::isinf(pos.y) || std::isinf(pos.z)) << "Frame " << frame << ": Inf in skinned position";
    }
  }
}

//==========================================================================
// CPU Morph Target Tests (SimpleMorph)
//
// SimpleMorph has 2 morph targets (position deltas only), 3 vertices.
// Setting non-zero weights and running computeMorphTargets should produce
// blended positions that differ from the base geometry.
//==========================================================================

TEST(ComputeAnimation, CpuMorphProducesBlendedPositions)
{
  nvvkgltf::Scene scene;
  if(!loadSampleModel(scene, "Models/SimpleMorph/glTF/SimpleMorph.gltf"))
    GTEST_SKIP() << "SimpleMorph not found";

  auto& anim = scene.animation();
  ASSERT_TRUE(anim.hasMorphTargets());
  ASSERT_GT(anim.getNumAnimations(), 0);

  // Advance animation to halfway so weights are non-trivial
  auto& info       = anim.getAnimationInfo(0);
  info.currentTime = info.start + (info.end - info.start) * 0.5f;
  anim.updateAnimation(0);
  scene.updateNodeWorldMatrices();

  anim.computeMorphTargets();

  const auto& morphPrims = anim.getMorphPrimitives();
  ASSERT_FALSE(morphPrims.empty());

  const auto& mr = anim.getMorphResult(0);
  EXPECT_FALSE(mr.blendedPositions.empty()) << "computeMorphTargets should allocate and fill blended positions";
  EXPECT_EQ(mr.blendedPositions.size(), mr.basePositions.size());

  // SimpleMorph has position-only targets; no normals or tangents
  EXPECT_TRUE(mr.blendedNormals.empty());
  EXPECT_TRUE(mr.blendedTangents.empty());
}

TEST(ComputeAnimation, CpuMorphBlendedPositionsDifferFromBase)
{
  nvvkgltf::Scene scene;
  if(!loadSampleModel(scene, "Models/SimpleMorph/glTF/SimpleMorph.gltf"))
    GTEST_SKIP() << "SimpleMorph not found";

  auto& anim = scene.animation();
  ASSERT_TRUE(anim.hasMorphTargets());

  // Set weights animation to mid-point
  auto& info       = anim.getAnimationInfo(0);
  info.currentTime = info.start + (info.end - info.start) * 0.25f;
  anim.updateAnimation(0);
  scene.updateNodeWorldMatrices();
  anim.computeMorphTargets();

  const auto& mr = anim.getMorphResult(0);
  ASSERT_FALSE(mr.blendedPositions.empty());

  bool anyDifferent = false;
  for(size_t i = 0; i < mr.basePositions.size(); i++)
  {
    if(glm::any(glm::epsilonNotEqual(mr.blendedPositions[i], mr.basePositions[i], 1e-6f)))
    {
      anyDifferent = true;
      break;
    }
  }
  EXPECT_TRUE(anyDifferent) << "Blended positions should differ from base when weights are active";
}

//==========================================================================
// CPU Morph Target Tests (AnimatedMorphCube)
//
// AnimatedMorphCube has 2 morph targets with POSITION + NORMAL + TANGENT
// deltas. This exercises the full morph blending path including normals
// and tangents.
//==========================================================================

TEST(ComputeAnimation, CpuMorphWithNormalsAndTangents)
{
  nvvkgltf::Scene scene;
  if(!loadSampleModel(scene, "Models/AnimatedMorphCube/glTF/AnimatedMorphCube.gltf"))
    GTEST_SKIP() << "AnimatedMorphCube not found";

  auto& anim = scene.animation();
  ASSERT_TRUE(anim.hasMorphTargets());
  ASSERT_GT(anim.getNumAnimations(), 0);

  // Advance to ~1/3 of animation
  auto& info       = anim.getAnimationInfo(0);
  info.currentTime = info.start + (info.end - info.start) * 0.33f;
  anim.updateAnimation(0);
  scene.updateNodeWorldMatrices();
  anim.computeMorphTargets();

  const auto& morphPrims = anim.getMorphPrimitives();
  ASSERT_FALSE(morphPrims.empty());

  const auto& mr = anim.getMorphResult(0);
  EXPECT_FALSE(mr.blendedPositions.empty()) << "Should have blended positions";
  EXPECT_EQ(mr.blendedPositions.size(), mr.basePositions.size());

  EXPECT_FALSE(mr.blendedNormals.empty()) << "AnimatedMorphCube has normal targets; blended normals should be filled";
  EXPECT_EQ(mr.blendedNormals.size(), mr.baseNormals.size());

  EXPECT_FALSE(mr.blendedTangents.empty()) << "AnimatedMorphCube has tangent targets; blended tangents should be filled";
  EXPECT_EQ(mr.blendedTangents.size(), mr.baseTangents.size());
}

TEST(ComputeAnimation, CpuMorphNormalsAreNormalized)
{
  nvvkgltf::Scene scene;
  if(!loadSampleModel(scene, "Models/AnimatedMorphCube/glTF/AnimatedMorphCube.gltf"))
    GTEST_SKIP() << "AnimatedMorphCube not found";

  auto& anim = scene.animation();
  ASSERT_TRUE(anim.hasMorphTargets());

  auto& info       = anim.getAnimationInfo(0);
  info.currentTime = info.start + (info.end - info.start) * 0.5f;
  anim.updateAnimation(0);
  scene.updateNodeWorldMatrices();
  anim.computeMorphTargets();

  const auto& mr = anim.getMorphResult(0);
  ASSERT_FALSE(mr.blendedNormals.empty());

  for(size_t i = 0; i < mr.blendedNormals.size(); i++)
  {
    float len = glm::length(mr.blendedNormals[i]);
    EXPECT_NEAR(len, 1.0f, 1e-4f) << "Blended normal at index " << i << " should be unit length, got " << len;
  }
}

TEST(ComputeAnimation, CpuMorphTangentWPreserved)
{
  nvvkgltf::Scene scene;
  if(!loadSampleModel(scene, "Models/AnimatedMorphCube/glTF/AnimatedMorphCube.gltf"))
    GTEST_SKIP() << "AnimatedMorphCube not found";

  auto& anim = scene.animation();
  ASSERT_TRUE(anim.hasMorphTargets());

  auto& info       = anim.getAnimationInfo(0);
  info.currentTime = info.start + (info.end - info.start) * 0.5f;
  anim.updateAnimation(0);
  scene.updateNodeWorldMatrices();
  anim.computeMorphTargets();

  const auto& mr = anim.getMorphResult(0);
  ASSERT_FALSE(mr.blendedTangents.empty());
  ASSERT_EQ(mr.blendedTangents.size(), mr.baseTangents.size());

  for(size_t i = 0; i < mr.blendedTangents.size(); i++)
  {
    EXPECT_FLOAT_EQ(mr.blendedTangents[i].w, mr.baseTangents[i].w) << "Tangent .w (handedness) should be preserved at index " << i;
  }
}

TEST(ComputeAnimation, CpuMorphStableOverMultipleFrames)
{
  nvvkgltf::Scene scene;
  if(!loadSampleModel(scene, "Models/AnimatedMorphCube/glTF/AnimatedMorphCube.gltf"))
    GTEST_SKIP() << "AnimatedMorphCube not found";

  auto& anim = scene.animation();
  ASSERT_TRUE(anim.hasMorphTargets());

  auto& info = anim.getAnimationInfo(0);
  for(int frame = 0; frame < 30; frame++)
  {
    info.incrementTime(1.0f / 30.0f, true);
    anim.updateAnimation(0);
    scene.updateNodeWorldMatrices();
    anim.computeMorphTargets();

    const auto& mr = anim.getMorphResult(0);
    ASSERT_FALSE(mr.blendedPositions.empty()) << "Frame " << frame;

    for(const auto& pos : mr.blendedPositions)
    {
      ASSERT_FALSE(std::isnan(pos.x) || std::isnan(pos.y) || std::isnan(pos.z)) << "Frame " << frame << ": NaN in morphed position";
    }
  }
}

//==========================================================================
// Integration: computeSkinning then computeMorphTargets on separate models
//
// Verify that both CPU paths can run independently in the same process
// without interference.
//==========================================================================

TEST(ComputeAnimation, BothCpuPathsRunIndependently)
{
  nvvkgltf::Scene skinScene;
  nvvkgltf::Scene morphScene;

  bool hasSkin  = loadSampleModel(skinScene, "Models/SimpleSkin/glTF/SimpleSkin.gltf");
  bool hasMorph = loadSampleModel(morphScene, "Models/SimpleMorph/glTF/SimpleMorph.gltf");

  if(!hasSkin && !hasMorph)
    GTEST_SKIP() << "Neither SimpleSkin nor SimpleMorph found";

  if(hasSkin)
  {
    auto& anim       = skinScene.animation();
    auto& info       = anim.getAnimationInfo(0);
    info.currentTime = info.start + (info.end - info.start) * 0.3f;
    anim.updateAnimation(0);
    skinScene.updateNodeWorldMatrices();
    anim.computeSkinning();

    EXPECT_FALSE(anim.getSkinningResult(0).positions.empty());
  }

  if(hasMorph)
  {
    auto& anim       = morphScene.animation();
    auto& info       = anim.getAnimationInfo(0);
    info.currentTime = info.start + (info.end - info.start) * 0.3f;
    anim.updateAnimation(0);
    morphScene.updateNodeWorldMatrices();
    anim.computeMorphTargets();

    EXPECT_FALSE(anim.getMorphResult(0).blendedPositions.empty());
  }
}
