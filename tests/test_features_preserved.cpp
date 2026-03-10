/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include "gltf_scene.hpp"
#include "gltf_scene_animation.hpp"
#include "gltf_scene_validator.hpp"
#include "common/test_utils.hpp"
#include <filesystem>

using namespace gltf_test;

// Phase 2A: Feature Preservation Tests
// Test that all features work after save/load

class FeaturePreservationTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    tempDir = std::filesystem::temp_directory_path() / "gltf_feature_test";
    std::filesystem::create_directories(tempDir);
  }

  void TearDown() override { std::filesystem::remove_all(tempDir); }

  std::filesystem::path tempDir;
};

TEST_F(FeaturePreservationTest, AnimationsPreserved)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  if(!scene.load(resourcePath))
  {
    GTEST_SKIP() << "Failed to load test resource";
  }

  // Get initial animation count
  int numAnimations = scene.animation().getNumAnimations();

  // Save and reload
  auto tempFile = tempDir / "test_anim.gltf";
  ASSERT_TRUE(scene.save(tempFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(tempFile));

  // Verify animations preserved
  EXPECT_EQ(scene2.animation().getNumAnimations(), numAnimations);

  // If there are animations, verify they're still valid
  if(numAnimations > 0)
  {
    EXPECT_TRUE(scene2.animation().hasAnimation());
    EXPECT_TRUE(scene2.validator().validateModel().valid);
  }
}

TEST_F(FeaturePreservationTest, MaterialsPreserved)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  if(!scene.load(resourcePath))
  {
    GTEST_SKIP() << "Failed to load test resource";
  }

  // Get initial material count
  size_t numMaterials = scene.getModel().materials.size();

  // Save and reload
  auto tempFile = tempDir / "test_materials.gltf";
  ASSERT_TRUE(scene.save(tempFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(tempFile));

  // Verify materials preserved
  EXPECT_EQ(scene2.getModel().materials.size(), numMaterials);
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

TEST_F(FeaturePreservationTest, VariantsPreserved)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  if(!scene.load(resourcePath))
  {
    GTEST_SKIP() << "Failed to load test resource";
  }

  // Get initial variants
  auto   variants    = scene.getVariants();
  size_t numVariants = variants.size();

  // Save and reload
  auto tempFile = tempDir / "test_variants.gltf";
  ASSERT_TRUE(scene.save(tempFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(tempFile));

  // Verify variants preserved
  EXPECT_EQ(scene2.getVariants().size(), numVariants);
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

TEST_F(FeaturePreservationTest, LightsPreserved)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  if(!scene.load(resourcePath))
  {
    GTEST_SKIP() << "Failed to load test resource";
  }

  // Parse the scene to get lights
  scene.setCurrentScene(0);
  auto   lights    = scene.getRenderLights();
  size_t numLights = lights.size();

  // Save and reload
  auto tempFile = tempDir / "test_lights.gltf";
  ASSERT_TRUE(scene.save(tempFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(tempFile));
  scene2.setCurrentScene(0);

  // Verify lights preserved
  EXPECT_EQ(scene2.getRenderLights().size(), numLights);
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

TEST_F(FeaturePreservationTest, CamerasPreserved)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  if(!scene.load(resourcePath))
  {
    GTEST_SKIP() << "Failed to load test resource";
  }

  // Get camera count from model
  size_t numCamerasModel = scene.getModel().cameras.size();

  // Save and reload
  auto tempFile = tempDir / "test_cameras.gltf";
  ASSERT_TRUE(scene.save(tempFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(tempFile));

  // Verify cameras preserved
  EXPECT_EQ(scene2.getModel().cameras.size(), numCamerasModel);
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

TEST_F(FeaturePreservationTest, RenderNodesPreserved)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  if(!scene.load(resourcePath))
  {
    GTEST_SKIP() << "Failed to load test resource";
  }

  // Parse scene to create render nodes
  scene.setCurrentScene(0);
  size_t numRenderNodes = scene.getRenderNodes().size();

  // Save and reload
  auto tempFile = tempDir / "test_render_nodes.gltf";
  ASSERT_TRUE(scene.save(tempFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(tempFile));
  scene2.setCurrentScene(0);

  // Verify render nodes preserved (should be recreated identically)
  EXPECT_EQ(scene2.getRenderNodes().size(), numRenderNodes);
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

TEST_F(FeaturePreservationTest, NodeHierarchyPreserved)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  if(!scene.load(resourcePath))
  {
    GTEST_SKIP() << "Failed to load test resource";
  }

  // Get node hierarchy info
  size_t              numNodes = scene.getModel().nodes.size();
  std::vector<size_t> childCounts;
  for(const auto& node : scene.getModel().nodes)
  {
    childCounts.push_back(node.children.size());
  }

  // Save and reload
  auto tempFile = tempDir / "test_hierarchy.gltf";
  ASSERT_TRUE(scene.save(tempFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(tempFile));

  // Verify hierarchy preserved
  EXPECT_EQ(scene2.getModel().nodes.size(), numNodes);

  for(size_t i = 0; i < std::min(numNodes, scene2.getModel().nodes.size()); ++i)
  {
    EXPECT_EQ(scene2.getModel().nodes[i].children.size(), childCounts[i]);
  }

  EXPECT_TRUE(scene2.validator().validateModel().valid);
}
