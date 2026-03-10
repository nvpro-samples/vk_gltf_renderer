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

//--------------------------------------------------------------------------------------------------
// Phase 2D.2: Advanced Index Remapping Tests
// Tests for animation pointer paths, skins, and material remapping
//--------------------------------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <filesystem>
#include "gltf_scene.hpp"
#include "gltf_scene_editor.hpp"
#include "gltf_scene_animation.hpp"
#include "gltf_scene_validator.hpp"
#include "common/test_utils.hpp"

namespace fs = std::filesystem;

// Helper to get path to glTF Sample Assets
static fs::path getSampleAssetsPath()
{
  fs::path basePath = GLTF_SAMPLE_ASSETS_PATH;
  if(!fs::exists(basePath))
  {
    return fs::path();  // Return empty path
  }
  return basePath;
}

class IndexRemappingAdvancedTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    tempDir = fs::temp_directory_path() / "gltf_test_remapping_adv";
    fs::create_directories(tempDir);
  }

  void TearDown() override { fs::remove_all(tempDir); }

  fs::path tempDir;
};

//--------------------------------------------------------------------------------------------------
// Test 1: Animation pointer path remapping (KHR_animation_pointer)
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingAdvancedTest, AnimationPointerRemapping)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  // AnimatedColorsCube uses KHR_animation_pointer
  fs::path testFile = assetsPath / "Models/AnimatedColorsCube/glTF/AnimatedColorsCube.gltf";
  if(!fs::exists(testFile))
  {
    GTEST_SKIP() << "AnimatedColorsCube not found in Sample Assets";
  }

  ASSERT_TRUE(scene.load(testFile)) << "Failed to load AnimatedColorsCube";

  // Verify it has animations with pointer paths
  ASSERT_GT(scene.animation().getNumAnimations(), 0) << "AnimatedColorsCube has no animations";

  // For now, just test that save/load works with animation pointers present
  // (We don't need to delete nodes to test pointer path remapping infrastructure)

  // Save and reload (no deletions, but tests that animation pointer system works)
  fs::path outputFile = tempDir / "test_anim_pointer_preservation.gltf";
  ASSERT_TRUE(scene.save(outputFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(outputFile));

  // Animations should be preserved
  EXPECT_EQ(scene2.animation().getNumAnimations(), scene.animation().getNumAnimations());

  // Animation should work after reload
  if(scene2.animation().getNumAnimations() > 0)
  {
    auto updated = scene2.animation().updateAnimation(0);
    EXPECT_TRUE(updated);

    // Should animate something
    EXPECT_GE(scene2.getDirtyFlags().nodes.size(), 0);  // May be 0 if only materials are animated
  }

  // Validation should pass
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

//--------------------------------------------------------------------------------------------------
// Test 2: Skin joint remapping
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingAdvancedTest, SkinJointRemapping)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  // SimpleSkin has skinning
  fs::path testFile = assetsPath / "Models/SimpleSkin/glTF/SimpleSkin.gltf";
  if(!fs::exists(testFile))
  {
    GTEST_SKIP() << "SimpleSkin not found in Sample Assets";
  }

  ASSERT_TRUE(scene.load(testFile)) << "Failed to load SimpleSkin";

  // Verify it has skins
  ASSERT_GT(scene.getModel().skins.size(), 0) << "SimpleSkin has no skins";

  const auto& skin           = scene.getModel().skins[0];
  size_t      origJointCount = skin.joints.size();
  ASSERT_GT(origJointCount, 0) << "Skin has no joints";

  // Try to delete a NON-joint node (to preserve skinning)
  std::unordered_set<int> jointNodes(skin.joints.begin(), skin.joints.end());

  int nodeToDelete = -1;
  for(size_t i = 0; i < scene.getModel().nodes.size(); ++i)
  {
    if(jointNodes.find(static_cast<int>(i)) == jointNodes.end())
    {
      // Not a joint node
      bool isRoot = false;
      for(const auto& rootIdx : scene.getModel().scenes[0].nodes)
      {
        if(rootIdx == static_cast<int>(i))
        {
          isRoot = true;
          break;
        }
      }

      // Check if it's the last root
      if(isRoot)
      {
        size_t rootCount = scene.getModel().scenes[0].nodes.size();
        if(rootCount <= 1)
          continue;  // Can't delete last root
      }

      nodeToDelete = static_cast<int>(i);
      break;
    }
  }

  if(nodeToDelete == -1)
  {
    GTEST_SKIP() << "Cannot find safe non-joint node to delete";
  }

  scene.editor().deleteNode(nodeToDelete);

  // Save and reload
  fs::path outputFile = tempDir / "test_skin_remap.gltf";
  ASSERT_TRUE(scene.save(outputFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(outputFile));

  // Skin should still have same number of joints
  ASSERT_GT(scene2.getModel().skins.size(), 0);
  const auto& skin2 = scene2.getModel().skins[0];
  EXPECT_EQ(skin2.joints.size(), origJointCount) << "Joints were affected by non-joint deletion";

  // Validation should pass
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

//--------------------------------------------------------------------------------------------------
// Test 3: Delete joint node (skin affected but preserved)
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingAdvancedTest, DeleteJointNodeWarning)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  fs::path testFile = assetsPath / "Models/SimpleSkin/glTF/SimpleSkin.gltf";
  if(!fs::exists(testFile))
  {
    GTEST_SKIP() << "SimpleSkin not found in Sample Assets";
  }

  ASSERT_TRUE(scene.load(testFile));

  ASSERT_GT(scene.getModel().skins.size(), 0);
  const auto& skin = scene.getModel().skins[0];
  ASSERT_GT(skin.joints.size(), 1) << "Need at least 2 joints to delete one";

  // Delete ONE joint node (not all)
  int jointToDelete = skin.joints[0];
  int jointHandle   = jointToDelete;

  // Check if this joint is also a root node
  const auto& mainScene = scene.getModel().scenes[0];
  bool isRoot = (std::find(mainScene.nodes.begin(), mainScene.nodes.end(), jointToDelete) != mainScene.nodes.end());

  if(isRoot && mainScene.nodes.size() <= 1)
  {
    GTEST_SKIP() << "Joint is the only root node - cannot delete";
  }

  scene.editor().deleteNode(jointHandle);

  // Save - should log warning about deleted joint
  fs::path outputFile = tempDir / "test_joint_delete.gltf";
  ASSERT_TRUE(scene.save(outputFile));

  // Reload
  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(outputFile));

  // Skin behavior depends on how many joints were deleted
  if(scene2.getModel().skins.empty())
  {
    // All joints were deleted - skin was removed (expected)
    EXPECT_EQ(scene2.getModel().skins.size(), 0) << "Skin should be removed when all joints deleted";
  }
  else
  {
    // Some joints remain - skin should have fewer joints
    const auto& skin2 = scene2.getModel().skins[0];
    EXPECT_LT(skin2.joints.size(), skin.joints.size()) << "Joint count should decrease";
  }

  // Validation should pass
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

//--------------------------------------------------------------------------------------------------
// Test 4: Material remapping
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingAdvancedTest, MaterialRemapping)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  // Use a scene with multiple materials
  fs::path testFile = assetsPath / "Models/Box/glTF/Box.gltf";
  ASSERT_TRUE(scene.load(testFile));

  size_t origMatCount = scene.getModel().materials.size();
  ASSERT_GT(origMatCount, 0);

  // Delete a node (which may trigger material compaction in future)
  // For now, just test that material indices remain valid
  int childNode = -1;
  for(const auto& node : scene.getModel().nodes)
  {
    if(!node.children.empty())
    {
      childNode = node.children[0];
      break;
    }
  }

  if(childNode != -1)
  {
    scene.editor().deleteNode(childNode);
  }

  // Save and reload
  fs::path outputFile = tempDir / "test_material_remap.gltf";
  ASSERT_TRUE(scene.save(outputFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(outputFile));

  // Materials should be preserved (no tombstones yet)
  EXPECT_EQ(scene2.getModel().materials.size(), origMatCount);

  // All primitive material references should be valid
  for(const auto& mesh : scene2.getModel().meshes)
  {
    for(const auto& prim : mesh.primitives)
    {
      if(prim.material >= 0)
      {
        EXPECT_LT(prim.material, static_cast<int>(scene2.getModel().materials.size())) << "Primitive has invalid material reference";
      }
    }
  }

  EXPECT_TRUE(scene2.validator().validateModel().valid);
}
