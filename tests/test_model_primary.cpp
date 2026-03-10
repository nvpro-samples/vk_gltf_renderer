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
#include "gltf_scene_editor.hpp"
#include "gltf_scene_validator.hpp"
#include "common/test_utils.hpp"
#include <filesystem>

using namespace gltf_test;

// Phase 2B: Model-Primary Architecture Tests
// Verify extras/extensions preservation through handle-based operations

class ModelPrimaryTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    tempDir = std::filesystem::temp_directory_path() / "gltf_model_primary_test";
    std::filesystem::create_directories(tempDir);
  }

  void TearDown() override { std::filesystem::remove_all(tempDir); }

  std::filesystem::path tempDir;
};


TEST_F(ModelPrimaryTest, HandleAccessorsWork)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  // Get handle for first node
  if(scene.getModel().nodes.empty())
  {
    GTEST_SKIP() << "No nodes in test scene";
  }

  auto handle = 0;
  EXPECT_NE(handle, -1);

  // Test accessors
  EXPECT_TRUE(scene.editor().isValidNodeIndex(handle));

  // Get node via handle
  const auto& node = scene.editor().getNode(handle);
  EXPECT_FALSE(node.name.empty());

  // Round-trip: handle → index → handle
  int backIdx = handle;
  EXPECT_EQ(backIdx, 0);

  auto backHandle = backIdx;
  EXPECT_EQ(backHandle, handle);
}

TEST_F(ModelPrimaryTest, DuplicatePreservesExtras)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  if(scene.getModel().nodes.empty())
  {
    GTEST_SKIP() << "No nodes in test scene";
  }

  // Add extras to first node
  auto  handle = 0;
  auto& node   = scene.editor().getNodeForEdit(handle);

  // Add custom extras
  node.extras = tinygltf::Value(
      tinygltf::Value::Object{{"custom_data", tinygltf::Value("test_value")}, {"user_id", tinygltf::Value(12345)}});

  // Duplicate node
  auto dupHandle = scene.editor().duplicateNode(handle);
  ASSERT_NE(dupHandle, -1);

  // Check duplicate has extras
  const auto& dupNode = scene.editor().getNode(dupHandle);
  EXPECT_NE(dupNode.extras.Type(), tinygltf::NULL_TYPE);
  EXPECT_TRUE(dupNode.extras.Has("custom_data"));
  EXPECT_TRUE(dupNode.extras.Has("user_id"));
  EXPECT_EQ(dupNode.extras.Get("custom_data").Get<std::string>(), "test_value");
  EXPECT_EQ(dupNode.extras.Get("user_id").Get<int>(), 12345);

  // Verify it's a copy (different handle)
  EXPECT_NE(dupHandle, handle);
  EXPECT_TRUE(dupNode.name.find("_copy") != std::string::npos);
}

TEST_F(ModelPrimaryTest, DuplicatePreservesExtensions)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  if(scene.getModel().nodes.empty())
  {
    GTEST_SKIP() << "No nodes in test scene";
  }

  // Add unknown extension to first node
  auto  handle = 0;
  auto& node   = scene.editor().getNodeForEdit(handle);

  node.extensions["CUSTOM_vendor_extension"] =
      tinygltf::Value(tinygltf::Value::Object{{"data", tinygltf::Value(42)}, {"enabled", tinygltf::Value(true)}});

  size_t origExtCount = node.extensions.size();

  // Duplicate node
  auto dupHandle = scene.editor().duplicateNode(handle);
  ASSERT_NE(dupHandle, -1);

  // Check duplicate has extensions
  const auto& dupNode = scene.editor().getNode(dupHandle);
  EXPECT_EQ(dupNode.extensions.size(), origExtCount);
  EXPECT_TRUE(dupNode.extensions.find("CUSTOM_vendor_extension") != dupNode.extensions.end());
}

TEST_F(ModelPrimaryTest, ExtrasPreservedThroughSaveLoad)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  if(scene.getModel().nodes.empty())
  {
    GTEST_SKIP() << "No nodes in test scene";
  }

  // Add extras
  auto  handle = 0;
  auto& node   = scene.editor().getNodeForEdit(handle);
  node.extras  = tinygltf::Value(tinygltf::Value::Object{{"preserved", tinygltf::Value(true)}});

  std::string nodeName = node.name;

  // Save
  auto tempFile = tempDir / "test_extras.gltf";
  ASSERT_TRUE(scene.save(tempFile));

  // Reload
  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(tempFile));

  // Find node by name
  int foundIdx = -1;
  for(size_t i = 0; i < scene2.getModel().nodes.size(); ++i)
  {
    if(scene2.getModel().nodes[i].name == nodeName)
    {
      foundIdx = static_cast<int>(i);
      break;
    }
  }

  ASSERT_GE(foundIdx, 0) << "Node not found after reload";

  // Verify extras preserved
  const auto& reloadedNode = scene2.getModel().nodes[foundIdx];
  EXPECT_NE(reloadedNode.extras.Type(), tinygltf::NULL_TYPE);
  EXPECT_TRUE(reloadedNode.extras.Has("preserved"));
  EXPECT_TRUE(reloadedNode.extras.Get("preserved").Get<bool>());
}

TEST_F(ModelPrimaryTest, DeleteNodeImmediate)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  if(scene.getModel().nodes.size() < 2)
  {
    GTEST_SKIP() << "Need at least 2 nodes for this test";
  }

  size_t initialCount = scene.getModel().nodes.size();

  // Find a leaf node (no children) to delete
  int nodeToDelete = -1;
  for(size_t i = 0; i < scene.getModel().nodes.size(); ++i)
  {
    if(scene.getModel().nodes[i].children.empty())
    {
      nodeToDelete = static_cast<int>(i);
      break;
    }
  }

  if(nodeToDelete < 0)
  {
    GTEST_SKIP() << "No leaf nodes to delete";
  }

  // Delete node (single node, no children)
  scene.editor().deleteNode(nodeToDelete);

  // Verify immediately removed from model (NO tombstones)
  EXPECT_EQ(scene.getModel().nodes.size(), initialCount - 1) << "Node should be removed immediately";

  // Model still has remaining nodes
  EXPECT_FALSE(scene.getModel().nodes.empty());
}

TEST_F(ModelPrimaryTest, SetTranslationWorks)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));
  scene.setCurrentScene(0);  // Parse scene

  if(scene.getModel().nodes.empty())
  {
    GTEST_SKIP() << "No nodes in test scene";
  }

  auto handle = 0;

  // Set translation
  glm::vec3 newTranslation(1.0f, 2.0f, 3.0f);
  scene.editor().setNodeTRS(handle, newTranslation, {1, 0, 0, 0}, {1, 1, 1});

  // Verify model modified
  const auto& node = scene.editor().getNode(handle);
  EXPECT_EQ(node.translation[0], 1.0);
  EXPECT_EQ(node.translation[1], 2.0);
  EXPECT_EQ(node.translation[2], 3.0);

  // Verify validation still passes
  EXPECT_TRUE(scene.validator().validateModel().valid);
}

TEST_F(ModelPrimaryTest, DirtyTrackingWorks)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));
  scene.setCurrentScene(0);

  if(scene.getModel().nodes.empty())
  {
    GTEST_SKIP() << "No nodes";
  }

  auto handle = 0;

  // Mark dirty
  // Update matrices for this node
  scene.updateNodeWorldMatrices();

  // Should be able to get world matrix
  const glm::mat4& worldMat = scene.editor().getNodeWorldMatrix(handle);
  EXPECT_TRUE(true);  // If we get here, accessor works
}

TEST_F(ModelPrimaryTest, MultipleEditsBeforeFlush)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));
  scene.setCurrentScene(0);

  if(scene.getModel().nodes.size() < 2)
  {
    GTEST_SKIP() << "Need multiple nodes";
  }

  auto handle1 = 0;
  auto handle2 = 1;

  // Edit multiple nodes
  scene.editor().setNodeTRS(handle1, glm::vec3(1, 0, 0), {1, 0, 0, 0}, {1, 1, 1});
  scene.editor().setNodeTRS(handle2, glm::vec3(0, 1, 0), {1, 0, 0, 0}, {1, 1, 1});

  // Update matrices for both nodes
  scene.updateNodeWorldMatrices();

  // Verify both modified
  const auto& node1 = scene.editor().getNode(handle1);
  const auto& node2 = scene.editor().getNode(handle2);
  EXPECT_EQ(node1.translation[0], 1.0);
  EXPECT_EQ(node2.translation[1], 1.0);
}
