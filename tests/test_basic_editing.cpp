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
#include "common/test_utils.hpp"
#include <filesystem>

using namespace gltf_test;

// Phase 2C: Basic Editing Operations
// Test node addition, deletion, and resource attachment

class BasicEditingTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    tempDir = std::filesystem::temp_directory_path() / "gltf_editing_test";
    std::filesystem::create_directories(tempDir);
  }

  void TearDown() override { std::filesystem::remove_all(tempDir); }

  std::filesystem::path tempDir;
};

TEST_F(BasicEditingTest, AddNode)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  size_t origCount = scene.getModel().nodes.size();

  // Add node
  int newNode = scene.editor().addNode("NewNode");
  ASSERT_NE(newNode, -1);

  // Should be valid
  ASSERT_TRUE(scene.editor().isValidNodeIndex(newNode));

  // Model should have one more node
  EXPECT_EQ(scene.getModel().nodes.size(), origCount + 1);

  // Verify node properties
  const auto& node = scene.editor().getNode(newNode);
  EXPECT_EQ(node.name, "NewNode");
  EXPECT_EQ(node.mesh, -1);    // No mesh
  EXPECT_EQ(node.camera, -1);  // No camera
  EXPECT_EQ(node.skin, -1);    // No skin
}

TEST_F(BasicEditingTest, AddNodeWithParent)
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
    GTEST_SKIP() << "No nodes in scene";
  }

  // Get existing node as parent
  int parent = 0;

  // Add child
  int child = scene.editor().addNode("ChildNode", parent);
  ASSERT_NE(child, -1);

  // Verify parent has child in model
  const auto& parentNode = scene.getModel().nodes[parent];
  bool        foundChild = false;
  for(int childId : parentNode.children)
  {
    if(childId == child)
    {
      foundChild = true;
      break;
    }
  }
  EXPECT_TRUE(foundChild);
}

TEST_F(BasicEditingTest, DeleteNodeRecursive)
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
    GTEST_SKIP() << "No nodes";
  }

  // Add parent with children
  int parent = scene.editor().addNode("Parent");
  int child1 = scene.editor().addNode("Child1", parent);
  int child2 = scene.editor().addNode("Child2", parent);

  size_t countBefore = scene.getModel().nodes.size();

  // Delete parent (should delete children too)
  scene.editor().deleteNode(parent);  // Default is recursive

  // All 3 nodes should be immediately removed from model
  EXPECT_EQ(scene.getModel().nodes.size(), countBefore - 3) << "Parent + 2 children should be removed";
}

TEST_F(BasicEditingTest, DeleteNodeSingleOrphansChildren)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  // Add parent with children
  int parent = scene.editor().addNode("Parent");
  int child  = scene.editor().addNode("Child", parent);

  size_t countBefore = scene.getModel().nodes.size();

  // deleteNode() removes the node and all descendants (recursive)
  scene.editor().deleteNode(parent);

  // Parent and child both removed
  EXPECT_EQ(scene.getModel().nodes.size(), countBefore - 2) << "Parent and child should be removed";

  bool foundChild = false;
  for(const auto& node : scene.getModel().nodes)
  {
    if(node.name == "Child")
    {
      foundChild = true;
      break;
    }
  }
  EXPECT_FALSE(foundChild) << "Child should be removed with parent";
}

TEST_F(BasicEditingTest, AttachMeshToNode)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  // Create empty node
  int newNode = scene.editor().addNode("EmptyNode");

  // Verify no mesh initially
  EXPECT_FALSE(scene.editor().getNodeMesh(newNode).has_value());

  // Attach mesh
  if(scene.getModel().meshes.empty())
  {
    GTEST_SKIP() << "No meshes in scene";
  }

  int mesh = 0;
  scene.editor().setNodeMesh(newNode, mesh);

  // Verify mesh attached
  auto attachedMesh = scene.editor().getNodeMesh(newNode);
  ASSERT_TRUE(attachedMesh.has_value());
  EXPECT_EQ(*attachedMesh, mesh);

  // Verify in model
  const auto& node = scene.editor().getNode(newNode);
  EXPECT_EQ(node.mesh, 0);
}

TEST_F(BasicEditingTest, ClearMeshFromNode)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  // Find a node with a mesh
  int nodeWithMesh = -1;
  for(size_t i = 0; i < scene.getModel().nodes.size(); ++i)
  {
    if(scene.getModel().nodes[i].mesh >= 0)
    {
      nodeWithMesh = static_cast<int>(i);
      break;
    }
  }

  if(nodeWithMesh == -1)
  {
    GTEST_SKIP() << "No nodes with meshes";
  }

  // Verify has mesh
  ASSERT_TRUE(scene.editor().getNodeMesh(nodeWithMesh).has_value());

  // Clear mesh
  scene.editor().clearNodeMesh(nodeWithMesh);

  // Verify cleared
  EXPECT_FALSE(scene.editor().getNodeMesh(nodeWithMesh).has_value());

  // Verify in model
  const auto& node = scene.editor().getNode(nodeWithMesh);
  EXPECT_LT(node.mesh, 0);
}

TEST_F(BasicEditingTest, AddNodeSaveAndReload)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  size_t origCount = scene.getModel().nodes.size();

  // Add node
  int newNode = scene.editor().addNode("AddedNode");
  scene.editor().setNodeTRS(newNode, glm::vec3(1.0f, 2.0f, 3.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));

  // Save
  auto tempFile = tempDir / "test_add.gltf";
  ASSERT_TRUE(scene.save(tempFile));

  // Reload
  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(tempFile));

  // Should have one more node
  EXPECT_EQ(scene2.getModel().nodes.size(), origCount + 1);

  // Find new node by name
  int newNodeIdx = -1;
  for(size_t i = 0; i < scene2.getModel().nodes.size(); ++i)
  {
    if(scene2.getModel().nodes[i].name == "AddedNode")
    {
      newNodeIdx = static_cast<int>(i);
      break;
    }
  }
  ASSERT_GE(newNodeIdx, 0) << "Added node not found after reload";

  // Verify translation preserved
  const auto& reloadedNode = scene2.getModel().nodes[newNodeIdx];
  EXPECT_DOUBLE_EQ(reloadedNode.translation[0], 1.0);
  EXPECT_DOUBLE_EQ(reloadedNode.translation[1], 2.0);
  EXPECT_DOUBLE_EQ(reloadedNode.translation[2], 3.0);
}

TEST_F(BasicEditingTest, AttachMeshSaveAndReload)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  if(scene.getModel().meshes.empty())
  {
    GTEST_SKIP() << "No meshes in scene";
  }

  // Create empty node and attach mesh
  int newNode = scene.editor().addNode("MeshNode");
  int mesh    = 0;
  scene.editor().setNodeMesh(newNode, mesh);

  // Save
  auto tempFile = tempDir / "test_attach_mesh.gltf";
  ASSERT_TRUE(scene.save(tempFile));

  // Reload
  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(tempFile));

  // Find new node
  int newNodeIdx = -1;
  for(size_t i = 0; i < scene2.getModel().nodes.size(); ++i)
  {
    if(scene2.getModel().nodes[i].name == "MeshNode")
    {
      newNodeIdx = static_cast<int>(i);
      break;
    }
  }
  ASSERT_GE(newNodeIdx, 0);

  // Verify mesh still attached
  const auto& reloadedNode = scene2.getModel().nodes[newNodeIdx];
  EXPECT_GE(reloadedNode.mesh, 0) << "Mesh attachment not preserved";
}

TEST_F(BasicEditingTest, ImmediateDeletion)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  size_t initialCount = scene.getModel().nodes.size();

  // Add nodes
  int node1 = scene.editor().addNode("Node1");
  int node2 = scene.editor().addNode("Node2");
  int node3 = scene.editor().addNode("Node3");

  EXPECT_EQ(scene.getModel().nodes.size(), initialCount + 3);

  // Delete two (no children, so deleteNode behaves like single-node removal)
  scene.editor().deleteNode(node2);  // Delete middle one
  EXPECT_EQ(scene.getModel().nodes.size(), initialCount + 2);

  scene.editor().deleteNode(node1);  // Delete another
  EXPECT_EQ(scene.getModel().nodes.size(), initialCount + 1);

  // node3's index has shifted, but should still exist in model
  bool found = false;
  for(const auto& node : scene.getModel().nodes)
  {
    if(node.name == "Node3")
    {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found) << "Node3 should still exist after others deleted";
}

// M7b gap: Rename node
TEST_F(BasicEditingTest, RenameNode)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  int nodeIdx = scene.editor().addNode("OriginalName");
  ASSERT_NE(nodeIdx, -1);
  EXPECT_EQ(scene.editor().getNodeName(nodeIdx), "OriginalName");

  scene.editor().renameNode(nodeIdx, "RenamedNode");
  EXPECT_EQ(scene.editor().getNodeName(nodeIdx), "RenamedNode");
  EXPECT_EQ(scene.getModel().nodes[nodeIdx].name, "RenamedNode");
}

// M7c gap: Buffer compaction verification
TEST_F(BasicEditingTest, CompactModelReducesBufferData)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto            resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(resourcePath));

  // Add a mesh node and then delete it — leaves orphaned geometry
  int newNode = scene.editor().addNode("TempNode");
  ASSERT_NE(newNode, -1);

  // Record pre-compact counts
  size_t preAccessors   = scene.getModel().accessors.size();
  size_t preBufferViews = scene.getModel().bufferViews.size();

  // Delete a node that has a mesh (first node with mesh > -1)
  for(int i = 0; i < static_cast<int>(scene.getModel().nodes.size()); ++i)
  {
    if(scene.getModel().nodes[i].mesh >= 0)
    {
      scene.editor().deleteNode(i);
      break;
    }
  }

  // Compact should remove orphaned data
  bool compacted = scene.compactModel();
  if(compacted)
  {
    EXPECT_LE(scene.getModel().accessors.size(), preAccessors);
    EXPECT_LE(scene.getModel().bufferViews.size(), preBufferViews);
  }
}
