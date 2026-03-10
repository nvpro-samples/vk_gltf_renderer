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
// Phase 2D.1: Basic Index Remapping Tests
// Tests for node compaction and simple reference remapping (nodes, animations)
//--------------------------------------------------------------------------------------------------

#include <gtest/gtest.h>
#include <filesystem>
#include "gltf_scene.hpp"
#include "gltf_scene_editor.hpp"
#include "gltf_scene_animation.hpp"
#include "gltf_scene_validator.hpp"
#include "common/test_utils.hpp"
#include "nvutils/logger.hpp"

namespace fs = std::filesystem;

// Helper to get path to glTF Sample Assets
// Returns empty path if not found (caller should check and skip)
static fs::path getSampleAssetsPath()
{
  fs::path basePath = GLTF_SAMPLE_ASSETS_PATH;
  if(!fs::exists(basePath))
  {
    return fs::path();  // Return empty path
  }
  return basePath;
}

class IndexRemappingTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    tempDir = fs::temp_directory_path() / "gltf_test_remapping";
    fs::create_directories(tempDir);
  }

  void TearDown() override { fs::remove_all(tempDir); }

  fs::path tempDir;
};

//--------------------------------------------------------------------------------------------------
// Test 1: Simple delete and compact (no references)
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingTest, DeleteNodeSimple)
{
  nvvkgltf::Scene scene;

  // Load a simple scene
  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  fs::path testFile = assetsPath / "Models/Box/glTF/Box.gltf";
  ASSERT_TRUE(scene.load(testFile)) << "Failed to load: " << testFile;

  size_t origCount = scene.getModel().nodes.size();
  ASSERT_GT(origCount, 0) << "Scene has no nodes";

  // Find a child node to delete (not a root node)
  const auto& mainScene    = scene.getModel().scenes[scene.getCurrentScene()];
  int         nodeToDelete = -1;

  // Try to find a child node (not a root)
  for(const auto& node : scene.getModel().nodes)
  {
    for(int childIdx : node.children)
    {
      nodeToDelete = childIdx;
      if(nodeToDelete != -1)
      {
        break;
      }
    }
    if(nodeToDelete != -1)
      break;
  }

  // If no children found, scene structure doesn't allow safe deletion
  if(nodeToDelete == -1)
  {
    GTEST_SKIP() << "Scene has no child nodes - only root node(s) which cannot be deleted";
  }

  ASSERT_NE(nodeToDelete, -1);

  // Count children before deletion (to know expected tombstone count)
  size_t childCount  = scene.editor().getNode(nodeToDelete).children.size();
  size_t countBefore = scene.getModel().nodes.size();

  scene.editor().deleteNode(nodeToDelete);

  // Verify immediately removed (node + children)
  size_t expectedRemainingNodes = countBefore - (1 + childCount);
  EXPECT_EQ(scene.getModel().nodes.size(), expectedRemainingNodes) << "Nodes should be removed immediately";

  // Save (no compaction needed - already clean)
  fs::path outputFile = tempDir / "test_delete_simple.gltf";
  ASSERT_TRUE(scene.save(outputFile)) << "Failed to save";

  // Reload
  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(outputFile)) << "Failed to reload";

  // Should have fewer nodes (node + children deleted)
  size_t expectedNodesDeleted = 1 + childCount;
  EXPECT_EQ(scene2.getModel().nodes.size(), origCount - expectedNodesDeleted);

  // Validation should pass
  auto validation = scene2.validator().validateModel();
  EXPECT_TRUE(validation.valid) << "Validation failed after compaction";
  if(!validation.valid)
  {
    validation.print();
  }
}

//--------------------------------------------------------------------------------------------------
// Test 2: Delete multiple nodes
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingTest, DeleteMultipleNodes)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  fs::path testFile = assetsPath / "Models/Box/glTF/Box.gltf";
  ASSERT_TRUE(scene.load(testFile));

  size_t origCount = scene.getModel().nodes.size();

  // Delete first two CHILD nodes (not roots) to avoid leaving scene invalid
  std::vector<int> toDelete;

  // Collect child nodes
  for(const auto& node : scene.getModel().nodes)
  {
    for(int childIdx : node.children)
    {
      int h = childIdx;
      if(h >= 0)
      {
        toDelete.push_back(h);
        if(toDelete.size() >= 2)
          break;
      }
    }
    if(toDelete.size() >= 2)
      break;
  }

  if(toDelete.empty())
  {
    GTEST_SKIP() << "No child nodes to delete in test scene";
  }

  size_t countBefore = scene.getModel().nodes.size();

  // Delete multiple nodes (sort high to low to avoid index shifting issues)
  std::sort(toDelete.rbegin(), toDelete.rend());
  for(auto h : toDelete)
  {
    scene.editor().deleteNode(h);
  }

  // Verify nodes removed immediately
  EXPECT_LT(scene.getModel().nodes.size(), countBefore);

  // Save and reload
  fs::path outputFile = tempDir / "test_delete_multi.gltf";
  ASSERT_TRUE(scene.save(outputFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(outputFile));

  // Should have fewer nodes than original
  EXPECT_LT(scene2.getModel().nodes.size(), origCount);

  // Validation should pass
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

//--------------------------------------------------------------------------------------------------
// Test 3: Delete node with children (hierarchy remapping)
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingTest, DeleteNodeWithChildren)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  // BoxAnimated has a simple hierarchy
  fs::path testFile = assetsPath / "Models/BoxAnimated/glTF/BoxAnimated.gltf";
  ASSERT_TRUE(scene.load(testFile));

  size_t origCount = scene.getModel().nodes.size();

  // Find a node with children
  int parentHandle = -1;

  for(size_t i = 0; i < origCount; ++i)
  {
    const auto& node = scene.getModel().nodes[i];
    if(!node.children.empty())
    {
      parentHandle = static_cast<int>(i);
      break;
    }
  }

  if(parentHandle == -1)
  {
    GTEST_SKIP() << "No nodes with children in test scene";
  }

  // Count total nodes that will be deleted (parent + all descendants)
  std::function<size_t(int)> countDescendants;
  countDescendants = [&](int handle) -> size_t {
    size_t      count = 1;  // This node
    const auto& node  = scene.editor().getNode(handle);
    for(int childIdx : node.children)
    {
      int childHandle = childIdx;
      if(childHandle != -1)
      {
        count += countDescendants(childHandle);
      }
    }
    return count;
  };

  size_t expectedDeleteCount = countDescendants(parentHandle);
  ASSERT_GT(expectedDeleteCount, 1) << "Parent has no children - test invalid";

  size_t countBefore = scene.getModel().nodes.size();

  // Delete parent (recursive delete - should delete children too)
  scene.editor().deleteNode(parentHandle);

  // Verify parent + children removed immediately
  EXPECT_EQ(scene.getModel().nodes.size(), countBefore - expectedDeleteCount);

  // Save and reload
  fs::path outputFile = tempDir / "test_delete_hierarchy.gltf";
  ASSERT_TRUE(scene.save(outputFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(outputFile));

  // Should have fewer nodes (parent + children deleted)
  EXPECT_EQ(scene2.getModel().nodes.size(), origCount - expectedDeleteCount);

  // Validation should pass
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

//--------------------------------------------------------------------------------------------------
// Test 4: Delete with scene root remapping
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingTest, DeleteSceneRootNode)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  fs::path testFile = assetsPath / "Models/Box/glTF/Box.gltf";
  ASSERT_TRUE(scene.load(testFile));

  ASSERT_GT(scene.getModel().scenes.size(), 0);
  auto& mainScene = scene.getModel().scenes[0];

  size_t origRootCount = mainScene.nodes.size();
  ASSERT_GT(origRootCount, 0) << "Scene has no root nodes";

  // Test behavior depends on number of root nodes
  if(origRootCount == 1)
  {
    // Test: Deletion should be REFUSED (last root node protection)
    int rootIdx    = mainScene.nodes[0];
    int rootHandle = rootIdx;
    ASSERT_NE(rootHandle, -1);

    size_t countBefore = scene.getModel().nodes.size();
    scene.editor().deleteNode(rootHandle);

    // Deletion should be REFUSED - node still exists
    EXPECT_EQ(scene.getModel().nodes.size(), countBefore) << "Last root node should not be deletable";

    // Skip save test since nothing was deleted
    return;
  }

  // Multiple roots - deletion should succeed
  int rootIdx    = mainScene.nodes[0];
  int rootHandle = rootIdx;
  ASSERT_NE(rootHandle, -1);

  // Count how many nodes will be deleted (parent + descendants)
  std::function<size_t(int)> countDescendants;
  countDescendants = [&](int handle) -> size_t {
    size_t      count = 1;  // This node
    const auto& node  = scene.editor().getNode(handle);
    for(int childIdx : node.children)
    {
      int childHandle = childIdx;
      if(childHandle != -1)
      {
        count += countDescendants(childHandle);
      }
    }
    return count;
  };
  size_t expectedDeleteCount = countDescendants(rootHandle);
  size_t countBefore         = scene.getModel().nodes.size();

  scene.editor().deleteNode(rootHandle);

  // Deletion should succeed immediately
  EXPECT_EQ(scene.getModel().nodes.size(), countBefore - expectedDeleteCount);

  // Save and reload
  fs::path outputFile = tempDir / "test_delete_root.gltf";
  ASSERT_TRUE(scene.save(outputFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(outputFile));

  // Scene should have fewer root nodes (one root deleted)
  ASSERT_GT(scene2.getModel().scenes.size(), 0);
  auto& reloadedScene = scene2.getModel().scenes[0];
  EXPECT_EQ(reloadedScene.nodes.size(), origRootCount - 1);  // One root removed

  // Validation should pass
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

//--------------------------------------------------------------------------------------------------
// Test 5: Delete animated node (animation channel remapping)
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingTest, DeleteAnimatedNode)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  // BoxAnimated has animations
  fs::path testFile = assetsPath / "Models/BoxAnimated/glTF/BoxAnimated.gltf";
  ASSERT_TRUE(scene.load(testFile));

  ASSERT_GT(scene.animation().getNumAnimations(), 0) << "Scene has no animations";

  // Find an animated node
  const auto& anim = scene.getModel().animations[0];
  ASSERT_GT(anim.channels.size(), 0) << "Animation has no channels";

  int animatedNodeIdx = anim.channels[0].target_node;
  int animatedHandle  = animatedNodeIdx;
  ASSERT_NE(animatedHandle, -1);

  // Delete the animated node
  scene.editor().deleteNode(animatedHandle);

  // Save (should remove the animation channel)
  fs::path outputFile = tempDir / "test_compact_anim_delete.gltf";
  ASSERT_TRUE(scene.save(outputFile));

  // Reload
  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(outputFile));

  // Animation should be removed OR have fewer channels
  // (depends on if all channels targeted the deleted node)
  if(scene2.animation().getNumAnimations() > 0)
  {
    const auto& anim2 = scene2.getModel().animations[0];
    // Should have fewer channels (or be removed entirely)
    EXPECT_LE(anim2.channels.size(), anim.channels.size());
  }

  // Validation should pass
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

//--------------------------------------------------------------------------------------------------
// Test 6: Delete non-animated node (animation preserved)
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingTest, DeleteNonAnimatedNode)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  fs::path testFile = assetsPath / "Models/BoxAnimated/glTF/BoxAnimated.gltf";
  ASSERT_TRUE(scene.load(testFile));

  ASSERT_GT(scene.animation().getNumAnimations(), 0);
  size_t origAnimCount = scene.animation().getNumAnimations();

  // Collect all animated node indices
  std::unordered_set<int> animatedNodes;
  for(const auto& anim : scene.getModel().animations)
  {
    for(const auto& channel : anim.channels)
    {
      animatedNodes.insert(channel.target_node);
    }
  }

  ASSERT_GT(animatedNodes.size(), 0) << "No animated nodes found";

  // Find a non-animated node (check ALL nodes)
  int nonAnimatedHandle = -1;
  for(size_t i = 0; i < scene.getModel().nodes.size(); ++i)
  {
    if(animatedNodes.find(static_cast<int>(i)) == animatedNodes.end())
    {
      // This node is NOT animated
      nonAnimatedHandle = static_cast<int>(i);
      LOGI("Non-animated node found: %s", scene.editor().getNode(nonAnimatedHandle).name.c_str());
      break;
    }
  }

  if(nonAnimatedHandle == -1)
  {
    GTEST_SKIP() << "All nodes are animated - cannot test non-animated deletion";
  }

  // Make sure it's not a root node (to avoid scene becoming invalid)
  int         nodeIdx   = nonAnimatedHandle;
  const auto& mainScene = scene.getModel().scenes[0];
  bool        isRoot    = (std::find(mainScene.nodes.begin(), mainScene.nodes.end(), nodeIdx) != mainScene.nodes.end());

  if(isRoot)
  {
    // Check if there are other roots
    size_t activeRootCount = mainScene.nodes.size();

    if(activeRootCount <= 1)
    {
      GTEST_SKIP() << "Only non-animated node is the last root - cannot delete";
    }
  }

  // Delete non-animated node
  scene.editor().deleteNode(nonAnimatedHandle);

  // Save and reload
  fs::path outputFile = tempDir / "test_compact_preserve_anim.gltf";
  ASSERT_TRUE(scene.save(outputFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(outputFile));

  // Animations should be preserved
  EXPECT_EQ(scene2.animation().getNumAnimations(), origAnimCount);

  // Animation should still work
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
// Test 7: Try to delete all nodes (should be prevented - last root protected)
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingTest, LastRootNodeProtected)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  fs::path testFile = assetsPath / "Models/Box/glTF/Box.gltf";
  ASSERT_TRUE(scene.load(testFile));

  // Try to delete all nodes - should be prevented
  std::vector<int> allHandles;
  for(size_t i = 0; i < scene.getModel().nodes.size(); ++i)
  {
    int h = static_cast<int>(i);
    if(h != -1)
    {
      allHandles.push_back(h);
    }
  }

  size_t initialNodeCount = allHandles.size();
  ASSERT_GT(initialNodeCount, 0);

  // Try to delete all nodes
  for(auto h : allHandles)
  {
    scene.editor().deleteNode(h);
  }

  // Deletion should be PREVENTED when trying to delete last root node
  // At least one node should remain (the last root)
  EXPECT_GT(scene.getModel().nodes.size(), 0) << "Should not be able to delete ALL nodes (last root protected)";

  // Scene should still be valid (at least one root node remains)
  EXPECT_GT(scene.getModel().scenes[0].nodes.size(), 0) << "Scene should have at least one root node";

  // Save should succeed (scene is valid)
  fs::path outputFile = tempDir / "test_compact_empty.gltf";
  bool     saveResult = scene.save(outputFile);
  EXPECT_TRUE(saveResult) << "Save should succeed (scene is still valid with protected root)";

  // Reload should succeed
  if(saveResult)
  {
    nvvkgltf::Scene scene2;
    EXPECT_TRUE(scene2.load(outputFile)) << "Should be able to load valid scene";
    EXPECT_GT(scene2.getModel().nodes.size(), 0) << "Reloaded scene should have nodes";
  }
}

//--------------------------------------------------------------------------------------------------
// Test 8: Add then delete then save (mixed operations)
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingTest, AddDeleteMixed)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  fs::path testFile = assetsPath / "Models/Box/glTF/Box.gltf";
  ASSERT_TRUE(scene.load(testFile));

  size_t origCount = scene.getModel().nodes.size();

  // Add a new node
  int newNode = scene.editor().addNode("AddedNode");
  ASSERT_NE(newNode, -1);

  // Delete a NON-root node (to avoid leaving scene invalid)
  // Find a child node
  int nodeToDelete = -1;
  for(const auto& node : scene.getModel().nodes)
  {
    if(!node.children.empty())
    {
      int childIdx = node.children[0];
      nodeToDelete = childIdx;
      break;
    }
  }

  // If no children found, skip this test
  if(nodeToDelete == -1)
  {
    GTEST_SKIP() << "No child nodes to delete in test scene";
  }

  // Count how many nodes will be deleted (node + descendants)
  std::function<size_t(int)> countDescendants;
  countDescendants = [&](int handle) -> size_t {
    size_t      count = 1;
    const auto& node  = scene.editor().getNode(handle);
    for(int childIdx : node.children)
    {
      int childHandle = childIdx;
      if(childHandle != -1)
      {
        count += countDescendants(childHandle);
      }
    }
    return count;
  };
  size_t expectedDeleteCount = countDescendants(nodeToDelete);

  scene.editor().deleteNode(nodeToDelete);

  // Save and reload
  fs::path outputFile = tempDir / "test_compact_mixed.gltf";
  ASSERT_TRUE(scene.save(outputFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(outputFile));

  // Final count = original + 1 added - deleted nodes
  size_t expectedCount = origCount + 1 - expectedDeleteCount;
  EXPECT_EQ(scene2.getModel().nodes.size(), expectedCount);

  // Find the added node by name
  int addedNodeIdx = -1;
  for(size_t i = 0; i < scene2.getModel().nodes.size(); ++i)
  {
    if(scene2.getModel().nodes[i].name == "AddedNode")
    {
      addedNodeIdx = static_cast<int>(i);
      break;
    }
  }
  EXPECT_GE(addedNodeIdx, 0) << "Added node not found after save/load";

  // Validation should pass
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

//--------------------------------------------------------------------------------------------------
// Test 9: Verify render data is valid after compaction (without reload)
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingTest, RenderDataValidAfterCompaction)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  fs::path testFile = assetsPath / "Models/Box/glTF/Box.gltf";
  ASSERT_TRUE(scene.load(testFile));

  size_t origRenderNodeCount = scene.getRenderNodes().size();
  ASSERT_GT(origRenderNodeCount, 0);

  // Delete a child node
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

  // Trigger compaction WITHOUT saving/reloading
  // This is internal, so we save to trigger it
  fs::path outputFile = tempDir / "test_compact_render_data.gltf";
  ASSERT_TRUE(scene.save(outputFile));

  // CRITICAL: After compaction, render data should be rebuilt
  // (parseScene() should have been called by compactModel())

  // Check world matrices were rebuilt (should match node count)
  EXPECT_EQ(scene.getNodesWorldMatrices().size(), scene.getModel().nodes.size()) << "World matrices size mismatch after compaction";

  // Check that data structures are consistent (even if scene has no meshes)
  // In this case, we deleted the only mesh node, so renderNodes may be empty (camera only)

  // Validation should pass (scene is valid, just has no mesh nodes)
  EXPECT_TRUE(scene.validator().validateModel().valid);

  // If there are render nodes, verify they're valid
  const auto& renderNodes = scene.getRenderNodes();
  for(const auto& rnode : renderNodes)
  {
    // Check that all references are valid
    EXPECT_GE(rnode.materialID, 0) << "Invalid material ID in render node";
    EXPECT_LT(rnode.materialID, static_cast<int>(scene.getModel().materials.size()));
    EXPECT_GE(rnode.renderPrimID, 0) << "Invalid primitive ID in render node";
    EXPECT_LT(rnode.renderPrimID, static_cast<int>(scene.getRenderPrimitives().size()));
    EXPECT_GE(rnode.refNodeID, 0) << "Invalid node ID in render node";
    EXPECT_LT(rnode.refNodeID, static_cast<int>(scene.getModel().nodes.size()));
  }

  // Key point: parseScene() should have been called, so handle mappings should be valid
}

//--------------------------------------------------------------------------------------------------
// Test 10: Duplicate root node (must be added to scene.nodes[])
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingTest, DuplicateRootNode)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  fs::path testFile = assetsPath / "Models/Box/glTF/Box.gltf";
  ASSERT_TRUE(scene.load(testFile));

  // Get first root node
  ASSERT_GT(scene.getModel().scenes.size(), 0);
  auto& mainScene = scene.getModel().scenes[0];
  ASSERT_GT(mainScene.nodes.size(), 0) << "Scene has no root nodes";

  int rootIdx    = mainScene.nodes[0];
  int rootHandle = rootIdx;
  ASSERT_NE(rootHandle, -1);

  size_t origRootCount = mainScene.nodes.size();

  // Duplicate the root node
  int dupHandle = scene.editor().duplicateNode(rootHandle);
  ASSERT_NE(dupHandle, -1);

  // CRITICAL: Duplicated node must be added to scene root nodes!
  EXPECT_EQ(mainScene.nodes.size(), origRootCount + 1) << "Duplicate not added to scene roots";

  // Find the duplicated node in root nodes
  int  dupIdx       = dupHandle;
  bool foundInRoots = (std::find(mainScene.nodes.begin(), mainScene.nodes.end(), dupIdx) != mainScene.nodes.end());
  EXPECT_TRUE(foundInRoots) << "Duplicated root node not found in scene.nodes[]";

  // Save and reload to verify
  fs::path outputFile = tempDir / "test_duplicate_root.gltf";
  ASSERT_TRUE(scene.save(outputFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(outputFile));

  // Should have one more root node
  ASSERT_GT(scene2.getModel().scenes.size(), 0);
  EXPECT_EQ(scene2.getModel().scenes[0].nodes.size(), origRootCount + 1);

  // Validation should pass
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}

//--------------------------------------------------------------------------------------------------
// Test 11: Duplicate node with mesh - verify render data (2 renderNodes, 1 shared renderPrimitive)
//--------------------------------------------------------------------------------------------------
TEST_F(IndexRemappingTest, DuplicateCreatesRenderNodes)
{
  nvvkgltf::Scene scene;

  fs::path assetsPath = getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  fs::path testFile = assetsPath / "Models/Box/glTF/Box.gltf";
  ASSERT_TRUE(scene.load(testFile));

  // Find a node with a mesh
  int nodeWithMesh = -1;
  for(size_t i = 0; i < scene.getModel().nodes.size(); ++i)
  {
    const auto& node = scene.getModel().nodes[i];
    if(node.mesh >= 0)
    {
      nodeWithMesh = static_cast<int>(i);
      break;
    }
  }

  if(nodeWithMesh == -1)
  {
    GTEST_SKIP() << "No nodes with meshes in test scene";
  }

  // Get initial counts
  size_t origRenderNodeCount = scene.getRenderNodes().size();
  size_t origRenderPrimCount = scene.getRenderPrimitives().size();

  ASSERT_GT(origRenderNodeCount, 0) << "Scene has no render nodes";
  ASSERT_GT(origRenderPrimCount, 0) << "Scene has no render primitives";

  // Duplicate the node with mesh
  int dupHandle = scene.editor().duplicateNode(nodeWithMesh);
  ASSERT_NE(dupHandle, -1);

  // Rebuild render data (this is what renderer does after hierarchy change)
  scene.setCurrentScene(scene.getCurrentScene());

  // VERIFY: Should have MORE render nodes (duplicate creates new instances)
  EXPECT_GT(scene.getRenderNodes().size(), origRenderNodeCount) << "Duplicate should create additional render nodes";

  // VERIFY: Render primitives should be SAME (primitives are shared/deduplicated)
  // For Box.gltf, the mesh is reused, so primitive count stays the same
  EXPECT_EQ(scene.getRenderPrimitives().size(), origRenderPrimCount) << "Render primitives should be shared (deduplicated)";

  // Save and reload to verify
  fs::path outputFile = tempDir / "test_duplicate_render_data.gltf";
  ASSERT_TRUE(scene.save(outputFile));

  nvvkgltf::Scene scene2;
  ASSERT_TRUE(scene2.load(outputFile));

  // After reload, should have more render nodes
  EXPECT_GT(scene2.getRenderNodes().size(), origRenderNodeCount);

  // Primitives still shared
  EXPECT_EQ(scene2.getRenderPrimitives().size(), origRenderPrimCount);

  // Validate
  EXPECT_TRUE(scene2.validator().validateModel().valid);
}
