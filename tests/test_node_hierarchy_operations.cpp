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
#include <map>
#include <set>
#include <fmt/core.h>

using namespace gltf_test;

//--------------------------------------------------------------------------------------------------
// Test suite for complex node hierarchy operations (add, duplicate, reparent, delete)
// These tests verify that render nodes remain consistent across complex editing operations
//--------------------------------------------------------------------------------------------------

class NodeHierarchyTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    tempDir = std::filesystem::temp_directory_path() / "gltf_hierarchy_test";
    std::filesystem::create_directories(tempDir);
  }

  void TearDown() override { std::filesystem::remove_all(tempDir); }

  std::filesystem::path tempDir;

  // Helper: Collect all render node IDs for a given node and its descendants
  std::set<int> collectAllRenderNodeIDs(nvvkgltf::Scene& scene, int nodeID)
  {
    std::set<int>    result;
    std::vector<int> renderNodeIDs;

    auto getChildren = [&scene](int nid) -> std::vector<int> {
      if(nid >= 0 && nid < static_cast<int>(scene.getModel().nodes.size()))
        return scene.getModel().nodes[nid].children;
      return {};
    };

    scene.getRenderNodeRegistry().getAllRenderNodesForNodeRecursive(nodeID, getChildren, renderNodeIDs);

    for(int rnID : renderNodeIDs)
      result.insert(rnID);

    return result;
  }

  // Helper: Get all meshes referenced by a node hierarchy
  std::set<int> collectAllMeshes(nvvkgltf::Scene& scene, int nodeID)
  {
    std::set<int>    meshes;
    std::vector<int> toVisit = {nodeID};
    std::set<int>    visited;

    while(!toVisit.empty())
    {
      int current = toVisit.back();
      toVisit.pop_back();

      if(visited.count(current))
        continue;
      visited.insert(current);

      if(current >= 0 && current < static_cast<int>(scene.getModel().nodes.size()))
      {
        const auto& node = scene.getModel().nodes[current];
        if(node.mesh >= 0)
          meshes.insert(node.mesh);

        for(int child : node.children)
          toVisit.push_back(child);
      }
    }

    return meshes;
  }

  // Helper: Verify render node registry consistency
  void verifyRenderNodeConsistency(nvvkgltf::Scene& scene)
  {
    const auto& renderNodes = scene.getRenderNodes();
    const auto& reg         = scene.getRenderNodeRegistry();

    for(int rnID = 0; rnID < static_cast<int>(renderNodes.size()); ++rnID)
    {
      // Verify bidirectional mapping
      auto nodeAndPrim = reg.getNodeAndPrim(rnID);
      ASSERT_TRUE(nodeAndPrim.has_value()) << "RenderNode " << rnID << " should have (nodeID, primID)";

      int nodeID  = nodeAndPrim->first;
      int primIdx = nodeAndPrim->second;

      // Verify node index is valid
      ASSERT_GE(nodeID, 0) << "RenderNode " << rnID << " has invalid nodeID";
      ASSERT_LT(nodeID, static_cast<int>(scene.getModel().nodes.size())) << "RenderNode " << rnID << " has out-of-bounds nodeID";

      // Verify render node data is valid
      const auto& rn = renderNodes[rnID];
      EXPECT_GE(rn.renderPrimID, 0) << "RenderNode " << rnID << " has invalid renderPrimID";
      EXPECT_LT(rn.renderPrimID, static_cast<int>(scene.getRenderPrimitives().size()))
          << "RenderNode " << rnID << " has out-of-bounds renderPrimID";
      EXPECT_EQ(rn.refNodeID, nodeID) << "RenderNode " << rnID << " refNodeID doesn't match registry";
    }
  }

  // Helper: Count total meshes that should have render nodes
  int countExpectedRenderNodes(nvvkgltf::Scene& scene)
  {
    int count = 0;
    for(const auto& node : scene.getModel().nodes)
    {
      if(node.mesh >= 0 && node.mesh < static_cast<int>(scene.getModel().meshes.size()))
      {
        const auto& mesh = scene.getModel().meshes[node.mesh];
        count += static_cast<int>(mesh.primitives.size());
      }
    }
    return count;
  }
};

//--------------------------------------------------------------------------------------------------
// Regression test for the exact scenario described:
// 1. Load a scene with ~5 nodes
// 2. Add a new node (node_6)
// 3. Move nodes 3 and 4 under node_6 (reparenting)
// 4. Duplicate node_6 (creating node_7)
// 5. Translate node_7
// 6. Delete node_7 -> verify no missing geometry (all render nodes present)
//--------------------------------------------------------------------------------------------------

TEST_F(NodeHierarchyTest, ComplexHierarchyOperationsPreservesRenderNodes)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  // Initial state
  size_t initialNodeCount       = scene.getModel().nodes.size();
  size_t initialRenderNodeCount = scene.getRenderNodes().size();
  int    expectedRenderNodes    = countExpectedRenderNodes(scene);

  ASSERT_GE(initialNodeCount, 1u) << "Need at least 1 node";
  ASSERT_EQ(initialRenderNodeCount, expectedRenderNodes) << "Initial render node count should match expected";

  // Step 1: Add a new node (node_6 in user's scenario)
  int newNode = scene.editor().addNode("node_6");
  ASSERT_NE(newNode, -1) << "Failed to add new node";
  EXPECT_EQ(scene.getModel().nodes.size(), initialNodeCount + 1);

  // New empty node should not add render nodes (no mesh)
  EXPECT_EQ(scene.getRenderNodes().size(), initialRenderNodeCount) << "Adding empty node should not create render nodes";

  // Step 2: Move existing nodes under new node (if we have enough nodes)
  if(initialNodeCount >= 2)
  {
    int node0 = 0;                                                    // Equivalent to node_3 in user's scenario
    int node1 = std::min(1, static_cast<int>(initialNodeCount - 1));  // Equivalent to node_4

    // Save mesh info before reparenting
    std::set<int> node0Meshes     = collectAllMeshes(scene, node0);
    std::set<int> node1Meshes     = collectAllMeshes(scene, node1);
    std::set<int> allMeshesBefore = node0Meshes;
    allMeshesBefore.insert(node1Meshes.begin(), node1Meshes.end());

    // Reparent nodes
    scene.editor().setNodeParent(node0, newNode);
    scene.editor().setNodeParent(node1, newNode);

    EXPECT_EQ(scene.editor().getNodeParent(node0), newNode) << "node0 should have newNode as parent";
    EXPECT_EQ(scene.editor().getNodeParent(node1), newNode) << "node1 should have newNode as parent";

    // Render nodes should still be present
    EXPECT_EQ(scene.getRenderNodes().size(), initialRenderNodeCount) << "Reparenting should not change render node count";
    verifyRenderNodeConsistency(scene);

    // Verify all meshes still have render nodes
    std::set<int> allMeshesAfter   = collectAllMeshes(scene, node0);
    std::set<int> node1MeshesAfter = collectAllMeshes(scene, node1);
    allMeshesAfter.insert(node1MeshesAfter.begin(), node1MeshesAfter.end());
    EXPECT_EQ(allMeshesAfter, allMeshesBefore) << "Reparenting should not lose meshes";
  }

  // Step 3: Duplicate the new node hierarchy (node_7 in user's scenario)
  int duplicatedNode = scene.editor().duplicateNode(newNode);
  ASSERT_NE(duplicatedNode, -1) << "Failed to duplicate node";

  size_t afterDuplicateRenderNodeCount = scene.getRenderNodes().size();
  int    afterDuplicateExpected        = countExpectedRenderNodes(scene);

  EXPECT_EQ(afterDuplicateRenderNodeCount, afterDuplicateExpected)
      << "After duplication, render node count should match expected node count";
  verifyRenderNodeConsistency(scene);

  // Step 4: Translate the duplicated node
  scene.editor().setNodeTRS(duplicatedNode, glm::vec3(10.0f, 0.0f, 0.0f), glm::quat(1, 0, 0, 0), glm::vec3(1, 1, 1));
  scene.updateNodeWorldMatrices();

  // Render nodes should still be intact
  EXPECT_EQ(scene.getRenderNodes().size(), afterDuplicateRenderNodeCount) << "Translation should not change render node count";

  // Step 5: Delete the duplicated node hierarchy (CRITICAL TEST)
  std::set<int> renderNodesBeforeDelete = collectAllRenderNodeIDs(scene, duplicatedNode);

  scene.editor().deleteNode(duplicatedNode);

  // Verify render node consistency after deletion
  verifyRenderNodeConsistency(scene);

  // CRITICAL: Verify that original geometry is still present
  // After deleting duplicated node, we should have the same render nodes as before duplication
  size_t afterDeleteRenderNodeCount = scene.getRenderNodes().size();
  int    afterDeleteExpected        = countExpectedRenderNodes(scene);

  EXPECT_EQ(afterDeleteRenderNodeCount, afterDeleteExpected) << "After deletion, render node count should match node count in model";

  // The render node count after deletion should be approximately equal to what we had before duplication
  // (may differ slightly due to render primitive sharing)
  EXPECT_GE(afterDeleteRenderNodeCount, initialRenderNodeCount - renderNodesBeforeDelete.size())
      << "Deletion removed too many render nodes - original geometry missing!";

  // Verify each remaining render node has valid data
  for(int rnID = 0; rnID < static_cast<int>(scene.getRenderNodes().size()); ++rnID)
  {
    const auto& rn = scene.getRenderNodes()[rnID];
    EXPECT_GE(rn.refNodeID, 0) << "RenderNode " << rnID << " has invalid refNodeID after deletion";
    EXPECT_LT(rn.refNodeID, static_cast<int>(scene.getModel().nodes.size()))
        << "RenderNode " << rnID << " references out-of-bounds node after deletion";
  }
}

//--------------------------------------------------------------------------------------------------
// Test: Multiple duplications and deletions in sequence
//--------------------------------------------------------------------------------------------------

TEST_F(NodeHierarchyTest, MultipleAddDuplicateDeleteCycles)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  size_t initialRenderNodeCount = scene.getRenderNodes().size();

  // Perform multiple cycles of add -> duplicate -> delete
  for(int cycle = 0; cycle < 3; ++cycle)
  {
    // Add new node
    int newNode = scene.editor().addNode(fmt::format("test_node_{}", cycle));
    ASSERT_NE(newNode, -1);

    // Duplicate it
    int duplicatedNode = scene.editor().duplicateNode(newNode);
    ASSERT_NE(duplicatedNode, -1);

    // Verify consistency
    verifyRenderNodeConsistency(scene);

    // Delete the duplicate
    scene.editor().deleteNode(duplicatedNode);

    // Verify consistency after deletion
    verifyRenderNodeConsistency(scene);

    // Verify render node count is still correct
    int expected = countExpectedRenderNodes(scene);
    EXPECT_EQ(scene.getRenderNodes().size(), expected) << "Cycle " << cycle << ": render node count mismatch";
  }

  // Final check: should still have consistent render nodes
  verifyRenderNodeConsistency(scene);
}

//--------------------------------------------------------------------------------------------------
// Test: Reparenting with meshes and then deleting
//--------------------------------------------------------------------------------------------------

TEST_F(NodeHierarchyTest, ReparentWithMeshThenDelete)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  if(scene.getModel().nodes.empty() || scene.getModel().meshes.empty())
    GTEST_SKIP() << "Need nodes with meshes";

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

  if(nodeWithMesh < 0)
    GTEST_SKIP() << "No node with mesh found";

  // Count render nodes for this node
  std::set<int> renderNodesForNode = collectAllRenderNodeIDs(scene, nodeWithMesh);
  ASSERT_FALSE(renderNodesForNode.empty()) << "Node with mesh should have render nodes";

  size_t initialTotal = scene.getRenderNodes().size();

  // Create new parent and reparent the mesh node
  int newParent = scene.editor().addNode("new_parent");
  scene.editor().setNodeParent(nodeWithMesh, newParent);

  // Verify render nodes are still present
  EXPECT_EQ(scene.getRenderNodes().size(), initialTotal) << "Reparenting should not change render node count";
  verifyRenderNodeConsistency(scene);

  // Duplicate the parent (which includes the mesh child)
  int duplicatedParent = scene.editor().duplicateNode(newParent);
  ASSERT_NE(duplicatedParent, -1);

  size_t afterDuplicate = scene.getRenderNodes().size();
  EXPECT_GT(afterDuplicate, initialTotal) << "Duplicating hierarchy with mesh should add render nodes";

  // Delete the duplicated hierarchy
  scene.editor().deleteNode(duplicatedParent);

  // Verify we're back to original count
  size_t afterDelete = scene.getRenderNodes().size();
  int    expected    = countExpectedRenderNodes(scene);
  EXPECT_EQ(afterDelete, expected) << "After deleting duplicate, render nodes should match model";

  // Verify original node's render nodes are still valid
  std::set<int> finalRenderNodes = collectAllRenderNodeIDs(scene, nodeWithMesh);
  EXPECT_FALSE(finalRenderNodes.empty()) << "Original node should still have render nodes after deleting duplicate";

  verifyRenderNodeConsistency(scene);
}

//--------------------------------------------------------------------------------------------------
// Test: Deep hierarchy duplication and deletion
//--------------------------------------------------------------------------------------------------

TEST_F(NodeHierarchyTest, DeepHierarchyDuplicateDelete)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  // Create a deep hierarchy: parent -> child1 -> child2 -> child3
  int parent = scene.editor().addNode("deep_parent");
  int child1 = scene.editor().addNode("deep_child1", parent);
  int child2 = scene.editor().addNode("deep_child2", child1);
  int child3 = scene.editor().addNode("deep_child3", child2);

  // Attach a mesh to the deepest child (if available)
  if(!scene.getModel().meshes.empty())
  {
    scene.editor().setNodeMesh(child3, 0);
  }

  size_t beforeDuplicate = scene.getRenderNodes().size();

  // Duplicate the entire deep hierarchy
  int duplicatedParent = scene.editor().duplicateNode(parent);
  ASSERT_NE(duplicatedParent, -1);

  size_t afterDuplicate = scene.getRenderNodes().size();
  if(!scene.getModel().meshes.empty())
  {
    EXPECT_GT(afterDuplicate, beforeDuplicate) << "Duplicating hierarchy with mesh should add render nodes";
  }

  verifyRenderNodeConsistency(scene);

  // Delete the duplicated hierarchy
  scene.editor().deleteNode(duplicatedParent);

  size_t afterDelete = scene.getRenderNodes().size();
  int    expected    = countExpectedRenderNodes(scene);
  EXPECT_EQ(afterDelete, expected) << "After deleting duplicate hierarchy, render nodes should match model";

  verifyRenderNodeConsistency(scene);
}
