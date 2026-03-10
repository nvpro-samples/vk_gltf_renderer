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
#include <fmt/core.h>

using namespace gltf_test;

//--------------------------------------------------------------------------------------------------
// Test to reproduce the exact children order issue
//--------------------------------------------------------------------------------------------------

class ChildrenOrderTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    tempDir = std::filesystem::temp_directory_path() / "gltf_children_order_test";
    std::filesystem::create_directories(tempDir);
  }

  void TearDown() override { std::filesystem::remove_all(tempDir); }

  std::filesystem::path tempDir;

  // Helper to print render node mappings
  void printRenderNodes(const std::string& label, nvvkgltf::Scene& scene)
  {
    fmt::print("\n=== {} ===\n", label);
    const auto& renderNodes = scene.getRenderNodes();
    const auto& registry    = scene.getRenderNodeRegistry();
    const auto& model       = scene.getModel();

    for(int rnID = 0; rnID < static_cast<int>(renderNodes.size()); ++rnID)
    {
      const auto& rn          = renderNodes[rnID];
      auto        nodeAndPrim = registry.getNodeAndPrim(rnID);

      if(nodeAndPrim.has_value() && rn.refNodeID >= 0 && rn.refNodeID < static_cast<int>(model.nodes.size()))
      {
        fmt::print("RN {:2d}: Node {:2d} ({:12s}) Prim {} Mat {}\n", rnID, rn.refNodeID, model.nodes[rn.refNodeID].name,
                   nodeAndPrim->second, rn.materialID);
      }
    }
  }

  // Helper to get render node info as a tuple for comparison
  struct RenderNodeInfo
  {
    int         nodeID;
    int         primID;
    int         materialID;
    std::string nodeName;

    bool operator==(const RenderNodeInfo& other) const
    {
      return nodeID == other.nodeID && primID == other.primID && materialID == other.materialID && nodeName == other.nodeName;
    }
  };

  std::vector<RenderNodeInfo> getRenderNodeInfos(nvvkgltf::Scene& scene)
  {
    std::vector<RenderNodeInfo> result;
    const auto&                 renderNodes = scene.getRenderNodes();
    const auto&                 registry    = scene.getRenderNodeRegistry();
    const auto&                 model       = scene.getModel();

    for(int rnID = 0; rnID < static_cast<int>(renderNodes.size()); ++rnID)
    {
      const auto& rn          = renderNodes[rnID];
      auto        nodeAndPrim = registry.getNodeAndPrim(rnID);

      if(nodeAndPrim.has_value() && rn.refNodeID >= 0 && rn.refNodeID < static_cast<int>(model.nodes.size()))
      {
        result.push_back({rn.refNodeID, nodeAndPrim->second, rn.materialID, model.nodes[rn.refNodeID].name});
      }
    }
    return result;
  }
};

//--------------------------------------------------------------------------------------------------
// Test: Reproduce exact user scenario with reverse-order reparenting
//--------------------------------------------------------------------------------------------------

TEST_F(ChildrenOrderTest, ReverseOrderReparentingPreservesRenderNodeOrder)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  // Get initial state
  auto initialRenderNodes = getRenderNodeInfos(scene);
  printRenderNodes("INITIAL STATE", scene);

  if(scene.getModel().nodes.size() < 2)
    GTEST_SKIP() << "Need at least 2 nodes for this test";

  // Add a new parent node
  int newParent = scene.editor().addNode("NewParent");
  ASSERT_NE(newParent, -1);

  // Simulate user's actions: move nodes in REVERSE order (5, then 4)
  // For Box model, we'll use nodes 0 and 1 if they exist
  std::vector<int> nodesToMove;
  for(int i = 0; i < std::min(2, static_cast<int>(scene.getModel().nodes.size())); ++i)
  {
    nodesToMove.push_back(i);
  }

  if(nodesToMove.size() < 2)
    GTEST_SKIP() << "Need at least 2 nodes to move";

  // Move in REVERSE order (like moving node 1 first, then node 0)
  scene.editor().setNodeParent(nodesToMove[1], newParent);
  scene.editor().setNodeParent(nodesToMove[0], newParent);

  // Check children order after moving
  const auto& parentNode = scene.getModel().nodes[newParent];
  fmt::print("\nParent children after moving: [");
  for(size_t i = 0; i < parentNode.children.size(); ++i)
  {
    fmt::print("{}{}", parentNode.children[i], (i < parentNode.children.size() - 1) ? ", " : "");
  }
  fmt::print("]\n");

  // Render nodes should still be in original order (just with different parent)
  auto afterReparentRenderNodes = getRenderNodeInfos(scene);
  printRenderNodes("AFTER REPARENTING", scene);

  // Duplicate the new parent (with its children)
  int duplicatedParent = scene.editor().duplicateNode(newParent);
  ASSERT_NE(duplicatedParent, -1);

  auto afterDuplicateRenderNodes = getRenderNodeInfos(scene);
  printRenderNodes("AFTER DUPLICATE", scene);

  // Delete the duplicated parent
  scene.editor().deleteNode(duplicatedParent);

  auto afterDeleteRenderNodes = getRenderNodeInfos(scene);
  printRenderNodes("AFTER DELETE", scene);

  // CRITICAL TEST: After delete, render nodes should match initial state
  // (same nodes, same order, same materials)
  ASSERT_EQ(afterDeleteRenderNodes.size(), initialRenderNodes.size()) << "Render node count should match after duplicate-delete cycle";

  for(size_t i = 0; i < initialRenderNodes.size(); ++i)
  {
    EXPECT_EQ(afterDeleteRenderNodes[i].nodeID, initialRenderNodes[i].nodeID)
        << "RenderNode " << i << " should reference same node ID as initial state";
    EXPECT_EQ(afterDeleteRenderNodes[i].materialID, initialRenderNodes[i].materialID)
        << "RenderNode " << i << " should have same material ID as initial state";
    EXPECT_EQ(afterDeleteRenderNodes[i].nodeName, initialRenderNodes[i].nodeName)
        << "RenderNode " << i << " should reference same node name as initial state";
  }
}

//--------------------------------------------------------------------------------------------------
// Test: Forward order reparenting (should work correctly)
//--------------------------------------------------------------------------------------------------

TEST_F(ChildrenOrderTest, ForwardOrderReparentingPreservesRenderNodeOrder)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  auto initialRenderNodes = getRenderNodeInfos(scene);
  printRenderNodes("INITIAL STATE (FORWARD ORDER)", scene);

  if(scene.getModel().nodes.size() < 2)
    GTEST_SKIP() << "Need at least 2 nodes for this test";

  int newParent = scene.editor().addNode("NewParent");
  ASSERT_NE(newParent, -1);

  std::vector<int> nodesToMove;
  for(int i = 0; i < std::min(2, static_cast<int>(scene.getModel().nodes.size())); ++i)
  {
    nodesToMove.push_back(i);
  }

  if(nodesToMove.size() < 2)
    GTEST_SKIP() << "Need at least 2 nodes to move";

  // Move in FORWARD order (node 0 first, then node 1)
  scene.editor().setNodeParent(nodesToMove[0], newParent);
  scene.editor().setNodeParent(nodesToMove[1], newParent);

  const auto& parentNode = scene.getModel().nodes[newParent];
  fmt::print("\nParent children after moving (forward order): [");
  for(size_t i = 0; i < parentNode.children.size(); ++i)
  {
    fmt::print("{}{}", parentNode.children[i], (i < parentNode.children.size() - 1) ? ", " : "");
  }
  fmt::print("]\n");

  auto afterReparentRenderNodes = getRenderNodeInfos(scene);
  printRenderNodes("AFTER REPARENTING (FORWARD)", scene);

  int duplicatedParent = scene.editor().duplicateNode(newParent);
  ASSERT_NE(duplicatedParent, -1);

  auto afterDuplicateRenderNodes = getRenderNodeInfos(scene);
  printRenderNodes("AFTER DUPLICATE (FORWARD)", scene);

  scene.editor().deleteNode(duplicatedParent);

  auto afterDeleteRenderNodes = getRenderNodeInfos(scene);
  printRenderNodes("AFTER DELETE (FORWARD)", scene);

  // This should pass with forward-order reparenting
  ASSERT_EQ(afterDeleteRenderNodes.size(), initialRenderNodes.size());

  for(size_t i = 0; i < initialRenderNodes.size(); ++i)
  {
    EXPECT_EQ(afterDeleteRenderNodes[i].nodeID, initialRenderNodes[i].nodeID)
        << "RenderNode " << i << " should reference same node ID as initial state (forward order)";
    EXPECT_EQ(afterDeleteRenderNodes[i].materialID, initialRenderNodes[i].materialID)
        << "RenderNode " << i << " should have same material ID as initial state (forward order)";
  }
}
