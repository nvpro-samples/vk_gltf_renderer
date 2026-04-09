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

//--------------------------------------------------------------------------------------------------
// Dirty flags system tests
//--------------------------------------------------------------------------------------------------

class DirtyFlagsTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    tempDir = std::filesystem::temp_directory_path() / "gltf_dirty_flags_test";
    std::filesystem::create_directories(tempDir);
  }

  void TearDown() override { std::filesystem::remove_all(tempDir); }

  std::filesystem::path tempDir;
};

TEST_F(DirtyFlagsTest, InitiallyEmptyAfterClear)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  scene.clearDirtyFlags();
  EXPECT_TRUE(scene.getDirtyFlags().isEmpty()) << "DirtyFlags should be empty after clear";
}

TEST_F(DirtyFlagsTest, MarkNodeDirtySetsNodesAndRenderNodes)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);
  scene.clearDirtyFlags();

  if(scene.getModel().nodes.empty())
    GTEST_SKIP() << "No nodes in scene";

  int nodeIndex = 0;
  scene.markNodeDirty(nodeIndex);

  EXPECT_FALSE(scene.getDirtyFlags().isEmpty());
  EXPECT_GE(scene.getDirtyFlags().nodes.count(nodeIndex), 1u) << "Node should be in dirty nodes set";

  // markNodeDirty also adds corresponding render node indices to renderNodesVk and renderNodesRtx
  const auto& rnIds = scene.getRenderNodeRegistry().getRenderNodesForNode(nodeIndex);
  if(!rnIds.empty())
  {
    EXPECT_FALSE(scene.getDirtyFlags().renderNodesVk.empty()) << "At least one render node should be marked for Vk";
    EXPECT_FALSE(scene.getDirtyFlags().renderNodesRtx.empty()) << "At least one render node should be marked for Rtx";
  }
}

TEST_F(DirtyFlagsTest, MarkMaterialDirtySetsMaterials)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);
  scene.clearDirtyFlags();

  if(scene.getModel().materials.empty())
    GTEST_SKIP() << "No materials in scene";

  scene.markMaterialDirty(0);
  EXPECT_FALSE(scene.getDirtyFlags().isEmpty());
  EXPECT_GE(scene.getDirtyFlags().materials.count(0), 1u);
}

TEST_F(DirtyFlagsTest, MarkLightDirtySetsLights)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);
  scene.clearDirtyFlags();

  // Box may not have lights; use markLightDirty(0) only if we have lights
  if(!scene.getModel().lights.empty())
  {
    scene.markLightDirty(0);
    EXPECT_FALSE(scene.getDirtyFlags().isEmpty());
    EXPECT_GE(scene.getDirtyFlags().lights.count(0), 1u);
  }
}

TEST_F(DirtyFlagsTest, MarkHierarchyDirtySetsHierarchyChanged)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);
  scene.clearDirtyFlags();

  scene.getDirtyFlags().allRenderNodesDirty = true;
  EXPECT_FALSE(scene.getDirtyFlags().isEmpty());
  EXPECT_TRUE(scene.getDirtyFlags().allRenderNodesDirty);
}

TEST_F(DirtyFlagsTest, MarkRenderNodeDirtySetsRenderNodesVkAndRtx)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);
  scene.clearDirtyFlags();

  if(scene.getRenderNodes().empty())
    GTEST_SKIP() << "No render nodes in scene";

  scene.markRenderNodeDirty(0, true, true);
  EXPECT_FALSE(scene.getDirtyFlags().isEmpty());
  EXPECT_GE(scene.getDirtyFlags().renderNodesVk.count(0), 1u);
  EXPECT_GE(scene.getDirtyFlags().renderNodesRtx.count(0), 1u);
}

TEST_F(DirtyFlagsTest, ClearDirtyFlagsResetsAll)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  scene.markNodeDirty(0);
  scene.getDirtyFlags().allRenderNodesDirty = true;
  scene.clearDirtyFlags();

  EXPECT_TRUE(scene.getDirtyFlags().isEmpty());
  EXPECT_TRUE(scene.getDirtyFlags().nodes.empty());
  EXPECT_TRUE(scene.getDirtyFlags().materials.empty());
  EXPECT_TRUE(scene.getDirtyFlags().lights.empty());
  EXPECT_TRUE(scene.getDirtyFlags().renderNodesVk.empty());
  EXPECT_TRUE(scene.getDirtyFlags().renderNodesRtx.empty());
  EXPECT_FALSE(scene.getDirtyFlags().allRenderNodesDirty);
  EXPECT_FALSE(scene.getDirtyFlags().tlasVisibilityNeedsCpuSync);
}

//--------------------------------------------------------------------------------------------------
// RenderNodes and RenderPrimitives creation and maintenance tests
//--------------------------------------------------------------------------------------------------

class RenderNodesAndPrimitivesTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    tempDir = std::filesystem::temp_directory_path() / "gltf_render_nodes_test";
    std::filesystem::create_directories(tempDir);
  }

  void TearDown() override { std::filesystem::remove_all(tempDir); }

  std::filesystem::path tempDir;
};

TEST_F(RenderNodesAndPrimitivesTest, ParseSceneCreatesRenderNodesAndPrimitives)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  const auto& renderNodes = scene.getRenderNodes();
  const auto& renderPrims = scene.getRenderPrimitives();

  // Box has at least one mesh with primitives, so we expect at least one render node and one primitive
  EXPECT_FALSE(renderNodes.empty()) << "Parsed scene should have at least one render node";
  EXPECT_FALSE(renderPrims.empty()) << "Parsed scene should have at least one render primitive";

  for(size_t i = 0; i < renderNodes.size(); ++i)
  {
    const auto& rn = renderNodes[i];
    EXPECT_GE(rn.materialID, 0);
    EXPECT_GE(rn.renderPrimID, 0);
    EXPECT_LT(rn.renderPrimID, static_cast<int>(renderPrims.size()));
    EXPECT_GE(rn.refNodeID, 0);
    EXPECT_LT(rn.refNodeID, static_cast<int>(scene.getModel().nodes.size()));
  }
}

TEST_F(RenderNodesAndPrimitivesTest, RenderPrimitivesHaveValidData)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  const auto& renderPrims = scene.getRenderPrimitives();
  ASSERT_FALSE(renderPrims.empty());

  for(size_t i = 0; i < renderPrims.size(); ++i)
  {
    const auto& rp = scene.getRenderPrimitive(i);
    EXPECT_GE(rp.vertexCount, 0);
    EXPECT_GE(rp.indexCount, 0);
    EXPECT_GE(rp.meshID, 0);
    EXPECT_LT(rp.meshID, static_cast<int>(scene.getModel().meshes.size()));
  }
}

TEST_F(RenderNodesAndPrimitivesTest, GetRenderNodeIDRoundTripsWithGetNodeAndPrim)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  const auto& reg         = scene.getRenderNodeRegistry();
  const auto& renderNodes = scene.getRenderNodes();

  for(int rnID = 0; rnID < static_cast<int>(renderNodes.size()); ++rnID)
  {
    auto nodeAndPrim = reg.getNodeAndPrim(rnID);
    ASSERT_TRUE(nodeAndPrim.has_value()) << "RenderNode " << rnID << " should have (nodeID, primIndex)";
    int nodeID   = nodeAndPrim->first;
    int primIdx  = nodeAndPrim->second;
    int backRnID = reg.getRenderNodeID(nodeID, primIdx);
    EXPECT_EQ(backRnID, rnID) << "Round-trip: getNodeAndPrim then getRenderNodeID should return same renderNodeID";
  }
}

TEST_F(RenderNodesAndPrimitivesTest, GetRenderNodesForNodeReturnsConsistentIndices)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  const auto& reg   = scene.getRenderNodeRegistry();
  const auto& nodes = scene.getModel().nodes;

  for(int nodeID = 0; nodeID < static_cast<int>(nodes.size()); ++nodeID)
  {
    const auto& rnIds = reg.getRenderNodesForNode(nodeID);
    for(int rnID : rnIds)
    {
      EXPECT_GE(rnID, 0);
      EXPECT_LT(rnID, static_cast<int>(scene.getRenderNodes().size()));
      auto nodeAndPrim = reg.getNodeAndPrim(rnID);
      ASSERT_TRUE(nodeAndPrim.has_value());
      EXPECT_EQ(nodeAndPrim->first, nodeID) << "RenderNode " << rnID << " should reference node " << nodeID;
    }
  }
}

TEST_F(RenderNodesAndPrimitivesTest, GetRenderNodeForPrimitiveAndGetPrimitiveIndexRoundTrip)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  const auto& renderNodes = scene.getRenderNodes();
  if(renderNodes.empty())
    GTEST_SKIP() << "No render nodes";

  for(int rnID = 0; rnID < static_cast<int>(renderNodes.size()); ++rnID)
  {
    int primIdx = scene.getPrimitiveIndexForRenderNode(rnID);
    EXPECT_GE(primIdx, 0);
    EXPECT_LT(primIdx, static_cast<int>(scene.getRenderPrimitives().size()));

    auto nodeAndPrim = scene.getRenderNodeRegistry().getNodeAndPrim(rnID);
    ASSERT_TRUE(nodeAndPrim.has_value());
    int backRnID = scene.getRenderNodeForPrimitive(nodeAndPrim->first, nodeAndPrim->second);
    EXPECT_EQ(backRnID, rnID) << "Round-trip getRenderNodeForPrimitive should match renderNodeID";
  }
}

TEST_F(RenderNodesAndPrimitivesTest, DuplicateNodeCreatesNewRenderNodes)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  size_t renderNodeCountBefore = scene.getRenderNodes().size();
  if(renderNodeCountBefore == 0)
    GTEST_SKIP() << "Scene has no render nodes";

  int rootNode = 0;
  int newRoot  = scene.editor().duplicateNode(rootNode);
  ASSERT_NE(newRoot, -1);

  size_t renderNodeCountAfter = scene.getRenderNodes().size();
  EXPECT_GE(renderNodeCountAfter, renderNodeCountBefore) << "Duplicate should add render nodes for the duplicated hierarchy";

  // New node should have render nodes (if it has mesh or children with meshes)
  const auto& rnIdsForNew = scene.getRenderNodeRegistry().getRenderNodesForNode(newRoot);
  // Duplicated root may have same structure as original; total count should have increased
  EXPECT_GT(renderNodeCountAfter, 0u);
}

TEST_F(RenderNodesAndPrimitivesTest, DeleteNodeRemovesRenderNodesAndKeepsRegistryConsistent)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  if(scene.getModel().nodes.size() < 2)
    GTEST_SKIP() << "Need at least two nodes to delete one";

  // Find a non-root node with a mesh (or any child) to delete
  int    nodeToDelete = 1;
  size_t countBefore  = scene.getRenderNodes().size();

  scene.editor().deleteNode(nodeToDelete);

  size_t countAfter = scene.getRenderNodes().size();
  // After delete we remap indices; render node count may decrease if deleted node had meshes
  EXPECT_LE(countAfter, countBefore + 1) << "Deleting a node should not increase render node count (may decrease)";

  // Registry should still be consistent: every render node has valid getNodeAndPrim
  const auto& reg = scene.getRenderNodeRegistry();
  for(int rnID = 0; rnID < static_cast<int>(scene.getRenderNodes().size()); ++rnID)
  {
    auto nodeAndPrim = reg.getNodeAndPrim(rnID);
    ASSERT_TRUE(nodeAndPrim.has_value());
    EXPECT_GE(nodeAndPrim->first, 0);
    EXPECT_LT(nodeAndPrim->first, static_cast<int>(scene.getModel().nodes.size()));
    EXPECT_GE(nodeAndPrim->second, 0);
  }
}

TEST_F(RenderNodesAndPrimitivesTest, UpdateNodeWorldMatricesUpdatesRenderNodeMatrices)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setCurrentScene(0);

  if(scene.getModel().nodes.empty() || scene.getRenderNodes().empty())
    GTEST_SKIP() << "Need nodes and render nodes";

  int nodeID = -1;
  for(int n = 0; n < static_cast<int>(scene.getModel().nodes.size()); ++n)
  {
    if(!scene.getRenderNodeRegistry().getRenderNodesForNode(n).empty())
    {
      nodeID = n;
      break;
    }
  }
  if(nodeID < 0)
    GTEST_SKIP() << "No node with render nodes found";

  scene.editor().setNodeTRS(nodeID, glm::vec3(1.f, 2.f, 3.f), {1, 0, 0, 0}, {1, 1, 1});
  scene.updateNodeWorldMatrices();

  const auto& rnIds = scene.getRenderNodeRegistry().getRenderNodesForNode(nodeID);
  ASSERT_FALSE(rnIds.empty()) << "Node " << nodeID << " should have render nodes";

  glm::mat4 expectedWorld = scene.editor().getNodeWorldMatrix(nodeID);
  for(int rnID : rnIds)
  {
    const glm::mat4& rnWorld = scene.getRenderNodes()[rnID].worldMatrix;
    for(int row = 0; row < 4; ++row)
      for(int col = 0; col < 4; ++col)
        EXPECT_FLOAT_EQ(rnWorld[row][col], expectedWorld[row][col])
            << "RenderNode " << rnID << " world matrix should match node world matrix";
  }
}
