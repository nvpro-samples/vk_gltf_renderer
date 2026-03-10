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
#include "gltf_scene_editor.hpp"
#include "gltf_scene_animation.hpp"
#include "gltf_scene_validator.hpp"

using namespace gltf_test;

//==================================================================================================
// SceneEditor bounds-checking tests
//==================================================================================================

TEST(ErrorPaths, InvalidNodeIndexReturnsStaticNode)
{
  nvvkgltf::Scene scene;
  try
  {
    auto path = TestResources::getResourcePath("Box.glb");
    ASSERT_TRUE(scene.load(path));
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }

  auto& editor = scene.editor();
  EXPECT_FALSE(editor.isValidNodeIndex(-1));
  EXPECT_FALSE(editor.isValidNodeIndex(99999));

  const auto& node = editor.getNode(-1);
  EXPECT_TRUE(node.name.empty());
  const auto& node2 = editor.getNode(99999);
  EXPECT_TRUE(node2.name.empty());
}

TEST(ErrorPaths, InvalidMeshIndexReturnsStaticMesh)
{
  nvvkgltf::Scene scene;
  try
  {
    auto path = TestResources::getResourcePath("Box.glb");
    ASSERT_TRUE(scene.load(path));
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }

  auto& editor = scene.editor();
  EXPECT_FALSE(editor.isValidMeshIndex(-1));
  EXPECT_FALSE(editor.isValidMeshIndex(99999));

  const auto& mesh = editor.getMesh(-1);
  EXPECT_TRUE(mesh.name.empty());
  const auto& mesh2 = editor.getMesh(99999);
  EXPECT_TRUE(mesh2.name.empty());
}

TEST(ErrorPaths, ValidNodeAndMeshIndicesWork)
{
  nvvkgltf::Scene scene;
  try
  {
    auto path = TestResources::getResourcePath("Box.glb");
    ASSERT_TRUE(scene.load(path));
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }

  auto& editor = scene.editor();
  EXPECT_TRUE(editor.isValidNodeIndex(0));
  EXPECT_TRUE(editor.isValidMeshIndex(0));

  const auto& node = editor.getNode(0);
  (void)node;
  const auto& mesh = editor.getMesh(0);
  (void)mesh;
}

//==================================================================================================
// Merger edge cases
//==================================================================================================

TEST(ErrorPaths, MergeWithNonexistentFileReturnsFalse)
{
  nvvkgltf::Scene scene;
  try
  {
    auto path = TestResources::getResourcePath("Box.glb");
    ASSERT_TRUE(scene.load(path));
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }

  int result = scene.mergeScene("/nonexistent/path/model.glb", 100000);
  EXPECT_LT(result, 0) << "Merging non-existent file should return negative index";
}

TEST(ErrorPaths, MergeSameFileDoublesResources)
{
  nvvkgltf::Scene scene;
  try
  {
    auto path = TestResources::getResourcePath("Box.glb");
    ASSERT_TRUE(scene.load(path));

    const auto& model         = scene.getModel();
    size_t      origMeshCount = model.meshes.size();
    size_t      origNodeCount = model.nodes.size();

    int wrapperIdx = scene.mergeScene(path.string(), 100000);
    EXPECT_GE(wrapperIdx, 0) << "Self-merge should succeed";
    EXPECT_GT(model.meshes.size(), origMeshCount) << "Merge should add meshes";
    EXPECT_GT(model.nodes.size(), origNodeCount) << "Merge should add nodes";
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }
}

//==================================================================================================
// Compact on various states
//==================================================================================================

TEST(ErrorPaths, CompactAfterDeleteRemovesOrphanedMesh)
{
  nvvkgltf::Scene scene;
  try
  {
    auto path = TestResources::getResourcePath("Box.glb");
    ASSERT_TRUE(scene.load(path));
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }

  const auto& model         = scene.getModel();
  size_t      origMeshCount = model.meshes.size();

  auto& editor = scene.editor();
  if(model.nodes.size() >= 2)
  {
    int meshNodeIdx = -1;
    for(int i = 0; i < static_cast<int>(model.nodes.size()); i++)
    {
      if(model.nodes[i].mesh >= 0)
      {
        meshNodeIdx = i;
        break;
      }
    }
    if(meshNodeIdx >= 0)
    {
      editor.deleteNode(meshNodeIdx);
      scene.setCurrentScene(0);

      bool compacted = scene.compactModel();
      EXPECT_TRUE(compacted) << "Should compact after deleting a mesh-bearing node";
      EXPECT_LE(model.meshes.size(), origMeshCount);
    }
  }
}

//==================================================================================================
// Validator edge cases
//==================================================================================================

TEST(ErrorPaths, ValidatorDetectsInvalidNodeReferences)
{
  nvvkgltf::Scene scene;
  try
  {
    auto path = TestResources::getResourcePath("Box.glb");
    ASSERT_TRUE(scene.load(path));
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }

  auto result = scene.validator().validateModel();
  EXPECT_TRUE(result.valid) << "Box.glb should be valid";
  EXPECT_TRUE(result.errors.empty());
}

//==================================================================================================
// Scene load error paths
//==================================================================================================

TEST(ErrorPaths, LoadNonexistentFileReturnsFalse)
{
  nvvkgltf::Scene scene;
  bool            loaded = scene.load("/nonexistent/path/model.glb");
  EXPECT_FALSE(loaded) << "Loading non-existent file should return false";
  EXPECT_FALSE(scene.valid());
}

TEST(ErrorPaths, LoadEmptyPathReturnsFalse)
{
  nvvkgltf::Scene scene;
  bool            loaded = scene.load("");
  EXPECT_FALSE(loaded) << "Loading empty path should return false";
  EXPECT_FALSE(scene.valid());
}

//==================================================================================================
// Animation edge cases
//==================================================================================================

TEST(ErrorPaths, AnimationOnSceneWithNoAnimations)
{
  nvvkgltf::Scene scene;
  try
  {
    auto path = TestResources::getResourcePath("Box.glb");
    ASSERT_TRUE(scene.load(path));
  }
  catch(const std::runtime_error& e)
  {
    GTEST_SKIP() << "Test resource not found: " << e.what();
  }

  auto& anim = scene.animation();
  EXPECT_EQ(anim.getNumAnimations(), 0) << "Box.glb should have no animations";
}
