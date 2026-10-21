/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// glTF 2.1 `uid` (#32): loaded and saved as it is, never checked or rewritten (the draft hasn't
// settled what a uid must satisfy). One rule is kept: a copy the editor makes is a new object and
// doesn't inherit the original's uid, so a save doesn't write the same uid twice.

#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "gltf_scene.hpp"
#include "gltf_scene_editor.hpp"
#include "gltf_scene_validator.hpp"
#include "common/test_utils.hpp"

using namespace gltf_test;

namespace {

class UidTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    const std::filesystem::path assets = TestResources::getSampleAssetsPath();
    if(assets.empty())
      GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
    m_box = assets / "Models/Box/glTF/Box.gltf";

    m_tempDir = std::filesystem::temp_directory_path() / "gltf_uid_tests";
    std::filesystem::create_directories(m_tempDir);
  }
  void TearDown() override
  {
    if(!m_tempDir.empty())
      std::filesystem::remove_all(m_tempDir);
  }

  // Box: node 0 (root) -> node 1 (mesh 0, material 0). Gives each a uid.
  void loadBoxWithUids(nvvkgltf::Scene& scene) const
  {
    ASSERT_TRUE(scene.load(m_box));
    tinygltf::Model& model = scene.getModel();
    ASSERT_GE(model.nodes.size(), 2u);
    model.nodes[0].uid     = "root";
    model.nodes[1].uid     = "box";
    model.meshes[0].uid    = "box-mesh";
    model.materials[0].uid = "box-material";
  }

  bool hasWarningContaining(const nvvkgltf::Scene& scene, const std::string& text) const
  {
    for(const std::string& warning : scene.validator().validateModel().warnings)
      if(warning.find(text) != std::string::npos)
        return true;
    return false;
  }

  std::filesystem::path m_box;
  std::filesystem::path m_tempDir;
};

// A copy is a new object: it gets no uid, and the original keeps its own.
TEST_F(UidTest, CopiesDropTheUid)
{
  nvvkgltf::Scene scene;
  loadBoxWithUids(scene);
  const tinygltf::Model& model = scene.getModel();

  // Duplicating the root copies the whole hierarchy.
  const size_t nodesBefore = model.nodes.size();
  const int    copy        = scene.editor().duplicateNode(0);
  ASSERT_GE(copy, 0);
  EXPECT_EQ(model.nodes[0].uid, "root");
  EXPECT_EQ(model.nodes[1].uid, "box");
  for(size_t i = nodesBefore; i < model.nodes.size(); ++i)
    EXPECT_TRUE(model.nodes[i].uid.empty()) << "copied node " << i;

  const int mesh = scene.editor().duplicateMeshForNode(0, 1);
  ASSERT_GE(mesh, 0);
  EXPECT_TRUE(model.meshes[mesh].uid.empty());
  EXPECT_EQ(model.meshes[0].uid, "box-mesh");

  const size_t materialsBefore = model.materials.size();
  ASSERT_GE(scene.editor().splitPrimitiveMaterial(1, 0), 0);
  ASSERT_EQ(model.materials.size(), materialsBefore + 1);
  EXPECT_TRUE(model.materials.back().uid.empty());
  EXPECT_EQ(model.materials[0].uid, "box-material");

  EXPECT_FALSE(hasWarningContaining(scene, "uid"));
}

// Merging keeps every uid as it is, even one that collides with the scene's, and reports nothing.
TEST_F(UidTest, MergeKeepsUids)
{
  const std::filesystem::path imported = m_tempDir / "imported.gltf";
  {
    nvvkgltf::Scene other;
    ASSERT_TRUE(other.load(m_box));
    other.getModel().nodes[0].uid = "root";  // the scene's root has this uid too
    ASSERT_TRUE(other.save(imported));
  }

  nvvkgltf::Scene scene;
  loadBoxWithUids(scene);
  const size_t nodesBefore = scene.getModel().nodes.size();
  ASSERT_GE(scene.mergeScene(imported), 0);

  const tinygltf::Model& model    = scene.getModel();
  int                    rootUids = 0;
  for(size_t i = 0; i < model.nodes.size(); ++i)
    rootUids += model.nodes[i].uid == "root" ? 1 : 0;
  EXPECT_EQ(rootUids, 2);
  EXPECT_GT(model.nodes.size(), nodesBefore);
  EXPECT_FALSE(hasWarningContaining(scene, "uid"));
  EXPECT_TRUE(scene.save(m_tempDir / "merged.gltf"));
}

}  // namespace
