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
#include "gltf_scene_validator.hpp"
#include "common/test_utils.hpp"
#include "nvutils/file_operations.hpp"
#include <filesystem>

using namespace gltf_test;

namespace {
// Plain tinygltf load of a .gltf, without the Scene: what is actually in the file.
tinygltf::Model readRawGltf(const std::filesystem::path& file)
{
  tinygltf::TinyGLTF loader;
  tinygltf::Model    model;
  std::string        err, warn;
  const std::string  fileUtf8 = nvutils::utf8FromPath(file);
  EXPECT_TRUE(loader.LoadASCIIFromFile(&model, &err, &warn, fileUtf8)) << fileUtf8 << ": " << err;
  return model;
}
}  // namespace

// Phase 2A: Round-Trip Testing
// Load → Validate → Save → Reload → Validate → Compare

class RoundTripTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    tempDir = std::filesystem::temp_directory_path() / "gltf_test";
    std::filesystem::create_directories(tempDir);
  }

  void TearDown() override { std::filesystem::remove_all(tempDir); }

  std::filesystem::path tempDir;

  // Load → Save → Load → Compare
  bool testRoundTrip(const std::filesystem::path& inputFile)
  {
    nvvkgltf::Scene scene1;
    if(!scene1.load(inputFile))
    {
      ADD_FAILURE() << "Failed to load input file: " << inputFile.string();
      return false;
    }

    // Validate after load
    auto validation1 = scene1.validator().validateModel();
    if(!validation1.valid)
    {
      ADD_FAILURE() << "Original file validation failed";
      validation1.print();
      return false;
    }

    // Save
    auto tempFile = tempDir / "roundtrip.gltf";
    if(!scene1.save(tempFile))
    {
      ADD_FAILURE() << "Failed to save";
      return false;
    }

    // Reload
    nvvkgltf::Scene scene2;
    if(!scene2.load(tempFile))
    {
      ADD_FAILURE() << "Failed to reload";
      return false;
    }

    // Validate after reload
    auto validation2 = scene2.validator().validateModel();
    if(!validation2.valid)
    {
      ADD_FAILURE() << "Reloaded file validation failed";
      validation2.print();
      return false;
    }

    // Compare structures
    EXPECT_EQ(scene1.getModel().nodes.size(), scene2.getModel().nodes.size());
    EXPECT_EQ(scene1.getModel().meshes.size(), scene2.getModel().meshes.size());
    EXPECT_EQ(scene1.getModel().materials.size(), scene2.getModel().materials.size());
    EXPECT_EQ(scene1.getModel().animations.size(), scene2.getModel().animations.size());
    EXPECT_EQ(scene1.getModel().skins.size(), scene2.getModel().skins.size());

    return true;
  }
};

TEST_F(RoundTripTest, SimpleCube)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  auto resourcePath = assetsPath / "Models/Box/glTF/Box.gltf";
  ASSERT_TRUE(testRoundTrip(resourcePath));
}

// Multiple scenes are deprecated in glTF 2.1 (#35), but must still load: the file's `scene` is the
// one shown, and a save keeps every scene and the default.
TEST_F(RoundTripTest, MultipleScenes)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/MultipleScenes/glTF/MultipleScenes.gltf"));
  ASSERT_EQ(scene.getModel().scenes.size(), 2u);
  // The file's "scene": 1, not the first one. Each scene holds one node: scene 0 node 0, scene 1 node 1.
  EXPECT_EQ(scene.getCurrentScene(), 1);
  ASSERT_EQ(scene.getRenderNodes().size(), 1u);
  EXPECT_EQ(scene.getRenderNodes()[0].refNodeID, 1);

  scene.setCurrentScene(0);
  EXPECT_EQ(scene.getCurrentScene(), 0);
  ASSERT_EQ(scene.getRenderNodes().size(), 1u);
  EXPECT_EQ(scene.getRenderNodes()[0].refNodeID, 0);

  ASSERT_TRUE(scene.save(tempDir / "multiple_scenes.gltf"));
  nvvkgltf::Scene reloaded;
  ASSERT_TRUE(reloaded.load(tempDir / "multiple_scenes.gltf"));
  EXPECT_EQ(reloaded.getModel().scenes.size(), 2u);
  EXPECT_EQ(reloaded.getModel().defaultScene, 1);
  EXPECT_EQ(reloaded.getModel().asset.version, "2.0");  // no 2.1 feature, so the deprecation doesn't apply
}

// Images stored as data: URIs must survive a .gltf save (they used to make the writer fail).
TEST_F(RoundTripTest, EmbeddedImages)
{
  const std::filesystem::path assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
  {
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }

  const std::filesystem::path resourcePath = assetsPath / "Models/BoxTextured/glTF-Embedded/BoxTextured.gltf";
  ASSERT_TRUE(testRoundTrip(resourcePath));

  // Still embedded with the same pixels: tinygltf decodes a data: URI on load and clears `uri`, so
  // an image with no buffer view and no file next to the .gltf (checked below) was a data: URI.
  const tinygltf::Model original = readRawGltf(resourcePath);
  const tinygltf::Model saved    = readRawGltf(tempDir / "roundtrip.gltf");
  ASSERT_EQ(saved.images.size(), original.images.size());
  for(size_t i = 0; i < saved.images.size(); i++)
  {
    EXPECT_EQ(saved.images[i].bufferView, -1) << "image " << i;
    EXPECT_EQ(saved.images[i].mimeType, original.images[i].mimeType) << "image " << i;
    EXPECT_EQ(saved.images[i].width, original.images[i].width) << "image " << i;
    EXPECT_EQ(saved.images[i].height, original.images[i].height) << "image " << i;
    EXPECT_TRUE(saved.images[i].image == original.images[i].image) << "image " << i << ": pixels differ";
  }
  for(const auto& entry : std::filesystem::directory_iterator(tempDir))
    EXPECT_NE(entry.path().extension(), ".png") << "image written to " << entry.path().string();
}

TEST_F(RoundTripTest, ValidationDetectsInvalidNode)
{
  nvvkgltf::Scene scene;

  // Create a minimal but invalid model
  scene.getModel().nodes.resize(2);
  scene.getModel().nodes[0].children.push_back(99);  // Invalid child index

  auto result = scene.validator().validateModel();
  EXPECT_FALSE(result.valid);
  EXPECT_FALSE(result.errors.empty());
}

TEST_F(RoundTripTest, ValidationDetectsSelfReference)
{
  nvvkgltf::Scene scene;

  // Create a node that references itself
  scene.getModel().nodes.resize(1);
  scene.getModel().nodes[0].children.push_back(0);  // Self-reference

  auto result = scene.validator().validateModel();
  EXPECT_FALSE(result.valid);
  EXPECT_FALSE(result.errors.empty());
}

TEST_F(RoundTripTest, ValidationDetectsCycle)
{
  nvvkgltf::Scene scene;

  // Create a cycle: 0 → 1 → 2 → 0
  scene.getModel().nodes.resize(3);
  scene.getModel().nodes[0].children.push_back(1);
  scene.getModel().nodes[1].children.push_back(2);
  scene.getModel().nodes[2].children.push_back(0);

  auto result = scene.validator().validateModel();
  EXPECT_FALSE(result.valid);
  EXPECT_FALSE(result.errors.empty());
}

TEST_F(RoundTripTest, ValidationPassesValidScene)
{
  nvvkgltf::Scene scene;

  // Create a valid simple hierarchy: 0 → 1, 0 → 2
  scene.getModel().nodes.resize(3);
  scene.getModel().nodes[0].children.push_back(1);
  scene.getModel().nodes[0].children.push_back(2);

  scene.getModel().scenes.resize(1);
  scene.getModel().scenes[0].nodes.push_back(0);
  scene.getModel().defaultScene = 0;

  auto result = scene.validator().validateModel();
  EXPECT_TRUE(result.valid);
  EXPECT_TRUE(result.errors.empty());
}

TEST_F(RoundTripTest, SaveRefusesInvalidModel)
{
  nvvkgltf::Scene scene;

  // Create an invalid model
  scene.getModel().nodes.resize(1);
  scene.getModel().nodes[0].mesh = 999;  // Invalid mesh reference

  auto tempFile   = tempDir / "invalid.gltf";
  bool saveResult = scene.save(tempFile);

  // Save should fail due to validation
  EXPECT_FALSE(saveResult);
}
