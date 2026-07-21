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

// Tests for tinygltf::utils::syncExtensionsUsed(): reconciling top-level
// extensionsUsed / extensionsRequired with the extensions actually present in a model,
// per the glTF 2.0 "Specifying Extensions" rules.

#include <filesystem>

#include <gtest/gtest.h>

#include "gltf_scene.hpp"
#include "tinygltf_utils.hpp"
#include "common/test_utils.hpp"

using namespace gltf_test;

namespace {

bool contains(const std::vector<std::string>& list, const std::string& name)
{
  for(size_t i = 0; i < list.size(); i++)
  {
    if(list[i] == name)
      return true;
  }
  return false;
}

// Number of times `name` appears in `list`.
int countOccurrences(const std::vector<std::string>& list, const std::string& name)
{
  int count = 0;
  for(size_t i = 0; i < list.size(); i++)
  {
    if(list[i] == name)
      count++;
  }
  return count;
}

// True when every entry of `required` is also present in `used` (spec invariant).
bool isSubset(const std::vector<std::string>& required, const std::vector<std::string>& used)
{
  for(size_t i = 0; i < required.size(); i++)
  {
    if(!contains(used, required[i]))
      return false;
  }
  return true;
}

// Marks an extension as present on an object by inserting an empty extension object.
void markExtension(tinygltf::ExtensionMap& ext, const std::string& name)
{
  ext[name] = tinygltf::Value(tinygltf::Value::Object());
}

}  // namespace

// extensionsUsed must be rebuilt from every place extensions can live: object-level
// material extensions, nested texture-info extensions, and node/light extensions.
TEST(ExtensionsMetadata, RecomputesUsedFromAllLocations)
{
  tinygltf::Model model;
  model.materials.push_back(tinygltf::Material{});
  model.nodes.push_back(tinygltf::Node{});

  markExtension(model.materials[0].extensions, KHR_MATERIALS_RETROREFLECTION_EXTENSION_NAME);
  markExtension(model.materials[0].pbrMetallicRoughness.baseColorTexture.extensions, KHR_TEXTURE_TRANSFORM_EXTENSION_NAME);
  markExtension(model.nodes[0].extensions, KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME);

  tinygltf::utils::syncExtensionsUsed(model);

  EXPECT_TRUE(contains(model.extensionsUsed, KHR_MATERIALS_RETROREFLECTION_EXTENSION_NAME));
  EXPECT_TRUE(contains(model.extensionsUsed, KHR_TEXTURE_TRANSFORM_EXTENSION_NAME));
  EXPECT_TRUE(contains(model.extensionsUsed, KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME));
}

// KHR_texture_transform used only on an extension-owned texture info (e.g. a clearcoat texture,
// stored as opaque JSON inside the material extension) must still be detected, not pruned.
TEST(ExtensionsMetadata, DetectsExtensionNestedOnExtensionOwnedTexture)
{
  tinygltf::Model model;
  model.materials.push_back(tinygltf::Material{});

  // Build KHR_materials_clearcoat { clearcoatTexture: { index:0, extensions:{ KHR_texture_transform:{} } } }
  tinygltf::Value::Object texExtensions;
  texExtensions[KHR_TEXTURE_TRANSFORM_EXTENSION_NAME] = tinygltf::Value(tinygltf::Value::Object());
  tinygltf::Value::Object clearcoatTexture;
  clearcoatTexture["index"]      = tinygltf::Value(0);
  clearcoatTexture["extensions"] = tinygltf::Value(texExtensions);
  tinygltf::Value::Object clearcoat;
  clearcoat["clearcoatTexture"]                                         = tinygltf::Value(clearcoatTexture);
  model.materials[0].extensions[KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME] = tinygltf::Value(clearcoat);

  tinygltf::utils::syncExtensionsUsed(model);

  EXPECT_TRUE(contains(model.extensionsUsed, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME));
  EXPECT_TRUE(contains(model.extensionsUsed, KHR_TEXTURE_TRANSFORM_EXTENSION_NAME));
}

// A name listed in extensionsUsed but not actually present anywhere must be dropped.
TEST(ExtensionsMetadata, PrunesStaleUsed)
{
  tinygltf::Model model;
  model.materials.push_back(tinygltf::Material{});
  markExtension(model.materials[0].extensions, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME);

  // Stale: no material actually uses sheen.
  model.extensionsUsed = {KHR_MATERIALS_SHEEN_EXTENSION_NAME};

  tinygltf::utils::syncExtensionsUsed(model);

  EXPECT_FALSE(contains(model.extensionsUsed, KHR_MATERIALS_SHEEN_EXTENSION_NAME));
  EXPECT_TRUE(contains(model.extensionsUsed, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME));
}

// Using an extension must never promote it into extensionsRequired.
TEST(ExtensionsMetadata, NeverAddsToRequired)
{
  tinygltf::Model model;
  model.materials.push_back(tinygltf::Material{});
  markExtension(model.materials[0].extensions, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME);
  markExtension(model.materials[0].pbrMetallicRoughness.baseColorTexture.extensions, KHR_TEXTURE_TRANSFORM_EXTENSION_NAME);

  tinygltf::utils::syncExtensionsUsed(model);

  EXPECT_TRUE(model.extensionsRequired.empty());
  EXPECT_TRUE(contains(model.extensionsUsed, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME));
}

// Footprint-less required extensions (e.g. KHR_mesh_quantization, signaled only by quantized
// accessor types) have no `extensions` entry anywhere. They must be preserved in
// extensionsRequired and forced into extensionsUsed to keep the subset invariant.
TEST(ExtensionsMetadata, PreservesFootprintlessRequired)
{
  tinygltf::Model model;
  model.materials.push_back(tinygltf::Material{});
  markExtension(model.materials[0].extensions, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME);

  model.extensionsUsed     = {};
  model.extensionsRequired = {"KHR_mesh_quantization"};

  tinygltf::utils::syncExtensionsUsed(model);

  EXPECT_TRUE(contains(model.extensionsRequired, "KHR_mesh_quantization"));
  EXPECT_TRUE(contains(model.extensionsUsed, "KHR_mesh_quantization"));
  EXPECT_TRUE(contains(model.extensionsUsed, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME));
  EXPECT_TRUE(isSubset(model.extensionsRequired, model.extensionsUsed));
}

// extensionsRequired is preserved even when the required extension is not detected by the walk;
// it is never silently dropped, and the subset invariant is maintained.
TEST(ExtensionsMetadata, RequiredAlwaysSubsetOfUsed)
{
  tinygltf::Model model;
  model.nodes.push_back(tinygltf::Node{});
  markExtension(model.nodes[0].extensions, KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME);

  model.extensionsRequired = {KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME, "KHR_mesh_quantization"};

  tinygltf::utils::syncExtensionsUsed(model);

  EXPECT_TRUE(isSubset(model.extensionsRequired, model.extensionsUsed));
  EXPECT_TRUE(contains(model.extensionsRequired, KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME));
  EXPECT_TRUE(contains(model.extensionsRequired, "KHR_mesh_quantization"));
}

// Duplicate entries in the input arrays must be collapsed.
TEST(ExtensionsMetadata, DeduplicatesEntries)
{
  tinygltf::Model model;
  model.nodes.push_back(tinygltf::Node{});
  markExtension(model.nodes[0].extensions, KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME);

  model.extensionsUsed     = {KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME, KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME};
  model.extensionsRequired = {KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME, KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME};

  tinygltf::utils::syncExtensionsUsed(model);

  EXPECT_EQ(countOccurrences(model.extensionsUsed, KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME), 1);
  EXPECT_EQ(countOccurrences(model.extensionsRequired, KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME), 1);
}

// End-to-end: an added material extension appears in the saved file's extensionsUsed and a stale
// entry is pruned, verified by reloading the written asset.
TEST(ExtensionsMetadata, SaveReloadReconcilesUsed)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  if(!scene.load(assetsPath / "Models/Box/glTF/Box.gltf"))
    GTEST_SKIP() << "Failed to load Box.gltf";

  tinygltf::Model& model = scene.getModel();
  if(model.materials.empty())
    GTEST_SKIP() << "Box.gltf has no materials";

  // Inject a stale used entry and add a real material extension.
  model.extensionsUsed.push_back(KHR_MATERIALS_SHEEN_EXTENSION_NAME);
  tinygltf::utils::setIor(model.materials[0], KHR_materials_ior{1.4f});

  auto tempDir = std::filesystem::temp_directory_path() / "gltf_ext_metadata_test";
  std::filesystem::create_directories(tempDir);
  auto tempFile = tempDir / "box_ext.gltf";
  ASSERT_TRUE(scene.save(tempFile));

  nvvkgltf::Scene reloaded;
  ASSERT_TRUE(reloaded.load(tempFile));
  const tinygltf::Model& m2 = reloaded.getModel();

  EXPECT_TRUE(contains(m2.extensionsUsed, KHR_MATERIALS_IOR_EXTENSION_NAME));
  EXPECT_FALSE(contains(m2.extensionsUsed, KHR_MATERIALS_SHEEN_EXTENSION_NAME));
  EXPECT_TRUE(isSubset(m2.extensionsRequired, m2.extensionsUsed));

  std::filesystem::remove_all(tempDir);
}
