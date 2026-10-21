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

// glTF 2.1 conformance against the fixtures that tests/gltf_2_1/generate_fixtures.py writes into the
// build tree (see tests/CMakeLists.txt).
// Each test names the GitLab issue it covers; docs/gltf_2_1.md tracks overall status.

#include <algorithm>
#include <filesystem>
#include <functional>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>

#include <gtest/gtest.h>

#include "gltf_scene.hpp"
#include "gltf_scene_animation.hpp"
#include "gltf_scene_editor.hpp"
#include "gltf_version.hpp"
#include "nvutils/file_operations.hpp"
#include "tinygltf_utils.hpp"

namespace {

const std::filesystem::path kFixtures = GLTF_2_1_FIXTURES_DIR;

// A per-test scratch folder, removed on teardown.
class Gltf21Test : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if(!std::filesystem::exists(kFixtures / ".generated"))
      GTEST_SKIP() << "glTF 2.1 fixtures not generated (needs Python 3 at configure time): " << kFixtures.string();
    const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
    m_tempDir                       = std::filesystem::temp_directory_path() / "gltf_2_1_tests"
                / (std::string(info->test_suite_name()) + "_" + info->name());
    std::filesystem::remove_all(m_tempDir);
    std::filesystem::create_directories(m_tempDir);
  }
  void TearDown() override
  {
    if(!m_tempDir.empty())
      std::filesystem::remove_all(m_tempDir);
  }

  std::filesystem::path m_tempDir{};
};

// Plain tinygltf load, without the Scene's external-asset resolution: what is actually on disk.
tinygltf::Model readRaw(const std::filesystem::path& file)
{
  tinygltf::TinyGLTF loader;
  tinygltf::Model    model;
  std::string        err, warn;
  const std::string  fileUtf8 = nvutils::utf8FromPath(file);
  const bool         ok       = file.extension() == ".glb" ? loader.LoadBinaryFromFile(&model, &err, &warn, fileUtf8) :
                                                             loader.LoadASCIIFromFile(&model, &err, &warn, fileUtf8);
  EXPECT_TRUE(ok) << fileUtf8 << ": " << err;
  return model;
}

int countExternalAssetNodes(const tinygltf::Model& model)
{
  int count = 0;
  for(size_t i = 0; i < model.nodes.size(); i++)
    count += model.nodes[i].externalAsset >= 0 ? 1 : 0;
  return count;
}

//--------------------------------------------------------------------------------------------------
// Core properties (#26, #32, #36, #37, #38)
//--------------------------------------------------------------------------------------------------

// Every 2.1 core property in core_properties.gltf, checked on a loaded model.
void expectCoreProperties(const tinygltf::Model& model)
{
  EXPECT_EQ(model.asset.version, "2.1");

  // #38 thumbnail: an image referenced only by asset.thumbnail.
  ASSERT_GE(model.asset.thumbnail, 0);
  ASSERT_LT(model.asset.thumbnail, static_cast<int>(model.images.size()));
  EXPECT_EQ(model.images[model.asset.thumbnail].uid, "thumbnail-image");

  // #36 shapes: one of each type, in order.
  ASSERT_EQ(model.shapes.size(), 5u);
  EXPECT_EQ(model.shapes[0].type, "box");
  EXPECT_EQ(model.shapes[0].box.size, (std::vector<double>{2.0, 2.0, 1.0}));
  EXPECT_EQ(model.shapes[1].type, "sphere");
  EXPECT_DOUBLE_EQ(model.shapes[1].sphere.radius, 0.75);
  EXPECT_EQ(model.shapes[2].type, "capsule");
  EXPECT_DOUBLE_EQ(model.shapes[2].capsule.radiusBottom, 0.5);
  EXPECT_EQ(model.shapes[3].type, "cylinder");
  EXPECT_EQ(model.shapes[4].type, "plane");
  EXPECT_DOUBLE_EQ(model.shapes[4].plane.sizeX, 4.0);

  // #32 uid and #37 bounding volumes, on the two-level hierarchy. Scene::load appends a default
  // camera node after them when the file has no camera.
  ASSERT_GE(model.nodes.size(), 2u);
  EXPECT_EQ(model.nodes[0].uid, "root-node");
  EXPECT_EQ(model.nodes[0].boundingVolume.shape, 0);
  EXPECT_EQ(model.nodes[0].boundingVolume.translation, (std::vector<double>{0.5, 0.0, 0.0}));
  EXPECT_EQ(model.nodes[1].uid, "quad-node");
  EXPECT_EQ(model.nodes[1].boundingVolume.shape, 1);
  ASSERT_EQ(model.meshes.size(), 1u);
  EXPECT_EQ(model.meshes[0].uid, "quad-mesh");
  ASSERT_EQ(model.materials.size(), 1u);
  EXPECT_EQ(model.materials[0].uid, "quad-mesh-material");
  ASSERT_EQ(model.scenes.size(), 1u);
  EXPECT_EQ(model.scenes[0].uid, "main-scene");
}

TEST_F(Gltf21Test, CoreProperties_Load)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "core_properties.gltf"));
  expectCoreProperties(scene.getModel());
}

// Load -> save -> reload keeps every 2.1 core property, in both container formats.
TEST_F(Gltf21Test, CoreProperties_RoundTrip)
{
  for(const char* name : {"core_properties.gltf", "core_properties.glb"})
  {
    SCOPED_TRACE(name);
    nvvkgltf::Scene scene;
    ASSERT_TRUE(scene.load(kFixtures / "core_properties.gltf"));
    const std::filesystem::path saved = m_tempDir / name;
    ASSERT_TRUE(scene.save(saved));
    expectCoreProperties(readRaw(saved));
  }
}

//--------------------------------------------------------------------------------------------------
// External assets (#33)
//--------------------------------------------------------------------------------------------------

// root.gltf: parts/part.gltf instanced three times, plus parts/nested.gltf -> parts/leaf/leaf.gltf.
constexpr size_t kRootRenderNodes      = 4;  // three part quads + one leaf quad
constexpr size_t kRootRenderPrimitives = 2;  // part geometry is shared by its three instances

TEST_F(Gltf21Test, ExternalAssets_InstancesShareGeometry)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "external/root.gltf"));
  ASSERT_TRUE(scene.valid());
  EXPECT_TRUE(scene.hasExternalAssets());

  // The nested reference resolved relative to parts/, not to the root: the leaf quad is there.
  EXPECT_EQ(scene.getRenderNodes().size(), kRootRenderNodes);
  // One RenderPrimitive (one BLAS) per unique geometry, however many instances.
  EXPECT_EQ(scene.getRenderPrimitives().size(), kRootRenderPrimitives);

  // The four instance nodes stay editable. The part is static, so its three placements draw it as
  // instance sets, without nodes; nested.gltf places a file itself, so it is copied as read-only nodes.
  const tinygltf::Model& model = scene.getModel();
  for(int i = 0; i < 4; ++i)
  {
    EXPECT_TRUE(scene.isExternalAssetInstance(i)) << "node " << i;
    EXPECT_FALSE(scene.isNodeReadOnly(i)) << "node " << i;
    EXPECT_EQ(scene.getInstanceSet(i) != nullptr, i < 3) << "node " << i;
    const std::vector<int>& children = model.nodes[i].children;
    EXPECT_EQ(children.empty(), i < 3) << "node " << i;
    for(size_t c = 0; c < children.size(); c++)
      EXPECT_TRUE(scene.isNodeReadOnly(children[c])) << "child " << children[c] << " of node " << i;
  }
}

// Default save re-externalizes: the written file holds the references, not the merged content.
TEST_F(Gltf21Test, ExternalAssets_SaveKeepsReferences)
{
  // Save next to the referenced files: saving elsewhere does not relocate them (docs/external_assets.md).
  std::filesystem::copy(kFixtures / "external", m_tempDir, std::filesystem::copy_options::recursive);
  const std::filesystem::path saved = m_tempDir / "root_saved.gltf";
  {
    nvvkgltf::Scene scene;
    ASSERT_TRUE(scene.load(m_tempDir / "root.gltf"));
    ASSERT_TRUE(scene.save(saved));
  }

  const tinygltf::Model raw = readRaw(saved);
  EXPECT_EQ(raw.files.size(), 2u);
  EXPECT_EQ(raw.externalAssets.size(), 2u);
  EXPECT_EQ(countExternalAssetNodes(raw), 4);
  EXPECT_TRUE(raw.meshes.empty());
  for(size_t i = 0; i < raw.nodes.size(); i++)
    EXPECT_FALSE(raw.nodes[i].extras.Has(nvvkgltf::Scene::kExternalAssetContentKey)) << raw.nodes[i].name;

  nvvkgltf::Scene reloaded;
  ASSERT_TRUE(reloaded.load(saved));
  EXPECT_EQ(reloaded.getRenderNodes().size(), kRootRenderNodes);
  EXPECT_EQ(reloaded.getRenderPrimitives().size(), kRootRenderPrimitives);
}

// Self-contained save flattens: no references left, same content.
TEST_F(Gltf21Test, ExternalAssets_SaveFlattened)
{
  const std::filesystem::path saved = m_tempDir / "root_flat.gltf";
  {
    nvvkgltf::Scene scene;
    ASSERT_TRUE(scene.load(kFixtures / "external/root.gltf"));
    ASSERT_TRUE(scene.save(saved, /*selfContained=*/true));
  }

  const tinygltf::Model raw = readRaw(saved);
  EXPECT_TRUE(raw.externalAssets.empty());
  EXPECT_EQ(countExternalAssetNodes(raw), 0);

  nvvkgltf::Scene reloaded;
  ASSERT_TRUE(reloaded.load(saved));
  EXPECT_FALSE(reloaded.hasExternalAssets());
  EXPECT_EQ(reloaded.getRenderNodes().size(), kRootRenderNodes);
}

// cycle_a.gltf <-> cycle_b.gltf: the back-edge is skipped, loading terminates, both quads remain.
TEST_F(Gltf21Test, ExternalAssets_CycleIsSkipped)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "external/cycle_a.gltf"));
  EXPECT_EQ(scene.getRenderNodes().size(), 2u);
}

// The image each texture of shared_images/root.gltf reads, by texture name (MSFT_texture_dds
// "b_dds" through its extension source).
std::map<std::string, int> textureImages(const tinygltf::Model& model)
{
  std::map<std::string, int> images;
  for(const tinygltf::Texture& texture : model.textures)
    images[texture.name] = tinygltf::utils::getTextureImageIndex(texture);
  return images;
}

// Files reading the same image file share one scene image; a data: image and another file with the
// same relative name do not.
void expectSharedImages(const tinygltf::Model& model)
{
  EXPECT_EQ(model.images.size(), 3u);
  ASSERT_EQ(model.textures.size(), 5u);
  const std::map<std::string, int> images = textureImages(model);
  for(const char* name : {"a_tex", "b_tex", "b_dds", "b_data", "c_tex"})
    ASSERT_TRUE(images.contains(name)) << name;  // at() below must not miss
  ASSERT_GE(images.at("a_tex"), 0);
  EXPECT_EQ(images.at("b_tex"), images.at("a_tex"));
  EXPECT_EQ(images.at("b_dds"), images.at("a_tex"));
  EXPECT_NE(images.at("b_data"), images.at("a_tex"));
  EXPECT_NE(images.at("c_tex"), images.at("a_tex"));
  EXPECT_NE(images.at("c_tex"), images.at("b_data"));
  ASSERT_GE(images.at("b_data"), 0);
  EXPECT_EQ(model.images[images.at("b_data")].uri.find("colormap"), std::string::npos);
}

TEST_F(Gltf21Test, ExternalAssets_FilesShareImages)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "external/shared_images/root.gltf"));
  EXPECT_EQ(scene.getRenderNodes().size(), 3u);
  expectSharedImages(scene.getModel());
}

// Both save forms keep the sharing: re-externalized, the files bring their images again on reload;
// flattened, the shared image is written once (and the two colormap.png files stay apart).
TEST_F(Gltf21Test, ExternalAssets_SharedImagesRoundTrip)
{
  std::filesystem::copy(kFixtures / "external/shared_images", m_tempDir, std::filesystem::copy_options::recursive);
  const std::filesystem::path saved = m_tempDir / "root_saved.gltf";
  const std::filesystem::path flat  = m_tempDir / "flat" / "root_flat.gltf";
  std::filesystem::create_directories(flat.parent_path());
  {
    nvvkgltf::Scene scene;
    ASSERT_TRUE(scene.load(m_tempDir / "root.gltf"));
    ASSERT_TRUE(scene.save(saved));
    ASSERT_TRUE(scene.save(flat, /*selfContained=*/true));
  }

  EXPECT_TRUE(readRaw(saved).images.empty());
  nvvkgltf::Scene reloaded;
  ASSERT_TRUE(reloaded.load(saved));
  expectSharedImages(reloaded.getModel());

  const tinygltf::Model raw = readRaw(flat);
  expectSharedImages(raw);
  for(const tinygltf::Image& image : raw.images)
    if(!image.uri.empty() && image.uri.rfind("data:", 0) != 0)
      EXPECT_TRUE(
          std::filesystem::exists(flat.parent_path() / nvutils::pathFromUtf8(tinygltf::utils::decodePathFromUri(image.uri))))
          << image.uri;
  nvvkgltf::Scene flatReloaded;
  ASSERT_TRUE(flatReloaded.load(flat));
  expectSharedImages(flatReloaded.getModel());
}

//--------------------------------------------------------------------------------------------------
// How placed content is drawn: fixtures in placement/, see generate_fixtures.py
//--------------------------------------------------------------------------------------------------

// What each render node draws, as "material@x,y,z" (world translation, rounded), sorted: a
// placement-independent picture of the scene to compare loads and saves with.
std::vector<std::string> drawn(const nvvkgltf::Scene& scene)
{
  const tinygltf::Model&   model = scene.getModel();
  std::vector<std::string> result;
  for(const nvvkgltf::RenderNode& rn : scene.getRenderNodes())
  {
    const glm::vec3   t        = glm::vec3(rn.worldMatrix[3]);
    const std::string material = rn.materialID >= 0 ? model.materials[rn.materialID].name : std::string("<default>");
    result.push_back(fmt::format("{}@{:.2f},{:.2f},{:.2f}", material, t.x, t.y, t.z));
  }
  std::sort(result.begin(), result.end());
  return result;
}

const std::vector<std::string> kStaticDrawn = {
    "local_material@0.00,3.00,5.00",                                // shelf_0's local child
    "piece_paint@0.00,0.50,0.00",    "piece_paint@2.00,0.50,0.00",  // placement . piece root offset
    "piece_paint@4.00,0.50,0.00",    "piece_trim@0.00,0.50,0.00",    "piece_trim@2.00,0.50,0.00",
    "piece_trim@4.00,0.50,0.00",     "shelf_material@0.00,1.00,5.00", "shelf_material@0.00,2.00,5.00",
    "shelf_material@4.00,2.00,5.00", "shelf_material@4.00,4.00,5.00",  // shelf_1 is scaled by 2
};

// static.gltf: static placed trees are instance sets, one entry per mesh node of the tree, with no
// node per placed node. The placing nodes and the local child stay editable; picking an entry
// selects the placing node.
TEST_F(Gltf21Test, Placement_StaticTreesAreInstanceSets)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "placement/static.gltf"));
  EXPECT_EQ(drawn(scene), kStaticDrawn);
  EXPECT_EQ(scene.getRenderPrimitives().size(), 5u);  // piece paint and trim, shelf, box, local quad

  const tinygltf::Model& model = scene.getModel();
  for(int placing = 0; placing < 5; ++placing)
  {
    const nvvkgltf::InstanceSet* set = scene.getInstanceSet(placing);
    ASSERT_NE(set, nullptr) << model.nodes[placing].name;
    EXPECT_EQ(set->entries.size(), placing < 3 ? 1u : 2u) << model.nodes[placing].name;
    EXPECT_FALSE(scene.isNodeReadOnly(placing));
    EXPECT_TRUE(scene.isMeshReadOnly(set->entries[0].mesh));  // placed content, though no node draws it
  }
  EXPECT_EQ(model.nodes[3].children, std::vector<int>{5});
  EXPECT_TRUE(model.nodes[4].children.empty());
  EXPECT_FALSE(scene.isNodeReadOnly(5));

  int setRenderNodes = 0;
  for(size_t rn = 0; rn < scene.getRenderNodes().size(); rn++)
    setRenderNodes += scene.isInstanceSetRenderNode(static_cast<int>(rn)) ? 1 : 0;
  EXPECT_EQ(setRenderNodes, 10);  // everything but the local child
}

// Make Editable turns every placement of the file into nodes of the scene, drawn as before and no
// longer read-only. Placements of other files keep their instance sets.
TEST_F(Gltf21Test, Placement_MakeEditableMaterializesInstanceSets)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "placement/static.gltf"));
  ASSERT_TRUE(scene.editor().makeExternalAssetEditable(0));
  EXPECT_EQ(drawn(scene), kStaticDrawn);

  const tinygltf::Model& model = scene.getModel();
  for(int placing = 0; placing < 3; ++placing)
  {
    EXPECT_EQ(scene.getInstanceSet(placing), nullptr);
    ASSERT_EQ(model.nodes[placing].children.size(), 1u);
    const int child = model.nodes[placing].children[0];
    EXPECT_FALSE(scene.isNodeReadOnly(child));
    EXPECT_FALSE(scene.isMeshReadOnly(model.nodes[child].mesh));
  }
  EXPECT_NE(scene.getInstanceSet(3), nullptr);
  EXPECT_NE(scene.getInstanceSet(4), nullptr);
}

// Default save writes the placing nodes and only the scene's own mesh; self-contained save writes
// the instance sets as plain nodes. Both reload to the same picture.
TEST_F(Gltf21Test, Placement_SaveRoundTrips)
{
  std::filesystem::copy(kFixtures / "placement", m_tempDir, std::filesystem::copy_options::recursive);
  const std::filesystem::path saved = m_tempDir / "static_saved.gltf";
  const std::filesystem::path flat  = m_tempDir / "static_flat.gltf";
  {
    nvvkgltf::Scene scene;
    ASSERT_TRUE(scene.load(m_tempDir / "static.gltf"));
    ASSERT_TRUE(scene.save(saved));
    ASSERT_TRUE(scene.save(flat, /*selfContained=*/true));
  }

  const tinygltf::Model raw = readRaw(saved);
  ASSERT_EQ(raw.meshes.size(), 1u);
  EXPECT_EQ(raw.meshes[0].name, "local");
  EXPECT_EQ(countExternalAssetNodes(raw), 5);
  const tinygltf::Model rawFlat = readRaw(flat);
  EXPECT_EQ(countExternalAssetNodes(rawFlat), 0);
  for(const tinygltf::Model* model : {&raw, &rawFlat})
    for(const tinygltf::Node& node : model->nodes)
      EXPECT_FALSE(node.extras.Has(nvvkgltf::Scene::kInstanceSetKey)) << node.name;

  for(const std::filesystem::path& file : {saved, flat})
  {
    nvvkgltf::Scene reloaded;
    ASSERT_TRUE(reloaded.load(file));
    EXPECT_EQ(drawn(reloaded), kStaticDrawn) << file.filename().string();
  }

  // The flattened nodes keep what the shelf file authored on them: name and extras, once per placement.
  int boxes = 0;
  for(size_t n = 0; n < rawFlat.nodes.size(); n++)
  {
    const tinygltf::Node& node = rawFlat.nodes[n];
    if(node.name != "box")
      continue;
    ASSERT_TRUE(node.extras.Has("sku")) << n;
    EXPECT_EQ(node.extras.Get("sku").Get<std::string>(), "box-1");
    boxes++;
  }
  EXPECT_EQ(boxes, 2);
}

// Compact Scene keeps the meshes only instance sets draw, and their entries follow the renumbering.
TEST_F(Gltf21Test, Placement_CompactKeepsInstanceSets)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "placement/static.gltf"));
  const size_t meshes = scene.getModel().meshes.size();
  // The placed files' buffers make it compact (into one buffer) though nothing is orphaned.
  ASSERT_TRUE(scene.compactModel());
  scene.setCurrentScene(scene.getCurrentScene());  // rebuild render nodes, as the app does after compaction
  EXPECT_EQ(scene.getModel().meshes.size(), meshes);
  EXPECT_EQ(drawn(scene), kStaticDrawn);
}

// pointer.gltf: the default save drops the placed content but keeps the scene's own
// KHR_animation_pointer channel, still pointing at its node.
TEST_F(Gltf21Test, Placement_SaveKeepsPointerChannels)
{
  std::filesystem::copy(kFixtures / "placement", m_tempDir, std::filesystem::copy_options::recursive);
  const std::filesystem::path saved = m_tempDir / "pointer_saved.gltf";
  {
    nvvkgltf::Scene scene;
    ASSERT_TRUE(scene.load(m_tempDir / "pointer.gltf"));
    ASSERT_TRUE(scene.save(saved));
  }
  const tinygltf::Model raw = readRaw(saved);
  ASSERT_EQ(raw.animations.size(), 1u);
  ASSERT_EQ(raw.animations[0].channels.size(), 1u);
  const tinygltf::AnimationChannel& channel = raw.animations[0].channels[0];
  EXPECT_EQ(channel.target_path, "pointer");
  ASSERT_TRUE(channel.target_extensions.contains("KHR_animation_pointer"));
  const std::string pointer = channel.target_extensions.at("KHR_animation_pointer").Get("pointer").Get<std::string>();
  ASSERT_EQ(pointer.rfind("/nodes/", 0), 0u) << pointer;
  const int node = tinygltf::utils::parsePointerIndexAndRest(pointer, 7).first;
  ASSERT_GE(node, 0);
  ASSERT_LT(node, static_cast<int>(raw.nodes.size()));
  EXPECT_EQ(raw.nodes[node].name, "own");
}

// shared_kit.gltf: two files place the same kit; it is loaded once for the scene, so its geometry
// is two RenderPrimitives (BLAS) for the eight drawn primitives, not one set per placing file.
TEST_F(Gltf21Test, Placement_KitSharedAcrossFiles)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "placement/shared_kit.gltf"));
  EXPECT_EQ(scene.getRenderNodes().size(), 8u);
  EXPECT_EQ(scene.getRenderPrimitives().size(), 2u);
  EXPECT_EQ(scene.getModel().meshes.size(), 1u);
}

// placement/library: files that each declare their own accessors over one parts.bin, the layout of a
// glTF 2.0 library. The .bin is held once; the parts both districts read in the same format are one
// RenderPrimitive each, though their accessors differ. The morphed flag and the skinned rigs keep one
// per file: morphing and skinning write the RenderPrimitive's vertices.
TEST_F(Gltf21Test, Placement_LibraryFilesShareGeometry)
{
  std::filesystem::copy(kFixtures / "placement/library", m_tempDir, std::filesystem::copy_options::recursive);
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(m_tempDir / "library.gltf"));
  const tinygltf::Model& model = scene.getModel();
  EXPECT_EQ(std::count_if(model.buffers.begin(), model.buffers.end(),
                          [](const tinygltf::Buffer& b) { return b.uri.find("parts") != std::string::npos; }),
            1);

  ASSERT_EQ(scene.getRenderNodes().size(), 12u);
  EXPECT_EQ(scene.getRenderPrimitives().size(), 6u);
  std::map<std::string, std::set<int>> primitivesByMaterial;
  for(const nvvkgltf::RenderNode& rn : scene.getRenderNodes())
    primitivesByMaterial[model.materials[rn.materialID].name].insert(rn.renderPrimID);
  EXPECT_EQ(primitivesByMaterial["part_material"].size(), 2u);   // big and small, from both districts
  EXPECT_EQ(primitivesByMaterial["flag_material"].size(), 2u);   // morphed: one per district
  ASSERT_EQ(primitivesByMaterial["rig_a_material"].size(), 1u);  // skinned: one per file
  EXPECT_NE(primitivesByMaterial["rig_a_material"], primitivesByMaterial["rig_b_material"]);

  // The per-mesh table agrees: both "part" meshes resolve to the same two render primitives.
  std::vector<std::vector<int>> parts;
  for(int mesh = 0; mesh < static_cast<int>(model.meshes.size()); mesh++)
    if(model.meshes[mesh].name == "part")
      parts.push_back({scene.getMeshRenderPrimitive(mesh, 0), scene.getMeshRenderPrimitive(mesh, 1)});
  ASSERT_EQ(parts.size(), 2u);
  EXPECT_EQ(parts[0], parts[1]);

  // Saved back to placements, the file reloads to the same picture.
  const std::vector<std::string> before = drawn(scene);
  const std::filesystem::path    saved  = m_tempDir / "library_saved.gltf";
  ASSERT_TRUE(scene.save(saved));
  nvvkgltf::Scene reloaded;
  ASSERT_TRUE(reloaded.load(saved));
  EXPECT_EQ(drawn(reloaded), before);
  EXPECT_EQ(reloaded.getRenderPrimitives().size(), 6u);
}

// placement/library/aliased.gltf: no external asset, so primitives keep their accessor identity even
// when two read the same bytes: rendering of plain glTF files never changes.
TEST_F(Gltf21Test, Placement_NoExternalAssetsKeepAccessorIdentity)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "placement/library/aliased.gltf"));
  EXPECT_EQ(scene.getRenderNodes().size(), 2u);
  EXPECT_EQ(scene.getRenderPrimitives().size(), 2u);
}

// The y of every walker body drawn (animated from 0 to 2) at `time` seconds of the walk, sorted.
std::vector<float> walkerHeights(nvvkgltf::Scene& scene, float time)
{
  nvvkgltf::AnimationInfo& info = scene.animation().getAnimationInfo(0);
  info.currentTime              = time;
  EXPECT_TRUE(scene.animation().updateAnimation(0));
  scene.updateNodeWorldMatrices();
  std::vector<float> heights;
  for(const nvvkgltf::RenderNode& rn : scene.getRenderNodes())
    if(scene.getModel().meshes[scene.getRenderPrimitive(rn.renderPrimID).meshID].name == "walker_body")
      heights.push_back(rn.worldMatrix[3].y);
  std::sort(heights.begin(), heights.end());
  return heights;
}

// animated.gltf: animated content is copied as nodes, not drawn as an instance set; each placement
// has a channel of its own, from the one animation the file brings.
TEST_F(Gltf21Test, Placement_AnimatedContentIsNodes)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "placement/animated.gltf"));
  ASSERT_EQ(scene.animation().getNumAnimations(), 1);
  const tinygltf::Model& model = scene.getModel();
  ASSERT_EQ(model.animations.size(), 1u);
  EXPECT_EQ(model.animations[0].channels.size(), 2u);
  for(int placing = 0; placing < 2; ++placing)
  {
    EXPECT_EQ(scene.getInstanceSet(placing), nullptr);
    EXPECT_EQ(model.nodes[placing].children.size(), 1u);
  }
  EXPECT_EQ(walkerHeights(scene, 0.25f), std::vector<float>(2, 0.5f));
  EXPECT_EQ(walkerHeights(scene, 0.75f), std::vector<float>(2, 1.5f));
}

// Where each render node draws its vertices, as "node@x,y,z" (world-space centre, rounded), sorted.
// Skinned and morphed vertices are the CPU path's, so every node shows the pose it actually draws.
std::vector<std::string> drawnCentres(nvvkgltf::Scene& scene)
{
  nvvkgltf::AnimationSystem& anim  = scene.animation();
  const tinygltf::Model&     model = scene.getModel();
  scene.updateNodeWorldMatrices();
  anim.computeMorphTargets();
  anim.computeSkinning();

  const auto deformed = [&anim](int renderPrimID) -> const std::vector<glm::vec3>* {
    const std::vector<nvvkgltf::SkinTask>& tasks = anim.getSkinTasks();
    for(size_t i = 0; i < tasks.size(); i++)
      if(tasks[i].renderPrimID == renderPrimID)
        return &anim.getSkinningResult(i).positions;
    const std::vector<uint32_t>& morphs = anim.getMorphPrimitives();
    for(size_t i = 0; i < morphs.size(); i++)
      if(static_cast<int>(morphs[i]) == renderPrimID)
        return &anim.getMorphResult(i).blendedPositions;
    return nullptr;
  };

  std::vector<std::string> result;
  for(const nvvkgltf::RenderNode& rn : scene.getRenderNodes())
  {
    std::vector<glm::vec3>        storage;
    const std::vector<glm::vec3>* positions = deformed(rn.renderPrimID);
    std::span<const glm::vec3>    local =
        positions != nullptr ?
               std::span<const glm::vec3>(*positions) :
               tinygltf::utils::getAttributeData3(model, *scene.getRenderPrimitive(rn.renderPrimID).pPrimitive, "POSITION", &storage);
    glm::vec3 centre(0.0f);
    for(const glm::vec3& p : local)
      centre += glm::vec3(rn.worldMatrix * glm::vec4(p, 1.0f));
    centre /= std::max<size_t>(local.size(), 1);
    result.push_back(fmt::format("{}@{:.2f},{:.2f},{:.2f}", model.nodes[rn.refNodeID].name, centre.x, centre.y, centre.z));
  }
  std::sort(result.begin(), result.end());
  return result;
}

// deform/two_skins.gltf: one skinned mesh drawn by two nodes with skins of their own, and by an
// unskinned node. Each skinned node draws its own skin's pose; the unskinned one the bind pose.
TEST_F(Gltf21Test, Deform_EachSkinDrawsItsOwnPose)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "deform/two_skins.gltf"));
  EXPECT_EQ(scene.getRenderPrimitives().size(), 3u);
  EXPECT_EQ(scene.animation().getSkinTasks().size(), 2u);
  EXPECT_EQ(drawnCentres(scene),
            (std::vector<std::string>{"body_a@0.00,1.00,0.00", "body_b@4.00,3.00,0.00", "still@8.00,0.00,0.00"}));
}

// deform/two_weights.gltf: one morphed mesh drawn by three nodes. A node's weights override the
// mesh's, and a weights channel animates the node it targets, not the other nodes drawing the mesh.
TEST_F(Gltf21Test, Deform_EachNodeDrawsItsOwnMorphWeights)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "deform/two_weights.gltf"));
  EXPECT_EQ(scene.getRenderPrimitives().size(), 3u);
  EXPECT_EQ(drawnCentres(scene), (std::vector<std::string>{"half@4.00,0.50,0.00", "raised@2.00,1.00,0.00", "rest@0.00,0.00,0.00"}));

  scene.animation().getAnimationInfo(0).currentTime = 0.5f;
  EXPECT_TRUE(scene.animation().updateAnimation(0));
  EXPECT_EQ(drawnCentres(scene), (std::vector<std::string>{"half@4.00,0.50,0.00", "raised@2.00,0.50,0.00", "rest@0.00,0.00,0.00"}));
}

// deform/mesh_weights.gltf: three meshes over one morphed primitive. Each draws its own mesh's
// default weights, and a mesh without weights (all 0) still animates when a channel gives its node some.
TEST_F(Gltf21Test, Deform_EachMeshDrawsItsOwnDefaultWeights)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "deform/mesh_weights.gltf"));
  EXPECT_EQ(scene.getRenderPrimitives().size(), 3u);
  EXPECT_EQ(drawnCentres(scene),
            (std::vector<std::string>{"one@2.00,1.00,0.00", "unweighted@4.00,0.00,0.00", "zero@0.00,0.00,0.00"}));

  scene.animation().getAnimationInfo(0).currentTime = 0.5f;
  EXPECT_TRUE(scene.animation().updateAnimation(0));
  EXPECT_EQ(drawnCentres(scene),
            (std::vector<std::string>{"one@2.00,1.00,0.00", "unweighted@4.00,0.50,0.00", "zero@0.00,0.00,0.00"}));
}

// deform/two_targets.gltf: two meshes over the same base accessors with different morph targets
// are different geometry, so they don't share a render primitive.
TEST_F(Gltf21Test, Deform_DifferentTargetsAreDifferentGeometry)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "deform/two_targets.gltf"));
  EXPECT_EQ(scene.getRenderPrimitives().size(), 2u);
  EXPECT_EQ(drawnCentres(scene), (std::vector<std::string>{"up_1@0.00,1.00,0.00", "up_2@2.00,2.00,0.00"}));
}

// placement/skinned.gltf: a skinned file placed twice. Read-only placements of one file animate in
// lockstep, so they share the file's skinned render primitive and its skinning, each drawn at its
// own placement. Once editable, each copy can take its own pose: moving one joint moves only its quad.
TEST_F(Gltf21Test, Placement_SkinnedFilePlacedTwice)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "placement/skinned.gltf"));
  EXPECT_EQ(scene.getRenderPrimitives().size(), 1u);
  EXPECT_EQ(scene.animation().getSkinTasks().size(), 1u);
  EXPECT_EQ(drawnCentres(scene), (std::vector<std::string>{"rig_body@0.00,1.00,0.00", "rig_body@4.00,1.00,0.00"}));

  ASSERT_TRUE(scene.editor().makeExternalAssetEditable(0));
  EXPECT_EQ(scene.getRenderPrimitives().size(), 2u);
  EXPECT_EQ(scene.animation().getSkinTasks().size(), 2u);
  tinygltf::Model& model = scene.getModel();
  int              joint = -1;
  for(int i = 0; i < static_cast<int>(model.nodes.size()); i++)
    if(model.nodes[i].name == "rig_joint" && scene.getNodesWorldMatrices()[i][3].x > 2.0f)
      joint = i;
  ASSERT_GE(joint, 0);
  model.nodes[joint].translation = {0.0, 3.0, 0.0};
  scene.markNodeDirty(joint);
  EXPECT_EQ(drawnCentres(scene), (std::vector<std::string>{"rig_body@0.00,1.00,0.00", "rig_body@4.00,3.00,0.00"}));
}

// runtime_rules.gltf: a missing file, a file placing itself and a too-deep chain are skipped; the
// load succeeds with what can be placed: the self-placing file's quad once, the 16 files of the
// chain the renderer nests, and the valid piece.
TEST_F(Gltf21Test, Placement_RuntimeRulesSkipAndContinue)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "placement/runtime_rules.gltf"));
  EXPECT_TRUE(scene.valid());
  EXPECT_EQ(scene.getRenderNodes().size(), 1u + 16u + 2u);
}

//--------------------------------------------------------------------------------------------------
// Versions (#26): what loads, and what a save declares
//--------------------------------------------------------------------------------------------------

// Asset version string syntax, from the spec's examples.
TEST(GltfVersion, ParseFollowsSpecSyntax)
{
  for(const char* valid : {"0.0", "0.12", "2.0", "2.1", "12.34", "999999999.999999999"})
    EXPECT_TRUE(nvvkgltf::parseGltfVersion(valid).has_value()) << valid;
  for(const char* invalid : {"", "2", "+1.0", "2.0.1", "2.00", "01.1", "2.", ".1", "2.a", "1000000000.0"})
    EXPECT_FALSE(nvvkgltf::parseGltfVersion(invalid).has_value()) << invalid;

  // Components compare as numbers: "2.1" < "2.10".
  EXPECT_LT(*nvvkgltf::parseGltfVersion("2.1"), *nvvkgltf::parseGltfVersion("2.10"));
  EXPECT_LT(*nvvkgltf::parseGltfVersion("2.10"), *nvvkgltf::parseGltfVersion("3.0"));
}

// Each 2.1 feature is detected on an otherwise valid 2.0 model, and 2.0 content is not.
TEST(GltfVersion, DetectsEachGltf21Feature)
{
  // One triangle: POSITION, plus attributes the cases below rename or retype.
  auto makeModel = []() {
    tinygltf::Model model;
    model.accessors.resize(3);
    for(tinygltf::Accessor& accessor : model.accessors)
    {
      accessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
      accessor.count         = 3;
    }
    tinygltf::Primitive primitive;
    primitive.attributes = {{"POSITION", 0}, {"TEXCOORD_0", 1}, {"COLOR_0", 2}};
    model.meshes.emplace_back().primitives.push_back(primitive);
    model.nodes.emplace_back().mesh = 0;
    return model;
  };
  auto attributes = [](tinygltf::Model& model) -> std::map<std::string, int>& {
    return model.meshes[0].primitives[0].attributes;
  };

  EXPECT_EQ(nvvkgltf::findGltf21Feature(makeModel()), "");
  EXPECT_EQ(nvvkgltf::gltfVersionForSave(makeModel()), "2.0");

  struct Case
  {
    const char*                           name;
    std::function<void(tinygltf::Model&)> apply;
  };
  const Case cases[] = {
      {"uid", [](tinygltf::Model& m) { m.nodes[0].uid = "node"; }},
      {"shape", [](tinygltf::Model& m) { m.shapes.emplace_back().type = "sphere"; }},
      {"thumbnail", [](tinygltf::Model& m) { m.asset.thumbnail = 0; }},
      {"float16 accessor", [](tinygltf::Model& m) { m.accessors[1].componentType = TINYGLTF_COMPONENT_TYPE_HALF_FLOAT; }},
      {"int32 attribute", [](tinygltf::Model& m) { m.accessors[2].componentType = TINYGLTF_COMPONENT_TYPE_INT; }},
      {"TEXCOORD_1 without TEXCOORD_0",
       [&attributes](tinygltf::Model& m) {
         attributes(m).erase("TEXCOORD_0");
         attributes(m)["TEXCOORD_1"] = 1;
       }},
      {"COLOR_2 without COLOR_1", [&attributes](tinygltf::Model& m) { attributes(m)["COLOR_2"] = 2; }},
  };
  for(const Case& c : cases)
  {
    tinygltf::Model model = makeModel();
    c.apply(model);
    EXPECT_NE(nvvkgltf::findGltf21Feature(model), "") << c.name;
    EXPECT_EQ(nvvkgltf::gltfVersionForSave(model), "2.1") << c.name;
  }

  // uint32 is valid 2.0 for indices, and only discouraged for attributes; sets 0..n-1 have no gap.
  tinygltf::Model model                        = makeModel();
  model.accessors.emplace_back().componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
  model.meshes[0].primitives[0].indices        = 3;
  model.accessors[2].componentType             = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
  attributes(model)["TEXCOORD_1"]              = 1;
  EXPECT_EQ(nvvkgltf::findGltf21Feature(model), "");
}

// Newer minor versions load; another major version, a too-new minVersion, a minVersion above version,
// or a malformed version do not.
TEST_F(Gltf21Test, Version_LoadPolicy)
{
  for(const char* name : {"plain_2_1.gltf", "min_version_2_1.gltf", "future_minor.gltf"})
  {
    nvvkgltf::Scene scene;
    EXPECT_TRUE(scene.load(kFixtures / "version" / name)) << name;
  }
  for(const char* name : {"future_major.gltf", "min_version_too_new.gltf", "min_version_above_version.gltf", "invalid_version.gltf"})
  {
    nvvkgltf::Scene scene;
    EXPECT_FALSE(scene.load(kFixtures / "version" / name)) << name;
  }
}

// A save declares "2.1" only when the content needs it (or minVersion asks for it), else "2.0".
TEST_F(Gltf21Test, Version_SavePolicy)
{
  struct Case
  {
    std::filesystem::path fixture;
    const char*           version;
    const char*           minVersion;
  };
  const Case cases[] = {
      {"version/plain_2_1.gltf", "2.0", ""},           // 2.1 file without 2.1 content
      {"version/min_version_2_1.gltf", "2.1", "2.1"},  // the author asked for 2.1
      {"version/future_minor.gltf", "2.0", ""},        // its unknown content is gone, don't claim 2.7
      {"core_properties.gltf", "2.1", ""},             // uses 2.1 features
  };
  for(const Case& c : cases)
  {
    SCOPED_TRACE(c.fixture.string());
    nvvkgltf::Scene scene;
    ASSERT_TRUE(scene.load(kFixtures / c.fixture));
    const std::filesystem::path saved = m_tempDir / c.fixture.filename();
    ASSERT_TRUE(scene.save(saved));
    const tinygltf::Model raw = readRaw(saved);
    EXPECT_EQ(raw.asset.version, c.version);
    EXPECT_EQ(raw.asset.minVersion, c.minVersion);
  }

  // Editing a 2.0-only scene into using a 2.1 feature upgrades the saved version.
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "version/plain_2_1.gltf"));
  scene.getModel().nodes[0].uid = "edited";
  ASSERT_TRUE(scene.save(m_tempDir / "edited.gltf"));
  EXPECT_EQ(readRaw(m_tempDir / "edited.gltf").asset.version, "2.1");
}

//--------------------------------------------------------------------------------------------------
// Component types (#29, #30): never crash, never read garbage
//--------------------------------------------------------------------------------------------------

// Positions or indices we cannot read: the file loads, the primitive is left out of the render
// primitives (so neither renderer uploads or traces it). These used to reach the GPU as a null
// vertex buffer and lose the device.
TEST_F(Gltf21Test, ComponentTypes_UnreadableGeometryIsSkipped)
{
  for(const char* name : {"position_int32.gltf", "position_float16.gltf", "indices_int32.gltf", "indices_uint64.gltf", "indices_vec3.gltf"})
  {
    SCOPED_TRACE(name);
    nvvkgltf::Scene scene;
    ASSERT_TRUE(scene.load(kFixtures / "component_types" / name));
    EXPECT_TRUE(scene.getRenderPrimitives().empty());
    EXPECT_TRUE(scene.getRenderNodes().empty());
  }
}

// Types we read or convert, and attributes we ignore or treat as absent, still render the quad.
// float16 / int64 / uint64 accessors parse at all only with the tinygltf fork from nv/all-fixes
// a76edf3 or later.
TEST_F(Gltf21Test, ComponentTypes_SupportedOrIgnoredRender)
{
  for(const char* name : {"position_float64.gltf", "feature_id_uint32.gltf", "feature_id_int32.gltf",
                          "custom_uint64.gltf", "normal_float16.gltf", "texcoord_float16.gltf", "texcoord_int64.gltf",
                          "color_int32.gltf", "unindexed_normal_map.gltf"})
  {
    SCOPED_TRACE(name);
    nvvkgltf::Scene scene;
    ASSERT_TRUE(scene.load(kFixtures / "component_types" / name));
    EXPECT_EQ(scene.getRenderPrimitives().size(), 1u);
    EXPECT_EQ(scene.getRenderNodes().size(), 1u);
  }
}

// Not 2.1-specific: a non-indexed primitive with a normal map gets generated tangents (this used
// to index accessors[-1]).
TEST_F(Gltf21Test, ComponentTypes_TangentsForUnindexedPrimitive)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(kFixtures / "component_types/unindexed_normal_map.gltf"));
  const tinygltf::Primitive& primitive = scene.getModel().meshes[0].primitives[0];
  ASSERT_TRUE(primitive.attributes.contains("TANGENT"));
  std::vector<glm::vec4> storage;
  std::span<const glm::vec4> tangents = tinygltf::utils::getAttributeData3(scene.getModel(), primitive, "TANGENT", &storage);
  ASSERT_EQ(tangents.size(), 6u);
  for(const glm::vec4& t : tangents)
    EXPECT_NEAR(glm::length(glm::vec3(t)), 1.0f, 1e-3f);  // generated, not left at zero
}

}  // namespace
