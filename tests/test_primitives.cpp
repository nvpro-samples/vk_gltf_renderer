/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Procedural primitives (plane/cube/sphere). CPU-only: exercises the model mutations in
// SceneEditor::addPrimitiveMesh / truncateGeometryTail, including adding the first primitive to a
// freshly-constructed Scene (the model side of createPrimitiveInNewScene). The GPU build lives in the
// renderer and is out of scope here.
//

#include <gtest/gtest.h>

#include <algorithm>

#include "gltf_scene.hpp"
#include "gltf_scene_editor.hpp"

using namespace nvvkgltf;

//--------------------------------------------------------------------------------------------------
// Adding a primitive to a freshly-constructed Scene (empty model) creates the scene container and a
// full glTF payload: one node/mesh/material and one buffer with 4 bufferViews/accessors. parseScene
// also synthesizes a default camera node, so the node count is 2 (primitive + camera).
//--------------------------------------------------------------------------------------------------
TEST(PrimitivesTest, AddToFreshSceneBuildsPayload)
{
  Scene scene;  // empty model: no scenes, no nodes

  const int node = scene.editor().addPrimitiveMesh(PrimitiveKind::eCube, {}, -1);
  ASSERT_GE(node, 0);

  const tinygltf::Model& m = scene.getModel();
  ASSERT_FALSE(m.scenes.empty());  // scene container was created
  EXPECT_EQ(m.meshes.size(), 1u);
  EXPECT_EQ(m.materials.size(), 1u);
  EXPECT_EQ(m.buffers.size(), 1u);
  EXPECT_EQ(m.bufferViews.size(), 4u);
  EXPECT_EQ(m.accessors.size(), 4u);
  EXPECT_EQ(m.nodes.size(), 2u);  // primitive node + synthesized default camera node

  ASSERT_EQ(m.meshes[0].primitives.size(), 1u);
  const tinygltf::Primitive& prim = m.meshes[0].primitives[0];
  EXPECT_EQ(prim.material, 0);
  EXPECT_GE(prim.indices, 0);
  EXPECT_EQ(prim.mode, TINYGLTF_MODE_TRIANGLES);
  EXPECT_EQ(prim.attributes.count("POSITION"), 1u);
  EXPECT_EQ(prim.attributes.count("NORMAL"), 1u);
  EXPECT_EQ(prim.attributes.count("TEXCOORD_0"), 1u);
  EXPECT_EQ(prim.attributes.count("TANGENT"), 0u);  // primitives carry no tangents

  // POSITION accessor must carry min/max (required by glTF; used for scene bounds).
  const tinygltf::Accessor& pos = m.accessors[prim.attributes.at("POSITION")];
  ASSERT_EQ(pos.minValues.size(), 3u);
  ASSERT_EQ(pos.maxValues.size(), 3u);
  EXPECT_LT(pos.minValues[0], pos.maxValues[0]);
  EXPECT_LT(pos.minValues[1], pos.maxValues[1]);
  EXPECT_LT(pos.minValues[2], pos.maxValues[2]);

  EXPECT_EQ(scene.getNumRenderPrimitives(), 1u);
  EXPECT_TRUE(scene.valid());
}

//--------------------------------------------------------------------------------------------------
// Every kind produces non-empty, triangle-list geometry.
//--------------------------------------------------------------------------------------------------
TEST(PrimitivesTest, AllKindsProduceGeometry)
{
  for(const auto& info : kPrimitiveKinds)
  {
    Scene     scene;
    const int node = scene.editor().addPrimitiveMesh(info.kind, {}, -1);
    ASSERT_GE(node, 0) << "kind=" << info.name;

    const tinygltf::Model&     m    = scene.getModel();
    const tinygltf::Primitive& prim = m.meshes[0].primitives[0];
    const tinygltf::Accessor&  pos  = m.accessors[prim.attributes.at("POSITION")];
    const tinygltf::Accessor&  idx  = m.accessors[prim.indices];

    EXPECT_GT(pos.count, 0) << info.name;       // has vertices
    EXPECT_GT(idx.count, 0) << info.name;       // has indices
    EXPECT_EQ(idx.count % 3, 0u) << info.name;  // whole triangles
    EXPECT_EQ(idx.componentType, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) << info.name;
  }
}

//--------------------------------------------------------------------------------------------------
// Parenting: -1 adds to the scene root; a valid parent index adds as its child.
//--------------------------------------------------------------------------------------------------
TEST(PrimitivesTest, AddToSceneRoot)
{
  Scene     scene;
  const int node = scene.editor().addPrimitiveMesh(PrimitiveKind::eSphere, {}, -1);
  ASSERT_GE(node, 0);

  const auto& roots = scene.getModel().scenes[scene.getCurrentScene()].nodes;
  EXPECT_NE(std::find(roots.begin(), roots.end(), node), roots.end());
}

TEST(PrimitivesTest, AddSecondPrimitiveAsChild)
{
  Scene scene;
  // First primitive bootstraps the scene (container + camera + geometry).
  const int parent = scene.editor().addPrimitiveMesh(PrimitiveKind::eCube, {}, -1);
  ASSERT_GE(parent, 0);

  // Second primitive as a child of the first's node.
  const int child = scene.editor().addPrimitiveMesh(PrimitiveKind::eSphere, {}, parent);
  ASSERT_GE(child, 0);

  const auto& children = scene.getModel().nodes[parent].children;
  EXPECT_NE(std::find(children.begin(), children.end(), child), children.end());
}

//--------------------------------------------------------------------------------------------------
// Undo: snapshot + tail-truncation (exactly what AddPrimitiveCommand::undo does) returns the model to
// its pre-add state. Truncate first, then restore (single reparse). The base scene already contains
// one primitive, so a scene container exists.
//--------------------------------------------------------------------------------------------------
TEST(PrimitivesTest, UndoRestoresPreAddState)
{
  Scene scene;
  ASSERT_GE(scene.editor().addPrimitiveMesh(PrimitiveKind::eCube, {}, -1), 0);  // base content

  const tinygltf::Model& m = scene.getModel();

  // Capture the pre-add tail sizes and node graph (mirrors AddPrimitiveCommand's constructor).
  ModelTailSizes sizes;
  sizes.meshes                = m.meshes.size();
  sizes.materials             = m.materials.size();
  sizes.accessors             = m.accessors.size();
  sizes.bufferViews           = m.bufferViews.size();
  sizes.buffers               = m.buffers.size();
  const size_t       nodes0   = m.nodes.size();
  const size_t       prims0   = scene.getNumRenderPrimitives();
  SceneGraphSnapshot snapshot = scene.editor().snapshotForDelete();

  ASSERT_GE(scene.editor().addPrimitiveMesh(PrimitiveKind::ePlane, {}, -1), 0);
  EXPECT_EQ(m.meshes.size(), sizes.meshes + 1);
  EXPECT_EQ(m.nodes.size(), nodes0 + 1);

  // Undo: truncate the geometry tail first, then restore the node graph (single reparse).
  // Order matters: buildPrimitiveKeyMap walks all meshes, so restoring before truncate would
  // leave an orphan mesh in the render-primitive list.
  scene.editor().truncateGeometryTail(sizes);
  scene.editor().restoreFromSnapshot(snapshot);

  EXPECT_EQ(m.nodes.size(), nodes0);
  EXPECT_EQ(m.meshes.size(), sizes.meshes);
  EXPECT_EQ(m.materials.size(), sizes.materials);
  EXPECT_EQ(m.accessors.size(), sizes.accessors);
  EXPECT_EQ(m.bufferViews.size(), sizes.bufferViews);
  EXPECT_EQ(m.buffers.size(), sizes.buffers);
  EXPECT_EQ(scene.getNumRenderPrimitives(), prims0);
}

//--------------------------------------------------------------------------------------------------
// Add -> undo -> add-different does not accumulate geometry (the add path is deterministic).
//--------------------------------------------------------------------------------------------------
TEST(PrimitivesTest, AddUndoAddDoesNotAccumulate)
{
  Scene scene;
  ASSERT_GE(scene.editor().addPrimitiveMesh(PrimitiveKind::eCube, {}, -1), 0);  // base content

  const tinygltf::Model& m = scene.getModel();

  ModelTailSizes sizes;
  sizes.meshes                = m.meshes.size();
  sizes.materials             = m.materials.size();
  sizes.accessors             = m.accessors.size();
  sizes.bufferViews           = m.bufferViews.size();
  sizes.buffers               = m.buffers.size();
  const size_t       nodes0   = m.nodes.size();
  SceneGraphSnapshot snapshot = scene.editor().snapshotForDelete();

  ASSERT_GE(scene.editor().addPrimitiveMesh(PrimitiveKind::eCube, {}, -1), 0);
  scene.editor().truncateGeometryTail(sizes);
  scene.editor().restoreFromSnapshot(snapshot);

  // Add a different primitive after undo: counts match a single fresh add, not two.
  ASSERT_GE(scene.editor().addPrimitiveMesh(PrimitiveKind::eSphere, {}, -1), 0);
  EXPECT_EQ(m.nodes.size(), nodes0 + 1);
  EXPECT_EQ(m.meshes.size(), sizes.meshes + 1);
  EXPECT_EQ(m.materials.size(), sizes.materials + 1);
  EXPECT_EQ(m.buffers.size(), sizes.buffers + 1);
  EXPECT_EQ(m.accessors.size(), sizes.accessors + 4);
}

//--------------------------------------------------------------------------------------------------
// Kind <-> name mapping (single source of truth used by UI, command and mesh naming).
//--------------------------------------------------------------------------------------------------
TEST(PrimitivesTest, KindNameLookup)
{
  EXPECT_STREQ(primitiveKindName(PrimitiveKind::ePlane), "Plane");
  EXPECT_STREQ(primitiveKindName(PrimitiveKind::eCube), "Cube");
  EXPECT_STREQ(primitiveKindName(PrimitiveKind::eSphere), "Sphere");

  for(const auto& info : kPrimitiveKinds)
  {
    ASSERT_NE(info.name, nullptr);
    EXPECT_STREQ(primitiveKindName(info.kind), info.name);
  }
}
