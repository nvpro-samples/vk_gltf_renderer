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

//
// Regression test: geometry compaction must preserve EXT_mesh_opacity_micromap.
//
// The micromap payload lives outside the normal mesh-attribute graph (root micromaps[] point at
// `data`/`triangles` bufferViews; each primitive references a `micromapIndices` accessor). Earlier,
// compactModel() collected only attribute/skin/animation/image references, so it stripped the
// micromap bufferViews/accessors and orphaned the extension. This drives the low-level compactor on
// a synthetic model and asserts the payload survives verbatim with correctly remapped indices.
//

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <tinygltf/tiny_gltf.h>

#include "gltf_compact_model.hpp"
#include "tinygltf_utils.hpp"

namespace {

// Append bytes to buffer[0] (4-byte aligned) and return the new bufferView index.
int appendBufferView(tinygltf::Model& model, const std::vector<uint8_t>& bytes)
{
  tinygltf::Buffer& buf = model.buffers[0];
  while(buf.data.size() % 4 != 0)
    buf.data.push_back(0);
  const size_t offset = buf.data.size();
  buf.data.insert(buf.data.end(), bytes.begin(), bytes.end());

  tinygltf::BufferView bv;
  bv.buffer     = 0;
  bv.byteOffset = offset;
  bv.byteLength = bytes.size();
  model.bufferViews.push_back(bv);
  return static_cast<int>(model.bufferViews.size() - 1);
}

// Read back the bytes a bufferView currently points at.
std::vector<uint8_t> readBufferView(const tinygltf::Model& model, int bvIdx)
{
  const tinygltf::BufferView& bv  = model.bufferViews[bvIdx];
  const tinygltf::Buffer&     buf = model.buffers[bv.buffer];
  return std::vector<uint8_t>(buf.data.begin() + bv.byteOffset, buf.data.begin() + bv.byteOffset + bv.byteLength);
}

}  // namespace

TEST(OpacityMicromapCompaction, PreservesExtensionData)
{
  tinygltf::Model model;
  model.buffers.emplace_back();  // buffer[0]

  // Distinctive payloads so survival can be verified byte-for-byte.
  const std::vector<uint8_t> posBytes(36, 0x01);  // 3 vertices * vec3
  const std::vector<uint8_t> mmData      = {0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44};
  const std::vector<uint8_t> mmTriangles = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE};
  const std::vector<uint8_t> mmIndices   = {0x00, 0x00, 0x01, 0x00, 0x02, 0x00};  // 3x ushort {0,1,2}
  const std::vector<uint8_t> orphanBytes(64, 0x77);

  const int bvPos    = appendBufferView(model, posBytes);
  const int bvMmData = appendBufferView(model, mmData);
  const int bvMmTris = appendBufferView(model, mmTriangles);
  const int bvMmIdx  = appendBufferView(model, mmIndices);
  const int bvOrphan = appendBufferView(model, orphanBytes);

  // Position accessor (kept, referenced by the primitive).
  {
    tinygltf::Accessor acc;
    acc.bufferView    = bvPos;
    acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
    acc.type          = TINYGLTF_TYPE_VEC3;
    acc.count         = 3;
    model.accessors.push_back(acc);
  }
  const int accPos = static_cast<int>(model.accessors.size() - 1);

  // micromapIndices accessor (kept only via the primitive extension).
  {
    tinygltf::Accessor acc;
    acc.bufferView    = bvMmIdx;
    acc.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT;
    acc.type          = TINYGLTF_TYPE_SCALAR;
    acc.count         = 3;
    model.accessors.push_back(acc);
  }
  const int accMmIdx = static_cast<int>(model.accessors.size() - 1);

  // Orphan accessor referencing the orphan bufferView -> forces compaction to actually run.
  {
    tinygltf::Accessor acc;
    acc.bufferView    = bvOrphan;
    acc.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
    acc.type          = TINYGLTF_TYPE_SCALAR;
    acc.count         = static_cast<int>(orphanBytes.size());
    model.accessors.push_back(acc);
  }

  // Mesh/primitive with the per-primitive micromap link.
  {
    tinygltf::Primitive prim;
    prim.attributes["POSITION"] = accPos;

    tinygltf::Value::Object primExt;
    primExt["micromap"]                                       = tinygltf::Value(0);
    primExt["micromapIndices"]                                = tinygltf::Value(accMmIdx);
    prim.extensions[EXT_MESH_OPACITY_MICROMAP_EXTENSION_NAME] = tinygltf::Value(primExt);

    tinygltf::Mesh mesh;
    mesh.primitives.push_back(prim);
    model.meshes.push_back(mesh);
  }

  // Root extension: one micromap build input pointing at the data/triangles bufferViews.
  {
    tinygltf::Value::Object mmEntry;
    mmEntry["data"]         = tinygltf::Value(bvMmData);
    mmEntry["triangles"]    = tinygltf::Value(bvMmTris);
    mmEntry["usageCounts"]  = tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(1)});
    mmEntry["usageLevels"]  = tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(3)});
    mmEntry["usageFormats"] = tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(1)});

    tinygltf::Value::Object rootExt;
    rootExt["micromaps"] = tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(mmEntry)});
    model.extensions[EXT_MESH_OPACITY_MICROMAP_EXTENSION_NAME] = tinygltf::Value(rootExt);
  }

  // Act: the orphan accessor makes compaction necessary.
  ASSERT_TRUE(compactModel(model));

  // The root extension must still be present with its micromaps[] entry.
  const tinygltf::Value* rootExt = tinygltf::utils::findExtension(model.extensions, EXT_MESH_OPACITY_MICROMAP_EXTENSION_NAME);
  ASSERT_NE(rootExt, nullptr);
  ASSERT_TRUE(rootExt->Has("micromaps") && rootExt->Get("micromaps").IsArray());
  ASSERT_EQ(rootExt->Get("micromaps").ArrayLen(), 1u);

  const tinygltf::Value& mmEntry = rootExt->Get("micromaps").Get(0);
  ASSERT_TRUE(mmEntry.Has("data") && mmEntry.Has("triangles"));

  // The remapped data/triangles bufferViews must still resolve to the original bytes.
  const int newBvData = mmEntry.Get("data").Get<int>();
  const int newBvTris = mmEntry.Get("triangles").Get<int>();
  ASSERT_GE(newBvData, 0);
  ASSERT_GE(newBvTris, 0);
  EXPECT_EQ(readBufferView(model, newBvData), mmData);
  EXPECT_EQ(readBufferView(model, newBvTris), mmTriangles);

  // The per-primitive micromapIndices accessor must survive and read back the original indices.
  const tinygltf::Value* primExt =
      tinygltf::utils::findExtension(model.meshes[0].primitives[0].extensions, EXT_MESH_OPACITY_MICROMAP_EXTENSION_NAME);
  ASSERT_NE(primExt, nullptr);
  ASSERT_TRUE(primExt->Has("micromapIndices"));
  const int newAccMmIdx = primExt->Get("micromapIndices").Get<int>();
  ASSERT_GE(newAccMmIdx, 0);
  ASSERT_LT(newAccMmIdx, static_cast<int>(model.accessors.size()));
  EXPECT_EQ(readBufferView(model, model.accessors[newAccMmIdx].bufferView), mmIndices);
}
