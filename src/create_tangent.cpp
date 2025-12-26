/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//////////////////////////////////////////////////////////////////////////
/*
    Tangent Space Generation for glTF Models
    =========================================

    This module provides two methods for computing tangent space information:

    1. SIMPLE METHOD (UV Gradient)
       - Fast computation using UV coordinate gradients
       - Modifies tangent buffer in-place
       - No vertex splitting - works within existing geometry

    2. MIKKTSPACE METHOD (High Quality)
       - Industry-standard tangent generation (Blender, Unity, Unreal use this)
       - Handles UV seams and mirrored UVs correctly by splitting vertices
       - Per-face-vertex tangent computation, then smart vertex deduplication

    MikkTSpace Algorithm:
    1. Read geometry into MikkContext (indices + all vertex attributes)
    2. Run genTangSpaceDefault() to compute per-face-vertex tangents
    3. Group face-vertices by compatible tangents (direction within ~11°, same handedness)
    4. Fast path: If all face-vertices at each vertex are compatible, write directly
    5. Slow path: Create new vertices for incompatible tangent groups
    6. Rebuild primitive with new vertex buffers and index buffer

    When vertex splitting occurs, the caller must:
    - Destroy and recreate SceneVk/SceneRtx
    - Re-parse the scene (scene.setCurrentScene)
    - Update UI scene graph
*/
//////////////////////////////////////////////////////////////////////////

#include <mikktspace.h>
#include <mutex>
#include <tinygltf/tiny_gltf.h>
#include <vector>
#include <glm/gtx/norm.hpp>

#include "nvshaders/functions.h.slang"

#include <nvutils/parallel_work.hpp>
#include <nvutils/timers.hpp>
#include <nvutils/logger.hpp>
#include <nvvkgltf/tinygltf_utils.hpp>

//==================================================================================================
// DATA STRUCTURES
//==================================================================================================

// All attributes for a single vertex - read from original model
struct OriginalVertex
{
  glm::vec3    position{0};
  glm::vec3    normal{0};
  glm::vec2    texcoord0{0};
  glm::vec2    texcoord1{0};
  glm::vec4    color{1};
  glm::vec4    weights{0};
  glm::u16vec4 joints{0};
};

// Context for MikkTSpace callbacks
struct MikkContext
{
  // Original indexed geometry (read from model)
  std::vector<uint32_t>       indices;
  std::vector<OriginalVertex> vertices;

  // Output: unindexed tangents [numFaces * 3]
  std::vector<glm::vec4> faceVertexTangents;
};

//==================================================================================================
// MIKKTSPACE CALLBACKS
//==================================================================================================

static int mikkGetNumFaces(const SMikkTSpaceContext* ctx)
{
  auto* data = static_cast<MikkContext*>(ctx->m_pUserData);
  return static_cast<int>(data->indices.size() / 3);
}

static int mikkGetNumVerticesOfFace(const SMikkTSpaceContext* ctx, int iFace)
{
  return 3;
}

static void mikkGetPosition(const SMikkTSpaceContext* ctx, float outPos[], int iFace, int iVert)
{
  auto*       data = static_cast<MikkContext*>(ctx->m_pUserData);
  uint32_t    idx  = data->indices[iFace * 3 + iVert];
  const auto& v    = data->vertices[idx];
  outPos[0]        = v.position.x;
  outPos[1]        = v.position.y;
  outPos[2]        = v.position.z;
}

static void mikkGetNormal(const SMikkTSpaceContext* ctx, float outNorm[], int iFace, int iVert)
{
  auto*       data = static_cast<MikkContext*>(ctx->m_pUserData);
  uint32_t    idx  = data->indices[iFace * 3 + iVert];
  const auto& v    = data->vertices[idx];
  outNorm[0]       = v.normal.x;
  outNorm[1]       = v.normal.y;
  outNorm[2]       = v.normal.z;
}

static void mikkGetTexCoord(const SMikkTSpaceContext* ctx, float outUV[], int iFace, int iVert)
{
  auto*       data = static_cast<MikkContext*>(ctx->m_pUserData);
  uint32_t    idx  = data->indices[iFace * 3 + iVert];
  const auto& v    = data->vertices[idx];
  outUV[0]         = v.texcoord0.x;
  outUV[1]         = v.texcoord0.y;
}

static void mikkSetTSpaceBasic(const SMikkTSpaceContext* ctx, const float tangent[], float sign, int iFace, int iVert)
{
  auto*    data      = static_cast<MikkContext*>(ctx->m_pUserData);
  uint32_t flatIndex = iFace * 3 + iVert;
  uint32_t vertIndex = data->indices[flatIndex];

  glm::vec3        t(tangent[0], tangent[1], tangent[2]);
  const glm::vec3& n = data->vertices[vertIndex].normal;

  // Validate tangent is not parallel to normal
  if(glm::abs(glm::dot(t, n)) < 0.9f)
  {
    // Valid - flip sign for Vulkan coordinate system
    data->faceVertexTangents[flatIndex] = glm::vec4(t, -sign);
  }
  else
  {
    // Fallback for degenerate case
    data->faceVertexTangents[flatIndex] = shaderio::makeFastTangent(n);
  }
}

//==================================================================================================
// HELPER FUNCTIONS
//==================================================================================================

// Check if two tangents are compatible (can share a vertex)
static bool areTangentsCompatible(const glm::vec4& a, const glm::vec4& b)
{
  glm::vec3 da = glm::vec3(a);
  glm::vec3 db = glm::vec3(b);

  float lenA = glm::length(da);
  float lenB = glm::length(db);

  // Degenerate tangents are compatible
  if(lenA < 1e-6f || lenB < 1e-6f)
    return true;

  da /= lenA;
  db /= lenB;

  // Direction must be similar (within ~11 degrees)
  if(glm::dot(da, db) < 0.98f)
    return false;

  // Handedness must match
  if(a.w * b.w < 0.0f)
    return false;

  return true;
}

// Read indices from primitive (zero-copy when format matches)
static void readIndices(const tinygltf::Model& model, const tinygltf::Primitive& prim, std::vector<uint32_t>& indices)
{
  if(prim.indices >= 0)
  {
    std::span<const uint32_t> data = tinygltf::utils::getAccessorData(model, model.accessors[prim.indices], &indices);
    if(indices.empty())  // getAccessorData returned direct pointer, need to copy
      indices.assign(data.begin(), data.end());
  }
  else
  {
    size_t count = tinygltf::utils::getVertexCount(model, prim);
    indices.resize(count);
    for(size_t i = 0; i < count; i++)
      indices[i] = static_cast<uint32_t>(i);
  }
}

// Read all vertex attributes from primitive into OriginalVertex array
// Uses getAttributeData3 for zero-copy access when data format matches (common case)
static void readVertices(const tinygltf::Model& model, const tinygltf::Primitive& prim, std::vector<OriginalVertex>& vertices)
{
  size_t count = tinygltf::utils::getVertexCount(model, prim);
  vertices.resize(count);

  // Position - always present (checked by caller)
  {
    std::vector<glm::vec3>     storage;
    std::span<const glm::vec3> data = tinygltf::utils::getAttributeData3(model, prim, "POSITION", &storage);
    for(size_t i = 0; i < data.size(); i++)
      vertices[i].position = data[i];
  }

  // Normal - always present (checked by caller)
  {
    std::vector<glm::vec3>     storage;
    std::span<const glm::vec3> data = tinygltf::utils::getAttributeData3(model, prim, "NORMAL", &storage);
    for(size_t i = 0; i < data.size(); i++)
      vertices[i].normal = data[i];
  }

  // Texcoord 0 - always present (checked by caller)
  {
    std::vector<glm::vec2>     storage;
    std::span<const glm::vec2> data = tinygltf::utils::getAttributeData3(model, prim, "TEXCOORD_0", &storage);
    for(size_t i = 0; i < data.size(); i++)
      vertices[i].texcoord0 = data[i];
  }

  // Texcoord 1 - optional
  {
    std::vector<glm::vec2>     storage;
    std::span<const glm::vec2> data = tinygltf::utils::getAttributeData3(model, prim, "TEXCOORD_1", &storage);
    for(size_t i = 0; i < data.size(); i++)
      vertices[i].texcoord1 = data[i];
  }

  // Color - optional, can be vec3 or vec4
  if(prim.attributes.count("COLOR_0"))
  {
    const auto& acc = model.accessors[prim.attributes.at("COLOR_0")];
    if(acc.type == TINYGLTF_TYPE_VEC3)
    {
      std::vector<glm::vec3>     storage;
      std::span<const glm::vec3> data = tinygltf::utils::getAttributeData3(model, prim, "COLOR_0", &storage);
      for(size_t i = 0; i < data.size(); i++)
        vertices[i].color = glm::vec4(data[i], 1.0f);
    }
    else
    {
      std::vector<glm::vec4>     storage;
      std::span<const glm::vec4> data = tinygltf::utils::getAttributeData3(model, prim, "COLOR_0", &storage);
      for(size_t i = 0; i < data.size(); i++)
        vertices[i].color = data[i];
    }
  }

  // Weights - optional
  {
    std::vector<glm::vec4>     storage;
    std::span<const glm::vec4> data = tinygltf::utils::getAttributeData3(model, prim, "WEIGHTS_0", &storage);
    for(size_t i = 0; i < data.size(); i++)
      vertices[i].weights = data[i];
  }

  // Joints - optional, can be u8vec4 or u16vec4
  if(prim.attributes.count("JOINTS_0"))
  {
    const auto& acc = model.accessors[prim.attributes.at("JOINTS_0")];
    if(acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
    {
      std::vector<glm::u8vec4>     storage;
      std::span<const glm::u8vec4> data = tinygltf::utils::getAttributeData3(model, prim, "JOINTS_0", &storage);
      for(size_t i = 0; i < data.size(); i++)
        vertices[i].joints = glm::u16vec4(data[i]);
    }
    else
    {
      std::vector<glm::u16vec4>     storage;
      std::span<const glm::u16vec4> data = tinygltf::utils::getAttributeData3(model, prim, "JOINTS_0", &storage);
      for(size_t i = 0; i < data.size(); i++)
        vertices[i].joints = data[i];
    }
  }
}

//==================================================================================================
// SIMPLE TANGENT GENERATION (UV Gradient Method)
//==================================================================================================

static void createTangentsSimple(tinygltf::Model& model, tinygltf::Primitive& prim)
{
  if(prim.attributes.find("TANGENT") == prim.attributes.end())
    tinygltf::utils::createTangentAttribute(model, prim);

  tinygltf::utils::simpleCreateTangents(model, prim);
}

//==================================================================================================
// MIKKTSPACE TANGENT GENERATION
//==================================================================================================

// Returns true if vertex splitting occurred (buffers grew)
static bool createTangentsMikkTSpace(tinygltf::Model& model, tinygltf::Primitive& prim)
{
  // --- Step 1: Read all geometry into MikkContext ---

  MikkContext mikkData;

  readIndices(model, prim, mikkData.indices);
  if(mikkData.indices.empty())
    return false;

  readVertices(model, prim, mikkData.vertices);
  if(mikkData.vertices.empty())
    return false;

  // Sanity check: ensure we have at least some valid geometry
  // (catches completely degenerate meshes with all-zero data)
  bool hasValidData = false;
  for(const auto& v : mikkData.vertices)
  {
    if(glm::length2(v.position) > 0 && glm::length2(v.normal) > 0)
    {
      hasValidData = true;
      break;  // Found valid vertex, no need to check more
    }
  }
  if(!hasValidData)
    return false;

  // Allocate output for MikkTSpace
  mikkData.faceVertexTangents.resize(mikkData.indices.size(), glm::vec4(1, 0, 0, 1));

  // --- Step 2: Run MikkTSpace ---

  SMikkTSpaceInterface iface   = {};
  iface.m_getNumFaces          = mikkGetNumFaces;
  iface.m_getNumVerticesOfFace = mikkGetNumVerticesOfFace;
  iface.m_getPosition          = mikkGetPosition;
  iface.m_getNormal            = mikkGetNormal;
  iface.m_getTexCoord          = mikkGetTexCoord;
  iface.m_setTSpaceBasic       = mikkSetTSpaceBasic;

  SMikkTSpaceContext mikkCtx = {};
  mikkCtx.m_pInterface       = &iface;
  mikkCtx.m_pUserData        = &mikkData;

  genTangSpaceDefault(&mikkCtx);

  // --- Step 3: Check if splitting is needed ---

  const size_t numOrigVerts = mikkData.vertices.size();

  // For each original vertex, collect all face-vertex indices that reference it
  std::vector<std::vector<uint32_t>> fvIndicesPerVertex(numOrigVerts);
  for(size_t fv = 0; fv < mikkData.indices.size(); fv++)
  {
    fvIndicesPerVertex[mikkData.indices[fv]].push_back(static_cast<uint32_t>(fv));
  }

  // Check for incompatible tangents
  bool needsSplitting = false;
  for(size_t v = 0; v < numOrigVerts && !needsSplitting; v++)
  {
    const auto& fvList = fvIndicesPerVertex[v];
    if(fvList.size() <= 1)
      continue;

    // Compare first tangent with all others
    const glm::vec4& firstTangent = mikkData.faceVertexTangents[fvList[0]];
    for(size_t i = 1; i < fvList.size(); i++)
    {
      if(!areTangentsCompatible(firstTangent, mikkData.faceVertexTangents[fvList[i]]))
      {
        needsSplitting = true;
        break;
      }
    }
  }

  // --- Step 4a: Fast path - no splitting needed ---

  if(!needsSplitting)
  {
    // Ensure tangent attribute exists
    if(prim.attributes.find("TANGENT") == prim.attributes.end())
      tinygltf::utils::createTangentAttribute(model, prim);

    // Write tangents directly to existing buffer
    auto tangentSpan = tinygltf::utils::getAttributeData3<glm::vec4>(model, prim, "TANGENT", nullptr);
    if(tangentSpan.empty())
      return false;

    // For each vertex, use the tangent from its first face-vertex (they're all compatible)
    for(size_t v = 0; v < numOrigVerts; v++)
    {
      if(!fvIndicesPerVertex[v].empty())
      {
        tangentSpan[v] = mikkData.faceVertexTangents[fvIndicesPerVertex[v][0]];
      }
    }
    return false;  // No splitting, buffers unchanged
  }

  // --- Step 4b: Slow path - smart vertex splitting ---
  // Only duplicate vertices that have incompatible tangents. Vertices with compatible
  // tangents across all their faces can share a single vertex slot.

  // Track which original attributes are present
  bool hasUV1     = prim.attributes.count("TEXCOORD_1") > 0;
  bool hasColor   = prim.attributes.count("COLOR_0") > 0;
  bool hasWeights = prim.attributes.count("WEIGHTS_0") > 0;
  bool hasJoints  = prim.attributes.count("JOINTS_0") > 0;

  // For each face-vertex, determine which new vertex index it maps to
  // Initially, all face-vertices pointing to the same original vertex share it
  std::vector<uint32_t> fvToNewVertex(mikkData.indices.size());

  // newVertexData[i] = {original vertex index, tangent to use}
  struct NewVertex
  {
    uint32_t  origIdx;
    glm::vec4 tangent;
  };
  std::vector<NewVertex> newVertexData;
  newVertexData.reserve(numOrigVerts * 2);  // Estimate: some splitting

  // Process each original vertex and its face-vertices
  for(size_t origV = 0; origV < numOrigVerts; origV++)
  {
    const auto& fvList = fvIndicesPerVertex[origV];
    if(fvList.empty())
      continue;

    // Group face-vertices by compatible tangents
    // Each group shares a single new vertex
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> groups;  // {newVertexIdx, [fvIndices]}

    for(uint32_t fvIdx : fvList)
    {
      const glm::vec4& tangent = mikkData.faceVertexTangents[fvIdx];

      // Find a compatible group
      bool found = false;
      for(auto& group : groups)
      {
        if(areTangentsCompatible(newVertexData[group.first].tangent, tangent))
        {
          group.second.push_back(fvIdx);
          found = true;
          break;
        }
      }

      if(!found)
      {
        // Create new vertex for this tangent
        uint32_t newIdx = static_cast<uint32_t>(newVertexData.size());
        newVertexData.push_back({static_cast<uint32_t>(origV), tangent});
        groups.push_back({newIdx, {fvIdx}});
      }
    }

    // Assign new vertex indices to face-vertices
    for(const auto& group : groups)
    {
      for(uint32_t fvIdx : group.second)
      {
        fvToNewVertex[fvIdx] = group.first;
      }
    }
  }

  LOGI("MikkTSpace: Vertices %zu -> %zu (split %zu for tangent discontinuities)\n", numOrigVerts, newVertexData.size(),
       newVertexData.size() - numOrigVerts);

  // Build new vertex arrays
  const size_t              newVertCount = newVertexData.size();
  std::vector<glm::vec3>    newPositions(newVertCount);
  std::vector<glm::vec3>    newNormals(newVertCount);
  std::vector<glm::vec4>    newTangents(newVertCount);
  std::vector<glm::vec2>    newTexcoord0(newVertCount);
  std::vector<glm::vec2>    newTexcoord1;
  std::vector<glm::vec4>    newColors;
  std::vector<glm::vec4>    newWeights;
  std::vector<glm::u16vec4> newJoints;

  if(hasUV1)
    newTexcoord1.resize(newVertCount);
  if(hasColor)
    newColors.resize(newVertCount);
  if(hasWeights)
    newWeights.resize(newVertCount);
  if(hasJoints)
    newJoints.resize(newVertCount);

  // Fill new vertex data
  for(size_t i = 0; i < newVertCount; i++)
  {
    const NewVertex&      nv   = newVertexData[i];
    const OriginalVertex& orig = mikkData.vertices[nv.origIdx];

    newPositions[i] = orig.position;
    newNormals[i]   = orig.normal;
    newTangents[i]  = nv.tangent;
    newTexcoord0[i] = orig.texcoord0;

    if(hasUV1)
      newTexcoord1[i] = orig.texcoord1;
    if(hasColor)
      newColors[i] = orig.color;
    if(hasWeights)
      newWeights[i] = orig.weights;
    if(hasJoints)
      newJoints[i] = orig.joints;
  }

  // Build new index buffer
  std::vector<uint32_t> newIndices(mikkData.indices.size());
  for(size_t fv = 0; fv < mikkData.indices.size(); fv++)
  {
    newIndices[fv] = fvToNewVertex[fv];
  }

  // --- Step 5: Write new geometry to model ---
  // NOTE: We append new data to the buffer rather than replacing in-place because:
  // 1. The new vertex count may differ from the original
  // 2. Other primitives may share the same buffer views
  // The old data becomes orphaned but this is acceptable for runtime tangent generation.
  // For minimal file size, save and reload the model after tangent generation.

  tinygltf::Buffer& buf = model.buffers[0];

  // Helper to add data to buffer and create accessor
  auto addBufferData = [&](const void* data, size_t dataBytes, int componentType, int glType, size_t count) -> int {
    // Align to 4 bytes
    size_t currentOffset = buf.data.size();
    size_t padding       = (4 - (currentOffset % 4)) % 4;
    buf.data.resize(currentOffset + padding);
    size_t dataOffset = buf.data.size();

    // Create buffer view
    tinygltf::BufferView bv;
    bv.buffer     = 0;
    bv.byteOffset = dataOffset;
    bv.byteLength = dataBytes;
    bv.byteStride = 0;  // Tightly packed
    int bvIndex   = static_cast<int>(model.bufferViews.size());
    model.bufferViews.push_back(bv);

    // Append data
    buf.data.resize(dataOffset + dataBytes);
    std::memcpy(buf.data.data() + dataOffset, data, dataBytes);

    // Create accessor
    tinygltf::Accessor acc;
    acc.bufferView    = bvIndex;
    acc.byteOffset    = 0;
    acc.componentType = componentType;
    acc.type          = glType;
    acc.count         = count;
    int accIndex      = static_cast<int>(model.accessors.size());
    model.accessors.push_back(acc);

    return accIndex;
  };

  // Write all vertex attributes to buffer
  prim.attributes["POSITION"]   = addBufferData(newPositions.data(), newPositions.size() * sizeof(glm::vec3),
                                                TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, newPositions.size());
  prim.attributes["NORMAL"]     = addBufferData(newNormals.data(), newNormals.size() * sizeof(glm::vec3),
                                                TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, newNormals.size());
  prim.attributes["TANGENT"]    = addBufferData(newTangents.data(), newTangents.size() * sizeof(glm::vec4),
                                                TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC4, newTangents.size());
  prim.attributes["TEXCOORD_0"] = addBufferData(newTexcoord0.data(), newTexcoord0.size() * sizeof(glm::vec2),
                                                TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC2, newTexcoord0.size());

  if(hasUV1)
    prim.attributes["TEXCOORD_1"] = addBufferData(newTexcoord1.data(), newTexcoord1.size() * sizeof(glm::vec2),
                                                  TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC2, newTexcoord1.size());
  if(hasColor)
    prim.attributes["COLOR_0"] = addBufferData(newColors.data(), newColors.size() * sizeof(glm::vec4),
                                               TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC4, newColors.size());
  if(hasWeights)
    prim.attributes["WEIGHTS_0"] = addBufferData(newWeights.data(), newWeights.size() * sizeof(glm::vec4),
                                                 TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC4, newWeights.size());
  if(hasJoints)
    prim.attributes["JOINTS_0"] = addBufferData(newJoints.data(), newJoints.size() * sizeof(glm::u16vec4),
                                                TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, TINYGLTF_TYPE_VEC4, newJoints.size());

  // Write index buffer
  prim.indices = addBufferData(newIndices.data(), newIndices.size() * sizeof(uint32_t),
                               TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_SCALAR, newIndices.size());

  return true;  // Splitting occurred, buffers grew
}

//==================================================================================================
// MAIN ENTRY POINT
//==================================================================================================

// Collect primitives that need processing
static std::vector<tinygltf::Primitive*> collectPrimitivesForTangents(tinygltf::Model& model, bool forceCreation)
{
  std::vector<tinygltf::Primitive*> result;
  for(auto& mesh : model.meshes)
  {
    for(auto& prim : mesh.primitives)
    {
      // Check required attributes
      if(prim.attributes.find("POSITION") == prim.attributes.end())
        continue;
      if(prim.attributes.find("NORMAL") == prim.attributes.end())
        continue;
      if(prim.attributes.find("TEXCOORD_0") == prim.attributes.end())
        continue;

      bool hasTangent = prim.attributes.find("TANGENT") != prim.attributes.end();
      if(!hasTangent && !forceCreation)
        continue;

      result.push_back(&prim);
    }
  }
  return result;
}

// Estimate buffer growth for primitives that may need splitting
// Returns estimated additional bytes needed (conservative upper bound)
static size_t estimateBufferGrowth(const tinygltf::Model& model, const std::vector<tinygltf::Primitive*>& primitives)
{
  size_t estimate = 0;
  for(const auto* prim : primitives)
  {
    // Worst case: every face-vertex becomes unique (fully unindexed)
    size_t indexCount = prim->indices >= 0 ? model.accessors[prim->indices].count : 0;
    if(indexCount == 0)
      continue;

    // Estimate bytes: position(12) + normal(12) + tangent(16) + uv0(8) + indices(4) = 52 bytes per vertex
    // Plus optional: uv1(8) + color(16) + weights(16) + joints(8) = 48 bytes
    size_t bytesPerVertex = 52;
    if(prim->attributes.count("TEXCOORD_1"))
      bytesPerVertex += 8;
    if(prim->attributes.count("COLOR_0"))
      bytesPerVertex += 16;
    if(prim->attributes.count("WEIGHTS_0"))
      bytesPerVertex += 16;
    if(prim->attributes.count("JOINTS_0"))
      bytesPerVertex += 8;

    estimate += indexCount * bytesPerVertex;
    estimate += 64;  // Alignment padding per primitive
  }
  return estimate;
}

bool recomputeTangents(tinygltf::Model& model, bool forceCreation, bool mikktspace)
{
  SCOPED_TIMER(__FUNCTION__);

  auto primitives = collectPrimitivesForTangents(model, forceCreation);
  if(primitives.empty())
    return false;

  // Pre-allocate buffer to avoid multiple reallocations during MikkTSpace processing
  if(mikktspace && !model.buffers.empty())
  {
    size_t estimatedGrowth = estimateBufferGrowth(model, primitives);
    model.buffers[0].data.reserve(model.buffers[0].data.size() + estimatedGrowth);
  }

  bool anySplitting = false;
  for(auto* prim : primitives)
  {
    if(mikktspace)
      anySplitting |= createTangentsMikkTSpace(model, *prim);
    else
      createTangentsSimple(model, *prim);
  }

  // Shrink buffer to actual size (release unused reserved memory)
  if(mikktspace && !model.buffers.empty())
  {
    model.buffers[0].data.shrink_to_fit();
  }

  return anySplitting;
}
