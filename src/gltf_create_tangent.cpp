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

#include <cstdlib>
#include <mikktspace.h>
#include <tinygltf/tiny_gltf.h>
#include <vector>
#include <glm/gtx/norm.hpp>

#include "nvshaders/functions.h.slang"

#include <nvutils/parallel_work.hpp>
#include <nvutils/timers.hpp>
#include <nvutils/logger.hpp>
#include "tinygltf_utils.hpp"

#include "gltf_compact_model.hpp"
#include "gltf_create_tangent.hpp"

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
  outPos[0] = outPos[1] = outPos[2] = 0.0f;
  auto*  data                       = static_cast<MikkContext*>(ctx->m_pUserData);
  size_t flatIdx                    = static_cast<size_t>(iFace) * 3 + iVert;
  if(flatIdx >= data->indices.size())
    return;
  uint32_t idx = data->indices[flatIdx];
  if(idx >= data->vertices.size())
    return;
  const auto& v = data->vertices[idx];
  outPos[0]     = v.position.x;
  outPos[1]     = v.position.y;
  outPos[2]     = v.position.z;
}

static void mikkGetNormal(const SMikkTSpaceContext* ctx, float outNorm[], int iFace, int iVert)
{
  outNorm[0] = outNorm[1] = outNorm[2] = 0.0f;
  auto*  data                          = static_cast<MikkContext*>(ctx->m_pUserData);
  size_t flatIdx                       = static_cast<size_t>(iFace) * 3 + iVert;
  if(flatIdx >= data->indices.size())
    return;
  uint32_t idx = data->indices[flatIdx];
  if(idx >= data->vertices.size())
    return;
  const auto& v = data->vertices[idx];
  outNorm[0]    = v.normal.x;
  outNorm[1]    = v.normal.y;
  outNorm[2]    = v.normal.z;
}

static void mikkGetTexCoord(const SMikkTSpaceContext* ctx, float outUV[], int iFace, int iVert)
{
  outUV[0] = outUV[1] = 0.0f;
  auto*  data         = static_cast<MikkContext*>(ctx->m_pUserData);
  size_t flatIdx      = static_cast<size_t>(iFace) * 3 + iVert;
  if(flatIdx >= data->indices.size())
    return;
  uint32_t idx = data->indices[flatIdx];
  if(idx >= data->vertices.size())
    return;
  const auto& v = data->vertices[idx];
  outUV[0]      = v.texcoord0.x;
  outUV[1]      = v.texcoord0.y;
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
// VERTEX SPLITTING & WRITE-BACK
//==================================================================================================

struct SplitVertex
{
  uint32_t  origIdx;
  glm::vec4 tangent;
};

// Per-attribute arrays produced by buildVertexArraysFromSplitData; consumed by writePrimitiveBuffers.
struct SplitVertexArrays
{
  std::vector<glm::vec3>    positions;
  std::vector<glm::vec3>    normals;
  std::vector<glm::vec4>    tangents;
  std::vector<glm::vec2>    texcoord0;
  std::vector<glm::vec2>    texcoord1;
  std::vector<glm::vec4>    colors;
  std::vector<glm::vec4>    weights;
  std::vector<glm::u16vec4> joints;
  std::vector<uint32_t>     indices;
  bool                      hasUV1     = false;
  bool                      hasColor   = false;
  bool                      hasWeights = false;
  bool                      hasJoints  = false;
};

// Append raw data to buffer[0], creating a new BufferView + Accessor. Returns accessor index.
static int appendToBuffer(tinygltf::Model& model, const void* data, size_t dataBytes, int componentType, int glType, size_t count)
{
  if(model.buffers.empty())
    model.buffers.emplace_back();

  tinygltf::Buffer& buf = model.buffers[0];

  if(data == nullptr && dataBytes != 0)
  {
    LOGE("appendToBuffer: data is nullptr but dataBytes is non-zero\n");
    std::abort();
  }

  size_t currentOffset = buf.data.size();
  size_t padding       = (4 - (currentOffset % 4)) % 4;
  buf.data.resize(currentOffset + padding);
  size_t dataOffset = buf.data.size();

  tinygltf::BufferView bv;
  bv.buffer     = 0;
  bv.byteOffset = dataOffset;
  bv.byteLength = dataBytes;
  bv.byteStride = 0;
  int bvIndex   = static_cast<int>(model.bufferViews.size());
  model.bufferViews.push_back(bv);

  buf.data.resize(dataOffset + dataBytes);
  if(dataBytes != 0)
    std::memcpy(buf.data.data() + dataOffset, data, dataBytes);

  tinygltf::Accessor acc;
  acc.bufferView    = bvIndex;
  acc.byteOffset    = 0;
  acc.componentType = componentType;
  acc.type          = glType;
  acc.count         = count;
  int accIndex      = static_cast<int>(model.accessors.size());
  model.accessors.push_back(acc);

  return accIndex;
}

// Group face-vertices by tangent compatibility and compute fv -> new vertex index.
// Returns (fvToNewVertex, newVertexData).
static std::pair<std::vector<uint32_t>, std::vector<SplitVertex>> computeTangentGroupsAndMapping(
    const MikkContext&                        mikkData,
    const std::vector<std::vector<uint32_t>>& fvIndicesPerVertex,
    size_t                                    numOrigVerts)
{
  std::vector<uint32_t>    fvToNewVertex(mikkData.indices.size());
  std::vector<SplitVertex> newVertexData;
  newVertexData.reserve(numOrigVerts * 2);

  for(size_t origV = 0; origV < numOrigVerts; origV++)
  {
    const auto& fvList = fvIndicesPerVertex[origV];
    if(fvList.empty())
      continue;

    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> tangentGroups;

    for(uint32_t fvIdx : fvList)
    {
      const glm::vec4& tangent = mikkData.faceVertexTangents[fvIdx];

      bool found = false;
      for(auto& tangentGroup : tangentGroups)
      {
        if(areTangentsCompatible(newVertexData[tangentGroup.first].tangent, tangent))
        {
          tangentGroup.second.push_back(fvIdx);
          found = true;
          break;
        }
      }

      if(!found)
      {
        uint32_t newIdx = static_cast<uint32_t>(newVertexData.size());
        newVertexData.push_back({static_cast<uint32_t>(origV), tangent});
        tangentGroups.push_back({newIdx, {fvIdx}});
      }
    }

    for(const auto& tangentGroup : tangentGroups)
      for(uint32_t fvIdx : tangentGroup.second)
        fvToNewVertex[fvIdx] = tangentGroup.first;
  }

  return {std::move(fvToNewVertex), std::move(newVertexData)};
}

// Build vertex attribute arrays from split data and original vertices.
static void buildVertexArraysFromSplitData(const std::vector<SplitVertex>&    newVertexData,
                                           const std::vector<OriginalVertex>& origVertices,
                                           const std::vector<uint32_t>&       fvToNewVertex,
                                           const tinygltf::Primitive&         prim,
                                           SplitVertexArrays&                 out)
{
  const size_t newVertCount = newVertexData.size();
  out.positions.resize(newVertCount);
  out.normals.resize(newVertCount);
  out.tangents.resize(newVertCount);
  out.texcoord0.resize(newVertCount);

  out.hasUV1     = prim.attributes.count("TEXCOORD_1") > 0;
  out.hasColor   = prim.attributes.count("COLOR_0") > 0;
  out.hasWeights = prim.attributes.count("WEIGHTS_0") > 0;
  out.hasJoints  = prim.attributes.count("JOINTS_0") > 0;

  if(out.hasUV1)
    out.texcoord1.resize(newVertCount);
  if(out.hasColor)
    out.colors.resize(newVertCount);
  if(out.hasWeights)
    out.weights.resize(newVertCount);
  if(out.hasJoints)
    out.joints.resize(newVertCount);

  for(size_t i = 0; i < newVertCount; i++)
  {
    const SplitVertex&    nv   = newVertexData[i];
    const OriginalVertex& orig = origVertices[nv.origIdx];

    out.positions[i] = orig.position;
    out.normals[i]   = orig.normal;
    out.tangents[i]  = nv.tangent;
    out.texcoord0[i] = orig.texcoord0;

    if(out.hasUV1)
      out.texcoord1[i] = orig.texcoord1;
    if(out.hasColor)
      out.colors[i] = orig.color;
    if(out.hasWeights)
      out.weights[i] = orig.weights;
    if(out.hasJoints)
      out.joints[i] = orig.joints;
  }

  out.indices.resize(fvToNewVertex.size());
  for(size_t fv = 0; fv < fvToNewVertex.size(); fv++)
    out.indices[fv] = fvToNewVertex[fv];
}

// Write vertex arrays and index buffer to model and set primitive attributes/indices.
static void writePrimitiveBuffers(tinygltf::Model& model, tinygltf::Primitive& prim, const SplitVertexArrays& arrays)
{
  prim.attributes["POSITION"] = appendToBuffer(model, arrays.positions.data(), arrays.positions.size() * sizeof(glm::vec3),
                                               TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, arrays.positions.size());
  prim.attributes["NORMAL"]  = appendToBuffer(model, arrays.normals.data(), arrays.normals.size() * sizeof(glm::vec3),
                                              TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC3, arrays.normals.size());
  prim.attributes["TANGENT"] = appendToBuffer(model, arrays.tangents.data(), arrays.tangents.size() * sizeof(glm::vec4),
                                              TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC4, arrays.tangents.size());
  prim.attributes["TEXCOORD_0"] = appendToBuffer(model, arrays.texcoord0.data(), arrays.texcoord0.size() * sizeof(glm::vec2),
                                                 TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC2, arrays.texcoord0.size());

  if(arrays.hasUV1)
    prim.attributes["TEXCOORD_1"] =
        appendToBuffer(model, arrays.texcoord1.data(), arrays.texcoord1.size() * sizeof(glm::vec2),
                       TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC2, arrays.texcoord1.size());
  if(arrays.hasColor)
    prim.attributes["COLOR_0"] = appendToBuffer(model, arrays.colors.data(), arrays.colors.size() * sizeof(glm::vec4),
                                                TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC4, arrays.colors.size());
  if(arrays.hasWeights)
    prim.attributes["WEIGHTS_0"] = appendToBuffer(model, arrays.weights.data(), arrays.weights.size() * sizeof(glm::vec4),
                                                  TINYGLTF_COMPONENT_TYPE_FLOAT, TINYGLTF_TYPE_VEC4, arrays.weights.size());
  if(arrays.hasJoints)
    prim.attributes["JOINTS_0"] =
        appendToBuffer(model, arrays.joints.data(), arrays.joints.size() * sizeof(glm::u16vec4),
                       TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, TINYGLTF_TYPE_VEC4, arrays.joints.size());

  prim.indices = appendToBuffer(model, arrays.indices.data(), arrays.indices.size() * sizeof(uint32_t),
                                TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_SCALAR, arrays.indices.size());
}

// Smart vertex splitting: duplicates only those vertices whose face-vertex tangents are incompatible,
// then writes the new geometry (all vertex attributes + index buffer) back to the model.
static bool splitAndWriteVertices(tinygltf::Model&                          model,
                                  tinygltf::Primitive&                      prim,
                                  const MikkContext&                        mikkData,
                                  const std::vector<std::vector<uint32_t>>& fvIndicesPerVertex,
                                  size_t                                    numOrigVerts)
{
  auto [fvToNewVertex, newVertexData] = computeTangentGroupsAndMapping(mikkData, fvIndicesPerVertex, numOrigVerts);

  LOGI("MikkTSpace: Vertices %zu -> %zu (split %zu for tangent discontinuities)\n", numOrigVerts, newVertexData.size(),
       newVertexData.size() - numOrigVerts);

  SplitVertexArrays arrays;
  buildVertexArraysFromSplitData(newVertexData, mikkData.vertices, fvToNewVertex, prim, arrays);
  writePrimitiveBuffers(model, prim, arrays);

  return true;
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
  return splitAndWriteVertices(model, prim, mikkData, fvIndicesPerVertex, numOrigVerts);
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
      if(forceCreation && !hasTangent)
        tinygltf::utils::createTangentAttribute(model, prim);


      result.push_back(&prim);
    }
  }
  return result;
}

bool recomputeTangents(tinygltf::Model& model, bool forceCreation, bool mikktspace)
{
  SCOPED_TIMER(__FUNCTION__);

  auto primitives = collectPrimitivesForTangents(model, forceCreation);
  if(primitives.empty())
    return false;

  bool anySplitting = false;
  if(mikktspace)
  {
    // MikkTSpace: sequential because split modifies shared buffer
    for(auto* prim : primitives)
    {
      anySplitting |= createTangentsMikkTSpace(model, *prim);
    }

    // Compact model if any splitting occurred
    if(anySplitting)
      compactModel(model);

    // Shrink buffer to actual size (release unused reserved memory)
    model.buffers[0].data.shrink_to_fit();
  }
  else
  {
    // Simple method: fully parallel
    nvutils::parallel_batches<1>(primitives.size(), [&](uint64_t primID) {
      tinygltf::utils::simpleCreateTangents(model, *primitives[primID]);
    });
  }

  return anySplitting;
}
