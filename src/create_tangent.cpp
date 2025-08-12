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

    This module provides functionality to compute tangent space information for glTF models
    using the MikkTSpace library. Key features include:

    - Tangent space computation for all primitives in a glTF model
    - Support for both MikkTSpace and simple tangent generation methods
    - Thread-safe parallel processing of multiple primitives
    - Proper handling of vertex attributes (position, normal, UV coordinates)
    - Orthogonal tangent vector correction for improved normal mapping
    - Integration with the glTF scene loading pipeline

    The implementation uses the MikkTSpace library to generate high-quality tangent
    space information, which is essential for normal mapping and other texture-based
    surface detail techniques.
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
#include <nvvkgltf/tinygltf_utils.hpp>

// Structure to hold context data for MikkTSpace interface
// Contains references to the glTF model and primitive being processed
struct UserData
{
  tinygltf::Model*     model            = nullptr;  // Reference to the glTF model
  tinygltf::Primitive* primitive        = nullptr;  // Current primitive being processed
  int32_t              posAccessorIndex = -1;       // Index of position attribute accessor
  int32_t              nrmAccessorIndex = -1;       // Index of normal attribute accessor
  int32_t              uvAccessorIndex  = -1;       // Index of UV coordinate attribute accessor
  int32_t              tanAccessorIndex = -1;       // Index of tangent attribute accessor
};

//--------------------------------------------------------------------------------------------------
// MikkTSpace Interface Functions
// These functions implement the required interface for the MikkTSpace library
// to access and modify vertex data during tangent space computation

// Get the total number of faces in the primitive
static int32_t getNumFaces(const SMikkTSpaceContext* pContext)
{
  const UserData*            userdata  = static_cast<const UserData*>(pContext->m_pUserData);
  const tinygltf::Primitive* primitive = userdata->primitive;
  const tinygltf::Model*     model     = userdata->model;
  const int32_t              count     = static_cast<int32_t>(tinygltf::utils::getIndexCount(*model, *primitive));
  return count / 3;  // Assuming triangles
}

// Get the number of vertices for a given face (always 3 for triangles)
static int32_t getNumVerticesOfFace(const SMikkTSpaceContext* pContext, const int32_t iFace)
{
  return 3;  // Assuming triangles
}

// Get the vertex index for a specific face and vertex
inline static int32_t getIndex(const SMikkTSpaceContext* pContext, const int32_t iFace, const int32_t iVert)
{
  const UserData*             userdata   = static_cast<const UserData*>(pContext->m_pUserData);
  const tinygltf::Primitive*  primitive  = userdata->primitive;
  const tinygltf::Model*      model      = userdata->model;
  const tinygltf::Accessor&   accessor   = model->accessors[primitive->indices];
  const tinygltf::BufferView& bufferView = model->bufferViews[accessor.bufferView];
  const tinygltf::Buffer&     buffer     = model->buffers[bufferView.buffer];
  const uint8_t*              data       = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
  const size_t                stride     = accessor.ByteStride(bufferView);

  const int32_t offset = iFace * 3 + iVert;
  assert(accessor.sparse.isSparse == false);
  switch(accessor.componentType)
  {
    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
      return *reinterpret_cast<const uint32_t*>(data + offset * stride);
    }
    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
      return *reinterpret_cast<const uint16_t*>(data + offset * stride);
    }
    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
      return *reinterpret_cast<const uint8_t*>(data + offset * stride);
    }
  }
  return 0;
}

// Helper function to get attribute data for a specific vertex
template <typename T>
inline static T* getAttributeData(const SMikkTSpaceContext* pContext, const int32_t iFace, const int32_t iVert, int32_t accessorIndex)
{
  auto             userdata = static_cast<const UserData*>(pContext->m_pUserData);
  tinygltf::Model* model    = userdata->model;
  const int32_t    index    = getIndex(pContext, iFace, iVert);

  const tinygltf::Accessor&   accessor   = model->accessors[accessorIndex];
  const tinygltf::BufferView& bufferView = model->bufferViews[accessor.bufferView];
  tinygltf::Buffer&           buffer     = model->buffers[bufferView.buffer];

  uint8_t*     data   = buffer.data.data() + bufferView.byteOffset + accessor.byteOffset;
  const size_t stride = accessor.ByteStride(bufferView);
  return reinterpret_cast<T*>(data + index * stride);
}

// Get position data for a vertex
inline static void getPosition(const SMikkTSpaceContext* pContext, float fvPosOut[], const int32_t iFace, const int32_t iVert)
{
  const glm::vec3* positions =
      getAttributeData<glm::vec3>(pContext, iFace, iVert, static_cast<const UserData*>(pContext->m_pUserData)->posAccessorIndex);
  std::memcpy(fvPosOut, positions, sizeof(glm::vec3));
}

// Get normal data for a vertex
inline static void getNormal(const SMikkTSpaceContext* pContext, float fvNormOut[], const int32_t iFace, const int32_t iVert)
{
  const glm::vec3* normals =
      getAttributeData<glm::vec3>(pContext, iFace, iVert, static_cast<const UserData*>(pContext->m_pUserData)->nrmAccessorIndex);
  std::memcpy(fvNormOut, normals, sizeof(glm::vec3));
}

// Get texture coordinate data for a vertex
inline static void getTexCoord(const SMikkTSpaceContext* pContext, float fvTexcOut[], const int32_t iFace, const int32_t iVert)
{
  const glm::vec2* texcoords =
      getAttributeData<glm::vec2>(pContext, iFace, iVert, static_cast<const UserData*>(pContext->m_pUserData)->uvAccessorIndex);
  std::memcpy(fvTexcOut, texcoords, sizeof(glm::vec2));
}

// Set the computed tangent space data for a vertex
// Includes validation to ensure tangent is orthogonal to normal
inline static void setTSpaceBasic(const SMikkTSpaceContext* pContext, const float fvTangent[], const float fSign, const int32_t iFace, const int32_t iVert)
{
  glm::vec4*       tangent = getAttributeData<glm::vec4>(pContext, iFace, iVert,
                                                         static_cast<const UserData*>(pContext->m_pUserData)->tanAccessorIndex);
  const glm::vec3* normal =
      getAttributeData<glm::vec3>(pContext, iFace, iVert, static_cast<const UserData*>(pContext->m_pUserData)->nrmAccessorIndex);

  // MikkTSpace uses the variation in texture coordinates to calculate the tangent and bitangent vectors.
  // In case of incorrect input values, the resulting tangent might not be orthogonal to the normal.
  // This additional check ensures the tangent is orthogonal to the normal and corrects it if necessary.
  glm::vec3 tng = {fvTangent[0], fvTangent[1], fvTangent[2]};
  if(glm::abs(glm::dot(tng, *normal)) < 0.9f)
  {
    // The sign is flipped for Vulkan as the texture coordinates are flipped from OpenGL
    *tangent = {fvTangent[0], fvTangent[1], fvTangent[2], -fSign};
  }
  else
  {
    *tangent = shaderio::makeFastTangent(*normal);
  }
}

//--------------------------------------------------------------------------------------------------
// Simple tangent space generation without MikkTSpace
// This is a fallback method that generates basic tangent vectors
void simpleCreateTangents(SMikkTSpaceContext* mikkContext)
{
  auto userData = static_cast<UserData*>(mikkContext->m_pUserData);
  tinygltf::utils::simpleCreateTangents(*userData->model, *userData->primitive);
}

//--------------------------------------------------------------------------------------------------
// Main function to recompute tangents for all primitives in the model
// Parameters:
//   model: The glTF model to process
//   forceCreation: If true, creates TANGENT attribute if it doesn't exist
//   mikktspace: If true, uses MikkTSpace for computation, otherwise uses simple method
void recomputeTangents(tinygltf::Model& model, bool forceCreation, bool mikktspace)
{
  SCOPED_TIMER(__FUNCTION__);

  // Set up MikkTSpace interface
  SMikkTSpaceInterface mikkInterface   = {};
  mikkInterface.m_getNumFaces          = getNumFaces;
  mikkInterface.m_getNumVerticesOfFace = getNumVerticesOfFace;
  mikkInterface.m_getPosition          = getPosition;
  mikkInterface.m_getNormal            = getNormal;
  mikkInterface.m_getTexCoord          = getTexCoord;
  mikkInterface.m_setTSpaceBasic       = setTSpaceBasic;

  // Collect all valid primitives that have required attributes
  std::vector<UserData> userDatas;
  for(auto& mesh : model.meshes)
  {
    for(auto& primitive : mesh.primitives)
    {
      if(primitive.attributes.find("POSITION") != primitive.attributes.end()
         && primitive.attributes.find("NORMAL") != primitive.attributes.end()
         && primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end())
      {
        bool hasTangent = (primitive.attributes.find("TANGENT") != primitive.attributes.end());

        if(forceCreation && !hasTangent)
        {
          tinygltf::utils::createTangentAttribute(model, primitive);
          hasTangent = true;
        }

        if(!hasTangent)
          continue;

        UserData userData{};
        userData.model            = &model;
        userData.primitive        = &primitive;
        userData.posAccessorIndex = primitive.attributes.at("POSITION");
        userData.nrmAccessorIndex = primitive.attributes.at("NORMAL");
        userData.uvAccessorIndex  = primitive.attributes.at("TEXCOORD_0");
        userData.tanAccessorIndex = primitive.attributes.at("TANGENT");

        userDatas.push_back(userData);
      }
    }
  }

  // Process primitives in parallel using multiple threads
  uint32_t num_threads = std::min((uint32_t)userDatas.size(), std::thread::hardware_concurrency());
  nvutils::parallel_batches<1>(
      userDatas.size(),
      [&](uint64_t i) {
        SMikkTSpaceContext mikkContext = {};
        mikkContext.m_pInterface       = &mikkInterface;
        mikkContext.m_pUserData        = &userDatas[i];

        if(mikktspace)
          genTangSpaceDefault(&mikkContext);
        else
          simpleCreateTangents(&mikkContext);
      },
      num_threads);
}
