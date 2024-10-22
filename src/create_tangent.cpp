/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2024 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


/**
 * This file goes over all primitives in a glTF model and recomputes the tangent space.
 * It uses the MikkTSpace library to generate the tangent space.
 */


#include <vector>
#include <mutex>
#include <tiny_gltf.h>
#include <mikktspace.h>
#include <glm/gtx/norm.hpp>

#include "fileformats/tinygltf_utils.hpp"
#include "nvh/parallel_work.hpp"
#include "nvvkhl/shaders/func.h"


struct UserData
{
  tinygltf::Model*     model            = nullptr;
  tinygltf::Primitive* primitive        = nullptr;
  int32_t              posAccessorIndex = -1;
  int32_t              nrmAccessorIndex = -1;
  int32_t              uvAccessorIndex  = -1;
  int32_t              tanAccessorIndex = -1;
  bool                 onlyFix          = false;  // Only fix null tangents
};


// MikkTSpace interface functions
static int32_t getNumFaces(const SMikkTSpaceContext* pContext)
{
  const UserData*            userdata  = static_cast<const UserData*>(pContext->m_pUserData);
  const tinygltf::Primitive* primitive = userdata->primitive;
  const tinygltf::Model*     model     = userdata->model;
  const int32_t              count     = static_cast<int32_t>(tinygltf::utils::getIndexCount(*model, *primitive));
  return count / 3;
}

static int32_t getNumVerticesOfFace(const SMikkTSpaceContext* pContext, const int32_t iFace)
{
  return 3;  // Assuming triangles
}

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

inline static void getPosition(const SMikkTSpaceContext* pContext, float fvPosOut[], const int32_t iFace, const int32_t iVert)
{
  const glm::vec3* positions =
      getAttributeData<glm::vec3>(pContext, iFace, iVert, static_cast<const UserData*>(pContext->m_pUserData)->posAccessorIndex);
  std::memcpy(fvPosOut, positions, sizeof(glm::vec3));
}

inline static void getNormal(const SMikkTSpaceContext* pContext, float fvNormOut[], const int32_t iFace, const int32_t iVert)
{
  const glm::vec3* normals =
      getAttributeData<glm::vec3>(pContext, iFace, iVert, static_cast<const UserData*>(pContext->m_pUserData)->nrmAccessorIndex);
  std::memcpy(fvNormOut, normals, sizeof(glm::vec3));
}

inline static void getTexCoord(const SMikkTSpaceContext* pContext, float fvTexcOut[], const int32_t iFace, const int32_t iVert)
{
  const glm::vec2* texcoords =
      getAttributeData<glm::vec2>(pContext, iFace, iVert, static_cast<const UserData*>(pContext->m_pUserData)->uvAccessorIndex);
  std::memcpy(fvTexcOut, texcoords, sizeof(glm::vec2));
}

inline static void setTSpaceBasic(const SMikkTSpaceContext* pContext, const float fvTangent[], const float fSign, const int32_t iFace, const int32_t iVert)
{
  glm::vec4*       tangent = getAttributeData<glm::vec4>(pContext, iFace, iVert,
                                                   static_cast<const UserData*>(pContext->m_pUserData)->tanAccessorIndex);
  const glm::vec3* normal =
      getAttributeData<glm::vec3>(pContext, iFace, iVert, static_cast<const UserData*>(pContext->m_pUserData)->nrmAccessorIndex);

  // If we only want to fix tangents, we skip the ones that are already valid
  UserData*        userdata = static_cast<UserData*>(pContext->m_pUserData);
  const glm::vec4& t        = *tangent;
  if(userdata->onlyFix && (glm::length2(glm::vec3(t)) > 0.01F) && (std::abs(t.w) > 0.5F))
  {
    // Fix non-ortho normal tangents
    if((glm::abs(glm::dot(glm::vec3(t), *normal)) > 0.1f))
      *tangent = makeFastTangent(*normal);
    return;
  }

  // MikkTSpace uses the variation in texture coordinates to calculate the tangent and bitangent vectors.
  // In case of incorrect input values, the resulting tangent might not be orthogonal to the normal.
  // This additional check ensures the tangent is orthogonal to the normal and corrects it if necessary.
  glm::vec3 tng = {fvTangent[0], fvTangent[1], fvTangent[2]};
  if(glm::abs(glm::dot(tng, *normal)) < 0.9f)
  {
    *tangent = {fvTangent[0], fvTangent[1], fvTangent[2], fSign};
  }
  else
  {
    *tangent = makeFastTangent(*normal);
  }
}

inline static void createMissingTangentSpace(tinygltf::Model& model)
{
  for(auto& mesh : model.meshes)
  {
    for(auto& primitive : mesh.primitives)
    {
      // Create TANGENT attribute if it doesn't exist
      if(primitive.attributes.find("TANGENT") == primitive.attributes.end())
      {
        // Create a new TANGENT attribute
        tinygltf::Accessor tangentAccessor;
        tangentAccessor.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
        tangentAccessor.type          = TINYGLTF_TYPE_VEC4;
        tangentAccessor.count         = tinygltf::utils::getVertexCount(model, primitive);

        tinygltf::BufferView tangentBufferView;
        tangentBufferView.buffer     = 0;  // Assume using the first buffer
        tangentBufferView.byteOffset = model.buffers[0].data.size();
        tangentBufferView.byteLength = tangentAccessor.count * 4 * sizeof(float);

        model.buffers[0].data.resize(tangentBufferView.byteOffset + tangentBufferView.byteLength);

        tangentAccessor.bufferView = static_cast<int32_t>(model.bufferViews.size());
        model.bufferViews.push_back(tangentBufferView);

        primitive.attributes["TANGENT"] = static_cast<int32_t>(model.accessors.size());
        model.accessors.push_back(tangentAccessor);
      }
    }
  }
}

// Recompute the tangents for all primitives in the model
// forceCreation: If true, it will create the TANGENT attribute if it doesn't exist
void recomputeTangents(tinygltf::Model& model, bool forceCreation, bool onlyFix)
{
  SMikkTSpaceInterface mikkInterface   = {};
  mikkInterface.m_getNumFaces          = getNumFaces;
  mikkInterface.m_getNumVerticesOfFace = getNumVerticesOfFace;
  mikkInterface.m_getPosition          = getPosition;
  mikkInterface.m_getNormal            = getNormal;
  mikkInterface.m_getTexCoord          = getTexCoord;
  mikkInterface.m_setTSpaceBasic       = setTSpaceBasic;

  if(forceCreation)
  {
    createMissingTangentSpace(model);
  }

  // Collect all valid primitives
  std::vector<UserData> userDatas;
  for(auto& mesh : model.meshes)
  {
    for(auto& primitive : mesh.primitives)
    {
      if(primitive.attributes.find("POSITION") != primitive.attributes.end()
         && primitive.attributes.find("NORMAL") != primitive.attributes.end()
         && primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()
         && primitive.attributes.find("TANGENT") != primitive.attributes.end())
      {
        UserData userData{};
        userData.model            = &model;
        userData.primitive        = &primitive;
        userData.posAccessorIndex = primitive.attributes.at("POSITION");
        userData.nrmAccessorIndex = primitive.attributes.at("NORMAL");
        userData.uvAccessorIndex  = primitive.attributes.at("TEXCOORD_0");
        userData.tanAccessorIndex = primitive.attributes.at("TANGENT");
        userData.onlyFix          = onlyFix;

        userDatas.push_back(userData);
      }
    }
  }

  uint32_t num_threads = std::min((uint32_t)userDatas.size(), std::thread::hardware_concurrency());
  nvh::parallel_batches<1>(
      userDatas.size(),
      [&](uint64_t i) {
        SMikkTSpaceContext mikkContext = {};
        mikkContext.m_pInterface       = &mikkInterface;
        mikkContext.m_pUserData        = &userDatas[i];

        genTangSpaceDefault(&mikkContext);
      },
      num_threads);
}
