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

#include <algorithm>
#include <execution>
#include <filesystem>
#include <unordered_set>

#include <glm/gtx/norm.hpp>
#include <fmt/format.h>
#include <meshoptimizer/src/meshoptimizer.h>

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <nvutils/timers.hpp>

#include "gltf_scene.hpp"
#include "gltf_animation_pointer.hpp"

//--------------------------------------------------------------------------------------------------
// Constructor
//
nvvkgltf::Scene::Scene()
    : m_animationPointer(m_model)
{
  // Base list of supported extensions; samples can add onto this for custom
  // image formats.
  m_supportedExtensions = {
      "KHR_animation_pointer",
      "KHR_lights_punctual",
      "KHR_materials_anisotropy",
      "KHR_materials_clearcoat",
      "KHR_materials_displacement",
      "KHR_materials_emissive_strength",
      "KHR_materials_ior",
      "KHR_materials_iridescence",
      "KHR_materials_sheen",
      "KHR_materials_specular",
      "KHR_materials_transmission",
      "KHR_materials_unlit",
      "KHR_materials_variants",
      "KHR_materials_volume",
      "KHR_materials_volume_scatter",
      "KHR_mesh_quantization",
      "KHR_texture_transform",
      "KHR_materials_dispersion",
      "KHR_node_visibility",
      "EXT_mesh_gpu_instancing",
      "NV_attributes_iray",
      "MSFT_texture_dds",
      "KHR_materials_pbrSpecularGlossiness",
      "KHR_materials_diffuse_transmission",
      "EXT_meshopt_compression",
#ifdef USE_DRACO
      "KHR_draco_mesh_compression",
#endif
#ifdef NVP_SUPPORTS_BASISU
      "KHR_texture_basisu",
#endif
  };
}

//--------------------------------------------------------------------------------------------------
// Loading a GLTF file and extracting all information
//
bool nvvkgltf::Scene::load(const std::filesystem::path& filename)
{
  nvutils::ScopedTimer st(std::string(__FUNCTION__) + "\n");
  const std::string    filenameUtf8 = nvutils::utf8FromPath(filename);
  LOGI("%s%s\n", nvutils::ScopedTimer::indent().c_str(), filenameUtf8.c_str());

  m_validSceneParsed = false;

  m_filename = filename;
  m_model    = {};
  tinygltf::TinyGLTF tcontext;
  std::string        warn;
  std::string        error;
  tcontext.SetMaxExternalFileSize(-1);  // No limit for external files (images, buffers, etc.)
  // We want to delay image loading until SceneVk::createTextureImages, so that
  // we can support DDS, KTX, and load images in parallel.
  // To do this, we give TinyGLTF a load callback that stores raw bytes without decoding.
  // This is especially important for data URIs (base64-encoded images in JSON), where
  // TinyGLTF decodes the base64 and passes the raw image bytes to this callback.
  tcontext.SetImageLoader(
      [](tinygltf::Image* image, const int /* image_idx */, std::string* /* err */, std::string* /* warn */,
         int /* req_width */, int /* req_height */, const unsigned char* bytes, int size, void* /*user_data */) {
        if(bytes != nullptr && size > 0)
        {
          image->image.assign(bytes, bytes + size);
        }
        return true;
      },
      nullptr);
  const std::string ext = nvutils::utf8FromPath(filename.extension());
  bool              result{false};
  if(ext == ".gltf")
  {
    result = tcontext.LoadASCIIFromFile(&m_model, &error, &warn, filenameUtf8);
  }
  else if(ext == ".glb")
  {
    result = tcontext.LoadBinaryFromFile(&m_model, &error, &warn, filenameUtf8);
  }
  else
  {
    LOGE("%sUnknown file extension: %s\n", st.indent().c_str(), ext.c_str());
    return false;
  }

  if(!result)
  {
    LOGW("%sError loading file: %s\n", st.indent().c_str(), filenameUtf8.c_str());
    LOGW("%s%s\n", st.indent().c_str(), warn.c_str());
    // This is LOGE because the user requested to load a (probably valid)
    // glTF file, but this loader can't do what the user asked it to.
    // Only the last one is LOGE so that all the messages print before the
    // breakpoint.
    LOGE("%s%s\n", st.indent().c_str(), error.c_str());
    clearParsedData();
    //assert(!"Error while loading scene");
    return result;
  }

  // Check for required extensions
  for(auto& extension : m_model.extensionsRequired)
  {
    if(m_supportedExtensions.find(extension) == m_supportedExtensions.end())
    {
      LOGE("%sRequired extension unsupported : %s\n", st.indent().c_str(), extension.c_str());
      clearParsedData();
      return false;
    }
  }

  // Check for used extensions
  for(auto& extension : m_model.extensionsUsed)
  {
    if(m_supportedExtensions.find(extension) == m_supportedExtensions.end())
    {
      LOGW("%sUsed extension unsupported : %s\n", st.indent().c_str(), extension.c_str());
    }
  }

  // Handle EXT_meshopt_compression by decompressing all buffer data at once
  if(std::find(m_model.extensionsUsed.begin(), m_model.extensionsUsed.end(), EXT_MESHOPT_COMPRESSION_EXTENSION_NAME)
     != m_model.extensionsUsed.end())
  {
    for(tinygltf::Buffer& buffer : m_model.buffers)
    {
      if(buffer.data.empty())
      {
        buffer.data.resize(buffer.byteLength);
        buffer.extensions.erase(EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
      }
    }

    // first used to tag buffers that can be removed after decompression
    std::vector<int> isFullyCompressedBuffer(m_model.buffers.size(), 1);

    for(auto& bufferView : m_model.bufferViews)
    {
      if(bufferView.buffer < 0)
        continue;

      bool warned = false;

      EXT_meshopt_compression mcomp;
      if(tinygltf::utils::getMeshoptCompression(bufferView, mcomp))
      {
        // this decoding logic was derived from `decompressMeshopt`
        // in https://github.com/zeux/meshoptimizer/blob/master/gltf/parsegltf.cpp


        const tinygltf::Buffer& sourceBuffer = m_model.buffers[mcomp.buffer];
        const unsigned char*    source       = &sourceBuffer.data[mcomp.byteOffset];
        assert(mcomp.byteOffset + mcomp.byteLength <= sourceBuffer.data.size());

        tinygltf::Buffer& resultBuffer = m_model.buffers[bufferView.buffer];
        unsigned char*    result       = &resultBuffer.data[bufferView.byteOffset];
        assert(bufferView.byteOffset + bufferView.byteLength <= resultBuffer.data.size());

        int  rc   = -1;
        bool warn = false;

        switch(mcomp.compressionMode)
        {
          case EXT_meshopt_compression::MESHOPT_COMPRESSION_MODE_ATTRIBUTES:
            warn = meshopt_decodeVertexVersion(source, mcomp.byteLength) < 0;
            rc   = meshopt_decodeVertexBuffer(result, mcomp.count, mcomp.byteStride, source, mcomp.byteLength);
            break;

          case EXT_meshopt_compression::MESHOPT_COMPRESSION_MODE_TRIANGLES:
            warn = meshopt_decodeIndexVersion(source, mcomp.byteLength) < 0;
            rc   = meshopt_decodeIndexBuffer(result, mcomp.count, mcomp.byteStride, source, mcomp.byteLength);
            break;

          case EXT_meshopt_compression::MESHOPT_COMPRESSION_MODE_INDICES:
            warn = meshopt_decodeIndexVersion(source, mcomp.byteLength) < 0;
            rc   = meshopt_decodeIndexSequence(result, mcomp.count, mcomp.byteStride, source, mcomp.byteLength);
            break;

          default:
            break;
        }

        if(rc != 0)
        {
          LOGW("EXT_meshopt_compression decompression failed\n");
          clearParsedData();
          return false;
        }

        if(warn && !warned)
        {
          LOGW("Warning: EXT_meshopt_compression data uses an unsupported or invalid encoding version\n");
          warned = true;
        }

        switch(mcomp.compressionFilter)
        {
          case EXT_meshopt_compression::MESHOPT_COMPRESSION_FILTER_OCTAHEDRAL:
            meshopt_decodeFilterOct(result, mcomp.count, mcomp.byteStride);
            break;

          case EXT_meshopt_compression::MESHOPT_COMPRESSION_FILTER_QUATERNION:
            meshopt_decodeFilterQuat(result, mcomp.count, mcomp.byteStride);
            break;

          case EXT_meshopt_compression::MESHOPT_COMPRESSION_FILTER_EXPONENTIAL:
            meshopt_decodeFilterExp(result, mcomp.count, mcomp.byteStride);
            break;

          default:
            break;
        }

        // remove extension for saving uncompressed
        bufferView.extensions.erase(EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
      }

      isFullyCompressedBuffer[bufferView.buffer] = 0;
    }

    // remove fully compressed buffers
    // isFullyCompressedBuffer is repurposed as buffer index remap table
    size_t writeIndex = 0;
    for(size_t readIndex = 0; readIndex < m_model.buffers.size(); readIndex++)
    {
      if(isFullyCompressedBuffer[readIndex])
      {
        // buffer is removed
        isFullyCompressedBuffer[readIndex] = -1;
      }
      else
      {
        // compacted index of buffer
        isFullyCompressedBuffer[readIndex] = int(writeIndex);

        if(readIndex != writeIndex)
        {
          m_model.buffers[writeIndex] = std::move(m_model.buffers[readIndex]);
        }
        writeIndex++;
      }
    }
    m_model.buffers.resize(writeIndex);

    // remap existing buffer views
    for(auto& bufferView : m_model.bufferViews)
    {
      if(bufferView.buffer < 0)
        continue;

      bufferView.buffer = isFullyCompressedBuffer[bufferView.buffer];
    }

    // remove extension
    std::erase(m_model.extensionsRequired, EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
    std::erase(m_model.extensionsUsed, EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
  }

  m_currentScene   = m_model.defaultScene > -1 ? m_model.defaultScene : 0;
  m_currentVariant = 0;        // Default KHR_materials_variants
  m_animationPointer.reset();  // Clear cached state from previous model
  parseScene();

  m_validSceneParsed = !m_renderNodes.empty();

  return m_validSceneParsed;
}

bool nvvkgltf::Scene::save(const std::filesystem::path& filename)
{
  namespace fs = std::filesystem;

  nvutils::ScopedTimer st(std::string(__FUNCTION__) + "\n");

  std::filesystem::path saveFilename = filename;

  // Make sure the extension is correct
  if(!nvutils::extensionMatches(filename, ".gltf") && !nvutils::extensionMatches(filename, ".glb"))
  {
    // replace the extension
    saveFilename = saveFilename.replace_extension(".gltf");
  }

  const bool saveBinary = nvutils::extensionMatches(filename, ".glb");

  // Copy the images to the destination folder
  if(!m_model.images.empty() && !saveBinary)
  {
    fs::path srcPath   = m_filename.parent_path();
    fs::path dstPath   = filename.parent_path();
    int      numCopied = 0;
    for(auto& image : m_model.images)
    {
      if(image.uri.empty())
        continue;
      std::string uri_decoded;
      tinygltf::URIDecode(image.uri, &uri_decoded, nullptr);  // ex. whitespace may be represented as %20

      fs::path srcFile = srcPath / uri_decoded;
      fs::path dstFile = dstPath / uri_decoded;
      if(srcFile != dstFile)
      {
        // Create the parent directory of the destination file if it doesn't exist
        fs::create_directories(dstFile.parent_path());

        try
        {
          if(fs::copy_file(srcFile, dstFile, fs::copy_options::update_existing))
            numCopied++;
        }
        catch(const fs::filesystem_error& e)
        {
          LOGW("%sError copying image: %s\n", st.indent().c_str(), e.what());
        }
      }
    }
    if(numCopied > 0)
      LOGI("%sImages copied: %d\n", st.indent().c_str(), numCopied);
  }

  // Save the glTF file
  tinygltf::TinyGLTF tcontext;
  const std::string  saveFilenameUtf8 = nvutils::utf8FromPath(saveFilename);
  bool result = tcontext.WriteGltfSceneToFile(&m_model, saveFilenameUtf8, saveBinary, saveBinary, true, saveBinary);
  LOGI("%sSaved: %s\n", st.indent().c_str(), saveFilenameUtf8.c_str());
  return result;
}


void nvvkgltf::Scene::takeModel(tinygltf::Model&& model)
{
  m_model = std::move(model);
  m_animationPointer.reset();  // Clear cached state from previous model
  parseScene();
}

void nvvkgltf::Scene::setCurrentScene(int sceneID)
{
  assert(sceneID >= 0 && sceneID < static_cast<int>(m_model.scenes.size()) && "Invalid scene ID");
  m_currentScene = sceneID;
  parseScene();
}

// Parses the scene from the glTF model, initializing and setting up scene elements, materials, animations, and the camera.
void nvvkgltf::Scene::parseScene()
{
  // Ensure there are nodes in the glTF model and the current scene ID is valid
  assert(m_model.nodes.size() > 0 && "No nodes in the glTF file");
  assert(m_currentScene >= 0 && m_currentScene < static_cast<int>(m_model.scenes.size()) && "Invalid scene ID");

  // Clear previous scene data and initialize scene elements
  clearParsedData();
  setSceneElementsDefaultNames();

  // There must be at least one material in the scene
  if(m_model.materials.empty())
  {
    m_model.materials.emplace_back();
  }

  // Collect all draw objects; RenderNode and RenderPrimitive
  // Also it will be used  to compute the scene bounds for the camera
  for(auto& sceneNode : m_model.scenes[m_currentScene].nodes)
  {
    tinygltf::utils::traverseSceneGraph(
        m_model, sceneNode, glm::mat4(1), nullptr,
        [this](int nodeID, const glm::mat4& worldMat) { return handleLightTraversal(nodeID, worldMat); },
        [this](int nodeID, const glm::mat4& worldMat) { return handleRenderNode(nodeID, worldMat); });
  }

  // Search for the first camera in the scene and exit traversal upon finding it
  for(auto& sceneNode : m_model.scenes[m_currentScene].nodes)
  {
    tinygltf::utils::traverseSceneGraph(
        m_model, sceneNode, glm::mat4(1),
        [&](int nodeID, glm::mat4) {
          m_sceneCameraNode = nodeID;
          return true;  // Stop traversal
        },
        nullptr, nullptr);
  }

  // Create a default camera if none is found in the scene
  if(m_sceneCameraNode == -1)
  {
    createSceneCamera();
  }

  // Parse various scene components
  parseVariants();
  parseAnimations();
  createMissingTangents();

  // We are updating the scene to the first state, animation, skinning, morph, ..
  updateRenderNodesFull();
}


// This function recursively updates the visibility of nodes in the scene graph.
// If a node is marked as not visible, all its children will also be marked as not visible,
// regardless of their individual visibility flags.
void nvvkgltf::Scene::updateVisibility(int nodeID)
{
  std::function<void(int, bool)> processNode;
  processNode = [&](int nodeID, bool visible) -> void {
    const tinygltf::Node& node = m_model.nodes[nodeID];
    if(visible)
    {
      // Changing the visibility only if the parent was visible
      visible = tinygltf::utils::getNodeVisibility(node).visible;
    }

    for(auto renderNodeID : m_nodeToRenderNodes[nodeID])
      m_renderNodes[renderNodeID].visible = visible;

    for(auto& child : node.children)
    {
      processNode(child, visible);
    }
  };

  const tinygltf::Node& node    = m_model.nodes[nodeID];
  bool                  visible = tinygltf::utils::getNodeVisibility(node).visible;
  processNode(nodeID, visible);
}

// Set the default names for the scene elements if they are empty
void nvvkgltf::Scene::setSceneElementsDefaultNames()
{
  auto setDefaultName = [](auto& elements, const std::string& prefix) {
    for(size_t i = 0; i < elements.size(); ++i)
    {
      if(elements[i].name.empty())
      {
        elements[i].name = fmt::format("{}-{}", prefix, i);
      }
    }
  };

  setDefaultName(m_model.scenes, "Scene");
  setDefaultName(m_model.meshes, "Mesh");
  setDefaultName(m_model.materials, "Material");
  setDefaultName(m_model.nodes, "Node");
  setDefaultName(m_model.cameras, "Camera");
  setDefaultName(m_model.lights, "Light");
}


// If there is no camera in the scene, we create one
// The camera is placed at the center of the scene, looking at the scene
void nvvkgltf::Scene::createSceneCamera()
{
  tinygltf::Camera& tcamera        = m_model.cameras.emplace_back();  // Add a camera
  int               newCameraIndex = static_cast<int>(m_model.cameras.size() - 1);
  tinygltf::Node&   tnode          = m_model.nodes.emplace_back();  // Add a node for the camera
  int               newNodeIndex   = static_cast<int>(m_model.nodes.size() - 1);
  tnode.name                       = "Camera";
  tnode.camera                     = newCameraIndex;
  m_model.scenes[m_currentScene].nodes.push_back(newNodeIndex);  // Add the camera node to the scene

  // Set the camera to look at the scene
  nvutils::Bbox bbox   = getSceneBounds();
  glm::vec3     center = bbox.center();
  glm::vec3 eye = center + glm::vec3(0, 0, bbox.radius() * 2.414f);  //2.414 units away from the center of the sphere to fit it within a 45 - degree FOV
  glm::vec3 up                    = glm::vec3(0, 1, 0);
  tcamera.type                    = "perspective";
  tcamera.name                    = "Camera";
  tcamera.perspective.aspectRatio = 16.0f / 9.0f;
  tcamera.perspective.yfov        = glm::radians(45.0f);
  tcamera.perspective.zfar        = bbox.radius() * 10.0f;
  tcamera.perspective.znear       = bbox.radius() * 0.1f;

  // Add extra information to the node/camera
  tinygltf::Value::Object extras;
  extras["camera::eye"]    = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(eye));
  extras["camera::center"] = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(center));
  extras["camera::up"]     = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(up));
  tnode.extras             = tinygltf::Value(extras);

  // Set the node transformation
  tnode.translation = {eye.x, eye.y, eye.z};
  glm::quat q       = glm::quatLookAt(glm::normalize(center - eye), up);
  tnode.rotation    = {q.x, q.y, q.z, q.w};
}

//--------------------------------------------------------------------------------------------
// This function will update the world matrices of the render nodes
//
void nvvkgltf::Scene::updateNodeWorldMatrices(const std::unordered_set<int>& dirtyNodeIds)
{
  const tinygltf::Scene& scene = m_model.scenes[m_currentScene];
  assert(scene.nodes.size() > 0 && "No nodes in the glTF file");

  if(dirtyNodeIds.empty())
  {
    // Full update
    updateRenderNodesFull();
    return;
  }

  // Partial update
  for(int nodeID : dirtyNodeIds)
  {
    tinygltf::Node& node         = m_model.nodes[nodeID];
    m_nodesLocalMatrices[nodeID] = tinygltf::utils::getNodeMatrix(node);
  }

  std::unordered_set<int> filteredDirtyNodes;
  filteredDirtyNodes.reserve(dirtyNodeIds.size());
  for(int nodeID : dirtyNodeIds)
  {
    bool hasParentInDirty = false;
    int  currentParent    = m_nodeParents[nodeID];
    while(currentParent >= 0)
    {
      if(dirtyNodeIds.contains(currentParent))
      {
        hasParentInDirty = true;
        break;
      }
      currentParent = m_nodeParents[currentParent];
    }

    if(!hasParentInDirty)
    {
      filteredDirtyNodes.insert(nodeID);
    }
  }

  std::function<void(int)> updateMatrix;
  updateMatrix = [&](int nodeID) -> void {
    tinygltf::Node& node = m_model.nodes[nodeID];
    glm::mat4 parentMat  = m_nodeParents[nodeID] >= 0 ? m_nodesWorldMatrices[m_nodeParents[nodeID]] : glm::mat4(1.0f);
    m_nodesWorldMatrices[nodeID] = parentMat * m_nodesLocalMatrices[nodeID];

    for(auto renderNodeID : m_nodeToRenderNodes[nodeID])
    {
      m_renderNodes[renderNodeID].worldMatrix = m_nodesWorldMatrices[nodeID];
    }

    if(node.light >= 0)
    {
      m_lights[node.light].worldMatrix = m_nodesWorldMatrices[nodeID];
    }

    for(const auto& child : node.children)
    {
      updateMatrix(child);
    }
  };

  for(auto nodeID : filteredDirtyNodes)
  {
    updateMatrix(nodeID);
  }
}

//--------------------------------------------------------------------------------------------------
// Update all the render nodes in the scene and collecting information about
// the node's parent,  and the render node indices for each node.
void nvvkgltf::Scene::updateRenderNodesFull()
{
  const tinygltf::Scene& scene = m_model.scenes[m_currentScene];
  m_nodesLocalMatrices.resize(m_model.nodes.size(), glm::mat4(1.0f));
  m_nodesWorldMatrices.resize(m_model.nodes.size());
  m_nodeParents.resize(m_model.nodes.size());
  m_nodeParents.assign(m_model.nodes.size(), -1);
  m_nodeToRenderNodes = {};
  m_nodeToRenderNodes.resize(m_model.nodes.size());

  int32_t renderNodeID = 0;  // Index of the render node

  // Recursive lambda function to traverse the scene nodes
  std::function<void(int, const glm::mat4&, bool)> traverseNodes;
  traverseNodes = [&](int nodeID, const glm::mat4& parentMat, bool visible) {
    const tinygltf::Node& node   = m_model.nodes[nodeID];
    m_nodesLocalMatrices[nodeID] = tinygltf::utils::getNodeMatrix(node);
    const glm::mat4 worldMat     = parentMat * m_nodesLocalMatrices[nodeID];
    tinygltf::Node& tnode        = m_model.nodes[nodeID];

    if(visible)
    {
      visible = tinygltf::utils::getNodeVisibility(tnode).visible;
    }

    if(tnode.light > -1)
    {
      m_lights[tnode.light].worldMatrix = worldMat;
    }

    if(tnode.mesh > -1)
    {
      const tinygltf::Mesh& mesh = m_model.meshes[tnode.mesh];
      for(const tinygltf::Primitive& primitive : mesh.primitives)
      {
        nvvkgltf::RenderNode& renderNode = m_renderNodes[renderNodeID];
        renderNode.worldMatrix           = worldMat;
        renderNode.materialID            = getMaterialVariantIndex(primitive, m_currentVariant);
        renderNode.visible               = visible;
        m_nodeToRenderNodes[nodeID].push_back(renderNodeID);
        renderNodeID++;
      }
    }

    m_nodesWorldMatrices[nodeID] = worldMat;
    for(const auto& child : tnode.children)
    {
      m_nodeParents[child] = nodeID;
      traverseNodes(child, worldMat, visible);
    }
  };

  // Traverse the scene nodes and collect the render node indices
  for(auto& sceneNode : scene.nodes)
  {
    bool visible = tinygltf::utils::getNodeVisibility(m_model.nodes[sceneNode]).visible;
    traverseNodes(sceneNode, glm::mat4(1), visible);
  }
}

//-----------------------------------------------------------
//
void nvvkgltf::Scene::setCurrentVariant(int variant, std::unordered_set<int>& dirtyRenderNodes)
{
  m_currentVariant = variant;
  dirtyRenderNodes.clear();

  for(size_t i = 0; i < m_nodeToRenderNodes.size(); i++)
  {
    if(m_nodeToRenderNodes[i].empty())
      continue;
    tinygltf::Node& tnode             = m_model.nodes[i];
    int             firstRenderNodeID = m_nodeToRenderNodes[i][0];
    if(tnode.mesh > -1)
    {
      tinygltf::Mesh& mesh = m_model.meshes[tnode.mesh];
      for(size_t primID = 0; primID < mesh.primitives.size(); primID++)
      {
        int renderNodeID = firstRenderNodeID + int(primID);
        int beforeMatID  = m_renderNodes[renderNodeID].materialID;
        int newMatId     = getMaterialVariantIndex(mesh.primitives[primID], m_currentVariant);
        if(beforeMatID != newMatId)
          dirtyRenderNodes.insert(renderNodeID);
        m_renderNodes[firstRenderNodeID + primID].materialID = newMatId;
      }
    }
  }
}


void nvvkgltf::Scene::clearParsedData()
{
  m_cameras.clear();
  m_lights.clear();
  m_animations.clear();
  m_renderNodes.clear();
  m_renderPrimitives.clear();
  m_uniquePrimitiveIndex.clear();
  m_variants.clear();
  m_nodeToRenderNodes.clear();
  m_nodeParents.clear();
  m_nodesLocalMatrices.clear();
  m_numTriangles    = 0;
  m_sceneBounds     = {};
  m_sceneCameraNode = -1;
}

void nvvkgltf::Scene::destroy()
{
  clearParsedData();
  m_filename.clear();
  m_validSceneParsed = false;
  m_model            = {};
  m_animationPointer.reset();  // Clear cached state when destroying the scene
}


// Get the unique index of a primitive, and add it to the list if it is not already there
int nvvkgltf::Scene::getUniqueRenderPrimitive(tinygltf::Primitive& primitive, int meshID)
{
  const std::string& key = tinygltf::utils::generatePrimitiveKey(primitive);

  // Attempt to insert the key with the next available index if it doesn't exist
  auto [it, inserted] = m_uniquePrimitiveIndex.try_emplace(key, static_cast<int>(m_uniquePrimitiveIndex.size()));

  // If the primitive was newly inserted, add it to the render primitives list
  if(inserted)
  {
    nvvkgltf::RenderPrimitive renderPrim;
    renderPrim.pPrimitive  = &primitive;
    renderPrim.vertexCount = int(tinygltf::utils::getVertexCount(m_model, primitive));
    renderPrim.indexCount  = int(tinygltf::utils::getIndexCount(m_model, primitive));
    renderPrim.meshID      = meshID;
    m_renderPrimitives.push_back(renderPrim);
  }

  return it->second;
}


// Function to extract eye, center, and up vectors from a view matrix
inline void extractCameraVectors(const glm::mat4& viewMatrix, const glm::vec3& sceneCenter, glm::vec3& eye, glm::vec3& center, glm::vec3& up)
{
  eye                    = glm::vec3(viewMatrix[3]);
  glm::mat3 rotationPart = glm::mat3(viewMatrix);
  glm::vec3 forward      = -rotationPart * glm::vec3(0.0f, 0.0f, 1.0f);

  // Project sceneCenter onto the forward vector
  glm::vec3 eyeToSceneCenter = sceneCenter - eye;
  float     projectionLength = std::abs(glm::dot(eyeToSceneCenter, forward));
  center                     = eye + projectionLength * forward;

  up = glm::vec3(0.0f, 1.0f, 0.0f);  // Assume the up vector is always (0, 1, 0)
}


// Retrieve the list of render cameras in the scene.
// This function returns a vector of render cameras present in the scene. If the `force`
// parameter is set to true, it clears and regenerates the list of cameras.
//
// Parameters:
// - force: If true, forces the regeneration of the camera list.
//
// Returns:
// - A const reference to the vector of render cameras.
const std::vector<nvvkgltf::RenderCamera>& nvvkgltf::Scene::getRenderCameras(bool force /*= false*/)
{
  if(force)
  {
    m_cameras.clear();
  }

  if(m_cameras.empty())
  {
    for(auto& sceneNode : m_model.scenes[m_currentScene].nodes)
    {
      tinygltf::utils::traverseSceneGraph(m_model, sceneNode, glm::mat4(1), [&](int nodeID, const glm::mat4& worldMatrix) {
        return handleCameraTraversal(nodeID, worldMatrix);
      });
    }
  }
  return m_cameras;
}


bool nvvkgltf::Scene::handleCameraTraversal(int nodeID, const glm::mat4& worldMatrix)
{
  tinygltf::Node& node = m_model.nodes[nodeID];
  m_sceneCameraNode    = nodeID;

  tinygltf::Camera&      tcam = m_model.cameras[node.camera];
  nvvkgltf::RenderCamera camera;
  if(tcam.type == "perspective")
  {
    camera.type  = nvvkgltf::RenderCamera::CameraType::ePerspective;
    camera.znear = tcam.perspective.znear;
    camera.zfar  = tcam.perspective.zfar;
    camera.yfov  = tcam.perspective.yfov;
  }
  else
  {
    camera.type  = nvvkgltf::RenderCamera::CameraType::eOrthographic;
    camera.znear = tcam.orthographic.znear;
    camera.zfar  = tcam.orthographic.zfar;
    camera.xmag  = tcam.orthographic.xmag;
    camera.ymag  = tcam.orthographic.ymag;
  }

  nvutils::Bbox bbox = getSceneBounds();

  // Validate zFar
  if(camera.zfar <= camera.znear)
  {
    camera.zfar = std::max(camera.znear * 2.0, 4.0 * bbox.radius());
    LOGW("glTF: Camera zFar is less than zNear, max(zNear * 2, 4 * bbos.radius()\n");
  }

  // From the view matrix, we extract the eye, center, and up vectors
  extractCameraVectors(worldMatrix, bbox.center(), camera.eye, camera.center, camera.up);

  // If the node/camera has extras, we extract the eye, center, and up vectors from the extras
  auto& extras = node.extras;
  if(extras.IsObject())
  {
    tinygltf::utils::getArrayValue(extras, "camera::eye", camera.eye);
    tinygltf::utils::getArrayValue(extras, "camera::center", camera.center);
    tinygltf::utils::getArrayValue(extras, "camera::up", camera.up);
  }

  m_cameras.push_back(camera);
  return false;
}

bool nvvkgltf::Scene::handleLightTraversal(int nodeID, const glm::mat4& worldMatrix)
{
  tinygltf::Node&       node = m_model.nodes[nodeID];
  nvvkgltf::RenderLight renderLight;
  renderLight.light      = node.light;
  tinygltf::Light& light = m_model.lights[node.light];
  // Add a default color if the light has no color
  if(light.color.empty())
  {
    light.color = {1.0f, 1.0f, 1.0f};
  }
  // Add a default radius if the light has no radius
  if(!light.extras.Has("radius"))
  {
    if(!light.extras.IsObject())
    {  // Avoid overwriting other extras
      light.extras = tinygltf::Value(tinygltf::Value::Object());
    }
    tinygltf::Value::Object extras = light.extras.Get<tinygltf::Value::Object>();
    extras["radius"]               = tinygltf::Value(0.);
    light.extras                   = tinygltf::Value(extras);
  }
  renderLight.worldMatrix = worldMatrix;

  m_lights.push_back(renderLight);
  return false;  // Continue traversal
}


// Return the bounding volume of the scene
nvutils::Bbox nvvkgltf::Scene::getSceneBounds()
{
  if(!m_sceneBounds.isEmpty())
    return m_sceneBounds;

  for(const nvvkgltf::RenderNode& rnode : m_renderNodes)
  {
    glm::vec3 minValues = {0.f, 0.f, 0.f};
    glm::vec3 maxValues = {0.f, 0.f, 0.f};

    const nvvkgltf::RenderPrimitive& rprim    = m_renderPrimitives[rnode.renderPrimID];
    const tinygltf::Accessor&        accessor = m_model.accessors[rprim.pPrimitive->attributes.at("POSITION")];
    if(!accessor.minValues.empty())
      minValues = glm::vec3(accessor.minValues[0], accessor.minValues[1], accessor.minValues[2]);
    if(!accessor.maxValues.empty())
      maxValues = glm::vec3(accessor.maxValues[0], accessor.maxValues[1], accessor.maxValues[2]);
    nvutils::Bbox bbox(minValues, maxValues);
    bbox = bbox.transform(rnode.worldMatrix);
    m_sceneBounds.insert(bbox);
  }

  if(m_sceneBounds.isEmpty() || !m_sceneBounds.isVolume())
  {
    LOGW("glTF: Scene bounding box invalid, Setting to: [-1,-1,-1], [1,1,1]\n");
    m_sceneBounds.insert({-1.0f, -1.0f, -1.0f});
    m_sceneBounds.insert({1.0f, 1.0f, 1.0f});
  }

  return m_sceneBounds;
}

// Handles the creation of render nodes for a given primitive in the scene.
// For each primitive in the node's mesh, it:
// - Generates a unique render primitive index.
// - Creates a render node with the appropriate world matrix, material ID, render primitive ID, primitive ID, and reference node ID.
// If the primitive has the EXT_mesh_gpu_instancing extension, multiple render nodes are created for instancing.
// Otherwise, a single render node is added to the render nodes list.
// Returns false to continue traversal of the scene graph.
bool nvvkgltf::Scene::handleRenderNode(int nodeID, glm::mat4 worldMatrix)
{
  const tinygltf::Node& node = m_model.nodes[nodeID];
  tinygltf::Mesh&       mesh = m_model.meshes[node.mesh];
  for(size_t primID = 0; primID < mesh.primitives.size(); primID++)
  {
    tinygltf::Primitive& primitive    = mesh.primitives[primID];
    int                  rprimID      = getUniqueRenderPrimitive(primitive, node.mesh);
    int                  numTriangles = m_renderPrimitives[rprimID].indexCount / 3;

    nvvkgltf::RenderNode renderNode;
    renderNode.worldMatrix  = worldMatrix;
    renderNode.materialID   = getMaterialVariantIndex(primitive, m_currentVariant);
    renderNode.renderPrimID = rprimID;
    renderNode.refNodeID    = nodeID;
    renderNode.skinID       = node.skin;

    if(tinygltf::utils::hasElementName(node.extensions, EXT_MESH_GPU_INSTANCING_EXTENSION_NAME))
    {
      const tinygltf::Value& ext = tinygltf::utils::getElementValue(node.extensions, EXT_MESH_GPU_INSTANCING_EXTENSION_NAME);
      const tinygltf::Value& attributes   = ext.Get("attributes");
      size_t                 numInstances = handleGpuInstancing(attributes, renderNode, worldMatrix);
      m_numTriangles += numTriangles * static_cast<uint32_t>(numInstances);  // Statistics
    }
    else
    {
      m_renderNodes.push_back(renderNode);
      m_numTriangles += numTriangles;  // Statistics
    }
  }
  return false;  // Continue traversal
}

// Handle GPU instancing : EXT_mesh_gpu_instancing
size_t nvvkgltf::Scene::handleGpuInstancing(const tinygltf::Value& attributes, nvvkgltf::RenderNode renderNode, glm::mat4 worldMatrix)
{
  std::vector<glm::vec3> tStorage;
  std::vector<glm::quat> rStorage;
  std::vector<glm::vec3> sStorage;
  std::span<glm::vec3> translations = tinygltf::utils::getAttributeData3(m_model, attributes, "TRANSLATION", &tStorage);
  std::span<glm::quat> rotations    = tinygltf::utils::getAttributeData3(m_model, attributes, "ROTATION", &rStorage);
  std::span<glm::vec3> scales       = tinygltf::utils::getAttributeData3(m_model, attributes, "SCALE", &sStorage);

  size_t numInstances = std::max({translations.size(), rotations.size(), scales.size()});

  // Note: the specification says, that the number of elements in the attributes should be the same if they are present
  for(size_t i = 0; i < numInstances; i++)
  {
    nvvkgltf::RenderNode instNode    = renderNode;
    glm::vec3            translation = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::quat            rotation    = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3            scale       = glm::vec3(1.0f, 1.0f, 1.0f);
    if(!translations.empty())
      translation = translations[i];
    if(!rotations.empty())
      rotation = rotations[i];
    if(!scales.empty())
      scale = scales[i];

    glm::mat4 mat = glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);

    instNode.worldMatrix = worldMatrix * mat;
    m_renderNodes.push_back(instNode);
  }
  return numInstances;
}

//-------------------------------------------------------------------------------------------------
// Add tangents on primitives that have normal maps but no tangents
void nvvkgltf::Scene::createMissingTangents()
{
  std::vector<int> missTangentPrimitives;

  for(const auto& renderNode : m_renderNodes)
  {
    // Check for missing tangents if the primitive has normalmap
    if(m_model.materials[renderNode.materialID].normalTexture.index >= 0)
    {
      int                  renderPrimID = renderNode.renderPrimID;
      tinygltf::Primitive& primitive    = *m_renderPrimitives[renderPrimID].pPrimitive;

      if(primitive.attributes.find("TANGENT") == primitive.attributes.end())
      {
        LOGW("Render Primitive %d has a normal map but no tangents. Generating tangents.\n", renderPrimID);
        tinygltf::utils::createTangentAttribute(m_model, primitive);
        missTangentPrimitives.push_back(renderPrimID);  // Will generate the tangents later
      }
    }
  }

  // Generate the tangents in parallel
  nvutils::parallel_batches<1>(missTangentPrimitives.size(), [&](uint64_t primID) {
    tinygltf::Primitive& primitive = *m_renderPrimitives[missTangentPrimitives[primID]].pPrimitive;
    tinygltf::utils::simpleCreateTangents(m_model, primitive);
  });
}

//-------------------------------------------------------------------------------------------------
// Find which render nodes use the given material variant IDs
//
std::unordered_set<int> nvvkgltf::Scene::getMaterialRenderNodes(const std::unordered_set<int>& materialVariantNodeIDs) const
{
  std::unordered_set<int> renderNodes;
  for(size_t i = 0; i < m_renderNodes.size(); i++)
  {
    if(materialVariantNodeIDs.contains(m_renderNodes[i].materialID))
    {
      renderNodes.insert(int(i));
    }
  }
  return renderNodes;
}

//-------------------------------------------------------------------------------------------------
// Find which nodes are solid or translucent, helps for raster rendering
//
std::vector<uint32_t> nvvkgltf::Scene::getShadedNodes(PipelineType type)
{
  std::vector<uint32_t> result;

  for(uint32_t i = 0; i < m_renderNodes.size(); i++)
  {
    const auto& tmat               = m_model.materials[m_renderNodes[i].materialID];
    float       transmissionFactor = 0;
    if(tinygltf::utils::hasElementName(tmat.extensions, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME))
    {
      const auto& ext = tinygltf::utils::getElementValue(tmat.extensions, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME);
      tinygltf::utils::getValue(ext, "transmissionFactor", transmissionFactor);
    }
    switch(type)
    {
      case eRasterSolid:
        if(tmat.alphaMode == "OPAQUE" && !tmat.doubleSided && (transmissionFactor == 0.0F))
          result.push_back(i);
        break;
      case eRasterSolidDoubleSided:
        if(tmat.alphaMode == "OPAQUE" && tmat.doubleSided)
          result.push_back(i);
        break;
      case eRasterBlend:
        if(tmat.alphaMode != "OPAQUE" || (transmissionFactor != 0))
          result.push_back(i);
        break;
      case eRasterAll:
        result.push_back(i);
        break;
    }
  }
  return result;
}


namespace {
void applyRenderCameraToNode(tinygltf::Node& tnode, tinygltf::Camera& tcamera, const nvvkgltf::RenderCamera& camera)
{
  glm::quat q       = glm::quatLookAt(glm::normalize(camera.center - camera.eye), camera.up);
  tnode.translation = {camera.eye.x, camera.eye.y, camera.eye.z};
  tnode.rotation    = {q.x, q.y, q.z, q.w};

  if(camera.type == nvvkgltf::RenderCamera::CameraType::eOrthographic)
  {
    tcamera.type               = "orthographic";
    tcamera.orthographic.znear = camera.znear;
    tcamera.orthographic.zfar  = camera.zfar;
    tcamera.orthographic.xmag  = camera.xmag;
    tcamera.orthographic.ymag  = camera.ymag;
  }
  else
  {
    tcamera.type              = "perspective";
    tcamera.perspective.znear = camera.znear;
    tcamera.perspective.zfar  = camera.zfar;
    tcamera.perspective.yfov  = camera.yfov;
  }

  tinygltf::Value::Object extras;
  extras["camera::eye"]    = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(camera.eye));
  extras["camera::center"] = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(camera.center));
  extras["camera::up"]     = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(camera.up));
  tnode.extras             = tinygltf::Value(extras);
}
}  // namespace

void nvvkgltf::Scene::setSceneCamera(const nvvkgltf::RenderCamera& camera)
{
  assert(m_sceneCameraNode != -1 && "No camera node found in the scene");

  tinygltf::Node&   tnode   = m_model.nodes[m_sceneCameraNode];
  tinygltf::Camera& tcamera = m_model.cameras[tnode.camera];
  applyRenderCameraToNode(tnode, tcamera, camera);
}

//--------------------------------------------------------------------------------------------------
// Set the scene cameras
// The cameras are stored in the model as nodes, and the camera index is stored in the node
void nvvkgltf::Scene::setSceneCameras(const std::vector<nvvkgltf::RenderCamera>& cameras)
{
  assert(!cameras.empty() && "cameras must not be empty");

  std::vector<int> cameraNodeIds;
  for(auto& sceneNode : m_model.scenes[m_currentScene].nodes)
  {
    tinygltf::utils::traverseSceneGraph(m_model, sceneNode, glm::mat4(1), [&](int nodeID, const glm::mat4&) {
      if(m_model.nodes[nodeID].camera >= 0)
        cameraNodeIds.push_back(nodeID);
      return false;
    });
  }

  // Adjust the number of cameras
  m_model.cameras.resize(cameras.size());

  for(size_t i = 0; i < cameras.size(); ++i)
  {
    int nodeIndex = -1;
    // If the node camera already exists, use it
    if(i < cameraNodeIds.size())
    {
      nodeIndex = cameraNodeIds[i];
    }
    // If the node camera does not exist, add a new node to hold the camera
    else
    {
      tinygltf::Node& tnode = m_model.nodes.emplace_back();
      nodeIndex             = static_cast<int>(m_model.nodes.size() - 1);
      tnode.name            = fmt::format("Camera-{}", i);
      m_model.scenes[m_currentScene].nodes.push_back(nodeIndex);
    }

    tinygltf::Node& tnode = m_model.nodes[nodeIndex];
    tnode.camera          = static_cast<int>(i);
    applyRenderCameraToNode(tnode, m_model.cameras[i], cameras[i]);
  }

  // Set all other cameras nodes to the first camera
  for(size_t i = cameras.size(); i < cameraNodeIds.size(); ++i)
  {
    m_model.nodes[cameraNodeIds[i]].camera = 0;  // Re-using the first camera
  }
}

// Collects all animation data
void nvvkgltf::Scene::parseAnimations()
{
  m_animations.clear();
  m_animations.reserve(m_model.animations.size());
  for(tinygltf::Animation& anim : m_model.animations)
  {
    Animation animation;
    animation.info.name = anim.name;
    if(animation.info.name.empty())
    {
      animation.info.name = "Animation" + std::to_string(m_animations.size());
    }

    // Samplers
    for(auto& samp : anim.samplers)
    {
      AnimationSampler sampler;

      if(samp.interpolation == "LINEAR")
      {
        sampler.interpolation = AnimationSampler::InterpolationType::eLinear;
      }
      if(samp.interpolation == "STEP")
      {
        sampler.interpolation = AnimationSampler::InterpolationType::eStep;
      }
      if(samp.interpolation == "CUBICSPLINE")
      {
        sampler.interpolation = AnimationSampler::InterpolationType::eCubicSpline;
      }

      // Read sampler input time values
      {
        const tinygltf::Accessor& accessor = m_model.accessors[samp.input];
        if(!tinygltf::utils::copyAccessorData(m_model, accessor, sampler.inputs))
        {
          LOGE("Invalid data type for animation input");
          continue;
        }

        // Protect against invalid values
        for(auto input : sampler.inputs)
        {
          if(input < animation.info.start)
          {
            animation.info.start = input;
          }
          if(input > animation.info.end)
          {
            animation.info.end = input;
          }
        }
      }

      // Read sampler output T/R/S values
      {
        const tinygltf::Accessor& accessor = m_model.accessors[samp.output];

        switch(accessor.type)
        {
          case TINYGLTF_TYPE_VEC2: {
            // copyAccessorData handles all cases: normal, sparse, and sparse-only (bufferView == -1)
            tinygltf::utils::copyAccessorData(m_model, accessor, sampler.outputsVec2);
            break;
          }
          case TINYGLTF_TYPE_VEC3: {
            tinygltf::utils::copyAccessorData(m_model, accessor, sampler.outputsVec3);
            break;
          }
          case TINYGLTF_TYPE_VEC4: {
            tinygltf::utils::copyAccessorData(m_model, accessor, sampler.outputsVec4);
            break;
          }
          case TINYGLTF_TYPE_SCALAR: {
            // This is for `sampler.inputs` vectors of `n` elements
            sampler.outputsFloat.resize(sampler.inputs.size());
            const size_t           elemPerKey = accessor.count / sampler.inputs.size();
            std::vector<float>     storage;
            std::span<const float> val     = tinygltf::utils::getAccessorData(m_model, accessor, &storage);
            const float*           dataPtr = val.data();

            for(size_t i = 0; i < sampler.inputs.size(); i++)
            {
              for(int j = 0; j < elemPerKey; j++)
              {
                sampler.outputsFloat[i].push_back(*dataPtr++);
              }
            }
            break;
          }
          default: {
            LOGW("Unknown animation type: %d\n", accessor.type);
            break;
          }
        }
      }

      animation.samplers.emplace_back(sampler);
    }

    // Channels
    for(auto& source : anim.channels)
    {
      AnimationChannel channel;

      if(source.target_path == "rotation")
      {
        channel.path = AnimationChannel::PathType::eRotation;
      }
      else if(source.target_path == "translation")
      {
        channel.path = AnimationChannel::PathType::eTranslation;
      }
      else if(source.target_path == "scale")
      {
        channel.path = AnimationChannel::PathType::eScale;
      }
      else if(source.target_path == "weights")
      {
        channel.path = AnimationChannel::PathType::eWeights;
      }
      else if(source.target_path == "pointer")
      {
        channel.path = AnimationChannel::PathType::ePointer;

        // Parse KHR_animation_pointer extension
        assert(tinygltf::utils::hasElementName(source.target_extensions, KHR_ANIMATION_POINTER));
        const tinygltf::Value& ext = tinygltf::utils::getElementValue(source.target_extensions, KHR_ANIMATION_POINTER);
        tinygltf::utils::getValue(ext, "pointer", channel.pointerPath);
      }
      channel.samplerIndex = source.sampler;
      channel.node         = source.target_node;

      animation.channels.emplace_back(channel);
    }

    animation.info.reset();
    m_animations.emplace_back(animation);
  }

  // Find all animated primitives (morph)
  m_morphPrimitives.clear();
  for(size_t renderPrimID = 0; renderPrimID < getRenderPrimitives().size(); renderPrimID++)
  {
    const auto&                renderPrimitive = getRenderPrimitive(renderPrimID);
    const tinygltf::Primitive& primitive       = *renderPrimitive.pPrimitive;
    const tinygltf::Mesh&      mesh            = getModel().meshes[renderPrimitive.meshID];

    if(!primitive.targets.empty() && !mesh.weights.empty())
    {
      m_morphPrimitives.push_back(uint32_t(renderPrimID));
    }
  }
  // Skin animated
  m_skinNodes.clear();
  for(size_t renderNodeID = 0; renderNodeID < m_renderNodes.size(); renderNodeID++)
  {
    if(m_renderNodes[renderNodeID].skinID > -1)
    {
      m_skinNodes.push_back(uint32_t(renderNodeID));
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Update the animation (index)
// The value of the animation is updated based on the current time
// - Node transformations are updated
// - Morph target weights are updated
std::unordered_set<int> nvvkgltf::Scene::updateAnimation(uint32_t animationIndex)
{
  Animation&              animation = m_animations[animationIndex];
  float                   time      = animation.info.currentTime;
  std::unordered_set<int> dirtyNodeIds;

  for(auto& channel : animation.channels)
  {
    AnimationSampler& sampler = animation.samplers[channel.samplerIndex];

    // Handle pointer animations (KHR_animation_pointer) - no node required
    if(channel.path == AnimationChannel::PathType::ePointer)
    {
      processAnimationChannel(nullptr, sampler, channel, time);
      continue;
    }

    // Standard animations require a valid node
    if(channel.node < 0 || channel.node >= static_cast<int>(m_model.nodes.size()))
    {
      continue;  // Invalid node
    }

    tinygltf::Node& gltfNode = m_model.nodes[channel.node];
    processAnimationChannel(&gltfNode, sampler, channel, time);
    if(channel.path != AnimationChannel::PathType::eWeights)
    {
      dirtyNodeIds.insert(channel.node);
    }
  }

  // Sync animated properties back to tinygltf::Model (for pointer animations)
  m_animationPointer.syncToModel();
  const auto& animDirtyNodes = m_animationPointer.getDirtyNodes();
  for(int nodeIndex : animDirtyNodes)
  {
    dirtyNodeIds.insert(nodeIndex);
    updateVisibility(nodeIndex);
  }

  return dirtyNodeIds;
}

//--------------------------------------------------------------------------------------------------
// Process the animation channel
// - Interpolates the keyframes
// - Updates the node transformation (if gltfNode is not nullptr)
// - Updates the morph target weights
// - Handles pointer animations (KHR_animation_pointer) when gltfNode is nullptr
bool nvvkgltf::Scene::processAnimationChannel(tinygltf::Node* gltfNode, AnimationSampler& sampler, const AnimationChannel& channel, float time)
{
  bool animated = false;

  for(size_t i = 0; i < sampler.inputs.size() - 1; i++)
  {
    float inputStart = sampler.inputs[i];
    float inputEnd   = sampler.inputs[i + 1];

    if(inputStart <= time && time <= inputEnd)
    {
      float t  = calculateInterpolationFactor(inputStart, inputEnd, time);
      animated = true;

      switch(sampler.interpolation)
      {
        case AnimationSampler::InterpolationType::eLinear:
          handleLinearInterpolation(gltfNode, sampler, channel, t, i);
          break;
        case AnimationSampler::InterpolationType::eStep:
          handleStepInterpolation(gltfNode, sampler, channel, i);
          break;
        case AnimationSampler::InterpolationType::eCubicSpline: {
          float keyDelta = inputEnd - inputStart;
          handleCubicSplineInterpolation(gltfNode, sampler, channel, t, keyDelta, i);
          break;
        }
      }
      break;  // Found the right time segment
    }
  }

  return animated;
}

//--------------------------------------------------------------------------------------------------
// Calculate the interpolation factor: [0..1] between two keyframes
float nvvkgltf::Scene::calculateInterpolationFactor(float inputStart, float inputEnd, float time)
{
  float keyDelta = inputEnd - inputStart;
  return std::clamp((time - inputStart) / keyDelta, 0.0f, 1.0f);
}

//--------------------------------------------------------------------------------------------------
// Interpolates the keyframes linearly
void nvvkgltf::Scene::handleLinearInterpolation(tinygltf::Node*         gltfNode,
                                                AnimationSampler&       sampler,
                                                const AnimationChannel& channel,
                                                float                   t,
                                                size_t                  index)
{
  switch(channel.path)
  {
    case AnimationChannel::PathType::eRotation: {
      const glm::quat q1 = glm::make_quat(glm::value_ptr(sampler.outputsVec4[index]));
      const glm::quat q2 = glm::make_quat(glm::value_ptr(sampler.outputsVec4[index + 1]));
      glm::quat       q  = glm::normalize(glm::slerp(q1, q2, t));
      if(gltfNode)
        gltfNode->rotation = {q.x, q.y, q.z, q.w};
      break;
    }
    case AnimationChannel::PathType::eTranslation: {
      glm::vec3 trans = glm::mix(sampler.outputsVec3[index], sampler.outputsVec3[index + 1], t);
      if(gltfNode)
        gltfNode->translation = {trans.x, trans.y, trans.z};
      break;
    }
    case AnimationChannel::PathType::eScale: {
      glm::vec3 s = glm::mix(sampler.outputsVec3[index], sampler.outputsVec3[index + 1], t);
      if(gltfNode)
        gltfNode->scale = {s.x, s.y, s.z};
      break;
    }
    case AnimationChannel::PathType::eWeights: {
      // Retrieve the mesh from the node
      if(gltfNode && gltfNode->mesh >= 0)
      {
        tinygltf::Mesh& mesh = m_model.meshes[gltfNode->mesh];

        // Make sure the weights vector is resized to match the number of morph targets
        if(mesh.weights.size() != sampler.outputsFloat[index].size())
        {
          mesh.weights.resize(sampler.outputsFloat[index].size());
        }

        // Interpolating between weights for morph targets
        for(size_t j = 0; j < mesh.weights.size(); j++)
        {
          float weight1   = sampler.outputsFloat[index][j];
          float weight2   = sampler.outputsFloat[index + 1][j];
          mesh.weights[j] = glm::mix(weight1, weight2, t);
        }
      }
      break;
    }
    case AnimationChannel::PathType::ePointer: {
      // Pointer animations (KHR_animation_pointer) - gltfNode should be nullptr
      if(!sampler.outputsVec4.empty() && index + 1 < sampler.outputsVec4.size())
      {
        glm::vec4 value = glm::mix(sampler.outputsVec4[index], sampler.outputsVec4[index + 1], t);
        m_animationPointer.applyValue(channel.pointerPath, value);
      }
      else if(!sampler.outputsVec3.empty() && index + 1 < sampler.outputsVec3.size())
      {
        glm::vec3 value = glm::mix(sampler.outputsVec3[index], sampler.outputsVec3[index + 1], t);
        m_animationPointer.applyValue(channel.pointerPath, value);
      }
      else if(!sampler.outputsVec2.empty() && index + 1 < sampler.outputsVec2.size())
      {
        glm::vec2 value = glm::mix(sampler.outputsVec2[index], sampler.outputsVec2[index + 1], t);
        m_animationPointer.applyValue(channel.pointerPath, value);
      }
      else if(!sampler.outputsFloat.empty() && index + 1 < sampler.outputsFloat.size() && !sampler.outputsFloat[index].empty())
      {
        float value = glm::mix(sampler.outputsFloat[index][0], sampler.outputsFloat[index + 1][0], t);
        m_animationPointer.applyValue(channel.pointerPath, value);
      }
      break;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Interpolates the keyframes with a step interpolation
void nvvkgltf::Scene::handleStepInterpolation(tinygltf::Node* gltfNode, AnimationSampler& sampler, const AnimationChannel& channel, size_t index)
{
  switch(channel.path)
  {
    case AnimationChannel::PathType::eRotation: {
      glm::quat q = glm::quat(sampler.outputsVec4[index]);
      if(gltfNode)
        gltfNode->rotation = {q.x, q.y, q.z, q.w};
      break;
    }
    case AnimationChannel::PathType::eTranslation: {
      glm::vec3 trans = glm::vec3(sampler.outputsVec3[index]);
      if(gltfNode)
        gltfNode->translation = {trans.x, trans.y, trans.z};
      break;
    }
    case AnimationChannel::PathType::eScale: {
      glm::vec3 s = glm::vec3(sampler.outputsVec3[index]);
      if(gltfNode)
        gltfNode->scale = {s.x, s.y, s.z};
      break;
    }
    case AnimationChannel::PathType::eWeights:
      // Not implemented for step interpolation
      break;
    case AnimationChannel::PathType::ePointer: {
      // Step interpolation for pointer animations (no blending, use exact value)
      if(!sampler.outputsVec4.empty() && index < sampler.outputsVec4.size())
      {
        m_animationPointer.applyValue(channel.pointerPath, sampler.outputsVec4[index]);
      }
      else if(!sampler.outputsVec3.empty() && index < sampler.outputsVec3.size())
      {
        m_animationPointer.applyValue(channel.pointerPath, sampler.outputsVec3[index]);
      }
      else if(!sampler.outputsVec2.empty() && index < sampler.outputsVec2.size())
      {
        m_animationPointer.applyValue(channel.pointerPath, sampler.outputsVec2[index]);
      }
      else if(!sampler.outputsFloat.empty() && index < sampler.outputsFloat.size() && !sampler.outputsFloat[index].empty())
      {
        m_animationPointer.applyValue(channel.pointerPath, sampler.outputsFloat[index][0]);
      }
      break;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Implements the logic in
// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#interpolation-cubic
// for general vectors. For quaternions, normalize after calling this function.
template <class T>
static T computeCubicInterpolation(const T* values, float t, float keyDelta, size_t index)
{
  const float tSq = t * t;
  const float tCb = tSq * t;
  const float tD  = keyDelta;

  // Compute each of the coefficient terms in the specification
  const float cV1 = -2 * tCb + 3 * tSq;        // -2 t^3 + 3 t^2
  const float cV0 = 1 - cV1;                   //  2 t^3 - 3 t^2 + 1
  const float cA  = tD * (tCb - tSq);          // t_d (t^3 - t^2)
  const float cB  = tD * (tCb - 2 * tSq + t);  // t_d (t^3 - 2 t^2 + t)

  const size_t prevIndex = index * 3;
  const size_t nextIndex = (index + 1) * 3;
  const size_t A         = 0;  // Offset for the in-tangent
  const size_t V         = 1;  // Offset for the value
  const size_t B         = 2;  // Offset for the out-tangent

  const T& v0 = values[prevIndex + V];  // v_k
  const T& a  = values[nextIndex + A];  // a_{k+1}
  const T& b  = values[prevIndex + B];  // b_k
  const T& v1 = values[nextIndex + V];  // v_{k+1}

  T result = v0 * cV0 + a * cA + b * cB + v1 * cV1;
  return result;
}

//--------------------------------------------------------------------------------------------------
// Interpolates the keyframes with a cubic spline interpolation
void nvvkgltf::Scene::handleCubicSplineInterpolation(tinygltf::Node*         gltfNode,
                                                     AnimationSampler&       sampler,
                                                     const AnimationChannel& channel,
                                                     float                   t,
                                                     float                   keyDelta,
                                                     size_t                  index)
{
  // Implements the logic in
  // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#interpolation-cubic
  // for quaternions (first case) and other values (second case).

  // Cubic spline data: each keyframe has 3 values (in-tangent, value, out-tangent)
  // We need to access up to (index+1)*3+1 (the next keyframe's value)
  const size_t maxRequiredIndex = (index + 1) * 3 + 1;

  // Handle pointer animations (KHR_animation_pointer extension)
  if(channel.path == AnimationChannel::PathType::ePointer)
  {
    // Pointer animations can use different vector types
    if(!sampler.outputsVec4.empty() && sampler.outputsVec4.size() > maxRequiredIndex)
    {
      glm::vec4 value = computeCubicInterpolation<glm::vec4>(sampler.outputsVec4.data(), t, keyDelta, index);
      m_animationPointer.applyValue(channel.pointerPath, value);
    }
    else if(!sampler.outputsVec3.empty() && sampler.outputsVec3.size() > maxRequiredIndex)
    {
      glm::vec3 value = computeCubicInterpolation<glm::vec3>(sampler.outputsVec3.data(), t, keyDelta, index);
      m_animationPointer.applyValue(channel.pointerPath, value);
    }
    else if(!sampler.outputsVec2.empty() && sampler.outputsVec2.size() > maxRequiredIndex)
    {
      glm::vec2 value = computeCubicInterpolation<glm::vec2>(sampler.outputsVec2.data(), t, keyDelta, index);
      m_animationPointer.applyValue(channel.pointerPath, value);
    }
    return;
  }

  // Standard node animations require a valid node
  if(!gltfNode)
    return;

  // Handle rotation (quaternion)
  if(channel.path == AnimationChannel::PathType::eRotation)
  {
    if(sampler.outputsVec4.size() > maxRequiredIndex)
    {
      glm::vec4 result     = computeCubicInterpolation<glm::vec4>(sampler.outputsVec4.data(), t, keyDelta, index);
      glm::quat quatResult = glm::make_quat(glm::value_ptr(result));
      quatResult           = glm::normalize(quatResult);
      gltfNode->rotation   = {quatResult.x, quatResult.y, quatResult.z, quatResult.w};
    }
  }
  // Handle translation and scale (vec3)
  else if(sampler.outputsVec3.size() > maxRequiredIndex)
  {
    glm::vec3 result = computeCubicInterpolation<glm::vec3>(sampler.outputsVec3.data(), t, keyDelta, index);

    if(channel.path == AnimationChannel::PathType::eTranslation)
    {
      gltfNode->translation = {result.x, result.y, result.z};
    }
    else if(channel.path == AnimationChannel::PathType::eScale)
    {
      gltfNode->scale = {result.x, result.y, result.z};
    }
  }
}

// Parse the variants of the materials
void nvvkgltf::Scene::parseVariants()
{
  if(m_model.extensions.find(KHR_MATERIALS_VARIANTS_EXTENSION_NAME) != m_model.extensions.end())
  {
    const auto& ext = m_model.extensions.find(KHR_MATERIALS_VARIANTS_EXTENSION_NAME)->second;
    if(ext.Has("variants"))
    {
      auto& variants = ext.Get("variants");
      for(size_t i = 0; i < variants.ArrayLen(); i++)
      {
        std::string name = variants.Get(int(i)).Get("name").Get<std::string>();
        m_variants.emplace_back(name);
      }
    }
  }
}

// Return the material index based on the variant, or the material set on the primitive
int nvvkgltf::Scene::getMaterialVariantIndex(const tinygltf::Primitive& primitive, int currentVariant)
{
  if(primitive.extensions.find(KHR_MATERIALS_VARIANTS_EXTENSION_NAME) != primitive.extensions.end())
  {
    const auto& ext     = primitive.extensions.find(KHR_MATERIALS_VARIANTS_EXTENSION_NAME)->second;
    auto&       mapping = ext.Get("mappings");
    for(auto& map : mapping.Get<tinygltf::Value::Array>())
    {
      auto& variants   = map.Get("variants");
      int   materialID = map.Get("material").Get<int>();
      for(auto& variant : variants.Get<tinygltf::Value::Array>())
      {
        int variantID = variant.Get<int>();
        if(variantID == currentVariant)
          return materialID;
      }
    }
  }

  return std::max(0, primitive.material);
}

//--------------------------------------------------------------------------------------------------
// Get the RenderNode index for a specific primitive within a node
// Returns -1 if not found
int nvvkgltf::Scene::getRenderNodeForPrimitive(int nodeId, int primitiveIndex) const
{
  if(nodeId < 0 || nodeId >= static_cast<int>(m_nodeToRenderNodes.size()))
    return -1;

  const auto& renderNodes = m_nodeToRenderNodes[nodeId];
  if(primitiveIndex < 0 || primitiveIndex >= static_cast<int>(renderNodes.size()))
    return -1;

  return renderNodes[primitiveIndex];
}

//--------------------------------------------------------------------------------------------------
// Get the primitive index within its node for a given RenderNode
// Returns -1 if not found
int nvvkgltf::Scene::getPrimitiveIndexForRenderNode(int renderNodeIndex) const
{
  if(renderNodeIndex < 0 || renderNodeIndex >= static_cast<int>(m_renderNodes.size()))
    return -1;

  const int nodeId = m_renderNodes[renderNodeIndex].refNodeID;
  if(nodeId < 0 || nodeId >= static_cast<int>(m_nodeToRenderNodes.size()))
    return -1;

  const auto& renderNodes = m_nodeToRenderNodes[nodeId];
  for(size_t i = 0; i < renderNodes.size(); ++i)
  {
    if(renderNodes[i] == renderNodeIndex)
      return static_cast<int>(i);
  }
  return -1;
}

//--------------------------------------------------------------------------------------------------
// Collect the render node indices for the given node IDs
//
bool nvvkgltf::Scene::collectRenderNodeIndices(const std::unordered_set<int>& nodeIds,
                                               std::unordered_set<int>&       outRenderNodeIndices,
                                               bool                           includeDescendants,
                                               float                          fullUpdateRatio) const
{
  // Update all render nodes if no node IDs are provided
  if(nodeIds.empty())
  {
    return true;
  }

  // Traverse the node graph and collect the render node indices
  std::function<void(int)> traverseNode;
  traverseNode = [&](int nodeId) {
    for(int rnodeId : m_nodeToRenderNodes.at(nodeId))
    {
      outRenderNodeIndices.insert(rnodeId);
    }

    // If including descendants, traverse child nodes
    if(includeDescendants)
    {
      const tinygltf::Node& node = m_model.nodes[nodeId];
      for(int childId : node.children)
      {
        traverseNode(childId);
      }
    }
  };

  // Add the render node indices for the given node IDs
  for(int nodeId : nodeIds)
  {
    traverseNode(nodeId);
  }

  // Check if the update is full
  if(fullUpdateRatio > float(outRenderNodeIndices.size()) / float(m_renderNodes.size()))
  {
    return false;
  }

  return true;
}
