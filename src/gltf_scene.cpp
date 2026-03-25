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

//
// Core Scene class: loads glTF files (via tinygltf), flattens the node
// hierarchy into render instances, computes bounding boxes, optimizes
// meshes (meshoptimizer), generates tangents, and provides the host-side
// data model that SceneVk uploads to the GPU for rendering.
//

#include <algorithm>
#include <execution>
#include <filesystem>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <glm/gtx/norm.hpp>
#include <fmt/format.h>
#include <meshoptimizer/src/meshoptimizer.h>

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <nvutils/timers.hpp>

#include "gltf_scene.hpp"
#include "gltf_scene_animation.hpp"
#include "gltf_scene_editor.hpp"
#include "gltf_scene_validator.hpp"
#include "gltf_animation_pointer.hpp"
#include "gltf_scene_merger.hpp"
#include "gltf_compact_model.hpp"
#include "version.hpp"

namespace {

//--------------------------------------------------------------------------------------------------
// Load .gltf or .glb into model using shared TinyGLTF config (no external file limit, image bytes stored raw).
// Returns true on success. On failure, outError and outWarn are set for caller to log.
//--------------------------------------------------------------------------------------------------
bool loadGltfFile(const std::filesystem::path& filename, tinygltf::Model& model, std::string* outError, std::string* outWarn)
{
  const std::string  filenameUtf8 = nvutils::utf8FromPath(filename);
  tinygltf::TinyGLTF tcontext;
  std::string        warn;
  std::string        error;
  tcontext.SetMaxExternalFileSize(-1);
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

  const std::string ext    = nvutils::utf8FromPath(filename.extension());
  bool              result = false;
  if(ext == ".gltf")
  {
    result = tcontext.LoadASCIIFromFile(&model, &error, &warn, filenameUtf8);
  }
  else if(ext == ".glb")
  {
    result = tcontext.LoadBinaryFromFile(&model, &error, &warn, filenameUtf8);
  }

  if(outError)
    *outError = std::move(error);
  if(outWarn)
    *outWarn = std::move(warn);
  return result;
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// RenderNodeRegistry
//--------------------------------------------------------------------------------------------------

const std::vector<int> nvvkgltf::RenderNodeRegistry::s_emptyRenderNodes;

int nvvkgltf::RenderNodeRegistry::addRenderNode(const RenderNode& node, int nodeID, int primIndex)
{
  const int renderNodeID = static_cast<int>(m_renderNodes.size());
  m_renderNodes.push_back(node);
  m_renderNodeToNodeAndPrim.emplace_back(nodeID, primIndex);
  // Only store first occurrence of (nodeID, primIndex) for getRenderNodeID (GPU instancing adds multiple per prim)
  m_nodeAndPrimToRenderNode.try_emplace(makeKey(nodeID, primIndex), renderNodeID);
  m_nodeToRenderNodes[nodeID].push_back(renderNodeID);
  return renderNodeID;
}

int nvvkgltf::RenderNodeRegistry::getRenderNodeID(int nodeID, int primIndex) const
{
  auto it = m_nodeAndPrimToRenderNode.find(makeKey(nodeID, primIndex));
  if(it == m_nodeAndPrimToRenderNode.end())
    return -1;
  return it->second;
}

std::optional<std::pair<int, int>> nvvkgltf::RenderNodeRegistry::getNodeAndPrim(int renderNodeID) const
{
  if(renderNodeID < 0 || static_cast<size_t>(renderNodeID) >= m_renderNodeToNodeAndPrim.size())
    return std::nullopt;
  return m_renderNodeToNodeAndPrim[renderNodeID];
}

const std::vector<int>& nvvkgltf::RenderNodeRegistry::getRenderNodesForNode(int nodeID) const
{
  auto it = m_nodeToRenderNodes.find(nodeID);
  if(it == m_nodeToRenderNodes.end())
    return s_emptyRenderNodes;
  return it->second;
}

void nvvkgltf::RenderNodeRegistry::getAllRenderNodesForNodeRecursive(int                                  nodeID,
                                                                     std::function<std::vector<int>(int)> getChildren,
                                                                     std::vector<int>& outRenderNodeIDs) const
{
  const std::vector<int>& direct = getRenderNodesForNode(nodeID);
  for(int rnID : direct)
    outRenderNodeIDs.push_back(rnID);
  for(int childID : getChildren(nodeID))
    getAllRenderNodesForNodeRecursive(childID, getChildren, outRenderNodeIDs);
}

void nvvkgltf::RenderNodeRegistry::clear()
{
  m_renderNodes.clear();
  m_nodeAndPrimToRenderNode.clear();
  m_renderNodeToNodeAndPrim.clear();
  m_nodeToRenderNodes.clear();
}

//--------------------------------------------------------------------------------------------------
// CONSTRUCTION / DESTRUCTION
//--------------------------------------------------------------------------------------------------

nvvkgltf::Scene::Scene()
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
      "KHR_meshopt_compression",
      "EXT_meshopt_compression",
#ifdef USE_DRACO
      "KHR_draco_mesh_compression",
#endif
#ifdef NVP_SUPPORTS_BASISU
      "KHR_texture_basisu",
#endif
  };
}

nvvkgltf::Scene::~Scene() = default;

nvvkgltf::SceneEditor& nvvkgltf::Scene::editor()
{
  if(!m_editor)
    m_editor = std::make_unique<SceneEditor>(*this);
  return *m_editor;
}

nvvkgltf::AnimationSystem& nvvkgltf::Scene::animation()
{
  if(!m_animation)
    m_animation = std::make_unique<AnimationSystem>(*this);
  return *m_animation;
}

const nvvkgltf::AnimationSystem& nvvkgltf::Scene::animation() const
{
  if(!m_animation)
    m_animation = std::make_unique<AnimationSystem>(const_cast<Scene&>(*this));
  return *m_animation;
}

nvvkgltf::SceneValidator& nvvkgltf::Scene::validator()
{
  if(!m_validator)
    m_validator = std::make_unique<SceneValidator>(*this);
  return *m_validator;
}

const nvvkgltf::SceneValidator& nvvkgltf::Scene::validator() const
{
  if(!m_validator)
    m_validator = std::make_unique<SceneValidator>(*this);
  return *m_validator;
}

//--------------------------------------------------------------------------------------------------
// FILE I/O
//--------------------------------------------------------------------------------------------------

bool nvvkgltf::Scene::load(const std::filesystem::path& filename)
{
  nvutils::ScopedTimer st(std::string(__FUNCTION__) + "\n");
  const std::string    filenameUtf8 = nvutils::utf8FromPath(filename);

  m_validSceneParsed = false;

  std::error_code ec;
  m_filename = std::filesystem::absolute(filename, ec);
  if(ec)
    m_filename = filename;
  m_model = {};

  std::string error, warn;
  if(!loadGltfFile(filename, m_model, &error, &warn))
  {
    if(error.empty())
      LOGW("%sUnknown file extension: %s\n", st.indent().c_str(), nvutils::utf8FromPath(filename.extension()).c_str());
    else
    {
      LOGW("%sError loading file: %s\n", st.indent().c_str(), filenameUtf8.c_str());
      LOGW("%s%s\n", st.indent().c_str(), warn.c_str());
      LOGW("%s%s\n", st.indent().c_str(), error.c_str());
    }
    clearParsedData();
    return false;
  }

  if(!validator().validateModelExtensions(m_model, nullptr))
  {
    clearParsedData();
    return false;
  }

  if(!decompressMeshoptExtension())
  {
    return false;
  }

  m_currentScene   = m_model.defaultScene > -1 ? m_model.defaultScene : 0;
  m_currentVariant = 0;
  if(m_animation)
    m_animation->resetPointer();
  parseScene();

  m_validSceneParsed = !m_model.nodes.empty();

  std::error_code pathEc;
  m_imageSearchPaths = {std::filesystem::absolute(m_filename.parent_path(), pathEc)};
  if(pathEc)
    m_imageSearchPaths = {m_filename.parent_path()};

  resolveImageURIs();

  return m_validSceneParsed;
}

//--------------------------------------------------------------------------------------------------
// Decompress KHR/EXT_meshopt_compression buffer views in-place; remove extension when done.
// Returns true if extension was not used or decompression succeeded, false on failure.
//--------------------------------------------------------------------------------------------------
bool nvvkgltf::Scene::decompressMeshoptExtension()
{
  bool hasKHR = std::find(m_model.extensionsUsed.begin(), m_model.extensionsUsed.end(), KHR_MESHOPT_COMPRESSION_EXTENSION_NAME)
                != m_model.extensionsUsed.end();
  bool hasEXT = std::find(m_model.extensionsUsed.begin(), m_model.extensionsUsed.end(), EXT_MESHOPT_COMPRESSION_EXTENSION_NAME)
                != m_model.extensionsUsed.end();
  if(!hasKHR && !hasEXT)
  {
    return true;
  }

  for(tinygltf::Buffer& buffer : m_model.buffers)
  {
    if(buffer.data.empty())
    {
      buffer.data.resize(buffer.byteLength);
      buffer.extensions.erase(KHR_MESHOPT_COMPRESSION_EXTENSION_NAME);
      buffer.extensions.erase(EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
    }
  }

  // First used to tag buffers that can be removed after decompression
  std::vector<int> isFullyCompressedBuffer(m_model.buffers.size(), 1);

  for(auto& bufferView : m_model.bufferViews)
  {
    if(bufferView.buffer < 0)
      continue;

    bool warned = false;

    KHR_meshopt_compression mcomp;
    if(tinygltf::utils::getMeshoptCompression(bufferView, mcomp))
    {
      // Decoding logic derived from `decompressMeshopt` in meshoptimizer/gltf/parsegltf.cpp

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
        case KHR_meshopt_compression::MESHOPT_COMPRESSION_MODE_ATTRIBUTES:
          warn = meshopt_decodeVertexVersion(source, mcomp.byteLength) < 0;
          rc   = meshopt_decodeVertexBuffer(result, mcomp.count, mcomp.byteStride, source, mcomp.byteLength);
          break;

        case KHR_meshopt_compression::MESHOPT_COMPRESSION_MODE_TRIANGLES:
          warn = meshopt_decodeIndexVersion(source, mcomp.byteLength) < 0;
          rc   = meshopt_decodeIndexBuffer(result, mcomp.count, mcomp.byteStride, source, mcomp.byteLength);
          break;

        case KHR_meshopt_compression::MESHOPT_COMPRESSION_MODE_INDICES:
          warn = meshopt_decodeIndexVersion(source, mcomp.byteLength) < 0;
          rc   = meshopt_decodeIndexSequence(result, mcomp.count, mcomp.byteStride, source, mcomp.byteLength);
          break;

        default:
          break;
      }

      if(rc != 0)
      {
        LOGW("meshopt_compression decompression failed\n");
        clearParsedData();
        return false;
      }

      if(warn && !warned)
      {
        LOGW("Warning: meshopt_compression data uses an unsupported or invalid encoding version\n");
        warned = true;
      }

      switch(mcomp.compressionFilter)
      {
        case KHR_meshopt_compression::MESHOPT_COMPRESSION_FILTER_OCTAHEDRAL:
          meshopt_decodeFilterOct(result, mcomp.count, mcomp.byteStride);
          break;

        case KHR_meshopt_compression::MESHOPT_COMPRESSION_FILTER_QUATERNION:
          meshopt_decodeFilterQuat(result, mcomp.count, mcomp.byteStride);
          break;

        case KHR_meshopt_compression::MESHOPT_COMPRESSION_FILTER_EXPONENTIAL:
          meshopt_decodeFilterExp(result, mcomp.count, mcomp.byteStride);
          break;

        default:
          break;
      }

      bufferView.extensions.erase(KHR_MESHOPT_COMPRESSION_EXTENSION_NAME);
      bufferView.extensions.erase(EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
    }

    isFullyCompressedBuffer[bufferView.buffer] = 0;
  }

  // Remove fully compressed buffers; isFullyCompressedBuffer is repurposed as buffer index remap table
  size_t writeIndex = 0;
  for(size_t readIndex = 0; readIndex < m_model.buffers.size(); readIndex++)
  {
    if(isFullyCompressedBuffer[readIndex])
    {
      isFullyCompressedBuffer[readIndex] = -1;
    }
    else
    {
      isFullyCompressedBuffer[readIndex] = static_cast<int>(writeIndex);
      if(readIndex != writeIndex)
      {
        m_model.buffers[writeIndex] = std::move(m_model.buffers[readIndex]);
      }
      writeIndex++;
    }
  }
  m_model.buffers.resize(writeIndex);

  for(auto& bufferView : m_model.bufferViews)
  {
    if(bufferView.buffer >= 0)
    {
      bufferView.buffer = isFullyCompressedBuffer[bufferView.buffer];
    }
  }

  std::erase(m_model.extensionsRequired, KHR_MESHOPT_COMPRESSION_EXTENSION_NAME);
  std::erase(m_model.extensionsUsed, KHR_MESHOPT_COMPRESSION_EXTENSION_NAME);
  std::erase(m_model.extensionsRequired, EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
  std::erase(m_model.extensionsUsed, EXT_MESHOPT_COMPRESSION_EXTENSION_NAME);
  return true;
}

//--------------------------------------------------------------------------------------------------
// Save scene to glTF/GLB file with validation
//
bool nvvkgltf::Scene::save(const std::filesystem::path& filename)
{
  namespace fs = std::filesystem;

  nvutils::ScopedTimer st(std::string(__FUNCTION__) + "\n");

  // VALIDATE BEFORE SAVE
  auto validation = validator().validateBeforeSave();
  validation.print();

  if(!validation.valid)
  {
    LOGW("Cannot save - validation failed\n");
    return false;  // STRICT: refuse to save invalid model
  }

  std::filesystem::path saveFilename = filename;

  // Make sure the extension is correct
  if(!nvutils::extensionMatches(filename, ".gltf") && !nvutils::extensionMatches(filename, ".glb"))
  {
    // replace the extension
    saveFilename = saveFilename.replace_extension(".gltf");
  }

  const bool saveBinary = nvutils::extensionMatches(filename, ".glb");

  // Copy the images to the destination folder using the same search paths as for loading.
  if(!m_model.images.empty() && !saveBinary && !getImageSearchPaths().empty())
  {
    const std::vector<fs::path>&    searchPaths = getImageSearchPaths();
    fs::path                        dstPath     = filename.parent_path();
    int                             numCopied   = 0;
    std::unordered_set<std::string> usedRelativeNames;
    for(size_t i = 0; i < m_model.images.size(); i++)
    {
      auto& image = m_model.images[i];
      if(image.uri.empty())
        continue;
      if(image.uri.size() >= 5 && image.uri.compare(0, 5, "data:") == 0)
        continue;  // data URI: no file to copy
      std::string uri_decoded;
      tinygltf::URIDecode(image.uri, &uri_decoded, nullptr);
      fs::path pathDecoded = nvutils::pathFromUtf8(uri_decoded);
      fs::path srcFile     = nvutils::findFile(pathDecoded, searchPaths, false);
      if(srcFile.empty())
        continue;
      std::string name      = pathDecoded.filename().string();
      std::string base      = pathDecoded.stem().string();
      std::string ext       = pathDecoded.extension().string();
      std::string candidate = "images/" + name;
      int         suffix    = 0;
      while(usedRelativeNames.count(candidate))
        candidate = "images/" + base + "_" + std::to_string(++suffix) + ext;
      usedRelativeNames.insert(candidate);
      fs::path dstRelative = fs::path(candidate);
      fs::path dstFile     = dstPath / dstRelative;
      if(srcFile != dstFile)
      {
        fs::create_directories(dstFile.parent_path());
        try
        {
          if(fs::copy_file(srcFile, dstFile, fs::copy_options::overwrite_existing))
            numCopied++;
        }
        catch(const fs::filesystem_error& e)
        {
          LOGW("%sError copying image: %s\n", st.indent().c_str(), e.what());
          continue;
        }
      }
      image.uri = dstRelative.generic_string();  // forward slashes for glTF
    }
    if(numCopied > 0)
      LOGI("%sImages copied: %d\n", st.indent().c_str(), numCopied);
  }

  // Append generator tag if not already present
  constexpr const char* generatorPrefix = "NVIDIA vk_gltf_renderer";
  if(m_model.asset.generator.find(generatorPrefix) == std::string::npos)
  {
    if(!m_model.asset.generator.empty())
      m_model.asset.generator += " + ";
    m_model.asset.generator += std::string(generatorPrefix) + " " APP_VERSION_STRING;
  }

  // Save the glTF file
  tinygltf::TinyGLTF tcontext;
  const std::string  saveFilenameUtf8 = nvutils::utf8FromPath(saveFilename);
  bool result = tcontext.WriteGltfSceneToFile(&m_model, saveFilenameUtf8, saveBinary, saveBinary, true, saveBinary);
  LOGI("%sSaved: %s\n", st.indent().c_str(), saveFilenameUtf8.c_str());

  // After a successful .gltf save, treat the save location as the new canonical home of the scene.
  // This ensures that subsequent operations (e.g. merging another scene) can resolve images that
  // were copied into the save directory's "images/" subfolder.
  if(result)
  {
    std::error_code pathEc;
    fs::path        saveDir = fs::absolute(saveFilename.parent_path(), pathEc);
    if(pathEc)
      saveDir = saveFilename.parent_path();
    m_filename = saveFilename;
    if(std::find(m_imageSearchPaths.begin(), m_imageSearchPaths.end(), saveDir) == m_imageSearchPaths.end())
      m_imageSearchPaths.push_back(saveDir);
  }

  return result;
}

//--------------------------------------------------------------------------------------------------
// Try alternative image extensions when the original URI doesn't resolve on disk.
// Updates m_model image URIs in-place so all consumers (Vulkan upload, save, display) stay in sync.
//
void nvvkgltf::Scene::resolveImageURIs()
{
  static constexpr std::string_view kFallbackExtensions[] = {".dds", ".ktx2", ".ktx", ".png", ".jpg", ".jpeg"};

  for(auto& image : m_model.images)
  {
    if(image.uri.empty() || image.bufferView >= 0)
      continue;
    if(image.uri.size() >= 5 && image.uri.compare(0, 5, "data:") == 0)
      continue;

    std::string uriDecoded;
    tinygltf::URIDecode(image.uri, &uriDecoded, nullptr);

    if(!nvutils::findFile(nvutils::pathFromUtf8(uriDecoded), m_imageSearchPaths, false).empty())
      continue;

    std::filesystem::path origPath(uriDecoded);
    std::string           origExt = origPath.extension().string();

    for(auto ext : kFallbackExtensions)
    {
      if(ext == origExt)
        continue;
      std::string candidate = std::filesystem::path(uriDecoded).replace_extension(ext).string();
      if(!nvutils::findFile(nvutils::pathFromUtf8(candidate), m_imageSearchPaths, false).empty())
      {
        LOGW("Image \"%s\" not found on disk; using \"%s\" instead.\n", image.uri.c_str(), candidate.c_str());
        image.uri = candidate;
        break;
      }
    }
  }
}


//--------------------------------------------------------------------------------------------------
// Take ownership of a pre-loaded model
//
void nvvkgltf::Scene::takeModel(tinygltf::Model&& model)
{
  m_model = std::move(model);
  m_imageSearchPaths.clear();
  if(m_animation)
    m_animation->resetPointer();
  parseScene();
}

//--------------------------------------------------------------------------------------------------
// Merge another glTF scene into this one. The imported scene is wrapped under a new root node.
// If maxTextureCount is set and the combined textures would exceed it, returns false.
// Returns true on success, false on failure.
//
int nvvkgltf::Scene::mergeScene(const std::filesystem::path& filename, std::optional<uint32_t> maxTextureCount)
{
  nvutils::ScopedTimer st(std::string(__FUNCTION__) + "\n");
  const std::string    filenameUtf8 = nvutils::utf8FromPath(filename);

  // Invalidate current scene until we know the merge succeeded
  m_validSceneParsed = false;

  tinygltf::Model importedModel;
  std::string     error, warn;
  if(!loadGltfFile(filename, importedModel, &error, &warn))
  {
    if(error.empty())
      LOGW("%sUnknown file extension: %s\n", st.indent().c_str(), nvutils::utf8FromPath(filename.extension()).c_str());
    else
    {
      LOGW("%sError loading file for merge: %s\n", st.indent().c_str(), filenameUtf8.c_str());
      LOGW("%s%s\n", st.indent().c_str(), error.c_str());
    }
    return -1;
  }

  if(!validator().validateModelExtensions(importedModel, "imported scene"))
  {
    return -1;
  }

  const size_t animationCountBeforeMerge = m_model.animations.size();
  const size_t importedAnimationCount    = importedModel.animations.size();

  int wrapperNodeIdx = SceneMerger::merge(m_model, importedModel, filename.stem().string(), maxTextureCount);
  if(wrapperNodeIdx < 0)
  {
    return -1;
  }

  std::error_code       pathEc;
  std::filesystem::path importDir = std::filesystem::absolute(filename.parent_path(), pathEc);
  if(pathEc)
    importDir = filename.parent_path();
  m_imageSearchPaths.push_back(importDir);
  resolveImageURIs();

  if(!decompressMeshoptExtension())
  {
    return -1;
  }

  parseScene();
  m_validSceneParsed = !m_model.nodes.empty();

  // First animation index that belongs to the merged file (clips are appended). Used by the renderer
  // to select merged motion; otherwise currentAnimation often stays on a base-scene clip (index 0).
  if(importedAnimationCount > 0 && m_model.animations.size() > animationCountBeforeMerge)
    m_pendingMergePreferredAnimationIndex = static_cast<int>(animationCountBeforeMerge);
  else
    m_pendingMergePreferredAnimationIndex = -1;

  return wrapperNodeIdx;
}

//--------------------------------------------------------------------------------------------------
int nvvkgltf::Scene::takeMergePreferredAnimationIndex()
{
  int v                                 = m_pendingMergePreferredAnimationIndex;
  m_pendingMergePreferredAnimationIndex = -1;
  return v;
}

//--------------------------------------------------------------------------------------------------
// SCENE MANAGEMENT
//--------------------------------------------------------------------------------------------------

void nvvkgltf::Scene::setCurrentScene(int sceneID)
{
  assert(sceneID >= 0 && sceneID < static_cast<int>(m_model.scenes.size()) && "Invalid scene ID");
  m_currentScene = sceneID;
  parseScene();
}

//--------------------------------------------------------------------------------------------------
// SCENE PARSING (PRIVATE)
//--------------------------------------------------------------------------------------------------

void nvvkgltf::Scene::parseScene()
{
  // Ensure there are nodes in the glTF model and the current scene ID is valid
  assert(m_model.nodes.size() > 0 && "No nodes in the glTF file");
  assert(m_currentScene >= 0 && m_currentScene < static_cast<int>(m_model.scenes.size()) && "Invalid scene ID");

  // Snapshot state before rebuild so we can diff afterward and set precise dirty flags.
  // Full state (worldMatrix, materialID, renderPrimID, visible) is captured because
  // clearParsedData() wipes pre-existing dirty flags -- the diff must detect everything.
  struct RenderNodeSnapshot
  {
    glm::mat4 worldMatrix;
    int       materialID;
    int       renderPrimID;
    bool      visible;
  };
  const auto&                     prevRNList = m_renderNodeRegistry.getRenderNodes();
  std::vector<RenderNodeSnapshot> prevRN;
  prevRN.reserve(prevRNList.size());
  for(const auto& rn : prevRNList)
    prevRN.push_back({rn.worldMatrix, rn.materialID, rn.renderPrimID, rn.visible});
  const size_t prevMatCount   = m_model.materials.size();
  const size_t prevPrimCount  = m_renderPrimitives.size();
  const size_t prevLightCount = m_lights.size();

  m_validSceneParsed = false;

  // Clear previous scene data and initialize scene elements
  clearParsedData();
  setSceneElementsDefaultNames();
  // Create tangents from model meshes so primitive keys are stable (no render nodes yet).
  createMissingTangentsForModel();

  // Build the list of unique RenderPrimitives in deterministic order (by mesh index, then primitive index).
  // CRITICAL: There is a direct correlation between BLAS and primitive index. BLAS are built once
  // using this order (m_blasAccel[renderPrimID]). The TLAS references BLAS by object.renderPrimID.
  // If primitive order ever changed without rebuilding BLAS, the TLAS would reference the wrong BLAS.
  // So we must never let primitive registration run only during traversal (visit order would vary).
  PrimitiveKeyMap primMap = buildPrimitiveKeyMap();


  // There must be at least one material in the scene
  if(m_model.materials.empty())
  {
    m_model.materials.emplace_back();
  }

  // Collect all draw objects; RenderNode and RenderPrimitive
  // Also it will be used to compute the scene bounds for the camera
  for(auto& sceneNode : m_model.scenes[m_currentScene].nodes)
  {
    tinygltf::utils::traverseSceneGraph(
        m_model, sceneNode, glm::mat4(1), nullptr,
        [this](int nodeID, const glm::mat4& worldMat) { return handleLightTraversal(nodeID, worldMat); },
        [this, &primMap](int nodeID, const glm::mat4& worldMat) { return handleRenderNode(nodeID, worldMat, primMap); });
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
  animation().parseAnimations();

  // We are updating the scene to the first state, animation, skinning, morph, ..
  updateRenderNodesFull();

  // Marking GPU elements dirty is deferred until the end of parseScene so that we can diff against the rebuilt state and set precise dirty flags.
  // Compare rebuilt state to snapshot and set precise dirty flags.
  // Two-pass for render nodes: count first, then only populate hash sets if below kFullUpdateRatio changed.
  {
    const auto& newRN = m_renderNodeRegistry.getRenderNodes();

    if(prevRN.size() != newRN.size())
    {
      m_dirtyFlags.allRenderNodesDirty = true;
    }
    else if(!newRN.empty())
    {
      const size_t     fullUpdateThreshold = static_cast<size_t>(float(newRN.size()) * kFullUpdateRatio);
      std::vector<int> dirtyIndices;
      dirtyIndices.reserve(std::min(fullUpdateThreshold + 1, newRN.size()));

      for(size_t i = 0; i < newRN.size(); i++)
      {
        if(newRN[i].worldMatrix != prevRN[i].worldMatrix || newRN[i].materialID != prevRN[i].materialID
           || newRN[i].renderPrimID != prevRN[i].renderPrimID || newRN[i].visible != prevRN[i].visible)
        {
          dirtyIndices.push_back(static_cast<int>(i));
          if(dirtyIndices.size() > fullUpdateThreshold)
          {
            m_dirtyFlags.allRenderNodesDirty = true;
            break;
          }
        }
      }
      if(!m_dirtyFlags.allRenderNodesDirty)
      {
        for(int idx : dirtyIndices)
          markRenderNodeDirty(idx);
      }
    }

    for(size_t i = prevMatCount; i < m_model.materials.size(); i++)
      markMaterialDirty(static_cast<int>(i));

    if(m_renderPrimitives.size() != prevPrimCount)
      m_dirtyFlags.primitivesChanged = true;

    if(m_lights.size() != prevLightCount)
    {
      for(size_t i = 0; i < m_lights.size(); i++)
        markLightDirty(static_cast<int>(i));
    }
  }

  m_validSceneParsed = !m_model.nodes.empty();
}


//--------------------------------------------------------------------------------------------------
// Dirty Tracking for GPU Updates
//--------------------------------------------------------------------------------------------------

void nvvkgltf::Scene::markMaterialDirty(int materialIndex)
{
  if(materialIndex >= 0 && materialIndex < static_cast<int>(m_model.materials.size()))
    m_dirtyFlags.materials.insert(materialIndex);
}

void nvvkgltf::Scene::markLightDirty(int lightIndex)
{
  if(lightIndex >= 0 && lightIndex < static_cast<int>(m_model.lights.size()))
    m_dirtyFlags.lights.insert(lightIndex);
}

void nvvkgltf::Scene::markRenderNodeDirty(int renderNodeIndex, bool forVk, bool forRtx)
{
  if(renderNodeIndex >= 0 && renderNodeIndex < static_cast<int>(m_renderNodeRegistry.getRenderNodes().size()))
  {
    if(forVk)
      m_dirtyFlags.renderNodesVk.insert(renderNodeIndex);
    if(forRtx)
      m_dirtyFlags.renderNodesRtx.insert(renderNodeIndex);
  }
}

void nvvkgltf::Scene::markNodeDirty(int nodeIndex)
{
  if(nodeIndex < 0 || nodeIndex >= static_cast<int>(m_model.nodes.size()))
    return;

  m_dirtyFlags.nodes.insert(nodeIndex);

  const tinygltf::Node& node = m_model.nodes[nodeIndex];

  if(node.light >= 0)
    markLightDirty(node.light);
}

// We mark RenderNodes as dirty for RTX if their materials changes features that affect ray tracing, such as alpha mode or double-sidedness.
void nvvkgltf::Scene::markRenderNodeRtxDirtyForMaterials(const std::unordered_set<int>& materialIds)
{
  std::unordered_set<int> rtxNodes = getMaterialRenderNodes(materialIds);
  for(int rn : rtxNodes)
    m_dirtyFlags.renderNodesRtx.insert(rn);
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

//--------------------------------------------------------------------------------------------------
// Update world matrices for dirty nodes and populate render-node dirty flags.
//
// Uses a parallel level-by-level walk that propagates dirty flags through the tree hierarchy
// and only computes matrix multiplications for nodes in dirty subtrees. Falls back to a serial
// filtered-root recursive walk when topological levels aren't built yet.
//
// Both paths insert affected render nodes into renderNodesVk/Rtx (preserving pre-existing
// flags from other sources), so callers never need a separate updateRenderNodeDirtyFromNodes().
//
void nvvkgltf::Scene::updateNodeWorldMatrices()
{
  const tinygltf::Scene& scene = m_model.scenes[m_currentScene];
  assert(scene.nodes.size() > 0 && "No nodes in the glTF file");

  if(m_dirtyFlags.nodes.empty())
    return;

  for(int nodeID : m_dirtyFlags.nodes)
    m_nodesLocalMatrices[nodeID] = tinygltf::utils::getNodeMatrix(m_model.nodes[nodeID]);

  // Use parallel level-by-level path when topological levels are available.
  // Falls back to serial recursive walk before first scene load.
  if(!m_topoLevels.nodeOrder.empty())
    updateWorldMatricesParallel();
  else
    updateWorldMatricesSerial();
}

//--------------------------------------------------------------------------------------------------
// Lightweight update for the GPU transform path: refreshes m_nodesLocalMatrices from TRS for dirty
// nodes and updates light world matrices. Skips the full world-matrix propagation and render-node
// dirty marking, which the GPU compute shader handles instead.
//
void nvvkgltf::Scene::updateLocalMatricesAndLights()
{
  if(m_dirtyFlags.nodes.empty())
    return;

  for(int nodeID : m_dirtyFlags.nodes)
    m_nodesLocalMatrices[nodeID] = tinygltf::utils::getNodeMatrix(m_model.nodes[nodeID]);

  for(auto& light : m_lights)
    light.worldMatrix = computeNodeWorldMatrix(light.nodeID);
}

//--------------------------------------------------------------------------------------------------
// Compute a single node's world matrix on demand by walking the parent chain.
// O(depth) per call; used for gizmo positioning and light updates on the GPU transform path
// where the full propagation is skipped.
//
glm::mat4 nvvkgltf::Scene::computeNodeWorldMatrix(int nodeID) const
{
  if(nodeID < 0 || nodeID >= static_cast<int>(m_nodesLocalMatrices.size()))
    return glm::mat4(1.0f);

  std::vector<int> chain;
  for(int n = nodeID; n >= 0; n = m_nodeParents[n])
    chain.push_back(n);

  glm::mat4 world(1.0f);
  for(auto it = chain.rbegin(); it != chain.rend(); ++it)
    world = world * m_nodesLocalMatrices[*it];
  return world;
}

//--------------------------------------------------------------------------------------------------
// Serial path: filtered-root recursive walk. Best for small dirty sets (leaf moves, editor).
//
void nvvkgltf::Scene::updateWorldMatricesSerial()
{
  const size_t numNodes = m_model.nodes.size();

  // Convert unordered_set to flat bit vector for O(1) parent-chain lookups
  std::vector<bool> isDirty(numNodes, false);
  for(int nodeID : m_dirtyFlags.nodes)
    isDirty[nodeID] = true;

  // Filter to root dirty nodes (skip nodes whose ancestor is also dirty), because updating a parent
  // implicitly updates the whole subtree and we want to avoid redundant updates.
  std::vector<int> filteredDirtyNodes;
  filteredDirtyNodes.reserve(m_dirtyFlags.nodes.size());
  for(int nodeID : m_dirtyFlags.nodes)
  {
    bool hasParentInDirty = false;
    int  currentParent    = m_nodeParents[nodeID];
    while(currentParent >= 0)
    {
      if(isDirty[currentParent])
      {
        hasParentInDirty = true;
        break;
      }
      currentParent = m_nodeParents[currentParent];
    }

    if(!hasParentInDirty)
      filteredDirtyNodes.push_back(nodeID);
  }

  const bool hasGpuInstancing = !m_gpuInstanceLocalMatrices.empty();  // Special case (KHR_instancing)

  // Lambda for recursive world matrix update walk. Captures filteredDirtyNodes by reference and walks the entire subtree of each entry.
  std::function<void(int)> updateMatrix;
  updateMatrix = [&](int nodeID) {
    const tinygltf::Node& node = m_model.nodes[nodeID];
    // Root node has no parent matrix, so use identity. Otherwise, multiply parent's world matrix by local matrix.
    glm::mat4 parentMat = m_nodeParents[nodeID] >= 0 ? m_nodesWorldMatrices[m_nodeParents[nodeID]] : glm::mat4(1.0f);
    m_nodesWorldMatrices[nodeID] = parentMat * m_nodesLocalMatrices[nodeID];

    // Only look up render nodes for mesh-bearing nodes (most skeleton joints have no mesh,
    // so this avoids expensive hash-map lookups for the vast majority of nodes)
    if(node.mesh >= 0)
    {
      const glm::mat4* instMatrices = nullptr;
      size_t           instCount    = 0;

      // Special case for KHR_instancing: look up instance matrices for this node if they exist, and apply them to each render node.
      if(hasGpuInstancing)
      {
        auto instIt = m_gpuInstanceLocalMatrices.find(nodeID);
        if(instIt != m_gpuInstanceLocalMatrices.end() && !instIt->second.empty())
        {
          instMatrices = instIt->second.data();
          instCount    = instIt->second.size();
        }
      }

      // Find all render nodes for this node and update their world matrices.
      // If "instance matrices" exist, multiply them with the node's world matrix; otherwise use the node's world matrix directly.
      size_t idx = 0;
      for(int renderNodeID : m_renderNodeRegistry.getRenderNodesForNode(nodeID))
      {
        m_renderNodeRegistry.getRenderNodes()[renderNodeID].worldMatrix =
            (instMatrices && instCount > 0) ? m_nodesWorldMatrices[nodeID] * instMatrices[idx % instCount] :
                                              m_nodesWorldMatrices[nodeID];
        m_dirtyFlags.renderNodesVk.insert(renderNodeID);
        m_dirtyFlags.renderNodesRtx.insert(renderNodeID);
        idx++;
      }
    }

    if(node.light >= 0)
      m_lights[node.light].worldMatrix = m_nodesWorldMatrices[nodeID];

    for(int child : node.children)
      updateMatrix(child);
  };

  for(int nodeID : filteredDirtyNodes)
    updateMatrix(nodeID);
}

//--------------------------------------------------------------------------------------------------
// Parallel path: level-by-level with dirty-subtree propagation. Best for large dirty sets
// Processes all nodes but only does matrix math for dirty subtrees.
// This is the same algorithm as for GPU in compute shader.
//
// Serial processes : 0 → 1 → 3 → 4 → 2 → 5 → 6(depth - first, one at a time)
//
// Parallel processes :
// Level 0 : [0]            (1 node, serial)
// Level 1 : [ 1, 2 ]       (2 nodes in parallel)
// Level 2 : [ 3, 4, 5, 6 ] (4 nodes in parallel)
//
// topoOrder = [ 0, 1, 2, 3, 4, 5, 6 ]      (sorted by depth)
// levels    = [(0, 1), (1, 2), (3, 4)]     ((offset, count) per level)
//
void nvvkgltf::Scene::updateWorldMatricesParallel()
{
  const size_t numNodes       = m_model.nodes.size();
  const size_t numRenderNodes = m_renderNodeRegistry.getRenderNodes().size();

  // Using uint8_t instead of bool: std::vector<bool> packs 8 bits per byte, so concurrent
  // writes to different indices can race on the same byte. uint8_t gives one byte per element,
  // making concurrent writes to different indices safe without synchronization.
  std::vector<uint8_t> isDirty(numNodes, 0);
  for(int nodeID : m_dirtyFlags.nodes)
    isDirty[nodeID] = 1;

  std::vector<uint8_t> subtreeDirty(numNodes, 0);
  std::vector<uint8_t> rnDirtyBits(numRenderNodes, 0);

  const bool hasGpuInstancing = !m_gpuInstanceLocalMatrices.empty();

  for(const auto& [offset, count] : m_topoLevels.levels)
  {
    nvutils::parallel_batches(static_cast<uint64_t>(count), [&](uint64_t i) {
      int                   nodeID = m_topoLevels.nodeOrder[offset + i];
      int                   parent = m_nodeParents[nodeID];
      const tinygltf::Node& node   = m_model.nodes[nodeID];

      bool dirty           = isDirty[nodeID] || (parent >= 0 && subtreeDirty[parent]);
      subtreeDirty[nodeID] = dirty ? 1 : 0;

      if(!dirty)
        return;

      glm::mat4 parentMat          = parent >= 0 ? m_nodesWorldMatrices[parent] : glm::mat4(1.0f);
      m_nodesWorldMatrices[nodeID] = parentMat * m_nodesLocalMatrices[nodeID];

      if(node.mesh >= 0)
      {
        const glm::mat4* instMatrices = nullptr;
        size_t           instCount    = 0;

        if(hasGpuInstancing)  // Special case for KHR_instancing: look up instance matrices for this node if they exist, and apply them to each render node.
        {
          auto instIt = m_gpuInstanceLocalMatrices.find(nodeID);
          if(instIt != m_gpuInstanceLocalMatrices.end() && !instIt->second.empty())
          {
            instMatrices = instIt->second.data();
            instCount    = instIt->second.size();
          }
        }

        // Find all render nodes for this node and update their world matrices.
        size_t idx = 0;
        for(int renderNodeID : m_renderNodeRegistry.getRenderNodesForNode(nodeID))
        {
          m_renderNodeRegistry.getRenderNodes()[renderNodeID].worldMatrix =
              (instMatrices && instCount > 0) ? m_nodesWorldMatrices[nodeID] * instMatrices[idx % instCount] :
                                                m_nodesWorldMatrices[nodeID];
          rnDirtyBits[renderNodeID] = 1;
          idx++;
        }
      }

      if(node.light >= 0)
        m_lights[node.light].worldMatrix = m_nodesWorldMatrices[nodeID];
    });
  }

  // Convert dirty bits to dirty sets (preserving pre-existing flags).
  // The downstream syncFromScene/syncTopLevelAS decide bulk vs surgical upload
  // based on the dirty ratio
  for(size_t i = 0; i < numRenderNodes; ++i)
  {
    if(rnDirtyBits[i])
    {
      m_dirtyFlags.renderNodesVk.insert(static_cast<int>(i));
      m_dirtyFlags.renderNodesRtx.insert(static_cast<int>(i));
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Build topological levels for parallel world-matrix propagation in the scene graph.
// Uses BFS from scene root nodes: level 0 = roots, level 1 = their children, and so on.
// The resulting topoOrder is a flat ordering of nodes grouped by depth;
// levels records (offset, count) pairs marking each level's range in topoOrder.
//
// Example:
//   topoOrder = [0, 1, 2, 3, 4, 5, 6]         // sorted by increasing depth
//   levels    = [(0, 1), (1, 2), (3, 4)]      // (offset, count) per level
void nvvkgltf::Scene::buildTopologicalLevels()
{
  const tinygltf::Scene& scene    = m_model.scenes[m_currentScene];
  const size_t           numNodes = m_model.nodes.size();

  m_topoLevels.nodeOrder.clear();
  m_topoLevels.nodeOrder.reserve(numNodes);
  m_topoLevels.levels.clear();

  // Perform breadth-first search (BFS) from each scene root to determine node levels.
  // Multiple scene roots are possible; we must visit all to correctly assign topological levels.
  // Level 0: roots; Level 1: their children; Level 2: grandchildren; etc.
  std::vector<int> currentLevel;
  currentLevel.reserve(numNodes);
  for(int rootID : scene.nodes)
    currentLevel.push_back(rootID);

  // Process current level: add to topological levels, and generate next level.
  while(!currentLevel.empty())
  {
    int offset = static_cast<int>(m_topoLevels.nodeOrder.size());
    int count  = static_cast<int>(currentLevel.size());
    m_topoLevels.levels.push_back({offset, count});
    m_topoLevels.nodeOrder.insert(m_topoLevels.nodeOrder.end(), currentLevel.begin(), currentLevel.end());

    // Generate next level: all children of current level nodes.
    std::vector<int> nextLevel;
    for(int nodeID : currentLevel)
    {
      const tinygltf::Node& node = m_model.nodes[nodeID];
      for(int childID : node.children)
        nextLevel.push_back(childID);
    }
    currentLevel = std::move(nextLevel);
  }
}

//--------------------------------------------------------------------------------------------------
// Traverse scene roots computing local/world matrices, visibility, and parent links.
// Calls callback(nodeID, worldMatrix, visible) for every node in the scene graph.
void nvvkgltf::Scene::traverseSceneWithVisibility(const std::function<void(int nodeID, const glm::mat4& worldMatrix, bool visible)>& callback)
{
  const tinygltf::Scene& scene = m_model.scenes[m_currentScene];
  m_nodesLocalMatrices.resize(m_model.nodes.size(), glm::mat4(1.0f));
  m_nodesWorldMatrices.resize(m_model.nodes.size());
  m_nodeParents.resize(m_model.nodes.size());
  m_nodeParents.assign(m_model.nodes.size(), -1);

  std::function<void(int, const glm::mat4&, bool)> traverse;
  traverse = [&](int nodeID, const glm::mat4& parentMat, bool visible) {
    const tinygltf::Node& node   = m_model.nodes[nodeID];
    m_nodesLocalMatrices[nodeID] = tinygltf::utils::getNodeMatrix(node);
    const glm::mat4 worldMat     = parentMat * m_nodesLocalMatrices[nodeID];
    tinygltf::Node& tnode        = m_model.nodes[nodeID];

    if(visible)
    {
      visible = tinygltf::utils::getNodeVisibility(tnode).visible;
    }

    callback(nodeID, worldMat, visible);

    m_nodesWorldMatrices[nodeID] = worldMat;
    for(const auto& child : tnode.children)
    {
      m_nodeParents[child] = nodeID;
      traverse(child, worldMat, visible);
    }
  };

  for(int sceneNode : scene.nodes)
  {
    bool visible = tinygltf::utils::getNodeVisibility(m_model.nodes[sceneNode]).visible;
    traverse(sceneNode, glm::mat4(1), visible);
  }

  buildTopologicalLevels();
  ++m_sceneGraphRevision;  // GPU transform static buffers must match this graph + render-node registry
}

//--------------------------------------------------------------------------------------------------
// Update all the render nodes in the scene and collecting information about
// the node's parent,  and the render node indices for each node.
void nvvkgltf::Scene::updateRenderNodesFull()
{
  std::vector<nvvkgltf::RenderNode>& renderNodes = m_renderNodeRegistry.getRenderNodes();

  traverseSceneWithVisibility([&](int nodeID, const glm::mat4& worldMat, bool visible) {
    tinygltf::Node& tnode = m_model.nodes[nodeID];

    if(tnode.light > -1)
    {
      m_lights[tnode.light].worldMatrix = worldMat;
    }

    if(tnode.mesh > -1)
    {
      const tinygltf::Mesh&   mesh         = m_model.meshes[tnode.mesh];
      const std::vector<int>& rnIDs        = m_renderNodeRegistry.getRenderNodesForNode(nodeID);
      auto                    instIt       = m_gpuInstanceLocalMatrices.find(nodeID);
      size_t                  numInstances = (instIt != m_gpuInstanceLocalMatrices.end()) ? instIt->second.size() : 0;
      size_t                  instIdx      = 0;

      for(int rnID : rnIDs)
      {
        if(rnID >= 0 && static_cast<size_t>(rnID) < renderNodes.size())
        {
          if(numInstances > 0)
            renderNodes[rnID].worldMatrix = worldMat * instIt->second[instIdx % numInstances];
          else
            renderNodes[rnID].worldMatrix = worldMat;

          renderNodes[rnID].visible = visible;
          auto nodeAndPrim          = m_renderNodeRegistry.getNodeAndPrim(rnID);
          if(nodeAndPrim && nodeAndPrim->first == nodeID && nodeAndPrim->second >= 0
             && static_cast<size_t>(nodeAndPrim->second) < mesh.primitives.size())
          {
            renderNodes[rnID].materialID = getMaterialVariantIndex(mesh.primitives[nodeAndPrim->second], m_currentVariant);
          }
          instIdx++;
        }
      }
    }
  });
}

//--------------------------------------------------------------------------------------------------
// Force rebuild of all render nodes (for debugging/validation). Forwards to rebuildRenderNodesAndLights().
//--------------------------------------------------------------------------------------------------
void nvvkgltf::Scene::rebuildRenderNodes()
{
  rebuildRenderNodesAndLights();
}

//--------------------------------------------------------------------------------------------------
// Rebuild only render nodes and lights (e.g. after node deletion). Does not clear or rebuild
// primitives, cameras, animations, variants. Use instead of parseScene() when only the node
// list or hierarchy changed.
//--------------------------------------------------------------------------------------------------
void nvvkgltf::Scene::rebuildRenderNodesAndLights()
{
  m_renderNodeRegistry.clear();
  m_lights.clear();
  m_gpuInstanceLocalMatrices.clear();
  m_numTriangles = 0;

  PrimitiveKeyMap primMap = buildPrimitiveKeyMap();

  traverseSceneWithVisibility([&](int nodeID, const glm::mat4& worldMat, bool visible) {
    tinygltf::Node& tnode = m_model.nodes[nodeID];

    if(tnode.light > -1)
    {
      handleLightTraversal(nodeID, worldMat);
    }

    if(tnode.mesh > -1)
    {
      createRenderNodesForNode(nodeID, worldMat, visible, primMap);
    }
  });
}

//--------------------------------------------------------------------------------------------------
// VARIANT MANAGEMENT
//--------------------------------------------------------------------------------------------------

void nvvkgltf::Scene::setCurrentVariant(int variant)
{
  m_currentVariant = variant;

  bool anyMaterialIdChanged = false;
  for(int nodeID = 0; nodeID < static_cast<int>(m_model.nodes.size()); nodeID++)
  {
    const std::vector<int>& rnIds = m_renderNodeRegistry.getRenderNodesForNode(nodeID);
    if(rnIds.empty())
      continue;
    tinygltf::Node& tnode = m_model.nodes[nodeID];
    if(tnode.mesh > -1)
    {
      tinygltf::Mesh& mesh = m_model.meshes[tnode.mesh];
      for(int rnID : rnIds)
      {
        auto nodeAndPrim = m_renderNodeRegistry.getNodeAndPrim(rnID);
        int  primIdx =
            (nodeAndPrim && nodeAndPrim->second >= 0 && static_cast<size_t>(nodeAndPrim->second) < mesh.primitives.size()) ?
                 nodeAndPrim->second :
                 0;
        int beforeMatID = m_renderNodeRegistry.getRenderNodes()[rnID].materialID;
        int newMatId    = getMaterialVariantIndex(mesh.primitives[primIdx], m_currentVariant);
        if(beforeMatID != newMatId)
        {
          anyMaterialIdChanged = true;
          m_dirtyFlags.renderNodesVk.insert(rnID);
        }
        m_renderNodeRegistry.getRenderNodes()[rnID].materialID = newMatId;
      }
    }
  }
  if(anyMaterialIdChanged)
    ++m_sceneGraphRevision;  // GPU RenderNodeGpuMapping SSBO must pick up new material IDs
}


void nvvkgltf::Scene::clearParsedData()
{
  m_cameras.clear();
  m_lights.clear();
  if(m_animation)
    m_animation->clear();
  m_renderNodeRegistry.clear();
  m_renderPrimitives.clear();
  m_variants.clear();
  m_nodeParents.clear();
  m_nodesLocalMatrices.clear();
  m_gpuInstanceLocalMatrices.clear();
  m_numTriangles    = 0;
  m_sceneBounds     = {};
  m_sceneCameraNode = -1;
  m_dirtyFlags.clear();
}

void nvvkgltf::Scene::destroy()
{
  clearParsedData();
  m_filename.clear();
  m_validSceneParsed = false;
  m_model            = {};
}


// Build the primitive key map and (re)populate m_renderPrimitives with unique primitives.
// Iterates meshes in deterministic order so indices match the BLAS build order.
nvvkgltf::Scene::PrimitiveKeyMap nvvkgltf::Scene::buildPrimitiveKeyMap()
{
  m_renderPrimitives.clear();
  PrimitiveKeyMap primMap;
  for(size_t i = 0; i < m_model.meshes.size(); ++i)
  {
    for(size_t j = 0; j < m_model.meshes[i].primitives.size(); ++j)
    {
      tinygltf::Primitive& primitive = m_model.meshes[i].primitives[j];
      const std::string&   key       = tinygltf::utils::generatePrimitiveKey(primitive);
      auto [it, inserted]            = primMap.try_emplace(key, static_cast<int>(primMap.size()));
      if(inserted)
      {
        nvvkgltf::RenderPrimitive renderPrim;
        renderPrim.pPrimitive  = &primitive;
        renderPrim.vertexCount = int(tinygltf::utils::getVertexCount(m_model, primitive));
        renderPrim.indexCount  = int(tinygltf::utils::getIndexCount(m_model, primitive));
        renderPrim.meshID      = static_cast<int>(i);
        m_renderPrimitives.push_back(renderPrim);
      }
    }
  }
  return primMap;
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


//--------------------------------------------------------------------------------------------------
// CAMERA MANAGEMENT
//--------------------------------------------------------------------------------------------------


//----------------------------------------------------------------------------------
// Retrieve the list of render cameras in the scene.
// This function returns a vector of render cameras present in the scene. If the `force`
// parameter is set to true, it clears and regenerates the list of cameras.
//
// Parameters:
// - force: If true, forces the regeneration of the camera list.
const std::vector<nvvkgltf::RenderCamera>& nvvkgltf::Scene::getRenderCameras(bool rebuild)
{
  if(rebuild)
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
  renderLight.light = node.light;
  if(node.light < 0 || node.light >= m_model.lights.size())
    return false;

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
  renderLight.nodeID      = nodeID;

  m_lights.push_back(renderLight);
  return false;  // Continue traversal
}


// Return the bounding volume of the scene
nvutils::Bbox nvvkgltf::Scene::getSceneBounds() const
{
  if(!m_sceneBounds.isEmpty())
    return m_sceneBounds;

  for(const nvvkgltf::RenderNode& rnode : m_renderNodeRegistry.getRenderNodes())
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

void nvvkgltf::Scene::createRenderNodesForNode(int nodeID, const glm::mat4& worldMatrix, bool visible, const PrimitiveKeyMap& primMap)
{
  const tinygltf::Node& node = m_model.nodes[nodeID];
  if(node.mesh < 0)
    return;
  tinygltf::Mesh& mesh = m_model.meshes[node.mesh];
  for(size_t primIdx = 0; primIdx < mesh.primitives.size(); primIdx++)
  {
    tinygltf::Primitive& primitive    = mesh.primitives[primIdx];
    int                  rprimID      = primMap.at(tinygltf::utils::generatePrimitiveKey(primitive));
    int                  numTriangles = m_renderPrimitives[rprimID].indexCount / 3;

    nvvkgltf::RenderNode renderNode;
    renderNode.worldMatrix  = worldMatrix;
    renderNode.materialID   = getMaterialVariantIndex(primitive, m_currentVariant);
    renderNode.renderPrimID = rprimID;
    renderNode.refNodeID    = nodeID;
    renderNode.skinID       = node.skin;
    renderNode.visible      = visible;

    if(tinygltf::utils::hasElementName(node.extensions, EXT_MESH_GPU_INSTANCING_EXTENSION_NAME))
    {
      const tinygltf::Value& ext = tinygltf::utils::getElementValue(node.extensions, EXT_MESH_GPU_INSTANCING_EXTENSION_NAME);
      const tinygltf::Value& attributes = ext.Get("attributes");
      size_t numInstances = handleGpuInstancing(attributes, renderNode, worldMatrix, nodeID, static_cast<int>(primIdx));
      m_numTriangles += numTriangles * static_cast<int32_t>(numInstances);
    }
    else
    {
      m_renderNodeRegistry.addRenderNode(renderNode, nodeID, static_cast<int>(primIdx));
      m_numTriangles += numTriangles;
    }
  }
}

// Handles the creation of render nodes for a given primitive in the scene.
// Delegates to createRenderNodesForNode with default visible=true (updated later in updateRenderNodesFull).
bool nvvkgltf::Scene::handleRenderNode(int nodeID, glm::mat4 worldMatrix, const PrimitiveKeyMap& primMap)
{
  const tinygltf::Node& node = m_model.nodes[nodeID];
  if(node.mesh < 0)
    return true;
  createRenderNodesForNode(nodeID, worldMatrix, true, primMap);
  return false;  // Continue traversal
}

// Handle GPU instancing : EXT_mesh_gpu_instancing
// Called once per primitive; the per-instance local transforms are shared across primitives
// of the same node and cached in m_gpuInstanceLocalMatrices for use by updateRenderNodesFull
// and updateNodeWorldMatrices.
size_t nvvkgltf::Scene::handleGpuInstancing(const tinygltf::Value& attributes,
                                            nvvkgltf::RenderNode   renderNode,
                                            glm::mat4              worldMatrix,
                                            int                    nodeID,
                                            int                    primIndex)
{
  // Build the per-instance local matrices once per node (shared across all primitives).
  auto [it, inserted] = m_gpuInstanceLocalMatrices.try_emplace(nodeID);
  if(inserted)
  {
    std::vector<glm::vec3> tStorage, sStorage;
    std::vector<glm::quat> rStorage;
    std::span<glm::vec3> translations = tinygltf::utils::getAttributeData3(m_model, attributes, "TRANSLATION", &tStorage);
    std::span<glm::quat> rotations = tinygltf::utils::getAttributeData3(m_model, attributes, "ROTATION", &rStorage);
    std::span<glm::vec3> scales    = tinygltf::utils::getAttributeData3(m_model, attributes, "SCALE", &sStorage);

    size_t n = std::max({translations.size(), rotations.size(), scales.size()});
    it->second.resize(n);
    for(size_t i = 0; i < n; i++)
    {
      glm::vec3 t   = i < translations.size() ? translations[i] : glm::vec3(0.0f);
      glm::quat r   = i < rotations.size() ? rotations[i] : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
      glm::vec3 s   = i < scales.size() ? scales[i] : glm::vec3(1.0f);
      it->second[i] = glm::translate(glm::mat4(1.0f), t) * glm::mat4_cast(r) * glm::scale(glm::mat4(1.0f), s);
    }
  }

  const std::vector<glm::mat4>& localMatrices = it->second;
  for(size_t i = 0; i < localMatrices.size(); i++)
  {
    nvvkgltf::RenderNode instNode = renderNode;
    instNode.worldMatrix          = worldMatrix * localMatrices[i];
    m_renderNodeRegistry.addRenderNode(instNode, nodeID, primIndex);
  }
  return localMatrices.size();
}

//-------------------------------------------------------------------------------------------------
// Create missing tangents by iterating all meshes/primitives in the model. Used before building
// the unique-primitive list (when render nodes are empty) so primitive keys are stable across
// parseScene() runs (e.g. after duplicate node). Prevents BLAS/TLAS index mismatch.
// Primitives that share the same key (same geometry) get the same TANGENT accessor so they
// remain deduplicated in the unique-primitive list.
void nvvkgltf::Scene::createMissingTangentsForModel()
{
  std::unordered_map<std::string, std::vector<tinygltf::Primitive*>> keyToPrimitives;

  for(size_t i = 0; i < m_model.meshes.size(); ++i)
  {
    for(auto& primitive : m_model.meshes[i].primitives)
    {
      const int matId =
          (primitive.material >= 0 && primitive.material < static_cast<int>(m_model.materials.size())) ? primitive.material : 0;
      if(matId >= static_cast<int>(m_model.materials.size()))
        continue;
      if(m_model.materials[matId].normalTexture.index < 0)
        continue;
      if(primitive.attributes.find("TANGENT") != primitive.attributes.end())
        continue;

      std::string key = tinygltf::utils::generatePrimitiveKey(primitive);
      keyToPrimitives[key].push_back(&primitive);
    }
  }

  for(auto& [key, primitives] : keyToPrimitives)
  {
    if(primitives.empty())
      continue;
    tinygltf::Primitive* first = primitives[0];
    tinygltf::utils::createTangentAttribute(m_model, *first);
    tinygltf::utils::simpleCreateTangents(m_model, *first);
    const int tangentAccessorIndex = first->attributes["TANGENT"];
    for(size_t i = 1; i < primitives.size(); ++i)
      primitives[i]->attributes["TANGENT"] = tangentAccessorIndex;
  }
}

//-------------------------------------------------------------------------------------------------
// Find which render nodes use the given material variant IDs
//
std::unordered_set<int> nvvkgltf::Scene::getMaterialRenderNodes(const std::unordered_set<int>& materialVariantNodeIDs) const
{
  std::unordered_set<int> renderNodes;
  const auto&             rnodes = m_renderNodeRegistry.getRenderNodes();
  for(size_t i = 0; i < rnodes.size(); i++)
  {
    if(materialVariantNodeIDs.contains(rnodes[i].materialID))
    {
      renderNodes.insert(int(i));
    }
  }
  return renderNodes;
}

//-------------------------------------------------------------------------------------------------
// Find which nodes are solid or translucent, helps for raster rendering
//
std::vector<uint32_t> nvvkgltf::Scene::getShadedNodes(PipelineType type) const
{
  std::vector<uint32_t> result;

  const auto& rnodes = m_renderNodeRegistry.getRenderNodes();
  for(uint32_t i = 0; i < rnodes.size(); i++)
  {
    const auto& tmat               = m_model.materials[rnodes[i].materialID];
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
  tnode.matrix      = {};  // Clear the matrix to avoid conflicts with translation/rotation/scale

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
int nvvkgltf::Scene::getRenderNodeForPrimitive(int nodeIndex, int primitiveIndex) const
{
  return m_renderNodeRegistry.getRenderNodeID(nodeIndex, primitiveIndex);
}

//--------------------------------------------------------------------------------------------------
// Get the primitive index within its node for a given RenderNode
// Returns -1 if not found
int nvvkgltf::Scene::getPrimitiveIndexForRenderNode(int renderNodeIndex) const
{
  auto nodeAndPrim = m_renderNodeRegistry.getNodeAndPrim(renderNodeIndex);
  if(!nodeAndPrim)
    return -1;
  return nodeAndPrim->second;
}

//--------------------------------------------------------------------------------------------------
// Expand a set of node indices to their associated render node IDs, optionally recursing into
// children. Calls callback(renderNodeID) for each render node found.
//
void nvvkgltf::Scene::expandNodesToRenderNodes(const std::unordered_set<int>&               nodeIndices,
                                               bool                                         includeDescendants,
                                               const std::function<void(int renderNodeID)>& callback) const
{
  std::function<void(int)> visitNode;
  visitNode = [&](int nodeIndex) {
    for(int rnodeId : m_renderNodeRegistry.getRenderNodesForNode(nodeIndex))
    {
      callback(rnodeId);
    }

    if(includeDescendants)
    {
      const tinygltf::Node& node = m_model.nodes[nodeIndex];
      for(int childIdx : node.children)
      {
        visitNode(childIdx);
      }
    }
  };

  for(int nodeIndex : nodeIndices)
  {
    visitNode(nodeIndex);
  }
}

//--------------------------------------------------------------------------------------------------
// Collect the render node indices for the given node indices
//
bool nvvkgltf::Scene::collectRenderNodeIndices(const std::unordered_set<int>& nodeIndices,
                                               std::unordered_set<int>&       outRenderNodeIndices,
                                               bool                           includeDescendants,
                                               float                          fullUpdateRatio) const
{
  if(nodeIndices.empty())
  {
    return true;
  }

  expandNodesToRenderNodes(nodeIndices, includeDescendants, [&](int rnodeId) { outRenderNodeIndices.insert(rnodeId); });

  if(fullUpdateRatio > float(outRenderNodeIndices.size()) / float(m_renderNodeRegistry.getRenderNodes().size()))
  {
    return false;
  }

  return true;
}

//--------------------------------------------------------------------------------------------------
// Populate renderNodesVk/Rtx from m_dirtyFlags.nodes (with optional descendant expansion).
// When dirty ratio is high, SceneVk/SceneRtx update methods do a full upload (heuristic there).
//
void nvvkgltf::Scene::updateRenderNodeDirtyFromNodes(bool includeDescendants)
{
  if(m_dirtyFlags.nodes.empty())
  {
    return;
  }

  m_dirtyFlags.renderNodesVk.clear();
  m_dirtyFlags.renderNodesRtx.clear();

  expandNodesToRenderNodes(m_dirtyFlags.nodes, includeDescendants, [&](int rnodeId) {
    m_dirtyFlags.renderNodesVk.insert(rnodeId);
    m_dirtyFlags.renderNodesRtx.insert(rnodeId);
  });
}

//--------------------------------------------------------------------------------------------------
// VALIDATION
//--------------------------------------------------------------------------------------------------

void nvvkgltf::Scene::ValidationResult::print() const
{
  if(valid)
  {
    LOGI("[OK] Validation passed\n");
  }
  else
  {
    LOGE("[FAIL] Validation failed\n");
  }

  for(const auto& err : errors)
  {
    LOGE("  ERROR: %s\n", err.c_str());
  }

  for(const auto& warn : warnings)
  {
    LOGW("  WARNING: %s\n", warn.c_str());
  }
}
