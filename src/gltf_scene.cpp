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
#include "gltf_compact_model.hpp"
#include "gltf_animation_pointer.hpp"
#include "gltf_scene_merger.hpp"
#include "gltf_compact_model.hpp"
#include "version.hpp"

namespace {

//--------------------------------------------------------------------------------------------------
// Load .gltf or .glb into model using shared TinyGLTF config (no external file limit, image bytes stored raw).
// Returns true on success. On failure, outError and outWarn are set for caller to log.
//--------------------------------------------------------------------------------------------------
// glTF 2.1 file aliases: while loading a referenced glTF, redirect any of its inner URIs that match
// an alias to another entry in the owner's `files` table (used for shared-resource packing or for
// overriding inner data). Aliases are declared on the owner's File entry that points to this glTF.
struct AliasContext
{
  const std::vector<tinygltf::FileAlias>* aliases    = nullptr;  // aliases declared on the file being loaded
  const std::vector<tinygltf::File>*      ownerFiles = nullptr;  // files[] that FileAlias::file indexes into
  std::filesystem::path                   ownerBaseDir;          // resolves the aliased target's URI
  std::filesystem::path                   childBaseDir;          // base dir of the file being loaded
};

bool loadGltfFile(const std::filesystem::path& filename,
                  tinygltf::Model&             model,
                  std::string*                 outError,
                  std::string*                 outWarn,
                  const AliasContext*          aliasCtx = nullptr)
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

  // glTF 2.1: apply this file's aliases by intercepting inner file reads (buffers/images). Any inner
  // URI matching an alias is served from the aliased entry in the owner's files[] table instead.
  if(aliasCtx != nullptr && aliasCtx->aliases != nullptr && !aliasCtx->aliases->empty() && aliasCtx->ownerFiles != nullptr)
  {
    const AliasContext    ctx = *aliasCtx;  // capture by value into the callback
    tinygltf::FsCallbacks fs{};
    fs.FileExists         = &tinygltf::FileExists;
    fs.ExpandFilePath     = &tinygltf::ExpandFilePath;
    fs.WriteWholeFile     = &tinygltf::WriteWholeFile;
    fs.GetFileSizeInBytes = &tinygltf::GetFileSizeInBytes;
    fs.user_data          = nullptr;
    fs.ReadWholeFile = [ctx](std::vector<unsigned char>* out, std::string* err, const std::string& filepath, void*) -> bool {
      std::error_code ec;
      const std::filesystem::path rel = std::filesystem::relative(nvutils::pathFromUtf8(filepath), ctx.childBaseDir, ec);
      const std::string innerUri = ec ? std::string{} : rel.generic_string();
      if(!innerUri.empty())
      {
        for(const tinygltf::FileAlias& a : *ctx.aliases)
        {
          if(a.alias != innerUri)
            continue;
          if(a.file < 0 || a.file >= static_cast<int>(ctx.ownerFiles->size()))
            break;
          const tinygltf::File& target = (*ctx.ownerFiles)[a.file];
          if(target.uri.empty())  // bufferView / data-URI aliased targets are not yet supported
            break;
          std::string targetDecoded;
          tinygltf::URIDecode(target.uri, &targetDecoded, nullptr);
          const std::filesystem::path targetPath = ctx.ownerBaseDir / nvutils::pathFromUtf8(targetDecoded);
          return tinygltf::ReadWholeFile(out, err, nvutils::utf8FromPath(targetPath), nullptr);
        }
      }
      return tinygltf::ReadWholeFile(out, err, filepath, nullptr);
    };
    tcontext.SetFsCallbacks(fs, nullptr);
  }

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
      "EXT_mesh_gpu_instancing",
      "EXT_meshopt_compression",
      "KHR_animation_pointer",
      "KHR_interactivity",
      "KHR_lights_punctual",
      "KHR_materials_anisotropy",
      "KHR_materials_clearcoat",
      "KHR_materials_diffuse_transmission",
      "KHR_materials_dispersion",
      "KHR_materials_displacement",
      "KHR_materials_emissive_strength",
      "KHR_materials_ior",
      "KHR_materials_iridescence",
      "KHR_materials_pbrSpecularGlossiness",
      "KHR_materials_retroreflection",
      "KHR_materials_sheen",
      "KHR_materials_specular",
      "KHR_materials_transmission",
      "KHR_materials_unlit",
      "KHR_materials_variants",
      "KHR_materials_volume_scatter",
      "KHR_materials_volume",
      "KHR_mesh_quantization",
      "KHR_meshopt_compression",
      "KHR_node_hoverability",
      "KHR_node_selectability",
      "KHR_node_visibility",
      "KHR_texture_transform",
      "MSFT_texture_dds",
      "NV_attributes_iray",
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

  // External-asset provenance persists across parseScene() (which clears only derived render
  // data); reset it here, once per file load. The per-node read-only markers live in the model
  // itself and are reset when m_model is cleared above.
  m_referencedAssets.clear();

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

  // Base image search path (asset directory). Set before resolving external assets, which append
  // their own import directories so images referenced by merged assets resolve correctly.
  std::error_code pathEc;
  m_imageSearchPaths = {std::filesystem::absolute(m_filename.parent_path(), pathEc)};
  if(pathEc)
    m_imageSearchPaths = {m_filename.parent_path()};

  // glTF 2.1: merge referenced external assets into the model before parsing the scene graph.
  if(resolveExternalAssets())
  {
    // Merged assets may bring their own KHR/EXT_meshopt_compression buffer views.
    if(!decompressMeshoptExtension())
      return false;
  }

  parseScene();

  m_validSceneParsed = !m_model.nodes.empty();

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
bool nvvkgltf::Scene::save(const std::filesystem::path& filename, bool selfContained)
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

  // glTF 2.1: when the scene references external assets, transform a copy of the model before
  // writing (the live scene is left untouched). Two forms:
  //   - selfContained=false (default): re-externalize -- write the complex-scene form (the
  //     files/externalAssets tables + instance nodes) instead of the flattened runtime model.
  //   - selfContained=true: flatten -- keep the merged content inline and strip every external
  //     reference, producing a portable file that can be shared without the referenced assets.
  // Everything below (image copy, generator tag, extension reconciliation, serialization) then
  // operates on `outModel`.
  tinygltf::Model  transformedModel;
  tinygltf::Model* outModelPtr = &m_model;
  if(!m_referencedAssets.empty())
  {
    transformedModel = m_model;
    if(selfContained)
      nvvkgltf::flattenExternalAssets(transformedModel);
    else
      nvvkgltf::removeExternalAssetContent(transformedModel);
    outModelPtr = &transformedModel;
  }
  tinygltf::Model& outModel = *outModelPtr;

  std::filesystem::path saveFilename = filename;

  // Make sure the extension is correct
  if(!nvutils::extensionMatches(filename, ".gltf") && !nvutils::extensionMatches(filename, ".glb"))
  {
    // replace the extension
    saveFilename = saveFilename.replace_extension(".gltf");
  }

  const bool saveBinary = nvutils::extensionMatches(filename, ".glb");

  // Copy the images to the destination folder using the same search paths as for loading.
  if(!outModel.images.empty() && !saveBinary && !getImageSearchPaths().empty())
  {
    const std::vector<fs::path>&    searchPaths = getImageSearchPaths();
    fs::path                        dstPath     = filename.parent_path();
    int                             numCopied   = 0;
    std::unordered_set<std::string> usedRelativeNames;
    for(size_t i = 0; i < outModel.images.size(); i++)
    {
      auto& image = outModel.images[i];
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
  if(outModel.asset.generator.find(generatorPrefix) == std::string::npos)
  {
    if(!outModel.asset.generator.empty())
      outModel.asset.generator += " + ";
    outModel.asset.generator += std::string(generatorPrefix) + " " APP_VERSION_STRING;
  }

  // Reconcile top-level extensionsUsed / extensionsRequired with what the model actually
  // contains, so the written asset complies with the glTF 2.0 "Specifying Extensions" rules
  // (tinygltf writes these arrays verbatim). This prunes stale entries left by edits/merges
  // and adds any extension that became used since load.
  tinygltf::utils::syncExtensionsUsed(outModel);

  // Save the glTF file
  tinygltf::TinyGLTF tcontext;
  const std::string  saveFilenameUtf8 = nvutils::utf8FromPath(saveFilename);
  bool result = tcontext.WriteGltfSceneToFile(&outModel, saveFilenameUtf8, saveBinary, saveBinary, true, saveBinary);
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
  m_referencedAssets.clear();
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
// True if the mesh is used by any read-only (referenced external-asset) node.
//--------------------------------------------------------------------------------------------------
bool nvvkgltf::Scene::isMeshReadOnly(int meshIndex) const
{
  if(meshIndex < 0)
    return false;
  for(const tinygltf::Node& node : m_model.nodes)
  {
    if(node.mesh == meshIndex && hasExternalAssetMarker(node))
      return true;
  }
  return false;
}

bool nvvkgltf::Scene::isNodeSelectable(int nodeIndex) const
{
  // A node is selectable exactly when it is its own nearest selectable ancestor, i.e. no node on the
  // path to the root opts out. Reuses the single upward pass below (invalid indices return -1 != node).
  return nodeIndex >= 0 && nearestSelectableAncestor(nodeIndex) == nodeIndex;
}

int nvvkgltf::Scene::nearestSelectableAncestor(int nodeIndex) const
{
  const int nodeCount = static_cast<int>(m_model.nodes.size());
  if(nodeIndex < 0 || nodeIndex >= nodeCount)
    return -1;

  // Fast path: if the asset never uses KHR_node_selectability, no node can opt out, so skip the walk.
  if(std::find(m_model.extensionsUsed.begin(), m_model.extensionsUsed.end(), KHR_NODE_SELECTABILITY_EXTENSION_NAME)
     == m_model.extensionsUsed.end())
    return nodeIndex;

  // Single upward pass (O(depth)): find the highest (closest to root) node that opts out of
  // selection. Everything at or below it is non-selectable, so the nearest selectable node is the
  // parent of that highest opt-out node (guaranteed selectable, since nothing above it blocks).
  // If no node on the path opts out, the node itself is selectable.
  int highestBlocker = -1;
  int current        = nodeIndex;
  while(current >= 0 && current < nodeCount)
  {
    if(!tinygltf::utils::getNodeSelectability(m_model.nodes[current]).selectable)
      highestBlocker = current;
    current = (current < static_cast<int>(m_nodeParents.size())) ? m_nodeParents[current] : -1;
  }

  if(highestBlocker < 0)
    return nodeIndex;  // Nothing opts out -> the node itself is selectable.
  return (highestBlocker < static_cast<int>(m_nodeParents.size())) ? m_nodeParents[highestBlocker] : -1;
}

//--------------------------------------------------------------------------------------------------
// glTF 2.1: recursively merge (in place) every external asset referenced by `model`, so it becomes
// self-contained before being merged into the scene. Applies file aliases (inner-URI redirection)
// and guards against reference cycles using `ancestry` (canonical paths currently on the chain).
// The referencing nodes have their externalAsset/mesh/camera links cleared once resolved.
//--------------------------------------------------------------------------------------------------
void nvvkgltf::Scene::flattenReferencedModel(tinygltf::Model&             model,
                                             const std::filesystem::path& modelDir,
                                             std::vector<std::string>&    ancestry,
                                             int                          depth)
{
  constexpr int kMaxExternalAssetDepth = 16;  // backstop beyond the cycle guard
  if(model.externalAssets.empty() || model.files.empty() || model.nodes.empty())
    return;
  if(depth > kMaxExternalAssetDepth)
  {
    LOGW("External asset: nesting deeper than %d levels; stopping recursion\n", kMaxExternalAssetDepth);
    return;
  }

  // Group referencing nodes by file so each unique file is loaded once (deterministic order).
  std::unordered_map<int, std::vector<int>> nodesByFile;
  std::vector<int>                          fileOrder;
  for(int i = 0; i < static_cast<int>(model.nodes.size()); ++i)
  {
    const int eaIdx = model.nodes[i].externalAsset;
    if(eaIdx < 0 || eaIdx >= static_cast<int>(model.externalAssets.size()))
      continue;
    const int fileIdx = model.externalAssets[eaIdx].file;
    if(fileIdx < 0 || fileIdx >= static_cast<int>(model.files.size()))
      continue;
    if(nodesByFile.find(fileIdx) == nodesByFile.end())
      fileOrder.push_back(fileIdx);
    nodesByFile[fileIdx].push_back(i);
  }

  for(int fileIdx : fileOrder)
  {
    const tinygltf::File& file = model.files[fileIdx];
    if(file.mimeType != "model/gltf-binary" && file.mimeType != "model/gltf+json")
      continue;
    if(file.uri.empty())  // embedded (bufferView/data:) external assets are not yet supported
      continue;

    std::string uriDecoded;
    tinygltf::URIDecode(file.uri, &uriDecoded, nullptr);
    const std::filesystem::path childPath = modelDir / nvutils::pathFromUtf8(uriDecoded);

    std::error_code ec;
    const std::filesystem::path canonical = std::filesystem::weakly_canonical(std::filesystem::absolute(childPath, ec), ec);
    const std::string canonKey = nvutils::utf8FromPath(canonical);
    if(std::find(ancestry.begin(), ancestry.end(), canonKey) != ancestry.end())
    {
      LOGW("External asset: reference cycle detected at '%s'; skipping\n", nvutils::utf8FromPath(childPath).c_str());
      continue;
    }

    AliasContext actx;
    actx.aliases      = &file.aliases;
    actx.ownerFiles   = &model.files;
    actx.ownerBaseDir = modelDir;
    actx.childBaseDir = childPath.parent_path();

    tinygltf::Model childModel;
    std::string     error, warn;
    if(!loadGltfFile(childPath, childModel, &error, &warn, file.aliases.empty() ? nullptr : &actx))
    {
      LOGW("External asset: failed to load nested '%s' (%s)\n", nvutils::utf8FromPath(childPath).c_str(),
           error.empty() ? "unknown error" : error.c_str());
      continue;
    }
    if(!validator().validateModelExtensions(childModel, "nested external asset"))
      continue;

    // Recurse first so the child is fully self-contained before we merge it in.
    ancestry.push_back(canonKey);
    flattenReferencedModel(childModel, childPath.parent_path(), ancestry, depth + 1);
    ancestry.pop_back();

    // Images of the merged asset resolve from its own directory.
    std::filesystem::path importDir = std::filesystem::absolute(childPath.parent_path(), ec);
    if(ec)
      importDir = childPath.parent_path();
    m_imageSearchPaths.push_back(importDir);

    const std::vector<int>& refNodes = nodesByFile[fileIdx];

    // Spec precedence: externalAsset overrides mesh/camera. Clear externalAsset too, so `model`
    // becomes self-contained (the content is now merged in as children of the referencing node).
    for(int nodeIdx : refNodes)
    {
      model.nodes[nodeIdx].mesh          = -1;
      model.nodes[nodeIdx].camera        = -1;
      model.nodes[nodeIdx].externalAsset = -1;
    }

    const int             firstNode  = refNodes.front();
    nvvkgltf::MergeResult firstMerge = SceneMerger::mergeIntoNode(model, childModel, firstNode);
    if(!firstMerge.valid())
    {
      LOGW("External asset: failed to merge nested '%s'\n", nvutils::utf8FromPath(childPath).c_str());
      continue;
    }
    for(size_t k = 1; k < refNodes.size(); ++k)
    {
      if(!SceneMerger::instanceSubtree(model, firstMerge, refNodes[k]).valid())
        LOGW("External asset: failed to instance nested '%s'\n", nvutils::utf8FromPath(childPath).c_str());
    }
  }

  // The model is now embedded/self-contained. Clear every node.externalAsset (the merger neither
  // copies the files/externalAssets tables nor remaps this index) so no stale reference — including
  // ones that failed to resolve above — leaks into the parent when this model is merged in.
  for(tinygltf::Node& node : model.nodes)
    node.externalAsset = -1;
}

//--------------------------------------------------------------------------------------------------
// glTF 2.1: resolve node.externalAsset references by loading each referenced file and merging it
// under the referencing node. Each unique file is loaded and merged ONCE; additional nodes that
// reference the same file are instanced from the first merged subtree so they share geometry
// (render primitives / BLAS) instead of duplicating it. Merged-in subtrees are tagged read-only
// and provenance is recorded in m_referencedAssets. Returns true if anything was merged.
//
// Limitation: only external file `uri` references are handled (bufferView / data: URIs are skipped).
//--------------------------------------------------------------------------------------------------
bool nvvkgltf::Scene::resolveExternalAssets()
{
  if(m_model.externalAssets.empty() || m_model.files.empty() || m_model.nodes.empty())
    return false;

  const std::filesystem::path baseDir = m_filename.parent_path();

  // Snapshot the referencing nodes up front (merging appends nodes to m_model.nodes), grouped by
  // the file they reference so each unique file is merged once and instanced for the rest. The
  // insertion-ordered list of files keeps behavior deterministic.
  std::unordered_map<int, std::vector<int>> nodesByFile;  // fileIndex -> referencing node indices
  std::vector<int>                          fileOrder;
  for(int i = 0; i < static_cast<int>(m_model.nodes.size()); ++i)
  {
    const int eaIdx = m_model.nodes[i].externalAsset;
    if(eaIdx < 0)
      continue;
    if(eaIdx >= static_cast<int>(m_model.externalAssets.size()))
    {
      LOGW("External asset: node %d references invalid externalAsset index %d\n", i, eaIdx);
      continue;
    }
    const int fileIdx = m_model.externalAssets[eaIdx].file;
    if(fileIdx < 0 || fileIdx >= static_cast<int>(m_model.files.size()))
    {
      LOGW("External asset %d references invalid file index %d\n", eaIdx, fileIdx);
      continue;
    }
    if(nodesByFile.find(fileIdx) == nodesByFile.end())
      fileOrder.push_back(fileIdx);
    nodesByFile[fileIdx].push_back(i);
  }
  if(fileOrder.empty())
    return false;

  // Cycle guard shared across nested references, seeded with the top-level scene file so a child
  // that references back to the root is detected.
  std::vector<std::string> ancestry;
  {
    std::error_code seedEc;
    ancestry.push_back(
        nvutils::utf8FromPath(std::filesystem::weakly_canonical(std::filesystem::absolute(m_filename, seedEc), seedEc)));
  }

  bool mergedAny = false;
  for(int fileIdx : fileOrder)
  {
    const tinygltf::File& file = m_model.files[fileIdx];

    // Only external glTF assets can be referenced as external assets.
    if(file.mimeType != "model/gltf-binary" && file.mimeType != "model/gltf+json")
    {
      LOGW("External asset file %d has unsupported mimeType '%s' (expected model/gltf+json or model/gltf-binary)\n",
           fileIdx, file.mimeType.c_str());
      continue;
    }
    // First pass: only external file URIs are supported.
    if(file.uri.empty())
    {
      LOGW("External asset file %d has no 'uri'; embedded (bufferView/data:) external assets are not yet supported\n", fileIdx);
      continue;
    }

    // #6: percent-decode the URI before resolving it to a path.
    std::string uriDecoded;
    tinygltf::URIDecode(file.uri, &uriDecoded, nullptr);
    const std::filesystem::path childPath     = baseDir / nvutils::pathFromUtf8(uriDecoded);
    const std::string           childPathUtf8 = nvutils::utf8FromPath(childPath);

    // Cycle guard across the reference chain.
    std::error_code             cycleEc;
    const std::filesystem::path canonical =
        std::filesystem::weakly_canonical(std::filesystem::absolute(childPath, cycleEc), cycleEc);
    const std::string canonKey = nvutils::utf8FromPath(canonical);
    if(std::find(ancestry.begin(), ancestry.end(), canonKey) != ancestry.end())
    {
      LOGW("External asset: reference cycle detected at '%s'; skipping\n", childPathUtf8.c_str());
      continue;
    }

    // #3: apply this file's glTF 2.1 aliases (inner-URI redirection) while loading.
    AliasContext actx;
    actx.aliases      = &file.aliases;
    actx.ownerFiles   = &m_model.files;
    actx.ownerBaseDir = baseDir;
    actx.childBaseDir = childPath.parent_path();

    tinygltf::Model childModel;
    std::string     error, warn;
    if(!loadGltfFile(childPath, childModel, &error, &warn, file.aliases.empty() ? nullptr : &actx))
    {
      LOGW("External asset: failed to load '%s' (%s)\n", childPathUtf8.c_str(), error.empty() ? "unknown error" : error.c_str());
      continue;
    }
    if(!validator().validateModelExtensions(childModel, "external asset"))
    {
      LOGW("External asset '%s' uses unsupported extensions; skipping\n", childPathUtf8.c_str());
      continue;
    }

    // #4: recursively resolve the child's own external assets so it is self-contained before merging.
    ancestry.push_back(canonKey);
    flattenReferencedModel(childModel, childPath.parent_path(), ancestry, 1);
    ancestry.pop_back();

    const std::vector<int>& refNodes = nodesByFile[fileIdx];

    // Spec: when a node has an externalAsset, its own mesh/camera are ignored (externalAsset takes
    // precedence). Clear them so referencing nodes contribute only their transform + children.
    for(int nodeIdx : refNodes)
    {
      m_model.nodes[nodeIdx].mesh   = -1;
      m_model.nodes[nodeIdx].camera = -1;
    }

    // Merge the file once under the first referencing node.
    const int             firstNode  = refNodes.front();
    const int             firstEa    = m_model.nodes[firstNode].externalAsset;
    nvvkgltf::MergeResult firstMerge = SceneMerger::mergeIntoNode(m_model, childModel, firstNode);
    if(!firstMerge.valid())
    {
      LOGW("External asset: failed to merge '%s'\n", childPathUtf8.c_str());
      continue;
    }

    // Make the merged asset's images resolvable from its own directory.
    std::error_code       importEc;
    std::filesystem::path importDir = std::filesystem::absolute(childPath.parent_path(), importEc);
    if(importEc)
      importDir = childPath.parent_path();
    m_imageSearchPaths.push_back(importDir);

    recordReferencedAsset(firstMerge.firstNode, firstMerge.lastNode, firstNode, firstEa, fileIdx, childPathUtf8);
    mergedAny = true;

    // Instance the merged subtree for every additional referencing node (shares geometry/BLAS).
    for(size_t k = 1; k < refNodes.size(); ++k)
    {
      const int             nodeIdx  = refNodes[k];
      const int             eaIdx    = m_model.nodes[nodeIdx].externalAsset;
      nvvkgltf::MergeResult instance = SceneMerger::instanceSubtree(m_model, firstMerge, nodeIdx);
      if(!instance.valid())
      {
        LOGW("External asset: failed to instance '%s' under node %d\n", childPathUtf8.c_str(), nodeIdx);
        continue;
      }
      recordReferencedAsset(instance.firstNode, instance.lastNode, nodeIdx, eaIdx, fileIdx, childPathUtf8);
    }

    LOGI("External asset: merged '%s' (%zu instance(s), %d nodes each)\n", childPathUtf8.c_str(), refNodes.size(),
         firstMerge.lastNode - firstMerge.firstNode);
  }

  return mergedAny;
}

//--------------------------------------------------------------------------------------------------
// Tag [firstNode,lastNode) read-only and record provenance. Returns the m_referencedAssets index.
//--------------------------------------------------------------------------------------------------
int nvvkgltf::Scene::recordReferencedAsset(int firstNode, int lastNode, int instanceNode, int externalAssetIndex, int fileIndex, const std::string& uri)
{
  const int       refIdx = static_cast<int>(m_referencedAssets.size());
  ReferencedAsset ref;
  ref.instanceNodeIndex  = instanceNode;
  ref.externalAssetIndex = externalAssetIndex;
  ref.fileIndex          = fileIndex;
  ref.sourceUri          = uri;
  ref.subtreeNodes.reserve(lastNode > firstNode ? static_cast<size_t>(lastNode - firstNode) : 0);
  for(int n = firstNode; n < lastNode; ++n)
  {
    ref.subtreeNodes.push_back(n);
    // Stamp the provenance marker into the node's extras (preserves any existing extras). This is
    // what isNodeReadOnly() reads, and it survives duplicate/delete renumbering intrinsically.
    setExternalAssetMarker(m_model.nodes[n], refIdx);
  }
  m_referencedAssets.push_back(std::move(ref));
  return refIdx;
}

//--------------------------------------------------------------------------------------------------
// glTF 2.1: add another glTF as a referenced external asset (read-only), instead of embedding it.
// Repeated references to the same file share geometry (placed by duplicating an existing instance).
//--------------------------------------------------------------------------------------------------
int nvvkgltf::Scene::referenceScene(const std::filesystem::path& filename)
{
  nvutils::ScopedTimer st(std::string(__FUNCTION__) + "\n");

  if(m_model.scenes.empty())
    m_model.scenes.emplace_back();

  std::error_code ec;
  const std::filesystem::path absTarget = std::filesystem::weakly_canonical(std::filesystem::absolute(filename, ec), ec);

  // If this file is already referenced and still has a live instance, place a shared copy by
  // duplicating that instance (shares meshes/materials/BLAS) rather than loading + merging again.
  int existingFileIndex = -1;
  for(int f = 0; f < static_cast<int>(m_model.files.size()); ++f)
  {
    const std::string& uri = m_model.files[f].uri;
    if(uri.empty())
      continue;
    std::string uriDecoded;
    tinygltf::URIDecode(uri, &uriDecoded, nullptr);
    std::filesystem::path p = nvutils::pathFromUtf8(uriDecoded);
    if(p.is_relative() && !m_filename.empty())
      p = m_filename.parent_path() / p;
    if(std::filesystem::weakly_canonical(std::filesystem::absolute(p, ec), ec) == absTarget)
    {
      existingFileIndex = f;
      break;
    }
  }
  if(existingFileIndex >= 0)
  {
    for(int n = 0; n < static_cast<int>(m_model.nodes.size()); ++n)
    {
      const int ea = m_model.nodes[n].externalAsset;
      if(ea >= 0 && ea < static_cast<int>(m_model.externalAssets.size()) && m_model.externalAssets[ea].file == existingFileIndex)
      {
        const int dup = editor().duplicateNode(n);  // shares geometry; copies read-only markers + link
        if(dup >= 0)
          LOGI("%sReferenced '%s' as a shared instance (node %d)\n", st.indent().c_str(),
               nvutils::utf8FromPath(filename.filename()).c_str(), dup);
        return dup;
      }
    }
    // File entry exists but no live instance remains: fall through and merge fresh (reusing the file).
  }

  // Load + validate the child file.
  tinygltf::Model childModel;
  std::string     error, warn;
  if(!loadGltfFile(filename, childModel, &error, &warn))
  {
    LOGW("%sReference: failed to load '%s' (%s)\n", st.indent().c_str(), nvutils::utf8FromPath(filename).c_str(),
         error.empty() ? "unknown error" : error.c_str());
    return -1;
  }
  if(!validator().validateModelExtensions(childModel, "referenced asset"))
    return -1;

  // glTF 2.1: recursively resolve the referenced asset's own external assets so it is self-contained
  // before merging it under the instance node (guards against cycles back to this scene / the file).
  {
    std::vector<std::string> ancestry;
    ancestry.push_back(nvutils::utf8FromPath(std::filesystem::weakly_canonical(std::filesystem::absolute(m_filename, ec), ec)));
    ancestry.push_back(nvutils::utf8FromPath(absTarget));
    flattenReferencedModel(childModel, filename.parent_path(), ancestry, 1);
  }

  // File entry: store a path relative to the scene's location when possible (portable), else absolute.
  std::string storedUri;
  {
    std::filesystem::path rel;
    if(!m_filename.empty())
      rel = std::filesystem::relative(absTarget, m_filename.parent_path(), ec);
    storedUri = (!rel.empty() && !ec) ? rel.generic_string() : absTarget.generic_string();
  }

  // Capture pre-mutation sizes so we can roll back all additions on any failure.
  const int origFilesSize            = static_cast<int>(m_model.files.size());
  const int origExternalAssetsSize   = static_cast<int>(m_model.externalAssets.size());
  const int origNodesSize            = static_cast<int>(m_model.nodes.size());
  auto&     sceneNodes               = m_model.scenes[m_currentScene >= 0 ? m_currentScene : 0].nodes;
  const int origSceneNodesSize       = static_cast<int>(sceneNodes.size());
  const int origImageSearchPathsSize = static_cast<int>(m_imageSearchPaths.size());

  auto rollback = [&]() {
    m_model.files.resize(origFilesSize);
    m_model.externalAssets.resize(origExternalAssetsSize);
    m_model.nodes.resize(origNodesSize);
    sceneNodes.resize(origSceneNodesSize);
    m_imageSearchPaths.resize(origImageSearchPathsSize);
  };

  int fileIndex = existingFileIndex;
  if(fileIndex < 0)
  {
    tinygltf::File file;
    file.uri      = storedUri;
    file.mimeType = nvutils::extensionMatches(filename, ".glb") ? "model/gltf-binary" : "model/gltf+json";
    fileIndex     = static_cast<int>(m_model.files.size());
    m_model.files.push_back(std::move(file));
  }

  const int externalAssetIndex = static_cast<int>(m_model.externalAssets.size());
  {
    tinygltf::ExternalAsset asset;
    asset.file = fileIndex;
    asset.name = filename.stem().string();
    m_model.externalAssets.push_back(std::move(asset));
  }

  // Create the instance node (editable transform node carrying the externalAsset link) as a root of
  // the current scene, then merge the child under it.
  const int instanceNode = static_cast<int>(m_model.nodes.size());
  {
    tinygltf::Node node;
    node.name          = filename.stem().string();
    node.translation   = {0.0, 0.0, 0.0};
    node.rotation      = {0.0, 0.0, 0.0, 1.0};
    node.scale         = {1.0, 1.0, 1.0};
    node.mesh          = -1;
    node.camera        = -1;
    node.skin          = -1;
    node.externalAsset = externalAssetIndex;
    m_model.nodes.push_back(std::move(node));
  }
  sceneNodes.push_back(instanceNode);

  nvvkgltf::MergeResult mr = SceneMerger::mergeIntoNode(m_model, childModel, instanceNode);
  if(!mr.valid())
  {
    LOGW("%sReference: failed to merge '%s'\n", st.indent().c_str(), nvutils::utf8FromPath(filename).c_str());
    rollback();
    return -1;
  }

  // Make the referenced asset's images resolvable from its own directory, then decompress meshopt.
  std::filesystem::path importDir = std::filesystem::absolute(filename.parent_path(), ec);
  if(ec)
    importDir = filename.parent_path();
  m_imageSearchPaths.push_back(importDir);
  if(!decompressMeshoptExtension())
  {
    rollback();
    return -1;
  }

  recordReferencedAsset(mr.firstNode, mr.lastNode, instanceNode, externalAssetIndex, fileIndex, nvutils::utf8FromPath(absTarget));

  resolveImageURIs();
  parseScene();
  m_validSceneParsed = !m_model.nodes.empty();

  LOGI("%sReferenced '%s' (%d nodes)\n", st.indent().c_str(), nvutils::utf8FromPath(filename.filename()).c_str(),
       mr.lastNode - mr.firstNode);
  return instanceNode;
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

  // Search for the first camera in the scene and exit traversal upon finding it. Cameras belonging
  // to a referenced (read-only) external asset are part of the asset, not the editable scene's
  // viewport camera, so they are skipped here (an editable default camera is created below instead).
  for(auto& sceneNode : m_model.scenes[m_currentScene].nodes)
  {
    tinygltf::utils::traverseSceneGraph(
        m_model, sceneNode, glm::mat4(1),
        [&](int nodeID, glm::mat4) {
          if(isNodeReadOnly(nodeID))
            return false;  // keep searching for an editable camera
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
  glm::dvec3    center = bbox.center();
  glm::dvec3 eye = center + glm::dvec3(0, 0, bbox.radius() * 2.414);  //2.414 units away from the center of the sphere to fit it within a 45 - degree FOV
  glm::dvec3 up                   = glm::dvec3(0, 1, 0);
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
  glm::dquat q      = glm::quatLookAt(glm::normalize(center - eye), up);
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
// Fold the nodes the GPU transform path moved on-device into the dirty set so the next
// updateNodeWorldMatrices() recomputes only those subtrees (restoring the CPU mirror the CPU sync path
// reads for render-node buffers / TLAS instances). Cheap: cost scales with what actually moved, not the
// whole scene. Returns true if any stale nodes were pending.
//
bool nvvkgltf::Scene::mergeGpuStaleNodesIntoDirty()
{
  if(m_gpuStaleNodes.empty())
    return false;

  m_dirtyFlags.nodes.insert(m_gpuStaleNodes.begin(), m_gpuStaleNodes.end());
  m_gpuStaleNodes.clear();
  return true;
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

    if(tnode.light > -1 && static_cast<size_t>(tnode.light) < m_lights.size())
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

  // Render-node indices in the shaded-nodes cache are about to become stale (node count and
  // ordering may both change). Force a rebuild on the next getShadedNodes() call.
  m_shadedCacheValid = false;

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
  m_renderPrimCenterObj.clear();
  m_variants.clear();
  m_nodeParents.clear();
  m_nodesLocalMatrices.clear();
  m_gpuInstanceLocalMatrices.clear();
  m_numTriangles    = 0;
  m_sceneBounds     = {};
  m_sceneCameraNode = -1;
  m_dirtyFlags.clear();
  m_gpuStaleNodes.clear();  // node IDs are scene-relative; must not survive into a newly parsed model

  // Invalidate the shaded-nodes cache: m_dirtyFlags.clear() wiped the signals the reconcile
  // path would have seen, so we must force the next getShadedNodes() call to do a full rebuild.
  for(int i = 0; i < kPipelineTypeCount; ++i)
    m_shadedNodesCache[i].clear();
  m_materialBucketKey.clear();
  m_shadedCacheValid              = false;
  m_hasTransmissionCache          = false;
  m_shadedCacheSceneGraphRevision = 0;
  // Note: intentionally NOT resetting m_shadedNodesRevision -- it's a monotonic external handle
  // and callers (e.g. Rasterizer::updateSortedBlendNodes) rely on it strictly increasing to
  // detect rebuilds. The next reconcile will ++ it naturally.
}

void nvvkgltf::Scene::destroy()
{
  clearParsedData();
  m_filename.clear();
  m_validSceneParsed = false;
  m_model            = {};
  m_referencedAssets.clear();
}


// Compute the object-space AABB centroid for a primitive from its POSITION accessor min/max.
// Returns (0,0,0) when POSITION is missing or has no min/max arrays -- same behavior the
// rasterizer's transparent-sort previously fell back to when parsing accessors inline.
glm::vec3 nvvkgltf::Scene::computePrimitiveCenterObj(const tinygltf::Primitive& primitive) const
{
  glm::vec3 minVal{0.f}, maxVal{0.f};
  auto      it = primitive.attributes.find("POSITION");
  if(it != primitive.attributes.end() && static_cast<size_t>(it->second) < m_model.accessors.size())
  {
    const tinygltf::Accessor& accessor = m_model.accessors[it->second];
    if(accessor.minValues.size() >= 3)
      minVal = glm::vec3(float(accessor.minValues[0]), float(accessor.minValues[1]), float(accessor.minValues[2]));
    if(accessor.maxValues.size() >= 3)
      maxVal = glm::vec3(float(accessor.maxValues[0]), float(accessor.maxValues[1]), float(accessor.maxValues[2]));
  }
  return 0.5f * (minVal + maxVal);
}

// Build the primitive key map and (re)populate m_renderPrimitives with unique primitives.
// Iterates meshes in deterministic order so indices match the BLAS build order. Also fills the
// parallel m_renderPrimCenterObj array so the rasterizer never has to re-parse POSITION
// accessor min/max for transparent depth sorting.
nvvkgltf::Scene::PrimitiveKeyMap nvvkgltf::Scene::buildPrimitiveKeyMap()
{
  m_renderPrimitives.clear();
  m_renderPrimCenterObj.clear();
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
        m_renderPrimCenterObj.push_back(computePrimitiveCenterObj(primitive));
      }
    }
  }
  return primMap;
}


// Function to extract eye, center, and up vectors from a view matrix
inline void extractCameraVectors(const glm::dmat4& viewMatrix, const glm::dvec3& sceneCenter, glm::dvec3& eye, glm::dvec3& center, glm::dvec3& up)
{
  eye                     = glm::dvec3(viewMatrix[3]);
  glm::dmat3 rotationPart = glm::dmat3(viewMatrix);
  glm::dvec3 forward      = -rotationPart * glm::dvec3(0.0, 0.0, 1.0);

  // Project sceneCenter onto the forward vector
  glm::dvec3 eyeToSceneCenter = sceneCenter - eye;
  double     projectionLength = std::abs(glm::dot(eyeToSceneCenter, forward));
  center                      = eye + projectionLength * forward;

  up = glm::dvec3(0.0, 1.0, 0.0);  // Assume the up vector is always (0, 1, 0)
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
  // Skip cameras that belong to a referenced (read-only) external asset. They are part of the asset,
  // not the editable scene's viewport cameras, and must not participate in the camera round-trip
  // (otherwise they surface as the wrong "camera 0" and get duplicated on save/reload).
  if(isNodeReadOnly(nodeID))
    return false;

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
      minValues = {tinygltf::utils::getAccessorNormalizedValue(accessor, accessor.minValues[0]),
                   tinygltf::utils::getAccessorNormalizedValue(accessor, accessor.minValues[1]),
                   tinygltf::utils::getAccessorNormalizedValue(accessor, accessor.minValues[2])};
    if(!accessor.maxValues.empty())
      maxValues = {tinygltf::utils::getAccessorNormalizedValue(accessor, accessor.maxValues[0]),
                   tinygltf::utils::getAccessorNormalizedValue(accessor, accessor.maxValues[1]),
                   tinygltf::utils::getAccessorNormalizedValue(accessor, accessor.maxValues[2])};
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
// Pack the three fields that decide which raster bucket a material's render nodes belong to
// (alphaMode==OPAQUE, doubleSided, KHR_materials_transmission factor > 0) into one byte. Kept
// minimal on purpose -- any material field NOT in this key can change every frame (e.g. base
// color, roughness, KHR_animation_pointer animating a transmission factor that stays > 0) and
// the shaded-nodes cache stays valid.
//
uint8_t nvvkgltf::Scene::computeMaterialBucketKey(const tinygltf::Material& mat)
{
  uint8_t key = 0;
  if(mat.alphaMode == "OPAQUE")
    key |= 0x1;
  if(mat.doubleSided)
    key |= 0x2;
  if(const auto* ext = tinygltf::utils::findExtension(mat.extensions, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME))
  {
    float transmissionFactor = 0.f;
    tinygltf::utils::getValue(*ext, "transmissionFactor", transmissionFactor);
    if(transmissionFactor > 0.f)
      key |= 0x4;
  }
  return key;
}

//-------------------------------------------------------------------------------------------------
// Lazily reconcile the per-bucket cache against the current dirty state. Called from the entry
// of getShadedNodes() and hasTransmissionMaterials().
//
// Invariants:
//  - The cache is valid when m_shadedCacheValid == true AND no bucket-affecting mutation has
//    happened since the last reconcile.
//  - m_materialBucketKey is parallel to m_model.materials and stores the key the cache was
//    built with. Dirty materials are re-keyed; only when the new key differs do we rebuild.
//  - m_dirtyFlags are NOT consumed here (SceneVk / SceneRTX still need them). This is a
//    read-only diff against the cached key columns.
//
void nvvkgltf::Scene::reconcileShadedNodesCache() const
{
  const size_t materialCount = m_model.materials.size();
  const auto&  renderNodes   = m_renderNodeRegistry.getRenderNodes();

  bool needsRebuild = !m_shadedCacheValid;

  // A structural reset (scene reload, primitive set rebuilt, or a bulk "everything dirty"
  // signal from the owning scene) forces a full rebuild.
  if(m_dirtyFlags.allRenderNodesDirty || m_dirtyFlags.primitivesChanged)
    needsRebuild = true;

  // A scene-graph revision bump captures *bucket-relevant* RenderNode edits: setCurrentVariant
  // flipping materialIDs, SceneEditor::setPrimitiveMaterial, and structural rebuilds. It does
  // NOT bump on animation world-matrix updates (line ~1073 inserts into renderNodesVk without
  // bumping the revision), so this signal lets us stay valid across an animated scene.
  if(m_sceneGraphRevision != m_shadedCacheSceneGraphRevision)
    needsRebuild = true;

  // Material count changed (editor added / removed materials): resize the key column and
  // force rebuild -- per-index keys may have shifted meaning.
  if(m_materialBucketKey.size() != materialCount)
  {
    m_materialBucketKey.assign(materialCount, 0);
    needsRebuild = true;
  }

  // Re-key dirty materials and detect whether any bucket actually flipped. Most material
  // edits (base color slider, KHR_animation_pointer animating a factor without crossing zero)
  // leave the key equal -- cache survives.
  if(m_shadedCacheValid && !needsRebuild)
  {
    for(int matIdx : m_dirtyFlags.materials)
    {
      if(matIdx < 0 || static_cast<size_t>(matIdx) >= materialCount)
        continue;
      const uint8_t newKey = computeMaterialBucketKey(m_model.materials[matIdx]);
      if(newKey != m_materialBucketKey[matIdx])
      {
        m_materialBucketKey[matIdx] = newKey;
        needsRebuild                = true;
      }
    }
  }

  if(!needsRebuild)
    return;

  // Full rebuild: re-key every material, repopulate the four lists, and refresh the
  // hasTransmissionMaterials() cache (equivalent to OR'ing bit2 across the key column).
  for(int i = 0; i < kPipelineTypeCount; ++i)
    m_shadedNodesCache[i].clear();

  m_materialBucketKey.assign(materialCount, 0);
  m_hasTransmissionCache = false;
  for(size_t i = 0; i < materialCount; ++i)
  {
    const uint8_t key      = computeMaterialBucketKey(m_model.materials[i]);
    m_materialBucketKey[i] = key;
    if(key & 0x4)
      m_hasTransmissionCache = true;
  }

  for(uint32_t i = 0; i < renderNodes.size(); ++i)
  {
    const int matID = renderNodes[i].materialID;
    if(matID < 0 || static_cast<size_t>(matID) >= materialCount)
      continue;
    const uint8_t key             = m_materialBucketKey[matID];
    const bool    isOpaque        = (key & 0x1) != 0;
    const bool    isDoubleSided   = (key & 0x2) != 0;
    const bool    hasTransmission = (key & 0x4) != 0;

    // Bucket classification matches the original getShadedNodes() switch exactly.
    if(isOpaque && !isDoubleSided && !hasTransmission)
      m_shadedNodesCache[eRasterSolid].push_back(i);
    if(isOpaque && isDoubleSided)
      m_shadedNodesCache[eRasterSolidDoubleSided].push_back(i);
    if(!isOpaque || hasTransmission)
      m_shadedNodesCache[eRasterBlend].push_back(i);
    m_shadedNodesCache[eRasterAll].push_back(i);
  }

  m_shadedCacheValid              = true;
  m_shadedCacheSceneGraphRevision = m_sceneGraphRevision;
  ++m_shadedNodesRevision;
}

//-------------------------------------------------------------------------------------------------
// Find which nodes are solid or translucent, helps for raster rendering. Backed by
// reconcileShadedNodesCache(); returns a reference valid until the next scene mutation.
//
const std::vector<uint32_t>& nvvkgltf::Scene::getShadedNodes(PipelineType type) const
{
  reconcileShadedNodesCache();
  return m_shadedNodesCache[type];
}

//-------------------------------------------------------------------------------------------------
// True when any material in the scene has a non-zero KHR_materials_transmission factor. Derived
// from the same bucket-key table built by reconcileShadedNodesCache(); O(1) after the first call
// of a frame (reconcile is lazy and no-op when nothing changed).
//
bool nvvkgltf::Scene::hasTransmissionMaterials() const
{
  reconcileShadedNodesCache();
  return m_hasTransmissionCache;
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

  // Preserve any existing extras (e.g. the external-asset read-only marker) and only update the
  // camera:: keys. Replacing the whole object would wipe provenance markers and un-tag merged
  // camera nodes, causing them to leak into saves and duplicate on reload.
  tinygltf::Value::Object extras =
      tnode.extras.IsObject() ? tnode.extras.Get<tinygltf::Value::Object>() : tinygltf::Value::Object{};
  extras["camera::eye"]    = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(camera.eye));
  extras["camera::center"] = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(camera.center));
  extras["camera::up"]     = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(camera.up));
  tnode.extras             = tinygltf::Value(extras);
}
}  // namespace

void nvvkgltf::Scene::setSceneCamera(const nvvkgltf::RenderCamera& camera)
{
  assert(m_sceneCameraNode != -1 && "No camera node found in the scene");
  // Never write into a referenced (read-only) external-asset camera node.
  if(m_sceneCameraNode < 0 || isNodeReadOnly(m_sceneCameraNode))
    return;

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

  // Collect the scene's editable camera nodes in traversal order. Cameras embedded in a referenced
  // (read-only) external asset are skipped: they belong to the asset, are re-merged on load and
  // stripped on save, so we must never overwrite them here.
  std::vector<int> cameraNodeIds;
  for(auto& sceneNode : m_model.scenes[m_currentScene].nodes)
  {
    tinygltf::utils::traverseSceneGraph(m_model, sceneNode, glm::mat4(1), [&](int nodeID, const glm::mat4&) {
      if(m_model.nodes[nodeID].camera >= 0 && !isNodeReadOnly(nodeID))
        cameraNodeIds.push_back(nodeID);
      return false;
    });
  }

  for(size_t i = 0; i < cameras.size(); ++i)
  {
    // Reuse an existing editable camera node, or add a new one when the user has more cameras.
    int nodeIndex = -1;
    if(i < cameraNodeIds.size())
    {
      nodeIndex = cameraNodeIds[i];
    }
    else
    {
      tinygltf::Node& tnode = m_model.nodes.emplace_back();
      nodeIndex             = static_cast<int>(m_model.nodes.size() - 1);
      tnode.name            = fmt::format("Camera-{}", i);
      m_model.scenes[m_currentScene].nodes.push_back(nodeIndex);
    }

    // Write into the node's own camera entry (append one for a brand-new node). Editable camera
    // entries are disjoint from referenced ones, so this leaves the referenced cameras untouched.
    tinygltf::Node& tnode = m_model.nodes[nodeIndex];
    if(tnode.camera < 0)
    {
      m_model.cameras.emplace_back();
      tnode.camera = static_cast<int>(m_model.cameras.size() - 1);
    }
    applyRenderCameraToNode(tnode, m_model.cameras[tnode.camera], cameras[i]);
  }

  // Fewer cameras than before (the user removed some): detach the surplus editable camera nodes so
  // they no longer appear as viewport cameras. Their now-unused camera entries are pruned on save.
  for(size_t i = cameras.size(); i < cameraNodeIds.size(); ++i)
    m_model.nodes[cameraNodeIds[i]].camera = -1;
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
