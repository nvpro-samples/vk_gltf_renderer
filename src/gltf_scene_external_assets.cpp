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
// glTF 2.1 external assets: resolves every node with `externalAsset` by placing the referenced
// file's content under it. See docs/external_assets.md.
//
// Each referenced file is loaded, and its resources (meshes, materials, accessors, ...) appended to
// the scene model, once per load, whatever the number of files and nodes that place it: one
// RenderPrimitive, so one BLAS, per unique geometry across the whole scene. A placed file whose
// content is static is drawn through an instance set on the placing node, without copying its
// nodes; other content is copied as read-only nodes pointing at the shared resources.
//
// Reading, parsing and decoding a file touch no scene state: every file a load places is read on
// the thread pool first (ExternalAssetPlacer::preload). Appending and placing then run on the
// loading thread, in order, so the result does not depend on the thread count.
//

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parallel_work.hpp>
#include <nvutils/timers.hpp>

#include "gltf_scene.hpp"
#include "gltf_scene_loading.hpp"
#include "gltf_scene_merger.hpp"
#include "gltf_scene_validator.hpp"
#include "load_progress.hpp"
#include "tinygltf_utils.hpp"

namespace nvvkgltf {

namespace {

// Files nested deeper than this are not placed (a backstop beyond the cycle check).
constexpr int kMaxPlacementDepth = 16;
// Nested placements multiply; stop creating nodes past this many in one load (and warn).
constexpr size_t kMaxPlacedNodes = size_t(1) << 24;

std::string canonicalKey(const std::filesystem::path& path)
{
  std::error_code ec;
  // Normalized first: a long unnormalized `../` chain can exceed MAX_PATH on Windows.
  return nvutils::utf8FromPath(std::filesystem::weakly_canonical(std::filesystem::absolute(path, ec).lexically_normal(), ec));
}

// The file a buffer of a file in `dir` reads; empty for GLB and data: buffers, which belong to their file.
std::string bufferFileKey(const tinygltf::Buffer& buffer, const std::filesystem::path& dir)
{
  if(buffer.uri.empty() || buffer.uri.rfind("data:", 0) == 0)
    return {};
  return canonicalKey(dir / nvutils::pathFromUtf8(tinygltf::utils::decodePathFromUri(buffer.uri)));
}

bool isPlaceable(const tinygltf::File& file)
{
  return (file.mimeType == "model/gltf-binary" || file.mimeType == "model/gltf+json") && !file.uri.empty();
}

// What a `files` entry points at.
struct FileRef
{
  std::filesystem::path path;       // normalized, so nested `../` never accumulates (past MAX_PATH on Windows)
  std::string           canonical;  // file identity, for the cycle check
  std::string           key;        // canonical + alias context: what a file is loaded once per
};

// Resolve `file`, an entry of `ownerJson`, a file in `ownerDir`. Touches the file system.
FileRef resolveFile(const tinygltf::Model& ownerJson, const std::filesystem::path& ownerDir, const tinygltf::File& file)
{
  FileRef ref;
  ref.path      = (ownerDir / nvutils::pathFromUtf8(tinygltf::utils::decodePathFromUri(file.uri))).lexically_normal();
  ref.canonical = canonicalKey(ref.path);
  // glTF 2.1 aliases belong to the referencing file: the same file reached with other aliases may
  // load other inner files, so it is another entry.
  ref.key = ref.canonical;
  for(const tinygltf::FileAlias& alias : file.aliases)
  {
    const bool valid = alias.file >= 0 && alias.file < static_cast<int>(ownerJson.files.size());
    ref.key +=
        "|" + alias.alias + "="
        + (valid ? canonicalKey(ownerDir
                                / nvutils::pathFromUtf8(tinygltf::utils::decodePathFromUri(ownerJson.files[alias.file].uri))) :
                   std::string("?"));
  }
  return ref;
}

// A referenced file to load, with what its aliases need from the referencing file.
struct LoadRequest
{
  std::string                      key;
  std::filesystem::path            path;
  std::vector<tinygltf::FileAlias> aliases;
  std::vector<tinygltf::File>      ownerFiles;  // the referencing file's files[], when there are aliases
  std::filesystem::path            ownerDir;
};

// A referenced file, read and decoded: all of its loading that needs no scene state.
struct LoadedFile
{
  bool                                 loaded = false;  // false: `error` says why
  std::string                          error;
  bool                                 decoded = false;  // meshopt decompression succeeded
  tinygltf::Model                      model;
  std::vector<std::string>             bufferKeys;  // bufferFileKey() of each buffer
  std::vector<int>                     roots;       // default-scene roots: what is placed
  std::vector<std::pair<int, FileRef>> placed;      // files it places (files index), in placement order
};

// Read, parse and decode a referenced file, and resolve the files it places. Thread-safe.
LoadedFile loadFile(const LoadRequest& request)
{
  LoadedFile                  out;
  const std::filesystem::path dir = request.path.parent_path();
  detail::AliasContext        aliasContext;
  aliasContext.aliases      = &request.aliases;
  aliasContext.ownerFiles   = &request.ownerFiles;
  aliasContext.ownerBaseDir = request.ownerDir;
  aliasContext.childBaseDir = dir;
  std::string warn;
  out.loaded = detail::loadGltfFile(request.path, out.model, &out.error, &warn, request.aliases.empty() ? nullptr : &aliasContext);
  if(!out.loaded)
    return out;
  out.decoded = detail::decompressMeshopt(out.model);

  const tinygltf::Model& model = out.model;
  for(const tinygltf::Buffer& buffer : model.buffers)
    out.bufferKeys.push_back(bufferFileKey(buffer, dir));

  const int nodeCount = static_cast<int>(model.nodes.size());
  if(!model.scenes.empty())
  {
    int scene = model.defaultScene >= 0 ? model.defaultScene : 0;
    if(scene >= static_cast<int>(model.scenes.size()))
      scene = 0;
    for(int root : model.scenes[scene].nodes)
      if(root >= 0 && root < nodeCount)
        out.roots.push_back(root);
  }

  // In the order ExternalAssetPlacer::copyTree asks for them: a node's children, then its own file.
  std::vector<char>        visited(nodeCount, 0);
  std::vector<char>        listed(model.files.size(), 0);
  std::function<void(int)> visit = [&](int n) {
    if(visited[n])
      return;
    visited[n] = 1;
    for(int child : model.nodes[n].children)
      if(child >= 0 && child < nodeCount)
        visit(child);
    const int externalAsset = model.nodes[n].externalAsset;
    if(externalAsset < 0 || externalAsset >= static_cast<int>(model.externalAssets.size()))
      return;
    const int fileIndex = model.externalAssets[externalAsset].file;
    if(fileIndex < 0 || fileIndex >= static_cast<int>(model.files.size()) || listed[fileIndex] || !isPlaceable(model.files[fileIndex]))
      return;
    listed[fileIndex] = 1;
    out.placed.emplace_back(fileIndex, resolveFile(model, dir, model.files[fileIndex]));
  };
  for(int root : out.roots)
    visit(root);
  return out;
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// Places referenced files into Scene::m_model. One instance per load (or per referenceScene call);
// its cache is what shares a file between all the nodes and files that place it.
//--------------------------------------------------------------------------------------------------
class ExternalAssetPlacer
{
public:
  explicit ExternalAssetPlacer(Scene& scene)
      : m_scene(scene)
      , m_model(scene.m_model)
  {
    m_sceneFile.isScene   = true;
    m_sceneFile.path      = scene.m_filename;
    m_sceneFile.dir       = scene.m_filename.parent_path();
    m_sceneFile.canonical = canonicalKey(scene.m_filename);
    m_sceneFile.loaded    = true;
  }

  // Place what node `nodeIndex` of the scene's own file references, under it: tag what was created
  // read-only and record its provenance. Returns true if the referenced file could be loaded.
  bool placeSceneNode(int nodeIndex)
  {
    const int    externalAsset = m_model.nodes[nodeIndex].externalAsset;
    const size_t firstNode     = m_model.nodes.size();
    int          fileIndex     = -1;

    m_ancestry.assign(1, m_sceneFile.canonical);
    const PlacedFile* placed = place(m_sceneFile, externalAsset, nodeIndex, 1, &fileIndex);
    if(placed == nullptr)
      return false;

    // Core glTF 2.1: a node with externalAsset doesn't draw its own mesh or camera.
    m_model.nodes[nodeIndex].mesh   = -1;
    m_model.nodes[nodeIndex].camera = -1;
    m_scene.recordReferencedAsset(static_cast<int>(firstNode), static_cast<int>(m_model.nodes.size()), nodeIndex,
                                  externalAsset, fileIndex, nvutils::utf8FromPath(placed->path));
    return true;
  }

  //------------------------------------------------------------------------------------------------
  // Load every file that the scene's own nodes `nodes` place, directly or nested, before placing
  // any: level by level on the thread pool, since a file's references are known once it is read.
  // Then make room in the scene model for all of their resources at once. Appending thousands of
  // files one after the other otherwise keeps growing the arrays, and every growth moves the
  // earlier elements again (each holding std::maps, which MSVC allocates for even when moving).
  //------------------------------------------------------------------------------------------------
  void preload(const std::vector<int>& nodes)
  {
    const nvutils::PerformanceTimer timer;
    std::unordered_set<std::string> requested;
    std::vector<LoadRequest>        level;
    for(int node : nodes)
    {
      const int externalAsset = m_model.nodes[node].externalAsset;
      if(externalAsset < 0 || externalAsset >= static_cast<int>(m_model.externalAssets.size()))
        continue;
      const int fileIndex = m_model.externalAssets[externalAsset].file;
      if(fileIndex < 0 || fileIndex >= static_cast<int>(m_model.files.size()) || !isPlaceable(m_model.files[fileIndex]))
        continue;
      const FileRef& ref = fileRef(m_sceneFile, fileIndex);
      if(ref.canonical != m_sceneFile.canonical && requested.insert(ref.key).second)
        level.push_back(loadRequest(m_model, m_sceneFile.dir, m_model.files[fileIndex], ref));
    }

    // Files nested deeper than placement goes are not loaded (cycles stop at `requested`).
    LoadProgress* progress = m_scene.loadProgress();
    for(int depth = 1; depth <= kMaxPlacementDepth && !level.empty(); depth++)
    {
      if(progress)
        progress->stage(depth == 1 ? "Loading external files" : "Loading nested external files",
                        static_cast<uint32_t>(level.size()));
      std::vector<LoadedFile> loaded(level.size());
      nvutils::parallel_batches<1>(level.size(), [&](uint64_t i) {
        loaded[i] = loadFile(level[i]);
        if(progress)
          progress->advance(1, nvutils::utf8FromPath(level[i].path.filename()));
      });
      std::vector<LoadRequest> next;
      for(size_t i = 0; i < level.size(); i++)
      {
        const std::filesystem::path dir = level[i].path.parent_path();
        for(const auto& [fileIndex, ref] : loaded[i].placed)
          if(requested.insert(ref.key).second)
            next.push_back(loadRequest(loaded[i].model, dir, loaded[i].model.files[fileIndex], ref));
        m_preloaded.emplace(std::move(level[i].key), std::move(loaded[i]));
      }
      level = std::move(next);
    }

    // At most what the files hold: shared images and buffers are not appended. Nodes are copied
    // only from files that place others (static content is drawn through instance sets), and a
    // node array growing one copy at a time moves millions of nodes, each holding std::maps.
    IndexRemapping total;
    size_t         copiedNodes = 0;
    for(const auto& [key, file] : m_preloaded)
    {
      const tinygltf::Model& model = file.model;
      if(!model.externalAssets.empty())
        copiedNodes += model.nodes.size();
      total.buffers += static_cast<int>(model.buffers.size());
      total.bufferViews += static_cast<int>(model.bufferViews.size());
      total.accessors += static_cast<int>(model.accessors.size());
      total.images += static_cast<int>(model.images.size());
      total.samplers += static_cast<int>(model.samplers.size());
      total.textures += static_cast<int>(model.textures.size());
      total.materials += static_cast<int>(model.materials.size());
      total.meshes += static_cast<int>(model.meshes.size());
      total.cameras += static_cast<int>(model.cameras.size());
      total.animations += static_cast<int>(model.animations.size());
    }
    m_model.buffers.reserve(m_model.buffers.size() + total.buffers);
    m_model.bufferViews.reserve(m_model.bufferViews.size() + total.bufferViews);
    m_model.accessors.reserve(m_model.accessors.size() + total.accessors);
    m_model.images.reserve(m_model.images.size() + total.images);
    m_model.samplers.reserve(m_model.samplers.size() + total.samplers);
    m_model.textures.reserve(m_model.textures.size() + total.textures);
    m_model.materials.reserve(m_model.materials.size() + total.materials);
    m_model.meshes.reserve(m_model.meshes.size() + total.meshes);
    m_model.cameras.reserve(m_model.cameras.size() + total.cameras);
    m_model.animations.reserve(m_model.animations.size() + total.animations);
    m_model.nodes.reserve(m_model.nodes.size() + copiedNodes);
    m_loadSeconds += timer.getSeconds();
  }

  // The placed files' JSON, only needed while placing; handed over to be freed elsewhere.
  auto takeFiles()
  {
    m_staticTrees.clear();  // keyed by the files
    m_resolvedPaths.clear();
    std::vector<std::unique_ptr<PlacedFile>> files;
    files.reserve(m_files.size());
    for(auto& [key, file] : m_files)
      files.push_back(std::move(file));
    m_files.clear();
    return files;
  }

  size_t loadedFiles() const { return m_loadedCanonical.size(); }  // unique files loaded, nested ones included

  void logSummary() const
  {
    const double total = m_timer.getSeconds();
    LOGI("External assets: %zu file(s) loaded once each (%.2f s), %zu image(s) and %zu buffer(s) shared, %zu placement(s), %zu drawn as instance sets, %zu node(s) and %zu animation channel(s) created (%.2f s)\n",
         m_loadedCanonical.size(), m_loadSeconds, m_sharedImages, m_sharedBuffers, m_placements,
         m_instanceSetPlacements, m_placedNodes, m_channelCopies, total - m_loadSeconds);
  }

private:
  // A referenced file, loaded once per (canonical path, alias context). Its resources live in the
  // scene model at `offsets`; its JSON stays here (payloads dropped) to copy nodes from.
  struct PlacedFile
  {
    bool                  isScene = false;  // the scene's own file: its JSON is m_model itself
    bool                  loaded  = false;
    std::filesystem::path path;
    std::filesystem::path dir;        // what its own references resolve from
    std::string           canonical;  // file identity, for the cycle check
    tinygltf::Model       model;
    IndexRemapping        offsets;               // zero for the scene's own file
    std::vector<int>      roots;                 // default-scene roots: what is placed
    std::vector<char>     animatedNode;          // targeted by an animation channel
    bool                  needsNodeMap = false;  // has skins or animations, which point at nodes
  };

  const tinygltf::Model& json(const PlacedFile& file) const { return file.isScene ? m_model : file.model; }

  void warnOnce(const std::string& key, const std::string& message)
  {
    if(m_warned.insert(key).second)
      LOGW("External asset: %s\n", message.c_str());
  }

  //------------------------------------------------------------------------------------------------
  // Place externalAssets[externalAsset] of `owner` under scene node `target`. Returns the placed
  // file, or null if it could not be loaded (the placing node then keeps its own mesh).
  //------------------------------------------------------------------------------------------------
  const PlacedFile* place(const PlacedFile& owner, int externalAsset, int target, int depth, int* outFileIndex)
  {
    const tinygltf::Model& ownerJson = json(owner);
    const std::string      ownerName = nvutils::utf8FromPath(owner.path.filename());
    if(externalAsset < 0 || externalAsset >= static_cast<int>(ownerJson.externalAssets.size()))
    {
      warnOnce(fmt::format("ea:{}:{}", owner.canonical, externalAsset),
               fmt::format("'{}' references invalid externalAsset index {}", ownerName, externalAsset));
      return nullptr;
    }
    const int fileIndex = ownerJson.externalAssets[externalAsset].file;
    if(fileIndex < 0 || fileIndex >= static_cast<int>(ownerJson.files.size()))
    {
      warnOnce(fmt::format("file:{}:{}", owner.canonical, externalAsset),
               fmt::format("'{}': external asset {} references invalid file index {}", ownerName, externalAsset, fileIndex));
      return nullptr;
    }
    if(outFileIndex != nullptr)
      *outFileIndex = fileIndex;

    const tinygltf::File& file    = ownerJson.files[fileIndex];
    const std::string     fileKey = fmt::format("{}#{}", owner.canonical, fileIndex);
    if(file.mimeType != "model/gltf-binary" && file.mimeType != "model/gltf+json")
    {
      warnOnce(fileKey, fmt::format("'{}': file {} has unsupported mimeType '{}' (expected model/gltf+json or model/gltf-binary)",
                                    ownerName, fileIndex, file.mimeType));
      return nullptr;
    }
    if(file.uri.empty())
    {
      warnOnce(fileKey, fmt::format("'{}': file {} has no 'uri'; embedded (bufferView/data:) external assets are not yet supported",
                                    ownerName, fileIndex));
      return nullptr;
    }

    const FileRef& ref = fileRef(owner, fileIndex);
    // A file already on its own placement path is a cycle. The same file in two branches is not.
    if(onPlacementPath(ref.canonical))
    {
      warnOnce("cycle:" + fileKey, fmt::format("reference cycle detected at '{}'; skipping", nvutils::utf8FromPath(ref.path)));
      return nullptr;
    }
    if(depth > kMaxPlacementDepth)
    {
      warnOnce("depth:" + fileKey, fmt::format("nesting deeper than {} levels at '{}'; not placed", kMaxPlacementDepth,
                                               nvutils::utf8FromPath(ref.path)));
      return nullptr;
    }

    const PlacedFile* placed = acquire(owner, file, ref);
    if(placed == nullptr)
      return nullptr;
    const PlacedFile& x = *placed;
    m_ancestry.push_back(ref.canonical);

    // Static content (nothing skinned, animated, lit, placing a file or carrying a
    // node extension) is drawn through an instance set: one entry per mesh node, no node copied.
    // Anything else is copied as nodes, which own their skins and animation channels.
    if(const StaticTree* tree = staticTree(x))
      addInstanceSet(x, *tree, target);
    else
    {
      std::vector<int> nodeMap;
      if(x.needsNodeMap)
        nodeMap.assign(x.model.nodes.size(), -1);
      for(int root : x.roots)
        copyTree(x, root, target, depth, x.needsNodeMap ? &nodeMap : nullptr);
      if(x.needsNodeMap)
        bindSkinsAndAnimations(x, nodeMap);
    }

    m_ancestry.pop_back();
    m_placements++;
    return placed;
  }

  //------------------------------------------------------------------------------------------------
  // The file `file` of `owner`, resolved to `ref`: loaded (preloaded, normally), validated and its
  // resources appended on first use.
  //------------------------------------------------------------------------------------------------
  const PlacedFile* acquire(const PlacedFile& owner, const tinygltf::File& file, const FileRef& ref)
  {
    const std::pair<PlacedFiles::iterator, bool> found = m_files.try_emplace(ref.key);
    const PlacedFiles::iterator                  it    = found.first;
    if(!found.second)
      return it->second->loaded ? it->second.get() : nullptr;  // warned on first use
    const nvutils::PerformanceTimer timer;
    struct AddTime  // counts every return below
    {
      const nvutils::PerformanceTimer& timer;
      double&                          seconds;
      ~AddTime() { seconds += timer.getSeconds(); }
    } addTime{timer, m_loadSeconds};
    it->second    = std::make_unique<PlacedFile>();
    PlacedFile& x = *it->second;
    x.path        = ref.path;
    x.dir         = ref.path.parent_path();
    x.canonical   = ref.canonical;

    const std::string pathUtf8 = nvutils::utf8FromPath(x.path);
    LoadedFile        loaded;
    if(const auto preloaded = m_preloaded.find(ref.key); preloaded != m_preloaded.end())
    {
      loaded = std::move(preloaded->second);
      m_preloaded.erase(preloaded);
    }
    else  // not preloaded
      loaded = loadFile(loadRequest(json(owner), owner.dir, file, ref));
    if(!loaded.loaded)
    {
      LOGW("External asset: failed to load '%s' (%s)\n", pathUtf8.c_str(),
           loaded.error.empty() ? "unknown error" : loaded.error.c_str());
      return nullptr;
    }
    x.model = std::move(loaded.model);
    if(!m_scene.validator().validateModelExtensions(x.model, "external asset"))
    {
      LOGW("External asset '%s' uses unsupported extensions; skipping\n", pathUtf8.c_str());
      return nullptr;
    }
    if(!loaded.decoded)
    {
      LOGW("External asset '%s': meshopt decompression failed; skipping\n", pathUtf8.c_str());
      return nullptr;
    }
    if(x.model.nodes.empty())
    {
      LOGW("External asset '%s' has no nodes; nothing placed\n", pathUtf8.c_str());
      return nullptr;
    }

    // Images of the file resolve from its own folder.
    std::error_code ec;
    m_scene.m_imageSearchPaths.push_back(std::filesystem::absolute(x.dir, ec));

    detail::rebaseImageUris(x.model, x.dir, m_sceneFile.dir);
    const std::vector<int> sharedImages  = shareImages(x.model);
    const std::vector<int> sharedBuffers = shareBuffers(x, loaded.bufferKeys);
    const size_t           firstBuffer   = m_model.buffers.size();
    x.offsets                            = SceneMerger::appendResources(m_model, x.model, sharedImages, sharedBuffers);
    indexImages();  // the ones just appended
    indexBuffers(loaded.bufferKeys, sharedBuffers, firstBuffer);

    const int nodeCount = static_cast<int>(x.model.nodes.size());
    x.roots             = std::move(loaded.roots);
    x.animatedNode.assign(nodeCount, 0);
    bool pointerChannels = false;
    for(const tinygltf::Animation& animation : x.model.animations)
      for(const tinygltf::AnimationChannel& channel : animation.channels)
      {
        if(channel.target_node >= 0 && channel.target_node < nodeCount)
          x.animatedNode[channel.target_node] = 1;
        pointerChannels = pointerChannels || channel.target_node < 0;
      }
    if(pointerChannels)
      LOGW("External asset '%s': KHR_animation_pointer channels of referenced files are not supported; ignored\n",
           pathUtf8.c_str());
    x.needsNodeMap = !x.model.skins.empty() || !x.model.animations.empty();

    // The payloads were copied into the scene model; only the JSON is needed from here on.
    for(tinygltf::Buffer& buffer : x.model.buffers)
      std::vector<unsigned char>().swap(buffer.data);
    for(tinygltf::Image& image : x.model.images)
      std::vector<unsigned char>().swap(image.image);

    x.loaded = true;
    m_loadedCanonical.insert(x.canonical);  // an alias context of a loaded file is not another file

    // The files it places, resolved while loading it.
    for(auto& [fileIndex, childRef] : loaded.placed)
      m_resolvedPaths.try_emplace({&x, fileIndex}, std::move(childRef));
    return &x;
  }

  bool onPlacementPath(const std::string& canonical) const
  {
    return std::find(m_ancestry.begin(), m_ancestry.end(), canonical) != m_ancestry.end();
  }

  // files[fileIndex] of `owner`, resolved once (a referenced file's while loading it).
  const FileRef& fileRef(const PlacedFile& owner, int fileIndex)
  {
    const std::pair<ResolvedPaths::iterator, bool> resolved = m_resolvedPaths.try_emplace({&owner, fileIndex});
    if(resolved.second)
      resolved.first->second = resolveFile(json(owner), owner.dir, json(owner).files[fileIndex]);
    return resolved.first->second;
  }

  // Loading `file`, an entry of `ownerJson`, a file in `ownerDir`, resolved to `ref`.
  static LoadRequest loadRequest(const tinygltf::Model&       ownerJson,
                                 const std::filesystem::path& ownerDir,
                                 const tinygltf::File&        file,
                                 const FileRef&               ref)
  {
    LoadRequest request;
    request.key  = ref.key;
    request.path = ref.path;
    if(!file.aliases.empty())
    {
      request.aliases    = file.aliases;
      request.ownerFiles = ownerJson.files;
      request.ownerDir   = ownerDir;
    }
    return request;
  }

  // The file an image entry of the scene model reads, as the GPU upload will find it through the
  // image search paths; empty for data: and bufferView images, and for files that can't be found.
  std::string imageFileKey(const tinygltf::Image& image) const
  {
    if(image.uri.empty() || image.bufferView >= 0 || image.uri.rfind("data:", 0) == 0)
      return {};
    const std::filesystem::path found =
        nvutils::findFile(nvutils::pathFromUtf8(tinygltf::utils::decodePathFromUri(image.uri)), m_scene.m_imageSearchPaths, false);
    return found.empty() ? std::string() : canonicalKey(found);
  }

  // Record the image files of the scene model's images not yet seen; the first entry of a file wins.
  void indexImages()
  {
    for(; m_indexedImages < m_model.images.size(); m_indexedImages++)
    {
      std::string key = imageFileKey(m_model.images[m_indexedImages]);
      if(!key.empty())
        m_imageByFile.try_emplace(std::move(key), static_cast<int>(m_indexedImages));
    }
  }

  // For each image of `model` (URIs already rebased to the scene's folder): the scene image that
  // reads the same file, or -1. Files that share a texture file then upload it once.
  std::vector<int> shareImages(const tinygltf::Model& model)
  {
    indexImages();
    std::vector<int> shared(model.images.size(), -1);
    for(size_t i = 0; i < model.images.size(); i++)
    {
      const std::string key = imageFileKey(model.images[i]);
      if(key.empty())
        continue;
      if(const auto it = m_imageByFile.find(key); it != m_imageByFile.end())
      {
        shared[i] = it->second;
        m_sharedImages++;
      }
    }
    return shared;
  }

  // For each buffer of `x` (`keys`: their bufferFileKey()): the scene buffer loaded from the same
  // file with the same bytes, or -1.
  // Files that read one .bin (a library of parts, each file declaring its own accessors into it) then
  // hold it once, and their accessors over the same bytes resolve to the same buffer range, which is
  // what lets Scene::primitiveKey see that they are the same geometry. The bytes are compared, not
  // assumed: decoding (meshopt, Draco) may have rewritten a file's copy.
  std::vector<int> shareBuffers(const PlacedFile& x, const std::vector<std::string>& keys)
  {
    if(!m_indexedSceneBuffers)
    {
      m_indexedSceneBuffers = true;
      for(size_t i = 0; i < m_model.buffers.size(); i++)
        if(std::string key = bufferFileKey(m_model.buffers[i], m_sceneFile.dir); !key.empty())
          m_bufferByFile.try_emplace(std::move(key), static_cast<int>(i));
    }
    std::vector<int> shared(x.model.buffers.size(), -1);
    for(size_t i = 0; i < x.model.buffers.size(); i++)
    {
      const tinygltf::Buffer& buffer = x.model.buffers[i];
      const auto              it     = m_bufferByFile.find(keys[i]);
      if(it != m_bufferByFile.end() && !buffer.data.empty() && m_model.buffers[it->second].data == buffer.data)
      {
        shared[i] = it->second;
        m_sharedBuffers++;
      }
    }
    return shared;
  }

  // Record the files (`keys`) of the buffers of a file just appended, from scene buffer `firstBuffer` on.
  void indexBuffers(const std::vector<std::string>& keys, const std::vector<int>& sharedBuffers, size_t firstBuffer)
  {
    int next = static_cast<int>(firstBuffer);
    for(size_t i = 0; i < keys.size(); i++)
    {
      if(sharedBuffers[i] >= 0)
        continue;
      if(!keys[i].empty())
        m_bufferByFile.try_emplace(keys[i], next);
      next++;
    }
  }

  //------------------------------------------------------------------------------------------------
  // Copy node `nodeIndex` of `x` and its descendants under scene node `parent`. A placing node of
  // `x` places its own file inside.
  //------------------------------------------------------------------------------------------------
  void copyTree(const PlacedFile& x, int nodeIndex, int parent, int depth, std::vector<int>* nodeMap)
  {
    if(m_placedNodes >= kMaxPlacedNodes)
    {
      if(!m_truncated)
        LOGW("External asset: more than %zu placed nodes; stopped placing (the scene is incomplete)\n", kMaxPlacedNodes);
      m_truncated = true;
      return;
    }

    const tinygltf::Node& src = x.model.nodes[nodeIndex];
    tinygltf::Node        node;
    node.name        = src.name;
    node.translation = src.translation;
    node.rotation    = src.rotation;
    node.scale       = src.scale;
    node.matrix      = src.matrix;
    node.weights     = src.weights;
    node.light       = src.light;
    // A placed file's cameras are not copied: a camera is the scene's to place, and a prop that
    // carries one (a stray export leftover) would otherwise add one per placement.
    node.camera     = -1;
    node.mesh       = src.mesh >= 0 && src.mesh < static_cast<int>(x.model.meshes.size()) ? src.mesh : -1;
    node.extensions = src.extensions;
    node.extras     = src.extras;
    SceneMerger::remapNodeResources(node, x.offsets);
    if(node.mesh >= 0)
    {
      tinygltf::Value::Object obj =
          node.extras.IsObject() ? node.extras.Get<tinygltf::Value::Object>() : tinygltf::Value::Object{};
      obj[Scene::kPlacedSourceKey] = tinygltf::Value(fmt::format("{}#{}", x.canonical, nodeIndex));
      node.extras                  = tinygltf::Value(std::move(obj));
    }

    const int index = addNode(std::move(node), parent);
    if(nodeMap != nullptr)
      (*nodeMap)[nodeIndex] = index;

    for(int child : src.children)
      if(child >= 0 && child < static_cast<int>(x.model.nodes.size()))
        copyTree(x, child, index, depth, nodeMap);

    if(src.externalAsset >= 0 && place(x, src.externalAsset, index, depth + 1, nullptr) != nullptr)
    {
      m_model.nodes[index].mesh   = -1;
      m_model.nodes[index].camera = -1;
    }
  }

  // Skins and animation channels of `x` pointing at the nodes of one copy (`nodeMap`: node of x ->
  // its copy) are given to that copy, so each placement animates on its own.
  void bindSkinsAndAnimations(const PlacedFile& x, const std::vector<int>& nodeMap)
  {
    std::unordered_map<int, int> skinCopies;  // skin of x -> skin of the copy (-1: can't be bound)
    for(size_t n = 0; n < nodeMap.size(); n++)
    {
      const int skin = x.model.nodes[n].skin;
      if(nodeMap[n] < 0 || skin < 0 || skin >= static_cast<int>(x.model.skins.size()))
        continue;
      const std::pair<std::unordered_map<int, int>::iterator, bool> copied = skinCopies.try_emplace(skin, -1);
      const std::unordered_map<int, int>::iterator                  it     = copied.first;
      if(copied.second)
      {
        tinygltf::Skin copy = x.model.skins[skin];
        copy.uid.clear();
        if(copy.inverseBindMatrices >= 0)
          copy.inverseBindMatrices += x.offsets.accessors;
        auto mapNode = [&nodeMap](int node) {
          return node >= 0 && node < static_cast<int>(nodeMap.size()) ? nodeMap[node] : -1;
        };
        bool complete = true;
        for(int& joint : copy.joints)
          complete = (joint = mapNode(joint)) >= 0 && complete;
        copy.skeleton = mapNode(copy.skeleton);
        if(complete)
        {
          it->second = static_cast<int>(m_model.skins.size());
          m_model.skins.push_back(std::move(copy));
        }
        else
          warnOnce("skin:" + x.canonical, fmt::format("'{}': a skin has joints outside the placed tree; drawn unskinned",
                                                      nvutils::utf8FromPath(x.path.filename())));
      }
      m_model.nodes[nodeMap[n]].skin = it->second;
    }

    for(size_t a = 0; a < x.model.animations.size(); a++)
    {
      std::vector<tinygltf::AnimationChannel>& channels = m_model.animations[x.offsets.animations + a].channels;
      for(const tinygltf::AnimationChannel& channel : x.model.animations[a].channels)
      {
        if(channel.target_node < 0 || channel.target_node >= static_cast<int>(nodeMap.size()) || nodeMap[channel.target_node] < 0)
          continue;
        tinygltf::AnimationChannel copy = channel;
        copy.target_node                = nodeMap[channel.target_node];
        channels.push_back(std::move(copy));
        m_channelCopies++;
      }
    }
  }

  // A mesh node of a static placed file, with its transform relative to the placing node.
  struct TreeMesh
  {
    int       mesh   = -1;  // in the placed file
    int       source = -1;  // into StaticTree::sources
    glm::mat4 relative{1.0f};
  };

  // What a static file's placements draw: its mesh nodes, and the authored node data (name,
  // weights, extras) that materializing an instance set writes back, shared by every placement.
  struct StaticTree
  {
    std::vector<TreeMesh>                              meshes;
    std::shared_ptr<const std::vector<tinygltf::Node>> sources;
  };

  // The static content of `x`, if it can be drawn through an instance set: nullptr if a node in it
  // is skinned, animated or a light (which the placement keeps), places a file, or has an
  // extension (which may need a node of its own, e.g. EXT_mesh_gpu_instancing). Computed once per
  // file.
  const StaticTree* staticTree(const PlacedFile& x)
  {
    const std::pair<StaticTrees::iterator, bool> found = m_staticTrees.try_emplace(&x);
    std::optional<StaticTree>&                   tree  = found.first->second;
    if(!found.second)
      return tree ? &*tree : nullptr;

    StaticTree                  result;
    std::vector<tinygltf::Node> sources;
    bool                        isStatic = true;
    for(size_t r = 0; r < x.roots.size() && isStatic; r++)
    {
      const int root = x.roots[r];
      isStatic = collectStaticMeshes(x, root, tinygltf::utils::getNodeMatrix(x.model.nodes[root]), 0, result.meshes, sources);
    }
    if(!isStatic)
      return nullptr;
    result.sources = std::make_shared<const std::vector<tinygltf::Node>>(std::move(sources));
    tree           = std::move(result);
    return &*tree;
  }

  // Appends the mesh nodes of node `n` of `x` and its descendants; false if one of them is not static.
  bool collectStaticMeshes(const PlacedFile&            x,
                           int                          n,
                           const glm::mat4&             relative,
                           int                          level,
                           std::vector<TreeMesh>&       meshes,
                           std::vector<tinygltf::Node>& sources) const
  {
    const tinygltf::Node& node  = x.model.nodes[n];
    const int             count = static_cast<int>(x.model.nodes.size());
    if(node.skin >= 0 || x.animatedNode[n] || node.light >= 0 || node.externalAsset >= 0 || !node.extensions.empty() || level > count)
      return false;  // (a level deeper than the node count is a malformed cycle)
    if(node.mesh >= 0 && node.mesh < static_cast<int>(x.model.meshes.size()))
    {
      tinygltf::Node source;
      source.name    = node.name;
      source.weights = node.weights;
      source.extras  = node.extras;
      meshes.push_back({node.mesh, static_cast<int>(sources.size()), relative});
      sources.push_back(std::move(source));
    }
    for(size_t c = 0; c < node.children.size(); c++)
    {
      const int child = node.children[c];
      if(child >= 0 && child < count
         && !collectStaticMeshes(x, child, relative * tinygltf::utils::getNodeMatrix(x.model.nodes[child]), level + 1, meshes, sources))
        return false;
    }
    return true;
  }

  // Give scene node `target` the instance set of the static content `tree` of `x`.
  void addInstanceSet(const PlacedFile& x, const StaticTree& tree, int target)
  {
    InstanceSet set;
    set.sources = tree.sources;
    set.entries.reserve(tree.meshes.size());
    for(size_t m = 0; m < tree.meshes.size(); m++)
      set.entries.push_back({x.offsets.meshes + tree.meshes[m].mesh, tree.meshes[m].relative, tree.meshes[m].source});
    m_instanceSetPlacements++;

    Scene::setInstanceSetMarker(m_model.nodes[target], static_cast<int>(m_scene.m_instanceSets.size()));
    m_scene.m_instanceSets.push_back(std::move(set));
  }

  int addNode(tinygltf::Node&& node, int parent)
  {
    const int index = static_cast<int>(m_model.nodes.size());
    m_model.nodes.push_back(std::move(node));
    m_model.nodes[parent].children.push_back(index);
    m_placedNodes++;
    return index;
  }

  using ResolvedPaths = std::map<std::pair<const PlacedFile*, int>, FileRef>;
  using PlacedFiles   = std::unordered_map<std::string, std::unique_ptr<PlacedFile>>;
  using StaticTrees   = std::unordered_map<const PlacedFile*, std::optional<StaticTree>>;

  Scene&           m_scene;
  tinygltf::Model& m_model;
  PlacedFile       m_sceneFile;

  PlacedFiles                     m_files;     // (canonical path, alias context)
  std::vector<std::string>        m_ancestry;  // files on the current placement path
  std::unordered_set<std::string> m_warned;
  ResolvedPaths                   m_resolvedPaths;  // (file, files index) -> what it points at
  StaticTrees                     m_staticTrees;    // empty optional: not static

  nvutils::PerformanceTimer m_timer;              // since construction
  double                    m_loadSeconds = 0.0;  // of which loading and appending files

  std::unordered_map<std::string, int> m_imageByFile;        // canonical image file -> scene image
  size_t                               m_indexedImages = 0;  // scene images recorded in m_imageByFile

  std::unordered_map<std::string, int> m_bufferByFile;                 // canonical buffer file -> scene buffer
  bool                                 m_indexedSceneBuffers = false;  // the scene's own buffers recorded

  std::unordered_map<std::string, LoadedFile> m_preloaded;  // by FileRef::key, until acquired

  std::unordered_set<std::string> m_loadedCanonical;  // canonical paths of the files loaded
  size_t                          m_sharedImages          = 0;
  size_t                          m_sharedBuffers         = 0;
  size_t                          m_placements            = 0;
  size_t                          m_instanceSetPlacements = 0;  // placements drawn through an instance set
  size_t                          m_placedNodes           = 0;
  size_t                          m_channelCopies         = 0;
  bool                            m_truncated             = false;
};

//--------------------------------------------------------------------------------------------------
// glTF 2.1: resolve every node.externalAsset of the loaded file by placing the referenced file
// under it. Placed nodes are tagged read-only and their provenance recorded in m_referencedAssets.
// Returns true if any referenced file was placed.
//
// Limitation: only external file `uri` references are handled (bufferView / data: URIs are skipped).
//--------------------------------------------------------------------------------------------------
bool Scene::resolveExternalAssets()
{
  if(m_model.externalAssets.empty() || m_model.files.empty() || m_model.nodes.empty())
    return false;

  // Snapshot the placing nodes: placing appends nodes.
  std::vector<int> placingNodes;
  for(int i = 0; i < static_cast<int>(m_model.nodes.size()); ++i)
    if(m_model.nodes[i].externalAsset >= 0)
      placingNodes.push_back(i);
  if(placingNodes.empty())
    return false;

  ExternalAssetPlacer placer(*this);
  placer.preload(placingNodes);
  if(m_loadProgress)
    m_loadProgress->stage("Placing external assets", static_cast<uint32_t>(placingNodes.size()));
  bool placedAny = false;
  for(int node : placingNodes)
  {
    placedAny = placer.placeSceneNode(node) || placedAny;
    if(m_loadProgress)
      m_loadProgress->advance(1, m_model.nodes[node].name);
  }
  placer.logSummary();
  m_externalFileCount = placer.loadedFiles();
  // Freeing thousands of parsed files takes as long as placing them (the heap serializes the frees,
  // so threads don't help): it happens on a thread of its own while the scene goes on to the GPU.
  if(m_externalAssetRelease.joinable())
    m_externalAssetRelease.join();
  m_externalAssetRelease = std::thread([files = placer.takeFiles()]() mutable { files.clear(); });
  return placedAny;
}

//--------------------------------------------------------------------------------------------------
// Place the file that the scene's own node `nodeIndex` references (runtime referenceScene()).
//--------------------------------------------------------------------------------------------------
bool Scene::placeExternalAsset(int nodeIndex)
{
  ExternalAssetPlacer placer(*this);
  placer.preload({nodeIndex});
  return placer.placeSceneNode(nodeIndex);
}

}  // namespace nvvkgltf
