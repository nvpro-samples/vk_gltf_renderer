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


#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>
#include <tinygltf/tiny_gltf.h>
#include <nvutils/bounding_box.hpp>

#include "tinygltf_utils.hpp"
#include "gltf_animation_pointer.hpp"
#include "gltf_interactivity_graph.hpp"
#include "gltf_interactivity_instance.hpp"

class LoadProgress;


namespace nvvkgltf {

// When the ratio of dirty elements to total elements exceeds this threshold,
// a full GPU buffer upload is used instead of individual per-element updates.
constexpr float kFullUpdateRatio = 0.3f;

// The render node is the instance of a primitive in the scene that will be rendered
struct RenderNode
{
  glm::mat4 worldMatrix  = glm::mat4(1.0f);
  int       materialID   = 0;   // Reference to the material
  int       renderPrimID = -1;  // Reference to the unique primitive
  int       refNodeID    = -1;  // Reference to the tinygltf::Node
  int       skinID       = -1;  // Reference to the skin, if the node is skinned, -1 if not skinned
  bool      visible      = true;
  bool      castShadows  = true;  // NV_node_shadow_casting: false keeps it out of shadow rays (TLAS mask)
};

// The RenderPrimitive is a unique primitive in the scene
struct RenderPrimitive
{
  tinygltf::Primitive* pPrimitive  = nullptr;
  int                  vertexCount = 0;
  int                  indexCount  = 0;
  int                  meshID      = 0;
  int                  nodeID      = -1;  // the node it is skinned or morphed for; -1 if only static nodes draw it
};

// glTF 2.1 external assets: what a placing node draws when the file it places is static (nothing
// skinned, animated, lit, with a camera, placing a file or carrying a node extension). One entry
// per mesh node of the file, with its transform relative to the placing node. The renderer expands
// the entries straight into render nodes, so the file's nodes are not copied (see
// docs/external_assets.md). Built by the placer and never changed; the placing node points at it
// through Scene::kInstanceSetKey in its extras.
struct InstanceSet
{
  struct Entry
  {
    int       mesh = -1;
    glm::mat4 local{1.0f};
    int       source = -1;  // into sources: the node data a materialized entry gets back
  };
  std::vector<Entry> entries;
  // The authored data of the placed file's mesh nodes (name, weights, extras), shared by every
  // placement of that file.
  std::shared_ptr<const std::vector<tinygltf::Node>> sources;
};

// glTF 2.1 external asset that was resolved and merged into the model. Records the provenance
// (source file + externalAsset/file indices) and the range of merged-in nodes so they can be
// treated as read-only and, later, re-externalized on save. The instance node itself (the node
// carrying `externalAsset`) stays editable and is NOT part of subtreeNodes.
struct ReferencedAsset
{
  int              instanceNodeIndex  = -1;  // node in m_model carrying externalAsset (editable)
  int              externalAssetIndex = -1;  // index into m_model.externalAssets
  int              fileIndex          = -1;  // index into m_model.files
  std::string      sourceUri;                // resolved source path/URI (provenance)
  std::vector<int> subtreeNodes;             // merged-in node indices (read-only)
};

struct RenderCamera
{
  enum CameraType
  {
    ePerspective,
    eOrthographic
  };

  CameraType type   = ePerspective;
  glm::dvec3 eye    = {0.0, 0.0, 0.0};
  glm::dvec3 center = {0.0, 0.0, 0.0};
  glm::dvec3 up     = {0.0, 1.0, 0.0};

  // Perspective
  double yfov = {0.0};  // in radians

  // Orthographic
  double xmag = {0.0};
  double ymag = {0.0};

  double znear = {0.0};
  double zfar  = {0.0};
};

// See: https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md
struct RenderLight
{
  glm::mat4 worldMatrix = glm::mat4(1.0f);
  int       light       = 0;
  int       nodeID      = -1;

  // EXT_lights_ies (read from the node, not the light): -1 = no profile attached.
  int       iesProfile    = -1;
  float     iesMultiplier = 1.0f;
  glm::vec3 iesColor      = glm::vec3(1.0f);
};

// Centralized registry for renderNode mappings (nodeID/primID <-> renderNodeID).
// The render nodes of one node are added together, so they are contiguous: each node keeps a range
// [first, first + count) into the flat arrays, and every lookup is an index, never a hash. Clearing
// keeps the capacity, so re-parsing a scene of millions of render nodes allocates and frees nothing.
class RenderNodeRegistry
{
public:
  // The render nodes of one node, in the order they were added.
  using RenderNodeRange = std::ranges::iota_view<int, int>;

  // Add a render node; returns the new renderNodeID. All render nodes of nodeID must be added one
  // after the other, and either all or none of them with a `local` matrix (see getLocalMatrices()).
  int addRenderNode(const RenderNode& node, int nodeID, int primIndex, const glm::mat4* local = nullptr);
  // Add the render node of primitive `primitive` of entry `entry` of nodeID's instance set. Its
  // primIndex is -1 (it draws no primitive of the node's own mesh); the draw is kept on the side.
  int addInstanceSetRenderNode(const RenderNode& node, int nodeID, int entry, int primitive, const glm::mat4& local);

  // Lookups
  int getRenderNodeID(int nodeID, int primIndex) const;  // (nodeID, primID) -> first RenderNodeID, -1 if not found
  std::optional<std::pair<int, int>> getNodeAndPrim(int renderNodeID) const;      // RenderNodeID -> (nodeID, primID)
  RenderNodeRange                    getRenderNodesForNode(int nodeID) const;     // nodeID -> [RenderNodeIDs]
  std::optional<std::pair<int, int>> getInstanceSetDraw(int renderNodeID) const;  // -> (entry, primitive)
  int getInstanceSetRenderNodeID(int nodeID, int entry, int primitive) const;     // -1 if not drawn

  // Nodes drawn more than once (EXT_mesh_gpu_instancing, an instance set) have one local matrix per
  // render node, in getRenderNodesForNode() order: the render node's world matrix is the node's world
  // matrix times it. Empty for a node drawn once.
  std::span<const glm::mat4> getLocalMatrices(int nodeID) const;
  const glm::mat4*           getLocalMatrix(int renderNodeID) const;  // nullptr for a node drawn once
  bool                       hasLocalMatrices() const { return !m_localMatrices.empty(); }

  // Batch: collect all renderNodeIDs for node and its descendants. getChildren(nodeID) returns child node indices.
  void getAllRenderNodesForNodeRecursive(int nodeID, std::function<std::vector<int>(int)> getChildren, std::vector<int>& outRenderNodeIDs) const;

  // Clear all mappings and the flat array.
  void clear();

  // Direct access to flat array (for GPU upload).
  const std::vector<RenderNode>& getRenderNodes() const { return m_renderNodes; }
  std::vector<RenderNode>&       getRenderNodes() { return m_renderNodes; }

private:
  std::vector<RenderNode> m_renderNodes;

  // Reverse: renderNodeID -> (nodeID, primIndex)
  std::vector<std::pair<int, int>> m_renderNodeToNodeAndPrim;

  // Per node (indexed by nodeID, grown on demand): its render nodes and, when drawn more than once,
  // where its local matrices start in m_localMatrices.
  struct NodeRange
  {
    int first      = 0;
    int count      = 0;
    int firstLocal = -1;
  };
  std::vector<NodeRange> m_nodeRanges;
  std::vector<glm::mat4> m_localMatrices;

  // renderNodeID -> (entry, primitive) of an instance-set render node, (-1, -1) for any other. Only
  // grown when an instance-set render node is added, so a scene without one pays nothing.
  std::vector<std::pair<int, int>> m_instanceSetDraws;

  const NodeRange* findRange(int nodeID) const
  {
    return nodeID >= 0 && static_cast<size_t>(nodeID) < m_nodeRanges.size() && m_nodeRanges[nodeID].count > 0 ?
               &m_nodeRanges[nodeID] :
               nullptr;
  }
};

// Animation data
struct AnimationInfo
{
  std::string name;
  float       start       = std::numeric_limits<float>::max();
  float       end         = std::numeric_limits<float>::lowest();
  float       currentTime = 0.0f;
  float       reset() { return currentTime = start; }
  float       incrementTime(float deltaTime, bool loop = true)
  {
    currentTime += deltaTime;
    if(loop)
    {
      float duration = end - start;
      // Wrap currentTime around using modulo arithmetic
      float wrapped = std::fmod(currentTime - start, duration);
      // fmod can return negative values if (currentTime - start) < 0, so fix that.
      if(wrapped < 0.0f)
        wrapped += duration;

      currentTime = start + wrapped;
    }
    else
    {
      if(currentTime > end)
      {
        currentTime = end;
      }
    }
    return currentTime;
  }
};


/*-------------------------------------------------------------------------------------------------

# nvh::nvvkgltf::Scene 

The Scene class is responsible for loading and managing a glTF scene.
- It is used to load a glTF file and parse it into a scene representation.
- It can be used to save the scene back to a glTF file.
- It can be used to manage the animations of the scene.
- What it returns is a list of RenderNodes, RenderPrimitives, RenderCameras, and RenderLights.
-  RenderNodes are the instances of the primitives in the scene that will be rendered.
-  RenderPrimitives are the unique primitives in the scene.

-------------------------------------------------------------------------------------------------*/

class SceneEditor;
class AnimationSystem;
class SceneValidator;
class ExternalAssetPlacer;

class Scene
{
public:
  //--------------------------------------------------------------------------------------------------
  // Types and Enums
  //--------------------------------------------------------------------------------------------------

  // Used to specify the type of pipeline to be used (Raster only)
  enum PipelineType
  {
    eRasterSolid,
    eRasterSolidDoubleSided,
    eRasterBlend,
    eRasterAll
  };

  // Validation result structure
  struct ValidationResult
  {
    bool                     valid = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;

    void addError(const std::string& msg)
    {
      errors.push_back(msg);
      valid = false;
    }

    void addWarning(const std::string& msg) { warnings.push_back(msg); }

    void print() const;
  };

  //--------------------------------------------------------------------------------------------------
  // Construction / Destruction
  //--------------------------------------------------------------------------------------------------

  Scene();
  ~Scene();

  SceneEditor&           editor();
  AnimationSystem&       animation();
  const AnimationSystem& animation() const;
  SceneValidator&        validator();
  const SceneValidator&  validator() const;

  //--------------------------------------------------------------------------------------------------
  // File I/O
  //--------------------------------------------------------------------------------------------------

  [[nodiscard]] bool load(const std::filesystem::path& filename);  // Load .gltf or .glb
  // Save .gltf or .glb. For scenes that reference external assets (glTF 2.1), selfContained=false
  // (default) re-externalizes them (small file, keeps references), while selfContained=true bakes
  // the merged content inline and drops all external references (portable, shareable file).
  [[nodiscard]] bool save(const std::filesystem::path& filename, bool selfContained = false);

  // Called by save() on the model that is actually written, just before extensionsUsed is
  // reconciled. Lets the application add extensions it owns but the scene graph does not hold
  // (the environment sky). Writing to the live model instead is a silent no-op whenever the
  // scene has external assets, because save() then serializes a transformed copy.
  void setPreSaveHook(std::function<void(tinygltf::Model&, const std::filesystem::path& destination)> hook)
  {
    m_preSaveHook = std::move(hook);
  }
  // Merge another glTF into this scene. Optional maxTextureCount validates combined texture limit (e.g. GPU descriptor limit).
  [[nodiscard]] int mergeScene(const std::filesystem::path& filename, std::optional<uint32_t> maxTextureCount = std::nullopt);  // Returns wrapper node index, or -1 on failure
  // glTF 2.1: add another glTF as a referenced external asset (read-only, re-externalized on save)
  // instead of embedding it. Repeated references to the same file share geometry. Returns the new
  // instance node index, or -1 on failure.
  [[nodiscard]] int referenceScene(const std::filesystem::path& filename);
  // After mergeScene(), returns the animation list index of the first clip from the merged file (for UI default); -1 if none.
  // Consumed once by the renderer so the animation dropdown selects merged motion instead of staying on a base-scene clip.
  [[nodiscard]] int                      takeMergePreferredAnimationIndex();
  // Use a pre-loaded model; sourceFilename (e.g. the converted .obj) anchors its relative image URIs
  void takeModel(tinygltf::Model&& model, const std::filesystem::path& sourceFilename = {});
  std::unordered_set<std::string>&       supportedExtensions() { return m_supportedExtensions; }
  const std::unordered_set<std::string>& supportedExtensions() const { return m_supportedExtensions; }
  const std::filesystem::path&           getFilename() const { return m_filename; }
  // Order: base scene directory first, then each imported scene's directory in merge order.
  // Used to resolve image URIs via findFile and when saving/copying images.
  const std::vector<std::filesystem::path>& getImageSearchPaths() const { return m_imageSearchPaths; }
  // Where loading reports its stages (load_progress.hpp); the GPU side (SceneVk) reports there too.
  // Not owned, may be null; reporting outside an active load is a no-op.
  void          setLoadProgress(LoadProgress* progress) { m_loadProgress = progress; }
  LoadProgress* loadProgress() const { return m_loadProgress; }

  //--------------------------------------------------------------------------------------------------
  // Model Access
  //--------------------------------------------------------------------------------------------------

  const tinygltf::Model& getModel() const { return m_model; }
  tinygltf::Model&       getModel() { return m_model; }
  [[nodiscard]] bool     valid() const { return m_validSceneParsed; }

  //--------------------------------------------------------------------------------------------------
  // External Assets (glTF 2.1)
  //--------------------------------------------------------------------------------------------------

  // External assets that were resolved and merged into the model (provenance + read-only ranges).
  const std::vector<nvvkgltf::ReferencedAsset>& getReferencedAssets() const { return m_referencedAssets; }
  // The instance set a placing node draws (see InstanceSet), or nullptr. Read through a marker in
  // the node's extras, like kExternalAssetContentKey, so it follows the node through renumbering and
  // duplication; the marker is never saved.
  static constexpr const char*     kInstanceSetKey = "NV_placed_instance_set";
  [[nodiscard]] const InstanceSet* getInstanceSet(int nodeIndex) const;
  [[nodiscard]] static int         getInstanceSetIndex(const tinygltf::Node& node);
  static void                      setInstanceSetMarker(tinygltf::Node& node, int instanceSetIndex);
  static void                      clearInstanceSetMarker(tinygltf::Node& node);
  // The (entry, primitive) of its node's instance set a render node draws, or nullopt when it draws
  // a primitive of its node's own mesh. getInstanceSetRenderNode is the reverse (-1 if not drawn).
  [[nodiscard]] std::optional<std::pair<int, int>> getInstanceSetDraw(int renderNodeIndex) const;
  [[nodiscard]] int getInstanceSetRenderNode(int nodeIndex, int entry, int primitive) const;
  // What a render node draws: its node, and the mesh and primitive index -- the node's own mesh, or
  // the mesh of the instance-set entry it draws. All -1 for an invalid render node.
  struct RenderNodePart
  {
    int node = -1, mesh = -1, primitive = -1;
  };
  [[nodiscard]] RenderNodePart getRenderNodePart(int renderNodeIndex) const;
  // Turn the instance sets of `nodes` (every marked node when empty) into real nodes
  // of `model`, one per entry, under the placing node, and drop the markers. For saves that need
  // plain glTF (flatten) and for "Make Editable". `model` is m_model or a copy of it.
  void materializeInstanceSets(tinygltf::Model& model, const std::unordered_set<int>& nodes = {}) const;
  // True if the scene references at least one external asset (glTF 2.1). When true, save() can
  // write either the referenced (complex) form or a flattened self-contained form.
  [[nodiscard]] bool hasExternalAssets() const { return !m_referencedAssets.empty(); }

  // Full path of the glTF a node comes from: the file an instance node's `externalAsset` points at, or
  // the source of the referenced asset a merged-in node is marked with. Both are read from the live
  // node, so the answer survives node deletion renumbering. Empty for ordinary nodes.
  [[nodiscard]] std::string referencedFilePathOf(int nodeIndex) const;

  // Provenance marker stamped into the `extras` of nodes merged in from a referenced external
  // asset. Stored in extras (not a Node struct field) so it travels with the node through
  // duplicate/delete renumbering for free, and is never persisted (re-externalization removes the
  // marked nodes before save). The value is the index into m_referencedAssets.
  static constexpr const char* kExternalAssetContentKey = "NV_external_asset_content";
  // Where a placed mesh node was copied from, as "<file>#<node>" (content copied as nodes: skinned,
  // animated). Copies of one node in several placements animate in lockstep and are read-only, so
  // they share one skinned or morphed render primitive (assignDeformedRenderPrimitives). Set by the
  // placer, cleared with the read-only marker.
  static constexpr const char* kPlacedSourceKey = "NV_placed_source";

  // Single source of truth for the read-only marker (set/query/clear on a node's `extras`). Static
  // so the merge/flatten/re-externalize passes (which operate on a bare tinygltf::Model) reuse them.
  static void setExternalAssetMarker(tinygltf::Node& node, int referencedAssetIndex)
  {
    tinygltf::Value::Object obj = node.extras.IsObject() ? node.extras.Get<tinygltf::Value::Object>() : tinygltf::Value::Object{};
    obj[kExternalAssetContentKey] = tinygltf::Value(referencedAssetIndex);
    node.extras                   = tinygltf::Value(std::move(obj));
  }
  [[nodiscard]] static bool hasExternalAssetMarker(const tinygltf::Node& node)
  {
    // Value::Has takes a std::string, and the key is too long for the small-string buffer: a key
    // made per call is a heap allocation per node, which a pass over millions of nodes notices.
    static const std::string s_externalAssetKey(kExternalAssetContentKey);
    return node.extras.Has(s_externalAssetKey);
  }
  static void clearExternalAssetMarker(tinygltf::Node& node)
  {
    if(!node.extras.IsObject() || (!node.extras.Has(kExternalAssetContentKey) && !node.extras.Has(kPlacedSourceKey)))
      return;
    tinygltf::Value::Object obj = node.extras.Get<tinygltf::Value::Object>();
    obj.erase(kExternalAssetContentKey);
    obj.erase(kPlacedSourceKey);
    // Reset to null when empty so tinygltf does not emit an empty "extras": {}.
    node.extras = obj.empty() ? tinygltf::Value() : tinygltf::Value(std::move(obj));
  }

  // True if the node was merged in from a referenced external asset and should be treated read-only.
  [[nodiscard]] bool isNodeReadOnly(int nodeIndex) const
  {
    return nodeIndex >= 0 && nodeIndex < static_cast<int>(m_model.nodes.size())
           && hasExternalAssetMarker(m_model.nodes[nodeIndex]);
  }
  // True if the mesh is used by a referenced (read-only) node or drawn by an instance set. Used to
  // guard mesh-keyed edits.
  [[nodiscard]] bool isMeshReadOnly(int meshIndex) const;
  // True if the material is used by a read-only mesh, i.e. it was loaded from a referenced file.
  [[nodiscard]] bool isMaterialReadOnly(int materialIndex) const;

  // Who uses an element, for the Inspector and the Elements list, which ask every frame. These and
  // the two read-only queries above read one index (ElementUsage), built in one pass over the nodes
  // and meshes and kept until the scene graph, a material assignment or an element count changes.
  [[nodiscard]] std::span<const int> getMeshNodes(int meshIndex) const;         // nodes with node.mesh == meshIndex
  [[nodiscard]] std::span<const int> getMeshPlacingNodes(int meshIndex) const;  // nodes whose instance set draws it
  [[nodiscard]] int                  getMaterialPrimitiveCount(int materialIndex) const;  // mesh primitives using it
  [[nodiscard]] int                  getCameraNode(int cameraIndex) const;  // first node carrying it, or -1
  [[nodiscard]] int                  getLightNode(int lightIndex) const;    // first node carrying it, or -1
  // Rebuild the usage index on its next query: for edits that change what it records without a
  // scene-graph revision bump (camera attachment).
  void invalidateElementUsage() { m_elementUsage.revision = ~uint64_t(0); }

  // KHR_node_selectability: a node is selectable iff neither it nor any ancestor sets
  // `selectable:false` (the flag cascades to the whole subtree). Default is selectable.
  [[nodiscard]] bool isNodeSelectable(int nodeIndex) const;
  // Returns `nodeIndex` if it is selectable, otherwise the closest selectable ancestor, or -1 if none.
  [[nodiscard]] int nearestSelectableAncestor(int nodeIndex) const;
  // KHR_node_hoverability: same cascading-opt-out shape as selectability, gating hover instead.
  [[nodiscard]] bool isNodeHoverable(int nodeIndex) const;
  // Returns `nodeIndex` if it is hoverable, otherwise the closest hoverable ancestor, or -1 if none.
  [[nodiscard]] int nearestHoverableAncestor(int nodeIndex) const;
  // True if the node is an external-asset instance node (carries an `externalAsset` link). These
  // nodes stay editable; "Make Editable" (SceneEditor::makeExternalAssetEditable) breaks the link.
  [[nodiscard]] bool isExternalAssetInstance(int nodeIndex) const
  {
    return nodeIndex >= 0 && nodeIndex < static_cast<int>(m_model.nodes.size()) && m_model.nodes[nodeIndex].externalAsset >= 0;
  }

  //--------------------------------------------------------------------------------------------------
  // Scene Management
  //--------------------------------------------------------------------------------------------------

  void                          setCurrentScene(int sceneID);  // Parse scene and create render nodes
  [[nodiscard]] int             getCurrentScene() const { return m_currentScene; }
  const std::vector<glm::mat4>& getNodesWorldMatrices() const { return m_nodesWorldMatrices; }
  const std::vector<glm::mat4>& getNodesLocalMatrices() const { return m_nodesLocalMatrices; }
  void                          updateNodeWorldMatrices();
  void                          updateLocalMatricesAndLights();
  glm::mat4                     computeNodeWorldMatrix(int nodeID) const;
  bool                          isEmissiveAreaLight(const RenderNode& rn) const;

  // The GPU transform path propagates world matrices on-device only, leaving the CPU mirror
  // (m_nodesWorldMatrices / RenderNode.worldMatrix) stale for the nodes it moved. That path records
  // those nodes here; the CPU sync path merges them back into the dirty set so only those subtrees are
  // recomputed (not the whole scene) before render-node / TLAS transforms are sourced from the mirror.
  void addGpuStaleNodes(const std::unordered_set<int>& nodes) { m_gpuStaleNodes.insert(nodes.begin(), nodes.end()); }
  bool mergeGpuStaleNodesIntoDirty();  // Moves recorded stale nodes into the dirty set; returns true if any were pending.

  // Topological BFS levels (see buildTopologicalLevels) — shared by CPU parallel propagation and GPU transform compute.
  [[nodiscard]] const std::vector<int>&                 getTopoNodeOrder() const { return m_topoLevels.nodeOrder; }
  [[nodiscard]] const std::vector<std::pair<int, int>>& getTopoLevels() const { return m_topoLevels.levels; }
  [[nodiscard]] const std::vector<int>&                 getNodeParents() const { return m_nodeParents; }
  // Bumped when traverseSceneWithVisibility completes (parents/m_topoLevels/world matrices refresh), when
  // variant changes alter render-node material IDs, editor assigns a primitive material, etc.
  // TransformComputeVk matches this to refresh static SSBOs (mappings, parents, topo, instancing locals)
  // without relying only on buildAccelerationStructures.
  [[nodiscard]] uint64_t getSceneGraphRevision() const { return m_sceneGraphRevision; }
  void                   bumpSceneGraphRevision() { ++m_sceneGraphRevision; }

  // Nodes drawn more than once: EXT_mesh_gpu_instancing nodes and placing nodes with an instance
  // set. For such a node, one local matrix per render node, in getRenderNodesForNode() order: the
  // render node's world matrix is the node's world matrix times it. Empty for a node drawn once.
  [[nodiscard]] std::span<const glm::mat4> getRenderNodeLocalMatrices(int nodeID) const
  {
    return m_renderNodeRegistry.getLocalMatrices(nodeID);
  }

  //--------------------------------------------------------------------------------------------------
  // Variant Management
  //--------------------------------------------------------------------------------------------------

  void                            setCurrentVariant(int variant);  // Marks dirty RenderNodes internally
  const std::vector<std::string>& getVariants() const { return m_variants; }
  [[nodiscard]] int               getCurrentVariant() const { return m_currentVariant; }
  [[nodiscard]] std::unordered_set<int> getMaterialRenderNodes(const std::unordered_set<int>& materialVariantNodeIDs) const;

  //--------------------------------------------------------------------------------------------------
  // KHR_interactivity Management
  //
  // m_interactivityGraphs is the compiled (parsed-once, immutable) form; m_interactivityInstances
  // holds one runtime InteractivityGraphInstance per graph, index-aligned. Only the default graph
  // (KHR_interactivity.graph, default 0) auto-runs each frame via tickInteractivityGraphs(); the
  // others exist so a future UI can preview/switch between them. See docs/interactivity.md.
  //--------------------------------------------------------------------------------------------------

  [[nodiscard]] const std::vector<InteractivityGraph>& getInteractivityGraphs() const { return m_interactivityGraphs; }
  [[nodiscard]] int                         getDefaultInteractivityGraph() const { return m_defaultInteractivityGraph; }
  [[nodiscard]] InteractivityGraphInstance* getInteractivityInstance(int graphIndex);

  // (Re-)builds m_interactivityGraphs + fresh m_interactivityInstances from the current
  // model.extensions["KHR_interactivity"]. Called automatically at the end of load(); public so
  // callers that mutate getModel().extensions directly (e.g. tests building a graph on top of an
  // already-loaded scene) can pick the change up without a full reload.
  void parseInteractivityGraphs();

  // Ticks the default graph's instance (starts it on first call) by deltaSeconds. No-op if the
  // model has no KHR_interactivity graphs. Returns true if anything about the graph's runtime
  // state changed (currently: any successful tick - see docs/interactivity.md's Phase A caveat
  // that variable/pointer writes are not yet GPU-visible).
  bool tickInteractivityGraphs(float deltaSeconds);

  // Fires event/onSelect (KHR_node_selectability) handlers bound anywhere on nodeIndex's ancestor
  // chain (inclusive), spec-mandated bubbling - every bound ancestor fires, not just the nearest.
  // controllerIndex is always 0 (this app has one interaction source: the desktop mouse).
  void notifyNodeSelected(int nodeIndex, const glm::vec3& selectionPoint, const glm::vec3& selectionRayOrigin);

  // Fires event/onHoverIn (for newNodeIndex's ancestor chain) and event/onHoverOut (for
  // previousNodeIndex's ancestor chain) handlers, bubbling up to but excluding their lowest common
  // ancestor (spec: a node only receives a new hover transition when the hover action actually
  // enters/exits its whole subtree, not on a transition between its own descendants). Pass -1 for
  // "nothing was/is hovered" (first hover, or the cursor left the viewport/all geometry).
  void notifyNodeHoverChanged(int previousNodeIndex, int newNodeIndex);

  //--------------------------------------------------------------------------------------------------
  // Camera Management
  //--------------------------------------------------------------------------------------------------

  // Rebuild: if true, clears and repopulates the camera list from the scene graph before returning.
  const std::vector<nvvkgltf::RenderCamera>& getRenderCameras(bool rebuild = false);
  void                                       setSceneCamera(const nvvkgltf::RenderCamera& camera);
  void                                       setSceneCameras(const std::vector<nvvkgltf::RenderCamera>& cameras);

  //--------------------------------------------------------------------------------------------------
  // Light Management
  //--------------------------------------------------------------------------------------------------

  const std::vector<nvvkgltf::RenderLight>& getRenderLights() const { return m_lights; }

  // Index into getRenderLights() of the sky's sun -- the light whose node carries the
  // OMI_environment_sky sun marker (gltf_environment_sky.hpp) -- or -1. Resolved when the lights
  // are rebuilt and when the marker is set or cleared, which must call updateSkySunLight().
  [[nodiscard]] int getSkySunLight() const { return m_skySunLight; }
  void              updateSkySunLight();

  //--------------------------------------------------------------------------------------------------
  // Render Node Management
  //--------------------------------------------------------------------------------------------------

  const std::vector<nvvkgltf::RenderNode>& getRenderNodes() const { return m_renderNodeRegistry.getRenderNodes(); }
  const RenderNodeRegistry&                getRenderNodeRegistry() const { return m_renderNodeRegistry; }
  RenderNodeRegistry&                      getRenderNodeRegistry() { return m_renderNodeRegistry; }
  [[nodiscard]] bool                       collectRenderNodeIndices(const std::unordered_set<int>& nodeIndices,
                                                                    std::unordered_set<int>&       outRenderNodeIndices,
                                                                    bool                           includeDescendants = true,
                                                                    float                          fullUpdateRatio = kFullUpdateRatio) const;
  // Uses m_dirtyFlags.nodes to populate renderNodesVk/Rtx
  void updateRenderNodeDirtyFromNodes(bool includeDescendants = true);
  // Whether this frame's dirty state can move geometry: render nodes already marked, a visibility
  // change, or a dirty node whose subtree holds a mesh. False when only mesh-less nodes changed --
  // a light (the sky's sun), a camera, an empty group -- which need a light upload at most, never
  // the transform compute or a TLAS update. Stops at the first mesh, so a large moved subtree
  // costs no more than a light does.
  [[nodiscard]] bool dirtyNodesReachGeometry() const;
  [[nodiscard]] int  getRenderNodeForPrimitive(int nodeIndex, int primitiveIndex) const;
  [[nodiscard]] int  getPrimitiveIndexForRenderNode(int renderNodeIndex) const;

  //--------------------------------------------------------------------------------------------------
  // Render Primitive Management
  //--------------------------------------------------------------------------------------------------

  const std::vector<nvvkgltf::RenderPrimitive>& getRenderPrimitives() const { return m_renderPrimitives; }
  const nvvkgltf::RenderPrimitive&              getRenderPrimitive(size_t ID) const { return m_renderPrimitives[ID]; }
  [[nodiscard]] size_t                          getNumRenderPrimitives() const { return m_renderPrimitives.size(); }
  // The render primitive drawing primitive `primitive` of mesh `mesh`, or -1 (not renderable, or a
  // mesh added since the last parse). Several mesh primitives may share one render primitive. A node
  // that skins or morphs the primitive may draw a render primitive of its own instead.
  [[nodiscard]] int getMeshRenderPrimitive(int mesh, size_t primitive) const;
  // The morph target weights a render primitive is blended with: its node's, or its mesh's when the
  // node has none (node.weights override mesh.weights).
  [[nodiscard]] const std::vector<double>& getMorphWeights(const nvvkgltf::RenderPrimitive& renderPrimitive) const;
  // Object-space AABB centroids, parallel to getRenderPrimitives(). Populated once when the
  // primitive set is built from the POSITION accessor min/max; reused by the rasterizer's
  // transparent depth sort to avoid re-parsing tinygltf accessors per frame.
  const std::vector<glm::vec3>& getRenderPrimCenterObj() const { return m_renderPrimCenterObj; }

  //--------------------------------------------------------------------------------------------------
  // Shading & Statistics
  //--------------------------------------------------------------------------------------------------

  // Returns render node indices that belong to the requested draw bucket. Backed by a cache
  // that is reconciled lazily from m_dirtyFlags on first call per frame; returned reference is
  // valid until the next scene mutation (material dirty, variant change, scene reload, ...).
  const std::vector<uint32_t>& getShadedNodes(PipelineType type) const;
  // True when any glTF material in the scene has a non-zero KHR_materials_transmission factor.
  // Used by the rasterizer to gate the screen-space-refraction capture pass: a scene that only
  // has plain alpha-blend draws does not need the opaque framebuffer captured. Backed by the
  // same cache reconciliation as getShadedNodes().
  [[nodiscard]] bool hasTransmissionMaterials() const;
  // Monotonic counter bumped whenever reconcileShadedNodesCache() actually rebuilds the lists.
  // Callers that cache a snapshot of getShadedNodes() can detect membership change (e.g. a
  // material alphaMode toggle that didn't change the list size) by watching this revision.
  [[nodiscard]] uint64_t getShadedNodesRevision() const
  {
    reconcileShadedNodesCache();
    return m_shadedNodesRevision;
  }
  [[nodiscard]] uint64_t getNumTriangles() const { return m_numTriangles; }
  // Returns cached bounds; lazily computes on first call (mutable cache, logically const).
  [[nodiscard]] nvutils::Bbox getSceneBounds() const;
  // AABB of a node's mesh and all its descendants' meshes, expressed in the node's own frame *before*
  // its TRS: multiply the extents by the node's scale for its displayed size. Uses accessor min/max
  // (bind pose; skinning and morph targets are not evaluated). Empty if the subtree has no geometry.
  [[nodiscard]] nvutils::Bbox computeNodeSubtreeBounds(int nodeID) const;

  //--------------------------------------------------------------------------------------------------
  // Resource Management
  //--------------------------------------------------------------------------------------------------

  void destroy();

  // Compact scene model - remove orphaned resources (meshes, materials, textures, images, samplers, skins, cameras, animations, lights, and geometry data)
  // Returns true if any resources were removed.
  // Use after operations that create orphaned resources: delete nodes, import then delete, etc.
  // IMPORTANT: After compaction, re-parse the active scene so render nodes / dirty flags match the
  // compacted model (e.g. setCurrentScene + parseScene), then rebuild GPU (e.g. rebuildVulkanSceneFull).
  [[nodiscard]] bool compactModel();

  //--------------------------------------------------------------------------------------------------
  // Debug and Diagnostics
  //--------------------------------------------------------------------------------------------------

  void rebuildRenderNodes();  // Force rebuild of all render nodes (for debugging/validation)

  //--------------------------------------------------------------------------------------------------
  // Dirty Tracking for GPU Updates
  //--------------------------------------------------------------------------------------------------

  struct DirtyFlags
  {
    std::unordered_set<int> renderNodesVk;   // RenderNode indices for SceneVk
    std::unordered_set<int> renderNodesRtx;  // RenderNode indices for SceneRTX
    std::unordered_set<int> materials;       // Material indices
    std::unordered_set<int> lights;          // Light indices (glTF light array)
    std::unordered_set<int> nodes;           // Node indices (for transform updates)
    std::unordered_set<int> samplers;  // glTF sampler indices whose wrap/filter changed in place (no image reload)
    bool                    allRenderNodesDirty = false;  // Full RN upload (count change or massive reorder)
    bool                    primitivesChanged   = false;  // BLAS rebuild needed (primitive set changed)
    bool                    texturesChanged = false;  // Full texture rebuild needed (structural image/texture change)
    bool texturesTailChanged = false;  // Only adding or removing from the end of the texture list (SceneVk::syncTextureTail)
    bool tlasVisibilityNeedsCpuSync = false;  // KHR_node_visibility / NV_node_shadow_casting: SceneEditor::updateVisibility / updateShadowCasting
    bool emissiveDirty = false;  // Emissive area-light list must be rebuilt (SceneVk::uploadEmissiveTriangles)

    void clear()
    {
      renderNodesVk.clear();
      renderNodesRtx.clear();
      materials.clear();
      lights.clear();
      nodes.clear();
      samplers.clear();
      allRenderNodesDirty        = false;
      primitivesChanged          = false;
      texturesChanged            = false;
      texturesTailChanged        = false;
      tlasVisibilityNeedsCpuSync = false;
      emissiveDirty              = false;
    }

    [[nodiscard]] bool isEmpty() const
    {
      return renderNodesVk.empty() && renderNodesRtx.empty() && materials.empty() && lights.empty() && nodes.empty()
             && samplers.empty() && !allRenderNodesDirty && !primitivesChanged && !texturesChanged
             && !texturesTailChanged && !tlasVisibilityNeedsCpuSync && !emissiveDirty;
    }
  };

  const DirtyFlags& getDirtyFlags() const { return m_dirtyFlags; }
  DirtyFlags&       getDirtyFlags() { return m_dirtyFlags; }
  void              clearDirtyFlags() { m_dirtyFlags.clear(); }

  void markMaterialDirty(int materialIndex);
  void markLightDirty(int lightIndex);
  void markRenderNodeDirty(int renderNodeIndex, bool forVk = true, bool forRtx = true);
  void markNodeDirty(int nodeIndex);
  // Bumped by every markNodeDirty(): a node's TRS (or what hangs off it) may have changed. Lets a
  // cache of something derived from node transforms, such as computeNodeSubtreeBounds(), be kept.
  [[nodiscard]] uint64_t getNodeTransformRevision() const { return m_nodeTransformRevision; }
  void markSamplerDirty(int samplerIndex);  // In-place wrap/filter edit; SceneVk updates the VkSampler only
  void markRenderNodeRtxDirtyForMaterials(const std::unordered_set<int>& materialIds);  // For TLAS instance flags

private:
  friend class SceneEditor;
  friend class ExternalAssetPlacer;

  // Application-supplied writer run by save() on the serialized model; see setPreSaveHook.
  std::function<void(tinygltf::Model&, const std::filesystem::path& destination)> m_preSaveHook;

  //--------------------------------------------------------------------------------------------------
  // Private Methods: Load and Extension Handling
  //--------------------------------------------------------------------------------------------------

  // Decompress KHR/EXT_meshopt_compression buffer views in-place; removes extension when done.
  // Returns false on decompression failure (caller should clear and return).
  bool decompressMeshoptExtension();

  //--------------------------------------------------------------------------------------------------
  // Private Methods: Scene Parsing
  //--------------------------------------------------------------------------------------------------

  void parseScene();
  void setSourceFilename(const std::filesystem::path& filename);
  void clearParsedData();
  // glTF 2.1: resolve every node.externalAsset by placing the referenced file's content under that
  // node, nested files too, each file loaded once. Tags the
  // placed nodes read-only and records provenance. Returns true if at least one file was placed.
  // Implemented in gltf_scene_external_assets.cpp.
  bool resolveExternalAssets();
  // The same for one node of the scene's own file (runtime referenceScene()).
  bool placeExternalAsset(int nodeIndex);
  // Tag the appended node range [firstNode, lastNode) read-only and record provenance. Shared by
  // load-time resolveExternalAssets() and runtime referenceScene(). Returns the m_referencedAssets index.
  int recordReferencedAsset(int firstNode, int lastNode, int instanceNode, int externalAssetIndex, int fileIndex, const std::string& uri);
  void parseVariants();
  void setSceneElementsDefaultNames();
  void createSceneCamera();

  //--------------------------------------------------------------------------------------------------
  // Private Methods: Render Node Helpers
  //--------------------------------------------------------------------------------------------------

  using PrimitiveKeyMap = std::map<std::string, int>;

  PrimitiveKeyMap buildPrimitiveKeyMap();
  std::string     primitiveKey(const tinygltf::Primitive& primitive) const;
  int             getMaterialVariantIndex(const tinygltf::Primitive& primitive, int currentVariant);
  void createRenderNodesForNode(int nodeID, const glm::mat4& worldMatrix, bool visible, const PrimitiveKeyMap& primMap);
  bool handleRenderNode(int nodeID, glm::mat4 worldMatrix, const PrimitiveKeyMap& primMap);
  std::vector<glm::mat4> gpuInstancingMatrices(const tinygltf::Value& attributes);
  void createInstanceSetRenderNodes(int nodeID, const InstanceSet& set, const glm::mat4& worldMatrix, bool visible, const PrimitiveKeyMap& primMap);
  int  renderPrimitiveOf(int nodeID, int mesh, size_t primitive, const PrimitiveKeyMap& primMap) const;
  void assignDeformedRenderPrimitives();
  std::string deformedPoseKey(int nodeID, size_t primitive, int sharedMesh, const std::unordered_set<int>& animatedWeights) const;
  bool handleCameraTraversal(int nodeID, const glm::mat4& worldMatrix);
  bool handleLightTraversal(int nodeID, const glm::mat4& worldMatrix);
  void updateRenderNodesFull();

  // Lightweight rebuild after structural change (e.g. node deletion): rebuilds the render primitives
  // and repopulates render nodes and lights; does not touch cameras, animations, variants.
  void rebuildRenderNodesAndLights();
  // Same, keeping the render primitives, and with dirty flags diffed as parseScene() does. Only for
  // an edit that changes no mesh and no skinned, morphed or camera node (see the definition).
  void rebuildRenderNodesKeepingPrimitives();
  void createRenderNodesAndLights(const PrimitiveKeyMap& primMap);  // one walk, both cleared first

  // A render node's state before a rebuild, to set precise dirty flags after it.
  struct RenderNodeState
  {
    glm::mat4 worldMatrix;
    int       materialID;
    int       renderPrimID;
    bool      visible;
  };
  std::vector<RenderNodeState> snapshotRenderNodes() const;
  void markRebuiltRenderNodesDirty(const std::vector<RenderNodeState>& previous, size_t previousLightCount);

  // Traverse scene roots computing local/world matrices, visibility, and parent links.
  void traverseSceneWithVisibility(const std::function<void(int nodeID, const glm::mat4& worldMatrix, bool visible)>& callback);

  // Expand a set of node indices to their associated render node IDs, optionally including descendants.
  void expandNodesToRenderNodes(const std::unordered_set<int>&               nodeIndices,
                                bool                                         includeDescendants,
                                const std::function<void(int renderNodeID)>& callback) const;

  //--------------------------------------------------------------------------------------------------
  // Private Methods: Utilities
  //--------------------------------------------------------------------------------------------------

  // Creates missing tangents for all primitives that need them (normal map, no TANGENT).
  // Deduplicates by primitive key so shared geometry keeps one TANGENT accessor. Safe to call anytime.
  void createMissingTangentsForModel();

  // For each image whose URI doesn't resolve on disk, try common alternative extensions
  // (.dds, .ktx2, etc.) and update the URI if a match is found. Called after load/import
  // so that all downstream consumers (Vulkan upload, save/copy, display) see the correct path.
  void resolveImageURIs();

  //--------------------------------------------------------------------------------------------------
  // Private Methods: Shaded-Nodes Cache (see getShadedNodes())
  //--------------------------------------------------------------------------------------------------

  // 3-bit key that determines which raster bucket a material's render nodes fall into.
  // bit0 = alphaMode == "OPAQUE", bit1 = doubleSided, bit2 = KHR_materials_transmission factor > 0.
  // Any field not in this key can change without invalidating the shaded-nodes cache.
  static uint8_t computeMaterialBucketKey(const tinygltf::Material& mat);

  // Lazily reconcile the per-bucket cache against m_dirtyFlags. Compares the current material
  // bucket key against the cached key; rebuilds the four lists only when a key actually flipped,
  // a render node's materialID was dirtied (renderNodesVk), or a structural reset flag is set.
  // Does NOT consume any dirty flags (SceneVk / SceneRTX still need them).
  void reconcileShadedNodesCache() const;

  // Compute the object-space AABB centroid for a render primitive from its POSITION accessor
  // min/max, falling back to (0,0,0) when the accessor is missing or has no min/max arrays.
  glm::vec3 computePrimitiveCenterObj(const tinygltf::Primitive& primitive) const;


  //--------------------------------------------------------------------------------------------------
  // Data Members: Core glTF Model
  //--------------------------------------------------------------------------------------------------

  tinygltf::Model                    m_model;             // The glTF model (source of truth)
  std::filesystem::path              m_filename;          // Loaded file path
  std::vector<std::filesystem::path> m_imageSearchPaths;  // Base dirs for image resolution (base first, then imports)
  std::unordered_set<std::string>    m_supportedExtensions;  // Extensions to load
  bool                               m_validSceneParsed = false;
  LoadProgress*                      m_loadProgress     = nullptr;  // see setLoadProgress()

  // glTF 2.1 external assets resolved into m_model (provenance). Read-only membership is derived
  // from the per-node extras marker (kExternalAssetContentKey), not tracked by index here.
  std::vector<nvvkgltf::ReferencedAsset> m_referencedAssets;
  std::vector<InstanceSet>               m_instanceSets;  // see getInstanceSet(); kept across parseScene()
  // Frees the JSON of the files placed by the last load: as long as placing them on large scenes, so
  // done off the load path. It owns what it frees; joined before the next load and on destruction.
  std::thread m_externalAssetRelease;

  //--------------------------------------------------------------------------------------------------
  // Data Members: Render Data (Built from Model)
  //--------------------------------------------------------------------------------------------------

  RenderNodeRegistry                     m_renderNodeRegistry;   // Centralized renderNode mappings
  std::vector<nvvkgltf::RenderPrimitive> m_renderPrimitives;     // Unique primitives
  std::vector<glm::vec3>                 m_renderPrimCenterObj;  // Object-space AABB centroid per render primitive
  std::vector<nvvkgltf::RenderCamera>    m_cameras;              // Cameras
  std::vector<nvvkgltf::RenderLight>     m_lights;               // Lights
  int                                    m_skySunLight = -1;     // Index into m_lights of the sky's sun, or -1

  //--------------------------------------------------------------------------------------------------
  // Data Members: Shaded-Nodes Cache
  //--------------------------------------------------------------------------------------------------

  // Four std::vector<uint32_t> indexed by PipelineType (eRasterSolid, eRasterSolidDoubleSided,
  // eRasterBlend, eRasterAll). Reconciled lazily by getShadedNodes() / hasTransmissionMaterials().
  static constexpr int          kPipelineTypeCount = 4;
  mutable std::vector<uint32_t> m_shadedNodesCache[kPipelineTypeCount];
  mutable std::vector<uint8_t>  m_materialBucketKey;  // Parallel to m_model.materials
  mutable bool                  m_shadedCacheValid     = false;
  mutable bool                  m_hasTransmissionCache = false;
  mutable uint64_t              m_shadedNodesRevision  = 0;  // Bumped on every rebuild, see getShadedNodesRevision()
  // Observed m_sceneGraphRevision at the point the cache was built. Re-observed in
  // reconcileShadedNodesCache() to catch materialID reassignments (setCurrentVariant,
  // SceneEditor::setPrimitiveMaterial) without reacting to animation world-matrix dirties
  // -- those do not bump m_sceneGraphRevision.
  mutable uint64_t m_shadedCacheSceneGraphRevision = 0;

  // The index behind isMeshReadOnly(), getMeshNodes() and the other usage queries, rebuilt by
  // elementUsage() when m_sceneGraphRevision or an element count changes: parseScene, reparenting
  // and material assignment bump the revision; adding or removing an element changes a count. An
  // edit that changes only an attachment the index records (a camera) calls invalidateElementUsage()
  // instead of bumping the revision, which would also refresh the GPU transform buffers.
  struct ElementUsage
  {
    uint64_t              revision = ~uint64_t(0);
    std::array<size_t, 5> counts{};  // nodes, meshes, materials, cameras, lights it was built for
    std::vector<uint8_t>  readOnlyMeshes;
    std::vector<uint8_t>  readOnlyMaterials;
    std::vector<int>      meshNodesOffset, meshNodes;      // mesh m: meshNodes[offset[m], offset[m + 1])
    std::vector<int>      meshPlacingOffset, meshPlacing;  // same layout, placing nodes of instance sets
    std::vector<int>      materialPrimitives;
    std::vector<int>      cameraNode, lightNode;
  };
  const ElementUsage&  elementUsage() const;
  mutable ElementUsage m_elementUsage;


  //--------------------------------------------------------------------------------------------------
  // Data Members: Node Transforms (Cached)
  //--------------------------------------------------------------------------------------------------

  std::vector<glm::mat4> m_nodesLocalMatrices;  // Per-node local transforms
  std::vector<glm::mat4> m_nodesWorldMatrices;  // Per-node world transforms
  std::vector<int>       m_nodeParents;         // Parent index for each node

  // Topological levels for parallel world-matrix propagation.
  // Built by buildTopologicalLevels() at scene load and on re-parent.
  struct TopoLevels
  {
    std::vector<int>                 nodeOrder;  // Node indices sorted by tree depth
    std::vector<std::pair<int, int>> levels;     // Per-level (offset, count) into nodeOrder
  };
  TopoLevels m_topoLevels;
  uint64_t   m_sceneGraphRevision    = 0;
  uint64_t   m_nodeTransformRevision = 0;  // see getNodeTransformRevision()
  void       buildTopologicalLevels();
  void       updateWorldMatricesSerial();    // Filtered-root recursive walk (small dirty sets)
  void       updateWorldMatricesParallel();  // Level-by-level parallel (large dirty sets)

  std::vector<std::vector<int>> m_meshRenderPrimitives;  // per mesh primitive, its render primitive or -1 (buildPrimitiveKeyMap)
  // Per node that skins or morphs a mesh other nodes draw too: its own render primitive per mesh
  // primitive, used in place of m_meshRenderPrimitives (assignDeformedRenderPrimitives).
  std::unordered_map<int, std::vector<int>> m_nodeRenderPrimitives;

  //--------------------------------------------------------------------------------------------------
  // Data Members: Material Variants
  //--------------------------------------------------------------------------------------------------

  std::vector<std::string> m_variants;  // KHR_materials_variants
  int                      m_currentVariant = 0;

  //--------------------------------------------------------------------------------------------------
  // Data Members: KHR_interactivity
  //--------------------------------------------------------------------------------------------------

  std::vector<InteractivityGraph> m_interactivityGraphs;
  std::vector<std::unique_ptr<InteractivityGraphInstance>> m_interactivityInstances;  // index-aligned with m_interactivityGraphs
  int m_defaultInteractivityGraph = 0;
  // Attached to every InteractivityGraphInstance in m_interactivityInstances (see .cpp) so
  // pointer/get and pointer/set can read/write this Scene's model. Concrete type is
  // ScenePointerResolver (gltf_interactivity_scene_pointer.hpp) - only .cpp needs that header.
  std::unique_ptr<InteractivityPointerResolver> m_interactivityPointerResolver;
  // Same idea, for animation/start's validity/duration queries. Concrete type is
  // SceneAnimationResolver (gltf_interactivity_scene_animation.hpp) - only .cpp needs that header.
  std::unique_ptr<InteractivityAnimationResolver> m_interactivityAnimationResolver;

  //--------------------------------------------------------------------------------------------------
  // Data Members: Scene State
  //--------------------------------------------------------------------------------------------------

  int                   m_currentScene    = 0;
  int                   m_sceneCameraNode = -1;
  uint64_t              m_numTriangles    = 0;  // 64-bit: instanced scenes exceed 2^32
  mutable nvutils::Bbox m_sceneBounds;

  //--------------------------------------------------------------------------------------------------
  // Data Members: Dirty Tracking for GPU Updates
  //--------------------------------------------------------------------------------------------------

  DirtyFlags m_dirtyFlags;

  // Nodes the GPU transform path updated on-device only; their CPU world matrices need reconciliation on
  // the next CPU sync. See addGpuStaleNodes() / mergeGpuStaleNodesIntoDirty().
  std::unordered_set<int> m_gpuStaleNodes;

  // Set by mergeScene when imported file contributes new animations (first new clip index). Cleared by takeMergePreferredAnimationIndex.
  int m_pendingMergePreferredAnimationIndex = -1;

  std::unique_ptr<SceneEditor>             m_editor;
  mutable std::unique_ptr<AnimationSystem> m_animation;
  mutable std::unique_ptr<SceneValidator>  m_validator;
};

}  // namespace nvvkgltf
