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
#include <cassert>
#include <functional>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>
#include <tinygltf/tiny_gltf.h>
#include <nvutils/bounding_box.hpp>

#include "tinygltf_utils.hpp"
#include "gltf_animation_pointer.hpp"


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
};

// The RenderPrimitive is a unique primitive in the scene
struct RenderPrimitive
{
  tinygltf::Primitive* pPrimitive  = nullptr;
  int                  vertexCount = 0;
  int                  indexCount  = 0;
  int                  meshID      = 0;
};

struct RenderCamera
{
  enum CameraType
  {
    ePerspective,
    eOrthographic
  };

  CameraType type   = ePerspective;
  glm::vec3  eye    = {0.0f, 0.0f, 0.0f};
  glm::vec3  center = {0.0f, 0.0f, 0.0f};
  glm::vec3  up     = {0.0f, 1.0f, 0.0f};

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
};

// Centralized registry for renderNode mappings (nodeID/primID <-> renderNodeID).
// Provides O(1) bidirectional lookups and sparse storage for nodes with meshes.
class RenderNodeRegistry
{
public:
  // Add a render node; returns the new renderNodeID.
  int addRenderNode(const RenderNode& node, int nodeID, int primIndex);

  // Lookups
  int getRenderNodeID(int nodeID, int primIndex) const;  // (nodeID, primID) -> RenderNodeID, -1 if not found
  std::optional<std::pair<int, int>> getNodeAndPrim(int renderNodeID) const;   // RenderNodeID -> (nodeID, primID)
  const std::vector<int>&            getRenderNodesForNode(int nodeID) const;  // nodeID -> [RenderNodeIDs]

  // Batch: collect all renderNodeIDs for node and its descendants. getChildren(nodeID) returns child node indices.
  void getAllRenderNodesForNodeRecursive(int nodeID, std::function<std::vector<int>(int)> getChildren, std::vector<int>& outRenderNodeIDs) const;

  // Clear all mappings and the flat array.
  void clear();

  // Direct access to flat array (for GPU upload).
  const std::vector<RenderNode>& getRenderNodes() const { return m_renderNodes; }
  std::vector<RenderNode>&       getRenderNodes() { return m_renderNodes; }

private:
  std::vector<RenderNode> m_renderNodes;

  // Forward: (nodeID, primIndex) -> renderNodeID
  std::unordered_map<uint64_t, int> m_nodeAndPrimToRenderNode;

  // Reverse: renderNodeID -> (nodeID, primIndex)
  std::vector<std::pair<int, int>> m_renderNodeToNodeAndPrim;

  // Grouped by node: nodeID -> [renderNodeIDs] (sparse, only nodes with meshes)
  std::unordered_map<int, std::vector<int>> m_nodeToRenderNodes;

  // Empty vector returned when node has no render nodes
  static const std::vector<int> s_emptyRenderNodes;

  static uint64_t makeKey(int nodeID, int primIndex)
  {
    return (static_cast<uint64_t>(static_cast<uint32_t>(nodeID)) << 32) | static_cast<uint32_t>(primIndex);
  }
};

// Animation data
struct AnimationInfo
{
  std::string name;
  float       start       = std::numeric_limits<float>::max();
  float       end         = std::numeric_limits<float>::min();
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
  [[nodiscard]] bool save(const std::filesystem::path& filename);  // Save .gltf or .glb
  // Merge another glTF into this scene. Optional maxTextureCount validates combined texture limit (e.g. GPU descriptor limit).
  [[nodiscard]] int mergeScene(const std::filesystem::path& filename, std::optional<uint32_t> maxTextureCount = std::nullopt);  // Returns wrapper node index, or -1 on failure
  // After mergeScene(), returns the animation list index of the first clip from the merged file (for UI default); -1 if none.
  // Consumed once by the renderer so the animation dropdown selects merged motion instead of staying on a base-scene clip.
  [[nodiscard]] int                      takeMergePreferredAnimationIndex();
  void                                   takeModel(tinygltf::Model&& model);  // Use pre-loaded model
  std::unordered_set<std::string>&       supportedExtensions() { return m_supportedExtensions; }
  const std::unordered_set<std::string>& supportedExtensions() const { return m_supportedExtensions; }
  const std::filesystem::path&           getFilename() const { return m_filename; }
  // Order: base scene directory first, then each imported scene's directory in merge order.
  // Used to resolve image URIs via findFile and when saving/copying images.
  const std::vector<std::filesystem::path>& getImageSearchPaths() const { return m_imageSearchPaths; }

  //--------------------------------------------------------------------------------------------------
  // Model Access
  //--------------------------------------------------------------------------------------------------

  const tinygltf::Model& getModel() const { return m_model; }
  tinygltf::Model&       getModel() { return m_model; }
  [[nodiscard]] bool     valid() const { return m_validSceneParsed; }

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

  // KHR_mesh_gpu_instancing: per-node instance locals (used by GPU transform path).
  [[nodiscard]] const std::unordered_map<int, std::vector<glm::mat4>>& getGpuInstanceLocalMatrices() const
  {
    return m_gpuInstanceLocalMatrices;
  }

  //--------------------------------------------------------------------------------------------------
  // Variant Management
  //--------------------------------------------------------------------------------------------------

  void                            setCurrentVariant(int variant);  // Marks dirty RenderNodes internally
  const std::vector<std::string>& getVariants() const { return m_variants; }
  [[nodiscard]] int               getCurrentVariant() const { return m_currentVariant; }
  [[nodiscard]] std::unordered_set<int> getMaterialRenderNodes(const std::unordered_set<int>& materialVariantNodeIDs) const;

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
  void              updateRenderNodeDirtyFromNodes(bool includeDescendants = true);
  [[nodiscard]] int getRenderNodeForPrimitive(int nodeIndex, int primitiveIndex) const;
  [[nodiscard]] int getPrimitiveIndexForRenderNode(int renderNodeIndex) const;

  //--------------------------------------------------------------------------------------------------
  // Render Primitive Management
  //--------------------------------------------------------------------------------------------------

  const std::vector<nvvkgltf::RenderPrimitive>& getRenderPrimitives() const { return m_renderPrimitives; }
  const nvvkgltf::RenderPrimitive&              getRenderPrimitive(size_t ID) const { return m_renderPrimitives[ID]; }
  [[nodiscard]] size_t                          getNumRenderPrimitives() const { return m_renderPrimitives.size(); }

  //--------------------------------------------------------------------------------------------------
  // Shading & Statistics
  //--------------------------------------------------------------------------------------------------

  std::vector<uint32_t> getShadedNodes(PipelineType type) const;
  [[nodiscard]] int     getNumTriangles() const { return m_numTriangles; }
  // Returns cached bounds; lazily computes on first call (mutable cache, logically const).
  [[nodiscard]] nvutils::Bbox getSceneBounds() const;

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
    std::unordered_set<int> renderNodesVk;                       // RenderNode indices for SceneVk
    std::unordered_set<int> renderNodesRtx;                      // RenderNode indices for SceneRTX
    std::unordered_set<int> materials;                           // Material indices
    std::unordered_set<int> lights;                              // Light indices (glTF light array)
    std::unordered_set<int> nodes;                               // Node indices (for transform updates)
    bool                    allRenderNodesDirty        = false;  // Full RN upload (count change or massive reorder)
    bool                    primitivesChanged          = false;  // BLAS rebuild needed (primitive set changed)
    bool                    tlasVisibilityNeedsCpuSync = false;  // KHR_node_visibility: SceneEditor::updateVisibility

    void clear()
    {
      renderNodesVk.clear();
      renderNodesRtx.clear();
      materials.clear();
      lights.clear();
      nodes.clear();
      allRenderNodesDirty        = false;
      primitivesChanged          = false;
      tlasVisibilityNeedsCpuSync = false;
    }

    [[nodiscard]] bool isEmpty() const
    {
      return renderNodesVk.empty() && renderNodesRtx.empty() && materials.empty() && lights.empty() && nodes.empty()
             && !allRenderNodesDirty && !primitivesChanged && !tlasVisibilityNeedsCpuSync;
    }
  };

  const DirtyFlags& getDirtyFlags() const { return m_dirtyFlags; }
  DirtyFlags&       getDirtyFlags() { return m_dirtyFlags; }
  void              clearDirtyFlags() { m_dirtyFlags.clear(); }

  void markMaterialDirty(int materialIndex);
  void markLightDirty(int lightIndex);
  void markRenderNodeDirty(int renderNodeIndex, bool forVk = true, bool forRtx = true);
  void markNodeDirty(int nodeIndex);
  void markRenderNodeRtxDirtyForMaterials(const std::unordered_set<int>& materialIds);  // For TLAS instance flags

private:
  friend class SceneEditor;

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
  void clearParsedData();
  void parseVariants();
  void setSceneElementsDefaultNames();
  void createSceneCamera();

  //--------------------------------------------------------------------------------------------------
  // Private Methods: Render Node Helpers
  //--------------------------------------------------------------------------------------------------

  using PrimitiveKeyMap = std::map<std::string, int>;

  PrimitiveKeyMap buildPrimitiveKeyMap();
  int             getMaterialVariantIndex(const tinygltf::Primitive& primitive, int currentVariant);
  void createRenderNodesForNode(int nodeID, const glm::mat4& worldMatrix, bool visible, const PrimitiveKeyMap& primMap);
  bool handleRenderNode(int nodeID, glm::mat4 worldMatrix, const PrimitiveKeyMap& primMap);
  size_t handleGpuInstancing(const tinygltf::Value& attributes, nvvkgltf::RenderNode renderNode, glm::mat4 worldMatrix, int nodeID, int primIndex);
  bool handleCameraTraversal(int nodeID, const glm::mat4& worldMatrix);
  bool handleLightTraversal(int nodeID, const glm::mat4& worldMatrix);
  void updateRenderNodesFull();

  // Lightweight rebuild after structural change (e.g. node deletion). Only clears and repopulates
  // render nodes and lights; does not touch primitives, cameras, animations, variants.
  void rebuildRenderNodesAndLights();

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
  // Data Members: Core glTF Model
  //--------------------------------------------------------------------------------------------------

  tinygltf::Model                    m_model;             // The glTF model (source of truth)
  std::filesystem::path              m_filename;          // Loaded file path
  std::vector<std::filesystem::path> m_imageSearchPaths;  // Base dirs for image resolution (base first, then imports)
  std::unordered_set<std::string>    m_supportedExtensions;  // Extensions to load
  bool                               m_validSceneParsed = false;

  //--------------------------------------------------------------------------------------------------
  // Data Members: Render Data (Built from Model)
  //--------------------------------------------------------------------------------------------------

  RenderNodeRegistry                     m_renderNodeRegistry;  // Centralized renderNode mappings
  std::vector<nvvkgltf::RenderPrimitive> m_renderPrimitives;    // Unique primitives
  std::vector<nvvkgltf::RenderCamera>    m_cameras;             // Cameras
  std::vector<nvvkgltf::RenderLight>     m_lights;              // Lights


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
  uint64_t   m_sceneGraphRevision = 0;
  void       buildTopologicalLevels();
  void       updateWorldMatricesSerial();    // Filtered-root recursive walk (small dirty sets)
  void       updateWorldMatricesParallel();  // Level-by-level parallel (large dirty sets)

  std::unordered_map<int, std::vector<glm::mat4>> m_gpuInstanceLocalMatrices;  // nodeID -> per-instance local transforms (EXT_mesh_gpu_instancing)

  //--------------------------------------------------------------------------------------------------
  // Data Members: Material Variants
  //--------------------------------------------------------------------------------------------------

  std::vector<std::string> m_variants;  // KHR_materials_variants
  int                      m_currentVariant = 0;

  //--------------------------------------------------------------------------------------------------
  // Data Members: Scene State
  //--------------------------------------------------------------------------------------------------

  int                   m_currentScene    = 0;
  int                   m_sceneCameraNode = -1;
  int                   m_numTriangles    = 0;
  mutable nvutils::Bbox m_sceneBounds;

  //--------------------------------------------------------------------------------------------------
  // Data Members: Dirty Tracking for GPU Updates
  //--------------------------------------------------------------------------------------------------

  DirtyFlags m_dirtyFlags;

  // Set by mergeScene when imported file contributes new animations (first new clip index). Cleared by takeMergePreferredAnimationIndex.
  int m_pendingMergePreferredAnimationIndex = -1;

  std::unique_ptr<SceneEditor>             m_editor;
  mutable std::unique_ptr<AnimationSystem> m_animation;
  mutable std::unique_ptr<SceneValidator>  m_validator;
};

}  // namespace nvvkgltf
