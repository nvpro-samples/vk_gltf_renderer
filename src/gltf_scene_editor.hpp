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

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>
#include <tinygltf/tiny_gltf.h>

#include "gltf_scene.hpp"

namespace nvvkgltf {

// Snapshot of model state needed to reliably undo structural operations.
// Captures all model vectors modified by deleteNode, addLightNode, etc.
struct SceneGraphSnapshot
{
  std::vector<tinygltf::Node>      nodes;
  std::vector<int>                 sceneRoots;
  std::vector<tinygltf::Animation> animations;
  std::vector<tinygltf::Skin>      skins;
  std::vector<tinygltf::Light>     lights;
};

// Procedural primitive kinds that can be appended to a live scene.
//
// To add a new kind (e.g. cone, torus):
//   1. add an enum value here,
//   2. add a row to kPrimitiveKinds below (drives every UI menu and the display name),
//   3. add a case to addPrimitiveMesh() that generates the geometry.
// No UI code needs to change - all menus iterate kPrimitiveKinds.
enum class PrimitiveKind
{
  ePlane,
  eCube,
  eSphere,
};

// Parameters for a procedural primitive. Interpretation depends on the kind:
//   ePlane  : size = width & depth, subdivU = steps
//   eCube   : size = edge length
//   eSphere : size = diameter, subdivU = sectors, subdivV = stacks
struct PrimitiveParams
{
  float size    = 1.0f;
  int   subdivU = 20;
  int   subdivV = 20;
};

// Single source of truth for the available primitives: pairs a kind with its display name (also used
// to name the generated mesh/material/node). UI menus iterate this; primitiveKindName() looks it up.
struct PrimitiveKindInfo
{
  PrimitiveKind kind;
  const char*   name;
};
inline constexpr std::array<PrimitiveKindInfo, 3> kPrimitiveKinds{{
    {PrimitiveKind::ePlane, "Plane"},
    {PrimitiveKind::eCube, "Cube"},
    {PrimitiveKind::eSphere, "Sphere"},
}};

// Human-readable name for a primitive kind (falls back to "Primitive" for an unknown value).
const char* primitiveKindName(PrimitiveKind kind);

// Single source of truth for the punctual light kinds (KHR_lights_punctual) that can be added to a
// live scene: pairs the glTF light.type string with a display name (also used to name the node/light).
// UI menus iterate this so the light list is defined in exactly one place, mirroring kPrimitiveKinds.
struct LightKindInfo
{
  const char* type;  // glTF light.type: "point" | "directional" | "spot"
  const char* name;  // display name and default node/light name
};
inline constexpr std::array<LightKindInfo, 3> kLightKinds{{
    {"point", "Point Light"},
    {"directional", "Directional Light"},
    {"spot", "Spot Light"},
}};

// Pre-append sizes of the model vectors touched by addPrimitiveMesh(). Used by the undo command
// to truncate the appended tail back to its original state (append is purely additive at the tail).
struct ModelTailSizes
{
  size_t meshes      = 0;
  size_t materials   = 0;
  size_t accessors   = 0;
  size_t bufferViews = 0;
  size_t buffers     = 0;
};

/*-------------------------------------------------------------------------------------------------
# class nvvkgltf::SceneEditor

>  Handles all mutation and read/edit access for a Scene: node lifecycle, hierarchy, transforms,
   resource attachment, material ops, visibility, and node/model query. Friend of Scene.
   Access via scene.editor().

 -------------------------------------------------------------------------------------------------*/
class SceneEditor
{
public:
  explicit SceneEditor(Scene& scene);

  // ---------- Node validation and access ----------
  [[nodiscard]] bool                  isValidNodeIndex(int nodeIndex) const;
  [[nodiscard]] const tinygltf::Node& getNode(int nodeIndex) const;
  tinygltf::Node&                     getNodeForEdit(int nodeIndex);
  [[nodiscard]] std::string           getNodeName(int nodeIndex) const;
  void                                renameNode(int nodeIndex, const std::string& name);
  [[nodiscard]] size_t                countDescendants(int nodeIndex) const;
  [[nodiscard]] const glm::mat4&      getNodeWorldMatrix(int nodeIndex) const;
  [[nodiscard]] int                   getNodeParent(int nodeIndex) const;

  // ---------- Mesh validation and access ----------
  [[nodiscard]] bool                  isValidMeshIndex(int meshIndex) const;
  [[nodiscard]] const tinygltf::Mesh& getMesh(int meshIndex) const;

  // ---------- Model edit access (caller should mark dirty) ----------
  tinygltf::Material& getMaterialForEdit(int materialIndex);
  tinygltf::Light&    getLightForEdit(int lightIndex);

  // ---------- Node resource queries ----------
  [[nodiscard]] std::optional<int> getNodeMesh(int nodeIndex) const;
  [[nodiscard]] std::optional<int> getNodeCamera(int nodeIndex) const;
  [[nodiscard]] std::optional<int> getNodeSkin(int nodeIndex) const;

  // ---------- Node lifecycle ----------
  [[nodiscard]] int addNode(const std::string& name = "", int parentIndex = -1);
  [[nodiscard]] int addLightNode(const std::string& lightType, const std::string& name, int parentIndex = -1);
  [[nodiscard]] int duplicateNode(int originalIndex, bool reparse = true);
  void              deleteNode(int nodeIndex);

  // ---------- Procedural primitives ----------
  // Appends a plane/cube/sphere as new glTF geometry (buffer + bufferViews + accessors + material +
  // mesh) plus a node (child of parentIndex, or a scene root when parentIndex < 0). Returns the new
  // node index, or -1 on failure. Calls parseScene(); the new geometry is picked up on the next GPU sync.
  [[nodiscard]] int addPrimitiveMesh(PrimitiveKind kind, const PrimitiveParams& params, int parentIndex = -1);

  // Truncate the appended tail of the model vectors back to the given sizes (undo of addPrimitiveMesh).
  // Safe because addPrimitiveMesh only appends at the tail. Does not call parseScene(); undo truncates
  // first, then restoreFromSnapshot() reparses once against the final model.
  void truncateGeometryTail(const ModelTailSizes& sizes);

  // ---------- Hierarchy ----------
  void               setNodeParent(int childIndex, int newParentIndex);
  [[nodiscard]] bool wouldCreateCycle(int childIndex, int newParentIndex) const;

  // ---------- External assets (glTF 2.1) ----------
  // "Break the lock": make a referenced external asset editable. Removes the read-only marker from
  // every merged node and drops the `externalAsset` link on the instance node(s). Because instances
  // of the same source share geometry/materials, this necessarily applies to ALL instances of that
  // asset at once (nodeIndex may be an instance node or any read-only node within it). The unlocked
  // content is then saved inline. Returns true if anything was unlocked.
  bool makeExternalAssetEditable(int nodeIndex);

  // ---------- Transforms ----------
  void setNodeTRS(int nodeIndex, const glm::vec3& translation, const glm::quat& rotation, const glm::vec3& scale);

  // ---------- Resource attachment ----------
  void setNodeMesh(int nodeIndex, int meshIndex);
  void setNodeCamera(int nodeIndex, int cameraIndex);
  void setNodeSkin(int nodeIndex, int skinIndex);
  void clearNodeMesh(int nodeIndex);
  void clearNodeCamera(int nodeIndex);
  void clearNodeSkin(int nodeIndex);

  // ---------- Material ops ----------
  void              setPrimitiveMaterial(int meshIndex, int primIndex, int newMaterialID);
  [[nodiscard]] int duplicateMaterial(int originalIndex);
  [[nodiscard]] int duplicateMeshForNode(int meshIndex, int nodeIndex);
  [[nodiscard]] int splitPrimitiveMaterial(int nodeIndex, int primIndex);
  [[nodiscard]] int mergePrimitiveMaterial(int nodeIndex);

  // ---------- Visibility ----------
  void updateVisibility(int nodeIndex);

  // ---------- Snapshot / Restore (for undo of structural operations) ----------
  [[nodiscard]] SceneGraphSnapshot snapshotForDelete() const;
  void                             restoreFromSnapshot(const SceneGraphSnapshot& snapshot);

private:
  Scene& m_scene;

  // Referenced external-asset subtrees (glTF 2.1) are read-only. Returns true (and logs) when an
  // edit on nodeIndex must be blocked; call sites early-out on true.
  [[nodiscard]] bool blockIfNodeReadOnly(int nodeIndex, const char* op) const;

  // Produce a unique node name from a base, using a " (N)" suffix (stripping any existing one).
  [[nodiscard]] std::string makeUniqueNodeName(const std::string& baseName) const;
  int  duplicateNodeRecursive(int originalIndex, int newParentIndex, std::unordered_map<int, int>& nodeMap);
  void collectDescendantIndices(int nodeIndex, std::vector<int>& indices) const;
  void deleteNodeRecursive(int nodeIndex);
  void deleteNodeSingle(int nodeIndex);
  void removeNodeFromParent(int nodeIndex);
  void removeNodeFromSceneRoots(int nodeIndex);
  void remapIndicesAfterNodeDeletion(int deletedIndex);
  int  findEquivalentMesh(int meshIndex) const;
};

}  // namespace nvvkgltf
