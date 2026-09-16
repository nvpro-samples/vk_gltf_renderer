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
#include <filesystem>
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
// NOTE: whenever deleteNode() starts mutating a new tinygltf::Model member, add it here too --
// otherwise undo silently drops the change. `cameras` and `lights` are here because deleteNode
// prunes orphan entries in both arrays (see SceneEditor::pruneOrphanLights/Cameras).
struct SceneGraphSnapshot
{
  std::vector<tinygltf::Node>      nodes;
  std::vector<int>                 sceneRoots;
  std::vector<tinygltf::Animation> animations;
  std::vector<tinygltf::Skin>      skins;
  std::vector<tinygltf::Light>     lights;
  std::vector<tinygltf::Camera>    cameras;
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

  // ---------- Texture / image import ----------
  // Import an image file (PNG/JPEG/KTX2/DDS/...) as a new texture appended at the tail of the model.
  // The file is validated/decoded up front; on failure returns -1 and fills *error (if given) without
  // mutating the model. On success returns the new texture index (== model.textures.size()-1) and:
  //   * appends one tinygltf::Image (referenced by absolute URI - Scene::save relocates/copies it) and
  //     one tinygltf::Texture pointing at it,
  //   * registers the file's directory as an image search path so the URI resolves,
  //   * sets DirtyFlags::texturesChanged so the next reconcile triggers a full GPU texture rebuild.
  // Color space (sRGB vs linear) is resolved on that rebuild from the material slot the texture is
  // assigned to, so callers should assign the returned index to a slot before the rebuild runs.
  [[nodiscard]] int importImageAsTexture(const std::filesystem::path& path, std::string* error = nullptr);

  // Replace the pixels of an existing image (by index) with those of a file, in place: every texture
  // and material keeps pointing at this image index. Validated up front; false + *error on failure.
  // Sets DirtyFlags::texturesChanged.
  [[nodiscard]] bool replaceImageFromFile(int imageIndex, const std::filesystem::path& path, std::string* error = nullptr);

  // Per-image count of textures whose (core or extension) source resolves to it (size == images.size()).
  // A 0 entry means the image is referenced by nothing and is safe to remove with removeImageAt().
  [[nodiscard]] std::vector<int> computeImageRefCounts() const;

  // Number of textures whose (core or extension) source resolves to imageIndex. 0 means the image is
  // referenced by nothing and is safe to remove with removeImageAt().
  [[nodiscard]] int countTextureRefsToImage(int imageIndex) const;

  // Remove an image no texture references (asserts the precondition), shifting higher image indices
  // down and remapping every texture source accordingly. Sets DirtyFlags::texturesChanged. Pair with
  // insertImageAt() for undo.
  void removeImageAt(int imageIndex);

  // Inverse of removeImageAt(): shift image indices at/after imageIndex up and insert `image` there,
  // remapping texture sources. Sets DirtyFlags::texturesChanged.
  void insertImageAt(int imageIndex, const tinygltf::Image& image);

  // ---------- Material ops ----------
  void setPrimitiveMaterial(int meshIndex, int primIndex, int newMaterialID);

  // Per-material count of references: both tinygltf::Primitive::material and any KHR_materials_variants
  // mapping targeting the material (size == materials.size()); a 0 entry means the material is referenced
  // by neither and is safe to remove with removeMaterialAt().
  [[nodiscard]] std::vector<int> computeMaterialRefCounts() const;

  // Insert `material` at `index` (clamped to [0, materials.size()]), shifting higher material indices up
  // and remapping every primitive's material reference -- and any KHR_materials_variants mapping -- that
  // pointed at or above the insertion point. Returns the effective (post-clamp) index the material was
  // inserted at; a caller that stores `index` for undo must use this return value instead, since a
  // requested index can be clamped. Inverse of removeMaterialAt(); pair them for undo. A tail insert
  // (index == materials.size()) is a cheap append (no reference remap / reparse).
  [[nodiscard]] int insertMaterialAt(int index, const tinygltf::Material& material);

  // Remove the material at `index`, shifting higher indices down and remapping primitive references and
  // KHR_materials_variants mappings accordingly. The caller must ensure the material is unreferenced
  // (computeMaterialRefCounts()[index] == 0); a tail removal is cheap (no remap / reparse). Returns false
  // (no mutation) when `index` is out of range, so a caller can tell whether the remove actually happened.
  // Pair with insertMaterialAt() for undo.
  bool              removeMaterialAt(int index);
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

  // Register an image file's directory as an image search path so its (absolute) URI resolves at load.
  void registerImageSearchDir(const std::filesystem::path& path);

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
  // After node deletion(s), sweep model.lights / model.cameras for entries no longer referenced by
  // any surviving node.light / node.camera; erase them and remap the surviving references so indices
  // stay contiguous. Prevents orphan lights from continuing to light the scene (via stale entries in
  // model.lights) and keeps saved glTF free of dangling definitions.
  void pruneOrphanLights();
  void pruneOrphanCameras();
  int  findEquivalentMesh(int meshIndex) const;
};

}  // namespace nvvkgltf
