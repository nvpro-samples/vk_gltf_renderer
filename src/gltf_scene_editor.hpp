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

  // ---------- Hierarchy ----------
  void               setNodeParent(int childIndex, int newParentIndex);
  [[nodiscard]] bool wouldCreateCycle(int childIndex, int newParentIndex) const;

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
