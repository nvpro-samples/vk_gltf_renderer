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
// SceneEditor: high-level editing operations on a glTF scene graph.
// Provides node manipulation (reparent, duplicate, delete), mesh and
// material assignment, transform editing, and undo-safe model mutations.
// All operations keep the underlying tinygltf::Model consistent.
//

#include <algorithm>
#include <functional>
#include <limits>

#include <fmt/format.h>

#include <nvutils/logger.hpp>
#include <nvutils/primitives.hpp>

#include "gltf_scene_editor.hpp"
#include "gltf_scene_animation.hpp"
#include "gltf_compact_model.hpp"
#include "tinygltf_utils.hpp"

namespace nvvkgltf {

//--------------------------------------------------------------------------------------------------
// Constructor
//--------------------------------------------------------------------------------------------------

SceneEditor::SceneEditor(Scene& scene)
    : m_scene(scene)
{
}

//--------------------------------------------------------------------------------------------------
// Node validation and access
//--------------------------------------------------------------------------------------------------

bool SceneEditor::isValidNodeIndex(int nodeIndex) const
{
  return nodeIndex >= 0 && nodeIndex < static_cast<int>(m_scene.m_model.nodes.size());
}

const tinygltf::Node& SceneEditor::getNode(int nodeIndex) const
{
  if(!isValidNodeIndex(nodeIndex))
  {
    static tinygltf::Node invalidNode;
    LOGW("Invalid node index: %d\n", nodeIndex);
    return invalidNode;
  }
  return m_scene.m_model.nodes[nodeIndex];
}

tinygltf::Node& SceneEditor::getNodeForEdit(int nodeIndex)
{
  assert(isValidNodeIndex(nodeIndex) && "Invalid node index in getNodeForEdit");
  return m_scene.m_model.nodes[nodeIndex];
}

bool SceneEditor::isValidMeshIndex(int meshIndex) const
{
  return meshIndex >= 0 && meshIndex < static_cast<int>(m_scene.m_model.meshes.size());
}

const tinygltf::Mesh& SceneEditor::getMesh(int meshIndex) const
{
  if(!isValidMeshIndex(meshIndex))
  {
    static tinygltf::Mesh invalidMesh;
    LOGW("Invalid mesh index: %d\n", meshIndex);
    return invalidMesh;
  }
  return m_scene.m_model.meshes[meshIndex];
}

std::string SceneEditor::getNodeName(int nodeIndex) const
{
  if(!isValidNodeIndex(nodeIndex))
    return "";
  return getNode(nodeIndex).name;
}

bool SceneEditor::blockIfNodeReadOnly(int nodeIndex, const char* op) const
{
  if(m_scene.isNodeReadOnly(nodeIndex))
  {
    LOGW("Cannot %s node %d: it belongs to a referenced external asset (read-only)\n", op, nodeIndex);
    return true;
  }
  return false;
}

void SceneEditor::renameNode(int nodeIndex, const std::string& name)
{
  if(!isValidNodeIndex(nodeIndex))
  {
    LOGW("Cannot rename invalid node\n");
    return;
  }
  if(blockIfNodeReadOnly(nodeIndex, "rename"))
    return;
  getNodeForEdit(nodeIndex).name = name;
}

size_t SceneEditor::countDescendants(int nodeIndex) const
{
  if(!isValidNodeIndex(nodeIndex))
    return 0;
  size_t      count = 0;
  const auto& node  = getNode(nodeIndex);
  for(int childIdx : node.children)
  {
    if(isValidNodeIndex(childIdx))
      count += 1 + countDescendants(childIdx);
  }
  return count;
}

const glm::mat4& SceneEditor::getNodeWorldMatrix(int nodeIndex) const
{
  assert(isValidNodeIndex(nodeIndex) && "Invalid node index in getNodeWorldMatrix");
  return m_scene.m_nodesWorldMatrices[nodeIndex];
}

int SceneEditor::getNodeParent(int nodeIndex) const
{
  if(nodeIndex < 0 || nodeIndex >= static_cast<int>(m_scene.m_nodeParents.size()))
    return -1;
  return m_scene.m_nodeParents[nodeIndex];
}

tinygltf::Material& SceneEditor::getMaterialForEdit(int materialIndex)
{
  assert(materialIndex >= 0 && materialIndex < static_cast<int>(m_scene.m_model.materials.size())
         && "Invalid material index in getMaterialForEdit");
  return m_scene.m_model.materials[materialIndex];
}

tinygltf::Light& SceneEditor::getLightForEdit(int lightIndex)
{
  assert(lightIndex >= 0 && lightIndex < static_cast<int>(m_scene.m_model.lights.size()) && "Invalid light index in getLightForEdit");
  return m_scene.m_model.lights[lightIndex];
}

std::optional<int> SceneEditor::getNodeMesh(int nodeIndex) const
{
  if(!isValidNodeIndex(nodeIndex))
    return std::nullopt;
  const auto& node = getNode(nodeIndex);
  if(node.mesh >= 0 && node.mesh < static_cast<int>(m_scene.m_model.meshes.size()))
    return node.mesh;
  return std::nullopt;
}

std::optional<int> SceneEditor::getNodeCamera(int nodeIndex) const
{
  if(!isValidNodeIndex(nodeIndex))
    return std::nullopt;
  const auto& node = getNode(nodeIndex);
  if(node.camera >= 0 && node.camera < static_cast<int>(m_scene.m_model.cameras.size()))
    return node.camera;
  return std::nullopt;
}

std::optional<int> SceneEditor::getNodeSkin(int nodeIndex) const
{
  if(!isValidNodeIndex(nodeIndex))
    return std::nullopt;
  const auto& node = getNode(nodeIndex);
  if(node.skin >= 0 && node.skin < static_cast<int>(m_scene.m_model.skins.size()))
    return node.skin;
  return std::nullopt;
}

//--------------------------------------------------------------------------------------------------
// Visibility
//
// Pushes KHR_node_visibility (and inherited visibility) into RenderNode::visible and marks each
// affected render node dirty for Vk + RTX. Raster reads visible on the CPU.
//
// GPU transform path: toggling visibility can dirty a huge fraction of render nodes; staging a full
// mapping SSBO would allocate proportional to the scene (OOM risk). We set tlasVisibilityNeedsCpuSync
// so dispatchTransformUpdate skips that upload and runs SceneRtx::syncTopLevelAS once to refresh
// TLAS instance references from CPU (same as the all-CPU path).
//--------------------------------------------------------------------------------------------------

void SceneEditor::updateVisibility(int nodeIndex)
{
  std::function<void(int, bool)> processNode;
  processNode = [&](int curNodeIndex, bool visible) -> void {
    const tinygltf::Node& node = m_scene.m_model.nodes[curNodeIndex];
    if(visible)
    {
      visible = tinygltf::utils::getNodeVisibility(node).visible;
    }
    for(int renderNodeID : m_scene.m_renderNodeRegistry.getRenderNodesForNode(curNodeIndex))
    {
      m_scene.m_renderNodeRegistry.getRenderNodes()[renderNodeID].visible = visible;
      m_scene.markRenderNodeDirty(renderNodeID);
    }

    for(auto& child : node.children)
    {
      processNode(child, visible);
    }
  };

  const tinygltf::Node& node    = m_scene.m_model.nodes[nodeIndex];
  bool                  visible = tinygltf::utils::getNodeVisibility(node).visible;
  processNode(nodeIndex, visible);
  // Root dirty is enough: updateNodeWorldMatrices() propagates through descendants.
  m_scene.markNodeDirty(nodeIndex);

  m_scene.m_dirtyFlags.tlasVisibilityNeedsCpuSync = true;
}

//--------------------------------------------------------------------------------------------------
// Snapshot / Restore (for undo of structural operations)
//--------------------------------------------------------------------------------------------------

SceneGraphSnapshot SceneEditor::snapshotForDelete() const
{
  SceneGraphSnapshot snapshot;
  snapshot.nodes      = m_scene.m_model.nodes;
  snapshot.sceneRoots = m_scene.m_model.scenes[m_scene.m_currentScene].nodes;
  snapshot.animations = m_scene.m_model.animations;
  snapshot.skins      = m_scene.m_model.skins;
  snapshot.lights     = m_scene.m_model.lights;
  return snapshot;
}

void SceneEditor::restoreFromSnapshot(const SceneGraphSnapshot& snapshot)
{
  m_scene.m_model.nodes                                = snapshot.nodes;
  m_scene.m_model.scenes[m_scene.m_currentScene].nodes = snapshot.sceneRoots;
  m_scene.m_model.animations                           = snapshot.animations;
  m_scene.m_model.skins                                = snapshot.skins;
  m_scene.m_model.lights                               = snapshot.lights;
  m_scene.parseScene();
}

//--------------------------------------------------------------------------------------------------
// Transforms
//--------------------------------------------------------------------------------------------------

void SceneEditor::setNodeTRS(int nodeIndex, const glm::vec3& translation, const glm::quat& rotation, const glm::vec3& scale)
{
  if(!isValidNodeIndex(nodeIndex))
  {
    LOGW("Cannot set transformation on invalid node index: %d\n", nodeIndex);
    return;
  }
  if(blockIfNodeReadOnly(nodeIndex, "transform"))
    return;

  tinygltf::Node& node = m_scene.m_model.nodes[nodeIndex];
  tinygltf::utils::setNodeTRS(node, translation, rotation, scale);
  node.matrix.clear();

  m_scene.markNodeDirty(nodeIndex);
}

//--------------------------------------------------------------------------------------------------
// Node lifecycle
//--------------------------------------------------------------------------------------------------

int SceneEditor::duplicateNode(int originalIndex, bool reparse)
{
  if(!isValidNodeIndex(originalIndex))
  {
    LOGW("Cannot duplicate invalid node index: %d\n", originalIndex);
    return -1;
  }
  if(blockIfNodeReadOnly(originalIndex, "duplicate"))
    return -1;

  int originalParent =
      (originalIndex < static_cast<int>(m_scene.m_nodeParents.size())) ? m_scene.m_nodeParents[originalIndex] : -1;

  std::unordered_map<int, int> nodeMap;  // original node index → duplicate node index
  int                          newIdx = duplicateNodeRecursive(originalIndex, originalParent, nodeMap);

  if(newIdx == -1)
    return -1;

  // Give the duplicated root a fresh, unique, human-friendly name ("fox1 (1)", "fox1 (2)", ...).
  m_scene.m_model.nodes[newIdx].name = makeUniqueNodeName(m_scene.m_model.nodes[newIdx].name);

  // Remap skins: create new skin objects with joint indices pointing to duplicated nodes
  for(auto& [origIdx, dupIdx] : nodeMap)
  {
    tinygltf::Node& dupNode = m_scene.m_model.nodes[dupIdx];
    if(dupNode.skin < 0 || dupNode.skin >= static_cast<int>(m_scene.m_model.skins.size()))
      continue;

    const tinygltf::Skin& origSkin = m_scene.m_model.skins[dupNode.skin];
    tinygltf::Skin        newSkin  = origSkin;
    newSkin.name                   = origSkin.name.empty() ? "skin_copy" : origSkin.name + "_copy";

    for(int& jointIdx : newSkin.joints)
    {
      auto it = nodeMap.find(jointIdx);
      if(it != nodeMap.end())
        jointIdx = it->second;
    }

    if(newSkin.skeleton >= 0)
    {
      auto it = nodeMap.find(newSkin.skeleton);
      if(it != nodeMap.end())
        newSkin.skeleton = it->second;
    }

    int newSkinIdx = static_cast<int>(m_scene.m_model.skins.size());
    m_scene.m_model.skins.push_back(std::move(newSkin));
    dupNode.skin = newSkinIdx;
  }

  // Remap animations: add new channels targeting the duplicate nodes (reusing the same samplers)
  for(auto& anim : m_scene.m_model.animations)
  {
    size_t originalChannelCount = anim.channels.size();
    for(size_t ci = 0; ci < originalChannelCount; ci++)
    {
      auto it = nodeMap.find(anim.channels[ci].target_node);
      if(it != nodeMap.end())
      {
        tinygltf::AnimationChannel newChannel = anim.channels[ci];
        newChannel.target_node                = it->second;
        anim.channels.push_back(newChannel);
      }
    }
  }

  if(originalParent == -1)
  {
    auto& currentScene = m_scene.m_model.scenes[m_scene.m_currentScene];
    currentScene.nodes.push_back(newIdx);
    LOGI("  Added duplicated root hierarchy to scene\n");
  }
  else
  {
    if(isValidNodeIndex(originalParent))
    {
      m_scene.m_model.nodes[originalParent].children.push_back(newIdx);
      LOGI("  Added duplicated hierarchy as child of parent %d\n", originalParent);
    }
  }

  LOGI("Duplicated node hierarchy starting at '%s' (new root: %d)\n", m_scene.m_model.nodes[newIdx].name.c_str(), newIdx);

  if(reparse)
    m_scene.parseScene();

  return newIdx;
}

// Build a name that is unique among all current node names, using a " (N)" suffix.
// A trailing " (N)" on the base is stripped first, so duplicating "fox1 (2)" gives
// "fox1 (3)" instead of "fox1 (2) (1)". Unnamed nodes get the "Node" stem.
std::string SceneEditor::makeUniqueNodeName(const std::string& baseName) const
{
  // Derive the stem: strip a trailing " (N)" where N is one or more digits.
  std::string stem = baseName.empty() ? "Node" : baseName;
  if(stem.back() == ')')
  {
    size_t open = stem.rfind(" (");
    if(open != std::string::npos)
    {
      bool allDigits = (open + 2 < stem.size() - 1);  // at least one char between "(" and ")"
      for(size_t i = open + 2; allDigits && i + 1 < stem.size(); i++)
        allDigits = (stem[i] >= '0' && stem[i] <= '9');
      if(allDigits)
        stem = stem.substr(0, open);
    }
  }

  // Collect existing names once, then pick the smallest free suffix.
  std::unordered_set<std::string> used;
  used.reserve(m_scene.m_model.nodes.size());
  for(const tinygltf::Node& n : m_scene.m_model.nodes)
    used.insert(n.name);

  for(int i = 1;; i++)
  {
    std::string candidate = stem + " (" + std::to_string(i) + ")";
    if(used.find(candidate) == used.end())
      return candidate;
  }
}

int SceneEditor::duplicateNodeRecursive(int originalIndex, int newParentIndex, std::unordered_map<int, int>& nodeMap)
{
  if(!isValidNodeIndex(originalIndex))
  {
    return -1;
  }

  tinygltf::Node   newNode             = m_scene.m_model.nodes[originalIndex];
  std::vector<int> originalChildren    = newNode.children;
  glm::mat4        originalLocalMatrix = m_scene.m_nodesLocalMatrices[originalIndex];
  glm::mat4        originalWorldMatrix = m_scene.m_nodesWorldMatrices[originalIndex];

  newNode.children.clear();

  int newIdx = static_cast<int>(m_scene.m_model.nodes.size());

  m_scene.m_model.nodes.push_back(newNode);
  nodeMap[originalIndex] = newIdx;

  m_scene.m_nodeParents.push_back(newParentIndex);
  m_scene.m_nodesLocalMatrices.push_back(originalLocalMatrix);
  m_scene.m_nodesWorldMatrices.push_back(originalWorldMatrix);

  for(int childIdx : originalChildren)
  {
    int newChildIdx = duplicateNodeRecursive(childIdx, newIdx, nodeMap);
    if(newChildIdx != -1)
    {
      m_scene.m_model.nodes[newIdx].children.push_back(newChildIdx);
    }
  }

  return newIdx;
}

int SceneEditor::addNode(const std::string& name, int parentIndex)
{
  tinygltf::Node newNode;
  newNode.name        = name.empty() ? fmt::format("Node_{}", m_scene.m_model.nodes.size()) : name;
  newNode.translation = {0.0, 0.0, 0.0};
  newNode.rotation    = {0.0, 0.0, 0.0, 1.0};
  newNode.scale       = {1.0, 1.0, 1.0};
  newNode.mesh        = -1;
  newNode.camera      = -1;
  newNode.skin        = -1;

  int newIdx = static_cast<int>(m_scene.m_model.nodes.size());
  m_scene.m_model.nodes.push_back(newNode);

  m_scene.m_nodesLocalMatrices.push_back(glm::mat4(1.0f));
  m_scene.m_nodesWorldMatrices.push_back(glm::mat4(1.0f));
  m_scene.m_nodeParents.push_back(-1);

  if(parentIndex >= 0 && isValidNodeIndex(parentIndex))
  {
    m_scene.m_model.nodes[parentIndex].children.push_back(newIdx);
    m_scene.m_nodeParents[newIdx] = parentIndex;
  }
  else
  {
    auto& currentScene = m_scene.m_model.scenes[m_scene.m_currentScene];
    currentScene.nodes.push_back(newIdx);
  }

  return newIdx;
}

int SceneEditor::addLightNode(const std::string& lightType, const std::string& name, int parentIndex)
{
  auto& model = m_scene.m_model;

  // Ensure the extension is registered
  if(std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), "KHR_lights_punctual") == model.extensionsUsed.end())
  {
    model.extensionsUsed.push_back("KHR_lights_punctual");
  }

  tinygltf::Light light;
  light.name      = name;
  light.type      = lightType;
  light.color     = {1.0, 1.0, 1.0};
  light.intensity = 1.0;

  if(lightType == "spot")
  {
    light.spot.innerConeAngle = 0.0;
    light.spot.outerConeAngle = 0.785398;  // pi/4
  }

  int lightIndex = static_cast<int>(model.lights.size());
  model.lights.push_back(light);

  int nodeIndex = addNode(name, parentIndex);
  if(nodeIndex < 0)
    return -1;

  model.nodes[nodeIndex].light = lightIndex;

  m_scene.rebuildRenderNodesAndLights();

  return nodeIndex;
}

//--------------------------------------------------------------------------------------------------
// Procedural primitives
//--------------------------------------------------------------------------------------------------

const char* primitiveKindName(PrimitiveKind kind)
{
  for(const auto& info : kPrimitiveKinds)
    if(info.kind == kind)
      return info.name;
  return "Primitive";
}

int SceneEditor::addPrimitiveMesh(PrimitiveKind kind, const PrimitiveParams& params, int parentIndex)
{
  if(parentIndex >= 0 && blockIfNodeReadOnly(parentIndex, "add primitive to"))
    return -1;

  // A freshly-constructed Scene has no scene container yet; addNode() below needs one to add a root.
  if(m_scene.m_model.scenes.empty())
    m_scene.m_model.scenes.emplace_back();

  const std::string baseName = primitiveKindName(kind);

  // 1. Generate the procedural geometry (positions/normals/uv only - no tangents or colors).
  nvutils::PrimitiveMesh mesh;
  switch(kind)
  {
    case PrimitiveKind::ePlane:
      mesh = nvutils::createPlane(std::max(1, params.subdivU), params.size, params.size);
      break;
    case PrimitiveKind::eCube:
      mesh = nvutils::createCube(params.size, params.size, params.size);
      break;
    case PrimitiveKind::eSphere:
      mesh = nvutils::createSphereUv(params.size * 0.5f, std::max(3, params.subdivU), std::max(2, params.subdivV));
      break;
  }

  if(mesh.vertices.empty() || mesh.triangles.empty())
  {
    LOGW("addPrimitiveMesh: generated an empty mesh\n");
    return -1;
  }

  tinygltf::Model& model = m_scene.m_model;

  // 2. De-interleave attributes, flatten indices, and compute the position AABB (required on POSITION).
  const size_t           vtxCount = mesh.vertices.size();
  std::vector<glm::vec3> positions(vtxCount);
  std::vector<glm::vec3> normals(vtxCount);
  std::vector<glm::vec2> texcoords(vtxCount);
  glm::vec3              bmin(std::numeric_limits<float>::max());
  glm::vec3              bmax(std::numeric_limits<float>::lowest());
  for(size_t i = 0; i < vtxCount; i++)
  {
    positions[i] = mesh.vertices[i].pos;
    normals[i]   = mesh.vertices[i].nrm;
    texcoords[i] = mesh.vertices[i].tex;
    bmin         = glm::min(bmin, mesh.vertices[i].pos);
    bmax         = glm::max(bmax, mesh.vertices[i].pos);
  }
  std::vector<uint32_t> indices(mesh.triangles.size() * 3);
  for(size_t t = 0; t < mesh.triangles.size(); t++)
  {
    indices[t * 3 + 0] = mesh.triangles[t].indices.x;
    indices[t * 3 + 1] = mesh.triangles[t].indices.y;
    indices[t * 3 + 2] = mesh.triangles[t].indices.z;
  }

  // 3. Dedicated buffer so an undo can pop the whole append cleanly (see truncateGeometryTail).
  const int bufferIndex = static_cast<int>(model.buffers.size());
  model.buffers.emplace_back();
  model.buffers[bufferIndex].name = baseName + " Buffer";

  const int posAcc                  = tinygltf::utils::appendAccessor(model, bufferIndex, positions.data(),
                                                                      positions.size() * sizeof(glm::vec3), TINYGLTF_COMPONENT_TYPE_FLOAT,
                                                                      TINYGLTF_TYPE_VEC3, positions.size(), TINYGLTF_TARGET_ARRAY_BUFFER);
  model.accessors[posAcc].minValues = {bmin.x, bmin.y, bmin.z};
  model.accessors[posAcc].maxValues = {bmax.x, bmax.y, bmax.z};

  const int nrmAcc = tinygltf::utils::appendAccessor(model, bufferIndex, normals.data(),
                                                     normals.size() * sizeof(glm::vec3), TINYGLTF_COMPONENT_TYPE_FLOAT,
                                                     TINYGLTF_TYPE_VEC3, normals.size(), TINYGLTF_TARGET_ARRAY_BUFFER);
  const int texAcc = tinygltf::utils::appendAccessor(model, bufferIndex, texcoords.data(),
                                                     texcoords.size() * sizeof(glm::vec2), TINYGLTF_COMPONENT_TYPE_FLOAT,
                                                     TINYGLTF_TYPE_VEC2, texcoords.size(), TINYGLTF_TARGET_ARRAY_BUFFER);
  const int idxAcc = tinygltf::utils::appendAccessor(model, bufferIndex, indices.data(), indices.size() * sizeof(uint32_t),
                                                     TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, TINYGLTF_TYPE_SCALAR,
                                                     indices.size(), TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER);

  // 4. One new material per primitive (mid-grey dielectric), so each can be recolored independently.
  tinygltf::Material mat;
  mat.name                                 = baseName + " Material";
  mat.pbrMetallicRoughness.baseColorFactor = {0.8, 0.8, 0.8, 1.0};
  mat.pbrMetallicRoughness.metallicFactor  = 0.0;
  mat.pbrMetallicRoughness.roughnessFactor = 0.5;
  mat.doubleSided                          = (kind == PrimitiveKind::ePlane);  // planes are visible from both sides
  const int matIndex                       = static_cast<int>(model.materials.size());
  model.materials.push_back(mat);

  // 5. Primitive + mesh.
  tinygltf::Primitive prim;
  prim.mode                     = TINYGLTF_MODE_TRIANGLES;
  prim.attributes["POSITION"]   = posAcc;
  prim.attributes["NORMAL"]     = nrmAcc;
  prim.attributes["TEXCOORD_0"] = texAcc;
  prim.indices                  = idxAcc;
  prim.material                 = matIndex;

  tinygltf::Mesh newMesh;
  newMesh.name = baseName;
  newMesh.primitives.push_back(prim);
  const int meshIndex = static_cast<int>(model.meshes.size());
  model.meshes.push_back(newMesh);

  // 6. Node referencing the new mesh (child of parent, or scene root).
  const int nodeIndex = addNode(baseName, parentIndex);
  if(nodeIndex < 0)
    return -1;
  model.nodes[nodeIndex].mesh = meshIndex;

  // 7. Rebuild render data; the primitive-count change flags a geometry rebuild on the next GPU sync.
  m_scene.parseScene();

  LOGI("Added %s primitive as node %d (mesh %d, material %d)\n", baseName.c_str(), nodeIndex, meshIndex, matIndex);
  return nodeIndex;
}

void SceneEditor::truncateGeometryTail(const ModelTailSizes& sizes)
{
  // Geometry-only resize: does not call parseScene(). Undo truncates first, then
  // restoreFromSnapshot() reparses once against the final model (see AddPrimitiveCommand::undo).
  auto& model = m_scene.m_model;
  if(sizes.meshes <= model.meshes.size())
    model.meshes.resize(sizes.meshes);
  if(sizes.materials <= model.materials.size())
    model.materials.resize(sizes.materials);
  if(sizes.accessors <= model.accessors.size())
    model.accessors.resize(sizes.accessors);
  if(sizes.bufferViews <= model.bufferViews.size())
    model.bufferViews.resize(sizes.bufferViews);
  if(sizes.buffers <= model.buffers.size())
    model.buffers.resize(sizes.buffers);
}

void SceneEditor::deleteNode(int nodeIndex)
{
  if(blockIfNodeReadOnly(nodeIndex, "delete"))
    return;
  deleteNodeRecursive(nodeIndex);
  m_scene.parseScene();
}

void SceneEditor::collectDescendantIndices(int nodeIndex, std::vector<int>& indices) const
{
  for(int childIdx : getNode(nodeIndex).children)
  {
    if(isValidNodeIndex(childIdx))
    {
      indices.push_back(childIdx);
      collectDescendantIndices(childIdx, indices);
    }
  }
}

void SceneEditor::deleteNodeRecursive(int nodeIndex)
{
  if(!isValidNodeIndex(nodeIndex))
  {
    return;
  }

  auto& currentScene = m_scene.m_model.scenes[m_scene.m_currentScene];

  bool isRootNode = (std::find(currentScene.nodes.begin(), currentScene.nodes.end(), nodeIndex) != currentScene.nodes.end());

  if(isRootNode && currentScene.nodes.size() == 1)
  {
    LOGW("Cannot delete node '%s' - it's the last root node in scene %d (would leave scene invalid)\n",
         getNode(nodeIndex).name.c_str(), m_scene.m_currentScene);
    return;
  }

  // Flatten all descendants, delete highest-index first so lower indices stay stable.
  std::vector<int> descendants;
  collectDescendantIndices(nodeIndex, descendants);
  std::sort(descendants.begin(), descendants.end(), std::greater<int>());

  for(int descIdx : descendants)
  {
    deleteNodeSingle(descIdx);
  }

  int currentIndex = nodeIndex;
  for(int descIdx : descendants)
  {
    if(descIdx < nodeIndex)
      currentIndex--;
  }

  deleteNodeSingle(currentIndex);
}

void SceneEditor::deleteNodeSingle(int nodeIndex)
{
  if(!isValidNodeIndex(nodeIndex))
  {
    return;
  }

  removeNodeFromParent(nodeIndex);
  removeNodeFromSceneRoots(nodeIndex);

  m_scene.m_model.nodes.erase(m_scene.m_model.nodes.begin() + nodeIndex);
  m_scene.m_nodesLocalMatrices.erase(m_scene.m_nodesLocalMatrices.begin() + nodeIndex);
  m_scene.m_nodesWorldMatrices.erase(m_scene.m_nodesWorldMatrices.begin() + nodeIndex);
  m_scene.m_nodeParents.erase(m_scene.m_nodeParents.begin() + nodeIndex);

  remapIndicesAfterNodeDeletion(nodeIndex);
}

//--------------------------------------------------------------------------------------------------
// Hierarchy
//--------------------------------------------------------------------------------------------------

bool SceneEditor::wouldCreateCycle(int childIndex, int newParentIndex) const
{
  if(newParentIndex == -1)
    return false;

  if(childIndex == newParentIndex)
    return true;

  if(!isValidNodeIndex(childIndex) || !isValidNodeIndex(newParentIndex))
    return false;

  int current = newParentIndex;
  while(current != -1)
  {
    if(current == childIndex)
      return true;

    current = m_scene.m_nodeParents[current];
  }

  return false;
}

//--------------------------------------------------------------------------------------------------
// External assets (glTF 2.1): "break the lock" -- make a referenced asset editable.
//--------------------------------------------------------------------------------------------------
bool SceneEditor::makeExternalAssetEditable(int nodeIndex)
{
  if(!isValidNodeIndex(nodeIndex))
  {
    LOGW("Cannot make editable: invalid node index %d\n", nodeIndex);
    return false;
  }

  // Resolve the instance node (the editable node carrying `externalAsset`). When a read-only merged
  // child was clicked, walk up to its instance-node ancestor.
  int instance = nodeIndex;
  while(instance >= 0 && m_scene.m_model.nodes[instance].externalAsset < 0)
    instance = (instance < static_cast<int>(m_scene.m_nodeParents.size())) ? m_scene.m_nodeParents[instance] : -1;

  if(instance < 0)
  {
    LOGW("Cannot make editable: node %d is not part of a referenced external asset\n", nodeIndex);
    return false;
  }

  const int ea = m_scene.m_model.nodes[instance].externalAsset;
  if(ea < 0 || ea >= static_cast<int>(m_scene.m_model.externalAssets.size()))
    return false;
  const int fileIndex = m_scene.m_model.externalAssets[ea].file;

  // Unlock every instance that shares this source file: the merged subtrees become editable and the
  // instance nodes drop their `externalAsset` link, so the now-owned content is saved inline instead
  // of being re-externalized. Geometry/materials remain SHARED between the instances by design --
  // editing a shared material therefore affects all instances of the asset, which is the intent.
  int unlockedInstances = 0;
  for(int n = 0; n < static_cast<int>(m_scene.m_model.nodes.size()); ++n)
  {
    const int nea = m_scene.m_model.nodes[n].externalAsset;
    if(nea < 0 || nea >= static_cast<int>(m_scene.m_model.externalAssets.size()))
      continue;
    if(m_scene.m_model.externalAssets[nea].file != fileIndex)
      continue;

    m_scene.m_model.nodes[n].externalAsset = -1;

    std::vector<int> subtree;
    collectDescendantIndices(n, subtree);
    for(int d : subtree)
      Scene::clearExternalAssetMarker(m_scene.m_model.nodes[d]);
    Scene::clearExternalAssetMarker(m_scene.m_model.nodes[n]);  // safety: instance node normally unmarked
    ++unlockedInstances;
  }

  // Drop the now-orphaned externalAssets/files entries and remap the survivors, so the .gltf no
  // longer carries a dead reference for the asset we just unlocked.
  nvvkgltf::compactExternalAssetReferences(m_scene.m_model);

  // Keep the provenance list's emptiness in sync with the model: once no external asset remains, the
  // scene is a plain model (nothing else consumes the provenance detail). Emptiness must never lead
  // model.externalAssets, or save() would flatten a scene that still has live references.
  if(m_scene.m_model.externalAssets.empty())
    m_scene.m_referencedAssets.clear();

  LOGI("Made external asset editable (file %d): unlocked %d instance(s)\n", fileIndex, unlockedInstances);
  return unlockedInstances > 0;
}

void SceneEditor::setNodeParent(int childIndex, int newParentIndex)
{
  if(!isValidNodeIndex(childIndex))
  {
    LOGW("Cannot move invalid node index: %d\n", childIndex);
    return;
  }
  if(blockIfNodeReadOnly(childIndex, "reparent"))
    return;

  if(newParentIndex != -1 && !isValidNodeIndex(newParentIndex))
  {
    LOGW("Cannot move to invalid parent index: %d\n", newParentIndex);
    return;
  }

  // Reparenting appends to the destination parent's children list, so a read-only parent must be
  // protected too (a read-only child is already blocked above). Scene-root moves (-1) have no parent.
  if(newParentIndex != -1 && blockIfNodeReadOnly(newParentIndex, "reparent into"))
    return;

  if(wouldCreateCycle(childIndex, newParentIndex))
  {
    LOGW("Cannot move node %d under %d - would create cycle in hierarchy!\n", childIndex, newParentIndex);
    return;
  }

  int oldParentIndex = m_scene.m_nodeParents[childIndex];

  if(oldParentIndex == newParentIndex)
  {
    return;
  }

  if(oldParentIndex == -1)
  {
    auto& sceneNodes = m_scene.m_model.scenes[m_scene.m_currentScene].nodes;
    sceneNodes.erase(std::remove(sceneNodes.begin(), sceneNodes.end(), childIndex), sceneNodes.end());
  }
  else
  {
    auto& oldParent = m_scene.m_model.nodes[oldParentIndex];
    oldParent.children.erase(std::remove(oldParent.children.begin(), oldParent.children.end(), childIndex),
                             oldParent.children.end());
  }

  if(newParentIndex == -1)
  {
    m_scene.m_model.scenes[m_scene.m_currentScene].nodes.push_back(childIndex);
  }
  else
  {
    m_scene.m_model.nodes[newParentIndex].children.push_back(childIndex);
  }

  m_scene.m_nodeParents[childIndex] = newParentIndex;
  m_scene.buildTopologicalLevels();

  // Bump scene-graph revision to update GPU parent links after reparenting.
  m_scene.bumpSceneGraphRevision();

  m_scene.markNodeDirty(childIndex);
  m_scene.updateNodeWorldMatrices();
}

//--------------------------------------------------------------------------------------------------
// Resource attachment
//--------------------------------------------------------------------------------------------------

void SceneEditor::setNodeMesh(int nodeIndex, int meshIndex)
{
  if(!isValidNodeIndex(nodeIndex))
  {
    LOGW("Cannot set mesh on invalid node index: %d\n", nodeIndex);
    return;
  }
  if(blockIfNodeReadOnly(nodeIndex, "set mesh on"))
    return;

  if(meshIndex < 0 || meshIndex >= static_cast<int>(m_scene.m_model.meshes.size()))
  {
    LOGW("Invalid mesh index: %d\n", meshIndex);
    return;
  }

  tinygltf::Node& node = getNodeForEdit(nodeIndex);
  node.mesh            = meshIndex;

  m_scene.parseScene();
}

void SceneEditor::clearNodeMesh(int nodeIndex)
{
  if(!isValidNodeIndex(nodeIndex))
  {
    return;
  }
  if(blockIfNodeReadOnly(nodeIndex, "clear mesh on"))
    return;

  tinygltf::Node& node = getNodeForEdit(nodeIndex);
  node.mesh            = -1;

  m_scene.parseScene();
}

void SceneEditor::setNodeCamera(int nodeIndex, int cameraIndex)
{
  if(!isValidNodeIndex(nodeIndex))
  {
    return;
  }
  if(blockIfNodeReadOnly(nodeIndex, "set camera on"))
    return;

  if(cameraIndex < 0 || cameraIndex >= static_cast<int>(m_scene.m_model.cameras.size()))
  {
    LOGW("Invalid camera index: %d\n", cameraIndex);
    return;
  }

  tinygltf::Node& node = getNodeForEdit(nodeIndex);
  node.camera          = cameraIndex;
}

void SceneEditor::clearNodeCamera(int nodeIndex)
{
  if(!isValidNodeIndex(nodeIndex))
  {
    return;
  }
  if(blockIfNodeReadOnly(nodeIndex, "clear camera on"))
    return;

  tinygltf::Node& node = getNodeForEdit(nodeIndex);
  node.camera          = -1;
}

void SceneEditor::setNodeSkin(int nodeIndex, int skinIndex)
{
  if(!isValidNodeIndex(nodeIndex))
  {
    return;
  }
  if(blockIfNodeReadOnly(nodeIndex, "set skin on"))
    return;

  if(skinIndex < 0 || skinIndex >= static_cast<int>(m_scene.m_model.skins.size()))
  {
    LOGW("Invalid skin index: %d\n", skinIndex);
    return;
  }

  tinygltf::Node& node = getNodeForEdit(nodeIndex);
  node.skin            = skinIndex;
}

void SceneEditor::clearNodeSkin(int nodeIndex)
{
  if(!isValidNodeIndex(nodeIndex))
  {
    return;
  }
  if(blockIfNodeReadOnly(nodeIndex, "clear skin on"))
    return;

  tinygltf::Node& node = getNodeForEdit(nodeIndex);
  node.skin            = -1;
}

//--------------------------------------------------------------------------------------------------
// Immediate deletion helpers
//--------------------------------------------------------------------------------------------------

void SceneEditor::removeNodeFromParent(int nodeIndex)
{
  if(nodeIndex < 0 || nodeIndex >= static_cast<int>(m_scene.m_nodeParents.size()))
    return;

  int parentIdx = m_scene.m_nodeParents[nodeIndex];
  if(parentIdx >= 0 && isValidNodeIndex(parentIdx))
  {
    auto& parentNode = m_scene.m_model.nodes[parentIdx];
    auto  it         = std::find(parentNode.children.begin(), parentNode.children.end(), nodeIndex);
    if(it != parentNode.children.end())
    {
      parentNode.children.erase(it);
    }
  }
}

void SceneEditor::removeNodeFromSceneRoots(int nodeIndex)
{
  auto& currentScene = m_scene.m_model.scenes[m_scene.m_currentScene];
  auto  it           = std::find(currentScene.nodes.begin(), currentScene.nodes.end(), nodeIndex);
  if(it != currentScene.nodes.end())
  {
    currentScene.nodes.erase(it);
  }
}

void SceneEditor::remapIndicesAfterNodeDeletion(int deletedIndex)
{
  auto shiftIndex = [deletedIndex](int& idx) {
    if(idx > deletedIndex)
    {
      idx--;
    }
    else if(idx == deletedIndex)
    {
      idx = -1;
    }
  };

  for(auto& node : m_scene.m_model.nodes)
  {
    std::vector<int> newChildren;
    for(int childIdx : node.children)
    {
      int remapped = childIdx;
      shiftIndex(remapped);
      if(remapped >= 0)
      {
        newChildren.push_back(remapped);
      }
    }
    node.children = std::move(newChildren);
  }

  for(auto& scene : m_scene.m_model.scenes)
  {
    std::vector<int> newRoots;
    for(int rootIdx : scene.nodes)
    {
      int remapped = rootIdx;
      shiftIndex(remapped);
      if(remapped >= 0)
      {
        newRoots.push_back(remapped);
      }
    }
    scene.nodes = std::move(newRoots);
  }

  std::vector<int> animationsToRemove;
  for(size_t animIdx = 0; animIdx < m_scene.m_model.animations.size(); ++animIdx)
  {
    auto&            anim = m_scene.m_model.animations[animIdx];
    std::vector<int> channelsToRemove;

    for(size_t chanIdx = 0; chanIdx < anim.channels.size(); ++chanIdx)
    {
      auto& channel = anim.channels[chanIdx];
      if(channel.target_path == "pointer")
        continue;

      shiftIndex(channel.target_node);
      if(channel.target_node < 0)
      {
        channelsToRemove.push_back(static_cast<int>(chanIdx));
      }
    }

    for(auto it = channelsToRemove.rbegin(); it != channelsToRemove.rend(); ++it)
    {
      anim.channels.erase(anim.channels.begin() + *it);
    }

    if(anim.channels.empty())
    {
      animationsToRemove.push_back(static_cast<int>(animIdx));
    }
  }

  for(auto it = animationsToRemove.rbegin(); it != animationsToRemove.rend(); ++it)
  {
    m_scene.m_model.animations.erase(m_scene.m_model.animations.begin() + *it);
  }

  std::vector<int> skinsToRemove;
  for(size_t skinIdx = 0; skinIdx < m_scene.m_model.skins.size(); ++skinIdx)
  {
    auto& skin = m_scene.m_model.skins[skinIdx];

    if(skin.skeleton >= 0)
    {
      shiftIndex(skin.skeleton);
    }

    std::vector<int> newJoints;
    for(int jointIdx : skin.joints)
    {
      int remapped = jointIdx;
      shiftIndex(remapped);
      if(remapped >= 0)
      {
        newJoints.push_back(remapped);
      }
    }

    if(newJoints.empty())
    {
      LOGW("  Skin %zu has no joints left - will be removed\n", skinIdx);
      skinsToRemove.push_back(static_cast<int>(skinIdx));
    }
    else
    {
      skin.joints = std::move(newJoints);
    }
  }

  for(auto it = skinsToRemove.rbegin(); it != skinsToRemove.rend(); ++it)
  {
    int skinIdx = *it;

    for(auto& node : m_scene.m_model.nodes)
    {
      if(node.skin == skinIdx)
      {
        LOGW("  Clearing node '%s' skin reference (skin %d being deleted)\n", node.name.c_str(), skinIdx);
        node.skin = -1;
      }
      else if(node.skin > skinIdx)
      {
        node.skin--;
      }
    }

    m_scene.m_model.skins.erase(m_scene.m_model.skins.begin() + skinIdx);
  }

  for(int& parentIdx : m_scene.m_nodeParents)
  {
    shiftIndex(parentIdx);
  }

  m_scene.animation().resetPointer();
}

//--------------------------------------------------------------------------------------------------
// Material ops
//--------------------------------------------------------------------------------------------------

// Updates glTF + CPU RenderNodes. GPU transform path reads materialID from RenderNodeGpuMapping;
// bumpSceneGraphRevision() forces TransformComputeVk to rebuild that SSBO (same as variant material changes).
void SceneEditor::setPrimitiveMaterial(int meshIndex, int primIndex, int newMaterialID)
{
  if(meshIndex < 0 || meshIndex >= static_cast<int>(m_scene.m_model.meshes.size()))
  {
    LOGE("Invalid mesh index: %d\n", meshIndex);
    return;
  }
  if(m_scene.isMeshReadOnly(meshIndex))
  {
    LOGW("Cannot set material on mesh %d: it belongs to a referenced external asset (read-only)\n", meshIndex);
    return;
  }

  tinygltf::Mesh& mesh = m_scene.m_model.meshes[meshIndex];
  if(primIndex < 0 || primIndex >= static_cast<int>(mesh.primitives.size()))
  {
    LOGE("Invalid primitive index: %d for mesh %d\n", primIndex, meshIndex);
    return;
  }

  if(newMaterialID < 0 || newMaterialID >= static_cast<int>(m_scene.m_model.materials.size()))
  {
    LOGE("Invalid material ID: %d\n", newMaterialID);
    return;
  }

  tinygltf::Primitive& primitive = mesh.primitives[primIndex];
  const int            prevMat   = primitive.material;
  primitive.material             = newMaterialID;

  std::vector<nvvkgltf::RenderNode>& rnodes = m_scene.m_renderNodeRegistry.getRenderNodes();
  for(size_t renderNodeIdx = 0; renderNodeIdx < rnodes.size(); ++renderNodeIdx)
  {
    const RenderNode& rn = rnodes[renderNodeIdx];

    if(rn.refNodeID >= 0 && rn.refNodeID < static_cast<int>(m_scene.m_model.nodes.size()))
    {
      const tinygltf::Node& node = m_scene.m_model.nodes[rn.refNodeID];
      if(node.mesh == meshIndex)
      {
        auto np = m_scene.m_renderNodeRegistry.getNodeAndPrim(static_cast<int>(renderNodeIdx));
        if(np && np->second == primIndex)
        {
          m_scene.m_dirtyFlags.renderNodesVk.insert(static_cast<int>(renderNodeIdx));
          rnodes[renderNodeIdx].materialID = newMaterialID;
        }
      }
    }
  }

  if(prevMat != newMaterialID)
    m_scene.bumpSceneGraphRevision();
}

int SceneEditor::duplicateMaterial(int originalIndex)
{
  if(originalIndex < 0 || originalIndex >= static_cast<int>(m_scene.m_model.materials.size()))
  {
    LOGW("Cannot duplicate invalid material index: %d\n", originalIndex);
    return -1;
  }

  tinygltf::Material newMat = m_scene.m_model.materials[originalIndex];
  newMat.name += "_copy";

  int newIndex = static_cast<int>(m_scene.m_model.materials.size());
  m_scene.m_model.materials.push_back(newMat);

  for(int i = 0; i < static_cast<int>(m_scene.m_model.materials.size()); ++i)
    m_scene.markMaterialDirty(i);

  LOGI("Duplicated material '%s' -> '%s' (index %d -> %d)\n", m_scene.m_model.materials[originalIndex].name.c_str(),
       newMat.name.c_str(), originalIndex, newIndex);

  return newIndex;
}

int SceneEditor::duplicateMeshForNode(int meshIndex, int nodeIndex)
{
  if(meshIndex < 0 || meshIndex >= static_cast<int>(m_scene.m_model.meshes.size()))
  {
    LOGW("Cannot duplicate invalid mesh index: %d\n", meshIndex);
    return -1;
  }
  if(!isValidNodeIndex(nodeIndex))
  {
    LOGW("Cannot assign mesh to invalid node index: %d\n", nodeIndex);
    return -1;
  }
  if(blockIfNodeReadOnly(nodeIndex, "duplicate mesh for"))
    return -1;

  tinygltf::Mesh newMesh = m_scene.m_model.meshes[meshIndex];
  newMesh.name += "_copy";

  int newIndex = static_cast<int>(m_scene.m_model.meshes.size());
  m_scene.m_model.meshes.push_back(newMesh);

  m_scene.m_model.nodes[nodeIndex].mesh = newIndex;

  LOGI("Duplicated mesh '%s' -> '%s' (index %d -> %d), assigned to node %d\n",
       m_scene.m_model.meshes[meshIndex].name.c_str(), newMesh.name.c_str(), meshIndex, newIndex, nodeIndex);

  m_scene.parseScene();

  return newIndex;
}

int SceneEditor::splitPrimitiveMaterial(int nodeIndex, int primIndex)
{
  if(!isValidNodeIndex(nodeIndex))
  {
    LOGW("splitPrimitiveMaterial: invalid node index %d\n", nodeIndex);
    return -1;
  }
  if(blockIfNodeReadOnly(nodeIndex, "split material on"))
    return -1;

  int meshIdx = m_scene.m_model.nodes[nodeIndex].mesh;
  if(meshIdx < 0 || meshIdx >= static_cast<int>(m_scene.m_model.meshes.size()))
  {
    LOGW("splitPrimitiveMaterial: node %d has no valid mesh\n", nodeIndex);
    return -1;
  }

  if(primIndex < 0 || primIndex >= static_cast<int>(m_scene.m_model.meshes[meshIdx].primitives.size()))
  {
    LOGW("splitPrimitiveMaterial: invalid primitive index %d for mesh %d\n", primIndex, meshIdx);
    return -1;
  }

  tinygltf::Mesh newMesh = m_scene.m_model.meshes[meshIdx];
  newMesh.name += "_copy";
  int newMeshIdx = static_cast<int>(m_scene.m_model.meshes.size());
  m_scene.m_model.meshes.push_back(newMesh);
  m_scene.m_model.nodes[nodeIndex].mesh = newMeshIdx;

  int origMatIdx = m_scene.m_model.meshes[newMeshIdx].primitives[primIndex].material;
  if(origMatIdx < 0 || origMatIdx >= static_cast<int>(m_scene.m_model.materials.size()))
  {
    LOGW("splitPrimitiveMaterial: primitive has no valid material\n");
    m_scene.parseScene();
    return -1;
  }

  tinygltf::Material newMat = m_scene.m_model.materials[origMatIdx];
  newMat.name += "_copy";
  int newMatIdx = static_cast<int>(m_scene.m_model.materials.size());
  m_scene.m_model.materials.push_back(newMat);

  m_scene.m_model.meshes[newMeshIdx].primitives[primIndex].material = newMatIdx;

  m_scene.parseScene();

  int rnId = m_scene.getRenderNodeForPrimitive(nodeIndex, primIndex);
  if(rnId >= 0)
    m_scene.markRenderNodeDirty(rnId, true, false);

  for(int i = 0; i < static_cast<int>(m_scene.m_model.materials.size()); ++i)
    m_scene.markMaterialDirty(i);

  LOGI("Split: node %d now has mesh %d (was %d), primitive %d has material %d (was %d)\n", nodeIndex, newMeshIdx,
       meshIdx, primIndex, newMatIdx, origMatIdx);

  return newMatIdx;
}

int SceneEditor::findEquivalentMesh(int meshIndex) const
{
  if(meshIndex < 0 || meshIndex >= static_cast<int>(m_scene.m_model.meshes.size()))
    return -1;

  const tinygltf::Mesh& meshA = m_scene.m_model.meshes[meshIndex];
  const size_t          nPrim = meshA.primitives.size();
  if(nPrim == 0)
    return -1;

  for(int other = 0; other < static_cast<int>(m_scene.m_model.meshes.size()); ++other)
  {
    if(other == meshIndex)
      continue;

    const tinygltf::Mesh& meshB = m_scene.m_model.meshes[other];
    if(meshB.primitives.size() != nPrim)
      continue;

    bool equivalent = true;
    for(size_t p = 0; p < nPrim; ++p)
    {
      if(tinygltf::utils::generatePrimitiveKey(meshA.primitives[p]) != tinygltf::utils::generatePrimitiveKey(meshB.primitives[p]))
      {
        equivalent = false;
        break;
      }
    }
    if(equivalent)
      return other;
  }

  return -1;
}

int SceneEditor::mergePrimitiveMaterial(int nodeIndex)
{
  if(!isValidNodeIndex(nodeIndex))
  {
    LOGW("mergePrimitiveMaterial: invalid node index %d\n", nodeIndex);
    return -1;
  }
  if(blockIfNodeReadOnly(nodeIndex, "merge material on"))
    return -1;

  int meshIdx = m_scene.m_model.nodes[nodeIndex].mesh;
  if(meshIdx < 0 || meshIdx >= static_cast<int>(m_scene.m_model.meshes.size()))
  {
    LOGW("mergePrimitiveMaterial: node %d has no valid mesh\n", nodeIndex);
    return -1;
  }

  int equivMeshIdx = findEquivalentMesh(meshIdx);
  if(equivMeshIdx < 0)
  {
    LOGI("mergePrimitiveMaterial: no equivalent mesh for mesh %d\n", meshIdx);
    return -1;
  }

  const int deletedMeshIdx = meshIdx;

  m_scene.m_model.nodes[nodeIndex].mesh = equivMeshIdx;

  bool orphaned = true;
  for(size_t n = 0; n < m_scene.m_model.nodes.size(); ++n)
  {
    if(static_cast<int>(n) == nodeIndex)
      continue;
    if(m_scene.m_model.nodes[n].mesh == deletedMeshIdx)
    {
      orphaned = false;
      break;
    }
  }

  if(orphaned)
  {
    m_scene.m_model.meshes.erase(m_scene.m_model.meshes.begin() + deletedMeshIdx);

    for(auto& node : m_scene.m_model.nodes)
    {
      if(node.mesh == deletedMeshIdx)
        node.mesh = -1;
      else if(node.mesh > deletedMeshIdx)
        node.mesh--;
    }

    LOGI("Merge: removed orphan mesh %d, reindexed node.mesh\n", deletedMeshIdx);
  }

  m_scene.parseScene();

  const std::vector<int>& rnIds = m_scene.m_renderNodeRegistry.getRenderNodesForNode(nodeIndex);
  for(int rn : rnIds)
    m_scene.markRenderNodeDirty(rn, true, false);

  const int finalMeshIdx = m_scene.m_model.nodes[nodeIndex].mesh;
  LOGI("Merge: node %d now uses mesh %d (was %d)\n", nodeIndex, finalMeshIdx, deletedMeshIdx);

  return finalMeshIdx;
}

}  // namespace nvvkgltf
