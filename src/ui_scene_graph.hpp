/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*

This is the ImGui UI for the glTF model.
It is used to render the scene graph and the details of the selected node, such as the transform and the materials.

*/


#include <string>
#include <vector>
#include <unordered_set>
#include <functional>

#include <imgui.h>
#include <tinygltf/tiny_gltf.h>
#include <glm/glm.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <nvutils/bounding_box.hpp>

class GltfRenderer;  // Forward declaration

class UiSceneGraph
{
public:
  // Structure to store the scene transform state
  struct SceneTransformState
  {
    glm::vec3              translation = glm::vec3(0.0f);
    glm::quat              rotation    = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3              scale       = glm::vec3(1.0f);
    std::vector<int>       nodeIds;
    std::vector<glm::mat4> baselineLocal;
  };

  // Callback types for camera operations
  using CameraApplyCallback       = std::function<void(int cameraIndex)>;
  using CameraSetFromViewCallback = std::function<void(int cameraIndex)>;
  // Callback to look up RenderNode index from node and primitive index
  using RenderNodeLookupCallback = std::function<int(int nodeIndex, int primitiveIndex)>;

  // Event types for better decoupling
  enum class EventType
  {
    CameraApply,        // Emitted when "Apply to Current View" is clicked
    CameraSetFromView,  // Emitted when "Set from Current View" is clicked
    NodeSelected,       // Emitted when a node is selected in the scene graph
    PrimitiveSelected,  // Emitted when a primitive is selected (via picking or UI)
    MaterialSelected    // Emitted when a material is selected
  };

  struct Event
  {
    EventType type;
    int       data;             // camera index, node index, material index, etc.
    int       renderNodeIndex;  // For PrimitiveSelected: the RenderNode index
  };

  using EventCallback = std::function<void(const Event&)>;

  UiSceneGraph() = default;

  void setModel(tinygltf::Model* model);

  void setBbox(nvutils::Bbox bbox) { m_bbox = bbox; }  // Use for translation of model

  // Set the event callback to handle UI events
  void setEventCallback(const EventCallback& callback) { m_eventCallback = callback; }

  // Set the render node lookup callback (for mapping node+primitive to RenderNode index)
  void setRenderNodeLookup(const RenderNodeLookupCallback& callback) { m_renderNodeLookup = callback; }

  void render(bool* showSceneGraph = nullptr, bool* showProperties = nullptr);  // Render the scene graph and details

  // Dirty tracking - check if specific types of changes occurred
  bool hasTransformChanged() const { return !m_dirty.nodes.empty(); }
  bool hasMaterialChanged() const { return !m_dirty.materials.empty(); }
  bool hasLightChanged() const { return !m_dirty.lights.empty(); }
  bool hasCameraChanged() const { return !m_dirty.cameras.empty(); }
  bool hasVisibilityChanged() const { return !m_dirty.visibilityNodes.empty(); }
  bool hasMaterialInstanceFlagChanges() const { return !m_dirty.materialInstanceFlagsChanged.empty(); }
  bool hasCameraApplyToView() const { return m_dirty.cameraApplyToView; }
  bool hasAnyChanges() const { return m_dirty.hasAny(); }
  void resetChanges() { m_dirty.clear(); }

  // Getters for dirty index sets - enables surgical GPU buffer updates
  const std::unordered_set<int>& getDirtyMaterials() const { return m_dirty.materials; }
  const std::unordered_set<int>& getMaterialInstanceFlagsChanged() const
  {
    return m_dirty.materialInstanceFlagsChanged;
  }
  const std::unordered_set<int>& getDirtyLights() const { return m_dirty.lights; }
  const std::unordered_set<int>& getDirtyNodes() const { return m_dirty.nodes; }
  const std::unordered_set<int>& getDirtyVisibilityNodes() const { return m_dirty.visibilityNodes; }
  const std::unordered_set<int>& getDirtyCameras() const { return m_dirty.cameras; }
  void                           selectNode(int nodeIndex);
  void                           selectPrimitive(int renderNodeIndex, int nodeIndex, int primitiveIndex);
  void                           selectMaterial(int materialIndex, int nodeIndex = -1);
  int                            selectedNode() const { return m_selectedIndex; }
  int                            selectedRenderNode() const { return m_selectedRenderNodeIndex; }
  int                            selectedPrimitiveIndex() const { return m_selectedPrimitiveIndex; }
  int                            selectedMaterial() const { return m_selectedMaterialIndex; }
  int                            selectedNodeForMaterial() const { return m_selectedNodeForMaterial; }


  // Get node for camera (made public for camera application)
  int getNodeForCamera(int cameraIndex);

private:
  void renderNode(int nodeIndex);
  void renderMesh(int meshIndex);
  void renderPrimitive(const tinygltf::Primitive& primitive, int primID, int nodeIndex, int renderNodeIndex);
  void renderLight(int lightIndex);
  void renderCamera(int cameraIndex);
  void renderMaterial(int materialIndex);

  void addButton(const char* extensionName, std::function<void()> addCallback);
  void removeButton(tinygltf::Material& material, const char* extensionName);
  void materialAnisotropy(tinygltf::Material& material);
  void materialClearcoat(tinygltf::Material& material);
  void materialDiffuseTransmission(tinygltf::Material& material);
  void materialDispersion(tinygltf::Material& material);
  void materialEmissiveStrength(tinygltf::Material& material);
  void materialIor(tinygltf::Material& material);
  void materialIridescence(tinygltf::Material& material);
  void materialSheen(tinygltf::Material& material);
  void materialSpecular(tinygltf::Material& material);
  void materialTransmission(tinygltf::Material& material);
  void materialUnlit(tinygltf::Material& material);
  void materialVolume(tinygltf::Material& material);
  void materialVolumeScatter(tinygltf::Material& material);

  void renderNodeDetails(int nodeIndex);
  void renderLightDetails(int lightIndex);
  void renderCameraDetails(int                              cameraIndex,
                           const CameraApplyCallback&       applyCameraCallback,
                           const CameraSetFromViewCallback& setCameraFromViewCallback);
  void renderCameraDetailsWithEvents(int cameraIndex);
  void renderMaterialSelector(int nodeIndex);

  void preprocessOpenNodes();
  bool markOpenNodes(int nodeIndex, int targetNodeIndex, std::unordered_set<int>& openNodes);
  void renderSceneGraph(bool* showSceneGraph);
  void renderDetails(bool* showProperties);
  void renderAssetInfo();  // Display glTF asset info (version, generator, copyright)
  void applySceneTransform(size_t sceneID);

  // Helper functions to get available materials for a node
  std::vector<int>                         getMaterialsForNode(int nodeIndex) const;
  std::vector<std::pair<int, std::string>> getPrimitiveInfoForNode(int nodeIndex) const;

  // Cache management
  void buildCache(std::unordered_map<int, int>& cache, bool& dirtyFlag, int(tinygltf::Node::* member)) const;
  int getNodeForElement(int elementIndex, std::unordered_map<int, int>& cache, bool& dirtyFlag, int(tinygltf::Node::* member)) const;

  // Convenience functions that use the template
  int getNodeForMesh(int meshIndex);
  int getNodeForLight(int lightIndex);

  std::unordered_set<int> m_openNodes;  // To keep track of nodes that should be open

  tinygltf::Model* m_model = nullptr;
  enum SelectType
  {
    eNode,
    eMaterial,
    eLight,
    eCamera
  };
  SelectType    m_selectType              = eNode;
  int           m_selectedIndex           = -1;
  int           m_selectedRenderNodeIndex = -1;  // Selected RenderNode index (for primitive selection)
  int           m_selectedPrimitiveIndex  = -1;  // Selected primitive index within the mesh
  int           m_selectedMaterialIndex   = -1;  // Currently selected material
  int           m_selectedNodeForMaterial = -1;  // Node context for the selected material
  nvutils::Bbox m_bbox;

  // Cache for efficient lookups
  std::unordered_map<int, int>     m_meshToNodeMap;
  std::unordered_map<int, int>     m_lightToNodeMap;
  std::unordered_map<int, int>     m_cameraToNodeMap;
  bool                             m_meshToNodeMapDirty   = true;
  bool                             m_lightToNodeMapDirty  = true;
  bool                             m_cameraToNodeMapDirty = true;
  std::vector<std::string>         m_textureNames;
  std::vector<SceneTransformState> m_sceneTransforms;

  // Dirty tracking - tracks which specific elements changed for surgical GPU updates
  struct DirtyTracking
  {
    std::unordered_set<int> materials;                     // Material indices that changed
    std::unordered_set<int> materialInstanceFlagsChanged;  // Subset of materials needing TLAS rebuild (alpha mode, double-sided)
    std::unordered_set<int> lights;                        // Light indices that changed
    std::unordered_set<int> nodes;                         // Node indices with transform changes
    std::unordered_set<int> visibilityNodes;               // Node indices with visibility changes
    std::unordered_set<int> cameras;                       // Camera indices that changed
    bool                    cameraApplyToView = false;  // Action flag (not a change)

    void clear()
    {
      materials.clear();
      materialInstanceFlagsChanged.clear();
      lights.clear();
      nodes.clear();
      visibilityNodes.clear();
      cameras.clear();
      cameraApplyToView = false;
    }

    bool hasAny() const
    {
      return !materials.empty() || !materialInstanceFlagsChanged.empty() || !lights.empty() || !nodes.empty()
             || !visibilityNodes.empty() || !cameras.empty() || cameraApplyToView;
    }
  };
  DirtyTracking m_dirty;

  bool m_doScroll = false;

  EventCallback            m_eventCallback;
  RenderNodeLookupCallback m_renderNodeLookup;
};
