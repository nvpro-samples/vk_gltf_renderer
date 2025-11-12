/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*

This is the ImGui UI for the glTF model.
It is used to render the scene graph and the details of the selected node, such as the transform and the materials.

*/


#include <string>
#include <vector>
#include <bitset>
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
  // Callback types for camera operations
  using CameraApplyCallback       = std::function<void(int cameraIndex)>;
  using CameraSetFromViewCallback = std::function<void(int cameraIndex)>;

  // Event types for better decoupling
  enum class EventType
  {
    CameraApply,        // Emitted when "Apply to Current View" is clicked
    CameraSetFromView,  // Emitted when "Set from Current View" is clicked
    NodeSelected,       // Emitted when a node is selected in the scene graph
    MaterialSelected    // Emitted when a material is selected
  };

  struct Event
  {
    EventType type;
    int       data;  // camera index, node index, material index, etc.
  };

  using EventCallback = std::function<void(const Event&)>;

  UiSceneGraph() = default;

  void setModel(tinygltf::Model* model)
  {
    m_model                 = model;
    m_selectedIndex         = -1;
    m_selectedMaterialIndex = -1;
    m_meshToNodeMapDirty    = true;  // Mark cache as dirty when model changes
    m_lightToNodeMapDirty   = true;
    m_cameraToNodeMapDirty  = true;
  }
  void setBbox(nvutils::Bbox bbox) { m_bbox = bbox; }  // Use for translation of model

  // Set the event callback to handle UI events
  void setEventCallback(const EventCallback& callback) { m_eventCallback = callback; }

  void render();  // Render the scene graph and details

  bool hasTransformChanged() { return m_changes.test(eNodeTransformDirty); }
  bool hasMaterialChanged() { return m_changes.test(eMaterialDirty); }
  bool hasLightChanged() { return m_changes.test(eLightDirty); }
  bool hasCameraChanged() { return m_changes.test(eCameraDirty); }
  bool hasVisibilityChanged() { return m_changes.test(eNodeVisibleDirty); }
  bool hasMaterialFlagChanges() { return m_changes.test(eMaterialFlagDirty); }
  bool hasCameraApplyToView() { return m_changes.test(eCameraApplyToView); }
  bool hasAnyChanges() { return m_changes.any(); }
  void resetChanges() { m_changes.reset(); }
  void selectNode(int nodeIndex);
  void selectMaterial(int materialIndex, int nodeIndex = -1);
  int  selectedNode() const { return m_selectedIndex; }
  int  selectedMaterial() const { return m_selectedMaterialIndex; }
  int  selectedNodeForMaterial() const { return m_selectedNodeForMaterial; }


  // Get node transform (made public for camera application)
  void getNodeTransform(const tinygltf::Node& node, glm::vec3& translation, glm::quat& rotation, glm::vec3& scale);

  // Get node for camera (made public for camera application)
  int getNodeForCamera(int cameraIndex);

private:
  void renderNode(int nodeIndex);
  void renderMesh(int meshIndex);
  void renderPrimitive(const tinygltf::Primitive& primitive, int primID, int nodeIndex);
  void renderLight(int lightIndex);
  void renderCamera(int cameraIndex);
  void renderMaterial(int materialIndex);

  void addButton(tinygltf::Material& material, const char* extensionName, std::function<void()> addCallback);
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

  void renderNodeDetails(int nodeIndex);
  void renderLightDetails(int lightIndex);
  void renderCameraDetails(int                              cameraIndex,
                           const CameraApplyCallback&       applyCameraCallback,
                           const CameraSetFromViewCallback& setCameraFromViewCallback);
  void renderCameraDetailsWithEvents(int cameraIndex);
  void renderMaterialSelector(int nodeIndex);

  void preprocessOpenNodes();
  bool markOpenNodes(int nodeIndex, int targetNodeIndex, std::unordered_set<int>& openNodes);
  void renderSceneGraph();
  void renderDetails();

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
  SelectType      m_selectType              = eNode;
  int             m_selectedIndex           = -1;
  int             m_selectedMaterialIndex   = -1;  // Currently selected material
  int             m_selectedNodeForMaterial = -1;  // Node context for the selected material
  std::bitset<32> m_changes;
  nvutils::Bbox   m_bbox;

  // Cache for efficient lookups
  std::unordered_map<int, int> m_meshToNodeMap;
  std::unordered_map<int, int> m_lightToNodeMap;
  std::unordered_map<int, int> m_cameraToNodeMap;
  bool                         m_meshToNodeMapDirty   = true;
  bool                         m_lightToNodeMapDirty  = true;
  bool                         m_cameraToNodeMapDirty = true;

  enum DirtyFlags
  {
    eNodeTransformDirty,
    eMaterialDirty,
    eLightDirty,
    eNodeVisibleDirty,
    eMaterialFlagDirty,
    eCameraDirty,
    eCameraApplyToView,
    // Add more flags as needed
  };
  bool m_doScroll = false;

  EventCallback m_eventCallback;
};
