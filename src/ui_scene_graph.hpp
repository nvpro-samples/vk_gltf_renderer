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

#include <imgui.h>
#include <tinygltf/tiny_gltf.h>
#include <glm/glm.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <nvutils/bounding_box.hpp>

class UiSceneGraph
{
public:
  UiSceneGraph() = default;

  void setModel(tinygltf::Model* model)
  {
    m_model         = model;
    m_selectedIndex = -1;
  }
  void setBbox(nvutils::Bbox bbox) { m_bbox = bbox; }  // Use for translation of model

  void render();
  bool hasTransformChanged() { return m_changes.test(eNodeTransformDirty); }
  bool hasMaterialChanged() { return m_changes.test(eMaterialDirty); }
  bool hasLightChanged() { return m_changes.test(eLightDirty); }
  bool hasVisibilityChanged() { return m_changes.test(eNodeVisibleDirty); }
  bool hasMaterialFlagChanges() { return m_changes.test(eMaterialFlagDirty); }
  bool hasAnyChanges() { return m_changes.any(); }
  void resetChanges() { m_changes.reset(); }
  void renderDetails();
  void selectNode(int nodeIndex);
  int  selectedNode() const { return (m_selectType == eNode) ? m_selectedIndex : -1; }
  int  selectedMaterial() const { return (m_selectType == eMaterial) ? m_selectedIndex : -1; }

private:
  void renderNode(int nodeIndex);
  void renderMesh(int meshIndex);
  void renderPrimitive(const tinygltf::Primitive& primitive, int primID);
  void renderLight(int lightIndex);
  void renderCamera(int cameraIndex);
  void renderMaterial(int materialIndex);

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
  void materialVolume(tinygltf::Material& material);

  void renderNodeDetails(int nodeIndex);
  void renderLightDetails(int lightIndex);

  void getNodeTransform(const tinygltf::Node& node, glm::vec3& translation, glm::quat& rotation, glm::vec3& scale);
  void preprocessOpenNodes();
  bool markOpenNodes(int nodeIndex, int targetNodeIndex, std::unordered_set<int>& openNodes);
  void renderSceneGraph();

  std::unordered_set<int> m_openNodes;  // To keep track of nodes that should be open

  tinygltf::Model* m_model = nullptr;
  enum SelectType
  {
    eNode,
    eMaterial,
    eLight
  };
  SelectType      m_selectType    = eNode;
  int             m_selectedIndex = -1;
  std::bitset<32> m_changes;
  nvutils::Bbox   m_bbox;

  enum DirtyFlags
  {
    eNodeTransformDirty,
    eMaterialDirty,
    eLightDirty,
    eNodeVisibleDirty,
    eMaterialFlagDirty,
    // Add more flags as needed
  };
  bool m_doScroll = false;
};
