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
 * UiInspector - Context-aware properties panel
 * 
 * Displays properties for the currently selected element:
 * - Node: Transform (TRS) + visibility + operations
 * - Primitive: Node transform + primitive stats + material assignment
 * - Material: All PBR properties + extensions
 * - Mesh: Mesh info + operations
 * - Camera: Camera properties + sync with view
 * - Light: Light properties
 */

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <nvutils/bounding_box.hpp>

#include "scene_selection.hpp"

class UndoStack;

namespace nvvkgltf {
class Scene;
}

namespace tinygltf {
struct Material;
struct Light;
}  // namespace tinygltf

class UiInspector
{
public:
  UiInspector() = default;
  ~UiInspector();

  void setScene(nvvkgltf::Scene* scene);
  void setSelection(SceneSelection* selection) { m_selection = selection; }
  void setUndoStack(UndoStack* undoStack) { m_undoStack = undoStack; }
  void setBbox(nvutils::Bbox bbox) { m_bbox = bbox; }

  void render(bool* show = nullptr, bool isBusy = false);

private:
  //==================================================================================================
  // PROPERTY EDITORS (context-aware)
  //==================================================================================================
  void renderNoSelection();
  void renderNodeProperties(int nodeIdx);
  void renderPrimitiveProperties(int nodeIdx, int primIdx, int meshIdx);
  void renderMaterialProperties(int matIdx);
  void renderMeshProperties(int meshIdx);
  void renderCameraProperties(int camIdx);
  void renderLightProperties(int lightIdx);

  //==================================================================================================
  // PROPERTY SECTIONS (reusable)
  //==================================================================================================
  void renderTransformSection(int nodeIdx);
  void renderMaterialSection(int matIdx, bool allowEdit = true);
  void renderMaterialAssignmentToolbar(int meshIdx, int primIdx, int nodeIdx, int matIdx);

  //==================================================================================================
  // OPERATIONS (buttons/actions)
  //==================================================================================================
  void renderMaterialOperations(int matIdx, int nodeContext = -1);

  //==================================================================================================
  // MATERIAL EXTENSIONS (reuse from original, DRY)
  //==================================================================================================
  bool renderMaterialExtensions(tinygltf::Material& material, int matIdx);
  bool addButton(const char* extensionName, std::function<void()> addCallback);
  bool removeButton(tinygltf::Material& material, const char* extensionName);

  // DRY helper: tree node + hasExt ? (remove + content) : add. Returns true if material changed.
  bool renderMaterialExtensionSection(tinygltf::Material&          material,
                                      const char*                  treeLabel,
                                      const char*                  extName,
                                      const std::function<bool()>& whenHasExt,
                                      const std::function<void()>& whenAdd);

  // All 12 material extension functions (from original)
  // Return true if material was modified
  bool materialAnisotropy(tinygltf::Material& material);
  bool materialClearcoat(tinygltf::Material& material);
  bool materialDiffuseTransmission(tinygltf::Material& material);
  bool materialDispersion(tinygltf::Material& material);
  bool materialEmissiveStrength(tinygltf::Material& material);
  bool materialIor(tinygltf::Material& material);
  bool materialIridescence(tinygltf::Material& material);
  bool materialSheen(tinygltf::Material& material);
  bool materialSpecular(tinygltf::Material& material);
  bool materialTransmission(tinygltf::Material& material);
  bool materialUnlit(tinygltf::Material& material);
  bool materialVolume(tinygltf::Material& material, int matIdx);  // Needs matIdx for special RTX dirty marking
  bool materialVolumeScatter(tinygltf::Material& material);

  //==================================================================================================
  // MEMBER VARIABLES
  //==================================================================================================
  nvvkgltf::Scene* m_scene     = nullptr;
  SceneSelection*  m_selection = nullptr;
  UndoStack*       m_undoStack = nullptr;
  nvutils::Bbox    m_bbox;

  // Cached texture names (for material dropdowns)
  std::vector<std::string> m_textureNames;

  // Persistent euler angles to avoid gimbal lock from quat<->euler round-trips.
  // Re-synced from quaternion only on node selection change or external modification (gizmo).
  struct CachedEuler
  {
    int       nodeIdx = -1;
    glm::vec3 euler{0.0f};
    glm::quat quat{1, 0, 0, 0};
  };
  CachedEuler m_cachedEuler;

  // Transform snapshot for undo: captures pre-edit TRS when a DragFloat3 edit begins.
  // Pushed as SetTransformCommand when the edit ends.
  struct TransformSnapshot
  {
    int       nodeIdx = -1;
    glm::vec3 translation{0.0f};
    glm::quat rotation{1, 0, 0, 0};
    glm::vec3 scale{1.0f};
  };
  TransformSnapshot m_transformSnapshot;
  bool              m_transformModifiedLastFrame = false;

  // Material snapshot for undo: captures full tinygltf::Material before editing starts.
  // Pushed as EditMaterialCommand when the edit cycle ends.
  int                                 m_materialSnapshotIdx = -1;
  std::unique_ptr<tinygltf::Material> m_materialSnapshotData;
  bool                                m_materialModifiedLastFrame = false;

  // Light snapshot for undo: captures full tinygltf::Light before editing starts.
  // Pushed as EditLightCommand when the edit cycle ends.
  int                              m_lightSnapshotIdx = -1;
  std::unique_ptr<tinygltf::Light> m_lightSnapshotData;
  bool                             m_lightModifiedLastFrame = false;
};
