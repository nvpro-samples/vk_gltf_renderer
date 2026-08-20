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

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <imgui.h>
#include <glm/gtc/quaternion.hpp>
#include <nvutils/bounding_box.hpp>

#include "scene_selection.hpp"
#include "ui_host_services.hpp"

class UndoStack;

namespace nvvkgltf {
class Scene;
}

namespace tinygltf {
struct Material;
struct Light;
struct Texture;
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

  // Host services shared with the scene browser: image file dialog, texture thumbnails, toasts. Without
  // them the "Load from file" action, slot/picker thumbnails, and error toasts are unavailable.
  void setHostServices(UiHostServices services) { m_host = std::move(services); }
  // Open the large image viewer for a glTF image index (wired to the scene browser's viewer). Lets a
  // click on a slot's thumbnail show the full-size image. No-op when unset.
  void setViewImageCallback(std::function<void(int)> cb) { m_onViewImage = std::move(cb); }
  // Resolve a glTF image index to a bounded ImGui thumbnail (0 if none) for the Image inspector.
  void setImageThumbnailCallback(std::function<ImTextureID(int)> cb) { m_getImageThumbnail = std::move(cb); }

  // Rebuild the cached texture-name list from the current model. Call after the texture set changes
  // outside the inspector (e.g. undo/redo of an import).
  void refreshTextureNames();

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
  void renderTextureProperties(int textureIdx);
  void renderImageProperties(int imageIdx);
  void renderSamplerProperties(int samplerIdx);
  void renderAnimationProperties(int animIdx);

  // Clickable cross-reference: renders `label` as a link and, when clicked, selects (jumps to) element
  // `index` of the given kind so the Inspector navigates the glTF graph. Negative index => disabled dash.
  bool elementLink(const char* label, SceneSelection::SelectionType kind, int index);
  // Canonical "<icon> [index] <name>" label for a cross-reference of `kind` (bounds-checked; empty if invalid).
  std::string elementRefLabel(SceneSelection::SelectionType kind, int index) const;
  // A "rowLabel:   <link>" row (or `emptyText` when index < 0), links aligned at labelWidth. The one place
  // every single-reference row is formatted, so the "icon [idx] name" convention can't drift per call site.
  void elementLinkRow(const char* rowLabel, SceneSelection::SelectionType kind, int index, float labelWidth = 120.0f, const char* emptyText = "-");
  // "<header>" + the textures satisfying `match`, each a jump link (or "(none)"). Shared by the Image and
  // Sampler inspectors' reverse "used by textures" lists.
  void renderTexturesUsing(const char* header, const std::function<bool(const tinygltf::Texture&)>& match);
  void renderNodeRelationships(int nodeIdx);  // parent / mesh / camera / light / skin / children as jump links
  void renderNodeExtensions(int nodeIdx);     // KHR_node_visibility / selectability / hoverability (+ other exts)
  // One primitive's detail (mode, vertex/triangle counts, attribute->accessor list). Shared by the Mesh
  // inspector (per primitive) and the composite primitive/pick inspector. The material link is not part of
  // this block: callers that want it append their own row (the pick inspector has a full MATERIAL section).
  void renderPrimitiveDetail(int meshIdx, int primIdx);

  //==================================================================================================
  // PROPERTY SECTIONS (reusable)
  //==================================================================================================
  void renderTransformSection(int nodeIdx);
  void renderMaterialSection(int matIdx, bool allowEdit = true);
  void renderMaterialAssignmentToolbar(int meshIdx, int primIdx, int nodeIdx, int matIdx);

  // One editable texture slot: name + actions (assign existing / load from file / clear). Returns true
  // when the slot's texture index changed (assign / switch / clear / import), so the caller writes back
  // copy-based rows and records the undo step.
  template <typename T>
  bool renderTextureEditRow(const char* label, T& info);

  // Renders the "SwitchTexture" modal (filterable list/grid picker of existing textures). Returns true
  // when a selection is committed into info via OK. Must be called within the slot's ImGui ID scope.
  template <typename T>
  bool renderTexturePicker(T& info, bool hasTexture);

  // "Load from file" action for a slot: opens the file dialog, imports the image as a new texture, and
  // assigns it to info. Returns true if a texture was imported (a normal material change for the caller).
  template <typename T>
  bool importTextureIntoSlot(T& info);

  // Per-binding KHR_texture_transform editor: a small button in the slot that opens an add/edit/remove
  // popup. Edited on the material's texture reference (not the shared texture). Returns true on change.
  template <typename T>
  bool renderTextureTransformButton(T& info);

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
                                      const std::function<void()>& whenAdd,
                                      const char*                  aliasExtName = nullptr);

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
  bool materialRetroreflection(tinygltf::Material& material);
  bool materialUnlit(tinygltf::Material& material);
  bool materialVolume(tinygltf::Material& material, int matIdx);  // Needs matIdx for special RTX dirty marking
  bool materialScatter(tinygltf::Material& material);

  //==================================================================================================
  // MEMBER VARIABLES
  //==================================================================================================
  nvvkgltf::Scene* m_scene     = nullptr;
  SceneSelection*  m_selection = nullptr;
  UndoStack*       m_undoStack = nullptr;
  nvutils::Bbox    m_bbox;

  // Renderer-provided host services (file dialog, thumbnails, toasts) shared with the scene browser.
  UiHostServices                  m_host;
  std::function<void(int)>        m_onViewImage;        // open the large image viewer for an image index
  std::function<ImTextureID(int)> m_getImageThumbnail;  // image index -> ImGui thumbnail (Image inspector)

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
