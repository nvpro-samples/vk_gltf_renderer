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
 * UiSceneBrowser - Tabbed scene browser window
 * 
 * Displays the glTF scene in multiple views:
 * - Asset Info: Version, generator, metadata
 * - Scene Graph: Hierarchical tree view with scene transform
 * - Scene List: Flat grouped view (nodes, meshes, materials, cameras, lights, textures, images, animations)
 */

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nvutils/bounding_box.hpp>

#include "scene_selection.hpp"
#include "ui_animation_control.hpp"

class UndoStack;

namespace nvvkgltf {
class Scene;
}

// Forward declarations (no need for full headers here)
namespace tinygltf {
class Node;  // Actually a struct, but forward declared as class in tinygltf
}

class UiSceneBrowser
{
public:
  enum class ViewTab
  {
    AssetInfo,   // Version, generator, metadata
    SceneGraph,  // Hierarchical tree view with scene transform
    SceneList,   // Flat list (grouped by type)
    Debug        // Debug information and tools
  };

  UiSceneBrowser() = default;

  void setScene(nvvkgltf::Scene* scene);
  void setSelection(SceneSelection* selection) { m_selection = selection; }
  void setUndoStack(UndoStack* undoStack) { m_undoStack = undoStack; }
  void setBbox(nvutils::Bbox bbox) { m_bbox = bbox; }
  void markCachesDirty();

  void render(bool* show = nullptr, bool isBusy = false);

  // Selection synchronization (called from external picker)
  void focusOnSelection();  // Auto-expand/scroll to selection

  // Animation control accessor (needed by renderer for update logic)
  AnimationControl& getAnimationControl() { return m_animControl; }

  // Node lookup helpers (public for camera operations)
  int getNodeForCamera(int camIdx);
  int getNodeForMesh(int meshIdx);
  int getNodeForLight(int lightIdx);

private:
  //==================================================================================================
  // TAB RENDERING
  //==================================================================================================
  void renderAssetInfoTab();
  void renderSceneGraphTab();
  void renderSceneListTab();

  //==================================================================================================
  // TAB RENDERERS
  //==================================================================================================
  void renderDebugTab();

  //==================================================================================================
  // SCENE GRAPH HELPERS
  //==================================================================================================
  void renderSceneTransformUI(size_t sceneID);
  void rebuildSceneTransformNodes(size_t sceneID);  // Rebuild node list from current scene (handles dynamic changes)
  void markSceneTransformsDirty();           // Mark all scene transforms for rebuild (call after hierarchy changes)
  void applySceneTransform(size_t sceneID);  // Apply transform to root nodes
  void renderNodeHierarchy(int nodeIdx);
  void renderMeshInHierarchy(int meshIdx, int nodeIdx);
  void renderPrimitiveInHierarchy(int primIdx, int meshIdx, int nodeIdx);
  void renderLightInHierarchy(int lightIdx);
  void renderCameraInHierarchy(int cameraIdx);

  //==================================================================================================
  // SCENE LIST HELPERS (grouped flat lists)
  //==================================================================================================
  void renderNodesGroup();
  void renderMeshesGroup();
  void renderMaterialsGroup();
  void renderCamerasGroup();
  void renderLightsGroup();
  void renderTexturesGroup();    // Display-only (non-selectable)
  void renderImagesGroup();      // Display-only (non-selectable)
  void renderAnimationsGroup();  // Display-only (non-selectable)

  //==================================================================================================
  // CONTEXT MENUS
  //==================================================================================================
  void showNodeContextMenu(int nodeIdx);
  void showMeshContextMenu(int meshIdx);
  void showPrimitiveContextMenu(int primIdx, int meshIdx, int nodeIdx);
  void showMaterialContextMenu(int matIdx);

  //==================================================================================================
  // DIALOG RENDERING
  //==================================================================================================
  void renderRenameDialog();
  void renderDeleteConfirmation();

  //==================================================================================================
  // ICON HELPERS
  //==================================================================================================
  const char* getNodeIcon(int nodeIdx) const;
  const char* getMaterialIcon(int matIdx) const;

  //==================================================================================================
  // CACHE MANAGEMENT
  //==================================================================================================
  void buildCache(std::unordered_map<int, int>& cache, bool& dirtyFlag, int(tinygltf::Node::* member)) const;

  //==================================================================================================
  // HIERARCHY EXPANSION HELPERS
  //==================================================================================================
  void expandParentPath(int targetNodeIdx);
  bool markParentNodes(int currentNodeIdx, int targetNodeIdx);

  //==================================================================================================
  // MEMBER VARIABLES
  //==================================================================================================
  nvvkgltf::Scene* m_scene     = nullptr;
  SceneSelection*  m_selection = nullptr;
  UndoStack*       m_undoStack = nullptr;
  nvutils::Bbox    m_bbox;

  ViewTab m_currentTab = ViewTab::SceneGraph;

  // UI state
  std::unordered_set<int> m_expandedNodes;     // Only force-open these nodes (from selection)
  bool                    m_doScroll = false;  // Auto-scroll to selection

  // Scene transform state (per scene)
  struct SceneTransformState
  {
    glm::vec3              translation = glm::vec3(0.0f);
    glm::quat              rotation    = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3              scale       = glm::vec3(1.0f);
    std::vector<int>       nodeIds;
    std::vector<glm::mat4> baselineLocal;
    bool                   needsRebuild = true;  // Rebuild node list when hierarchy changes
  };
  std::vector<SceneTransformState> m_sceneTransforms;

  // Cached texture display names (for material texture dropdowns)
  std::vector<std::string> m_textureNames;

  // Performance caches: element → first node containing it
  std::unordered_map<int, int> m_meshToNodeMap;
  std::unordered_map<int, int> m_lightToNodeMap;
  std::unordered_map<int, int> m_cameraToNodeMap;
  bool                         m_meshToNodeMapDirty   = true;
  bool                         m_lightToNodeMapDirty  = true;
  bool                         m_cameraToNodeMapDirty = true;

  // Dialog state
  struct RenameState
  {
    std::string* targetName  = nullptr;
    int          nodeIndex   = -1;  // >= 0 when renaming a node (for undo support)
    char         buffer[256] = {};
  };
  RenameState m_renameState;
  bool        m_openRenamePopupNextFrame = false;

  int  m_pendingDeleteNode        = -1;
  bool m_openDeletePopupNextFrame = false;

  // Animation control
  AnimationControl m_animControl;
};
