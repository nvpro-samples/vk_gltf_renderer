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

//
// Scene browser UI panel. Renders the glTF node hierarchy as a tree view
// with drag-and-drop reparenting, context-menu editing (duplicate, delete,
// rename), node transform editing, material variant selection, and
// integration with the selection and animation systems.
//

#include "ui_scene_browser.hpp"
#include "ui_xmp.hpp"
#include "undo_redo.hpp"
#include "gltf_scene.hpp"
#include "gltf_scene_editor.hpp"
#include "gltf_scene_animation.hpp"
#include "tinygltf_utils.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <nvgui/fonts.hpp>
#include <nvutils/logger.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>
#include "ui_animation_control.hpp"

//==================================================================================================
// CONSTANTS
//==================================================================================================

static ImGuiTreeNodeFlags s_treeNodeFlags = ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_SpanFullWidth
                                            | ImGuiTreeNodeFlags_SpanTextWidth | ImGuiTreeNodeFlags_OpenOnArrow
                                            | ImGuiTreeNodeFlags_OpenOnDoubleClick;

//==================================================================================================
// HELPER FUNCTIONS
//==================================================================================================

static std::string getTextureDisplayName(const tinygltf::Model& model, int textureIndex)
{
  if(textureIndex < 0 || textureIndex >= static_cast<int>(model.textures.size()))
    return "None";

  const tinygltf::Texture& texture = model.textures[textureIndex];
  if(!texture.name.empty())
    return texture.name + " (tex " + std::to_string(textureIndex) + ")";

  if(texture.source >= 0 && texture.source < static_cast<int>(model.images.size()))
  {
    const tinygltf::Image& image = model.images[texture.source];
    if(!image.name.empty())
      return image.name + " (tex " + std::to_string(textureIndex) + ")";
  }

  return "Texture " + std::to_string(textureIndex);
}

//==================================================================================================
// INITIALIZATION
//==================================================================================================

void UiSceneBrowser::markCachesDirty()
{
  m_meshToNodeMapDirty   = true;
  m_lightToNodeMapDirty  = true;
  m_cameraToNodeMapDirty = true;
  markSceneTransformsDirty();
}

void UiSceneBrowser::setScene(nvvkgltf::Scene* scene)
{
  m_scene = scene;

  if(!scene)
    return;

  const tinygltf::Model& model = scene->getModel();

  // Build texture names cache
  m_textureNames.clear();
  m_textureNames.reserve(model.textures.size());
  for(int i = 0; i < static_cast<int>(model.textures.size()); ++i)
  {
    m_textureNames.push_back(getTextureDisplayName(model, i));
  }

  // Mark caches as dirty
  m_meshToNodeMapDirty   = true;
  m_lightToNodeMapDirty  = true;
  m_cameraToNodeMapDirty = true;

  // Initialize scene transforms (only TRS state, node list will be rebuilt dynamically)
  m_sceneTransforms.clear();
  m_sceneTransforms.resize(model.scenes.size());
  for(size_t sceneID = 0; sceneID < model.scenes.size(); ++sceneID)
  {
    SceneTransformState& state = m_sceneTransforms[sceneID];
    state                      = {};
    state.needsRebuild         = true;  // Mark for rebuild on first use
    // Note: nodeIds and baselineLocal are rebuilt dynamically in rebuildSceneTransformNodes()
    // to handle runtime node additions/deletions/moves
  }
}

//==================================================================================================
// MAIN RENDER
//==================================================================================================

void UiSceneBrowser::render(bool* show, bool isBusy)
{
  if(show && !*show)
    return;

  if(ImGui::Begin("Scene Browser", show))
  {
    if(!m_scene)
    {
      ImGui::TextDisabled("No scene loaded");
      ImGui::End();
      return;
    }

    // Show progress message if scene is being modified (merge/load operation in progress)
    if(isBusy)
    {
      ImGui::TextDisabled("Scene operation in progress...");
      ImGui::Text("Please wait while the scene is being modified.");
      ImGui::End();
      return;
    }

    if(ImGui::CollapsingHeader("Asset Info"))
    {
      renderAssetInfoTab();
    }

    // Animation Widget (at the top of the Scene Browser)
    if(m_scene->animation().hasAnimation())
    {
      if(ImGui::CollapsingHeader("Animation"))
      {
        m_animControl.onUI(m_scene);
      }
    }

    // Material Variant selection (at the top of the Scene Browser)
    if(m_scene->getVariants().size() > 0)
    {
      if(ImGui::CollapsingHeader("Material Variants"))
      {
        ImGui::PushID("Variants");
        for(size_t i = 0; i < m_scene->getVariants().size(); i++)
        {
          if(ImGui::Selectable(m_scene->getVariants()[i].c_str(), m_scene->getCurrentVariant() == i))
            m_scene->setCurrentVariant(int(i));  // Marks dirty in Scene
        }
        ImGui::PopID();
      }
    }

    // Tab bar
    if(ImGui::BeginTabBar("SceneBrowserTabs"))
    {

      if(ImGui::BeginTabItem("Scene Graph"))
      {
        m_currentTab = ViewTab::SceneGraph;
        renderSceneGraphTab();
        ImGui::EndTabItem();
      }

      if(ImGui::BeginTabItem("Scene List"))
      {
        m_currentTab = ViewTab::SceneList;
        renderSceneListTab();
        ImGui::EndTabItem();
      }

#ifndef NDEBUG
      if(ImGui::BeginTabItem(ICON_MS_BUG_REPORT " Debug"))
      {
        m_currentTab = ViewTab::Debug;
        renderDebugTab();
        ImGui::EndTabItem();
      }
#endif

      ImGui::EndTabBar();
    }

    // Keyboard shortcuts (when window is focused)
    if(ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && m_selection && m_selection->hasSelection())
    {
      auto sel = m_selection->getSelection();

      // Ctrl+D: Duplicate node
      if(ImGui::IsKeyPressed(ImGuiKey_D) && ImGui::GetIO().KeyCtrl)
      {
        if(sel.type == SceneSelection::SelectionType::eNode && sel.nodeIndex >= 0)
        {
          auto cmd = std::make_unique<DuplicateNodeCommand>(*m_scene, sel.nodeIndex, m_selection);
          m_undoStack->executeCommand(std::move(cmd));
          LOGI("Duplicated node %d (via Ctrl+D)\n", sel.nodeIndex);
          markCachesDirty();
        }
      }

      // Del: Delete node
      if(ImGui::IsKeyPressed(ImGuiKey_Delete))
      {
        if(sel.type == SceneSelection::SelectionType::eNode && sel.nodeIndex >= 0)
        {
          m_pendingDeleteNode        = sel.nodeIndex;
          m_openDeletePopupNextFrame = true;
        }
      }
    }

    // Dialogs (rendered outside tabs)
    renderRenameDialog();
    renderDeleteConfirmation();
  }
  ImGui::End();
}

//==================================================================================================
// ASSET INFO TAB
//==================================================================================================

void UiSceneBrowser::renderDebugTab()
{
  const tinygltf::Model& model = m_scene->getModel();

  ImGui::Text(ICON_MS_INFO " Render Node Statistics");
  ImGui::Separator();

  const auto& renderNodes = m_scene->getRenderNodes();
  const auto& renderPrims = m_scene->getRenderPrimitives();
  const auto& registry    = m_scene->getRenderNodeRegistry();

  ImGui::Text("Render Nodes: %zu", renderNodes.size());
  ImGui::Text("Render Primitives: %zu", renderPrims.size());
  ImGui::Text("Scene Nodes: %zu", model.nodes.size());
  ImGui::Text("Meshes: %zu", model.meshes.size());

  // Count expected render nodes (nodes with meshes * primitives per mesh)
  int nodesWithMeshes          = 0;
  int totalExpectedRenderNodes = 0;
  for(const auto& node : model.nodes)
  {
    if(node.mesh >= 0 && node.mesh < static_cast<int>(model.meshes.size()))
    {
      nodesWithMeshes++;
      totalExpectedRenderNodes += static_cast<int>(model.meshes[node.mesh].primitives.size());
    }
  }

  ImGui::Text("Nodes with Meshes: %d", nodesWithMeshes);
  ImGui::Text("Expected Render Nodes: %d", totalExpectedRenderNodes);

  // Warning if mismatch
  if(renderNodes.size() != static_cast<size_t>(totalExpectedRenderNodes))
  {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
    ImGui::Text(ICON_MS_WARNING " MISMATCH DETECTED!");
    ImGui::Text("  Actual: %zu render nodes", renderNodes.size());
    ImGui::Text("  Expected: %d render nodes", totalExpectedRenderNodes);
    ImGui::PopStyleColor();
  }
  else
  {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 1.0f, 0.3f, 1.0f));
    ImGui::Text(ICON_MS_CHECK_CIRCLE " Render nodes match expected count");
    ImGui::PopStyleColor();
  }

  ImGui::Separator();
  ImGui::Text(ICON_MS_BUILD " Repair Tools");
  ImGui::Separator();

#ifndef NDEBUG
  ImGui::Text(ICON_MS_VERIFIED " GPU sync validation active (auto-checked after every scene change)");
#endif

  if(ImGui::Button(ICON_MS_REFRESH " Rebuild All Render Nodes", ImVec2(-1, 0)))
  {
    m_scene->rebuildRenderNodes();
    LOGI("Render nodes manually rebuilt from Scene Browser\n");
  }

  if(ImGui::IsItemHovered())
  {
    ImGui::SetTooltip(
        "Force complete regeneration of all render nodes from the current scene state.\n\n"
        "Use this if you notice missing geometry after operations like:\n"
        "- Duplicate and delete node hierarchies\n"
        "- Complex reparenting operations\n"
        "- Any unexpected rendering issues\n\n"
        "This will rebuild the entire render node registry from scratch.");
  }

  ImGui::Separator();
  ImGui::Text(ICON_MS_LIST " Render Node Details");
  ImGui::Separator();

  // Show details in a scrollable region
  if(ImGui::BeginChild("RenderNodeDetails", ImVec2(0, 300), true))
  {
    static ImGuiTableFlags tableFlags = ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter
                                        | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable;

    if(ImGui::BeginTable("RenderNodesTable", 5, tableFlags))
    {
      ImGui::TableSetupScrollFreeze(0, 1);
      ImGui::TableSetupColumn("RN ID", ImGuiTableColumnFlags_WidthFixed, 50);
      ImGui::TableSetupColumn("Node", ImGuiTableColumnFlags_WidthFixed, 50);
      ImGui::TableSetupColumn("Node Name", ImGuiTableColumnFlags_WidthStretch);
      ImGui::TableSetupColumn("Prim", ImGuiTableColumnFlags_WidthFixed, 50);
      ImGui::TableSetupColumn("Material", ImGuiTableColumnFlags_WidthFixed, 70);
      ImGui::TableHeadersRow();

      for(int rnID = 0; rnID < static_cast<int>(renderNodes.size()); ++rnID)
      {
        const auto& rn          = renderNodes[rnID];
        auto        nodeAndPrim = registry.getNodeAndPrim(rnID);

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%d", rnID);

        ImGui::TableNextColumn();
        if(nodeAndPrim.has_value())
        {
          ImGui::Text("%d", nodeAndPrim->first);
        }
        else
        {
          ImGui::TextColored(ImVec4(1, 0, 0, 1), "ERR");
        }

        ImGui::TableNextColumn();
        if(rn.refNodeID >= 0 && rn.refNodeID < static_cast<int>(model.nodes.size()))
        {
          ImGui::Text("%s", model.nodes[rn.refNodeID].name.c_str());
        }
        else
        {
          ImGui::TextColored(ImVec4(1, 0, 0, 1), "INVALID");
        }

        ImGui::TableNextColumn();
        if(nodeAndPrim.has_value())
        {
          ImGui::Text("%d", rn.renderPrimID);
        }
        else
        {
          ImGui::TextColored(ImVec4(1, 0, 0, 1), "ERR");
        }

        ImGui::TableNextColumn();
        ImGui::Text("%d", rn.materialID);
      }

      ImGui::EndTable();
    }
  }
  ImGui::EndChild();
}

void UiSceneBrowser::renderAssetInfoTab()
{
  const tinygltf::Model& model = m_scene->getModel();
  const tinygltf::Asset& asset = model.asset;

  ImGui::Text("glTF Version: %s", asset.version.c_str());

  if(!asset.generator.empty())
    ImGui::TextWrapped("Generator: %s", asset.generator.c_str());

  if(!asset.copyright.empty())
    ImGui::Text("Copyright: %s", asset.copyright.c_str());

  if(!asset.minVersion.empty())
    ImGui::Text("Min Version: %s", asset.minVersion.c_str());

  // XMP button for asset-level metadata
  ui_xmp::renderInfoButton(&m_scene->getModel(), asset.extensions, "asset_xmp_popup");

  ImGui::Separator();

  // Full metadata panel (collapsible)
  ui_xmp::renderMetadataPanel(&m_scene->getModel());
}

//==================================================================================================
// SCENE GRAPH TAB
//==================================================================================================

void UiSceneBrowser::renderSceneGraphTab()
{
  const tinygltf::Model& model         = m_scene->getModel();
  static const float     textBaseWidth = ImGui::CalcTextSize("A").x;
  static ImGuiTableFlags s_tableFlags =
      ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV;

  if(ImGui::BeginTable("SceneGraphTable", 2, s_tableFlags))
  {
    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoHide);
    ImGui::TableSetupColumn(" ", ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_WidthFixed, textBaseWidth * 2.3f);
    ImGui::TableHeadersRow();

    const ImGuiTreeNodeFlags sceneTreeFlags =
        ImGuiTreeNodeFlags_SpanTextWidth | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_DefaultOpen;

    for(size_t sceneID = 0; sceneID < model.scenes.size(); sceneID++)
    {
      const tinygltf::Scene& scene = model.scenes[sceneID];

      ImGui::PushID(static_cast<int>(sceneID));
      ImGui::TableNextRow();
      ImGui::TableNextColumn();

      if(ImGui::TreeNodeEx("Scene", sceneTreeFlags, "%s", scene.name.c_str()))
      {
        // Context menu on Scene (add root-level nodes and lights)
        if(ImGui::BeginPopupContextItem())
        {
          if(m_scene)
          {
            if(ImGui::MenuItem(ICON_MS_ADD " Add Node"))
            {
              int newIndex = m_scene->editor().addNode("", -1);  // -1 = root level
              if(newIndex != -1)
              {
                LOGI("Added root node to scene\n");
                markSceneTransformsDirty();  // Scene root nodes have changed
                if(m_selection)
                  m_selection->selectNode(newIndex);
              }
            }

            ImGui::Separator();

            if(ImGui::MenuItem(ICON_MS_LIGHTBULB " Add Point Light"))
            {
              auto cmd = std::make_unique<AddLightCommand>(*m_scene, "point", "Point Light", -1, m_selection);
              m_undoStack->executeCommand(std::move(cmd));
              markCachesDirty();
            }
            if(ImGui::MenuItem(ICON_MS_LIGHTBULB " Add Directional Light"))
            {
              auto cmd = std::make_unique<AddLightCommand>(*m_scene, "directional", "Directional Light", -1, m_selection);
              m_undoStack->executeCommand(std::move(cmd));
              markCachesDirty();
            }
            if(ImGui::MenuItem(ICON_MS_LIGHTBULB " Add Spot Light"))
            {
              auto cmd = std::make_unique<AddLightCommand>(*m_scene, "spot", "Spot Light", -1, m_selection);
              m_undoStack->executeCommand(std::move(cmd));
              markCachesDirty();
            }
          }
          ImGui::EndPopup();
        }

        // Accept drops on Scene (move to root level)
        if(ImGui::BeginDragDropTarget())
        {
          const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("NODE_MOVE");
          if(payload)
          {
            int  draggedNodeIndex = *(int*)payload->Data;
            int  oldParent        = m_scene->editor().getNodeParent(draggedNodeIndex);
            auto cmd              = std::make_unique<ReparentNodeCommand>(*m_scene, draggedNodeIndex, oldParent, -1);
            m_undoStack->executeCommand(std::move(cmd));
            LOGI("Moved node %d to scene root\n", draggedNodeIndex);
            markCachesDirty();
          }
          ImGui::EndDragDropTarget();
        }

        ImGui::TableNextColumn();

        // Scene transform button
        if(ImGui::SmallButton(ICON_MS_TRANSFORM "##scene"))
        {
          ImGui::OpenPopup("scene_transform_popup");
        }
        renderSceneTransformUI(sceneID);

        // Render nodes in scene
        std::vector<int> nodesToRender = scene.nodes;
        for(int nodeId : nodesToRender)
        {
          renderNodeHierarchy(nodeId);
        }

        ImGui::TreePop();
        m_expandedNodes.clear();
      }
      ImGui::PopID();
    }

    ImGui::EndTable();
  }
}

//==================================================================================================
// SCENE LIST TAB
//==================================================================================================

void UiSceneBrowser::renderSceneListTab()
{
  renderNodesGroup();
  renderMeshesGroup();
  renderMaterialsGroup();
  renderCamerasGroup();
  renderLightsGroup();
  renderTexturesGroup();
  renderImagesGroup();
  renderAnimationsGroup();
}

//==================================================================================================
// SCENE TRANSFORM UI
//==================================================================================================

void UiSceneBrowser::renderSceneTransformUI(size_t sceneID)
{
  if(sceneID >= m_sceneTransforms.size())
    return;

  ImGui::SetNextWindowSizeConstraints(ImVec2(260.0f, 0.0f), ImVec2(600.0f, 600.0f));
  if(ImGui::BeginPopup("scene_transform_popup"))
  {
    SceneTransformState& state = m_sceneTransforms[sceneID];
    glm::vec3            euler = glm::degrees(glm::eulerAngles(state.rotation));
    bool                 modif = false;

    ImGui::Text("Scene Transform");
    ImGui::Separator();

    modif |= ImGui::DragFloat3("Translation", glm::value_ptr(state.translation), 0.01f * m_bbox.radius());
    modif |= ImGui::DragFloat3("Rotation", glm::value_ptr(euler), 0.1f);
    modif |= ImGui::DragFloat3("Scale", glm::value_ptr(state.scale), 0.01f);

    if(modif)
    {
      state.rotation = glm::quat(glm::radians(euler));
      applySceneTransform(sceneID);
    }

    if(ImGui::Button("Reset"))
    {
      state.translation = glm::vec3(0.0f);
      state.rotation    = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
      state.scale       = glm::vec3(1.0f);
      applySceneTransform(sceneID);
    }

    ImGui::EndPopup();
  }
}

//--------------------------------------------------------------------------------------------------
// Mark all scene transforms as needing rebuild (call after hierarchy changes)
//
void UiSceneBrowser::markSceneTransformsDirty()
{
  for(SceneTransformState& state : m_sceneTransforms)
  {
    state.needsRebuild = true;

    // Check if we have an active transform that needs to be frozen
    const bool hasActiveTransform = (state.translation != glm::vec3(0.0f))
                                    || (state.rotation != glm::quat(1.0f, 0.0f, 0.0f, 0.0f))
                                    || (state.scale != glm::vec3(1.0f));

    if(hasActiveTransform)
    {
      LOGW("Scene hierarchy modified while scene transform is active - freezing current transform.\n");
      LOGW("The transform has been permanently applied to existing nodes. Use 'Reset' to clear.\n");

      // Reset transform state to identity (freeze the current transform)
      state.translation = glm::vec3(0.0f);
      state.rotation    = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
      state.scale       = glm::vec3(1.0f);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Rebuild the node list and baseline matrices for a scene
// This is called before applying transforms to handle runtime node additions/deletions/moves
//
void UiSceneBrowser::rebuildSceneTransformNodes(size_t sceneID)
{
  if(!m_scene || sceneID >= m_sceneTransforms.size())
    return;

  const tinygltf::Model& model = m_scene->getModel();
  if(sceneID >= model.scenes.size())
    return;

  const tinygltf::Scene& gltfScene = model.scenes[sceneID];
  SceneTransformState&   state     = m_sceneTransforms[sceneID];

  // Clear and rebuild node list from current scene state
  // All nodes (both old transformed ones and new ones) are captured as-is
  state.nodeIds.clear();
  state.baselineLocal.clear();
  state.nodeIds.reserve(gltfScene.nodes.size());
  state.baselineLocal.reserve(gltfScene.nodes.size());

  for(int nodeId : gltfScene.nodes)
  {
    // Validate node index before accessing
    if(nodeId >= 0 && nodeId < static_cast<int>(model.nodes.size()))
    {
      state.nodeIds.push_back(nodeId);
      state.baselineLocal.push_back(tinygltf::utils::getNodeMatrix(model.nodes[nodeId]));
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Apply the scene transform to the root nodes in the scene
//
void UiSceneBrowser::applySceneTransform(size_t sceneID)
{
  if(!m_scene || sceneID >= m_sceneTransforms.size())
    return;

  SceneTransformState& state = m_sceneTransforms[sceneID];

  // Rebuild node list from current scene state if hierarchy changed
  if(state.needsRebuild)
  {
    rebuildSceneTransformNodes(sceneID);
    state.needsRebuild = false;
  }

  const glm::mat4 sceneMat = glm::translate(glm::mat4(1.0f), state.translation) * glm::mat4_cast(state.rotation)
                             * glm::scale(glm::mat4(1.0f), state.scale);

  const tinygltf::Model& model = m_scene->getModel();

  // Loop over all root nodes in the scene and apply the scene transform
  for(size_t i = 0; i < state.nodeIds.size(); ++i)
  {
    const int       nodeId   = state.nodeIds[i];
    tinygltf::Node& node     = m_scene->editor().getNodeForEdit(nodeId);
    const glm::mat4 newLocal = sceneMat * state.baselineLocal[i];

    glm::vec3 translation;
    glm::quat rotation;
    glm::vec3 scale;
    glm::vec3 skew;
    glm::vec4 perspective;

    if(glm::decompose(newLocal, scale, rotation, translation, skew, perspective))
    {
      // Successfully decomposed - set as TRS
      tinygltf::utils::setNodeTRS(node, translation, rotation, scale);
      node.matrix.clear();
    }
    else
    {
      // Decomposition failed - store as matrix
      node.matrix.resize(16);
      const float* data = glm::value_ptr(newLocal);
      for(size_t matIndex = 0; matIndex < 16; ++matIndex)
        node.matrix[matIndex] = static_cast<double>(data[matIndex]);
      node.translation.clear();
      node.rotation.clear();
      node.scale.clear();
    }

    m_scene->markNodeDirty(nodeId);
  }

  LOGI("Applied scene transform to %zu root nodes\n", state.nodeIds.size());
}

//==================================================================================================
// NODE HIERARCHY RENDERING
//==================================================================================================

void UiSceneBrowser::renderNodeHierarchy(int nodeIdx)
{
  if(!m_scene || nodeIdx < 0)
    return;

  const tinygltf::Model& model = m_scene->getModel();

  // Validate node index BEFORE accessing to prevent crashes during concurrent modifications
  if(nodeIdx >= static_cast<int>(model.nodes.size()))
    return;

  ImGui::TableNextRow();
  ImGui::TableNextColumn();

  ImGuiTreeNodeFlags  flags      = s_treeNodeFlags;
  bool                nodeOpen   = false;
  KHR_node_visibility visibility = {};

  // Scope for initial node access (before context menu that could reallocate vector)
  {
    const tinygltf::Node& node = model.nodes[nodeIdx];

    // Check if this node is selected
    if(m_selection)
    {
      auto sel = m_selection->getSelection();
      if(sel.type == SceneSelection::SelectionType::eNode && sel.nodeIndex == nodeIdx)
      {
        flags |= ImGuiTreeNodeFlags_Selected;
        if(m_doScroll)
        {
          ImGui::SetScrollHereY();
          m_doScroll = false;
        }
      }
    }

    // Only force open if in expansion set (from selection)
    if(m_expandedNodes.find(nodeIdx) != m_expandedNodes.end())
    {
      ImGui::SetNextItemOpen(true);
    }

    // Get visibility state
    visibility = tinygltf::utils::getNodeVisibility(node);

    // Get visibility icon
    const char* icon = visibility.visible ? ICON_MS_CATEGORY : ICON_MS_VISIBILITY_OFF;

    // Render tree node
    nodeOpen = ImGui::TreeNodeEx((void*)(intptr_t)nodeIdx, flags, "%s [%d] %s", icon, nodeIdx, node.name.c_str());

    // Handle node selection on click
    if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
    {
      if(m_selection)
      {
        m_selection->selectNode(nodeIdx);
      }
    }

    // Drag-and-drop source (for moving nodes)
    if(ImGui::BeginDragDropSource(ImGuiDragDropFlags_None))
    {
      ImGui::SetDragDropPayload("NODE_MOVE", &nodeIdx, sizeof(int));
      ImGui::Text("Move: %s", node.name.c_str());
      ImGui::EndDragDropSource();
    }

    // Accept drops on this node (makes it the new parent)
    if(ImGui::BeginDragDropTarget())
    {
      const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("NODE_MOVE");
      if(payload)
      {
        int draggedNodeIndex = *(int*)payload->Data;

        // Check for cycle
        if(m_scene->editor().wouldCreateCycle(draggedNodeIndex, nodeIdx))
        {
          LOGW("Cannot move node: would create a cycle\n");
        }
        else if(draggedNodeIndex != nodeIdx)
        {
          int  oldParent = m_scene->editor().getNodeParent(draggedNodeIndex);
          auto cmd       = std::make_unique<ReparentNodeCommand>(*m_scene, draggedNodeIndex, oldParent, nodeIdx);
          m_undoStack->executeCommand(std::move(cmd));
          LOGI("Moved node %d under node %d\n", draggedNodeIndex, nodeIdx);

          markCachesDirty();
        }
      }
      ImGui::EndDragDropTarget();
    }

    // Context menu (may reallocate nodes vector via duplicate/add/delete)
    showNodeContextMenu(nodeIdx);
  }  // End scope - 'node' reference no longer valid (context menu may have reallocated)

  // Re-fetch node reference after context menu (may have reallocated)
  if(nodeIdx < 0 || nodeIdx >= static_cast<int>(model.nodes.size()))
    return;
  const tinygltf::Node& node = model.nodes[nodeIdx];

  ImGui::TableNextColumn();

  // XMP button
  std::string popupId = "node_xmp_" + std::to_string(nodeIdx);
  ui_xmp::renderInfoButton(&m_scene->getModel(), node.extensions, popupId.c_str());

  if(nodeOpen)
  {
    // Render mesh
    if(node.mesh >= 0)
    {
      renderMeshInHierarchy(node.mesh, nodeIdx);
    }

    // Render light
    if(node.light >= 0)
    {
      renderLightInHierarchy(node.light);
    }

    // Render camera
    if(node.camera >= 0)
    {
      renderCameraInHierarchy(node.camera);
    }

    // Render children
    for(int child : node.children)
    {
      renderNodeHierarchy(child);
    }

    ImGui::TreePop();
  }
}

//==================================================================================================
// MESH HIERARCHY RENDERING
//==================================================================================================

void UiSceneBrowser::renderMeshInHierarchy(int meshIdx, int nodeIdx)
{
  if(!m_scene)
    return;

  const tinygltf::Model& model = m_scene->getModel();
  if(meshIdx < 0 || meshIdx >= static_cast<int>(model.meshes.size()))
    return;

  const tinygltf::Mesh& mesh = model.meshes[meshIdx];

  ImGui::TableNextRow();
  ImGui::TableNextColumn();

  ImGuiTreeNodeFlags flags = s_treeNodeFlags;

  // Force open if a primitive in this mesh is selected
  if(m_selection)
  {
    auto sel = m_selection->getSelection();
    if(sel.type == SceneSelection::SelectionType::ePrimitive && sel.nodeIndex == nodeIdx && sel.meshIndex == meshIdx)
    {
      ImGui::SetNextItemOpen(true);
    }
  }

  bool meshOpen = ImGui::TreeNodeEx((void*)(intptr_t)(meshIdx + 1000000), flags, "%s [%d] %s", ICON_MS_VIEW_IN_AR,
                                    meshIdx, mesh.name.c_str());

  // Context menu (must be right after TreeNodeEx)
  showMeshContextMenu(meshIdx);

  ImGui::TableNextColumn();

  // XMP button
  std::string popupId = "mesh_xmp_" + std::to_string(meshIdx);
  ui_xmp::renderInfoButton(&m_scene->getModel(), mesh.extensions, popupId.c_str());

  if(meshOpen)
  {
    // Render primitives
    for(int primIdx = 0; primIdx < static_cast<int>(mesh.primitives.size()); ++primIdx)
    {
      renderPrimitiveInHierarchy(primIdx, meshIdx, nodeIdx);
    }

    ImGui::TreePop();
  }
}

//==================================================================================================
// PRIMITIVE HIERARCHY RENDERING
//==================================================================================================

void UiSceneBrowser::renderPrimitiveInHierarchy(int primIdx, int meshIdx, int nodeIdx)
{
  if(!m_scene)
    return;

  const tinygltf::Model& model = m_scene->getModel();
  if(meshIdx < 0 || meshIdx >= static_cast<int>(model.meshes.size()))
    return;

  const tinygltf::Mesh& mesh = model.meshes[meshIdx];
  if(primIdx < 0 || primIdx >= static_cast<int>(mesh.primitives.size()))
    return;

  const tinygltf::Primitive& primitive = mesh.primitives[primIdx];
  int                        matIdx    = std::max(0, primitive.material);

  ImGui::TableNextRow();
  ImGui::TableNextColumn();

  // Get render node index for this primitive
  int renderNodeIdx = m_scene->getRenderNodeForPrimitive(nodeIdx, primIdx);

  bool isSelected = false;
  if(m_selection)
  {
    auto sel = m_selection->getSelection();
    isSelected =
        (sel.type == SceneSelection::SelectionType::ePrimitive && sel.renderNodeIndex == renderNodeIdx && renderNodeIdx >= 0);
  }

  // Scroll to selection
  if(isSelected && m_doScroll)
  {
    ImGui::SetScrollHereY();
    m_doScroll = false;
  }

  std::string primName = std::string(ICON_MS_SHAPE_LINE) + " Primitive " + std::to_string(primIdx);
  if(ImGui::Selectable(primName.c_str(), isSelected, ImGuiSelectableFlags_SpanAllColumns))
  {
    if(m_selection)
    {
      if(isSelected)
      {
        m_selection->clearSelection();
      }
      else
      {
        m_selection->selectPrimitive(renderNodeIdx, nodeIdx, primIdx, meshIdx);
      }
    }
  }

  // Context menu
  showPrimitiveContextMenu(primIdx, meshIdx, nodeIdx);

  ImGui::TableNextColumn();

  // Material icon and name
  if(matIdx >= 0 && matIdx < static_cast<int>(model.materials.size()))
  {
    ImGui::Text("%s", ICON_MS_BRUSH);
    if(ImGui::IsItemHovered())
    {
      ImGui::SetTooltip("Material: %s", model.materials[matIdx].name.c_str());
    }
  }
}

//==================================================================================================
// LIGHT/CAMERA HIERARCHY RENDERING
//==================================================================================================

void UiSceneBrowser::renderLightInHierarchy(int lightIdx)
{
  if(!m_scene)
    return;

  const tinygltf::Model& model = m_scene->getModel();
  if(lightIdx < 0 || lightIdx >= static_cast<int>(model.lights.size()))
    return;

  const tinygltf::Light& light = model.lights[lightIdx];

  ImGui::TableNextRow();
  ImGui::TableNextColumn();

  std::string label = std::string(ICON_MS_LIGHTBULB) + " Light: " + light.name;
  if(ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_SpanAllColumns))
  {
    if(m_selection)
    {
      m_selection->selectLight(lightIdx);
    }
  }

  ImGui::TableNextColumn();
}

void UiSceneBrowser::renderCameraInHierarchy(int cameraIdx)
{
  if(!m_scene)
    return;

  const tinygltf::Model& model = m_scene->getModel();
  if(cameraIdx < 0 || cameraIdx >= static_cast<int>(model.cameras.size()))
    return;

  const tinygltf::Camera& camera = model.cameras[cameraIdx];

  ImGui::TableNextRow();
  ImGui::TableNextColumn();

  std::string label = std::string(ICON_MS_CAMERA_ALT) + " Camera: " + camera.name;
  if(ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_SpanAllColumns))
  {
    if(m_selection)
    {
      m_selection->selectCamera(cameraIdx);
    }
  }

  ImGui::TableNextColumn();
}

//==================================================================================================
// SCENE LIST GROUPS
//==================================================================================================

void UiSceneBrowser::renderNodesGroup()
{
  if(!m_scene)
    return;

  const tinygltf::Model& model = m_scene->getModel();

  if(ImGui::CollapsingHeader((std::string(ICON_MS_CATEGORY) + " Nodes (" + std::to_string(model.nodes.size()) + ")").c_str()))
  {
    // Add scrollable child region with max height
    ImGui::BeginChild("NodesScrollRegion", ImVec2(0, 200), false, ImGuiWindowFlags_HorizontalScrollbar);

    for(int i = 0; i < static_cast<int>(model.nodes.size()); ++i)
    {
      const tinygltf::Node& node = model.nodes[i];

      bool isSelected = false;
      if(m_selection)
      {
        auto sel   = m_selection->getSelection();
        isSelected = (sel.type == SceneSelection::SelectionType::eNode && sel.nodeIndex == i);
      }

      std::string label = "[" + std::to_string(i) + "] " + node.name;
      if(ImGui::Selectable(label.c_str(), isSelected))
      {
        if(m_selection)
        {
          m_selection->selectNode(i);
        }
      }

      // Context menu must be right after Selectable
      showNodeContextMenu(i);
    }

    ImGui::EndChild();
  }
}

void UiSceneBrowser::renderMeshesGroup()
{
  if(!m_scene)
    return;

  const tinygltf::Model& model = m_scene->getModel();

  if(ImGui::CollapsingHeader((std::string(ICON_MS_VIEW_IN_AR) + " Meshes (" + std::to_string(model.meshes.size()) + ")").c_str()))
  {
    // Add scrollable child region with max height
    ImGui::BeginChild("MeshesScrollRegion", ImVec2(0, 200), false, ImGuiWindowFlags_HorizontalScrollbar);

    for(int i = 0; i < static_cast<int>(model.meshes.size()); ++i)
    {
      const tinygltf::Mesh& mesh = model.meshes[i];

      bool isSelected = false;
      if(m_selection)
      {
        auto sel   = m_selection->getSelection();
        isSelected = (sel.type == SceneSelection::SelectionType::eMesh && sel.meshIndex == i);
      }

      std::string label = "[" + std::to_string(i) + "] " + mesh.name;
      if(ImGui::Selectable(label.c_str(), isSelected))
      {
        if(m_selection)
        {
          m_selection->selectMesh(i);
        }
      }

      // Context menu must be right after Selectable
      showMeshContextMenu(i);
    }

    ImGui::EndChild();
  }
}

void UiSceneBrowser::renderMaterialsGroup()
{
  if(!m_scene)
    return;

  const tinygltf::Model& model = m_scene->getModel();

  if(ImGui::CollapsingHeader(
         (std::string(ICON_MS_BRUSH) + " Materials (" + std::to_string(model.materials.size()) + ")").c_str()))
  {
    // Add scrollable child region with max height
    ImGui::BeginChild("MaterialsScrollRegion", ImVec2(0, 200), false, ImGuiWindowFlags_HorizontalScrollbar);

    for(int i = 0; i < static_cast<int>(model.materials.size()); ++i)
    {
      const tinygltf::Material& material = model.materials[i];

      bool isSelected = false;
      if(m_selection)
      {
        auto sel   = m_selection->getSelection();
        isSelected = (sel.type == SceneSelection::SelectionType::eMaterial && sel.materialIndex == i);
      }

      std::string label = "[" + std::to_string(i) + "] " + material.name;
      if(ImGui::Selectable(label.c_str(), isSelected))
      {
        if(m_selection)
        {
          m_selection->selectMaterial(i);
        }
      }

      // Context menu must be right after Selectable
      showMaterialContextMenu(i);
    }

    ImGui::EndChild();
  }
}

void UiSceneBrowser::renderCamerasGroup()
{
  if(!m_scene)
    return;

  const tinygltf::Model& model = m_scene->getModel();

  if(ImGui::CollapsingHeader(
         (std::string(ICON_MS_CAMERA_ALT) + " Cameras (" + std::to_string(model.cameras.size()) + ")").c_str()))
  {
    // Add scrollable child region with max height
    ImGui::BeginChild("CamerasScrollRegion", ImVec2(0, 200), false, ImGuiWindowFlags_HorizontalScrollbar);

    for(int i = 0; i < static_cast<int>(model.cameras.size()); ++i)
    {
      const tinygltf::Camera& camera = model.cameras[i];

      bool isSelected = false;
      if(m_selection)
      {
        auto sel   = m_selection->getSelection();
        isSelected = (sel.type == SceneSelection::SelectionType::eCamera && sel.cameraIndex == i);
      }

      std::string label = "[" + std::to_string(i) + "] " + camera.name;
      if(ImGui::Selectable(label.c_str(), isSelected))
      {
        if(m_selection)
        {
          m_selection->selectCamera(i);
        }
      }
    }

    ImGui::EndChild();
  }
}

void UiSceneBrowser::renderLightsGroup()
{
  if(!m_scene)
    return;

  const tinygltf::Model& model = m_scene->getModel();

  bool lightsOpen = ImGui::CollapsingHeader(
      (std::string(ICON_MS_LIGHTBULB) + " Lights (" + std::to_string(model.lights.size()) + ")").c_str());

  // "+" button on the same line as the header for quick light creation
  if(m_undoStack && m_scene)
  {
    ImGui::SameLine(ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX() - ImGui::GetFrameHeight());
    ImGui::PushID("AddLightBtn");
    if(ImGui::SmallButton(ICON_MS_ADD))
    {
      ImGui::OpenPopup("AddLightPopup");
    }
    if(ImGui::BeginPopup("AddLightPopup"))
    {
      if(ImGui::MenuItem(ICON_MS_LIGHTBULB " Point Light"))
      {
        auto cmd = std::make_unique<AddLightCommand>(*m_scene, "point", "Point Light", -1, m_selection);
        m_undoStack->executeCommand(std::move(cmd));
        markCachesDirty();
      }
      if(ImGui::MenuItem(ICON_MS_LIGHTBULB " Directional Light"))
      {
        auto cmd = std::make_unique<AddLightCommand>(*m_scene, "directional", "Directional Light", -1, m_selection);
        m_undoStack->executeCommand(std::move(cmd));
        markCachesDirty();
      }
      if(ImGui::MenuItem(ICON_MS_LIGHTBULB " Spot Light"))
      {
        auto cmd = std::make_unique<AddLightCommand>(*m_scene, "spot", "Spot Light", -1, m_selection);
        m_undoStack->executeCommand(std::move(cmd));
        markCachesDirty();
      }
      ImGui::EndPopup();
    }
    ImGui::PopID();
  }

  if(lightsOpen)
  {
    ImGui::BeginChild("LightsScrollRegion", ImVec2(0, 200), false, ImGuiWindowFlags_HorizontalScrollbar);

    for(int i = 0; i < static_cast<int>(model.lights.size()); ++i)
    {
      const tinygltf::Light& light = model.lights[i];

      bool isSelected = false;
      if(m_selection)
      {
        auto sel   = m_selection->getSelection();
        isSelected = (sel.type == SceneSelection::SelectionType::eLight && sel.lightIndex == i);
      }

      std::string label = "[" + std::to_string(i) + "] " + light.name;
      if(ImGui::Selectable(label.c_str(), isSelected))
      {
        if(m_selection)
        {
          m_selection->selectLight(i);
        }
      }
    }

    ImGui::EndChild();
  }
}

void UiSceneBrowser::renderTexturesGroup()
{
  if(!m_scene)
    return;

  const tinygltf::Model& model = m_scene->getModel();

  if(ImGui::CollapsingHeader((std::string(ICON_MS_IMAGE) + " Textures (" + std::to_string(model.textures.size()) + ")").c_str()))
  {
    ImGui::TextDisabled("(Display only - not selectable)");

    // Add scrollable child region with max height
    ImGui::BeginChild("TexturesScrollRegion", ImVec2(0, 200), false, ImGuiWindowFlags_HorizontalScrollbar);

    for(int i = 0; i < static_cast<int>(model.textures.size()); ++i)
    {
      std::string label = "[" + std::to_string(i) + "] " + m_textureNames[i];
      ImGui::BulletText("%s", label.c_str());
    }

    ImGui::EndChild();
  }
}

void UiSceneBrowser::renderImagesGroup()
{
  if(!m_scene)
    return;

  const tinygltf::Model& model = m_scene->getModel();

  if(ImGui::CollapsingHeader((std::string(ICON_MS_PHOTO) + " Images (" + std::to_string(model.images.size()) + ")").c_str()))
  {
    ImGui::TextDisabled("(Display only - not selectable)");

    // Add scrollable child region with max height
    ImGui::BeginChild("ImagesScrollRegion", ImVec2(0, 200), false, ImGuiWindowFlags_HorizontalScrollbar);

    for(int i = 0; i < static_cast<int>(model.images.size()); ++i)
    {
      const tinygltf::Image& image = model.images[i];
      std::string label = "[" + std::to_string(i) + "] " + (image.uri.empty() ? "EMB " + std::to_string(i) : image.uri);
      if(image.width > 0 && image.height > 0)
      {
        label += " (" + std::to_string(image.width) + "x" + std::to_string(image.height) + ")";
      }
      ImGui::BulletText("%s", label.c_str());
    }

    ImGui::EndChild();
  }
}

void UiSceneBrowser::renderAnimationsGroup()
{
  if(!m_scene)
    return;

  const tinygltf::Model& model = m_scene->getModel();

  if(ImGui::CollapsingHeader(
         (std::string(ICON_MS_MOVIE) + " Animations (" + std::to_string(model.animations.size()) + ")").c_str()))
  {
    ImGui::TextDisabled("(Display only - not selectable)");

    // Add scrollable child region with max height
    ImGui::BeginChild("AnimationsScrollRegion", ImVec2(0, 200), false, ImGuiWindowFlags_HorizontalScrollbar);

    for(int i = 0; i < static_cast<int>(model.animations.size()); ++i)
    {
      const tinygltf::Animation& anim = model.animations[i];
      std::string label = "[" + std::to_string(i) + "] " + (anim.name.empty() ? "Animation " + std::to_string(i) : anim.name);
      ImGui::BulletText("%s", label.c_str());
    }

    ImGui::EndChild();
  }
}

//==================================================================================================
// CONTEXT MENUS
//==================================================================================================

void UiSceneBrowser::showNodeContextMenu(int nodeIdx)
{
  if(ImGui::BeginPopupContextItem())
  {
    if(m_scene)
    {
      if(ImGui::MenuItem(ICON_MS_CONTENT_COPY " Duplicate", "Ctrl+D"))
      {
        auto cmd = std::make_unique<DuplicateNodeCommand>(*m_scene, nodeIdx, m_selection);
        m_undoStack->executeCommand(std::move(cmd));
        LOGI("Duplicated node %d\n", nodeIdx);
        markCachesDirty();
      }

      if(ImGui::MenuItem(ICON_MS_DELETE " Delete", "Del"))
      {
        m_pendingDeleteNode        = nodeIdx;
        m_openDeletePopupNextFrame = true;
      }

      ImGui::Separator();

      // Add Child submenu
      if(ImGui::BeginMenu(ICON_MS_ADD " Add Child"))
      {
        if(ImGui::MenuItem("Empty Node"))
        {
          auto cmd = std::make_unique<AddNodeCommand>(*m_scene, "", nodeIdx, m_selection);
          m_undoStack->executeCommand(std::move(cmd));
          LOGI("Added child node to node %d\n", nodeIdx);
          markCachesDirty();
        }

        ImGui::Separator();

        if(ImGui::MenuItem(ICON_MS_LIGHTBULB " Point Light"))
        {
          auto cmd = std::make_unique<AddLightCommand>(*m_scene, "point", "Point Light", nodeIdx, m_selection);
          m_undoStack->executeCommand(std::move(cmd));
          markCachesDirty();
        }
        if(ImGui::MenuItem(ICON_MS_LIGHTBULB " Directional Light"))
        {
          auto cmd = std::make_unique<AddLightCommand>(*m_scene, "directional", "Directional Light", nodeIdx, m_selection);
          m_undoStack->executeCommand(std::move(cmd));
          markCachesDirty();
        }
        if(ImGui::MenuItem(ICON_MS_LIGHTBULB " Spot Light"))
        {
          auto cmd = std::make_unique<AddLightCommand>(*m_scene, "spot", "Spot Light", nodeIdx, m_selection);
          m_undoStack->executeCommand(std::move(cmd));
          markCachesDirty();
        }

        ImGui::EndMenu();
      }

      ImGui::Separator();

      if(ImGui::MenuItem(ICON_MS_EDIT " Rename"))
      {
        m_renameState.targetName   = &m_scene->editor().getNodeForEdit(nodeIdx).name;
        m_renameState.nodeIndex    = nodeIdx;
        m_openRenamePopupNextFrame = true;
      }
    }

    ImGui::EndPopup();
  }
}

void UiSceneBrowser::showMeshContextMenu(int meshIdx)
{
  if(ImGui::BeginPopupContextItem())
  {
    if(ImGui::MenuItem(ICON_MS_EDIT " Rename"))
    {
      m_renameState.targetName   = &m_scene->getModel().meshes[meshIdx].name;
      m_openRenamePopupNextFrame = true;
    }

    ImGui::EndPopup();
  }
}

void UiSceneBrowser::showPrimitiveContextMenu(int primIdx, int meshIdx, int nodeIdx)
{
  if(ImGui::BeginPopupContextItem())
  {
    ImGui::TextDisabled("Primitive %d operations", primIdx);
    ImGui::Separator();

    if(ImGui::MenuItem("Assign Material..."))
    {
      LOGI("Assign material to primitive (not yet implemented)\n");
    }

    ImGui::EndPopup();
  }
}

void UiSceneBrowser::showMaterialContextMenu(int matIdx)
{
  if(ImGui::BeginPopupContextItem())
  {
    if(ImGui::MenuItem(ICON_MS_EDIT " Rename"))
    {
      m_renameState.targetName   = &m_scene->editor().getMaterialForEdit(matIdx).name;
      m_openRenamePopupNextFrame = true;
    }

    ImGui::Separator();

    if(ImGui::MenuItem(ICON_MS_CONTENT_COPY " Copy Material ID"))
    {
      if(m_selection)
      {
        m_selection->copyMaterialToClipboard(matIdx);
      }
    }

    ImGui::EndPopup();
  }
}

//==================================================================================================
// DIALOGS
//==================================================================================================

void UiSceneBrowser::renderRenameDialog()
{
  // Open popup if flagged
  if(m_openRenamePopupNextFrame)
  {
    ImGui::OpenPopup("RenameDialog");
    m_openRenamePopupNextFrame = false;
    if(m_renameState.targetName)
    {
      size_t copyLen = std::min(m_renameState.targetName->length(), sizeof(m_renameState.buffer) - 1);
      std::memcpy(m_renameState.buffer, m_renameState.targetName->c_str(), copyLen);
      m_renameState.buffer[copyLen] = '\0';
    }
  }

  // Rename popup
  if(ImGui::BeginPopupModal("RenameDialog", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
  {
    ImGui::Text("Rename:");

    auto commitRename = [&]() {
      if(m_renameState.buffer[0] != '\0' && m_renameState.targetName)
      {
        std::string oldName = *m_renameState.targetName;
        std::string newName = std::string(m_renameState.buffer);
        if(m_renameState.nodeIndex >= 0)
        {
          auto cmd = std::make_unique<RenameNodeCommand>(*m_scene, m_renameState.nodeIndex, oldName, newName);
          m_undoStack->executeCommand(std::move(cmd));
        }
        else
        {
          *m_renameState.targetName = newName;
        }
        LOGI("Renamed: '%s' -> '%s'\n", oldName.c_str(), m_renameState.buffer);
        m_renameState = {};
        ImGui::CloseCurrentPopup();
      }
    };

    if(ImGui::InputText("##name", m_renameState.buffer, sizeof(m_renameState.buffer), ImGuiInputTextFlags_EnterReturnsTrue))
    {
      commitRename();
    }

    if(ImGui::Button(ICON_MS_CHECK " OK"))
    {
      commitRename();
    }
    ImGui::SameLine();
    if(ImGui::Button(ICON_MS_CANCEL " Cancel"))
    {
      m_renameState = {};
      ImGui::CloseCurrentPopup();
    }

    ImGui::EndPopup();
  }
}

void UiSceneBrowser::renderDeleteConfirmation()
{
  // Open popup if flagged
  if(m_openDeletePopupNextFrame)
  {
    ImGui::OpenPopup("DeleteConfirmation");
    m_openDeletePopupNextFrame = false;
  }

  // Delete confirmation popup
  if(ImGui::BeginPopupModal("DeleteConfirmation", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
  {
    if(m_pendingDeleteNode >= 0 && m_scene && m_pendingDeleteNode < static_cast<int>(m_scene->getModel().nodes.size()))
    {
      const tinygltf::Model& model = m_scene->getModel();
      ImGui::Text("Delete node '%s'?", model.nodes[m_pendingDeleteNode].name.c_str());
      ImGui::Text("This will delete the node and all its children.");
      ImGui::Text("You can undo this with Ctrl+Z.");
      ImGui::Separator();

      if(ImGui::Button(ICON_MS_DELETE " Delete", ImVec2(120, 0)))
      {
        LOGI("Deleting node %d: %s\n", m_pendingDeleteNode, model.nodes[m_pendingDeleteNode].name.c_str());

        auto cmd = std::make_unique<DeleteNodeCommand>(*m_scene, m_pendingDeleteNode, m_selection);
        m_undoStack->executeCommand(std::move(cmd));
        m_pendingDeleteNode = -1;

        markCachesDirty();

        ImGui::CloseCurrentPopup();
      }
      ImGui::SameLine();
      if(ImGui::Button(ICON_MS_CANCEL " Cancel", ImVec2(120, 0)))
      {
        m_pendingDeleteNode = -1;
        ImGui::CloseCurrentPopup();
      }
    }

    ImGui::EndPopup();
  }
}

//==================================================================================================
// SELECTION SYNCHRONIZATION
//==================================================================================================

void UiSceneBrowser::focusOnSelection()
{
  if(!m_selection->hasSelection())
  {
    return;
  }

  auto sel = m_selection->getSelection();

  if(sel.type == SceneSelection::SelectionType::ePrimitive || sel.type == SceneSelection::SelectionType::eNode)
  {
    // Expand entire parent path from root to selected node
    expandParentPath(sel.nodeIndex);

    // Switch to scene graph tab
    if(m_currentTab != ViewTab::SceneGraph)
      m_currentTab = ViewTab::SceneGraph;

    // Flag for scroll on next render
    m_doScroll = true;
  }
}

//--------------------------------------------------------------------------------------------------
// Expand all parent nodes in the path to targetNodeIdx (simple walk-up approach)
//
void UiSceneBrowser::expandParentPath(int targetNodeIdx)
{
  if(targetNodeIdx < 0 || !m_scene)
    return;

  m_expandedNodes.insert(targetNodeIdx);

  // Simple while loop: walk up parent chain and add all to expanded set
  int nodeIdx = targetNodeIdx;
  while(nodeIdx >= 0)
  {
    int parentIdx = m_scene->editor().getNodeParent(nodeIdx);
    if(parentIdx >= 0)
    {
      m_expandedNodes.insert(parentIdx);  // Mark parent as expanded
    }
    nodeIdx = parentIdx;  // Move up to parent
  }
}

//--------------------------------------------------------------------------------------------------
// Recursive helper: no longer needed with simple walk-up approach
//
bool UiSceneBrowser::markParentNodes(int currentNodeIdx, int targetNodeIdx)
{
  // Kept for interface compatibility but not used
  return false;
}

//==================================================================================================
// CACHE MANAGEMENT
//==================================================================================================

void UiSceneBrowser::buildCache(std::unordered_map<int, int>& cache, bool& dirtyFlag, int(tinygltf::Node::* member)) const
{
  if(!dirtyFlag || !m_scene)
    return;

  cache.clear();
  const tinygltf::Model& model = m_scene->getModel();

  for(size_t i = 0; i < model.nodes.size(); ++i)
  {
    const tinygltf::Node& node         = model.nodes[i];
    int                   elementIndex = node.*member;
    if(elementIndex >= 0)
    {
      cache[elementIndex] = static_cast<int>(i);  // First node with this element
    }
  }

  dirtyFlag = false;
}

int UiSceneBrowser::getNodeForMesh(int meshIdx)
{
  buildCache(m_meshToNodeMap, m_meshToNodeMapDirty, &tinygltf::Node::mesh);
  auto it = m_meshToNodeMap.find(meshIdx);
  return (it != m_meshToNodeMap.end()) ? it->second : -1;
}

int UiSceneBrowser::getNodeForLight(int lightIdx)
{
  buildCache(m_lightToNodeMap, m_lightToNodeMapDirty, &tinygltf::Node::light);
  auto it = m_lightToNodeMap.find(lightIdx);
  return (it != m_lightToNodeMap.end()) ? it->second : -1;
}

int UiSceneBrowser::getNodeForCamera(int camIdx)
{
  buildCache(m_cameraToNodeMap, m_cameraToNodeMapDirty, &tinygltf::Node::camera);
  auto it = m_cameraToNodeMap.find(camIdx);
  return (it != m_cameraToNodeMap.end()) ? it->second : -1;
}

//==================================================================================================
// ICON HELPERS
//==================================================================================================

const char* UiSceneBrowser::getNodeIcon(int nodeIdx) const
{
  if(!m_scene)
    return ICON_MS_CATEGORY;

  const tinygltf::Model& model = m_scene->getModel();
  if(nodeIdx < 0 || nodeIdx >= static_cast<int>(model.nodes.size()))
    return ICON_MS_CATEGORY;

  const tinygltf::Node& node = model.nodes[nodeIdx];
  KHR_node_visibility   vis  = tinygltf::utils::getNodeVisibility(node);

  return vis.visible ? ICON_MS_CATEGORY : ICON_MS_VISIBILITY_OFF;
}

const char* UiSceneBrowser::getMaterialIcon(int matIdx) const
{
  return ICON_MS_BRUSH;
}
