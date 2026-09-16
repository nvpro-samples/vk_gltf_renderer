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
#include "ui_gltf_labels.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <nvgui/fonts.hpp>
#include <nvutils/logger.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>

#include <algorithm>

//==================================================================================================
// CONSTANTS
//==================================================================================================

static ImGuiTreeNodeFlags s_treeNodeFlags = ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_SpanFullWidth
                                            | ImGuiTreeNodeFlags_SpanTextWidth | ImGuiTreeNodeFlags_OpenOnArrow
                                            | ImGuiTreeNodeFlags_OpenOnDoubleClick;

//==================================================================================================
// HELPER FUNCTIONS
//==================================================================================================

// Above this many siblings a scene-graph level -- the scene's root list or one node's children --
// is virtualized with ImGuiListClipper (see walkSiblingRows). Below it we render the level in full
// every frame, which is what makes scrolling and arbitrarily deep expansion "just work" -- the
// clipper cannot, because it models the scroll extent as (sibling count x row height) and so
// ignores the height of expanded subtrees. That undercount is harmless only when there are enough
// siblings that a scroll region exists regardless (the huge-flat-list case this threshold targets);
// for ordinary levels we must render fully or the scroll bar collapses. The full walk is trivial at
// this size and only runs while the Scene Graph tab is open.
static constexpr size_t kSceneGraphVirtualizeThreshold = 200;

// Uniform height, in pixels, of a single scene-graph table row.
//
// Only used on the virtualized path: inside a table ImGuiListClipper requires every clipped item to
// be exactly one row of a constant, known height. It otherwise measures item 0 to infer the height
// and asserts (ImGuiListClipper_StepInternal, table branch) if that item did not advance the cursor
// by exactly one row -- which an expanded root does, since it emits several rows. We sidestep the
// measurement by pinning each root row to this height and handing the same value to clipper.Begin().
// Under the default style this equals the natural text row height (ItemSpacing.y == 2*CellPadding.y),
// so it does not change how the graph looks; the max() guards themes with larger cell padding,
// keeping the forced height >= the natural content.
static float sceneGraphRowHeight()
{
  const ImGuiStyle& style = ImGui::GetStyle();
  return std::max(ImGui::GetTextLineHeightWithSpacing(), ImGui::GetTextLineHeight() + style.CellPadding.y * 2.0f);
}

// Emits one level of sibling rows, calling render(index, rowHeight) per row that must be built.
// When `virtualize` is set only the rows inside the scroll region are built, keeping the per-frame
// cost O(visible rows) instead of O(siblings) -- the difference between a responsive and an
// unusable Scene Graph on a city-scale scene, where a single node can own hundreds of thousands of
// children. The fixed row height lets the clipper skip measuring item 0, which is required because
// an expanded sibling emits several rows and would otherwise trip the clipper's one-row-per-item
// assert inside a table (ImGuiListClipper_StepInternal, table branch).
//
// TODO: known limitation of the uniform-height model. An expanded sibling still renders its whole
// subtree, but the clipper keeps modelling it as one row, so the scroll extent is short by those
// extra rows and the rows it positions from that model can shift or overlap near an expanded
// sibling while scrolling. It is bounded by the height of what is expanded and only shows up past
// the threshold; fixing it properly means clipping a flattened list of visible rows rather than a
// sibling list, which is a restructure of the whole hierarchy walk. Do not "fix" it by declining to
// virtualize a list that contains an open node: focusOnSelection() opens the selected node, so
// every viewport pick would drop a huge list straight back to the full walk this exists to avoid.
template <class RenderRow>
static void walkSiblingRows(size_t count, bool virtualize, RenderRow&& render)
{
  if(!virtualize)
  {
    for(size_t i = 0; i < count; ++i)
      render(i, 0.0f);
    return;
  }

  const float      rowHeight = sceneGraphRowHeight();
  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(count), rowHeight);
  while(clipper.Step())
  {
    for(int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i)
      render(static_cast<size_t>(i), rowHeight);
  }
}

//==================================================================================================
// INITIALIZATION
//==================================================================================================

void UiSceneBrowser::markCachesDirty()
{
  m_meshToNodeMapDirty   = true;
  m_lightToNodeMapDirty  = true;
  m_cameraToNodeMapDirty = true;
  ++m_revision;  // invalidate the Elements view + derived-stat tables (see ui_scene_browser_elements.cpp)
  markSceneTransformsDirty();
}

void UiSceneBrowser::setScene(nvvkgltf::Scene* scene)
{
  m_scene = scene;

  if(!scene)
  {
    // Unload: the authoritative invalidation point (GltfRenderer::cleanupScene always passes
    // through here before the next scene is handed over). Retire the id the tree rows are scoped
    // under, so the incoming scene starts collapsed instead of inheriting this one's expansion.
    // A merge keeps the same Scene and never unloads, so it keeps the tree the user had open.
    ++m_treeStateGeneration;
    m_expandedNodes.clear();
    m_doScroll = false;
    return;
  }

  // Mark caches as dirty
  m_meshToNodeMapDirty   = true;
  m_lightToNodeMapDirty  = true;
  m_cameraToNodeMapDirty = true;
  m_elementTypes.clear();  // rebuilt lazily for the new scene by ensureElementRegistry()
  ++m_revision;

  // Initialize scene transforms (only TRS state, node list will be rebuilt dynamically)
  const tinygltf::Model& model = scene->getModel();
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

      if(ImGui::BeginTabItem("Elements"))
      {
        m_currentTab = ViewTab::Elements;
        renderElementsTab();
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

    // Dialogs (rendered outside tabs). Note: the "Add Primitive" modal is intentionally NOT rendered
    // here -- it is driven from an always-rendered top-level UI path (see GltfRenderer::renderUI)
    // so it still surfaces when this window is hidden/collapsed (e.g. triggered from the menu bar).
    renderRenameDialog();
  }
  ImGui::End();
}

// Public entry points for the image viewer. Rendered from an always-on top-level path (see
// GltfRenderer::renderUI) so the modal surfaces regardless of the active tab or window visibility --
// e.g. when opened by a click on an inspector texture thumbnail.
void UiSceneBrowser::openImageViewer(int imageIndex)
{
  m_viewerImageIndex = imageIndex;
  m_openImageViewer  = true;
}

void UiSceneBrowser::showImageViewer()
{
  renderImageViewer();
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

    // Scope every row under the current load id. ImGui keeps tree open/closed state in per-window
    // storage keyed by the id we hand TreeNodeEx -- the node index -- so without this a freshly
    // loaded scene inherits the previous one's expansion ("node 0 was open, so node 0 is open").
    // Bumping the id on unload (see setScene) gives each loaded scene a collapsed tree. It sits
    // inside BeginTable so the table keeps its own identity, and therefore its column widths.
    ImGui::PushID(m_treeStateGeneration);

    const ImGuiTreeNodeFlags sceneTreeFlags =
        ImGuiTreeNodeFlags_SpanTextWidth | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_DefaultOpen;

    for(size_t sceneID = 0; sceneID < model.scenes.size(); sceneID++)
    {
      const tinygltf::Scene& scene = model.scenes[sceneID];

      ImGui::PushID(static_cast<int>(sceneID));
      ImGui::TableNextRow();
      ImGui::TableNextColumn();

      // A focus request lives for exactly this frame (it is consumed at the end of the pass, see
      // below), so it has to be able to reach its row now. A collapsed scene row would swallow it
      // and the selection would never be revealed -- open the scene rows for that one frame.
      if(m_doScroll)
        ImGui::SetNextItemOpen(true);

      if(ImGui::TreeNodeEx("Scene", sceneTreeFlags, "%s", scene.name.c_str()))
      {
        // Context menu on Scene (add objects at the scene root)
        if(ImGui::BeginPopupContextItem())
        {
          if(m_scene)
          {
            renderAddObjectMenu(-1, ICON_MS_ADD " Add");  // -1 = scene root
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

        // Render the scene's root nodes. We iterate a snapshot, not scene.nodes directly, because
        // rendering a row can immediately edit that same vector: the per-node context menu
        // (Duplicate / Add) and drag-drop reparenting run their editor commands inline, and those
        // push/erase in model.scenes[...].nodes (see gltf_scene_editor.cpp). Iterating the live
        // vector would reallocate it mid-loop and desync the clipper's item count. Copying is O(root
        // count) but a trivial memcpy, and the edit simply takes effect next frame -- the same
        // apply-after-traversal contract Delete/Rename already use via their "next frame" flags.
        std::vector<int> nodesToRender = scene.nodes;

        // Virtualize only huge root lists. m_doScroll forces a full walk so SetScrollHereY() on the
        // (possibly clipped-away) selected row can fire; small graphs also render fully so scrolling
        // and deep expansion stay correct -- the clipper models scroll extent as (root count x row
        // height) and would collapse the scroll bar for an expanded ordinary scene.
        const bool virtualizeRoots = !m_doScroll && nodesToRender.size() > kSceneGraphVirtualizeThreshold;
        walkSiblingRows(nodesToRender.size(), virtualizeRoots, [&](size_t i, float rowHeight) {
          renderNodeHierarchy(nodesToRender[i], rowHeight, /*canVirtualizeChildren=*/!virtualizeRoots);
        });

        ImGui::TreePop();
      }
      ImGui::PopID();
    }

    ImGui::PopID();  // m_treeStateGeneration
    ImGui::EndTable();
  }

  // Both are one-shot requests from focusOnSelection(), consumed by the walk above: the force-open
  // set has done its job once the parent chain is open (ImGui remembers the open state from there),
  // and the scroll has fired on the selected row. Clearing them here rather than at the row that
  // consumed them is what guarantees they cannot get stuck: a selection that never reaches a row --
  // cleared again before this frame, or pointing at a node the model no longer has -- would
  // otherwise leave m_doScroll set forever, permanently disabling virtualization. The walk above
  // force-opens the scene rows while the request is live, so a collapsed scene cannot be the reason
  // a row goes unvisited.
  m_expandedNodes.clear();
  m_doScroll = false;
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
    const int nodeId = state.nodeIds[i];
    // glTF 2.1: referenced external-asset nodes are read-only; the gizmo must not move them.
    if(m_scene->isNodeReadOnly(nodeId))
      continue;
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

// rowHeight > 0 pins this node's table row to a fixed height (used only when this level is
// virtualized with ImGuiListClipper, which needs a constant row height); 0 = natural height.
// canVirtualizeChildren is false when this row was emitted from inside a clipper, so the child list
// below does not open a second clipper nested inside it.
void UiSceneBrowser::renderNodeHierarchy(int nodeIdx, float rowHeight, bool canVirtualizeChildren)
{
  if(!m_scene || nodeIdx < 0)
    return;

  const tinygltf::Model& model = m_scene->getModel();

  // Validate node index BEFORE accessing to prevent crashes during concurrent modifications
  if(nodeIdx >= static_cast<int>(model.nodes.size()))
    return;

  // rowHeight > 0 (virtualized path) pins the row to the constant height the clipper was told (see
  // sceneGraphRowHeight); 0 leaves the row at its natural height. Children recurse with 0 -- only
  // the clipped root rows need a fixed height.
  ImGui::TableNextRow(ImGuiTableRowFlags_None, rowHeight);
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

    // Render children, virtualizing long sibling lists the same way the root list is (a single
    // node holding a whole city's worth of children is common in exported scenes). The child list
    // is re-read from the model on every row because a row can edit it inline (context-menu
    // Duplicate / Add, drag-drop reparent), which reallocates model.nodes and would dangle a
    // reference taken before the loop.
    const bool virtualizeChildren =
        canVirtualizeChildren && !m_doScroll && node.children.size() > kSceneGraphVirtualizeThreshold;
    walkSiblingRows(node.children.size(), virtualizeChildren, [&](size_t i, float childRowHeight) {
      const std::vector<int>& children = m_scene->getModel().nodes[nodeIdx].children;
      if(i < children.size())
        renderNodeHierarchy(children[i], childRowHeight, canVirtualizeChildren && !virtualizeChildren);
    });

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

//--------------------------------------------------------------------------------------------------
// Image viewer: large aspect-fit preview + metadata, with replace-from-file and reload. Opened from
// the Images panel. Model edits set DirtyFlags::texturesChanged; the GPU rebuild fires at end of render().
void UiSceneBrowser::renderImageViewer()
{
  if(!m_scene)
    return;
  const tinygltf::Model& model = m_scene->getModel();

  if(m_openImageViewer)
  {
    ImGui::OpenPopup("Image Viewer");
    m_openImageViewer = false;
  }

  ImGui::SetNextWindowSize(ImVec2(560.0f, 640.0f), ImGuiCond_Appearing);
  bool open = true;
  if(!ImGui::BeginPopupModal("Image Viewer", &open, ImGuiWindowFlags_None))
    return;

  const int i = m_viewerImageIndex;
  if(i < 0 || i >= static_cast<int>(model.images.size()))
  {
    ImGui::TextDisabled("No image.");
    if(ImGui::Button("Close"))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
    return;
  }

  const tinygltf::Image& image = model.images[i];

  // Metadata
  ImGui::Text("Image %d", i);
  ImGui::SameLine();
  ImGui::TextDisabled("(%s)", image.uri.empty() ? "embedded" : "external");
  ImGui::TextWrapped("%s", uigltf::imageDisplayName(model, i).c_str());
  if(image.width > 0 && image.height > 0)
    ImGui::Text("Resolution: %d x %d", image.width, image.height);
  ImGui::Separator();

  // Aspect-fit preview into the region above the button row.
  const float buttonsH    = ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y;
  ImVec2      avail       = ImGui::GetContentRegionAvail();
  avail.y                 = std::max(64.0f, avail.y - buttonsH);
  const ImTextureID thumb = m_getImageThumbnail ? m_getImageThumbnail(i) : 0;
  if(thumb != 0 && image.width > 0 && image.height > 0)
  {
    const float aspect = static_cast<float>(image.width) / static_cast<float>(image.height);
    float       w      = avail.x;
    float       h      = w / aspect;
    if(h > avail.y)
    {
      h = avail.y;
      w = h * aspect;
    }
    if(const float offset = (avail.x - w) * 0.5f; offset > 0.0f)
      ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);
    ImGui::Image(thumb, ImVec2(w, h));
  }
  else
  {
    ImGui::TextDisabled("(no GPU preview - image not resident)");
  }

  // Actions
  ImGui::BeginDisabled(!m_host.canPickImage());
  if(ImGui::Button(ICON_MS_FILE_OPEN " Replace..."))
  {
    const std::filesystem::path path = m_host.pickImage();
    if(!path.empty())
    {
      const tinygltf::Image oldImage = image;  // copy before replace
      std::string           err;
      if(m_scene->editor().replaceImageFromFile(i, path, &err))
      {
        if(m_undoStack)
          m_undoStack->pushExecuted(std::make_unique<ReplaceImageCommand>(*m_scene, i, oldImage, m_scene->getModel().images[i],
                                                                          "Replace image " + std::to_string(i)));
      }
      else
      {
        LOGE("Replace image failed: %s\n", err.c_str());
        m_host.toast("Replace image failed: " + err, true);
      }
    }
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if(ImGui::Button(ICON_MS_REFRESH " Reload"))
  {
    m_scene->getDirtyFlags().texturesChanged = true;  // force re-decode from the (external) URI on rebuild
  }
  ImGui::SameLine();
  if(ImGui::Button("Close"))
    ImGui::CloseCurrentPopup();

  ImGui::EndPopup();
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
      const bool readOnly = m_scene->isNodeReadOnly(nodeIdx);

      // glTF 2.1: "break the lock" -- make a referenced external asset editable. Offered on read-only
      // merged nodes and on the instance node itself, and kept enabled while the rest of the menu is
      // disabled below for read-only nodes.
      if(readOnly || m_scene->isExternalAssetInstance(nodeIdx))
      {
        if(ImGui::MenuItem(ICON_MS_LOCK_OPEN " Make Editable"))
        {
          if(m_scene->editor().makeExternalAssetEditable(nodeIdx))
          {
            m_undoStack->clear();  // model mutated outside the command system
            markCachesDirty();
          }
        }
        ImGui::SetItemTooltip("Break the external-asset link and unlock all instances of this asset for editing (saved inline on next save)");
        ImGui::Separator();
      }

      // glTF 2.1: referenced external-asset subtrees are read-only -- disable structural edits.
      ImGui::BeginDisabled(readOnly);

      if(ImGui::MenuItem(ICON_MS_CONTENT_COPY " Duplicate", "Ctrl+D"))
      {
        auto cmd = std::make_unique<DuplicateNodeCommand>(*m_scene, nodeIdx, m_selection);
        m_undoStack->executeCommand(std::move(cmd));
        LOGI("Duplicated node %d\n", nodeIdx);
        markCachesDirty();
      }

      if(ImGui::MenuItem(ICON_MS_DELETE " Delete", "Del"))
      {
        if(m_pendingDeleteNode && m_openDeletePopupNextFrame)
        {
          *m_pendingDeleteNode        = nodeIdx;
          *m_openDeletePopupNextFrame = true;
        }
      }

      ImGui::Separator();

      renderAddObjectMenu(nodeIdx, ICON_MS_ADD " Add Child");  // Empty / Mesh / Light under this node

      ImGui::Separator();

      if(ImGui::MenuItem(ICON_MS_EDIT " Rename"))
      {
        m_renameState.targetName   = &m_scene->editor().getNodeForEdit(nodeIdx).name;
        m_renameState.nodeIndex    = nodeIdx;
        m_openRenamePopupNextFrame = true;
      }

      ImGui::EndDisabled();
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


//--------------------------------------------------------------------------------------------------
// Create catalog: Empty Node + Mesh submenu + Light submenu. Shared by the menu-bar "Create" menu,
// the node context "Add Child" and the scene-root context "Add", so meshes and lights live in the
// same taxonomy and the object list is defined in exactly one place. parentIndex = -1 = scene root.
//--------------------------------------------------------------------------------------------------
void UiSceneBrowser::renderCreateCatalog(int parentIndex)
{
  if(ImGui::MenuItem("Empty Node"))
    addEmptyNode(parentIndex);

  if(ImGui::BeginMenu(ICON_MS_DEPLOYED_CODE " Mesh"))
  {
    renderAddPrimitiveItems(parentIndex);
    ImGui::EndMenu();
  }

  if(ImGui::BeginMenu(ICON_MS_LIGHTBULB " Light"))
  {
    renderAddLightItems(parentIndex);
    ImGui::EndMenu();
  }
}

// Wrap the catalog in a labeled submenu (used by the context menus).
void UiSceneBrowser::renderAddObjectMenu(int parentIndex, const char* menuLabel)
{
  if(ImGui::BeginMenu(menuLabel))
  {
    renderCreateCatalog(parentIndex);
    ImGui::EndMenu();
  }
}

// One item per nvvkgltf::kPrimitiveKinds; clicking opens the size/subdivision popup.
void UiSceneBrowser::renderAddPrimitiveItems(int parentIndex)
{
  for(const auto& info : nvvkgltf::kPrimitiveKinds)
  {
    if(ImGui::MenuItem(info.name))
      requestAddPrimitive(info.kind, parentIndex);
  }
}

// One item per nvvkgltf::kLightKinds; clicking adds the light immediately (no parameter popup).
void UiSceneBrowser::renderAddLightItems(int parentIndex)
{
  for(const auto& info : nvvkgltf::kLightKinds)
  {
    if(ImGui::MenuItem(info.name))
      addLight(info.type, info.name, parentIndex);
  }
}

//--------------------------------------------------------------------------------------------------
// Undoable empty-node creation. Ensures a scene exists first (menu-bar "Create" with nothing loaded)
// and asks the renderer to build GPU resources if this is the first object in a fresh scene.
//--------------------------------------------------------------------------------------------------
void UiSceneBrowser::addEmptyNode(int parentIndex)
{
  if(m_onBeforeCreate)
    m_onBeforeCreate();
  if(!m_scene || !m_undoStack)
    return;

  auto cmd = std::make_unique<AddNodeCommand>(*m_scene, "", parentIndex, m_selection);
  m_undoStack->executeCommand(std::move(cmd));
  if(m_onGeometryChanged)
    m_onGeometryChanged();  // first object in an empty scene needs the initial GPU build
  markSceneTransformsDirty();
  markCachesDirty();
}

//--------------------------------------------------------------------------------------------------
// Undoable light creation. Same ensure-scene / first-build handling as addEmptyNode.
//--------------------------------------------------------------------------------------------------
void UiSceneBrowser::addLight(const char* lightType, const char* lightName, int parentIndex)
{
  if(m_onBeforeCreate)
    m_onBeforeCreate();
  if(!m_scene || !m_undoStack)
    return;

  auto cmd = std::make_unique<AddLightCommand>(*m_scene, lightType, lightName, parentIndex, m_selection);
  m_undoStack->executeCommand(std::move(cmd));
  if(m_onGeometryChanged)
    m_onGeometryChanged();  // first object in an empty scene needs the initial GPU build
  markSceneTransformsDirty();
  markCachesDirty();
}

//--------------------------------------------------------------------------------------------------
// Stash the requested primitive and open the parameter popup on the next frame (deferred-open pattern).
// Ensures a scene exists first so the menu-bar "Create > Mesh" works with nothing loaded.
//--------------------------------------------------------------------------------------------------
void UiSceneBrowser::requestAddPrimitive(nvvkgltf::PrimitiveKind kind, int parentIndex)
{
  if(m_onBeforeCreate)
    m_onBeforeCreate();

  m_pendingPrimitiveKind   = kind;
  m_pendingPrimitiveParent = parentIndex;

  // Default the size to a fraction of the scene so the primitive lands at a sensible scale.
  const float radius            = m_bbox.radius();
  const float defaultSize       = (radius > 0.0f) ? radius * 0.2f : 1.0f;
  m_pendingPrimitiveParams      = nvvkgltf::PrimitiveParams{};
  m_pendingPrimitiveParams.size = defaultSize;

  m_openAddPrimitivePopupNextFrame = true;
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

//--------------------------------------------------------------------------------------------------
// Parameter popup for adding a procedural primitive. On "Add", runs an undoable AddPrimitiveCommand
// and asks the renderer (via callback) to (re)create the GPU geometry for the new primitive.
//--------------------------------------------------------------------------------------------------
void UiSceneBrowser::showAddPrimitivePopup()
{
  if(m_openAddPrimitivePopupNextFrame)
  {
    ImGui::OpenPopup("Add Primitive");
    m_openAddPrimitivePopupNextFrame = false;
  }

  if(ImGui::BeginPopupModal("Add Primitive", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
  {
    ImGui::Text("Add %s", nvvkgltf::primitiveKindName(m_pendingPrimitiveKind));
    if(m_pendingPrimitiveParent >= 0)
      ImGui::TextDisabled("Parent: node %d", m_pendingPrimitiveParent);
    else
      ImGui::TextDisabled("Parent: scene root");
    ImGui::Separator();

    ImGui::DragFloat("Size", &m_pendingPrimitiveParams.size, 0.05f, 0.001f, 1.0e6f, "%.3f");
    if(m_pendingPrimitiveKind == nvvkgltf::PrimitiveKind::eSphere)
    {
      ImGui::DragInt("Sectors", &m_pendingPrimitiveParams.subdivU, 1.0f, 3, 256);
      ImGui::DragInt("Stacks", &m_pendingPrimitiveParams.subdivV, 1.0f, 2, 256);
    }
    else if(m_pendingPrimitiveKind == nvvkgltf::PrimitiveKind::ePlane)
    {
      ImGui::DragInt("Subdivisions", &m_pendingPrimitiveParams.subdivU, 1.0f, 1, 256);
    }

    ImGui::Separator();

    const bool canAdd = m_scene && m_undoStack;
    ImGui::BeginDisabled(!canAdd);
    if(ImGui::Button(ICON_MS_CHECK " Add", ImVec2(120, 0)))
    {
      auto cmd = std::make_unique<AddPrimitiveCommand>(*m_scene, m_pendingPrimitiveKind, m_pendingPrimitiveParams,
                                                       m_pendingPrimitiveParent, m_selection);
      m_undoStack->executeCommand(std::move(cmd));
      if(m_onGeometryChanged)
        m_onGeometryChanged();  // renderer (re)creates GPU vertex/index buffers + acceleration structures
      markSceneTransformsDirty();
      markCachesDirty();
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if(ImGui::Button(ICON_MS_CANCEL " Cancel", ImVec2(120, 0)))
      ImGui::CloseCurrentPopup();

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
