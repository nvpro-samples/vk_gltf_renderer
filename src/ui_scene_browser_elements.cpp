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
// Elements tab of the Scene Browser: a single data-driven, editor-grade list.
//
// One ElementTypeDesc per glTF collection (built in ensureElementRegistry) drives ONE generic
// renderer - icon tab bar, columnar table, Add/Duplicate/Delete/Rename toolbar, inline rename -
// so every category looks and behaves the same and a new collection is one descriptor.
//
// Performance: the whole thing rests on m_revision (bumped by markCachesDirty on every edit). The
// filtered view (buildElementView) and the derived-stat tables (ensureElementStats) rebuild ONLY
// when the revision/filter/tab changes; the table itself renders through an ImGuiListClipper, so
// only on-screen rows do any work - the list stays cheap into the millions of nodes.
//

#include "ui_scene_browser.hpp"
#include "undo_redo.hpp"
#include "gltf_scene.hpp"
#include "gltf_scene_editor.hpp"
#include "tinygltf_utils.hpp"
#include "ui_gltf_labels.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <nvgui/fonts.hpp>
#include <nvutils/logger.hpp>
#include <glm/glm.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>

//==================================================================================================
// LOCAL HELPERS
//==================================================================================================

namespace {

// Leading icon for a node row, chosen by the resource it carries (mesh / camera / light / group).
const char* nodeResourceIcon(const tinygltf::Node& node)
{
  if(node.mesh >= 0)
    return ICON_MS_VIEW_IN_AR;
  if(node.camera >= 0)
    return ICON_MS_CAMERA_ALT;
  if(node.light >= 0)
    return ICON_MS_LIGHTBULB;
  if(tinygltf::utils::getNodeIesLight(node).light >= 0)
    return ICON_MS_FLASHLIGHT_ON;  // pure EXT_lights_ies (no KHR_lights_punctual)
  return ICON_MS_CATEGORY;         // empty / pure transform group
}

// Right-align a short text within the current table cell (numeric columns read better right-aligned).
void cellRightText(const char* text)
{
  const float w     = ImGui::CalcTextSize(text).x;
  const float avail = ImGui::GetContentRegionAvail().x;
  if(avail > w)
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (avail - w));
  ImGui::TextUnformatted(text);
}

void cellRightInt(int value)
{
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%d", value);
  cellRightText(buf);
}

// Right-aligned compact count (e.g. 92.4K, 1.2M) so large values (triangles) fit a narrow cell.
void cellRightCompact(long long value)
{
  char buf[32];
  if(value >= 1000000)
    std::snprintf(buf, sizeof(buf), "%.1fM", double(value) / 1e6);
  else if(value >= 10000)
    std::snprintf(buf, sizeof(buf), "%.1fK", double(value) / 1e3);
  else
    std::snprintf(buf, sizeof(buf), "%lld", value);
  cellRightText(buf);
}

// A small non-interactive color swatch (clicks fall through to the row selectable).
void drawSwatch(const ImVec4& color)
{
  const ImGuiColorEditFlags flags = ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoDragDrop
                                    | ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_NoBorder;
  ImGui::ColorButton("##sw", color, flags, ImVec2(12.0f, 12.0f));
}

ImVec4 baseColorOf(const tinygltf::Material& m)
{
  const auto& c = m.pbrMetallicRoughness.baseColorFactor;  // linear RGBA, size 4
  if(c.size() < 4)
    return ImVec4(1, 1, 1, 1);
  return ImVec4(float(c[0]), float(c[1]), float(c[2]), float(c[3]));
}

ImVec4 lightColorOf(const tinygltf::Light& l)
{
  if(l.color.size() < 3)
    return ImVec4(1, 1, 1, 1);
  return ImVec4(float(l.color[0]), float(l.color[1]), float(l.color[2]), 1.0f);
}

std::string toLower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// Animation length in seconds: the max end-time across its samplers' input (time) accessors, read from
// each accessor's stored maxValues[0] (cheap; no buffer decode).
float animationDuration(const tinygltf::Model& model, const tinygltf::Animation& anim)
{
  float duration = 0.0f;
  for(const tinygltf::AnimationSampler& s : anim.samplers)
  {
    if(s.input < 0 || s.input >= int(model.accessors.size()))
      continue;
    const std::vector<double>& mx = model.accessors[s.input].maxValues;
    if(!mx.empty())
      duration = std::max(duration, float(mx[0]));
  }
  return duration;
}

// Compact human counts for footer aggregates.
std::string formatTriangleCount(long long tris)
{
  char buf[64];
  if(tris >= 1000000)
    std::snprintf(buf, sizeof(buf), "%.1fM triangles", double(tris) / 1e6);
  else if(tris >= 1000)
    std::snprintf(buf, sizeof(buf), "%.1fK triangles", double(tris) / 1e3);
  else
    std::snprintf(buf, sizeof(buf), "%lld triangles", tris);
  return buf;
}

std::string formatBytes(long long bytes)
{
  char buf[64];
  if(bytes >= (1LL << 30))
    std::snprintf(buf, sizeof(buf), "%.1f GB", double(bytes) / double(1LL << 30));
  else if(bytes >= (1LL << 20))
    std::snprintf(buf, sizeof(buf), "%.1f MB", double(bytes) / double(1LL << 20));
  else if(bytes >= (1LL << 10))
    std::snprintf(buf, sizeof(buf), "%.1f KB", double(bytes) / double(1LL << 10));
  else
    std::snprintf(buf, sizeof(buf), "%lld B", bytes);
  return buf;
}

// Draw a small square thumbnail (or a placeholder) sized to the row height, then keep the cursor on the
// same line so the caller can append the name. `tex` may be 0 (no GPU thumbnail resident).
void thumbnailCell(ImTextureID tex)
{
  const float sz = ImGui::GetTextLineHeight();
  if(tex != ImTextureID(0))
    ImGui::Image(tex, ImVec2(sz, sz));
  else
    ImGui::Dummy(ImVec2(sz, sz));
  ImGui::SameLine();
}

}  // namespace

//==================================================================================================
// REGISTRY - one descriptor per glTF collection (built once, lambdas close over `this`)
//==================================================================================================

void UiSceneBrowser::ensureElementRegistry()
{
  if(!m_elementTypes.empty() || !m_scene)
    return;

  m_elementTypes.clear();
  const tinygltf::Model& model = m_scene->getModel();
  (void)model;  // captured lazily inside lambdas via m_scene

  //------------------------------------------------------------------------------------------------
  // NODES - the scene graph. Owns instance lifecycle (add empty/mesh/light, duplicate, delete).
  //------------------------------------------------------------------------------------------------
  {
    ElementTypeDesc d;
    d.icon     = ICON_MS_CATEGORY;
    d.singular = "Node";
    d.plural   = "Nodes";
    d.selKind  = SceneSelection::SelectionType::eNode;
    d.count    = [this] { return int(m_scene->getModel().nodes.size()); };
    d.name     = [this](int i) { return m_scene->getModel().nodes[i].name; };
    d.select   = [this](int i) { m_selection->selectNode(i); };
    d.columns.push_back({"Name", 0.0f, false, [this](int i) {
                           const tinygltf::Node& n = m_scene->getModel().nodes[i];
                           ImGui::Text("%s %s", nodeResourceIcon(n), n.name.c_str());
                         }});
    d.columns.push_back(
        {"Children", 70.0f, true, [this](int i) { cellRightInt(int(m_scene->getModel().nodes[i].children.size())); }});
    d.columns.back().sortKey = [this](int i) { return double(m_scene->getModel().nodes[i].children.size()); };
    // Inline visibility toggle (KHR_node_visibility): hide/show the node and its subtree from rendering.
    d.columns.push_back(
        {"", 26.0f, false, [this](int i) {
           tinygltf::Node&           node = m_scene->editor().getNodeForEdit(i);
           const KHR_node_visibility vis  = tinygltf::utils::getNodeVisibility(node);
           ImGui::BeginDisabled(m_scene->isNodeReadOnly(i));
           if(ImGui::SmallButton((std::string(vis.visible ? ICON_MS_VISIBILITY : ICON_MS_VISIBILITY_OFF) + "###vis").c_str()))
           {
             // Snapshot the prior extension payload (or "absent") and compute the new one via the typed
             // setter on a scratch node, so the toggle is a normal undoable edit instead of a direct write.
             tinygltf::Value oldValue = tinygltf::utils::hasElementName(node.extensions, KHR_NODE_VISIBILITY_EXTENSION_NAME) ?
                                            node.extensions.at(KHR_NODE_VISIBILITY_EXTENSION_NAME) :
                                            tinygltf::Value{};
             tinygltf::Node scratch;
             tinygltf::utils::setNodeVisibility(scratch, {!vis.visible});
             m_undoStack->executeCommand(std::make_unique<SetNodeExtensionCommand>(
                 *m_scene, i, KHR_NODE_VISIBILITY_EXTENSION_NAME, oldValue,
                 scratch.extensions.at(KHR_NODE_VISIBILITY_EXTENSION_NAME), vis.visible ? "Hide node" : "Show node"));
             // render-only change; no list rebuild needed
           }
           if(ImGui::IsItemHovered())
             ImGui::SetTooltip(vis.visible ? "Hide node" : "Show node");
           ImGui::EndDisabled();
         }});

    // Add: Empty + one item per procedural mesh + one item per punctual light (flat menu).
    d.addVariants.push_back({"Empty Node", ICON_MS_CATEGORY, [this] { addEmptyNode(-1); }});
    for(const auto& info : nvvkgltf::kPrimitiveKinds)
      d.addVariants.push_back({info.name, ICON_MS_VIEW_IN_AR, [this, info] { requestAddPrimitive(info.kind, -1); }});
    for(const auto& info : nvvkgltf::kLightKinds)
      d.addVariants.push_back({info.name, ICON_MS_LIGHTBULB, [this, info] { addLight(info.type, info.name, -1); }});

    d.duplicate = [this](int i) {
      if(m_scene->isNodeReadOnly(i))
        return;
      m_undoStack->executeCommand(std::make_unique<DuplicateNodeCommand>(*m_scene, i, m_selection));
      markCachesDirty();
    };
    d.canDelete            = [this](int i) { return !m_scene->isNodeReadOnly(i); };
    d.deleteDisabledReason = [](int) { return std::string("Read-only (external asset) - Make Editable first"); };
    d.remove               = [this](int i) {
      if(m_pendingDeleteNode && m_openDeletePopupNextFrame)
      {
        *m_pendingDeleteNode        = i;
        *m_openDeletePopupNextFrame = true;
      }
    };
    d.rename = [this](int i, const std::string& n) {
      if(m_scene->isNodeReadOnly(i))
        return;
      m_undoStack->executeCommand(std::make_unique<RenameNodeCommand>(*m_scene, i, m_scene->getModel().nodes[i].name, n));
    };
    m_elementTypes.push_back(std::move(d));
  }

  //------------------------------------------------------------------------------------------------
  // MESHES - resource pool. Triangle / instance counts make it an asset-weight view.
  //------------------------------------------------------------------------------------------------
  {
    ElementTypeDesc d;
    d.icon     = ICON_MS_VIEW_IN_AR;
    d.singular = "Mesh";
    d.plural   = "Meshes";
    d.selKind  = SceneSelection::SelectionType::eMesh;
    d.count    = [this] { return int(m_scene->getModel().meshes.size()); };
    d.name     = [this](int i) { return m_scene->getModel().meshes[i].name; };
    d.select   = [this](int i) { m_selection->selectMesh(i); };
    d.columns.push_back({"Name", 0.0f, false, [this](int i) {
                           ImGui::Text("%s %s", ICON_MS_VIEW_IN_AR, m_scene->getModel().meshes[i].name.c_str());
                         }});
    d.columns.push_back(
        {"Prims", 56.0f, true, [this](int i) { cellRightInt(int(m_scene->getModel().meshes[i].primitives.size())); }});
    d.columns.push_back({"Triangles", 96.0f, true, [this](int i) {
                           ensureElementStats();
                           cellRightCompact(i < int(m_meshTriangles.size()) ? m_meshTriangles[i] : 0);
                         }});
    d.columns.push_back({"Instances", 76.0f, true, [this](int i) {
                           ensureElementStats();
                           cellRightInt(i < int(m_meshInstances.size()) ? m_meshInstances[i] : 0);
                         }});
    d.columns[1].sortKey = [this](int i) { return double(m_scene->getModel().meshes[i].primitives.size()); };
    d.columns[2].sortKey = [this](int i) {
      ensureElementStats();
      return double(i < int(m_meshTriangles.size()) ? m_meshTriangles[i] : 0);
    };
    d.columns[3].sortKey = [this](int i) {
      ensureElementStats();
      return double(i < int(m_meshInstances.size()) ? m_meshInstances[i] : 0);
    };
    d.footerAggregate = [this]() {
      ensureElementStats();
      long long tris = 0;
      for(int t : m_meshTriangles)
        tris += t;
      return formatTriangleCount(tris);
    };
    d.rename = [this](int i, const std::string& n) {
      m_undoStack->executeCommand(std::make_unique<RenameMeshCommand>(*m_scene, i, m_scene->getModel().meshes[i].name, n));
      markCachesDirty();
    };
    m_elementTypes.push_back(std::move(d));
  }

  //------------------------------------------------------------------------------------------------
  // MATERIALS - pure resource pool: full CRUD, refcount-aware delete, all undoable.
  //------------------------------------------------------------------------------------------------
  {
    ElementTypeDesc d;
    d.icon     = ICON_MS_BRUSH;
    d.singular = "Material";
    d.plural   = "Materials";
    d.selKind  = SceneSelection::SelectionType::eMaterial;
    d.count    = [this] { return int(m_scene->getModel().materials.size()); };
    d.name     = [this](int i) { return m_scene->getModel().materials[i].name; };
    d.select   = [this](int i) { m_selection->selectMaterial(i); };
    d.columns.push_back({"Name", 0.0f, false, [this](int i) {
                           const tinygltf::Material& m = m_scene->getModel().materials[i];
                           drawSwatch(baseColorOf(m));
                           ImGui::SameLine();
                           ImGui::TextUnformatted(m.name.c_str());
                         }});
    d.columns.push_back({"Alpha", 90.0f, false, [this](int i) {
                           const std::string& a = m_scene->getModel().materials[i].alphaMode;
                           ImGui::TextUnformatted(a.empty() ? "OPAQUE" : a.c_str());
                         }});
    d.columns.push_back({"Used by", 64.0f, true, [this](int i) {
                           ensureElementStats();
                           cellRightInt(i < int(m_materialRefs.size()) ? m_materialRefs[i] : 0);
                         }});
    d.columns[2].sortKey = [this](int i) {
      ensureElementStats();
      return double(i < int(m_materialRefs.size()) ? m_materialRefs[i] : 0);
    };

    // Add a default PBR material (undoable), then select it.
    d.addVariants.push_back({"Material", ICON_MS_BRUSH, [this] {
                               const int          idx = int(m_scene->getModel().materials.size());
                               tinygltf::Material mat;  // tinygltf defaults: baseColor 1,1,1,1; metallic 1; roughness 1
                               mat.name = "Material";
                               m_undoStack->executeCommand(
                                   std::make_unique<MaterialLifecycleCommand>(*m_scene, idx, mat, true, "Add material"));
                               markCachesDirty();
                               m_selection->selectMaterial(idx);
                             }});
    d.duplicate = [this](int i) {
      const int          idx = int(m_scene->getModel().materials.size());
      tinygltf::Material mat = m_scene->getModel().materials[i];
      mat.name += "_copy";
      m_undoStack->executeCommand(std::make_unique<MaterialLifecycleCommand>(*m_scene, idx, mat, true, "Duplicate material"));
      markCachesDirty();
      m_selection->selectMaterial(idx);
    };
    // A material is deletable only when no primitive references it (refcount 0).
    d.canDelete = [this](int i) {
      ensureElementStats();
      return i < int(m_materialPrimRefs.size()) && m_materialPrimRefs[i] == 0;
    };
    d.deleteDisabledReason = [](int) { return std::string("In use - assign its primitives to another material first"); };
    d.remove = [this](int i) {
      m_undoStack->executeCommand(std::make_unique<MaterialLifecycleCommand>(*m_scene, i, m_scene->getModel().materials[i],
                                                                             false, "Delete material"));
      markCachesDirty();
      if(m_selection->getSelection().type == SceneSelection::SelectionType::eMaterial)
        m_selection->clearSelection();
    };
    d.rename = [this](int i, const std::string& n) {
      m_undoStack->executeCommand(std::make_unique<RenameMaterialCommand>(*m_scene, i, m_scene->getModel().materials[i].name, n));
      markCachesDirty();
    };
    m_elementTypes.push_back(std::move(d));
  }

  //------------------------------------------------------------------------------------------------
  // CAMERAS - node-attached resource. Lifecycle (add/duplicate/delete) is node-coupled and handled
  // through the scene graph in a dedicated pass; the list is browse + rename + select for now.
  //------------------------------------------------------------------------------------------------
  {
    ElementTypeDesc d;
    d.icon     = ICON_MS_CAMERA_ALT;
    d.singular = "Camera";
    d.plural   = "Cameras";
    d.selKind  = SceneSelection::SelectionType::eCamera;
    d.count    = [this] { return int(m_scene->getModel().cameras.size()); };
    d.name     = [this](int i) { return m_scene->getModel().cameras[i].name; };
    d.select   = [this](int i) { m_selection->selectCamera(i); };
    d.columns.push_back({"Name", 0.0f, false, [this](int i) {
                           ImGui::Text("%s %s", ICON_MS_CAMERA_ALT, m_scene->getModel().cameras[i].name.c_str());
                         }});
    d.columns.push_back({"Type", 100.0f, false, [this](int i) {
                           const std::string& t = m_scene->getModel().cameras[i].type;
                           ImGui::TextUnformatted(t.empty() ? "perspective" : t.c_str());
                         }});
    d.columns.push_back({"FOV / Mag", 84.0f, true, [this](int i) {
                           const tinygltf::Camera& c = m_scene->getModel().cameras[i];
                           char                    buf[32];
                           if(c.type == "orthographic")
                             std::snprintf(buf, sizeof(buf), "%.2f", c.orthographic.ymag);
                           else
                             std::snprintf(buf, sizeof(buf), "%.0f\xc2\xb0", glm::degrees(c.perspective.yfov));
                           cellRightText(buf);
                         }});
    d.columns[2].sortKey = [this](int i) {
      const tinygltf::Camera& c = m_scene->getModel().cameras[i];
      return c.type == "orthographic" ? c.orthographic.ymag : glm::degrees(c.perspective.yfov);
    };
    d.rename = [this](int i, const std::string& n) {
      m_undoStack->executeCommand(std::make_unique<RenameCameraCommand>(*m_scene, i, m_scene->getModel().cameras[i].name, n));
      markCachesDirty();
    };
    m_elementTypes.push_back(std::move(d));
  }

  //------------------------------------------------------------------------------------------------
  // LIGHTS - KHR_lights_punctual and pure EXT_lights_ies lights in one unified list.
  //   Indices [0, khrCount)          → KHR_lights_punctual (model.lights[i])
  //   Indices [khrCount, khrCount+n) → pure EXT_lights_ies nodes (no KHR light)
  // The two sub-ranges use different selection types: eLight for KHR, eNode for IES.
  // selectedIndexFor bridges both into the list's linear index space.
  //------------------------------------------------------------------------------------------------
  {
    // Helper: list of node indices that carry EXT_lights_ies but no KHR_lights_punctual.
    auto getIesOnlyNodes = [this]() {
      std::vector<int>       nodes;
      const tinygltf::Model& model = m_scene->getModel();
      for(int n = 0; n < int(model.nodes.size()); ++n)
        if(model.nodes[n].light < 0 && tinygltf::utils::getNodeIesLight(model.nodes[n]).light >= 0)
          nodes.push_back(n);
      return nodes;
    };

    ElementTypeDesc d;
    d.icon     = ICON_MS_LIGHTBULB;
    d.singular = "Light";
    d.plural   = "Lights";
    d.selKind  = SceneSelection::SelectionType::eLight;  // default (KHR range)

    d.count = [this, getIesOnlyNodes] { return int(m_scene->getModel().lights.size()) + int(getIesOnlyNodes().size()); };

    d.name = [this, getIesOnlyNodes](int i) -> std::string {
      const tinygltf::Model& model    = m_scene->getModel();
      const int              khrCount = int(model.lights.size());
      if(i < khrCount)
        return model.lights[i].name;
      // IES-only: prefer node name, then profile name/URI.
      const auto iesNodes = getIesOnlyNodes();
      const int  iesIdx   = i - khrCount;
      if(iesIdx >= int(iesNodes.size()))
        return {};
      const tinygltf::Node& node = model.nodes[iesNodes[iesIdx]];
      if(!node.name.empty())
        return node.name;
      EXT_lights_ies_ref                  ref      = tinygltf::utils::getNodeIesLight(node);
      std::vector<EXT_lights_ies_profile> profiles = tinygltf::utils::getIesProfiles(model);
      if(ref.light >= 0 && ref.light < int(profiles.size()))
      {
        if(!profiles[ref.light].name.empty())
          return profiles[ref.light].name;
        if(!profiles[ref.light].uri.empty())
          return profiles[ref.light].uri;
      }
      return "IES Light " + std::to_string(iesIdx);
    };

    d.select = [this, getIesOnlyNodes](int i) {
      const tinygltf::Model& model    = m_scene->getModel();
      const int              khrCount = int(model.lights.size());
      if(i < khrCount)
      {
        // KHR+IES hybrid: EXT_lights_ies wins over the KHR attachment on that node (per spec), and
        // IES properties only appear in the node inspector. Route to the node so the inspector
        // surfaces the IES multiplier/color that actually affect rendering.
        for(int n = 0; n < int(model.nodes.size()); ++n)
        {
          if(model.nodes[n].light == i && tinygltf::utils::getNodeIesLight(model.nodes[n]).light >= 0)
          {
            m_selection->selectNode(n);
            return;
          }
        }
        m_selection->selectLight(i);
      }
      else
      {
        const auto iesNodes = getIesOnlyNodes();
        const int  iesIdx   = i - khrCount;
        if(iesIdx < int(iesNodes.size()))
          m_selection->selectNode(iesNodes[iesIdx]);
      }
    };

    d.selectedIndexFor = [this, getIesOnlyNodes](const SceneSelection::SelectionContext& s, const tinygltf::Model& model) -> int {
      const int khrCount = int(model.lights.size());
      using Sel          = SceneSelection::SelectionType;
      if(s.type == Sel::eLight && s.lightIndex >= 0 && s.lightIndex < khrCount)
        return s.lightIndex;
      if(s.type == Sel::eNode && s.nodeIndex >= 0)
      {
        const auto iesNodes = getIesOnlyNodes();
        for(int k = 0; k < int(iesNodes.size()); ++k)
          if(iesNodes[k] == s.nodeIndex)
            return khrCount + k;
      }
      // Contextual: a node selection highlights its KHR light if it has one.
      if(s.type == Sel::eNode && s.nodeIndex >= 0 && s.nodeIndex < int(model.nodes.size()))
      {
        const int khrLight = model.nodes[s.nodeIndex].light;
        if(khrLight >= 0 && khrLight < khrCount)
          return khrLight;
      }
      return -1;
    };

    // Name column: color swatch + name, differentiated by icon for IES-only.
    // Capture d.name so the displayed label uses the same fallback as search/sort.
    auto rowName = d.name;
    d.columns.push_back({"Name", 0.0f, false, [this, getIesOnlyNodes, rowName](int i) {
                           const tinygltf::Model& model    = m_scene->getModel();
                           const int              khrCount = int(model.lights.size());
                           if(i < khrCount)
                           {
                             const tinygltf::Light& l = model.lights[i];
                             drawSwatch(lightColorOf(l));
                             ImGui::SameLine();
                             ImGui::TextUnformatted(l.name.c_str());
                           }
                           else
                           {
                             const auto            iesNodes = getIesOnlyNodes();
                             const int             iesIdx   = i - khrCount;
                             const tinygltf::Node& node     = model.nodes[iesNodes[iesIdx]];
                             EXT_lights_ies_ref    ref      = tinygltf::utils::getNodeIesLight(node);
                             glm::vec3             col      = ref.color;
                             drawSwatch(ImVec4(col.x, col.y, col.z, 1.0f));
                             ImGui::SameLine();
                             ImGui::Text("%s %s", ICON_MS_FLASHLIGHT_ON, rowName(i).c_str());
                           }
                         }});

    // Type column: KHR type (with +IES suffix when both present), or "IES" for pure IES.
    d.columns.push_back({"Type", 100.0f, false, [this, getIesOnlyNodes](int i) {
                           const tinygltf::Model& model    = m_scene->getModel();
                           const int              khrCount = int(model.lights.size());
                           if(i < khrCount)
                           {
                             const std::string& t      = model.lights[i].type;
                             bool               hasIes = false;
                             for(const auto& node : model.nodes)
                               if(node.light == i && tinygltf::utils::getNodeIesLight(node).light >= 0)
                               {
                                 hasIes = true;
                                 break;
                               }
                             std::string label = t.empty() ? "point" : t;
                             if(hasIes)
                               label += "+IES";
                             ImGui::TextUnformatted(label.c_str());
                           }
                           else
                           {
                             ImGui::TextUnformatted("IES");
                           }
                         }});

    // Intensity column: KHR intensity, or IES multiplier for pure IES.
    d.columns.push_back({"Intensity", 84.0f, true, [this, getIesOnlyNodes](int i) {
                           const tinygltf::Model& model    = m_scene->getModel();
                           const int              khrCount = int(model.lights.size());
                           char                   buf[32];
                           if(i < khrCount)
                             std::snprintf(buf, sizeof(buf), "%.1f", model.lights[i].intensity);
                           else
                           {
                             const auto iesNodes = getIesOnlyNodes();
                             EXT_lights_ies_ref ref = tinygltf::utils::getNodeIesLight(model.nodes[iesNodes[i - khrCount]]);
                             std::snprintf(buf, sizeof(buf), "%.2f", ref.multiplier);
                           }
                           cellRightText(buf);
                         }});
    d.columns[2].sortKey = [this, getIesOnlyNodes](int i) -> double {
      const tinygltf::Model& model    = m_scene->getModel();
      const int              khrCount = int(model.lights.size());
      if(i < khrCount)
        return model.lights[i].intensity;
      const auto         iesNodes = getIesOnlyNodes();
      EXT_lights_ies_ref ref      = tinygltf::utils::getNodeIesLight(model.nodes[iesNodes[i - khrCount]]);
      return ref.multiplier;
    };

    for(const auto& info : nvvkgltf::kLightKinds)
      d.addVariants.push_back({info.name, ICON_MS_LIGHTBULB, [this, info] { addLight(info.type, info.name, -1); }});

    d.editableName = [this, getIesOnlyNodes](int i) -> std::string {
      const tinygltf::Model& model    = m_scene->getModel();
      const int              khrCount = int(model.lights.size());
      if(i < khrCount)
        return model.lights[i].name;
      const auto iesNodes = getIesOnlyNodes();
      const int  iesIdx   = i - khrCount;
      return iesIdx < int(iesNodes.size()) ? model.nodes[iesNodes[iesIdx]].name : std::string{};
    };

    d.rename = [this, getIesOnlyNodes](int i, const std::string& n) {
      const tinygltf::Model& model    = m_scene->getModel();
      const int              khrCount = int(model.lights.size());
      if(i < khrCount)
      {
        m_undoStack->executeCommand(std::make_unique<RenameLightCommand>(*m_scene, i, model.lights[i].name, n));
      }
      else
      {
        const auto iesNodes = getIesOnlyNodes();
        const int  iesIdx   = i - khrCount;
        if(iesIdx < int(iesNodes.size()))
          m_undoStack->executeCommand(
              std::make_unique<RenameNodeCommand>(*m_scene, iesNodes[iesIdx], model.nodes[iesNodes[iesIdx]].name, n));
      }
      markCachesDirty();
    };

    m_elementTypes.push_back(std::move(d));
  }

  //------------------------------------------------------------------------------------------------
  // TEXTURES - image + sampler reference. Edited in the Inspector; delete rides the image refcount.
  //------------------------------------------------------------------------------------------------
  {
    ElementTypeDesc d;
    d.icon         = ICON_MS_IMAGE;
    d.singular     = "Texture";
    d.plural       = "Textures";
    d.selKind      = SceneSelection::SelectionType::eTexture;
    d.count        = [this] { return int(m_scene->getModel().textures.size()); };
    d.name         = [this](int i) { return tinygltf::utils::getTextureUiLabel(m_scene->getModel(), i); };
    d.editableName = [this](int i) { return m_scene->getModel().textures[i].name; };
    d.select       = [this](int i) { m_selection->selectTexture(i); };
    d.columns.push_back({"Name", 0.0f, false, [this](int i) {
                           thumbnailCell(m_host.thumbnail(i));
                           ImGui::TextUnformatted(tinygltf::utils::getTextureUiLabel(m_scene->getModel(), i).c_str());
                         }});
    d.columns.push_back({"Image", 92.0f, false, [this](int i) {
                           const tinygltf::Model& model  = m_scene->getModel();
                           const int              imgIdx = tinygltf::utils::getTextureImageIndex(model.textures[i]);
                           if(imgIdx >= 0 && imgIdx < int(model.images.size()) && model.images[imgIdx].width > 0)
                             ImGui::Text("%dx%d", model.images[imgIdx].width, model.images[imgIdx].height);
                           else
                             ImGui::TextDisabled("-");
                         }});
    d.columns.push_back({"Sampler", 120.0f, false, [this](int i) {
                           const tinygltf::Model& model = m_scene->getModel();
                           const int              s     = model.textures[i].sampler;
                           if(s >= 0 && s < int(model.samplers.size()))
                             ImGui::TextUnformatted(uigltf::samplerSummary(model.samplers[s]).c_str());
                           else
                             ImGui::TextDisabled("default");
                         }});
    d.rename = [this](int i, const std::string& n) {
      m_undoStack->executeCommand(std::make_unique<RenameTextureCommand>(*m_scene, i, m_scene->getModel().textures[i].name, n));
      markCachesDirty();
    };
    m_elementTypes.push_back(std::move(d));
  }

  //------------------------------------------------------------------------------------------------
  // IMAGES - pixel source. Resolution + refcount make this the texture-memory / unused-asset view.
  //------------------------------------------------------------------------------------------------
  {
    ElementTypeDesc d;
    d.icon         = ICON_MS_PHOTO;
    d.singular     = "Image";
    d.plural       = "Images";
    d.selKind      = SceneSelection::SelectionType::eImage;
    d.count        = [this] { return int(m_scene->getModel().images.size()); };
    d.name         = [this](int i) { return uigltf::imageDisplayName(m_scene->getModel(), i); };
    d.editableName = [this](int i) { return m_scene->getModel().images[i].name; };
    d.select       = [this](int i) { m_selection->selectImage(i); };
    d.columns.push_back({"Name", 0.0f, false, [this](int i) {
                           thumbnailCell(m_getImageThumbnail ? m_getImageThumbnail(i) : ImTextureID(0));
                           ImGui::TextUnformatted(uigltf::imageDisplayName(m_scene->getModel(), i).c_str());
                         }});
    d.columns.push_back({"Resolution", 96.0f, false, [this](int i) {
                           const tinygltf::Image& img = m_scene->getModel().images[i];
                           if(img.width > 0 && img.height > 0)
                             ImGui::Text("%dx%d", img.width, img.height);
                           else
                             ImGui::TextDisabled("-");
                         }});
    d.columns.push_back({"Refs", 56.0f, true, [this](int i) {
                           ensureElementStats();
                           cellRightInt(i < int(m_imageRefs.size()) ? m_imageRefs[i] : 0);
                         }});
    d.columns[1].sortKey = [this](int i) {
      const tinygltf::Image& img = m_scene->getModel().images[i];
      return double(img.width) * double(img.height);
    };
    d.columns[2].sortKey = [this](int i) {
      ensureElementStats();
      return double(i < int(m_imageRefs.size()) ? m_imageRefs[i] : 0);
    };
    // Approximate decoded texture memory (RGBA8) across all images: sort-by-heaviest / find-unused view.
    d.footerAggregate = [this]() {
      long long bytes = 0;
      for(const tinygltf::Image& img : m_scene->getModel().images)
        if(img.width > 0 && img.height > 0)
          bytes += static_cast<long long>(img.width) * img.height * 4;
      return formatBytes(bytes);
    };
    // An image is deletable only when no texture references it (refcount 0).
    d.canDelete = [this](int i) {
      ensureElementStats();
      return i < int(m_imageRefs.size()) && m_imageRefs[i] == 0;
    };
    d.deleteDisabledReason = [](int) { return std::string("In use - referenced by a texture"); };
    d.remove               = [this](int i) {
      m_undoStack->executeCommand(std::make_unique<RemoveImageCommand>(*m_scene, i, m_scene->getModel().images[i]));
      markCachesDirty();
      if(m_selection->getSelection().type == SceneSelection::SelectionType::eImage)
        m_selection->clearSelection();
    };
    d.rename = [this](int i, const std::string& n) {
      m_undoStack->executeCommand(std::make_unique<RenameImageCommand>(*m_scene, i, m_scene->getModel().images[i].name, n));
      markCachesDirty();
    };
    m_elementTypes.push_back(std::move(d));
  }

  //------------------------------------------------------------------------------------------------
  // SAMPLERS - wrap / filter. Edited in the Inspector.
  //------------------------------------------------------------------------------------------------
  {
    ElementTypeDesc d;
    d.icon     = ICON_MS_TUNE;
    d.singular = "Sampler";
    d.plural   = "Samplers";
    d.selKind  = SceneSelection::SelectionType::eSampler;
    d.count    = [this] { return int(m_scene->getModel().samplers.size()); };
    d.name     = [this](int i) {
      const std::string& n = m_scene->getModel().samplers[i].name;
      return n.empty() ? ("Sampler " + std::to_string(i)) : n;
    };
    d.editableName = [this](int i) { return m_scene->getModel().samplers[i].name; };
    d.select       = [this](int i) { m_selection->selectSampler(i); };
    d.columns.push_back({"Name", 0.0f, false, [this](int i) {
                           const tinygltf::Sampler& s = m_scene->getModel().samplers[i];
                           ImGui::Text("%s %s", ICON_MS_TUNE,
                                       s.name.empty() ? ("Sampler " + std::to_string(i)).c_str() : s.name.c_str());
                         }});
    d.columns.push_back({"Wrap S/T", 150.0f, false, [this](int i) {
                           const tinygltf::Sampler& s = m_scene->getModel().samplers[i];
                           ImGui::Text("%s / %s", uigltf::wrapName(s.wrapS), uigltf::wrapName(s.wrapT));
                         }});
    d.columns.push_back({"Filter", 130.0f, false, [this](int i) {
                           const tinygltf::Sampler& s = m_scene->getModel().samplers[i];
                           ImGui::Text("%s / %s", uigltf::minName(s.minFilter), uigltf::magName(s.magFilter));
                         }});
    d.columns.push_back({"Used by", 64.0f, true, [this](int i) {
                           ensureElementStats();
                           cellRightInt(i < int(m_samplerRefs.size()) ? m_samplerRefs[i] : 0);
                         }});
    d.columns[3].sortKey = [this](int i) {
      ensureElementStats();
      return double(i < int(m_samplerRefs.size()) ? m_samplerRefs[i] : 0);
    };
    d.rename = [this](int i, const std::string& n) {
      m_undoStack->executeCommand(std::make_unique<RenameSamplerCommand>(*m_scene, i, m_scene->getModel().samplers[i].name, n));
      markCachesDirty();
    };
    m_elementTypes.push_back(std::move(d));
  }

  //------------------------------------------------------------------------------------------------
  // ANIMATIONS - channels + duration (read summary in the Inspector; playback lives in its own panel).
  //------------------------------------------------------------------------------------------------
  {
    ElementTypeDesc d;
    d.icon     = ICON_MS_MOVIE;
    d.singular = "Animation";
    d.plural   = "Animations";
    d.selKind  = SceneSelection::SelectionType::eAnimation;
    d.count    = [this] { return int(m_scene->getModel().animations.size()); };
    d.name     = [this](int i) {
      const std::string& n = m_scene->getModel().animations[i].name;
      return n.empty() ? ("Animation " + std::to_string(i)) : n;
    };
    d.editableName = [this](int i) { return m_scene->getModel().animations[i].name; };
    d.select       = [this](int i) { m_selection->selectAnimation(i); };
    d.columns.push_back({"Name", 0.0f, false, [this](int i) {
                           const tinygltf::Animation& a = m_scene->getModel().animations[i];
                           ImGui::Text("%s %s", ICON_MS_MOVIE,
                                       a.name.empty() ? ("Animation " + std::to_string(i)).c_str() : a.name.c_str());
                         }});
    d.columns.push_back({"Channels", 74.0f, true,
                         [this](int i) { cellRightInt(int(m_scene->getModel().animations[i].channels.size())); }});
    d.columns.push_back({"Duration", 84.0f, true, [this](int i) {
                           char buf[32];
                           std::snprintf(buf, sizeof(buf), "%.2fs",
                                         animationDuration(m_scene->getModel(), m_scene->getModel().animations[i]));
                           cellRightText(buf);
                         }});
    d.columns[1].sortKey = [this](int i) { return double(m_scene->getModel().animations[i].channels.size()); };
    d.columns[2].sortKey = [this](int i) {
      return double(animationDuration(m_scene->getModel(), m_scene->getModel().animations[i]));
    };
    d.rename = [this](int i, const std::string& n) {
      m_undoStack->executeCommand(
          std::make_unique<RenameAnimationCommand>(*m_scene, i, m_scene->getModel().animations[i].name, n));
      markCachesDirty();
    };
    m_elementTypes.push_back(std::move(d));
  }
}

//==================================================================================================
// DERIVED STAT TABLES - recomputed once per revision (cheap; shared by columns and delete gates)
//==================================================================================================

void UiSceneBrowser::ensureElementStats()
{
  if(!m_scene || m_statsRevision == m_revision)
    return;

  const tinygltf::Model& model = m_scene->getModel();

  m_meshTriangles.assign(model.meshes.size(), 0);
  m_meshInstances.assign(model.meshes.size(), 0);
  m_materialRefs.assign(model.materials.size(), 0);
  m_samplerRefs.assign(model.samplers.size(), 0);

  // Triangles per mesh: sum over unique render primitives, mode-aware (0 for POINTS/LINE*, count-2 for
  // TRIANGLE_STRIP/FAN) so a mesh built from non-triangle primitives doesn't get a fabricated count.
  for(const nvvkgltf::RenderPrimitive& rp : m_scene->getRenderPrimitives())
  {
    if(rp.meshID < 0 || rp.meshID >= int(m_meshTriangles.size()))
      continue;
    const int mode = rp.pPrimitive ? rp.pPrimitive->mode : TINYGLTF_MODE_TRIANGLES;
    const int tris = int(uigltf::primitiveTriangleCountForMode(mode, rp.indexCount > 0 ? rp.indexCount : rp.vertexCount));
    m_meshTriangles[rp.meshID] += tris;
  }

  // Instances + material usage: walk nodes carrying a mesh, count instanced primitive->material usages.
  for(const tinygltf::Node& node : model.nodes)
  {
    if(node.mesh < 0 || node.mesh >= int(model.meshes.size()))
      continue;
    m_meshInstances[node.mesh]++;
    for(const tinygltf::Primitive& prim : model.meshes[node.mesh].primitives)
      if(prim.material >= 0 && prim.material < int(m_materialRefs.size()))
        m_materialRefs[prim.material]++;
  }

  // Sampler usage: textures referencing each sampler.
  for(const tinygltf::Texture& t : model.textures)
    if(t.sampler >= 0 && t.sampler < int(m_samplerRefs.size()))
      m_samplerRefs[t.sampler]++;

  // Image refcounts (core + extension sources) and per-material primitive refcounts (delete gate)
  // come from the editor's shared helpers.
  m_imageRefs        = m_scene->editor().computeImageRefCounts();
  m_materialPrimRefs = m_scene->editor().computeMaterialRefCounts();

  m_statsRevision = m_revision;
}

//==================================================================================================
// LAZY FILTERED VIEW - rebuilt only when (revision, tab, filter) changes
//==================================================================================================

void UiSceneBrowser::buildElementView(const ElementTypeDesc& desc, int tabIndex, int sortCol, bool sortAsc)
{
  const bool filterActive = m_elementFilter[0] != '\0';

  // sortCol is the TABLE column index: 0 = the leading "#" index column, >=1 = desc.columns[sortCol-1].
  // Index-ascending is the natural order, so it needs no materialized list (identity fast-path).
  const bool sortByIndex = (sortCol <= 0);
  const bool sortActive  = !(sortByIndex && sortAsc);

  if(m_elementView.builtRevision == m_revision && m_elementView.builtTab == tabIndex && m_elementView.builtFilter == m_elementFilter
     && m_elementView.builtSortCol == sortCol && m_elementView.builtSortAsc == sortAsc)
    return;  // still valid

  m_elementView.builtRevision = m_revision;
  m_elementView.builtTab      = tabIndex;
  m_elementView.builtFilter   = m_elementFilter;
  m_elementView.builtSortCol  = sortCol;
  m_elementView.builtSortAsc  = sortAsc;
  m_elementView.rows.clear();

  if(!filterActive && !sortActive)
  {
    m_elementView.identity = true;  // iterate 0..count directly - no per-frame scan
    return;
  }

  m_elementView.identity = false;
  const int n            = desc.count();

  // Base set: filtered subset, or all indices.
  if(filterActive)
  {
    const std::string needle = toLower(m_elementFilter);
    for(int i = 0; i < n; ++i)
      if(toLower(desc.name(i)).find(needle) != std::string::npos)
        m_elementView.rows.push_back(i);
  }
  else
  {
    m_elementView.rows.resize(n);
    for(int i = 0; i < n; ++i)
      m_elementView.rows[i] = i;
  }

  // Sort. Keys are precomputed once (O(n)) so the sort itself does no per-compare string lowercasing /
  // lambda calls - keeps a sort of a huge node list to O(n log n) comparisons.
  std::vector<int>& rows = m_elementView.rows;
  auto order = [sortAsc](const auto& a, const auto& b) { return sortAsc ? (a.first < b.first) : (a.first > b.first); };

  const int            descCol = sortCol - 1;  // -1 => the "#" index column
  const ElementColumn* col = (descCol >= 0 && descCol < int(desc.columns.size())) ? &desc.columns[descCol] : nullptr;

  if(sortByIndex)  // "#" column: sort by element index
  {
    std::stable_sort(rows.begin(), rows.end(), [sortAsc](int a, int b) { return sortAsc ? (a < b) : (a > b); });
  }
  else if(col && col->sortKey)  // numeric column
  {
    std::vector<std::pair<double, int>> keyed;
    keyed.reserve(rows.size());
    for(int idx : rows)
      keyed.emplace_back(col->sortKey(idx), idx);
    std::stable_sort(keyed.begin(), keyed.end(), order);
    for(size_t i = 0; i < rows.size(); ++i)
      rows[i] = keyed[i].second;
  }
  else  // name column (descCol == 0): lexical
  {
    std::vector<std::pair<std::string, int>> keyed;
    keyed.reserve(rows.size());
    for(int idx : rows)
      keyed.emplace_back(toLower(desc.name(idx)), idx);
    std::stable_sort(keyed.begin(), keyed.end(), order);
    for(size_t i = 0; i < rows.size(); ++i)
      rows[i] = keyed[i].second;
  }
}

//==================================================================================================
// SELECTION HELPERS
//==================================================================================================

// The "current" element index for a category, given the shared selection - used both to highlight the
// row and as the toolbar's target. Beyond the direct match (the selected element IS of this category),
// a selection also lights up the elements it unambiguously references, so picking a primitive in the
// viewport shows its node / mesh / material as current across the corresponding lists (and a node its
// mesh/camera/light, a texture its image/sampler). This mirrors the Inspector's RELATIONSHIPS view.
int UiSceneBrowser::selectedElementIndex(const ElementTypeDesc& desc) const
{
  if(!m_selection || !m_scene)
    return -1;
  using Sel                                     = SceneSelection::SelectionType;
  const SceneSelection::SelectionContext& s     = m_selection->getSelection();
  const tinygltf::Model&                  model = m_scene->getModel();

  // Category-specific override (e.g. a list mixing multiple selection types).
  if(desc.selectedIndexFor)
    return desc.selectedIndexFor(s, model);

  // Direct: the selected element is itself of this category.
  if(s.type == desc.selKind)
  {
    switch(desc.selKind)
    {
      case Sel::eNode:
        return s.nodeIndex;
      case Sel::eMesh:
        return s.meshIndex;
      case Sel::eMaterial:
        return s.materialIndex;
      case Sel::eCamera:
        return s.cameraIndex;
      case Sel::eLight:
        return s.lightIndex;
      case Sel::eTexture:
        return s.textureIndex;
      case Sel::eImage:
        return s.imageIndex;
      case Sel::eSampler:
        return s.samplerIndex;
      case Sel::eAnimation:
        return s.animationIndex;
      default:
        return -1;
    }
  }

  // Contextual: an element implies the ones it references (each is a single, unambiguous target).
  switch(s.type)
  {
    case Sel::ePrimitive:  // a picked primitive resolves to exactly one node, mesh and material
      if(desc.selKind == Sel::eNode)
        return s.nodeIndex;
      if(desc.selKind == Sel::eMesh)
        return s.meshIndex;
      if(desc.selKind == Sel::eMaterial)  // the material lives on the primitive, not the selection context
      {
        if(s.meshIndex >= 0 && s.meshIndex < int(model.meshes.size()) && s.primitiveIndex >= 0
           && s.primitiveIndex < int(model.meshes[s.meshIndex].primitives.size()))
          return model.meshes[s.meshIndex].primitives[s.primitiveIndex].material;
      }
      break;
    case Sel::eNode:  // a node references at most one mesh / camera / light
      if(s.nodeIndex >= 0 && s.nodeIndex < int(model.nodes.size()))
      {
        const tinygltf::Node& n = model.nodes[s.nodeIndex];
        if(desc.selKind == Sel::eMesh)
          return n.mesh;
        if(desc.selKind == Sel::eCamera)
          return n.camera;
        if(desc.selKind == Sel::eLight)
          return n.light;
      }
      break;
    case Sel::eTexture:  // a texture references one image and one sampler
      if(s.textureIndex >= 0 && s.textureIndex < int(model.textures.size()))
      {
        const tinygltf::Texture& t = model.textures[s.textureIndex];
        if(desc.selKind == Sel::eImage)
          return tinygltf::utils::getTextureImageIndex(t);
        if(desc.selKind == Sel::eSampler)
          return t.sampler;
      }
      break;
    default:
      break;
  }
  return -1;
}

//==================================================================================================
// RENAME (modal, routed through the active descriptor's rename handler)
//==================================================================================================

void UiSceneBrowser::beginElementRename(const ElementTypeDesc& desc, int index)
{
  if(index < 0 || !desc.rename)
    return;
  m_elementRenameIndex     = index;
  m_elementRenameTab       = m_activeElementTab;
  m_openElementRenamePopup = true;
  // Seed from the raw editable name when the category's `name` is a derived display label (image URI,
  // "Sampler N" placeholder, ...) -- otherwise a no-op rename would write that derived text back as the
  // real glTF name.
  const std::string cur = desc.editableName ? desc.editableName(index) : desc.name(index);
  const size_t      len = std::min(cur.size(), sizeof(m_elementRenameBuffer) - 1);
  std::memcpy(m_elementRenameBuffer, cur.c_str(), len);
  m_elementRenameBuffer[len] = '\0';
}

void UiSceneBrowser::renderElementRenameDialog()
{
  if(m_openElementRenamePopup)
  {
    ImGui::OpenPopup("RenameElement");
    m_openElementRenamePopup = false;
  }
  if(!ImGui::BeginPopupModal("RenameElement", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    return;

  ImGui::TextUnformatted("Rename:");
  if(ImGui::IsWindowAppearing())
    ImGui::SetKeyboardFocusHere();  // focus the field once, when the modal opens
  const bool valid = m_elementRenameTab >= 0 && m_elementRenameTab < int(m_elementTypes.size()) && m_elementRenameIndex >= 0;

  auto commit = [&]() {
    if(valid && m_elementRenameBuffer[0] != '\0')
    {
      const ElementTypeDesc& d = m_elementTypes[m_elementRenameTab];
      if(d.rename)
        d.rename(m_elementRenameIndex, std::string(m_elementRenameBuffer));
    }
    m_elementRenameIndex = -1;
    m_elementRenameTab   = -1;
    ImGui::CloseCurrentPopup();
  };

  if(ImGui::InputText("##name", m_elementRenameBuffer, sizeof(m_elementRenameBuffer), ImGuiInputTextFlags_EnterReturnsTrue))
    commit();
  if(ImGui::Button(ICON_MS_CHECK " OK###renameOk"))
    commit();
  ImGui::SameLine();
  if(ImGui::Button(ICON_MS_CANCEL " Cancel"))
  {
    m_elementRenameIndex = -1;
    m_elementRenameTab   = -1;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

//==================================================================================================
// TOOLBAR - filter + Add / Duplicate / Delete / Rename (uniform across categories)
//==================================================================================================

void UiSceneBrowser::renderElementToolbar(const ElementTypeDesc& desc)
{
  const int sel = selectedElementIndex(desc);

  // Filter
  ImGui::SetNextItemWidth(180.0f);
  const std::string hint = std::string("Filter ") + desc.plural;
  ImGui::InputTextWithHint("##elemFilter", hint.c_str(), m_elementFilter, sizeof(m_elementFilter));
  ImGui::SameLine();

  // Add (single button, or a variant popup when there is more than one default). Stable ### id "add"
  // so UI-test scenarios can address it despite the leading icon glyph.
  if(!desc.addVariants.empty())
  {
    if(desc.addVariants.size() == 1)
    {
      if(ImGui::Button(ICON_MS_ADD " Add###add"))
        desc.addVariants[0].create();
    }
    else
    {
      if(ImGui::Button(ICON_MS_ADD " Add###add"))
        ImGui::OpenPopup("##addVariants");
      if(ImGui::BeginPopup("##addVariants"))
      {
        for(const ElementAddVariant& v : desc.addVariants)
        {
          // Icon + label for display, with a ### id (the label) so UI-test scenarios can address the item.
          const std::string label = std::string(v.icon) + " " + v.label + "###" + v.label;
          if(ImGui::MenuItem(label.c_str()))
            v.create();
        }
        ImGui::EndPopup();
      }
    }
    ImGui::SameLine();
  }

  // Duplicate (icon-only; stable ### id "dup" for UI-test scenarios)
  ImGui::BeginDisabled(sel < 0 || !desc.duplicate);
  if(ImGui::Button(ICON_MS_CONTENT_COPY "###dup") && desc.duplicate)
    desc.duplicate(sel);
  ImGui::EndDisabled();
  ImGui::SetItemTooltip("Duplicate %s", desc.singular);
  ImGui::SameLine();

  // Delete (refcount / read-only gated; stable ### id "del")
  const bool deletable = sel >= 0 && desc.remove && (!desc.canDelete || desc.canDelete(sel));
  ImGui::BeginDisabled(!deletable);
  if(ImGui::Button(ICON_MS_DELETE "###del") && desc.remove)
    desc.remove(sel);
  ImGui::EndDisabled();
  if(sel >= 0 && desc.remove && desc.canDelete && !desc.canDelete(sel) && desc.deleteDisabledReason
     && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("%s", desc.deleteDisabledReason(sel).c_str());
  else
    ImGui::SetItemTooltip("Delete %s", desc.singular);
  ImGui::SameLine();

  // Rename (stable ### id "ren")
  ImGui::BeginDisabled(sel < 0 || !desc.rename);
  if(ImGui::Button(ICON_MS_EDIT "###ren"))
    beginElementRename(desc, sel);
  ImGui::EndDisabled();
  ImGui::SetItemTooltip("Rename %s", desc.singular);
}

//==================================================================================================
// TABLE - clippered columnar list; whole-row selectable, per-type context menu
//==================================================================================================

void UiSceneBrowser::renderElementTable(const ElementTypeDesc& desc)
{
  const int    ncol    = int(desc.columns.size());
  const int    total   = desc.count();
  const float  footerH = ImGui::GetFrameHeightWithSpacing();
  const ImVec2 tableSize(0.0f, std::max(64.0f, ImGui::GetContentRegionAvail().y - footerH));

  constexpr ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersInnerV
                                    | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Sortable;

  // Per-category table ID: each category has its own column set, so they must not share persisted
  // column settings (a shared ID would mismatch column counts across categories and corrupt widths).
  const std::string tableId = std::string("##elemTable_") + desc.plural;
  if(ImGui::BeginTable(tableId.c_str(), ncol + 1, flags, tableSize))  // +1 for the leading "#" index column
  {
    const float cellPad    = ImGui::GetStyle().CellPadding.x * 2.0f;
    const float arrowWidth = ImGui::GetFontSize() + ImGui::GetStyle().ItemInnerSpacing.x;  // sort-arrow allowance

    // Leading "#" column: the glTF index of each element (glTF cross-references by index), sortable so a
    // click restores natural order. Sized to the widest index it will show.
    {
      char widest[16];
      std::snprintf(widest, sizeof(widest), "%d", total > 0 ? total - 1 : 0);
      const float idxWidth = std::max(ImGui::CalcTextSize("#").x, ImGui::CalcTextSize(widest).x) + cellPad + arrowWidth;
      ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, idxWidth);
    }
    for(int c = 0; c < ncol; ++c)
    {
      const ElementColumn& col      = desc.columns[c];
      const bool           sortable = col.sortKey || c == 0;
      ImGuiTableColumnFlags colFlag = (col.width > 0.0f) ? ImGuiTableColumnFlags_WidthFixed : ImGuiTableColumnFlags_WidthStretch;
      // Name (column 0) sorts lexically; numeric columns with a sortKey sort by value; the rest opt out.
      if(!sortable)
        colFlag |= ImGuiTableColumnFlags_NoSort;
      // Grow a fixed column so its header (plus the sort arrow) never clips; the user can still resize.
      float width = col.width;
      if(width > 0.0f)
        width = std::max(width, ImGui::CalcTextSize(col.header).x + cellPad + (sortable ? arrowWidth : 0.0f));
      ImGui::TableSetupColumn(col.header, colFlag, width);
    }
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableHeadersRow();

    // Read the clicked sort column (single-column sort) and rebuild the view for it.
    int  sortCol = -1;
    bool sortAsc = true;
    if(ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs(); specs && specs->SpecsCount > 0)
    {
      sortCol = specs->Specs[0].ColumnIndex;
      sortAsc = specs->Specs[0].SortDirection != ImGuiSortDirection_Descending;
    }
    buildElementView(desc, m_activeElementTab, sortCol, sortAsc);

    const bool useRows  = !m_elementView.identity;
    const int  rowCount = useRows ? int(m_elementView.rows.size()) : total;
    const int  selected = selectedElementIndex(desc);

    ImGuiListClipper clipper;
    clipper.Begin(rowCount);
    while(clipper.Step())
    {
      for(int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row)
      {
        const int idx = useRows ? m_elementView.rows[row] : row;
        ImGui::PushID(idx);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();  // "#" index column

        // Whole-row selectable (spans all columns); real cell content is drawn on the same line / next columns.
        const ImGuiSelectableFlags sflags = ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap;
        if(ImGui::Selectable("##sel", idx == selected, sflags))
          desc.select(idx);

        if(ImGui::BeginPopupContextItem("##rowctx"))
        {
          desc.select(idx);
          if(desc.contextMenuExtra)
            desc.contextMenuExtra(idx);
          if(desc.duplicate && ImGui::MenuItem(ICON_MS_CONTENT_COPY " Duplicate"))
            desc.duplicate(idx);
          if(desc.rename && ImGui::MenuItem(ICON_MS_EDIT " Rename"))
            beginElementRename(desc, idx);
          if(desc.remove)
          {
            const bool canDel = !desc.canDelete || desc.canDelete(idx);
            ImGui::BeginDisabled(!canDel);
            if(ImGui::MenuItem(ICON_MS_DELETE " Delete"))
              desc.remove(idx);
            ImGui::EndDisabled();
          }
          ImGui::EndPopup();
        }

        ImGui::SameLine();
        ImGui::TextDisabled("%d", idx);  // the glTF index of this element

        for(int c = 0; c < ncol; ++c)
        {
          ImGui::TableNextColumn();
          desc.columns[c].draw(idx);
        }
        ImGui::PopID();
      }
    }
    ImGui::EndTable();
  }

  // Footer summary: count (filtered "shown / total" when a filter is active) + optional aggregate.
  const bool filtered = m_elementFilter[0] != '\0';
  if(filtered)
    ImGui::Text("%d / %d %s", int(m_elementView.rows.size()), total, desc.plural);
  else
    ImGui::Text("%d %s", total, desc.plural);
  if(desc.footerAggregate)
  {
    const std::string agg = desc.footerAggregate();
    if(!agg.empty())
    {
      ImGui::SameLine();
      ImGui::TextDisabled("\xc2\xb7 %s", agg.c_str());  // " · <aggregate>"
    }
  }
}

//==================================================================================================
// ELEMENTS TAB - icon tab bar (one per category) + toolbar + table
//==================================================================================================

void UiSceneBrowser::renderElementsTab()
{
  if(!m_scene)
  {
    ImGui::TextDisabled("No scene loaded");
    return;
  }

  ensureElementRegistry();
  if(m_elementTypes.empty())
    return;

  if(ImGui::BeginTabBar("##elemCats", ImGuiTabBarFlags_FittingPolicyScroll | ImGuiTabBarFlags_TabListPopupButton))
  {
    for(int t = 0; t < int(m_elementTypes.size()); ++t)
    {
      const ElementTypeDesc& d = m_elementTypes[t];
      // Icon-only display, but a stable ASCII id (### => id is the plural) so UI-test scenarios and
      // tooltips can address the tab by name regardless of the glyph.
      const std::string label = std::string(d.icon) + "###" + d.plural;
      if(ImGui::BeginTabItem(label.c_str()))
      {
        if(m_activeElementTab != t)
          m_activeElementTab = t;  // switching category invalidates the filtered view on next build
        ImGui::EndTabItem();
      }
      if(ImGui::IsItemHovered())
        ImGui::SetTooltip("%s (%d)", d.plural, d.count());
    }
    ImGui::EndTabBar();
  }

  if(m_activeElementTab < 0 || m_activeElementTab >= int(m_elementTypes.size()))
    m_activeElementTab = 0;

  const ElementTypeDesc& desc = m_elementTypes[m_activeElementTab];
  renderElementToolbar(desc);
  ImGui::Separator();
  renderElementTable(desc);
  renderElementRenameDialog();
}
