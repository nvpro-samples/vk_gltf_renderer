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
// Property inspector UI panel. Displays and edits properties of the
// currently selected glTF element (node transforms, mesh info, material
// parameters, texture references, extensions, XMP metadata) using
// ImGui property-editor widgets.
//

#include "ui_inspector.hpp"
#include "ui_xmp.hpp"
#include "undo_redo.hpp"
#include "gltf_scene.hpp"
#include "gltf_scene_editor.hpp"
#include "tinygltf_utils.hpp"
#include "ui_helpers.hpp"
#include "ui_linear_color.hpp"
#include "ui_gltf_labels.hpp"

#include <algorithm>
#include <cstdio>

#include <imgui.h>
#include <imgui_internal.h>
#include <nvgui/property_editor.hpp>
#include <nvgui/fonts.hpp>
#include <nvutils/logger.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/color_space.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace PE = nvgui::PropertyEditor;

//==================================================================================================
// CONSTANTS
//==================================================================================================

static const double f64_zero = 0., f64_one = 1., f64_ten = 10., f64_179 = 179., f64_001 = 0.001, f64_1000 = 1000.,
                    f64_10000 = 10000., f64_01 = 0.1, f64_100 = 100., f64_neg1000 = -1000.;


// Linear color-factor editors (linear numbers + perceptual sRGB swatch/wheel). See header for details.
using uicolor::colorEdit3Linear;
using uicolor::colorEdit4Linear;

//==================================================================================================
// HELPER FUNCTIONS
//==================================================================================================

// "Load from file" for a texture slot: pick an image, import it as a new texture appended to the model,
// and assign it to this slot. Returns true when a texture was imported so the caller propagates it as a
// normal material change (write-back of copy-based rows + per-frame EditMaterialCommand for undo of the
// slot). importImageAsTexture sets DirtyFlags::texturesChanged; the renderer does the GPU rebuild after
// the panels render, so material write-backs have flushed and it can resolve sRGB from final usage.
template <typename T>
bool UiInspector::importTextureIntoSlot(T& info)
{
  if(!m_host.canPickImage() || !m_scene)
    return false;

  const std::filesystem::path path = m_host.pickImage();
  if(path.empty())
    return false;  // cancelled

  std::string error;
  const int   textureIndex = m_scene->editor().importImageAsTexture(path, &error);
  if(textureIndex < 0)
  {
    LOGE("Texture import failed: %s\n", error.c_str());
    m_host.toast("Texture import failed: " + error, true);
    return false;
  }

  // Make the resource append undoable (the slot assignment below rides the caller's EditMaterialCommand).
  if(m_undoStack)
  {
    const int imageIndex = m_scene->getModel().textures[textureIndex].source;
    m_undoStack->pushExecuted(std::make_unique<ImportImageAsTextureCommand>(*m_scene, imageIndex, textureIndex,
                                                                            "Import texture " + path.stem().string()));
  }

  info.index = textureIndex;  // keep the slot's existing texCoord (UV set); only the image changes
  refreshTextureNames();      // new texture appears in the picker/rows immediately
  return true;
}

// Texture editing UI - one editable slot. Returns true when the slot's texture index changed (assign /
// switch / clear / import), so the caller writes back copy-based rows and records the undo step.
template <typename T>
bool UiInspector::renderTextureEditRow(const char* label, T& info)
{
  const tinygltf::Model& model       = m_scene->getModel();
  const bool             canPickFile = m_host.canPickImage();

  bool changed = false;

  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::Text("%s %s", ICON_MS_IMAGE, label);

  ImGui::TableNextColumn();
  const bool hasTexture = info.index >= 0;
  if(hasTexture)
  {
    // Use pre-computed texture name from m_textureNames
    const std::string displayName = (info.index >= 0 && info.index < static_cast<int>(m_textureNames.size())) ?
                                        m_textureNames[info.index] :
                                        "Invalid texture " + std::to_string(info.index);

    // Small thumbnail swatch. Advancing the cursor past it means the name/button width math below
    // (which reads GetContentRegionAvail) already accounts for it.
    const ImTextureID thumb = m_host.thumbnail(info.index);
    if(thumb != 0)
    {
      // A button (not a plain Image) so the click reliably registers and the swatch shows a hover
      // highlight. Zero frame padding keeps its footprint at one text line, so the name/button width
      // math below is unchanged. str_id = label keeps it unique across the material's slots.
      const float sz = ImGui::GetTextLineHeight();
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
      const bool clicked = ImGui::ImageButton(label, thumb, ImVec2(sz, sz));
      ImGui::PopStyleVar();
      if(ImGui::IsItemHovered())
      {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::SetTooltip("Click to view full image");
      }
      // Click the swatch to open the full-size image viewer (resolving the texture to its backing image).
      if(m_onViewImage && clicked && info.index < static_cast<int>(model.textures.size()))
      {
        const int imgIdx = tinygltf::utils::getTextureImageIndex(model.textures[info.index]);
        if(imgIdx >= 0)
          m_onViewImage(imgIdx);
      }
      ImGui::SameLine(0.0f, 4.0f);
    }

    // Reserve room for the action buttons that follow (switch, [load], UV transform, delete) so the
    // ellipsized name never overlaps them.
    const int    buttonCount  = canPickFile ? 4 : 3;
    const float  buttonWidth  = ImGui::CalcTextSize(ICON_MS_DELETE).x + ImGui::GetStyle().FramePadding.x * 2.0f;
    const float  spacing      = ImGui::GetStyle().ItemSpacing.x;
    const float  available    = ImGui::GetContentRegionAvail().x;
    const float  textMaxWidth = std::max(0.0f, available - buttonCount * (buttonWidth + spacing));
    const ImVec2 textStart    = ImGui::GetCursorScreenPos();
    const ImVec2 textEnd(textStart.x + textMaxWidth, textStart.y + ImGui::GetTextLineHeight());

    // Draw text over the button
    ImGui::RenderTextEllipsis(ImGui::GetWindowDrawList(), textStart, textEnd, textEnd.x, displayName.c_str(), nullptr, nullptr);
    ImGui::Dummy(ImVec2(textMaxWidth, ImGui::GetTextLineHeight()));

    // Enhanced tooltip showing texture -> image -> URI
    if(ImGui::IsItemHovered())
    {
      ImGui::BeginTooltip();
      ImGui::TextUnformatted(displayName.c_str());

      // Enlarged preview so the user can actually see the image, not just its name.
      if(thumb != 0)
        ImGui::Image(thumb, ImVec2(160.0f, 160.0f));

      // Show texture -> image -> URI path
      if(info.index >= 0 && info.index < static_cast<int>(model.textures.size()))
      {
        const tinygltf::Texture& texture = model.textures[info.index];
        ImGui::Separator();

        const int imageIdx = tinygltf::utils::getTextureImageIndex(texture);
        if(imageIdx >= 0 && imageIdx < static_cast<int>(model.images.size()))
        {
          const tinygltf::Image& image = model.images[imageIdx];

          // Check if image is embedded or has a URI
          if(!image.uri.empty())
          {
            ImGui::Text("Image: %s", image.uri.c_str());
          }
          else
          {
            ImGui::Text("Image: %d (EMB)", imageIdx);
          }
        }
        else
        {
          ImGui::TextDisabled("No image source");
        }
      }

      ImGui::EndTooltip();
    }

    ImGui::SameLine();
  }

  ImGui::PushID(label);
  if(hasTexture)
  {
    if(ImGui::SmallButton(ICON_MS_SWAP_HORIZ))
    {
      ImGui::OpenPopup("SwitchTexture");
    }
    if(ImGui::IsItemHovered())
    {
      ImGui::BeginTooltip();
      ImGui::TextUnformatted("Switch texture");
      ImGui::EndTooltip();
    }
    if(canPickFile)
    {
      ImGui::SameLine(0.0f, 2.0f);
      if(ImGui::SmallButton(ICON_MS_FILE_OPEN))
      {
        changed |= importTextureIntoSlot(info);
      }
      if(ImGui::IsItemHovered())
      {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted("Load from file");
        ImGui::EndTooltip();
      }
    }
    ImGui::SameLine(0.0f, 2.0f);
    changed |= renderTextureTransformButton(info);
    ImGui::SameLine(0.0f, 2.0f);  // Reduce spacing between buttons
    if(ImGui::SmallButton(ICON_MS_DELETE))
    {
      info.index = -1;
      changed    = true;
    }
    if(ImGui::IsItemHovered())
    {
      ImGui::BeginTooltip();
      ImGui::TextUnformatted("Remove texture");
      ImGui::EndTooltip();
    }
  }
  else
  {
    // Empty slot: assign an existing texture (only when the scene has any) and always allow loading a
    // new one from file. The load path works even on a scene with zero textures.
    bool needSpacing = false;
    if(!m_textureNames.empty())
    {
      if(ImGui::SmallButton(ICON_MS_IMAGE_SEARCH))
      {
        ImGui::OpenPopup("SwitchTexture");
      }
      if(ImGui::IsItemHovered())
      {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted("Assign existing texture");
        ImGui::EndTooltip();
      }
      needSpacing = true;
    }
    if(canPickFile)
    {
      if(needSpacing)
        ImGui::SameLine(0.0f, 2.0f);
      if(ImGui::SmallButton(ICON_MS_FILE_OPEN))
      {
        changed |= importTextureIntoSlot(info);
      }
      if(ImGui::IsItemHovered())
      {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted("Load from file");
        ImGui::EndTooltip();
      }
    }
    else if(m_textureNames.empty())
    {
      ImGui::TextDisabled(ICON_MS_ADD_CIRCLE);  // no import hook wired: nothing to do
    }
  }

  // Existing-texture selection popup (assign / switch). Only meaningful when the scene has textures.
  changed |= renderTexturePicker(info, hasTexture);
  ImGui::PopID();

  return changed;
}

// Small per-slot UV-transform (KHR_texture_transform) editor: a button that opens a popup. glTF stores
// the transform on this material's texture reference, so it is edited here per binding -- not on the
// shared texture, whose other references may use a different transform. Absent extension -> the popup
// offers "Add"; present -> offset/rotation/scale + "Remove". The returned flag folds into the row's
// change flag, so the edit rides the material's existing EditMaterialCommand undo step.
template <typename T>
bool UiInspector::renderTextureTransformButton(T& info)
{
  bool       changed = false;
  const bool hasT    = tinygltf::utils::hasTextureTransform(info);

  if(hasT)
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.40f, 0.70f, 1.00f, 1.00f));  // tint when a transform is present
  const bool open = ImGui::SmallButton(ICON_MS_TRANSFORM);
  if(hasT)
    ImGui::PopStyleColor();
  if(open)
    ImGui::OpenPopup("UVTransform");
  if(ImGui::IsItemHovered())
  {
    ImGui::BeginTooltip();
    ImGui::TextUnformatted(hasT ? "UV transform (KHR_texture_transform)" : "Add UV transform (KHR_texture_transform)");
    ImGui::EndTooltip();
  }

  if(ImGui::BeginPopup("UVTransform"))
  {
    if(tinygltf::utils::hasTextureTransform(info))
    {
      KHR_texture_transform tt     = tinygltf::utils::getTextureTransform(info);
      bool                  edited = false;
      ImGui::SetNextItemWidth(160.0f);
      edited |= ImGui::DragFloat2("Offset", glm::value_ptr(tt.offset), 0.001f, 0.0f, 0.0f, "%.4f");
      ImGui::SetNextItemWidth(160.0f);
      edited |= ImGui::DragFloat("Rotation", &tt.rotation, 0.005f, 0.0f, 0.0f, "%.4f rad");
      ImGui::SetNextItemWidth(160.0f);
      edited |= ImGui::DragFloat2("Scale", glm::value_ptr(tt.scale), 0.01f, 0.0f, 0.0f, "%.4f");
      if(edited)
      {
        tinygltf::utils::setTextureTransform(info, tt);
        changed = true;
      }
      ImGui::Separator();
      if(ImGui::SmallButton("Remove"))
      {
        tinygltf::utils::removeTextureTransform(info);
        changed = true;
        ImGui::CloseCurrentPopup();
      }
    }
    else
    {
      ImGui::TextDisabled("No KHR_texture_transform on this binding.");
      if(ImGui::SmallButton("Add"))
      {
        tinygltf::utils::setTextureTransform(info, KHR_texture_transform{});  // identity; fields appear next frame
        changed = true;
      }
    }
    ImGui::EndPopup();
  }
  return changed;
}

template <typename T>
bool UiInspector::renderTexturePicker(T& info, bool hasTexture)
{
  if(m_textureNames.empty())
    return false;

  bool changed = false;
  bool open    = true;
  ImGui::SetNextWindowSize(ImVec2(520.0f, 440.0f), ImGuiCond_Once);
  if(!ImGui::BeginPopupModal("SwitchTexture", &open, ImGuiWindowFlags_None))
    return false;

  ImGuiStorage* storage = ImGui::GetStateStorage();
  ImGuiID       listId  = ImGui::GetID("texture_select_index");
  int           selIdx  = storage->GetInt(listId, hasTexture ? info.index : 0);
  selIdx                = std::clamp(selIdx, 0, static_cast<int>(m_textureNames.size() - 1));

  ImGui::TextUnformatted(hasTexture ? "Switch to texture:" : "Select texture:");

  // Text filter first: at thousands of textures, typing a name is the fast way to find one.
  static ImGuiTextFilter s_filter;
  s_filter.Draw(ICON_MS_SEARCH " Filter", 240.0f);

  // List / grid view toggle (persisted for the session). A pressed-looking button marks the active view.
  static bool s_gridView = false;
  auto        toggleView = [&](const char* icon, bool active) {
    if(active)
      ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    const bool clicked = ImGui::Button(icon);
    if(active)
      ImGui::PopStyleColor();
    return clicked;
  };
  ImGui::SameLine();
  if(toggleView(ICON_MS_VIEW_LIST, !s_gridView))
    s_gridView = false;
  ImGui::SameLine(0.0f, 2.0f);
  if(toggleView(ICON_MS_GRID_VIEW, s_gridView))
    s_gridView = true;

  // Filtered index list. Both views clip so only on-screen entries request a thumbnail (bounding the
  // number of live descriptor sets on huge scenes).
  std::vector<int> filtered;
  filtered.reserve(m_textureNames.size());
  for(int i = 0; i < static_cast<int>(m_textureNames.size()); ++i)
    if(s_filter.PassFilter(m_textureNames[i].c_str()))
      filtered.push_back(i);

  // The list fills the popup, leaving only the OK/Cancel row below, so it grows when the window resizes.
  const float  lineH = ImGui::GetTextLineHeight();
  const ImVec2 listSize(-FLT_MIN, -ImGui::GetFrameHeightWithSpacing());
  if(ImGui::BeginChild("##TextureList", listSize, ImGuiChildFlags_FrameStyle))
  {
    if(s_gridView)
    {
      const float tile  = 72.0f;
      const float pad   = ImGui::GetStyle().ItemSpacing.x;
      const int   cols  = std::max(1, static_cast<int>((ImGui::GetContentRegionAvail().x + pad) / (tile + pad)));
      const int   nrows = (static_cast<int>(filtered.size()) + cols - 1) / cols;

      ImGuiListClipper clipper;
      clipper.Begin(nrows, tile + lineH + ImGui::GetStyle().ItemSpacing.y);
      while(clipper.Step())
      {
        for(int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r)
        {
          for(int c = 0; c < cols; ++c)
          {
            const int flat = r * cols + c;
            if(flat >= static_cast<int>(filtered.size()))
              break;
            const int i = filtered[flat];
            if(c > 0)
              ImGui::SameLine();
            ImGui::PushID(i);
            const bool   selected = (selIdx == i);
            const ImVec2 p0       = ImGui::GetCursorScreenPos();
            if(ImGui::Selectable("##tile", selected, ImGuiSelectableFlags_None, ImVec2(tile, tile + lineH)))
              selIdx = i;
            if(selected)
              ImGui::SetItemDefaultFocus();
            if(ImGui::IsItemHovered())
              ImGui::SetTooltip("%s", m_textureNames[i].c_str());
            // Draw the thumbnail + ellipsized name on the draw list so only the selectable drives layout.
            ImDrawList*       dl = ImGui::GetWindowDrawList();
            const ImTextureID th = m_host.thumbnail(i);
            if(th != 0)
              dl->AddImage(th, p0, ImVec2(p0.x + tile, p0.y + tile));
            ImGui::RenderTextEllipsis(dl, ImVec2(p0.x, p0.y + tile), ImVec2(p0.x + tile, p0.y + tile + lineH),
                                      p0.x + tile, m_textureNames[i].c_str(), nullptr, nullptr);
            ImGui::PopID();
          }
        }
      }
      clipper.End();
    }
    else
    {
      const float      thumbSz = 22.0f;
      const float      rowH    = std::max(24.0f, lineH) + ImGui::GetStyle().ItemSpacing.y;
      ImGuiListClipper clipper;
      clipper.Begin(static_cast<int>(filtered.size()), rowH);
      while(clipper.Step())
      {
        for(int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row)
        {
          const int i = filtered[row];
          ImGui::PushID(i);
          const bool selected = (selIdx == i);
          if(ImGui::Selectable("##row", selected, ImGuiSelectableFlags_None, ImVec2(0.0f, thumbSz)))
            selIdx = i;
          if(selected)
            ImGui::SetItemDefaultFocus();
          ImGui::SameLine(0.0f, 4.0f);
          const ImTextureID th = m_host.thumbnail(i);
          if(th != 0)
            ImGui::Image(th, ImVec2(thumbSz, thumbSz));
          else
            ImGui::Dummy(ImVec2(thumbSz, thumbSz));
          ImGui::SameLine(0.0f, 6.0f);
          ImGui::AlignTextToFramePadding();
          ImGui::TextUnformatted(m_textureNames[i].c_str());
          ImGui::PopID();
        }
      }
      clipper.End();
    }
  }
  ImGui::EndChild();
  storage->SetInt(listId, selIdx);

  if(ImGui::Button(ICON_MS_CHECK " OK"))
  {
    info.index = selIdx;  // keep the slot's existing texCoord (UV set); only the referenced texture changes
    changed    = true;
    ImGui::CloseCurrentPopup();
  }
  ImGui::SameLine();
  if(ImGui::Button(ICON_MS_CANCEL " Cancel"))
  {
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();

  return changed;
}

//==================================================================================================
// INITIALIZATION
//==================================================================================================

UiInspector::~UiInspector() = default;

void UiInspector::setScene(nvvkgltf::Scene* scene)
{
  m_scene = scene;

  if(!scene)
  {
    m_textureNames.clear();
    return;
  }

  refreshTextureNames();
}

void UiInspector::refreshTextureNames()
{
  m_textureNames.clear();
  if(!m_scene)
    return;

  const tinygltf::Model& model = m_scene->getModel();
  m_textureNames.reserve(model.textures.size());
  for(int i = 0; i < static_cast<int>(model.textures.size()); ++i)
  {
    m_textureNames.push_back(tinygltf::utils::getTextureUiLabel(model, i));
  }
}

//==================================================================================================
// MAIN RENDER
//==================================================================================================

void UiInspector::render(bool* show, bool isBusy)
{
  if(show && !*show)
    return;

  if(ImGui::Begin("Inspector", show))
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
      ImGui::Text("Inspector will be available once the operation completes.");
      ImGui::End();
      return;
    }

    if(!m_selection || !m_selection->hasSelection())
    {
      renderNoSelection();
      ImGui::End();
      return;
    }

    // Route to appropriate property editor based on selection type
    auto sel = m_selection->getSelection();

    switch(sel.type)
    {
      case SceneSelection::SelectionType::eNode:
        renderNodeProperties(sel.nodeIndex);
        break;

      case SceneSelection::SelectionType::ePrimitive:
        renderPrimitiveProperties(sel.nodeIndex, sel.primitiveIndex, sel.meshIndex);
        break;

      case SceneSelection::SelectionType::eMaterial:
        renderMaterialProperties(sel.materialIndex);
        break;

      case SceneSelection::SelectionType::eMesh:
        renderMeshProperties(sel.meshIndex);
        break;

      case SceneSelection::SelectionType::eCamera:
        renderCameraProperties(sel.cameraIndex);
        break;

      case SceneSelection::SelectionType::eLight:
        renderLightProperties(sel.lightIndex);
        break;

      case SceneSelection::SelectionType::eTexture:
        renderTextureProperties(sel.textureIndex);
        break;

      case SceneSelection::SelectionType::eImage:
        renderImageProperties(sel.imageIndex);
        break;

      case SceneSelection::SelectionType::eSampler:
        renderSamplerProperties(sel.samplerIndex);
        break;

      case SceneSelection::SelectionType::eAnimation:
        renderAnimationProperties(sel.animationIndex);
        break;

      default:
        renderNoSelection();
        break;
    }
  }
  ImGui::End();
}

//==================================================================================================
// NO SELECTION
//==================================================================================================

void UiInspector::renderNoSelection()
{
  ImGui::TextDisabled("No selection");
  ImGui::Separator();
  ImGui::TextWrapped("Select an element in the Scene Browser or 3D view to view its properties.");
}

//==================================================================================================
// IES EDITOR (used by the Node inspector for any node carrying EXT_lights_ies -- see header for
// contract. Per spec the extension is standalone and does not compose with KHR_lights_punctual, so
// this is the only IES editing entry point in the UI; if a node also carries a KHR attachment the
// caller displays a warning line above this editor.)
//==================================================================================================

bool UiInspector::renderIesEditor(int nodeIdx)
{
  if(!m_scene || nodeIdx < 0)
    return false;
  const tinygltf::Model& model = m_scene->getModel();
  if(nodeIdx >= int(model.nodes.size()))
    return false;

  EXT_lights_ies_ref iesRef = tinygltf::utils::getNodeIesLight(model.nodes[nodeIdx]);
  if(iesRef.light < 0)
    return false;

  // Profile identity line -- read-only (index refers to the loaded .ies array).
  std::vector<EXT_lights_ies_profile> profiles = tinygltf::utils::getIesProfiles(model);
  std::string                         profileLabel;
  if(iesRef.light < int(profiles.size()))
    profileLabel = profiles[iesRef.light].name.empty() ? profiles[iesRef.light].uri : profiles[iesRef.light].name;
  ImGui::TextDisabled("%s IES profile [%d]: %s", ICON_MS_FLASHLIGHT_ON, iesRef.light, profileLabel.c_str());

  // Capture pre-edit state before any widget can modify iesRef (for undo snapshot).
  const EXT_lights_ies_ref preEditIesRef = iesRef;
  const bool               readOnly      = m_scene->isNodeReadOnly(nodeIdx);

  bool iesModif = false;
  ImGui::BeginDisabled(readOnly);
  if(PE::begin())
  {
    if(PE::DragFloat("Multiplier", &iesRef.multiplier, 0.01f, 0.0f, 100.0f, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat,
                     "Non-negative scale on the IES candela distribution.\n"
                     "Acts as a brightness multiplier on top of the profile."))
      iesModif = true;
    if(colorEdit3Linear("Color", glm::value_ptr(iesRef.color),
                        "RGB tint in linear space, clamped to [0,1].\n"
                        "Applied as a per-channel multiplier on the IES output."))
      iesModif = true;
    PE::end();
  }
  ImGui::EndDisabled();

  if(iesModif)
  {
    tinygltf::utils::setNodeIesLight(m_scene->editor().getNodeForEdit(nodeIdx), iesRef);
    m_scene->markNodeDirty(nodeIdx);
  }

  // Snapshot on first edit frame (uses preEditIesRef, before any write-back).
  // Push undo command when the drag/edit cycle ends.
  if(iesModif && !m_iesModifiedLastFrame)
  {
    m_iesNodeSnapshotIdx = nodeIdx;
    m_iesSnapshotMult    = preEditIesRef.multiplier;
    m_iesSnapshotColor   = preEditIesRef.color;
  }
  if(!iesModif && m_iesModifiedLastFrame && m_iesNodeSnapshotIdx == nodeIdx && m_undoStack)
  {
    EditNodeIesCommand::IesParams before{m_iesSnapshotMult, m_iesSnapshotColor};
    EditNodeIesCommand::IesParams after{iesRef.multiplier, iesRef.color};
    m_undoStack->pushExecuted(std::make_unique<EditNodeIesCommand>(*m_scene, nodeIdx, before, after));
  }
  m_iesModifiedLastFrame = iesModif;
  return true;
}

//==================================================================================================
// NODE PROPERTIES
//==================================================================================================

void UiInspector::renderNodeProperties(int nodeIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  if(nodeIdx < 0 || nodeIdx >= static_cast<int>(model.nodes.size()))
    return;

  const tinygltf::Node& node = model.nodes[nodeIdx];

  ImGui::Text("%s Node[%d]: %s", ICON_MS_CATEGORY, nodeIdx, node.name.c_str());

  // XMP button
  std::string popupId = "inspector_node_xmp_" + std::to_string(nodeIdx);
  ImGui::SameLine();
  ui_xmp::renderInfoButton(&m_scene->getModel(), node.extensions, popupId.c_str());

  ImGui::Separator();

  // glTF 2.1: nodes merged in from a referenced external asset are read-only.
  const bool readOnly = m_scene->isNodeReadOnly(nodeIdx);
  if(readOnly)
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s Referenced asset (read-only)", ICON_MS_LOCK);

  ImGui::BeginDisabled(readOnly);
  if(ImGui::CollapsingHeader("TRANSFORM", ImGuiTreeNodeFlags_DefaultOpen))
    renderTransformSection(nodeIdx);
  ImGui::EndDisabled();

  // Pure navigation (jump links), not an edit -- stays clickable on a read-only node.
  if(ImGui::CollapsingHeader("RELATIONSHIPS", ImGuiTreeNodeFlags_DefaultOpen))
    renderNodeRelationships(nodeIdx);

  // EXT_lights_ies is standalone per spec. If the node also carries KHR_lights_punctual, the KHR
  // attachment is ignored (see Scene::handleLightTraversal) -- surface that inline so the author
  // knows why edits to the KHR light have no effect on this node.
  if(tinygltf::utils::getNodeIesLight(node).light >= 0)
  {
    if(ImGui::CollapsingHeader((std::string(ICON_MS_FLASHLIGHT_ON) + " IES LIGHT (EXT_lights_ies)").c_str(), ImGuiTreeNodeFlags_DefaultOpen))
    {
      if(node.light >= 0)
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                           "%s KHR_lights_punctual on this node is ignored (EXT_lights_ies is standalone).", ICON_MS_WARNING);
      renderIesEditor(nodeIdx);
    }
  }

  ImGui::BeginDisabled(readOnly);
  if(ImGui::CollapsingHeader("NODE EXTENSIONS"))
    renderNodeExtensions(nodeIdx);
  ImGui::EndDisabled();
}

//==================================================================================================
// ELEMENT CROSS-REFERENCE LINK + NODE RELATIONSHIPS / EXTENSIONS
//==================================================================================================

bool UiInspector::elementLink(const char* label, SceneSelection::SelectionType kind, int index)
{
  if(index < 0)
  {
    ImGui::TextDisabled("%s", label);
    return false;
  }
  ImGui::PushID(label);
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.42f, 0.72f, 1.00f, 1.0f));  // link blue
  const bool clicked = ImGui::Selectable(label, false, ImGuiSelectableFlags_None, ImGui::CalcTextSize(label));
  ImGui::PopStyleColor();
  ImGui::PopID();
  if(ImGui::IsItemHovered())
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
  if(clicked && m_selection)
  {
    switch(kind)
    {
      case SceneSelection::SelectionType::eNode:
        m_selection->selectNode(index);
        break;
      case SceneSelection::SelectionType::eMesh:
        m_selection->selectMesh(index);
        break;
      case SceneSelection::SelectionType::eMaterial:
        m_selection->selectMaterial(index);
        break;
      case SceneSelection::SelectionType::eCamera:
        m_selection->selectCamera(index);
        break;
      case SceneSelection::SelectionType::eLight:
        m_selection->selectLight(index);
        break;
      case SceneSelection::SelectionType::eTexture:
        m_selection->selectTexture(index);
        break;
      case SceneSelection::SelectionType::eImage:
        m_selection->selectImage(index);
        break;
      case SceneSelection::SelectionType::eSampler:
        m_selection->selectSampler(index);
        break;
      case SceneSelection::SelectionType::eAnimation:
        m_selection->selectAnimation(index);
        break;
      default:
        break;
    }
  }
  return clicked;
}

std::string UiInspector::elementRefLabel(SceneSelection::SelectionType kind, int index) const
{
  if(index < 0)
    return {};
  const tinygltf::Model& model = m_scene->getModel();
  const char*            icon  = ICON_MS_CATEGORY;
  std::string            name;
  switch(kind)
  {
    case SceneSelection::SelectionType::eNode:
      if(index < int(model.nodes.size()))
      {
        const tinygltf::Node& n = model.nodes[index];
        icon                    = n.mesh >= 0   ? ICON_MS_VIEW_IN_AR :
                                  n.camera >= 0 ? ICON_MS_CAMERA_ALT :
                                  n.light >= 0  ? ICON_MS_LIGHTBULB :
                                                  ICON_MS_CATEGORY;
        name                    = n.name;
      }
      break;
    case SceneSelection::SelectionType::eMesh:
      icon = ICON_MS_VIEW_IN_AR;
      if(index < int(model.meshes.size()))
        name = model.meshes[index].name;
      break;
    case SceneSelection::SelectionType::eMaterial:
      icon = ICON_MS_BRUSH;
      if(index < int(model.materials.size()))
        name = model.materials[index].name;
      break;
    case SceneSelection::SelectionType::eCamera:
      icon = ICON_MS_CAMERA_ALT;
      if(index < int(model.cameras.size()))
        name = model.cameras[index].name;
      break;
    case SceneSelection::SelectionType::eLight:
      icon = ICON_MS_LIGHTBULB;
      if(index < int(model.lights.size()))
        name = model.lights[index].name;
      break;
    case SceneSelection::SelectionType::eTexture:
      icon = ICON_MS_IMAGE;
      if(index < int(model.textures.size()))
        name = tinygltf::utils::getTextureUiLabel(model, index);
      break;
    case SceneSelection::SelectionType::eImage:
      icon = ICON_MS_PHOTO;
      name = uigltf::imageDisplayName(model, index);
      break;
    case SceneSelection::SelectionType::eSampler:
      icon = ICON_MS_TUNE;
      name = "Sampler " + std::to_string(index);
      break;
    case SceneSelection::SelectionType::eAnimation:
      icon = ICON_MS_MOVIE;
      if(index < int(model.animations.size()))
        name = model.animations[index].name;
      break;
    default:
      break;
  }
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%s [%d] %s", icon, index, name.c_str());
  return buf;
}

void UiInspector::elementLinkRow(const char* rowLabel, SceneSelection::SelectionType kind, int index, float labelWidth, const char* emptyText)
{
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(rowLabel);
  ImGui::SameLine(labelWidth);
  if(index < 0)
    ImGui::TextDisabled("%s", emptyText);
  else
    elementLink(elementRefLabel(kind, index).c_str(), kind, index);
}

void UiInspector::renderTexturesUsing(const char* header, const std::function<bool(const tinygltf::Texture&)>& match)
{
  const tinygltf::Model& model = m_scene->getModel();
  ImGui::TextUnformatted(header);
  ImGui::Indent();
  bool any = false;
  for(int t = 0; t < int(model.textures.size()); ++t)
    if(match(model.textures[t]))
    {
      any = true;
      elementLink(elementRefLabel(SceneSelection::SelectionType::eTexture, t).c_str(), SceneSelection::SelectionType::eTexture, t);
    }
  if(!any)
    ImGui::TextDisabled("(none)");
  ImGui::Unindent();
}

void UiInspector::renderNodeRelationships(int nodeIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  const tinygltf::Node&  node  = model.nodes[nodeIdx];

  elementLinkRow("Parent", SceneSelection::SelectionType::eNode, m_scene->editor().getNodeParent(nodeIdx), 120.0f, "Scene root");
  elementLinkRow("Mesh", SceneSelection::SelectionType::eMesh, node.mesh);
  elementLinkRow("Camera", SceneSelection::SelectionType::eCamera, node.camera);
  elementLinkRow("Light", SceneSelection::SelectionType::eLight, node.light);

  // Skin has no dedicated inspector/selection kind yet; show its index/name for reference.
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Skin");
  ImGui::SameLine(120.0f);
  if(node.skin < 0)
    ImGui::TextDisabled("-");
  else
    ImGui::Text("[%d] %s", node.skin, node.skin < int(model.skins.size()) ? model.skins[node.skin].name.c_str() : "");

  // Children (each a jump link)
  ImGui::Spacing();
  ImGui::Text("Children (%zu)", node.children.size());
  ImGui::Indent();
  if(node.children.empty())
    ImGui::TextDisabled("none");
  for(int c : node.children)
    if(c >= 0 && c < int(model.nodes.size()))
      elementLink(elementRefLabel(SceneSelection::SelectionType::eNode, c).c_str(), SceneSelection::SelectionType::eNode, c);
  ImGui::Unindent();
}

void UiInspector::renderNodeExtensions(int nodeIdx)
{
  tinygltf::Node& node = m_scene->editor().getNodeForEdit(nodeIdx);

  // Snapshot the current value of `extName` (or a null-typed sentinel if absent), compute the new value
  // by calling `setter` on a scratch node, and push an undoable SetNodeExtensionCommand -- so toggling or
  // adding a KHR_node_* flag is a normal reversible edit instead of a direct node.extensions mutation.
  auto pushExtensionEdit = [&](const char* extName, const auto& setter, const char* description) {
    tinygltf::Value oldValue =
        tinygltf::utils::hasElementName(node.extensions, extName) ? node.extensions.at(extName) : tinygltf::Value{};
    tinygltf::Node scratch;
    setter(scratch);
    m_undoStack->executeCommand(std::make_unique<SetNodeExtensionCommand>(*m_scene, nodeIdx, extName, oldValue,
                                                                          scratch.extensions.at(extName), description));
  };

  // Remove `extName` from the node. Emits a SetNodeExtensionCommand with a NULL-typed newValue, which
  // SetNodeExtensionCommand::apply() interprets as "erase". Undo restores the previous value verbatim.
  auto popExtensionEdit = [&](const char* extName, const char* description) {
    if(!tinygltf::utils::hasElementName(node.extensions, extName))
      return;
    tinygltf::Value oldValue = node.extensions.at(extName);
    m_undoStack->executeCommand(
        std::make_unique<SetNodeExtensionCommand>(*m_scene, nodeIdx, extName, oldValue, tinygltf::Value{}, description));
  };

  // All three KHR_node_* extensions we expose here (visibility, selectability, hoverability) share the
  // same shape: a struct with a single bool flag, a get/set utility pair, and identical UI (checkbox +
  // inline Remove when present, Add when absent). This aggregate is the *data* -- one entry per row.
  struct NodeExtRow
  {
    const char* extName;     // KHR_NODE_*_EXTENSION_NAME
    const char* label;       // property-editor label (also the ImGui id root)
    const char* toggleTip;   // tooltip shown on the checkbox row
    const char* addTip;      // tooltip shown on the Add row (empty for none)
    const char* idTag;       // ImGui id suffix so Add##<tag> / Remove##<tag> don't collide
    const char* onDesc;      // undo-history label when the flag turns ON
    const char* offDesc;     // undo-history label when the flag turns OFF
    const char* addDesc;     // undo-history label when the extension is added
    const char* removeDesc;  // undo-history label when the extension is removed
  };

  // The *pattern*, defined once. `Ext` is deduced from the pointer-to-member `flag`; `get`/`set` are
  // taken as auto (their signatures are still checked at the call sites in the body). Adding a fourth
  // KHR_node_* boolean-flag extension is one more data row below -- no change to this helper.
  // NOTE: `get`/`set` are intentionally not typed as function pointers here -- MSVC's C++20 template-
  // lambda parser fails to resolve the `Ext` template parameter inside a function-pointer type in the
  // parameter list (C4430 "missing type specifier - int assumed"), even though clang/gcc accept it.
  auto renderNodeExtRow = [&]<class Ext>(const NodeExtRow& r, auto get, auto set, bool Ext::* flag) {
    if(tinygltf::utils::hasElementName(node.extensions, r.extName))
    {
      Ext current = get(node);
      if(PE::Checkbox(r.label, &(current.*flag), r.toggleTip))
      {
        pushExtensionEdit(r.extName, [&](tinygltf::Node& n) { set(n, current); }, (current.*flag) ? r.onDesc : r.offDesc);
      }
      ImGui::SameLine();
      const std::string removeId = std::string("Remove##") + r.idTag;
      if(ImGui::SmallButton(removeId.c_str()))
      {
        popExtensionEdit(r.extName, r.removeDesc);
      }
    }
    else
    {
      const std::string addId = std::string("Add##") + r.idTag;
      if(PE::entry(r.label, [&] { return ImGui::SmallButton(addId.c_str()); }, r.addTip))
      {
        pushExtensionEdit(r.extName, [&](tinygltf::Node& n) { set(n, Ext{}); }, r.addDesc);
      }
    }
  };

  ImGui::TextDisabled("glTF KHR_node_* extensions. Toggling adds the extension to the node if absent; use Remove to strip it.");
  if(PE::begin())
  {
    renderNodeExtRow(
        {
            .extName    = KHR_NODE_VISIBILITY_EXTENSION_NAME,
            .label      = "Visible (KHR_node_visibility)",
            .toggleTip  = "Hide the node and its children from rendering.",
            .addTip     = "Add KHR_node_visibility so the node can be hidden.",
            .idTag      = "vis",
            .onDesc     = "Show node",
            .offDesc    = "Hide node",
            .addDesc    = "Add node visibility",
            .removeDesc = "Remove node visibility",
        },
        &tinygltf::utils::getNodeVisibility, &tinygltf::utils::setNodeVisibility, &KHR_node_visibility::visible);

    renderNodeExtRow(
        {
            .extName    = KHR_NODE_SELECTABILITY_EXTENSION_NAME,
            .label      = "Selectable (KHR_node_selectability)",
            .toggleTip  = "When off, clicking this node (or a child) selects the nearest selectable ancestor instead.",
            .addTip     = "",
            .idTag      = "sel",
            .onDesc     = "Make selectable",
            .offDesc    = "Make unselectable",
            .addDesc    = "Add node selectability",
            .removeDesc = "Remove node selectability",
        },
        &tinygltf::utils::getNodeSelectability, &tinygltf::utils::setNodeSelectability, &KHR_node_selectability::selectable);

    renderNodeExtRow(
        {
            .extName    = KHR_NODE_HOVERABILITY_EXTENSION_NAME,
            .label      = "Hoverable (KHR_node_hoverability)",
            .toggleTip  = "Whether this node and its children can be hovered (consumed by KHR_interactivity).",
            .addTip     = "",
            .idTag      = "hov",
            .onDesc     = "Make hoverable",
            .offDesc    = "Make unhoverable",
            .addDesc    = "Add node hoverability",
            .removeDesc = "Remove node hoverability",
        },
        &tinygltf::utils::getNodeHoverability, &tinygltf::utils::setNodeHoverability, &KHR_node_hoverability::hoverable);

    PE::end();
  }
}

//==================================================================================================
// PRIMITIVE PROPERTIES
//==================================================================================================

void UiInspector::renderPrimitiveProperties(int nodeIdx, int primIdx, int meshIdx)
{
  const tinygltf::Model& model    = m_scene->getModel();
  const bool             readOnly = m_scene->isNodeReadOnly(nodeIdx);

  // --- NODE section: transform + extensions + relationships ---
  {
    const char* nodeName = (nodeIdx >= 0 && nodeIdx < int(model.nodes.size())) ? model.nodes[nodeIdx].name.c_str() : "";
    char        nodeHdr[160];
    std::snprintf(nodeHdr, sizeof(nodeHdr), "NODE  [%d] %s##node_%d", nodeIdx, nodeName, nodeIdx);
    if(ImGui::CollapsingHeader(nodeHdr))
    {
      if(readOnly)
        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "%s Referenced asset (read-only)", ICON_MS_LOCK);

      ImGui::BeginDisabled(readOnly);
      if(ImGui::TreeNodeEx("Transform##prim_transform", ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_SpanAvailWidth))
      {
        renderTransformSection(nodeIdx);
        ImGui::TreePop();
      }
      ImGui::EndDisabled();

      if(ImGui::TreeNodeEx("Node extensions##prim_ext", ImGuiTreeNodeFlags_SpanAvailWidth))
      {
        ImGui::BeginDisabled(readOnly);
        renderNodeExtensions(nodeIdx);
        ImGui::EndDisabled();
        ImGui::TreePop();
      }

      if(ImGui::TreeNodeEx("Relationships##prim_rel", ImGuiTreeNodeFlags_SpanAvailWidth))
      {
        renderNodeRelationships(nodeIdx);
        ImGui::TreePop();
      }
    }
  }

  // --- MESH section: all primitives, selected one open by default ---
  if(meshIdx >= 0 && meshIdx < int(model.meshes.size()))
  {
    const tinygltf::Mesh& mesh = model.meshes[meshIdx];
    char                  meshHdr[160];
    std::snprintf(meshHdr, sizeof(meshHdr), "MESH  [%d] %s##mesh_%d", meshIdx, mesh.name.c_str(), meshIdx);
    if(ImGui::CollapsingHeader(meshHdr))
    {
      for(int i = 0; i < int(mesh.primitives.size()); ++i)
      {
        ImGui::PushID(i);
        const bool isSelected = (i == primIdx);
        char       primHdr[80];
        if(isSelected)
          std::snprintf(primHdr, sizeof(primHdr), "%s Primitive %d  [selected]", ICON_MS_SHAPE_LINE, i);
        else
          std::snprintf(primHdr, sizeof(primHdr), "%s Primitive %d", ICON_MS_SHAPE_LINE, i);
        const ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth | (isSelected ? ImGuiTreeNodeFlags_DefaultOpen : 0);
        if(ImGui::TreeNodeEx(primHdr, flags))
        {
          renderPrimitiveDetail(meshIdx, i);
          elementLinkRow("Material", SceneSelection::SelectionType::eMaterial, mesh.primitives[i].material, 100.0f, "none");
          ImGui::TreePop();
        }
        ImGui::PopID();
      }
    }
  }

  // --- MATERIAL section: assignment toolbar + full property editor (unchanged content) ---
  if(meshIdx >= 0 && meshIdx < int(model.meshes.size()))
  {
    const tinygltf::Mesh& mesh = model.meshes[meshIdx];
    if(primIdx >= 0 && primIdx < int(mesh.primitives.size()))
    {
      const int matIdx = mesh.primitives[primIdx].material;
      {
        char matHdr[160];
        if(matIdx >= 0 && matIdx < int(model.materials.size()))
          std::snprintf(matHdr, sizeof(matHdr), "MATERIAL  [%d] %s##material_section", matIdx,
                        model.materials[matIdx].name.c_str());
        else
          std::snprintf(matHdr, sizeof(matHdr), "MATERIAL  (none)##material_section");
        if(ImGui::CollapsingHeader(matHdr, ImGuiTreeNodeFlags_DefaultOpen))
        {
          ImGui::BeginDisabled(readOnly);
          renderMaterialAssignmentToolbar(meshIdx, primIdx, nodeIdx, matIdx);
          ImGui::EndDisabled();
          if(matIdx >= 0 && matIdx < int(model.materials.size()))
          {
            ImGui::Separator();
            renderMaterialSection(matIdx, !readOnly);
          }
        }
      }
    }
  }
}

//==================================================================================================
// MATERIAL PROPERTIES
//==================================================================================================

void UiInspector::renderMaterialProperties(int matIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  if(matIdx < 0 || matIdx >= static_cast<int>(model.materials.size()))
    return;

  const tinygltf::Material& material = model.materials[matIdx];

  ImGui::Text("%s Material[%d]: %s", ICON_MS_BRUSH, matIdx, material.name.c_str());

  // XMP button
  std::string popupId = "inspector_mat_xmp_" + std::to_string(matIdx);
  ImGui::SameLine();
  ui_xmp::renderInfoButton(&m_scene->getModel(), material.extensions, popupId.c_str());

  // How many primitives reference this material.
  int usedByPrims = 0;
  for(const tinygltf::Mesh& mesh : model.meshes)
    for(const tinygltf::Primitive& prim : mesh.primitives)
      if(prim.material == matIdx)
        usedByPrims++;
  ImGui::TextDisabled("Used by %d primitive(s)", usedByPrims);

  ImGui::Separator();

  renderMaterialSection(matIdx, true);

  ImGui::Separator();

  renderMaterialOperations(matIdx, -1);
}

//==================================================================================================
// MESH PROPERTIES
//==================================================================================================

// glTF primitive.mode -> name.
static const char* primitiveModeName(int mode)
{
  switch(mode)
  {
    case TINYGLTF_MODE_POINTS:
      return "POINTS";
    case TINYGLTF_MODE_LINE:
      return "LINES";
    case TINYGLTF_MODE_LINE_LOOP:
      return "LINE_LOOP";
    case TINYGLTF_MODE_LINE_STRIP:
      return "LINE_STRIP";
    case TINYGLTF_MODE_TRIANGLES:
      return "TRIANGLES";
    case TINYGLTF_MODE_TRIANGLE_STRIP:
      return "TRIANGLE_STRIP";
    case TINYGLTF_MODE_TRIANGLE_FAN:
      return "TRIANGLE_FAN";
    default:
      return "?";
  }
}

// Accessor element count (0 if the accessor index is invalid) and the derived vertex/triangle counts of
// a primitive - shared by the mesh totals and the per-primitive detail section.
static long long accessorElementCount(const tinygltf::Model& model, int accessor)
{
  return (accessor >= 0 && accessor < int(model.accessors.size())) ? static_cast<long long>(model.accessors[accessor].count) : 0;
}
static long long primitiveVertexCount(const tinygltf::Model& model, const tinygltf::Primitive& p)
{
  auto it = p.attributes.find("POSITION");
  return it != p.attributes.end() ? accessorElementCount(model, it->second) : 0;
}
static long long primitiveTriangleCount(const tinygltf::Model& model, const tinygltf::Primitive& p)
{
  const long long count = p.indices >= 0 ? accessorElementCount(model, p.indices) : primitiveVertexCount(model, p);
  return uigltf::primitiveTriangleCountForMode(p.mode, count);
}

void UiInspector::renderPrimitiveDetail(int meshIdx, int primIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  if(meshIdx < 0 || meshIdx >= int(model.meshes.size()))
    return;
  const tinygltf::Mesh& mesh = model.meshes[meshIdx];
  if(primIdx < 0 || primIdx >= int(mesh.primitives.size()))
    return;
  const tinygltf::Primitive& prim = mesh.primitives[primIdx];

  ImGui::Text("%s    %lld verts    %lld tris", primitiveModeName(prim.mode), primitiveVertexCount(model, prim),
              primitiveTriangleCount(model, prim));

  if(ImGui::BeginTable("##acc", 3, ImGuiTableFlags_None))
  {
    ImGui::TableSetupColumn("##name", ImGuiTableColumnFlags_WidthFixed, 110.0f);
    ImGui::TableSetupColumn("##acc", ImGuiTableColumnFlags_WidthFixed, 58.0f);
    ImGui::TableSetupColumn("##cnt", ImGuiTableColumnFlags_WidthStretch);
    for(const auto& [name, acc] : prim.attributes)
    {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::TextDisabled("%s", name.c_str());
      ImGui::TableSetColumnIndex(1);
      ImGui::TextDisabled("acc[%d]", acc);
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("%lld", accessorElementCount(model, acc));
    }
    if(prim.indices >= 0)
    {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::TextDisabled("INDICES");
      ImGui::TableSetColumnIndex(1);
      ImGui::TextDisabled("acc[%d]", prim.indices);
      ImGui::TableSetColumnIndex(2);
      ImGui::Text("%lld", accessorElementCount(model, prim.indices));
    }
    ImGui::EndTable();
  }
  if(!prim.targets.empty())
    ImGui::TextDisabled("Morph targets: %zu", prim.targets.size());
}

void UiInspector::renderMeshProperties(int meshIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  if(meshIdx < 0 || meshIdx >= static_cast<int>(model.meshes.size()))
    return;

  const tinygltf::Mesh& mesh = model.meshes[meshIdx];

  ImGui::Text("%s Mesh[%d]: %s", ICON_MS_VIEW_IN_AR, meshIdx, mesh.name.c_str());
  std::string popupId = "inspector_mesh_xmp_" + std::to_string(meshIdx);
  ImGui::SameLine();
  ui_xmp::renderInfoButton(&m_scene->getModel(), mesh.extensions, popupId.c_str());
  ImGui::Separator();

  // Totals across primitives.
  long long totalVerts = 0, totalTris = 0;
  for(const tinygltf::Primitive& p : mesh.primitives)
  {
    totalVerts += primitiveVertexCount(model, p);
    totalTris += primitiveTriangleCount(model, p);
  }
  ImGui::Text("Primitives: %zu    Vertices: %lld    Triangles: %lld", mesh.primitives.size(), totalVerts, totalTris);

  ImGui::Separator();

  // Per-primitive: compact collapsible rows, first open by default.
  for(int i = 0; i < int(mesh.primitives.size()); ++i)
  {
    ImGui::PushID(i);
    char hdr[64];
    std::snprintf(hdr, sizeof(hdr), "%s Primitive %d", ICON_MS_SHAPE_LINE, i);
    const ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth | (i == 0 ? ImGuiTreeNodeFlags_DefaultOpen : 0);
    if(ImGui::TreeNodeEx(hdr, flags))
    {
      renderPrimitiveDetail(meshIdx, i);
      elementLinkRow("Material", SceneSelection::SelectionType::eMaterial, mesh.primitives[i].material, 100.0f, "none (default material)");
      ImGui::TreePop();
    }
    ImGui::PopID();
  }

  ImGui::Separator();

  // Used by nodes: all instancing nodes as jump links.
  int instances = 0;
  for(int n = 0; n < int(model.nodes.size()); ++n)
    if(model.nodes[n].mesh == meshIdx)
      instances++;
  char usedHdr[64];
  std::snprintf(usedHdr, sizeof(usedHdr), "Used by nodes (%d)##used_by_%d", instances, meshIdx);
  if(ImGui::CollapsingHeader(usedHdr))
  {
    for(int n = 0; n < int(model.nodes.size()); ++n)
      if(model.nodes[n].mesh == meshIdx)
        elementLink(elementRefLabel(SceneSelection::SelectionType::eNode, n).c_str(), SceneSelection::SelectionType::eNode, n);
  }
}

//==================================================================================================
// CAMERA PROPERTIES
//==================================================================================================

void UiInspector::renderCameraProperties(int camIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  if(camIdx < 0 || camIdx >= static_cast<int>(model.cameras.size()))
    return;

  tinygltf::Camera& camera = m_scene->getModel().cameras[camIdx];  // No getCameraForEdit yet; direct access

  ImGui::Text("%s Camera[%d]: %s", ICON_MS_CAMERA_ALT, camIdx, camera.name.c_str());
  ImGui::TextDisabled("Type: %s", camera.type.empty() ? "perspective" : camera.type.c_str());
  int camNode = -1;
  for(int n = 0; n < int(model.nodes.size()); ++n)
    if(model.nodes[n].camera == camIdx)
    {
      camNode = n;
      break;
    }
  elementLinkRow("Attached to", SceneSelection::SelectionType::eNode, camNode, 100.0f, "(unattached)");
  ImGui::Separator();

  if(PE::begin())
  {
    bool modif = false;

    if(camera.type == "perspective")
    {
      tinygltf::PerspectiveCamera& persp = camera.perspective;
      modif |= PE::DragScalar("Y FOV (rad)", ImGuiDataType_Double, &persp.yfov, 0.01f, &f64_001, &f64_179);
      modif |= PE::DragScalar("Aspect Ratio", ImGuiDataType_Double, &persp.aspectRatio, 0.01f, &f64_01, &f64_ten);
      modif |= PE::DragScalar("Z Near", ImGuiDataType_Double, &persp.znear, 0.01f, &f64_001, &f64_1000);
      modif |= PE::DragScalar("Z Far", ImGuiDataType_Double, &persp.zfar, 0.01f, &f64_001, &f64_10000);
    }
    else if(camera.type == "orthographic")
    {
      tinygltf::OrthographicCamera& ortho = camera.orthographic;
      modif |= PE::DragScalar("X Mag", ImGuiDataType_Double, &ortho.xmag, 0.01f, &f64_neg1000, &f64_1000);
      modif |= PE::DragScalar("Y Mag", ImGuiDataType_Double, &ortho.ymag, 0.01f, &f64_neg1000, &f64_1000);
      modif |= PE::DragScalar("Z Near", ImGuiDataType_Double, &ortho.znear, 0.01f, &f64_neg1000, &f64_1000);
      double ortho_zfar_min = ortho.znear + 1.0;
      modif |= PE::DragScalar("Z Far", ImGuiDataType_Double, &ortho.zfar, 0.01f, &ortho_zfar_min, &f64_10000);
    }

    PE::end();
  }

  ImGui::Separator();
  ImGui::Text("Camera Sync:");

  if(ImGui::Button("Apply to Current View"))
  {
    if(m_selection)
      m_selection->emitCameraApply(camIdx);
  }

  ImGui::SameLine();
  if(ImGui::Button("Set from Current View"))
  {
    if(m_selection)
      m_selection->emitCameraSetFromView(camIdx);
  }
}

//==================================================================================================
// LIGHT PROPERTIES
//==================================================================================================

void UiInspector::renderLightProperties(int lightIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  if(lightIdx < 0 || lightIdx >= static_cast<int>(model.lights.size()))
    return;

  tinygltf::Light& light = m_scene->editor().getLightForEdit(lightIdx);

  // Snapshot before any widget can modify the light (for undo)
  tinygltf::Light preEditLight = light;

  ImGui::Text("%s Light[%d]: %s", ICON_MS_LIGHTBULB, lightIdx, light.name.c_str());
  int lightNode = -1;
  for(int n = 0; n < int(model.nodes.size()); ++n)
    if(model.nodes[n].light == lightIdx)
    {
      lightNode = n;
      break;
    }
  elementLinkRow("Attached to", SceneSelection::SelectionType::eNode, lightNode, 100.0f, "(unattached)");

  // EXT_lights_ies is standalone per spec: even if the attached node also carries a profile, it
  // does not compose with this KHR light -- see Scene::handleLightTraversal and the Node inspector.
  ImGui::Separator();

  bool modif = false;

  if(PE::begin())
  {
    // Type combo
    static const char* lightTypes[] = {"point", "directional", "spot"};
    int                currentType  = 0;
    if(light.type == "directional")
      currentType = 1;
    else if(light.type == "spot")
      currentType = 2;

    if(PE::Combo("Type", &currentType, lightTypes, 3, -1,
                 "Point: emits in all directions from position, attenuates with distance squared\n"
                 "Directional: infinitely far, emits along local -Z axis, no attenuation\n"
                 "Spot: cone along local -Z axis, attenuates with distance squared"))
    {
      light.type = lightTypes[currentType];
      if(light.type == "spot" && light.spot.outerConeAngle == 0.0)
        light.spot.outerConeAngle = 0.785398;  // pi/4 default per spec
      modif = true;
    }

    // KHR_lights_punctual color is a linear multiplier on intensity. Shown/edited in linear; the
    // swatch and wheel are perceptual (sRGB) so picking is intuitive.
    glm::vec3 color = glm::vec3(light.color[0], light.color[1], light.color[2]);
    if(colorEdit3Linear("Color", glm::value_ptr(color),
                        "RGB light color (shown/edited in linear; swatch/wheel perceptual).\n"
                        "Acts as a wavelength-specific multiplier on intensity."))
    {
      light.color[0] = color.x;
      light.color[1] = color.y;
      light.color[2] = color.z;
      modif          = true;
    }

    float intensity = static_cast<float>(light.intensity);
    if(light.type == "directional")
    {
      if(PE::DragFloat("Intensity", &intensity, 0.1f, 0.0f, 100000.0f, "%.3f", 0,
                       "Illuminance in lux (lm/m\xc2\xb2).\n"
                       "Brightness of a pure white directional light."))
      {
        light.intensity = intensity;
        modif           = true;
      }
    }
    else
    {
      if(PE::DragFloat("Intensity", &intensity, 0.1f, 0.0f, 100000.0f, "%.3f", 0,
                       "Luminous intensity in candela (lm/sr).\n"
                       "Brightness at the light source; attenuates with distance squared."))
      {
        light.intensity = intensity;
        modif           = true;
      }
    }

    // Range (point and spot only; 0 = undefined/infinite per spec)
    if(light.type == "point" || light.type == "spot")
    {
      float range = static_cast<float>(light.range);
      if(PE::DragFloat("Range", &range, 0.1f, 0.0f, 10000.0f, "%.3f", 0,
                       "Distance cutoff where light intensity reaches zero.\n"
                       "0 = undefined (infinite range, inverse-square only).\n"
                       "When set, attenuation uses: (1-(d/range)^4)^2 / d^2.\n"
                       "Not affected by node scale."))
      {
        light.range = range;
        modif       = true;
      }
    }

    // Spot angles (KHR_lights_punctual: innerConeAngle >= 0 and < outerConeAngle <= pi/2)
    if(light.type == "spot")
    {
      float innerAngle = static_cast<float>(light.spot.innerConeAngle);
      float outerAngle = static_cast<float>(light.spot.outerConeAngle);

      if(PE::DragFloat("Inner Cone Angle", &innerAngle, 0.01f, 0.0f, outerAngle, "%.3f", 0,
                       "Angle in radians from the spotlight axis where falloff begins.\n"
                       "Full intensity inside this cone. Must be >= 0 and < outer.\n"
                       "Default: 0"))
      {
        light.spot.innerConeAngle = std::min(innerAngle, outerAngle);
        modif                     = true;
      }

      if(PE::DragFloat("Outer Cone Angle", &outerAngle, 0.01f, innerAngle, 1.5708f, "%.3f", 0,
                       "Angle in radians from the spotlight axis where falloff ends.\n"
                       "No light beyond this cone. Must be > inner and <= pi/2.\n"
                       "Default: pi/4 (45 deg)"))
      {
        light.spot.outerConeAngle = std::max(outerAngle, innerAngle);
        modif                     = true;
      }
    }

    if(modif && m_scene)
      m_scene->markLightDirty(lightIdx);

    PE::end();
  }

  // Undo tracking: snapshot on first modified frame, push command when editing stops
  if(modif && !m_lightModifiedLastFrame)
  {
    m_lightSnapshotIdx  = lightIdx;
    m_lightSnapshotData = std::make_unique<tinygltf::Light>(preEditLight);
  }

  if(!modif && m_lightModifiedLastFrame && m_scene && m_undoStack && m_lightSnapshotData && m_lightSnapshotIdx == lightIdx)
  {
    auto cmd = std::make_unique<EditLightCommand>(*m_scene, lightIdx, *m_lightSnapshotData, light);
    m_undoStack->pushExecuted(std::move(cmd));
    m_lightSnapshotData.reset();
  }

  m_lightModifiedLastFrame = modif;
}

//==================================================================================================
// TEXTURE / IMAGE / SAMPLER / ANIMATION PROPERTIES (resource pools)
//==================================================================================================

void UiInspector::renderTextureProperties(int textureIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  if(textureIdx < 0 || textureIdx >= static_cast<int>(model.textures.size()))
  {
    renderNoSelection();
    return;
  }
  const tinygltf::Texture& tex = model.textures[textureIdx];

  ImGui::Text("%s Texture %d", ICON_MS_IMAGE, textureIdx);
  ImGui::TextDisabled("%s", tinygltf::utils::getTextureUiLabel(model, textureIdx).c_str());
  ImGui::Separator();

  const int         imgIdx = tinygltf::utils::getTextureImageIndex(tex);
  const ImTextureID thumb  = m_host.thumbnail(textureIdx);
  if(thumb != ImTextureID(0))
  {
    if(ImGui::ImageButton("texthumb", thumb, ImVec2(96.0f, 96.0f)) && m_onViewImage && imgIdx >= 0)
      m_onViewImage(imgIdx);
    if(ImGui::IsItemHovered())
      ImGui::SetTooltip("Open the full image");
  }

  // One undo step per image/sampler-reference change (matches the old texture "tune" popup).
  auto commitTexture = [&](const tinygltf::Texture& edited) {
    if(m_undoStack)
      m_undoStack->executeCommand(std::make_unique<EditTextureCommand>(*m_scene, textureIdx, tex, edited,
                                                                       "Edit texture " + std::to_string(textureIdx)));
  };

  // getTextureImageIndex() (imgIdx, above) prefers a vendor extension's own `source` over this base
  // field when one is present (EXT_texture_webp / MSFT_texture_dds / KHR_texture_basisu) -- writing
  // tex.source in that case would have no visible effect, so gate the field off instead.
  const bool imageSourceOverridden = tinygltf::utils::hasTextureImageSourceOverride(tex);
  int        imageIdx              = tex.source;
  ImGui::BeginDisabled(imageSourceOverridden);
  ImGui::SetNextItemWidth(120.0f);
  if(ImGui::InputInt("Image", &imageIdx) && !model.images.empty())
  {
    imageIdx = std::clamp(imageIdx, 0, static_cast<int>(model.images.size()) - 1);
    if(imageIdx != tex.source)
    {
      tinygltf::Texture edited = tex;
      edited.source            = imageIdx;
      commitTexture(edited);
    }
  }
  ImGui::EndDisabled();
  if(imageSourceOverridden)
  {
    ImGui::SameLine();
    ImGui::TextDisabled("(from vendor extension)");
    if(ImGui::IsItemHovered())
      ImGui::SetTooltip("The active image comes from a WebP/DDS/BasisU extension override, not this base source field.");
  }
  if(imgIdx >= 0)
  {
    ImGui::SameLine();
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s go to image", ICON_MS_IMAGE);
    elementLink(buf, SceneSelection::SelectionType::eImage, imgIdx);
  }

  int samplerIdx = tex.sampler;
  ImGui::SetNextItemWidth(120.0f);
  if(ImGui::InputInt("Sampler (-1=default)", &samplerIdx))
  {
    samplerIdx = std::clamp(samplerIdx, -1, static_cast<int>(model.samplers.size()) - 1);
    if(samplerIdx != tex.sampler)
    {
      tinygltf::Texture edited = tex;
      edited.sampler           = samplerIdx;
      commitTexture(edited);
    }
  }
  if(tex.sampler >= 0 && tex.sampler < static_cast<int>(model.samplers.size()))
  {
    ImGui::SameLine();
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s go to sampler", ICON_MS_TUNE);
    elementLink(buf, SceneSelection::SelectionType::eSampler, tex.sampler);
  }

  // Inline wrap/filter editing of the referenced sampler (one undo step per field).
  if(tex.sampler >= 0 && tex.sampler < static_cast<int>(model.samplers.size()))
  {
    ImGui::SeparatorText("Sampler");
    const int               si  = tex.sampler;
    const tinygltf::Sampler cur = model.samplers[si];
    uigltf::renderSamplerFields(cur, [&](const tinygltf::Sampler& edited) {
      if(m_undoStack)
        m_undoStack->executeCommand(
            std::make_unique<EditSamplerCommand>(*m_scene, si, cur, edited, "Edit sampler " + std::to_string(si)));
    });
  }
}

void UiInspector::renderImageProperties(int imageIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  if(imageIdx < 0 || imageIdx >= static_cast<int>(model.images.size()))
  {
    renderNoSelection();
    return;
  }
  const tinygltf::Image& image = model.images[imageIdx];

  ImGui::Text("%s Image %d", ICON_MS_PHOTO, imageIdx);
  ImGui::TextDisabled("%s", uigltf::imageDisplayName(model, imageIdx).c_str());
  ImGui::Text("Source: %s", image.uri.empty() ? "embedded" : "external");
  if(image.width > 0 && image.height > 0)
    ImGui::Text("Resolution: %d x %d", image.width, image.height);
  ImGui::Text("Referenced by %d texture(s)", m_scene->editor().countTextureRefsToImage(imageIdx));
  ImGui::Separator();

  const ImTextureID thumb = m_getImageThumbnail ? m_getImageThumbnail(imageIdx) : ImTextureID(0);
  if(thumb != ImTextureID(0) && image.width > 0 && image.height > 0)
  {
    const float w = std::min(192.0f, static_cast<float>(image.width));
    const float h = w * static_cast<float>(image.height) / static_cast<float>(image.width);
    if(ImGui::ImageButton("imgview", thumb, ImVec2(w, h)) && m_onViewImage)
      m_onViewImage(imageIdx);
  }

  ImGui::BeginDisabled(!m_host.canPickImage());
  if(ImGui::Button(ICON_MS_FILE_OPEN " Replace..."))
  {
    const std::filesystem::path path = m_host.pickImage();
    if(!path.empty())
    {
      const tinygltf::Image oldImage = image;  // copy before replace
      std::string           err;
      if(m_scene->editor().replaceImageFromFile(imageIdx, path, &err))
      {
        if(m_undoStack)
          m_undoStack->pushExecuted(std::make_unique<ReplaceImageCommand>(*m_scene, imageIdx, oldImage,
                                                                          m_scene->getModel().images[imageIdx],
                                                                          "Replace image " + std::to_string(imageIdx)));
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
  if(ImGui::Button(ICON_MS_VISIBILITY " View") && m_onViewImage)
    m_onViewImage(imageIdx);
  ImGui::SameLine();
  if(ImGui::Button(ICON_MS_REFRESH " Reload"))
    m_scene->getDirtyFlags().texturesChanged = true;  // force re-decode from the URI on the next rebuild

  // Textures that reference this image (jump links).
  ImGui::Separator();
  renderTexturesUsing("Used by textures:",
                      [&](const tinygltf::Texture& tx) { return tinygltf::utils::getTextureImageIndex(tx) == imageIdx; });
}

void UiInspector::renderSamplerProperties(int samplerIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  if(samplerIdx < 0 || samplerIdx >= static_cast<int>(model.samplers.size()))
  {
    renderNoSelection();
    return;
  }
  ImGui::Text("%s Sampler %d", ICON_MS_TUNE, samplerIdx);
  ImGui::Separator();

  const tinygltf::Sampler cur = model.samplers[samplerIdx];
  uigltf::renderSamplerFields(cur, [&](const tinygltf::Sampler& edited) {
    if(m_undoStack)
      m_undoStack->executeCommand(std::make_unique<EditSamplerCommand>(*m_scene, samplerIdx, cur, edited,
                                                                       "Edit sampler " + std::to_string(samplerIdx)));
  });

  // Textures that reference this sampler (jump links).
  ImGui::Separator();
  renderTexturesUsing("Used by textures:", [&](const tinygltf::Texture& tx) { return tx.sampler == samplerIdx; });
}

void UiInspector::renderAnimationProperties(int animIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  if(animIdx < 0 || animIdx >= static_cast<int>(model.animations.size()))
  {
    renderNoSelection();
    return;
  }
  const tinygltf::Animation& anim = model.animations[animIdx];

  ImGui::Text("%s Animation[%d]: %s", ICON_MS_MOVIE, animIdx, anim.name.empty() ? "(unnamed)" : anim.name.c_str());
  ImGui::Separator();
  ImGui::Text("Channels: %d    Samplers: %d", static_cast<int>(anim.channels.size()), static_cast<int>(anim.samplers.size()));

  // Per-channel target: node (jump link) + animated path (translation/rotation/scale/weights/pointer).
  ImGui::Text("Targets:");
  ImGui::Indent();
  for(const tinygltf::AnimationChannel& ch : anim.channels)
  {
    if(ch.target_node >= 0 && ch.target_node < static_cast<int>(model.nodes.size()))
    {
      elementLink(elementRefLabel(SceneSelection::SelectionType::eNode, ch.target_node).c_str(),
                  SceneSelection::SelectionType::eNode, ch.target_node);
      ImGui::SameLine();
      ImGui::TextDisabled("\xc2\xb7 %s", ch.target_path.c_str());
    }
    else
    {
      ImGui::TextDisabled("(pointer) \xc2\xb7 %s", ch.target_path.c_str());
    }
  }
  ImGui::Unindent();
  ImGui::TextDisabled("Playback is controlled from the Animations panel.");
}

//==================================================================================================
// TRANSFORM SECTION
//==================================================================================================

void UiInspector::renderTransformSection(int nodeIdx)
{
  tinygltf::Node& node = m_scene->editor().getNodeForEdit(nodeIdx);

  glm::vec3 translation, scale;
  glm::quat rotation;
  tinygltf::utils::getNodeTRS(node, translation, rotation, scale);

  // Re-decompose euler from quaternion only on node selection change or external
  // modification (gizmo, animation). This avoids the gimbal lock that occurs when
  // round-tripping through glm::eulerAngles() every frame (Y clamped to [-90,90]).
  bool externalQuatChange = glm::abs(glm::dot(rotation, m_cachedEuler.quat)) < (1.0f - 1e-4f);
  if(nodeIdx != m_cachedEuler.nodeIdx || externalQuatChange)
  {
    m_cachedEuler.nodeIdx = nodeIdx;
    m_cachedEuler.euler   = glm::degrees(glm::eulerAngles(rotation));
    m_cachedEuler.quat    = rotation;
  }

  // Capture pre-edit TRS for undo snapshot (before DragFloat3 modifies values)
  glm::vec3 preEditT = translation;
  glm::quat preEditR = rotation;
  glm::vec3 preEditS = scale;

  if(PE::begin())
  {
    bool modif = false;
    modif |= PE::DragFloat3("Translation", glm::value_ptr(translation), 0.01f * m_bbox.radius());
    modif |= PE::DragFloat3("Rotation", glm::value_ptr(m_cachedEuler.euler), 0.1f);
    modif |= PE::DragFloat3("Scale", glm::value_ptr(scale), 0.01f);

    // Undo tracking: detect the start and end of a DragFloat3 edit cycle.
    // On the first frame where modif becomes true, snapshot the pre-edit TRS.
    // On the first frame where modif becomes false after editing, push the command.
    if(modif && !m_transformModifiedLastFrame)
    {
      m_transformSnapshot = {nodeIdx, preEditT, preEditR, preEditS};
    }

    if(modif)
    {
      rotation           = glm::quat(glm::radians(m_cachedEuler.euler));
      m_cachedEuler.quat = rotation;
      if(m_scene)
        m_scene->editor().setNodeTRS(nodeIdx, translation, rotation, scale);
    }

    if(!modif && m_transformModifiedLastFrame && m_scene && m_transformSnapshot.nodeIdx == nodeIdx)
    {
      auto cmd = std::make_unique<SetTransformCommand>(*m_scene, nodeIdx, m_transformSnapshot.translation,
                                                       m_transformSnapshot.rotation, m_transformSnapshot.scale,
                                                       translation, rotation, scale);
      m_undoStack->pushExecuted(std::move(cmd));
    }

    m_transformModifiedLastFrame = modif;

    PE::end();
  }
}

//==================================================================================================
// MATERIAL SECTION
//==================================================================================================

void UiInspector::renderMaterialSection(int matIdx, bool allowEdit)
{
  tinygltf::Material& material = m_scene->editor().getMaterialForEdit(matIdx);

  ImGui::Text("Material[%d]: %s", matIdx, material.name.c_str());

  if(!allowEdit)
  {
    ImGui::TextDisabled("(Read-only view)");
    return;
  }

  // Snapshot before any widget can modify the material (for undo)
  tinygltf::Material preEditMaterial = material;

  bool modif = false;

  if(PE::begin())
  {
    // Base PBR properties — specular-glossiness replaces metallic-roughness when present
    bool useSpecGloss = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_PBR_SPECULAR_GLOSSINESS_EXTENSION_NAME);
    if(useSpecGloss)
    {
      KHR_materials_pbrSpecularGlossiness sg = tinygltf::utils::getPbrSpecularGlossiness(material);

      glm::vec4 diffuse = sg.diffuseFactor;
      if(colorEdit4Linear("Diffuse Color", glm::value_ptr(diffuse),
                          "RGBA diffuse color factor (RGB shown/edited in linear; swatch/wheel perceptual; A is linear).\n"
                          "(KHR_materials_pbrSpecularGlossiness)"))
      {
        sg.diffuseFactor = diffuse;
        modif            = true;
      }
      modif |= renderTextureEditRow("Diffuse", sg.diffuseTexture);

      glm::vec3 specular = sg.specularFactor;
      if(colorEdit3Linear("Specular", glm::value_ptr(specular),
                          "RGB specular color factor (shown/edited in linear; swatch/wheel perceptual).\n"
                          "(KHR_materials_pbrSpecularGlossiness)"))
      {
        sg.specularFactor = specular;
        modif             = true;
      }

      if(PE::DragFloat("Glossiness", &sg.glossinessFactor, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                       "Glossiness factor. 1.0 = fully smooth, 0.0 = fully rough.\n"
                       "(KHR_materials_pbrSpecularGlossiness)"))
      {
        modif = true;
      }
      modif |= renderTextureEditRow("Specular-Glossiness", sg.specularGlossinessTexture);

      if(modif)
        tinygltf::utils::setPbrSpecularGlossiness(material, sg);
    }
    else
    {
      tinygltf::PbrMetallicRoughness& pbr = material.pbrMetallicRoughness;

      glm::vec4 baseColor = glm::make_vec4(pbr.baseColorFactor.data());
      if(colorEdit4Linear("Base Color", glm::value_ptr(baseColor),
                          "RGBA base color factor (RGB shown/edited in linear; swatch/wheel perceptual; A is linear).\n"
                          "RGB modulates diffuse/specular; A controls alpha coverage."))
      {
        pbr.baseColorFactor[0] = baseColor.x;
        pbr.baseColorFactor[1] = baseColor.y;
        pbr.baseColorFactor[2] = baseColor.z;
        pbr.baseColorFactor[3] = baseColor.w;
        modif                  = true;
      }
      modif |= renderTextureEditRow("Base Color", pbr.baseColorTexture);

      float metallic = static_cast<float>(pbr.metallicFactor);
      if(PE::DragFloat("Metallic", &metallic, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                       "Metallic-ness of the material. 0.0 = dielectric, 1.0 = metal.\n"
                       "Blends between dielectric and metallic BRDF."))
      {
        pbr.metallicFactor = metallic;
        modif              = true;
      }

      float roughness = static_cast<float>(pbr.roughnessFactor);
      if(PE::DragFloat("Roughness", &roughness, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                       "Perceptual roughness. 0.0 = smooth (sharp reflections), 1.0 = rough.\n"
                       "Squared internally (alphaRoughness = roughness^2)."))
      {
        pbr.roughnessFactor = roughness;
        modif               = true;
      }
      modif |= renderTextureEditRow("Metallic-Roughness", pbr.metallicRoughnessTexture);
    }

    // Emissive
    // emissiveFactor is a linear [0,1] multiplier on the emissive texel radiance (glTF 5.19.8).
    // Shown/edited in linear; the swatch and wheel are perceptual (sRGB) so picking is intuitive.
    glm::vec3 emissive = glm::make_vec3(material.emissiveFactor.data());
    if(colorEdit3Linear("Emissive", glm::value_ptr(emissive),
                        "RGB emissive factor [0,1] (shown/edited in linear; swatch/wheel perceptual).\n"
                        "Linear multiplier on the emissive texture; scaled by emissiveStrength for HDR emission."))
    {
      material.emissiveFactor[0] = emissive.x;
      material.emissiveFactor[1] = emissive.y;
      material.emissiveFactor[2] = emissive.z;
      modif                      = true;
    }
    modif |= renderTextureEditRow("Emissive", material.emissiveTexture);
    modif |= renderTextureEditRow("Normal", material.normalTexture);
    modif |= renderTextureEditRow("Occlusion", material.occlusionTexture);

    // Alpha mode
    const char* alphaModes[] = {"OPAQUE", "MASK", "BLEND"};
    int         currentMode  = 0;
    if(material.alphaMode == "OPAQUE")
      currentMode = 0;
    else if(material.alphaMode == "MASK")
      currentMode = 1;
    else if(material.alphaMode == "BLEND")
      currentMode = 2;

    if(PE::Combo("Alpha Mode", &currentMode, alphaModes, 3, -1,
                 "OPAQUE: alpha ignored, fully opaque.\n"
                 "MASK: alpha below cutoff is discarded.\n"
                 "BLEND: alpha blending with background."))
    {
      material.alphaMode = alphaModes[currentMode];
      if(m_scene)
        m_scene->markRenderNodeRtxDirtyForMaterials({matIdx});
      modif = true;
    }

    if(material.alphaMode == "MASK")
    {
      float alphaCutoff = static_cast<float>(material.alphaCutoff);
      if(PE::DragFloat("Alpha Cutoff", &alphaCutoff, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                       "Threshold for MASK mode.\nFragments with alpha below this value are discarded."))
      {
        material.alphaCutoff = alphaCutoff;
        modif                = true;
      }
    }

    // Double sided
    bool doubleSided = material.doubleSided;
    if(PE::Checkbox("Double Sided", &doubleSided,
                    "When enabled, back faces are rendered.\n"
                    "Affects both rasterizer culling and ray tracing instance flags."))
    {
      material.doubleSided = doubleSided;
      if(m_scene)
        m_scene->markRenderNodeRtxDirtyForMaterials({matIdx});
      modif = true;
    }

    if(modif && m_scene)
      m_scene->markMaterialDirty(matIdx);

    PE::end();
  }

  // Material extensions (returns true if any extension property changed)
  bool extensionChange = renderMaterialExtensions(material, matIdx);

  // Undo tracking: combine PBR and extension changes into one flag per frame.
  // Snapshot on first modified frame, push command when editing stops.
  bool materialModifiedThisFrame = modif || extensionChange;

  if(materialModifiedThisFrame && !m_materialModifiedLastFrame)
  {
    m_materialSnapshotIdx  = matIdx;
    m_materialSnapshotData = std::make_unique<tinygltf::Material>(preEditMaterial);
  }

  if(!materialModifiedThisFrame && m_materialModifiedLastFrame && m_scene && m_materialSnapshotData && m_materialSnapshotIdx == matIdx)
  {
    auto cmd = std::make_unique<EditMaterialCommand>(*m_scene, matIdx, *m_materialSnapshotData, material);
    m_undoStack->pushExecuted(std::move(cmd));
    m_materialSnapshotData.reset();
  }

  m_materialModifiedLastFrame = materialModifiedThisFrame;
}

//==================================================================================================
// MATERIAL ASSIGNMENT TOOLBAR (for primitive context)
//==================================================================================================

void UiInspector::renderMaterialAssignmentToolbar(int meshIdx, int primIdx, int nodeIdx, int matIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  ImGui::Text("Primitive/Material Operations");
  ImGui::SameLine();

  ImGui::PushID("mat_toolbar");

  // [Split] -- duplicate mesh + material for independent editing (### id for UI-test scenarios)
  if(ImGui::SmallButton(ICON_MS_CALL_SPLIT "###split"))
  {
    int newMatIdx = m_scene->editor().splitPrimitiveMaterial(nodeIdx, primIdx);
    if(newMatIdx >= 0 && m_selection)
    {
      int newMeshIdx = m_scene->getModel().nodes[nodeIdx].mesh;

      // Find the new render node index for this node+primitive
      int newRenderNodeIdx = m_scene->getRenderNodeForPrimitive(nodeIdx, primIdx);

      m_selection->selectPrimitive(newRenderNodeIdx, nodeIdx, primIdx, newMeshIdx);
    }
  }
  if(ImGui::IsItemHovered())
  {
    ImGui::SetTooltip("Split\nDuplicate mesh and material for independent editing");
  }

  // [Merge] -- reverse of Split: use shared mesh, remove duplicate (search done only on click to avoid cost in large scenes)
  ImGui::SameLine(0.0f, 2.0f);
  if(ImGui::SmallButton(ICON_MS_CALL_MERGE "###merge"))
  {
    int result = m_scene->editor().mergePrimitiveMaterial(nodeIdx);
    if(result >= 0 && m_selection)
    {
      int newMeshIdx = m_scene->getModel().nodes[nodeIdx].mesh;
      int newRnId    = m_scene->getRenderNodeForPrimitive(nodeIdx, primIdx);
      m_selection->selectPrimitive(newRnId, nodeIdx, primIdx, newMeshIdx);
    }
    // On failure, mergePrimitiveMaterial logs "no equivalent mesh for mesh N"
  }
  if(ImGui::IsItemHovered())
  {
    ImGui::SetTooltip("Merge\nUse shared mesh if one exists, remove duplicate (searches on click)");
  }

  // [Assign] -- pick a different existing material
  ImGui::SameLine(0.0f, 2.0f);
  if(ImGui::SmallButton(ICON_MS_SWAP_HORIZ))
  {
    ImGui::OpenPopup("AssignMaterial");
  }
  if(ImGui::IsItemHovered())
  {
    ImGui::SetTooltip("Assign a different material");
  }

  // [Copy] -- copy material reference to clipboard (disabled when no material assigned)
  ImGui::SameLine(0.0f, 2.0f);
  ImGui::BeginDisabled(matIdx < 0);
  if(ImGui::SmallButton(ICON_MS_CONTENT_COPY))
  {
    if(m_selection)
      m_selection->copyMaterialToClipboard(matIdx);
  }
  if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip(matIdx < 0 ? "Copy material reference\n(no material assigned)" : "Copy material reference");
  ImGui::EndDisabled();

  // [Paste] -- assign clipboard material to this primitive
  ImGui::SameLine(0.0f, 2.0f);
  bool hasClip = m_selection && m_selection->hasClipboardMaterial();
  ImGui::BeginDisabled(!hasClip);
  if(ImGui::SmallButton(ICON_MS_CONTENT_PASTE))
  {
    if(m_selection)
    {
      int clipMat = m_selection->getMaterialFromClipboard();
      if(clipMat >= 0 && clipMat < static_cast<int>(model.materials.size()))
      {
        LOGI("Pasting material %d (%s) to primitive %d of mesh %d\n", clipMat, model.materials[clipMat].name.c_str(), primIdx, meshIdx);
        m_scene->editor().setPrimitiveMaterial(meshIdx, primIdx, clipMat);
        m_scene->markMaterialDirty(matIdx);
        m_scene->markMaterialDirty(clipMat);
      }
    }
  }
  if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
  {
    if(hasClip)
    {
      int clipMat = m_selection->getMaterialFromClipboard();
      if(clipMat >= 0 && clipMat < static_cast<int>(model.materials.size()))
        ImGui::SetTooltip("Paste: %s (mat %d)", model.materials[clipMat].name.c_str(), clipMat);
      else
        ImGui::SetTooltip("Paste material\n(invalid clipboard)");
    }
    else
    {
      ImGui::SetTooltip("Paste material\n(no material copied)");
    }
  }
  ImGui::EndDisabled();

  // Assign material popup (modal)
  bool popupOpen = true;
  ImGui::SetNextWindowSize(ImVec2(520.0f, 0.0f), ImGuiCond_Once);
  if(ImGui::BeginPopupModal("AssignMaterial", &popupOpen, ImGuiWindowFlags_None))
  {
    ImGuiStorage* storage     = ImGui::GetStateStorage();
    ImGuiID       listId      = ImGui::GetID("material_select_index");
    int           selectedIdx = storage->GetInt(listId, matIdx);
    selectedIdx               = std::clamp(selectedIdx, 0, static_cast<int>(model.materials.size() - 1));

    ImGui::TextUnformatted("Select material:");
    ImVec2 listSize(-FLT_MIN, 8.0f * ImGui::GetTextLineHeightWithSpacing());
    if(ImGui::BeginListBox("##MaterialList", listSize))
    {
      for(int i = 0; i < static_cast<int>(model.materials.size()); ++i)
      {
        const tinygltf::Material& mat     = model.materials[i];
        std::string               matName = mat.name;
        if(matName.empty())
          matName = "Material " + std::to_string(i);
        matName += " (mat " + std::to_string(i) + ")";

        const bool selected = (selectedIdx == i);
        if(ImGui::Selectable(matName.c_str(), selected))
          selectedIdx = i;
        if(selected)
          ImGui::SetItemDefaultFocus();
      }
      ImGui::EndListBox();
    }
    storage->SetInt(listId, selectedIdx);

    if(ImGui::Button(ICON_MS_CHECK " OK"))
    {
      if(selectedIdx >= 0 && selectedIdx < static_cast<int>(model.materials.size()))
      {
        LOGI("Assigning material %d (%s) to primitive %d of mesh %d\n", selectedIdx,
             model.materials[selectedIdx].name.c_str(), primIdx, meshIdx);
        m_scene->editor().setPrimitiveMaterial(meshIdx, primIdx, selectedIdx);
        m_scene->markMaterialDirty(matIdx);
        m_scene->markMaterialDirty(selectedIdx);
      }
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if(ImGui::Button(ICON_MS_CANCEL " Cancel"))
    {
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  }

  ImGui::PopID();
}

//==================================================================================================
// OPERATIONS
//==================================================================================================

void UiInspector::renderMaterialOperations(int matIdx, int nodeContext)
{
  ImGui::Text("Material Operations:");

  if(ImGui::SmallButton(ICON_MS_CONTENT_COPY " Duplicate"))
  {
    const int          newIdx = static_cast<int>(m_scene->getModel().materials.size());
    tinygltf::Material mat    = m_scene->getModel().materials[matIdx];
    mat.name += "_copy";
    if(m_undoStack)
      m_undoStack->executeCommand(std::make_unique<MaterialLifecycleCommand>(*m_scene, newIdx, mat, true, "Duplicate material"));
    else
      (void)m_scene->editor().insertMaterialAt(newIdx, mat);  // tail append: newIdx is always the effective index
    LOGI("Duplicated material %d -> %d\n", matIdx, newIdx);
    if(m_selection)
      m_selection->selectMaterial(newIdx);
  }
  if(ImGui::IsItemHovered())
  {
    ImGui::SetTooltip("Create a copy of this material");
  }

  ImGui::SameLine();
  if(ImGui::SmallButton(ICON_MS_CONTENT_COPY " Copy"))
  {
    if(m_selection)
      m_selection->copyMaterialToClipboard(matIdx);
  }
  if(ImGui::IsItemHovered())
  {
    ImGui::SetTooltip("Copy material reference to clipboard");
  }
}

//==================================================================================================
// MATERIAL EXTENSIONS
//==================================================================================================

bool UiInspector::renderMaterialExtensions(tinygltf::Material& material, int matIdx)
{
  ImGui::Separator();
  ImGui::Text("Material Extensions:");

  // All material extensions - collect changes
  bool anyChange = false;
  anyChange |= materialAnisotropy(material);
  anyChange |= materialClearcoat(material);
  anyChange |= materialDiffuseTransmission(material);
  anyChange |= materialDispersion(material);
  anyChange |= materialEmissiveStrength(material);
  anyChange |= materialIor(material);
  anyChange |= materialIridescence(material);
  anyChange |= materialSheen(material);
  anyChange |= materialSpecular(material);
  anyChange |= materialTransmission(material);
  anyChange |= materialRetroreflection(material);
  anyChange |= materialUnlit(material);
  anyChange |= materialVolume(material, matIdx);  // Volume needs matIdx for special RTX dirty marking
  anyChange |= materialScatter(material);
  anyChange |= materialDlssNr(material);

  // Single point of truth for dirty marking
  if(anyChange)
    m_scene->markMaterialDirty(matIdx);

  return anyChange;
}

bool UiInspector::addButton(const char* extensionName, std::function<void()> addCallback)
{
  ImGui::PushID(extensionName);
  bool clicked = ImGui::SmallButton("Add");
  if(clicked)
    addCallback();
  ImGui::PopID();
  return clicked;
}

bool UiInspector::removeButton(tinygltf::Material& material, const char* extensionName)
{
  ImGui::SameLine();
  ImGui::PushID(extensionName);
  bool clicked = ImGui::SmallButton("Remove");
  if(clicked)
    material.extensions.erase(extensionName);
  ImGui::PopID();
  return clicked;
}

bool UiInspector::renderMaterialExtensionSection(tinygltf::Material&          material,
                                                 const char*                  treeLabel,
                                                 const char*                  extName,
                                                 const std::function<bool()>& whenHasExt,
                                                 const std::function<void()>& whenAdd,
                                                 const char*                  aliasExtName)
{
  // aliasExtName lets an extension be recognized under a legacy name too (e.g.
  // KHR_materials_volume_scatter for KHR_materials_scatter), so legacy assets show the populated
  // section instead of an "Add" button. The section's editor callback migrates to extName on edit.
  bool hasExt = tinygltf::utils::hasElementName(material.extensions, extName)
                || (aliasExtName && tinygltf::utils::hasElementName(material.extensions, aliasExtName));
  bool changed = false;
  if(ImGui::TreeNodeEx(treeLabel, hasExt ? ImGuiTreeNodeFlags_DefaultOpen : 0))
  {
    if(hasExt)
    {
      if(removeButton(material, extName))
      {
        if(aliasExtName)
          material.extensions.erase(aliasExtName);  // remove the legacy-named copy too
        changed = true;
      }
      changed |= whenHasExt();
    }
    else
      changed |= addButton(extName, whenAdd);
    ImGui::TreePop();
  }
  return changed;
}

//==================================================================================================
// MATERIAL EXTENSION IMPLEMENTATIONS (Simplified - showing pattern)
//==================================================================================================

bool UiInspector::materialAnisotropy(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "Anisotropy", KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME,
      [this, &material]() {
        KHR_materials_anisotropy anisotropy = tinygltf::utils::getAnisotropy(material);
        bool                     modif      = false;
        if(PE::begin())
        {
          float strength = anisotropy.anisotropyStrength;
          float rotation = anisotropy.anisotropyRotation;
          modif |= PE::DragFloat("Strength", &strength, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                                 "Anisotropy strength [0,1]. Increases roughness along the tangent direction.\n"
                                 "Multiplied by the anisotropy texture blue channel.");
          modif |= PE::DragFloat("Rotation", &rotation, 0.01f, -3.14f, 3.14f, "%.3f", 0,
                                 "Rotation of the anisotropy direction in radians, counter-clockwise from tangent.\n"
                                 "Additional rotation on top of the anisotropy texture direction.");
          modif |= renderTextureEditRow("Anisotropy", anisotropy.anisotropyTexture);
          if(modif)
          {
            anisotropy.anisotropyStrength = strength;
            anisotropy.anisotropyRotation = rotation;
          }
          PE::end();
        }
        if(modif)
          tinygltf::utils::setAnisotropy(material, anisotropy);
        return modif;
      },
      [&material]() { tinygltf::utils::setAnisotropy(material, {}); });
}

bool UiInspector::materialClearcoat(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "Clearcoat", KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME,
      [this, &material]() {
        KHR_materials_clearcoat clearcoat = tinygltf::utils::getClearcoat(material);
        bool                    modif     = false;
        if(PE::begin())
        {
          modif |= PE::DragFloat("Factor", &clearcoat.factor, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                                 "Clearcoat layer intensity [0,1]. 0 = disabled.\n"
                                 "Models a protective coating (e.g. varnish, car paint) with IOR 1.5.");
          modif |= PE::DragFloat("Roughness", &clearcoat.roughnessFactor, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                                 "Clearcoat layer roughness [0,1].\n"
                                 "Independent from base material roughness. Usually very low.");
          modif |= renderTextureEditRow("Clearcoat", clearcoat.texture);
          modif |= renderTextureEditRow("Clearcoat Roughness", clearcoat.roughnessTexture);
          modif |= renderTextureEditRow("Clearcoat Normal", clearcoat.normalTexture);
          PE::end();
        }
        if(modif)
          tinygltf::utils::setClearcoat(material, clearcoat);
        return modif;
      },
      [&material]() { tinygltf::utils::setClearcoat(material, {}); });
}

bool UiInspector::materialTransmission(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "Transmission", KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME,
      [this, &material]() {
        KHR_materials_transmission transmission = tinygltf::utils::getTransmission(material);
        bool                       modif        = false;
        if(PE::begin())
        {
          modif |= PE::DragFloat("Factor", &transmission.factor, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                                 "Percentage of non-specularly-reflected light transmitted through the surface [0,1].\n"
                                 "For thin-wall transparency (glass, plastic). Tinted by base color.");
          modif |= renderTextureEditRow("Transmission", transmission.texture);
          PE::end();
        }
        if(modif)
          tinygltf::utils::setTransmission(material, transmission);
        return modif;
      },
      [&material]() { tinygltf::utils::setTransmission(material, {}); });
}

bool UiInspector::materialRetroreflection(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "Retroreflection", KHR_MATERIALS_RETROREFLECTION_EXTENSION_NAME,
      [this, &material]() {
        KHR_materials_retroreflection retro = tinygltf::utils::getRetroreflection(material);
        bool                          modif = false;
        if(PE::begin())
        {
          modif |= PE::DragFloat("Factor", &retro.retroreflectionFactor, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                                 "Blend between forward microfacet (0) and the MRM retroreflective\n"
                                 "microfacet (1). Modulated per-texel by Retroreflection texture (R channel).\n"
                                 "Portsmouth et al. 2026 (JCGT, MRM model).");

          modif |= renderTextureEditRow("Retroreflection", retro.retroreflectionTexture);
          PE::end();
        }
        if(modif)
          tinygltf::utils::setRetroreflection(material, retro);
        return modif;
      },
      [&material]() { tinygltf::utils::setRetroreflection(material, {}); });
}

bool UiInspector::materialDiffuseTransmission(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "Diffuse Transmission", KHR_MATERIALS_DIFFUSE_TRANSMISSION_EXTENSION_NAME,
      [this, &material]() {
        KHR_materials_diffuse_transmission dt    = tinygltf::utils::getDiffuseTransmission(material);
        bool                               modif = false;
        if(PE::begin())
        {
          modif |= PE::DragFloat("Factor", &dt.diffuseTransmissionFactor, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                                 "Percentage of penetrating light that is diffusely transmitted [0,1].\n"
                                 "For thin translucent materials (leaves, paper, candle wax).");
          modif |= colorEdit3Linear("Color", glm::value_ptr(dt.diffuseTransmissionColor),
                                    "Color that modulates the diffusely transmitted light (shown/edited in linear; swatch/wheel perceptual).\n"
                                    "Acts as a transmission-side tint, independent of base color.");
          modif |= renderTextureEditRow("Diffuse Transmission", dt.diffuseTransmissionTexture);
          modif |= renderTextureEditRow("Diffuse Transmission Color", dt.diffuseTransmissionColorTexture);
          PE::end();
        }
        if(modif)
          tinygltf::utils::setDiffuseTransmission(material, dt);
        return modif;
      },
      [&material]() { tinygltf::utils::setDiffuseTransmission(material, {}); });
}

bool UiInspector::materialDispersion(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "Dispersion", KHR_MATERIALS_DISPERSION_EXTENSION_NAME,
      [&material]() {
        KHR_materials_dispersion dispersion = tinygltf::utils::getDispersion(material);
        bool                     modif      = false;
        if(PE::begin())
        {
          modif |= PE::DragFloat("Dispersion Factor", &dispersion.dispersion, 0.01f, 0.0f, 10.0f, "%.3f", 0,
                                 "Chromatic aberration strength, defined as 20/Abbe number.\n"
                                 "0 = no dispersion. 1.0 ~ Abbe 20 (strong). Requires volume+transmission.\n"
                                 "Examples: diamond 0.36, glass 0.33, polycarbonate 0.63.");
          PE::end();
        }
        if(modif)
          tinygltf::utils::setDispersion(material, dispersion);
        return modif;
      },
      [&material]() { tinygltf::utils::setDispersion(material, {}); });
}

bool UiInspector::materialEmissiveStrength(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "Emissive Strength", KHR_MATERIALS_EMISSIVE_STRENGTH_EXTENSION_NAME,
      [&material]() {
        KHR_materials_emissive_strength strength = tinygltf::utils::getEmissiveStrength(material);
        bool                            modif    = false;
        if(PE::begin())
        {
          modif |= PE::DragFloat("Emissive Strength", &strength.emissiveStrength,
                                 logarithmicStep(strength.emissiveStrength), 0.0f, FLT_MAX, "%.3f", 0,
                                 "Unitless multiplier on emissiveFactor * emissiveTexture.\n"
                                 "Allows HDR emission beyond the core [0,1] range.\n"
                                 "Affects reflections, bloom, and tonemapping.");
          PE::end();
        }
        if(modif)
          tinygltf::utils::setEmissiveStrength(material, strength);
        return modif;
      },
      [&material]() { tinygltf::utils::setEmissiveStrength(material, {}); });
}

bool UiInspector::materialIor(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "IOR", KHR_MATERIALS_IOR_EXTENSION_NAME,
      [&material]() {
        KHR_materials_ior ior   = tinygltf::utils::getIor(material);
        bool              modif = false;
        if(PE::begin())
        {
          modif |= PE::DragFloat("IOR", &ior.ior, 0.01f, 0.0f, 10.0f, "%.3f", 0,
                                 "Index of refraction. Default 1.5 (glass/plastic).\n"
                                 "Examples: water 1.33, glass 1.52, diamond 2.42.");
          PE::end();
        }
        if(modif)
          tinygltf::utils::setIor(material, ior);
        return modif;
      },
      [&material]() { tinygltf::utils::setIor(material, {}); });
}

bool UiInspector::materialIridescence(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "Iridescence", KHR_MATERIALS_IRIDESCENCE_EXTENSION_NAME,
      [this, &material]() {
        KHR_materials_iridescence iridescence = tinygltf::utils::getIridescence(material);
        bool                      modif       = false;
        if(PE::begin())
        {
          modif |= PE::DragFloat("Iridescence Factor", &iridescence.iridescenceFactor, 0.01f, 0.0f, 10.0f, "%.3f", 0,
                                 "Iridescence intensity [0,1]. 0 = disabled.\n"
                                 "Thin-film interference effect (soap bubbles, oil films, insect wings).");
          modif |= PE::DragFloat("Iridescence Ior", &iridescence.iridescenceIor, 0.01f, 0.0f, 10.0f, "%.3f", 0,
                                 "Index of refraction of the thin-film layer.\n"
                                 "The further from the base IOR, the stronger the iridescence. Default: 1.3.");
          modif |= PE::DragFloat("Thickness Min", &iridescence.iridescenceThicknessMinimum, 0.01f, 0.0f, 1000.0f, "%.3f nm", 0,
                                 "Minimum thin-film thickness in nanometers.\n"
                                 "Maps to thickness texture value 0.0. Default: 100 nm.");
          modif |= PE::DragFloat("Thickness Max", &iridescence.iridescenceThicknessMaximum, 0.01f, 0.0f, 1000.0f, "%.3f nm", 0,
                                 "Maximum thin-film thickness in nanometers.\n"
                                 "Maps to thickness texture value 1.0. Default: 400 nm.\n"
                                 "Visible light is 380-750 nm; half-wavelength gives strongest effect.");
          modif |= renderTextureEditRow("Iridescence", iridescence.iridescenceTexture);
          modif |= renderTextureEditRow("Iridescence Thickness", iridescence.iridescenceThicknessTexture);
          PE::end();
        }
        if(modif)
          tinygltf::utils::setIridescence(material, iridescence);
        return modif;
      },
      [&material]() { tinygltf::utils::setIridescence(material, {}); });
}

bool UiInspector::materialSheen(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "Sheen", KHR_MATERIALS_SHEEN_EXTENSION_NAME,
      [this, &material]() {
        KHR_materials_sheen sheen = tinygltf::utils::getSheen(material);
        bool                modif = false;
        if(PE::begin())
        {
          modif |= colorEdit3Linear("Sheen Color", glm::value_ptr(sheen.sheenColorFactor),
                                    "Sheen color (shown/edited in linear; swatch/wheel perceptual). Black = disabled.\n"
                                    "Models back-scattering from fabric micro-fibers (velvet, cloth).");
          modif |= PE::DragFloat("Sheen Roughness", &sheen.sheenRoughnessFactor, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                                 "Sheen roughness [0,1]. Controls micro-fiber divergence.\n"
                                 "Low = sharp grazing-angle highlights. High = soft, broad sheen.");
          modif |= renderTextureEditRow("Sheen Color", sheen.sheenColorTexture);
          modif |= renderTextureEditRow("Sheen Roughness", sheen.sheenRoughnessTexture);
          PE::end();
        }
        if(modif)
          tinygltf::utils::setSheen(material, sheen);
        return modif;
      },
      [&material]() { tinygltf::utils::setSheen(material, {}); });
}

bool UiInspector::materialSpecular(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "Specular", KHR_MATERIALS_SPECULAR_EXTENSION_NAME,
      [this, &material]() {
        KHR_materials_specular specular = tinygltf::utils::getSpecular(material);
        bool                   modif    = false;
        if(PE::begin())
        {
          modif |= colorEdit3Linear("Specular Color", glm::value_ptr(specular.specularColorFactor),
                                    "F0 color tint for dielectric specular reflection (shown/edited in linear; swatch/wheel perceptual).\n"
                                    "At normal incidence, multiplies the IOR-derived F0. At grazing, remains white.");
          modif |= PE::DragFloat("Specular Factor", &specular.specularFactor, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                                 "Strength of dielectric specular reflection [0,1]. 0 = pure diffuse.\n"
                                 "Does not affect metals. Scales both F0 and F90.");
          modif |= renderTextureEditRow("Specular", specular.specularTexture);
          modif |= renderTextureEditRow("Specular Color", specular.specularColorTexture);
          PE::end();
        }
        if(modif)
          tinygltf::utils::setSpecular(material, specular);
        return modif;
      },
      [&material]() { tinygltf::utils::setSpecular(material, {}); });
}

bool UiInspector::materialUnlit(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "Unlit", KHR_MATERIALS_UNLIT_EXTENSION_NAME,
      []() {
        ImGui::TextWrapped("Material is unlit (no lighting applied)");
        return false;
      },
      [&material]() { tinygltf::utils::setUnlit(material, {}); });
}

bool UiInspector::materialVolume(tinygltf::Material& material, int matIdx)
{
  return renderMaterialExtensionSection(
      material, "Volume", KHR_MATERIALS_VOLUME_EXTENSION_NAME,
      [this, &material, matIdx]() {
        KHR_materials_volume volume          = tinygltf::utils::getVolume(material);
        bool                 modif           = false;
        float                thicknessFactor = volume.thicknessFactor;
        if(PE::begin())
        {
          modif |= PE::DragFloat("Thickness", &volume.thicknessFactor, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                                 "Volume thickness beneath the surface (mesh coordinate space).\n"
                                 "0 = thin-walled. > 0 = volumetric (requires closed mesh).\n"
                                 "Path tracer: acts as an on/off switch only -- the magnitude doesn't matter (0.1 looks\n"
                                 "the same as 5.0), because absorption uses the real ray-traced hit distance instead.\n"
                                 "Rasterizer: has no ray-traced exit point, so this value's magnitude directly scales\n"
                                 "the approximate transmission ray length.");
          modif |= colorEdit3Linear("Attenuation Color", glm::value_ptr(volume.attenuationColor),
                                    "Color that white light becomes after traveling the attenuation distance\n"
                                    "(shown/edited in linear; swatch/wheel perceptual). Models wavelength-dependent absorption via Beer's law.");

          bool isInfinite = (volume.attenuationDistance >= FLT_MAX);
          if(PE::Checkbox("Infinite Attenuation", &isInfinite, "No light absorption (infinite distance)"))
          {
            volume.attenuationDistance = isInfinite ? FLT_MAX : 1.0f;
            modif                      = true;
          }
          if(!isInfinite)
          {
            modif |= PE::DragFloat("Attenuation Distance", &volume.attenuationDistance,
                                   logarithmicStep(volume.attenuationDistance), 1e-6f, FLT_MAX, "%.4g", ImGuiSliderFlags_None,
                                   "Average distance light travels before interacting with a particle (world space).\n"
                                   "Smaller = denser medium, faster color absorption.");
          }

          modif |= renderTextureEditRow("Volume Thickness", volume.thicknessTexture);
          PE::end();
        }
        if(modif)
        {
          tinygltf::utils::setVolume(material, volume);
          if(thicknessFactor == 0.0f && volume.thicknessFactor != 0.0f)
            m_scene->markRenderNodeRtxDirtyForMaterials({matIdx});
        }
        return modif;
      },
      [&material]() { tinygltf::utils::setVolume(material, {}); });
}

bool UiInspector::materialScatter(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "Scatter", KHR_MATERIALS_SCATTER_EXTENSION_NAME,
      [this, &material]() {
        KHR_materials_scatter scatter = tinygltf::utils::getScatter(material);
        bool                  modif   = false;
        if(PE::begin())
        {
          modif |= PE::SliderFloat("Scatter Strength", &scatter.scatterStrengthFactor, 0.0f, 1.0f, "%.3f", 0,
                                   "Strength of the scattering effect [0, 1].\n"
                                   "0 = no scattering (extension has no effect). 1 = full scattering.\n"
                                   "With KHR_materials_volume (thickness > 0): volumetric scattering.\n"
                                   "Without volume (thin-walled): anisotropy-controlled diffuse transmission/reflection.");
          modif |= colorEdit3Linear("Multiscatter Color", glm::value_ptr(scatter.multiscatterColorFactor),
                                    "Multi-scatter albedo (shown/edited in linear; swatch/wheel perceptual).\n"
                                    "Approximates the perceived color after many scattering bounces.");
          // The range is open, and the loader clamps to +/-0.999; keep the slider inside it so the
          // edited value survives a save/reload round-trip unchanged.
          modif |= PE::SliderFloat("Scatter Anisotropy", &scatter.scatterAnisotropy, -0.999f, 0.999f, "%.3f", 0,
                                   "Henyey-Greenstein phase function parameter (-1, 1).\n"
                                   "0 = isotropic. Positive = forward scattering. Negative = backward scattering.\n"
                                   "Thin-walled: splits the scattered energy between transmission (+1) and reflection (-1).");
          modif |= renderTextureEditRow("Scatter Strength", scatter.scatterStrengthTexture);
          modif |= renderTextureEditRow("Multiscatter Color", scatter.multiscatterColorTexture);
          PE::end();
        }
        if(modif)
        {
          // Editing migrates to the current extension name; drop any legacy-named copy.
          material.extensions.erase(KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME);
          tinygltf::utils::setScatter(material, scatter);
        }
        return modif;
      },
      [&material]() {
        material.extensions.erase(KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME);
        tinygltf::utils::setScatter(material, {});
      },
      KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME);
}

bool UiInspector::materialDlssNr(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "DLSS-NR Mask", EXT_DLSS_NR_EXTENSION_NAME,
      [&material]() {
        EXT_DLSS_NR ext   = tinygltf::utils::getExtDlssNr(material);
        bool        modif = false;
        if(PE::begin())
        {
          modif |= PE::SliderFloat("Intensity", &ext.nrMask.x, 0.0f, 1.0f, "%.3f", 0,
                                   "Scales the global NR intensity for this material. 1 = global value, 0 = NR disabled.");
          modif |= PE::SliderFloat("Local Tone", &ext.nrMask.y, 0.0f, 1.0f, "%.3f", 0,
                                   "Scales the local tone strength for this material. 1 = global value, 0 = disabled.");
          modif |= PE::SliderFloat("Local Structure", &ext.nrMask.z, 0.0f, 1.0f, "%.3f", 0,
                                   "Scales the local structure strength for this material. 1 = global value, 0 = disabled.");
          modif |= PE::SliderFloat("Global Tone", &ext.nrMask.w, 0.0f, 1.0f, "%.3f", 0,
                                   "Scales the global tone strength for this material. 1 = global value, 0 = disabled.");
          PE::end();
        }
        if(modif)
          tinygltf::utils::setExtDlssNr(material, ext);
        return modif;
      },
      [&material]() { tinygltf::utils::setExtDlssNr(material, {}); });
}
