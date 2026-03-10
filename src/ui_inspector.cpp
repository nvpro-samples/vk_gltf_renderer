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

#include <imgui.h>
#include <imgui_internal.h>
#include <nvgui/property_editor.hpp>
#include <nvgui/fonts.hpp>
#include <nvutils/logger.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace PE = nvgui::PropertyEditor;

//==================================================================================================
// CONSTANTS
//==================================================================================================

static const double f64_zero = 0., f64_one = 1., f64_ten = 10., f64_179 = 179., f64_001 = 0.001, f64_1000 = 1000.,
                    f64_10000 = 10000., f64_01 = 0.1, f64_100 = 100., f64_neg1000 = -1000.;

static float logarithmicStep(float value)
{
  return std::max(0.1f * std::pow(10.0f, std::floor(std::log10(value))), 0.001f);
}

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

// Texture editing UI - handles add/remove/change texture
template <typename T>
static bool renderTextureEditRow(const char* label, T& info, const tinygltf::Model& model, const std::vector<std::string>& textureItems)
{
  bool changed = false;

  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::Text("%s %s", ICON_MS_IMAGE, label);

  ImGui::TableNextColumn();
  const bool hasTexture = info.index >= 0;
  if(hasTexture)
  {
    // Use pre-computed texture name from textureItems
    const std::string displayName = (info.index >= 0 && info.index < static_cast<int>(textureItems.size())) ?
                                        textureItems[info.index] :
                                        "Invalid texture " + std::to_string(info.index);

    const float  buttonWidth  = ImGui::CalcTextSize(ICON_MS_DELETE).x + ImGui::GetStyle().FramePadding.x * 2.0f;
    const float  spacing      = ImGui::GetStyle().ItemSpacing.x;
    const float  available    = ImGui::GetContentRegionAvail().x;
    const float  textMaxWidth = std::max(0.0f, available - 2 * buttonWidth - 2 * spacing);
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

      // Show texture -> image -> URI path
      if(info.index >= 0 && info.index < static_cast<int>(model.textures.size()))
      {
        const tinygltf::Texture& texture = model.textures[info.index];
        ImGui::Separator();

        if(texture.source >= 0 && texture.source < static_cast<int>(model.images.size()))
        {
          const tinygltf::Image& image = model.images[texture.source];

          // Check if image is embedded or has a URI
          if(!image.uri.empty())
          {
            ImGui::Text("Image: %s", image.uri.c_str());
          }
          else
          {
            ImGui::Text("Image: %d (EMB)", texture.source);
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
    if(textureItems.empty())
    {
      ImGui::TextDisabled(ICON_MS_ADD_CIRCLE);
    }
    else
    {
      if(ImGui::SmallButton(ICON_MS_ADD_CIRCLE))
      {
        ImGui::OpenPopup("SwitchTexture");
      }
      if(ImGui::IsItemHovered())
      {
        ImGui::BeginTooltip();
        ImGui::TextUnformatted("Add texture");
        ImGui::EndTooltip();
      }
    }
  }

  // Unified texture selection popup (works for both add and switch)
  if(!textureItems.empty())
  {
    bool open = true;
    ImGui::SetNextWindowSize(ImVec2(520.0f, 0.0f), ImGuiCond_Once);
    if(ImGui::BeginPopupModal("SwitchTexture", &open, ImGuiWindowFlags_None))
    {
      ImGuiStorage* storage = ImGui::GetStateStorage();
      ImGuiID       listId  = ImGui::GetID("texture_select_index");
      int           selIdx  = storage->GetInt(listId, hasTexture ? info.index : 0);
      selIdx                = std::clamp(selIdx, 0, static_cast<int>(textureItems.size() - 1));

      ImGui::TextUnformatted(hasTexture ? "Switch to texture:" : "Select texture:");
      ImVec2 listSize(-FLT_MIN, 8.0f * ImGui::GetTextLineHeightWithSpacing());
      if(ImGui::BeginListBox("##TextureList", listSize))
      {
        for(int i = 0; i < static_cast<int>(textureItems.size()); ++i)
        {
          const bool selected = (selIdx == i);
          if(ImGui::Selectable(textureItems[i].c_str(), selected))
            selIdx = i;
          if(selected)
            ImGui::SetItemDefaultFocus();
        }
        ImGui::EndListBox();
      }
      storage->SetInt(listId, selIdx);

      if(ImGui::Button(ICON_MS_CHECK " OK"))
      {
        info.index    = selIdx;
        info.texCoord = 0;
        changed       = true;
        ImGui::CloseCurrentPopup();
      }
      ImGui::SameLine();
      if(ImGui::Button(ICON_MS_CANCEL " Cancel"))
      {
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }
  }
  ImGui::PopID();

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
    return;

  const tinygltf::Model& model = scene->getModel();

  // Build texture names cache
  m_textureNames.clear();
  m_textureNames.reserve(model.textures.size());
  for(int i = 0; i < static_cast<int>(model.textures.size()); ++i)
  {
    m_textureNames.push_back(getTextureDisplayName(model, i));
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
// NODE PROPERTIES
//==================================================================================================

void UiInspector::renderNodeProperties(int nodeIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  if(nodeIdx < 0 || nodeIdx >= static_cast<int>(model.nodes.size()))
    return;

  const tinygltf::Node& node = model.nodes[nodeIdx];

  ImGui::Text("%s Node: %s", ICON_MS_CATEGORY, node.name.c_str());

  // XMP button
  std::string popupId = "inspector_node_xmp_" + std::to_string(nodeIdx);
  ImGui::SameLine();
  ui_xmp::renderInfoButton(&m_scene->getModel(), node.extensions, popupId.c_str());

  ImGui::Separator();

  renderTransformSection(nodeIdx);
  if(node.light > -1)
    renderLightProperties(node.light);
  if(node.camera > -1)
    renderCameraProperties(node.camera);
}

//==================================================================================================
// PRIMITIVE PROPERTIES
//==================================================================================================

void UiInspector::renderPrimitiveProperties(int nodeIdx, int primIdx, int meshIdx)
{
  const tinygltf::Model& model = m_scene->getModel();

  // Transform section (from parent node)
  if(ImGui::CollapsingHeader("TRANSFORM (Node)"))
  {
    renderTransformSection(nodeIdx);
  }

  // Primitive -- clean: identity + geometry stats only
  if(ImGui::CollapsingHeader("PRIMITIVE"))
  {
    ImGui::Text("%s Primitive %d", ICON_MS_SHAPE_LINE, primIdx);
    if(meshIdx >= 0 && meshIdx < static_cast<int>(model.meshes.size()))
      ImGui::Text("%s Mesh(%d): %s", ICON_MS_SUBDIRECTORY_ARROW_RIGHT, meshIdx, model.meshes[meshIdx].name.c_str());
    if(nodeIdx >= 0 && nodeIdx < static_cast<int>(model.nodes.size()))
      ImGui::Text("%s Node(%d): %s", ICON_MS_SUBDIRECTORY_ARROW_RIGHT, nodeIdx, model.nodes[nodeIdx].name.c_str());
    ImGui::Separator();

    if(meshIdx >= 0 && meshIdx < static_cast<int>(model.meshes.size()))
    {
      const tinygltf::Mesh& mesh = model.meshes[meshIdx];
      if(primIdx >= 0 && primIdx < static_cast<int>(mesh.primitives.size()))
      {
        const tinygltf::Primitive& primitive = mesh.primitives[primIdx];

        int  vertexCount = 0;
        auto posIt       = primitive.attributes.find("POSITION");
        if(posIt != primitive.attributes.end())
        {
          int accessorIdx = posIt->second;
          if(accessorIdx >= 0 && accessorIdx < static_cast<int>(model.accessors.size()))
            vertexCount = static_cast<int>(model.accessors[accessorIdx].count);
        }

        int triangleCount = 0;
        if(primitive.indices >= 0 && primitive.indices < static_cast<int>(model.accessors.size()))
          triangleCount = static_cast<int>(model.accessors[primitive.indices].count) / 3;

        ImGui::Text("Vertices: %d   Triangles: %d", vertexCount, triangleCount);
      }
    }
  }

  // Material -- single home for assignment toolbar + property editor
  if(meshIdx >= 0 && meshIdx < static_cast<int>(model.meshes.size()))
  {
    const tinygltf::Mesh& mesh = model.meshes[meshIdx];
    if(primIdx >= 0 && primIdx < static_cast<int>(mesh.primitives.size()))
    {
      int matIdx = mesh.primitives[primIdx].material;
      if(matIdx >= 0 && matIdx < static_cast<int>(model.materials.size()))
      {
        if(ImGui::CollapsingHeader("MATERIAL", ImGuiTreeNodeFlags_DefaultOpen))
        {
          renderMaterialAssignmentToolbar(meshIdx, primIdx, nodeIdx, matIdx);
          ImGui::Separator();
          renderMaterialSection(matIdx, true);
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

  ImGui::Text("%s Material: %s", ICON_MS_BRUSH, material.name.c_str());

  // XMP button
  std::string popupId = "inspector_mat_xmp_" + std::to_string(matIdx);
  ImGui::SameLine();
  ui_xmp::renderInfoButton(&m_scene->getModel(), material.extensions, popupId.c_str());

  ImGui::Separator();

  renderMaterialSection(matIdx, true);

  ImGui::Separator();

  renderMaterialOperations(matIdx, -1);
}

//==================================================================================================
// MESH PROPERTIES
//==================================================================================================

void UiInspector::renderMeshProperties(int meshIdx)
{
  const tinygltf::Model& model = m_scene->getModel();
  if(meshIdx < 0 || meshIdx >= static_cast<int>(model.meshes.size()))
    return;

  const tinygltf::Mesh& mesh = model.meshes[meshIdx];

  ImGui::Text("%s Mesh: %s", ICON_MS_VIEW_IN_AR, mesh.name.c_str());

  // XMP button
  std::string popupId = "inspector_mesh_xmp_" + std::to_string(meshIdx);
  ImGui::SameLine();
  ui_xmp::renderInfoButton(&m_scene->getModel(), mesh.extensions, popupId.c_str());

  ImGui::Separator();

  ImGui::Text("Primitives: %zu", mesh.primitives.size());

  // List primitives
  for(int i = 0; i < static_cast<int>(mesh.primitives.size()); ++i)
  {
    const tinygltf::Primitive& prim   = mesh.primitives[i];
    int                        matIdx = prim.material;

    std::string matName = "None";
    if(matIdx >= 0 && matIdx < static_cast<int>(model.materials.size()))
    {
      matName = model.materials[matIdx].name;
    }

    ImGui::BulletText("Primitive %d: %s", i, matName.c_str());
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

  ImGui::Text("%s Camera: %s", ICON_MS_CAMERA_ALT, camera.name.c_str());
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

  ImGui::Text("%s Light: %s", ICON_MS_LIGHTBULB, light.name.c_str());
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

    // Convert color to sRGB for editing
    glm::vec3 color = glm::vec3(light.color[0], light.color[1], light.color[2]);
    color           = glm::pow(color, glm::vec3(1.0f / 2.2f));  // Linear to sRGB

    if(PE::ColorEdit3("Color", glm::value_ptr(color), 0,
                      "RGB color in linear space.\n"
                      "Acts as a wavelength-specific multiplier on intensity."))
    {
      color          = glm::pow(color, glm::vec3(2.2f));
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

    // Visibility extension
    bool hasVisibility = tinygltf::utils::hasElementName(node.extensions, KHR_NODE_VISIBILITY_EXTENSION_NAME);
    if(hasVisibility)
    {
      KHR_node_visibility visibility = tinygltf::utils::getNodeVisibility(node);
      if(PE::Checkbox("Visible", &visibility.visible, "KHR_node_visibility: hide the node and its children from rendering."))
      {
        tinygltf::utils::setNodeVisibility(node, visibility);
        if(m_scene)
        {
          m_scene->editor().updateVisibility(nodeIdx);
          m_scene->markNodeDirty(nodeIdx);
        }
      }
    }
    else
    {
      if(ImGui::SmallButton("Add Visibility"))
      {
        tinygltf::utils::setNodeVisibility(node, {});  // Default: visible = true
        if(m_scene)
        {
          m_scene->editor().updateVisibility(nodeIdx);
          m_scene->markNodeDirty(nodeIdx);
        }
      }
    }

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
      if(PE::ColorEdit4("Diffuse Color", glm::value_ptr(diffuse), 0,
                        "RGBA diffuse color factor in linear space.\n"
                        "(KHR_materials_pbrSpecularGlossiness)"))
      {
        sg.diffuseFactor = diffuse;
        modif            = true;
      }
      modif |= renderTextureEditRow("Diffuse", sg.diffuseTexture, m_scene->getModel(), m_textureNames);

      glm::vec3 specular = sg.specularFactor;
      if(PE::ColorEdit3("Specular", glm::value_ptr(specular), 0,
                        "RGB specular color factor in linear space.\n"
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
      modif |= renderTextureEditRow("Specular-Glossiness", sg.specularGlossinessTexture, m_scene->getModel(), m_textureNames);

      if(modif)
        tinygltf::utils::setPbrSpecularGlossiness(material, sg);
    }
    else
    {
      tinygltf::PbrMetallicRoughness& pbr = material.pbrMetallicRoughness;

      glm::vec4 baseColor = glm::make_vec4(pbr.baseColorFactor.data());
      if(PE::ColorEdit4("Base Color", glm::value_ptr(baseColor), 0,
                        "RGBA base color factor in linear space.\n"
                        "RGB modulates diffuse/specular; A controls alpha coverage."))
      {
        pbr.baseColorFactor[0] = baseColor.x;
        pbr.baseColorFactor[1] = baseColor.y;
        pbr.baseColorFactor[2] = baseColor.z;
        pbr.baseColorFactor[3] = baseColor.w;
        modif                  = true;
      }
      modif |= renderTextureEditRow("Base Color", pbr.baseColorTexture, m_scene->getModel(), m_textureNames);

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
      modif |= renderTextureEditRow("Metallic-Roughness", pbr.metallicRoughnessTexture, m_scene->getModel(), m_textureNames);
    }

    // Emissive
    glm::vec3 emissive = glm::make_vec3(material.emissiveFactor.data());
    if(PE::ColorEdit3("Emissive", glm::value_ptr(emissive), 0,
                      "RGB emissive color in linear space.\n"
                      "Multiplied by emissiveStrength (if present) for HDR emission."))
    {
      material.emissiveFactor[0] = emissive.x;
      material.emissiveFactor[1] = emissive.y;
      material.emissiveFactor[2] = emissive.z;
      modif                      = true;
    }
    modif |= renderTextureEditRow("Emissive", material.emissiveTexture, m_scene->getModel(), m_textureNames);
    modif |= renderTextureEditRow("Normal", material.normalTexture, m_scene->getModel(), m_textureNames);
    modif |= renderTextureEditRow("Occlusion", material.occlusionTexture, m_scene->getModel(), m_textureNames);

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

  // [Split] -- duplicate mesh + material for independent editing
  if(ImGui::SmallButton(ICON_MS_CALL_SPLIT))
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
  if(ImGui::SmallButton(ICON_MS_CALL_MERGE))
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

  // [Copy] -- copy material reference to clipboard
  ImGui::SameLine(0.0f, 2.0f);
  if(ImGui::SmallButton(ICON_MS_CONTENT_COPY))
  {
    if(m_selection)
      m_selection->copyMaterialToClipboard(matIdx);
  }
  if(ImGui::IsItemHovered())
  {
    ImGui::SetTooltip("Copy material reference");
  }

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
    int newIdx = m_scene->editor().duplicateMaterial(matIdx);
    LOGI("Duplicated material %d -> %d\n", matIdx, newIdx);
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

  // All 12 material extensions - collect changes
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
  anyChange |= materialUnlit(material);
  anyChange |= materialVolume(material, matIdx);  // Volume needs matIdx for special RTX dirty marking
  anyChange |= materialVolumeScatter(material);

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
                                                 const std::function<void()>& whenAdd)
{
  bool hasExt  = tinygltf::utils::hasElementName(material.extensions, extName);
  bool changed = false;
  if(ImGui::TreeNodeEx(treeLabel, hasExt ? ImGuiTreeNodeFlags_DefaultOpen : 0))
  {
    if(hasExt)
    {
      changed |= removeButton(material, extName);
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
          modif |= renderTextureEditRow("Anisotropy", anisotropy.anisotropyTexture, m_scene->getModel(), m_textureNames);
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
          modif |= renderTextureEditRow("Clearcoat", clearcoat.texture, m_scene->getModel(), m_textureNames);
          modif |= renderTextureEditRow("Clearcoat Roughness", clearcoat.roughnessTexture, m_scene->getModel(), m_textureNames);
          modif |= renderTextureEditRow("Clearcoat Normal", clearcoat.normalTexture, m_scene->getModel(), m_textureNames);
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
          modif |= renderTextureEditRow("Transmission", transmission.texture, m_scene->getModel(), m_textureNames);
          PE::end();
        }
        if(modif)
          tinygltf::utils::setTransmission(material, transmission);
        return modif;
      },
      [&material]() { tinygltf::utils::setTransmission(material, {}); });
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
          modif |= PE::ColorEdit3("Color", glm::value_ptr(dt.diffuseTransmissionColor), 0,
                                  "Color that modulates the diffusely transmitted light.\n"
                                  "Acts as a transmission-side tint, independent of base color.");
          modif |= renderTextureEditRow("Diffuse Transmission", dt.diffuseTransmissionTexture, m_scene->getModel(), m_textureNames);
          modif |= renderTextureEditRow("Diffuse Transmission Color", dt.diffuseTransmissionColorTexture,
                                        m_scene->getModel(), m_textureNames);
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
          modif |= renderTextureEditRow("Iridescence", iridescence.iridescenceTexture, m_scene->getModel(), m_textureNames);
          modif |= renderTextureEditRow("Iridescence Thickness", iridescence.iridescenceThicknessTexture,
                                        m_scene->getModel(), m_textureNames);
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
          modif |= PE::ColorEdit3("Sheen Color", glm::value_ptr(sheen.sheenColorFactor), 0,
                                  "Sheen color in linear space. Black = disabled.\n"
                                  "Models back-scattering from fabric micro-fibers (velvet, cloth).");
          modif |= PE::DragFloat("Sheen Roughness", &sheen.sheenRoughnessFactor, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                                 "Sheen roughness [0,1]. Controls micro-fiber divergence.\n"
                                 "Low = sharp grazing-angle highlights. High = soft, broad sheen.");
          modif |= renderTextureEditRow("Sheen Color", sheen.sheenColorTexture, m_scene->getModel(), m_textureNames);
          modif |= renderTextureEditRow("Sheen Roughness", sheen.sheenRoughnessTexture, m_scene->getModel(), m_textureNames);
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
          modif |= PE::ColorEdit3("Specular Color", glm::value_ptr(specular.specularColorFactor), 0,
                                  "F0 color tint for dielectric specular reflection (linear RGB).\n"
                                  "At normal incidence, multiplies the IOR-derived F0. At grazing, remains white.");
          modif |= PE::DragFloat("Specular Factor", &specular.specularFactor, 0.01f, 0.0f, 1.0f, "%.3f", 0,
                                 "Strength of dielectric specular reflection [0,1]. 0 = pure diffuse.\n"
                                 "Does not affect metals. Scales both F0 and F90.");
          modif |= renderTextureEditRow("Specular", specular.specularTexture, m_scene->getModel(), m_textureNames);
          modif |= renderTextureEditRow("Specular Color", specular.specularColorTexture, m_scene->getModel(), m_textureNames);
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
                                 "Ray tracers use actual distance; rasterizers use this as approximation.");
          modif |= PE::ColorEdit3("Attenuation Color", glm::value_ptr(volume.attenuationColor), 0,
                                  "Color that white light becomes after traveling the attenuation distance.\n"
                                  "Models wavelength-dependent absorption via Beer's law.");

          bool isInfinite = (volume.attenuationDistance >= FLT_MAX);
          if(PE::Checkbox("Infinite Attenuation", &isInfinite, "No light absorption (infinite distance)"))
          {
            volume.attenuationDistance = isInfinite ? FLT_MAX : 1.0f;
            modif                      = true;
          }
          if(!isInfinite)
          {
            modif |= PE::DragFloat("Attenuation Distance", &volume.attenuationDistance,
                                   logarithmicStep(volume.attenuationDistance), 0.0, FLT_MAX, "%.3f", ImGuiSliderFlags_None,
                                   "Average distance light travels before interacting with a particle (world space).\n"
                                   "Smaller = denser medium, faster color absorption.");
          }

          ImGui::BeginDisabled(true);
          modif |= renderTextureEditRow("Thickness", volume.thicknessTexture, m_scene->getModel(), m_textureNames);
          ImGui::EndDisabled();
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

bool UiInspector::materialVolumeScatter(tinygltf::Material& material)
{
  return renderMaterialExtensionSection(
      material, "Volume Scatter", KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME,
      [&material]() {
        KHR_materials_volume_scatter volumeScatter = tinygltf::utils::getVolumeScatter(material);
        bool                         modif         = false;
        if(PE::begin())
        {
          modif |= PE::ColorEdit3("Multiscatter Color", glm::value_ptr(volumeScatter.multiscatterColor), 0,
                                  "Multi-scatter albedo (linear RGB). Black = no scattering.\n"
                                  "Approximates the perceived color after many scattering bounces.\n"
                                  "Requires KHR_materials_volume.");
          modif |= PE::SliderFloat("Scatter Anisotropy", &volumeScatter.scatterAnisotropy, -1.0f, 1.0f, "%.3f", 0,
                                   "Henyey-Greenstein phase function parameter (-1, 1).\n"
                                   "0 = isotropic. Positive = forward scattering. Negative = backward scattering.");
          PE::end();
        }
        if(modif)
          tinygltf::utils::setVolumeScatter(material, volumeScatter);
        return modif;
      },
      [&material]() { tinygltf::utils::setVolumeScatter(material, {}); });
}
