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

#include <algorithm>
#include <functional>

#include "ui_xmp.hpp"
#include <glm/glm.hpp>

namespace glm {
// vec3 specialization
GLM_FUNC_DECL vec3      fma(vec3 const& a, vec3 const& b, vec3 const& c);
GLM_FUNC_QUALIFIER vec3 fma(vec3 const& a, vec3 const& b, vec3 const& c)
{
  return {a.x * b.x + c.x, a.y * b.y + c.y, a.z * b.z + c.z};
}
}  // namespace glm


#include <nvgui/property_editor.hpp>
#include <nvvkgltf/tinygltf_utils.hpp>
#include <nvgui/fonts.hpp>

#include "ui_scene_graph.hpp"

#include <nvshaders/tonemap_functions.h.slang>

namespace PE = nvgui::PropertyEditor;

static ImGuiTreeNodeFlags s_treeNodeFlags = ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_SpanFullWidth
                                            | ImGuiTreeNodeFlags_SpanTextWidth | ImGuiTreeNodeFlags_OpenOnArrow
                                            | ImGuiTreeNodeFlags_OpenOnDoubleClick;

static const double f64_zero = 0., f64_one = 1., f64_ten = 10., f64_179 = 179., f64_001 = 0.001, f64_1000 = 1000.,
                    f64_10000 = 10000., f64_01 = 0.1, f64_100 = 100., f64_neg1000 = -1000.;

//--------------------------------------------------------------------------------------------------
// Entry point for rendering the scene graph
// Loop over all scenes
// - Loop over all nodes in the scene
//Following, in the second part, is the details:
// - Display the node details (transform)
//   OR Display the material details
//
void UiSceneGraph::render(bool* showSceneGraph, bool* showProperties)
{
  renderSceneGraph(showSceneGraph);
  renderDetails(showProperties);
}

void UiSceneGraph::renderSceneGraph(bool* showSceneGraph)
{
  if(showSceneGraph && !*showSceneGraph)
    return;

  static const float     textBaseWidth = ImGui::CalcTextSize("A").x;
  static ImGuiTableFlags s_tableFlags =
      ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV;

  if(ImGui::Begin("Scene Graph", showSceneGraph))
  {
    if(m_model == nullptr)
    {
      ImGui::End();
      return;
    }

    // Display asset info and XMP metadata at top (collapsible)
    renderAssetInfo();
    ui_xmp::renderMetadataPanel(m_model);

    if(ImGui::BeginTable("SceneGraphTable", 3, s_tableFlags))
    {
      ImGui::TableSetupScrollFreeze(1, 1);
      ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoHide);
      ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_WidthFixed, textBaseWidth * 8.0f);
      ImGui::TableSetupColumn(" ", ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_WidthFixed, textBaseWidth * 2.2f);
      ImGui::TableHeadersRow();

      for(size_t sceneID = 0; sceneID < m_model->scenes.size(); sceneID++)
      {
        tinygltf::Scene& scene = m_model->scenes[sceneID];
        ImGui::SetNextItemOpen(true);  // Scene is always open
        ImGui::PushID(static_cast<int>(sceneID));
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        if(ImGui::TreeNodeEx("Scene", s_treeNodeFlags, "%s", scene.name.c_str()))
        {
          ImGui::TableNextColumn();
          ImGui::Text("Scene %ld", sceneID);
          for(int node : scene.nodes)
          {
            renderNode(node);
          }
          ImGui::TreePop();
        }
        ImGui::PopID();
      }

      ImGui::EndTable();
    }
  }
  ImGui::End();
}

//--------------------------------------------------------------------------------------------------
// Display glTF asset information (version, generator, copyright)
void UiSceneGraph::renderAssetInfo()
{
  if(!m_model)
    return;

  const auto& asset = m_model->asset;

  if(ImGui::CollapsingHeader("Asset Info"))
  {
    if(!asset.version.empty())
      ImGui::Text("glTF Version: %s", asset.version.c_str());
    if(!asset.generator.empty())
      ImGui::TextWrapped("Generator: %s", asset.generator.c_str());
    if(!asset.copyright.empty())
      ImGui::TextWrapped("Copyright: %s", asset.copyright.c_str());
    if(!asset.minVersion.empty())
      ImGui::Text("Min Version: %s", asset.minVersion.c_str());

    ui_xmp::renderInfoButton(m_model, asset.extensions, "asset_xmp_popup");
  }
}

//--------------------------------------------------------------------------------------------------
// This function renders the details of the selected node, light, camera, or material
void UiSceneGraph::renderDetails(bool* showProperties)
{
  if(showProperties && !*showProperties)
    return;

  if(ImGui::Begin("Properties", showProperties))
  {
    if(m_model == nullptr)
    {
      ImGui::TextDisabled("No model loaded");
      ImGui::End();
      return;
    }

    // Handle light or camera selection
    if((m_selectType == eLight || m_selectType == eCamera) && m_selectedIndex >= 0)
    {
      // Show node properties first
      renderNodeDetails(m_selectedIndex);

      // Add a separator between node and light/camera properties
      ImGui::Separator();

      // Show light or camera specific properties
      if(m_selectType == eLight)
      {
        // Find the light index for the selected node
        const tinygltf::Node& node = m_model->nodes[m_selectedIndex];
        if(node.light >= 0)
        {
          renderLightDetails(node.light);
        }
      }
      else if(m_selectType == eCamera)
      {
        // Find the camera index for the selected node
        const tinygltf::Node& node = m_model->nodes[m_selectedIndex];
        if(node.camera >= 0)
        {
          renderCameraDetailsWithEvents(node.camera);
        }
      }
    }
    else
    {
      // Always show node properties if a node is selected
      if(m_selectedIndex >= 0)
      {
        renderNodeDetails(m_selectedIndex);

        // Add a separator between node and material properties
        ImGui::Separator();
      }

      // Show material properties if a material is selected
      if(m_selectedMaterialIndex >= 0)
      {
        // Show material selector if we have a node context
        if(m_selectedNodeForMaterial >= 0)
        {
          renderMaterialSelector(m_selectedNodeForMaterial);
        }

        renderMaterial(m_selectedMaterialIndex);
      }
      else if(m_selectedIndex >= 0)
      {
        // If no material is selected but we have a node, show a message
        ImGui::TextDisabled("No material selected for this node");
      }
      else
      {
        ImGui::TextDisabled("No selection");
      }
    }
  }
  ImGui::End();
}

//--------------------------------------------------------------------------------------------------
// This function is called when a node is selected
// It will open all the parents of the selected node and select the first primitive
//
void UiSceneGraph::selectNode(int nodeIndex)
{
  m_selectType    = eNode;
  m_selectedIndex = nodeIndex;

  // Look up the first RenderNode for this node (primitive index 0)
  int renderNodeIndex = -1;
  if(nodeIndex >= 0 && m_renderNodeLookup)
  {
    renderNodeIndex = m_renderNodeLookup(nodeIndex, 0);
  }
  m_selectedRenderNodeIndex = renderNodeIndex;
  m_selectedPrimitiveIndex  = (renderNodeIndex >= 0) ? 0 : -1;

  // Emit node selection event with the first RenderNode index
  if(m_eventCallback)
  {
    m_eventCallback({EventType::NodeSelected, nodeIndex, renderNodeIndex});
  }

  m_openNodes.clear();
  if(nodeIndex >= 0)
  {
    preprocessOpenNodes();
    // Auto-select the first available material for this node
    auto materials = getMaterialsForNode(nodeIndex);
    if(!materials.empty())
    {
      m_selectedMaterialIndex   = materials[0];
      m_selectedNodeForMaterial = nodeIndex;
    }
    else
    {
      m_selectedMaterialIndex   = -1;
      m_selectedNodeForMaterial = -1;
    }
  }
  else
  {
    m_selectedMaterialIndex   = -1;
    m_selectedNodeForMaterial = -1;
  }
  m_doScroll = true;
}

//--------------------------------------------------------------------------------------------------
// This function is called when a primitive is selected (via picking or UI)
// It stores the RenderNode index for proper silhouette/framing and selects the parent node
//
void UiSceneGraph::selectPrimitive(int renderNodeIndex, int nodeIndex, int primitiveIndex)
{
  m_selectType              = eNode;
  m_selectedRenderNodeIndex = renderNodeIndex;
  m_selectedPrimitiveIndex  = primitiveIndex;
  m_selectedIndex           = nodeIndex;

  // Emit primitive selection event with the RenderNode index
  if(m_eventCallback)
  {
    m_eventCallback({EventType::PrimitiveSelected, nodeIndex, renderNodeIndex});
  }

  m_openNodes.clear();
  if(nodeIndex >= 0)
  {
    preprocessOpenNodes();
    // Select the material for this specific primitive
    if(primitiveIndex >= 0 && m_model)
    {
      const tinygltf::Node& node = m_model->nodes[nodeIndex];
      if(node.mesh >= 0)
      {
        const tinygltf::Mesh& mesh = m_model->meshes[node.mesh];
        if(primitiveIndex < static_cast<int>(mesh.primitives.size()))
        {
          int materialID            = std::max(0, mesh.primitives[primitiveIndex].material);
          m_selectedMaterialIndex   = materialID;
          m_selectedNodeForMaterial = nodeIndex;
        }
      }
    }
  }
  else
  {
    m_selectedMaterialIndex   = -1;
    m_selectedNodeForMaterial = -1;
  }
  m_doScroll = true;
}

//--------------------------------------------------------------------------------------------------
// This function is called when a material is selected
// It will also select the node that contains this material if nodeIndex is provided
//
void UiSceneGraph::selectMaterial(int materialIndex, int nodeIndex)
{
  m_selectedMaterialIndex = materialIndex;
  if(nodeIndex >= 0)
  {
    m_selectedNodeForMaterial = nodeIndex;
    // Also select the node if it's not already selected
    if(m_selectedIndex != nodeIndex)
    {
      selectNode(nodeIndex);
    }
  }

  // Emit material selection event
  if(m_eventCallback)
  {
    m_eventCallback({EventType::MaterialSelected, materialIndex, -1});
  }
}

//--------------------------------------------------------------------------------------------------
// This is rendering the node and its children
// If it has the command to open the node, it will open it,
// when it finds the selected node, it will highlight it and will scroll to it. (done once)
//
void UiSceneGraph::renderNode(int nodeIndex)
{
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  const tinygltf::Node& node = m_model->nodes[nodeIndex];

  ImGuiTreeNodeFlags flags = s_treeNodeFlags;

  // Ensure the selected node is visible (open parent nodes in the path)
  // Also open the selected node itself if it contains the selected primitive
  if(m_openNodes.find(nodeIndex) != m_openNodes.end() || (m_selectedIndex == nodeIndex && m_selectedRenderNodeIndex >= 0))
  {
    ImGui::SetNextItemOpen(true);
  }

  // Highlight the selected node only if no primitive is selected under it
  if((m_selectType == eNode) && (m_selectedIndex == nodeIndex) && (m_selectedRenderNodeIndex < 0))
  {
    flags |= ImGuiTreeNodeFlags_Selected;
    if(m_doScroll)
    {
      ImGui::SetScrollHereY();
      m_doScroll = false;
    }
  }

  // Append "(invisible)" to the name if the node isn't visible
  KHR_node_visibility visibility = tinygltf::utils::getNodeVisibility(node);

  // Handle node selection
  bool nodeOpen = ImGui::TreeNodeEx((void*)(intptr_t)nodeIndex, flags, "%s", node.name.c_str());

  if(ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
  {
    // Use selectNode to ensure proper material selection
    if((m_selectType == eNode) && (m_selectedIndex == nodeIndex))
    {
      selectNode(-1);  // Deselect if clicking the same node
    }
    else
    {
      selectNode(nodeIndex);  // Select the new node
    }
  }

  ImGui::TableNextColumn();
  ImGui::Text("Node %d", nodeIndex);

  ImGui::TableNextColumn();
  if(!visibility.visible)
  {
    ImGui::Text("%s", ICON_MS_VISIBILITY_OFF);
  }
  // Show XMP info button if node has XMP metadata
  const std::string popupId = "node_xmp_" + std::to_string(nodeIndex);
  ui_xmp::renderInfoButton(m_model, node.extensions, popupId.c_str());

  // Render the mesh, children, light, and camera if the node is open
  if(nodeOpen)
  {
    if(node.mesh >= 0)
    {
      renderMesh(node.mesh);
    }

    if(node.light >= 0)
    {
      renderLight(node.light);
    }

    if(node.camera >= 0)
    {
      renderCamera(node.camera);
    }

    for(int child : node.children)
    {
      renderNode(child);
    }

    ImGui::TreePop();
  }
}


//--------------------------------------------------------------------------------------------------
// Convenience functions using the template
//
int UiSceneGraph::getNodeForMesh(int meshIndex)
{
  return getNodeForElement(meshIndex, m_meshToNodeMap, m_meshToNodeMapDirty, &tinygltf::Node::mesh);
}

int UiSceneGraph::getNodeForLight(int lightIndex)
{
  return getNodeForElement(lightIndex, m_lightToNodeMap, m_lightToNodeMapDirty, &tinygltf::Node::light);
}

int UiSceneGraph::getNodeForCamera(int cameraIndex)
{
  return getNodeForElement(cameraIndex, m_cameraToNodeMap, m_cameraToNodeMapDirty, &tinygltf::Node::camera);
}

void UiSceneGraph::renderMesh(int meshIndex)
{
  const tinygltf::Mesh& mesh = m_model->meshes[meshIndex];
  ImGui::TableNextRow();
  ImGui::TableNextColumn();

  // Get the node index that contains this mesh
  int nodeIndex = getNodeForMesh(meshIndex);

  // Create a selectable mesh item (only highlight if no primitive is selected)
  std::string meshLabel = "Mesh: " + mesh.name;
  bool meshSelected     = (m_selectedIndex == nodeIndex) && (m_selectType == eNode) && (m_selectedRenderNodeIndex < 0);
  if(ImGui::Selectable(meshLabel.c_str(), meshSelected))
  {
    // Select the parent node and its first material
    if(nodeIndex >= 0)
    {
      selectNode(nodeIndex);
    }
  }

  ImGui::TableNextColumn();
  ImGui::Text("Mesh %d", meshIndex);
  ImGui::TableNextColumn();
  // Show XMP info button if mesh has XMP metadata
  const std::string popupId = "mesh_xmp_" + std::to_string(meshIndex);
  ui_xmp::renderInfoButton(m_model, mesh.extensions, popupId.c_str());

  // Force open the Primitives tree if a primitive in this mesh is selected
  if(m_selectedIndex == nodeIndex && m_selectedRenderNodeIndex >= 0)
  {
    ImGui::SetNextItemOpen(true);
  }

  // Render primitives as a tree node
  if(ImGui::TreeNodeEx("Primitives", s_treeNodeFlags, "Primitives (%zu)", mesh.primitives.size()))
  {
    int primID{};
    for(const auto& primitive : mesh.primitives)
    {
      // Look up the RenderNode index for this primitive using the callback
      int renderNodeIndex = m_renderNodeLookup(nodeIndex, primID);
      renderPrimitive(primitive, primID++, nodeIndex, renderNodeIndex);
    }
    ImGui::TreePop();
  }
}

void UiSceneGraph::renderPrimitive(const tinygltf::Primitive& primitive, int primID, int nodeIndex, int renderNodeIndex)
{
  ImGui::TableNextRow();
  ImGui::TableNextColumn();

  const int         materialID = std::clamp(primitive.material, 0, static_cast<int>(m_model->materials.size() - 1));
  const std::string primName   = "Prim " + std::to_string(primID);
  const bool        isSelected = (m_selectedRenderNodeIndex == renderNodeIndex) && (renderNodeIndex >= 0);

  // Scroll to the selected primitive
  if(isSelected && m_doScroll)
  {
    ImGui::SetScrollHereY();
    m_doScroll = false;
  }

  if(ImGui::Selectable(primName.c_str(), isSelected))
  {
    // Toggle: deselect if clicking the same primitive, otherwise select
    if(isSelected)
    {
      selectPrimitive(-1, -1, -1);
    }
    else
    {
      selectPrimitive(renderNodeIndex, nodeIndex, primID);
    }
  }

  ImGui::TableNextColumn();
  ImGui::Text("Primitive");
  ImGui::TableNextColumn();
  if(materialID >= 0)
  {
    ImGui::Text("%s", ICON_MS_SHAPES);
  }
}

void UiSceneGraph::renderLight(int lightIndex)
{
  const tinygltf::Light& light = m_model->lights[lightIndex];
  ImGui::TableNextRow();
  ImGui::TableNextColumn();

  // Get the node index that contains this light
  int nodeIndex = getNodeForLight(lightIndex);

  if(ImGui::Selectable(light.name.c_str(), (m_selectedIndex == nodeIndex) && (m_selectType == eLight)))
  {
    if(nodeIndex >= 0)
    {
      m_selectType    = eLight;
      m_selectedIndex = nodeIndex;
      // Clear material selection when selecting a light
      m_selectedMaterialIndex   = -1;
      m_selectedNodeForMaterial = -1;
    }
  }
  ImGui::TableNextColumn();
  ImGui::Text("Light %d", lightIndex);
  ImGui::TableNextColumn();
  ImGui::Text("%s", ICON_MS_LIGHTBULB);
}

void UiSceneGraph::renderCamera(int cameraIndex)
{
  const tinygltf::Camera& camera = m_model->cameras[cameraIndex];
  ImGui::TableNextRow();
  ImGui::TableNextColumn();

  // Get the node index that contains this camera
  int nodeIndex = getNodeForCamera(cameraIndex);

  if(ImGui::Selectable(camera.name.c_str(), (m_selectedIndex == nodeIndex) && (m_selectType == eCamera)))
  {
    if(nodeIndex >= 0)
    {
      m_selectType    = eCamera;
      m_selectedIndex = nodeIndex;
      // Clear material selection when selecting a camera
      m_selectedMaterialIndex   = -1;
      m_selectedNodeForMaterial = -1;
    }
  }
  ImGui::TableNextColumn();
  ImGui::Text("Camera %d", cameraIndex);
  ImGui::TableNextColumn();
  ImGui::Text("%s", ICON_MS_PHOTO_CAMERA);
}
// Utility struct to handle material UI
struct MaterialUI
{
  glm::vec4 baseColorFactor;
  glm::vec3 emissiveFactor;
  int       alphaMode;

  static constexpr const char* alphaModes[] = {"OPAQUE", "MASK", "BLEND"};

  void toUI(tinygltf::Material& material)
  {
    baseColorFactor = glm::make_vec4(material.pbrMetallicRoughness.baseColorFactor.data());
    emissiveFactor  = glm::make_vec3(material.emissiveFactor.data());
    alphaMode       = material.alphaMode == "OPAQUE" ? 0 : material.alphaMode == "MASK" ? 1 : 2;
  }
  void fromUI(tinygltf::Material& material) const
  {
    material.pbrMetallicRoughness.baseColorFactor = {baseColorFactor.x, baseColorFactor.y, baseColorFactor.z,
                                                     baseColorFactor.w};
    material.emissiveFactor                       = {emissiveFactor.x, emissiveFactor.y, emissiveFactor.z};
    material.alphaMode                            = alphaModes[alphaMode];
  }
};

// Utility struct to handle material UI
struct LightUI
{
  glm::vec3 color;
  int       type;
  float     innerAngle;
  float     outerAngle;
  float     intensity;
  float     radius;

  static constexpr const char* lightType[] = {"point", "spot", "directional"};

  void toUI(const tinygltf::Light& light)
  {
    color      = shaderio::toSrgb(glm::make_vec3(light.color.data()));
    type       = light.type == "point" ? 0 : light.type == "spot" ? 1 : 2;
    intensity  = static_cast<float>(light.intensity);
    innerAngle = static_cast<float>(light.spot.innerConeAngle);
    outerAngle = static_cast<float>(light.spot.outerConeAngle);
    radius     = light.extras.Has("radius") ? float(light.extras.Get("radius").GetNumberAsDouble()) : 0.0f;
  }

  void fromUI(tinygltf::Light& light) const
  {
    glm::vec3 linearColor     = shaderio::toLinear(color);
    light.color               = {linearColor.x, linearColor.y, linearColor.z};
    light.type                = lightType[type];
    light.intensity           = intensity;
    light.spot.innerConeAngle = innerAngle;
    light.spot.outerConeAngle = outerAngle;
    if(!light.extras.IsObject())
    {
      light.extras = tinygltf::Value(tinygltf::Value::Object());
    }
    tinygltf::Value::Object extras = light.extras.Get<tinygltf::Value::Object>();
    extras["radius"]               = tinygltf::Value(radius);
    light.extras                   = tinygltf::Value(extras);
  }
};

static float logarithmicStep(float value)
{
  return std::max(0.1f * std::pow(10.0f, std::floor(std::log10(value))), 0.001f);
}

//--------------------------------------------------------------------------------------------------
// Node details is the transform of the node
// It will show the translation, rotation and scale
//
void UiSceneGraph::renderNodeDetails(int nodeIndex)
{
  tinygltf::Node&     node = m_model->nodes[nodeIndex];
  glm::vec3           translation, scale;
  glm::quat           rotation;
  KHR_node_visibility visibility;

  bool hasVisibility = tinygltf::utils::hasElementName(node.extensions, KHR_NODE_VISIBILITY_EXTENSION_NAME);
  if(hasVisibility)
  {
    visibility = tinygltf::utils::getNodeVisibility(node);
  }

  getNodeTransform(node, translation, rotation, scale);

  ImGui::Text("Node: %s", node.name.c_str());

  glm::vec3 euler = glm::degrees(glm::eulerAngles(rotation));

  if(PE::begin())
  {
    bool modif = false;
    modif |= PE::DragFloat3("Translation", glm::value_ptr(translation), 0.01f * m_bbox.radius());
    modif |= PE::DragFloat3("Rotation", glm::value_ptr(euler), 0.1f);
    modif |= PE::DragFloat3("Scale", glm::value_ptr(scale), 0.01f);
    if(modif)
    {
      m_changes.set(eNodeTransformDirty);
      node.translation = {translation.x, translation.y, translation.z};
      rotation         = glm::quat(glm::radians(euler));
      node.rotation    = {rotation.x, rotation.y, rotation.z, rotation.w};
      node.scale       = {scale.x, scale.y, scale.z};
      node.matrix.clear();  // Clear the matrix, has its been converted to translation, rotation and scale
    }
    if(hasVisibility)
    {
      if(PE::Checkbox("Visible", &visibility.visible))
      {
        tinygltf::utils::setNodeVisibility(node, visibility);
        m_changes.set(eNodeVisibleDirty);
      }
    }
    else if(ImGui::SmallButton("Add Visibility"))
    {
      tinygltf::utils::setNodeVisibility(node, {});
    }
    PE::end();
  }
}

//--------------------------------------------------------------------------------------------------
// Returning the translation, rotation and scale of a node
// If the node has a matrix, it will decompose it
void UiSceneGraph::getNodeTransform(const tinygltf::Node& node, glm::vec3& translation, glm::quat& rotation, glm::vec3& scale)
{
  translation = glm::vec3(0.0f);
  rotation    = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
  scale       = glm::vec3(1.0f);

  if(node.matrix.size() == 16)
  {
    glm::mat4 matrix = glm::make_mat4(node.matrix.data());
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(matrix, scale, rotation, translation, skew, perspective);

    return;
  }
  if(node.translation.size() == 3)
  {
    translation = glm::make_vec3(node.translation.data());
  }
  if(node.rotation.size() == 4)
  {
    rotation = glm::make_quat(node.rotation.data());
  }
  if(node.scale.size() == 3)
  {
    scale = glm::make_vec3(node.scale.data());
  }
}

//--------------------------------------------------------------------------------------------------
// Rendering the material properties
// - Base color
// - Metallic
// - Roughness
// - Emissive
void UiSceneGraph::renderMaterial(int materialIndex)
{
  tinygltf::Material& material = m_model->materials[materialIndex];

  ImGui::Text("Material: %s", material.name.c_str());
  // Show XMP info button if material has XMP metadata
  const std::string popupId = "mat_xmp_" + std::to_string(materialIndex);
  ui_xmp::renderInfoButton(m_model, material.extensions, popupId.c_str());

  // Example: Basic PBR properties
  if(PE::begin())
  {
    bool       modif      = false;
    MaterialUI materialUI = {};
    materialUI.toUI(material);
    modif |= PE::ColorEdit4("Base Color", glm::value_ptr(materialUI.baseColorFactor));
    modif |= PE::DragScalar("Metallic", ImGuiDataType_Double, &material.pbrMetallicRoughness.metallicFactor, 0.01f,
                            &f64_zero, &f64_one);
    modif |= PE::DragScalar("Roughness", ImGuiDataType_Double, &material.pbrMetallicRoughness.roughnessFactor, 0.01f,
                            &f64_zero, &f64_one);
    modif |= PE::ColorEdit3("Emissive", glm::value_ptr(materialUI.emissiveFactor));
    modif |= PE::DragScalar("Alpha Cutoff", ImGuiDataType_Double, &material.alphaCutoff, 0.01f, &f64_zero, &f64_one);

    if(PE::Combo("Alpha Mode", &materialUI.alphaMode, MaterialUI::alphaModes, IM_ARRAYSIZE(MaterialUI::alphaModes)))
    {
      m_changes.set(eMaterialFlagDirty);
      modif = true;
    }

    if(PE::Checkbox("Double Sided", &material.doubleSided))
    {
      m_changes.set(eMaterialFlagDirty);
      modif = true;
    }

    if(modif)
    {
      materialUI.fromUI(material);
      m_changes.set(eMaterialDirty);
      modif = false;
    }

    // Extensions
    materialAnisotropy(material);
    materialClearcoat(material);
    materialDiffuseTransmission(material);
    materialDispersion(material);
    materialEmissiveStrength(material);
    materialIor(material);
    materialIridescence(material);
    materialSheen(material);
    materialSpecular(material);
    materialTransmission(material);
    materialUnlit(material);
    materialVolume(material);
    materialVolumeScatter(material);

    PE::end();
  }
}

void UiSceneGraph::addButton(tinygltf::Material& material, const char* extensionName, std::function<void()> addCallback)
{
  ImGui::TableNextColumn();
  ImGui::PushID(extensionName);
  if(ImGui::Button("Add"))
  {
    addCallback();
    m_changes.set(eMaterialDirty);
  }
  ImGui::PopID();
}

void UiSceneGraph::removeButton(tinygltf::Material& material, const char* extensionName)
{
  ImGui::TableNextColumn();
  ImGui::PushID(extensionName);
  if(ImGui::Button("Remove"))
  {
    material.extensions.erase(extensionName);
    m_changes.set(eMaterialDirty);
  }
  ImGui::PopID();
}


void UiSceneGraph::materialDiffuseTransmission(tinygltf::Material& material)
{
  bool hasMaterialDiffuseTransmission =
      tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_DIFFUSE_TRANSMISSION_EXTENSION_NAME);
  if(PE::treeNode("Diffuse Transmission"))
  {
    if(hasMaterialDiffuseTransmission)
    {
      removeButton(material, KHR_MATERIALS_DIFFUSE_TRANSMISSION_EXTENSION_NAME);
      KHR_materials_diffuse_transmission diffuseTransmission = tinygltf::utils::getDiffuseTransmission(material);
      bool                               modif               = false;
      modif |= PE::DragFloat("Factor", &diffuseTransmission.diffuseTransmissionFactor, 0.01f, 0.0f, 1.0f);
      modif |= PE::ColorEdit3("Color", glm::value_ptr(diffuseTransmission.diffuseTransmissionColor));
      if(modif)
      {
        tinygltf::utils::setDiffuseTransmission(material, diffuseTransmission);
        m_changes.set(eMaterialDirty);
      }
    }
    PE::treePop();
  }
  if(!hasMaterialDiffuseTransmission)
  {
    addButton(material, KHR_MATERIALS_DIFFUSE_TRANSMISSION_EXTENSION_NAME,
              [&]() { tinygltf::utils::setDiffuseTransmission(material, {}); });
  }
}


void UiSceneGraph::materialDispersion(tinygltf::Material& material)
{
  bool hasMaterialDispersion = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_DISPERSION_EXTENSION_NAME);
  if(PE::treeNode("Dispersion"))
  {
    if(hasMaterialDispersion)
    {
      removeButton(material, KHR_MATERIALS_DISPERSION_EXTENSION_NAME);
      KHR_materials_dispersion dispersion = tinygltf::utils::getDispersion(material);
      bool                     modif      = false;
      modif |= PE::DragFloat("Dispersion Factor", &dispersion.dispersion, 0.01f, 0.0f, 10.0f);
      if(modif)
      {
        tinygltf::utils::setDispersion(material, dispersion);
        m_changes.set(eMaterialDirty);
      }
    }
    PE::treePop();
  }
  if(!hasMaterialDispersion)
  {
    addButton(material, KHR_MATERIALS_DISPERSION_EXTENSION_NAME, [&]() { tinygltf::utils::setDispersion(material, {}); });
  }
}

void UiSceneGraph::materialIridescence(tinygltf::Material& material)
{
  bool hasMaterialIridescence = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_IRIDESCENCE_EXTENSION_NAME);
  if(PE::treeNode("Iridescence"))
  {
    if(hasMaterialIridescence)
    {
      removeButton(material, KHR_MATERIALS_IRIDESCENCE_EXTENSION_NAME);
      KHR_materials_iridescence iridescence = tinygltf::utils::getIridescence(material);
      bool                      modif       = false;
      modif |= PE::DragFloat("Iridescence Factor", &iridescence.iridescenceFactor, 0.01f, 0.0f, 10.0f);
      modif |= PE::DragFloat("Iridescence Ior", &iridescence.iridescenceIor, 0.01f, 0.0f, 10.0f);
      modif |= PE::DragFloat("Thickness Min", &iridescence.iridescenceThicknessMinimum, 0.01f, 0.0f, 1000.0f, "%.3f nm");
      modif |= PE::DragFloat("Thickness Max", &iridescence.iridescenceThicknessMaximum, 0.01f, 0.0f, 1000.0f, "%.3f nm");
      if(modif)
      {
        tinygltf::utils::setIridescence(material, iridescence);
        m_changes.set(eMaterialDirty);
      }
    }
    PE::treePop();
  }
  if(!hasMaterialIridescence)
  {
    addButton(material, KHR_MATERIALS_IRIDESCENCE_EXTENSION_NAME,
              [&]() { tinygltf::utils::setIridescence(material, {}); });
  }
}

void UiSceneGraph::materialAnisotropy(tinygltf::Material& material)
{
  bool hasMaterialAnisotropy = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME);
  if(PE::treeNode("Anisotropy"))
  {
    if(hasMaterialAnisotropy)
    {
      removeButton(material, KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME);
      KHR_materials_anisotropy anisotropy = tinygltf::utils::getAnisotropy(material);
      bool                     modif      = false;
      modif |= PE::DragFloat("Anisotropy Strength", &anisotropy.anisotropyStrength, 0.01f, 0.0f, 1.0f);
      modif |= PE::DragFloat("Anisotropy Rotation", &anisotropy.anisotropyRotation, 0.01f, -glm::pi<float>(), glm::pi<float>());
      if(modif)
      {
        tinygltf::utils::setAnisotropy(material, anisotropy);
        m_changes.set(eMaterialDirty);
      }
    }
    PE::treePop();
  }
  if(!hasMaterialAnisotropy)
  {
    addButton(material, KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME, [&]() { tinygltf::utils::setAnisotropy(material, {}); });
  }
}

void UiSceneGraph::materialVolume(tinygltf::Material& material)
{
  bool hasMaterialVolume = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_VOLUME_EXTENSION_NAME);
  if(PE::treeNode("Volume"))
  {
    if(hasMaterialVolume)
    {
      removeButton(material, KHR_MATERIALS_VOLUME_EXTENSION_NAME);
      KHR_materials_volume volume          = tinygltf::utils::getVolume(material);
      bool                 modif           = false;
      float                thicknessFactor = volume.thicknessFactor;
      modif |= PE::DragFloat("Thickness", &volume.thicknessFactor, 0.01f, 0.0f, 1.0f);
      modif |= PE::ColorEdit3("Attenuation Color", glm::value_ptr(volume.attenuationColor));
      bool isInfinite = (volume.attenuationDistance >= FLT_MAX);
      if(PE::Checkbox("Infinite Attenuation", &isInfinite, "No light absorption (infinite distance)"))
      {
        volume.attenuationDistance = isInfinite ? FLT_MAX : 1.0f;  // Default to 1.0 when toggling off infinite
        modif                      = true;
      }
      if(!isInfinite)
      {
        modif |= PE::DragFloat("Attenuation Distance", &volume.attenuationDistance,
                               logarithmicStep(volume.attenuationDistance), 0.0, FLT_MAX, "%.3f", ImGuiSliderFlags_None,
                               "Distance light travels before absorption (smaller = more opaque)");
      }
      if(modif)
      {
        tinygltf::utils::setVolume(material, volume);
        m_changes.set(eMaterialDirty);
        if(thicknessFactor == 0.0f && volume.thicknessFactor != 0.0f)
          m_changes.set(eMaterialFlagDirty);
      }
    }

    PE::treePop();
  }
  if(!hasMaterialVolume)
  {
    addButton(material, KHR_MATERIALS_VOLUME_EXTENSION_NAME, [&]() { tinygltf::utils::setVolume(material, {}); });
  }
}

void UiSceneGraph::materialVolumeScatter(tinygltf::Material& material)
{
  bool hasMaterialVolumeScatter = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME);
  if(PE::treeNode("Volume Scatter"))
  {
    if(hasMaterialVolumeScatter)
    {
      removeButton(material, KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME);
      KHR_materials_volume_scatter volumeScatter = tinygltf::utils::getVolumeScatter(material);
      bool                         modif         = false;
      modif |= PE::ColorEdit3("Multiscatter Color", glm::value_ptr(volumeScatter.multiscatterColor));
      modif |= PE::SliderFloat("Scatter Anisotropy", &volumeScatter.scatterAnisotropy, -1.0f, 1.0f);
      if(modif)
      {
        tinygltf::utils::setVolumeScatter(material, volumeScatter);
        m_changes.set(eMaterialDirty);
      }
    }
    PE::treePop();
  }
  if(!hasMaterialVolumeScatter)
  {
    addButton(material, KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME,
              [&]() { tinygltf::utils::setVolumeScatter(material, {}); });
  }
}

void UiSceneGraph::materialSpecular(tinygltf::Material& material)
{
  bool hasMaterialSpecular = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_SPECULAR_EXTENSION_NAME);
  if(PE::treeNode("Specular"))
  {
    if(hasMaterialSpecular)
    {
      removeButton(material, KHR_MATERIALS_SPECULAR_EXTENSION_NAME);
      KHR_materials_specular specular = tinygltf::utils::getSpecular(material);
      bool                   modif    = false;
      modif |= PE::ColorEdit3("Specular Color", glm::value_ptr(specular.specularColorFactor));
      modif |= PE::DragFloat("Specular Factor", &specular.specularFactor, 0.01f, 0.0f, 1.0f);
      if(modif)
      {
        tinygltf::utils::setSpecular(material, specular);
        m_changes.set(eMaterialDirty);
      }
    }
    PE::treePop();
  }
  if(!hasMaterialSpecular)
  {
    addButton(material, KHR_MATERIALS_SPECULAR_EXTENSION_NAME, [&]() { tinygltf::utils::setSpecular(material, {}); });
  }
}

void UiSceneGraph::materialIor(tinygltf::Material& material)
{
  bool hasMaterialIor = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_IOR_EXTENSION_NAME);
  if(PE::treeNode("IOR"))
  {
    if(hasMaterialIor)
    {
      removeButton(material, KHR_MATERIALS_IOR_EXTENSION_NAME);
      KHR_materials_ior ior   = tinygltf::utils::getIor(material);
      bool              modif = false;
      modif |= PE::DragFloat("IOR", &ior.ior, 0.01f, 0.0f, 10.0f);
      if(modif)
      {
        tinygltf::utils::setIor(material, ior);
        m_changes.set(eMaterialDirty);
      }
    }
    PE::treePop();
  }
  if(!hasMaterialIor)
  {
    addButton(material, KHR_MATERIALS_IOR_EXTENSION_NAME, [&]() { tinygltf::utils::setIor(material, {}); });
  }
}

void UiSceneGraph::materialTransmission(tinygltf::Material& material)
{
  bool hasMaterialTransmission = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME);
  if(PE::treeNode("Transmission"))
  {
    if(hasMaterialTransmission)
    {
      removeButton(material, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME);
      KHR_materials_transmission transmission       = tinygltf::utils::getTransmission(material);
      bool                       modif              = false;
      float                      transmissionFactor = transmission.factor;
      modif |= PE::DragFloat("Transmission Factor", &transmission.factor, 0.01f, 0.0f, 1.0f);
      if(modif)
      {
        tinygltf::utils::setTransmission(material, transmission);
        m_changes.set(eMaterialDirty);
        if(transmissionFactor == 0.0f && transmission.factor != 0.0f)
          m_changes.set(eMaterialFlagDirty);
      }
    }
    PE::treePop();
  }
  if(!hasMaterialTransmission)
  {
    addButton(material, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME,
              [&]() { tinygltf::utils::setTransmission(material, {}); });
  }
}

void UiSceneGraph::materialSheen(tinygltf::Material& material)
{
  bool hasMaterialSheen = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_SHEEN_EXTENSION_NAME);
  if(PE::treeNode("Sheen"))
  {
    if(hasMaterialSheen)
    {
      removeButton(material, KHR_MATERIALS_SHEEN_EXTENSION_NAME);
      KHR_materials_sheen sheen = tinygltf::utils::getSheen(material);
      bool                modif = false;
      modif |= PE::ColorEdit3("Sheen Color", glm::value_ptr(sheen.sheenColorFactor));
      modif |= PE::DragFloat("Sheen Roughness", &sheen.sheenRoughnessFactor, 0.01f, 0.0f, 1.0f);
      if(modif)
      {
        tinygltf::utils::setSheen(material, sheen);
        m_changes.set(eMaterialDirty);
      }
    }
    PE::treePop();
  }
  if(!hasMaterialSheen)
  {
    addButton(material, KHR_MATERIALS_SHEEN_EXTENSION_NAME, [&]() { tinygltf::utils::setSheen(material, {}); });
  }
}

void UiSceneGraph::materialUnlit(tinygltf::Material& material)
{
  bool hasMaterialUnlit = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_UNLIT_EXTENSION_NAME);
  if(PE::treeNode("Unlit"))
  {
    if(hasMaterialUnlit)
    {
      ImGui::TextWrapped("Material is unlit (no lighting applied)");
      removeButton(material, KHR_MATERIALS_UNLIT_EXTENSION_NAME);
    }
    PE::treePop();
  }
  if(!hasMaterialUnlit)
  {
    addButton(material, KHR_MATERIALS_UNLIT_EXTENSION_NAME, [&]() { tinygltf::utils::setUnlit(material, {}); });
  }
}

void UiSceneGraph::materialClearcoat(tinygltf::Material& material)
{
  bool hasMaterialClearcoat = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME);
  if(PE::treeNode("Clearcoat"))
  {
    if(hasMaterialClearcoat)
    {
      removeButton(material, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME);
      KHR_materials_clearcoat clearcoat = tinygltf::utils::getClearcoat(material);
      bool                    modif     = false;
      modif |= PE::DragFloat("Clearcoat Factor", &clearcoat.factor, 0.01f, 0.0f, 1.0f);
      modif |= PE::DragFloat("Clearcoat Roughness", &clearcoat.roughnessFactor, 0.01f, 0.0f, 1.0f);
      if(modif)
      {
        tinygltf::utils::setClearcoat(material, clearcoat);
        m_changes.set(eMaterialDirty);
      }
    }
    PE::treePop();
  }
  if(!hasMaterialClearcoat)
  {
    addButton(material, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME, [&]() { tinygltf::utils::setClearcoat(material, {}); });
  }
}

void UiSceneGraph::materialEmissiveStrength(tinygltf::Material& material)
{
  bool hasEmissiveStrength = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_EMISSIVE_STRENGTH_EXTENSION_NAME);
  if(PE::treeNode("Emissive Strength"))
  {
    if(hasEmissiveStrength)
    {
      removeButton(material, KHR_MATERIALS_EMISSIVE_STRENGTH_EXTENSION_NAME);
      KHR_materials_emissive_strength strenght = tinygltf::utils::getEmissiveStrength(material);
      if(PE::DragFloat("Emissive Strength", &strenght.emissiveStrength, logarithmicStep(strenght.emissiveStrength), 0.0f, FLT_MAX))
      {
        tinygltf::utils::setEmissiveStrength(material, strenght);
        m_changes.set(eMaterialDirty);
      }
    }
    PE::treePop();
  }
  if(!hasEmissiveStrength)
  {
    addButton(material, KHR_MATERIALS_EMISSIVE_STRENGTH_EXTENSION_NAME,
              [&]() { tinygltf::utils::setEmissiveStrength(material, {}); });
  }
}

// This function is called when a node is selected
// It will open all the parents of the selected node
void UiSceneGraph::preprocessOpenNodes()
{
  m_openNodes.clear();
  if((m_selectedIndex < 0) || (m_selectType != eNode))
  {
    return;
  }
  for(int rootIndex : m_model->scenes[0].nodes)  // Assuming sceneNodes contains root node indices
  {
    if(markOpenNodes(rootIndex, m_selectedIndex, m_openNodes))
    {
      break;
    }
  }
}

// Recursive function to mark all the nodes that are in the path to the target node
bool UiSceneGraph::markOpenNodes(int nodeIndex, int targetNodeIndex, std::unordered_set<int>& openNodes)
{
  if(nodeIndex == targetNodeIndex)
  {
    return true;
  }

  const tinygltf::Node& node = m_model->nodes[nodeIndex];
  for(int child : node.children)
  {
    if(markOpenNodes(child, targetNodeIndex, openNodes))
    {
      openNodes.insert(nodeIndex);  // Mark the current node as open if any child path leads to the target
      return true;
    }
  }
  return false;
}

//--------------------------------------------------------------------------------------------------
// Helper function to get all materials used by a node
//
std::vector<int> UiSceneGraph::getMaterialsForNode(int nodeIndex) const
{
  std::vector<int> materials;
  if(nodeIndex < 0 || nodeIndex >= static_cast<int>(m_model->nodes.size()))
    return materials;

  const tinygltf::Node& node = m_model->nodes[nodeIndex];
  if(node.mesh < 0 || node.mesh >= static_cast<int>(m_model->meshes.size()))
    return materials;

  const tinygltf::Mesh& mesh = m_model->meshes[node.mesh];
  for(const auto& primitive : mesh.primitives)
  {
    if(primitive.material >= 0)
    {
      materials.push_back(primitive.material);
    }
  }

  // Remove duplicates while preserving order
  std::vector<int> uniqueMaterials;
  for(int material : materials)
  {
    if(std::find(uniqueMaterials.begin(), uniqueMaterials.end(), material) == uniqueMaterials.end())
    {
      uniqueMaterials.push_back(material);
    }
  }

  return uniqueMaterials;
}

//--------------------------------------------------------------------------------------------------
// Helper function to get primitive information for a node
//
std::vector<std::pair<int, std::string>> UiSceneGraph::getPrimitiveInfoForNode(int nodeIndex) const
{
  std::vector<std::pair<int, std::string>> primitiveInfo;
  if(nodeIndex < 0 || nodeIndex >= static_cast<int>(m_model->nodes.size()))
    return primitiveInfo;

  const tinygltf::Node& node = m_model->nodes[nodeIndex];
  if(node.mesh < 0 || node.mesh >= static_cast<int>(m_model->meshes.size()))
    return primitiveInfo;

  const tinygltf::Mesh& mesh = m_model->meshes[node.mesh];
  for(size_t i = 0; i < mesh.primitives.size(); ++i)
  {
    const auto& primitive = mesh.primitives[i];
    std::string primName  = "Primitive " + std::to_string(i);
    if(primitive.material >= 0 && primitive.material < static_cast<int>(m_model->materials.size()))
    {
      const std::string& materialName = m_model->materials[primitive.material].name;
      if(!materialName.empty())
      {
        primName += " (" + materialName + ")";
      }
    }
    primitiveInfo.emplace_back(primitive.material, primName);
  }

  return primitiveInfo;
}

//--------------------------------------------------------------------------------------------------
// Render material selector for a node
//
void UiSceneGraph::renderMaterialSelector(int nodeIndex)
{
  auto primitiveInfo = getPrimitiveInfoForNode(nodeIndex);
  if(primitiveInfo.empty())
    return;

  ImGui::Text("Material Selection:");

  // Create a combo box for material selection
  std::vector<const char*> items;
  std::vector<int>         materialIndices;

  for(const auto& [materialIndex, primName] : primitiveInfo)
  {
    if(materialIndex >= 0)
    {
      items.push_back(primName.c_str());
      materialIndices.push_back(materialIndex);
    }
  }

  if(items.empty())
    return;

  // Find current selection index
  int currentSelection = 0;
  for(size_t i = 0; i < materialIndices.size(); ++i)
  {
    if(materialIndices[i] == m_selectedMaterialIndex)
    {
      currentSelection = static_cast<int>(i);
      break;
    }
  }

  if(ImGui::Combo("##MaterialSelector", &currentSelection, items.data(), static_cast<int>(items.size())))
  {
    if(currentSelection >= 0 && currentSelection < static_cast<int>(materialIndices.size()))
    {
      m_selectedMaterialIndex = materialIndices[currentSelection];
    }
  }

  ImGui::Separator();
}

void UiSceneGraph::renderLightDetails(int lightIndex)
{
  tinygltf::Light& light = m_model->lights[lightIndex];

  ImGui::Text("Light: %s", light.name.c_str());

  if(PE::begin())
  {
    bool    modif   = false;
    LightUI lightUI = {};
    lightUI.toUI(light);

    modif |= PE::Combo("Type", &lightUI.type, LightUI::lightType, IM_ARRAYSIZE(LightUI::lightType));
    modif |= PE::ColorEdit3("Color", glm::value_ptr(lightUI.color));
    modif |= PE::SliderAngle("Intensity", &lightUI.intensity, 0.0f, 1000000.f, "%.3f", ImGuiSliderFlags_Logarithmic);
    modif |= PE::SliderAngle("Inner Cone Angle", &lightUI.innerAngle, 0.0f, 180.f);
    lightUI.outerAngle = std::max(lightUI.innerAngle, lightUI.outerAngle);  // Outer angle should be larger than inner angle
    modif |= PE::SliderAngle("Outer Cone Angle", &lightUI.outerAngle, 0.0f, 180.f);
    lightUI.innerAngle = std::min(lightUI.innerAngle, lightUI.outerAngle);  // Inner angle should be smaller than outer angle
    modif |= PE::SliderAngle("Radius", &lightUI.radius, 0.0f, 1000000.f, "%.3f", ImGuiSliderFlags_Logarithmic);

    lightUI.fromUI(light);

    if(modif)
    {
      m_changes.set(eLightDirty);
    }

    PE::end();
  }
}

//--------------------------------------------------------------------------------------------------
// Renders the details of a camera
//
// This function displays the properties of a camera in a collapsible section.
// It supports both perspective and orthographic cameras.
//
// Parameters:
//   cameraIndex - The index of the camera to render
//
void UiSceneGraph::renderCameraDetails(int                              cameraIndex,
                                       const CameraApplyCallback&       applyCameraCallback,
                                       const CameraSetFromViewCallback& setCameraFromViewCallback)
{
  tinygltf::Camera& camera = m_model->cameras[cameraIndex];

  ImGui::Text("Camera: %s", camera.name.c_str());

  if(PE::begin())
  {
    bool modif = false;

    if(camera.type == "perspective")
    {
      ImGui::Text("Type: Perspective");
      ImGui::Separator();

      tinygltf::PerspectiveCamera& persp = camera.perspective;

      // Convert FOV from radians (GLTF) to degrees (UI)
      double fovDegrees = glm::degrees(persp.yfov);
      modif |= PE::DragScalar("Y FOV (degrees)", ImGuiDataType_Double, &fovDegrees, 0.1f, &f64_one, &f64_179);
      // Convert back to radians when saving
      if(modif)
      {
        persp.yfov = glm::radians(fovDegrees);
      }

      modif |= PE::DragScalar("Z Near", ImGuiDataType_Double, &persp.znear, 0.01f, &f64_001, &f64_1000);
      modif |= PE::DragScalar("Z Far", ImGuiDataType_Double, &persp.zfar, 1.0f, &persp.znear, &f64_10000);
    }
    else if(camera.type == "orthographic")
    {
      ImGui::Text("Type: Orthographic");
      ImGui::Separator();

      tinygltf::OrthographicCamera& ortho = camera.orthographic;
      modif |= PE::DragScalar("X Magnification", ImGuiDataType_Double, &ortho.xmag, 0.1f, &f64_01, &f64_100);
      modif |= PE::DragScalar("Y Magnification", ImGuiDataType_Double, &ortho.ymag, 0.1f, &f64_01, &f64_100);
      modif |= PE::DragScalar("Z Near", ImGuiDataType_Double, &ortho.znear, 0.01f, &f64_neg1000, &f64_1000);
      double ortho_zfar_min = ortho.znear + 1.0;
      modif |= PE::DragScalar("Z Far", ImGuiDataType_Double, &ortho.zfar, 0.01f, &ortho_zfar_min, &f64_10000);
    }

    if(modif)
    {
      m_changes.set(eCameraDirty);
    }

    // Add buttons to sync between GLTF camera and current view
    ImGui::Separator();
    ImGui::Text("Camera Sync:");

    if(ImGui::Button("Apply to Current View"))
    {
      if(applyCameraCallback)
      {
        applyCameraCallback(cameraIndex);
      }
      m_changes.set(eCameraApplyToView);
    }

    ImGui::SameLine();
    if(ImGui::Button("Set from Current View"))
    {
      if(setCameraFromViewCallback)
      {
        setCameraFromViewCallback(cameraIndex);
      }
      m_changes.set(eCameraDirty);
    }

    PE::end();
  }
}

void UiSceneGraph::renderCameraDetailsWithEvents(int cameraIndex)
{
  tinygltf::Camera& camera = m_model->cameras[cameraIndex];

  ImGui::Text("Camera: %s", camera.name.c_str());

  if(PE::begin())
  {
    bool modif = false;

    if(camera.type == "perspective")
    {
      ImGui::Text("Type: Perspective");
      ImGui::Separator();

      tinygltf::PerspectiveCamera& persp = camera.perspective;

      // Convert FOV from radians (GLTF) to degrees (UI)
      double fovDegrees = glm::degrees(persp.yfov);
      modif |= PE::DragScalar("Y FOV (degrees)", ImGuiDataType_Double, &fovDegrees, 0.1f, &f64_one, &f64_179);
      // Convert back to radians when saving
      if(modif)
      {
        persp.yfov = glm::radians(fovDegrees);
      }

      modif |= PE::DragScalar("Z Near", ImGuiDataType_Double, &persp.znear, 0.01f, &f64_001, &f64_1000);
      modif |= PE::DragScalar("Z Far", ImGuiDataType_Double, &persp.zfar, 1.0f, &persp.znear, &f64_10000);
    }
    else if(camera.type == "orthographic")
    {
      ImGui::Text("Type: Orthographic");
      ImGui::Separator();

      tinygltf::OrthographicCamera& ortho = camera.orthographic;
      modif |= PE::DragScalar("X Magnification", ImGuiDataType_Double, &ortho.xmag, 0.1f, &f64_01, &f64_100);
      modif |= PE::DragScalar("Y Magnification", ImGuiDataType_Double, &ortho.ymag, 0.1f, &f64_01, &f64_100);
      modif |= PE::DragScalar("Z Near", ImGuiDataType_Double, &ortho.znear, 0.01f, &f64_neg1000, &f64_1000);
      double ortho_zfar_min = ortho.znear + 1.0;
      modif |= PE::DragScalar("Z Far", ImGuiDataType_Double, &ortho.zfar, 0.01f, &ortho_zfar_min, &f64_10000);
    }

    if(modif)
    {
      m_changes.set(eCameraDirty);
    }

    // Add buttons to sync between GLTF camera and current view
    ImGui::Separator();
    ImGui::Text("Camera Sync:");

    if(ImGui::Button("Apply to Current View"))
    {
      if(m_eventCallback)
      {
        m_eventCallback({EventType::CameraApply, cameraIndex, -1});
      }
      m_changes.set(eCameraApplyToView);
    }

    ImGui::SameLine();
    if(ImGui::Button("Set from Current View"))
    {
      if(m_eventCallback)
      {
        m_eventCallback({EventType::CameraSetFromView, cameraIndex, -1});
      }
      m_changes.set(eCameraDirty);
    }

    PE::end();
  }
}

//--------------------------------------------------------------------------------------------------
// Builds a lookup cache mapping element indices to their containing node indices
//
// This function creates a reverse lookup table that maps from element indices
// (mesh, light, or camera) to the node index that contains them.
//
// Parameters:
//   cache     - The cache map to populate (elementIndex -> nodeIndex)
//   dirtyFlag - Flag indicating if the cache needs to be rebuilt
//   member    - Pointer to member function that returns the element index (e.g., &tinygltf::Node::mesh)
//
void UiSceneGraph::buildCache(std::unordered_map<int, int>& cache, bool& dirtyFlag, int(tinygltf::Node::* member)) const
{
  if(!dirtyFlag || m_model == nullptr)
    return;

  cache.clear();

  for(size_t i = 0; i < m_model->nodes.size(); ++i)
  {
    const tinygltf::Node& node         = m_model->nodes[i];
    int                   elementIndex = node.*member;
    if(elementIndex >= 0)
    {
      cache[elementIndex] = static_cast<int>(i);
    }
  }

  dirtyFlag = false;
}

//--------------------------------------------------------------------------------------------------
// Gets the node index that contains a specific element using cached lookup
//
// Parameters:
//   elementIndex - The index of the element to find (mesh, light, or camera index)
//   cache        - The cache map for fast element->node lookup
//   dirtyFlag    - Flag indicating if the cache needs to be rebuilt
//   member       - Pointer to member function that returns the element index
//
// Returns:
//   The node index that contains the specified element, or -1 if not found
//
int UiSceneGraph::getNodeForElement(int elementIndex, std::unordered_map<int, int>& cache, bool& dirtyFlag, int(tinygltf::Node::* member)) const
{
  buildCache(cache, dirtyFlag, member);

  auto it = cache.find(elementIndex);
  return (it != cache.end()) ? it->second : -1;
}
