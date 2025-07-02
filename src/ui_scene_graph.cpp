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

namespace shaderio {
using namespace glm;
#include <nvshaders/tonemap_functions.h.slang>
}  // namespace shaderio


namespace PE = nvgui::PropertyEditor;

static ImGuiTreeNodeFlags s_treeNodeFlags = ImGuiTreeNodeFlags_SpanAllColumns | ImGuiTreeNodeFlags_SpanFullWidth
                                            | ImGuiTreeNodeFlags_SpanTextWidth | ImGuiTreeNodeFlags_OpenOnArrow
                                            | ImGuiTreeNodeFlags_OpenOnDoubleClick;


//--------------------------------------------------------------------------------------------------
// Entry point for rendering the scene graph
// Loop over all scenes
// - Loop over all nodes in the scene
//Following, in the second part, is the details:
// - Display the node details (transform)
//   OR Display the material details
//
void UiSceneGraph::render()
{
  renderSceneGraph();
  renderDetails();
}

void UiSceneGraph::renderSceneGraph()
{
  static const float     textBaseWidth = ImGui::CalcTextSize("A").x;
  static ImGuiTableFlags s_tableFlags =
      ImGuiTableFlags_ScrollY | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV;

  if(ImGui::Begin("Scene Graph"))
  {
    if(m_model == nullptr)
    {
      ImGui::End();
      return;
    }

    if(ImGui::BeginTable("SceneGraphTable", 3, s_tableFlags))
    {
      ImGui::TableSetupScrollFreeze(1, 1);
      ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_NoHide);
      ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_WidthFixed, textBaseWidth * 8.0f);
      ImGui::TableSetupColumn("-", ImGuiTableColumnFlags_NoHide | ImGuiTableColumnFlags_WidthFixed, textBaseWidth * 1.0f);
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

void UiSceneGraph::renderDetails()
{
  if(ImGui::Begin("Properties"))
  {
    if(m_selectedIndex < 0 || m_model == nullptr)
    {
      ImGui::TextDisabled("No selection");
      ImGui::End();
      return;
    }
    switch(m_selectType)
    {
      case UiSceneGraph::eNode:
        renderNodeDetails(m_selectedIndex);
        break;
      case UiSceneGraph::eMaterial:
        renderMaterial(m_selectedIndex);
        break;
      case UiSceneGraph::eLight:
        renderLightDetails(m_selectedIndex);
        break;
      default:
        break;
    }
  }
  ImGui::End();
}

//--------------------------------------------------------------------------------------------------
// This function is called when a node is selected
// It will open all the parents of the selected node
//
void UiSceneGraph::selectNode(int nodeIndex)
{
  m_selectType    = eNode;
  m_selectedIndex = nodeIndex;
  m_openNodes.clear();
  if(nodeIndex >= 0)
    preprocessOpenNodes();
  m_doScroll = true;
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

  // Ensure the selected node is visible
  if(m_openNodes.find(nodeIndex) != m_openNodes.end())
  {
    ImGui::SetNextItemOpen(true);
  }

  // Highlight the selected node
  if((m_selectType == eNode) && (m_selectedIndex == nodeIndex))
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
    m_selectedIndex = ((m_selectType == eNode) && (m_selectedIndex == nodeIndex)) ? -1 : nodeIndex;
    m_selectType    = eNode;
  }

  ImGui::TableNextColumn();
  ImGui::Text("Node %d", nodeIndex);

  if(!visibility.visible)
  {
    ImGui::TableNextColumn();
    ImGui::PushFont(nvgui::getIconicFont());
    ImGui::Text("%s", nvgui::icon_ban);
    ImGui::PopFont();
  }
  else
  {
    ImGui::TableNextColumn();
  }

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

void UiSceneGraph::renderMesh(int meshIndex)
{
  const tinygltf::Mesh& mesh = m_model->meshes[meshIndex];
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  bool meshOpen = ImGui::TreeNodeEx("Mesh", s_treeNodeFlags, "%s", mesh.name.c_str());
  ImGui::TableNextColumn();
  ImGui::Text("Mesh %d", meshIndex);
  ImGui::TableNextColumn();

  if(meshOpen)
  {
    int primID{};
    for(const auto& primitive : mesh.primitives)
    {
      renderPrimitive(primitive, primID++);
    }
    ImGui::TreePop();
  }
}

void UiSceneGraph::renderPrimitive(const tinygltf::Primitive& primitive, int primID)
{
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  const int         materialID = std::clamp(primitive.material, 0, static_cast<int>(m_model->materials.size() - 1));
  const std::string primName   = "Prim " + std::to_string(primID);
  if(ImGui::Selectable(primName.c_str(), (m_selectedIndex == materialID) && (m_selectType == eMaterial)))
  {
    m_selectType    = eMaterial;
    m_selectedIndex = materialID;
  }
  ImGui::TableNextColumn();
  ImGui::Text("Primitive");
  ImGui::TableNextColumn();
}

void UiSceneGraph::renderLight(int lightIndex)
{
  const tinygltf::Light& light = m_model->lights[lightIndex];
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  if(ImGui::Selectable(light.name.c_str(), (m_selectedIndex == lightIndex) && (m_selectType == eLight)))
  {
    m_selectType    = eLight;
    m_selectedIndex = lightIndex;
  }
  ImGui::TableNextColumn();
  ImGui::Text("Light %d", lightIndex);
  ImGui::TableNextColumn();
}

void UiSceneGraph::renderCamera(int cameraIndex)
{
  const tinygltf::Camera& camera = m_model->cameras[cameraIndex];
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::Text("%s", camera.name.c_str());
  ImGui::TableNextColumn();
  ImGui::Text("Camera %d", cameraIndex);
  ImGui::TableNextColumn();
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
// Rendering the material properties
// - Base color
// - Metallic
// - Roughness
// - Emissive
void UiSceneGraph::renderMaterial(int materialIndex)
{
  tinygltf::Material& material = m_model->materials[materialIndex];

  ImGui::Text("Material: %s", material.name.c_str());

  // Example: Basic PBR properties
  if(PE::begin())
  {
    const double f64_zero = 0., f64_one = 1.;

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
    materialVolume(material);

    PE::end();
  }
}


void UiSceneGraph::materialDiffuseTransmission(tinygltf::Material& material)
{
  bool hasMaterialDiffuseTransmission =
      tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_DIFFUSE_TRANSMISSION_EXTENSION_NAME);
  if(PE::treeNode("Diffuse Transmission"))
  {
    if(hasMaterialDiffuseTransmission)
    {
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
    ImGui::TableNextColumn();
    if(ImGui::Button("Add##Add_DiffuseTransmission"))
    {
      tinygltf::utils::setDiffuseTransmission(material, {});
    }
  }
}


void UiSceneGraph::materialDispersion(tinygltf::Material& material)
{
  bool hasMaterialDispersion = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_DISPERSION_EXTENSION_NAME);
  if(PE::treeNode("Dispersion"))
  {
    if(hasMaterialDispersion)
    {
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
    ImGui::TableNextColumn();
    if(ImGui::Button("Add##Add_Dispersion"))
    {
      tinygltf::utils::setDispersion(material, {});
    }
  }
}

void UiSceneGraph::materialIridescence(tinygltf::Material& material)
{
  bool hasMaterialIridescence = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_IRIDESCENCE_EXTENSION_NAME);
  if(PE::treeNode("Iridescence"))
  {
    if(hasMaterialIridescence)
    {
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
    ImGui::TableNextColumn();
    if(ImGui::Button("Add##Add_9"))
    {
      tinygltf::utils::setIridescence(material, {});
    }
  }
}

void UiSceneGraph::materialAnisotropy(tinygltf::Material& material)
{
  bool hasMaterialAnisotropy = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME);
  if(PE::treeNode("Anisotropy"))
  {
    if(hasMaterialAnisotropy)
    {
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
    ImGui::TableNextColumn();
    if(ImGui::Button("Add##Add_8"))
    {
      tinygltf::utils::setAnisotropy(material, {});
    }
  }
}

void UiSceneGraph::materialVolume(tinygltf::Material& material)
{
  bool hasMaterialVolume = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_VOLUME_EXTENSION_NAME);
  if(PE::treeNode("Volume"))
  {
    if(hasMaterialVolume)
    {
      KHR_materials_volume volume          = tinygltf::utils::getVolume(material);
      bool                 modif           = false;
      float                thicknessFactor = volume.thicknessFactor;
      modif |= PE::DragFloat("Thickness", &volume.thicknessFactor, 0.01f, 0.0f, 1.0f);
      modif |= PE::ColorEdit3("Attenuation Color", glm::value_ptr(volume.attenuationColor));
      modif |= PE::DragFloat("Attenuation", &volume.attenuationDistance);
      volume.attenuationDistance = std::max(0.0F, volume.attenuationDistance);
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
    ImGui::TableNextColumn();
    if(ImGui::Button("Add##Add_7"))
    {
      tinygltf::utils::setVolume(material, {});
    }
  }
}

void UiSceneGraph::materialSpecular(tinygltf::Material& material)
{
  bool hasMaterialSpecular = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_SPECULAR_EXTENSION_NAME);
  if(PE::treeNode("Specular"))
  {
    if(hasMaterialSpecular)
    {
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
    ImGui::TableNextColumn();
    if(ImGui::Button("Add##Add_6"))
    {
      tinygltf::utils::setSpecular(material, {});
    }
  }
}

void UiSceneGraph::materialIor(tinygltf::Material& material)
{
  bool hasMaterialIor = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_IOR_EXTENSION_NAME);
  if(PE::treeNode("IOR"))
  {
    if(hasMaterialIor)
    {
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
    ImGui::TableNextColumn();
    if(ImGui::Button("Add##Add_5"))
    {
      tinygltf::utils::setIor(material, {});
    }
  }
}

void UiSceneGraph::materialTransmission(tinygltf::Material& material)
{
  bool hasMaterialTransmission = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME);
  if(PE::treeNode("Transmission"))
  {
    if(hasMaterialTransmission)
    {
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
    ImGui::TableNextColumn();
    if(ImGui::Button("Add##Add_4"))
    {
      tinygltf::utils::setTransmission(material, {});
    }
  }
}

void UiSceneGraph::materialSheen(tinygltf::Material& material)
{
  bool hasMaterialSheen = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_SHEEN_EXTENSION_NAME);
  if(PE::treeNode("Sheen"))
  {
    if(hasMaterialSheen)
    {
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
    ImGui::TableNextColumn();
    if(ImGui::Button("Add##Add_3"))
    {
      tinygltf::utils::setSheen(material, {});
    }
  }
}

void UiSceneGraph::materialClearcoat(tinygltf::Material& material)
{
  bool hasMaterialClearcoat = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME);
  if(PE::treeNode("Clearcoat"))
  {
    if(hasMaterialClearcoat)
    {
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
    ImGui::TableNextColumn();
    if(ImGui::Button("Add##Add_2"))
    {
      tinygltf::utils::setClearcoat(material, {});
    }
  }
}

void UiSceneGraph::materialEmissiveStrength(tinygltf::Material& material)
{
  bool hasEmissiveStrength = tinygltf::utils::hasElementName(material.extensions, KHR_MATERIALS_EMISSIVE_STRENGTH_EXTENSION_NAME);
  if(PE::treeNode("Emissive Strength"))
  {
    if(hasEmissiveStrength)
    {

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
    ImGui::TableNextColumn();
    if(ImGui::Button("Add##Add_1"))
    {
      tinygltf::utils::setEmissiveStrength(material, {});
    }
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
