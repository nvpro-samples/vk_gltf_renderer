/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Main renderer UI and application loop. Orchestrates scene loading,
// environment map setup, camera control, renderer switching (path tracer /
// rasterizer), picking, gizmo interaction, animation playback, and
// the top-level ImGui layout with viewport, menus, and side panels.
//

#include <algorithm>
#include <filesystem>
#include <fmt/format.h>
#include <cmath>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

#include <nvvk/check_error.hpp>
#include <nvvk/helpers.hpp>
#include <nvutils/bounding_box.hpp>
#include <nvgui/tonemapper.hpp>
#include <nvgui/file_dialog.hpp>
#include <nvgui/axis.hpp>
#include "tinygltf_utils.hpp"

#include <nvapp/elem_camera.hpp>

#include "renderer.hpp"
#include "gltf_create_tangent.hpp"
#include "scoped_banner.hpp"
#include "ui_mouse_state.hpp"
#include "version.hpp"

void GltfRenderer::mouseClickedInViewport()
{
  static UiMouseState s_mouseClickState;  // Mouse click state

  // The mouse click state is only updated when the "Viewport" window is hovered.
  if(!ImGui::IsWindowHovered(ImGuiFocusedFlags_RootWindow))
    return;

  // Do not pick if the scene is not valid or the renderer is busy
  if(!m_resources.getScene() || !m_resources.getScene()->valid() || m_busy.isBusy())
  {
    return;
  }

  s_mouseClickState.update();

  // Process gizmo input before ray picking (gizmo gets priority)
  if(m_resources.settings.showGizmo && m_visualHelpers.transform.isAttached())
  {
    ImVec2 mousePos  = ImGui::GetMousePos();
    ImVec2 cursorPos = ImGui::GetCursorScreenPos();
    ImVec2 avail     = ImGui::GetContentRegionAvail();

    glm::vec2 localMouse(mousePos.x - cursorPos.x, mousePos.y - cursorPos.y);
    glm::vec2 mouseDelta(ImGui::GetIO().MouseDelta.x, ImGui::GetIO().MouseDelta.y);
    glm::vec2 viewportSize(avail.x, avail.y);

    bool mouseDown     = ImGui::IsMouseDown(ImGuiMouseButton_Left);
    bool mousePressed  = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    bool mouseReleased = ImGui::IsMouseReleased(ImGuiMouseButton_Left);

    bool gizmoActive = m_visualHelpers.transform.processInput(localMouse, mouseDelta, mouseDown, mousePressed,
                                                              mouseReleased, m_cameraManip->getViewMatrix(),
                                                              m_cameraManip->getPerspectiveMatrix(), viewportSize);

    if(gizmoActive || m_visualHelpers.transform.isDragging())
      return;
  }

  // If double-clicking in the "Viewport", shoot a ray to the scene under the mouse.
  // If the ray hit something, set the camera center to the hit position.
  if(s_mouseClickState.isMouseClicked(ImGuiMouseButton_Left))
  {
    nvutils::ScopedTimer st("RayPicker");
    VkCommandBuffer      cmd = m_app->createTempCmdBuffer();
    // Convert screen coordinates to normalized viewport coordinates [0,1]
    ImVec2 mousePos      = ImGui::GetMousePos();
    ImVec2 cursorPos     = ImGui::GetCursorScreenPos();
    ImVec2 avail         = ImGui::GetContentRegionAvail();
    ImVec2 localMousePos = ImVec2((mousePos.x - cursorPos.x) / avail.x, (mousePos.y - cursorPos.y) / avail.y);

    m_rayPicker.run(cmd, {.modelViewInv   = glm::inverse(m_cameraManip->getViewMatrix()),
                          .perspectiveInv = glm::inverse(m_cameraManip->getPerspectiveMatrix()),
                          .isOrthographic = (m_cameraManip->getProjectionType() == nvutils::CameraManipulator::Orthographic) ? 1 : 0,
                          .pickPos = {localMousePos.x, localMousePos.y},
                          .tlas    = m_resources.sceneRtx.topLevelAS()});
    m_app->submitAndWaitTempCmdBuffer(cmd);
    nvvk::RayPicker::PickResult pickResult = m_rayPicker.getResult();

    // Set or de-select the selected object (primitive-level selection)
    // pickResult.instanceID is the TLAS instance index = render node index
    if(s_mouseClickState.isMouseSingleClicked(ImGuiMouseButton_Left))
    {
      if(pickResult.instanceID > -1)
      {
        const int        renderNodeIdx = pickResult.instanceID;
        nvvkgltf::Scene* scene         = m_resources.getScene();
        const auto&      renderNodes   = scene->getRenderNodes();
        const bool       valid         = renderNodeIdx >= 0 && renderNodeIdx < static_cast<int>(renderNodes.size());

        if(!valid)
        {
          m_resources.selectedRenderNodes.clear();
          m_sceneSelection.clearSelection();
        }
        else if(m_resources.selectedRenderNodes.size() == 1 && m_resources.selectedRenderNodes.count(renderNodeIdx))
        {
          m_resources.selectedRenderNodes.clear();
          m_sceneSelection.clearSelection();
        }
        else
        {
          const auto& renderNode = renderNodes[renderNodeIdx];
          int         nodeID     = renderNode.refNodeID;
          int         primIndex  = scene->getPrimitiveIndexForRenderNode(renderNodeIdx);
          int         meshID     = scene->getModel().nodes[nodeID].mesh;

          m_resources.selectedRenderNodes = {renderNodeIdx};
          m_sceneSelection.selectPrimitive(renderNodeIdx, nodeID, primIndex, meshID);
          m_sceneBrowser.focusOnSelection();
        }
      }
      else
      {
        m_resources.selectedRenderNodes.clear();
        m_sceneSelection.clearSelection();
      }
    }

    // Environment was picked (no hit)
    if(pickResult.instanceID < 0)
      return;

    glm::vec3 worldPos = pickResult.worldRayOrigin + pickResult.worldRayDirection * pickResult.hitT;
    if(s_mouseClickState.isMouseDoubleClicked(ImGuiMouseButton_Left))
    {
      // Set the camera CENTER to the hit position
      glm::vec3 eye, center, up;
      m_cameraManip->getLookat(eye, center, up);
      m_cameraManip->setLookat(eye, worldPos, up, false);  // Nice with CameraManip.updateAnim();
      m_cameraManip->setSpeed(pickResult.hitT);            // Re-adjust speed based on the new distance
    }

    {
      const int        renderNodeIdx = pickResult.instanceID;
      nvvkgltf::Scene* scene         = m_resources.getScene();
      const auto&      renderNodes   = scene->getRenderNodes();
      if(renderNodeIdx >= 0 && renderNodeIdx < static_cast<int>(renderNodes.size()))
      {
        const nvvkgltf::RenderNode& renderNode = renderNodes[renderNodeIdx];
        const tinygltf::Node&       node       = scene->getModel().nodes[renderNode.refNodeID];
        LOGI("Node Name: %s\n", node.name.c_str());
        LOGI(" - GLTF: NodeID: %d, MeshID: %d, TriangleId: %d\n", renderNode.refNodeID, node.mesh, pickResult.primitiveID);
        LOGI(" - Render: renderNode: %d, RenderPrim: %d\n", renderNodeIdx, pickResult.instanceCustomIndex);
        LOGI("{%3.2f, %3.2f, %3.2f}, Dist: %3.2f\n", worldPos.x, worldPos.y, worldPos.z, pickResult.hitT);
      }
    }
  }
}

nvutils::Bbox GltfRenderer::getRenderNodeBbox(int renderNodeIndex)
{
  nvutils::Bbox    worldBbox({-1, -1, -1}, {1, 1, 1});
  nvvkgltf::Scene* scene = m_resources.getScene();
  if(renderNodeIndex < 0 || !scene || !scene->valid())
    return worldBbox;

  const auto& renderNodes = scene->getRenderNodes();
  if(renderNodeIndex >= static_cast<int>(renderNodes.size()))
    return worldBbox;

  const nvvkgltf::RenderNode&      renderNode      = renderNodes[renderNodeIndex];
  const nvvkgltf::RenderPrimitive& renderPrimitive = scene->getRenderPrimitive(renderNode.renderPrimID);
  const tinygltf::Model&           model           = scene->getModel();
  const tinygltf::Accessor&        accessor = model.accessors[renderPrimitive.pPrimitive->attributes.at("POSITION")];

  glm::vec3 minValues = {-1.f, -1.f, -1.f};
  glm::vec3 maxValues = {1.f, 1.f, 1.f};
  if(!accessor.minValues.empty())
    minValues = glm::vec3(accessor.minValues[0], accessor.minValues[1], accessor.minValues[2]);
  if(!accessor.maxValues.empty())
    maxValues = glm::vec3(accessor.maxValues[0], accessor.maxValues[1], accessor.maxValues[2]);
  nvutils::Bbox objBbox(minValues, maxValues);
  worldBbox = objBbox.transform(renderNode.worldMatrix);

  return worldBbox;
}

nvutils::Bbox GltfRenderer::getRenderNodesBbox(const std::unordered_set<int>& renderNodeIndices)
{
  nvutils::Bbox unionBbox;
  for(int idx : renderNodeIndices)
    unionBbox.insert(getRenderNodeBbox(idx));
  if(unionBbox.isEmpty())
    return {};
  return unionBbox;
}

void GltfRenderer::windowTitle()
{
  static float dirty_timer = 0.0F;
  dirty_timer += ImGui::GetIO().DeltaTime;
  if(dirty_timer > 1.0F)  // Refresh every seconds
  {
    const VkExtent2D&     size = m_app->getViewportSize();
    std::filesystem::path filename;
    if(m_resources.getScene())
      filename = m_resources.getScene()->getFilename().filename();
    if(filename.empty())
    {
      filename = "No Scene";
    }
    std::string text = fmt::format("{} {} - {}x{} | {:.0f} FPS / {:.3f}ms | Frame {}", nvutils::utf8FromPath(filename),
                                   APP_VERSION_STRING, size.width, size.height, ImGui::GetIO().Framerate,
                                   1000.F / ImGui::GetIO().Framerate, m_resources.frameCount);

    glfwSetWindowTitle(m_app->getWindowHandle(), text.c_str());
    dirty_timer = 0;
  }
}

// Helper function to load HDR files
void GltfRenderer::loadHdrFileDialog()
{
  std::filesystem::path filename =
      nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load HDR Environment", "HDR Image|*.hdr", m_lastHdrDirectory);
  if(!filename.empty())
  {
    onFileDrop(filename.c_str());
  }
}

void GltfRenderer::renderUI()
{
  static int frameCount = 0;

  namespace PE = nvgui::PropertyEditor;
  {  // Setting menu
    bool changed{false};

    if(m_resources.settings.showCameraWindow)
    {
      if(ImGui::Begin("Camera", &m_resources.settings.showCameraWindow))
      {
        nvgui::tooltip("Press F1 to toggle this window");
        nvgui::CameraWidget(m_cameraManip);
      }
      ImGui::End();  // End Camera
    }

    // Scene Browser system: translate selection events into Resources for silhouette and fit-object
    {
      m_sceneSelection.setEventCallback([&](const SceneSelection::Event& event) {
        switch(event.type)
        {
          case SceneSelection::EventType::CameraApply:
            applyGltfCamera(event.data);
            break;
          case SceneSelection::EventType::CameraSetFromView:
            setGltfCameraFromView(event.data);
            break;
          case SceneSelection::EventType::NodeSelected: {
            m_resources.selectedRenderNodes.clear();
            nvvkgltf::Scene* scn = m_resources.getScene();
            if(scn && scn->valid())
            {
              const int        nodeIdx = event.data;
              std::vector<int> renderNodeIDs;
              scn->getRenderNodeRegistry().getAllRenderNodesForNodeRecursive(
                  nodeIdx,
                  [scn](int nid) {
                    const auto& node = scn->getModel().nodes[static_cast<size_t>(nid)];
                    return node.children;
                  },
                  renderNodeIDs);
              for(int id : renderNodeIDs)
                m_resources.selectedRenderNodes.insert(id);
            }
            break;
          }
          case SceneSelection::EventType::PrimitiveSelected:
            m_resources.selectedRenderNodes = {event.renderNodeIndex};
            break;
          case SceneSelection::EventType::MaterialSelected:
            // Material selection event
            break;
        }
      });

      m_sceneBrowser.render(&m_resources.settings.showSceneBrowserWindow, m_busy.isBusy());
      m_inspector.render(&m_resources.settings.showInspectorWindow, m_busy.isBusy());
    }

    if(m_resources.settings.showSettingsWindow)
    {
      if(frameCount < 2)  // This is a hack to make the settings window focus on the first frame
      {
        ImGui::SetNextWindowFocus();
        frameCount++;
      }

      if(ImGui::Begin("Settings", &m_resources.settings.showSettingsWindow))
      {
        nvgui::tooltip("Press F3 to toggle this window");
        // Add renderer selection at the top level of the Settings panel for better visibility
        if(PE::begin())
        {
          static const char* rendererItems[] = {"Path Tracer", "Rasterizer"};
          int                currentItem     = static_cast<int>(m_resources.settings.renderSystem);
          if(PE::Combo("Active Renderer", &currentItem, rendererItems, IM_ARRAYSIZE(rendererItems)))
          {
            m_resources.settings.renderSystem = static_cast<RenderingMode>(currentItem);
            changed                           = true;  // Reset frame counter when switching renderers
          }
          changed |= PE::Combo("Debug Method", (int32_t*)(&m_resources.settings.debugMethod),
                               "None\0BaseColor\0Metallic\0Roughness\0Normal\0Tangent\0Bitangent\0Emissive\0Opacity\0TexCoord0\0TexCoord1\0\0");
          PE::end();
          if(m_resources.settings.renderSystem == RenderingMode::ePathtracer)
          {
            if(ImGui::CollapsingHeader("Path Tracer", ImGuiTreeNodeFlags_DefaultOpen))
            {
              changed |= m_pathTracer.onUIRender(m_resources);
            }
          }
          else  // Rasterizer
          {
            if(ImGui::CollapsingHeader("Rasterizer", ImGuiTreeNodeFlags_DefaultOpen))
            {
              changed |= m_rasterizer.onUIRender(m_resources);
            }
          }
        }
        ImGui::Separator();

        // Multiple scenes (glTF scenes within one file)
        if(m_resources.getScene() && m_resources.getScene()->getModel().scenes.size() > 1)
        {
          if(ImGui::CollapsingHeader("Multiple Scenes"))
          {
            ImGui::PushID("Scenes");
            for(size_t i = 0; i < m_resources.getScene()->getModel().scenes.size(); i++)
            {
              if(ImGui::RadioButton(m_resources.getScene()->getModel().scenes[i].name.c_str(),
                                    m_resources.getScene()->getCurrentScene() == i))
              {
                m_resources.getScene()->setCurrentScene(int(i));
                // SYNC NOTE: User-initiated scene switch — wait before full GPU resource rebuild.
                NVVK_CHECK(vkQueueWaitIdle(m_app->getQueue(0).queue));
                createVulkanScene();
                updateTextures();
                changed = true;
              }
            }
            ImGui::PopID();
          }
        }
      }
      ImGui::End();  // End Settings

      if(changed)
        resetFrame();
    }
  }


  // Show the rendered tonemapped image in the viewport
  {  // Rendering Viewport
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");

    // Camera smooth animation runs always
    m_cameraManip->updateAnim();

    // Camera orbit/pan/zoom only when gizmo is not capturing input
    bool gizmoCapturedInput = m_visualHelpers.transform.isDragging();
    if(!gizmoCapturedInput)
    {
      nvapp::ElementCamera::updateCamera(m_cameraManip, ImGui::GetCurrentWindow());
    }

    // Handle mouse clicks and gizmo interaction in the viewport
    GltfRenderer::mouseClickedInViewport();

    // Viewport-scoped keyboard shortcuts (only when hovered or focused)
    if(ImGui::IsWindowHovered(ImGuiHoveredFlags_RootWindow) || ImGui::IsWindowFocused())
    {
      if(ImGui::IsKeyPressed(ImGuiKey_G, false))
        m_resources.settings.showGrid = !m_resources.settings.showGrid;
      if(ImGui::IsKeyPressed(ImGuiKey_T, false))
        m_resources.settings.showGizmo = !m_resources.settings.showGizmo;
    }

    // Display the G-Buffer tonemapped image
    ImGui::Image(ImTextureID(m_resources.gBuffers.getDescriptorSet(Resources::eImgTonemapped)), ImGui::GetContentRegionAvail());

    // Adding Axis at the bottom left corner of the viewport
    if(m_resources.settings.showAxis)
    {
      nvgui::Axis(m_cameraManip->getViewMatrix(), 25.f);
    }

    ImGui::End();
    ImGui::PopStyleVar();
  }


  // Show the busy window if the busy flag is set
  if(m_busy.isBusy())
  {
    m_busy.show();
  }

  // Display memory statistics window
  renderMemoryStatistics();

  // Display environment, tonemapper, and statistics windows
  renderEnvironmentWindow();
  renderTonemapperWindow();
  renderStatisticsWindow();

#ifndef NDEBUG
  if(m_resources.settings.showGridStyleWindow)
  {
    if(ImGui::Begin("Grid Style", &m_resources.settings.showGridStyleWindow))
    {
      GridHelperVk::onDebugUI(m_visualHelpers.grid.style());
    }
    ImGui::End();
  }
  if(m_resources.settings.showGizmoStyleWindow)
  {
    if(ImGui::Begin("Gizmo Style", &m_resources.settings.showGizmoStyleWindow))
    {
      TransformHelperVk::onDebugUI(m_visualHelpers.transform.style());
    }
    ImGui::End();
  }
#endif
}

void GltfRenderer::renderFileMenu(bool                   validScene,
                                  bool&                  newScene,
                                  bool&                  openFile,
                                  bool&                  mergeFile,
                                  bool&                  loadHdrFile,
                                  bool&                  saveFile,
                                  bool&                  saveScreenFile,
                                  bool&                  saveImageFile,
                                  bool&                  closeApp,
                                  std::filesystem::path& sceneToLoadFilename,
                                  std::filesystem::path& sceneToMergeFilename)
{
  if(!ImGui::BeginMenu("File"))
    return;
  newScene = ImGui::MenuItem(ICON_MS_FILTER_NONE " New Scene", "Ctrl+N");
  openFile |= ImGui::MenuItem(ICON_MS_FILE_OPEN " Open", "Ctrl+O");
  ImGui::BeginDisabled(!validScene);
  mergeFile |= ImGui::MenuItem(ICON_MS_ADD " Import/Merge Scene...");
  ImGui::EndDisabled();
  loadHdrFile |= ImGui::MenuItem(ICON_MS_IMAGE " Load HDR Environment", "Ctrl+Shift+O");
  if(ImGui::BeginMenu(ICON_MS_HISTORY " Open Recent"))
  {
    for(const auto& file : m_recentFiles)
    {
      if(ImGui::MenuItem(file.string().c_str()))
        sceneToLoadFilename = file;
    }
    ImGui::EndMenu();
  }
  ImGui::BeginDisabled(!validScene);
  saveFile |= ImGui::MenuItem(ICON_MS_FILE_SAVE " Save As", "Ctrl+S");
  ImGui::EndDisabled();
  ImGui::Separator();
  saveImageFile |= ImGui::MenuItem(ICON_MS_IMAGE " Save Image", "Ctrl+Shift+S");
  saveScreenFile |= ImGui::MenuItem(ICON_MS_DESKTOP_WINDOWS " Save Screen", "Ctrl+Alt+Shift+S");
  ImGui::Separator();
  closeApp |= ImGui::MenuItem(ICON_MS_POWER_SETTINGS_NEW " Exit", "Ctrl+Q");
  ImGui::EndMenu();
}

void GltfRenderer::renderEditMenu(bool validScene)
{
  if(!ImGui::BeginMenu("Edit"))
    return;

  ImGui::BeginDisabled(!validScene || !m_undoStack.canUndo());
  std::string undoLabel = m_undoStack.canUndo() ? ICON_MS_UNDO " Undo " + m_undoStack.undoDescription() : ICON_MS_UNDO " Undo";
  if(ImGui::MenuItem(undoLabel.c_str(), "Ctrl+Z"))
  {
    m_undoStack.undo();
    onUndoRedo();
  }
  ImGui::EndDisabled();

  ImGui::BeginDisabled(!validScene || !m_undoStack.canRedo());
  std::string redoLabel = m_undoStack.canRedo() ? ICON_MS_REDO " Redo " + m_undoStack.redoDescription() : ICON_MS_REDO " Redo";
  if(ImGui::MenuItem(redoLabel.c_str(), "Ctrl+Y"))
  {
    m_undoStack.redo();
    onUndoRedo();
  }
  ImGui::EndDisabled();

  ImGui::EndMenu();
}

void GltfRenderer::renderViewMenu(bool validScene, bool& fitScene, bool& fitObject, bool& toggleVsync)
{
  if(!ImGui::BeginMenu("View"))
    return;
  ImGui::BeginDisabled(!validScene);
  fitScene |= ImGui::MenuItem(ICON_MS_ZOOM_OUT " Fit Scene", "Ctrl+Shift+F");
  ImGui::BeginDisabled(m_resources.selectedRenderNodes.empty());
  fitObject |= ImGui::MenuItem(ICON_MS_ZOOM_IN " Fit Object", "Ctrl+F");
  ImGui::EndDisabled();
  ImGui::EndDisabled();
  ImGui::Separator();
  ImGui::MenuItem(ICON_MS_BOTTOM_PANEL_OPEN " V-Sync", "Ctrl+Shift+V", &toggleVsync);
  ImGui::MenuItem(ICON_MS_VIEW_IN_AR " 3D-Axis", nullptr, &m_resources.settings.showAxis);
  ImGui::Separator();
  ImGui::MenuItem(ICON_MS_GRID_ON " Grid", "G", &m_resources.settings.showGrid);
  ImGui::MenuItem(ICON_MS_3D_ROTATION " Gizmo", "T", &m_resources.settings.showGizmo);
  ImGui::EndMenu();
}

void GltfRenderer::renderWindowsMenu()
{
  struct WindowToggleInfo
  {
    const char* icon;
    const char* name;
    const char* shortcut;
    bool*       visible;
  };
  static const std::array<WindowToggleInfo, 6> toggles = {{
      {ICON_MS_PHOTO_CAMERA, "Camera", "F1", &m_resources.settings.showCameraWindow},
      {ICON_MS_ACCOUNT_TREE, "Scene Browser", "F2", &m_resources.settings.showSceneBrowserWindow},
      {ICON_MS_SETTINGS, "Settings", "F3", &m_resources.settings.showSettingsWindow},
      {ICON_MS_LIST_ALT, "Inspector", "F4", &m_resources.settings.showInspectorWindow},
      {ICON_MS_PUBLIC, "Environment", "F5", &m_resources.settings.showEnvironmentWindow},
      {ICON_MS_TONALITY, "Tonemapper", "F6", &m_resources.settings.showTonemapperWindow},
  }};
  if(!ImGui::BeginMenu("Windows"))
    return;
  for(const auto& t : toggles)
  {
    std::string label = std::string(t.icon) + " " + t.name;
    ImGui::MenuItem(label.c_str(), t.shortcut, t.visible);
  }
  ImGui::MenuItem(ICON_MS_ANALYTICS " Statistics", nullptr, &m_resources.settings.showStatisticsWindow);
  ImGui::Separator();
  ImGui::MenuItem(ICON_MS_MONITORING " Memory Usage", nullptr, &m_resources.settings.showMemStats);
  ImGui::EndMenu();
}

void GltfRenderer::renderToolsMenu(bool validScene, bool& reloadShader, bool& compactScene)
{
  if(!ImGui::BeginMenu("Tools"))
    return;
  reloadShader |= ImGui::MenuItem(ICON_MS_REFRESH " Reload Shaders", "Ctrl+Shift+R");
  ImGui::Separator();
  ImGui::BeginDisabled(!validScene);
  if(ImGui::MenuItem(ICON_MS_BUILD " Recreate Tangents - Simple"))
  {
    SCOPED_BANNER("Recreate Tangents - Simple");
    recomputeTangents(m_resources.getScene()->getModel(), true, false);
    m_resources.dirtyFlags.set(DirtyFlags::eDirtyTangents);
  }
  ImGui::SetItemTooltip("This recreate tangents using UV gradient method");
  if(ImGui::MenuItem(ICON_MS_BUILD " Recreate Tangents - MikkTSpace"))
  {
    bool buffersChanged = false;
    {
      SCOPED_BANNER("Recreate Tangents - MikkTSpace");
      buffersChanged = recomputeTangents(m_resources.getScene()->getModel(), true, true);
    }
    if(buffersChanged)
      rebuildSceneFromModel();
    else
      m_resources.dirtyFlags.set(DirtyFlags::eDirtyTangents);
  }
  ImGui::SetItemTooltip("Recreate tangents using MikkTSpace (may split vertices at UV seams)");
  ImGui::Separator();
  compactScene |= ImGui::MenuItem(ICON_MS_COMPRESS " Compact Scene", "Ctrl+K");
  ImGui::SetItemTooltip(
      "Remove orphaned resources (meshes, materials, textures, images, geometry).\n"
      "Use after deleting nodes or importing/merging scenes.");
  ImGui::EndDisabled();
  ImGui::EndMenu();
}

void GltfRenderer::renderDebugMenu()
{
#ifndef NDEBUG
  if(ImGui::BeginMenu("Debug"))
  {
    ImGui::MenuItem("Grid Style", nullptr, &m_resources.settings.showGridStyleWindow);
    ImGui::MenuItem("Gizmo Style", nullptr, &m_resources.settings.showGizmoStyleWindow);
    ImGui::EndMenu();
  }
#endif
}

void GltfRenderer::renderMenuToolbarAndGizmos()
{
  struct WindowToggleInfo
  {
    const char* icon;
    const char* shortcut;
    bool*       visible;
    const char* tooltip;
  };
  static const std::array<WindowToggleInfo, 6> windowToggles  = {{
      {ICON_MS_PHOTO_CAMERA, "F1", &m_resources.settings.showCameraWindow, "Camera"},
      {ICON_MS_ACCOUNT_TREE, "F2", &m_resources.settings.showSceneBrowserWindow, "Scene Browser"},
      {ICON_MS_SETTINGS, "F3", &m_resources.settings.showSettingsWindow, "Settings"},
      {ICON_MS_LIST_ALT, "F4", &m_resources.settings.showInspectorWindow, "Inspector"},
      {ICON_MS_PUBLIC, "F5", &m_resources.settings.showEnvironmentWindow, "Environment"},
      {ICON_MS_TONALITY, "F6", &m_resources.settings.showTonemapperWindow, "Tonemapper"},
  }};
  float                                        buttonSize     = ImGui::GetFrameHeight();
  const ImGuiStyle&                            style          = ImGui::GetStyle();
  float                                        separatorWidth = 2.0f;
  float totalWidth  = separatorWidth + static_cast<float>(windowToggles.size()) * (buttonSize + style.ItemSpacing.x);
  float windowWidth = ImGui::GetWindowWidth();
  float offsetX     = windowWidth * 0.5f - totalWidth * 0.5f;
  ImGui::SameLine(offsetX);
  ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
  ImGui::SameLine();
  for(size_t i = 0; i < windowToggles.size(); ++i)
  {
    const auto& toggle = windowToggles[i];
    ImVec4      buttonColor =
        *toggle.visible ? ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] : ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
    ImGui::PushStyleColor(ImGuiCol_Button, buttonColor);
    if(ImGui::Button(toggle.icon))
      *toggle.visible = !*toggle.visible;
    ImGui::PopStyleColor();
    if(ImGui::IsItemHovered())
      ImGui::SetTooltip("Toggle %s Window (%s)", toggle.tooltip, toggle.shortcut);
    if(i < windowToggles.size() - 1)
      ImGui::SameLine(0, 0);
  }
  ImGui::SameLine();
  ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
  ImGui::SameLine();
  struct GizmoToggle
  {
    const char* icon;
    const char* tooltip;
    const char* shortcut;
    bool*       visible;
  };
  GizmoToggle gizmoToggles[] = {
      {ICON_MS_GRID_ON, "Grid", "G", &m_resources.settings.showGrid},
      {ICON_MS_3D_ROTATION, "Gizmo", "T", &m_resources.settings.showGizmo},
  };
  for(size_t i = 0; i < std::size(gizmoToggles); ++i)
  {
    const auto& toggle = gizmoToggles[i];
    ImVec4      buttonColor =
        *toggle.visible ? ImGui::GetStyle().Colors[ImGuiCol_ButtonActive] : ImGui::GetStyle().Colors[ImGuiCol_ChildBg];
    ImGui::PushStyleColor(ImGuiCol_Button, buttonColor);
    if(ImGui::Button(toggle.icon, ImVec2(buttonSize, buttonSize)))
      *toggle.visible = !*toggle.visible;
    ImGui::PopStyleColor();
    if(ImGui::IsItemHovered())
      ImGui::SetTooltip("Toggle %s (%s in viewport)", toggle.tooltip, toggle.shortcut);
    if(i < std::size(gizmoToggles) - 1)
      ImGui::SameLine(0, 0);
  }
  ImGui::SameLine();
  ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
}

void GltfRenderer::renderMenu()
{
  bool v_sync = m_app->isVsync();
  struct WindowToggleInfo
  {
    const char* icon;
    const char* name;
    const char* shortcut;
    ImGuiKey    key;
    bool        useShift;
    bool*       visible;
    const char* tooltip;
  };
  static const std::array<WindowToggleInfo, 6> windowToggles = {{
      {ICON_MS_PHOTO_CAMERA, "Camera", "F1", ImGuiKey_F1, false, &m_resources.settings.showCameraWindow, "Camera"},
      {ICON_MS_ACCOUNT_TREE, "Scene Browser", "F2", ImGuiKey_F2, false, &m_resources.settings.showSceneBrowserWindow, "Scene Browser"},
      {ICON_MS_SETTINGS, "Settings", "F3", ImGuiKey_F3, false, &m_resources.settings.showSettingsWindow, "Settings"},
      {ICON_MS_LIST_ALT, "Inspector", "F4", ImGuiKey_F4, false, &m_resources.settings.showInspectorWindow, "Inspector"},
      {ICON_MS_PUBLIC, "Environment", "F5", ImGuiKey_F5, false, &m_resources.settings.showEnvironmentWindow, "Environment"},
      {ICON_MS_TONALITY, "Tonemapper", "F6", ImGuiKey_F6, false, &m_resources.settings.showTonemapperWindow, "Tonemapper"},
  }};
  for(const auto& toggle : windowToggles)
  {
    if(toggle.useShift)
    {
      if(ImGui::IsKeyChordPressed(ImGuiMod_Shift | toggle.key))
        *toggle.visible = !*toggle.visible;
    }
    else if(toggle.key != ImGuiKey_None && ImGui::IsKeyPressed(toggle.key))
      *toggle.visible = !*toggle.visible;
  }
  std::filesystem::path sceneToLoadFilename{};
  std::filesystem::path sceneToMergeFilename{};
  GltfRenderer::windowTitle();
  bool newScene       = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_N);
  bool openFile       = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_O);
  bool mergeFile      = false;
  bool loadHdrFile    = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_O);
  bool saveFile       = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S);
  bool saveScreenFile = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiMod_Alt | ImGuiKey_S);
  bool saveImageFile  = !saveScreenFile && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_S);
  bool closeApp       = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Q);
  bool fitScene       = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_F);
  bool fitObject      = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_F);
  bool toggleVsyc     = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_V);
  bool reloadShader   = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_R);
  bool compactScene   = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_K);
  bool undoCmd        = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Z);
  bool redoCmd        = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Y);
  if(toggleVsyc)
    v_sync = !v_sync;
  bool validScene = m_resources.getScene() && m_resources.getScene()->valid();

  renderFileMenu(validScene, newScene, openFile, mergeFile, loadHdrFile, saveFile, saveScreenFile, saveImageFile,
                 closeApp, sceneToLoadFilename, sceneToMergeFilename);
  renderEditMenu(validScene);

  if(ImGui::IsKeyPressed(ImGuiKey_Escape))
  {
    m_resources.selectedRenderNodes.clear();
    m_sceneSelection.clearSelection();
  }

  if(undoCmd && m_undoStack.canUndo())
  {
    m_undoStack.undo();
    onUndoRedo();
  }
  if(redoCmd && m_undoStack.canRedo())
  {
    m_undoStack.redo();
    onUndoRedo();
  }

  renderViewMenu(validScene, fitScene, fitObject, v_sync);
  renderWindowsMenu();
  renderToolsMenu(validScene, reloadShader, compactScene);
  renderDebugMenu();
  renderMenuToolbarAndGizmos();

  auto getSaveImage = [this]() {
    return nvgui::windowSaveFileDialog(m_app->getWindowHandle(), "Save Image",
                                       "Image Files|*.png;*.jpg;*.hdr|PNG Files|*.png|JPG Files|*.jpg|HDR Files|*.hdr");
  };

  if(newScene)
  {
    // SYNC NOTE: User-initiated new scene — wait before full cleanup.
    vkQueueWaitIdle(m_app->getQueue(0).queue);
    cleanupScene();
    m_sceneSelection.clearSelection();
  }

  if(reloadShader)
  {
    SCOPED_BANNER("Reload Shaders");
    // SYNC NOTE: User-initiated shader recompile (F5) — wait before destroying old pipelines.
    vkQueueWaitIdle(m_app->getQueue(0).queue);
    compileShaders();
    resetFrame();
  }

  if(openFile)
  {
    sceneToLoadFilename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load 3D Scene",
                                                      "3D Scene Files|*.gltf;*.glb;*.obj;*.scene.json;*.glxf|glTF|*.gltf;*.glb|OBJ|*.obj|Scene Descriptor|*.scene.json;*.glxf",
                                                      m_lastSceneDirectory);
  }
  if(!sceneToLoadFilename.empty())
  {
    onFileDrop(sceneToLoadFilename.c_str());
  }

  if(mergeFile)
  {
    sceneToMergeFilename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Import/Merge Scene",
                                                       "3D Scene Files|*.gltf;*.glb|glTF Text|*.gltf|glTF Binary|*.glb",
                                                       m_lastSceneDirectory);
  }
  if(!sceneToMergeFilename.empty())
  {
    onMergeScene(sceneToMergeFilename);
    sceneToMergeFilename.clear();
  }

  if(loadHdrFile)
  {
    loadHdrFileDialog();
  }

  if(saveFile && validScene)
  {
    std::filesystem::path filename = nvgui::windowSaveFileDialog(m_app->getWindowHandle(), "Save glTF",
                                                                 "glTF Files|*.gltf;*.glb|glTF Text|*.gltf|glTF Binary|*.glb");
    if(!filename.empty())
      save(filename);
  }

  if(saveScreenFile)
  {
    std::filesystem::path filename = getSaveImage();
    if(!filename.empty())
    {
      m_app->requestScreenShot(filename, 100);
    }
  }

  if(saveImageFile)
  {
    std::filesystem::path filename = getSaveImage();
    if(!filename.empty())
    {
      Resources::ImageType imageType =
          (filename.extension() == ".hdr") ? Resources::ImageType::eImgRendered : Resources::ImageType::eImgTonemapped;

      m_app->saveImageToFile(m_resources.gBuffers.getColorImage(imageType), m_resources.gBuffers.getSize(), filename);
    }
  }

  if(validScene && (fitScene || (fitObject && !m_resources.selectedRenderNodes.empty())))
  {
    nvutils::Bbox bbox = fitScene ? m_resources.getScene()->getSceneBounds() :
                                    GltfRenderer::getRenderNodesBbox(m_resources.selectedRenderNodes);
    m_cameraManip->fit(bbox.min(), bbox.max(), false, true, m_cameraManip->getAspectRatio());
  }

  if(compactScene && validScene)
  {
    SCOPED_BANNER("Compact Scene");
    if(m_resources.getScene()->compactModel())
    {
      // Model compacted - need full GPU rebuild including textures
      rebuildVulkanSceneFull();
      resetFrame();  // Reset path tracer accumulation
    }
  }

  if(closeApp)
  {
    m_app->close();
  }

  if(m_app->isVsync() != v_sync)
  {
    m_app->setVsync(v_sync);
  }

  // Let both renderers handle their menus
  m_pathTracer.onUIMenu();
  m_rasterizer.onUIMenu();
}


void GltfRenderer::addToRecentFiles(const std::filesystem::path& filePath, int historySize)
{
  if(filePath.empty())
    return;

  auto it = std::find(m_recentFiles.begin(), m_recentFiles.end(), filePath);
  if(it != m_recentFiles.end())
  {
    m_recentFiles.erase(it);
  }
  m_recentFiles.insert(m_recentFiles.begin(), filePath);
  if(m_recentFiles.size() > historySize)
  {
    m_recentFiles.pop_back();
  }
}

void GltfRenderer::removeFromRecentFiles(const std::filesystem::path& filePath)
{
  auto it = std::find(m_recentFiles.begin(), m_recentFiles.end(), filePath);
  if(it != m_recentFiles.end())
  {
    m_recentFiles.erase(it);
  }
}

// Register handler
void GltfRenderer::registerRecentFilesHandler()
{
  // mandatory to work, see ImGui::DockContextInitialize as an example
  auto readOpen = [](ImGuiContext*, ImGuiSettingsHandler*, const char* name) -> void* {
    if(strcmp(name, "Data") != 0)
      return nullptr;
    return (void*)1;
  };

  // Save settings handler, not using capture so can be used as a function pointer
  auto saveRecentFilesToIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
    auto* self = static_cast<GltfRenderer*>(handler->UserData);
    buf->appendf("[%s][Data]\n", handler->TypeName);
    for(const auto& file : self->m_recentFiles)
    {
      buf->appendf("File=%s\n", file.string().c_str());
    }
    // Save directory preferences
    if(!self->m_lastSceneDirectory.empty())
    {
      buf->appendf("SceneDir=%s\n", self->m_lastSceneDirectory.string().c_str());
    }
    if(!self->m_lastHdrDirectory.empty())
    {
      buf->appendf("HdrDir=%s\n", self->m_lastHdrDirectory.string().c_str());
    }
    buf->append("\n");
  };

  // Load settings handler, not using capture so can be used as a function pointer
  auto loadRecentFilesFromIni = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const char* line) {
    auto* self = static_cast<GltfRenderer*>(handler->UserData);
    if(strncmp(line, "File=", 5) == 0)
    {
      const char* filePath = line + 5;
      auto        it       = std::find(self->m_recentFiles.begin(), self->m_recentFiles.end(), filePath);
      if(it == self->m_recentFiles.end())
      {
        self->m_recentFiles.push_back(filePath);
      }
    }
    else if(strncmp(line, "SceneDir=", 9) == 0)
    {
      const char* dirPath        = line + 9;
      self->m_lastSceneDirectory = std::filesystem::path(dirPath);
    }
    else if(strncmp(line, "HdrDir=", 7) == 0)
    {
      const char* dirPath      = line + 7;
      self->m_lastHdrDirectory = std::filesystem::path(dirPath);
    }
  };

  //
  ImGuiSettingsHandler iniHandler;
  iniHandler.TypeName   = "RecentFiles";
  iniHandler.TypeHash   = ImHashStr(iniHandler.TypeName);
  iniHandler.ReadOpenFn = readOpen;
  iniHandler.WriteAllFn = saveRecentFilesToIni;
  iniHandler.ReadLineFn = loadRecentFilesFromIni;
  iniHandler.UserData   = this;  // Pass the current instance to the handler
  ImGui::GetCurrentContext()->SettingsHandlers.push_back(iniHandler);
}

//--------------------------------------------------------------------------------------------------
// Apply a GLTF camera to the camera manipulator
// This function converts GLTF camera parameters to camera manipulator settings
// and applies them to the current view
//
void GltfRenderer::applyGltfCamera(int cameraIndex)
{
  if(!m_resources.getScene() || !m_resources.getScene()->valid())
    return;

  // Get the node that contains this camera to clear its extras
  int nodeIndex = m_sceneBrowser.getNodeForCamera(cameraIndex);
  if(nodeIndex >= 0)
  {
    // Clear the camera extras so that eye/center/up are recalculated from the world matrix
    tinygltf::Node& node = m_resources.getScene()->getModel().nodes[nodeIndex];
    if(node.extras.IsObject())
    {
      tinygltf::Value::Object extras = node.extras.Get<tinygltf::Value::Object>();
      extras.erase("camera::eye");
      extras.erase("camera::center");
      extras.erase("camera::up");
      node.extras = tinygltf::Value(extras);
    }
  }

  // Force refresh of render cameras to reflect latest UI changes
  const std::vector<nvvkgltf::RenderCamera>& cameras = m_resources.getScene()->getRenderCameras(true);

  if(cameraIndex < 0 || cameraIndex >= static_cast<int>(cameras.size()))
    return;

  const nvvkgltf::RenderCamera& camera = cameras[cameraIndex];

  if(camera.type == nvvkgltf::RenderCamera::CameraType::ePerspective)
  {
    float                              fov = static_cast<float>(glm::degrees(camera.yfov));
    nvutils::CameraManipulator::Camera cam{camera.eye, camera.center, camera.up, fov};
    cam.projectionType = nvutils::CameraManipulator::Perspective;
    m_cameraManip->setCamera(cam);
    m_cameraManip->setClipPlanes({static_cast<float>(camera.znear), static_cast<float>(camera.zfar)});
  }
  else if(camera.type == nvvkgltf::RenderCamera::CameraType::eOrthographic)
  {
    float                              fov = 45.0f;
    nvutils::CameraManipulator::Camera cam{camera.eye, camera.center, camera.up, fov};
    cam.projectionType = nvutils::CameraManipulator::Orthographic;
    cam.orthMag.x      = static_cast<float>(camera.xmag);
    cam.orthMag.y      = static_cast<float>(camera.ymag);
    m_cameraManip->setCamera(cam);
    m_cameraManip->adjustOrthographicAspect();
    m_cameraManip->setClipPlanes({static_cast<float>(camera.znear), static_cast<float>(camera.zfar)});
  }

  // Also update the scene's camera to keep extras (eye, center, up) in sync
  m_resources.getScene()->setSceneCamera(camera);
}

//--------------------------------------------------------------------------------------------------
// Set a GLTF camera from the current camera manipulator state
// This function updates the GLTF camera parameters and node extras with the current view
//
void GltfRenderer::setGltfCameraFromView(int cameraIndex)
{
  if(!m_resources.getScene() || !m_resources.getScene()->valid() || !m_cameraManip)
  {
    return;
  }

  // Get the node that contains this camera
  int nodeIndex = m_sceneBrowser.getNodeForCamera(cameraIndex);
  if(nodeIndex < 0)
  {
    return;
  }

  tinygltf::Node&   node   = m_resources.getScene()->getModel().nodes[nodeIndex];
  tinygltf::Camera& camera = m_resources.getScene()->getModel().cameras[cameraIndex];

  // Get current camera state from manipulator
  nvutils::CameraManipulator::Camera cameraState = m_cameraManip->getCamera();
  glm::vec2                          clipPlanes  = m_cameraManip->getClipPlanes();

  // Update the camera parameters
  if(cameraState.projectionType == nvutils::CameraManipulator::Orthographic)
  {
    camera.type                         = "orthographic";
    tinygltf::OrthographicCamera& ortho = camera.orthographic;
    ortho.xmag                          = static_cast<double>(cameraState.orthMag.x);
    ortho.ymag                          = static_cast<double>(cameraState.orthMag.y);
    ortho.znear                         = static_cast<double>(clipPlanes.x);
    ortho.zfar                          = static_cast<double>(clipPlanes.y);
  }
  else
  {
    camera.type                        = "perspective";
    tinygltf::PerspectiveCamera& persp = camera.perspective;
    // Convert FOV from degrees (manipulator) to radians (GLTF)
    persp.yfov  = glm::radians(cameraState.fov);
    persp.znear = static_cast<double>(clipPlanes.x);
    persp.zfar  = static_cast<double>(clipPlanes.y);
    // Aspect ratio will be calculated from viewport when camera is applied
  }

  // Update the node transformation to match the current camera position
  // The camera's eye position becomes the node's translation
  node.translation = {cameraState.eye.x, cameraState.eye.y, cameraState.eye.z};

  // Calculate the rotation that points the camera from eye to center with the given up vector
  glm::vec3 forward = glm::normalize(cameraState.ctr - cameraState.eye);
  glm::vec3 right   = glm::normalize(glm::cross(forward, cameraState.up));
  glm::vec3 up      = glm::normalize(glm::cross(right, forward));

  // Create rotation matrix and convert to quaternion
  glm::mat3 rotationMatrix(right, up, -forward);  // -forward because GLTF cameras look down -Z
  glm::quat rotation = glm::quat_cast(rotationMatrix);

  node.rotation = {rotation.x, rotation.y, rotation.z, rotation.w};
  node.scale    = {1.0, 1.0, 1.0};

  // Clear any existing matrix since we're using TRS
  node.matrix.clear();

  // Update the node extras with the current eye, center, and up vectors
  if(!node.extras.IsObject())
  {
    node.extras = tinygltf::Value(tinygltf::Value::Object());
  }

  tinygltf::Value::Object extras = node.extras.Get<tinygltf::Value::Object>();

  // Store eye vector
  extras["camera::eye"]    = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(cameraState.eye));
  extras["camera::center"] = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(cameraState.ctr));
  extras["camera::up"]     = tinygltf::utils::convertToTinygltfValue(3, glm::value_ptr(cameraState.up));
  node.extras              = tinygltf::Value(extras);
}

//--------------------------------------------------------------------------------------------------
// Display GPU memory statistics in an ImGui window
//
void GltfRenderer::renderMemoryStatistics()
{
  using namespace nvvkgltf;

  if(!m_resources.settings.showMemStats)
    return;

  if(!ImGui::Begin("Memory Statistics", &m_resources.settings.showMemStats))
  {
    ImGui::End();
    return;
  }

  // Helper function to format bytes
  auto formatBytes = [](uint64_t bytes) -> std::string {
    if(bytes >= 1024ULL * 1024ULL * 1024ULL)
      return fmt::format("{:.2f} GB", bytes / (1024.0 * 1024.0 * 1024.0));
    if(bytes >= 1024ULL * 1024ULL)
      return fmt::format("{:.2f} MB", bytes / (1024.0 * 1024.0));
    if(bytes >= 1024ULL)
      return fmt::format("{:.2f} KB", bytes / 1024.0);
    return fmt::format("{} B", bytes);
  };

  // Get memory trackers from scene
  const auto& vkTracker  = m_resources.sceneVk.getMemoryTracker();
  const auto& rtxTracker = m_resources.sceneRtx.getMemoryTracker();

  // Create sortable table with memory statistics
  ImGuiTableFlags tableFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Sortable;
  if(ImGui::BeginTable("MemoryStatsTable", 3, tableFlags))
  {
    // Setup sortable columns
    ImGui::TableSetupColumn("Category", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 120.0f);
    ImGui::TableSetupColumn("Current", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 50.0f);
    ImGui::TableHeadersRow();

    // Get sort specifications
    const ImGuiTableColumnSortSpecs& spec      = ImGui::TableGetSortSpecs()->Specs[0];
    const CategorySortBy             sortBy    = CategorySortBy(spec.ColumnIndex);
    const bool                       ascending = (spec.SortDirection == ImGuiSortDirection_Ascending);

    // --- Scene (SceneVk) Section ---
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "SCENE");
    ImGui::TableNextColumn();
    ImGui::TableNextColumn();

    // Display SceneVk categories with sorting
    for(const auto& categoryName : vkTracker.getActiveCategories(sortBy, ascending))
    {
      GpuMemoryStats stats = vkTracker.getStats(categoryName);

      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("  %s", categoryName.c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatBytes(stats.currentBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%u", stats.currentCount);
    }

    // Scene subtotal
    auto totalVk = vkTracker.getTotalStats();
    if(totalVk.currentBytes > 0)
    {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "  Scene Subtotal");
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "%s", formatBytes(totalVk.currentBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(0.7f, 0.9f, 1.0f, 1.0f), "%u", totalVk.currentCount);
    }

    // --- RTX (SceneRtx) Section ---
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "RTX (Acceleration)");
    ImGui::TableNextColumn();
    ImGui::TableNextColumn();

    // Display SceneRtx categories with sorting
    for(const auto& categoryName : rtxTracker.getActiveCategories(sortBy, ascending))
    {
      GpuMemoryStats stats = rtxTracker.getStats(categoryName);

      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::Text("  %s", categoryName.c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%s", formatBytes(stats.currentBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::Text("%u", stats.currentCount);
    }

    // RTX subtotal
    auto totalRtx = rtxTracker.getTotalStats();
    if(totalRtx.currentBytes > 0)
    {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.7f, 1.0f), "  RTX Subtotal");
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.7f, 1.0f), "%s", formatBytes(totalRtx.currentBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.7f, 1.0f), "%u", totalRtx.currentCount);
    }

    // --- Combined Total ---
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "TOTAL");
    ImGui::TableNextColumn();
    auto combinedBytes = totalVk.currentBytes + totalRtx.currentBytes;
    auto combinedCount = totalVk.currentCount + totalRtx.currentCount;
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%s", formatBytes(combinedBytes).c_str());
    ImGui::TableNextColumn();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%u", combinedCount);

    ImGui::EndTable();
  }

  ImGui::End();
}

//--------------------------------------------------------------------------------------------------
// Environment window - controls for environment settings (HDR, Sky, Solid Color)
//
void GltfRenderer::renderEnvironmentWindow()
{
  if(!m_resources.settings.showEnvironmentWindow)
    return;

  if(!ImGui::Begin("Environment", &m_resources.settings.showEnvironmentWindow))
  {
    ImGui::End();
    return;
  }
  nvgui::tooltip("Press F5 to toggle this window");
  bool changed = false;
  namespace PE = nvgui::PropertyEditor;

  if(PE::begin())
  {
    if(PE::Combo("Environment Type", (int*)&m_resources.settings.envSystem, "Sky\0HDR\0\0"))  // 0: Sky, 1: HDR
    {
      m_pathTracer.m_pushConst.fireflyClampThreshold =
          (m_resources.settings.envSystem == shaderio::EnvSystem::eSky) ? 10.0f : m_resources.hdrIbl.getIntegral();
      changed |= true;
    }
    changed |= PE::Checkbox("Solid Color", &m_resources.settings.useSolidBackground);
    if(m_resources.settings.useSolidBackground)
    {
      changed |= PE::ColorEdit3("Background Color", glm::value_ptr(m_resources.settings.solidBackgroundColor));
    }
    PE::end();
  }

  if(m_resources.settings.envSystem == shaderio::EnvSystem::eHdr)
  {
    if(PE::begin("HDR"))
    {
      if(PE::entry("", [&] { return ImGui::SmallButton("load"); }, "Load HDR Image"))
      {
        loadHdrFileDialog();
        changed = true;
      }
      changed |= PE::SliderFloat("Intensity", &m_resources.settings.hdrEnvIntensity, 0, 100, "%.3f",
                                 ImGuiSliderFlags_Logarithmic, "HDR intensity");
      changed |= PE::SliderAngle("Rotation", &m_resources.settings.hdrEnvRotation, -360, 360, "%.0f deg", 0, "Rotating the environment");
      changed |= PE::SliderFloat("Blur", &m_resources.settings.hdrBlur, 0, 1, "%.3f", 0, "Blur the environment");
      PE::end();
    }
  }
  else
  {
    changed |= nvgui::skyPhysicalParameterUI(m_resources.skyParams);
  }

  if(changed)
    resetFrame();

  ImGui::End();
}

//--------------------------------------------------------------------------------------------------
// Tonemapper window - controls for tone mapping settings
//
void GltfRenderer::renderTonemapperWindow()
{
  if(!m_resources.settings.showTonemapperWindow)
    return;

  if(!ImGui::Begin("Tonemapper", &m_resources.settings.showTonemapperWindow))
  {
    ImGui::End();
    return;
  }
  nvgui::tooltip("Press F6 to toggle this window");
  nvgui::tonemapperWidget(m_resources.tonemapperData);

  ImGui::End();
}

//--------------------------------------------------------------------------------------------------
// Statistics window - displays scene statistics
//
void GltfRenderer::renderStatisticsWindow()
{
  if(!m_resources.settings.showStatisticsWindow)
    return;

  if(!ImGui::Begin("Statistics", &m_resources.settings.showStatisticsWindow))
  {
    ImGui::End();
    return;
  }

  if(!m_resources.getScene() || !m_resources.getScene()->valid())
  {
    ImGui::TextDisabled("No scene loaded");
    ImGui::End();
    return;
  }

  const tinygltf::Model& tiny = m_resources.getScene()->getModel();

  // Create a table for better organization
  if(ImGui::BeginTable("StatisticsTable", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
  {
    ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, 80.0f);
    ImGui::TableHeadersRow();

    // Lambda function to add table rows
    auto addStatRow = [](const char* icon, const char* label, size_t value) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("%s %s", icon, label);
      ImGui::TableSetColumnIndex(1);
      ImGui::Text("%zu", value);
    };

    // Scene Structure
    addStatRow(ICON_MS_LAYERS, "Nodes", tiny.nodes.size());
    addStatRow(ICON_MS_VIEW_LIST, "Render Nodes", m_resources.getScene()->getRenderNodes().size());
    addStatRow(ICON_MS_EXTENSION, "Render Primitives", m_resources.getScene()->getNumRenderPrimitives());

    // Materials & Geometry
    addStatRow(ICON_MS_BRUSH, "Materials", tiny.materials.size());
    addStatRow(ICON_MS_SIGNAL_CELLULAR_NULL, "Triangles", m_resources.getScene()->getNumTriangles());

    // Lighting & Assets
    addStatRow(ICON_MS_LIGHTBULB, "Lights", tiny.lights.size());
    addStatRow(ICON_MS_IMAGE_INSET, "Textures", tiny.textures.size());
    addStatRow(ICON_MS_IMAGE, "Images", tiny.images.size());

    ImGui::EndTable();
  }

  // Copy to clipboard button with proper spacing
  ImGui::Spacing();

  if(ImGui::Button(ICON_MS_CONTENT_COPY " Copy to Clipboard"))
  {
    ImGui::LogToClipboard();
    // Log all statistics to clipboard
    ImGui::LogText("Scene Statistics:\n");
    ImGui::LogText("Nodes: %zu\n", tiny.nodes.size());
    ImGui::LogText("Render Nodes: %zu\n", m_resources.getScene()->getRenderNodes().size());
    ImGui::LogText("Render Primitives: %zu\n", m_resources.getScene()->getNumRenderPrimitives());
    ImGui::LogText("Materials: %zu\n", tiny.materials.size());
    ImGui::LogText("Triangles: %d\n", m_resources.getScene()->getNumTriangles());
    ImGui::LogText("Lights: %zu\n", tiny.lights.size());
    ImGui::LogText("Textures: %zu\n", tiny.textures.size());
    ImGui::LogText("Images: %zu\n", tiny.images.size());
    ImGui::LogFinish();
  }
  if(ImGui::IsItemHovered())
  {
    ImGui::SetTooltip("Copy statistics to clipboard");
  }

  // Compact Scene button
  ImGui::SameLine();
  if(ImGui::Button(ICON_MS_COMPRESS " Compact Scene"))
  {
    SCOPED_BANNER("Compact Scene");
    if(m_resources.getScene()->compactModel())
    {
      // Model compacted - need full GPU rebuild including textures
      rebuildVulkanSceneFull();
      resetFrame();  // Reset path tracer accumulation
    }
  }
  if(ImGui::IsItemHovered())
  {
    ImGui::SetTooltip("Remove orphaned resources - see counts update immediately");
  }

  ImGui::End();
}