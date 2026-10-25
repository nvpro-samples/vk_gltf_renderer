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
#include <cfloat>
#include <filesystem>
#include <limits>
#include <fmt/format.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

#include <nvgui/axis.hpp>
#include <nvgui/file_dialog.hpp>
#include <nvgui/fonts.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/hover_scrolling.hpp>
#include <nvgui/tonemapper.hpp>
#include <nvgui/tooltip.hpp>
#include <nvutils/bounding_box.hpp>
#include <nvutils/file_operations.hpp>
#include <nvvk/check_error.hpp>


#include "env_image_loader.hpp"
#include "renderer.hpp"
#include "gltf_create_tangent.hpp"
#include "gltf_scene_editor.hpp"
#include "scoped_banner.hpp"
#include "tinygltf_utils.hpp"
#include "ui_animation.hpp"
#include "ui_commands.hpp"
#include "ui_dock_layout.hpp"
#include "ui_helpers.hpp"
#include "ui_interactivity.hpp"
#include "ui_mouse_state.hpp"
#include "version.hpp"

// The window toggles: the "Windows" commands with an on/off state. The Windows menu and the toolbar
// both list them, in the order registerCommands() adds them.
static bool isWindowToggle(const ui::Command& cmd)
{
  return cmd.group != nullptr && std::strcmp(cmd.group, "Windows") == 0 && cmd.checked;
}

// Command actions over a bool the renderer owns: raise a request, flip a toggle, read its state.
static std::function<void()> requestFlag(bool& flag)
{
  return [&flag] { flag = true; };
}
static std::function<void()> toggleFlag(bool& flag)
{
  return [&flag] { flag = !flag; };
}
static std::function<bool()> flagIsOn(const bool& flag)
{
  return [&flag] { return flag; };
}

void GltfRenderer::mouseClickedInViewport()
{
  static UiMouseState s_mouseClickState;  // Mouse click state

  // Allow input when hovered, or when the gizmo is mid-drag (mouse may have left the viewport).
  if(!ImGui::IsWindowHovered(ImGuiFocusedFlags_RootWindow) && !m_visualHelpers.transform.isDragging())
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

  // A click in the "Viewport" shoots a ray to the scene under the mouse: a click selects what it
  // hits, a double-click sets the camera center to the hit position. Which button and gesture do which
  // comes from the camera_controls table (Action::eSelect, Action::eSetPointOfInterest).
  //
  // The actual ray-pick GPU work is deferred (see m_pendingClickPick's comment in renderer.hpp) -
  // this just captures everything it needs at the moment of the click (camera state included, so
  // a camera move between now and when onRender() records the pick doesn't skew the ray) and hands
  // it off. No GPU work, no wait - this function is now as cheap as updateHoverCursorPosition().
  //
  // Two ways a click can be recognized:
  //  - Normal editing: s_mouseClickState's single/double-click debounce (waits io.MouseDoubleClickTime,
  //    ImGui's default 0.3s, to see if a second click follows) so a double-click can mean "recenter
  //    camera" instead of "select".
  //  - While a KHR_interactivity graph is playing: double-click-to-recenter isn't something a
  //    click-driven graph needs, and that debounce is exactly the "still feels laggy" delay users
  //    notice even after the GPU wait was removed - so skip it entirely and react on the release
  //    itself (isMouseReleasedNoDrag still correctly excludes camera-orbit drags).
  const camera_controls::Binding* selectBinding = camera_controls::findAction(camera_controls::Action::eSelect);
  const camera_controls::Binding* centerBinding = camera_controls::findAction(camera_controls::Action::eSetPointOfInterest);
  if(selectBinding == nullptr || centerBinding == nullptr)
  {
    assert(false && "The camera_controls table must bind eSelect and eSetPointOfInterest");
    return;
  }
  auto gestured = [&](const camera_controls::Binding& binding) {
    const ImGuiMouseButton button = camera_controls::clickButton(binding.input);
    const bool held = camera_controls::matches(binding, camera_controls::heldMods(), m_cameraManip->getMode(),
                                               m_cameraManip->getProjectionType());
    return held
           && (binding.input == camera_controls::Input::eLeftDoubleClick ? s_mouseClickState.isMouseDoubleClicked(button) :
                                                                           s_mouseClickState.isMouseSingleClicked(button));
  };
  const bool interactivityPlaying = isInteractivityPlaying();
  const bool setCenter            = !interactivityPlaying && gestured(*centerBinding);
  const bool clicked              = interactivityPlaying ?
                                        s_mouseClickState.isMouseReleasedNoDrag(camera_controls::clickButton(selectBinding->input)) :
                                        (gestured(*selectBinding) || setCenter);
  if(clicked)
  {
    // Convert screen coordinates to normalized viewport coordinates [0,1]
    ImVec2 mousePos      = ImGui::GetMousePos();
    ImVec2 cursorPos     = ImGui::GetCursorScreenPos();
    ImVec2 avail         = ImGui::GetContentRegionAvail();
    ImVec2 localMousePos = ImVec2((mousePos.x - cursorPos.x) / avail.x, (mousePos.y - cursorPos.y) / avail.y);

    m_pendingClickPick = PendingClickPick{
        .modelViewInv   = glm::inverse(m_cameraManip->getViewMatrix()),
        .perspectiveInv = glm::inverse(m_cameraManip->getPerspectiveMatrix()),
        .isOrthographic = (m_cameraManip->getProjectionType() == nvutils::CameraManipulator::Orthographic) ? 1 : 0,
        .pickPos        = {localMousePos.x, localMousePos.y},
        .isDoubleClick  = setCenter,
    };
  }
}

// AnchorCameraManipulator::AnchorRequest: called from camera_controls::update() inside the
// "Viewport" window when an anchored drag or wheel step begins. Like a click, it only captures the ray; onRender()
// records the pick and updateAnchorPickState() answers the manipulator.
bool GltfRenderer::requestAnchorPick(const glm::dvec2& screenPos)
{
  const nvvkgltf::Scene* scene = m_resources.getScene();
  if(m_busy.isBusy() || scene == nullptr || !scene->valid() || m_resources.sceneRtx.topLevelAS() == VK_NULL_HANDLE)
    return false;

  // A new request supersedes one still in flight: the manipulator only waits for the latest, and an
  // older answer would otherwise be taken for it (e.g. a wheel pick becoming a drag's anchor).
  m_pendingAnchorResult.reset();

  const ImVec2 cursorPos = ImGui::GetCursorScreenPos();
  const ImVec2 avail     = ImGui::GetContentRegionAvail();
  m_pendingAnchorPick    = nvvk::RayPicker::PickInfo{
         .modelViewInv   = glm::inverse(m_cameraManip->getViewMatrix()),
         .perspectiveInv = glm::inverse(m_cameraManip->getPerspectiveMatrix()),
         .isOrthographic = (m_cameraManip->getProjectionType() == nvutils::CameraManipulator::Orthographic) ? 1 : 0,
         .pickPos        = {(screenPos.x - cursorPos.x) / avail.x, (screenPos.y - cursorPos.y) / avail.y},
  };
  return true;
}

// Applies a completed async ray-pick result (see m_pendingClickPick's comment in renderer.hpp for
// the full pipeline) - selection, and on a double click, camera recenter. Exactly the logic that
// used to run inline in mouseClickedInViewport() right after its now-removed blocking GPU wait.
void GltfRenderer::applyClickPickResult(const nvvk::RayPicker::PickResult& pickResult, bool isDoubleClick)
{
  // Environment was picked (no hit)
  const bool hasHit = pickResult.instanceID >= 0;
  // KHR_interactivity event/onSelect's selectionPoint/selectionRayOrigin - real ray-pick hit
  // point/origin (global space) when this click actually hit something, so a bound onSelect
  // handler sees finite values instead of the "no ray info" NaN fallback (KHR_node_selectability
  // README ~110 spec-sanctions NaN when there's no ray, but this IS a ray, so it must be used).
  glm::vec3 worldPos = hasHit ? pickResult.worldRayOrigin + pickResult.worldRayDirection * pickResult.hitT : glm::vec3(0);

  // Set or de-select the selected object (primitive-level selection)
  // pickResult.instanceID is the TLAS instance index = render node index
  if(!isDoubleClick)
  {
    updateSelectionFromPick(pickResult.instanceID, worldPos, pickResult.worldRayOrigin);
  }

  if(!hasHit)
    return;
  if(isDoubleClick)
  {
    // Set the camera CENTER to the hit position
    glm::dvec3 eye, center, up;
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

// Captures whether the cursor is currently over the 3D viewport and, if so, its pixel position in
// render-target space - consumed by onRender()'s updateHoverState()/HoverPicker later this same
// frame (docs/interactivity.md Phase E). Must be called from the viewport window's ImGui code,
// right where mouseClickedInViewport() computes the equivalent position for click-picking.
void GltfRenderer::updateHoverCursorPosition()
{
  m_hoverCursorInViewport = ImGui::IsWindowHovered(ImGuiFocusedFlags_RootWindow) && m_resources.getScene()
                            && m_resources.getScene()->valid() && !m_busy.isBusy();
  if(!m_hoverCursorInViewport)
    return;

  const ImVec2     mousePos  = ImGui::GetMousePos();
  const ImVec2     cursorPos = ImGui::GetCursorScreenPos();
  const ImVec2     avail     = ImGui::GetContentRegionAvail();
  const VkExtent2D imgSize   = m_resources.gBuffers.getSize();
  m_hoverCursorPixel = {static_cast<int>((mousePos.x - cursorPos.x) / avail.x * static_cast<float>(imgSize.width)),
                        static_cast<int>((mousePos.y - cursorPos.y) / avail.y * static_cast<float>(imgSize.height))};
}

bool GltfRenderer::isInteractivityPlaying() const
{
  const nvvkgltf::Scene* scene = m_resources.getScene();
  return scene && !scene->getInteractivityGraphs().empty() && m_resources.interactivityControl.play;
}

// Update the current selection from a ray-pick result. `renderNodeIdx` is the TLAS instance index
// (== render node index), or < 0 when the environment/nothing was hit. `selectionPoint`/
// `selectionRayOrigin` (global space) are threaded through to SceneSelection::selectPrimitive() for
// KHR_interactivity event/onSelect; default NaN for callers with no real ray (e.g. script-driven tests).
void GltfRenderer::updateSelectionFromPick(int renderNodeIdx, const glm::vec3& selectionPoint, const glm::vec3& selectionRayOrigin)
{
  const auto clearSelection = [&]() {
    m_resources.selectedRenderNodes.clear();
    m_sceneSelection.clearSelection();
  };

  nvvkgltf::Scene* scene       = m_resources.getScene();
  const auto&      renderNodes = scene->getRenderNodes();

  // Nothing hit, or invalid instance: clear the selection.
  if(renderNodeIdx < 0 || renderNodeIdx >= static_cast<int>(renderNodes.size()))
  {
    clearSelection();
    return;
  }

  // Clicking the currently selected primitive again de-selects it - standard editor behavior, but
  // actively harmful while a KHR_interactivity graph is playing: every other click on the same node
  // (e.g. re-clicking a lever/button several times in a row) would silently eat the click as a
  // deselect instead of re-firing event/onSelect, making the graph feel like it needs "two clicks"
  // to react. Skip the toggle while a graph is actually running, so every click re-selects and
  // re-fires the event; click-to-deselect still works normally once the graph is paused/absent.
  if(!isInteractivityPlaying() && m_resources.selectedRenderNodes.size() == 1 && m_resources.selectedRenderNodes.count(renderNodeIdx))
  {
    clearSelection();
    return;
  }

  // KHR_node_selectability: honor author intent. If the picked node opts out of selection (itself or
  // via an ancestor with `selectable:false`), redirect to the nearest selectable ancestor, or clear.
  if(!scene->isNodeSelectable(renderNodes[renderNodeIdx].refNodeID))
  {
    int selectableNodeID = scene->nearestSelectableAncestor(renderNodes[renderNodeIdx].refNodeID);
    if(selectableNodeID >= 0)
    {
      m_sceneSelection.selectNode(selectableNodeID, selectionPoint, selectionRayOrigin);
      m_sceneBrowser.focusOnSelection();
    }
    else
    {
      clearSelection();
    }
    return;
  }

  // The picked node is read-only (e.g. part of a referenced external asset). Select the first
  // editable ancestor instead, so the user can still change its transformation.
  if(scene->isNodeReadOnly(renderNodes[renderNodeIdx].refNodeID))
  {
    int editableNodeID = renderNodes[renderNodeIdx].refNodeID;
    while(editableNodeID >= 0 && scene->isNodeReadOnly(editableNodeID))
      editableNodeID = scene->editor().getNodeParent(editableNodeID);

    if(editableNodeID >= 0)
    {
      m_sceneSelection.selectNode(editableNodeID, selectionPoint, selectionRayOrigin);
      m_sceneBrowser.focusOnSelection();
    }
    else
    {
      clearSelection();
    }
    return;
  }

  // An instance of a placed external asset drawn without a node of its own: select its placing node.
  if(scene->isInstanceSetRenderNode(renderNodeIdx))
  {
    m_sceneSelection.selectNode(renderNodes[renderNodeIdx].refNodeID, selectionPoint, selectionRayOrigin);
    m_sceneBrowser.focusOnSelection();
    return;
  }

  // Regular case: select the picked primitive.
  const auto& renderNode = renderNodes[renderNodeIdx];
  int         nodeID     = renderNode.refNodeID;
  int         primIndex  = scene->getPrimitiveIndexForRenderNode(renderNodeIdx);
  int         meshID     = scene->getModel().nodes[nodeID].mesh;

  m_resources.selectedRenderNodes = {renderNodeIdx};
  m_sceneSelection.selectPrimitive(renderNodeIdx, nodeID, primIndex, meshID, selectionPoint, selectionRayOrigin);
  m_sceneBrowser.focusOnSelection();
}

nvutils::Bbox GltfRenderer::getRenderNodeBbox(int renderNodeIndex)
{
  // Empty when the render node is invalid or its primitive has no readable bounds, so a union
  // over a selection is never widened by a made-up box.
  nvvkgltf::Scene* scene = m_resources.getScene();
  if(renderNodeIndex < 0 || !scene || !scene->valid())
    return {};

  const auto& renderNodes = scene->getRenderNodes();
  if(renderNodeIndex >= static_cast<int>(renderNodes.size()))
    return {};

  const nvvkgltf::RenderNode&      renderNode      = renderNodes[renderNodeIndex];
  const nvvkgltf::RenderPrimitive& renderPrimitive = scene->getRenderPrimitive(renderNode.renderPrimID);

  glm::vec3 minValues, maxValues;
  if(!tinygltf::utils::getPrimitivePositionBounds(scene->getModel(), *renderPrimitive.pPrimitive, minValues, maxValues))
    return {};
  nvutils::Bbox objBbox(minValues, maxValues);

  glm::mat4 nodeWorld = scene->computeNodeWorldMatrix(renderNode.refNodeID);

  // Nodes drawn more than once (EXT_mesh_gpu_instancing, instance sets): one local matrix per render node.
  const auto& gpuInstMap = scene->getRenderNodeLocalMatrices();
  auto        instIt     = gpuInstMap.find(renderNode.refNodeID);
  if(instIt != gpuInstMap.end() && !instIt->second.empty())
  {
    const auto& rnList = scene->getRenderNodeRegistry().getRenderNodesForNode(renderNode.refNodeID);
    for(size_t k = 0; k < rnList.size(); ++k)
    {
      if(rnList[k] == renderNodeIndex)
      {
        if(k < instIt->second.size())
          nodeWorld = nodeWorld * instIt->second[k];
        break;
      }
    }
  }
  return objBbox.transform(nodeWorld);
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

// Centered warning banner drawn over the viewport image when Environment Type = None leaves the
// scene with no light source (no punctual lights, no emissive materials), so the frame is fully
// black. Anchored to the top-center of the passed image rectangle.
static void drawNoLightOverlay(const ImVec2& imageTopLeft, const ImVec2& imageSize)
{
  const char* line1 = ICON_MS_WARNING " Environment Type is \"None\" and the scene has no lights";
  const char* line2 = "Add a light or an emissive material, or set Environment to Sky / HDR";

  ImDrawList*  dl = ImGui::GetWindowDrawList();
  const ImVec2 pad(14.f, 10.f);
  const ImVec2 sz1     = ImGui::CalcTextSize(line1);
  const ImVec2 sz2     = ImGui::CalcTextSize(line2);
  const float  spacing = ImGui::GetStyle().ItemSpacing.y;
  const float  boxW    = std::max(sz1.x, sz2.x) + pad.x * 2.f;
  const float  boxH    = sz1.y + sz2.y + spacing + pad.y * 2.f;

  const float  cx  = imageTopLeft.x + imageSize.x * 0.5f;
  const float  top = imageTopLeft.y + imageSize.y * 0.12f;  // upper area, not flush against the edge
  const ImVec2 boxMin(cx - boxW * 0.5f, top);
  const ImVec2 boxMax(boxMin.x + boxW, boxMin.y + boxH);

  dl->AddRectFilled(boxMin, boxMax, IM_COL32(30, 25, 0, 210), 6.f);
  dl->AddRect(boxMin, boxMax, IM_COL32(230, 190, 40, 230), 6.f, 0, 1.5f);

  dl->AddText(ImVec2(cx - sz1.x * 0.5f, boxMin.y + pad.y), IM_COL32(255, 210, 70, 255), line1);
  dl->AddText(ImVec2(cx - sz2.x * 0.5f, boxMin.y + pad.y + sz1.y + spacing), IM_COL32(220, 220, 220, 235), line2);
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
    // The path tracer's progress is its samples per pixel: frames hold a varying number of them
    std::string progress = fmt::format("Frame {}", m_resources.frameCount);
    if(m_resources.settings.renderSystem == RenderingMode::ePathtracer)
      progress = fmt::format("{} / {} spp | {}", m_resources.accumulatedSamples, m_resources.settings.maxSamples, progress);
    std::string text =
        fmt::format("{} {} - {}x{} | {} | {:.0f} FPS / {:.3f}ms", nvutils::utf8FromPath(filename), APP_VERSION_STRING,
                    size.width, size.height, progress, ImGui::GetIO().Framerate, 1000.F / ImGui::GetIO().Framerate);

    glfwSetWindowTitle(m_app->getWindowHandle(), text.c_str());
    dirty_timer = 0;
  }
}

// Helper function to load HDR files
void GltfRenderer::loadHdrFileDialog()
{
  // "hdr,exr" -> "Environment Image|*.hdr;*.exr". Derived from the loader's own list so a newly
  // supported format cannot end up unofferable in the dialog.
  const std::string      extensions = envImageExtensions();
  std::string            filter     = "Environment Image|";
  std::string::size_type start      = 0;
  while(start < extensions.size())
  {
    const std::string::size_type comma = extensions.find(',', start);
    if(start != 0)
      filter += ";";
    filter += "*." + extensions.substr(start, comma - start);
    start = (comma == std::string::npos) ? extensions.size() : comma + 1;
  }

  std::filesystem::path filename =
      nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load Environment Image", filter.c_str(), m_lastHdrDirectory);
  if(!filename.empty())
  {
    onFileDrop(filename.c_str());
  }
}

//--------------------------------------------------------------------------------------------------
// The two `.sky.json` pickers. Thin on purpose: the work is in sky_preset, and these only decide
// where the file goes.
//
void GltfRenderer::loadSkyPresetDialog()
{
  const std::filesystem::path filename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load Sky Preset",
                                                                     "Sky Preset|*.sky.json;*.json", m_lastSkyPresetDirectory);
  if(filename.empty())
    return;

  if(loadSkyPreset(filename))
    m_lastSkyPresetDirectory = filename.parent_path();
}

void GltfRenderer::saveSkyPresetDialog()
{
  std::filesystem::path suggested = m_lastSkyPresetDirectory;
  suggested /= "sky.sky.json";

  std::filesystem::path filename =
      nvgui::windowSaveFileDialog(m_app->getWindowHandle(), "Save Sky Preset", "Sky Preset|*.sky.json;*.json", suggested);
  if(filename.empty())
    return;

  // The double extension is the convention the drop handler recognises, so a file saved without it
  // would not come back by being dragged in -- worth fixing up rather than explaining.
  const std::string name = nvutils::utf8FromPath(filename.filename());
  if(!name.ends_with(".sky.json"))
    filename.replace_extension(".sky.json");

  if(saveSkyPreset(filename))
    m_lastSkyPresetDirectory = filename.parent_path();
}

// Image picker for the inspector's "Load from file" texture action (see UiInspector import hooks).
std::filesystem::path GltfRenderer::pickImageFile()
{
  const std::filesystem::path filename =
      nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load Texture Image",
                                  "Images|*.png;*.jpg;*.jpeg;*.ktx;*.ktx2;*.dds;*.webp;*.tga;*.bmp;*.hdr;*.exr",
                                  m_lastImageDirectory);
  if(!filename.empty())
    m_lastImageDirectory = filename.parent_path();
  return filename;
}

void GltfRenderer::renderBenchmarkViewport()
{
  // GPU rendering runs in onRender(); the swapchain only shows content when ImGui draws
  // the tonemapped G-buffer (see Application::renderToSwapchain).
  m_cameraManip->updateAnim();

  const ImGuiViewport* vp = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(vp->WorkPos);
  ImGui::SetNextWindowSize(vp->WorkSize);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
  ImGui::Begin("Viewport", nullptr,
               ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize
                   | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoBringToFrontOnFocus);

  if(m_resources.getScene() && m_resources.getScene()->valid())
  {
    const ImVec2 imageSize = ImGui::GetContentRegionAvail();
    ImGui::Image(m_resources.tonemappedUi, imageSize);
  }

  ImGui::End();
  ImGui::PopStyleVar();
}

void GltfRenderer::renderUI()
{
  static int frameCount = 0;

  namespace PE = nvgui::PropertyEditor;
  {  // Setting menu
    bool changed{false};

    // Auto-poll + heartbeat refresh in one call (throttled internally).
#ifdef USE_AGENTIC
    m_agentic.tick();
#endif

    if(m_resources.settings.showCameraWindow)
    {
      if(ImGui::Begin("Camera", &m_resources.settings.showCameraWindow))
      {
        nvgui::tooltip("Press F1 to toggle this window");
        nvgui::CameraWidget(m_cameraManip);
      }
      ImGui::End();  // End Camera
    }

    // KHR_interactivity Graphs window: closed by default, opt-in via the View menu or toolbar
    // indicator. renderWindow() itself handles the ImGui::Begin/End pair and every no-op case
    // (no scene, no graph, window closed) - this is the whole call site.
    ui::interactivity::renderWindow(m_resources.getScene(), m_resources.interactivityControl,
                                    &m_resources.settings.showInteractivityWindow);

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
              // KHR_interactivity event/onSelect: real ray-pick hit point/origin when this selection
              // was redirected here from an actual 3D-viewport click (e.g. KHR_node_selectability's
              // nearest-selectable-ancestor fallback in updateSelectionFromPick()) - NaN (the
              // spec-sanctioned "no ray info" fallback) otherwise (Scene Browser click, script-driven
              // `selectnode`, ...; SceneSelection::selectNode's default parameters already are NaN).
              scn->notifyNodeSelected(nodeIdx, event.selectionPoint, event.selectionRayOrigin);
            }
            break;
          }
          case SceneSelection::EventType::PrimitiveSelected: {
            m_resources.selectedRenderNodes = {event.renderNodeIndex};
            nvvkgltf::Scene* scn            = m_resources.getScene();
            if(scn && scn->valid())
            {
              // KHR_interactivity event/onSelect: real ray-pick hit point/origin when this came from
              // an actual 3D-viewport click (SceneSelection::selectPrimitive's default parameters
              // are already the spec-sanctioned NaN fallback for callers with no ray - e.g. tests).
              scn->notifyNodeSelected(event.data, event.selectionPoint, event.selectionRayOrigin);
            }
            break;
          }
          case SceneSelection::EventType::MaterialSelected:
            // Material selection event
            break;
        }
      });

      m_sceneBrowser.render(&m_resources.settings.showSceneBrowserWindow, m_busy.isBusy());
      m_inspector.render(&m_resources.settings.showInspectorWindow, m_busy.isBusy());

      // Top-level so the "Add Primitive" modal surfaces even when the Scene Browser window is
      // hidden/collapsed (e.g. triggered from the menu-bar "Create"). requestAddPrimitive() remains
      // the trigger; this just renders the pending modal.
      m_sceneBrowser.showAddPrimitivePopup();
      // Same rationale for the image viewer: it can be opened from the inspector (thumbnail click), so
      // render it here rather than inside the Scene Browser's tab body.
      m_sceneBrowser.showImageViewer();
    }

#ifdef USE_AGENTIC
    agentic::renderAgenticWindow(m_agentic, m_resources);
#endif

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
            onRenderSystemChanged();
            changed = true;
          }
          changed |= PE::Combo("Visualization", (int32_t*)(&m_resources.settings.visualization),
                               "Rendered\0BaseColor\0Metallic\0Roughness\0NormalShd\0NormalGeo\0Tangent\0Bitangent\0Emissive\0Opacity\0TexCoord0\0TexCoord1\0Clay\0TriangleID\0FaceOrientation\0"
                               // Khronos glTF-Sample-Renderer DEBUG_* parity views (must match
                               // the Visualization enum order in shaderio.h).
                               "Occlusion\0Clearcoat Factor\0Clearcoat Roughness\0Clearcoat Normal\0"
                               "Sheen Color\0Sheen Roughness\0Specular Factor\0Specular Color\0"
                               "Transmission Factor\0Iridescence Factor\0Iridescence Thickness\0"
                               "Anisotropy Strength\0Diffuse Transmission Factor\0Diffuse Transmission Color\0"
                               // Opacity-micromap debug view (must match the Visualization enum order).
                               "Opacity Micromap\0\0");
          changed |= nvgui::hoverScrolling(*reinterpret_cast<int32_t*>(&m_resources.settings.visualization), 0,
                                           int32_t(shaderio::Visualization::eOpacityMicromap));
          changed |= PE::Checkbox("Wireframe", &m_resources.settings.wireframe, "Overlay wireframe on rendered meshes");
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

    // Camera orbit/pan/zoom, as camera_controls.cpp's table maps the input, unless the gizmo is
    // capturing it. The table leaves the movement keys alone while Ctrl or Alt is held, since those
    // combinations are application shortcuts (Ctrl+S, Ctrl+D, ...).
    if(!m_visualHelpers.transform.isDragging())
      camera_controls::update(*m_cameraManip, ImGui::GetCurrentWindow());

    // Handle mouse clicks and gizmo interaction in the viewport
    GltfRenderer::mouseClickedInViewport();

    // Capture the cursor's viewport pixel position for KHR_interactivity hover detection
    // (docs/interactivity.md Phase E). ImGui state (IsWindowHovered/GetCursorScreenPos) is only
    // valid here, in the UI-render phase - onRender()'s later Vulkan-recording pass can't query
    // it directly, so it's stashed on GltfRenderer for updateHoverState() to consume this frame.
    GltfRenderer::updateHoverCursorPosition();

    // Viewport-scoped keyboard shortcuts (G, T, Home, Space): live while the viewport is hovered or
    // focused. Called every frame from inside this window; the routing decides (ui_commands.cpp).
    m_commands.dispatch(ui::KeyScope::eViewport);

    // Display the tonemapped viewport image. The Animation Strip below is an
    // overlay (drawn on top of the image), so the image uses the full content
    // region and always matches what is displayed. Capture the
    // image's top-left before drawing so we can anchor the overlay with no
    // item-spacing fudge factor.
    const ImVec2 imageTopLeft = ImGui::GetCursorScreenPos();
    const ImVec2 imageSize    = ImGui::GetContentRegionAvail();
    ImGui::Image(m_resources.tonemappedUi, imageSize);

    // Black-viewport guard: with Environment Type = None there is no sky/HDR light. If the scene
    // also has no punctual lights and no emissive materials, nothing is lit. Unless a Solid Color
    // background is set (which still fills the frame), the viewport is fully black -- warn the user
    // rather than leaving them staring at an empty frame.
    if(m_resources.settings.envSystem == shaderio::EnvSystem::eNone && !m_resources.settings.useSolidBackground)
    {
      const nvvkgltf::Scene* scene = m_resources.getScene();
      const bool hasLight = scene && (!scene->getRenderLights().empty() || m_resources.sceneVk.numEmissiveTriangles() > 0);
      if(scene && !hasLight)
        drawNoLightOverlay(imageTopLeft, imageSize);
    }

    // Axis gizmo: anchored to the 3D image's bottom-left, nudged up by one
    // frame-height when the Animation Strip is showing so the two don't overlap.
    const bool stripVisible = m_resources.animationControl.showStrip;
    if(m_resources.settings.showAxis)
    {
      const float  axisSize   = 25.f;
      const float  dpi        = ImGui::GetWindowDpiScale();
      const ImVec2 windowPos  = ImGui::GetWindowPos();
      const ImVec2 windowSize = ImGui::GetWindowSize();
      const float  stripOff   = stripVisible ? ImGui::GetFrameHeight() : 0.0F;
      const ImVec2 axisPos(windowPos.x + axisSize * 1.1F * dpi, windowPos.y + windowSize.y - stripOff - axisSize * 1.1F * dpi);
      nvgui::Axis(axisPos, m_cameraManip->getViewMatrix(), axisSize);
    }

    // Animation Strip overlay: hover-reactive strip flush with the image's bottom.
    if(stripVisible)
    {
      const float  h = ImGui::GetFrameHeight();
      const ImVec2 stripTopLeft(imageTopLeft.x, imageTopLeft.y + imageSize.y - h);
      if(ui::animation::renderStripOverlay(m_resources.getScene(), m_resources.animationControl, stripTopLeft,
                                           ImVec2(imageSize.x, h)))
        m_animPlay = m_resources.animationControl.play;
    }

    ImGui::End();
    ImGui::PopStyleVar();
  }


  ui::help::renderWindows(m_help, *m_cameraManip, m_commands, m_app->getPhysicalDevice());

  // Show the busy window if the busy flag is set
  if(m_busy.isBusy())
  {
    m_busy.show();
  }

  // Display memory statistics window
  renderMemoryStatistics();

  // Display environment, tonemapper, and statistics windows
  m_uiEnvironment.render();
  renderTonemapperWindow();
  renderStatisticsWindow();

  if(m_resources.settings.showGridSettingsWindow)
  {
    if(ImGui::Begin(ICON_MS_GRID_VIEW " Grid & Snap", &m_resources.settings.showGridSettingsWindow))
    {
      Settings& s = m_resources.settings;
      if(PE::begin())
      {
        PE::DragFloat("Grid Unit", &s.gridUnit, 0.01f, 0.001f, 1000.0f, "%.3f");
        ImGui::Separator();
        PE::Checkbox("Enable Snapping", &s.snapEnabled);
        ImGui::BeginDisabled(!s.snapEnabled);
        PE::Text("Translation", "%.3f (= Grid Unit)", s.gridUnit);
        PE::DragFloat("Rotation (deg)", &s.snapRotation, 1.0f, 1.0f, 180.0f, "%.1f");
        PE::DragFloat("Scale", &s.snapScale, 0.01f, 0.001f, 10.0f, "%.3f");
        ImGui::EndDisabled();
        PE::end();
      }
    }
    ImGui::End();
  }

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

  // Application-level delete confirmation (works regardless of which windows are open)
  if(m_openDeletePopupNextFrame)
  {
    ImGui::OpenPopup("DeleteConfirmation");
    m_openDeletePopupNextFrame = false;
  }
  if(ImGui::BeginPopupModal("DeleteConfirmation", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
  {
    if(m_pendingDeleteNode >= 0 && m_resources.getScene()
       && m_pendingDeleteNode < static_cast<int>(m_resources.getScene()->getModel().nodes.size()))
    {
      const tinygltf::Model& model = m_resources.getScene()->getModel();
      ImGui::Text("Delete node '%s'?", model.nodes[m_pendingDeleteNode].name.c_str());

      // One local matrix per render node, so per primitive for an EXT_mesh_gpu_instancing mesh.
      const tinygltf::Node& pending    = model.nodes[m_pendingDeleteNode];
      const auto&           gpuInstMap = m_resources.getScene()->getRenderNodeLocalMatrices();
      auto                  instIt     = gpuInstMap.find(m_pendingDeleteNode);
      if(instIt != gpuInstMap.end() && !instIt->second.empty() && pending.mesh >= 0)
      {
        const size_t primitives = std::max<size_t>(1, model.meshes[pending.mesh].primitives.size());
        ImGui::TextColored(ImVec4(1.f, 0.6f, 0.2f, 1.f),
                           ICON_MS_WARNING
                           " This node uses EXT_mesh_gpu_instancing (%zu instances).\n"
                           "Deleting it will remove ALL instances.",
                           instIt->second.size() / primitives);
      }

      ImGui::Text("This will delete the node and all its children.");
      ImGui::Text("You can undo this with Ctrl+Z.");
      ImGui::Separator();

      if(ImGui::Button(ICON_MS_DELETE " Delete###confirmDelete", ImVec2(120, 0)))  // ### id for UI-test scenarios
      {
        auto cmd = std::make_unique<DeleteNodeCommand>(*m_resources.getScene(), m_pendingDeleteNode, &m_sceneSelection);
        m_undoStack.executeCommand(std::move(cmd));
        m_pendingDeleteNode = -1;
        onUndoRedo();
        ImGui::CloseCurrentPopup();
      }
      ImGui::SameLine();
      if(ImGui::Button(ICON_MS_CANCEL " Cancel", ImVec2(120, 0)))
      {
        m_pendingDeleteNode = -1;
        ImGui::CloseCurrentPopup();
      }
    }
    else
    {
      ImGui::Text("Nothing to delete.");
      if(ImGui::Button("Close", ImVec2(120, 0)))
      {
        m_pendingDeleteNode = -1;
        ImGui::CloseCurrentPopup();
      }
    }
    ImGui::EndPopup();
  }

  // Windows > Reset All to Default: not undoable, so confirm before discarding the user's tuning.
  if(m_openResetAllPopupNextFrame)
  {
    ImGui::OpenPopup("ResetAllConfirmation");
    m_openResetAllPopupNextFrame = false;
  }
  if(ImGui::BeginPopupModal("ResetAllConfirmation", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
  {
    ImGui::Text("Reset every setting to its default value and restore the default panel layout?");
    ImGui::Text("The loaded scene and environment image stay loaded, but the settings that drive them");
    ImGui::Text("-- including the sky/HDR choice, lighting and background -- go back to their defaults.");
    ImGui::TextDisabled("This cannot be undone with Ctrl+Z.");
    ImGui::Separator();
    if(ImGui::Button(ICON_MS_SETTINGS_BACKUP_RESTORE " Reset###confirmResetAll", ImVec2(120, 0)))
    {
      m_pendingResetSettings = true;
      m_pendingResetLayout   = true;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if(ImGui::Button(ICON_MS_CANCEL " Cancel", ImVec2(120, 0)))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
}

void GltfRenderer::renderFileMenu(std::filesystem::path& sceneToLoadFilename)
{
  if(!ImGui::BeginMenu("File"))
    return;
  m_commands.menuItem("file.new");
  m_commands.menuItem("file.open");
  // Import/Merge and Reference always available: with no scene they create one from the imported file
  // (addSceneFromFile builds a fresh Scene), otherwise they add into the current scene.
  m_commands.menuItem("file.merge");
  m_commands.menuItem("file.reference");
  m_commands.menuItem("file.loadHdr");
  if(ImGui::BeginMenu(ICON_MS_HISTORY " Open Recent"))
  {
    for(const auto& file : m_recentFiles)
    {
      if(ImGui::MenuItem(file.string().c_str()))
        sceneToLoadFilename = file;
    }
    ImGui::EndMenu();
  }
  m_commands.menuItem("file.save");
  m_commands.menuItem("file.saveAs");
  // glTF 2.1: only meaningful when the scene references external assets.
  if(hasValidScene() && m_resources.getScene()->hasExternalAssets())
    m_commands.menuItem("file.saveSelfContained");
  ImGui::Separator();
  m_commands.menuItem("file.saveImage");
  m_commands.menuItem("file.saveScreen");
  ImGui::Separator();
  m_commands.menuItem("file.exit");
  ImGui::EndMenu();
}

void GltfRenderer::renderEditMenu()
{
  if(!ImGui::BeginMenu("Edit"))
    return;
  const std::string undoLabel = m_undoStack.canUndo() ? "Undo " + m_undoStack.undoDescription() : "Undo";
  const std::string redoLabel = m_undoStack.canRedo() ? "Redo " + m_undoStack.redoDescription() : "Redo";
  m_commands.menuItem("edit.undo", undoLabel.c_str());
  m_commands.menuItem("edit.redo", redoLabel.c_str());
  ImGui::Separator();
  m_commands.menuItem("edit.duplicate");
  m_commands.menuItem("edit.delete");
  ImGui::EndMenu();
}

void GltfRenderer::renderCreateMenu()
{
  if(!ImGui::BeginMenu("Create"))
    return;

  // Global creation: always adds at the scene root (never under the current selection). Each catalog
  // action runs the browser's before-create hook (ensureEmptyScene), so "Create" works with nothing
  // loaded; the first object then triggers the initial GPU build via reconcileGeometryIfNeeded().
  m_sceneBrowser.renderCreateCatalog(-1);

  ImGui::EndMenu();
}

void GltfRenderer::renderViewMenu()
{
  if(!ImGui::BeginMenu("View"))
    return;
  m_commands.menuItem("view.fitScene");
  m_commands.menuItem("view.fitObject");
  ImGui::Separator();
  m_commands.menuItem("view.vsync");
  m_commands.menuItem("view.axis");
  ImGui::Separator();
  m_commands.menuItem("view.grid");
  m_commands.menuItem("view.gizmo");
  m_commands.menuItem("view.snap");
  m_commands.menuItem("view.animationStrip");
  ImGui::Separator();
  // Only a request: updatePresentationMode() applies it at the top of the next UI pass.
  m_commands.menuItem("view.presentation");
  ImGui::EndMenu();
}

void GltfRenderer::renderWindowsMenu()
{
  if(!ImGui::BeginMenu("Windows"))
    return;
  for(const ui::Command& cmd : m_commands.all())
    if(isWindowToggle(cmd))
      m_commands.menuItem(cmd.id);
  ImGui::MenuItem(ICON_MS_ANALYTICS " Statistics", nullptr, &m_resources.settings.showStatisticsWindow);
  ImGui::MenuItem(ICON_MS_GRID_VIEW " Grid & Snap", nullptr, &m_resources.settings.showGridSettingsWindow);
  ImGui::Separator();
  ImGui::MenuItem(ICON_MS_MONITORING " Memory Usage", nullptr, &m_resources.settings.showMemStats);
  ImGui::Separator();
  if(ImGui::MenuItem(ICON_MS_DASHBOARD " Reset UI Layout"))
    m_pendingResetLayout = true;
  ImGui::SetItemTooltip("Dock every panel back where a fresh run puts it. Settings are left alone.");
  if(ImGui::MenuItem(ICON_MS_SETTINGS_BACKUP_RESTORE " Reset All to Default"))
    m_openResetAllPopupNextFrame = true;
  ImGui::SetItemTooltip(
      "Put the application back in its out-of-the-box state: every setting at its default and the "
      "default panel layout, as if starting with no ImGui.ini. The loaded scene and environment "
      "image stay loaded, but the settings that drive them reset too.");
  ImGui::EndMenu();
}

// Consume the deferred Windows > Reset requests. Called at the top of the UI pass, so the docking
// tree is rebuilt (and the settings are back at their defaults) before any panel is submitted this
// frame.
//
// Resetting the settings writes storage directly, exactly as an ImGui.ini restore does, so the
// same follow-up applies: replay the post-restore hooks for derived state (see settings_registry.hpp)
// and restart accumulation. Everything else the renderers read per frame -- active renderer,
// visualization mode, shader specialization -- is sampled from the settings each frame, so there is
// nothing further to invalidate.
void GltfRenderer::applyPendingResets()
{
  // Both resets rewrite panel visibility and the dock tree, which presentation mode has saved and
  // hidden. Leave it first (only reachable from a script or MCP: the menu is hidden while presenting),
  // so the reset lands on the real layout instead of being overwritten by the restore.
  if((m_pendingResetSettings || m_pendingResetLayout) && m_presentation.isActive())
    exitPresentationMode();
  if(m_pendingResetSettings)
  {
    m_pendingResetSettings = false;
    m_settings.resetToDefaults();
    m_settings.runPostRestoreHooks();
    resetFrame();
    notify("Settings reset to default", false);
  }
  if(m_pendingResetLayout)
  {
    m_pendingResetLayout = false;
    ui::resetDockLayout();
  }
}

void GltfRenderer::updatePresentationMode()
{
  // Harmless in headless runs: there is no window to make full screen and no panel to hide.
  if(isHeadlessMode())
  {
    m_presentationModeRequested = false;
    return;
  }

  // F11 toggles: its command is polled here (KeyScope::eAlways), not by the menu bar, which is
  // hidden in this mode. Esc leaves (KeyScope::ePresentation), polled only while the mode is on: Esc
  // otherwise clears the selection, a command polled by renderMenu(), which does not run while the
  // menu bar is hidden, so the two never claim Esc in the same frame.
  m_commands.dispatch(ui::KeyScope::eAlways);
  if(m_presentation.isActive())
    m_commands.dispatch(ui::KeyScope::ePresentation);

  m_presentation.setActive(m_app, m_presentationModeRequested);
  m_presentationModeRequested = m_presentation.isActive();  // reads back the real state (e.g. if entering was refused)
}

void GltfRenderer::exitPresentationMode()
{
  m_presentation.setActive(m_app, false);
  m_presentationModeRequested = m_presentation.isActive();
}

void GltfRenderer::addPresentationWindow(std::function<bool()> isVisible, std::function<void(bool)> setVisible)
{
  m_presentation.addWindow(std::move(isVisible), std::move(setVisible));
}

// (The Agentic feature has no top-level menu; its window is toggled from the
//  Windows menu / F7 and rendered by agentic::renderAgenticWindow in ui_agentic.cpp.)

void GltfRenderer::renderToolsMenu(bool validScene)
{
  if(!ImGui::BeginMenu("Tools"))
    return;
  m_commands.menuItem("tools.reloadShaders");
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
  m_commands.menuItem("tools.compactScene");
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
    ImGui::Separator();
    ImGui::MenuItem("GPU Compute Animation", nullptr, &m_resources.sceneGpu.useComputeAnimation);
    ImGui::MenuItem("GPU Compute Transformation", nullptr, &m_resources.sceneGpu.useComputeTransformation);
    ImGui::EndMenu();
  }
#endif
}

void GltfRenderer::renderMenuToolbarAndGizmos()
{
  // A row of icon buttons, one per command, lit while the command is checked.
  auto toggleButtons = [this](std::span<const char* const> ids, ImVec2 buttonSize, auto&& tooltip) {
    const ImGuiStyle& style = ImGui::GetStyle();
    for(size_t i = 0; i < ids.size(); ++i)
    {
      const ui::Command& cmd = m_commands.get(ids[i]);
      ImGui::PushStyleColor(ImGuiCol_Button, cmd.checked() ? style.Colors[ImGuiCol_ButtonActive] : style.Colors[ImGuiCol_ChildBg]);
      if(ImGui::Button(cmd.icon, buttonSize))
        cmd.run();
      ImGui::PopStyleColor();
      if(ImGui::IsItemHovered())
        tooltip(cmd);
      if(i < ids.size() - 1)
        ImGui::SameLine(0, 0);
    }
  };

  std::vector<const char*> windowIds;
  for(const ui::Command& cmd : m_commands.all())
    if(isWindowToggle(cmd))
      windowIds.push_back(cmd.id);

  float             buttonSize     = ImGui::GetFrameHeight();
  const ImGuiStyle& style          = ImGui::GetStyle();
  float             separatorWidth = 2.0f;
  float totalWidth  = separatorWidth + static_cast<float>(windowIds.size()) * (buttonSize + style.ItemSpacing.x);
  float windowWidth = ImGui::GetWindowWidth();
  float offsetX     = windowWidth * 0.5f - totalWidth * 0.5f;
  ImGui::SameLine(offsetX);
  ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
  ImGui::SameLine();
  toggleButtons(windowIds, ImVec2(0, 0), [](const ui::Command& cmd) {
    ImGui::SetTooltip("Toggle %s Window (%s)", cmd.label, ui::shortcutText(cmd.binding).c_str());
  });
  ImGui::SameLine();
  ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
  ImGui::SameLine();
  static constexpr const char* kGizmoIds[] = {"view.grid", "view.gizmo", "view.snap", "view.animationStrip"};
  toggleButtons(kGizmoIds, ImVec2(buttonSize, buttonSize), [](const ui::Command& cmd) {
    if(cmd.binding.isBound())
      ImGui::SetTooltip("Toggle %s (%s in viewport)", cmd.label, ui::shortcutText(cmd.binding).c_str());
    else
      ImGui::SetTooltip("Toggle %s", cmd.label);
  });

  // KHR_interactivity Play/Pause indicator - only drawn when the loaded scene actually has a graph
  // (unlike the toggles above, which are always present). Toggles InteractivityControl::play
  // directly; does not open the Interactivity window (View > Windows > Interactivity, or F8).
  if(ui::interactivity::hasGraph(m_resources.getScene()))
  {
    ImGui::SameLine(0, 0);
    ui::interactivity::renderToolbarIndicator(m_resources.getScene(), m_resources.interactivityControl);
  }

  ImGui::SameLine();
  ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
}

//--------------------------------------------------------------------------------------------------
// The commands of the menus, the toolbar and the keyboard (ui_commands.hpp). The order is the menus'
// order, which is also the order Help lists the shortcuts in.
void GltfRenderer::registerCommands()
{
  using ui::KeyScope;
  constexpr ImGuiKeyChord kCtrl  = ImGuiMod_Ctrl;
  constexpr ImGuiKeyChord kShift = ImGuiMod_Shift;
  constexpr ImGuiKeyChord kAlt   = ImGuiMod_Alt;

  auto validScene = [this] { return hasValidScene(); };
  auto window     = [&](const char* id, const char* label, const char* icon, ImGuiKey key, bool& visible) {
    m_commands.add(
        {.id = id, .label = label, .icon = icon, .group = "Windows", .binding = {key}, .run = toggleFlag(visible), .checked = flagIsOn(visible)});
  };
  Settings& settings = m_resources.settings;

  // File
  m_commands.add({.id      = "file.new",
                  .label   = "New Scene",
                  .icon    = ICON_MS_FILTER_NONE,
                  .group   = "File",
                  .binding = {kCtrl | ImGuiKey_N},
                  .run     = requestFlag(m_menuRequests.newScene)});
  m_commands.add({.id      = "file.open",
                  .label   = "Open",
                  .icon    = ICON_MS_FILE_OPEN,
                  .group   = "File",
                  .binding = {kCtrl | ImGuiKey_O},
                  .run     = requestFlag(m_menuRequests.openFile)});
  m_commands.add({.id      = "file.merge",
                  .label   = "Import/Merge Scene...",
                  .icon    = ICON_MS_ADD,
                  .group   = "File",
                  .tooltip = "Embed another glTF into this scene, or open it as a new scene if none is loaded",
                  .binding = {.chord = kShift, .text = "Drop file"},
                  .run     = requestFlag(m_menuRequests.mergeFile)});
  m_commands.add({.id    = "file.reference",
                  .label = "Reference Scene...",
                  .icon  = ICON_MS_LINK,
                  .group = "File",
                  .tooltip = "Link another glTF as a read-only external asset (glTF 2.1). Repeats share geometry; kept as a reference on save",
                  .binding = {.chord = kCtrl | kShift, .text = "Drop file"},
                  .run     = requestFlag(m_menuRequests.referenceFile)});
  m_commands.add({.id      = "file.loadHdr",
                  .label   = "Load HDR Environment",
                  .icon    = ICON_MS_IMAGE,
                  .group   = "File",
                  .binding = {kCtrl | kShift | ImGuiKey_O},
                  .run     = requestFlag(m_menuRequests.loadHdrFile)});
  m_commands.add({.id      = "file.save",
                  .label   = "Save",
                  .icon    = ICON_MS_SAVE,
                  .group   = "File",
                  .binding = {kCtrl | ImGuiKey_S},
                  .run     = requestFlag(m_menuRequests.saveFile),
                  .enabled = validScene});
  m_commands.add({.id      = "file.saveAs",
                  .label   = "Save As...",
                  .icon    = ICON_MS_FILE_SAVE,
                  .group   = "File",
                  .binding = {kCtrl | kShift | ImGuiKey_S},
                  .run     = requestFlag(m_menuRequests.saveAsFile),
                  .enabled = validScene});
  // glTF 2.1: bakes the external assets inline and drops all references, producing a portable file
  // that can be shared without the referenced files.
  m_commands.add({.id    = "file.saveSelfContained",
                  .label = "Save Self-Contained As...",
                  .icon  = ICON_MS_FILE_SAVE,
                  .group = "File",
                  .tooltip = "Bake referenced external assets into the file and remove all external references (shareable copy)",
                  .run     = requestFlag(m_menuRequests.saveSelfContainedAsFile),
                  .enabled = validScene});
  m_commands.add({.id      = "file.saveImage",
                  .label   = "Save Image",
                  .icon    = ICON_MS_IMAGE,
                  .group   = "File",
                  .binding = {kCtrl | kAlt | ImGuiKey_I},
                  .run     = requestFlag(m_menuRequests.saveImageFile)});
  m_commands.add({.id      = "file.saveScreen",
                  .label   = "Save Screen Image",
                  .icon    = ICON_MS_DESKTOP_WINDOWS,
                  .group   = "File",
                  .binding = {kCtrl | kAlt | kShift | ImGuiKey_I},
                  .run     = requestFlag(m_menuRequests.saveScreenFile)});
  m_commands.add(
      {.id = "file.exit", .label = "Exit", .icon = ICON_MS_POWER_SETTINGS_NEW, .group = "File", .binding = {kCtrl | ImGuiKey_Q}, .run = [this] {
         m_app->close();
       }});

  // Edit. Duplicate and Delete act on the node of a node or primitive selection.
  m_commands.add({.id      = "edit.undo",
                  .label   = "Undo",
                  .icon    = ICON_MS_UNDO,
                  .group   = "Edit",
                  .binding = {kCtrl | ImGuiKey_Z},
                  .run =
                      [this] {
                        m_undoStack.undo();
                        onUndoRedo();
                      },
                  .enabled = [this] { return hasValidScene() && m_undoStack.canUndo(); }});
  m_commands.add({.id      = "edit.redo",
                  .label   = "Redo",
                  .icon    = ICON_MS_REDO,
                  .group   = "Edit",
                  .binding = {kCtrl | ImGuiKey_Y},
                  .run =
                      [this] {
                        m_undoStack.redo();
                        onUndoRedo();
                      },
                  .enabled = [this] { return hasValidScene() && m_undoStack.canRedo(); }});
  auto nodeSelected = [this] { return selectedNodeIndex() >= 0; };
  m_commands.add({.id      = "edit.duplicate",
                  .label   = "Duplicate",
                  .icon    = ICON_MS_CONTENT_COPY,
                  .group   = "Edit",
                  .binding = {kCtrl | ImGuiKey_D},
                  .run =
                      [this] {
                        m_undoStack.executeCommand(std::make_unique<DuplicateNodeCommand>(*m_resources.getScene(),
                                                                                          selectedNodeIndex(), &m_sceneSelection));
                        onUndoRedo();
                      },
                  .enabled = nodeSelected});
  m_commands.add({.id      = "edit.delete",
                  .label   = "Delete",
                  .icon    = ICON_MS_DELETE,
                  .group   = "Edit",
                  .binding = {ImGuiKey_Delete},
                  .run =
                      [this] {
                        m_pendingDeleteNode        = selectedNodeIndex();
                        m_openDeletePopupNextFrame = true;
                      },
                  .enabled = nodeSelected});
  m_commands.add({.id = "edit.clearSelection", .label = "Clear Selection", .group = "Edit", .binding = {ImGuiKey_Escape}, .run = [this] {
                    m_resources.selectedRenderNodes.clear();
                    m_sceneSelection.clearSelection();
                  }});

  // View
  m_commands.add({.id      = "view.fitScene",
                  .label   = "Fit Scene",
                  .icon    = ICON_MS_ZOOM_OUT,
                  .group   = "View",
                  .binding = {kCtrl | kShift | ImGuiKey_F},
                  .run     = [this] { fitCamera(true); },
                  .enabled = validScene});
  m_commands.add({.id      = "view.fitObject",
                  .label   = "Fit Object",
                  .icon    = ICON_MS_ZOOM_IN,
                  .group   = "View",
                  .binding = {kCtrl | ImGuiKey_F},
                  .run     = [this] { fitCamera(false); },
                  .enabled = [this] { return hasValidScene() && !m_resources.selectedRenderNodes.empty(); }});
  m_commands.add({.id      = "view.vsync",
                  .label   = "V-Sync",
                  .icon    = ICON_MS_BOTTOM_PANEL_OPEN,
                  .group   = "View",
                  .binding = {kCtrl | kShift | ImGuiKey_V},
                  .run     = [this] { m_app->setVsync(!m_app->isVsync()); },
                  .checked = [this] { return m_app->isVsync(); }});
  m_commands.add({.id      = "view.axis",
                  .label   = "3D-Axis",
                  .icon    = ICON_MS_VIEW_IN_AR,
                  .group   = "View",
                  .run     = toggleFlag(settings.showAxis),
                  .checked = flagIsOn(settings.showAxis)});
  m_commands.add({.id      = "view.grid",
                  .label   = "Grid",
                  .icon    = ICON_MS_GRID_ON,
                  .group   = "View",
                  .binding = {ImGuiKey_G, KeyScope::eViewport},
                  .run     = toggleFlag(settings.showGrid),
                  .checked = flagIsOn(settings.showGrid)});
  m_commands.add({.id      = "view.gizmo",
                  .label   = "Gizmo",
                  .icon    = ICON_MS_3D_ROTATION,
                  .group   = "View",
                  .binding = {ImGuiKey_T, KeyScope::eViewport},
                  .run     = toggleFlag(settings.showGizmo),
                  .checked = flagIsOn(settings.showGizmo)});
  m_commands.add({.id      = "view.snap",
                  .label   = "Snap",
                  .icon    = ICON_MS_STRAIGHTEN,
                  .group   = "View",
                  .run     = toggleFlag(settings.snapEnabled),
                  .checked = flagIsOn(settings.snapEnabled)});
  m_commands.add({.id      = "view.animationStrip",
                  .label   = "Animation Strip",
                  .icon    = ICON_MS_MOVIE,
                  .group   = "View",
                  .run     = toggleFlag(m_resources.animationControl.showStrip),
                  .checked = flagIsOn(m_resources.animationControl.showStrip)});
  // Only a request: updatePresentationMode() applies it, and polls F11 itself (KeyScope::eAlways)
  // since the menu bar is hidden while it is on.
  m_commands.add({.id    = "view.presentation",
                  .label = "Presentation Mode",
                  .icon  = ICON_MS_PRESENT_TO_ALL,
                  .group = "View",
                  .tooltip = "Viewport only: hide every panel and the menu bar, borderless full screen on this monitor.\n"
                             "F11 or Esc to leave.",
                  .binding = {ImGuiKey_F11, KeyScope::eAlways},
                  .run     = [this] { m_presentationModeRequested = !m_presentation.isActive(); },
                  .checked = flagIsOn(m_presentationModeRequested)});
  // Esc is left to ImGui while it closes a popup or cancels a text edit.
  m_commands.add({.id      = "view.leavePresentation",
                  .label   = "Leave Presentation Mode",
                  .group   = "View",
                  .tooltip = "Only in presentation mode.",
                  .binding = {ImGuiKey_Escape, KeyScope::ePresentation},
                  .run     = [this] { m_presentationModeRequested = false; },
                  .enabled =
                      [this] {
                        return m_presentationModeRequested && !ImGui::GetIO().WantTextInput
                               && !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId);
                      }});
  // Same as the Camera window's Home button.
  m_commands.add({.id      = "view.homeCamera",
                  .label   = "Home Camera",
                  .group   = "View",
                  .binding = {ImGuiKey_Home, KeyScope::eViewport},
                  .run     = [this] { m_cameraManip->setCamera(nvgui::GetCameras()[0], false); },
                  .enabled = [] { return !nvgui::GetCameras().empty(); }});
  m_commands.add({.id      = "view.playPause",
                  .label   = "Play / Pause Animation",
                  .group   = "View",
                  .binding = {ImGuiKey_Space, KeyScope::eViewport},
                  .run =
                      [this] {
                        m_resources.animationControl.togglePlay();
                        m_animPlay = m_resources.animationControl.play;  // Remembered for the next scene
                      },
                  .enabled = [this] { return ui::animation::hasPlayableAnimation(m_resources.getScene()); }});

  // Windows, in menu and toolbar order. The Agentic window only exists when USE_AGENTIC is built, so
  // a non-Agentic build has no dead control.
  window("window.camera", "Camera", ICON_MS_PHOTO_CAMERA, ImGuiKey_F1, settings.showCameraWindow);
  window("window.sceneBrowser", "Scene Browser", ICON_MS_ACCOUNT_TREE, ImGuiKey_F2, settings.showSceneBrowserWindow);
  window("window.settings", "Settings", ICON_MS_SETTINGS, ImGuiKey_F3, settings.showSettingsWindow);
  window("window.inspector", "Inspector", ICON_MS_LIST_ALT, ImGuiKey_F4, settings.showInspectorWindow);
  window("window.environment", "Environment", ICON_MS_PUBLIC, ImGuiKey_F5, settings.showEnvironmentWindow);
  window("window.tonemapper", "Tonemapper", ICON_MS_TONALITY, ImGuiKey_F6, settings.showTonemapperWindow);
  window("window.interactivity", "Interactivity", ICON_MS_EXTENSION, ImGuiKey_F8, settings.showInteractivityWindow);
#ifdef USE_AGENTIC
  window("window.agentic", "Agentic", ICON_MS_AUTO_AWESOME, ImGuiKey_F7, settings.showAgenticWindow);
#endif
  // Handled by nvapp::ElementLogger; listed so Help shows it.
  m_commands.add({.id = "window.log", .label = "Log", .group = "Windows", .binding = {kCtrl | kShift | ImGuiKey_L}});

  // Tools
  m_commands.add({.id      = "tools.reloadShaders",
                  .label   = "Reload Shaders",
                  .icon    = ICON_MS_REFRESH,
                  .group   = "Tools",
                  .binding = {kCtrl | kShift | ImGuiKey_R},
                  .run     = requestFlag(m_menuRequests.reloadShaders)});
  m_commands.add({.id      = "tools.compactScene",
                  .label   = "Compact Scene",
                  .icon    = ICON_MS_COMPRESS,
                  .group   = "Tools",
                  .tooltip = "Remove orphaned resources (meshes, materials, textures, images, geometry).\n"
                             "Use after deleting nodes or importing/merging scenes.",
                  .binding = {kCtrl | ImGuiKey_K},
                  .run     = requestFlag(m_menuRequests.compactScene),
                  .enabled = validScene});

  // Help. '?' is matched as a typed character by ui::help::renderMenu(), so it follows the keyboard
  // layout; listed here for the menu text and for Help itself.
  m_commands.add(
      {.id = "help.keyboard", .label = "Keyboard & Mouse", .icon = ICON_MS_KEYBOARD, .group = "Help", .binding = {.text = "?"}});
}

bool GltfRenderer::hasValidScene()
{
  return m_resources.getScene() && m_resources.getScene()->valid();
}

int GltfRenderer::selectedNodeIndex()
{
  if(!hasValidScene() || !m_sceneSelection.hasSelection())
    return -1;
  const auto sel = m_sceneSelection.getSelection();
  if(sel.type == SceneSelection::SelectionType::eNode || sel.type == SceneSelection::SelectionType::ePrimitive)
    return sel.nodeIndex;
  return -1;
}

// Frame the whole scene, or the selected render nodes. A selection with no readable bounds leaves the camera alone.
void GltfRenderer::fitCamera(bool wholeScene)
{
  const nvutils::Bbox bbox = wholeScene ? m_resources.getScene()->getSceneBounds() :
                                          GltfRenderer::getRenderNodesBbox(m_resources.selectedRenderNodes);
  if(!bbox.isEmpty())
    m_cameraManip->fit(bbox.min(), bbox.max(), false, true, m_cameraManip->getAspectRatio());
}

void GltfRenderer::renderMenu()
{
  // The global key bindings; the viewport's are polled by the viewport window, F11 by
  // updatePresentationMode(). A command only raises a request (m_menuRequests) where its work has
  // to wait for the menus to close.
  m_commands.dispatch(ui::KeyScope::eGlobal);

  std::filesystem::path sceneToLoadFilename{};
  GltfRenderer::windowTitle();
  const bool validScene = hasValidScene();

  renderFileMenu(sceneToLoadFilename);
  renderEditMenu();
  renderCreateMenu();
  renderViewMenu();
  renderWindowsMenu();
  renderToolsMenu(validScene);
  renderDebugMenu();
  ui::help::renderMenu(m_help, m_commands);
  renderMenuToolbarAndGizmos();

  MenuRequests req = std::exchange(m_menuRequests, {});

  // Shared "Save As"-style dialog wrapper: opens with defaultPath pre-filled and, if given a
  // remember-slot, updates it with the user's choice so later saves keep offering that name.
  auto saveFileDialog = [this](const char* title, const char* filter, const std::filesystem::path& defaultPath,
                               std::filesystem::path* rememberInto = nullptr) {
    std::filesystem::path filename = nvgui::windowSaveFileDialog(m_app->getWindowHandle(), title, filter, defaultPath);
    if(!filename.empty() && rememberInto)
      *rememberInto = filename;
    return filename;
  };
  auto isGltfExt = [](const std::filesystem::path& p) {
    return nvutils::extensionMatches(p, ".gltf") || nvutils::extensionMatches(p, ".glb");
  };

  auto getSaveImage = [this, &saveFileDialog]() {
    return saveFileDialog("Save Image", "Image Files|*.png;*.jpg;*.hdr|PNG Files|*.png|JPG Files|*.jpg|HDR Files|*.hdr",
                          m_imageSaveFilename, &m_imageSaveFilename);
  };

  if(req.newScene)
  {
    // SYNC NOTE: User-initiated new scene — wait before full cleanup.
    vkQueueWaitIdle(m_app->getQueue(0).queue);
    cleanupScene();
    m_sceneSelection.clearSelection();
    m_imageSaveFilename.clear();
  }

  if(req.reloadShaders)
  {
    SCOPED_BANNER("Reload Shaders");
    reloadShaders();  // owns the GPU drain and the frame reset
  }

  if(req.openFile)
  {
    const std::filesystem::path& currentScene =
        m_resources.getScene() ? m_resources.getScene()->getFilename() : std::filesystem::path{};
    sceneToLoadFilename = nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Load 3D Scene",
                                                      "3D Scene Files|*.gltf;*.glb;*.obj;*.scene.json|glTF|*.gltf;*.glb|OBJ|*.obj|Scene Descriptor|*.scene.json",
                                                      m_lastSceneDirectory, currentScene);
  }
  if(!sceneToLoadFilename.empty())
  {
    onFileDrop(sceneToLoadFilename.c_str());
  }

  if(req.mergeFile)
  {
    std::filesystem::path filename =
        nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Import/Merge Scene",
                                    "3D Scene Files|*.gltf;*.glb|glTF Text|*.gltf|glTF Binary|*.glb", m_lastSceneDirectory);
    if(!filename.empty())
      onMergeScene(filename);
  }

  if(req.referenceFile)
  {
    std::filesystem::path filename =
        nvgui::windowOpenFileDialog(m_app->getWindowHandle(), "Reference Scene (External Asset)",
                                    "glTF Files|*.gltf;*.glb|glTF Text|*.gltf|glTF Binary|*.glb", m_lastSceneDirectory);
    if(!filename.empty())
      onReferenceScene(filename);
  }

  if(req.loadHdrFile)
  {
    loadHdrFileDialog();
  }

  if(req.saveFile && validScene)
  {
    // "Save": write directly to the current scene file when it is a glTF/glb we can overwrite.
    // Otherwise (no path yet, or a non-glTF source like .obj/.scene.json) fall back to "Save As".
    const std::filesystem::path& currentFilename = m_resources.getScene()->getFilename();
    if(!currentFilename.empty() && isGltfExt(currentFilename))
      save(currentFilename);
    else
      req.saveAsFile = true;
  }

  // Default filename offered by the "Save As" dialogs: the current scene file when it's already a
  // glTF/glb we can round-trip, otherwise the loaded file's stem with a .gltf extension (e.g. after
  // loading an .obj) — matches getSaveImage()'s "default name follows the loaded scene" behavior.
  auto getSceneDefaultFilename = [this, &isGltfExt]() {
    const std::filesystem::path& currentFilename = m_resources.getScene()->getFilename();
    return isGltfExt(currentFilename) ? currentFilename : std::filesystem::path(currentFilename.stem()).replace_extension(".gltf");
  };
  static constexpr const char* kGltfFilter = "glTF Files|*.gltf;*.glb|glTF Text|*.gltf|glTF Binary|*.glb";

  if(req.saveAsFile && validScene)
  {
    std::filesystem::path filename = saveFileDialog("Save glTF", kGltfFilter, getSceneDefaultFilename());
    if(!filename.empty())
      save(filename);
  }

  if(req.saveSelfContainedAsFile && validScene)
  {
    // glTF 2.1: write a portable copy with external assets baked inline (no external references).
    std::filesystem::path filename = saveFileDialog("Save Self-Contained glTF", kGltfFilter, getSceneDefaultFilename());
    if(!filename.empty())
      save(filename, /*selfContained=*/true);
  }

  if(req.saveScreenFile)
  {
    std::filesystem::path filename = getSaveImage();
    if(!filename.empty())
    {
      m_app->requestScreenShot(filename, 100);
    }
  }

  if(req.saveImageFile)
  {
    std::filesystem::path filename = getSaveImage();
    if(!filename.empty())
    {
      saveViewportImage(filename);
    }
  }

  if(req.compactScene && validScene)
  {
    SCOPED_BANNER("Compact Scene");
    if(m_resources.getScene()->compactModel())
    {
      // Compact renumbers every index-based reference (meshes, materials, textures, lights, ...),
      // so any queued undo command holding pre-compact indices is now unsafe -- drop history.
      m_undoStack.clear();
      // Compact rewires accessors — re-parse so render nodes / dirty flags match before GPU rebuild.
      refreshCpuSceneGraphFromModel();
      rebuildVulkanSceneFull();
      resetFrame();  // Reset path tracer accumulation
    }
  }

  // Let both renderers handle their menus
  m_pathTracer.onUIMenu();
  m_rasterizer.onUIMenu();
}

// (renderAgenticWindow moved to agentic::renderAgenticWindow in src/ui_agentic.cpp.)


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
void GltfRenderer::applyCameraString(const std::string& text)
{
  if(text.empty())
    return;
  nvutils::CameraManipulator::Camera cam = m_cameraManip->getCamera();
  if(!cam.setFromString(text))
  {
    LOGW("cameraString not understood: %s\n", text.c_str());
    return;
  }
  m_cameraManip->setCamera(cam);
  resetFrame();
}

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
    double                             fov = glm::degrees(camera.yfov);
    nvutils::CameraManipulator::Camera cam{camera.eye, camera.center, camera.up, fov};
    cam.projectionType = nvutils::CameraManipulator::Perspective;
    m_cameraManip->setCamera(cam);
    m_cameraManip->setClipPlanes({camera.znear, camera.zfar});
  }
  else if(camera.type == nvvkgltf::RenderCamera::CameraType::eOrthographic)
  {
    double                             fov = 45.0;
    nvutils::CameraManipulator::Camera cam{camera.eye, camera.center, camera.up, fov};
    cam.projectionType = nvutils::CameraManipulator::Orthographic;
    cam.orthMag        = {camera.xmag, camera.ymag};
    m_cameraManip->setCamera(cam);
    m_cameraManip->adjustOrthographicAspect();
    m_cameraManip->setClipPlanes({camera.znear, camera.zfar});
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
  glm::dvec2                         clipPlanes  = m_cameraManip->getClipPlanes();

  // Update the camera parameters
  if(cameraState.projectionType == nvutils::CameraManipulator::Orthographic)
  {
    camera.type                         = "orthographic";
    tinygltf::OrthographicCamera& ortho = camera.orthographic;
    ortho.xmag                          = cameraState.orthMag.x;
    ortho.ymag                          = cameraState.orthMag.y;
    ortho.znear                         = clipPlanes.x;
    ortho.zfar                          = clipPlanes.y;
  }
  else
  {
    camera.type                        = "perspective";
    tinygltf::PerspectiveCamera& persp = camera.perspective;
    // Convert FOV from degrees (manipulator) to radians (GLTF)
    persp.yfov  = glm::radians(cameraState.fov);
    persp.znear = clipPlanes.x;
    persp.zfar  = clipPlanes.y;
    // Aspect ratio will be calculated from viewport when camera is applied
  }

  // Update the node transformation to match the current camera position
  // The camera's eye position becomes the node's translation
  node.translation = {cameraState.eye.x, cameraState.eye.y, cameraState.eye.z};

  // Calculate the rotation that points the camera from eye to center with the given up vector
  glm::dvec3 forward = glm::normalize(cameraState.ctr - cameraState.eye);
  glm::dvec3 right   = glm::normalize(glm::cross(forward, cameraState.up));
  glm::dvec3 up      = glm::normalize(glm::cross(right, forward));

  // Create rotation matrix and convert to quaternion
  glm::dmat3 rotationMatrix(right, up, -forward);  // -forward because GLTF cameras look down -Z
  glm::dquat rotation = glm::quat_cast(rotationMatrix);

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

  // Get memory trackers from subsystems
  const auto& vkTracker        = m_resources.sceneVk.getMemoryTracker();
  const auto& rtxTracker       = m_resources.sceneRtx.getMemoryTracker();
  const auto& transformTracker = m_resources.transformCompute.getMemoryTracker();
  const auto& animationTracker = m_resources.animationVk.getMemoryTracker();
  const auto& appTracker       = m_resources.appMemoryTracker;

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

    // --- Transform (GPU compute) Section ---
    auto totalTransform = transformTracker.getTotalStats();
    if(totalTransform.currentBytes > 0)
    {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "TRANSFORM (GPU)");
      ImGui::TableNextColumn();
      ImGui::TableNextColumn();

      for(const auto& categoryName : transformTracker.getActiveCategories(sortBy, ascending))
      {
        GpuMemoryStats stats = transformTracker.getStats(categoryName);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("  %s", categoryName.c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatBytes(stats.currentBytes).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%u", stats.currentCount);
      }

      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "  Transform Subtotal");
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "%s", formatBytes(totalTransform.currentBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(0.7f, 1.0f, 0.7f, 1.0f), "%u", totalTransform.currentCount);
    }

    // --- Animation (GPU compute) Section ---
    auto totalAnimation = animationTracker.getTotalStats();
    if(totalAnimation.currentBytes > 0)
    {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "ANIMATION (GPU)");
      ImGui::TableNextColumn();
      ImGui::TableNextColumn();

      for(const auto& categoryName : animationTracker.getActiveCategories(sortBy, ascending))
      {
        GpuMemoryStats stats = animationTracker.getStats(categoryName);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("  %s", categoryName.c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatBytes(stats.currentBytes).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%u", stats.currentCount);
      }

      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.6f, 1.0f), "  Animation Subtotal");
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.6f, 1.0f), "%s", formatBytes(totalAnimation.currentBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.6f, 1.0f), "%u", totalAnimation.currentCount);
    }

    // --- Application (frame targets, denoisers) Section ---
    auto totalApp = appTracker.getTotalStats();
    // OptiX export memory is only compiled in when USE_OPTIX_DENOISER is enabled (see CMakeLists.txt).
    GpuMemoryStats totalOptixExport{};
#if defined(USE_OPTIX_DENOISER)
    if(const OptiXDenoiser* optix = m_pathTracer.getOptiXDenoiser())
      totalOptixExport = optix->getExportMemoryTracker().getTotalStats();
#endif

    uint64_t appCombinedBytes = totalApp.currentBytes + totalOptixExport.currentBytes;
    uint32_t appCombinedCount = totalApp.currentCount + totalOptixExport.currentCount;
    if(appCombinedBytes > 0)
    {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1.0f), "APPLICATION");
      ImGui::TableNextColumn();
      ImGui::TableNextColumn();

      for(const auto& categoryName : appTracker.getActiveCategories(sortBy, ascending))
      {
        GpuMemoryStats stats = appTracker.getStats(categoryName);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("  %s", categoryName.c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%s", formatBytes(stats.currentBytes).c_str());
        ImGui::TableNextColumn();
        ImGui::Text("%u", stats.currentCount);
      }

#if defined(USE_OPTIX_DENOISER)
      if(const OptiXDenoiser* optix = m_pathTracer.getOptiXDenoiser())
      {
        for(const auto& categoryName : optix->getExportMemoryTracker().getActiveCategories(sortBy, ascending))
        {
          GpuMemoryStats stats = optix->getExportMemoryTracker().getStats(categoryName);
          ImGui::TableNextRow();
          ImGui::TableNextColumn();
          ImGui::Text("  %s", categoryName.c_str());
          ImGui::TableNextColumn();
          ImGui::Text("%s", formatBytes(stats.currentBytes).c_str());
          ImGui::TableNextColumn();
          ImGui::Text("%u", stats.currentCount);
        }
      }
#endif

      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(0.9f, 0.8f, 1.0f, 1.0f), "  Application Subtotal");
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(0.9f, 0.8f, 1.0f, 1.0f), "%s", formatBytes(appCombinedBytes).c_str());
      ImGui::TableNextColumn();
      ImGui::TextColored(ImVec4(0.9f, 0.8f, 1.0f, 1.0f), "%u", appCombinedCount);
    }

    // --- Combined Total ---
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "TOTAL");
    ImGui::TableNextColumn();
    auto combinedBytes = totalVk.currentBytes + totalRtx.currentBytes + totalTransform.currentBytes
                         + totalAnimation.currentBytes + appCombinedBytes;
    auto combinedCount = totalVk.currentCount + totalRtx.currentCount + totalTransform.currentCount
                         + totalAnimation.currentCount + appCombinedCount;
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%s", formatBytes(combinedBytes).c_str());
    ImGui::TableNextColumn();
    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "%u", combinedCount);

    ImGui::EndTable();
  }

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
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed);  // auto-fit: counts can be long
    ImGui::TableHeadersRow();

    // Lambda function to add table rows
    auto addStatRow = [](const char* icon, const char* label, uint64_t value) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::Text("%s %s", icon, label);
      ImGui::TableSetColumnIndex(1);
      ImGui::TextUnformatted(formatThousands(value).c_str());
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
    auto logStat = [](const char* label, uint64_t value) {
      ImGui::LogText("%s: %s\n", label, formatThousands(value).c_str());
    };
    ImGui::LogText("Scene Statistics:\n");
    logStat("Nodes", tiny.nodes.size());
    logStat("Render Nodes", m_resources.getScene()->getRenderNodes().size());
    logStat("Render Primitives", m_resources.getScene()->getNumRenderPrimitives());
    logStat("Materials", tiny.materials.size());
    logStat("Triangles", m_resources.getScene()->getNumTriangles());
    logStat("Lights", tiny.lights.size());
    logStat("Textures", tiny.textures.size());
    logStat("Images", tiny.images.size());
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
      // Compact renumbers every index-based reference; queued undo commands would restore stale
      // indices. Mirror the Tools > Compact Scene path and drop history.
      m_undoStack.clear();
      refreshCpuSceneGraphFromModel();
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
