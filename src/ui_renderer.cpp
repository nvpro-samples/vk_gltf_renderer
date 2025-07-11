/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <filesystem>
#include <fmt/format.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>

#include "create_tangent.hpp"
#include "nvgui/axis.hpp"
#include "nvgui/file_dialog.hpp"
#include "nvgui/tonemapper.hpp"
#include "nvutils/bounding_box.hpp"
#include "renderer.hpp"
#include "ui_collapsing_header_manager.h"
#include "ui_mouse_state.hpp"
#include "ui_renderer.hpp"

void GltfRendererUI::mouseClickedInViewport(GltfRenderer& renderer)
{
  static UiMouseState s_mouseClickState;  // Mouse click state

  s_mouseClickState.update();

  if(!renderer.m_resources.scene.valid())
  {
    return;
  }

  // If double-clicking in the "Viewport", shoot a ray to the scene under the mouse.
  // If the ray hit something, set the camera center to the hit position.
  if(ImGui::IsWindowHovered(ImGuiFocusedFlags_RootWindow) && (s_mouseClickState.isMouseClicked(ImGuiMouseButton_Left)))
  {
    nvutils::ScopedTimer st("RayPicker");
    VkCommandBuffer      cmd = renderer.m_app->createTempCmdBuffer();
    // Convert screen coordinates to normalized viewport coordinates [0,1]
    ImVec2 mousePos      = ImGui::GetMousePos();
    ImVec2 cursorPos     = ImGui::GetCursorScreenPos();
    ImVec2 avail         = ImGui::GetContentRegionAvail();
    ImVec2 localMousePos = ImVec2((mousePos.x - cursorPos.x) / avail.x, (mousePos.y - cursorPos.y) / avail.y);

    renderer.m_rayPicker.run(cmd, {.modelViewInv = glm::inverse(renderer.m_resources.cameraManip->getViewMatrix()),
                                   .perspectiveInv = glm::inverse(renderer.m_resources.cameraManip->getPerspectiveMatrix()),
                                   .pickPos = {localMousePos.x, localMousePos.y},
                                   .tlas    = renderer.m_resources.sceneRtx.tlas()});
    renderer.m_app->submitAndWaitTempCmdBuffer(cmd);
    nvvk::RayPicker::PickResult pickResult = renderer.m_rayPicker.getResult();

    // Set or de-select the selected object
    if(s_mouseClickState.isMouseSingleClicked(ImGuiMouseButton_Left))
    {
      renderer.m_resources.selectedObject = pickResult.instanceID;
      int nodeID                          = -1;
      if(pickResult.instanceID > -1)
      {
        const nvvkgltf::RenderNode& renderNode = renderer.m_resources.scene.getRenderNodes()[pickResult.instanceID];
        nodeID                                 = renderNode.refNodeID;
      }
      renderer.m_uiSceneGraph.selectNode(nodeID);
    }

    // Environment was picked (no hit)
    if(pickResult.instanceID < 0)
      return;

    glm::vec3 worldPos = pickResult.worldRayOrigin + pickResult.worldRayDirection * pickResult.hitT;
    if(s_mouseClickState.isMouseDoubleClicked(ImGuiMouseButton_Left))
    {
      // Set the camera CENTER to the hit position
      glm::vec3 eye, center, up;
      renderer.m_resources.cameraManip->getLookat(eye, center, up);
      renderer.m_resources.cameraManip->setLookat(eye, worldPos, up, false);  // Nice with CameraManip.updateAnim();
    }

    {
      // Logging picking info.
      const nvvkgltf::RenderNode& renderNode = renderer.m_resources.scene.getRenderNodes()[pickResult.instanceID];
      const tinygltf::Node&       node       = renderer.m_resources.scene.getModel().nodes[renderNode.refNodeID];
      LOGI("Node Name: %s\n", node.name.c_str());
      LOGI(" - GLTF: NodeID: %d, MeshID: %d, TriangleId: %d\n", renderNode.refNodeID, node.mesh, pickResult.primitiveID);
      LOGI(" - Render: RenderNode: %d, RenderPrim: %d\n", pickResult.instanceID, pickResult.instanceCustomIndex);
      LOGI("{%3.2f, %3.2f, %3.2f}, Dist: %3.2f\n", worldPos.x, worldPos.y, worldPos.z, pickResult.hitT);

      renderer.m_uiSceneGraph.selectNode(renderNode.refNodeID);
    }
  }
}

nvutils::Bbox GltfRendererUI::getRenderNodeBbox(GltfRenderer& renderer, int nodeID)
{
  nvutils::Bbox worldBbox({-1, -1, -1}, {1, 1, 1});
  if(nodeID < 0 || !renderer.m_resources.scene.valid())
    return worldBbox;

  const nvvkgltf::RenderNode& renderNode = renderer.m_resources.scene.getRenderNodes()[nodeID];
  const nvvkgltf::RenderPrimitive& renderPrimitive = renderer.m_resources.scene.getRenderPrimitive(renderNode.renderPrimID);
  const tinygltf::Model&    model    = renderer.m_resources.scene.getModel();
  const tinygltf::Accessor& accessor = model.accessors[renderPrimitive.pPrimitive->attributes.at("POSITION")];

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

void GltfRendererUI::windowTitle(GltfRenderer& renderer)
{
  static float dirty_timer = 0.0F;
  dirty_timer += ImGui::GetIO().DeltaTime;
  if(dirty_timer > 1.0F)  // Refresh every seconds
  {
    const VkExtent2D&     size     = renderer.m_app->getViewportSize();
    std::filesystem::path filename = renderer.m_resources.scene.getFilename().filename();
    if(filename.empty())
    {
      filename = "No Scene";
    }
    std::string text = fmt::format("{} - {}x{} | {:.0f} FPS / {:.3f}ms | Frame {}", nvutils::utf8FromPath(filename),
                                   size.width, size.height, ImGui::GetIO().Framerate, 1000.F / ImGui::GetIO().Framerate,
                                   renderer.m_resources.frameCount);

    glfwSetWindowTitle(renderer.m_app->getWindowHandle(), text.c_str());
    dirty_timer = 0;
  }
}

void GltfRendererUI::renderUI(GltfRenderer& renderer)
{
  static int frameCount    = 0;
  auto&      headerManager = CollapsingHeaderManager::getInstance();

  namespace PE = nvgui::PropertyEditor;
  {  // Setting menu
    bool changed{false};

    if(ImGui::Begin("Camera"))
    {
      nvgui::CameraWidget(renderer.m_resources.cameraManip);
    }
    ImGui::End();  // End Camera

    // Scene Graph UI
    {
      int selectedNode = renderer.m_uiSceneGraph.selectedNode();

      // Use event-based approach for maximum decoupling
      // The scene graph UI emits events instead of calling renderer methods directly
      // This eliminates the circular dependency and makes the code more modular
      renderer.m_uiSceneGraph.setEventCallback([&renderer](const UiSceneGraph::Event& event) {
        switch(event.type)
        {
          case UiSceneGraph::EventType::CameraApply:
            GltfRendererUI::applyGltfCamera(renderer, event.data);
            break;
          case UiSceneGraph::EventType::CameraSetFromView:
            GltfRendererUI::setGltfCameraFromView(renderer, event.data);
            break;
          case UiSceneGraph::EventType::NodeSelected:
            // Update the selected render node index when a node is selected
            {
              auto it = renderer.m_nodeToRenderNodeMap.find(event.data);
              if(it != renderer.m_nodeToRenderNodeMap.end())
              {
                renderer.m_resources.selectedObject = it->second;
              }
              else
              {
                renderer.m_resources.selectedObject = -1;  // No matching render node
              }
            }
            break;
          case UiSceneGraph::EventType::MaterialSelected:
            // [TODO] Handle material selection
            break;
        }
      });

      renderer.m_uiSceneGraph.render();
    }

    if(frameCount < 2)  // This is a hack to make the settings window focus on the first frame
    {
      ImGui::SetNextWindowFocus();
      frameCount++;
    }

    if(ImGui::Begin("Settings"))
    {
      // Add renderer selection at the top level of the Settings panel for better visibility
      if(PE::begin())
      {
        static const char* rendererItems[] = {"Path Tracer", "Rasterizer"};
        int                currentItem     = static_cast<int>(renderer.m_resources.settings.renderSystem);
        if(PE::Combo("Active Renderer", &currentItem, rendererItems, IM_ARRAYSIZE(rendererItems)))
        {
          renderer.m_resources.settings.renderSystem = static_cast<RenderingMode>(currentItem);
          changed                                    = true;  // Reset frame counter when switching renderers
        }
        changed |= PE::Combo("Debug Method", reinterpret_cast<int32_t*>(&renderer.m_resources.settings.debugMethod),
                             "None\0BaseColor\0Metallic\0Roughness\0Normal\0Tangent\0Bitangent\0Emissive\0Opacity\0TexCoord0\0TexCoord1\0\0");
        PE::end();
        if(renderer.m_resources.settings.renderSystem == RenderingMode::ePathtracer)
        {
          if(headerManager.beginHeader("Path Tracer"))
          {
            changed |= renderer.m_pathTracer.onUIRender(renderer.m_resources);
            PE::begin();
            PE::Text("Current frame", std::to_string(renderer.m_resources.frameCount));
            changed |= PE::SliderInt("Max Iterations", &renderer.m_resources.settings.maxFrames, 0, 10000, "%d", 0,
                                     "Maximum number of iterations");
            PE::end();
          }
        }
        else
        {
          if(headerManager.beginHeader("Rasterizer"))
            changed |= renderer.m_rasterizer.onUIRender(renderer.m_resources);
        }
      }
      ImGui::Separator();


      if(headerManager.beginHeader("Environment"))
      {
        if(PE::begin())
        {
          if(PE::Combo("Environment Type", (int*)&renderer.m_resources.settings.envSystem, "Sky\0HDR\0\0"))  // 0: Sky, 1: HDR
          {
            renderer.m_pathTracer.m_pushConst.fireflyClampThreshold =
                (renderer.m_resources.settings.envSystem == shaderio::EnvSystem::eSky) ?
                    10.0f :
                    renderer.m_resources.hdrIbl.getIntegral();
            changed |= true;
          }
          changed |= PE::Checkbox("Solid Color", &renderer.m_resources.settings.useSolidBackground);
          if(renderer.m_resources.settings.useSolidBackground)
          {
            changed |= PE::ColorEdit3("Background Color", glm::value_ptr(renderer.m_resources.settings.solidBackgroundColor));
          }
          PE::end();
        }

        if(renderer.m_resources.settings.envSystem == shaderio::EnvSystem::eHdr)
        {
          if(PE::begin("HDR"))
          {
            if(PE::entry("", [&] { return ImGui::SmallButton("load"); }, "Load HDR Image"))
            {
              std::filesystem::path filename =
                  nvgui::windowOpenFileDialog(renderer.m_app->getWindowHandle(), "Load HDR Image", "HDR(.hdr)|*.hdr");
              if(!filename.empty())
              {
                renderer.onFileDrop(filename.c_str());
              }
              changed = true;
            }
            changed |= PE::SliderFloat("Intensity", &renderer.m_resources.settings.hdrEnvIntensity, 0, 100, "%.3f",
                                       ImGuiSliderFlags_Logarithmic, "HDR intensity");
            changed |= PE::SliderAngle("Rotation", &renderer.m_resources.settings.hdrEnvRotation, -360, 360, "%.0f deg",
                                       0, "Rotating the environment");
            changed |= PE::SliderFloat("Blur", &renderer.m_resources.settings.hdrBlur, 0, 1, "%.3f", 0, "Blur the environment");
            PE::end();
          }
        }
        else
        {
          changed |= nvgui::skyPhysicalParameterUI(renderer.m_resources.skyParams);
        }
      }

      if(headerManager.beginHeader("Tonemapper"))
      {
        nvgui::tonemapperWidget(renderer.m_resources.tonemapperData);
      }

      // Multiple scenes
      if(renderer.m_resources.scene.getModel().scenes.size() > 1)
      {
        if(headerManager.beginHeader("Multiple Scenes"))
        {
          ImGui::PushID("Scenes");
          for(size_t i = 0; i < renderer.m_resources.scene.getModel().scenes.size(); i++)
          {
            if(ImGui::RadioButton(renderer.m_resources.scene.getModel().scenes[i].name.c_str(),
                                  renderer.m_resources.scene.getCurrentScene() == i))
            {
              renderer.m_resources.scene.setCurrentScene(int(i));
              vkDeviceWaitIdle(renderer.m_device);
              renderer.createVulkanScene();
              renderer.updateTextures();
              changed = true;
            }
          }
          ImGui::PopID();
        }
      }

      // Variant selection
      if(renderer.m_resources.scene.getVariants().size() > 0)
      {
        if(headerManager.beginHeader("Variants"))
        {
          ImGui::PushID("Variants");
          for(size_t i = 0; i < renderer.m_resources.scene.getVariants().size(); i++)
          {
            if(ImGui::Selectable(renderer.m_resources.scene.getVariants()[i].c_str(),
                                 renderer.m_resources.scene.getCurrentVariant() == i))
            {
              renderer.m_resources.scene.setCurrentVariant(int(i));
              renderer.m_resources.dirtyFlags.set(DirtyFlags::eVulkanScene);
              changed = true;
            }
          }
          ImGui::PopID();
        }
      }

      // Animation
      if(renderer.m_resources.scene.hasAnimation())
      {
        if(headerManager.beginHeader("Animation"))
        {
          renderer.m_animControl.onUI(&renderer.m_resources.scene);
        }
      }

      if(renderer.m_resources.scene.valid() && headerManager.beginHeader("Statistics"))
      {
        if(PE::begin("Stat_Val"))
        {
          const tinygltf::Model& tiny = renderer.m_resources.scene.getModel();
          PE::Text("Nodes", std::to_string(tiny.nodes.size()));
          PE::Text("Render Nodes", std::to_string(renderer.m_resources.scene.getRenderNodes().size()));
          PE::Text("Render Primitives", std::to_string(renderer.m_resources.scene.getNumRenderPrimitives()));
          PE::Text("Materials", std::to_string(tiny.materials.size()));
          PE::Text("Triangles", std::to_string(renderer.m_resources.scene.getNumTriangles()));
          PE::Text("Lights", std::to_string(tiny.lights.size()));
          PE::Text("Textures", std::to_string(tiny.textures.size()));
          PE::Text("Images", std::to_string(tiny.images.size()));
          PE::end();
        }
      }
    }
    ImGui::End();  // End Settings

    if(changed)
      renderer.resetFrame();
  }


  // Show the rendered tonemapped image in the viewport
  {  // Rendering Viewport
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");

    // Handle mouse clicks in the viewport
    GltfRendererUI::mouseClickedInViewport(renderer);

    // Display the G-Buffer tonemapped image
    ImGui::Image(ImTextureID(renderer.m_resources.gBuffers.getDescriptorSet(Resources::eImgTonemapped)),
                 ImGui::GetContentRegionAvail());

    // Adding Axis at the bottom left corner of the viewport
    if(renderer.m_resources.settings.showAxis)
    {
      nvgui::Axis(renderer.m_resources.cameraManip->getViewMatrix(), 25.f);
    }

    ImGui::End();
    ImGui::PopStyleVar();
  }


  // Show the busy window if the busy flag is set
  if(renderer.m_busy.isBusy())
  {
    renderer.m_busy.show();
  }
}

void GltfRendererUI::renderMenu(GltfRenderer& renderer)
{
  bool v_sync = renderer.m_app->isVsync();

  auto getSaveImage = [&]() {
    std::filesystem::path filename =
        nvgui::windowSaveFileDialog(renderer.m_app->getWindowHandle(), "Save Image", "PNG(.png),JPG(.jpg)|*.png;*.jpg");
    if(!filename.empty())
    {
      std::filesystem::path ext = std::filesystem::path(filename).extension();
    }
    return filename;
  };

  GltfRendererUI::windowTitle(renderer);
  bool clearScene     = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_N);
  bool loadFile       = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_O);
  bool saveFile       = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S);
  bool saveScreenFile = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiMod_Alt | ImGuiKey_S);
  bool saveImageFile  = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_S);
  bool closeApp       = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Q);
  bool fitScene       = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_F);
  bool fitObject      = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_F);
  bool toggleVsyc     = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_V);
  bool reloadShader   = ImGui::IsKeyPressed(ImGuiKey_F5);

  if(toggleVsyc)
  {
    v_sync = !v_sync;
  }
  bool validScene = renderer.m_resources.scene.valid();
  if(ImGui::BeginMenu("File"))
  {
    clearScene = ImGui::MenuItem("Clear Scene", "Ctrl+N");
    loadFile |= ImGui::MenuItem("Load", "Ctrl+O");
    ImGui::BeginDisabled(!validScene);  // Disable menu item if no scene is loaded
    saveFile |= ImGui::MenuItem("Save As", "Ctrl+S");
    ImGui::EndDisabled();
    ImGui::Separator();
    saveImageFile |= ImGui::MenuItem("Save Image", "Ctrl+Shift+S");
    saveScreenFile |= ImGui::MenuItem("Save Screen", "Ctrl+Alt+Shift+S");
    ImGui::Separator();
    closeApp |= ImGui::MenuItem("Exit", "Ctrl+Q");
    ImGui::EndMenu();
  }

  // De-selecting the object
  if(ImGui::IsKeyPressed(ImGuiKey_Escape))
  {
    renderer.m_resources.selectedObject = -1;
    renderer.m_uiSceneGraph.selectNode(-1);
  }

  if(ImGui::BeginMenu("View"))
  {
    ImGui::BeginDisabled(!validScene);  // Disable menu item if no scene is loaded)
    fitScene |= ImGui::MenuItem("Fit Scene", "Ctrl+Shift+F");
    ImGui::BeginDisabled(renderer.m_resources.selectedObject < 0);  // Disable menu item if no object is selected
    fitObject |= ImGui::MenuItem("Fit Object", "Ctrl+F");
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    ImGui::Separator();
    ImGui::MenuItem("V-Sync", "Ctrl+Shift+V", &v_sync);
    ImGui::MenuItem("3D-Axis", nullptr, &renderer.m_resources.settings.showAxis);
    ImGui::EndMenu();
  }

  if(ImGui::BeginMenu("Tools"))
  {
    reloadShader |= ImGui::MenuItem("Reload Shaders", "F5");
    ImGui::Separator();
    ImGui::BeginDisabled(!validScene);  // Disable menu item if no scene is loaded)

    if(ImGui::MenuItem("Recreate Tangents - Simple"))
    {
      recomputeTangents(renderer.m_resources.scene.getModel(), true, false);
      renderer.m_resources.dirtyFlags.set(DirtyFlags::eVulkanScene);
    }
    ImGui::SetItemTooltip("This recreate tangents using UV gradient method");
    if(ImGui::MenuItem("Recreate Tangents - MikkTSpace"))
    {
      recomputeTangents(renderer.m_resources.scene.getModel(), true, true);
      renderer.m_resources.dirtyFlags.set(DirtyFlags::eVulkanScene);
    }
    ImGui::SetItemTooltip("This recreate tangents using MikkTSpace");

    ImGui::EndDisabled();

    ImGui::EndMenu();
  }

  if(clearScene)
  {
    vkQueueWaitIdle(renderer.m_app->getQueue(0).queue);
    renderer.m_resources.scene.destroy();
    renderer.m_resources.sceneVk.destroy();
    renderer.m_resources.sceneRtx.destroy();
    renderer.m_resources.dirtyFlags.set(DirtyFlags::eVulkanScene);
    renderer.m_resources.selectedObject = -1;
    renderer.m_uiSceneGraph.selectNode(-1);
  }

  if(reloadShader)
  {
    vkQueueWaitIdle(renderer.m_app->getQueue(0).queue);
    renderer.compileShaders();
    renderer.resetFrame();
  }

  if(loadFile)
  {
    std::filesystem::path filename = nvgui::windowOpenFileDialog(renderer.m_app->getWindowHandle(), "Load glTF | HDR",
                                                                 "glTF(.gltf, .glb), OBJ(.obj), "
                                                                 "HDR(.hdr)|*.gltf;*.glb;*.obj;*.hdr");
    renderer.onFileDrop(filename.c_str());
  }

  if(saveFile && validScene)
  {
    std::filesystem::path filename =
        nvgui::windowSaveFileDialog(renderer.m_app->getWindowHandle(), "Save glTF", "glTF(.gltf, .glb)|*.gltf;*.glb");
    if(!filename.empty())
      renderer.save(filename);
  }

  if(saveScreenFile)
  {
    std::filesystem::path filename = getSaveImage();
    if(!filename.empty())
    {
      renderer.m_app->screenShot(filename, 100);
    }
  }

  if(saveImageFile)
  {
    std::filesystem::path filename = getSaveImage();
    if(!filename.empty())
    {
      renderer.m_app->saveImageToFile(renderer.m_resources.gBuffers.getColorImage(Resources::eImgTonemapped),
                                      renderer.m_resources.gBuffers.getSize(), filename);
    }
  }

  if(validScene && (fitScene || (fitObject && renderer.m_resources.selectedObject > -1)))
  {
    nvutils::Bbox bbox = fitScene ? renderer.m_resources.scene.getSceneBounds() :
                                    GltfRendererUI::getRenderNodeBbox(renderer, renderer.m_resources.selectedObject);
    renderer.m_resources.cameraManip->fit(bbox.min(), bbox.max(), false, true, renderer.m_resources.cameraManip->getAspectRatio());
  }

  if(closeApp)
  {
    renderer.m_app->close();
  }

  if(renderer.m_app->isVsync() != v_sync)
  {
    renderer.m_app->setVsync(v_sync);
  }

  // Let both renderers handle their menus
  renderer.m_pathTracer.onUIMenu();
  renderer.m_rasterizer.onUIMenu();
}

//--------------------------------------------------------------------------------------------------
// Apply a GLTF camera to the camera manipulator
// This function converts GLTF camera parameters to camera manipulator settings
// and applies them to the current view
//
void GltfRendererUI::applyGltfCamera(GltfRenderer& renderer, int cameraIndex)
{
  if(!renderer.m_resources.scene.valid())
    return;

  // Get the node that contains this camera to clear its extras
  int nodeIndex = renderer.m_uiSceneGraph.getNodeForCamera(cameraIndex);
  if(nodeIndex >= 0)
  {
    // Clear the camera extras so that eye/center/up are recalculated from the world matrix
    tinygltf::Node& node = renderer.m_resources.scene.getModel().nodes[nodeIndex];
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
  const std::vector<nvvkgltf::RenderCamera>& cameras = renderer.m_resources.scene.getRenderCameras(true);

  if(cameraIndex < 0 || cameraIndex >= static_cast<int>(cameras.size()))
    return;

  const nvvkgltf::RenderCamera& camera = cameras[cameraIndex];

  if(camera.type == nvvkgltf::RenderCamera::CameraType::ePerspective)
  {
    float fov = static_cast<float>(glm::degrees(camera.yfov));
    renderer.m_resources.cameraManip->setCamera({camera.eye, camera.center, camera.up, fov});
    renderer.m_resources.cameraManip->setClipPlanes({static_cast<float>(camera.znear), static_cast<float>(camera.zfar)});
  }
  else if(camera.type == nvvkgltf::RenderCamera::CameraType::eOrthographic)
  {
    float fov = 45.0f;
    renderer.m_resources.cameraManip->setCamera({camera.eye, camera.center, camera.up, fov});
    renderer.m_resources.cameraManip->setClipPlanes({static_cast<float>(camera.znear), static_cast<float>(camera.zfar)});
  }

  // Also update the scene's camera to keep extras (eye, center, up) in sync
  renderer.m_resources.scene.setSceneCamera(camera);
}

//--------------------------------------------------------------------------------------------------
// Set a GLTF camera from the current camera manipulator state
// This function updates the GLTF camera parameters and node extras with the current view
//
void GltfRendererUI::setGltfCameraFromView(GltfRenderer& renderer, int cameraIndex)
{
  if(!renderer.m_resources.scene.valid() || !renderer.m_resources.cameraManip)
  {
    return;
  }

  // Get the node that contains this camera
  int nodeIndex = renderer.m_uiSceneGraph.getNodeForCamera(cameraIndex);
  if(nodeIndex < 0)
  {
    return;
  }

  tinygltf::Node&   node   = renderer.m_resources.scene.getModel().nodes[nodeIndex];
  tinygltf::Camera& camera = renderer.m_resources.scene.getModel().cameras[cameraIndex];

  // Get current camera state from manipulator
  nvutils::CameraManipulator::Camera cameraState = renderer.m_resources.cameraManip->getCamera();
  glm::vec2                          clipPlanes  = renderer.m_resources.cameraManip->getClipPlanes();

  // Update the camera parameters
  if(camera.type == "perspective")
  {
    tinygltf::PerspectiveCamera& persp = camera.perspective;
    // Convert FOV from degrees (manipulator) to radians (GLTF)
    persp.yfov  = glm::radians(cameraState.fov);
    persp.znear = static_cast<double>(clipPlanes.x);
    persp.zfar  = static_cast<double>(clipPlanes.y);
    // Aspect ratio will be calculated from viewport when camera is applied
  }
  else if(camera.type == "orthographic")
  {
    tinygltf::OrthographicCamera& ortho = camera.orthographic;
    // For orthographic cameras, is it not yet supported
    ortho.znear = static_cast<double>(clipPlanes.x);
    ortho.zfar  = static_cast<double>(clipPlanes.y);
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