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
// Scene selection manager. Tracks the currently selected node, primitive,
// or material in the glTF scene, converts between selection types
// (e.g. primitive pick to node selection), and emits selection-change
// events that the UI and gizmo systems subscribe to.
//

#include "scene_selection.hpp"
#include <imgui.h>
#include <string>
#include <cstring>

//--------------------------------------------------------------------------------------------------
// Select a node in the hierarchy
//
void SceneSelection::selectNode(int nodeIdx)
{
  m_selection           = {};  // Clear previous selection
  m_selection.type      = SelectionType::eNode;
  m_selection.nodeIndex = nodeIdx;

  emitEvent({EventType::NodeSelected, nodeIdx, -1});
}

//--------------------------------------------------------------------------------------------------
// Select a specific primitive instance (from 3D picking or UI)
//
void SceneSelection::selectPrimitive(int renderNodeIdx, int nodeIdx, int primIdx, int meshIdx)
{
  m_selection                 = {};
  m_selection.type            = SelectionType::ePrimitive;
  m_selection.renderNodeIndex = renderNodeIdx;
  m_selection.nodeIndex       = nodeIdx;
  m_selection.primitiveIndex  = primIdx;
  m_selection.meshIndex       = meshIdx;

  emitEvent({EventType::PrimitiveSelected, nodeIdx, renderNodeIdx});
}

//--------------------------------------------------------------------------------------------------
// Select a material (from scene list or inspector)
//
void SceneSelection::selectMaterial(int matIdx, int nodeContext)
{
  m_selection               = {};
  m_selection.type          = SelectionType::eMaterial;
  m_selection.materialIndex = matIdx;
  m_selection.nodeIndex     = nodeContext;  // Optional context

  emitEvent({EventType::MaterialSelected, matIdx, -1});
}

//--------------------------------------------------------------------------------------------------
// Select a mesh from flat list
//
void SceneSelection::selectMesh(int meshIdx)
{
  m_selection           = {};
  m_selection.type      = SelectionType::eMesh;
  m_selection.meshIndex = meshIdx;

  // No event emitted for mesh selection (internal UI only)
}

//--------------------------------------------------------------------------------------------------
// Select a camera
//
void SceneSelection::selectCamera(int camIdx)
{
  m_selection             = {};
  m_selection.type        = SelectionType::eCamera;
  m_selection.cameraIndex = camIdx;

  // No event emitted for camera selection (events are for Apply/SetFromView)
}

//--------------------------------------------------------------------------------------------------
// Select a light
//
void SceneSelection::selectLight(int lightIdx)
{
  m_selection            = {};
  m_selection.type       = SelectionType::eLight;
  m_selection.lightIndex = lightIdx;

  // No event emitted for light selection (internal UI only)
}

//--------------------------------------------------------------------------------------------------
// Clear selection
//
void SceneSelection::clearSelection()
{
  m_selection = {};
}

//--------------------------------------------------------------------------------------------------
// Copy material to system clipboard
//
void SceneSelection::copyMaterialToClipboard(int matIdx)
{
  m_clipboardMaterialIndex = matIdx;

  // Store in system clipboard as "gltf_material:index"
  std::string clipData = "gltf_material:" + std::to_string(matIdx);
  ImGui::SetClipboardText(clipData.c_str());
}

//--------------------------------------------------------------------------------------------------
// Get material from system clipboard
//
int SceneSelection::getMaterialFromClipboard() const
{
  const char* clipText = ImGui::GetClipboardText();
  if(clipText && std::strncmp(clipText, "gltf_material:", 14) == 0)
  {
    return std::atoi(clipText + 14);
  }
  return -1;  // Invalid
}

//--------------------------------------------------------------------------------------------------
// Check if clipboard contains a material
//
bool SceneSelection::hasClipboardMaterial() const
{
  return getMaterialFromClipboard() >= 0;
}

//--------------------------------------------------------------------------------------------------
// Emit event to external systems (internal helper)
//
void SceneSelection::emitEvent(const Event& event)
{
  if(m_eventCallback)
  {
    m_eventCallback(event);
  }
}

//--------------------------------------------------------------------------------------------------
// Emit camera action events (for UI buttons)
//
void SceneSelection::emitCameraApply(int cameraIdx)
{
  emitEvent({EventType::CameraApply, cameraIdx, -1});
}

void SceneSelection::emitCameraSetFromView(int cameraIdx)
{
  emitEvent({EventType::CameraSetFromView, cameraIdx, -1});
}
