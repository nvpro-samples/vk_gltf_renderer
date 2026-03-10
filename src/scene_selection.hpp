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

#pragma once

/*
 * SceneSelection - Central state manager for scene element selection
 * 
 * This class manages all selection state for the Scene Browser and Inspector.
 * It provides a single source of truth for which element is currently selected
 * and emits events to external systems (like the renderer) when selection changes.
 */

#include <functional>

class SceneSelection
{
public:
  // Types of elements that can be selected
  enum class SelectionType
  {
    eNone,
    eNode,       // Node in hierarchy
    ePrimitive,  // Specific primitive instance (node+mesh+prim)
    eMaterial,   // Material (for editing)
    eMesh,       // Mesh (flat list)
    eCamera,     // Camera
    eLight       // Light
    // Note: Animations, Textures, Images are NOT selectable (display-only)
  };

  // Complete selection context - contains all relevant indices
  struct SelectionContext
  {
    SelectionType type            = SelectionType::eNone;
    int           nodeIndex       = -1;  // For nodes, primitives (parent node)
    int           renderNodeIndex = -1;  // For primitives (render instance)
    int           primitiveIndex  = -1;  // For primitives (which prim in mesh)
    int           meshIndex       = -1;  // For meshes, primitives
    int           materialIndex   = -1;  // For materials, primitives
    int           cameraIndex     = -1;  // For cameras
    int           lightIndex      = -1;  // For lights
  };

  // Event types for external notification (decouples UI from renderer)
  enum class EventType
  {
    NodeSelected,
    PrimitiveSelected,
    MaterialSelected,
    CameraApply,       // User wants to apply camera to view
    CameraSetFromView  // User wants to set camera from current view
  };

  struct Event
  {
    EventType type;
    int       data;             // nodeIndex, materialIndex, cameraIndex, etc.
    int       renderNodeIndex;  // For PrimitiveSelected
  };

  using EventCallback = std::function<void(const Event&)>;

  SceneSelection() = default;

  // Selection API
  void selectNode(int nodeIdx);
  void selectPrimitive(int renderNodeIdx, int nodeIdx, int primIdx, int meshIdx);
  void selectMaterial(int matIdx, int nodeContext = -1);
  void selectMesh(int meshIdx);
  void selectCamera(int camIdx);
  void selectLight(int lightIdx);
  void clearSelection();

  // Event callback for external systems
  void setEventCallback(const EventCallback& callback) { m_eventCallback = callback; }

  // Queries
  [[nodiscard]] const SelectionContext& getSelection() const { return m_selection; }
  [[nodiscard]] bool                    hasSelection() const { return m_selection.type != SelectionType::eNone; }

  // Material clipboard (system clipboard integration)
  void               copyMaterialToClipboard(int matIdx);
  [[nodiscard]] int  getMaterialFromClipboard() const;
  [[nodiscard]] bool hasClipboardMaterial() const;

  // Camera action events (for UI buttons)
  void emitCameraApply(int cameraIdx);
  void emitCameraSetFromView(int cameraIdx);

private:
  SelectionContext m_selection;
  int              m_clipboardMaterialIndex = -1;
  EventCallback    m_eventCallback;

  void emitEvent(const Event& event);
};
