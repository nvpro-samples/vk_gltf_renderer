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
 * Undo/Redo - Command-pattern undo/redo system for scene editing
 *
 * Provides ICommand interface, UndoStack for history management, and concrete
 * command classes for all undoable scene operations: transforms, node lifecycle
 * (add/duplicate/delete), hierarchy (reparent), and rename.
 *
 * Two usage patterns:
 * - Discrete ops: undoStack.executeCommand(cmd) -- executes then pushes
 * - Continuous ops: undoStack.pushExecuted(cmd) -- pushes already-executed command
 *   (used for gizmo drag and inspector DragFloat3 where mutation happens live)
 */

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

// Forward declarations
namespace nvvkgltf {
class Scene;
struct SceneGraphSnapshot;
}  // namespace nvvkgltf

namespace tinygltf {
struct Material;
struct Light;
}  // namespace tinygltf

class SceneSelection;

//--------------------------------------------------------------------------------------------------
// ICommand - Base interface for all undoable operations
//--------------------------------------------------------------------------------------------------

class ICommand
{
public:
  virtual ~ICommand()                                   = default;
  virtual void                      execute()           = 0;
  virtual void                      undo()              = 0;
  [[nodiscard]] virtual std::string description() const = 0;

  // Merge support: consecutive commands of the same kind (e.g. dragging a slider)
  // can be merged into one undo step. Override in continuous-edit commands only.
  [[nodiscard]] virtual bool canMergeWith(const ICommand& /*other*/) const { return false; }
  virtual void               mergeWith(const ICommand& /*other*/) {}
};

//--------------------------------------------------------------------------------------------------
// UndoStack - Linear undo/redo history manager
//
// New commands clear the redo stack (linear history model).
// History is capped at m_maxSize entries; oldest commands are discarded.
//--------------------------------------------------------------------------------------------------

class UndoStack
{
public:
  void executeCommand(std::unique_ptr<ICommand> cmd);
  void pushExecuted(std::unique_ptr<ICommand> cmd);

  bool undo();
  bool redo();
  void clear();

  [[nodiscard]] bool        canUndo() const;
  [[nodiscard]] bool        canRedo() const;
  [[nodiscard]] std::string undoDescription() const;
  [[nodiscard]] std::string redoDescription() const;

private:
  std::vector<std::unique_ptr<ICommand>> m_undoStack;
  std::vector<std::unique_ptr<ICommand>> m_redoStack;
  size_t                                 m_maxSize = 100;

  // Time-limited merge: consecutive pushExecuted() calls within this window
  // are merged if the commands are compatible (same target, same type).
  static constexpr auto                 kMergeWindow = std::chrono::milliseconds(500);
  std::chrono::steady_clock::time_point m_lastPushTime{};

  void trimToMaxSize();
};

//--------------------------------------------------------------------------------------------------
// SetTransformCommand - Undo/redo for node TRS changes (gizmo + inspector)
//--------------------------------------------------------------------------------------------------

class SetTransformCommand : public ICommand
{
public:
  SetTransformCommand(nvvkgltf::Scene& scene,
                      int              nodeIndex,
                      const glm::vec3& oldT,
                      const glm::quat& oldR,
                      const glm::vec3& oldS,
                      const glm::vec3& newT,
                      const glm::quat& newR,
                      const glm::vec3& newS);

  void                      execute() override;
  void                      undo() override;
  [[nodiscard]] std::string description() const override;
  [[nodiscard]] bool        canMergeWith(const ICommand& other) const override;
  void                      mergeWith(const ICommand& other) override;

private:
  nvvkgltf::Scene& m_scene;
  int              m_nodeIndex;
  glm::vec3        m_oldTranslation, m_newTranslation;
  glm::quat        m_oldRotation, m_newRotation;
  glm::vec3        m_oldScale, m_newScale;
};

//--------------------------------------------------------------------------------------------------
// RenameNodeCommand - Undo/redo for node rename
//--------------------------------------------------------------------------------------------------

class RenameNodeCommand : public ICommand
{
public:
  RenameNodeCommand(nvvkgltf::Scene& scene, int nodeIndex, const std::string& oldName, const std::string& newName);

  void                      execute() override;
  void                      undo() override;
  [[nodiscard]] std::string description() const override;

private:
  nvvkgltf::Scene& m_scene;
  int              m_nodeIndex;
  std::string      m_oldName;
  std::string      m_newName;
};

//--------------------------------------------------------------------------------------------------
// DuplicateNodeCommand - Undo/redo for node duplication
//--------------------------------------------------------------------------------------------------

class DuplicateNodeCommand : public ICommand
{
public:
  DuplicateNodeCommand(nvvkgltf::Scene& scene, int originalIndex, SceneSelection* selection);

  void                      execute() override;
  void                      undo() override;
  [[nodiscard]] std::string description() const override;
  [[nodiscard]] int         getNewIndex() const { return m_newIndex; }

private:
  nvvkgltf::Scene& m_scene;
  int              m_originalIndex;
  int              m_newIndex = -1;
  std::string      m_nodeName;
  SceneSelection*  m_selection;
};

//--------------------------------------------------------------------------------------------------
// DeleteNodeCommand - Undo/redo for node deletion (uses snapshot for safe restore)
//
// Takes a full SceneGraphSnapshot before deletion so undo can reliably restore
// the exact pre-delete state including all index remapping side-effects.
//--------------------------------------------------------------------------------------------------

class DeleteNodeCommand : public ICommand
{
public:
  DeleteNodeCommand(nvvkgltf::Scene& scene, int nodeIndex, SceneSelection* selection);
  ~DeleteNodeCommand() override;

  void                      execute() override;
  void                      undo() override;
  [[nodiscard]] std::string description() const override;

private:
  nvvkgltf::Scene&                              m_scene;
  int                                           m_nodeIndex;
  std::string                                   m_nodeName;
  SceneSelection*                               m_selection;
  std::unique_ptr<nvvkgltf::SceneGraphSnapshot> m_snapshot;
};

//--------------------------------------------------------------------------------------------------
// AddNodeCommand - Undo/redo for adding a child node
//--------------------------------------------------------------------------------------------------

class AddNodeCommand : public ICommand
{
public:
  AddNodeCommand(nvvkgltf::Scene& scene, const std::string& name, int parentIndex, SceneSelection* selection);

  void                      execute() override;
  void                      undo() override;
  [[nodiscard]] std::string description() const override;
  [[nodiscard]] int         getNewIndex() const { return m_newIndex; }

private:
  nvvkgltf::Scene& m_scene;
  std::string      m_name;
  int              m_parentIndex;
  int              m_newIndex = -1;
  SceneSelection*  m_selection;
};

//--------------------------------------------------------------------------------------------------
// ReparentNodeCommand - Undo/redo for drag-and-drop reparenting
//--------------------------------------------------------------------------------------------------

class ReparentNodeCommand : public ICommand
{
public:
  ReparentNodeCommand(nvvkgltf::Scene& scene, int childIndex, int oldParent, int newParent);

  void                      execute() override;
  void                      undo() override;
  [[nodiscard]] std::string description() const override;

private:
  nvvkgltf::Scene& m_scene;
  int              m_childIndex;
  int              m_oldParent;
  int              m_newParent;
  std::string      m_nodeName;
};

//--------------------------------------------------------------------------------------------------
// EditMaterialCommand - Undo/redo for material property changes
//
// Uses full tinygltf::Material snapshots (before/after) so a single command
// covers all properties: PBR factors, textures, emissive, alpha, double-sided,
// and all material extensions. No per-property tracking needed.
//--------------------------------------------------------------------------------------------------

class EditMaterialCommand : public ICommand
{
public:
  EditMaterialCommand(nvvkgltf::Scene& scene, int materialIndex, const tinygltf::Material& oldMaterial, const tinygltf::Material& newMaterial);
  ~EditMaterialCommand() override;

  void                      execute() override;
  void                      undo() override;
  [[nodiscard]] std::string description() const override;
  [[nodiscard]] bool        canMergeWith(const ICommand& other) const override;
  void                      mergeWith(const ICommand& other) override;

private:
  nvvkgltf::Scene&                    m_scene;
  int                                 m_materialIndex;
  std::string                         m_materialName;
  std::unique_ptr<tinygltf::Material> m_oldMaterial;
  std::unique_ptr<tinygltf::Material> m_newMaterial;

  void restore(const tinygltf::Material& mat);
};

//--------------------------------------------------------------------------------------------------
// AddLightCommand - Undo/redo for adding a light node (snapshot-based)
//
// Takes a SceneGraphSnapshot before creation so undo can reliably restore
// the pre-add state including both the node and the light definition.
//--------------------------------------------------------------------------------------------------

class AddLightCommand : public ICommand
{
public:
  AddLightCommand(nvvkgltf::Scene& scene, const std::string& lightType, const std::string& name, int parentIndex, SceneSelection* selection);
  ~AddLightCommand() override;

  void                      execute() override;
  void                      undo() override;
  [[nodiscard]] std::string description() const override;
  [[nodiscard]] int         getNewNodeIndex() const { return m_newNodeIndex; }

private:
  nvvkgltf::Scene&                              m_scene;
  std::string                                   m_lightType;
  std::string                                   m_name;
  int                                           m_parentIndex;
  int                                           m_newNodeIndex = -1;
  SceneSelection*                               m_selection;
  std::unique_ptr<nvvkgltf::SceneGraphSnapshot> m_snapshot;
};

//--------------------------------------------------------------------------------------------------
// EditLightCommand - Undo/redo for light property changes
//
// Uses full tinygltf::Light snapshots (before/after) so a single command
// covers all properties: type, color, intensity, range, spot angles.
//--------------------------------------------------------------------------------------------------

class EditLightCommand : public ICommand
{
public:
  EditLightCommand(nvvkgltf::Scene& scene, int lightIndex, const tinygltf::Light& oldLight, const tinygltf::Light& newLight);
  ~EditLightCommand() override;

  void                      execute() override;
  void                      undo() override;
  [[nodiscard]] std::string description() const override;
  [[nodiscard]] bool        canMergeWith(const ICommand& other) const override;
  void                      mergeWith(const ICommand& other) override;

private:
  nvvkgltf::Scene&                 m_scene;
  int                              m_lightIndex;
  std::string                      m_lightName;
  std::unique_ptr<tinygltf::Light> m_oldLight;
  std::unique_ptr<tinygltf::Light> m_newLight;

  void restore(const tinygltf::Light& light);
};
