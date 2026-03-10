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
// Undo/Redo: command implementations for all undoable scene editing operations.
// Commands call SceneEditor methods for mutations and manage snapshots for
// reliable undo of structural operations (delete, duplicate, add node).
//

#include "undo_redo.hpp"
#include "gltf_scene_editor.hpp"
#include "scene_selection.hpp"

#include <fmt/format.h>
#include <nvutils/logger.hpp>

//--------------------------------------------------------------------------------------------------
// UndoStack
//--------------------------------------------------------------------------------------------------

void UndoStack::executeCommand(std::unique_ptr<ICommand> cmd)
{
  if(!cmd)
    return;
  cmd->execute();
  m_redoStack.clear();
  m_undoStack.push_back(std::move(cmd));
  trimToMaxSize();
}

void UndoStack::pushExecuted(std::unique_ptr<ICommand> cmd)
{
  if(!cmd)
    return;

  auto now = std::chrono::steady_clock::now();
  if(!m_undoStack.empty() && (now - m_lastPushTime) < kMergeWindow && m_undoStack.back()->canMergeWith(*cmd))
  {
    m_undoStack.back()->mergeWith(*cmd);
  }
  else
  {
    m_undoStack.push_back(std::move(cmd));
    trimToMaxSize();
  }

  m_redoStack.clear();
  m_lastPushTime = now;
}

bool UndoStack::undo()
{
  if(m_undoStack.empty())
    return false;
  auto cmd = std::move(m_undoStack.back());
  m_undoStack.pop_back();
  cmd->undo();
  m_redoStack.push_back(std::move(cmd));
  return true;
}

bool UndoStack::redo()
{
  if(m_redoStack.empty())
    return false;
  auto cmd = std::move(m_redoStack.back());
  m_redoStack.pop_back();
  cmd->execute();
  m_undoStack.push_back(std::move(cmd));
  return true;
}

void UndoStack::clear()
{
  m_undoStack.clear();
  m_redoStack.clear();
}

bool UndoStack::canUndo() const
{
  return !m_undoStack.empty();
}

bool UndoStack::canRedo() const
{
  return !m_redoStack.empty();
}

std::string UndoStack::undoDescription() const
{
  return m_undoStack.empty() ? "" : m_undoStack.back()->description();
}

std::string UndoStack::redoDescription() const
{
  return m_redoStack.empty() ? "" : m_redoStack.back()->description();
}

void UndoStack::trimToMaxSize()
{
  while(m_undoStack.size() > m_maxSize)
    m_undoStack.erase(m_undoStack.begin());
}

//--------------------------------------------------------------------------------------------------
// SetTransformCommand
//--------------------------------------------------------------------------------------------------

SetTransformCommand::SetTransformCommand(nvvkgltf::Scene& scene,
                                         int              nodeIndex,
                                         const glm::vec3& oldT,
                                         const glm::quat& oldR,
                                         const glm::vec3& oldS,
                                         const glm::vec3& newT,
                                         const glm::quat& newR,
                                         const glm::vec3& newS)
    : m_scene(scene)
    , m_nodeIndex(nodeIndex)
    , m_oldTranslation(oldT)
    , m_newTranslation(newT)
    , m_oldRotation(oldR)
    , m_newRotation(newR)
    , m_oldScale(oldS)
    , m_newScale(newS)
{
}

void SetTransformCommand::execute()
{
  m_scene.editor().setNodeTRS(m_nodeIndex, m_newTranslation, m_newRotation, m_newScale);
}

void SetTransformCommand::undo()
{
  m_scene.editor().setNodeTRS(m_nodeIndex, m_oldTranslation, m_oldRotation, m_oldScale);
}

std::string SetTransformCommand::description() const
{
  return fmt::format("Transform '{}'", m_scene.editor().getNodeName(m_nodeIndex));
}

bool SetTransformCommand::canMergeWith(const ICommand& other) const
{
  auto* o = dynamic_cast<const SetTransformCommand*>(&other);
  return o && o->m_nodeIndex == m_nodeIndex;
}

void SetTransformCommand::mergeWith(const ICommand& other)
{
  auto& o          = dynamic_cast<const SetTransformCommand&>(other);
  m_newTranslation = o.m_newTranslation;
  m_newRotation    = o.m_newRotation;
  m_newScale       = o.m_newScale;
}

//--------------------------------------------------------------------------------------------------
// RenameNodeCommand
//--------------------------------------------------------------------------------------------------

RenameNodeCommand::RenameNodeCommand(nvvkgltf::Scene& scene, int nodeIndex, const std::string& oldName, const std::string& newName)
    : m_scene(scene)
    , m_nodeIndex(nodeIndex)
    , m_oldName(oldName)
    , m_newName(newName)
{
}

void RenameNodeCommand::execute()
{
  m_scene.editor().renameNode(m_nodeIndex, m_newName);
}

void RenameNodeCommand::undo()
{
  m_scene.editor().renameNode(m_nodeIndex, m_oldName);
}

std::string RenameNodeCommand::description() const
{
  return fmt::format("Rename '{}'", m_oldName);
}

//--------------------------------------------------------------------------------------------------
// DuplicateNodeCommand
//--------------------------------------------------------------------------------------------------

DuplicateNodeCommand::DuplicateNodeCommand(nvvkgltf::Scene& scene, int originalIndex, SceneSelection* selection)
    : m_scene(scene)
    , m_originalIndex(originalIndex)
    , m_selection(selection)
{
  m_nodeName = m_scene.editor().getNodeName(originalIndex);
}

void DuplicateNodeCommand::execute()
{
  m_newIndex = m_scene.editor().duplicateNode(m_originalIndex);
  if(m_newIndex >= 0 && m_selection)
    m_selection->selectNode(m_newIndex);
}

void DuplicateNodeCommand::undo()
{
  if(m_newIndex >= 0)
  {
    if(m_selection)
      m_selection->clearSelection();
    m_scene.editor().deleteNode(m_newIndex);
    m_newIndex = -1;
    if(m_selection)
      m_selection->selectNode(m_originalIndex);
  }
}

std::string DuplicateNodeCommand::description() const
{
  return fmt::format("Duplicate '{}'", m_nodeName);
}

//--------------------------------------------------------------------------------------------------
// DeleteNodeCommand
//--------------------------------------------------------------------------------------------------

DeleteNodeCommand::DeleteNodeCommand(nvvkgltf::Scene& scene, int nodeIndex, SceneSelection* selection)
    : m_scene(scene)
    , m_nodeIndex(nodeIndex)
    , m_selection(selection)
{
  m_nodeName = m_scene.editor().getNodeName(nodeIndex);
  m_snapshot = std::make_unique<nvvkgltf::SceneGraphSnapshot>(m_scene.editor().snapshotForDelete());
}

DeleteNodeCommand::~DeleteNodeCommand() = default;

void DeleteNodeCommand::execute()
{
  if(m_selection)
    m_selection->clearSelection();
  m_scene.editor().deleteNode(m_nodeIndex);
}

void DeleteNodeCommand::undo()
{
  if(m_snapshot)
    m_scene.editor().restoreFromSnapshot(*m_snapshot);
  if(m_selection)
    m_selection->selectNode(m_nodeIndex);
}

std::string DeleteNodeCommand::description() const
{
  return fmt::format("Delete '{}'", m_nodeName);
}

//--------------------------------------------------------------------------------------------------
// AddNodeCommand
//--------------------------------------------------------------------------------------------------

AddNodeCommand::AddNodeCommand(nvvkgltf::Scene& scene, const std::string& name, int parentIndex, SceneSelection* selection)
    : m_scene(scene)
    , m_name(name)
    , m_parentIndex(parentIndex)
    , m_selection(selection)
{
}

void AddNodeCommand::execute()
{
  m_newIndex = m_scene.editor().addNode(m_name, m_parentIndex);
  if(m_newIndex >= 0 && m_selection)
    m_selection->selectNode(m_newIndex);
}

void AddNodeCommand::undo()
{
  if(m_newIndex >= 0)
  {
    if(m_selection)
      m_selection->clearSelection();
    m_scene.editor().deleteNode(m_newIndex);
    m_newIndex = -1;
  }
}

std::string AddNodeCommand::description() const
{
  return fmt::format("Add Node '{}'", m_name.empty() ? "Node" : m_name);
}

//--------------------------------------------------------------------------------------------------
// ReparentNodeCommand
//--------------------------------------------------------------------------------------------------

ReparentNodeCommand::ReparentNodeCommand(nvvkgltf::Scene& scene, int childIndex, int oldParent, int newParent)
    : m_scene(scene)
    , m_childIndex(childIndex)
    , m_oldParent(oldParent)
    , m_newParent(newParent)
{
  m_nodeName = m_scene.editor().getNodeName(childIndex);
}

void ReparentNodeCommand::execute()
{
  m_scene.editor().setNodeParent(m_childIndex, m_newParent);
}

void ReparentNodeCommand::undo()
{
  m_scene.editor().setNodeParent(m_childIndex, m_oldParent);
}

std::string ReparentNodeCommand::description() const
{
  return fmt::format("Move '{}'", m_nodeName);
}

//--------------------------------------------------------------------------------------------------
// EditMaterialCommand
//--------------------------------------------------------------------------------------------------

EditMaterialCommand::EditMaterialCommand(nvvkgltf::Scene&          scene,
                                         int                       materialIndex,
                                         const tinygltf::Material& oldMaterial,
                                         const tinygltf::Material& newMaterial)
    : m_scene(scene)
    , m_materialIndex(materialIndex)
    , m_materialName(oldMaterial.name)
    , m_oldMaterial(std::make_unique<tinygltf::Material>(oldMaterial))
    , m_newMaterial(std::make_unique<tinygltf::Material>(newMaterial))
{
}

EditMaterialCommand::~EditMaterialCommand() = default;

void EditMaterialCommand::execute()
{
  restore(*m_newMaterial);
}

void EditMaterialCommand::undo()
{
  restore(*m_oldMaterial);
}

void EditMaterialCommand::restore(const tinygltf::Material& mat)
{
  m_scene.editor().getMaterialForEdit(m_materialIndex) = mat;
  m_scene.markMaterialDirty(m_materialIndex);
  m_scene.markRenderNodeRtxDirtyForMaterials({m_materialIndex});
}

std::string EditMaterialCommand::description() const
{
  return fmt::format("Edit Material '{}'", m_materialName);
}

bool EditMaterialCommand::canMergeWith(const ICommand& other) const
{
  auto* o = dynamic_cast<const EditMaterialCommand*>(&other);
  return o && o->m_materialIndex == m_materialIndex;
}

void EditMaterialCommand::mergeWith(const ICommand& other)
{
  auto& o        = dynamic_cast<const EditMaterialCommand&>(other);
  *m_newMaterial = *o.m_newMaterial;
}

//--------------------------------------------------------------------------------------------------
// AddLightCommand
//--------------------------------------------------------------------------------------------------

AddLightCommand::AddLightCommand(nvvkgltf::Scene& scene, const std::string& lightType, const std::string& name, int parentIndex, SceneSelection* selection)
    : m_scene(scene)
    , m_lightType(lightType)
    , m_name(name)
    , m_parentIndex(parentIndex)
    , m_selection(selection)
{
  m_snapshot = std::make_unique<nvvkgltf::SceneGraphSnapshot>(m_scene.editor().snapshotForDelete());
}

AddLightCommand::~AddLightCommand() = default;

void AddLightCommand::execute()
{
  m_newNodeIndex = m_scene.editor().addLightNode(m_lightType, m_name, m_parentIndex);
  if(m_newNodeIndex >= 0 && m_selection)
    m_selection->selectNode(m_newNodeIndex);
}

void AddLightCommand::undo()
{
  if(m_snapshot)
  {
    if(m_selection)
      m_selection->clearSelection();
    m_scene.editor().restoreFromSnapshot(*m_snapshot);
    m_newNodeIndex = -1;
  }
}

std::string AddLightCommand::description() const
{
  return fmt::format("Add {} Light '{}'", m_lightType, m_name);
}

//--------------------------------------------------------------------------------------------------
// EditLightCommand
//--------------------------------------------------------------------------------------------------

EditLightCommand::EditLightCommand(nvvkgltf::Scene& scene, int lightIndex, const tinygltf::Light& oldLight, const tinygltf::Light& newLight)
    : m_scene(scene)
    , m_lightIndex(lightIndex)
    , m_lightName(oldLight.name)
    , m_oldLight(std::make_unique<tinygltf::Light>(oldLight))
    , m_newLight(std::make_unique<tinygltf::Light>(newLight))
{
}

EditLightCommand::~EditLightCommand() = default;

void EditLightCommand::execute()
{
  restore(*m_newLight);
}

void EditLightCommand::undo()
{
  restore(*m_oldLight);
}

void EditLightCommand::restore(const tinygltf::Light& light)
{
  m_scene.editor().getLightForEdit(m_lightIndex) = light;
  m_scene.markLightDirty(m_lightIndex);
}

std::string EditLightCommand::description() const
{
  return fmt::format("Edit Light '{}'", m_lightName);
}

bool EditLightCommand::canMergeWith(const ICommand& other) const
{
  auto* o = dynamic_cast<const EditLightCommand*>(&other);
  return o && o->m_lightIndex == m_lightIndex;
}

void EditLightCommand::mergeWith(const ICommand& other)
{
  auto& o     = dynamic_cast<const EditLightCommand&>(other);
  *m_newLight = *o.m_newLight;
}
