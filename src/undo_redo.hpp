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

#include "gltf_scene_editor.hpp"  // PrimitiveKind, PrimitiveParams, ModelTailSizes, SceneGraphSnapshot

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
// ReplaceImageCommand - Undo/redo for replacing an image's pixels in place (SceneEditor::replaceImageFromFile)
//--------------------------------------------------------------------------------------------------

// Undo/redo for an in-place swap of one element of a model resource vector (image / sampler / texture).
// All three edits just flag DirtyFlags::texturesChanged; the renderer does the GPU rebuild. GetVec
// returns the target vector from the model. See the aliases below.
template <typename T, std::vector<T>& (*GetVec)(tinygltf::Model&)>
class TextureResourceEditCommand : public ICommand
{
public:
  TextureResourceEditCommand(nvvkgltf::Scene& scene, int index, const T& oldValue, const T& newValue, std::string description)
      : m_scene(scene)
      , m_index(index)
      , m_old(std::make_unique<T>(oldValue))
      , m_new(std::make_unique<T>(newValue))
      , m_description(std::move(description))
  {
  }

  void                      execute() override { restore(*m_new); }
  void                      undo() override { restore(*m_old); }
  [[nodiscard]] std::string description() const override { return m_description; }

private:
  void restore(const T& value)
  {
    std::vector<T>& vec = GetVec(m_scene.getModel());
    if(m_index >= 0 && m_index < static_cast<int>(vec.size()))
    {
      vec[m_index]                            = value;
      m_scene.getDirtyFlags().texturesChanged = true;
    }
  }

  nvvkgltf::Scene&   m_scene;
  int                m_index;
  std::unique_ptr<T> m_old;
  std::unique_ptr<T> m_new;
  std::string        m_description;
};

inline std::vector<tinygltf::Image>& modelImages(tinygltf::Model& m)
{
  return m.images;
}
inline std::vector<tinygltf::Sampler>& modelSamplers(tinygltf::Model& m)
{
  return m.samplers;
}
inline std::vector<tinygltf::Texture>& modelTextures(tinygltf::Model& m)
{
  return m.textures;
}

using ReplaceImageCommand = TextureResourceEditCommand<tinygltf::Image, &modelImages>;     // replace an image's pixels
using EditSamplerCommand = TextureResourceEditCommand<tinygltf::Sampler, &modelSamplers>;  // edit a sampler's wrap/filter
using EditTextureCommand = TextureResourceEditCommand<tinygltf::Texture, &modelTextures>;  // edit a texture's image/sampler ref

//--------------------------------------------------------------------------------------------------
// RemoveImageCommand - Undo/redo for removing an unreferenced image (SceneEditor::removeImageAt)
//
// The image (referenced by no texture) is removed live; undo re-inserts the stored copy at its
// original index. Both directions remap texture image sources via the editor.
//--------------------------------------------------------------------------------------------------

class RemoveImageCommand : public ICommand
{
public:
  RemoveImageCommand(nvvkgltf::Scene& scene, int imageIndex, const tinygltf::Image& removedImage);
  ~RemoveImageCommand() override;

  void                      execute() override;  // redo: remove again
  void                      undo() override;     // re-insert the stored image
  [[nodiscard]] std::string description() const override;

private:
  nvvkgltf::Scene&                 m_scene;
  int                              m_imageIndex;
  std::unique_ptr<tinygltf::Image> m_image;
};

//--------------------------------------------------------------------------------------------------
// ImportImageAsTextureCommand - Undo/redo for importing a file as a new image + texture.
//
// SceneEditor::importImageAsTexture appends both at the tail of model.images / model.textures. This
// command (pushed via pushExecuted after the import already succeeded) makes that undoable: undo()
// removes the texture then the image (both at the tail); execute() re-appends the stored copies for
// redo. The slot assignment that follows an import is recorded separately as an EditMaterialCommand.
//--------------------------------------------------------------------------------------------------

class ImportImageAsTextureCommand : public ICommand
{
public:
  ImportImageAsTextureCommand(nvvkgltf::Scene& scene, int imageIndex, int textureIndex, std::string description);
  ~ImportImageAsTextureCommand() override;

  void                      execute() override;  // redo: re-append the stored image + texture
  void                      undo() override;     // remove the appended texture + image
  [[nodiscard]] std::string description() const override { return m_description; }

private:
  nvvkgltf::Scene&                   m_scene;
  int                                m_imageIndex;
  int                                m_textureIndex;
  std::unique_ptr<tinygltf::Image>   m_image;
  std::unique_ptr<tinygltf::Texture> m_texture;
  std::string                        m_description;
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
// AddPrimitiveCommand - Undo/redo for adding a procedural primitive (plane/cube/sphere)
//
// Adding a primitive appends geometry (buffer/bufferViews/accessors), a material, a mesh and a node.
// Undo truncates the appended geometry tail first, then restores the node graph from a snapshot
// (one parseScene). Truncate-before-restore matters because buildPrimitiveKeyMap walks all meshes.
//--------------------------------------------------------------------------------------------------

class AddPrimitiveCommand : public ICommand
{
public:
  AddPrimitiveCommand(nvvkgltf::Scene&                 scene,
                      nvvkgltf::PrimitiveKind          kind,
                      const nvvkgltf::PrimitiveParams& params,
                      int                              parentIndex,
                      SceneSelection*                  selection);
  ~AddPrimitiveCommand() override;

  void                      execute() override;
  void                      undo() override;
  [[nodiscard]] std::string description() const override;

private:
  nvvkgltf::Scene&                              m_scene;
  nvvkgltf::PrimitiveKind                       m_kind;
  nvvkgltf::PrimitiveParams                     m_params;
  int                                           m_parentIndex;
  int                                           m_newNodeIndex = -1;
  SceneSelection*                               m_selection;
  nvvkgltf::ModelTailSizes                      m_tailSizes;
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
