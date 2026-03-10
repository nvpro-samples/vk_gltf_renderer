/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use it except in compliance with the License.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Validates a glTF scene for structural correctness before rendering.
// Checks node hierarchy integrity, mesh/material/texture index bounds,
// accessor validity, skin references, and other constraints. Reports
// errors and warnings through a ValidationResult structure.
//

#include "gltf_scene_validator.hpp"

#include <functional>

#include <fmt/format.h>
#include <tinygltf/tiny_gltf.h>

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>

namespace nvvkgltf {

SceneValidator::SceneValidator(const Scene& scene)
    : m_scene(scene)
{
}

Scene::ValidationResult SceneValidator::validateModel() const
{
  Scene::ValidationResult result;

  validateNodes(result);
  validateScenes(result);
  validateAnimations(result);
  validateSkins(result);
  validateMeshReferences(result);
  validateMaterialReferences(result);

  return result;
}

void SceneValidator::validateNodes(Scene::ValidationResult& result) const
{
  const auto& model = m_scene.getModel();

  // Check parent-child consistency
  for(size_t i = 0; i < model.nodes.size(); ++i)
  {
    const auto& node = model.nodes[i];

    // Validate children indices
    for(int childIdx : node.children)
    {
      if(childIdx < 0 || childIdx >= static_cast<int>(model.nodes.size()))
      {
        result.addError(fmt::format("Node {} has invalid child index {}", i, childIdx));
      }

      // Check for self-reference
      if(childIdx == static_cast<int>(i))
      {
        result.addError(fmt::format("Node {} references itself as child", i));
      }
    }

    // Validate mesh reference
    if(node.mesh >= 0 && node.mesh >= static_cast<int>(model.meshes.size()))
    {
      result.addError(fmt::format("Node {} has invalid mesh index {}", i, node.mesh));
    }

    // Validate camera reference
    if(node.camera >= 0 && node.camera >= static_cast<int>(model.cameras.size()))
    {
      result.addError(fmt::format("Node {} has invalid camera index {}", i, node.camera));
    }

    // Validate skin reference
    if(node.skin >= 0 && node.skin >= static_cast<int>(model.skins.size()))
    {
      result.addError(fmt::format("Node {} has invalid skin index {}", i, node.skin));
    }
  }

  // Check for cycles in hierarchy
  std::vector<bool> visited(model.nodes.size(), false);
  std::vector<bool> inStack(model.nodes.size(), false);

  std::function<bool(int)> hasCycle = [&](int nodeIdx) -> bool {
    if(inStack[nodeIdx])
      return true;  // Cycle detected
    if(visited[nodeIdx])
      return false;

    visited[nodeIdx] = true;
    inStack[nodeIdx] = true;

    const auto& node = model.nodes[nodeIdx];
    for(int childIdx : node.children)
    {
      if(childIdx >= 0 && childIdx < static_cast<int>(model.nodes.size()))
      {
        if(hasCycle(childIdx))
          return true;
      }
    }

    inStack[nodeIdx] = false;
    return false;
  };

  for(size_t i = 0; i < model.nodes.size(); ++i)
  {
    if(hasCycle(static_cast<int>(i)))
    {
      result.addError(fmt::format("Cycle detected in node hierarchy at node {}", i));
    }
  }
}

void SceneValidator::validateScenes(Scene::ValidationResult& result) const
{
  const auto& model = m_scene.getModel();

  for(size_t i = 0; i < model.scenes.size(); ++i)
  {
    const auto& scene = model.scenes[i];

    // Check for empty scene (no root nodes) - this is invalid for rendering
    if(scene.nodes.empty())
    {
      result.addError(fmt::format("Scene {} has no root nodes - scene must have at least one root node", i));
    }

    for(int nodeIdx : scene.nodes)
    {
      if(nodeIdx < 0 || nodeIdx >= static_cast<int>(model.nodes.size()))
      {
        result.addError(fmt::format("Scene {} has invalid root node index {}", i, nodeIdx));
      }
    }
  }

  // Check default scene
  if(model.defaultScene >= 0 && model.defaultScene >= static_cast<int>(model.scenes.size()))
  {
    result.addError(fmt::format("Invalid default scene index {}", model.defaultScene));
  }
}

void SceneValidator::validateAnimations(Scene::ValidationResult& result) const
{
  const auto& model = m_scene.getModel();

  for(size_t animIdx = 0; animIdx < model.animations.size(); ++animIdx)
  {
    const auto& anim = model.animations[animIdx];

    for(size_t chanIdx = 0; chanIdx < anim.channels.size(); ++chanIdx)
    {
      const auto& channel = anim.channels[chanIdx];

      // Validate target node (pointer animations can have target_node = -1)
      int targetNode = channel.target_node;
      if(channel.target_path == "pointer")
      {
        // Pointer animations don't use target_node, they use JSON pointers
        // target_node can be -1, which is valid
      }
      else
      {
        // Standard animations must have valid target node
        if(targetNode < 0 || targetNode >= static_cast<int>(model.nodes.size()))
        {
          result.addError(fmt::format("Animation {} channel {} has invalid target node {}", animIdx, chanIdx, targetNode));
        }
      }

      // Validate sampler index
      if(channel.sampler < 0 || channel.sampler >= static_cast<int>(anim.samplers.size()))
      {
        result.addError(fmt::format("Animation {} channel {} has invalid sampler index {}", animIdx, chanIdx, channel.sampler));
      }
    }
  }
}

void SceneValidator::validateSkins(Scene::ValidationResult& result) const
{
  const auto& model = m_scene.getModel();

  for(size_t skinIdx = 0; skinIdx < model.skins.size(); ++skinIdx)
  {
    const auto& skin = model.skins[skinIdx];

    // Validate joint indices
    for(size_t jointIdx = 0; jointIdx < skin.joints.size(); ++jointIdx)
    {
      int nodeIdx = skin.joints[jointIdx];
      if(nodeIdx < 0 || nodeIdx >= static_cast<int>(model.nodes.size()))
      {
        result.addError(fmt::format("Skin {} has invalid joint index {} (node {})", skinIdx, jointIdx, nodeIdx));
      }
    }

    // Validate skeleton root
    if(skin.skeleton >= 0 && skin.skeleton >= static_cast<int>(model.nodes.size()))
    {
      result.addError(fmt::format("Skin {} has invalid skeleton root {}", skinIdx, skin.skeleton));
    }
  }
}

void SceneValidator::validateMeshReferences(Scene::ValidationResult& result) const
{
  const auto& model = m_scene.getModel();

  for(size_t meshIdx = 0; meshIdx < model.meshes.size(); ++meshIdx)
  {
    const auto& mesh = model.meshes[meshIdx];

    for(size_t primIdx = 0; primIdx < mesh.primitives.size(); ++primIdx)
    {
      const auto& prim = mesh.primitives[primIdx];

      // Validate material
      if(prim.material >= 0 && prim.material >= static_cast<int>(model.materials.size()))
      {
        result.addError(fmt::format("Mesh {} primitive {} has invalid material index {}", meshIdx, primIdx, prim.material));
      }
    }
  }
}

void SceneValidator::validateMaterialReferences(Scene::ValidationResult& result) const
{
  const auto& model = m_scene.getModel();

  // Check texture references in materials
  for(size_t matIdx = 0; matIdx < model.materials.size(); ++matIdx)
  {
    const auto& mat = model.materials[matIdx];

    auto checkTexture = [&](int texIndex, const char* name) {
      if(texIndex >= 0 && texIndex >= static_cast<int>(model.textures.size()))
      {
        result.addError(fmt::format("Material {} {} has invalid texture index {}", matIdx, name, texIndex));
      }
    };

    checkTexture(mat.pbrMetallicRoughness.baseColorTexture.index, "baseColor");
    checkTexture(mat.pbrMetallicRoughness.metallicRoughnessTexture.index, "metallicRoughness");
    checkTexture(mat.normalTexture.index, "normal");
    checkTexture(mat.occlusionTexture.index, "occlusion");
    checkTexture(mat.emissiveTexture.index, "emissive");
  }
}

Scene::ValidationResult SceneValidator::validateBeforeSave() const
{
  auto result = validateModel();

  const auto& model = m_scene.getModel();

  // Additional checks for saving - these are ERRORS not warnings
  if(model.scenes.empty())
  {
    result.addError("No scenes defined - cannot save empty glTF");
  }

  if(model.nodes.empty())
  {
    result.addError("No nodes defined - cannot save scene without nodes");
  }

  return result;
}

bool SceneValidator::validateModelExtensions(const tinygltf::Model& model, const char* logContext) const
{
  const std::string indent     = nvutils::ScopedTimer::indent();
  const bool        hasContext = (logContext != nullptr && logContext[0] != '\0');

  for(const auto& extension : model.extensionsRequired)
  {
    if(m_scene.supportedExtensions().find(extension) == m_scene.supportedExtensions().end())
    {
      if(hasContext)
        LOGW("%sRequired extension unsupported in %s: %s\n", indent.c_str(), logContext, extension.c_str());
      else
        LOGW("%sRequired extension unsupported : %s\n", indent.c_str(), extension.c_str());
      return false;
    }
  }

  for(const auto& extension : model.extensionsUsed)
  {
    if(m_scene.supportedExtensions().find(extension) == m_scene.supportedExtensions().end())
    {
      if(hasContext)
        LOGW("%sUsed extension unsupported in %s: %s\n", indent.c_str(), logContext, extension.c_str());
      else
        LOGW("%sUsed extension unsupported : %s\n", indent.c_str(), extension.c_str());
    }
  }
  return true;
}

}  // namespace nvvkgltf
