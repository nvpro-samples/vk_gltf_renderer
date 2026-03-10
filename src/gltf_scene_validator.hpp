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

#pragma once

#include "gltf_scene.hpp"

namespace nvvkgltf {

/*-------------------------------------------------------------------------------------------------
# class nvvkgltf::SceneValidator

  Read-only validation of a Scene's model: validateModel, validateBeforeSave, validateModelExtensions.
  Friend of Scene for access to m_model and m_supportedExtensions. Scene retains forwarding
  wrappers so external call sites are unchanged.
--------------------------------------------------------------------------------------------------*/
class SceneValidator
{
public:
  explicit SceneValidator(const Scene& scene);

  [[nodiscard]] Scene::ValidationResult validateModel() const;
  [[nodiscard]] Scene::ValidationResult validateBeforeSave() const;
  // Validate model extensions against scene's supported extensions. logContext used in messages (e.g. "" or "imported scene").
  // Returns false if any required extension is unsupported; logs for used-but-unsupported.
  [[nodiscard]] bool validateModelExtensions(const tinygltf::Model& model, const char* logContext) const;

private:
  const Scene& m_scene;

  void validateNodes(Scene::ValidationResult& result) const;
  void validateScenes(Scene::ValidationResult& result) const;
  void validateAnimations(Scene::ValidationResult& result) const;
  void validateSkins(Scene::ValidationResult& result) const;
  void validateMeshReferences(Scene::ValidationResult& result) const;
  void validateMaterialReferences(Scene::ValidationResult& result) const;
};

}  // namespace nvvkgltf
