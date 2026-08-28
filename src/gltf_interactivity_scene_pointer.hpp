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
// Scene-backed InteractivityPointerResolver: resolves KHR_interactivity pointer/get and
// pointer/set against the live tinygltf::Model owned by a Scene.
//
// Coverage: node TRS/matrix/globalMatrix/mesh/camera/skin/parent/children/weights +
// KHR_node_visibility, core + extension material properties (any /extensions/<NAME>/... path,
// including KHR_texture_transform under any texture slot) via a generic tinygltf::Value walk,
// punctual-light color/intensity/range/spot cone angles, perspective-camera basics, core read-only
// array/length/ref pointers (animations/cameras/materials/meshes/nodes/scenes/skins), and the
// spec's own asset/limits self-description pointers. Still not the full glTF Object Model (no
// accessor/buffer/image/sampler/texture-index-level pointers, no skin inverseBindMatrices) - see
// docs/interactivity.md for what's covered and grep this file's get()/set() for the exact path list.
//

#pragma once

#include "gltf_interactivity_pointer.hpp"
#include "gltf_scene.hpp"

namespace nvvkgltf {

class ScenePointerResolver : public InteractivityPointerResolver
{
public:
  explicit ScenePointerResolver(Scene& scene)
      : m_scene(scene)
  {
  }

  std::optional<InteractivityValue> get(const std::string& concretePath) const override;
  bool                              set(const std::string& concretePath, const InteractivityValue& value) override;

private:
  Scene& m_scene;
};

}  // namespace nvvkgltf
