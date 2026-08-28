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
// Scene-backed InteractivityAnimationResolver: answers animation/start's validity/duration queries
// and applies the CPU-side pose (AnimationSystem::updateAnimation()) against the live Scene. The
// heavier GPU-side reconciliation (world matrices, GPU sync, BLAS update) still happens once per
// frame in GltfRenderer, driven by InteractivityGraphInstance::pendingAnimationApplies() - see
// gltf_interactivity_animation.hpp for why the CPU/GPU split is where it is.
//

#pragma once

#include "gltf_interactivity_animation.hpp"
#include "gltf_scene.hpp"

namespace nvvkgltf {

class SceneAnimationResolver : public InteractivityAnimationResolver
{
public:
  explicit SceneAnimationResolver(Scene& scene)
      : m_scene(scene)
  {
  }

  bool  isValidAnimation(int animationIndex) const override;
  float animationMaxTime(int animationIndex) const override;
  bool  applyPose(int animationIndex, float effectiveTime) override;

private:
  Scene& m_scene;
};

}  // namespace nvvkgltf
