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

#include "gltf_interactivity_scene_animation.hpp"

#include "gltf_scene_animation.hpp"

namespace nvvkgltf {

bool SceneAnimationResolver::isValidAnimation(int animationIndex) const
{
  return animationIndex >= 0 && animationIndex < m_scene.animation().getNumAnimations();
}

float SceneAnimationResolver::animationMaxTime(int animationIndex) const
{
  if(!isValidAnimation(animationIndex))
    return 0.0f;
  return m_scene.animation().getAnimationInfo(animationIndex).end;
}

bool SceneAnimationResolver::applyPose(int animationIndex, float effectiveTime)
{
  if(!isValidAnimation(animationIndex))
    return false;
  m_scene.animation().getAnimationInfo(animationIndex).currentTime = effectiveTime;
  return m_scene.animation().updateAnimation(animationIndex);
}

}  // namespace nvvkgltf
