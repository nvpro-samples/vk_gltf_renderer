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
// Abstract glTF-animation query surface for animation/start, animation/stop, animation/stopAt.
//
// Kept dependency-light on purpose, mirroring gltf_interactivity_pointer.hpp's split: this header
// must NOT pull in gltf_scene.hpp, so the core graph engine (gltf_interactivity_instance/eval) stays
// unit-testable without a Scene. The concrete Scene-backed implementation lives in
// gltf_interactivity_scene_animation.hpp/cpp.
//
// The timestamp math (spec's requested-to-effective-timestamp remap, elapsed-time tracking,
// start/end/stop-time comparisons) lives in InteractivityGraphInstance itself (pure, no Scene
// needed). applyPose() is the one Scene-touching call it makes, and it's synchronous and CPU-only
// (writes the new pose into the live tinygltf::Model, no VkCommandBuffer/GPU work) - required
// because the spec's own animation/start algorithm applies the pose to the asset BEFORE activating
// `done` (Specification.adoc's "on each asset animation update" steps), so a pointer/get or
// debug/log reached from that same `done` activation must already see the fresh value. The
// GPU-side reconciliation (world matrices, GPU sync, BLAS update) is a separate, heavier pipeline
// that genuinely does need batching/cmd access - that part alone stays in GltfRenderer, driven by
// pendingAnimationApplies() (see InteractivityGraphInstance).
//

#pragma once

namespace nvvkgltf {

class InteractivityAnimationResolver
{
public:
  virtual ~InteractivityAnimationResolver() = default;

  // True if `animationIndex` is a valid index into the glTF asset's animations array.
  virtual bool isValidAnimation(int animationIndex) const = 0;

  // Spec's "T": the maximum value of all animation sampler input accessors for this animation -
  // i.e. its duration, assuming (as the spec does) sampler data spans [0, T].
  virtual float animationMaxTime(int animationIndex) const = 0;

  // Evaluates `animationIndex`'s channels at `effectiveTime` and writes the result into the live
  // model immediately (CPU-side only - node TRS/weights, morph/skin-affecting flags; no GPU sync).
  // Returns false if `animationIndex` is invalid. Called once per active animation/start entry,
  // per tick, before InteractivityGraphInstance fires that entry's `done`/stop-`done` flow.
  virtual bool applyPose(int animationIndex, float effectiveTime) = 0;
};

}  // namespace nvvkgltf
