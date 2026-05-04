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

// Animation UI for the glTF renderer.
//
// Contains:
//   1. AnimationControl -- playback state + domain logic (play/pause, clip
//      selection, scrub, frame step, reset, speed). No ImGui in this header
//      so other call sites can read the state without pulling the UI layer.
//   2. ui::animation::renderStripWidgets -- the single view, a slim transport
//      row drawn as a semi-transparent overlay at the bottom of the Viewport's
//      3D image. Visibility is driven by AnimationControl::showStrip
//      (auto-enabled by GltfRenderer when a scene containing animations is
//      loaded; toggleable from the View menu and viewport toolbar).
//
// State owner: Resources::animationControl. The renderer's update path
// (GltfRenderer::updateAnimation) reads the queries (doAnimation, deltaTime,
// isReset) and calls clearStates() after consuming a one-shot step/reset.
// GltfRenderer::cleanupScene resets the struct to defaults on scene load.

#include <string>

#include <imgui/imgui.h>  // ImVec2 in renderStripOverlay's signature

namespace nvvkgltf {
class Scene;
}

//-----------------------------------------------------------------------------
// Playback state + domain logic
//-----------------------------------------------------------------------------

struct AnimationControl
{
  // State
  bool  play             = true;   // Continuous playback
  bool  runOnce          = false;  // One-shot step (timeline scrub or frame advance)
  bool  reset            = false;  // Jump back to start on next update
  float speed            = 1.0F;   // Playback multiplier
  int   currentAnimation = 0;      // Index into AnimationSystem clips
  bool  showStrip        = false;  // Viewport Animation Strip overlay visible (auto-enabled on animated scene load)

  // Domain actions (used by the Animation Strip widget and keyboard shortcut)
  void togglePlay() { play = !play; }
  void stepOne()
  {
    runOnce = true;
    play    = false;
  }
  void resetToStart() { reset = true; }

  // Move the current clip to `time`, pause playback, and request a one-shot
  // update so the scene reflects the new pose on the next frame.
  // No-op if the scene has no animation or the selection is out of range.
  void scrubTo(float time, nvvkgltf::Scene* scene);

  // Queries consumed by the animation update path
  [[nodiscard]] bool  doAnimation() const { return play || runOnce || reset; }
  [[nodiscard]] float deltaTime() const;
  [[nodiscard]] bool  isReset() const { return reset; }

  // Called by the animation update path after consuming a one-shot step/reset.
  void clearStates() { runOnce = reset = false; }

  // Format a time value in seconds as mm:ss.ms (used by the time readout).
  static std::string formatTime(float seconds);
};

//-----------------------------------------------------------------------------
// Animation Strip widget
//-----------------------------------------------------------------------------

namespace ui::animation {

// A scene has "playable animation" when at least one clip declares a positive duration.
bool hasPlayableAnimation(nvvkgltf::Scene* scene);

// Render the Animation Strip as a hover-reactive overlay at the given screen rect
// (typically over the bottom edge of the 3D image in the Viewport). Fully
// visible while the mouse is over the strip; eases to a dim idle alpha when
// the mouse leaves, so it stays out of the way of the 3D content until the
// user wants it. Owns its own child region, background, and fade state.
void renderStripOverlay(nvvkgltf::Scene* scene, AnimationControl& ctrl, const ImVec2& topLeft, const ImVec2& size);

}  // namespace ui::animation
