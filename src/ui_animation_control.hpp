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

// Animation Control UI Component
//
// Provides a UI for controlling glTF animations: selection, play/pause,
// frame stepping, reset, speed control, and timeline scrubbing.

#include <imgui/imgui.h>

#include <nvgui/fonts.hpp>
#include "gltf_scene.hpp"
#include "gltf_scene_animation.hpp"
#include <nvgui/property_editor.hpp>
#include <nvgui/tooltip.hpp>


// Simple structure to hold the animation controls
struct AnimationControl
{
  //--------------------------------------------------------------------------------------------------
  // Render the animation control UI
  // This method creates a complete animation control interface including:
  // 1. Animation selection dropdown
  // 2. Play/pause, step forward, and reset buttons
  // 3. Speed control slider
  // 4. Timeline slider for precise control
  // The UI is designed to be intuitive and provide immediate visual feedback
  void onUI(nvvkgltf::Scene* gltfScene)
  {
    namespace PE = nvgui::PropertyEditor;
    std::vector<const char*> animationNames;
    for(int i = 0; i < gltfScene->animation().getNumAnimations(); i++)
    {
      animationNames.push_back(gltfScene->animation().getAnimationInfo(i).name.c_str());
    }
    if(PE::begin(""))
    {
      PE::Combo("Animations", &currentAnimation, animationNames.data(), static_cast<int>(animationNames.size()));
      PE::end();
    }

    ImGui::SeparatorText("Animation Controls");
    if(ImGui::Button(play ? ICON_MS_PAUSE: ICON_MS_PLAY_ARROW))
    {
      play = !play;
    }
    nvgui::tooltip("Play/Pause the animation");
    ImGui::SameLine();
    if(ImGui::Button(ICON_MS_SKIP_NEXT))
    {
      runOnce = true;
      play    = false;
    }
    nvgui::tooltip("Advance one frame");
    ImGui::SameLine();
    if(ImGui::Button(ICON_MS_REPLAY))
    {
      reset = true;
    }
    nvgui::tooltip("Reset animation to start");

    ImGui::SameLine(0, 10.0f);
    ImGui::PushItemWidth(60.0f);  // Adjust width to make it compact
    ImGui::DragFloat("##speed", &speed, 0.01f, 0.0f, 100.0f);
    // Tooltip or additional indicator for the speed control
    if(ImGui::IsItemHovered())
      ImGui::SetTooltip("Playback speed multiplier");
    ImGui::PopItemWidth();  // Reset width

    // Add a small label next to the speed input for clarity
    ImGui::SameLine(0.0, 1.0);
    ImGui::TextUnformatted("x");

    // Show the timeline slider
    nvvkgltf::AnimationInfo& animInfo = gltfScene->animation().getAnimationInfo(currentAnimation);
    ImGui::TextDisabled("Timeline");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
    if(ImGui::SliderFloat("##no-label", &animInfo.currentTime, animInfo.start, animInfo.end, "Time: %.2f"))
      runOnce = true;
    ImGui::PopItemWidth();
  }

  //--------------------------------------------------------------------------------------------------
  // Check if animation should be updated
  // Returns true if any animation state is active (playing, single step, or reset)
  bool doAnimation() const { return play || runOnce || reset; }

  //--------------------------------------------------------------------------------------------------
  // Calculate the time step for animation
  // Returns either a fixed step for single-frame advancement or a scaled delta time for continuous playback
  float deltaTime() const { return runOnce ? speed / 60.f : ImGui::GetIO().DeltaTime * speed; }

  //--------------------------------------------------------------------------------------------------
  // Check if animation should be reset
  // Returns true if the reset button was pressed
  bool isReset() const { return reset; }

  //--------------------------------------------------------------------------------------------------
  // Clear one-time animation states
  // Resets the runOnce and reset flags after they've been processed
  void clearStates() { runOnce = reset = false; }

  bool  play             = true;   // Simulation running?
  bool  runOnce          = false;  // Simulation step-once
  bool  reset            = false;  // Reset the animation
  float speed            = 1.0F;   // Simulation speed
  int   currentAnimation = 0;
};
