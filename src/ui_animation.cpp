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

#include "ui_animation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include <imgui/imgui.h>
#include <nvgui/fonts.hpp>
#include <nvgui/tooltip.hpp>

#include "gltf_scene.hpp"
#include "gltf_scene_animation.hpp"

//=============================================================================
// AnimationControl -- state + domain logic
//=============================================================================

// The 1/60 second step for single-frame advancement matches the nominal rate
// used by glTF viewers and keeps frame-step behavior independent of display FPS.
static constexpr float kStepDeltaSeconds = 1.0F / 60.0F;

float AnimationControl::deltaTime() const
{
  // runOnce produces a fixed-size step; continuous playback scales real delta time.
  return runOnce ? speed * kStepDeltaSeconds : ImGui::GetIO().DeltaTime * speed;
}

void AnimationControl::scrubTo(float time, nvvkgltf::Scene* scene)
{
  if(!scene || !scene->animation().hasAnimation())
    return;

  const int nAnim = scene->animation().getNumAnimations();
  if(currentAnimation < 0 || currentAnimation >= nAnim)
    return;

  nvvkgltf::AnimationInfo& info = scene->animation().getAnimationInfo(currentAnimation);
  info.currentTime              = std::clamp(time, info.start, info.end);
  runOnce                       = true;   // Request a single evaluation on next update
  play                          = false;  // Scrubbing implies pause
}

std::string AnimationControl::formatTime(float seconds)
{
  if(!std::isfinite(seconds) || seconds < 0.0F)
    seconds = 0.0F;
  const int total   = static_cast<int>(seconds);
  const int minutes = total / 60;
  const int secs    = total % 60;
  const int ms      = static_cast<int>((seconds - static_cast<float>(total)) * 1000.0F);
  char      buffer[16];
  std::snprintf(buffer, sizeof(buffer), "%02d:%02d.%03d", minutes, secs, ms);
  return buffer;
}

//=============================================================================
// Animation Strip widget
//=============================================================================

namespace ui::animation {
namespace {

// Right-side horizontal budgets. The scrubber fills whatever remains.
constexpr float kTimeReadoutWidth = 110.0F;  // "00:12 / 01:00"
constexpr float kSpeedFieldWidth  = 65.0F;   // "1.00x" drag
constexpr float kClipSelectorMinW = 140.0F;

// Hover-reactive fade tuning. Exponential smoothing gives a natural ease:
//   new = old + (target - old) * (1 - exp(-rate * dt))
// rate = ln(2) / half_life; 4.0 corresponds to ~173 ms half-life.
constexpr float kIdleAlpha   = 0.25F;
constexpr float kActiveAlpha = 1.0F;
constexpr float kFadeRate    = 4.0F;

// Single-instance transition state for the overlay strip. UI-only, so it
// stays out of AnimationControl (which models domain/playback state).
struct FadeState
{
  float alpha         = kIdleAlpha;  // start dim; hover eases up
  bool  clipPopupOpen = false;       // last-frame snapshot: is the clip combo dropdown open?
};

FadeState& fadeState()
{
  static FadeState s;
  return s;
}

float easeTowards(float current, float target, float dt, float rate)
{
  const float t = 1.0F - std::exp(-rate * dt);
  return current + (target - current) * t;
}

}  // namespace

bool hasPlayableAnimation(nvvkgltf::Scene* scene)
{
  if(!scene || !scene->animation().hasAnimation())
    return false;

  // A clip is "playable" only when its timeline spans a meaningful positive
  // duration. Require the span to exceed a microsecond: well below any real
  // animation frame (~16 ms at 60 fps), well above numerical noise. This
  // defends against malformed files with all keyframes at the same time,
  // NaN-poisoned timelines (all comparisons return false), and any future
  // regression of the default sentinel in AnimationInfo.
  constexpr float            kMinPlayableDurationSeconds = 1e-6F;
  nvvkgltf::AnimationSystem& system                      = scene->animation();
  for(int i = 0; i < system.getNumAnimations(); ++i)
  {
    const nvvkgltf::AnimationInfo& info = system.getAnimationInfo(i);
    if(info.end - info.start > kMinPlayableDurationSeconds)
      return true;
  }
  return false;
}

namespace {

// Render the Animation Strip widget row. Placement-agnostic: the caller (currently
// renderStripOverlay) opens whatever wrapping container it wants, then calls
// into this function to fill it with the play/pause/scrubber/speed widgets.
void renderStripWidgets(nvvkgltf::Scene* scene, AnimationControl& ctrl)
{
  const bool enabled = hasPlayableAnimation(scene);

  ImGui::PushID("##animation_strip");
  ImGui::BeginDisabled(!enabled);

  // Clip info resolved once so every widget uses the same snapshot.
  int                      nAnim = 0;
  nvvkgltf::AnimationInfo* info  = nullptr;
  if(enabled)
  {
    nAnim = scene->animation().getNumAnimations();
    if(ctrl.currentAnimation < 0 || ctrl.currentAnimation >= nAnim)
      ctrl.currentAnimation = 0;
    info = &scene->animation().getAnimationInfo(ctrl.currentAnimation);
  }

  // Play / pause
  if(ImGui::Button(ctrl.play ? ICON_MS_PAUSE : ICON_MS_PLAY_ARROW))
    ctrl.togglePlay();
  if(enabled)
    nvgui::tooltip("Play/Pause (Space)");

  // Step one frame forward
  ImGui::SameLine();
  if(ImGui::Button(ICON_MS_SKIP_NEXT))
    ctrl.stepOne();
  if(enabled)
    nvgui::tooltip("Advance one frame");

  // Reset to start
  ImGui::SameLine();
  if(ImGui::Button(ICON_MS_REPLAY))
    ctrl.resetToStart();
  if(enabled)
    nvgui::tooltip("Reset to start");

  // Clip selector (hidden label, compact combo)
  ImGui::SameLine();
  ImGui::SetNextItemWidth(kClipSelectorMinW);
  const char* label = info ? info->name.c_str() : "No animation";
  if(info && info->name.empty())
    label = "Animation";
  // Record the popup state so the overlay's fade can keep the strip visible
  // while the user picks a clip (the dropdown opens outside our hover rect).
  const bool clipPopupOpen  = ImGui::BeginCombo("##clip", label);
  fadeState().clipPopupOpen = clipPopupOpen;
  if(clipPopupOpen)
  {
    for(int i = 0; i < nAnim; ++i)
    {
      const std::string& n   = scene->animation().getAnimationInfo(i).name;
      const std::string  txt = n.empty() ? ("Animation " + std::to_string(i)) : n;
      if(ImGui::Selectable(txt.c_str(), ctrl.currentAnimation == i))
        ctrl.currentAnimation = i;
    }
    ImGui::EndCombo();
  }
  if(enabled && nAnim > 1)
    nvgui::tooltip("Animation clip");

  // Scrubber (fills remaining space, minus time + speed reservations)
  ImGui::SameLine();
  const float reserveRight = kTimeReadoutWidth + kSpeedFieldWidth + ImGui::GetStyle().ItemSpacing.x * 2.0F;
  const float sliderWidth  = std::max(60.0F, ImGui::GetContentRegionAvail().x - reserveRight);
  ImGui::SetNextItemWidth(sliderWidth);
  float t     = info ? info->currentTime : 0.0F;
  float start = info ? info->start : 0.0F;
  float end   = info ? info->end : 1.0F;
  if(ImGui::SliderFloat("##progress", &t, start, end))
    ctrl.scrubTo(t, scene);

  // Time readout
  ImGui::SameLine();
  const float relative = info ? std::max(0.0F, info->currentTime - info->start) : 0.0F;
  const float duration = info ? std::max(0.0F, info->end - info->start) : 0.0F;
  ImGui::AlignTextToFramePadding();
  ImGui::Text("%s / %s", AnimationControl::formatTime(relative).c_str(), AnimationControl::formatTime(duration).c_str());

  // Speed (compact inline drag)
  ImGui::SameLine();
  ImGui::SetNextItemWidth(kSpeedFieldWidth);
  ImGui::DragFloat("##speed", &ctrl.speed, 0.01F, 0.0F, 100.0F, "%.2fx");
  if(enabled)
    nvgui::tooltip("Playback speed (drag)");

  ImGui::EndDisabled();
  ImGui::PopID();
}

}  // namespace

void renderStripOverlay(nvvkgltf::Scene* scene, AnimationControl& ctrl, const ImVec2& topLeft, const ImVec2& size)
{
  // Decide target alpha BEFORE opening the child so we can push the style
  // var for both the child background and the widgets inside. The strip is
  // kept fully visible while the clip dropdown is open, since that popup
  // sits outside our hover rect.
  const ImVec2 bottomRight(topLeft.x + size.x, topLeft.y + size.y);
  const bool   hovered = ImGui::IsMouseHoveringRect(topLeft, bottomRight);

  FadeState&  fs     = fadeState();
  const float dt     = ImGui::GetIO().DeltaTime;
  const float target = (hovered || fs.clipPopupOpen) ? kActiveAlpha : kIdleAlpha;
  fs.alpha           = easeTowards(fs.alpha, target, dt, kFadeRate);

  // Alpha multiplies through to every widget and the child bg, giving a
  // uniform fade. ItemSpacing y=0 keeps the widget row tight against the
  // bottom of the child. ChildBg is our semi-transparent base.
  ImGui::PushStyleVar(ImGuiStyleVar_Alpha, fs.alpha);
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(ImGui::GetStyle().ItemSpacing.x, 0.0F));
  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08F, 0.08F, 0.08F, 0.50F));

  ImGui::SetCursorScreenPos(topLeft);
  ImGui::BeginChild("##anim_strip", size, ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
  renderStripWidgets(scene, ctrl);
  ImGui::EndChild();

  ImGui::PopStyleColor();
  ImGui::PopStyleVar(2);
}

}  // namespace ui::animation
