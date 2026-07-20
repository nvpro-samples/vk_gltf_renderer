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

//==================================================================================================
// LINEAR COLOR EDITING
//==================================================================================================
//
// glTF color *factors* (baseColor, emissive, sheen, specular tint, punctual light color, ...) are all
// stored in linear space. So are the renderer's own linear colors (infinite-plane base color, solid
// background radiance). We always DISPLAY and EDIT the raw linear numbers so that what the user reads
// in the panel matches exactly what is written/used - no surprising sRGB values.
//
// ImGui's swatch and color wheel, however, are perceptual and have no sRGB awareness: feeding them a
// linear value makes the swatch look washed out and the wheel pick non-intuitively. Because a single
// ImGui::ColorEdit drives its numeric fields, swatch and wheel from one buffer, we cannot show linear
// numbers and a perceptual swatch at the same time with one call. So we compose the row ourselves:
//   - a DragFloat showing the true linear value (what gets stored), and
//   - a swatch/wheel fed the sRGB-encoded value (converted back to linear on pick, via glm's
//     IEC 61966-2-1 helpers).
// The sRGB value is re-derived from the stored linear value every frame, so idle colors never drift;
// only an actual pick or drag writes back.
//
// If ImGui ever gains an ImGuiColorEditFlags_LinearSpace flag, these helpers collapse back to a
// single ColorEdit call with that flag set.

#include <string>

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/color_space.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <nvgui/property_editor.hpp>

namespace uicolor {

// Draws a swatch (perceptual sRGB) next to the linear DragFloat and writes any picked color back to
// the linear buffer. Returns true if the picker changed the color.
inline bool linearColorSwatch(float* col, int components)
{
  glm::vec4 disp(glm::convertLinearToSRGB(glm::make_vec3(col)), components == 4 ? col[3] : 1.0f);

  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
  bool changed = (components == 4) ? ImGui::ColorEdit4("##pick", glm::value_ptr(disp), ImGuiColorEditFlags_NoInputs) :
                                     ImGui::ColorEdit3("##pick", glm::value_ptr(disp), ImGuiColorEditFlags_NoInputs);
  if(changed)
  {
    glm::vec3 lin = glm::convertSRGBToLinear(glm::vec3(disp));
    col[0]        = lin.x;
    col[1]        = lin.y;
    col[2]        = lin.z;
    if(components == 4)
      col[3] = disp.w;  // alpha stays linear
  }
  return changed;
}

// Property-editor color row for a linear RGB factor: numeric fields show/edit the linear value while
// the swatch and wheel work perceptually (sRGB). Storage stays linear.
inline bool colorEdit3Linear(const char* label, float col[3], const std::string& tooltip = {})
{
  return nvgui::PropertyEditor::entry(
      label,
      [&] {
        const float swatchWidth = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - swatchWidth);
        bool changed = ImGui::DragFloat3("##lin", col, 0.01f, 0.0f, 1.0f, "%.3f");
        changed |= linearColorSwatch(col, 3);
        return changed;
      },
      tooltip);
}

// Same as above for RGBA factors. Alpha is linear coverage, not a color, so it is never gamma-converted.
inline bool colorEdit4Linear(const char* label, float col[4], const std::string& tooltip = {})
{
  return nvgui::PropertyEditor::entry(
      label,
      [&] {
        const float swatchWidth = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - swatchWidth);
        bool changed = ImGui::DragFloat4("##lin", col, 0.01f, 0.0f, 1.0f, "%.3f");
        changed |= linearColorSwatch(col, 4);
        return changed;
      },
      tooltip);
}

}  // namespace uicolor
