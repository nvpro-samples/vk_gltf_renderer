/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <glm/glm.hpp>
#include <imgui.h>

namespace nvvkgltf {
class Scene;
}

//--------------------------------------------------------------------------------------------------
// Viewport icons: the glyphs of the lights and cameras of the scene, drawn over the viewport image.
//
// A light or a camera has no geometry, so nothing in the render shows where it is or which way it points.
// This unit draws each as a small wireframe through an ImGui draw list, so it is vector-sharp at any
// resolution, needs no GPU resources, and stays out of the render and its captures. The selected one is
// always drawn, highlighted; the others are drawn when asked. A click on a glyph picks its node.
//--------------------------------------------------------------------------------------------------
namespace ui::viewport_icons {

// The viewport image the icons are drawn over, and the camera that looks at it.
struct View
{
  glm::mat4 view        = glm::mat4(1.0F);  // world to view, as CameraManipulator::getViewMatrix()
  glm::mat4 projection  = glm::mat4(1.0F);  // view to clip, Vulkan convention, as getPerspectiveMatrix()
  glm::vec2 imageOrigin = glm::vec2(0.0F);  // top-left of the viewport image, in ImGui screen pixels
  glm::vec2 imageSize   = glm::vec2(1.0F);  // size of the viewport image, in ImGui screen pixels
};

// What drawIcons() and pickIcon() work on: the view, the scene whose lights and cameras to show, and the
// selection. A null `scene` means the glyphs are not shown, and both functions do nothing.
struct Request
{
  View                   view;
  const nvvkgltf::Scene* scene        = nullptr;
  int                    selectedNode = -1;     // glTF node whose light and camera are always drawn, highlighted
  bool                   showAll      = false;  // also draw every other light and camera
};

// Draws the glyph of each light and camera over the viewport image: all of them when `request.showAll`, and
// the ones that `request.selectedNode` carries in any case, highlighted and on top.
//
// The glyphs are wireframes built in world space around the node and scaled so that they keep the same
// size on screen, so they stay readable at any distance and resolution, and their perspective shows which
// way the light or camera points. A point light is a disc inside a ring, a spot light a disc with its real
// cone, a directional light a disc with rays and an arrow, a camera its frustum with an "up" triangle.
//
// A glyph is a few dozen segments and a scene can hold thousands of lights, so only a fixed number of
// lights and of cameras is considered per frame, and a glyph far outside the image is skipped.
void drawIcons(ImDrawList* drawList, const Request& request);

// The glTF node whose glyph lies under `mousePos` (ImGui screen pixels), or -1. Takes the same request as
// drawIcons() and tests the glyphs that call draws, so what can be clicked is what is seen. A glyph is
// hit on its lines or its disc, within a few pixels, so it does not take clicks from the geometry around
// it. Rebuilds the glyphs, so it is for acting on a click, not for every frame.
int pickIcon(const Request& request, const glm::vec2& mousePos);

}  // namespace ui::viewport_icons
