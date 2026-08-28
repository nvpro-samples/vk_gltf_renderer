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

// KHR_interactivity Graphs panel (Phase F), plus a viewport toolbar indicator.
//
// Contains:
//   1. InteractivityControl -- play/pause/reset state for the default graph's runtime
//      instance. No ImGui in this struct, matching AnimationControl's split (ui_animation.hpp)
//      so other call sites can read/drive it without pulling the UI layer.
//   2. ui::interactivity::renderWindow -- the standalone "Interactivity" window (transport
//      controls, live graph/node/variable stats, a custom event sender, and a debug/log history
//      surface). Closed by default (Resources::settings.showInteractivityWindow) - developer-
//      facing content a user never has to open. Opened via the View menu or
//      GltfRenderer::registerRecentFilesHandler's window-toggle wiring, same as the Camera/Scene
//      Browser/Inspector windows.
//   3. ui::interactivity::renderToolbarIndicator -- a small Play/Pause icon button drawn in the
//      viewport toolbar next to the Grid/Gizmo/Snap/Animation Strip toggles. Only exists when the
//      loaded scene actually has a KHR_interactivity graph; click toggles InteractivityControl::
//      play directly. Does not open the window - that's a separate, deliberate action.
//
// State owner: Resources::interactivityControl. GltfRenderer::updateInteractivityGraphs reads
// `play` to gate ticking and consumes a one-shot `resetRequested` by resetting the default
// graph's InteractivityGraphInstance. Every control uses a stable ImGui label so it's a
// reliable scripted `click <label>` target.

namespace nvvkgltf {
class Scene;
}

//-----------------------------------------------------------------------------
// Playback state
//-----------------------------------------------------------------------------

struct InteractivityControl
{
  bool play           = true;   // Default graph auto-ticks every frame while true
  bool resetRequested = false;  // One-shot: reset the default graph's instance on next update

  int selectedEventIndex = 0;  // Graphs panel's custom "Send Event" combo selection

  void togglePlay() { play = !play; }
  void requestReset() { resetRequested = true; }
  void clearResetRequest() { resetRequested = false; }
};

//-----------------------------------------------------------------------------
// Graphs panel
//-----------------------------------------------------------------------------

namespace ui::interactivity {

// True when `scene` has at least one parsed KHR_interactivity graph - the shared gate both
// renderWindow() and renderToolbarIndicator() use to stay invisible on ordinary scenes.
bool hasGraph(nvvkgltf::Scene* scene);

// Renders the standalone "Interactivity" window for `scene`'s default KHR_interactivity graph
// (play/pause/reset, node/variable/event stats, custom event sender, debug/log history).
// No-op if `scene` has no parsed interactivity graphs, `open` is null, or `*open` is false.
// `open` is passed straight to ImGui::Begin() so the window's title-bar close button works.
void renderWindow(nvvkgltf::Scene* scene, InteractivityControl& ctrl, bool* open);

// Renders the viewport toolbar's Play/Pause indicator button (see file header comment).
// No-op if `scene` has no parsed interactivity graphs.
void renderToolbarIndicator(nvvkgltf::Scene* scene, InteractivityControl& ctrl);

}  // namespace ui::interactivity
