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

#include <optional>

#include "renderer_pathtracer.hpp"

//--------------------------------------------------------------------------------------------------
// Status bar along the bottom of the window: what the renderer is doing, for a developer to read at a
// glance. It is a viewport side bar, so the docked panels and the viewport give up its row instead of
// being covered. Each field of Content is one part of the bar; a field left empty has none.
//--------------------------------------------------------------------------------------------------
namespace ui::statusbar {

struct Content
{
  // The path trace: how far the image is, what a frame costs, what renders it and which shader (see
  // PathTracer::getStatus()). Empty with the rasterizer, which has nothing to report.
  std::optional<PathTracer::Status> pathTracer;
};

// Draw the bar. Call it every frame the bar is shown: ImGui carries the room it takes to the next
// frame, so the first frame after turning it on has no row of its own yet.
void render(const Content& content);

}  // namespace ui::statusbar
