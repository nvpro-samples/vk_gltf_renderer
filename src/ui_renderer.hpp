/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nvutils/bounding_box.hpp"

class GltfRenderer;  // Forward declaration

struct GltfRendererUI
{
  static void          renderUI(GltfRenderer& renderer);
  static void          renderMenu(GltfRenderer& renderer);
  static void          mouseClickedInViewport(GltfRenderer& renderer);
  static nvutils::Bbox getRenderNodeBbox(GltfRenderer& renderer, int nodeID);
  static void          windowTitle(GltfRenderer& renderer);
};
