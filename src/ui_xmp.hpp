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

/*
 * UI functions for displaying KHR_xmp_json_ld metadata
 * See: https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_xmp_json_ld
 */

#include <tinygltf/tiny_gltf.h>

namespace ui_xmp {

// Renders an XMP info icon button for a glTF object's extensions.
// Shows a popup with packet contents when clicked.
// Returns true if an XMP info button was rendered for this object.
bool renderInfoButton(const tinygltf::Model* model, const tinygltf::ExtensionMap& extensions, const char* popupId);

// Render collapsible header showing all XMP packets in the model
void renderMetadataPanel(const tinygltf::Model* model);

}  // namespace ui_xmp
