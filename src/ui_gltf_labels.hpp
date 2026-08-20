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
 * Shared glTF label / sampler-field helpers used by both the Elements list (browse columns) and the
 * Inspector (resource editors). Keeps the sampler wrap/filter enum<->name tables and the image display
 * name in exactly one place so the list summary and the editor never drift.
 */

#include <functional>
#include <string>

#include <tinygltf/tiny_gltf.h>

namespace uigltf {

// Display name for a glTF image: its URI, else its name, else "Embedded image <i>".
std::string imageDisplayName(const tinygltf::Model& model, int imageIndex);

// Short human names for glTF sampler enum values ("?" if unrecognized).
const char* wrapName(int value);  // wrapS / wrapT
const char* magName(int value);   // magFilter (-1 => "Default")
const char* minName(int value);   // minFilter (-1 => "Default")

// One-line sampler summary for a list column, e.g. "Repeat / Linear".
std::string samplerSummary(const tinygltf::Sampler& s);

// Renders the four wrap/filter combos for `cur`; when a field changes, invokes commit(edited) with a
// copy of cur carrying that change (one field per frame). Shared by the sampler and texture editors.
bool renderSamplerFields(const tinygltf::Sampler& cur, const std::function<void(const tinygltf::Sampler&)>& commit);

// Triangle count implied by a primitive's mode and its indexed-or-vertex element count (indexCount if
// indexed, else vertexCount). TRIANGLES -> count/3; TRIANGLE_STRIP / TRIANGLE_FAN -> count-2 (0 if
// count < 3); every other mode (POINTS, LINE*) has no triangles. Shared by the Elements list's mesh
// triangle totals and the Inspector's per-primitive/mesh counts so the two views can't disagree.
long long primitiveTriangleCountForMode(int mode, long long count);

}  // namespace uigltf
