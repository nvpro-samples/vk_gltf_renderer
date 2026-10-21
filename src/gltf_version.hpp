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

// glTF asset versions: which files the renderer may load, and which version a saved file declares.
// Rules: glTF 2.0 spec "Versioning" and "Asset"; the 2.1 feature list follows the 2.1 draft. See
// docs/gltf_2_1.md.

#include <compare>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include <tinygltf/tiny_gltf.h>

namespace nvvkgltf {

// A parsed asset version string ("<major>.<minor>", decimal, no leading zeros).
struct GltfVersion
{
  uint32_t majorVersion = 0;
  uint32_t minorVersion = 0;

  auto        operator<=>(const GltfVersion&) const = default;
  std::string toString() const;
};

// The newest version whose features this renderer reads and writes.
inline constexpr GltfVersion kMaxSupportedGltfVersion{2, 1};

// Parses a version string with the spec's syntax; nullopt when it does not conform.
std::optional<GltfVersion> parseGltfVersion(std::string_view text);

// True if a client supporting kMaxSupportedGltfVersion may load the asset: `minVersion`, when present,
// must not exceed it; otherwise only the major `version` must match. Newer minor versions load
// (unknown features are ignored). On false, `reason` says why.
bool isGltfVersionLoadable(const tinygltf::Asset& asset, std::string* reason);

// The first glTF 2.1 feature `model` uses, as a short description, or empty if it is valid 2.0.
std::string findGltf21Feature(const tinygltf::Model& model);

// The `asset.version` a save should write: "2.1" if the model uses a 2.1 feature or its minVersion
// asks for 2.1, otherwise "2.0", so files without 2.1 content stay readable by 2.0-only loaders.
std::string gltfVersionForSave(const tinygltf::Model& model);

}  // namespace nvvkgltf
