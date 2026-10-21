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

// File-level helpers shared by Scene loading (gltf_scene.cpp) and the glTF 2.1 external-asset
// resolver (gltf_scene_external_assets.cpp). Internal to the scene; not a public API.

#include <filesystem>
#include <string>
#include <vector>

#include <tinygltf/tiny_gltf.h>

namespace nvvkgltf::detail {

// glTF 2.1 file aliases: while loading a referenced glTF, redirect any of its inner URIs that match
// an alias to another entry in the owner's `files` table (used for shared-resource packing or for
// overriding inner data). Aliases are declared on the owner's File entry that points to this glTF.
struct AliasContext
{
  const std::vector<tinygltf::FileAlias>* aliases    = nullptr;  // aliases declared on the file being loaded
  const std::vector<tinygltf::File>*      ownerFiles = nullptr;  // files[] that FileAlias::file indexes into
  std::filesystem::path                   ownerBaseDir;          // resolves the aliased target's URI
  std::filesystem::path                   childBaseDir;          // base dir of the file being loaded
};

// Load a .gltf or .glb with the scene's tinygltf settings (no external file limit, image bytes kept
// encoded), refusing glTF versions this renderer cannot read. On failure, outError/outWarn are set
// for the caller to log.
bool loadGltfFile(const std::filesystem::path& filename,
                  tinygltf::Model&             model,
                  std::string*                 outError,
                  std::string*                 outWarn,
                  const AliasContext*          aliasCtx = nullptr);

// Rewrite the relative image URIs of `child`, loaded from `childDir`, so they resolve from `ownerDir`.
void rebaseImageUris(tinygltf::Model& child, const std::filesystem::path& childDir, const std::filesystem::path& ownerDir);

// Decompress KHR/EXT_meshopt_compression buffer views in place and drop the extension. Returns true
// if the extension was not used or decoding succeeded.
bool decompressMeshopt(tinygltf::Model& model);

}  // namespace nvvkgltf::detail
