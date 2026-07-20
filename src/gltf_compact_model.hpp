/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <tinygltf/tiny_gltf.h>

/*-------------------------------------------------------------------------------------------------
## Function `compactModel`

Removes orphaned data from a glTF model, reducing memory usage and file size.

After operations that create new accessors/buffer views (like MikkTSpace tangent generation
with vertex splitting), old data becomes orphaned - still present in buffers but no longer
referenced.

Returns true if any data was removed, false if model was already compact.
-------------------------------------------------------------------------------------------------*/
bool compactModel(tinygltf::Model& model);

/*-------------------------------------------------------------------------------------------------
## Function `removeExternalAssetContent`

Re-externalization for glTF 2.1 complex scenes: removes every node that was merged in from a
referenced external asset (identified by the Scene::kExternalAssetContentKey extras marker),
remaps all node-index references, and compacts the now-orphaned meshes/materials/skins/animations/
accessors/etc. The instance nodes (which keep their `externalAsset` reference) and the top-level
`files`/`externalAssets` tables are preserved, so the model can be written back as a complex scene
instead of the flattened runtime model. Operates in place; intended to run on a save-time copy.
-------------------------------------------------------------------------------------------------*/
namespace nvvkgltf {
void removeExternalAssetContent(tinygltf::Model& model);

/*-------------------------------------------------------------------------------------------------
## Function `flattenExternalAssets`

Self-contained (shareable) counterpart to `removeExternalAssetContent`: keeps all merged-in
external-asset content inline and instead strips every trace of the external-asset relationship --
the per-node read-only `extras` marker (Scene::kExternalAssetContentKey), the `externalAsset`
reference on the instance nodes, and the top-level `files`/`externalAssets` tables. The result is
a plain glTF 2.0-style scene with no external references, so it can be shared without shipping the
referenced files. Operates in place; intended to run on a save-time copy.
-------------------------------------------------------------------------------------------------*/
void flattenExternalAssets(tinygltf::Model& model);

/*-------------------------------------------------------------------------------------------------
## Function `compactExternalAssetReferences`

Removes `externalAssets` and `files` entries that are no longer referenced -- e.g. after breaking an
external-asset link with `SceneEditor::makeExternalAssetEditable` -- and remaps the survivors. An
`externalAssets` entry is kept iff some node still references it; a `files` entry is kept iff it is
reachable from a surviving `externalAsset` or (transitively) from a surviving file's `aliases`. Node
`externalAsset` indices and `ExternalAsset::file` / `FileAlias::file` indices are remapped. No-op
when nothing is orphaned. Operates in place.
-------------------------------------------------------------------------------------------------*/
void compactExternalAssetReferences(tinygltf::Model& model);
}  // namespace nvvkgltf
