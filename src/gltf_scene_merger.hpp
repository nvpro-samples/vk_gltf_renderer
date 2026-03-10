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

#include <optional>
#include <string>
#include <tinygltf/tiny_gltf.h>

namespace nvvkgltf {

// Tracks index offsets when merging two models (current array sizes before append).
struct IndexRemapping
{
  int nodes       = 0;
  int meshes      = 0;
  int materials   = 0;
  int textures    = 0;
  int images      = 0;
  int samplers    = 0;
  int accessors   = 0;
  int bufferViews = 0;
  int buffers     = 0;
  int skins       = 0;
  int cameras     = 0;
  int animations  = 0;
  int lights      = 0;  // KHR_lights_punctual extension
};

class SceneMerger
{
public:
  // Merge importedModel into baseModel. Imported scene roots become children of a new wrapper node.
  // The caller must set base model's "imported image" state so the loader uses the right basedir
  // per image: Scene stores baseImageCount and importedSceneBasePath for that.
  // If maxTextureCount is set and the combined texture count would exceed it, returns -1.
  // Returns the index of the new wrapper node in baseModel.nodes on success, or -1 on failure.
  static int merge(tinygltf::Model&        baseModel,
                   const tinygltf::Model&  importedModel,
                   const std::string&      importedSceneName,
                   std::optional<uint32_t> maxTextureCount = std::nullopt);

private:
  // Implementation in .cpp (helpers in anonymous namespace).
};

}  // namespace nvvkgltf
