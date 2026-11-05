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

#include <filesystem>
#include <functional>
#include <string>

#include <imgui.h>

#include "gltf_material_cache.hpp"  // nvvkgltf::TextureBinding

// Host-provided services shared by the UI panels (inspector + scene browser). The renderer owns the
// implementations (file dialog, GPU thumbnails, toast overlay) and hands one filled struct to each
// panel via setHostServices(), instead of wiring the same three callbacks into both separately. The
// accessor helpers fold the "is it set?" null-check so call sites stay terse.
struct UiHostServices
{
  std::function<std::filesystem::path()> pickImageFile;     // open an image file dialog (empty = cancelled/unavailable)
  std::function<ImTextureID(int)>        textureThumbnail;  // glTF texture index -> bounded ImGui thumbnail (0 = none)
  std::function<void(const std::string&, bool)> notify;     // transient toast; second arg = isError (red)

  // Texture isolate (Visualization::eTextureIsolate): show one texture of a material raw and unlit.
  // Arguments are the glTF material index and the texture binding of the row; `srgb` marks color data
  // (base color, emissive...).
  std::function<bool(int, const nvvkgltf::TextureBinding&)> isTextureIsolated;  // is this the one shown?
  std::function<void(int, const nvvkgltf::TextureBinding&, bool)> toggleTextureIsolate;  // show it, or go back to the rendered view

  bool                  canIsolateTexture() const { return static_cast<bool>(toggleTextureIsolate); }
  bool                  canPickImage() const { return static_cast<bool>(pickImageFile); }
  std::filesystem::path pickImage() const { return pickImageFile ? pickImageFile() : std::filesystem::path{}; }
  ImTextureID           thumbnail(int textureIndex) const
  {
    return textureThumbnail ? textureThumbnail(textureIndex) : ImTextureID(0);
  }
  void toast(const std::string& message, bool isError = false) const
  {
    if(notify)
      notify(message, isError);
  }
};
