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

//==================================================================================================
// `.sky.json` -- a sky on its own, outside any scene
//==================================================================================================
//
// One sky per scene is the invariant; this is how a sky gets from one scene to another, or to a
// colleague. The file is **exactly one entry of the OMI `skies[]` array**, with the NV blocks nested
// where they already belong. No wrapper, no version field, no format of our own: a preset is a
// fragment of a glTF extension, and anything that can read one can read the other.
//
// That is why there is no serializer here. `gltf_environment_sky::toValue` / `fromValue` do the
// work, and this file is only the text layer around them -- what a glTF writer would otherwise do.
// A field added to the extension appears in presets the same day, without anyone remembering to.
//
// The one addition is `sunRotation`, which OMI has no place for. A sky and the angle of its sun are
// one look; a preset that restored the atmosphere and left the sun where it was would be half the
// thing you saved. It is written beside the OMI keys rather than inside them, and a reader that
// does not know it ignores it -- which is what `extras`-style additions are for.
//
// Panorama presets name their image relative to the `.sky.json`, the same rule glTF external assets
// follow (docs/external_assets.md), so a preset and its `.hdr` travel as a pair.

#include <filesystem>
#include <optional>

#include <glm/gtc/quaternion.hpp>

#include "gltf_environment_sky.hpp"

namespace sky_preset {

// What a preset carries: the sky itself, and the sun angle that went with it.
//
// `sunRotation` is optional because a file may omit it, and because two of the four sky types have
// no sun to speak of. Absent means "leave the sun alone", not "point it at the default".
struct Preset
{
  SkyDescriptor            sky;
  std::optional<glm::quat> sunRotation;
};

// Writes `preset` as a `.sky.json`. False on any failure, having logged what went wrong.
//
// The path is taken as given -- the caller owns the extension, since this is also reached from a
// command line where the user typed a name.
bool save(const std::filesystem::path& path, const Preset& preset);

// Reads one back. nullopt on a file that is missing, unreadable, or not a JSON object, in each case
// with a log line naming the file: a preset that silently does nothing is worse than one that fails.
[[nodiscard]] std::optional<Preset> load(const std::filesystem::path& path);

// True for a path this module claims, by extension. Used by the viewport's drop handler to tell a
// preset from the `.hdr` and `.gltf` it also accepts.
[[nodiscard]] bool isPresetPath(const std::filesystem::path& path);

}  // namespace sky_preset
