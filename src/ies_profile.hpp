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

#include <cstdint>
#include <span>
#include <vector>

namespace nvvkgltf {

// Number of samples in the normalized candela distribution returned by parseIesProfile(),
// evenly spaced across the vertical angle range [0, 180] degrees (1-degree steps). Must match
// shaderio::GltfScene::iesProfileSampleCount and the stride used when uploading the GPU table.
constexpr int kIesProfileSampleCount = 181;

/*-------------------------------------------------------------------------------------------------
## Function `parseIesProfile`
> Parses an IESNA LM-63 (`.ies`) photometric-web file (EXT_lights_ies) into a normalized,
> azimuthally-averaged candela distribution: `outSamples[i]` is the relative intensity at
> vertical angle `i` degrees off the light's photometric (nadir) axis, for `i` in
> `[0, kIesProfileSampleCount)`. The peak value is normalized to 1.0 -- KHR_lights_punctual's
> `intensity` still carries the light's absolute physical scale; this only reshapes it angularly.

Simplification: multiple horizontal (azimuth) angle planes are averaged into one 1D vertical-angle
curve, so genuinely asymmetric fixtures (e.g. rectangular floodlights) lose their azimuthal
variation. This matches how most real-time engines treat IES profiles for standard
axially-symmetric luminaires and keeps the GPU lookup a simple 1D table -- a full 2D (theta, phi)
table is not implemented. TILT=INCLUDE inline blocks are consumed (to stay positioned in the token
stream) but not applied. TILT=<filename> references an external file that is not loaded; the
photometric data in the main file is parsed normally (no inline block to skip).

Returns false (leaving `outSamples` untouched) if the file cannot be parsed as LM-63.
-------------------------------------------------------------------------------------------------*/
bool parseIesProfile(std::span<const uint8_t> fileBytes, std::vector<float>& outSamples);

}  // namespace nvvkgltf
