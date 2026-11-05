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
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

//--------------------------------------------------------------------------------------------------
// SPIR-V cache on disk
//
// The path tracer builds a specialized shader per scene and per denoiser state, each one a Slang
// compile of about a second and a half. The result only depends on the shader sources, the macros
// and the compiler, so it is kept in `spirv/` of the cache folder (cache_dir.hpp): the next time the same
// variant is wanted, in this run or in a later one, the compile is a file read. Together with the
// pipeline cache (pipeline_cache_util.hpp), which makes the driver's part a lookup, a variant seen
// before is ready in milliseconds.
//
// Entries are keyed by a hash of everything that shapes the SPIR-V, so an edited shader, a new Slang
// or another GPU vendor never reads a stale file. The directory is trimmed to the most recently used.
//--------------------------------------------------------------------------------------------------
namespace spirvcache {

// The key of one compile. `variantText` carries what is specific to it (the macros, the entry point);
// the sources under `shaderDirs` (by name, size and modification time), the Slang library next to the executable and `vendorId` (it picks
// the debug-info level) are added here. Only stats the source files: cheap next to a compile.
std::string makeKey(std::span<const std::filesystem::path> shaderDirs, std::string_view variantText, uint32_t vendorId);

// The SPIR-V stored under `key`, if any. A hit counts as a use for the trim.
bool load(const std::string& key, std::vector<uint32_t>& spirv);

// Store `spirv` under `key` (atomically, so a reader never sees half a file), then trim the directory.
void store(const std::string& key, std::span<const uint32_t> spirv);

}  // namespace spirvcache
