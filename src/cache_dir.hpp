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

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <random>
#include <system_error>

#include <nvutils/file_operations.hpp>

//--------------------------------------------------------------------------------------------------
// Where the application keeps what it learns to start faster on the next run: the compiled shaders
// (spirv_cache.hpp), the driver's compiled pipelines (pipeline_cache_util.hpp) and NGX's answer about
// its device extensions (dlss_wrapper.cpp). One folder next to the executable, safe to delete: every
// file in it is rebuilt on demand.
//--------------------------------------------------------------------------------------------------
namespace nvsamples {

// The cache folder: `cache/` next to the executable.
inline std::filesystem::path getCacheDir()
{
  return nvutils::getExecutablePath().parent_path() / "cache";
}

// An earlier build kept its caches loose next to the executable. Move one (a file or a folder) into
// the cache folder, so what it holds is not redone and the old copy does not linger. Does nothing
// when `legacy` is gone or `current` already exists.
inline void adoptLegacyCache(const std::filesystem::path& legacy, const std::filesystem::path& current)
{
  std::error_code ec;
  if(!std::filesystem::exists(legacy, ec) || std::filesystem::exists(current, ec))
    return;
  std::filesystem::create_directories(current.parent_path(), ec);
  std::filesystem::rename(legacy, current, ec);
}

// A temporary path beside `target`, different on every call. A cache file is written there and renamed
// over the target, so a reader never sees half a file; the name has to be the writer's own, since two
// instances of the application share this folder and would otherwise truncate each other's file.
inline std::filesystem::path uniqueTempPath(const std::filesystem::path& target)
{
  std::random_device random;
  const uint64_t     id = (static_cast<uint64_t>(random()) << 32) ^ random();
  char               suffix[32];
  std::snprintf(suffix, sizeof(suffix), ".%016llx.tmp", static_cast<unsigned long long>(id));
  std::filesystem::path temp = target;
  temp += suffix;
  return temp;
}

// Temporary files a writer that was killed left behind: older than any write in progress.
inline void removeStaleTempFiles(const std::filesystem::path& dir)
{
  std::error_code                       ec;
  const std::filesystem::file_time_type now = std::filesystem::file_time_type::clock::now();
  std::filesystem::directory_iterator   it(dir, ec);
  for(; !ec && it != std::filesystem::directory_iterator{}; it.increment(ec))
  {
    if(it->path().extension() == ".tmp" && now - it->last_write_time(ec) > std::chrono::hours(1))
      std::filesystem::remove(it->path(), ec);
  }
}

}  // namespace nvsamples
