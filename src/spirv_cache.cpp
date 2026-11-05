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

#include "spirv_cache.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <mutex>
#include <string>

#include <fmt/format.h>
#include <nvutils/file_operations.hpp>

#include "cache_dir.hpp"

namespace fs = std::filesystem;

namespace spirvcache {

namespace {

constexpr size_t kMaxEntries  = 64;  // Each is a megabyte or two; the least recently used go first
constexpr size_t kMaxFileSize = 64ull * 1024 * 1024;

std::mutex g_mutex;  // The prefetch thread and the compile worker may both hit the directory

// The folder of the compiled shaders, in the cache folder; a shader_cache/ of an earlier build is moved into it.
fs::path cacheDir()
{
  static const fs::path dir = [] {
    fs::path path = nvsamples::getCacheDir() / "spirv";
    nvsamples::adoptLegacyCache(nvutils::getExecutablePath().parent_path() / "shader_cache", path);
    return path;
  }();
  return dir;
}

// 64-bit FNV-1a, fed in pieces
class Hasher
{
public:
  void add(const void* data, size_t size)
  {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for(size_t i = 0; i < size; ++i)
      m_hash = (m_hash ^ bytes[i]) * 0x100000001b3ull;
  }
  void add(std::string_view text) { add(text.data(), text.size()); }
  template <class T>
  void addValue(const T& value)
  {
    add(&value, sizeof(value));
  }
  uint64_t value() const { return m_hash; }

private:
  uint64_t m_hash = 0xcbf29ce484222325ull;
};

// Whether a file is a source Slang can include, so one that goes into the key.
bool isShaderSource(const fs::path& path)
{
  const std::string ext = path.extension().string();
  return ext == ".slang" || ext == ".h" || ext == ".hpp" || ext == ".inl";
}

// A source is identified by its name, size and modification time, as a build system does: reading
// forty files to hash them costs more than the cache saves (antivirus scans each open).
void hashFile(Hasher& hasher, const fs::path& path)
{
  std::error_code ec;
  hasher.add(path.filename().string());
  hasher.addValue(uint64_t(fs::file_size(path, ec)));
  hasher.addValue(int64_t(fs::last_write_time(path, ec).time_since_epoch().count()));
}

// The shader sources an iterator reaches
template <class Iterator>
void appendSources(Iterator it, std::vector<fs::path>& sources)
{
  std::error_code ec;
  for(; !ec && it != Iterator{}; it.increment(ec))
  {
    if(it->is_regular_file(ec) && isShaderSource(it->path()))
      sources.push_back(it->path());
  }
}

}  // namespace

// See spirv_cache.hpp.
std::string makeKey(std::span<const fs::path> shaderDirs, std::string_view variantText, uint32_t vendorId)
{
  Hasher hasher;
  hasher.add("spirv-cache-v1");  // Bump when the layout of what is hashed changes
  hasher.add(variantText);
  hasher.addValue(vendorId);

  // Every source Slang can pull in, in a fixed order
  // (the executable's own folder is one of the search paths: only its top level, not utils/ or the caches)
  const fs::path        exeDir = nvutils::getExecutablePath().parent_path().lexically_normal();
  std::vector<fs::path> sources;
  std::error_code       ec;
  for(size_t i = 0; i < shaderDirs.size(); ++i)
  {
    if(shaderDirs[i].lexically_normal() == exeDir)
      appendSources(fs::directory_iterator(shaderDirs[i], fs::directory_options::skip_permission_denied, ec), sources);
    else
      appendSources(fs::recursive_directory_iterator(shaderDirs[i], fs::directory_options::skip_permission_denied, ec), sources);
    ec.clear();
  }
  // The same file can be reachable through two search paths
  std::sort(sources.begin(), sources.end());
  sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
  for(size_t i = 0; i < sources.size(); ++i)
    hashFile(hasher, sources[i]);

  // The compiler: another build of Slang may emit other SPIR-V from the same sources
  for(const fs::directory_entry& entry : fs::directory_iterator(nvutils::getExecutablePath().parent_path(), ec))
  {
    if(entry.path().filename().string().find("slang-compiler") == std::string::npos)
      continue;
    hasher.add(entry.path().filename().string());
    hasher.addValue(uint64_t(entry.file_size(ec)));
    hasher.addValue(int64_t(entry.last_write_time(ec).time_since_epoch().count()));
  }

  return fmt::format("{:016x}", hasher.value());
}

// See spirv_cache.hpp.
bool load(const std::string& key, std::vector<uint32_t>& spirv)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  const fs::path              path = cacheDir() / (key + ".spv");
  std::error_code             ec;
  const uintmax_t             size = fs::file_size(path, ec);
  if(ec || size == 0 || size > kMaxFileSize || size % sizeof(uint32_t) != 0)
    return false;

  std::ifstream file(path, std::ios::binary);
  if(!file)
    return false;
  std::vector<uint32_t> data(size / sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(data.data()), std::streamsize(size));
  if(!file)
    return false;
  constexpr uint32_t kSpirvMagic = 0x07230203u;
  if(data[0] != kSpirvMagic)
    return false;  // Damaged: the caller compiles, and store() replaces it

  fs::last_write_time(path, fs::file_time_type::clock::now(), ec);  // Used now, for the trim
  spirv = std::move(data);
  return true;
}

// See spirv_cache.hpp.
void store(const std::string& key, std::span<const uint32_t> spirv)
{
  if(spirv.empty())
    return;
  std::lock_guard<std::mutex> lock(g_mutex);
  std::error_code             ec;
  fs::create_directories(cacheDir(), ec);

  const fs::path path = cacheDir() / (key + ".spv");
  const fs::path temp = nvsamples::uniqueTempPath(path);  // this writer's own: other instances share the folder
  {
    std::ofstream file(temp, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(spirv.data()), std::streamsize(spirv.size_bytes()));
    if(!file)
    {
      file.close();
      fs::remove(temp, ec);
      return;
    }
  }
  fs::rename(temp, path, ec);  // Replaces an existing file on Windows too
  if(ec)
  {
    fs::remove(temp, ec);
    return;
  }

  // Trim: the least recently used beyond kMaxEntries
  std::vector<std::pair<fs::file_time_type, fs::path>> entries;
  for(const fs::directory_entry& entry : fs::directory_iterator(cacheDir(), ec))
  {
    if(entry.path().extension() == ".spv")
      entries.push_back({entry.last_write_time(ec), entry.path()});
  }
  if(entries.size() > kMaxEntries)
  {
    std::sort(entries.begin(), entries.end());  // Oldest first
    for(size_t i = 0; i + kMaxEntries < entries.size(); ++i)
      fs::remove(entries[i].second, ec);
  }
  nvsamples::removeStaleTempFiles(cacheDir());
}

}  // namespace spirvcache
