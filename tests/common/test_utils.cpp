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

#include "test_utils.hpp"
#include <chrono>
#include <ctime>

namespace gltf_test {

std::filesystem::path TestResources::getResourcePath(const std::string& filename)
{
  // Try several locations (from most specific to most general)
  std::vector<std::filesystem::path> searchPaths = {
      // When running from build/tests/Debug
      std::filesystem::current_path() / "resources",
      // When running from _bin/Debug (nvpro2 structure)
      std::filesystem::current_path() / ".." / ".." / "resources",
      // When running from build/tests
      std::filesystem::current_path() / ".." / "resources",
      // When running from project root
      std::filesystem::current_path() / "resources",
      // Absolute path to source (as backup) - try to find via executable path
      std::filesystem::current_path().parent_path().parent_path() / "resources",
  };

  for(const auto& base : searchPaths)
  {
    auto fullPath = base / filename;
    if(std::filesystem::exists(fullPath))
    {
      return fullPath;
    }
  }

  // Fallback: glTF-Sample-Assets layout (e.g. C:/src/work/glTF-Sample-Assets)
  std::filesystem::path sampleBase = getSampleAssetsPath();
  if(!sampleBase.empty())
  {
    std::filesystem::path samplePath;
    if(filename == "Box.glb")
      samplePath = sampleBase / "Models" / "Box" / "glTF-Binary" / "Box.glb";
    else if(filename == "Box.gltf")
      samplePath = sampleBase / "Models" / "Box" / "glTF" / "Box.gltf";
    else if(filename == "shader_ball.gltf")
      samplePath = sampleBase / "Models" / "ShaderBall" / "glTF" / "ShaderBall.gltf";
    if(!samplePath.empty() && std::filesystem::exists(samplePath))
      return samplePath;
  }

  throw std::runtime_error("Test resource not found: " + filename);
}

std::filesystem::path TestResources::getTempPath(const std::string& filename)
{
  auto tempDir = std::filesystem::temp_directory_path() / "gltf_renderer_tests";
  std::filesystem::create_directories(tempDir);
  return tempDir / filename;
}

std::filesystem::path TestResources::getSampleAssetsPath()
{
#ifdef GLTF_SAMPLE_ASSETS_PATH
  std::filesystem::path basePath = GLTF_SAMPLE_ASSETS_PATH;
  if(std::filesystem::exists(basePath))
  {
    return basePath;
  }
#endif
  return std::filesystem::path();  // Return empty if not found
}

void TestResources::cleanupTempFiles()
{
  auto tempDir = std::filesystem::temp_directory_path() / "gltf_renderer_tests";
  if(std::filesystem::exists(tempDir))
  {
    std::filesystem::remove_all(tempDir);
  }
}

#ifdef GTEST_API_
void SceneTestFixture::SetUp()
{
  m_tempDir = std::filesystem::temp_directory_path() / ("gltf_test_" + std::to_string(std::time(nullptr)));
  std::filesystem::create_directories(m_tempDir);
}

void SceneTestFixture::TearDown()
{
  if(std::filesystem::exists(m_tempDir))
  {
    std::filesystem::remove_all(m_tempDir);
  }
}
#endif

bool VisualValidator::compareScreenshots(const std::filesystem::path& expected, const std::filesystem::path& actual, float tolerance)
{
  // TODO: Implement visual comparison using image library
  // For now, just check that files exist
  return std::filesystem::exists(expected) && std::filesystem::exists(actual);
}

void PerformanceTimer::start()
{
  m_start = std::chrono::high_resolution_clock::now();
}

double PerformanceTimer::stop()
{
  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
  return duration.count() / 1000.0;  // Convert to milliseconds
}

}  // namespace gltf_test
