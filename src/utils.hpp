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

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include <nvutils/file_operations.hpp>
#include <vulkan/vulkan_core.h>

namespace nvsamples {

// A templated rolling average calculator similar to ImGui's FPS calculation
template <typename T, size_t N>
class RollingAverage
{
public:
  static constexpr size_t SAMPLE_COUNT = N;

  RollingAverage()
      : m_frameIdx(0)
      , m_frameCount(0)
      , m_accum(T{})
  {
    std::fill(std::begin(m_values), std::end(m_values), T{});
  }

  // Add a new value to the rolling average
  void addValue(T value)
  {
    m_accum += value - m_values[m_frameIdx];
    m_values[m_frameIdx] = value;
    m_frameIdx           = (m_frameIdx + 1) % N;
    m_frameCount         = std::min(m_frameCount + 1, static_cast<int>(N));
  }

  // Get the current rolling average
  T getAverage() const
  {
    if(m_frameCount <= 0 || m_accum <= T{})
      return T{};

    return m_accum / static_cast<T>(m_frameCount);
  }

  // Get the number of samples currently in the average
  int getSampleCount() const { return m_frameCount; }

  // Check if the rolling average has enough samples
  bool hasData() const { return m_frameCount > 0; }

  // Reset the rolling average to initial state
  void reset()
  {
    m_frameIdx   = 0;
    m_frameCount = 0;
    m_accum      = T{};
    std::fill(std::begin(m_values), std::end(m_values), T{});
  }

  // Get the total accumulated value
  T getAccumulated() const { return m_accum; }

private:
  T   m_values[N];   // Circular buffer of values
  int m_frameIdx;    // Current frame index in circular buffer
  int m_frameCount;  // Number of frames accumulated
  T   m_accum;       // Accumulated values
};

inline static std::vector<std::filesystem::path> getResourcesDirs()
{
  std::filesystem::path exePath = nvutils::getExecutablePath().parent_path();
  return {
      std::filesystem::absolute(exePath / TARGET_EXE_TO_SOURCE_DIRECTORY / "resources"),
      std::filesystem::absolute(exePath / "resources"),                         //
      std::filesystem::absolute(exePath / TARGET_NAME "_files" / "resources"),  //
      std::filesystem::absolute(exePath)                                        //
  };
}

inline static std::vector<std::filesystem::path> getShaderDirs()
{
  std::filesystem::path exePath = nvutils::getExecutablePath().parent_path();
  return {
      std::filesystem::absolute(exePath / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders"),
      std::filesystem::absolute(exePath / TARGET_EXE_TO_NVSHADERS_DIRECTORY),
      std::filesystem::absolute(exePath / TARGET_NAME "_files" / "shaders"),
      std::filesystem::absolute(exePath),
  };
}

// Helper to display a little (?) mark which shows a tooltip when hovered.
// In your own code you may want to display an actual icon if you are using a merged icon fonts (see docs/FONTS.md)
inline void HelpMarker(const char* desc)
{
  ImGui::SameLine();
  ImGui::TextDisabled("(?)");
  if(ImGui::BeginItemTooltip())
  {
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
    ImGui::TextUnformatted(desc);
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
  }
}


}  // namespace nvsamples
