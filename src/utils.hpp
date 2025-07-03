/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <vector>
#include <string>
#include <filesystem>

#include <vulkan/vulkan_core.h>
#include <nvutils/file_operations.hpp>

namespace nvsamples {

inline static std::vector<std::filesystem::path> getResourcesDirs()
{
  std::filesystem::path exePath = nvutils::getExecutablePath().parent_path();
  return {
      std::filesystem::absolute(exePath / std::filesystem::path(PROJECT_EXE_TO_ROOT_DIRECTORY) / "resources"),
      std::filesystem::absolute(exePath / "resources"),                 //
      std::filesystem::absolute(exePath / PROJECT_NAME / "resources"),  //
      std::filesystem::absolute(exePath)                                //
  };
}

inline static std::vector<std::filesystem::path> getShaderDirs()
{
  std::filesystem::path exePath = nvutils::getExecutablePath().parent_path();
  std::filesystem::path exeName = nvutils::getExecutablePath().stem();
  return {
      std::filesystem::absolute(exePath / std::filesystem::path(PROJECT_EXE_TO_SOURCE_DIRECTORY) / "shaders"),
      std::filesystem::absolute(exePath / std::filesystem::path(PROJECT_EXE_TO_NVSHADERS_DIRECTORY)),
      std::filesystem::absolute(exePath / std::filesystem::path(PROJECT_EXE_TO_ROOT_DIRECTORY) / "common"),
      std::filesystem::absolute(exePath / exeName / "shaders"),
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
