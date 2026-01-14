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

//////////////////////////////////////////////////////////////////////////
/*
    Collapsing Header Manager for ImGui UI

    This class implements a singleton manager for ImGui collapsing headers that ensures
    only one header can be open at a time. 
    
*/
//////////////////////////////////////////////////////////////////////////

#pragma once
#include "imgui.h"
#include <string_view>
#include <unordered_set>

class CollapsingHeaderManager
{
public:
  // Deleted copy/move operations to enforce singleton pattern
  CollapsingHeaderManager(const CollapsingHeaderManager&)            = delete;
  CollapsingHeaderManager& operator=(const CollapsingHeaderManager&) = delete;
  CollapsingHeaderManager(CollapsingHeaderManager&&)                 = delete;
  CollapsingHeaderManager& operator=(CollapsingHeaderManager&&)      = delete;

  // Singleton accessor - provides global access to the single instance
  static CollapsingHeaderManager& getInstance()
  {
    static CollapsingHeaderManager instance;
    return instance;
  }

  // Main header function
  // Returns true if the header is open, false otherwise
  // Parameters:
  //   name: The name/label of the collapsing header
  //   flags: Optional ImGui tree node flags for customizing header behavior
  [[nodiscard]] bool beginHeader(std::string_view name, ImGuiTreeNodeFlags flags = 0)
  {
    const bool wasOpen = (m_openedHeader == name);

    ImGui::SetNextItemOpen(wasOpen);
    const bool isOpen = ImGui::CollapsingHeader(name.data());

    if(isOpen)
      m_openedHeader = name;
    else if(wasOpen)
      m_openedHeader = {};

    return isOpen;
  }

private:
  // Private constructor and destructor to enforce singleton pattern
  CollapsingHeaderManager()  = default;
  ~CollapsingHeaderManager() = default;

  // Tracks the currently open header using string_view for efficient string comparison
  std::string_view m_openedHeader{};
};