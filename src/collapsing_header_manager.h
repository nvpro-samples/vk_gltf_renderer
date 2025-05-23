#pragma once


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

  // Singleton accessor
  static CollapsingHeaderManager& getInstance()
  {
    static CollapsingHeaderManager instance;
    return instance;
  }

  // Main header function using string_view for efficient string handling
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
  CollapsingHeaderManager()  = default;
  ~CollapsingHeaderManager() = default;

  std::string_view m_openedHeader{};
};