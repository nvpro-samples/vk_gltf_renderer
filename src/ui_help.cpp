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

#include "ui_help.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cfloat>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <imgui.h>

#include <nvgui/fonts.hpp>
#include <nvutils/camera_manipulator.hpp>

#include "camera_controls.hpp"
#include "ui_commands.hpp"
#include "version.hpp"

namespace ui::help {
namespace {

constexpr const char* kRepositoryUrl = "https://github.com/nvpro-samples/vk_gltf_renderer";
constexpr const char* kUserGuideUrl  = "https://github.com/nvpro-samples/vk_gltf_renderer/blob/main/docs/user-guide.md";
constexpr const char* kReleasesUrl   = "https://github.com/nvpro-samples/vk_gltf_renderer/releases";
constexpr const char* kIssuesUrl     = "https://github.com/nvpro-samples/vk_gltf_renderer/issues";

constexpr const char* kShortcutsWindow = ICON_MS_KEYBOARD " Keyboard & Mouse###HelpShortcuts";
constexpr const char* kAboutWindow     = ICON_MS_INFO " About###HelpAbout";

//--------------------------------------------------------------------------------------------------
// Bindings

// A row of the keyboard and mouse lists: a command's binding (commandRow()), or camera table rows.
struct KeyRow
{
  const char*                  action{nullptr};
  std::array<ImGuiKeyChord, 4> chords{};  // Alternatives, drawn "W / S"; unused entries are ImGuiKey_None
  const char* text{nullptr};              // After chords[0]'s modifiers instead of chords: a typed character, a gesture
  bool        viewportOnly{false};
  const char* tooltip{nullptr};
};

KeyRow commandRow(const ui::Command& cmd)
{
  return {cmd.label, {cmd.binding.chord}, cmd.binding.text, cmd.binding.scope == ui::KeyScope::eViewport, cmd.tooltip};
}

// The registry commands that move the camera, listed under the camera keys as well.
constexpr const char* kCameraCommands[] = {"view.homeCamera", "view.fitObject", "view.fitScene"};

// The mouse table's rows: the drag and wheel inputs of the camera binding table.
constexpr std::array<std::pair<camera_controls::Input, const char*>, 4> kMouseRows = {{
    {camera_controls::Input::eLeftDrag, "Left drag"},
    {camera_controls::Input::eMiddleDrag, "Middle drag"},
    {camera_controls::Input::eRightDrag, "Right drag"},
    {camera_controls::Input::eWheel, "Wheel"},
}};

// The mouse table's columns: every modifier combination a mouse binding names, fewest keys first and
// then Shift, Ctrl, Alt, so a binding added to the table gets its column.
std::vector<camera_controls::Mods> mouseColumns()
{
  std::vector<camera_controls::Mods>        columns{camera_controls::kModNone};
  std::span<const camera_controls::Binding> bindings = camera_controls::bindings();
  for(size_t i = 0; i < bindings.size(); ++i)
  {
    const camera_controls::Binding& b = bindings[i];
    if(camera_controls::isMotion(b.input) && std::find(columns.begin(), columns.end(), b.mods) == columns.end())
      columns.push_back(b.mods);
  }
  const auto order = [](camera_controls::Mods m) {
    const unsigned keys = ((m & ImGuiMod_Shift) ? 1U : 0U) | ((m & ImGuiMod_Ctrl) ? 2U : 0U) | ((m & ImGuiMod_Alt) ? 4U : 0U);
    return std::pair(std::popcount(keys), keys);
  };
  std::stable_sort(columns.begin(), columns.end(),
                   [&](camera_controls::Mods a, camera_controls::Mods b) { return order(a) < order(b); });
  return columns;
}

// The clicks for the current mode, drawn as their modifiers, if any, then "Click" / "Double-click".
std::vector<KeyRow> clickRows(const nvutils::CameraManipulator& camera)
{
  std::vector<KeyRow>                       rows;
  std::span<const camera_controls::Binding> bindings = camera_controls::bindings();
  for(size_t i = 0; i < bindings.size(); ++i)
  {
    const camera_controls::Binding& b           = bindings[i];
    const bool                      doubleClick = b.input == camera_controls::Input::eLeftDoubleClick;
    if((b.input != camera_controls::Input::eLeftClick && !doubleClick)
       || !camera_controls::matches(b, b.mods, camera.getMode(), camera.getProjectionType()))
      continue;
    rows.push_back({b.label, {b.mods}, doubleClick ? "Double-click" : "Click", false, b.tooltip});
  }
  return rows;
}

// The camera keys for the current mode, one row per label, "W / S" for "Move forward / back".
std::vector<KeyRow> cameraKeyRows(const nvutils::CameraManipulator& camera)
{
  std::vector<KeyRow>                       rows;
  std::span<const camera_controls::Binding> bindings = camera_controls::bindings();
  for(size_t i = 0; i < bindings.size(); ++i)
  {
    const camera_controls::Binding& b = bindings[i];
    if(b.input != camera_controls::Input::eKey || !camera_controls::matches(b, b.mods, camera.getMode(), camera.getProjectionType()))
      continue;
    if(rows.empty() || std::strcmp(rows.back().action, b.label) != 0)
      rows.push_back({b.label, {}, nullptr, true, b.tooltip});
    // The first free slot of the row's alternatives.
    std::array<ImGuiKeyChord, 4>& chords = rows.back().chords;
    size_t                        slot   = 0;
    while(slot < chords.size() && chords[slot] != ImGuiKey_None)
      ++slot;
    assert(slot < chords.size() && "More keys share a label than a KeyRow holds");
    if(slot < chords.size())
      chords[slot] = b.mods | b.key;
  }
  return rows;
}

//--------------------------------------------------------------------------------------------------
// Keycaps

// On a keycap the arrows are drawn as arrows; the search matches them by name.
const char* keycapLabel(ImGuiKey key)
{
  switch(key)
  {
    case ImGuiKey_LeftArrow:
      return ICON_MS_KEYBOARD_ARROW_LEFT;
    case ImGuiKey_RightArrow:
      return ICON_MS_KEYBOARD_ARROW_RIGHT;
    case ImGuiKey_UpArrow:
      return ICON_MS_KEYBOARD_ARROW_UP;
    case ImGuiKey_DownArrow:
      return ICON_MS_KEYBOARD_ARROW_DOWN;
    default:
      return ui::keyName(key);
  }
}

// One key drawn as a keycap; filled with the accent color while it is held, so pressing keys with
// the window open shows which entries they belong to.
void keycap(const char* label, bool held)
{
  const ImGuiStyle& style    = ImGui::GetStyle();
  const ImVec2      textSize = ImGui::CalcTextSize(label);
  const float       height   = ImGui::GetFrameHeight();
  const float       padX     = style.FramePadding.x * 0.75F;
  const ImVec2      size(std::max(textSize.x + 2.0F * padX, height), height);
  const ImVec2      pos = ImGui::GetCursorScreenPos();
  ImGui::Dummy(size);

  ImDrawList*  drawList = ImGui::GetWindowDrawList();
  const float  rounding = std::max(style.FrameRounding, 3.0F);
  const ImVec2 min(pos.x, pos.y + 1.0F);
  const ImVec2 max(pos.x + size.x, pos.y + size.y - 1.0F);
  // A darker band under the cap reads as its side; a held key sits flush with it.
  const float lift = held ? 0.0F : 2.0F;
  drawList->AddRectFilled(min, max, ImGui::GetColorU32(ImGuiCol_Border), rounding);
  drawList->AddRectFilled(min, ImVec2(max.x, max.y - lift),
                          ImGui::GetColorU32(held ? ImGuiCol_ButtonActive : ImGuiCol_FrameBgHovered), rounding);
  drawList->AddRect(min, max, ImGui::GetColorU32(ImGuiCol_Border), rounding);
  drawList->AddText(ImVec2(pos.x + (size.x - textSize.x) * 0.5F, min.y + (max.y - lift - min.y - textSize.y) * 0.5F),
                    ImGui::GetColorU32(ImGuiCol_Text), label);
}

bool isHeld(ImGuiKeyChord chord)
{
  const ImGuiIO& io   = ImGui::GetIO();
  const ImGuiKey key  = ImGuiKey(chord & ~ImGuiMod_Mask_);
  const bool modsHeld = ((chord & ImGuiMod_Ctrl) == 0 || io.KeyCtrl) && ((chord & ImGuiMod_Shift) == 0 || io.KeyShift)
                        && ((chord & ImGuiMod_Alt) == 0 || io.KeyAlt);
  return modsHeld && (key == ImGuiKey_None || ImGui::IsKeyDown(key));
}

void separatorText(const char* text)
{
  ImGui::SameLine(0.0F, ImGui::GetStyle().ItemInnerSpacing.x);
  ImGui::AlignTextToFramePadding();
  ImGui::TextDisabled("%s", text);
  ImGui::SameLine(0.0F, ImGui::GetStyle().ItemInnerSpacing.x);
}

void chordCaps(ImGuiKeyChord chord)
{
  const ImGuiIO& io    = ImGui::GetIO();
  bool           first = true;
  auto           cap   = [&](const char* label, bool held) {
    if(!first)
      separatorText("+");
    keycap(label, held);
    first = false;
  };
  // The modifiers in the order ui::chordText() spells them.
  if(chord & ImGuiMod_Ctrl)
    cap("Ctrl", io.KeyCtrl);
  if(chord & ImGuiMod_Alt)
    cap("Alt", io.KeyAlt);
  if(chord & ImGuiMod_Shift)
    cap("Shift", io.KeyShift);
  const ImGuiKey key = ImGuiKey(chord & ~ImGuiMod_Mask_);
  if(key != ImGuiKey_None)
    cap(keycapLabel(key), isHeld(chord));
}

void shortcutCaps(const KeyRow& s)
{
  // A text binding: the chord's modifiers, if any, then the text ("Shift + Drop file").
  if(s.text != nullptr)
  {
    if(s.chords[0] != ImGuiKey_None)
    {
      chordCaps(s.chords[0]);
      separatorText("+");
    }
    keycap(s.text, false);
    return;
  }
  for(size_t i = 0; i < s.chords.size() && s.chords[i] != ImGuiKey_None; ++i)
  {
    if(i > 0)
      separatorText("/");
    chordCaps(s.chords[i]);
  }
}

// What the search box matches a shortcut against: its group, its action, and its keys spelled as
// they are drawn ("Ctrl+Shift+S"), so typing either a name or a key combination finds it.
std::string searchText(const char* group, const KeyRow& s)
{
  std::string text = fmt::format("{} {} ", group, s.action);
  if(s.text != nullptr)
    return text + ui::chordText(s.chords[0]) + s.text;
  for(ImGuiKeyChord chord : s.chords)
    if(chord != ImGuiKey_None)
      text += " " + ui::chordText(chord);
  return text;
}

//--------------------------------------------------------------------------------------------------
// Tables

constexpr ImGuiTableFlags kListTableFlags =
    ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_PadOuterX | ImGuiTableFlags_SizingStretchProp;

void viewportTag()
{
  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  ImGui::TextDisabled(ICON_MS_PAN_TOOL);
  ImGui::SetItemTooltip("Only while the cursor is over the viewport, or the viewport has focus.");
}

// One row of action | keys. showViewportTag marks a viewport-only shortcut, for lists that mix them
// with global ones.
void shortcutRow(const KeyRow& s, bool showViewportTag)
{
  ImGui::PushID(s.action);
  ImGui::TableNextRow();
  ImGui::TableNextColumn();
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(s.action);
  if(s.tooltip != nullptr)
    ImGui::SetItemTooltip("%s", s.tooltip);
  if(showViewportTag && s.viewportOnly)
    viewportTag();
  ImGui::TableNextColumn();
  shortcutCaps(s);
  ImGui::PopID();
}

bool beginListTable(const char* id)
{
  if(!ImGui::BeginTable(id, 2, kListTableFlags))
    return false;
  ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch, 1.0F);
  ImGui::TableSetupColumn("Keys", ImGuiTableColumnFlags_WidthStretch, 1.0F);
  return true;
}

//--------------------------------------------------------------------------------------------------
// Camera tab

// The mouse table follows the input live: the column of the modifiers being held and the row of the
// button being dragged or the wheel being turned light up, and so does the cell of the binding they
// trigger. The wheel has no "held" state, so its row stays lit for a moment after each step.
struct LiveInput
{
  camera_controls::Mods mods{camera_controls::kModNone};
  int                   row{-1};  // Into kMouseRows; -1: no mouse input
};

LiveInput liveInput(bool overThisWindow)
{
  // The wheel has no held state, so its row stays lit for a moment after the last step: the time of
  // that step has to outlive the frame.
  static double  s_lastWheelTime = -1.0;
  const ImGuiIO& io              = ImGui::GetIO();

  LiveInput live{camera_controls::heldMods()};
  if(overThisWindow)
    return live;
  if(io.MouseWheel != 0.0F)
    s_lastWheelTime = ImGui::GetTime();
  if(ImGui::IsMouseDown(ImGuiMouseButton_Left))
    live.row = 0;
  else if(ImGui::IsMouseDown(ImGuiMouseButton_Middle))
    live.row = 1;
  else if(ImGui::IsMouseDown(ImGuiMouseButton_Right))
    live.row = 2;
  else if(s_lastWheelTime >= 0.0 && ImGui::GetTime() - s_lastWheelTime < 0.4)
    live.row = 3;
  return live;
}

void cameraTab(const nvutils::CameraManipulator& camera, const ui::CommandRegistry& commands, bool overThisWindow)
{
  static constexpr std::array<const char*, 3>      kModeNames = {"Examine", "Fly", "Walk"};
  const nvutils::CameraManipulator::Modes          mode       = camera.getMode();
  const nvutils::CameraManipulator::ProjectionType projection = camera.getProjectionType();
  const char* modeName = kModeNames[std::min<size_t>(size_t(mode), kModeNames.size() - 1)];

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted(ICON_MS_MOUSE " Mouse");
  ImGui::SameLine();
  ImGui::TextDisabled("%s mode%s", modeName, projection == nvutils::CameraManipulator::Orthographic ? ", orthographic" : "");
  ImGui::SetItemTooltip("The current navigation mode, set in the Camera window (F1). The table follows it.");

  const std::vector<camera_controls::Mods> columns  = mouseColumns();
  const LiveInput                          live     = liveInput(overThisWindow);
  auto                                     columnOf = [&](camera_controls::Mods mods) {
    const auto it = std::find(columns.begin(), columns.end(), mods);
    return it != columns.end() ? int(it - columns.begin()) : -1;
  };
  // The column of the held modifiers, if they have one. With nothing held, "No modifier" would stay
  // lit for good; light it only while the mouse is in use.
  const int litColumn = (live.mods != camera_controls::kModNone || live.row >= 0) ? columnOf(live.mods) : -1;
  // The binding the live input triggers. When the held modifiers have no column (Ctrl+Shift), its
  // cell is the one of the modifiers it names.
  const camera_controls::Binding* litBinding =
      live.row >= 0 ? camera_controls::find(kMouseRows[live.row].first, live.mods, mode, projection) : nullptr;
  const int litCell = litColumn >= 0 ? litColumn : (litBinding != nullptr ? columnOf(litBinding->mods) : -1);

  const ImU32 columnColor = ImGui::GetColorU32(ImGuiCol_HeaderHovered, 0.35F);
  const ImU32 cellColor   = ImGui::GetColorU32(ImGuiCol_ButtonActive);

  constexpr ImGuiTableFlags kGridFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_PadOuterX;
  if(ImGui::BeginTable("CameraMouse", int(columns.size()) + 1, kGridFlags))
  {
    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed,
                            ImGui::CalcTextSize("Middle drag").x + ImGui::GetStyle().CellPadding.x * 2.0F);
    for(size_t c = 0; c < columns.size(); ++c)
      ImGui::TableSetupColumn("");

    // Header: modifier keycaps, the held ones lit.
    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
    ImGui::TableNextColumn();
    for(int c = 0; c < int(columns.size()); ++c)
    {
      ImGui::TableNextColumn();
      if(c == litColumn)
        ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, columnColor);
      if(columns[c] == camera_controls::kModNone)
      {
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("No modifier");
      }
      else
        chordCaps(columns[c]);
    }

    for(int r = 0; r < int(kMouseRows.size()); ++r)
    {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::AlignTextToFramePadding();
      if(r == live.row)
        ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, columnColor);
      ImGui::TextUnformatted(kMouseRows[r].second);
      for(int c = 0; c < int(columns.size()); ++c)
      {
        ImGui::TableNextColumn();
        const camera_controls::Binding* binding = camera_controls::find(kMouseRows[r].first, columns[c], mode, projection);
        if(r == live.row && c == litCell && binding != nullptr && binding == litBinding)
          ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, cellColor);
        else if(c == litColumn)
          ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, columnColor);
        if(binding == nullptr)
          continue;
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(binding->label);
        if(binding->tooltip != nullptr)
          ImGui::SetItemTooltip("%s", binding->tooltip);
      }
    }
    ImGui::EndTable();
  }
  ImGui::TextDisabled("Hold a modifier or drag in the viewport to see where you are in the table.");

  ImGui::Spacing();
  if(beginListTable("CameraClicks"))
  {
    for(const KeyRow& row : clickRows(camera))
      shortcutRow(row, false);
    ImGui::EndTable();
  }

  ImGui::Spacing();
  ImGui::SeparatorText(ICON_MS_KEYBOARD " Keyboard");
  if(beginListTable("CameraKeys"))
  {
    for(const KeyRow& row : cameraKeyRows(camera))
      shortcutRow(row, false);
    for(const char* id : kCameraCommands)
      shortcutRow(commandRow(commands.get(id)), false);
    ImGui::EndTable();
  }
  ImGui::TextDisabled("The movement keys and Home work over the viewport. Ctrl or Alt turns the movement keys off.");
}

//--------------------------------------------------------------------------------------------------
// Shortcuts tab

void shortcutsTab(const ui::CommandRegistry& commands)
{
  // The search text, kept while the window is closed and reopened.
  static ImGuiTextFilter s_filter;
  ImGui::SetNextItemWidth(-FLT_MIN);
  if(ImGui::IsWindowAppearing())
    ImGui::SetKeyboardFocusHere();
  s_filter.Draw("##Search", 0.0F);
  // The hint, drawn over the empty box; ImGuiTextFilter::Draw() takes no hint of its own.
  if(!s_filter.IsActive() && !ImGui::IsItemActive())
  {
    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 pad = ImGui::GetStyle().FramePadding;
    ImGui::GetWindowDrawList()->AddText(ImVec2(min.x + pad.x, min.y + pad.y), ImGui::GetColorU32(ImGuiCol_TextDisabled),
                                        ICON_MS_SEARCH " Search actions or keys, e.g. \"save\" or \"ctrl+shift\"");
  }

  if(ImGui::BeginChild("ShortcutList"))
  {
    const ImGuiTextFilter* activeFilter = s_filter.IsActive() ? &s_filter : nullptr;
    size_t                 total        = 0;
    // The groups in the order they first appear, which is the menus' order.
    std::vector<const char*> groups;
    for(const ui::Command& cmd : commands.all())
      if(cmd.binding.isBound()
         && std::none_of(groups.begin(), groups.end(), [&](const char* g) { return std::strcmp(g, cmd.group) == 0; }))
        groups.push_back(cmd.group);

    for(const char* group : groups)
    {
      // The group's bound commands that pass the filter; a group with none is left out.
      std::vector<KeyRow> rows;
      for(const ui::Command& cmd : commands.all())
      {
        if(!cmd.binding.isBound() || std::strcmp(cmd.group, group) != 0)
          continue;
        const KeyRow row = commandRow(cmd);
        if(activeFilter == nullptr || activeFilter->PassFilter(searchText(group, row).c_str()))
          rows.push_back(row);
      }
      if(rows.empty())
        continue;

      ImGui::SeparatorText(group);
      if(beginListTable(group))
      {
        for(const KeyRow& row : rows)
          shortcutRow(row, true);
        ImGui::EndTable();
      }
      total += rows.size();
    }
    if(total == 0)
      ImGui::TextDisabled("No shortcut matches \"%s\".", s_filter.InputBuf);

    ImGui::Spacing();
    ImGui::TextDisabled(ICON_MS_PAN_TOOL " Only while the cursor is over the viewport, or the viewport has focus.");
  }
  ImGui::EndChild();
}

//--------------------------------------------------------------------------------------------------
// About

void openUrl(const char* url)
{
  const ImGuiPlatformIO& platformIO = ImGui::GetPlatformIO();
  if(platformIO.Platform_OpenInShellFn != nullptr)
    platformIO.Platform_OpenInShellFn(ImGui::GetCurrentContext(), url);
}

struct SystemInfo
{
  std::string gpu;
  std::string driver;
  std::string vulkan;  // Driver, loader and header versions: the three often differ
};

std::string versionText(uint32_t version)
{
  return fmt::format("{}.{}.{}", VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version), VK_API_VERSION_PATCH(version));
}

SystemInfo systemInfo(VkPhysicalDevice physicalDevice)
{
  SystemInfo info;
  if(physicalDevice == VK_NULL_HANDLE)
    return info;
  VkPhysicalDeviceDriverProperties driverProps{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
  VkPhysicalDeviceProperties2 props{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &driverProps};
  vkGetPhysicalDeviceProperties2(physicalDevice, &props);
  info.gpu    = props.properties.deviceName;
  info.driver = fmt::format("{} {}", driverProps.driverName, driverProps.driverInfo);
  // The driver's version is what the app can use on this GPU; the loader comes with the installed
  // Vulkan runtime/SDK, and the headers are the ones this build was compiled against.
  uint32_t loader = VK_API_VERSION_1_0;
  vkEnumerateInstanceVersion(&loader);
  info.vulkan = fmt::format("{} (driver), loader {}, built with {}", versionText(props.properties.apiVersion),
                            versionText(loader), versionText(VK_HEADER_VERSION_COMPLETE));
  return info;
}

void aboutWindow(HelpState& state, VkPhysicalDevice physicalDevice)
{
  ImGui::SetNextWindowSize(ImVec2(ImGui::GetFontSize() * 30.0F, 0.0F), ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));
  if(!ImGui::Begin(kAboutWindow, &state.showAbout, ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize))
  {
    ImGui::End();
    return;
  }

  ImGui::TextUnformatted("Vulkan glTF Renderer");
  ImGui::SameLine();
  ImGui::TextDisabled("%s", APP_VERSION_STRING);
  ImGui::TextWrapped("A glTF 2.0 viewer and editor with an RTX path tracer and a rasterizer, built on nvpro_core2 and Slang.");
  ImGui::Spacing();

  // Queried once: the driver and loader do not change while the application runs.
  static const SystemInfo s_info = systemInfo(physicalDevice);
  if(ImGui::BeginTable("AboutInfo", 2, ImGuiTableFlags_SizingFixedFit))
  {
    for(const auto& [name, value] : {std::pair{"GPU", s_info.gpu.c_str()}, std::pair{"Driver", s_info.driver.c_str()},
                                     std::pair{"Vulkan", s_info.vulkan.c_str()}, std::pair{"Dear ImGui", IMGUI_VERSION}})
    {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextDisabled("%s", name);
      ImGui::TableNextColumn();
      ImGui::TextUnformatted(value);
    }
    ImGui::EndTable();
  }
  if(ImGui::Button(ICON_MS_CONTENT_COPY " Copy"))
  {
    ImGui::SetClipboardText(fmt::format("vk_gltf_renderer {}\nGPU: {}\nDriver: {}\nVulkan: {}\n", APP_VERSION_STRING,
                                        s_info.gpu, s_info.driver, s_info.vulkan)
                                .c_str());
  }
  ImGui::SetItemTooltip("Copy the version and system details, e.g. for an issue report.");

  ImGui::Separator();
  ImGui::TextLinkOpenURL("Repository", kRepositoryUrl);
  ImGui::SameLine();
  ImGui::TextLinkOpenURL("User Guide", kUserGuideUrl);
  ImGui::SameLine();
  ImGui::TextLinkOpenURL("Release Notes", kReleasesUrl);
  ImGui::SameLine();
  ImGui::TextLinkOpenURL("Report an Issue", kIssuesUrl);
  ImGui::TextDisabled("Copyright (c) 2026, NVIDIA CORPORATION. Apache License 2.0.");
  ImGui::End();
}

void shortcutsWindow(HelpState& state, const nvutils::CameraManipulator& camera, const ui::CommandRegistry& commands)
{
  const float em = ImGui::GetFontSize();
  ImGui::SetNextWindowSize(ImVec2(em * 40.0F, em * 36.0F), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5F, 0.5F));
  if(!ImGui::Begin(kShortcutsWindow, &state.showShortcuts, ImGuiWindowFlags_NoCollapse))
  {
    ImGui::End();
    return;
  }
  const bool overThisWindow = ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);
  if(ImGui::BeginTabBar("HelpTabs"))
  {
    if(ImGui::BeginTabItem(ICON_MS_VIDEOCAM " Camera"))
    {
      cameraTab(camera, commands, overThisWindow);
      ImGui::EndTabItem();
    }
    if(ImGui::BeginTabItem(ICON_MS_KEYBOARD " Shortcuts"))
    {
      shortcutsTab(commands);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
  ImGui::End();
}

}  // namespace

void renderMenu(HelpState& state, const ui::CommandRegistry& commands)
{
  // '?' is matched as a typed character, not a key, so it follows the keyboard layout. The modifiers
  // are not tested: the queued character already proves '?' was typed, and on layouts where it takes
  // AltGr, Windows reports Ctrl and Alt held as well.
  const ui::Command& keyboardHelp = commands.get("help.keyboard");
  const ImGuiIO&     io           = ImGui::GetIO();
  const ImWchar      key          = ImWchar(keyboardHelp.binding.text[0]);
  if(!io.WantTextInput)
  {
    for(ImWchar c : io.InputQueueCharacters)
    {
      if(c == key)
      {
        state.showShortcuts = !state.showShortcuts;
        if(state.showShortcuts)
          ImGui::SetWindowFocus(kShortcutsWindow);
      }
    }
  }

  if(!ImGui::BeginMenu("Help"))
    return;
  const std::string keyboardHelpLabel = std::string(keyboardHelp.icon) + " " + keyboardHelp.label;
  if(ImGui::MenuItem(keyboardHelpLabel.c_str(), keyboardHelp.binding.text, &state.showShortcuts) && state.showShortcuts)
    ImGui::SetWindowFocus(kShortcutsWindow);
  ImGui::Separator();
  if(ImGui::MenuItem(ICON_MS_MENU_BOOK " User Guide"))
    openUrl(kUserGuideUrl);
  ImGui::SetItemTooltip("%s", kUserGuideUrl);
  if(ImGui::MenuItem(ICON_MS_NEW_RELEASES " Release Notes"))
    openUrl(kReleasesUrl);
  ImGui::SetItemTooltip("%s", kReleasesUrl);
  if(ImGui::MenuItem(ICON_MS_BUG_REPORT " Report an Issue"))
    openUrl(kIssuesUrl);
  ImGui::SetItemTooltip("%s", kIssuesUrl);
  ImGui::Separator();
  if(ImGui::MenuItem(ICON_MS_INFO " About"))
    state.showAbout = true;
  ImGui::EndMenu();
}

void renderWindows(HelpState& state, const nvutils::CameraManipulator& camera, const ui::CommandRegistry& commands, VkPhysicalDevice physicalDevice)
{
  if(state.showShortcuts)
    shortcutsWindow(state, camera, commands);
  if(state.showAbout)
    aboutWindow(state, physicalDevice);
}

}  // namespace ui::help
