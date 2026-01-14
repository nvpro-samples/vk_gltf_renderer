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

#include "ui_xmp.hpp"

#include <cctype>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <imgui.h>

namespace ui_xmp {

//--------------------------------------------------------------------------------------------------
// KHR_xmp_json_ld parsing helpers
// See: https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_xmp_json_ld

static constexpr const char* KHR_XMP_JSON_LD = "KHR_xmp_json_ld";

//--------------------------------------------------------------------------------------------------
// Core extension accessors - centralized lookup logic

// Get XMP extension value from any extension map, returns nullptr if not found
static const tinygltf::Value* getXmpExtension(const tinygltf::ExtensionMap& extensions)
{
  auto it = extensions.find(KHR_XMP_JSON_LD);
  return (it != extensions.end()) ? &it->second : nullptr;
}

// Get packets array from model's root extensions, returns nullptr if not found or empty
static const tinygltf::Value* getXmpPacketsArray(const tinygltf::Model* model)
{
  if(!model)
    return nullptr;

  const tinygltf::Value* ext = getXmpExtension(model->extensions);
  if(!ext || !ext->Has("packets"))
    return nullptr;

  const auto& packets = ext->Get("packets");
  return (packets.IsArray() && packets.ArrayLen() > 0) ? &packets : nullptr;
}

//--------------------------------------------------------------------------------------------------
// Value extraction helpers

// Format a scalar value (string, number, or boolean) as a display string.
// Returns std::nullopt if the value is not a recognized scalar type.
// Note: empty strings are valid values and are preserved (not treated as "no value").
static std::optional<std::string> formatScalarValue(const tinygltf::Value& val)
{
  if(val.IsString())
    return val.Get<std::string>();
  if(val.IsInt() || val.IsNumber())
    return fmt::format("{:g}", val.GetNumberAsDouble());
  if(val.IsBool())
    return val.Get<bool>() ? "true" : "false";
  return std::nullopt;
}

// Extract string: either directly from a scalar value, or from the "@value" field of an object.
// Returns std::nullopt if no value could be extracted.
static std::optional<std::string> getXmpStringValue(const tinygltf::Value& val)
{
  // Try direct scalar extraction
  if(auto result = formatScalarValue(val))
    return result;

  // Try "@value" field for JSON-LD typed values
  if(val.IsObject() && val.Has("@value"))
    return formatScalarValue(val.Get("@value"));

  return std::nullopt;
}

// Extract all values from @list or @set array.
// Empty strings are preserved as valid values.
static std::vector<std::string> getXmpArrayValues(const tinygltf::Value& val)
{
  std::vector<std::string> result;

  if(val.IsString())
  {
    result.push_back(val.Get<std::string>());
    return result;
  }

  if(!val.IsObject())
    return result;

  // Direct @value
  if(val.Has("@value"))
  {
    if(auto s = getXmpStringValue(val))
      result.push_back(*s);
    return result;
  }

  // @list or @set arrays
  const char* arrayKey = val.Has("@list") ? "@list" : (val.Has("@set") ? "@set" : nullptr);
  if(arrayKey)
  {
    const auto& arr = val.Get(std::string(arrayKey));
    if(arr.IsArray())
    {
      const int arrLen = int(arr.ArrayLen());
      for(int i = 0; i < arrLen; i++)
      {
        if(auto item = getXmpStringValue(arr.Get(i)))
          result.push_back(*item);
      }
    }
    return result;
  }

  // rdf:Alt language alternatives - collect all rdf:_N values
  if(val.Has("@type"))
  {
    const auto& type = val.Get("@type");
    if(type.IsString() && type.Get<std::string>() == "rdf:Alt")
    {
      for(const auto& key : val.Keys())
      {
        if(key.rfind("rdf:_", 0) == 0)
        {
          if(auto s = getXmpStringValue(val.Get(key)))
            result.push_back(*s);
        }
      }
    }
  }

  return result;
}

// Convert XMP field key to display label (e.g., "dc:title" -> "Title", "xmpRights:Owner" -> "Owner")
static std::string xmpKeyToLabel(const std::string& key)
{
  // Find the colon separator (namespace:fieldName)
  size_t      colonPos  = key.find(':');
  std::string fieldName = (colonPos != std::string::npos) ? key.substr(colonPos + 1) : key;

  // Capitalize first letter
  if(!fieldName.empty())
    fieldName[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(fieldName[0])));

  return fieldName;
}

// Display XMP field - handles single and multi-value
static void displayXmpField(const std::string& key, const tinygltf::Value& val)
{
  std::vector<std::string> values = getXmpArrayValues(val);
  if(values.empty())
    return;

  std::string label = xmpKeyToLabel(key);

  if(values.size() == 1)
  {
    ImGui::TextWrapped("%s: %s", label.c_str(), values[0].c_str());
  }
  else
  {
    ImGui::Text("%s:", label.c_str());
    ImGui::Indent();
    for(const auto& v : values)
    {
      ImGui::BulletText("%s", v.c_str());
    }
    ImGui::Unindent();
  }
}

// Display XMP packet contents - fully generic, displays all fields
static void displayXmpPacketContents(const tinygltf::Value& packet)
{
  if(!packet.IsObject())
    return;

  // Display all fields except JSON-LD metadata keys
  for(const auto& key : packet.Keys())
  {
    // Skip JSON-LD structural keys
    if(key[0] == '@')  // @context, @type, @id, etc.
      continue;
    displayXmpField(key, packet.Get(key));
  }
}

// Get XMP packet index from an object's extensions, returns -1 if not found
static int getXmpPacketIndex(const tinygltf::ExtensionMap& extensions)
{
  const tinygltf::Value* ext = getXmpExtension(extensions);
  if(!ext || !ext->Has("packet"))
    return -1;

  return ext->Get("packet").GetNumberAsInt();
}

// Get XMP packet by index from model's root extensions, returns nullptr if not found
static const tinygltf::Value* getXmpPacket(const tinygltf::Model* model, int packetIndex)
{
  if(packetIndex < 0)
    return nullptr;

  const tinygltf::Value* packets = getXmpPacketsArray(model);
  if(!packets || packetIndex >= static_cast<int>(packets->ArrayLen()))
    return nullptr;

  return &packets->Get(packetIndex);
}

//--------------------------------------------------------------------------------------------------
// Public API

bool renderInfoButton(const tinygltf::Model* model, const tinygltf::ExtensionMap& extensions, const char* popupId)
{
  int packetIndex = getXmpPacketIndex(extensions);
  if(packetIndex < 0)
    return false;

  const tinygltf::Value* packet = getXmpPacket(model, packetIndex);
  if(!packet)
    return false;

  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_HeaderHovered));
  if(ImGui::SmallButton("i"))
  {
    ImGui::OpenPopup(popupId);
  }
  ImGui::PopStyleColor(2);

  if(ImGui::IsItemHovered())
    ImGui::SetTooltip("XMP Metadata (Packet %d)", packetIndex);

  if(ImGui::BeginPopup(popupId))
  {
    ImGui::Text("XMP Metadata (Packet %d)", packetIndex);
    ImGui::Separator();
    displayXmpPacketContents(*packet);
    ImGui::EndPopup();
  }

  return true;
}

void renderMetadataPanel(const tinygltf::Model* model)
{
  const tinygltf::Value* packets = getXmpPacketsArray(model);
  if(!packets)
    return;

  const size_t packetCount = packets->ArrayLen();
  const bool   multiPacket = (packetCount > 1);

  ImGui::Separator();
  if(ImGui::CollapsingHeader("Asset Metadata (XMP)"))
  {
    for(size_t i = 0; i < packetCount; i++)
    {
      const auto& packet = packets->Get(static_cast<int>(i));
      if(!packet.IsObject())
        continue;

      // If multiple packets, show them in separate tree nodes
      if(multiPacket)
      {
        ImGui::PushID(static_cast<int>(i));
        if(!ImGui::TreeNodeEx("Packet", ImGuiTreeNodeFlags_DefaultOpen, "Packet %zu", i))
        {
          ImGui::PopID();
          continue;
        }
      }

      displayXmpPacketContents(packet);

      if(multiPacket)
      {
        ImGui::TreePop();
        ImGui::PopID();
      }
    }
  }
}

}  // namespace ui_xmp
