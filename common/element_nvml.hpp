/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <imgui.h>
#include <imgui_internal.h>
#include "nvvkhl/application.hpp"
#include "nvml_monitor.hpp"
#include "imgui/imgui_helper.h"
#include <numeric>

namespace nvvkhl {


//extern SampleAppLog g_logger;
struct ElementNvml : public nvvkhl::IAppElement
{
  explicit ElementNvml(bool show = false)
      : m_showWindow(show)
  {
#if defined(NVP_SUPPORTS_NVML)
    m_nvmlMonitor = std::make_unique<NvmlMonitor>();
#endif
    addSettingsHandler();
  }

  virtual ~ElementNvml() = default;

  void onUIRender() override
  {
#if defined(NVP_SUPPORTS_NVML)
    m_nvmlMonitor->refresh();
#endif

    ImGui::SetNextWindowCollapsed(false, ImGuiCond_Appearing);
    ImGui::SetNextWindowSize({400, 200}, ImGuiCond_Appearing);
    ImGui::SetNextWindowBgAlpha(0.7F);
    if(m_showWindow && ImGui::Begin("NVML Monitor", &m_showWindow))
    {
      guiGpuMeasures();
      ImGui::End();
    }
  }

  void onUIMenu() override
  {
    if(ImGui::BeginMenu("Help"))
    {
      ImGui::MenuItem("NVML Monitor", nullptr, &m_showWindow);
      ImGui::EndMenu();
    }
  }  // This is the menubar to create


  //--------------------------------------------------------------------------------------------------
  //
  //
  bool guiGpuMeasures()
  {
    static const std::vector<const char*> t{"KB", "MB", "GB", "TB"};

#if defined(NVP_SUPPORTS_NVML)
    if(m_nvmlMonitor->isValid() == false)
    {
      ImGui::Text("NVML wasn't loaded");
      return false;
    }

    uint32_t offset = m_nvmlMonitor->getOffset();

    for(uint32_t g = 0; g < m_nvmlMonitor->nbGpu(); g++)  // Number of gpu
    {
      const auto& i = m_nvmlMonitor->getInfo(g);
      const auto& m = m_nvmlMonitor->getMeasures(g);
      char        progtext[64];
      float       divider = 1.0F;
      int         level   = 0;
      while(i.max_mem / divider > 1000)
      {
        divider *= 1000;
        level++;
      }
      sprintf(progtext, "%3.2f/%3.2f %s", m.memory[offset] / divider, i.max_mem / divider, t[level]);

      // Load
      ImGui::Text("GPU: %s", i.name.c_str());
      ImGuiH::PropertyEditor::begin();
      ImGuiH::PropertyEditor::entry("Load", [&] {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, (ImVec4)ImColor::HSV(0.3F, 0.5F, 0.5F));
        ImGui::ProgressBar(m.load[offset] / 100.F);
        ImGui::PopStyleColor();
        return false;
      });
      // Memory
      ImGuiH::PropertyEditor::entry("Memory", [&] {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, (ImVec4)ImColor::HSV(0.6F, 0.5F, 0.5F));
        ImGui::ProgressBar(m.memory[offset] / float(i.max_mem), ImVec2(-1.f, 0.f), progtext);
        ImGui::PopStyleColor();
        return false;
      });

      //ImGui::Unindent();
      ImGuiH::PropertyEditor::end();
    }

    // CPU - refreshing only every second and average the last 5 values
    static float  average      = 0;
    static double refresh_time = ImGui::GetTime();
    if(refresh_time < ImGui::GetTime() - 1)  // Create data at fixed 60 Hz rate for the demo
    {
      average           = 0;
      int values_offset = 0;
      for(int i = 0; i < 5; i++)
      {
        values_offset = (offset - i) % m_nvmlMonitor->getSysInfo().cpu.size();
        average += m_nvmlMonitor->getSysInfo().cpu[values_offset];
      }
      average /= 5.0F;
      refresh_time = ImGui::GetTime();
    }

    ImGuiH::PropertyEditor::begin();
    ImGuiH::PropertyEditor::entry("CPU", [&] {
      ImGui::ProgressBar(average / 100.F);
      return false;
    });
    ImGuiH::PropertyEditor::end();

    // Display Graphs
    for(uint32_t g = 0; g < m_nvmlMonitor->nbGpu(); g++)  // Number of gpu
    {
      const auto& i = m_nvmlMonitor->getInfo(g);
      const auto& m = m_nvmlMonitor->getMeasures(g);

      if(ImGui::TreeNode("Graph", "Graph: %s", i.name.c_str()))
      {
        ImGui::ImPlotMulti datas[2];
        datas[0].plot_type     = static_cast<ImGuiPlotType>(ImGuiPlotType_Area);
        datas[0].name          = "Load";
        datas[0].color         = ImColor(0.07f, 0.9f, 0.06f, 1.0f);
        datas[0].thickness     = 1.5;
        datas[0].data          = m.load.data();
        datas[0].values_count  = (int)m.load.size();
        datas[0].values_offset = offset + 1;
        datas[0].scale_min     = 0;
        datas[0].scale_max     = 100;

        datas[1].plot_type     = ImGuiPlotType_Histogram;
        datas[1].name          = "Mem";
        datas[1].color         = ImColor(0.06f, 0.6f, 0.97f, 0.8f);
        datas[1].thickness     = 2.0;
        datas[1].data          = m.memory.data();
        datas[1].values_count  = (int)m.memory.size();
        datas[1].values_offset = offset + 1;
        datas[1].scale_min     = 0;
        datas[1].scale_max     = float(i.max_mem);


        std::string overlay = "Load: " + std::to_string((int)m.load[offset]) + " %";
        ImGui::PlotMultiEx("##NoName", 2, datas, overlay.c_str(), ImVec2(ImGui::GetContentRegionAvail().x, 100));
        ImGui::TreePop();
      }
    }
#else
    ImGui::Text("NVML wasn't loaded");
#endif
    return false;
  }


  // This goes in the .ini file and remember the state of the window [open/close]
  void addSettingsHandler()
  {
    // Persisting the window
    ImGuiSettingsHandler ini_handler{};
    ini_handler.TypeName   = "ElementNvml";
    ini_handler.TypeHash   = ImHashStr("ElementNvml");
    ini_handler.ClearAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler*) {};
    ini_handler.ApplyAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler*) {};
    ini_handler.ReadOpenFn = [](ImGuiContext*, ImGuiSettingsHandler*, const char* name) -> void* { return (void*)1; };
    ini_handler.ReadLineFn = [](ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line) {
      ElementNvml* s = (ElementNvml*)handler->UserData;
      int          x;
      if(sscanf(line, "ShowLoader=%d", &x) == 1)
      {
        s->m_showWindow = (x == 1);
      }
    };
    ini_handler.WriteAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
      ElementNvml* s = (ElementNvml*)handler->UserData;
      buf->appendf("[%s][State]\n", handler->TypeName);
      buf->appendf("ShowLoader=%d\n", s->m_showWindow ? 1 : 0);
      buf->appendf("\n");
    };
    ini_handler.UserData = this;
    ImGui::AddSettingsHandler(&ini_handler);
  }

private:
  bool                         m_showWindow{false};
#if defined(NVP_SUPPORTS_NVML)
  std::unique_ptr<NvmlMonitor> m_nvmlMonitor;
#endif
};


}  // namespace nvvkhl
