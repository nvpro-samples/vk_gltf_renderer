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

#include "ui_status_bar.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <imgui.h>
#include <imgui_internal.h>  // BeginViewportSideBar

namespace ui::statusbar {

namespace {

// One piece of text, with a colored dot before it when `dot` is set (it pulses while `pulse` is).
// Hovering it explains it.
struct Segment
{
  std::string text;
  std::string tooltip;
  ImU32       dot    = 0;
  bool        pulse  = false;
  bool        dimmed = false;  // The disabled text color: on, but not what is showing now
  bool        joined = false;  // Follows the previous segment with a gap, not a "|"
};

// The dot before a segment's text, and the gap after it, scale with the font.
float dotRadius()
{
  return ImGui::GetFontSize() * 0.3F;
}
// See dotRadius().
float dotGap()
{
  return ImGui::GetFontSize() * 0.4F;
}
float separatorGap()  // each side of the "|" between two segments
{
  return ImGui::GetFontSize() * 0.6F;
}

// What a segment takes: its text, and its dot when it has one.
float segmentWidth(const Segment& segment)
{
  return ImGui::CalcTextSize(segment.text.c_str()).x + (segment.dot ? dotRadius() * 2.0F + dotGap() : 0.0F);
}

float separatorWidth()  // a "|" and the gap each side of it
{
  return ImGui::CalcTextSize("|").x + separatorGap() * 2.0F;
}

// The segments side by side, with a faint "|" between them
float groupWidth(const std::vector<Segment>& group)
{
  float width = 0.0F;
  for(size_t i = 0; i < group.size(); ++i)
    width += segmentWidth(group[i]) + (i == 0 ? 0.0F : group[i].joined ? separatorGap() : separatorWidth());
  return width;
}

// One segment: the dot, then the text (dimmed if asked), and its tooltip on hover.
void drawSegment(const Segment& segment)
{
  ImU32 dot = segment.dot;
  if(segment.pulse)  // alpha follows a sine of the frame clock: no ticker to drive
  {
    const float alpha = 0.45F + 0.55F * (0.5F + 0.5F * std::sin(static_cast<float>(ImGui::GetTime()) * 5.0F));
    dot               = (dot & 0x00FFFFFFu) | (static_cast<ImU32>(255.0F * alpha) << 24);
  }

  ImGui::BeginGroup();
  ImGui::AlignTextToFramePadding();
  if(segment.dot)
  {
    const ImVec2 pos   = ImGui::GetCursorScreenPos();
    const float  lineH = ImGui::GetTextLineHeight();
    const float  r     = dotRadius();
    ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(pos.x + r, pos.y + ImGui::GetStyle().FramePadding.y + lineH * 0.5F), r, dot);
    ImGui::Dummy(ImVec2(r * 2.0F, lineH));
    ImGui::SameLine(0.0F, dotGap());
  }
  if(segment.dimmed)
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
  ImGui::TextUnformatted(segment.text.c_str());
  if(segment.dimmed)
    ImGui::PopStyleColor();
  ImGui::EndGroup();

  if(!segment.tooltip.empty() && ImGui::IsItemHovered())
    ImGui::SetTooltip("%s", segment.tooltip.c_str());
}

// Draw the segments side by side; `afterSomething` puts a "|" before the first, since the line already holds
// an item.
void drawGroup(const std::vector<Segment>& group, bool afterSomething)
{
  for(size_t i = 0; i < group.size(); ++i)
  {
    if(group[i].joined && i > 0)
    {
      ImGui::SameLine(0.0F, separatorGap());
    }
    else if(i > 0 || afterSomething)
    {
      ImGui::SameLine(0.0F, separatorGap());
      ImGui::TextDisabled("|");
      ImGui::SameLine(0.0F, separatorGap());
    }
    drawSegment(group[i]);
  }
}

// Width of the accumulation bar.
float progressWidth()
{
  return ImGui::GetFontSize() * 13.0F;
}

// The accumulation as a bar with its count over it: the same look as ImGui::ProgressBar, drawn by hand to
// sit centered in the bar's row.
void drawProgress(int samples, int maxSamples)
{
  const std::string text  = fmt::format("{} / {} spp", samples, maxSamples);
  const float       width = progressWidth();
  const float       lineH = ImGui::GetTextLineHeight();

  ImGui::BeginGroup();
  ImGui::AlignTextToFramePadding();
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const ImVec2 min(pos.x, pos.y + ImGui::GetStyle().FramePadding.y);
  const ImVec2 max(min.x + width, min.y + lineH);
  const float  fraction = std::min(static_cast<float>(samples) / static_cast<float>(std::max(maxSamples, 1)), 1.0F);
  ImDrawList*  dl       = ImGui::GetWindowDrawList();
  dl->AddRectFilled(min, max, ImGui::GetColorU32(ImGuiCol_FrameBg), 2.0F);
  if(fraction > 0.0F)
    dl->AddRectFilled(min, ImVec2(min.x + width * fraction, max.y), ImGui::GetColorU32(ImGuiCol_PlotHistogram), 2.0F);
  const ImVec2 textSize = ImGui::CalcTextSize(text.c_str());
  dl->AddText(ImVec2(min.x + (width - textSize.x) * 0.5F, min.y), ImGui::GetColorU32(ImGuiCol_Text), text.c_str());
  ImGui::Dummy(ImVec2(width, lineH));
  ImGui::EndGroup();

  if(ImGui::IsItemHovered())
    ImGui::SetTooltip("Samples per pixel in the image, out of Max Samples");
}

// The shader state as a segment: the dot's color says which shader renders, and pulses while the optimized one builds.
Segment shaderSegment(PathTracer::ShaderStatus status)
{
  switch(status)
  {
    case PathTracer::ShaderStatus::eOptimized:
      return {"Optimized shader", "The path tracer shader is compiled with only the features this scene uses.",
              IM_COL32(110, 200, 90, 255)};
    case PathTracer::ShaderStatus::eBuilding:
      return {"All-features shader  |  building optimized...",
              "The all-features shader renders the same image, a little slower, while the optimized one\n"
              "builds in the background. It swaps in without restarting the accumulation.",
              IM_COL32(240, 180, 50, 255), true};
    case PathTracer::ShaderStatus::eAllFeatures:
      return {"All-features shader", "Optimize Shader is off: the all-features shader renders.", IM_COL32(150, 150, 150, 255)};
    case PathTracer::ShaderStatus::eFailed:
      return {"All-features shader  |  optimized build failed",
              "The optimized shader did not compile (see the log), so the all-features one keeps rendering.\n"
              "Reload Shaders retries.",
              IM_COL32(230, 80, 70, 255)};
  }
  return {};
}

// Left: how far the image is and what a frame costs. Right: what renders it.
void drawPathTracerStatus(const PathTracer::Status& pt)
{
  std::vector<Segment> speed;
  // "-- ms" while the profiler has no time (nothing was traced lately), so the figure keeps its place
  speed.push_back({pt.pathTraceMs > 0.0F ? fmt::format("{:.1f} ms", pt.pathTraceMs) : std::string("-- ms"),
                   "GPU time of the path trace (average)"});
  speed.push_back({fmt::format("{} spp/frame", pt.samplesPerFrame), "Samples per pixel the last frame traced"});
  speed.push_back({fmt::format("{:.0f} MSPP/s", pt.msppPerSecond), "Mega-sample-pixels per second of wall time (rolling average)"});

  std::vector<Segment> rendering;
  // Each denoiser that is on, the one whose image is showing in the normal text color, the other dimmed
  for(size_t i = 0; i < pt.denoisers.size(); ++i)
  {
    const PathTracer::Denoiser& denoiser = pt.denoisers[i];
    rendering.push_back({denoiser.name,
                         denoiser.active ? "The denoiser whose image is showing" :
                                           "On, but not the image showing now. With DLSS-RR and OptiX or OIDN together, DLSS-RR "
                                           "denoises the first frame after a change and the other the accumulation after it.",
                         0, false, !denoiser.active, i > 0});
  }
  if(pt.renderSize.width != pt.outputSize.width || pt.renderSize.height != pt.outputSize.height)
    rendering.push_back({fmt::format("{} x {} -> {} x {}", pt.renderSize.width, pt.renderSize.height,
                                     pt.outputSize.width, pt.outputSize.height),
                         "The path tracer's resolution, then the output's: DLSS upscales between them"});
  else
    rendering.push_back({fmt::format("{} x {}", pt.outputSize.width, pt.outputSize.height), "Resolution of the image"});
  rendering.push_back(shaderSegment(pt.shader));

  // The speed figures give way when the window is too narrow for both groups
  const float padding   = ImGui::GetStyle().WindowPadding.x;
  const float room      = ImGui::GetWindowWidth() - padding * 2.0F;
  const float leftWidth = progressWidth() + separatorWidth() + groupWidth(speed);
  const bool  showSpeed = leftWidth + groupWidth(rendering) + ImGui::GetFontSize() * 2.0F <= room;

  drawProgress(pt.samples, pt.maxSamples);
  if(showSpeed)
    drawGroup(speed, true);

  ImGui::SameLine(padding + room - groupWidth(rendering) - ImGui::GetFontSize() * 0.5F);
  drawGroup(rendering, false);
}

}  // namespace

// See ui_status_bar.hpp.
void render(const Content& content)
{
  // A menu-bar window, as ImGui's own status bar sample: it centers the text vertically in the row
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_MenuBar;
  if(ImGui::BeginViewportSideBar("##StatusBar", ImGui::GetMainViewport(), ImGuiDir_Down, ImGui::GetFrameHeight(), flags))
  {
    // A thin line along the top edge separates the bar from the panels above it. The text color at low
    // alpha, since the theme's separator and border colors sit too close to the background to show.
    // The window is only a menu bar, so its content clip rect is empty: clip to the whole window.
    const ImVec2 pos  = ImGui::GetWindowPos();
    const ImVec2 size = ImGui::GetWindowSize();
    ImDrawList*  dl   = ImGui::GetWindowDrawList();
    dl->PushClipRect(pos, ImVec2(pos.x + size.x, pos.y + size.y), false);
    dl->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + 1.0F), ImGui::GetColorU32(ImGuiCol_Text, 0.25F));
    dl->PopClipRect();

    if(ImGui::BeginMenuBar())
    {
      if(content.pathTracer)
        drawPathTracerStatus(*content.pathTracer);
      ImGui::EndMenuBar();
    }
  }
  ImGui::End();  // whatever BeginViewportSideBar returned
}

}  // namespace ui::statusbar
