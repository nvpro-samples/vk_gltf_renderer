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

// ImGui rendering for the Agentic feature, paired with agentic::Controller in
// agentic.cpp. The window (F7, reached from the Windows menu) is the only UI;
// there is no top-level menu for this optional feature.
//
// Layout, top to bottom:
//   - Services: two status lights, "Bridge" (the adapter process) and "ComfyUI".
//     ComfyUI's state is only known through the bridge heartbeat, so it reads
//     "unknown" until the bridge is running. When the bridge is down, a
//     copy-pasteable start command is shown.
//   - Beautify last render (primary), then HDRI from prompt: compact auto-growing
//     prompt, an explicit action button gated on the services being ready, and
//     always-visible progress so a running job is never ambiguous.
//   - Advanced (collapsed): bridge folder, converter Python, steps, seed, 4x, polling.

#include "agentic.hpp"
#include "resources.hpp"

#include <cfloat>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include <imgui.h>
#include <nvgui/fonts.hpp>    // for ICON_MS_* glyphs (transitively pulls in IconsMaterialSymbols.h)
#include <nvgui/tooltip.hpp>  // nvgui::tooltip

namespace agentic {
namespace {

// Docs are not packaged with binary Releases, so the window links to GitHub for
// the setup/install guide — the only discoverable path for someone who opened the
// app without reading anything first.
constexpr const char* kSetupGuideUrl = "https://github.com/nvpro-samples/vk_gltf_renderer/blob/main/docs/comfyui-agentic-setup.md";

constexpr ImU32 kLightGreen = IM_COL32(90, 200, 90, 255);
constexpr ImU32 kLightAmber = IM_COL32(230, 180, 60, 255);
constexpr ImU32 kLightRed   = IM_COL32(220, 80, 80, 255);
constexpr ImU32 kLightGray  = IM_COL32(160, 160, 160, 255);

const ImVec4 kTextGreen{0.42f, 0.78f, 0.42f, 1.0f};
const ImVec4 kTextAmber{0.90f, 0.70f, 0.24f, 1.0f};
const ImVec4 kTextRed{0.86f, 0.31f, 0.31f, 1.0f};
const ImVec4 kTextGray{0.60f, 0.60f, 0.60f, 1.0f};

// A small filled status light drawn inline; leaves the cursor on the same line so
// a label follows immediately after it.
void statusLight(ImU32 color)
{
  const ImVec2 p  = ImGui::GetCursorScreenPos();
  const float  th = ImGui::GetTextLineHeight();
  const float  r  = th * 0.34f;
  ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.x + r + 2.0f, p.y + th * 0.5f), r, color);
  ImGui::Dummy(ImVec2(r * 2.0f + 8.0f, th));
  ImGui::SameLine();
}

// A hoverable "(?)" that shows an explanatory tooltip for the preceding control.
void helpMarker(const char* desc)
{
  ImGui::SameLine();
  ImGui::TextDisabled("(?)");
  if(ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    ImGui::SetTooltip("%s", desc);
}

// Compact prompt field (fixed, scrollable). Prompts can be long, so a small box
// keeps the window tidy; the card's "Edit…" button opens promptPopup() for a
// full-size editor.
void compactPrompt(const char* id, char* buffer, size_t bufferSize)
{
  const float h = ImGui::GetTextLineHeight() * 2.5f + ImGui::GetStyle().FramePadding.y * 2.0f;
  ImGui::InputTextMultiline(id, buffer, bufferSize, ImVec2(-FLT_MIN, h), ImGuiInputTextFlags_WordWrap);
}

// Large modal editor for a prompt. Edits the same buffer live, so "Close" just
// dismisses the popup. Opened via ImGui::OpenPopup(id) from the card header.
void promptPopup(const char* id, char* buffer, size_t bufferSize)
{
  const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(560.0f, 360.0f), ImGuiCond_Appearing);
  if(ImGui::BeginPopupModal(id, nullptr))
  {
    ImGui::InputTextMultiline("##editor", buffer, bufferSize, ImVec2(-FLT_MIN, -ImGui::GetFrameHeightWithSpacing()),
                              ImGuiInputTextFlags_WordWrap);
    if(ImGui::Button("Close"))
      ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
}

// Presets dropdown. Selecting an entry replaces the prompt buffer with the preset
// text (still freely editable afterwards). Renders nothing when no presets loaded.
void renderPresetCombo(const char* id, const std::vector<PromptPreset>& presets, char* buffer, size_t bufferSize)
{
  if(presets.empty())
    return;
  ImGui::SetNextItemWidth(150.0f);
  if(ImGui::BeginCombo(id, "Presets"))
  {
    for(const PromptPreset& preset : presets)
    {
      if(ImGui::Selectable(preset.name.c_str()))
        std::snprintf(buffer, bufferSize, "%s", preset.prompt.c_str());
      if(ImGui::IsItemHovered())
        ImGui::SetTooltip("%.200s%s", preset.prompt.c_str(), preset.prompt.size() > 200 ? " ..." : "");
    }
    ImGui::EndCombo();
  }
}

// Always-visible job progress: a real bar when ComfyUI reports value/max, else a
// live spinner + phase text, so an in-flight job is never a blank silence.
void renderJobProgress(const std::optional<JobProgress>& progress)
{
  if(progress && progress->max > 0)
  {
    float frac = float(progress->value) / float(progress->max);
    frac       = frac < 0.0f ? 0.0f : (frac > 1.0f ? 1.0f : frac);
    char overlay[64];
    if(progress->message.empty())
      std::snprintf(overlay, sizeof(overlay), "step %d / %d", progress->value, progress->max);
    else
      std::snprintf(overlay, sizeof(overlay), "%s", progress->message.c_str());
    ImGui::ProgressBar(frac, ImVec2(-FLT_MIN, 0), overlay);
    return;
  }

  const char  spinner[] = {'|', '/', '-', '\\'};
  const char  glyph     = spinner[int(ImGui::GetTime() * 8.0) & 3];
  const char* label     = "Queued\xE2\x80\xA6";  // "Queued…"
  if(progress)
  {
    if(!progress->message.empty())
      label = progress->message.c_str();
    else if(!progress->phase.empty())
      label = progress->phase.c_str();
    else
      label = "Working\xE2\x80\xA6";
  }
  ImGui::Text("%c  %s", glyph, label);
}

// Build the copy-pasteable adapter command from the controller's current paths.
std::string buildAdapterCommand(Controller& ctl)
{
  const std::filesystem::path script    = ctl.adapterScriptPath();
  const std::filesystem::path workflows = script.parent_path() / "workflows";
  const std::filesystem::path root      = ctl.bridgeRootPath();
  std::string                 cmd = "python \"" + script.string() + "\"" + " --bridge-root \"" + root.string() + "\""
                    + " --workflow-dir \"" + workflows.string() + "\"" + " --comfy-url http://127.0.0.1:8188";
  if(ctl.converterPython[0] != '\0')
    cmd += " --converter-python \"" + std::string(ctl.converterPython) + "\"";
  return cmd;
}

// One service row: a status light, a name aligned into a column, and a colored
// state label. `stateXOffset` aligns the state text across rows.
void serviceRow(ImU32 lightColor, const char* name, const ImVec4& stateColor, const char* state, float stateXOffset, const char* help)
{
  statusLight(lightColor);
  const float xAfterLight = ImGui::GetCursorPosX();
  ImGui::TextUnformatted(name);
  ImGui::SameLine(xAfterLight + stateXOffset);
  ImGui::TextColored(stateColor, "%s", state);
  helpMarker(help);
}

// Segmented "Render | Beautified" view toggle. Greyed out (tri-state) when there
// is no beautified result to switch to.
void renderViewToggle(Controller& ctl, Resources& resources)
{
  const bool hasImage = ctl.hasBeautifiedImage();
  const bool showing  = resources.settings.displayBuffer == DisplayBuffer::eAgenticBeautified;

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("View");
  ImGui::SameLine();

  ImGui::BeginDisabled(!hasImage);
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1.0f, ImGui::GetStyle().ItemSpacing.y));

  const ImU32 activeBg   = IM_COL32(60, 120, 200, 255);
  const ImU32 activeText = IM_COL32(255, 255, 255, 255);

  const bool renderActive = hasImage && !showing;
  if(renderActive)
  {
    ImGui::PushStyleColor(ImGuiCol_Button, activeBg);
    ImGui::PushStyleColor(ImGuiCol_Text, activeText);
  }
  if(ImGui::Button("Render"))
    resources.settings.displayBuffer = DisplayBuffer::eRendered;
  if(renderActive)
    ImGui::PopStyleColor(2);

  ImGui::SameLine();
  const bool beautifiedActive = hasImage && showing;
  if(beautifiedActive)
  {
    ImGui::PushStyleColor(ImGuiCol_Button, activeBg);
    ImGui::PushStyleColor(ImGuiCol_Text, activeText);
  }
  if(ImGui::Button("Beautified"))
    resources.settings.displayBuffer = DisplayBuffer::eAgenticBeautified;
  if(beautifiedActive)
    ImGui::PopStyleColor(2);

  ImGui::PopStyleVar();
  ImGui::EndDisabled();

  if(hasImage)
  {
    ImGui::SameLine();
    if(ImGui::SmallButton(ICON_MS_CLOSE " Clear"))
      ctl.destroyBeautifiedImage(true);
  }
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// Dockable "Agentic" window (F7).
//--------------------------------------------------------------------------------------------------

void renderAgenticWindow(Controller& ctl, Resources& resources)
{
  if(!resources.settings.showAgenticWindow)
    return;

  if(!ImGui::Begin("Agentic", &resources.settings.showAgenticWindow))
  {
    ImGui::End();
    return;
  }
  nvgui::tooltip("Press F7 to toggle this window");

  // Refresh up-front so the lights reflect the current state even when the window
  // was reopened between auto-poll ticks (throttled internally, so cheap).
  ctl.refreshAdapterStatus();
  const AdapterStatusInfo& a            = ctl.adapterStatus();
  const AdapterStatus      st           = a.status;
  const bool               bridgeActive = st == AdapterStatus::eActive;
  const bool               bridgeUp = st == AdapterStatus::eActive || st == AdapterStatus::eStale;  // light color only
  const bool               comfyOk  = a.heartbeat && a.heartbeat->comfyReachable;
  const bool               ready    = bridgeActive && comfyOk;  // require a fresh heartbeat to submit

  ImGui::TextWrapped(
      "Generate an HDRI environment from a prompt, or make a photoreal version of your render. "
      "Optional — it drives a local ComfyUI through a small bridge process.");
  ImGui::TextUnformatted("New here?");
  ImGui::SameLine();
  ImGui::TextLinkOpenURL("Setup & ComfyUI install guide", kSetupGuideUrl);
  ImGui::Spacing();

  // ----- Services: Bridge + ComfyUI -----------------------------------------
  const float stateX = ImGui::CalcTextSize("ComfyUI").x + 12.0f;

  serviceRow(bridgeActive ? kLightGreen : (bridgeUp ? kLightAmber : kLightRed),         //
             "Bridge",                                                                  //
             bridgeActive ? kTextGreen : (bridgeUp ? kTextAmber : kTextRed),            //
             bridgeActive ? "running" : (bridgeUp ? "not responding" : "not running"),  //
             stateX,
             "The bridge is a small helper process that passes your requests to ComfyUI and brings the "
             "generated images back. Start it in a terminal with the command below (it is not launched by the app). "
             "'Not responding' means its heartbeat has gone stale — usually it was stopped or killed.");

  // ComfyUI's state is only trustworthy while the bridge is actively reporting;
  // a stale/dead bridge can't tell us anything, so show 'unknown'.
  serviceRow(bridgeActive ? (comfyOk ? kLightGreen : kLightRed) : kLightGray,  //
             "ComfyUI",                                                        //
             bridgeActive ? (comfyOk ? kTextGreen : kTextRed) : kTextGray,     //
             bridgeActive ? (comfyOk ? "reachable" : "not reachable — start ComfyUI") : "unknown — start the bridge to check",  //
             stateX,
             "ComfyUI is the image generator. Its status is reported by the bridge, so it stays 'unknown' "
             "until the bridge is running.");

  // Whenever the bridge is not confirmed running (down or stale), show the exact
  // start command inline so it's never missing.
  if(!bridgeActive)
  {
    ImGui::Spacing();
    ImGui::TextWrapped("The bridge isn't confirmed running — start (or restart) it in a terminal. The command is pre-filled with your paths:");
    const std::string        cmd = buildAdapterCommand(ctl);
    static std::vector<char> sScratch;
    sScratch.assign(cmd.begin(), cmd.end());
    sScratch.push_back('\0');
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextMultiline("##adapterCmd", sScratch.data(), sScratch.size(),
                              ImVec2(0.0f, ImGui::GetTextLineHeight() * 3.0f), ImGuiInputTextFlags_ReadOnly);
    if(ImGui::Button(ICON_MS_CONTENT_COPY " Copy start command"))
      ImGui::SetClipboardText(cmd.c_str());
    ImGui::SameLine();
    ImGui::TextLinkOpenURL("Setup guide (GitHub)", kSetupGuideUrl);
  }

  if(!a.error.empty())
    ImGui::TextColored(kTextRed, "Heartbeat read error: %s", a.error.c_str());

  // Inline reason shared by both action buttons when they are disabled.
  auto notReadyReason = [&](bool promptEmpty) -> const char* {
    if(!bridgeActive)
      return "Start the bridge first (see above).";
    if(!comfyOk)
      return "ComfyUI isn't reachable — start ComfyUI.";
    if(promptEmpty)
      return "Enter a prompt to enable.";
    return nullptr;
  };

  // ----- Beautify (primary) --------------------------------------------------
  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  ImGui::TextUnformatted(ICON_MS_AUTO_FIX_HIGH " Beautify");
  helpMarker("Sends your last render to ComfyUI and displays a photoreal version back in the viewport.");
  ImGui::SameLine();
  if(ImGui::SmallButton("Edit\xE2\x80\xA6##beautifyPromptEdit"))
    ImGui::OpenPopup("Beautify prompt");
  ImGui::SameLine();
  renderPresetCombo("##beautifyPresets", ctl.beautifyPresets, ctl.beautifyPrompt, sizeof(ctl.beautifyPrompt));
  compactPrompt("##BeautifyPrompt", ctl.beautifyPrompt, sizeof(ctl.beautifyPrompt));
  promptPopup("Beautify prompt", ctl.beautifyPrompt, sizeof(ctl.beautifyPrompt));

  const bool beautifyBusy  = ctl.beautifyJob().active;
  const bool beautifyEmpty = ctl.beautifyPrompt[0] == '\0';
  ImGui::BeginDisabled(beautifyBusy || !ready || beautifyEmpty);
  if(ImGui::Button(ICON_MS_AUTO_FIX_HIGH " Beautify last render"))
    ctl.queueBeautifyJob();
  ImGui::EndDisabled();

  if(beautifyBusy)
  {
    renderJobProgress(ctl.beautifyProgress());
  }
  else if(const char* reason = notReadyReason(beautifyEmpty))
  {
    ImGui::TextDisabled("%s", reason);
  }

  renderViewToggle(ctl, resources);
  if(ctl.hasBeautifiedImage())
  {
    const VkExtent2D e  = ctl.beautifiedExtent();
    const VkExtent2D vp = resources.gBuffers.getSize();
    if(e.width != vp.width || e.height != vp.height)
      ImGui::TextDisabled("Result %ux%u · viewport %ux%u (display stretched)", e.width, e.height, vp.width, vp.height);
    else
      ImGui::TextDisabled("Result %ux%u · shown 1:1", e.width, e.height);
  }

  // ----- HDRI ----------------------------------------------------------------
  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  ImGui::TextUnformatted(ICON_MS_PUBLIC " HDRI from prompt");
  helpMarker("Generates an equirectangular HDR environment map and applies it as the scene lighting.");
  ImGui::SameLine();
  if(ImGui::SmallButton("Edit\xE2\x80\xA6##hdriPromptEdit"))
    ImGui::OpenPopup("HDRI prompt");
  ImGui::SameLine();
  renderPresetCombo("##hdriPresets", ctl.hdriPresets, ctl.hdriPrompt, sizeof(ctl.hdriPrompt));
  compactPrompt("##HdriPrompt", ctl.hdriPrompt, sizeof(ctl.hdriPrompt));
  promptPopup("HDRI prompt", ctl.hdriPrompt, sizeof(ctl.hdriPrompt));

  const bool hdriBusy  = ctl.hdriJob().active;
  const bool hdriEmpty = ctl.hdriPrompt[0] == '\0';
  ImGui::BeginDisabled(hdriBusy || !ready || hdriEmpty);
  if(ImGui::Button(ICON_MS_PUBLIC " Generate HDRI from prompt"))
    ctl.queueHdriJob();
  ImGui::EndDisabled();

  if(hdriBusy)
  {
    renderJobProgress(ctl.hdriProgress());
  }
  else if(const char* reason = notReadyReason(hdriEmpty))
  {
    ImGui::TextDisabled("%s", reason);
  }

  if(!ctl.lastHdrPath().empty())
    ImGui::TextDisabled("Applied: %s", ctl.lastHdrPath().filename().string().c_str());

  // ----- Advanced (collapsed) ------------------------------------------------
  ImGui::Spacing();
  if(ImGui::CollapsingHeader("Advanced"))
  {
    if(ImGui::Button(ICON_MS_CONTENT_COPY " Copy adapter command"))
      ImGui::SetClipboardText(buildAdapterCommand(ctl).c_str());
    nvgui::tooltip("Copy the terminal command that starts the bridge adapter.");
    ImGui::Spacing();

    ImGui::InputText("Bridge folder", ctl.bridgeRoot, sizeof(ctl.bridgeRoot));
    nvgui::tooltip("Folder shared with the bridge process for jobs, results, and assets. Must match the adapter's --bridge-root.");

    ImGui::TextUnformatted("Converter Python");
    nvgui::tooltip(
        "Only used when an HDRI comes back as PNG and must be converted to .hdr. Needs numpy + Pillow; "
        "auto-detected from a ComfyUI portable install at startup.");
    const float autoBtnW = ImGui::CalcTextSize("Auto-detect").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - autoBtnW - ImGui::GetStyle().ItemSpacing.x);
    ImGui::InputTextWithHint("##converterPython", "(optional) python_embeded\\python.exe", ctl.converterPython,
                             sizeof(ctl.converterPython));
    ImGui::SameLine();
    if(ImGui::Button("Auto-detect"))
    {
      const std::filesystem::path detected = ctl.detectComfyEmbeddedPython();
      if(!detected.empty())
        std::snprintf(ctl.converterPython, sizeof(ctl.converterPython), "%s", detected.string().c_str());
    }

    ImGui::SliderInt("Steps", &ctl.generationSteps, 1, 50);
    nvgui::tooltip("Sampling steps for both HDRI and Beautify. Higher = slower, usually cleaner.");

    ImGui::InputScalar("Seed", ImGuiDataType_U64, &ctl.generationSeed, nullptr, nullptr, "%llu");
    ImGui::SameLine();
    if(ImGui::Button(ICON_MS_CASINO))
      ctl.randomizeGenerationSeed();
    nvgui::tooltip("Randomize the seed. Same seed + steps + prompt reproduces a result.");

    ImGui::Checkbox("High-res 4× upscale (HDRI)", &ctl.hdriUpscale4x);
    nvgui::tooltip(
        "Both output 4096×2048. On: model-based PixelDiT upscale (needs extra ComfyUI models). "
        "Off: bicubic upscale (no extra models).");

    ImGui::Checkbox("Auto poll", &ctl.autoPoll);
    ImGui::SameLine();
    ImGui::BeginDisabled(!ctl.enabled);
    if(ImGui::Button(ICON_MS_SYNC " Poll now"))
      ctl.pollNow();
    ImGui::EndDisabled();
  }

  // Last status / error detail, muted, at the bottom.
  if(!ctl.lastMessage.empty())
  {
    ImGui::Spacing();
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("%s", ctl.lastMessage.c_str());
    ImGui::PopStyleColor();
  }

  ImGui::End();
}

}  // namespace agentic
