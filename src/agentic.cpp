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

//
// agentic::Controller — optional filesystem bridge to external image generation.
//
// Queues HDRI and beautify jobs under <bridge>/requests/, polls responses/, applies
// HDR environments and uploads beautified viewport images for display. UI lives in
// ui_agentic.cpp; JSON contracts and manifest I/O live in agentic_bridge.cpp.
//

#include "agentic.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <span>
#include <sstream>

#include <fmt/format.h>
#include <stb/stb_image.h>
#include <tinygltf/json.hpp>

#include <backends/imgui_impl_vulkan.h>
#include <nvapp/application.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/frame_uploader.hpp>

#include "resources.hpp"

namespace agentic {

Controller::Controller()  = default;
Controller::~Controller() = default;

//--------------------------------------------------------------------------------------------------
// Lifecycle
//--------------------------------------------------------------------------------------------------

void Controller::init(Dependencies deps)
{
  m_deps = std::move(deps);
  randomizeGenerationSeed();

  // Default bridge root: <exe_dir>/agentic_bridge. Editable from the UI.
  const std::string defaultRoot = (nvutils::getExecutablePath().parent_path() / "agentic_bridge").string();
  std::snprintf(bridgeRoot, sizeof(bridgeRoot), "%s", defaultRoot.c_str());

  // Pre-fill the converter Python if we can find one of the well-known portable
  // ComfyUI installs at startup, so the "How to start the adapter" hint is
  // already a complete copy-paste even before the user clicks Initialize/Enable.
  if(const std::filesystem::path detected = detectComfyEmbeddedPython(); !detected.empty())
  {
    std::snprintf(converterPython, sizeof(converterPython), "%s", detected.string().c_str());
  }

  loadPromptPresets();
}

void Controller::deinit()
{
  destroyBeautifiedImage(false);
  m_deps = {};
}

//--------------------------------------------------------------------------------------------------
// Path helpers
//--------------------------------------------------------------------------------------------------

void Controller::setBridgeRoot(const std::filesystem::path& root)
{
  if(root.empty())
    return;
  std::snprintf(bridgeRoot, sizeof(bridgeRoot), "%s", root.string().c_str());
  // Presets can be overridden per project via <bridgeRoot>/prompts.json, so pick
  // up the new root's set (e.g. after the --agenticBridgeRoot override is applied).
  loadPromptPresets();
  refreshAdapterStatus(/*force=*/true);
}

std::filesystem::path Controller::bridgeRootPath() const
{
  if(bridgeRoot[0] == '\0')
    return nvutils::getExecutablePath().parent_path() / "agentic_bridge";
  return std::filesystem::path(bridgeRoot);
}

static std::filesystem::path resolveAgenticPath(const std::filesystem::path& base, const std::filesystem::path& path)
{
  if(path.empty() || path.is_absolute())
    return path;
  return base / path;
}

std::filesystem::path Controller::promptsFilePath() const
{
  // Per-project override in the bridge root wins; otherwise the copy shipped next
  // to the adapter script (<exe>/utils/comfy_bridge/prompts.json).
  std::error_code             ec;
  const std::filesystem::path overridePath = bridgeRootPath() / "prompts.json";
  if(std::filesystem::exists(overridePath, ec) && !ec)
    return overridePath;
  return adapterScriptPath().parent_path() / "prompts.json";
}

void Controller::loadPromptPresets()
{
  hdriPresets.clear();
  beautifyPresets.clear();

  const std::filesystem::path path = promptsFilePath();
  std::error_code             ec;
  if(!std::filesystem::exists(path, ec) || ec)
    return;

  std::ifstream  f(path, std::ios::binary);
  std::string    payload((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  nlohmann::json doc = nlohmann::json::parse(payload, nullptr, /*allow_exceptions=*/false);
  if(doc.is_discarded() || !doc.is_object())
  {
    LOGW("Agentic: prompts.json is not valid JSON: %s\n", path.string().c_str());
    return;
  }

  auto readList = [](const nlohmann::json& node, std::vector<PromptPreset>& out) {
    if(!node.is_array())
      return;
    for(const auto& entry : node)
    {
      if(!entry.is_object())
        continue;
      PromptPreset preset{entry.value("name", ""), entry.value("prompt", "")};
      if(!preset.name.empty() && !preset.prompt.empty())
        out.push_back(std::move(preset));
    }
  };
  // A present-but-mistyped field (e.g. "name": 123) makes json::value throw; catch
  // it so a hand-edited or malicious prompts.json can't abort startup.
  try
  {
    readList(doc.value("hdri", nlohmann::json::array()), hdriPresets);
    readList(doc.value("beautify", nlohmann::json::array()), beautifyPresets);
  }
  catch(const nlohmann::json::exception& e)
  {
    hdriPresets.clear();
    beautifyPresets.clear();
    LOGW("Agentic: prompts.json has a mistyped field, ignoring presets: %s\n", e.what());
    return;
  }
  LOGI("Agentic: loaded %zu HDRI + %zu beautify prompt presets from %s\n", hdriPresets.size(), beautifyPresets.size(),
       path.string().c_str());
}

std::filesystem::path Controller::adapterScriptPath() const
{
  // CMake POST_BUILD copies utils/comfy_bridge/comfy_bridge.py next to the
  // executable (and the same files are installed at INSTALL/utils/...). Fall
  // back to the same path even when missing - it's still a meaningful hint.
  const std::filesystem::path exeDir   = nvutils::getExecutablePath().parent_path();
  const std::filesystem::path adjacent = exeDir / "utils" / "comfy_bridge" / "comfy_bridge.py";
  return adjacent;
}

std::filesystem::path Controller::detectComfyEmbeddedPython() const
{
  // ComfyUI portable bundles its own python at <install>\python_embeded\python.exe;
  // that interpreter already has numpy + Pillow which our png_to_hdr.py needs.
  // Try environment overrides first, then a short list of common install paths.
  auto check = [](std::filesystem::path p) -> std::filesystem::path {
    std::error_code ec;
    return (std::filesystem::exists(p, ec) && !ec) ? p : std::filesystem::path{};
  };

  auto readEnv = [](const char* name) -> std::string {
#if defined(_WIN32)
    char*  buf = nullptr;
    size_t len = 0;
    if(_dupenv_s(&buf, &len, name) == 0 && buf)
    {
      std::string value(buf);
      std::free(buf);
      return value;
    }
    return {};
#else
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string{};
#endif
  };

  // Environment overrides only — no hardcoded absolute install paths. A user
  // points COMFYUI_PORTABLE / COMFYUI_HOME at their ComfyUI portable install, or
  // types the interpreter into the "Converter Python" field. This keeps startup
  // free of guessing across drive letters and avoids shipping one machine's layout.
  for(const char* name : {"COMFYUI_PORTABLE", "COMFYUI_HOME"})
  {
    if(std::string env = readEnv(name); !env.empty())
    {
      if(auto hit = check(std::filesystem::path(env) / "python_embeded" / "python.exe"); !hit.empty())
        return hit;
    }
  }
  return {};
}

//--------------------------------------------------------------------------------------------------
// Bridge enable / disable
//--------------------------------------------------------------------------------------------------

void Controller::initializeBridge()
{
  std::string                 error;
  const std::filesystem::path manifestPath = agentic::writeBridgeManifest(bridgeRootPath(), &error);
  if(manifestPath.empty())
  {
    initialized = false;
    enabled     = false;
    status      = "Bridge initialization failed";
    lastMessage = error;
    LOGE("Agentic bridge initialization failed: %s\n", error.c_str());
    return;
  }

  initialized = true;
  enabled     = true;
  status      = "Bridge enabled";
  lastMessage = "Manifest written to " + manifestPath.string();
  LOGI("Agentic bridge initialized at %s\n", bridgeRootPath().string().c_str());

  // Best-effort auto-detect of ComfyUI portable's embedded Python for the
  // optional PNG -> HDR converter. Never overwrite a value the user typed in.
  if(converterPython[0] == '\0')
  {
    const std::filesystem::path detected = detectComfyEmbeddedPython();
    if(!detected.empty())
    {
      std::snprintf(converterPython, sizeof(converterPython), "%s", detected.string().c_str());
      LOGI("Detected ComfyUI embedded Python: %s\n", detected.string().c_str());
    }
  }

  refreshAdapterStatus(/*force=*/true);
}

//--------------------------------------------------------------------------------------------------
// Job queuing
//--------------------------------------------------------------------------------------------------

void Controller::queueHdriJob()
{
  if(!enabled || !initialized)
    initializeBridge();
  if(!enabled)
    return;

  agentic::GenerationJob job;
  job.id                  = agentic::makeGenerationJobId("hdri");
  job.task                = agentic::GenerationTask::eHdriFromPrompt;
  job.prompt              = hdriPrompt;
  job.preferredOutputPath = agentic::bridgeLayout({}).assets / (job.id + ".hdr");
  // Both shipped HDRI workflows generate at 1024×512 and upscale 4× to 4096×2048;
  // they differ only in the upscaler (bicubic vs. model-based PixelDiT). width/height
  // are informational metadata in the request — the output size is baked into the
  // workflow template, not patched by the adapter.
  if(hdriUpscale4x)
    job.workflowFile = "hdri_from_prompt_4x.json";
  else
    job.workflowFile = agentic::defaultWorkflowFile(job.task);
  job.parameters = {{"width", "4096"}, {"height", "2048"}, {"format", "hdr"}};
  appendGenerationSamplerParameters(job.parameters);

  std::string                 error;
  const std::filesystem::path requestPath = agentic::writeGenerationJob(bridgeRootPath(), job, &error);
  if(requestPath.empty())
  {
    status      = "Failed to queue HDRI job";
    lastMessage = error;
    LOGE("Agentic HDRI job failed: %s\n", error.c_str());
    return;
  }

  m_hdriProgress.reset();
  m_hdriJob = {
      .active              = true,
      .task                = job.task,
      .id                  = job.id,
      .requestPath         = requestPath,
      .responsePath        = agentic::bridgeLayout(bridgeRootPath()).responses / (job.id + ".json"),
      .preferredOutputPath = resolveAgenticPath(bridgeRootPath(), job.preferredOutputPath),
  };
  status      = "HDRI generation queued";
  lastMessage = fmt::format("{} | steps {} | seed {} | waiting for {}", job.workflowFile, generationSteps,
                            generationSeed, m_hdriJob.responsePath.string());
  LOGI("Queued agentic HDRI job %s workflow %s (steps %d, seed %llu)\n", job.id.c_str(), job.workflowFile.c_str(),
       generationSteps, static_cast<unsigned long long>(generationSeed));
}

//--------------------------------------------------------------------------------------------------
// Shared job parameters
//--------------------------------------------------------------------------------------------------

void Controller::appendGenerationSamplerParameters(std::map<std::string, std::string>& parameters) const
{
  parameters["steps"]      = std::to_string(generationSteps);
  parameters["seed"]       = std::to_string(generationSeed);
  parameters["noise_seed"] = std::to_string(generationSeed);
}

void Controller::randomizeGenerationSeed()
{
  std::random_device                           rd;
  std::uniform_int_distribution<std::uint64_t> dist(0, std::numeric_limits<std::uint64_t>::max());
  generationSeed = dist(rd);
}

void Controller::queueBeautifyJob()
{
  if(!enabled || !initialized)
    initializeBridge();
  if(!enabled || !m_deps.app || !m_deps.resources)
    return;

  agentic::GenerationJob job;
  job.id                  = agentic::makeGenerationJobId("beautify");
  job.task                = agentic::GenerationTask::eImageToImage;
  job.workflowFile        = agentic::defaultWorkflowFile(job.task);
  job.prompt              = beautifyPrompt;
  job.inputPath           = agentic::bridgeLayout({}).assets / (job.id + "_input.jpg");
  job.preferredOutputPath = agentic::bridgeLayout({}).assets / (job.id + "_beautified.png");
  job.parameters          = {{"format", "png"}, {"match_input_size", "true"}};

  const std::filesystem::path inputPath = resolveAgenticPath(bridgeRootPath(), job.inputPath);
  std::filesystem::create_directories(inputPath.parent_path());

  saveBeautifyInputImage(inputPath);
  const VkExtent2D imageSize = m_deps.resources->gBuffers.getSize();

  job.parameters["width"]  = std::to_string(imageSize.width);
  job.parameters["height"] = std::to_string(imageSize.height);
  appendGenerationSamplerParameters(job.parameters);

  std::string                 error;
  const std::filesystem::path requestPath = agentic::writeGenerationJob(bridgeRootPath(), job, &error);
  if(requestPath.empty())
  {
    status      = "Failed to queue beautify job";
    lastMessage = error;
    LOGE("Agentic beautify job failed: %s\n", error.c_str());
    return;
  }

  m_beautifyProgress.reset();
  m_beautifyJob = {
      .active              = true,
      .task                = job.task,
      .id                  = job.id,
      .requestPath         = requestPath,
      .responsePath        = agentic::bridgeLayout(bridgeRootPath()).responses / (job.id + ".json"),
      .inputPath           = inputPath,
      .preferredOutputPath = resolveAgenticPath(bridgeRootPath(), job.preferredOutputPath),
  };
  status      = "Beautify job queued";
  lastMessage = fmt::format("Input {} | steps {} | seed {}", inputPath.string(), generationSteps, generationSeed);
  LOGI("Queued agentic beautify job %s\n", job.id.c_str());
}

//--------------------------------------------------------------------------------------------------
// Beautify input capture
//--------------------------------------------------------------------------------------------------

void Controller::saveBeautifyInputImage(const std::filesystem::path& path)
{
  if(!m_deps.app || !m_deps.resources)
    return;

  // When the viewport shows the beautified overlay, tonemap() copies that into eImgTonemapped.
  // Re-run tonemap from the live render so ComfyUI conditions on the scene, not the prior beautify.
  const bool viewingBeautified =
      m_deps.resources->settings.displayBuffer == DisplayBuffer::eAgenticBeautified && hasBeautifiedImage();
  if(viewingBeautified && m_deps.runTonemapPass && m_deps.device != VK_NULL_HANDLE && m_deps.transientCmdPool != VK_NULL_HANDLE)
  {
    VkCommandBuffer cmd{};
    nvvk::beginSingleTimeCommands(cmd, m_deps.device, m_deps.transientCmdPool);
    m_deps.runTonemapPass(cmd, /*skipBeautifiedOverlay=*/true);
    nvvk::endSingleTimeCommands(cmd, m_deps.device, m_deps.transientCmdPool, m_deps.app->getQueue(0).queue);
  }

  const VkExtent2D imageSize = m_deps.resources->gBuffers.getSize();
  // JPG (not PNG): tonemapped buffer alpha is 0 on sky/background hits; Comfy LoadImage
  // would treat that as black. JPG drops alpha so RGB matches the visible viewport.
  // quality=95 keeps artifacts below what diffusion conditioning can resolve.
  m_deps.app->saveImageToFile(m_deps.resources->gBuffers.getColorImage(Resources::eImgTonemapped), imageSize, path,
                              /*quality=*/95);
}

//--------------------------------------------------------------------------------------------------
// Polling + heartbeat
//--------------------------------------------------------------------------------------------------

void Controller::refreshJobProgress(const PendingJob& job, std::optional<agentic::JobProgress>& progress)
{
  if(!job.active || !initialized)
  {
    progress.reset();
    return;
  }
  progress = agentic::readJobProgressFile(bridgeRootPath(), job.id);
}

void Controller::pollNow()
{
  auto pollJob = [this](PendingJob& job, std::optional<agentic::JobProgress>& progress) {
    std::error_code ec;
    if(!job.active || !std::filesystem::exists(job.responsePath, ec) || ec)
      return;

    std::string error;
    bool        stillWriting = false;
    const auto  result       = agentic::readGenerationResultFile(job.responsePath, &error, &stillWriting);
    if(!result)
    {
      if(stillWriting)
        return;  // adapter is mid-publish; retry next poll (no thread sleep, no spam)

      // Fully written but unparseable: terminal. Clear the job so we don't
      // re-read the same bad file every frame.
      status      = "Agentic response parse failed";
      lastMessage = error;
      job.active  = false;
      progress.reset();
      LOGE("Agentic response parse failed (%s): %s\n", job.responsePath.string().c_str(), error.c_str());
      return;
    }

    // Guard against a stale / mis-named response file being applied to this job.
    if(!result->jobId.empty() && result->jobId != job.id)
    {
      status      = "Agentic response id mismatch";
      lastMessage = fmt::format("Response for {} does not match pending job {}", result->jobId, job.id);
      job.active  = false;
      progress.reset();
      LOGW("Agentic response id mismatch: got %s, expected %s\n", result->jobId.c_str(), job.id.c_str());
      return;
    }

    handleResult(*result, job.task);
    job.active = false;
    progress.reset();
  };

  pollJob(m_hdriJob, m_hdriProgress);
  pollJob(m_beautifyJob, m_beautifyProgress);
}

void Controller::refreshAdapterStatus(bool force)
{
  const auto now = std::chrono::steady_clock::now();
  if(!force && (now - m_lastAdapterCheck) < std::chrono::seconds(1))
    return;
  m_lastAdapterCheck = now;
  m_adapterStatus    = agentic::queryAdapterStatus(bridgeRootPath());
}

void Controller::tick()
{
  if(enabled && autoPoll)
    pollNow();
  refreshJobProgress(m_hdriJob, m_hdriProgress);
  refreshJobProgress(m_beautifyJob, m_beautifyProgress);

  // Refresh the heartbeat only when the feature is in use, or the window is open
  // (renderAgenticWindow refreshes it too). Users who never touch the bridge
  // shouldn't pay a per-second stat+read of a file that doesn't exist. The
  // throttle inside collapses repeat calls when several of these are true.
  const bool windowOpen = m_deps.resources && m_deps.resources->settings.showAgenticWindow;
  if(enabled || initialized || windowOpen)
    refreshAdapterStatus(/*force=*/false);
}

//--------------------------------------------------------------------------------------------------
// Response handling
//--------------------------------------------------------------------------------------------------

void Controller::handleResult(const agentic::GenerationResult& result, agentic::GenerationTask task)
{
  if(result.status != agentic::GenerationStatus::eSucceeded)
  {
    status      = fmt::format("{} job {}", agentic::toString(task), agentic::toString(result.status));
    lastMessage = result.message;
    LOGW("Agentic job %s finished with status %s: %s\n", result.jobId.c_str(), agentic::toString(result.status),
         result.message.c_str());
    return;
  }

  // The response comes from the adapter, which writes into a directory a separate
  // process controls. Confine the output path to the bridge root before handing
  // it to an image / HDR loader — reject absolute paths and "../" traversal so a
  // response can never make the renderer read a file outside the bridge.
  const std::optional<std::filesystem::path> confined = agentic::confineToBridgeRoot(bridgeRootPath(), result.outputPath);
  if(!confined)
  {
    status      = "Agentic output path rejected";
    lastMessage = "Adapter output path is not inside the bridge root: " + result.outputPath.generic_string();
    LOGE("Rejected agentic output path outside bridge root: %s\n", result.outputPath.generic_string().c_str());
    return;
  }
  const std::filesystem::path outputPath = *confined;

  std::error_code ec;
  if(outputPath.empty() || !std::filesystem::exists(outputPath, ec) || ec)
  {
    status      = "Agentic output missing";
    lastMessage = "Expected output path does not exist: " + outputPath.string();
    LOGE("Agentic output missing: %s\n", outputPath.string().c_str());
    return;
  }

  if(task == agentic::GenerationTask::eHdriFromPrompt)
  {
    if(m_deps.applyHdri)
      m_deps.applyHdri(outputPath);
    m_lastHdrPath = outputPath;
    status        = "HDRI applied";
    lastMessage   = outputPath.string();
    if(m_deps.resetFrame)
      m_deps.resetFrame();
    LOGI("Applied agentic HDRI: %s\n", outputPath.string().c_str());
    return;
  }

  if(task == agentic::GenerationTask::eImageToImage)
  {
    if(loadBeautifiedImage(outputPath))
    {
      if(m_deps.resources)
        m_deps.resources->settings.displayBuffer = DisplayBuffer::eAgenticBeautified;
      m_lastBeautifiedPath = outputPath;
      status               = "Beautified output displayed";
      lastMessage          = outputPath.string();
      if(m_deps.resetFrame)
        m_deps.resetFrame();
      LOGI("Loaded agentic beautified output: %s\n", outputPath.string().c_str());
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Beautified image: GPU upload + ImGui registration + destroy
//--------------------------------------------------------------------------------------------------

bool Controller::loadBeautifiedImage(const std::filesystem::path& filename)
{
  // Reached from pollNow()->handleResult(); guard the renderer deps like the
  // queue path does, so a result applied without a wired controller can't crash.
  if(!m_deps.resources || !m_deps.app || m_deps.device == VK_NULL_HANDLE || m_deps.transientCmdPool == VK_NULL_HANDLE)
    return false;

  int      width    = 0;
  int      height   = 0;
  int      channels = 0;
  stbi_uc* pixels   = stbi_load(filename.string().c_str(), &width, &height, &channels, 4);
  if(!pixels || width <= 0 || height <= 0)
  {
    status      = "Failed to load beautified output";
    lastMessage = filename.string();
    if(pixels)
      stbi_image_free(pixels);
    return false;
  }

  // Comfy output should match the saved input (viewport size at queue time).
  // Warn only when the current G-buffer size differs — usually the user resized
  // the window while the job was running. Still load and display the result.
  const VkExtent2D gbufferSize = m_deps.resources->gBuffers.getSize();
  const bool       sizeMatch =
      static_cast<uint32_t>(width) == gbufferSize.width && static_cast<uint32_t>(height) == gbufferSize.height;

  // No vkQueueWaitIdle: destroyBeautifiedImage() defers the actual free of
  // the previous descriptor + image until the in-flight frame is retired (see
  // its comment), so we can immediately reuse the member fields for the new image.
  destroyBeautifiedImage(true);

  VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
  imageInfo.extent            = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
  imageInfo.format            = VK_FORMAT_R8G8B8A8_UNORM;
  imageInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imageInfo.mipLevels         = 1;

  NVVK_CHECK(m_deps.resources->allocator.createImage(m_beautifiedImage, imageInfo, DEFAULT_VkImageViewCreateInfo));
  m_beautifiedImage.descriptor.sampler     = m_deps.resources->linearSampler;
  m_beautifiedImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  m_beautifiedExtent                       = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

  VkCommandBuffer     cmd{};
  nvvk::FrameUploader uploader;
  NVVK_CHECK(uploader.init({.allocator = &m_deps.resources->allocator, .debugName = "agenticUpload"}));
  nvvk::beginSingleTimeCommands(cmd, m_deps.device, m_deps.transientCmdPool);
  nvvk::cmdImageMemoryBarrier(cmd, {m_beautifiedImage.image, VK_IMAGE_LAYOUT_UNDEFINED, m_beautifiedImage.descriptor.imageLayout});
  const size_t byteCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
  NVVK_CHECK(uploader.appendImage(m_beautifiedImage, std::span(pixels, byteCount), m_beautifiedImage.descriptor.imageLayout));
  uploader.cmdUploadAppended(cmd);
  m_beautifiedImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  nvvk::cmdImageMemoryBarrier(cmd, {m_beautifiedImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    m_beautifiedImage.descriptor.imageLayout});
  nvvk::endSingleTimeCommands(cmd, m_deps.device, m_deps.transientCmdPool, m_deps.app->getQueue(0).queue);
  uploader.deinit();
  stbi_image_free(pixels);

  // Register the image with ImGui so we can draw it as a thumbnail / toggle
  // button in the Agentic Window, mirroring the OptiX denoiser pattern.
  m_beautifiedImguiDescriptor =
      ImGui_ImplVulkan_AddTexture(m_beautifiedImage.descriptor.imageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  if(!sizeMatch)
  {
    lastMessage = fmt::format(
        "Beautified image is {}x{} but the current viewport is {}x{}; displaying stretched. "
        "Click the thumbnail to toggle.",
        width, height, gbufferSize.width, gbufferSize.height);
    LOGW("%s\n", lastMessage.c_str());
  }
  return true;
}

void Controller::destroyBeautifiedImage(bool deferred)
{
  // Deferred destruction. The thumbnail in the Agentic Window was rendered earlier
  // this frame, so ImGui has a draw call referencing m_beautifiedImguiDescriptor
  // queued in its draw data. Freeing the descriptor / image right now would
  // invalidate that handle before ImGui_ImplVulkan_RenderDrawData consumes it,
  // tripping the VUID-vkCmdBindDescriptorSets-pDescriptorSets-parameter
  // validation error and crashing in the validation layer.
  //
  // submitResourceFree() queues the callback against the current frame slot;
  // nvapp runs it after that frame's command buffer has retired (i.e. after
  // ImGui's draw data has been fully consumed). Outside the frame loop (e.g.
  // during onDetach after vkDeviceWaitIdle) submitResourceFree falls back to
  // running the callback immediately, so this path is also safe at shutdown.
  if(m_deps.resources && m_deps.resources->settings.displayBuffer == DisplayBuffer::eAgenticBeautified)
    m_deps.resources->settings.displayBuffer = DisplayBuffer::eRendered;

  if(m_beautifiedImguiDescriptor == VK_NULL_HANDLE && m_beautifiedImage.image == VK_NULL_HANDLE)
    return;  // nothing to free

  // Capture the handles by value, then clear the members so a follow-up
  // loadBeautifiedImage() can immediately allocate fresh resources into them.
  VkDescriptorSet          oldDescriptor = m_beautifiedImguiDescriptor;
  nvvk::Image              oldImage      = m_beautifiedImage;
  nvvk::ResourceAllocator* allocator     = m_deps.resources ? &m_deps.resources->allocator : nullptr;

  m_beautifiedImguiDescriptor = VK_NULL_HANDLE;
  m_beautifiedImage           = {};
  m_beautifiedExtent          = {};

  auto freeFn = [oldDescriptor, oldImage, allocator]() mutable {
    if(oldDescriptor != VK_NULL_HANDLE)
      ImGui_ImplVulkan_RemoveTexture(oldDescriptor);
    if(oldImage.image != VK_NULL_HANDLE && allocator)
      allocator->destroyImage(oldImage);
  };

  if(m_deps.app && deferred)
    m_deps.app->submitResourceFree(std::move(freeFn));
  else
    freeFn();
}

}  // namespace agentic
