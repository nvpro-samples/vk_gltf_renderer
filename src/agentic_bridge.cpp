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
// Filesystem agentic bridge — JSON manifest, job requests, and response parsing.
//
// Defines the renderer ↔ adapter contract (schemas, bridge layout, generation tasks).
// The Python ComfyUI adapter (utils/comfy_bridge/comfy_bridge.py) watches the same
// directories; this module does not talk to ComfyUI directly.
//

#include "agentic_bridge.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>

#include <tinygltf/json.hpp>

namespace agentic {
namespace {

//--------------------------------------------------------------------------------------------------
// Internal helpers
//--------------------------------------------------------------------------------------------------

using json = nlohmann::json;

constexpr int         kSchemaVersion  = 1;
constexpr const char* kManifestSchema = "vk_gltf_renderer.agentic_bridge.manifest";
constexpr const char* kRequestSchema  = "vk_gltf_renderer.external_generation.request";
constexpr const char* kResponseSchema = "vk_gltf_renderer.external_generation.response";
constexpr const char* kProgressSchema = "vk_gltf_renderer.agentic_bridge.job_progress";

void setError(std::string* error, std::string message)
{
  if(error)
  {
    *error = std::move(message);
  }
}

std::string pathToString(const std::filesystem::path& path)
{
  return path.generic_string();
}

std::filesystem::path pathFromJson(const json& value)
{
  if(!value.is_string())
  {
    return {};
  }
  return std::filesystem::path(value.get<std::string>());
}

std::string sanitizeIdPrefix(std::string_view prefix)
{
  std::string result;
  result.reserve(prefix.size());
  for(unsigned char c : prefix)
  {
    if(std::isalnum(c) || c == '-' || c == '_')
    {
      result.push_back(static_cast<char>(c));
    }
    else
    {
      result.push_back('_');
    }
  }
  return result.empty() ? "generation" : result;
}

bool createDirectory(const std::filesystem::path& path, std::string* error)
{
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  if(ec)
  {
    setError(error, "Failed to create directory '" + pathToString(path) + "': " + ec.message());
    return false;
  }
  return true;
}

std::string slurpTextFile(const std::filesystem::path& path)
{
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void stripUtf8Bom(std::string& payload)
{
  if(payload.size() >= 3 && static_cast<unsigned char>(payload[0]) == 0xEF
     && static_cast<unsigned char>(payload[1]) == 0xBB && static_cast<unsigned char>(payload[2]) == 0xBF)
  {
    payload.erase(0, 3);
  }
}

std::filesystem::path responseTempPath(const std::filesystem::path& responsePath)
{
  // Matches comfy_bridge.write_json_atomic: foo.json -> foo.json.tmp
  return responsePath.parent_path() / (responsePath.filename().string() + ".tmp");
}

std::filesystem::path writeTextFile(const std::filesystem::path& path, std::string_view payload, std::string* error)
{
  std::ofstream file(path, std::ios::binary);
  if(!file)
  {
    setError(error, "Failed to open '" + pathToString(path) + "' for writing");
    return {};
  }

  file << payload << '\n';
  if(!file)
  {
    setError(error, "Failed to write '" + pathToString(path) + "'");
    return {};
  }

  return path;
}

// Write through a sibling ".tmp" then rename into place, mirroring the adapter's
// write_json_atomic(). A directory-watching adapter must never observe a
// half-written request, so requests use this rather than a plain stream.
std::filesystem::path writeTextFileAtomic(const std::filesystem::path& path, std::string_view payload, std::string* error)
{
  const std::filesystem::path tmp = path.parent_path() / (path.filename().string() + ".tmp");
  if(writeTextFile(tmp, payload, error).empty())
  {
    return {};
  }

  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if(ec)
  {
    std::filesystem::remove(tmp, ec);
    setError(error, "Failed to publish '" + pathToString(path) + "': " + ec.message());
    return {};
  }
  return path;
}

json directoriesJson()
{
  return {{"requests", "requests"}, {"responses", "responses"}, {"assets", "assets"}};
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// Bridge layout
//--------------------------------------------------------------------------------------------------

BridgeLayout bridgeLayout(const std::filesystem::path& root)
{
  return {
      .root      = root,
      .requests  = root / "requests",
      .responses = root / "responses",
      .assets    = root / "assets",
  };
}

//--------------------------------------------------------------------------------------------------
// Task and status naming
//--------------------------------------------------------------------------------------------------

const char* toString(GenerationTask task)
{
  switch(task)
  {
    case GenerationTask::eTextToImage:
      return "text_to_image";
    case GenerationTask::eImageToImage:
      return "image_to_image";
    case GenerationTask::eHdriFromPrompt:
      return "hdri_from_prompt";
  }
  return "unknown";
}

const char* defaultWorkflowFile(GenerationTask task)
{
  switch(task)
  {
    case GenerationTask::eTextToImage:
      return "text_to_image.json";
    case GenerationTask::eImageToImage:
      return "image_beautifier.json";
    case GenerationTask::eHdriFromPrompt:
      return "hdri_from_prompt.json";
  }
  return "";
}

const char* toString(GenerationStatus status)
{
  switch(status)
  {
    case GenerationStatus::eQueued:
      return "queued";
    case GenerationStatus::eRunning:
      return "running";
    case GenerationStatus::eSucceeded:
      return "succeeded";
    case GenerationStatus::eFailed:
      return "failed";
  }
  return "unknown";
}

std::optional<GenerationTask> generationTaskFromString(std::string_view value)
{
  if(value == "text_to_image")
    return GenerationTask::eTextToImage;
  if(value == "image_to_image")
    return GenerationTask::eImageToImage;
  if(value == "hdri_from_prompt")
    return GenerationTask::eHdriFromPrompt;
  return std::nullopt;
}

std::optional<GenerationStatus> generationStatusFromString(std::string_view value)
{
  if(value == "queued")
    return GenerationStatus::eQueued;
  if(value == "running")
    return GenerationStatus::eRunning;
  if(value == "succeeded")
    return GenerationStatus::eSucceeded;
  if(value == "failed")
    return GenerationStatus::eFailed;
  return std::nullopt;
}

//--------------------------------------------------------------------------------------------------
// Job validation and layout setup
//--------------------------------------------------------------------------------------------------

std::string makeGenerationJobId(std::string_view prefix)
{
  static std::atomic_uint64_t counter{0};

  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();

  std::ostringstream id;
  id << sanitizeIdPrefix(prefix) << "-" << ms << "-" << counter.fetch_add(1, std::memory_order_relaxed);
  return id.str();
}

bool validateGenerationJob(const GenerationJob& job, std::string* error)
{
  if(job.id.empty())
  {
    setError(error, "Generation job id is empty");
    return false;
  }

  if(job.prompt.empty())
  {
    setError(error, "Generation job prompt is empty");
    return false;
  }

  if(job.workflowFile.empty())
  {
    setError(error, "Generation job workflow file is empty");
    return false;
  }

  if(job.task == GenerationTask::eImageToImage && job.inputPath.empty())
  {
    setError(error, "Image-to-image generation requires an input path");
    return false;
  }

  return true;
}

std::optional<std::filesystem::path> confineToBridgeRoot(const std::filesystem::path& root, const std::filesystem::path& relative)
{
  if(root.empty() || relative.empty() || relative.is_absolute())
  {
    return std::nullopt;
  }

  // Reject any traversal component outright. lexically_normal() below would also
  // collapse "assets/../.." but an explicit ".." check keeps the intent obvious
  // and refuses even paths that happen to normalize back inside the root.
  for(const std::filesystem::path& part : relative)
  {
    if(part == "..")
    {
      return std::nullopt;
    }
  }

  const std::filesystem::path resolved = (root / relative).lexically_normal();

  // Is `p` equal to `dir` or a child of it? Requires the match to end on a path
  // separator so a sibling like "<root>2/..." is not mistaken for "<root>".
  const auto underDir = [](std::string dir, const std::string& p) {
    while(dir.size() > 1 && dir.back() == '/')
      dir.pop_back();
    return p == dir || (p.size() > dir.size() && p.compare(0, dir.size(), dir) == 0 && p[dir.size()] == '/');
  };

  // Lexical containment (defense in depth on top of the per-component ".." check).
  if(!underDir(root.lexically_normal().generic_string(), resolved.generic_string()))
    return std::nullopt;

  // Symlink-aware containment: weakly_canonical resolves symlinks in the existing
  // prefix, so a symlink planted inside the bridge dir that points outside is
  // caught here even though the lexical path looked contained. If the root can't
  // be canonicalized (e.g. does not exist yet) we keep the lexical result, which
  // already passed the check above.
  std::error_code             ec;
  const std::filesystem::path canonRoot = std::filesystem::weakly_canonical(root, ec);
  if(!ec)
  {
    const std::filesystem::path canonResolved = std::filesystem::weakly_canonical(resolved, ec);
    if(!ec && !underDir(canonRoot.generic_string(), canonResolved.generic_string()))
      return std::nullopt;  // a symlink component escapes the bridge root
  }

  return resolved;
}

bool ensureBridgeLayout(const std::filesystem::path& root, std::string* error)
{
  if(root.empty())
  {
    setError(error, "Bridge root path is empty");
    return false;
  }

  const BridgeLayout layout = bridgeLayout(root);
  return createDirectory(layout.root, error) && createDirectory(layout.requests, error)
         && createDirectory(layout.responses, error) && createDirectory(layout.assets, error)
         && createDirectory(root / ".job_progress", error);
}

//--------------------------------------------------------------------------------------------------
// JSON serialization
//--------------------------------------------------------------------------------------------------

std::string serializeBridgeManifest()
{
  json manifest = {
      {"schema", kManifestSchema},
      {"schemaVersion", kSchemaVersion},
      {"application", "vk_gltf_renderer"},
      {"transport", "filesystem"},
      {"directories", directoriesJson()},
      {"capabilities", json::array({
                           {{"name", toString(GenerationTask::eHdriFromPrompt)},
                            {"description", "Generate an HDR or EXR environment map from a text prompt."},
                            {"inputs", json::array({"prompt"})},
                            {"outputs", json::array({"hdr", "exr"})},
                            {"optionalAdapters", json::array({"ComfyUI"})}},
                           {{"name", toString(GenerationTask::eTextToImage)},
                            {"description", "Generate a texture or reference image from a text prompt."},
                            {"inputs", json::array({"prompt"})},
                            {"outputs", json::array({"png", "jpg", "exr"})},
                            {"optionalAdapters", json::array({"ComfyUI"})}},
                           {{"name", toString(GenerationTask::eImageToImage)},
                            {"description", "Enhance or transform an existing image asset."},
                            {"inputs", json::array({"prompt", "image"})},
                            {"outputs", json::array({"png", "jpg", "exr"})},
                            {"optionalAdapters", json::array({"ComfyUI"})}},
                       })},
      {"controlPlane",
       {{"style", "mcp"}, {"status", "planned"}, {"phase", 2}, {"description", "Scene inspection and mutation tools are reserved for the next phase."}}},
  };

  return manifest.dump(2);
}

std::string serializeGenerationJob(const GenerationJob& job)
{
  json request = {
      {"schema", kRequestSchema},
      {"schemaVersion", kSchemaVersion},
      {"job",
       {
           {"id", job.id},
           {"kind", toString(job.task)},
           {"workflow", job.workflowFile},
           {"prompt", job.prompt},
           {"parameters", job.parameters},
           {"inputs", json::object()},
           {"outputs", json::object()},
       }},
  };

  if(!job.inputPath.empty())
  {
    request["job"]["inputs"]["image"] = pathToString(job.inputPath);
  }
  if(!job.preferredOutputPath.empty())
  {
    request["job"]["outputs"]["preferredPath"] = pathToString(job.preferredOutputPath);
  }

  return request.dump(2);
}

//--------------------------------------------------------------------------------------------------
// Response parsing
//--------------------------------------------------------------------------------------------------

std::optional<GenerationResult> parseGenerationResult(std::string_view payload, std::string* error)
{
  json resultJson = json::parse(payload.begin(), payload.end(), nullptr, false);
  if(resultJson.is_discarded() || !resultJson.is_object())
  {
    setError(error, "Generation result is not a valid JSON object (" + std::to_string(payload.size()) + " bytes)");
    return std::nullopt;
  }

  // A parseable JSON object can still carry wrong-typed fields; json::value/get
  // throw type_error then. Catch it so a malformed-but-valid file degrades to an
  // error rather than aborting the caller (pollNow runs on the render/UI thread).
  try
  {
    const std::string schema = resultJson.value("schema", "");
    if(schema != kResponseSchema)
    {
      setError(error, "Generation result schema is not '" + std::string(kResponseSchema) + "'");
      return std::nullopt;
    }

    const std::string               statusString = resultJson.value("status", "");
    std::optional<GenerationStatus> status       = generationStatusFromString(statusString);
    if(!status)
    {
      setError(error, "Generation result has unknown status '" + statusString + "'");
      return std::nullopt;
    }

    GenerationResult result;
    result.jobId   = resultJson.value("jobId", "");
    result.status  = *status;
    result.message = resultJson.value("message", "");
    if(const auto outputsIt = resultJson.find("outputs"); outputsIt != resultJson.end() && outputsIt->is_object())
    {
      result.outputPath = pathFromJson(outputsIt->value("image", json{}));
    }

    if(result.jobId.empty())
    {
      setError(error, "Generation result jobId is empty");
      return std::nullopt;
    }

    return result;
  }
  catch(const nlohmann::json::exception& e)
  {
    setError(error, std::string("Generation result has a mistyped field: ") + e.what());
    return std::nullopt;
  }
}

std::optional<GenerationResult> readGenerationResultFile(const std::filesystem::path& path, std::string* error, bool* stillWriting)
{
  const auto setPending = [&](bool pending) {
    if(stillWriting)
      *stillWriting = pending;
  };
  setPending(false);

  // The adapter publishes responses atomically (write foo.json.tmp, then rename).
  // If the temp sibling is present the final file is mid-publish: signal "retry
  // next poll" rather than blocking the render thread with a sleep loop.
  std::error_code ec;
  if(std::filesystem::exists(responseTempPath(path), ec) && !ec)
  {
    setPending(true);
    setError(error, "Generation result is still being written");
    return std::nullopt;
  }

  std::string payload = slurpTextFile(path);
  stripUtf8Bom(payload);
  if(payload.empty())
  {
    // An empty file with no temp sibling can still be a momentary state between
    // create and write on some filesystems; treat as transient.
    setPending(true);
    setError(error, "Generation result file is empty");
    return std::nullopt;
  }

  std::string parseError;
  if(auto result = parseGenerationResult(payload, &parseError))
  {
    return result;
  }

  // Fully written (no temp sibling) but not parseable — terminal.
  setError(error, parseError.empty() ? "Failed to read generation result" : parseError);
  return std::nullopt;
}

//--------------------------------------------------------------------------------------------------
// Job progress (ComfyUI WebSocket sidecar)
//--------------------------------------------------------------------------------------------------

std::filesystem::path jobProgressPath(const std::filesystem::path& root, std::string_view jobId)
{
  if(root.empty() || jobId.empty())
  {
    return {};
  }
  return root / ".job_progress" / (std::string(jobId) + ".json");
}

std::optional<JobProgress> parseJobProgress(std::string_view payload, std::string* error)
{
  json doc = json::parse(payload, nullptr, false);
  if(doc.is_discarded() || !doc.is_object())
  {
    setError(error, "Job progress is not valid JSON");
    return std::nullopt;
  }

  try
  {
    if(doc.value("schema", "") != kProgressSchema)
    {
      setError(error, "Job progress schema mismatch");
      return std::nullopt;
    }

    JobProgress progress;
    progress.jobId    = doc.value("jobId", "");
    progress.promptId = doc.value("promptId", "");
    progress.phase    = doc.value("phase", "");
    progress.value    = doc.value("value", 0);
    progress.max      = doc.value("max", 0);
    progress.node     = doc.value("node", "");
    progress.message  = doc.value("message", "");

    if(progress.jobId.empty())
    {
      setError(error, "Job progress jobId is empty");
      return std::nullopt;
    }

    return progress;
  }
  catch(const nlohmann::json::exception& e)
  {
    setError(error, std::string("Job progress has a mistyped field: ") + e.what());
    return std::nullopt;
  }
}

std::optional<JobProgress> readJobProgressFile(const std::filesystem::path& root, std::string_view jobId)
{
  const std::filesystem::path path = jobProgressPath(root, jobId);
  if(path.empty())
  {
    return std::nullopt;
  }

  std::error_code ec;
  if(!std::filesystem::exists(path, ec) || ec)
  {
    return std::nullopt;
  }

  std::string payload = slurpTextFile(path);
  stripUtf8Bom(payload);
  if(payload.empty())
  {
    return std::nullopt;
  }

  return parseJobProgress(payload);
}

//--------------------------------------------------------------------------------------------------
// Bridge file I/O
//--------------------------------------------------------------------------------------------------

std::filesystem::path writeBridgeManifest(const std::filesystem::path& root, std::string* error)
{
  if(!ensureBridgeLayout(root, error))
  {
    return {};
  }

  return writeTextFile(bridgeLayout(root).root / "manifest.json", serializeBridgeManifest(), error);
}

std::filesystem::path writeGenerationJob(const std::filesystem::path& root, const GenerationJob& job, std::string* error)
{
  if(!validateGenerationJob(job, error))
  {
    return {};
  }

  if(!ensureBridgeLayout(root, error))
  {
    return {};
  }

  return writeTextFileAtomic(bridgeLayout(root).requests / (job.id + ".json"), serializeGenerationJob(job), error);
}

//--------------------------------------------------------------------------------------------------
// Adapter heartbeat
//--------------------------------------------------------------------------------------------------

std::filesystem::path adapterHeartbeatPath(const std::filesystem::path& root)
{
  return root.empty() ? std::filesystem::path{} : (root / ".adapter_heartbeat.json");
}

const char* toString(AdapterStatus status)
{
  switch(status)
  {
    case AdapterStatus::eUnknown:
      return "unknown";
    case AdapterStatus::eDead:
      return "dead";
    case AdapterStatus::eStale:
      return "stale";
    case AdapterStatus::eActive:
      return "active";
  }
  return "unknown";
}

AdapterStatusInfo queryAdapterStatus(const std::filesystem::path& root)
{
  AdapterStatusInfo info;
  if(root.empty())
  {
    return info;  // eUnknown
  }

  const std::filesystem::path path = adapterHeartbeatPath(root);
  std::error_code             ec;
  if(!std::filesystem::exists(path, ec) || ec)
  {
    info.status = AdapterStatus::eDead;
    return info;
  }

  const auto writeTime = std::filesystem::last_write_time(path, ec);
  if(ec)
  {
    info.status = AdapterStatus::eDead;
    info.error  = "stat failed: " + ec.message();
    return info;
  }

  // Convert file_time_type -> system_clock via clock_cast where available, else
  // via the documented epoch-shift trick. C++20 clock_cast is the clean path.
  const auto sysTime = std::chrono::clock_cast<std::chrono::system_clock>(writeTime);
  info.age           = std::chrono::system_clock::now() - sysTime;

  // Read the payload best-effort. Even if parsing fails, the file's mtime alone
  // is enough to classify alive/stale/dead — the JSON only adds metadata.
  std::ifstream f(path, std::ios::binary);
  if(f)
  {
    std::ostringstream ss;
    ss << f.rdbuf();
    json hb = json::parse(ss.str(), nullptr, false);
    if(hb.is_discarded() || !hb.is_object())
    {
      info.error = "heartbeat JSON parse failed";
    }
    else
    {
      // Mistyped fields (json::value throws type_error) are non-fatal here — the
      // mtime already classifies the status, so just skip the metadata.
      try
      {
        AdapterHeartbeat h;
        h.pid            = hb.value("pid", 0);
        h.startedAt      = hb.value("startedAt", "");
        h.lastPollAt     = hb.value("lastPollAt", "");
        h.comfyUrl       = hb.value("comfyUrl", "");
        h.comfyReachable = hb.value("comfyReachable", false);
        h.adapter        = hb.value("adapter", "");
        h.adapterVersion = hb.value("adapterVersion", "");
        info.heartbeat   = std::move(h);
      }
      catch(const nlohmann::json::exception& e)
      {
        info.error = std::string("heartbeat has a mistyped field: ") + e.what();
      }
    }
  }

  if(info.age >= kAdapterDeadThreshold)
    info.status = AdapterStatus::eDead;
  else if(info.age >= kAdapterStaleThreshold)
    info.status = AdapterStatus::eStale;
  else
    info.status = AdapterStatus::eActive;

  return info;
}

}  // namespace agentic
