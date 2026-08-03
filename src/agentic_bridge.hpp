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

#pragma once

#include <chrono>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace agentic {

// Phase-one bridge task kinds. They describe what the renderer can ask an
// external generation tool to do, without naming or linking a specific tool.
enum class GenerationTask
{
  eTextToImage,
  eImageToImage,
  eHdriFromPrompt,
};

enum class GenerationStatus
{
  eQueued,
  eRunning,
  eSucceeded,
  eFailed,
};

struct BridgeLayout
{
  std::filesystem::path root;
  std::filesystem::path requests;
  std::filesystem::path responses;
  std::filesystem::path assets;
};

struct GenerationJob
{
  std::string    id;
  GenerationTask task{GenerationTask::eTextToImage};
  std::string    prompt;
  // ComfyUI API workflow filename under the adapter's --workflow-dir (e.g. image_beautifier.json).
  std::string                        workflowFile;
  std::filesystem::path              inputPath;
  std::filesystem::path              preferredOutputPath;
  std::map<std::string, std::string> parameters;
};

struct GenerationResult
{
  std::string           jobId;
  GenerationStatus      status{GenerationStatus::eQueued};
  std::filesystem::path outputPath;
  std::string           message;
};

// Live ComfyUI progress while a job is running (written by comfy_bridge.py).
struct JobProgress
{
  std::string jobId;
  std::string promptId;
  std::string phase;
  int         value{0};
  int         max{0};
  std::string node;
  std::string message;
};

[[nodiscard]] BridgeLayout bridgeLayout(const std::filesystem::path& root);
[[nodiscard]] const char*  toString(GenerationTask task);
[[nodiscard]] const char*  toString(GenerationStatus status);
// Default ComfyUI workflow JSON for each bridge task (basename only).
[[nodiscard]] const char*                     defaultWorkflowFile(GenerationTask task);
[[nodiscard]] std::optional<GenerationTask>   generationTaskFromString(std::string_view value);
[[nodiscard]] std::optional<GenerationStatus> generationStatusFromString(std::string_view value);
[[nodiscard]] std::string                     makeGenerationJobId(std::string_view prefix = "generation");

[[nodiscard]] bool validateGenerationJob(const GenerationJob& job, std::string* error = nullptr);
[[nodiscard]] bool ensureBridgeLayout(const std::filesystem::path& root, std::string* error = nullptr);

// Resolve an adapter-supplied, bridge-relative path (e.g. the "outputs.image" of
// a response) into an absolute path that is guaranteed to stay under `root`.
// The bridge directory is written to by a separate, untrusted process, so an
// absolute path or one containing a ".." component is rejected (returns nullopt)
// rather than letting a response read/overwrite files outside the bridge. On
// success the returned path is lexically normalized and confirmed to be within
// `root`. Used before handing a path to an image loader / HDR loader.
[[nodiscard]] std::optional<std::filesystem::path> confineToBridgeRoot(const std::filesystem::path& root,
                                                                       const std::filesystem::path& relative);

[[nodiscard]] std::string serializeBridgeManifest();
[[nodiscard]] std::string serializeGenerationJob(const GenerationJob& job);
[[nodiscard]] std::optional<GenerationResult> parseGenerationResult(std::string_view payload, std::string* error = nullptr);
// Read and parse a response file in a single, non-blocking pass (safe to call
// from the render/UI thread every frame). If the adapter is still publishing the
// file atomically (a sibling ".tmp" is present) or the file is momentarily empty,
// `stillWriting` is set to true and nullopt is returned — the caller should retry
// on the next poll rather than treating it as a failure. A fully written file
// that fails to parse returns nullopt with `stillWriting` false (terminal).
[[nodiscard]] std::optional<GenerationResult> readGenerationResultFile(const std::filesystem::path& path,
                                                                       std::string*                 error = nullptr,
                                                                       bool* stillWriting                 = nullptr);

[[nodiscard]] std::filesystem::path      jobProgressPath(const std::filesystem::path& root, std::string_view jobId);
[[nodiscard]] std::optional<JobProgress> parseJobProgress(std::string_view payload, std::string* error = nullptr);
[[nodiscard]] std::optional<JobProgress> readJobProgressFile(const std::filesystem::path& root, std::string_view jobId);

[[nodiscard]] std::filesystem::path writeBridgeManifest(const std::filesystem::path& root, std::string* error = nullptr);
[[nodiscard]] std::filesystem::path writeGenerationJob(const std::filesystem::path& root,
                                                       const GenerationJob&         job,
                                                       std::string*                 error = nullptr);

// Adapter liveness reporting --------------------------------------------------
// The optional sidecar process (e.g. utils/comfy_bridge/comfy_bridge.py) writes
// a heartbeat JSON into <root>/.adapter_heartbeat.json once per poll tick. The
// renderer reads it to show a green/yellow/red indicator: it answers "is anything
// actually listening on the other side of my filesystem queue?" without any IPC.
enum class AdapterStatus
{
  eUnknown,  // bridge not initialized / heartbeat path not yet known
  eDead,     // heartbeat file missing or older than kAdapterDeadThreshold
  eStale,    // heartbeat file present but older than kAdapterStaleThreshold
  eActive,   // heartbeat file is fresh
};

inline constexpr std::chrono::seconds kAdapterStaleThreshold{5};
// The adapter refreshes its heartbeat every ~1s (idle) and ~2s (mid-job), so a
// gap beyond this reliably means it stopped — kept short so a killed adapter is
// reported "not running" quickly rather than lingering as "not responding".
inline constexpr std::chrono::seconds kAdapterDeadThreshold{15};

struct AdapterHeartbeat
{
  int         pid{0};
  std::string startedAt;   // ISO-8601 string from the adapter (informational)
  std::string lastPollAt;  // ISO-8601 string from the adapter (informational)
  std::string comfyUrl;
  bool        comfyReachable{false};
  std::string adapter;  // e.g. "comfy_bridge.py"
  std::string adapterVersion;
};

struct AdapterStatusInfo
{
  AdapterStatus                   status{AdapterStatus::eUnknown};
  std::chrono::duration<double>   age{};      // wall-clock age of heartbeat file
  std::optional<AdapterHeartbeat> heartbeat;  // parsed payload, if present
  std::string                     error;      // populated if the file exists but parsing failed
};

[[nodiscard]] std::filesystem::path adapterHeartbeatPath(const std::filesystem::path& root);
[[nodiscard]] AdapterStatusInfo     queryAdapterStatus(const std::filesystem::path& root);
[[nodiscard]] const char*           toString(AdapterStatus status);

}  // namespace agentic
