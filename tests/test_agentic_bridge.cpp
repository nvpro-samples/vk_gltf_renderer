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

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>

#include <agentic_bridge.hpp>
#include <tinygltf/json.hpp>

#include "common/test_utils.hpp"

namespace {

using json = nlohmann::json;

std::string readTextFile(const std::filesystem::path& path)
{
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST(AgenticBridge, SerializesManifestWithOptionalGenerationCapabilities)
{
  const json manifest = json::parse(agentic::serializeBridgeManifest());

  EXPECT_EQ(manifest["schema"], "vk_gltf_renderer.agentic_bridge.manifest");
  EXPECT_EQ(manifest["schemaVersion"], 1);
  EXPECT_EQ(manifest["transport"], "filesystem");
  EXPECT_EQ(manifest["directories"]["requests"], "requests");
  EXPECT_EQ(manifest["controlPlane"]["style"], "mcp");
  EXPECT_EQ(manifest["controlPlane"]["status"], "planned");

  const auto& capabilities = manifest["capabilities"];
  ASSERT_EQ(capabilities.size(), 3);
  EXPECT_EQ(capabilities[0]["name"], "hdri_from_prompt");
  EXPECT_EQ(capabilities[0]["optionalAdapters"][0], "ComfyUI");
}

TEST(AgenticBridge, SerializesHdriGenerationJob)
{
  agentic::GenerationJob job;
  job.id                  = "test-hdri";
  job.task                = agentic::GenerationTask::eHdriFromPrompt;
  job.workflowFile        = agentic::defaultWorkflowFile(job.task);
  job.prompt              = "soft sunset studio lighting";
  job.preferredOutputPath = "assets/studio_sunset.hdr";
  job.parameters          = {{"height", "1024"}, {"width", "2048"}};

  const json request = json::parse(agentic::serializeGenerationJob(job));

  EXPECT_EQ(request["schema"], "vk_gltf_renderer.external_generation.request");
  EXPECT_EQ(request["job"]["id"], "test-hdri");
  EXPECT_EQ(request["job"]["kind"], "hdri_from_prompt");
  EXPECT_EQ(request["job"]["workflow"], "hdri_from_prompt.json");
  EXPECT_EQ(request["job"]["prompt"], "soft sunset studio lighting");
  EXPECT_EQ(request["job"]["outputs"]["preferredPath"], "assets/studio_sunset.hdr");
  EXPECT_EQ(request["job"]["parameters"]["width"], "2048");
}

TEST(AgenticBridge, ValidatesImageToImageInput)
{
  agentic::GenerationJob job;
  job.id           = "img2img";
  job.task         = agentic::GenerationTask::eImageToImage;
  job.workflowFile = agentic::defaultWorkflowFile(job.task);
  job.prompt       = "remove compression artifacts";

  std::string error;
  EXPECT_FALSE(agentic::validateGenerationJob(job, &error));
  EXPECT_NE(error.find("input path"), std::string::npos);

  job.inputPath = "assets/input.png";
  EXPECT_TRUE(agentic::validateGenerationJob(job, &error));
}

TEST(AgenticBridge, WritesBridgeLayoutAndRequest)
{
  const std::filesystem::path root = gltf_test::TestResources::getTempPath("agentic_bridge");
  std::filesystem::remove_all(root);

  std::string error;
  const auto  manifestPath = agentic::writeBridgeManifest(root, &error);
  ASSERT_FALSE(manifestPath.empty()) << error;

  const agentic::BridgeLayout layout = agentic::bridgeLayout(root);
  EXPECT_TRUE(std::filesystem::exists(layout.requests));
  EXPECT_TRUE(std::filesystem::exists(layout.responses));
  EXPECT_TRUE(std::filesystem::exists(layout.assets));
  EXPECT_TRUE(std::filesystem::exists(manifestPath));

  agentic::GenerationJob job;
  job.id                  = "queued-job";
  job.task                = agentic::GenerationTask::eTextToImage;
  job.workflowFile        = agentic::defaultWorkflowFile(job.task);
  job.prompt              = "brushed metal material reference";
  job.preferredOutputPath = layout.assets / "brushed_metal.png";

  const auto requestPath = agentic::writeGenerationJob(root, job, &error);
  ASSERT_FALSE(requestPath.empty()) << error;
  EXPECT_TRUE(std::filesystem::exists(requestPath));

  const json request = json::parse(readTextFile(requestPath));
  EXPECT_EQ(request["job"]["id"], "queued-job");
  EXPECT_EQ(request["job"]["kind"], "text_to_image");
  EXPECT_EQ(request["job"]["workflow"], "text_to_image.json");
}

TEST(AgenticBridge, ReadsGenerationResultFileWithRetries)
{
  const std::filesystem::path root         = gltf_test::TestResources::getTempPath("agentic_bridge_read");
  const std::filesystem::path responsePath = root / "responses" / "job-1.json";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(responsePath.parent_path());

  const char* payload = R"json({
    "schema": "vk_gltf_renderer.external_generation.response",
    "schemaVersion": 1,
    "jobId": "job-1",
    "status": "succeeded",
    "outputs": { "image": "assets/out.png" },
    "message": "done"
  })json";

  std::ofstream(responsePath, std::ios::binary) << payload;

  std::string error;
  const auto  result = agentic::readGenerationResultFile(responsePath, &error);
  ASSERT_TRUE(result.has_value()) << error;
  EXPECT_EQ(result->jobId, "job-1");
}

TEST(AgenticBridge, ParsesJobProgress)
{
  const char* payload = R"json({
    "schema": "vk_gltf_renderer.agentic_bridge.job_progress",
    "schemaVersion": 1,
    "jobId": "hdri-1",
    "promptId": "abc-123",
    "phase": "progress",
    "value": 3,
    "max": 10,
    "node": "7",
    "message": "ComfyUI 3/10"
  })json";

  std::string error;
  const auto  progress = agentic::parseJobProgress(payload, &error);
  ASSERT_TRUE(progress.has_value()) << error;
  EXPECT_EQ(progress->jobId, "hdri-1");
  EXPECT_EQ(progress->promptId, "abc-123");
  EXPECT_EQ(progress->value, 3);
  EXPECT_EQ(progress->max, 10);
  EXPECT_EQ(progress->message, "ComfyUI 3/10");
}

TEST(AgenticBridge, ParsesGenerationResult)
{
  const char* payload = R"json({
    "schema": "vk_gltf_renderer.external_generation.response",
    "schemaVersion": 1,
    "jobId": "queued-job",
    "status": "succeeded",
    "outputs": {
      "image": "assets/brushed_metal.png"
    },
    "message": "done"
  })json";

  std::string error;
  const auto  result = agentic::parseGenerationResult(payload, &error);
  ASSERT_TRUE(result.has_value()) << error;
  EXPECT_EQ(result->jobId, "queued-job");
  EXPECT_EQ(result->status, agentic::GenerationStatus::eSucceeded);
  EXPECT_EQ(result->outputPath.generic_string(), "assets/brushed_metal.png");
  EXPECT_EQ(result->message, "done");
}

TEST(AgenticBridge, RejectsMalformedAndFailedResults)
{
  std::string error;

  // Not JSON at all.
  EXPECT_FALSE(agentic::parseGenerationResult("{ not json", &error).has_value());
  // Wrong schema.
  EXPECT_FALSE(agentic::parseGenerationResult(R"json({"schema":"other","status":"succeeded","jobId":"x"})json", &error).has_value());
  // Unknown status.
  EXPECT_FALSE(agentic::parseGenerationResult(R"json({"schema":"vk_gltf_renderer.external_generation.response","status":"bogus","jobId":"x"})json",
                                              &error)
                   .has_value());

  // A well-formed "failed" response must parse (so the renderer can surface it).
  const auto failed = agentic::parseGenerationResult(R"json({"schema":"vk_gltf_renderer.external_generation.response","schemaVersion":1,"jobId":"j","status":"failed","message":"missing model"})json",
                                                     &error);
  ASSERT_TRUE(failed.has_value()) << error;
  EXPECT_EQ(failed->status, agentic::GenerationStatus::eFailed);
  EXPECT_EQ(failed->message, "missing model");
}

// Valid JSON with wrong-typed fields must degrade to nullopt, never throw (the
// caller runs on the render thread). Covers the type_error hardening.
TEST(AgenticBridge, SurvivesMistypedJsonFields)
{
  std::string error;
  // Top-level non-object.
  EXPECT_NO_THROW({ EXPECT_FALSE(agentic::parseGenerationResult("123", &error).has_value()); });
  // Present-but-mistyped fields (schema is a number; status is a number).
  EXPECT_NO_THROW({ EXPECT_FALSE(agentic::parseGenerationResult(R"json({"schema":123})json", &error).has_value()); });
  EXPECT_NO_THROW({
    EXPECT_FALSE(agentic::parseGenerationResult(R"json({"schema":"vk_gltf_renderer.external_generation.response","status":7,"jobId":"j"})json",
                                                &error)
                     .has_value());
  });
  // Job progress with a string where an int is expected.
  EXPECT_NO_THROW({
    EXPECT_FALSE(agentic::parseJobProgress(R"json({"schema":"vk_gltf_renderer.agentic_bridge.job_progress","jobId":"j","value":"nan"})json",
                                           &error)
                     .has_value());
  });
}

TEST(AgenticBridge, ReadResultReportsStillWritingForTempSibling)
{
  const std::filesystem::path root         = gltf_test::TestResources::getTempPath("agentic_bridge_tmp");
  const std::filesystem::path responsePath = root / "responses" / "job-9.json";
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(responsePath.parent_path());

  // Final file present, but the adapter's atomic-write temp sibling still exists:
  // the reader must report "still writing" (transient), not a terminal failure.
  std::ofstream(responsePath, std::ios::binary) << "{}";
  std::ofstream(responsePath.parent_path() / "job-9.json.tmp", std::ios::binary) << "partial";

  std::string error;
  bool        stillWriting = false;
  EXPECT_FALSE(agentic::readGenerationResultFile(responsePath, &error, &stillWriting).has_value());
  EXPECT_TRUE(stillWriting);
}

TEST(AgenticBridge, ConfinesAdapterPathsToBridgeRoot)
{
  const std::filesystem::path root = std::filesystem::path("bridge_root");

  // A normal bridge-relative output resolves under the root.
  const auto ok = agentic::confineToBridgeRoot(root, "assets/out.hdr");
  ASSERT_TRUE(ok.has_value());
  EXPECT_EQ(ok->generic_string(), (root / "assets" / "out.hdr").lexically_normal().generic_string());

  // Absolute paths and traversal are rejected.
  EXPECT_FALSE(agentic::confineToBridgeRoot(root, "../escape.hdr").has_value());
  EXPECT_FALSE(agentic::confineToBridgeRoot(root, "assets/../../escape.hdr").has_value());
#if defined(_WIN32)
  EXPECT_FALSE(agentic::confineToBridgeRoot(root, "C:/Windows/system32/x").has_value());
#else
  EXPECT_FALSE(agentic::confineToBridgeRoot(root, "/etc/passwd").has_value());
#endif
  EXPECT_FALSE(agentic::confineToBridgeRoot(root, "").has_value());
}
