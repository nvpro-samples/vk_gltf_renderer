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

//
// EnvBaker -- see env_baker.hpp for what this class is for and why it exists.
//
// Measured cost of a commit, on an RTX 6000 Ada:
//
//   env_bake dispatch                 ~0.4 ms at kEnvBakeSize (1024x512)
//   + mipmap chain                    (folded into the dispatch submit)
//   HdrIbl::updateFromGpuImage        ~0.5 ms: the alias table is built over kEnvSamplingGrid,
//                                     not the image
//   HdrEnvDome::updateEnvironment     ~28 ms; rasterizer only
//
// The mip chain is not an optimization detail to be dropped: the GGX prefilter samples the
// lat-long image thousands of times per output texel and thrashes cache without it -- ~63 ms
// unmipped against ~28 ms mipped.
//
// Known limitation: commit() is synchronous. Pipelining the importance readback across frames
// (dispatch on N, poll the fence on N+1, run Vose on N+2) would keep the application thread off
// the critical path, but HdrIbl::updateFromGpuImage submits and waits internally, so it needs an
// asynchronous entry point in nvpro_core2 that does not exist yet. The cost lands on a parameter
// commit rather than per frame, and preview() is what keeps slider drags smooth meanwhile.
//

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <span>
#include <vector>

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/mipmaps.hpp>

#include <chrono>
#include <cstdlib>

#include "env_baker.hpp"
#include "resources.hpp"

#include "_autogen/env_bake.slang.h"

//--------------------------------------------------------------------------------------------------
// One-time setup: compute pipeline, descriptor layout, the parameter uniform buffer, and the
// transient command pool commit() submits through. No lat-long image yet -- that waits for
// setResolution(), because its size is a user-facing setting.
//
void EnvBaker::init(Resources& res, VkDescriptorSetLayout brunetonSetLayout, VkDescriptorSet brunetonSet)
{
  assert(!m_initialized && "init called twice");
  m_brunetonSetLayout = brunetonSetLayout;
  m_brunetonSet       = brunetonSet;
  m_device            = res.allocator.getDevice();

  const VkCommandPoolCreateInfo poolInfo{
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = res.app->getQueue(0).familyIndex,
  };
  NVVK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_transientCmdPool));
  NVVK_DBG_NAME(m_transientCmdPool);

  createPipeline(res);

  // Device-local and written by recordBake() through the command buffer: a preview bake is
  // recorded while earlier frames may still be running an earlier bake that reads it.
  static_assert(sizeof(shaderio::SkyOmiParameters) % 4 == 0 && sizeof(shaderio::SkyOmiParameters) <= 65536, "vkCmdUpdateBuffer limits");
  NVVK_CHECK(res.allocator.createBuffer(m_skyParamsBuf, sizeof(shaderio::SkyOmiParameters),
                                        VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
  NVVK_DBG_NAME(m_skyParamsBuf.buffer);

  // Device-local, and addressed rather than bound: the bake writes it and the background passes
  // read it every frame, which is why it stays device-local rather than becoming host-visible --
  // the per-ray read is the hot path and BAR memory would make it slower. The host copy it needs
  // for the firefly clamp is taken by a 32-byte readback on commit instead, hence TRANSFER_SRC.
  // Zeroed at creation so a background pass that runs before the first bake draws no disk instead
  // of a random one.
  NVVK_CHECK(res.allocator.createBuffer(m_sunDiskBuf, sizeof(shaderio::SkySunDisk),
                                        VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
                                            | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                        VMA_MEMORY_USAGE_AUTO));
  NVVK_DBG_NAME(m_sunDiskBuf.buffer);
  {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool));
    vkCmdFillBuffer(cmd, m_sunDiskBuf.buffer, 0, VK_WHOLE_SIZE, 0);
    NVVK_CHECK(nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, res.app->getQueue(0).queue));
  }

  m_initialized = true;
}

//--------------------------------------------------------------------------------------------------
//
void EnvBaker::deinit(Resources& res)
{
  if(!m_initialized)
    return;

  destroyResources(res);
  res.allocator.destroyBuffer(m_skyParamsBuf);
  res.allocator.destroyBuffer(m_sunDiskBuf);

  vkDestroyPipeline(m_device, m_pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  m_descriptorPack.deinit();
  vkDestroyCommandPool(m_device, m_transientCmdPool, nullptr);

  m_pipeline         = VK_NULL_HANDLE;
  m_pipelineLayout   = VK_NULL_HANDLE;
  m_transientCmdPool = VK_NULL_HANDLE;
  m_initialized      = false;
}

//--------------------------------------------------------------------------------------------------
// Descriptor set and compute pipeline for env_bake.slang. The set mirrors EnvBakeBindings
// exactly; keeping the two in one header is what stops them drifting.
//
void EnvBaker::createPipeline(Resources& res)
{
  nvvk::DescriptorBindings bindings;
  bindings.addBinding(shaderio::EnvBakeBindings::eEnvBakeColor, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  bindings.addBinding(shaderio::EnvBakeBindings::eEnvBakeImportance, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  bindings.addBinding(shaderio::EnvBakeBindings::eEnvBakeSkyParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  NVVK_CHECK(m_descriptorPack.init(bindings, m_device, 1));
  NVVK_DBG_NAME(m_descriptorPack.getLayout());
  NVVK_DBG_NAME(m_descriptorPack.getPool());
  NVVK_DBG_NAME(m_descriptorPack.getSet(0));

  const VkPushConstantRange pushRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset     = 0,
      .size       = sizeof(shaderio::EnvBakePushConstant),
  };
  // Set 0 is ours; set 1 is SkyBruneton's LUTs, which only the PhysicalBruneton case of the shader
  // samples. One pipeline reads both, so both are always bound.
  NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorPack.getLayout(), m_brunetonSetLayout}, {pushRange}));
  NVVK_DBG_NAME(m_pipelineLayout);

  VkShaderModuleCreateInfo moduleInfo{
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = std::span(env_bake_slang).size_bytes(),
      .pCode    = env_bake_slang,
  };
  VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.pName = "main";
  stageInfo.pNext = &moduleInfo;

  VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  pipelineInfo.layout = m_pipelineLayout;
  pipelineInfo.stage  = stageInfo;
  NVVK_CHECK(vkCreateComputePipelines(m_device, {}, 1, &pipelineInfo, nullptr, &m_pipeline));
  NVVK_DBG_NAME(m_pipeline);
}

//--------------------------------------------------------------------------------------------------
// Allocate the lat-long image + importance buffer for `size`.
//
// The image carries a full mip chain (see the file header for why) and stays in GENERAL for its
// whole life: env_bake writes it as a storage image, env_write_pdf writes alpha as a storage
// image, and the prefilter samples it. One layout end to end means no transitions to get wrong.
//
void EnvBaker::setResolution(Resources& res, VkExtent2D size)
{
  assert(m_initialized && "init must be called first");
  assert(size.width > 0 && size.height > 0);
  if(m_size.width == size.width && m_size.height == size.height)
    return;

  // The previous image is still referenced by HdrIbl descriptors and may be in flight.
  NVVK_CHECK(vkDeviceWaitIdle(m_device));
  destroyResources(res);

  m_size      = size;
  m_mipLevels = nvvk::mipLevels(size);

  VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
  imageInfo.extent            = {size.width, size.height, 1};
  imageInfo.format            = VK_FORMAT_R32G32B32A32_SFLOAT;
  imageInfo.mipLevels         = m_mipLevels;
  imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                    | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

  // The sampled view spans every mip so the prefilter can trilinear-sample the chain; storage
  // writes implicitly target the view's first level, which is mip 0.
  VkImageViewCreateInfo viewInfo       = DEFAULT_VkImageViewCreateInfo;
  viewInfo.subresourceRange.levelCount = m_mipLevels;
  NVVK_CHECK(res.allocator.createImage(m_colorImage, imageInfo, viewInfo));
  // The layout every bake sees: commit() transitions the fresh image before its first bake, and
  // preview() refuses to run until that commit has happened.
  m_colorImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  NVVK_DBG_NAME(m_colorImage.image);
  NVVK_DBG_NAME(m_colorImage.descriptor.imageView);

  const VkSamplerCreateInfo samplerInfo{
      .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter    = VK_FILTER_LINEAR,
      .minFilter    = VK_FILTER_LINEAR,
      .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod       = VK_LOD_CLAMP_NONE,
  };
  NVVK_CHECK(res.samplerPool.acquireSampler(m_colorImage.descriptor.sampler, samplerInfo));

  NVVK_CHECK(res.allocator.createBuffer(m_importanceBuf,
                                        VkDeviceSize(kEnvSamplingGrid.width) * kEnvSamplingGrid.height * sizeof(float),
                                        VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT));
  NVVK_DBG_NAME(m_importanceBuf.buffer);

  // Written here, once per allocation, and never while a bake could be in flight: the set is not
  // update-after-bind, and the wait above is what makes rewriting it legal.
  nvvk::WriteSetContainer writes;
  writes.append(m_descriptorPack.makeWrite(shaderio::EnvBakeBindings::eEnvBakeColor), m_colorImage);
  writes.append(m_descriptorPack.makeWrite(shaderio::EnvBakeBindings::eEnvBakeImportance), m_importanceBuf);
  writes.append(m_descriptorPack.makeWrite(shaderio::EnvBakeBindings::eEnvBakeSkyParams), m_skyParamsBuf);
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  // Fresh image: HdrIbl still points at whatever it held before, and the mip chain is undefined.
  m_needsCommit = true;
}

//--------------------------------------------------------------------------------------------------
//
void EnvBaker::destroyResources(Resources& res)
{
  if(m_colorImage.descriptor.sampler != VK_NULL_HANDLE)
    res.samplerPool.releaseSampler(m_colorImage.descriptor.sampler);
  res.allocator.destroyImage(m_colorImage);
  res.allocator.destroyBuffer(m_importanceBuf);
  m_colorImage    = {};
  m_importanceBuf = {};
  m_size          = {0, 0};
  m_mipLevels     = 1;
  m_needsCommit   = true;
}

//--------------------------------------------------------------------------------------------------
// Record the color bake plus the mip chain. Shared by preview() and commit() so the two cannot
// produce different images from the same parameters.
//
void EnvBaker::recordBake(VkCommandBuffer cmd)
{
  // Behind a barrier that waits for any earlier bake still reading the buffer.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  vkCmdUpdateBuffer(cmd, m_skyParamsBuf.buffer, 0, sizeof(m_skyParams), &m_skyParams);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

  shaderio::EnvBakePushConstant push{};
  push.imageSize        = {m_size.width, m_size.height};
  push.skyType          = m_skyType;
  push.sunDirection     = m_sunDirection;
  push.observerAltitude = m_observerAltitude;
  push.sunDiskAddress   = m_sunDiskBuf.address;
  push.gridSize         = {kEnvSamplingGrid.width, kEnvSamplingGrid.height};

  // One pipeline for every sky: env_bake.slang switches on pc.skyType. Set 1 carries the LUTs only
  // the Bruneton case reads, and is bound regardless -- SkyBruneton precomputes at startup, so it
  // is always valid, and a set the shader may sample has to be bound whether this dispatch will or
  // not.
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_descriptorPack.getSetPtr(), 0, nullptr);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 1, 1, &m_brunetonSet, 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
  vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

  const VkExtent2D groups = nvvk::getGroupCounts(m_size, HDR_WORKGROUP_SIZE);
  vkCmdDispatch(cmd, groups.width, groups.height, 1);

  // cmdGenerateMipmaps transitions mip 0 out of `currentLayout` behind a barrier that already
  // orders against the dispatch above, blits the chain, and restores every level to that same
  // layout -- so the GENERAL-everywhere contract holds with no extra barriers of our own.
  nvvk::cmdGenerateMipmaps(cmd, m_colorImage.image, m_size, m_mipLevels, /*layerCount*/ 1, VK_IMAGE_LAYOUT_GENERAL);
}

//--------------------------------------------------------------------------------------------------
// Color-only re-bake, recorded into the caller's frame command buffer.
//
// Deliberately skips the alias rebuild and the PDF write. The bake writes alpha = 0, so until the
// next commit every texel's PDF is zero: the path tracer's environment next-event estimation
// rejects each sample (it needs a positive PDF), and BSDF rays that escape take the full MIS
// weight. The environment is still lit, only through BSDF sampling -- unbiased, but noisier than
// with NEE, most visibly around a small bright feature. The visible background does not come from
// here at all, so what the user watches still follows the slider exactly.
//
void EnvBaker::preview(VkCommandBuffer cmd, Resources& res, bool refreshDome)
{
  assert(m_initialized && "init must be called first");
  if(m_needsCommit)
    return;  // HdrIbl descriptors do not point at our image yet; nothing would be read.

  recordBake(cmd);

  // The bake also rewrote the sun-disk buffer, which the background, raster, and path-tracing
  // passes later in this same command buffer read through frameInfo->sunDisk. The mip barrier
  // inside recordBake orders the image only. (commit() needs no equivalent: it submits and waits.)
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                             | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

  // Diffuse cube only: the glossy one is twice the cost and this runs every frame of a drag. A
  // colour change shows up mostly in the diffuse term, and the commit on release refreshes both,
  // so what lags during the interaction is specular reflection sharpness -- not the sky's colour.
  if(refreshDome)
    res.hdrDome.updateEnvironment(cmd, res.hdrIbl.getDescriptorSet(), nvshaders::HdrEnvDome::PrefilterSet::eDiffuseOnly);
}

//--------------------------------------------------------------------------------------------------
// Full refresh. Three submits, matching the three steps Phase 1 benchmarked independently.
//
void EnvBaker::commit(Resources& res, bool refreshDome)
{
  assert(m_initialized && "init must be called first");
  if(m_size.width == 0 || m_size.height == 0)
  {
    LOGW("EnvBaker::commit called before setResolution; skipping\n");
    return;
  }

  const nvvk::QueueInfo queueInfo = res.app->getQueue(0);

  // Wait for in-flight frames before touching the environment.
  //
  // updateFromGpuImage() below begins by destroying the image and alias buffer the descriptor sets
  // still point at, and commit() runs from the top of onRender -- so without this, resources the
  // GPU may still be reading from a previous frame are freed underneath it, and the descriptors
  // referencing them dangle until they are rewritten further down. createHDR() already guards the
  // file-load path the same way, for the same reason; the baked path simply never got the same
  // treatment. commit() is synchronous and runs on a parameter change rather than per frame, so
  // the stall costs nothing that matters.
  NVVK_CHECK(vkDeviceWaitIdle(m_device));

  // Stage timings. Each stage below submits and waits, so reading the clock between them measures
  // the stage rather than the queue depth -- see CommitTimings for why the GPU profiler cannot.
  using Clock        = std::chrono::steady_clock;
  const auto started = Clock::now();
  auto       mark    = [](Clock::time_point& since) {
    const auto now = Clock::now();
    const auto ms  = std::chrono::duration<double, std::milli>(now - since).count();
    since          = now;
    return ms;
  };
  auto          stage = started;
  CommitTimings timings{};

  // -- Bake color + importance ---------------------------------------------------------------
  {
    nvutils::ScopedTimer st("EnvBaker::bake");
    VkCommandBuffer      cmd = VK_NULL_HANDLE;
    NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool));

    // The whole chain goes UNDEFINED -> GENERAL on the first bake after (re)allocation; later
    // bakes re-enter from GENERAL, which is also a legal source for a discarding transition.
    const VkImageSubresourceRange fullRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, m_mipLevels, 0, 1};
    nvvk::cmdImageMemoryBarrier(cmd, {m_colorImage.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, fullRange});

    recordBake(cmd);
    nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, queueInfo.queue);
  }
  timings.bakeMs = mark(stage);

  // -- Alias table + PDF into alpha ----------------------------------------------------------
  // Rebinds the HdrIbl `eHdr` descriptor to our image and `eImpSamples` to the freshly built
  // alias buffer. Submits and waits internally (the Vose construction is CPU-side).
  res.hdrIbl.updateFromGpuImage(queueInfo, m_colorImage, m_importanceBuf, m_size, kEnvSamplingGrid);
  timings.aliasMs = mark(stage);

  // -- Prefiltered cubemaps for raster IBL ---------------------------------------------------
  // Reuses the cubes allocated by HdrEnvDome::create() at renderer init; only the two prefilter
  // dispatches rerun, so the cube descriptors written once at init stay valid and must not be
  // rewritten here.
  if(refreshDome)
  {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool));
    res.hdrDome.updateEnvironment(cmd, res.hdrIbl.getDescriptorSet());
    nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, queueInfo.queue);
    timings.prefilterMs = mark(stage);
  }

  timings.totalMs = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  LOGI("EnvBaker::commit %.2f ms (bake %.2f, alias+pdf %.2f, prefilter %.2f)\n", timings.totalMs, timings.bakeMs,
       timings.aliasMs, timings.prefilterMs);

  m_needsCommit = false;
}
