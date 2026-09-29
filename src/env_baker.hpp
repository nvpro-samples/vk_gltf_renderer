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

#pragma once

#include <filesystem>
#include <mutex>

#include <nvvk/descriptors.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>

#include "nvshaders/hdr_io.h.slang"  // HDR_WORKGROUP_SIZE, the bake's workgroup
#include "shaders/env_bake_io.h.slang"
#include "shaders/shaderio.h"

struct Resources;

// Lat-long size an analytic sky is baked into. Fixed rather than a setting, because nothing a user
// would choose it for depends on it any more: sampling quality comes from kEnvSamplingGrid below,
// not from this size, and the path tracer evaluates the visible sky per ray. What it still sets is
// the sharpness of what rays read from the image -- rough reflections, and the physical sky's
// horizon in the rasterizer -- and 1024x512 keeps both where they were. A loaded HDR panorama
// ignores this and keeps its native file resolution.
static constexpr VkExtent2D kEnvBakeSize{1024, 512};

// The grid the alias table is built over, independently of the bake size (see
// nvvk::HdrIbl::updateFromGpuImage). A baked sky's lighting field is smooth -- its sun is sampled
// as a light of its own, never baked -- so a coarse grid costs next to nothing in variance, and the
// CPU alias build costs what the grid costs rather than what the image costs.
static constexpr VkExtent2D kEnvSamplingGrid{256, 128};
static_assert(kEnvBakeSize.width % kEnvSamplingGrid.width == 0 && kEnvBakeSize.height % kEnvSamplingGrid.height == 0,
              "kEnvSamplingGrid must divide kEnvBakeSize");
// The bake sums each cell's importance inside one workgroup (env_bake.slang), so a cell's block of
// texels must tile the workgroup.
static_assert(HDR_WORKGROUP_SIZE % (kEnvBakeSize.width / kEnvSamplingGrid.width) == 0
                  && HDR_WORKGROUP_SIZE % (kEnvBakeSize.height / kEnvSamplingGrid.height) == 0,
              "each sampling-grid cell must fit whole inside one env_bake workgroup");

//--------------------------------------------------------------------------------------------------
// EnvBaker -- turns an analytic sky into the lighting environment every renderer path already
// knows how to consume.
//
// It owns one lat-long image and one importance buffer, dispatches `env_bake.slang` into them,
// and hands the pair to `nvvk::HdrIbl::updateFromGpuImage` + `nvshaders::HdrEnvDome::
// updateEnvironment`. The point of the exercise is that afterwards a plain or gradient sky *is*
// an HDR environment as far as importance sampling, MIS and raster IBL are concerned -- no sky
// type earns a branch in any sampling path. A loaded HDR panorama skips this class entirely and
// keeps going through HdrIbl's file-load path at its native resolution.
//
// What it produces is the lighting field only: `env_bake.slang` omits the sun, because the sun is
// a KHR_lights_punctual directional light that next-event estimation samples directly. The
// visible sun disk belongs to the background field, which the renderer evaluates analytically per
// primary ray and which never comes from this class.
//
// Two entry points, deliberately named rather than one call with a mode enum:
//
//   * `preview(cmd)` re-bakes color only. Cheap enough to run every frame while a slider is
//     being dragged. The bake leaves the PDF in alpha at zero, so until the commit the path tracer
//     lights the environment by BSDF sampling alone -- unbiased, but noisier on a moving image --
//     and the raster path loses nothing. Records into the caller's command buffer.
//   * `commit()` runs the whole pipeline -- bake, alias rebuild, PDF write, prefilter refresh --
//     on slider release, on load, and on any MCP or command-line write. It submits and waits
//     internally, because `HdrIbl::updateFromGpuImage` reads the per-cell importance back to the
//     CPU to build the alias table; it cannot be folded into a caller's command buffer.
//
// Application-thread only: it touches Vulkan state and, through `commit()`, blocks on a queue
// submit. Do not call it from a background thread or an MCP tool that is not marked
// `runOnApplicationThread`.
//--------------------------------------------------------------------------------------------------
class EnvBaker
{
public:
  EnvBaker() = default;
  ~EnvBaker() { assert(!m_initialized && "deinit must be called"); }

  // `brunetonSet` is SkyBruneton's runtime descriptor set and its layout. The Bruneton sky is the
  // only one that bakes from textures, and it is one case of the single bake shader, so that set
  // becomes set 1 of the one pipeline layout and is bound on every dispatch -- see recordBake.
  //
  // Passing the layout at init means SkyBruneton has to be initialised first.
  void init(Resources& res, VkDescriptorSetLayout brunetonSetLayout, VkDescriptorSet brunetonSet);
  void deinit(Resources& res);

  // Allocates (or reallocates) the lat-long image and importance buffer. A resolution change
  // invalidates the descriptors HdrIbl holds, so the next environment refresh has to be a
  // `commit()` -- `needsCommit()` reports that.
  void       setResolution(Resources& res, VkExtent2D size);
  VkExtent2D getResolution() const { return m_size; }

  // Where the sun is, for the sky types whose *lighting* depends on it. Only Bruneton does: the
  // gradient bakes without its sun, and plain has none.
  void setSunDirection(const glm::vec3& direction) { m_sunDirection = direction; }

  // Observer height above the planet surface, in kilometres. Fixed rather than tracking the
  // camera -- see the fixed-observer note on brunetonObserver() in shaders/env_bake.slang.
  void setObserverAltitude(float km) { m_observerAltitude = km; }

  // Device address of the solar disk the Bruneton bake publishes: radiance and angular size for
  // whoever draws the visible sky. The value is GPU-written and never read back, so the renderer
  // hands this address to its background passes rather than a number. Zero before init().
  VkDeviceAddress sunDiskAddress() const { return m_sunDiskBuf.address; }

  // The parameter block the shader reads. Fill it, then call `preview()` or `commit()`; the
  // contents are uploaded as part of the dispatch.
  shaderio::SkyOmiParameters&       skyParams() { return m_skyParams; }
  const shaderio::SkyOmiParameters& skyParams() const { return m_skyParams; }

  void              setSkyType(shaderio::SkyType type) { m_skyType = type; }
  shaderio::SkyType getSkyType() const { return m_skyType; }

  // True until the first successful `commit()` at the current resolution. `preview()` is a no-op
  // while this holds, because HdrIbl's descriptors do not yet point at our image.
  bool needsCommit() const { return m_needsCommit; }

  // Color-only re-bake into the caller's command buffer. `refreshDome` reruns the prefilter
  // dispatches, which only the raster path needs -- the path tracer samples the lat-long image
  // directly and would pay ~5 ms per frame for nothing.
  void preview(VkCommandBuffer cmd, Resources& res, bool refreshDome);

  // Full refresh: bake, alias table, PDF, and (when `refreshDome`) the prefiltered cubemaps.
  // Submits and waits internally. Callers must reset frame accumulation afterwards -- a new
  // environment landing mid-accumulation blends two different skies.
  void commit(Resources& res, bool refreshDome);

  // Wall-clock cost of a `commit()`, per stage, for the log line it prints.
  //
  // The GPU profiler cannot see any of this: `commit()` records into its own single-time command
  // buffers and submits them outside the frame, so none of it lands in a profiled frame section
  // and `vk_gltf_measure` has nothing to time. Wall-clock is honest here precisely because the
  // commit is synchronous -- each stage submits and waits before the next begins.
  struct CommitTimings
  {
    double bakeMs{0.0};       // env_bake dispatch (+ mip chain)
    double aliasMs{0.0};      // PDF into alpha, Vose table build, descriptor rebind
    double prefilterMs{0.0};  // diffuse + glossy cubes; 0 when the dome was not refreshed
    double totalMs{0.0};
  };

private:
  void createPipeline(Resources& res);
  void destroyResources(Resources& res);
  void recordBake(VkCommandBuffer cmd);

  bool       m_initialized{false};
  bool       m_needsCommit{true};
  VkDevice   m_device{VK_NULL_HANDLE};
  VkExtent2D m_size{0, 0};
  uint32_t   m_mipLevels{1};

  shaderio::SkyType          m_skyType{shaderio::SkyType::ePlain};
  shaderio::SkyOmiParameters m_skyParams{};
  glm::vec3                  m_sunDirection{0.0F, 0.707F, 0.707F};
  float                      m_observerAltitude{0.001F};  // 1 m

  // Set and layout borrowed from SkyBruneton; this class does not own them.
  VkDescriptorSetLayout m_brunetonSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet       m_brunetonSet{VK_NULL_HANDLE};

  // Both are handed to HdrIbl by reference and must outlive every environment refresh; HdrIbl
  // never takes ownership of either.
  nvvk::Image  m_colorImage;     // Lat-long RGB + PDF in alpha, GENERAL layout end-to-end
  nvvk::Buffer m_importanceBuf;  // One float per kEnvSamplingGrid cell: sum of solidAngle(row) * max(rgb)
  nvvk::Buffer m_skyParamsBuf;   // Device-local SkyOmiParameters, updated in the command buffer per bake
  nvvk::Buffer m_sunDiskBuf;     // Device-local; written by the bake, read by the background passes


  VkPipeline           m_pipeline{VK_NULL_HANDLE};
  VkPipelineLayout     m_pipelineLayout{VK_NULL_HANDLE};
  nvvk::DescriptorPack m_descriptorPack;
  VkCommandPool        m_transientCmdPool{VK_NULL_HANDLE};
};
