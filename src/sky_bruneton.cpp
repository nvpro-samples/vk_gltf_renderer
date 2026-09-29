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
// SkyBruneton -- see sky_bruneton.hpp for what this owns and why it owns it.
//
// This file is the Vulkan half of the port of `atmosphere/model.cc` from
// github.com/ebruneton/precomputed_atmospheric_scattering. Upstream drives the precomputation with
// OpenGL fragment shaders rendering into FBOs, and -- the part that does not survive the port at
// all -- synthesizes its GLSL at runtime, pasting every atmosphere constant into the source as a
// literal before calling glCompileShader. Slang here is compiled at CMake time, so the constants
// travel in a uniform buffer instead and the passes are compute dispatches into storage images.
// The maths is untouched; it lives in shaders/sky_bruneton_functions.h.slang.
//

#include <cstring>
#include <string>
#include <vector>

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/helpers.hpp>


#include <cmath>

#include "resources.hpp"
#include "_autogen/sky_bruneton_precompute.slang.h"

#include "sky_bruneton.hpp"

// The built-in worlds.
//
// Earth is upstream's apart from the aerosol, which upstream sets cleaner than any real sky --
// see Settings::atmoMieScattering for the measurement that moved it. It must stay identical to
// skyBrunetonEarthDefaults(), or the panel opens on "Custom" for a scene nobody edited.
//
// Mars and the alien world are plausible rather than authoritative: Mars uses its real radius,
// solar distance and scale height, and a dust loading in the range Viking and MER measured, which
// is what makes the sky butterscotch and the sunsets blue. Nobody should cite them in a paper;
// they exist to show that the parameters reach that far and to give someone a place to start.
static constexpr AtmospherePreset kPresets[] = {
    {
        .name                = "Earth",
        .solarIrradiance     = {1.474F, 1.8504F, 1.91198F},
        .rayleighScattering  = {0.00580234F, 0.0135578F, 0.0331F},
        .rayleighScaleHeight = 8.0F,
        .mieScattering       = {0.06F, 0.06F, 0.06F},  // see Settings::atmoMieScattering
        .mieScaleHeight      = 1.2F,
        .mieAnisotropy       = 0.8F,
        .mieAlbedo           = 0.9F,
        .ozoneExtinction     = {0.000649717F, 0.0018809F, 8.50167e-05F},
        .ozoneCenter         = 25.0F,
        .ozoneWidth          = 30.0F,
        .groundAlbedo        = {0.1F, 0.1F, 0.1F},
        .bottomRadius        = 6360.0F,
        .thickness           = 60.0F,
        .sunAngularRadius    = 0.004675F,
    },
    {
        // Thin, dusty and further from the sun. The dust, not the air, is what you see: Rayleigh is
        // ~1/100 of Earth's because the surface pressure is, while the aerosol is an order of
        // magnitude stronger than Earth's and mixed through the whole column rather than hugging
        // the ground. No ozone layer.
        .name                = "Mars",
        .solarIrradiance     = {0.635F, 0.798F, 0.824F},  // Earth's, at 1.52 AU
        .rayleighScattering  = {5.75e-05F, 0.000134F, 0.000328F},
        .rayleighScaleHeight = 11.1F,
        .mieScattering       = {0.050F, 0.036F, 0.022F},
        .mieScaleHeight      = 11.0F,
        .mieAnisotropy       = 0.65F,
        .mieAlbedo           = 0.92F,
        .ozoneExtinction     = {0.0F, 0.0F, 0.0F},
        .ozoneCenter         = 25.0F,
        .ozoneWidth          = 30.0F,
        .groundAlbedo        = {0.25F, 0.14F, 0.08F},
        .bottomRadius        = 3389.5F,
        .thickness           = 100.0F,
        .sunAngularRadius    = 0.003068F,
    },
    {
        // A bigger world whose air scatters green rather than blue, under a high layer that absorbs
        // both ends of the spectrum. The sky lands green, which no amount of editing Earth will
        // reach -- that is the point of it being here. The absorber is an order of magnitude
        // stronger than Earth's ozone, because at Earth's loading it tints nothing.
        .name                = "Alien",
        .solarIrradiance     = {1.6F, 1.75F, 1.6F},
        .rayleighScattering  = {0.007F, 0.030F, 0.013F},
        .rayleighScaleHeight = 12.0F,
        .mieScattering       = {0.006F, 0.006F, 0.006F},
        .mieScaleHeight      = 2.0F,
        .mieAnisotropy       = 0.75F,
        .mieAlbedo           = 0.85F,
        .ozoneExtinction     = {0.012F, 0.0005F, 0.008F},
        .ozoneCenter         = 30.0F,
        .ozoneWidth          = 40.0F,
        .groundAlbedo        = {0.15F, 0.12F, 0.08F},
        .bottomRadius        = 8200.0F,
        .thickness           = 90.0F,
        .sunAngularRadius    = 0.006F,
    },
};
static_assert(std::size(kPresets) == eAtmospherePresetCount, "preset table and enum disagree");

std::span<const AtmospherePreset> atmospherePresets()
{
  return kPresets;
}

void applyAtmospherePreset(Settings& settings, int preset)
{
  if(preset < 0 || preset >= eAtmospherePresetCount)
    return;

  const AtmospherePreset& a        = kPresets[preset];
  settings.atmoSolarIrradiance     = a.solarIrradiance;
  settings.atmoRayleighScattering  = a.rayleighScattering;
  settings.atmoRayleighScaleHeight = a.rayleighScaleHeight;
  settings.atmoMieScattering       = a.mieScattering;
  settings.atmoMieScaleHeight      = a.mieScaleHeight;
  settings.atmoMieAnisotropy       = a.mieAnisotropy;
  settings.atmoMieAlbedo           = a.mieAlbedo;
  settings.atmoOzoneExtinction     = a.ozoneExtinction;
  settings.atmoOzoneCenter         = a.ozoneCenter;
  settings.atmoOzoneWidth          = a.ozoneWidth;
  settings.atmoGroundAlbedo        = a.groundAlbedo;
  settings.atmoBottomRadius        = a.bottomRadius;
  settings.atmoThickness           = a.thickness;
  settings.atmoSunAngularRadius    = a.sunAngularRadius;
}

int matchingAtmospherePreset(const Settings& settings)
{
  // Relative, because a glTF round trip returns these through decimal text and a preset that
  // survives a save/load should still read as that preset rather than silently becoming Custom.
  const auto almostEqual  = [](float a, float b) { return std::fabs(a - b) <= 1e-5F * std::max(1.0F, std::fabs(b)); };
  const auto almostEqual3 = [&](const glm::vec3& a, const glm::vec3& b) {
    return almostEqual(a.x, b.x) && almostEqual(a.y, b.y) && almostEqual(a.z, b.z);
  };

  for(int i = 0; i < eAtmospherePresetCount; ++i)
  {
    const AtmospherePreset& a = kPresets[i];
    if(almostEqual3(settings.atmoSolarIrradiance, a.solarIrradiance) && almostEqual3(settings.atmoRayleighScattering, a.rayleighScattering)
       && almostEqual(settings.atmoRayleighScaleHeight, a.rayleighScaleHeight)
       && almostEqual3(settings.atmoMieScattering, a.mieScattering) && almostEqual(settings.atmoMieScaleHeight, a.mieScaleHeight)
       && almostEqual(settings.atmoMieAnisotropy, a.mieAnisotropy) && almostEqual(settings.atmoMieAlbedo, a.mieAlbedo)
       && almostEqual3(settings.atmoOzoneExtinction, a.ozoneExtinction) && almostEqual(settings.atmoOzoneCenter, a.ozoneCenter)
       && almostEqual(settings.atmoOzoneWidth, a.ozoneWidth) && almostEqual3(settings.atmoGroundAlbedo, a.groundAlbedo)
       && almostEqual(settings.atmoBottomRadius, a.bottomRadius) && almostEqual(settings.atmoThickness, a.thickness)
       && almostEqual(settings.atmoSunAngularRadius, a.sunAngularRadius))
    {
      return i;
    }
  }
  return -1;
}

shaderio::SkyAtmosphereParameters skyBrunetonEarthDefaults()
{
  shaderio::SkyAtmosphereParameters p{};  // scalars already carry Earth values; profiles do not

  // Rayleigh: exponential falloff with an 8 km scale height. Layer 0 is the zero padding upstream
  // inserts, so the real profile is layer 1.
  p.rayleighDensity.layers[1].expTerm  = 1.0f;
  p.rayleighDensity.layers[1].expScale = -0.125f;  // -1/8 km

  // Mie: the same shape with a 1.2 km scale height.
  p.mieDensity.layers[1].expTerm  = 1.0f;
  p.mieDensity.layers[1].expScale = -0.833333333f;  // -1/1.2 km

  // Ozone: a tent centred on 25 km, rising linearly from 10 km and falling back to zero at 40 km.
  // Genuinely two layers, so neither is padding.
  p.absorptionDensity.layers[0].width        = 25.0f;
  p.absorptionDensity.layers[0].linearTerm   = 0.0666666667f;  // 1/15 km
  p.absorptionDensity.layers[0].constantTerm = -0.666666667f;
  p.absorptionDensity.layers[1].linearTerm   = -0.0666666667f;
  p.absorptionDensity.layers[1].constantTerm = 2.66666667f;

  return p;
}

void SkyBruneton::init(Resources& res, const Shaders& shaders)
{
  assert(!m_initialized && "init called twice");
  m_device     = res.allocator.getDevice();
  m_parameters = skyBrunetonEarthDefaults();

  const VkCommandPoolCreateInfo poolInfo{
      .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
      .queueFamilyIndex = res.app->getQueue(0).familyIndex,
  };
  NVVK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_transientCmdPool));
  NVVK_DBG_NAME(m_transientCmdPool);

  // Written by recordPrecompute() through the command buffer; see m_paramsBuffer.
  static_assert(sizeof(shaderio::SkyAtmosphereParameters) % 4 == 0 && sizeof(shaderio::SkyAtmosphereParameters) <= 65536,
                "vkCmdUpdateBuffer limits");
  NVVK_CHECK(res.allocator.createBuffer(m_paramsBuffer, sizeof(shaderio::SkyAtmosphereParameters),
                                        VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
  NVVK_DBG_NAME(m_paramsBuffer.buffer);

  createLuts(res);
  createPipelines(res, shaders);
  // Once, here: the LUT and scratch images keep their identity for the object's lifetime, so only
  // their contents ever change. Rewriting the sets per precompute -- which the preview path now
  // does inside a frame -- would race against a previously submitted one still reading them.
  writeDescriptors();

  m_initialized = true;
}

namespace {

// Every LUT is RGBA32F. Upstream offers a half-float variant to halve the footprint; this build
// keeps full float because the precompute is a one-off and the scattering LUTs store values whose
// magnitude varies by several orders across the table.
nvvk::Image createLut(Resources& res, VkExtent3D extent, const char* debugName)
{
  const bool is3D = extent.depth > 1;

  VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
  imageInfo.imageType         = is3D ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
  imageInfo.extent            = extent;
  imageInfo.format            = VK_FORMAT_R32G32B32A32_SFLOAT;
  imageInfo.mipLevels         = 1;
  imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

  VkImageViewCreateInfo viewInfo = DEFAULT_VkImageViewCreateInfo;
  viewInfo.viewType              = is3D ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;

  nvvk::Image image;
  NVVK_CHECK(res.allocator.createImage(image, imageInfo, viewInfo));
  nvvk::DebugUtil::getInstance().setObjectName(image.image, debugName);

  // CLAMP_TO_EDGE on every axis. The LUT mappings already put f(0) and f(1) at the first and last
  // texel centres, so a sample outside is an out-of-domain query that should saturate rather than
  // wrap -- a repeat would join the zenith to the horizon, or one nu slab to the next.
  const VkSamplerCreateInfo samplerInfo{
      .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter    = VK_FILTER_LINEAR,
      .minFilter    = VK_FILTER_LINEAR,
      .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
  };
  NVVK_CHECK(res.samplerPool.acquireSampler(image.descriptor.sampler, samplerInfo));

  // GENERAL for life: each LUT is written as a storage image by one pass and sampled by later
  // ones, and that is the only layout legal for both. Declared here rather than after the first
  // barrier because descriptor writes validate against this field and happen earlier.
  image.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  return image;
}

constexpr VkExtent3D kTransmittanceExtent{TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT, 1};
constexpr VkExtent3D kIrradianceExtent{IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT, 1};
constexpr VkExtent3D kScatteringExtent{SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH};

}  // namespace

void SkyBruneton::createLuts(Resources& res)
{
  m_transmittance = createLut(res, kTransmittanceExtent, "SkyBruneton::transmittance");
  m_irradiance    = createLut(res, kIrradianceExtent, "SkyBruneton::irradiance");
  m_scattering    = createLut(res, kScatteringExtent, "SkyBruneton::scattering");
  m_singleMie     = createLut(res, kScatteringExtent, "SkyBruneton::singleMie");

  m_deltaIrradiance        = createLut(res, kIrradianceExtent, "SkyBruneton::deltaIrradiance");
  m_deltaRayleigh          = createLut(res, kScatteringExtent, "SkyBruneton::deltaRayleigh");
  m_deltaMie               = createLut(res, kScatteringExtent, "SkyBruneton::deltaMie");
  m_deltaScatteringDensity = createLut(res, kScatteringExtent, "SkyBruneton::deltaScatteringDensity");
  m_deltaMultiple          = createLut(res, kScatteringExtent, "SkyBruneton::deltaMultiple");
}

void SkyBruneton::createPipelines(Resources& res, const Shaders& shaders)
{
  // Runtime set: the four finished LUTs.
  {
    nvvk::DescriptorBindings bindings;
    for(uint32_t i = 0; i < 4; ++i)
      bindings.addBinding(i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    // The parameters travel with the LUTs: every evaluation function takes an `atmosphere`
    // argument, so a consumer that binds the textures without them cannot call anything.
    bindings.addBinding(4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    NVVK_CHECK(m_runtimePack.init(bindings, m_device, 1));
    NVVK_DBG_NAME(m_runtimePack.getLayout());
  }

  // Precompute set: one layout for all six passes, so every binding exists for every dispatch even
  // where a pass ignores it. Simpler to reason about than six layouts differing by two entries.
  {
    using B = shaderio::SkyBrunetonPrecomputeBindings;
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(B::eBrunetonParams, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    for(uint32_t b = B::eOutTransmittance; b <= B::eOutDeltaMultiple; ++b)
      bindings.addBinding(b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    for(uint32_t b = B::eInTransmittance; b <= B::eInScatteringDensity; ++b)
      bindings.addBinding(b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(m_precomputePack.init(bindings, m_device, 1));
    NVVK_DBG_NAME(m_precomputePack.getLayout());
  }

  const VkPushConstantRange pushRange{
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset     = 0,
      .size       = sizeof(shaderio::SkyBrunetonPushConstant),
  };
  NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_precomputeLayout, {m_precomputePack.getLayout()}, {pushRange}));
  NVVK_DBG_NAME(m_precomputeLayout);

  // One module, six entrypoints, in ePass order. Slang names them exactly as written in
  // shaders/sky_bruneton_precompute.slang.
  static constexpr std::array<const char*, ePassCount> kPrecomputeEntryPoints{
      "Transmittance",     "DirectIrradiance",   "SingleScattering",
      "ScatteringDensity", "IndirectIrradiance", "MultipleScattering",
  };

  VkShaderModuleCreateInfo moduleInfo{
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = shaders.precompute.size_bytes(),
      .pCode    = shaders.precompute.data(),
  };

  for(uint32_t i = 0; i < ePassCount; ++i)
  {
    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.pName = kPrecomputeEntryPoints[i];
    stageInfo.pNext = &moduleInfo;

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.layout = m_precomputeLayout;
    pipelineInfo.stage  = stageInfo;
    NVVK_CHECK(vkCreateComputePipelines(m_device, {}, 1, &pipelineInfo, nullptr, &m_pipelines[i]));
  }
}

void SkyBruneton::writeDescriptors()
{
  using B = shaderio::SkyBrunetonPrecomputeBindings;
  nvvk::WriteSetContainer writes;
  writes.append(m_precomputePack.makeWrite(B::eBrunetonParams), m_paramsBuffer);

  writes.append(m_precomputePack.makeWrite(B::eOutTransmittance), m_transmittance);
  writes.append(m_precomputePack.makeWrite(B::eOutDeltaIrradiance), m_deltaIrradiance);
  writes.append(m_precomputePack.makeWrite(B::eOutIrradiance), m_irradiance);
  writes.append(m_precomputePack.makeWrite(B::eOutDeltaRayleigh), m_deltaRayleigh);
  writes.append(m_precomputePack.makeWrite(B::eOutDeltaMie), m_deltaMie);
  writes.append(m_precomputePack.makeWrite(B::eOutScattering), m_scattering);
  writes.append(m_precomputePack.makeWrite(B::eOutSingleMie), m_singleMie);
  writes.append(m_precomputePack.makeWrite(B::eOutScatteringDensity), m_deltaScatteringDensity);
  writes.append(m_precomputePack.makeWrite(B::eOutDeltaMultiple), m_deltaMultiple);

  writes.append(m_precomputePack.makeWrite(B::eInTransmittance), m_transmittance);
  writes.append(m_precomputePack.makeWrite(B::eInDeltaIrradiance), m_deltaIrradiance);
  writes.append(m_precomputePack.makeWrite(B::eInDeltaRayleigh), m_deltaRayleigh);
  writes.append(m_precomputePack.makeWrite(B::eInDeltaMie), m_deltaMie);
  writes.append(m_precomputePack.makeWrite(B::eInDeltaMultiple), m_deltaMultiple);
  writes.append(m_precomputePack.makeWrite(B::eInScatteringDensity), m_deltaScatteringDensity);
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  nvvk::WriteSetContainer runtimeWrites;
  runtimeWrites.append(m_runtimePack.makeWrite(0), m_transmittance);
  runtimeWrites.append(m_runtimePack.makeWrite(1), m_scattering);
  runtimeWrites.append(m_runtimePack.makeWrite(2), m_singleMie);
  runtimeWrites.append(m_runtimePack.makeWrite(3), m_irradiance);
  runtimeWrites.append(m_runtimePack.makeWrite(4), m_paramsBuffer);
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(runtimeWrites.size()), runtimeWrites.data(), 0, nullptr);
}

// Every pass reads what the previous one wrote, so they serialise on a full compute barrier. There
// is no parallelism to recover here: the dependency chain is the algorithm.
void SkyBruneton::computeBarrier(VkCommandBuffer cmd) const
{
  const VkMemoryBarrier memoryBarrier{
      .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
  };
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &memoryBarrier, 0, nullptr, 0, nullptr);
}

// The 3D LUTs are filled a slice at a time, matching upstream's layer-by-layer rendering: the
// scattering parameterisation folds nu and mu_s into the width, so a slice is one radius shell.
void SkyBruneton::dispatch3DLayers(VkCommandBuffer cmd, VkPipeline pipeline, int scatteringOrder, int densitySamples)
{
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  const uint32_t groupsX = (SCATTERING_TEXTURE_WIDTH + SKY_BRUNETON_WORKGROUP_SIZE - 1) / SKY_BRUNETON_WORKGROUP_SIZE;
  const uint32_t groupsY = (SCATTERING_TEXTURE_HEIGHT + SKY_BRUNETON_WORKGROUP_SIZE - 1) / SKY_BRUNETON_WORKGROUP_SIZE;
  for(int layer = 0; layer < SCATTERING_TEXTURE_DEPTH; ++layer)
  {
    const shaderio::SkyBrunetonPushConstant push{.layer = layer, .scatteringOrder = scatteringOrder, .densitySamples = densitySamples};
    vkCmdPushConstants(cmd, m_precomputeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
  }
}

void SkyBruneton::setParameters(const shaderio::SkyAtmosphereParameters& parameters)
{
  m_parameters = parameters;
}

//--------------------------------------------------------------------------------------------------
// Record a full rebuild of the LUTs. No submit, no wait -- the caller owns the command buffer.
//
void SkyBruneton::recordPrecompute(VkCommandBuffer cmd, Quality quality)
{
  assert(m_initialized && "init must be called first");

  // The parameters go in through the command buffer: earlier frames may still be reading the
  // buffer through the runtime set, and the first barrier waits for them.
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  vkCmdUpdateBuffer(cmd, m_paramsBuffer.buffer, 0, sizeof(m_parameters), &m_parameters);
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_UNIFORM_READ_BIT);

  // UNDEFINED is a legal discarding source and every precompute rewrites each LUT in full.
  for(const nvvk::Image* image : {&m_transmittance, &m_irradiance, &m_scattering, &m_singleMie, &m_deltaIrradiance,
                                  &m_deltaRayleigh, &m_deltaMie, &m_deltaScatteringDensity, &m_deltaMultiple})
  {
    nvvk::cmdImageMemoryBarrier(cmd, {image->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});
  }

  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_precomputeLayout, 0, 1, m_precomputePack.getSetPtr(), 0, nullptr);

  const auto dispatch2D = [&](Pass pass, uint32_t width, uint32_t height, int order) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[pass]);
    const shaderio::SkyBrunetonPushConstant push{.layer = 0, .scatteringOrder = order, .densitySamples = quality.densitySamples};
    vkCmdPushConstants(cmd, m_precomputeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (width + SKY_BRUNETON_WORKGROUP_SIZE - 1) / SKY_BRUNETON_WORKGROUP_SIZE,
                  (height + SKY_BRUNETON_WORKGROUP_SIZE - 1) / SKY_BRUNETON_WORKGROUP_SIZE, 1);
  };

  // The order below is upstream's, and it is a dependency chain rather than a preference.
  mark(cmd, nullptr);

  dispatch2D(eTransmittance, TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT, 0);
  computeBarrier(cmd);
  mark(cmd, "transmittance");

  dispatch2D(eDirectIrradiance, IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT, 0);
  computeBarrier(cmd);
  mark(cmd, "directIrradiance");

  dispatch3DLayers(cmd, m_pipelines[eSingleScattering], 1, quality.densitySamples);
  computeBarrier(cmd);
  mark(cmd, "singleScattering");

  // Each further order needs the previous one complete: density, then the ground irradiance it
  // implies, then the integration along the view ray.
  for(int order = 2; order <= quality.scatteringOrders; ++order)
  {
    dispatch3DLayers(cmd, m_pipelines[eScatteringDensity], order, quality.densitySamples);
    computeBarrier(cmd);
    mark(cmd, "scatteringDensity");

    dispatch2D(eIndirectIrradiance, IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT, order - 1);
    computeBarrier(cmd);
    mark(cmd, "indirectIrradiance");

    dispatch3DLayers(cmd, m_pipelines[eMultipleScattering], order, quality.densitySamples);
    computeBarrier(cmd);
    mark(cmd, "multipleScattering");
  }

  // Recorded, therefore owed: by the time anything sampling these images through this command
  // buffer runs, they hold exactly this.
  m_lutParameters = m_parameters;
  m_lutQuality    = quality;
}

bool SkyBruneton::lutsSatisfy(const shaderio::SkyAtmosphereParameters& parameters, Quality quality) const
{
  return m_lutQuality.densitySamples >= quality.densitySamples && m_lutQuality.scatteringOrders >= quality.scatteringOrders
         && std::memcmp(&parameters, &m_lutParameters, sizeof(parameters)) == 0;
}

//--------------------------------------------------------------------------------------------------
// Rebuild the LUTs and wait for them. For init, and for the commit that follows a drag.
//
void SkyBruneton::precompute(Resources& res, Quality quality)
{
  assert(m_initialized && "init must be called first");

  // Per-pass GPU timing, on the blocking path only.
  //
  // The passes serialise on a full barrier, so a single timestamp after each one turns the chain
  // into a set of durations -- no bracketing needed. The wall-clock timer below cannot do this:
  // it also contains the queue wait, and it cannot see inside the chain. The preview path leaves
  // m_timingPool null and pays none of it.
  const VkQueryPoolCreateInfo queryInfo{.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                        .queryType  = VK_QUERY_TYPE_TIMESTAMP,
                                        .queryCount = kMaxTimestamps};
  NVVK_CHECK(vkCreateQueryPool(m_device, &queryInfo, nullptr, &m_timingPool));
  m_timingCount = 0;

  // Scoped so the timer closes its log line before reportPassTimings writes one of its own.
  {
    nvutils::ScopedTimer st("SkyBruneton::precompute");

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, m_transientCmdPool));
    {
      NVVK_DBG_SCOPE(cmd);  // closes before endSingleTimeCommands frees the buffer
      vkCmdResetQueryPool(cmd, m_timingPool, 0, kMaxTimestamps);
      recordPrecompute(cmd, quality);
    }
    nvvk::endSingleTimeCommands(cmd, m_device, m_transientCmdPool, res.app->getQueue(0).queue);
  }

  reportPassTimings(res);
  vkDestroyQueryPool(m_device, m_timingPool, nullptr);
  m_timingPool = VK_NULL_HANDLE;
}

// `label` names the pass that ended here; the first mark opens the chain and names nothing.
void SkyBruneton::mark(VkCommandBuffer cmd, const char* label)
{
  if(m_timingPool == VK_NULL_HANDLE || m_timingCount >= kMaxTimestamps)
    return;
  m_timingLabels[m_timingCount] = label;
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_timingPool, m_timingCount);
  ++m_timingCount;
}

// Fold the timestamp chain into one line: total GPU time, and what each kind of pass took.
// Repeated passes are summed and counted, since what matters is "scattering density costs X
// across its three orders", not what each individual order cost.
void SkyBruneton::reportPassTimings(Resources& res) const
{
  if(m_timingCount < 2)
    return;

  std::array<uint64_t, kMaxTimestamps> stamps{};
  if(vkGetQueryPoolResults(m_device, m_timingPool, 0, m_timingCount, sizeof(uint64_t) * m_timingCount, stamps.data(),
                           sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
     != VK_SUCCESS)
    return;

  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(res.app->getPhysicalDevice(), &props);
  if(props.limits.timestampPeriod == 0.0F)
    return;
  const double toMs = double(props.limits.timestampPeriod) * 1e-6;

  // Distinct labels in the order they first appear, so the log reads in execution order.
  std::vector<const char*> names;
  std::vector<double>      totals;
  std::vector<int>         counts;
  for(uint32_t i = 1; i < m_timingCount; ++i)
  {
    const double ms  = double(stamps[i] - stamps[i - 1]) * toMs;
    size_t       idx = 0;
    for(; idx < names.size(); ++idx)
      if(names[idx] == m_timingLabels[i])
        break;
    if(idx == names.size())
    {
      names.push_back(m_timingLabels[i]);
      totals.push_back(0.0);
      counts.push_back(0);
    }
    totals[idx] += ms;
    counts[idx] += 1;
  }

  std::string line;
  for(size_t i = 0; i < names.size(); ++i)
  {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s%s %.2f ms", i ? ", " : "", names[i], totals[i]);
    line += buf;
    if(counts[i] > 1)
      line += " (x" + std::to_string(counts[i]) + ")";
  }
  const double total = double(stamps[m_timingCount - 1] - stamps[0]) * toMs;
  LOGI("SkyBruneton::precompute GPU %.2f ms -- %s\n", total, line.c_str());
}

void SkyBruneton::destroyScratch(Resources& res)
{
  for(nvvk::Image* image : {&m_deltaIrradiance, &m_deltaRayleigh, &m_deltaMie, &m_deltaScatteringDensity, &m_deltaMultiple})
  {
    if(image->descriptor.sampler != VK_NULL_HANDLE)
      res.samplerPool.releaseSampler(image->descriptor.sampler);
    res.allocator.destroyImage(*image);
    *image = {};
  }
}

void SkyBruneton::destroyResources(Resources& res)
{
  destroyScratch(res);
  for(nvvk::Image* image : {&m_transmittance, &m_irradiance, &m_scattering, &m_singleMie})
  {
    if(image->descriptor.sampler != VK_NULL_HANDLE)
      res.samplerPool.releaseSampler(image->descriptor.sampler);
    res.allocator.destroyImage(*image);
    *image = {};
  }
  res.allocator.destroyBuffer(m_paramsBuffer);
  m_paramsBuffer = {};
}

void SkyBruneton::deinit(Resources& res)
{
  if(!m_initialized)
    return;

  destroyResources(res);
  for(VkPipeline& pipeline : m_pipelines)
  {
    vkDestroyPipeline(m_device, pipeline, nullptr);
    pipeline = VK_NULL_HANDLE;
  }
  vkDestroyPipelineLayout(m_device, m_precomputeLayout, nullptr);
  m_runtimePack.deinit();
  m_precomputePack.deinit();
  vkDestroyCommandPool(m_device, m_transientCmdPool, nullptr);

  m_precomputeLayout = VK_NULL_HANDLE;
  m_transientCmdPool = VK_NULL_HANDLE;
  m_device           = VK_NULL_HANDLE;
  m_initialized      = false;
}
