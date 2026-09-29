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

#include <array>
#include <cassert>
#include <filesystem>
#include <span>

#include <nvvk/descriptors.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>

#include "shaders/shaderio.h"

struct Resources;
struct Settings;

// A whole atmosphere, named. These are the values `atmoPreset` writes into the settings.
//
// Data rather than code because the set is open: adding a world should mean adding a row, and
// every field a preset carries is one the user can then edit individually.
struct AtmospherePreset
{
  const char* name;
  glm::vec3   solarIrradiance;      // W/m^2 at the top of the atmosphere
  glm::vec3   rayleighScattering;   // 1/km at the surface
  float       rayleighScaleHeight;  // km
  glm::vec3   mieScattering;        // 1/km at the surface
  float       mieScaleHeight;       // km
  float       mieAnisotropy;
  float       mieAlbedo;
  glm::vec3   ozoneExtinction;  // 1/km at the peak of the tent
  float       ozoneCenter;      // km
  float       ozoneWidth;       // km
  glm::vec3   groundAlbedo;
  float       bottomRadius;      // km
  float       thickness;         // km
  float       sunAngularRadius;  // radians
};

// The built-in worlds, indexed by AtmospherePresetIndex.
std::span<const AtmospherePreset> atmospherePresets();

enum AtmospherePresetIndex
{
  eAtmosphereEarth = 0,
  eAtmosphereMars,
  eAtmosphereAlien,
  eAtmospherePresetCount
};

// Copy a preset into the settings. Out-of-range indices are ignored rather than clamped: a preset
// is an action, and applying the wrong world is worse than applying none.
void applyAtmospherePreset(Settings& settings, int preset);

// Which preset the settings currently match exactly, or -1 for "none of them" -- what the UI shows
// as Custom. Compared by value so that editing a slider, loading a scene, restoring an .ini and an
// MCP write all reach the same answer without any of them having to announce it.
[[nodiscard]] int matchingAtmospherePreset(const Settings& settings);

// Earth, with the density profiles filled in.
//
// `shaderio::SkyAtmosphereParameters{}` alone is not usable: its three DensityProfile members are
// arrays of structs, which cannot carry an in-struct default both the host and Slang agree on, so
// a value-initialised block describes an atmosphere with no air in it and renders black. Everything
// that needs parameters starts from here.
//
// The layer values are upstream's, converted from 1/m to 1/km. Note the ordering: a profile that
// upstream declares with one layer is padded by inserting an all-zero layer *before* it, so the
// real layer is index 1 and the zero-width layer 0 never wins the `altitude < layers[0].width`
// test. Ozone is the exception -- it genuinely has two layers, a linear ramp up to 25 km and a
// linear ramp back down above it.
shaderio::SkyAtmosphereParameters skyBrunetonEarthDefaults();

//--------------------------------------------------------------------------------------------------
// SkyBruneton -- the precomputed atmospheric scattering LUTs and the passes that build them.
//
// Owns its Vulkan resources and hands evaluation a descriptor set, the same shape nvvk::HdrIbl and
// nvshaders::HdrEnvDome use in this subsystem (`hdrDome.updateEnvironment(cmd,
// hdrIbl.getDescriptorSet())`). That is deliberate: it keeps the class liftable into nvpro_core2
// later -- SPIR-V spans in, LUTs out, no knowledge of this renderer -- if another sample ever wants
// Bruneton. Nothing here touches Resources beyond the allocator, sampler pool and queue.
//
// The LUTs depend only on the atmosphere parameters, never on the sun or the camera, so they are
// built once at init and rebuilt only when someone edits the atmosphere. Sun movement costs
// nothing, which is what makes a time-of-day scrub cheap.
//
class SkyBruneton
{
public:
  SkyBruneton() = default;
  ~SkyBruneton() { assert(!m_initialized && "deinit must be called first"); }

  SkyBruneton(const SkyBruneton&)            = delete;
  SkyBruneton& operator=(const SkyBruneton&) = delete;

  // The six precompute passes live in one Slang module as six entrypoints, so there is one blob
  // and the pass is chosen by entrypoint name -- see kPrecomputeEntryPoints in the .cpp.
  struct Shaders
  {
    std::span<const uint32_t> precompute;
  };

  void init(Resources& res, const Shaders& shaders);
  void deinit(Resources& res);

  bool isInitialized() const { return m_initialized; }

  // How thoroughly to build the LUTs.
  //
  // Only the scattering-density integral and the number of orders are worth varying: measured on
  // an RTX PRO 4500 Blackwell, density is most of the whole precomputation and its cost is quadratic in the
  // sample count, so halving that quarters the bill. The LUT *dimensions* deliberately are not
  // here -- they are baked into the parameterisation the sampling functions invert, so changing
  // them at runtime would mean recompiling the shaders, not rebinding a texture.
  struct Quality
  {
    int densitySamples;    // zenith samples in the density integral; azimuth is twice this
    int scatteringOrders;  // bounces accumulated, including the first

    bool operator==(const Quality&) const = default;
  };

  // The reference build, and upstream's numbers. ~34 ms.
  static constexpr Quality kQualityFinal{16, 4};
  // What a drag can afford. ~6 ms: a sixteenth of the directions (a quarter per axis) and two
  // bounces instead of four.
  // Visibly softer in the multiple-scattering term at twilight, indistinguishable by day, and
  // replaced by kQualityFinal the moment the drag stops.
  static constexpr Quality kQualityPreview{4, 2};

  // Replace the atmosphere. Does not rebuild anything -- call precompute() for that, which is the
  // expensive half and should not happen implicitly from a setter.
  void                                     setParameters(const shaderio::SkyAtmosphereParameters& parameters);
  const shaderio::SkyAtmosphereParameters& getParameters() const { return m_parameters; }

  // Build every LUT from the current parameters. Submits and waits: this runs at init and on an
  // atmosphere edit, never per frame.
  void precompute(Resources& res, Quality quality = kQualityFinal);

  // Record a rebuild into a command buffer the caller will submit, and do not wait.
  //
  // For the preview path, which runs inside the frame. Everything it writes -- the parameter
  // buffer and the LUTs -- is written by the GPU behind barriers that wait for earlier frames, so
  // nothing on the host touches a resource a frame in flight may still be reading.
  void recordPrecompute(VkCommandBuffer cmd, Quality quality);

  // Whether the LUTs already hold this atmosphere at this quality or better. Lets both the preview
  // and the commit skip work without either having to track what the other did -- including the
  // case that matters most, a settle whose parameters are unchanged but which still owes the
  // upgrade from preview quality to final.
  [[nodiscard]] bool lutsSatisfy(const shaderio::SkyAtmosphereParameters& parameters, Quality quality) const;

  // The finished LUTs, for evaluation to bind.
  VkDescriptorSetLayout getDescriptorSetLayout() const { return m_runtimePack.getLayout(); }
  VkDescriptorSet       getDescriptorSet() const { return m_runtimePack.getSet(0); }

private:
  void createLuts(Resources& res);
  void createPipelines(Resources& res, const Shaders& shaders);
  void destroyResources(Resources& res);
  void destroyScratch(Resources& res);
  void writeDescriptors();
  void dispatch3DLayers(VkCommandBuffer cmd, VkPipeline pipeline, int scatteringOrder, int densitySamples);
  void computeBarrier(VkCommandBuffer cmd) const;
  void mark(VkCommandBuffer cmd, const char* label);
  void reportPassTimings(Resources& res) const;

  // Per-pass GPU timing, active only while the blocking precompute() is running.
  static constexpr uint32_t               kMaxTimestamps = 32;
  VkQueryPool                             m_timingPool{VK_NULL_HANDLE};
  std::array<const char*, kMaxTimestamps> m_timingLabels{};
  uint32_t                                m_timingCount{0};

  bool     m_initialized{false};
  VkDevice m_device{VK_NULL_HANDLE};

  shaderio::SkyAtmosphereParameters m_parameters{};

  // What the LUTs actually contain, as opposed to what has merely been asked for.
  shaderio::SkyAtmosphereParameters m_lutParameters{};
  Quality                           m_lutQuality{0, 0};

  // Final LUTs, sampled by evaluation.
  nvvk::Image m_transmittance;  // 2D, (radius, view-zenith cosine)
  nvvk::Image m_irradiance;     // 2D, (radius, sun-zenith cosine), summed over orders
  nvvk::Image m_scattering;     // 3D, Rayleigh plus every order above the first
  nvvk::Image m_singleMie;      // 3D, kept separate so its phase function stays exact at render time

  // Scratch for the precomputation: each order is built from the previous one. Kept for the
  // object's lifetime rather than freed after the first build -- the atmosphere is editable, so
  // every rebuild needs them again, and a preview cannot afford to reallocate four 3D images.
  nvvk::Image m_deltaIrradiance;
  nvvk::Image m_deltaRayleigh;
  nvvk::Image m_deltaMie;
  nvvk::Image m_deltaScatteringDensity;
  nvvk::Image m_deltaMultiple;

  // Device-local, written only by the vkCmdUpdateBuffer at the top of recordPrecompute(). The
  // runtime set binds it too, so a host write would race frames still in flight; recording it
  // orders it after them. Its contents are therefore always m_lutParameters.
  nvvk::Buffer m_paramsBuffer;

  // Two sets, as the class comment describes: the runtime one outlives precompute, the precompute
  // one binds storage images that only exist while the LUTs are being written.
  nvvk::DescriptorPack m_runtimePack;
  nvvk::DescriptorPack m_precomputePack;

  VkPipelineLayout m_precomputeLayout{VK_NULL_HANDLE};
  // Indexed by Pass below; one pipeline per precompute shader.
  enum Pass : uint32_t
  {
    eTransmittance = 0,
    eDirectIrradiance,
    eSingleScattering,
    eScatteringDensity,
    eIndirectIrradiance,
    eMultipleScattering,
    ePassCount
  };
  std::array<VkPipeline, ePassCount> m_pipelines{};
  VkCommandPool                      m_transientCmdPool{VK_NULL_HANDLE};
};
