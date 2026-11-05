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

#include <functional>
#include <vector>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include "resources.hpp"
#include "settings_registry.hpp"

//-------------------------------------------------------------------------------------------------------
// Caustics -- beta. The host side of the path tracer's light tracing and photon map
// (docs/caustics.md, shaders/caustics.h.slang).
//
// Photons leave the emitters and follow the smooth casters. The ones that land splat the receiver into the
// camera when it sees it directly, and are kept in a photon map for the camera paths that reach their
// receivers through casters or after a bounce. This class owns what that needs apart from the shader: the settings,
// the spheres around clusters of the casters the photons aim at, the buffers the photon passes write and the path tracer reads,
// and the frame's grid. The PathTracer owns the shader variant (USE_CAUSTICS follows enabled()) and the
// photon passes' pipelines.
//
//   if(caustics.enabled())                                    // the shader is the caustics variant
//     push.caustics = caustics.cmdPrepare(cmd, resources, renderSize, canTrace);  // null: nothing to trace
//   ... PathTracer dispatches photonGroups() groups of the photon entry point, then the grid's counting sort
//       (mapGroups() and cellGroups()), then the paths ...
//
// The map is this frame's photons and nothing else: the paths gather them in a radius that shrinks as the
// accumulation grows, and the average of the frames does the rest. Nothing outlives the frame, so a changed
// scene, material or light needs no cache to invalidate, whatever the number of nodes.
//-------------------------------------------------------------------------------------------------------
class Caustics
{
public:
  // `restart` starts a new accumulation: a setting changed from the command line or a script has no UI to do it
  void registerParameters(SettingsRegistry* settings, std::function<void()> restart);
  void deinit(Resources& resources);

  // The settings and their controls. Returns true when something changed that restarts the accumulation.
  bool enabled() const { return m_enabled; }
  bool onUiSettings();
  // Calls callback(storage) for each setting the panel shows now (a hidden one is left out), for the Settings
  // panel's changed-from-default dot and reset
  void forEachShownSetting(const std::function<void(const void*)>& callback) const
  {
    callback(&m_enabled);
    if(!m_enabled)
      return;
    callback(&m_autoPhotons);
    if(!m_autoPhotons)
      callback(&m_photonsPerFrame);
    callback(&m_gather);
    if(m_gather)
      callback(&m_gatherPixels);
  }

  // Prepare this frame's photon passes: bounds and their clusters, buffers (cleared), the gather radius, parameters. Returns
  // what the shaders read through PathtracePushConstant::caustics, or null when there are no casters to
  // trace, or `canTrace` is false (the live shader has no photon entry points yet).
  shaderio::CausticParams* cmdPrepare(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderSize, bool canTrace);

  // Workgroups of the photon entry point for this frame (64 photons each); of the grid's passes over the
  // photons (a slot each) and over its hash cells (256 each): none without a map
  uint32_t photonGroups() const { return (m_photonCount + kGroupSize - 1) / kGroupSize; }
  // Builds the map's grid over the photons the photon pass wrote (the counting sort of caustics_grid.slang),
  // recorded after it, and the barrier that orders them. The passes' pipelines belong to this class and are built
  // once: they do not depend on the scene, so a new path tracer shader leaves them alone.
  void cmdBuildGrid(VkCommandBuffer cmd);

  uint32_t mapGroups() const { return (m_mapPhotons + kGroupSize - 1) / kGroupSize; }
  uint32_t cellGroups() const { return m_cells / kScanGroupSize; }

  static constexpr uint32_t kGroupSize     = 64;   // numthreads of causticMain and the grid's passes over the photons
  static constexpr uint32_t kScanGroupSize = 256;  // numthreads of the grid's passes over the hash cells
  static constexpr uint32_t kMaxPhotons    = 65535u * kGroupSize;  // One dispatch row: the guaranteed group count

private:
  struct Sphere
  {
    glm::vec3 center{0.0F};
    float     radius = 0.0F;  // 0: empty
  };

  // A node whose material may be a smooth caster, the box of its primitive and the cluster it belongs to
  struct CasterNode
  {
    int       renderNode = 0;
    bool      wholeScene = false;  // Skinned or morphed: the bind-pose box says nothing
    uint32_t  cluster    = 0;
    glm::vec3 lo{0.0F};
    glm::vec3 hi{0.0F};
  };

  // The casters' bounding sphere and the spheres of their clusters, up to date with the scene. The list of nodes
  // that may be casters, and their clusters, are made when the graph or the materials change; a node that only
  // moves costs its own box.
  void updateBounds(Resources& resources);
  void createGridPipelines(VkDevice device);
  void assignClusters(const std::vector<nvvkgltf::RenderNode>& renderNodes);
  void createMap(Resources& resources);

  bool  m_enabled         = false;    // ptCaustics
  bool  m_autoPhotons     = true;     // ptCausticPhotonsAuto: a quarter of the pixels
  int   m_photonsPerFrame = 1 << 18;  // ptCausticPhotonsPerFrame, when not automatic
  bool  m_gather          = true;     // ptCausticGather: keep a photon map, for the receivers behind casters
  float m_gatherPixels    = 4.0F;     // ptCausticGatherRadius: the gather's first radius, in pixels; it shrinks

  Sphere                  m_casters;   // Of the nodes whose material may be a smooth caster
  std::vector<Sphere>     m_clusters;  // Of clusters of them, which the photons aim at (as many as CAUSTIC_MAX_CASTERS)
  uint32_t                m_clusterCount = 0;  // Clusters of m_casterNodes
  Sphere                  m_scene;
  std::vector<CasterNode> m_casterNodes;
  uint64_t                m_structureKey[4]{};  // What the list was made for; see updateBounds()
  uint64_t                m_transformKey = ~0ull;
  const void*             m_boundsScene  = nullptr;

  static constexpr uint32_t kGridPasses = 5;  // Count, scan blocks, scan sums, add offsets, scatter
  VkPipelineLayout          m_gridLayout{};
  VkPipeline                m_gridPipelines[kGridPasses]{};

  nvvk::Buffer m_params{};  // shaderio::CausticParams
  nvvk::Buffer m_splats{};  // 3 uints (float bits) per pixel of the largest render size so far
  uint32_t     m_photonCount = 0;

  // The photon map: one slot per photon of the frame, then the grid over the ones that landed
  nvvk::Buffer m_photons{};       // shaderio::CausticPhoton per slot
  nvvk::Buffer m_sorted{};        // The photons that landed, sorted by hash cell
  nvvk::Buffer m_cellStarts{};    // The grid: where each hash cell's photons start in m_sorted
  uint32_t     m_cells      = 0;  // Hash cells this frame
  uint32_t     m_mapPhotons = 0;  // Photons the map keeps this frame
  uint32_t     m_emitted = 0;  // Photons traced so far; wraps. Numbers the photons, so a frame never repeats another's

  static constexpr uint32_t kMinAutoPhotons  = 1u << 16;  // Of a small render
  static constexpr uint32_t kMapSlots        = 1u << 20;  // Photons of a frame the map keeps
  static constexpr uint32_t kMaxCells        = 1u << 21;  // Hash cells of the grid; a power of two
  static constexpr uint32_t kGatherMax       = 512;       // Photons a gather sums at most (its radius shrinks to fit)
  static constexpr float    kCellRadii       = 4.0F;      // Cells of the grid, in gather radii
  static constexpr float    kMinGatherPixels = 1.0F;      // Where the shrinking radius stops: a pixel
  static constexpr float kWorldRadius = 0.05F;  // The first radius at most, in casters' radii; shrinks like the other
};
