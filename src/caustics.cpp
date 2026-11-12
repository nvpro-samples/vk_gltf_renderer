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


#include "caustics.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <unordered_map>

#include <nvgui/property_editor.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/debug_util.hpp>

#include "tinygltf_utils.hpp"
#include "_autogen/caustics_grid.slang.h"

static_assert(sizeof(shaderio::CausticPhoton) == 20, "the photon map's slot");

void Caustics::registerParameters(SettingsRegistry* settings, std::function<void()> restart)
{
  const auto restarts = [restart](const nvutils::ParameterBase* const) {
    if(restart)
      restart();
  };
  settings->add({.name = "ptCaustics",
                 .help = "PathTracer: Beta caustics by light tracing and photon mapping (off by default; see docs/caustics.md)",
                 .callbackSuccess = restarts},
                &m_enabled, Persist::eYes);
  settings->add({.name = "ptCausticPhotonsAuto",
                 .help = "PathTracer: Caustic photons per frame follow the render size (a quarter of the pixels)",
                 .callbackSuccess = restarts},
                &m_autoPhotons, Persist::eYes);
  settings->add({.name            = "ptCausticPhotonsPerFrame",
                 .help            = "PathTracer: Caustic photons traced per frame, when not automatic",
                 .callbackSuccess = restarts},
                &m_photonsPerFrame, Persist::eYes, 256, int(kMaxPhotons));
  settings->add({.name = "ptCausticGather",
                 .help = "PathTracer: Caustics keep a photon map, which covers receivers seen through glass or in a mirror "
                         "(a pool's floor) or after a bounce",
                 .callbackSuccess = restarts},
                &m_gather, Persist::eYes);
  settings->add({.name = "ptCausticGatherRadius",
                 .help = "PathTracer: First radius of the photon map's gather, in pixels; it shrinks as the accumulation grows "
                         "(more: smoother, blurrier at first)",
                 .callbackSuccess = restarts},
                &m_gatherPixels, Persist::eYes, 1.0F, 64.0F);
}

void Caustics::deinit(Resources& resources)
{
  const VkDevice device = resources.allocator.getDevice();
  for(size_t i = 0; i < kGridPasses; i++)
  {
    vkDestroyPipeline(device, m_gridPipelines[i], nullptr);
    m_gridPipelines[i] = {};
  }
  vkDestroyPipelineLayout(device, m_gridLayout, nullptr);
  m_gridLayout = {};
  resources.allocator.destroyBuffer(m_params);
  resources.allocator.destroyBuffer(m_splats);
  resources.allocator.destroyBuffer(m_photons);
  resources.allocator.destroyBuffer(m_sorted);
  resources.allocator.destroyBuffer(m_cellStarts);
}

bool Caustics::onUiSettings()
{
  namespace PE = nvgui::PropertyEditor;
  bool changed = false;
  changed |= PE::Checkbox("Caustics (beta)", &m_enabled,
                          "(Beta) Light tracing and photon mapping for the caustics of smooth glass and polished metal:\n"
                          "photons leave the lights, follow the casters and land on what they light. The camera sees\n"
                          "them splatted into the image, or gathered from the photon map behind glass.\n"
                          "Casters become opaque to the light samples of those surfaces, so the photons carry that light,\n"
                          "focused. Off: nothing is traced and the render is unchanged. Changing it recompiles the shader.\n"
                          "Not covered: volume scattering, rough or partly diffuse casters.\n"
                          "See docs/caustics.md.");
  if(m_enabled)
  {
    changed |= PE::Checkbox("Auto Photons", &m_autoPhotons,
                            "Trace a quarter as many photons per frame as the render has pixels, for the splats and\n"
                            "the photon map alike. A photon that reaches what the camera sees lights one pixel, so\n"
                            "fewer leave the image speckled.");
    if(!m_autoPhotons)
    {
      changed |= PE::SliderInt("Caustic Photons", &m_photonsPerFrame, 256, int(kMaxPhotons), "%d", ImGuiSliderFlags_Logarithmic,
                               "Photons traced in each frame. More photons: smoother caustics, a longer frame.");
    }
    changed |= PE::Checkbox("Photon Map", &m_gather,
                            "Gather the frame's photons where the camera sees a receiver through glass or in a mirror\n"
                            "(the floor of a pool). Off: only what the camera sees directly has a caustic.");
    if(m_gather)
    {
      changed |= PE::SliderFloat("Gather Radius", &m_gatherPixels, 1.0F, 64.0F, "%.1f px", ImGuiSliderFlags_Logarithmic,
                                 "The caustic of a point is made of the photons around it, in a disk of this many pixels\n"
                                 "at first. The disk shrinks as the accumulation grows, to a pixel. Smaller: sharper and\n"
                                 "noisier at first.");
    }
  }
  return changed;
}

// CAUSTIC_MAX_ROUGHNESS in shaders/caustics.h.slang: the host test must not be stricter than the shader's
static constexpr float kMaxCasterRoughness = 0.1F;

//--------------------------------------------------------------------------------------------------
// The world-space box of an object-space box
static nvutils::Bbox transformedBox(const glm::vec3& lo, const glm::vec3& hi, const glm::mat4& objectToWorld)
{
  nvutils::Bbox box;
  for(int corner = 0; corner < 8; corner++)
  {
    const glm::vec3 p(corner & 1 ? hi.x : lo.x, corner & 2 ? hi.y : lo.y, corner & 4 ? hi.z : lo.z);
    box.insert(glm::vec3(objectToWorld * glm::vec4(p, 1.0F)));
  }
  return box;
}

//--------------------------------------------------------------------------------------------------
// The bounding sphere of the casters. Casters are found by their material's factors alone: a texture
// only scales them down, so a material whose factors cannot make a smooth caster (a rough one, a plain
// dielectric) never is one, and the shader's per-hit test (isCausticCaster) is stricter. The photons aim at
// the sphere around every node that may be one.
//
// Two keys, because a scene of millions of nodes must not be walked for what one node did. The list of the
// nodes that may be casters (and their primitives' boxes) follows the graph and the materials; moving a node
// only walks that list.
void Caustics::updateBounds(Resources& resources)
{
  nvvkgltf::Scene* scene = resources.getScene();
  if(!scene || !scene->valid())
  {
    m_casters = {};
    m_scene   = {};
    m_casterNodes.clear();
    m_boundsScene = nullptr;
    return;
  }

  const std::vector<nvvkgltf::RenderNode>& renderNodes = scene->getRenderNodes();
  const uint64_t structure[4] = {scene->getSceneGraphRevision(), resources.sceneVk.shadeMaterialsRevision(),
                                 renderNodes.size(), scene->getModel().materials.size()};
  const uint64_t transform    = scene->getNodeTransformRevision();
  const bool     restructured =
      m_boundsScene != scene || !std::equal(std::begin(structure), std::end(structure), std::begin(m_structureKey));
  if(!restructured && transform == m_transformKey)
    return;
  m_transformKey = transform;

  const nvutils::Bbox sceneBounds = scene->getSceneBounds();
  m_scene = sceneBounds.isEmpty() ? Sphere{} : Sphere{sceneBounds.center(), sceneBounds.radius()};

  if(restructured)
  {
    m_boundsScene = scene;
    std::copy(std::begin(structure), std::end(structure), std::begin(m_structureKey));
    m_casterNodes.clear();

    const std::vector<shaderio::GltfShadeMaterial>& shadeMaterials = resources.sceneVk.shadeMaterials();
    const tinygltf::Model&                          model          = scene->getModel();

    // One verdict per material, and one object-space box per primitive, however many nodes use them
    std::vector<int8_t>                    materialMayCast(shadeMaterials.size(), -1);
    std::unordered_map<int, nvutils::Bbox> primitiveBounds;
    for(size_t nodeIndex = 0; nodeIndex < renderNodes.size(); nodeIndex++)
    {
      const nvvkgltf::RenderNode& node = renderNodes[nodeIndex];
      if(node.materialID < 0 || node.materialID >= static_cast<int>(shadeMaterials.size()))
        continue;
      int8_t& mayCast = materialMayCast[node.materialID];
      if(mayCast < 0)
      {
        const shaderio::GltfShadeMaterial& m = shadeMaterials[node.materialID];
        // The shader's roughness is the factor squared (evaluateMaterial); a texture only lowers it
        const bool smooth =
            m.pbrRoughnessFactor * m.pbrRoughnessFactor <= kMaxCasterRoughness || m.pbrMetallicRoughnessTexture != 0;
        mayCast = (smooth && (m.transmissionFactor > 0.0F || m.pbrMetallicFactor > 0.0F)) ? 1 : 0;
      }
      if(!mayCast)
        continue;

      CasterNode caster;
      caster.renderNode = static_cast<int>(nodeIndex);

      // Skinning and morph targets move the vertices away from the accessor bounds, which are the bind
      // pose: a deformed caster is bounded by the whole scene
      const tinygltf::Primitive* primitive =
          node.renderPrimID >= 0 ? scene->getRenderPrimitive(node.renderPrimID).pPrimitive : nullptr;
      if(node.skinID >= 0 || (primitive != nullptr && !primitive->targets.empty()))
      {
        caster.wholeScene = true;
        m_casterNodes.push_back(caster);
        continue;
      }

      std::unordered_map<int, nvutils::Bbox>::iterator it = primitiveBounds.find(node.renderPrimID);
      if(it == primitiveBounds.end())
      {
        nvutils::Bbox bounds;
        glm::vec3     bmin, bmax;
        if(primitive != nullptr && tinygltf::utils::getPrimitivePositionBounds(model, *primitive, bmin, bmax))
          bounds = nvutils::Bbox(bmin, bmax);
        it = primitiveBounds.emplace(node.renderPrimID, bounds).first;
      }
      if(it->second.isEmpty())
        continue;
      caster.lo = it->second.min();
      caster.hi = it->second.max();
      m_casterNodes.push_back(caster);
    }
    assignClusters(renderNodes);
  }

  // One box per cluster, and all of them
  std::array<nvutils::Bbox, CAUSTIC_MAX_CASTERS> boxes;
  nvutils::Bbox                                  casters;
  for(size_t i = 0; i < m_casterNodes.size(); i++)
  {
    const CasterNode&           caster = m_casterNodes[i];
    const nvvkgltf::RenderNode& node   = renderNodes[caster.renderNode];
    if(!node.visible)
      continue;
    if(caster.wholeScene)
      boxes[caster.cluster].insert(
          nvutils::Bbox(m_scene.center - glm::vec3(m_scene.radius), m_scene.center + glm::vec3(m_scene.radius)));
    else
      boxes[caster.cluster].insert(transformedBox(caster.lo, caster.hi, node.worldMatrix));
  }
  m_clusters.clear();
  for(uint32_t i = 0; i < m_clusterCount; i++)
  {
    if(boxes[i].isEmpty())
      continue;
    m_clusters.push_back({boxes[i].center(), boxes[i].radius()});
    casters.insert(boxes[i]);
  }
  m_casters = casters.isEmpty() ? Sphere{} : Sphere{casters.center(), casters.radius()};
}

//--------------------------------------------------------------------------------------------------
// The photons aim at spheres around clusters of the casters, not one around all of them: two casters far apart
// would waste nearly every photon on the space between, and the camera looking at one would get few on it.
// The clusters come from splitting the set of caster nodes, the one whose box is largest first, in two at the
// median of the centers along its longest axis, until there are CAUSTIC_MAX_CASTERS. The assignment is made
// when the casters change; where the nodes are afterwards only moves the spheres (updateBounds).
void Caustics::assignClusters(const std::vector<nvvkgltf::RenderNode>& renderNodes)
{
  struct Group
  {
    std::vector<uint32_t> nodes = {};  // Indices into m_casterNodes
    nvutils::Bbox         box   = {};
  };

  // Deformed casters are bounded by the whole scene: they share a cluster of their own
  std::vector<nvutils::Bbox> boxes(m_casterNodes.size());
  Group                      first;
  bool                       hasWholeScene = false;
  for(uint32_t i = 0; i < m_casterNodes.size(); i++)
  {
    if(m_casterNodes[i].wholeScene)
    {
      hasWholeScene = true;
      continue;
    }
    boxes[i] = transformedBox(m_casterNodes[i].lo, m_casterNodes[i].hi, renderNodes[m_casterNodes[i].renderNode].worldMatrix);
    first.nodes.push_back(i);
    first.box.insert(boxes[i]);
  }

  std::vector<Group> groups;
  if(!first.nodes.empty())
    groups.push_back(std::move(first));
  const size_t limit = CAUSTIC_MAX_CASTERS - (hasWholeScene ? 1 : 0);
  while(groups.size() < limit)
  {
    // The largest group that can still be split
    size_t largest = groups.size();
    float  extent  = 0.0F;
    for(size_t g = 0; g < groups.size(); g++)
    {
      const float diagonal = groups[g].nodes.size() >= 2 ? glm::length(groups[g].box.max() - groups[g].box.min()) : 0.0F;
      if(diagonal > extent)
      {
        extent  = diagonal;
        largest = g;
      }
    }
    if(largest == groups.size())
      break;

    Group     group = std::move(groups[largest]);
    glm::vec3 lo(std::numeric_limits<float>::max()), hi(std::numeric_limits<float>::lowest());
    for(size_t n = 0; n < group.nodes.size(); n++)
    {
      lo = glm::min(lo, boxes[group.nodes[n]].center());
      hi = glm::max(hi, boxes[group.nodes[n]].center());
    }
    const glm::vec3 spread = hi - lo;
    const int       axis   = spread.x >= spread.y && spread.x >= spread.z ? 0 : (spread.y >= spread.z ? 1 : 2);
    const std::vector<uint32_t>::iterator middle = group.nodes.begin() + group.nodes.size() / 2;
    std::nth_element(group.nodes.begin(), middle, group.nodes.end(),
                     [&boxes, axis](uint32_t a, uint32_t b) { return boxes[a].center()[axis] < boxes[b].center()[axis]; });
    Group second;
    second.nodes.assign(middle, group.nodes.end());
    group.nodes.erase(middle, group.nodes.end());
    group.box = {};
    for(size_t n = 0; n < group.nodes.size(); n++)
      group.box.insert(boxes[group.nodes[n]]);
    for(size_t n = 0; n < second.nodes.size(); n++)
      second.box.insert(boxes[second.nodes[n]]);
    groups[largest] = std::move(group);
    groups.push_back(std::move(second));
  }

  for(uint32_t g = 0; g < groups.size(); g++)
    for(size_t n = 0; n < groups[g].nodes.size(); n++)
      m_casterNodes[groups[g].nodes[n]].cluster = g;
  m_clusterCount = uint32_t(groups.size());
  if(hasWholeScene)
  {
    for(size_t i = 0; i < m_casterNodes.size(); i++)
    {
      if(m_casterNodes[i].wholeScene)
        m_casterNodes[i].cluster = m_clusterCount;
    }
    m_clusterCount++;
  }
}

// How much of the photons a sphere around casters gets: in proportion to the share of the view it fills,
// and a floor for what is off screen or small, which still casts caustics into the view
static float casterWeight(const glm::mat4& viewProj, const glm::vec3& center, float radius)
{
  const glm::vec4 clip   = viewProj * glm::vec4(center, 1.0F);
  const float     extent = radius
                       * std::max(glm::length(glm::vec3(viewProj[0][0], viewProj[1][0], viewProj[2][0])),
                                  glm::length(glm::vec3(viewProj[0][1], viewProj[1][1], viewProj[2][1])));
  constexpr float kFloor = 0.05F;
  if(clip.w <= 0.0F || std::abs(clip.x) > clip.w + extent || std::abs(clip.y) > clip.w + extent)
    return kFloor;
  return kFloor + std::min(1.0F, extent * extent / std::max(clip.w * clip.w, 1e-20F));
}

//--------------------------------------------------------------------------------------------------
// The grid's passes: one small shader (caustics_grid.slang), a pipeline per entry point
void Caustics::createGridPipelines(VkDevice device)
{
  static constexpr const char* kEntries[kGridPasses] = {"causticCountMain", "causticScanBlocksMain", "causticScanSumsMain",
                                                        "causticAddOffsetsMain", "causticScatterMain"};
  const VkPushConstantRange        range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkDeviceAddress)};  // The CausticParams
  const VkPipelineLayoutCreateInfo layoutInfo{.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                              .pushConstantRangeCount = 1,
                                              .pPushConstantRanges    = &range};
  NVVK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_gridLayout));
  NVVK_DBG_NAME(m_gridLayout);

  VkShaderModuleCreateInfo moduleInfo{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                      .codeSize = caustics_grid_slang_sizeInBytes,
                                      .pCode    = caustics_grid_slang};
  for(uint32_t i = 0; i < kGridPasses; i++)
  {
    const VkComputePipelineCreateInfo info{.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                           .stage  = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                      .pNext = &moduleInfo,
                                                      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                                      .pName = kEntries[i]},
                                           .layout = m_gridLayout};
    NVVK_CHECK(vkCreateComputePipelines(device, nullptr, 1, &info, nullptr, &m_gridPipelines[i]));
    NVVK_DBG_NAME(m_gridPipelines[i]);
  }
}

void Caustics::cmdBuildGrid(VkCommandBuffer cmd)
{
  if(mapGroups() == 0)
    return;
  const VkDeviceAddress params = m_params.address;
  vkCmdPushConstants(cmd, m_gridLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
  const uint32_t groups[kGridPasses] = {mapGroups(), cellGroups(), 1, cellGroups(), mapGroups()};
  for(uint32_t i = 0; i < kGridPasses; i++)
  {
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_gridPipelines[i]);
    vkCmdDispatch(cmd, groups[i], 1, 1);
  }
}

void Caustics::createMap(Resources& resources)
{
  constexpr VkBufferUsageFlags2KHR usage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                           | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT;
  if(m_photons.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(resources.allocator.createBuffer(m_photons, VkDeviceSize(kMapSlots) * sizeof(shaderio::CausticPhoton), usage));
    NVVK_DBG_NAME(m_photons.buffer);
  }
  if(m_sorted.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(resources.allocator.createBuffer(m_sorted, VkDeviceSize(kMapSlots) * sizeof(shaderio::CausticPhoton), usage));
    NVVK_DBG_NAME(m_sorted.buffer);
  }
  if(m_cellStarts.buffer == VK_NULL_HANDLE)
  {
    // The cells and the end, then the totals of the blocks of cells the prefix sum goes by
    NVVK_CHECK(resources.allocator.createBuffer(
        m_cellStarts, VkDeviceSize(kMaxCells + 1 + kMaxCells / kScanGroupSize) * sizeof(uint32_t), usage));
    NVVK_DBG_NAME(m_cellStarts.buffer);
  }
}

//--------------------------------------------------------------------------------------------------
shaderio::CausticParams* Caustics::cmdPrepare(VkCommandBuffer cmd, Resources& resources, VkExtent2D renderSize, bool canTrace)
{
  m_photonCount = 0;
  m_mapPhotons  = 0;
  m_cells       = 0;
  if(!m_enabled || !canTrace)
    return nullptr;

  // The splats project with a perspective camera's measure
  if(resources.cameraManip->getProjectionType() == nvutils::CameraManipulator::ProjectionType::Orthographic)
    return nullptr;

  updateBounds(resources);
  if(m_casters.radius <= 0.0F)
    return nullptr;  // Nothing can cast a caustic: the paths alone, as if the setting were off

  // One splat per pixel of the render size. The buffer only grows: a render size that comes back (DLSS and an
  // accumulated render alternate in some modes) reuses it, and the shaders index it by the frame's own width. Growing
  // waits for the device, since the frames in flight read the old one.
  const VkDeviceSize splatBytes = VkDeviceSize(renderSize.width) * renderSize.height * 3 * sizeof(uint32_t);
  if(m_splats.buffer == VK_NULL_HANDLE || m_splats.bufferSize < splatBytes)
  {
    if(m_splats.buffer != VK_NULL_HANDLE)
    {
      NVVK_CHECK(vkDeviceWaitIdle(resources.allocator.getDevice()));
      resources.allocator.destroyBuffer(m_splats);
    }
    NVVK_CHECK(resources.allocator.createBuffer(m_splats, splatBytes,
                                                VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                                    | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
    NVVK_DBG_NAME(m_splats.buffer);
  }
  if(m_params.buffer == VK_NULL_HANDLE)
  {
    NVVK_CHECK(resources.allocator.createBuffer(m_params, sizeof(shaderio::CausticParams),
                                                VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                                    | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
    NVVK_DBG_NAME(m_params.buffer);
  }

  const int64_t wanted =
      m_autoPhotons ? std::max<int64_t>(int64_t(renderSize.width) * renderSize.height / 4, kMinAutoPhotons) : m_photonsPerFrame;
  m_photonCount = uint32_t(std::clamp<int64_t>(wanted, 1, kMaxPhotons));

  shaderio::CausticParams params{};
  const glm::mat4         viewProj =
      glm::mat4(resources.cameraManip->getPerspectiveMatrix()) * glm::mat4(resources.cameraManip->getViewMatrix());
  params.casterCount = uint32_t(m_clusters.size());
  for(uint32_t i = 0; i < params.casterCount; i++)
    params.casters[i] = {m_clusters[i].center, m_clusters[i].radius,
                         casterWeight(viewProj, m_clusters[i].center, m_clusters[i].radius)};
  params.sceneCenter = m_scene.center;
  params.sceneRadius = std::max(m_scene.radius, m_casters.radius);
  params.splats      = reinterpret_cast<uint32_t*>(m_splats.address);
  params.photonCount = m_photonCount;
  params.photonFirst = m_emitted;  // The photons are numbered for as long as the app runs, whatever resets
  m_emitted += m_photonCount;

  if(m_gather)
  {
    createMap(resources);
    if(m_gridLayout == VK_NULL_HANDLE)
      createGridPipelines(resources.allocator.getDevice());
    m_mapPhotons = std::min(m_photonCount, kMapSlots);
    m_cells      = std::clamp(std::bit_ceil(std::max(2u * m_mapPhotons, 1u)), 1u << 12, kMaxCells);

    // The gather's radius starts wide, which is smooth, and shrinks with the frames accumulated (the radius of
    // progressive photon mapping goes with k^(-1/6) in 3D) down to a pixel, which is sharp
    const float shrink  = std::pow(float(std::max(resources.frameCount, 0) + 1), -1.0F / 6.0F);
    params.gatherPixels = std::max(m_gatherPixels * shrink, kMinGatherPixels);
    params.gatherWorld  = std::max(kWorldRadius * m_casters.radius * shrink, 1e-6F);
    params.gatherMax    = kGatherMax;

    // Cells several times as wide as the typical gather disk: the casters' distance from the camera gives the
    // pixel's size there. A gather reads the cells its disk overlaps, and each read of a cell is a cache miss, so
    // a few big cells beat many small ones even though they hold photons the disk rejects (measured: 1x the
    // radius reads twice as long as 4x, 16x twice as long again).
    const float pixelAngle = 2.0F * std::abs(glm::inverse(glm::mat4(resources.cameraManip->getPerspectiveMatrix()))[1][1])
                             / std::max(float(renderSize.height), 1.0F);
    const float distance =
        std::max(glm::length(glm::vec3(resources.cameraManip->getEye()) - m_casters.center), 0.5F * m_casters.radius);
    params.cellSize = kCellRadii * std::max(std::min(params.gatherPixels * pixelAngle * distance, params.gatherWorld), 1e-6F);

    params.photons    = reinterpret_cast<shaderio::CausticPhoton*>(m_photons.address);
    params.sorted     = reinterpret_cast<shaderio::CausticPhoton*>(m_sorted.address);
    params.cellStarts = reinterpret_cast<uint32_t*>(m_cellStarts.address);
    params.mapSlots   = kMapSlots;
    params.cellMask   = m_cells - 1;
  }

  // The last frame's paths still read these buffers
  nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT);
  vkCmdUpdateBuffer(cmd, m_params.buffer, 0, sizeof(params), &params);
  vkCmdFillBuffer(cmd, m_splats.buffer, 0, splatBytes, 0);  // Only what this render size uses
  if(m_cells > 0)                                           // The counts of the sort start at zero
    vkCmdFillBuffer(cmd, m_cellStarts.buffer, 0, VkDeviceSize(m_cells + 1) * sizeof(uint32_t), 0u);
  return reinterpret_cast<shaderio::CausticParams*>(m_params.address);
}
