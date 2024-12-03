/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2024 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "scene.hpp"

#include "fileformats/tiny_converter.hpp"
#include "nvh/cameramanipulator.hpp"
#include "nvh/timesampler.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/scene_camera.hpp"
#include "settings.hpp"
#include "shaders/dh_bindings.h"
#include "tiny_obj_loader.h"
#include "create_tangent.hpp"
#include "nvvkhl/shaders/dh_tonemap.h"
#include "collapsing_header_manager.h"

extern std::shared_ptr<nvvkhl::ElementCamera> g_elemCamera;  // Is accessed elsewhere in the App
namespace PE = ImGuiH::PropertyEditor;

constexpr uint32_t MAXTEXTURES = 1000;  // Maximum textures allowed in the application


//--------------------------------------------------------------------------------------------------
// Initialization of the scene object
// - Create the buffers for the scene frame information
// - Create the sky
// - Create the empty HDR environment
void gltfr::Scene::init(Resources& res)
{
  nvvk::ResourceAllocator* alloc = res.m_allocator.get();

  createHdr(res, "");  // Initialize the environment with nothing (constant white: for now)
  m_sky = std::make_unique<nvvkhl::PhysicalSkyDome>();  // Sun&Sky
  m_sky->setup(res.ctx.device, alloc);

  // Create the buffer of the current frame, changing at each frame
  {
    m_sceneFrameInfoBuffer =
        res.m_allocator->createBuffer(sizeof(DH::SceneFrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    nvvk::DebugUtil(res.ctx.device).DBG_NAME(m_sceneFrameInfoBuffer.buffer);
  }

  VkDevice device = res.ctx.device;
  createDescriptorPool(device);
  createDescriptorSet(device);
}

//--------------------------------------------------------------------------------------------------
// De-initialization of the scene object
// - Destroy the buffers
// - Reset the scene objects
// - Reset the HDR environment
// - Reset the sky
void gltfr::Scene::deinit(Resources& res)
{
  res.m_allocator->destroy(m_sceneFrameInfoBuffer);

  destroyDescriptorSet(res.ctx.device);

  m_gltfSceneVk.reset();
  m_gltfSceneRtx.reset();
  m_gltfScene.reset();
  m_hdrEnv.reset();
  m_hdrDome.reset();
  m_sky.reset();
}

//--------------------------------------------------------------------------------------------------
// Position the camera to fit the scene
//
void gltfr::Scene::fitSceneToView() const
{
  if(m_gltfScene)
  {
    auto bbox = m_gltfScene->getSceneBounds();
    CameraManip.fit(bbox.min(), bbox.max(), false, true, CameraManip.getAspectRatio());
  }
}

//--------------------------------------------------------------------------------------------------
// Position the camera to fit the selected object
//
void gltfr::Scene::fitObjectToView() const
{
  if(m_selectedRenderNode >= 0)
  {
    nvh::Bbox worldBbox = getRenderNodeBbox(m_selectedRenderNode);
    CameraManip.fit(worldBbox.min(), worldBbox.max(), false, true, CameraManip.getAspectRatio());
  }
}

//--------------------------------------------------------------------------------------------------
// Select a render node
// - tells the scene graph to select the node
void gltfr::Scene::selectRenderNode(int renderNodeIndex)
{
  m_selectedRenderNode = renderNodeIndex;
  if(m_sceneGraph && m_gltfScene && renderNodeIndex > -1)
  {
    const nvh::gltf::RenderNode& renderNode = m_gltfScene->getRenderNodes()[renderNodeIndex];
    m_sceneGraph->selectNode(renderNode.refNodeID);
  }
  else if(m_sceneGraph)
  {
    m_sceneGraph->selectNode(-1);
  }
}

//--------------------------------------------------------------------------------------------------
// Return the filename of the scene
//
std::string gltfr::Scene::getFilename() const
{
  if(m_gltfScene != nullptr)
    return m_gltfScene->getFilename();
  return "empty";
}

//--------------------------------------------------------------------------------------------------
// Recreating the tangents of the scene
void gltfr::Scene::recreateTangents(bool onlyFix)
{
  if(m_gltfScene && m_gltfScene->valid())
  {
    {
      nvh::ScopedTimer st(std::string("\n") + __FUNCTION__);
      recomputeTangents(m_gltfScene->getModel(), false, onlyFix);
    }
    m_dirtyFlags.set(eVulkanAttributes);
    resetFrameCount();
  }
}

//--------------------------------------------------------------------------------------------------
// Load a scene or HDR environment
//
bool gltfr::Scene::load(Resources& resources, const std::string& filename)
{
  const std::string extension   = std::filesystem::path(filename).extension().string();
  bool              sceneloaded = false;

  if(extension == ".gltf" || extension == ".glb")
  {
    m_gltfScene = std::make_unique<nvh::gltf::Scene>();
    m_sceneGraph.reset();
    m_selectedRenderNode = -1;

    // Loading the scene
    if(m_gltfScene->load(filename))
    {
      sceneloaded  = true;
      m_sceneGraph = std::make_unique<GltfModelUI>(m_gltfScene->getModel(), m_gltfScene->getSceneBounds());
      createVulkanScene(resources);
    }
    else
    {
      m_gltfScene.release();
      return false;
    }
  }
  else if(extension == ".obj")
  {
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = std::filesystem::path(filename).parent_path().string();
    tinyobj::ObjReader reader;

    bool        result = reader.ParseFromFile(filename, reader_config);
    std::string warn   = reader.Warning();
    std::string error  = reader.Error();

    if(result)
    {
      sceneloaded = true;
      TinyConverter   converter;
      tinygltf::Model model;
      converter.convert(model, reader);
      m_gltfScene = std::make_unique<nvh::gltf::Scene>();
      m_gltfScene->takeModel(std::move(model));
      m_sceneGraph = std::make_unique<GltfModelUI>(m_gltfScene->getModel(), m_gltfScene->getSceneBounds());
      createVulkanScene(resources);
    }
    else
    {
      m_gltfScene.release();
      LOGE("Error loading OBJ: %s\n", error.c_str());
      LOGW("Warning: %s\n", warn.c_str());
      return false;
    }
  }
  else if(extension == ".hdr")
  {
    createHdr(resources, filename);
    sceneloaded = false;
  }

  if(sceneloaded)
  {
    postSceneCreateProcess(resources, filename);
  }

  resetFrameCount();
  return true;
}

//--------------------------------------------------------------------------------------------------
// After the scene is loaded, we need to create the descriptor set and write the information
// - This is done after the scene is loaded, and the camera is fitted
//
void gltfr::Scene::postSceneCreateProcess(Resources& resources, const std::string& filename)
{
  if(filename.empty())
    return;

  setDirtyFlag(Scene::eNewScene, true);

  writeDescriptorSet(resources);

  // Scene camera fitting
  nvh::Bbox                                   bbox    = m_gltfScene->getSceneBounds();
  const std::vector<nvh::gltf::RenderCamera>& cameras = m_gltfScene->getRenderCameras();
  nvvkhl::setCamera(filename, cameras, bbox);   // Camera auto-scene-fitting
  g_elemCamera->setSceneRadius(bbox.radius());  // Navigation help
}

//--------------------------------------------------------------------------------------------------
// Save the scene
//
bool gltfr::Scene::save(const std::string& filename) const
{
  if(m_gltfScene && m_gltfScene->valid() && !filename.empty())
  {
    // First, copy the camera
    nvh::gltf::RenderCamera camera;
    CameraManip.getLookat(camera.eye, camera.center, camera.up);
    camera.yfov  = glm::radians(CameraManip.getFov());
    camera.znear = CameraManip.getClipPlanes().x;
    camera.zfar  = CameraManip.getClipPlanes().y;
    m_gltfScene->setSceneCamera(camera);
    // Saving the scene
    return m_gltfScene->save(filename);
  }
  return false;
}

void gltfr::Scene::createDescriptorPool(VkDevice device)
{
  const std::vector<VkDescriptorPoolSize> poolSizes{
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAXTEXTURES},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
  };

  const VkDescriptorPoolCreateInfo poolInfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |  //  allows descriptor sets to be updated after they have been bound to a command buffer
               VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,  // individual descriptor sets can be freed from the descriptor pool
      .maxSets       = MAXTEXTURES,  // Allowing to create many sets (ImGui uses this for textures)
      .poolSizeCount = uint32_t(poolSizes.size()),
      .pPoolSizes    = poolSizes.data(),
  };
  NVVK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool));
  // nvvk::DebugUtil(device).DBG_NAME(m_descriptorPool));
}

//--------------------------------------------------------------------------------------------------
// Create the descriptor set for the scene
//
void gltfr::Scene::createDescriptorSet(VkDevice device)
{
  std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
  layoutBindings.push_back({.binding         = SceneBindings::eFrameInfo,
                            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            .descriptorCount = 1,
                            .stageFlags      = VK_SHADER_STAGE_ALL});
  layoutBindings.push_back({.binding         = SceneBindings::eSceneDesc,
                            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .descriptorCount = 1,
                            .stageFlags      = VK_SHADER_STAGE_ALL});
  layoutBindings.push_back({.binding         = SceneBindings::eTextures,
                            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            .descriptorCount = MAXTEXTURES,  // Not all will be filled - but pipeline will be cached
                            .stageFlags      = VK_SHADER_STAGE_ALL});

  const VkDescriptorBindingFlags flags[] = {
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,  // Flags for binding 0 (uniform buffer)
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,  // Flags for binding 1 (storage buffer)
      // Flags for binding 2 (texture array):
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |                // Can update while in use
          VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT |  // Can update unused entries
          VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,  // Not all array elements need to be valid (0,2,3 vs 0,1,2,3)
  };
  const VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{
      .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .bindingCount  = uint32_t(layoutBindings.size()),  // matches our number of bindings
      .pBindingFlags = flags,                            // the flags for each binding
  };

  const VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &bindingFlags,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,  // Allows to update the descriptor set after it has been bound
      .bindingCount = uint32_t(layoutBindings.size()),
      .pBindings    = layoutBindings.data(),
  };
  NVVK_CHECK(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutInfo, nullptr, &m_sceneDescriptorSetLayout));
  nvvk::DebugUtil(device).DBG_NAME(m_sceneDescriptorSetLayout);

  // Allocate the descriptor set, needed only for larger descriptor sets
  const VkDescriptorSetAllocateInfo allocInfo = {
      .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool     = m_descriptorPool,
      .descriptorSetCount = 1,
      .pSetLayouts        = &m_sceneDescriptorSetLayout,
  };
  NVVK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &m_sceneDescriptorSet));
  nvvk::DebugUtil(device).DBG_NAME(m_sceneDescriptorSet);
}

//--------------------------------------------------------------------------------------------------
// Write the descriptor set for the scene
//
void gltfr::Scene::writeDescriptorSet(Resources& resources) const
{
  if(!m_gltfScene->valid())
  {
    return;
  }

  // Write to descriptors
  const VkDescriptorBufferInfo frameBufferInfo{m_sceneFrameInfoBuffer.buffer, 0, VK_WHOLE_SIZE};
  const VkDescriptorBufferInfo sceneBufferInfo{m_gltfSceneVk->sceneDesc().buffer, 0, VK_WHOLE_SIZE};

  std::vector<VkWriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.push_back({.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet          = m_sceneDescriptorSet,
                                 .dstBinding      = SceneBindings::eFrameInfo,
                                 .dstArrayElement = 0,
                                 .descriptorCount = 1,
                                 .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                 .pBufferInfo     = &frameBufferInfo});
  writeDescriptorSets.push_back({.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet          = m_sceneDescriptorSet,
                                 .dstBinding      = SceneBindings::eSceneDesc,
                                 .dstArrayElement = 0,
                                 .descriptorCount = 1,
                                 .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                 .pBufferInfo     = &sceneBufferInfo});

  std::vector<VkDescriptorImageInfo> descImageInfos;
  descImageInfos.reserve(m_gltfSceneVk->nbTextures());
  for(const nvvk::Texture& texture : m_gltfSceneVk->textures())  // All texture samplers
  {
    descImageInfos.emplace_back(texture.descriptor);
  }
  writeDescriptorSets.push_back({.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                 .dstSet          = m_sceneDescriptorSet,
                                 .dstBinding      = SceneBindings::eTextures,
                                 .dstArrayElement = 0,
                                 .descriptorCount = m_gltfSceneVk->nbTextures(),
                                 .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                 .pImageInfo      = descImageInfos.data()});

  vkUpdateDescriptorSets(resources.ctx.device, static_cast<uint32_t>(writeDescriptorSets.size()),
                         writeDescriptorSets.data(), 0, nullptr);
}

void gltfr::Scene::destroyDescriptorSet(VkDevice device)
{
  if(m_descriptorPool)
  {
    vkFreeDescriptorSets(device, m_descriptorPool, 1, &m_sceneDescriptorSet);
    vkDestroyDescriptorSetLayout(device, m_sceneDescriptorSetLayout, nullptr);
    vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
  }
}

//--------------------------------------------------------------------------------------------------
// Processing the frame is something call at each frame
// If something has changed, we need to update one of the following:
// - Update the animation
// - Update the camera
// - Update the environment
// - Update the Vulkan scene
// - Update the RTX scene
//
bool gltfr::Scene::processFrame(VkCommandBuffer cmdBuf, Settings& settings)
{
  // Dealing with animation
  if(m_gltfScene->hasAnimation() && m_animControl.doAnimation())
  {
    float                     deltaTime = m_animControl.deltaTime();
    nvh::gltf::AnimationInfo& animInfo  = m_gltfScene->getAnimationInfo(m_animControl.currentAnimation);
    if(m_animControl.isReset())
    {
      animInfo.reset();
    }
    else
    {
      animInfo.incrementTime(deltaTime);
    }

    m_gltfScene->updateAnimation(m_animControl.currentAnimation);
    m_gltfScene->updateRenderNodes();

    m_animControl.clearStates();

    {
      m_dirtyFlags.set(eVulkanScene);
      m_dirtyFlags.set(eRtxScene);
      resetFrameCount();
    }
  }

  // Increase the frame count and return if we reached the maximum
  if(!updateFrameCount(settings))
    return false;

  // Check for scene changes
  if(m_dirtyFlags.test(eVulkanScene))
  {
    m_gltfSceneVk->updateRenderNodesBuffer(cmdBuf, *m_gltfScene);       // Animation, changing nodes transform
    m_gltfSceneVk->updateRenderPrimitivesBuffer(cmdBuf, *m_gltfScene);  // Animation
    m_gltfSceneVk->updateRenderLightsBuffer(cmdBuf, *m_gltfScene);      // changing lights data
    m_dirtyFlags.reset(eVulkanScene);
  }
  if(m_dirtyFlags.test(eVulkanMaterial))
  {
    m_gltfSceneVk->updateMaterialBuffer(cmdBuf, *m_gltfScene);
    m_dirtyFlags.reset(eVulkanMaterial);
  }
  if(m_dirtyFlags.test(eVulkanAttributes))
  {
    m_gltfSceneVk->updateVertexBuffers(cmdBuf, *m_gltfScene);
    m_dirtyFlags.reset(eVulkanAttributes);
  }
  if(m_dirtyFlags.test(eRtxScene))
  {
    m_gltfSceneRtx->updateTopLevelAS(cmdBuf, *m_gltfScene);
    m_gltfSceneRtx->updateBottomLevelAS(cmdBuf, *m_gltfScene);

    m_dirtyFlags.reset(eRtxScene);
  }

  // Update the camera
  const glm::vec2& clip       = CameraManip.getClipPlanes();
  m_sceneFrameInfo.viewMatrix = CameraManip.getMatrix();
  m_sceneFrameInfo.projMatrix =
      glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), CameraManip.getAspectRatio(), clip.x, clip.y);
  m_sceneFrameInfo.projMatrix[1][1] *= -1;
  m_sceneFrameInfo.projMatrixI = glm::inverse(m_sceneFrameInfo.projMatrix);
  m_sceneFrameInfo.viewMatrixI = glm::inverse(m_sceneFrameInfo.viewMatrix);
  m_sceneFrameInfo.camPos      = CameraManip.getEye();

  // Update the environment
  m_sceneFrameInfo.envIntensity = glm::vec4(settings.hdrEnvIntensity, settings.hdrEnvIntensity, settings.hdrEnvIntensity, 1.0F);
  m_sceneFrameInfo.envRotation = settings.hdrEnvRotation;
  m_sceneFrameInfo.envBlur     = settings.hdrBlur;
  m_sceneFrameInfo.flags       = 0;
  if(settings.envSystem == Settings::eSky)
  {
    SET_FLAG(m_sceneFrameInfo.flags, USE_SKY_FLAG);
    m_sceneFrameInfo.nbLights = 1;  //static_cast<int>(settings.lights.size());
    m_sceneFrameInfo.light[0] = m_sky->getSun();
  }
  else
  {
    SET_FLAG(m_sceneFrameInfo.flags, USE_HDR_FLAG);
    m_sceneFrameInfo.nbLights = 0;
  }
  if(settings.useSolidBackground)
  {
    SET_FLAG(m_sceneFrameInfo.flags, USE_SOLID_BACKGROUND_FLAG);
    m_sceneFrameInfo.backgroundColor = nvvkhl_shaders::toLinear(settings.solidBackgroundColor);
  }


  vkCmdUpdateBuffer(cmdBuf, m_sceneFrameInfoBuffer.buffer, 0, sizeof(DH::SceneFrameInfo), &m_sceneFrameInfo);

  // Barrier to ensure the buffer is updated before rendering
  VkBufferMemoryBarrier bufferBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  bufferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  bufferBarrier.buffer        = m_sceneFrameInfoBuffer.buffer;
  bufferBarrier.size          = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0,
                       nullptr, 1, &bufferBarrier, 0, nullptr);

  // Update the sky
  m_sky->skyParams().yIsUp = CameraManip.getUp().y > CameraManip.getUp().z;
  m_sky->updateParameterBuffer(cmdBuf);

  return true;
}

//--------------------------------------------------------------------------------------------------
// Create the Vulkan scene representation
// This means that the glTF scene is converted into buffers and acceleration structures
// The sceneVk is the Vulkan representation of the scene
// - Materials
// - Textures
// - RenderNodes and RenderPrimitives
// The sceneRtx is the Vulkan representation of the scene for ray tracing
// - Bottom-level acceleration structures
// - Top-level acceleration structure
void gltfr::Scene::createVulkanScene(Resources& res)
{
  nvh::ScopedTimer st(std::string("\n") + __FUNCTION__);

  nvvk::ResourceAllocator* alloc = res.m_allocator.get();

  m_gltfSceneVk = std::make_unique<nvvkhl::SceneVk>(res.ctx.device, res.ctx.physicalDevice, alloc);
  m_gltfSceneRtx = std::make_unique<nvvkhl::SceneRtx>(res.ctx.device, res.ctx.physicalDevice, alloc, res.ctx.compute.familyIndex);

  if(m_gltfScene->valid())
  {
    // Create the Vulkan side of the scene
    // Since we load and display simultaneously, we need to use a second GTC queue
    nvvk::CommandPool cmd_pool(res.ctx.device, res.ctx.compute.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                               res.ctx.compute.queue);
    VkCommandBuffer   cmd;
    {  // Creating the scene in Vulkan buffers
      cmd = cmd_pool.createCommandBuffer();
      m_gltfSceneVk->create(cmd, *m_gltfScene, false);
      // This method is simpler, but it is not as efficient as the while-loop below
      // m_sceneRtx->create(cmd, *m_scene, *m_sceneVk, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
      cmd_pool.submitAndWait(cmd);
      res.m_allocator->finalizeAndReleaseStaging();  // Make sure there are no pending staging buffers and clear them up
    }

    // Create the acceleration structure, and compact the BLAS
    VkBuildAccelerationStructureFlagsKHR blasBuildFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                          | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR
                                                          | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    m_gltfSceneRtx->createBottomLevelAccelerationStructure(*m_gltfScene, *m_gltfSceneVk, blasBuildFlags);
    bool finished = false;
    do
    {
      {  // Building the BLAS
        cmd      = cmd_pool.createCommandBuffer();
        finished = m_gltfSceneRtx->cmdBuildBottomLevelAccelerationStructure(cmd, 512'000'000);
        cmd_pool.submitAndWait(cmd);
      }
      {  // Compacting the BLAS
        cmd = cmd_pool.createCommandBuffer();
        m_gltfSceneRtx->cmdCompactBlas(cmd);
        cmd_pool.submitAndWait(cmd);
      }
      m_gltfSceneRtx->destroyNonCompactedBlas();
    } while(!finished);

    {  // Creating the top-level acceleration structure
      cmd = cmd_pool.createCommandBuffer();
      m_gltfSceneRtx->cmdCreateBuildTopLevelAccelerationStructure(cmd, *m_gltfScene);
      cmd_pool.submitAndWait(cmd);
    }
    if(!m_gltfScene->hasAnimation())
    {
      m_gltfSceneRtx->destroyScratchBuffers();
    }
  }
  else
  {
    m_gltfSceneRtx.release();
    m_gltfSceneVk.release();
  }
}

//--------------------------------------------------------------------------------------------------
// Create the HDR environment
//
void gltfr::Scene::createHdr(Resources& res, const std::string& filename)
{
  nvh::ScopedTimer st(std::string("\n") + __FUNCTION__);

  nvvk::ResourceAllocator* alloc          = res.m_allocator.get();
  const uint32_t           c_family_queue = res.ctx.compute.familyIndex;

  m_hdrEnv  = std::make_unique<nvvkhl::HdrEnv>(res.ctx.device, res.ctx.physicalDevice, alloc, c_family_queue);
  m_hdrDome = std::make_unique<nvvkhl::HdrEnvDome>(res.ctx.device, res.ctx.physicalDevice, alloc, c_family_queue);
  m_hdrEnv->loadEnvironment(filename, true);
  m_hdrDome->create(m_hdrEnv->getDescriptorSet(), m_hdrEnv->getDescriptorSetLayout());
  alloc->finalizeAndReleaseStaging();

  m_hdrFilename = std::filesystem::path(filename).filename().string();
  setDirtyFlag(Scene::eHdrEnv, true);
}

void gltfr::Scene::generateHdrMipmap(VkCommandBuffer cmd, Resources& res)
{
  vkQueueWaitIdle(res.ctx.GCT0.queue);
  nvvk::cmdGenerateMipmaps(cmd, m_hdrEnv->getHdrTexture().image, VK_FORMAT_R32G32B32A32_SFLOAT,
                           m_hdrEnv->getHdrImageSize(), nvvk::mipLevels(m_hdrEnv->getHdrImageSize()));
}

//--------------------------------------------------------------------------------------------------
// Update the frame counter only if the camera has NOT changed
// otherwise, reset the frame counter
//
bool gltfr::Scene::updateFrameCount(Settings& settings)
{
  static glm::mat4 ref_cam_matrix;
  static float     ref_fov{CameraManip.getFov()};

  const glm::mat4& m   = CameraManip.getMatrix();
  const float      fov = CameraManip.getFov();

  if(ref_cam_matrix != m || ref_fov != fov)
  {
    resetFrameCount();
    ref_cam_matrix = m;
    ref_fov        = fov;
  }

  if(m_sceneFrameInfo.frameCount >= settings.maxFrames)
  {
    return false;
  }
  m_sceneFrameInfo.frameCount++;
  return true;
}

//--------------------------------------------------------------------------------------------------
// Reset the frame counter
void gltfr::Scene::resetFrameCount()
{
  m_sceneFrameInfo.frameCount = -1;
}

nvh::Bbox gltfr::Scene::getRenderNodeBbox(int nodeID) const
{
  nvh::Bbox worldBbox({-1, -1, -1}, {1, 1, 1});
  if(nodeID < 0)
    return worldBbox;

  const nvh::gltf::RenderNode&      renderNode      = m_gltfScene->getRenderNodes()[nodeID];
  const nvh::gltf::RenderPrimitive& renderPrimitive = m_gltfScene->getRenderPrimitive(renderNode.renderPrimID);
  const tinygltf::Model&            model           = m_gltfScene->getModel();
  const tinygltf::Accessor&         accessor = model.accessors[renderPrimitive.primitive.attributes.at("POSITION")];

  glm::vec3 minValues = {-1.f, -1.f, -1.f};
  glm::vec3 maxValues = {1.f, 1.f, 1.f};
  if(!accessor.minValues.empty())
    minValues = glm::vec3(accessor.minValues[0], accessor.minValues[1], accessor.minValues[2]);
  if(!accessor.maxValues.empty())
    maxValues = glm::vec3(accessor.maxValues[0], accessor.maxValues[1], accessor.maxValues[2]);
  nvh::Bbox objBbox(minValues, maxValues);
  worldBbox = objBbox.transform(renderNode.worldMatrix);

  return worldBbox;
}

//--------------------------------------------------------------------------------------------------
// Rendering the UI of the scene
// - Environment
//  - Sky
//  - HDR
// - Scene
//   - Multiple Scenes
//   - Variants
//   - Animation
//   - Scene Graph
//   - Statistics
//
bool gltfr::Scene::onUI(Resources& resources, Settings& settings, GLFWwindow* winHandle)
{
  auto& headerManager = CollapsingHeaderManager::getInstance();

  bool reset = false;

  if(headerManager.beginHeader("Environment"))
  {
    const bool          skyOnly          = !(m_hdrEnv && m_hdrEnv->isValid());
    Settings::EnvSystem cache_env_system = settings.envSystem;
    reset |= ImGui::RadioButton("Sky", reinterpret_cast<int*>(&settings.envSystem), Settings::eSky);
    ImGui::SameLine();
    ImGui::BeginDisabled(skyOnly);
    reset |= ImGui::RadioButton("Hdr", reinterpret_cast<int*>(&settings.envSystem), Settings::eHdr);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if(ImGui::SmallButton("Load##env"))
    {
      std::string filename = NVPSystem::windowOpenFileDialog(winHandle, "Load HDR", "HDR(.hdr)|*.hdr");
      if(!filename.empty())
      {
        vkDeviceWaitIdle(resources.ctx.device);
        createHdr(resources, filename);
        settings.envSystem = Settings::eHdr;
        reset              = true;
      }
    }

    // When switching the environment, reset Firefly max luminance
    if(cache_env_system != settings.envSystem)
    {
      settings.setDefaultLuminance(m_hdrEnv->getIntegral());
    }

    PE::begin();
    if(settings.envSystem == Settings::eSky)
    {
      reset |= m_sky->onUI();
    }
    else  // HDR
    {
      PE::Text("HDR File", m_hdrFilename);
      reset |= PE::SliderFloat("Intensity", &settings.hdrEnvIntensity, 0, 100, "%.3f", ImGuiSliderFlags_Logarithmic, "HDR intensity");
      reset |= PE::SliderAngle("Rotation", &settings.hdrEnvRotation, -360, 360, "%.0f deg", 0, "Rotating the environment");
      reset |= PE::SliderFloat("Blur", &settings.hdrBlur, 0, 1, "%.3f", 0, "Blur the environment");
    }
    PE::end();
    PE::begin();
    reset |= PE::Checkbox("Use Solid Background", &settings.useSolidBackground);
    if(settings.useSolidBackground)
    {
      reset |= PE::ColorEdit3("Background Color", glm::value_ptr(settings.solidBackgroundColor),
                              ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_Float);
    }
    PE::end();
  }

  if(m_gltfScene && m_gltfScene->valid())
  {
    // Multiple scenes
    if(m_gltfScene->getModel().scenes.size() > 1)
    {
      if(headerManager.beginHeader("Multiple Scenes"))
      {
        ImGui::PushID("Scenes");
        for(size_t i = 0; i < m_gltfScene->getModel().scenes.size(); i++)
        {
          if(ImGui::RadioButton(m_gltfScene->getModel().scenes[i].name.c_str(), m_gltfScene->getCurrentScene() == i))
          {
            m_gltfScene->setCurrentScene(int(i));
            vkDeviceWaitIdle(resources.ctx.device);
            createVulkanScene(resources);
            postSceneCreateProcess(resources, m_gltfScene->getFilename());
            reset = true;
            setDirtyFlag(Scene::eNewScene, true);
          }
        }
        ImGui::PopID();
      }
    }

    // Variant selection
    if(m_gltfScene->getVariants().size() > 0)
    {
      if(headerManager.beginHeader("Variants"))
      {
        ImGui::PushID("Variants");
        for(size_t i = 0; i < m_gltfScene->getVariants().size(); i++)
        {
          if(ImGui::Selectable(m_gltfScene->getVariants()[i].c_str(), m_gltfScene->getCurrentVariant() == i))
          {
            m_gltfScene->setCurrentVariant(int(i));
            m_dirtyFlags.set(eVulkanScene);
            reset = true;
          }
        }
        ImGui::PopID();
      }
    }

    // Animation
    if(m_gltfScene->hasAnimation())
    {
      if(headerManager.beginHeader("Animation"))
      {
        m_animControl.onUI(m_gltfScene.get());
      }
    }


    if(m_sceneGraph && headerManager.beginHeader("Scene Graph"))
    {
      int selectedNode = m_sceneGraph->selectedNode();
      m_sceneGraph->render();

      // Find the `render node` corresponding to the selected node
      // The `render node` is the node that is rendered, and different from the `scene node`
      if(m_sceneGraph->selectedNode() > -1 && selectedNode != m_selectedRenderNode)
      {
        selectedNode      = m_sceneGraph->selectedNode();
        auto& renderNodes = m_gltfScene->getRenderNodes();
        for(size_t i = 0; i < renderNodes.size(); i++)
        {
          if(renderNodes[i].refNodeID == selectedNode)
          {
            m_selectedRenderNode = int(i);
            break;
          }
        }
      }
      else if(selectedNode == -1)
      {
        m_selectedRenderNode = -1;  // No node selected
      }

      // Check for scene graph changes
      bool transformChanged  = m_sceneGraph->hasTransformChanged();
      bool lightChanged      = m_sceneGraph->hasLightChanged();
      bool visibilityChanged = m_sceneGraph->hasVisibilityChanged();
      bool materialChanged   = m_sceneGraph->hasMaterialChanged();

      if(m_sceneGraph->hasMaterialFlagChanges())
      {
        m_dirtyFlags.set(eRtxScene);
        reset = true;
      }

      if(transformChanged || lightChanged || visibilityChanged)
      {
        m_dirtyFlags.set(eVulkanScene);
        m_dirtyFlags.set(eRtxScene);

        if(visibilityChanged)
          m_dirtyFlags.set(eNodeVisibility);

        m_gltfScene->updateRenderNodes();
        reset = true;
      }

      if(materialChanged)
      {
        m_dirtyFlags.set(eVulkanMaterial);
        reset = true;
      }

      m_sceneGraph->resetChanges();
    }


    if(headerManager.beginHeader("Statistics"))
    {
      const tinygltf::Model& tiny = m_gltfScene->getModel();
      PE::begin("Stat_Val");
      PE::Text("Nodes", std::to_string(tiny.nodes.size()));
      PE::Text("Render Nodes", std::to_string(m_gltfScene->getRenderNodes().size()));
      PE::Text("Render Primitives", std::to_string(m_gltfScene->getNumRenderPrimitives()));
      PE::Text("Materials", std::to_string(tiny.materials.size()));
      PE::Text("Triangles", std::to_string(m_gltfScene->getNumTriangles()));
      PE::Text("Lights", std::to_string(tiny.lights.size()));
      PE::Text("Textures", std::to_string(tiny.textures.size()));
      PE::Text("Images", std::to_string(tiny.images.size()));
      PE::end();
    }


    if(reset)
    {
      resetFrameCount();
    }
  }

  return reset;
}
