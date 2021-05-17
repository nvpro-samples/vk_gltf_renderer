/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */



#pragma once
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <vector>
#include <vulkan/vulkan.hpp>


#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "vkalloc.hpp"

// Load an environment image (HDR) and create the cubic textures for glossy reflection and diffuse illumination.
// Creates also the BRDF lookup table and an acceleration structure for lights
// It also has the ability to render a cube with the environment, use by the rasterizer.
class SkydomePbr
{
public:
  SkydomePbr() = default;

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator)
  {
    m_device     = device;
    m_alloc      = allocator;
    m_queueIndex = familyIndex;
    m_debug.setup(device);
  }

  void loadEnvironment(const std::string& hrdImage);
  void create(const vk::DescriptorBufferInfo& sceneBufferDesc, const vk::RenderPass& renderPass);
  void draw(const vk::CommandBuffer& commandBuffer);
  void destroy();

  struct Textures
  {
    nvvk::Texture txtHdr;
    nvvk::Texture lutBrdf;
    nvvk::Texture accelImpSmpl;
    nvvk::Texture irradianceCube;
    nvvk::Texture prefilteredCube;
  } m_textures;

  enum Descriptors
  {
    eScene,
    eMaterial
  };

  vk::DescriptorSet       m_descriptorSet[2];
  vk::DescriptorSetLayout m_descriptorSetLayout[2];
  vk::DescriptorPool      m_descriptorpool;
  vk::Pipeline            m_pipeline;
  vk::PipelineLayout      m_pipelineLayout;
  vk::RenderPass          m_renderPass;
  vk::Device              m_device;

private:
  nvvk::Buffer m_vertices;
  nvvk::Buffer m_indices;

  uint32_t         m_queueIndex{0};
  nvvk::ResourceAllocator* m_alloc{nullptr};
  nvvk::DebugUtil  m_debug;

  void createCube();
  void createEnvironmentAccelTexture(const float* pixels, vk::Extent2D& size, nvvk::Texture& accelTex);
  void integrateBrdf(uint32_t dim);
  void createPipelines(const vk::DescriptorBufferInfo& sceneBufferDesc);
  void prefilterDiffuse(uint32_t dim);
  void prefilterGlossy(uint32_t dim);
  void renderToCube(const vk::RenderPass& renderpass,
                    nvvk::Texture&        filteredEnv,
                    vk::PipelineLayout    pipelinelayout,
                    vk::Pipeline          pipeline,
                    vk::DescriptorSet     descSet,
                    uint32_t              dim,
                    vk::Format            format,
                    const uint32_t        numMips);

  struct Offscreen
  {
    nvvk::Image             image;
    vk::DescriptorImageInfo descriptor;
    vk::Framebuffer         framebuffer;
  };

  Offscreen createOffscreen(int dim, vk::Format format, const vk::RenderPass& renderpass);
};
