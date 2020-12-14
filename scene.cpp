/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


//--------------------------------------------------------------------------------------------------
// This example is loading a glTF scene and renders it with a very simple material
//

#include <vulkan/vulkan.hpp>

// Only once - defining VMA functions
#define VMA_IMPLEMENTATION

#include <iostream>

#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/profiler_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "scene.hpp"


// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include <fileformats/tiny_gltf.h>


#include "imgui/imgui_orient.h"
#include "imgui_impl_glfw.h"
#include "nvh/fileoperations.hpp"
#include "shaders/gltf.glsl"

nvvk::ProfilerVK g_profilerVK;
struct stats
{
  double loadScene{0};
  double scenePrep{0};
  double record{0};
} s_stats;


extern std::vector<std::string> defaultSearchPaths;

void VkScene::setup(const vk::Instance& instance, const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t graphicsQueueIndex)
{
  AppBase::setup(instance, device, physicalDevice, graphicsQueueIndex);

#ifdef NVVK_ALLOC_DEDICATED
  m_alloc.init(device, physicalDevice);
#elif defined(NVVK_ALLOC_DMA)
  m_memAllocator.init(device, physicalDevice);
  m_alloc.init(device, physicalDevice, &m_memAllocator);
#elif defined(NVVK_ALLOC_VMA)
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.physicalDevice         = physicalDevice;
  allocatorInfo.device                 = device;
  allocatorInfo.instance               = instance;
  vmaCreateAllocator(&allocatorInfo, &m_memAllocator);

  m_alloc.init(device, physicalDevice, m_memAllocator);
#endif

  m_debug.setup(device);
  m_skydome.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
}

//--------------------------------------------------------------------------------------------------
// Overridden function that is called after the base class create()
//
void VkScene::initExample()
{
  g_profilerVK.init(m_device, m_physicalDevice);

  // Loading the glTF file, it will allocate 3 buffers: vertex, index and matrices
  tinygltf::Model    gltfModel;
  tinygltf::TinyGLTF gltfContext;
  std::string        warn, error;
  bool               fileLoaded = false;
  {
    LOGI("Loading Scene: %s ", m_filename.c_str());
    s_stats.loadScene = -g_profilerVK.getMicroSeconds();
    fileLoaded        = gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warn, m_filename);
    if(!error.empty())
    {
      throw std::runtime_error(error.c_str());
    }

    s_stats.loadScene += g_profilerVK.getMicroSeconds();
    LOGI(" (%f s)\n", s_stats.loadScene / 1000000);
  }

  {
    s_stats.scenePrep = -g_profilerVK.getMicroSeconds();
    if(fileLoaded)
    {
      auto t = g_profilerVK.getMicroSeconds();
      LOGI("glTF to Vulkan");
      m_gltfScene.importMaterials(gltfModel);
      m_gltfScene.importDrawableNodes(gltfModel, nvh::GltfAttributes::Normal | nvh::GltfAttributes::Color_0
                                                     | nvh::GltfAttributes::Texcoord_0 | nvh::GltfAttributes::Tangent);
      m_gltfScene.computeSceneDimensions();
      LOGI(" (%f ms)\n", (g_profilerVK.getMicroSeconds() - t) / 1000);
      importImages(gltfModel);
    }

    // Set the camera as to see the model
    fitCamera(m_gltfScene.m_dimensions.min, m_gltfScene.m_dimensions.max);

    // Light Direction
    m_ubo.lightDirection = {-.5f, -.35f, -.75f};

    prepareUniformBuffers();
    setupDescriptorSetLayout();
    preparePipelines();

    s_stats.scenePrep += g_profilerVK.getMicroSeconds();
  }

  // Other elements
  m_skydome.loadEnvironment(m_hdrFilename);
  m_skydome.create({m_sceneBuffer.buffer, 0, VK_WHOLE_SIZE}, m_renderPassSky);
  m_axis.init(m_device, m_renderPass, 0, 40.f);

  setupDescriptorSets();
  recordCommandBuffer();

  m_alloc.finalizeAndReleaseStaging();
}

//--------------------------------------------------------------------------------------------------
// Overridden function called on shutdown
//
void VkScene::destroy()
{
  m_device.waitIdle();

  m_gltfScene.destroy();

  m_alloc.destroy(m_colorBuffer);
  m_alloc.destroy(m_indexBuffer);
  m_alloc.destroy(m_matrixBuffer);
  m_alloc.destroy(m_normalBuffer);
  m_alloc.destroy(m_uvBuffer);
  m_alloc.destroy(m_vertexBuffer);
  m_alloc.destroy(m_pixelBuffer);
  m_alloc.destroy(m_sceneBuffer);

  g_profilerVK.deinit();

  m_device.destroyRenderPass(m_renderPassUI);
  m_device.destroyRenderPass(m_renderPassSky);


  m_device.destroyPipeline(m_drawPipeline);
  m_device.destroyPipelineLayout(m_pipelineLayout);
  for(int i = 0; i < NB_DSET; i++)
  {
    m_device.destroyDescriptorSetLayout(m_descSetLayout[i]);
    m_device.destroyDescriptorPool(m_descPool[i]);
  }
  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

  m_axis.deinit();
  m_skydome.destroy();
  m_alloc.deinit();

#if defined(NVVK_ALLOC_DMA)
  m_memAllocator.deinit();
#elif defined(NVVK_ALLOC_VMA)
  vmaDestroyAllocator(m_memAllocator);
#endif

  AppBase::destroy();
}


//--------------------------------------------------------------------------------
// Called at each frame, as fast as possible
//
void VkScene::display()
{
  g_profilerVK.beginFrame();

  drawUI();

  // render the scene
  prepareFrame();
  const vk::CommandBuffer& cmdBuff = m_commandBuffers[getCurFrame()];
  cmdBuff.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  std::string name("Render-" + std::to_string(getCurFrame()));
  m_debug.setObjectName(cmdBuff, name.c_str());
  auto dbgLabel = m_debug.scopeLabel(cmdBuff, "Start rendering");

  // Updating the matrices of the camera
  updateUniformBuffer(cmdBuff);

  vk::ClearValue clearValues[2];
  clearValues[0].setColor(std::array<float, 4>({0.1f, 0.1f, 0.4f, 0.f}));
  clearValues[1].setDepthStencil({1.0f, 0});

  {
    auto scopeTime = g_profilerVK.timeRecurring("frame", cmdBuff);

    // Skybox
    vk::RenderPassBeginInfo renderPassBeginInfo{m_renderPassSky, m_framebuffers[getCurFrame()], {{}, m_size}, 2, clearValues};
    {
      auto dbgLabel = m_debug.scopeLabel(cmdBuff, "Skybox");
      cmdBuff.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
      setViewport(cmdBuff);
      m_skydome.draw(cmdBuff);
      cmdBuff.endRenderPass();
    }

    // Pre-recorded scene
    {
      auto dbgLabel = m_debug.scopeLabel(cmdBuff, "Recorded Scene");
      renderPassBeginInfo.setRenderPass(m_renderPass);
      cmdBuff.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eSecondaryCommandBuffers);
      cmdBuff.executeCommands(m_recordedCmdBuffer);
      cmdBuff.endRenderPass();
    }


    // Rendering UI

    {
      auto dbgLabel = m_debug.scopeLabel(cmdBuff, "Rendering UI");
      renderPassBeginInfo.setRenderPass(m_renderPassUI);
      cmdBuff.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
      ImGui::RenderDrawDataVK(cmdBuff, ImGui::GetDrawData());
    }

    // Rendering axis in same render pass
    dbgLabel.setLabel("Drawing Axis");
    m_axis.display(cmdBuff, CameraManip.getMatrix(), m_size);
    cmdBuff.endRenderPass();
  }

  // End of the frame and present the one which is ready
  cmdBuff.end();
  submitFrame();

  g_profilerVK.endFrame();
}


//--------------------------------------------------------------------------------------------------
// Building the command buffer, is in fact, recording all the calls  needed to draw the frame in a
// command buffer.This need to be  call only if the number of objects in the scene is changing or
// if the viewport is changing
//
void VkScene::recordCommandBuffer()
{
  s_stats.record = -g_profilerVK.getMicroSeconds();

  if(m_recordedCmdBuffer)
  {
    m_device.freeCommandBuffers(m_cmdPool, {m_recordedCmdBuffer});
  }

  m_recordedCmdBuffer = m_device.allocateCommandBuffers({m_cmdPool, vk::CommandBufferLevel::eSecondary, 1})[0];

  vk::CommandBufferInheritanceInfo inheritance_info{m_renderPass};
  vk::CommandBufferBeginInfo       begin_info{vkCB::eSimultaneousUse | vkCB::eRenderPassContinue, &inheritance_info};

  m_recordedCmdBuffer.begin(begin_info);
  {
    setViewport(m_recordedCmdBuffer);
    render(m_recordedCmdBuffer);
  }
  m_recordedCmdBuffer.end();

  s_stats.record += g_profilerVK.getMicroSeconds();
}


//--------------------------------------------------------------------------------------------------
// Creating the Uniform Buffers, only for the scene camera matrices
// The one holding all all matrices of the scene nodes was created in glTF.load()
//
void VkScene::prepareUniformBuffers()
{
  nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
  auto              cmdBuf = genCmdBuf.createCommandBuffer();
  {

    vk::BufferCreateInfo info{{}, sizeof(SceneUBO), vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst};
    m_sceneBuffer = m_alloc.createBuffer(info);

    // Creating the GPU buffer of the vertices
    m_vertexBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_positions, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
    m_normalBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_normals, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
    m_colorBuffer  = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_colors0, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
    m_uvBuffer     = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_texcoords0, vkBU::eVertexBuffer | vkBU::eStorageBuffer);


    // Creating the GPU buffer of the indices
    m_indexBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_indices, vkBU::eIndexBuffer | vkBU::eStorageBuffer);


    // Adding all matrices of the scene in a single buffer
    std::vector<NodeMatrices> allMatrices;
    allMatrices.reserve(m_gltfScene.m_nodes.size());
    for(auto& node : m_gltfScene.m_nodes)
    {
      NodeMatrices nm;
      nm.world   = node.worldMatrix;
      nm.worldIT = nm.world;
      nm.worldIT = nvmath::transpose(nvmath::invert(nm.worldIT));
      allMatrices.push_back(nm);
    }
    m_matrixBuffer = m_alloc.createBuffer(cmdBuf, allMatrices, vkBU::eStorageBuffer);
  }

  m_pixelBuffer = m_alloc.createBuffer(4 * sizeof(float), vkBU::eUniformBuffer | vkBU::eTransferDst,
                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

  m_debug.setObjectName(m_sceneBuffer.buffer, "SceneUBO");
  m_debug.setObjectName(m_vertexBuffer.buffer, "Vertex");
  m_debug.setObjectName(m_indexBuffer.buffer, "Index");
  m_debug.setObjectName(m_normalBuffer.buffer, "Normal");
  m_debug.setObjectName(m_colorBuffer.buffer, "Color");
  m_debug.setObjectName(m_matrixBuffer.buffer, "Matrix");
  m_debug.setObjectName(m_pixelBuffer.buffer, "Pixel");

  genCmdBuf.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void VkScene::preparePipelines()
{
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_renderPass);
  gpb.depthStencilState.depthTestEnable = true;

  gpb.addShader(nvh::loadFile("shaders/vert_shader.vert.spv", true, paths), vk::ShaderStageFlagBits::eVertex);
  gpb.addShader(nvh::loadFile("shaders/metallic-roughness.frag.spv", true, paths), vk::ShaderStageFlagBits::eFragment);
  gpb.addBindingDescriptions(
      {{0, sizeof(nvmath::vec3)}, {1, sizeof(nvmath::vec3)}, {2, sizeof(nvmath::vec4)}, {3, sizeof(nvmath::vec2)}});
  gpb.addAttributeDescriptions({{0, 0, vk::Format::eR32G32B32Sfloat, 0},     // Position
                                {1, 1, vk::Format::eR32G32B32Sfloat, 0},     // Normal
                                {2, 2, vk::Format::eR32G32B32A32Sfloat, 0},  // Color
                                {3, 3, vk::Format::eR32G32Sfloat, 0}});      // UV
  gpb.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_drawPipeline = gpb.createPipeline();

  m_debug.setObjectName(m_drawPipeline, "ShadingPipeline");
  m_debug.setObjectName(gpb.getShaderModule(0), "VertexShader");
  m_debug.setObjectName(gpb.getShaderModule(1), "FragmentShader");
}


//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void VkScene::setupDescriptorSetLayout()
{
  m_descSetLayoutBind[eScene].addBinding(vk::DescriptorSetLayoutBinding(0, vkDT::eUniformBuffer, 1, vkSS::eVertex | vkSS::eFragment));
  m_descSetLayout[eScene] = m_descSetLayoutBind[eScene].createLayout(m_device);
  m_descPool[eScene]      = m_descSetLayoutBind[eScene].createPool(m_device, 1);
  m_descSet[eScene]       = nvvk::allocateDescriptorSet(m_device, m_descPool[eScene], m_descSetLayout[eScene]);
  m_debug.setObjectName(m_descSet[eScene], "Scene Desc");

  m_descSetLayoutBind[eMatrix].addBinding(vk::DescriptorSetLayoutBinding(0, vkDT::eStorageBufferDynamic, 1, vkSS::eVertex));
  m_descSetLayout[eMatrix] = m_descSetLayoutBind[eMatrix].createLayout(m_device);
  m_descPool[eMatrix]      = m_descSetLayoutBind[eMatrix].createPool(m_device, 1);
  m_descSet[eMatrix]       = nvvk::allocateDescriptorSet(m_device, m_descPool[eMatrix], m_descSetLayout[eMatrix]);
  m_debug.setObjectName(m_descSet[eScene], "Matrices Desc");


  auto nbTextures = static_cast<uint32_t>(m_textures.size());
  m_descSetLayoutBind[eMaterial].addBinding(
      vk::DescriptorSetLayoutBinding(0, vkDT::eCombinedImageSampler, nbTextures, vkSS::eFragment));  // textures
  m_descSetLayout[eMaterial] = m_descSetLayoutBind[eMaterial].createLayout(m_device);
  m_descPool[eMaterial] =
      m_descSetLayoutBind[eMaterial].createPool(m_device, static_cast<uint32_t>(m_gltfScene.m_materials.size()));
  m_descSet[eMaterial] = nvvk::allocateDescriptorSet(m_device, m_descPool[eMaterial], m_descSetLayout[eMaterial]);

  m_descSetLayoutBind[eEnv].addBinding(vk::DescriptorSetLayoutBinding(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));  // irradiance
  m_descSetLayoutBind[eEnv].addBinding(vk::DescriptorSetLayoutBinding(1, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));  // brdfLut
  m_descSetLayoutBind[eEnv].addBinding(vk::DescriptorSetLayoutBinding(2, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));  // prefilteredMap
  m_descSetLayout[eEnv] = m_descSetLayoutBind[eEnv].createLayout(m_device);
  m_descPool[eEnv]      = m_descSetLayoutBind[eEnv].createPool(m_device, 1);
  m_descSet[eEnv]       = nvvk::allocateDescriptorSet(m_device, m_descPool[eEnv], m_descSetLayout[eEnv]);
  m_debug.setObjectName(m_descSet[eEnv], "Env Desc");

  // Push constants in the fragment shader
  vk::PushConstantRange pushConstantRanges = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(GltfShadeMaterial)};

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(m_descSetLayout.size()));
  pipelineLayoutCreateInfo.setPSetLayouts(m_descSetLayout.data());
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);

  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);
}

//--------------------------------------------------------------------------------------------------
// The layout of the descriptor is done, but we have to allocate a descriptor from this template
// a set the pBufferInfo to the buffer descriptor that was previously allocated (see prepareUniformBuffers)
//
void VkScene::setupDescriptorSets()
{
  vk::DescriptorBufferInfo dbiScene{m_sceneBuffer.buffer, 0, VK_WHOLE_SIZE};
  vk::DescriptorBufferInfo dbiMatrix{m_matrixBuffer.buffer, 0, VK_WHOLE_SIZE};

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(m_descSetLayoutBind[eScene].makeWrite(m_descSet[eScene], 0, &dbiScene));
  writes.emplace_back(m_descSetLayoutBind[eMatrix].makeWrite(m_descSet[eMatrix], 0, &dbiMatrix));

  // Textures
  std::vector<vk::DescriptorImageInfo> dbiImages;
  for(const auto& imageDesc : m_textures)
    dbiImages.emplace_back(imageDesc.descriptor);

  for(int i = 0; i < dbiImages.size(); i++)
    writes.emplace_back(m_descSet[eMaterial], 0, i, 1, vk::DescriptorType::eCombinedImageSampler, &dbiImages[i]);


  writes.emplace_back(m_descSetLayoutBind[eEnv].makeWrite(m_descSet[eEnv], 0, &m_skydome.m_textures.prefilteredCube.descriptor));
  writes.emplace_back(m_descSetLayoutBind[eEnv].makeWrite(m_descSet[eEnv], 1, &m_skydome.m_textures.lutBrdf.descriptor));
  writes.emplace_back(m_descSetLayoutBind[eEnv].makeWrite(m_descSet[eEnv], 2, &m_skydome.m_textures.irradianceCube.descriptor));


  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Rendering all glTF nodes
//
void VkScene::render(const vk::CommandBuffer& cmdBuff)
{
  if(!m_drawPipeline)
  {
    return;
  }

  m_debug.setObjectName(cmdBuff, "Recored");
  auto dgbLabel = m_debug.scopeLabel(cmdBuff, "Recording Scene");

  // Pipeline to use for rendering the current scene
  cmdBuff.bindPipeline(vk::PipelineBindPoint::eGraphics, m_drawPipeline);

  // Offsets for the descriptor set and vertex buffer
  std::vector<vk::DeviceSize> offsets = {0, 0, 0, 0};
  std::vector<uint32_t>       doffset = {0};

  // Keeping track of the last material to avoid binding them again
  uint32_t lastMaterial = -1;

  std::vector<vk::Buffer> vertexBuffers = {m_vertexBuffer.buffer, m_normalBuffer.buffer, m_colorBuffer.buffer,
                                           m_uvBuffer.buffer};
  cmdBuff.bindVertexBuffers(0, static_cast<uint32_t>(vertexBuffers.size()), vertexBuffers.data(), offsets.data());
  cmdBuff.bindIndexBuffer(m_indexBuffer.buffer, 0, vk::IndexType::eUint32);

  uint32_t idxNode = 0;
  for(auto& node : m_gltfScene.m_nodes)
  {
    // The offset for the matrix descriptor set, such that it points to the matrix of the gltfNode
    doffset[0] = static_cast<uint32_t>(idxNode * sizeof(NodeMatrices));


    auto dgbLabel = m_debug.scopeLabel(cmdBuff, std::string("Draw Mesh: " + std::to_string(node.primMesh)).c_str());

    auto& primitive = m_gltfScene.m_primMeshes[node.primMesh];

    if(lastMaterial != primitive.materialIndex)
    {
      lastMaterial = primitive.materialIndex;
      nvh::GltfMaterial& m(m_gltfScene.m_materials[lastMaterial]);

      GltfShadeMaterial shadeMat = GltfShadeMaterial{m.shadingModel,
                                                     m.pbrBaseColorFactor,
                                                     m.pbrBaseColorTexture,
                                                     m.pbrMetallicFactor,
                                                     m.pbrRoughnessFactor,
                                                     m.pbrMetallicRoughnessTexture,
                                                     m.khrDiffuseFactor,
                                                     m.khrDiffuseTexture,
                                                     m.khrSpecularFactor,
                                                     m.khrGlossinessFactor,
                                                     m.khrSpecularGlossinessTexture,
                                                     m.emissiveTexture,
                                                     m.emissiveFactor,
                                                     m.alphaMode,
                                                     m.alphaCutoff,
                                                     m.doubleSided,
                                                     m.normalTexture,
                                                     m.normalTextureScale,
                                                     m.occlusionTexture,
                                                     m.occlusionTextureStrength

      };
      cmdBuff.pushConstants<GltfShadeMaterial>(m_pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, shadeMat);
    }

    // The pipeline uses four descriptor set, one for the scene information, one for the matrix of the instance, one for the textures and for the environment
    std::vector<vk::DescriptorSet> descriptorSets = {m_descSet[eScene], m_descSet[eMatrix], m_descSet[eMaterial], m_descSet[eEnv]};
    cmdBuff.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, descriptorSets, doffset);

    cmdBuff.drawIndexed(primitive.indexCount, 1, primitive.firstIndex, primitive.vertexOffset, 0);


    idxNode++;
  }
}

//--------------------------------------------------------------------------------------------------
// When the frames are redone, we also need to re-record the command buffer
//
void VkScene::onResize(int w, int h)
{
  recordCommandBuffer();
}

//--------------------------------------------------------------------------------------------------
// Setting which scene to load. Check arguments in main.cpp
//
void VkScene::setScene(const std::string& filename)
{
  m_filename = filename;
}

void VkScene::setEnvironmentHdr(const std::string& hdrFilename)
{
  m_hdrFilename = hdrFilename;
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void VkScene::updateUniformBuffer(const vk::CommandBuffer& cmdBuffer)
{
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);
  float       nearPlane   = m_gltfScene.m_dimensions.radius / 100.0f;
  float       farPlane    = m_gltfScene.m_dimensions.radius * 50.0f;

  m_ubo.model      = CameraManip.getMatrix();
  m_ubo.projection = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, nearPlane, farPlane);
  nvmath::vec3f pos, center, up;
  CameraManip.getLookat(pos, center, up);
  m_ubo.cameraPosition = pos;

  auto dbgLabel = m_debug.scopeLabel(cmdBuffer, "Update Uniform Buffer");
  cmdBuffer.updateBuffer<VkScene::SceneUBO>(m_sceneBuffer.buffer, 0, m_ubo);
}

//--------------------------------------------------------------------------------------------------
// The display will render the recorded command buffer, then in a sub-pass, render the UI
//
void VkScene::createRenderPass()
{
  m_renderPass    = nvvk::createRenderPass(m_device, {getColorFormat()}, m_depthFormat, 1, false, true,
                                        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eColorAttachmentOptimal);
  m_renderPassSky = nvvk::createRenderPass(m_device, {getColorFormat()}, m_depthFormat, 1, true, true,
                                           vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal);
  m_renderPassUI  = nvvk::createRenderPass(m_device, {getColorFormat()}, m_depthFormat, 1, false, false,
                                          vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);

  m_debug.setObjectName(m_renderPass, "General Render Pass");
  m_debug.setObjectName(m_renderPassSky, "Environment Render Pass");
  m_debug.setObjectName(m_renderPassUI, "UIRender Pass");
}

//--------------------------------------------------------------------------------------------------
// Overload callback when a key gets hit
//
void VkScene::onKeyboardChar(unsigned char key)
{
  AppBase::onKeyboardChar(key);

  if(key == 'f')
    fitCamera(m_gltfScene.m_dimensions.min, m_gltfScene.m_dimensions.max, false);

  if(key == GLFW_KEY_SPACE)
  {
    double x, y;
    glfwGetCursorPos(m_window, &x, &y);
    float z = getDepth(x, y);
    if(z < 1.0f)  // Not the background
    {
      nvmath::vec3f worldPos = unProjectScreenPosition({x, y, z});
      // Set the interest position
      nvmath::vec3f eye, center, up;
      CameraManip.getLookat(eye, center, up);
      CameraManip.setLookat(eye, worldPos, up, false);
    }
  }
}


//--------------------------------------------------------------------------------------------------
// IMGUI UI display
//
void VkScene::drawUI()
{
  static int e = m_upVector;

  // Start the Dear ImGui frame
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  ImGui::SetNextWindowBgAlpha(0.8);
  ImGui::SetNextWindowSize(ImVec2(450, 0), ImGuiCond_FirstUseEver);

  ImGui::Begin("Hello, Vulkan!", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::Text("%s", &m_physicalDevice.getProperties().deviceName[0]);

  if(ImGui::CollapsingHeader("Camera Up Vector"))
  {

    ImGui::RadioButton("X", &e, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Y", &e, 1);
    ImGui::SameLine();
    ImGui::RadioButton("Z", &e, 2);
    if(e != m_upVector)
    {
      nvmath::vec3f eye, center, up;
      CameraManip.getLookat(eye, center, up);
      CameraManip.setLookat(eye, center, nvmath::vec3f(e == 0, e == 1, e == 2));
      m_upVector = e;
    }
  }

  if(ImGui::CollapsingHeader("Lighting"))
  {
    ImVec3 vecDir(m_ubo.lightDirection.x, m_ubo.lightDirection.y, m_ubo.lightDirection.z);
    ImGui::DirectionGizmo("Directional Light", &m_ubo.lightDirection.x);
    ImGui::SliderFloat("Light Intensity", &m_ubo.lightIntensity, 0.0f, 10.f);
    ImGui::SliderFloat("Exposure", &m_ubo.exposure, 0.0f, 10.f);
    ImGui::SliderFloat("Gamma", &m_ubo.gamma, 1.0f, 2.2f);
    static const char* tmItem[] = {"Linear", "Uncharted 2", "Hejl Richard", "ACES"};
    ImGui::Combo("Tone Map", &m_ubo.tonemap, tmItem, 4);
    ImGui::SliderFloat("Environment Intensity", &m_ubo.envIntensity, 0.0f, 2.f, "%.3f", 2.f);
  }

  if(ImGui::CollapsingHeader("Debug"))
  {
    static const char* dbgItem[] = {"None",     "Metallic", "Normal", "Base Color", "Occlusion",
                                    "Emissive", "F0",       "Alpha",  "Roughness"};
    ImGui::Combo("Debug Mode", &m_ubo.materialMode, dbgItem, 9);
  }

  // For performance graph
  static float  valuesFPS[90] = {0};  // FPS value
  static float  valuesRnd[90] = {0};  // Rendering values in ms
  static int    values_offset = 0;    // Used for the graph
  static float  valueMax{0};          // Auto adjusting max
  static float  valueMSMax{0};        //  " " "
  static double perfTime = g_profilerVK.getMicroSeconds();


  // Get the average values every 50 frames
  if(g_profilerVK.getTotalFrames() % 50 == 49)
  {
    double                   frameCpu, frameGpu /*, dummy*/;
    nvh::Profiler::TimerInfo info;
    g_profilerVK.getTimerInfo("frame", info);
    frameCpu                 = info.cpu.average;
    frameGpu                 = info.gpu.average;
    double curTime           = g_profilerVK.getMicroSeconds();
    double diffTime          = curTime - perfTime;
    perfTime                 = curTime;
    valuesFPS[values_offset] = 50.0f / diffTime * 1000000.0f;
    valuesRnd[values_offset] = frameGpu;
    valueMax                 = std::min(std::max(valueMax, valuesFPS[values_offset]), 1000.0f);
    valueMSMax               = std::max(valueMSMax, valuesRnd[values_offset]);
    values_offset            = (values_offset + 1) % IM_ARRAYSIZE(valuesFPS);
  }

  if(ImGui::CollapsingHeader("Performance"))
  {
    // Displaying performance graph
    char strbuf[80];
    int  last = (values_offset - 1) % IM_ARRAYSIZE(valuesFPS);
    sprintf(strbuf, "Render\n%3.2fms", valuesRnd[last]);
    ImGui::PlotLines(strbuf, valuesRnd, IM_ARRAYSIZE(valuesFPS), values_offset, nullptr, 0.0f, valueMSMax, ImVec2(0, 80));
    sprintf(strbuf, "FPS\n%3.1f", valuesFPS[last]);
    ImGui::PlotLines(strbuf, valuesFPS, IM_ARRAYSIZE(valuesFPS), values_offset, nullptr, 0.0f, valueMax, ImVec2(0, 80));
    if(ImGui::TreeNode("Extra"))
    {
      ImGui::Text("Scene loading time:     %3.2f ms", s_stats.loadScene / 1000.0);
      ImGui::Text("Scene preparation time: %3.2f ms", s_stats.scenePrep / 1000.0);
      ImGui::Text("Scene recording time:   %3.2f ms", s_stats.record / 1000.0);
      ImGui::TreePop();
    }
  }

  if(ImGui::CollapsingHeader("Statistics"))
  {
    ImGui::Text("Nb instances  : %zu", m_gltfScene.m_nodes.size());
    ImGui::Text("Nb meshes     : %zu", m_gltfScene.m_primMeshes.size());
    ImGui::Text("Nb materials  : %zu", m_gltfScene.m_materials.size());
    ImGui::Text("Nb triangles  : %zu", m_gltfScene.m_indices.size() / 3);
  }

  ImGui::End();
  ImGui::Render();
}

//--------------------------------------------------------------------------------------------------
// Convert all images to textures
//
void VkScene::importImages(tinygltf::Model& gltfModel)
{
  if(gltfModel.images.empty())
  {
    // Make dummy image(1,1), needed as we cannot have an empty array
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
    std::array<uint8_t, 4>   white = {255, 255, 255, 255};
    m_textures.emplace_back(m_alloc.createTexture(cmdBuf, 4, white.data(), nvvk::makeImage2DCreateInfo(vk::Extent2D{1, 1}), {}));
    m_debug.setObjectName(m_textures[0].image, "dummy");
    return;
  }

  m_textures.resize(gltfModel.images.size());

  LOGI("Loading %d images ", gltfModel.images.size());
  auto t = g_profilerVK.getMicroSeconds();

  vk::SamplerCreateInfo samplerCreateInfo{{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
  vk::Format            format = vk::Format::eR8G8B8A8Unorm;
  samplerCreateInfo.maxLod     = FLT_MAX;

  // Get available Command Buffer
  nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
  vk::CommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();

  g_profilerVK.beginFrame();
  auto sectionID = g_profilerVK.beginSection("test", cmdBuf, true);
  for(size_t i = 0; i < gltfModel.images.size(); i++)
  {
    LOGI(".");
    auto&               gltfimage       = gltfModel.images[i];
    void*               buffer          = &gltfimage.image[0];
    vk::DeviceSize      bufferSize      = gltfimage.image.size();
    auto                imgSize         = vk::Extent2D(gltfimage.width, gltfimage.height);
    vk::ImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, vkIU::eSampled, true);

    // Creating an image with staging buffer
    nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
    nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
    vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    m_textures[i]                  = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

    std::string name(gltfimage.name.empty() ? "Txt" + std::to_string(i) : gltfimage.name);
    m_debug.setObjectName(m_textures[i].image, name.c_str());
  }

  g_profilerVK.endSection(sectionID, cmdBuf);
  g_profilerVK.endFrame();
  // Submit current command buffer
  genCmdBuf.submitAndWait(cmdBuf);
  // Adds all staging buffers to garbage collection, delete what it can
  m_alloc.finalizeAndReleaseStaging();

  double gpuTime{0.0};
  g_profilerVK.getSectionTime(sectionID, 0, gpuTime);


  LOGI("CPU (%f ms), GPU (%f ms)\n", (g_profilerVK.getMicroSeconds() - t) / 1000, gpuTime / 1000);
}

//--------------------------------------------------------------------------------------------------
// Overload keyboard hit
//
void VkScene::onKeyboard(int key, int scancode, int action, int mods)
{
  nvvk::AppBase::onKeyboard(key, scancode, action, mods);

  if(key == GLFW_KEY_HOME)
  {
    // Set the camera as to see the model
    fitCamera(m_gltfScene.m_dimensions.min, m_gltfScene.m_dimensions.max, false);
  }
  return;
}

//--------------------------------------------------------------------------------------------------
// Return the depth under the screen 2D position
//
float VkScene::getDepth(int x, int y)
{
  nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
  vk::CommandBuffer cmdBuf = genCmdBuf.createCommandBuffer();

  // Transit the depth buffer image in eTransferSrcOptimal
  vk::ImageSubresourceRange subresourceRange(vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1);
  nvvk::cmdBarrierImageLayout(cmdBuf, m_depthImage, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                              vk::ImageLayout::eTransferSrcOptimal, subresourceRange);

  // Copy the pixel under the cursor
  vk::BufferImageCopy copyRegion;
  copyRegion.setImageSubresource({vk::ImageAspectFlagBits::eDepth, 0, 0, 1});
  copyRegion.setImageOffset({x, y, 0});
  copyRegion.setImageExtent({1, 1, 1});
  cmdBuf.copyImageToBuffer(m_depthImage, vk::ImageLayout::eTransferSrcOptimal, m_pixelBuffer.buffer, {copyRegion});

  // Put back the depth buffer as  it was
  nvvk::cmdBarrierImageLayout(cmdBuf, m_depthImage, vk::ImageLayout::eTransferSrcOptimal,
                              vk::ImageLayout::eDepthStencilAttachmentOptimal, subresourceRange);
  genCmdBuf.submitAndWait(cmdBuf);

  // Grab the value
  void* mapped = m_alloc.map(m_pixelBuffer);
  float value;
  memcpy(&value, mapped, sizeof(float));
  m_alloc.unmap(m_pixelBuffer);


  return value;
}

//--------------------------------------------------------------------------------------------------
// Return the 3D position of the screen 2D + depth
//
nvmath::vec3f VkScene::unProjectScreenPosition(const nvmath::vec3f& screenPos)
{
  // Transformation of normalized coordinates between -1 and 1
  nvmath::vec4f winNorm;
  winNorm[0] = screenPos.x / (float)m_size.width * 2.0 - 1.0;
  winNorm[1] = screenPos.y / (float)m_size.height * 2.0 - 1.0;
  winNorm[2] = screenPos.z;
  winNorm[3] = 1.0;

  // Transform to world space
  nvmath::mat4f mat     = m_ubo.projection * m_ubo.model;
  nvmath::mat4f matInv  = nvmath::invert(mat);
  nvmath::vec4f worlPos = matInv * winNorm;
  worlPos.w             = 1.0f / worlPos.w;
  worlPos.x             = worlPos.x * worlPos.w;
  worlPos.y             = worlPos.y * worlPos.w;
  worlPos.z             = worlPos.z * worlPos.w;

  return worlPos;
}
