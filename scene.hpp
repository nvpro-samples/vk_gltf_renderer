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


#include <array>
#include <nvmath/nvmath.h>

#include "nvvkpp/commands_vkpp.hpp"
#include "nvvkpp/debug_util_vkpp.hpp"
#include "nvvkpp/utilities_vkpp.hpp"
#include "skydome.hpp"
#include <nvh/gltfscene.hpp>
#include <nvvk/memorymanagement_vk.hpp>
#include <nvvkpp/appbase_vkpp.hpp>
#include <nvvkpp/axis_vkpp.hpp>

struct gltfScene : nvh::gltf::Scene
{
  std::vector<vk::DescriptorSet>       m_materialDSets;
  std::vector<vk::DescriptorImageInfo> m_textureDescriptors;

  void getMaterials(tinygltf::Model& gltfModel)
  {
    Scene::loadMaterials(gltfModel);
    m_textureDescriptors.resize(m_numTextures);
    m_materialDSets.resize(m_materials.size());
  }

  vk::DescriptorImageInfo& getDescriptor(nvh::gltf::TextureIDX idx) { return m_textureDescriptors[idx]; }
};

//--------------------------------------------------------------------------------------------------
// Simple example showing a cube, camera movement and post-process
//
class VkScene : public nvvkpp::AppBase
{
  using nvvkAlloc    = nvvkpp::AllocatorDma;
  using nvvkMemAlloc = nvvk::DeviceMemoryAllocator;

public:
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t graphicsQueueIndex) override
  {
    AppBase::setup(device, physicalDevice, graphicsQueueIndex);


    m_memAllocator.init(device, physicalDevice);
    m_alloc.init(device, &m_memAllocator);

    m_cmdBufs.setup(device, graphicsQueueIndex);
    m_debug.setup(device);
    m_skydome.setup(device, physicalDevice, graphicsQueueIndex, m_memAllocator);
  }


  void initExample();
  void display();


  void destroy() override;
  void onResize(int w, int h) override;
  void createRenderPass() override;
  void onKeyboardChar(unsigned char key, int mods, int x, int y) override;
  void onKeyboard(NVPWindow::KeyCode key, ButtonAction action, int mods, int x, int y) override;

  float         getDepth(int x, int y);
  nvmath::vec3f unProjectScreenPosition(const nvmath::vec3f& screenPos);


  struct SceneUBO
  {
    nvmath::mat4f projection;
    nvmath::mat4f model;
    nvmath::vec4f cameraPosition = {0.f, 0.f, 0.f};
    nvmath::vec4f lightDirection = {1.f, 1.f, 1.f};
    float         lightIntensity = 1.0f;
    float         exposure       = 0.5f;
    float         gamma          = 2.2f;
    int           materialMode   = 0;
    int           tonemap        = 1;
    float         envIntensity   = 0.2f;
  } m_ubo;

  struct PushC
  {
    nvmath::vec4f albedo;
    nvmath::vec3f emissiveFactor  = {0, 0, 0};
    float         metallicFactor  = 1.0f;
    float         roughnessFactor = 1.0f;
    float         alphaCutoff     = 0.0f;
    int           alphamode{0};
  };


  enum
  {
    eScene,
    eMatrix,
    eMaterial,
    eEnv,
    NB_DSET
  };

  std::vector<std::vector<vk::DescriptorSetLayoutBinding>> m_descSetLayoutBind{NB_DSET};
  std::vector<vk::DescriptorSetLayout>                     m_descSetLayout{NB_DSET};
  std::vector<vk::DescriptorPool>                          m_descPool{NB_DSET};
  std::vector<vk::DescriptorSet>                           m_descSet{NB_DSET};

  vk::PipelineLayout m_pipelineLayout    = {};
  vk::Pipeline       m_drawPipeline      = {};
  vk::CommandBuffer  m_recordedCmdBuffer = {};

  nvmath::vec4f m_clearColor = nvmath::vec4f(0.07f, 0.07f, 0.07f, 1.f);

  // GLTF scene model
  gltfScene             m_gltfScene;
  nvvkpp::TextureDma    m_emptyTexture[2];
  nvh::gltf::VertexData m_vertices;
  std::vector<uint32_t> m_indices;


  VkScene()          = default;
  virtual ~VkScene() = default;

  void setScene(const std::string& filename);
  void setEnvironmentHdr(const std::string& hdrFilename);


private:
  void setupDescriptorSetLayout();
  void setupDescriptorSets();
  void createEmptyTexture();
  void prepareUniformBuffers();
  void preparePipelines();
  void recordCommandBuffer();
  void render(const vk::CommandBuffer& cmdBuff);
  void updateUniformBuffer(const vk::CommandBuffer& cmdBuffer);
  void drawUI();
  void loadImages(tinygltf::Model& gltfModel);

  std::string    m_filename;
  std::string    m_hdrFilename;
  int            m_upVector = 1;
  nvvkpp::AxisVK m_axis;

  vk::RenderPass m_renderPassSky;
  vk::RenderPass m_renderPassUI;

  nvvkpp::BufferDma  m_sceneBuffer;
  nvvkpp::BufferDma  m_vertexBuffer;
  nvvkpp::BufferDma  m_normalBuffer;
  nvvkpp::BufferDma  m_colorBuffer;
  nvvkpp::BufferDma  m_uvBuffer;
  nvvkpp::BufferDma  m_indexBuffer;
  nvvkpp::BufferDma  m_matrixBuffer;
  nvvkpp::BufferDma  m_pixelBuffer;  // Picking
  nvvk::AllocationID m_pixelAlloc;

  std::vector<nvvkpp::TextureDma> m_textures;

  SkydomePbr m_skydome;

  nvvkMemAlloc m_memAllocator{};
  nvvkAlloc    m_alloc;

  nvvkpp::MultipleCommandBuffers m_cmdBufs;
  nvvkpp::DebugUtil              m_debug;
};
