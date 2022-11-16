/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

//////////////////////////////////////////////////////////////////////////
/*

 This sample can load GLTF scene and render using the raster or RTX (path tracer)

*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <filesystem>
#include <thread>
#include <vulkan/vulkan_core.h>

#include <glm/glm.hpp>

#define VMA_IMPLEMENTATION
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvh/commandlineparser.hpp"
#include "nvp/nvpsystem.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/gizmos_vk.hpp"
#include "nvvk/raypicker_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/gltf_scene.hpp"
#include "nvvkhl/gltf_scene_rtx.hpp"
#include "nvvkhl/gltf_scene_vk.hpp"
#include "nvvkhl/hdr_env.hpp"
#include "nvvkhl/hdr_env_dome.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/scene_camera.hpp"
#include "nvvkhl/shaders/dh_scn_desc.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/sky.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"

#include "_autogen/pathtrace.rahit.h"
#include "_autogen/pathtrace.rchit.h"
#include "_autogen/pathtrace.rgen.h"
#include "_autogen/pathtrace.rmiss.h"
#include "_autogen/raster.frag.h"
#include "_autogen/raster.vert.h"
#include "_autogen/raster_overlay.frag.h"
#include "shaders/device_host.h"
#include "shaders/dh_bindings.h"

#include "GLFW/glfw3.h"
#include "busy_window.hpp"
#include "nvh/fileoperations.hpp"

#define RASTER_SS_SIZE 2.0F

std::shared_ptr<nvvkhl::ElementCamera> g_elem_camera;

namespace {

//////////////////////////////////////////////////////////////////////////
/// </summary> Ray trace multiple primitives
class GltfViewer : public nvvkhl::IAppElement
{
  struct Settings
  {
    Settings()
    {
      // Default light
      lights.resize(1);
      lights[0] = defaultLight();
    }

    enum EnvSystem
    {
      eSky,
      eHdr,
    };

    enum RenderSystem
    {
      ePathtracer,
      eRaster,
    };

    int           maxFrames     = 200000;
    int           maxSamples    = 1;
    int           maxDepth      = 5;
    bool          showAxis      = true;
    bool          showWireframe = false;
    EnvSystem     envSystem     = eSky;
    RenderSystem  renderSystem  = ePathtracer;
    nvmath::vec4f clearColor{1.F};
    float         envRotation = 0.F;

    std::vector<Light> lights;
  } m_settings;

public:
  GltfViewer()
  {
    m_frameInfo.nbLights     = 1;
    m_frameInfo.useSky       = 1;
    m_frameInfo.maxLuminance = 10.0F;
    m_frameInfo.clearColor   = nvmath::vec4f(1.F);
    m_frameInfo.dbgMethod    = eDbgMethod_none;
  };

  ~GltfViewer() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    auto scope_t = nvh::ScopedTimer("onAttach\n");

    m_app    = app;
    m_device = m_app->getDevice();

    const auto&    ctx             = app->getContext();
    const uint32_t gct_queue_index = ctx->m_queueGCT.familyIndex;
    const uint32_t t_queue_index   = ctx->m_queueT.familyIndex;
    const uint32_t c_queue_index   = ctx->m_queueC.familyIndex;

    m_dutil   = std::make_unique<nvvk::DebugUtil>(m_device);                                  // Debug utility
    m_alloc   = std::make_unique<nvvkhl::AllocVma>(m_app->getContext().get());                // Allocator
    m_scene   = std::make_unique<nvvkhl::Scene>();                                            // GLTF scene
    m_sceneVk = std::make_unique<nvvkhl::SceneVk>(m_app->getContext().get(), m_alloc.get());  // GLTF Scene buffers
    m_sceneRtx = std::make_unique<nvvkhl::SceneRtx>(m_app->getContext().get(), m_alloc.get(), c_queue_index);  // GLTF Scene BLAS/TLAS
    m_tonemapper = std::make_unique<nvvkhl::TonemapperPostProcess>(m_app->getContext().get(), m_alloc.get());
    m_sbt        = std::make_unique<nvvk::SBTWrapper>();  // Shader Binding Table
    m_sky        = std::make_unique<nvvkhl::SkyDome>(m_app->getContext().get(), m_alloc.get());
    m_picker     = std::make_unique<nvvk::RayPickerKHR>(m_app->getContext().get(), m_alloc.get(), c_queue_index);
    m_vkAxis     = std::make_unique<nvvk::AxisVK>();
    m_hdrEnv = std::make_unique<nvvkhl::HdrEnv>(m_app->getContext().get(), m_alloc.get(), c_queue_index);  // HDR Generic
    m_hdrDome = std::make_unique<nvvkhl::HdrEnvDome>(m_app->getContext().get(), m_alloc.get(), c_queue_index);  // HDR raster
    m_rtxSet   = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_sceneSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    // Create an extra queue for loading in parallel
    m_qGCT1 = ctx->createQueue(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, "GCT1", 1.0F);

    m_hdrEnv->loadEnvironment("");
    m_hdrDome->create(m_hdrEnv->getDescriptorSet(), m_hdrEnv->getDescriptorSetLayout());

    // Requesting ray tracing properties
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_prop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &rt_prop;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create the Shading Binding Table (SBT)
    m_sbt->setup(m_app->getDevice(), t_queue_index, m_alloc.get(), rt_prop);

    // Create resources
    createGbuffers(m_viewSize);
    createVulkanBuffers();

    // Axis in the bottom left corner
    nvvk::AxisVK::CreateAxisInfo ainfo;
    ainfo.colorFormat = {m_gBuffers->getColorFormat(0)};
    ainfo.depthFormat = m_gBuffers->getDepthFormat();
    m_vkAxis->init(m_device, ainfo);

    m_tonemapper->createComputePipeline();
  }

  void onDetach() override { destroyResources(); }

  void onResize(uint32_t width, uint32_t height) override
  {
    createGbuffers({width, height});
    // Tonemapper is using GBuffer-1 as input and output to GBuffer-0
    m_tonemapper->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(1), m_gBuffers->getDescriptorImageInfo(0));

    if(m_busy)
      return;

    writeRtxSet();
  }

  void onUIMenu() override
  {
    bool load_file{false};

    windowTitle();

    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Load", "Ctrl+O"))
      {
        load_file = true;
      }
      ImGui::Separator();
      ImGui::EndMenu();
    }

    if(m_busy)
      return;

    if(ImGui::IsKeyPressed(ImGuiKey_O) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
    {
      load_file = true;
    }

    if(load_file)
    {
      auto filename = NVPSystem::windowOpenFileDialog(m_app->getWindowHandle(), "Load glTF | HDR",
                                                      "glTF(.gltf, .glb), HDR(.hdr)|*.gltf;*.glb;*.hdr");
      onFileDrop(filename.c_str());
    }
  }

  void onFileDrop(const char* filename) override
  {
    if(m_busy)
      return;

    namespace fs = std::filesystem;
    vkDeviceWaitIdle(m_device);
    m_busy                  = true;
    const std::string tfile = filename;
    std::thread([&, tfile]() {
      const std::string extension = fs::path(tfile).extension().string();
      if(extension == ".gltf" || extension == ".glb")
      {
        createScene(tfile);
      }
      else if(extension == ".hdr")
      {
        createHdr(tfile);
        m_settings.envSystem = Settings::eHdr;
        resetFrame();
      }

      resetFrame();
      m_busy = false;
    }).detach();
  }

  void onUIRender() override
  {
    using PE = ImGuiH::PropertyEditor;

    bool reset{false};
    // Pick under mouse cursor
    if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
    {
      screenPicking();
    }
    if(ImGui::IsKeyPressed(ImGuiKey_R))  // Toggle renders
    {
      m_settings.renderSystem = m_settings.renderSystem == Settings::ePathtracer ? Settings::eRaster : Settings::ePathtracer;
      onResize(m_app->getViewportSize().width, m_app->getViewportSize().height);  // Force recreation of G-Buffers
    }

    {  // Setting menu
      ImGui::Begin("Settings");

      if(ImGui::CollapsingHeader("Camera"))
      {
        ImGuiH::CameraWidget();
      }

      if(ImGui::CollapsingHeader("Rendering"))
      {
        auto rs = m_settings.renderSystem;
        reset |= ImGui::RadioButton("RTX", reinterpret_cast<int*>(&m_settings.renderSystem), Settings::ePathtracer);
        ImGui::SameLine();
        reset |= ImGui::RadioButton("Raster", reinterpret_cast<int*>(&m_settings.renderSystem), Settings::eRaster);
        ImGui::SameLine();
        ImGui::TextDisabled("(R) Toggle render");
        if(rs != m_settings.renderSystem)
        {
          // Force recreation of G-Buffers because raster used 2x the size and display downscaled, making
          // cheap AA
          onResize(m_app->getViewportSize().width, m_app->getViewportSize().height);
        }

        PE::begin();
        if(PE::treeNode("Ray Tracing"))
        {
          reset |= PE::entry("Depth", [&] { return ImGui::SliderInt("#1", &m_settings.maxDepth, 1, 10); });
          reset |= PE::entry("Samples", [&] { return ImGui::SliderInt("#2", &m_settings.maxSamples, 1, 5); });
          reset |= PE::entry("Frames", [&] { return ImGui::DragInt("#3", &m_settings.maxFrames, 5.0F, 1, 1000000); });
          PE::treePop();
        }
        if(PE::treeNode("Raster"))
        {
          const bool b = m_settings.showWireframe;
          PE::entry("Show Wireframe", [&] { return ImGui::Checkbox("##4", &m_settings.showWireframe); });
          if(b != m_settings.showWireframe)
            freeRecordCommandBuffer();
          PE::treePop();
        }
        static const std::array<char*, 6> dbgItems = {"None",   "Metallic",   "Roughness",
                                                      "Normal", "Base Color", "Emissive"};
        reset |= PE::entry("Debug Method", [&] {
          return ImGui::Combo("##DebugMode", &m_frameInfo.dbgMethod, dbgItems.data(), int(dbgItems.size()));
        });
        PE::entry("Show Axis", [&] { return ImGui::Checkbox("##4", &m_settings.showAxis); });

        PE::end();
      }

      if(ImGui::CollapsingHeader("Environment"))
      {
        const bool sky_only = !(m_hdrEnv && m_hdrEnv->isValid());
        reset |= ImGui::RadioButton("Sky", reinterpret_cast<int*>(&m_settings.envSystem), Settings::eSky);
        ImGui::SameLine();
        ImGui::BeginDisabled(sky_only);
        reset |= ImGui::RadioButton("Hdr", reinterpret_cast<int*>(&m_settings.envSystem), Settings::eHdr);
        ImGui::EndDisabled();
        PE::begin();
        if(PE::treeNode("Sky"))
        {
          reset |= m_sky->onUI();
          PE::treePop();
        }
        ImGui::BeginDisabled(sky_only);
        if(PE::treeNode("Hdr"))
        {
          reset |= PE::entry(
              "Color", [&] { return ImGui::ColorEdit3("##Color", &m_settings.clearColor.x, ImGuiColorEditFlags_Float); },
              "Color multiplier");

          reset |= PE::entry(
              "Rotation", [&] { return ImGui::SliderAngle("Rotation", &m_settings.envRotation); }, "Rotating the environment");
          PE::treePop();
        }
        ImGui::EndDisabled();
        PE::end();
      }

      if(ImGui::CollapsingHeader("Tonemapper"))
      {
        m_tonemapper->onUI();
      }

      if(ImGui::CollapsingHeader("Statistics"))
      {
        if(m_scene->valid())
        {
          const auto& gltf = m_scene->scene();
          const auto& tiny = m_scene->model();
          ImGui::Text("Instances:  %zu", gltf.m_nodes.size());
          ImGui::Text("Mesh:  %zu", gltf.m_primMeshes.size());
          ImGui::Text("Materials:  %zu", gltf.m_materials.size());
          ImGui::Text("Triangles:  %zu", gltf.m_indices.size() / 3);
          ImGui::Text("Lights:  %zu", gltf.m_lights.size());
          ImGui::Text("Textures/Images:  %zu/%zu", tiny.textures.size(), tiny.images.size());
        }
      }

      ImGui::End();

      if(reset)
      {
        resetFrame();
      }
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }

    if(m_busy)
      showBusyWindow("Loading");
  }

  void onRender(VkCommandBuffer cmd) override
  {
    if(!m_scene->valid() || !updateFrame() || m_busy)
    {
      return;
    }

    auto scope_dbg = m_dutil->DBG_SCOPE(cmd);

    // Get camera info
    const float view_aspect_ratio = m_viewSize.x / m_viewSize.y;

    // Update Frame buffer uniform buffer
    const auto& clip    = CameraManip.getClipPlanes();
    m_frameInfo.view    = CameraManip.getMatrix();
    m_frameInfo.proj    = nvmath::perspectiveVK(CameraManip.getFov(), view_aspect_ratio, clip.x, clip.y);
    m_frameInfo.projInv = nvmath::inverse(m_frameInfo.proj);
    m_frameInfo.viewInv = nvmath::inverse(m_frameInfo.view);
    m_frameInfo.camPos  = CameraManip.getEye();
    if(m_settings.envSystem == Settings::eSky)
    {
      m_frameInfo.useSky   = 1;
      m_frameInfo.nbLights = static_cast<int>(m_settings.lights.size());
      m_frameInfo.light[0] = m_sky->getSun();
    }
    else
    {
      m_frameInfo.useSky      = 0;
      m_frameInfo.nbLights    = 0;
      m_frameInfo.envRotation = m_settings.envRotation;
      m_frameInfo.clearColor  = m_settings.clearColor;
    }
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(FrameInfo), &m_frameInfo);

    // Push constant
    m_pushConst.maxDepth   = m_settings.maxDepth;
    m_pushConst.maxSamples = m_settings.maxSamples;
    m_pushConst.frame      = m_frame;

    // Update the sky
    m_sky->updateParameterBuffer(cmd);

    if(m_settings.renderSystem == Settings::ePathtracer)
    {
      raytraceScene(cmd);
    }
    else
    {
      rasterScene(cmd);
    }

    // Apply tonemapper - take GBuffer-1 and output to GBuffer-0
    m_tonemapper->runCompute(cmd, m_gBuffers->getSize());

    // Render corner axis
    if(m_settings.showAxis)
    {
      renderAxis(cmd);
    }
  }

private:
  void createScene(const std::string& filename)
  {
    m_scene->load(filename);
    nvvkhl::setCameraFromScene(filename, m_scene->scene());               // Camera auto-scene-fitting
    g_elem_camera->setSceneRadius(m_scene->scene().m_dimensions.radius);  // Navigation help

    {  // Create the Vulkan side of the scene
      // Since we load and display simultaneously, we need to use a second GTC queue
      nvvk::CommandPool cmd_pool(m_device, m_qGCT1.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, m_qGCT1.queue);
      {
        VkCommandBuffer cmd = cmd_pool.createCommandBuffer();
        m_sceneVk->create(cmd, *m_scene);
        cmd_pool.submitAndWait(cmd);
        m_alloc->finalizeAndReleaseStaging();  // Make sure there are no pending staging buffers and clear them up
      }

      m_sceneRtx->create(*m_scene, *m_sceneVk);  // Create BLAS / TLAS

      m_picker->setTlas(m_sceneRtx->tlas());
    }

    // Find which nodes are solid or translucent
    const auto& gltf_scene = m_scene->scene();
    m_solidMatNodes.clear();
    m_blendMatNodes.clear();
    m_allNodes.clear();
    for(uint32_t i = 0; i < gltf_scene.m_nodes.size(); i++)
    {
      const auto  prim_mesh = gltf_scene.m_nodes[i].primMesh;
      const auto  mat_id    = gltf_scene.m_primMeshes[prim_mesh].materialIndex;
      const auto& mat       = gltf_scene.m_materials[mat_id];
      m_allNodes.push_back(i);
      if(mat.alphaMode == 0)
        m_solidMatNodes.push_back(i);
      else
        m_blendMatNodes.push_back(i);
    }

    // Need to record the scene
    freeRecordCommandBuffer();

    // Descriptor Set and Pipelines
    createSceneSet();
    createRtxSet();
    createRtxPipeline();  // must recreate due to texture changes
    writeSceneSet();
    writeRtxSet();
    createRasterPipeline();
  }

  void createGbuffers(const nvmath::vec2f& size)
  {
    static auto depth_format = nvvk::findDepthFormat(m_app->getPhysicalDevice());  // Not all depth are supported

    m_viewSize = size;

    // For raster we are rendering in a 2x image, which is making nice AA
    if(m_settings.renderSystem == Settings::eRaster)
    {
      m_viewSize *= RASTER_SS_SIZE;
    }

    // Two GBuffers: RGBA8 and RGBA32F, rendering to RGBA32F and tone mapped to RGBA8
    const std::vector<VkFormat> color_buffers = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32G32B32A32_SFLOAT};
    // Creation of the GBuffers
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(
        m_device, m_alloc.get(), VkExtent2D{static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)},
        color_buffers, depth_format);

    m_sky->setOutImage(m_gBuffers->getDescriptorImageInfo(1));
    m_hdrDome->setOutImage(m_gBuffers->getDescriptorImageInfo(1));

    // Indicate the renderer to reset its frame
    resetFrame();
    freeRecordCommandBuffer();
  }

  // Create all Vulkan buffer data
  void createVulkanBuffers()
  {
    auto* cmd = m_app->createTempCmdBuffer();

    // Create the buffer of the current frame, changing at each frame
    m_bFrameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  void createRtxSet()
  {
    m_rtxSet->deinit();

    // This descriptor set, holds the top level acceleration structure and the output image
    m_rtxSet->addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_rtxSet->addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_rtxSet->initLayout();
    m_rtxSet->initPool(1);
    m_dutil->DBG_NAME(m_rtxSet->getLayout());
    m_dutil->DBG_NAME(m_rtxSet->getSet());
  }

  void createSceneSet()
  {
    m_sceneSet->deinit();

    // This descriptor set, holds scene information and the textures
    m_sceneSet->addBinding(SceneBindings::eFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_sceneSet->addBinding(SceneBindings::eSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_sceneSet->addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_sceneVk->nbTextures(),
                           VK_SHADER_STAGE_ALL);
    m_sceneSet->initLayout();
    m_sceneSet->initPool(1);
    m_dutil->DBG_NAME(m_sceneSet->getLayout());
    m_dutil->DBG_NAME(m_sceneSet->getSet());
  }

  void createRasterPipeline()
  {
    m_rasterPipe.destroy(m_device);

    // Creating the Pipeline Layout
    std::vector<VkDescriptorSetLayout> layouts{m_sceneSet->getLayout(), m_hdrDome->getDescLayout(), m_sky->getDescriptorSetLayout()};
    const VkPushConstantRange  push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                                       sizeof(PushConstant)};
    VkPipelineLayoutCreateInfo create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    create_info.setLayoutCount         = static_cast<uint32_t>(layouts.size());
    create_info.pSetLayouts            = layouts.data();
    create_info.pushConstantRangeCount = 1;
    create_info.pPushConstantRanges    = &push_constant_ranges;
    vkCreatePipelineLayout(m_device, &create_info, nullptr, &m_rasterPipe.layout);

    // Shader source (Spir-V)
    const std::vector<uint32_t> vertex_shader(std::begin(raster_vert), std::end(raster_vert));
    const std::vector<uint32_t> frag_shader(std::begin(raster_frag), std::end(raster_frag));

    auto                             color_format = m_gBuffers->getColorFormat(1);  // Using the RGBA32F
    VkPipelineRenderingCreateInfoKHR rf_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    rf_info.colorAttachmentCount    = 1;
    rf_info.pColorAttachmentFormats = &color_format;
    rf_info.depthAttachmentFormat   = m_gBuffers->getDepthFormat();

    // Creating the Pipeline
    nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_rasterPipe.layout, {} /*m_offscreenRenderPass*/);
    gpb.createInfo.pNext = &rf_info;
    gpb.addBindingDescriptions({{0, sizeof(Vertex)}});
    gpb.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, position)},  // Position + texcoord U
        {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, normal)},    // Normal + texcoord V
        {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent)},   // Tangents
    });

    {
      // Solid
      gpb.rasterizationState.depthBiasEnable         = VK_TRUE;
      gpb.rasterizationState.depthBiasConstantFactor = -1;
      gpb.rasterizationState.depthBiasSlopeFactor    = 1;
      gpb.addShader(vertex_shader, VK_SHADER_STAGE_VERTEX_BIT);
      gpb.addShader(frag_shader, VK_SHADER_STAGE_FRAGMENT_BIT);
      m_rasterPipe.plines.push_back(gpb.createPipeline());
      m_dutil->DBG_NAME(m_rasterPipe.plines[0]);

      // Blend
      gpb.rasterizationState.cullMode = VK_CULL_MODE_NONE;
      VkPipelineColorBlendAttachmentState blend_state{};
      blend_state.blendEnable = VK_TRUE;
      blend_state.colorWriteMask =
          VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
      blend_state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      blend_state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blend_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      gpb.setBlendAttachmentState(0, blend_state);
      m_rasterPipe.plines.push_back(gpb.createPipeline());
      m_dutil->DBG_NAME(m_rasterPipe.plines[1]);

      // Revert Blend Mode
      blend_state.blendEnable = VK_FALSE;
      gpb.setBlendAttachmentState(0, blend_state);
    }

    // Wireframe
    {
      gpb.clearShaders();
      const std::vector<uint32_t> frag_shader(std::begin(raster_overlay_frag), std::end(raster_overlay_frag));
      gpb.addShader(vertex_shader, VK_SHADER_STAGE_VERTEX_BIT);
      gpb.addShader(frag_shader, VK_SHADER_STAGE_FRAGMENT_BIT);
      gpb.rasterizationState.depthBiasEnable = VK_FALSE;
      gpb.rasterizationState.polygonMode     = VK_POLYGON_MODE_LINE;
      gpb.rasterizationState.lineWidth       = 2.0F;
      gpb.depthStencilState.depthWriteEnable = VK_FALSE;
      m_rasterPipe.plines.push_back(gpb.createPipeline());
      m_dutil->DBG_NAME(m_rasterPipe.plines[2]);
    }
  }

  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    auto& p = m_rtxPipe;
    p.destroy(m_device);
    p.plines.resize(1);

    // Creating all shaders
    enum StageIndices
    {
      eRaygen,
      eMiss,
      eClosestHit,
      eAnyHit,
      eShaderGroupCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.pName = "main";  // All the same entry point
    // Raygen
    stage.module    = nvvk::createShaderModule(m_device, pathtrace_rgen, sizeof(pathtrace_rgen));
    stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eRaygen] = stage;
    m_dutil->setObjectName(stage.module, "Raygen");
    // Miss
    stage.module  = nvvk::createShaderModule(m_device, pathtrace_rmiss, sizeof(pathtrace_rmiss));
    stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eMiss] = stage;
    m_dutil->setObjectName(stage.module, "Miss");
    // Hit Group - Closest Hit
    stage.module        = nvvk::createShaderModule(m_device, pathtrace_rchit, sizeof(pathtrace_rchit));
    stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eClosestHit] = stage;
    m_dutil->setObjectName(stage.module, "Closest Hit");
    // AnyHit
    stage.module    = nvvk::createShaderModule(m_device, pathtrace_rahit, sizeof(pathtrace_rahit));
    stage.stage     = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[eAnyHit] = stage;
    m_dutil->setObjectName(stage.module, "Any Hit");

    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group.anyHitShader       = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;
    // Raygen
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    shader_groups.push_back(group);

    // Miss
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss;
    shader_groups.push_back(group);

    // Hit Group-0
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    group.anyHitShader     = eAnyHit;
    shader_groups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    const VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant)};

    VkPipelineLayoutCreateInfo pipeline_layout_create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_create_info.pushConstantRangeCount = 1;
    pipeline_layout_create_info.pPushConstantRanges    = &push_constant;

    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::vector<VkDescriptorSetLayout> rt_desc_set_layouts = {m_rtxSet->getLayout(), m_sceneSet->getLayout(),
                                                              m_sky->getDescriptorSetLayout(),
                                                              m_hdrEnv->getDescriptorSetLayout()};
    pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(rt_desc_set_layouts.size());
    pipeline_layout_create_info.pSetLayouts                = rt_desc_set_layouts.data();
    vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &p.layout);
    m_dutil->DBG_NAME(p.layout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    ray_pipeline_info.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    ray_pipeline_info.pStages                      = stages.data();
    ray_pipeline_info.groupCount                   = static_cast<uint32_t>(shader_groups.size());
    ray_pipeline_info.pGroups                      = shader_groups.data();
    ray_pipeline_info.maxPipelineRayRecursionDepth = 2;  // Ray depth
    ray_pipeline_info.layout                       = p.layout;
    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, (p.plines).data());
    m_dutil->DBG_NAME(p.plines[0]);

    // Creating the SBT
    m_sbt->create(p.plines[0], ray_pipeline_info);

    // Removing temp modules
    for(auto& s : stages)
    {
      vkDestroyShaderModule(m_device, s.module, nullptr);
    }
  }

  void writeRtxSet()
  {
    if(!m_scene->valid())
    {
      return;
    }

    auto& d = m_rtxSet;

    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_sceneRtx->tlas();
    VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    desc_as_info.accelerationStructureCount = 1;
    desc_as_info.pAccelerationStructures    = &tlas;
    const VkDescriptorImageInfo image_info{{}, m_gBuffers->getColorImageView(1), VK_IMAGE_LAYOUT_GENERAL};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(d->makeWrite(0, RtxBindings::eTlas, &desc_as_info));
    writes.emplace_back(d->makeWrite(0, RtxBindings::eOutImage, &image_info));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void writeSceneSet()
  {
    if(!m_scene->valid())
    {
      return;
    }

    auto& d = m_sceneSet;

    // Write to descriptors
    const VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo scene_desc{m_sceneVk->sceneDesc().buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(d->makeWrite(0, SceneBindings::eFrameInfo, &dbi_unif));
    writes.emplace_back(d->makeWrite(0, SceneBindings::eSceneDesc, &scene_desc));
    std::vector<VkDescriptorImageInfo> diit;
    for(const auto& texture : m_sceneVk->textures())  // All texture samplers
    {
      diit.emplace_back(texture.descriptor);
    }
    writes.emplace_back(d->makeWriteArray(0, SceneBindings::eTextures, diit.data()));

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  //--------------------------------------------------------------------------------------------------
  // If the camera matrix has changed, resets the frame.
  // otherwise, increments frame.
  //
  bool updateFrame()
  {
    static nvmath::mat4f ref_cam_matrix;
    static float         ref_fov{CameraManip.getFov()};

    const auto& m   = CameraManip.getMatrix();
    const auto  fov = CameraManip.getFov();

    if(memcmp(&ref_cam_matrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0 || ref_fov != fov)
    {
      resetFrame();
      ref_cam_matrix = m;
      ref_fov        = fov;
    }

    if(m_frame >= m_settings.maxFrames)
    {
      return false;
    }
    m_frame++;
    return true;
  }

  //--------------------------------------------------------------------------------------------------
  // To be call when renderer need to re-start
  //
  void resetFrame() { m_frame = -1; }

  void windowTitle()
  {
    // Window Title
    static float dirty_timer = 0.0F;
    dirty_timer += ImGui::GetIO().DeltaTime;
    if(dirty_timer > 1.0F)  // Refresh every seconds
    {
      const auto&           size = m_app->getViewportSize();
      std::array<char, 256> buf{};
      const int ret = snprintf(buf.data(), buf.size(), "%s %dx%d | %d FPS / %.3fms | Frame %d", PROJECT_NAME,
                               static_cast<int>(size.width), static_cast<int>(size.height),
                               static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate, m_frame);
      glfwSetWindowTitle(m_app->getWindowHandle(), buf.data());
      dirty_timer = 0;
    }
  }

  //--------------------------------------------------------------------------------------------------
  // Send a ray under mouse coordinates, and retrieve the information
  // - Set new camera interest point on hit position
  //
  void screenPicking()
  {
    auto* tlas = m_sceneRtx->tlas();
    if(tlas == VK_NULL_HANDLE)
      return;

    ImGui::Begin("Viewport");  // ImGui, picking within "viewport"
    auto        mouse_pos        = ImGui::GetMousePos();
    const auto  main_size        = ImGui::GetContentRegionAvail();
    const auto  corner           = ImGui::GetCursorScreenPos();  // Corner of the viewport
    const float aspect_ratio     = main_size.x / main_size.y;
    mouse_pos                    = mouse_pos - corner;
    const ImVec2 local_mouse_pos = mouse_pos / main_size;
    ImGui::End();


    // Finding current camera matrices
    const auto& view = CameraManip.getMatrix();
    auto        proj = nvmath::perspectiveVK(CameraManip.getFov(), aspect_ratio, 0.1F, 1000.0F);

    float hitT{0.0F};

    // Setting up the data to do picking
    auto*                        cmd = m_app->createTempCmdBuffer();
    nvvk::RayPickerKHR::PickInfo pick_info;
    pick_info.pickX          = local_mouse_pos.x;
    pick_info.pickY          = local_mouse_pos.y;
    pick_info.modelViewInv   = nvmath::invert(view);
    pick_info.perspectiveInv = nvmath::invert(proj);

    // Run and wait for result
    m_picker->run(cmd, pick_info);
    m_app->submitAndWaitTempCmdBuffer(cmd);


    // Retrieving picking information
    const nvvk::RayPickerKHR::PickResult pr = m_picker->getResult();
    if(pr.instanceID == ~0)
    {
      LOGI("Nothing Hit\n");
      return;
    }

    if(pr.hitT <= 0.F)
    {
      LOGI("Hit Distance == 0.0\n");
      return;
    }

    // Find where the hit point is and set the interest position
    const nvmath::vec3f world_pos = nvmath::vec3f(pr.worldRayOrigin + pr.worldRayDirection * pr.hitT);
    nvmath::vec3f       eye;
    nvmath::vec3f       center;
    nvmath::vec3f       up;
    CameraManip.getLookat(eye, center, up);
    CameraManip.setLookat(eye, world_pos, up, false);

    auto float_as_uint = [](float f) { return *reinterpret_cast<uint32_t*>(&f); };

    // Logging picking info.
    const auto& prim = m_scene->scene().m_primMeshes[pr.instanceCustomIndex];
    LOGI("Hit(%d): %s, PrimId: %d, ", pr.instanceCustomIndex, prim.name.c_str(), pr.primitiveID);
    LOGI("{%3.2f, %3.2f, %3.2f}, Dist: %3.2f\n", world_pos.x, world_pos.y, world_pos.z, pr.hitT);
    LOGI("PrimitiveID: %d\n", pr.primitiveID);
  }

  //--------------------------------------------------------------------------------------------------
  // Render the axis in the bottom left corner of the screen
  //
  void renderAxis(const VkCommandBuffer& cmd)
  {
    auto _sdbg = m_dutil->DBG_SCOPE(cmd);

    float axis_size = 50.0F;
    if(m_settings.renderSystem == Settings::eRaster)
    {
      axis_size *= RASTER_SS_SIZE;
    }

    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView(0)},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_CLEAR);
    r_info.pStencilAttachment = nullptr;
    // Rendering the axis
    vkCmdBeginRendering(cmd, &r_info);
    m_vkAxis->setAxisSize(axis_size);
    m_vkAxis->display(cmd, CameraManip.getMatrix(), m_gBuffers->getSize());
    vkCmdEndRendering(cmd);
  }

  void raytraceScene(VkCommandBuffer cmd)
  {
    const nvvk::DebugUtil::ScopedCmdLabel scope_dbg = m_dutil->DBG_SCOPE(cmd);

    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_rtxSet->getSet(), m_sceneSet->getSet(), m_sky->getDescriptorSet(),
                                           m_hdrEnv->getDescriptorSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtxPipe.plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtxPipe.layout, 0,
                            static_cast<uint32_t>(desc_sets.size()), desc_sets.data(), 0, nullptr);
    vkCmdPushConstants(cmd, m_rtxPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant), &m_pushConst);

    const auto& regions = m_sbt->getRegions();
    const auto& size    = m_gBuffers->getSize();
    vkCmdTraceRaysKHR(cmd, regions.data(), &regions[1], &regions[2], &regions[3], size.width, size.height, 1);

    // Making sure the rendered image is ready to be used
    auto* out_image = m_gBuffers->getColorImage(1);
    auto image_memory_barrier = nvvk::makeImageMemoryBarrier(out_image, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                                             VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &image_memory_barrier);
  }

  void createRecordCommandBuffer()
  {
    VkCommandBufferAllocateInfo alloc_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc_info.commandPool        = m_app->getCommandPool();
    alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    alloc_info.commandBufferCount = 1;
    vkAllocateCommandBuffers(m_device, &alloc_info, &m_recordedSceneCmd);
  }

  void freeRecordCommandBuffer()
  {
    vkDeviceWaitIdle(m_device);
    vkFreeCommandBuffers(m_device, m_app->getCommandPool(), 1, &m_recordedSceneCmd);
    m_recordedSceneCmd = VK_NULL_HANDLE;
  }

  void recordRasterScene()
  {
    createRecordCommandBuffer();

    auto color_format = m_gBuffers->getColorFormat(1);  // Using the RGBA32F

    VkCommandBufferInheritanceRenderingInfoKHR inheritance_rendering_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO_KHR};
    inheritance_rendering_info.colorAttachmentCount    = 1;
    inheritance_rendering_info.pColorAttachmentFormats = &color_format;
    inheritance_rendering_info.depthAttachmentFormat   = m_gBuffers->getDepthFormat();
    inheritance_rendering_info.rasterizationSamples    = VK_SAMPLE_COUNT_1_BIT;

    VkCommandBufferInheritanceInfo inherit_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
    inherit_info.pNext = &inheritance_rendering_info;

    VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    begin_info.pInheritanceInfo = &inherit_info;
    vkBeginCommandBuffer(m_recordedSceneCmd, &begin_info);
    renderRasterScene(m_recordedSceneCmd);
    vkEndCommandBuffer(m_recordedSceneCmd);
  }

  // Render the scene
  void renderRasterScene(VkCommandBuffer cmd)
  {
    auto _sdbg = m_dutil->DBG_SCOPE(cmd);

    const VkExtent2D& render_size = m_gBuffers->getSize();

    const VkViewport viewport{0.0F, 0.0F, static_cast<float>(render_size.width), static_cast<float>(render_size.height),
                              0.0F, 1.0F};
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    const VkRect2D scissor{{0, 0}, {render_size.width, render_size.height}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    std::vector dset = {m_sceneSet->getSet(), m_hdrDome->getDescSet(), m_sky->getDescriptorSet()};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_rasterPipe.layout, 0,
                            static_cast<uint32_t>(dset.size()), dset.data(), 0, nullptr);
    // Draw solid
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_rasterPipe.plines[0]);
    renderNodes(cmd, m_solidMatNodes);
    // Draw blend-able
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_rasterPipe.plines[1]);
    renderNodes(cmd, m_blendMatNodes);

    if(m_settings.showWireframe)
    {
      // Draw wireframe
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_rasterPipe.plines[2]);
      renderNodes(cmd, m_allNodes);
    }
  }

  void rasterScene(VkCommandBuffer cmd)
  {
    auto scope_dbg = m_dutil->DBG_SCOPE(cmd);

    // Rendering Dome/Background
    {
      const auto& viewport_size = m_gBuffers->getSize();
      const float aspect_ratio  = static_cast<float>(viewport_size.width) / static_cast<float>(viewport_size.height);
      const auto& view          = CameraManip.getMatrix();
      const auto  proj          = nvmath::perspectiveVK(CameraManip.getFov(), aspect_ratio, 0.1F, 1000.0F);

      auto img_size = m_gBuffers->getSize();
      if(m_settings.envSystem == Settings::eSky)
        m_sky->draw(cmd, view, proj, img_size);
      else
        m_hdrDome->draw(cmd, view, proj, img_size, &m_settings.clearColor.x, m_settings.envRotation);
    }

    if(m_recordedSceneCmd == VK_NULL_HANDLE)
    {
      recordRasterScene();
    }

    // Execute recorded command buffer
    {
      // Drawing the primitives in the RGBA32F G-Buffer, don't clear or it will erase the
      nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView(1)},
                                       m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_CLEAR,
                                       m_clearColor, {1.0F, 0}, VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT_KHR);
      r_info.pStencilAttachment = nullptr;

      vkCmdBeginRendering(cmd, &r_info);
      vkCmdExecuteCommands(cmd, 1, &m_recordedSceneCmd);
      vkCmdEndRendering(cmd);
    }
  }

  // Rendering GLTF nodes of the list
  void renderNodes(VkCommandBuffer cmd, std::vector<uint32_t>& nodeIDs)
  {
    auto _sdbg = m_dutil->DBG_SCOPE(cmd);

    const VkDeviceSize offsets{0};
    const auto&        gltf_scene = m_scene->scene();

    for(auto& node_id : nodeIDs)
    {
      const auto& node      = gltf_scene.m_nodes[node_id];
      const auto& primitive = gltf_scene.m_primMeshes[node.primMesh];

      m_pushConst.materialId = primitive.materialIndex;
      m_pushConst.instanceId = static_cast<int>(node_id);
      vkCmdPushConstants(cmd, m_rasterPipe.layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(PushConstant), &m_pushConst);

      vkCmdBindVertexBuffers(cmd, 0, 1, &m_sceneVk->vertices()[node.primMesh].buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m_sceneVk->indices()[node.primMesh].buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(cmd, primitive.indexCount, 1, 0, 0, 0);
    }
  }

  void createHdr(const std::string& filename)
  {
    uint32_t c_family_queue = m_app->getContext()->m_queueC.familyIndex;
    m_hdrEnv  = std::make_unique<nvvkhl::HdrEnv>(m_app->getContext().get(), m_alloc.get(), c_family_queue);
    m_hdrDome = std::make_unique<nvvkhl::HdrEnvDome>(m_app->getContext().get(), m_alloc.get(), c_family_queue);

    m_hdrEnv->loadEnvironment(filename);
    m_hdrDome->create(m_hdrEnv->getDescriptorSet(), m_hdrEnv->getDescriptorSetLayout());
    m_hdrDome->setOutImage(m_gBuffers->getDescriptorImageInfo(1));
    freeRecordCommandBuffer();
  }


  void destroyResources()
  {
    m_alloc->destroy(m_bFrameInfo);

    freeRecordCommandBuffer();

    m_gBuffers.reset();

    m_rasterPipe.destroy(m_device);
    m_rtxPipe.destroy(m_device);
    m_rtxSet->deinit();
    m_sceneSet->deinit();
    m_sbt->destroy();
    m_sky->destroy();
    m_picker->destroy();
    m_vkAxis->deinit();

    m_tonemapper.reset();
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*              m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>  m_dutil;
  std::unique_ptr<nvvkhl::AllocVma> m_alloc;
  VkCommandBuffer                   m_recordedSceneCmd{VK_NULL_HANDLE};

  nvmath::vec2f                    m_viewSize{1, 1};
  VkClearColorValue                m_clearColor{{0.3F, 0.3F, 0.3F, 1.0F}};  // Clear color
  VkDevice                         m_device{VK_NULL_HANDLE};                // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                              // G-Buffers: color + depth

  // Resources
  nvvk::Buffer         m_bFrameInfo;
  nvvk::Context::Queue m_qGCT1{};

  // Pipeline
  PushConstant              m_pushConst{};  // Information sent to the shader
  nvvkhl::PipelineContainer m_rasterPipe;
  nvvkhl::PipelineContainer m_rtxPipe;
  int                       m_frame{-1};
  FrameInfo                 m_frameInfo{};

  std::unique_ptr<nvvkhl::HdrEnv>                m_hdrEnv;
  std::unique_ptr<nvvkhl::HdrEnvDome>            m_hdrDome;
  std::unique_ptr<nvvk::AxisVK>                  m_vkAxis;
  std::unique_ptr<nvvk::DescriptorSetContainer>  m_rtxSet;    // Descriptor set
  std::unique_ptr<nvvk::DescriptorSetContainer>  m_sceneSet;  // Descriptor set
  std::unique_ptr<nvvk::RayPickerKHR>            m_picker;    // For ray picking info
  std::unique_ptr<nvvk::SBTWrapper>              m_sbt;       // Shading binding table wrapper
  std::unique_ptr<nvvkhl::Scene>                 m_scene;
  std::unique_ptr<nvvkhl::SceneRtx>              m_sceneRtx;
  std::unique_ptr<nvvkhl::SceneVk>               m_sceneVk;
  std::unique_ptr<nvvkhl::SkyDome>               m_sky;
  std::unique_ptr<nvvkhl::TonemapperPostProcess> m_tonemapper;

  // For rendering all nodes
  std::vector<uint32_t> m_solidMatNodes;
  std::vector<uint32_t> m_blendMatNodes;
  std::vector<uint32_t> m_allNodes;
  bool                  m_busy{false};
};

}  // namespace

//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char** argv) -> int
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name             = PROJECT_NAME " Example";
  spec.vSync            = false;
  spec.vkSetup.apiMajor = 1;
  spec.vkSetup.apiMinor = 3;

  spec.vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false,
                                  &accel_feature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false,
                                  &rt_pipeline_feature);                            // To use vkCmdTraceRaysKHR
  spec.vkSetup.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
  VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &ray_query_features);  // Used for picking
  spec.vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

  // Request for extra Queue for loading in parallel
  spec.vkSetup.addRequestedQueue(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, 1, 1.0F);

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // Create Elements of the application
  auto raytracing = std::make_shared<GltfViewer>();
  g_elem_camera   = std::make_shared<nvvkhl::ElementCamera>();

  app->addElement(g_elem_camera);
  app->addElement(raytracing);
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // Menu / Quit

  // Search paths
  const std::vector<std::string> default_search_paths = {NVPSystem::exePath() + PROJECT_DOWNLOAD_RELDIRECTORY};

  // Load scene elements
  std::string in_filename = nvh::findFile(R"(FlightHelmet/FlightHelmet.gltf)", default_search_paths, true);
  std::string in_hdr      = nvh::findFile("daytime.hdr", default_search_paths, true);

  // Parsing arguments
  bool                   print_help{false};
  nvh::CommandLineParser args("Test Parser");
  args.addArgument({"-f", "--filename"}, &in_filename, "Input filename");
  args.addArgument({"--hdr"}, &in_hdr, "Input HDR");
  args.addArgument({"-h", "--help"}, &print_help, "Print Help");
  const bool result = args.parse(argc, argv);
  if(!result || print_help)
  {
    args.printHelp();
    return 1;
  }

  raytracing->onFileDrop(in_filename.c_str());
  raytracing->onFileDrop(in_hdr.c_str());

  app->run();
  raytracing.reset();
  app.reset();

  return 0;
}
