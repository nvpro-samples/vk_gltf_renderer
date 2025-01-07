/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <thread>

#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// ImGui headers
#include "imgui/imgui_axis.hpp"
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"

// NV Vulkan headers
#include "nvvk/extensions_vk.hpp"
#include "nvvk/raypicker_vk.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_dbgprintf.hpp"
#include "nvvkhl/element_logger.hpp"
#include "nvvkhl/element_nvml.hpp"
#include "nvvkhl/element_profiler.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"

// Application specific headers
#include "busy_window.hpp"
#include "renderer.hpp"
#include "scene.hpp"
#include "settings.hpp"
#include "settings_handler.hpp"
#include "utilities.hpp"
#include "vk_context.hpp"
#include "stb_image.h"
#include "doc/app_icon_png.h"
#include "collapsing_header_manager.h"
#include "perproject_globals.hpp"
#include "nvvk/nsight_aftermath_vk.hpp"

// #define USE_AFTERMATH
// For debugging GPU crashes
//  In CMake, set -DNSIGHT_AFTERMATH_SKD=<path_to_sdk> to have "Aftermath available"

// #define USE_DGBPRINTF
// With USE_DGBPRINTF defined, the application will have the capability to use the debug printf extension.
// And it is possible to use the debug printf in the shaders, by including the following:
// if(useDebug)
//   debugPrintfEXT("Hello from shader\n");
// This will print "Hello from shader" in the debug printf window ONLY for the pixels under the mouse cursor, when the mouse button is pressed.


// Benchmark Testing
// To run benchmark testing, the following command line arguments can be used:
// <path-to-gltf-scene> -test -test-frames 1000 -screenshot "screenshots.png"
// The above command will run the benchmark test for 1000 frames and save the screenshot to "screenshots.png".


std::shared_ptr<nvvkhl::ElementCamera>              g_elemCamera;     // The camera element (UI and movement)
std::shared_ptr<nvvkhl::ElementProfiler>            g_elemProfiler;   // GPU profiler
std::shared_ptr<nvvkhl::ElementBenchmarkParameters> g_elemBenchmark;  // Benchmark parameters
std::shared_ptr<nvvkhl::ElementDbgPrintf>           g_elemDebugPrintf;
std::shared_ptr<nvvkhl::SampleAppLog>               g_elemLogger;              // Log window
std::vector<std::string>                            g_applicationSearchPaths;  // Search paths for resources


// Default scene elements
std::string g_inFilename;
std::string g_inHdr;

namespace PE = ImGuiH::PropertyEditor;

namespace gltfr {
bool g_forceExternalShaders = false;

extern PathtraceSettings g_pathtraceSettings;

class GltfRendererElement : public nvvkhl::IAppElement
{
public:
  GltfRendererElement() { addSettingsHandler(); }

  //--------------------------------------------------------------------------------------------------
  // Called at the beginning of the application
  //
  void onAttach(nvvkhl::Application* app) override
  {
    m_app = app;

    // Override the way benchmark count frames, to only use valid ones
    g_elemBenchmark->setCurrentFrame([&] { return m_scene.m_sceneFrameInfo.frameCount; });

    // Getting all required resources
    gltfr::VulkanInfo ctx;
    ctx.device         = app->getDevice();
    ctx.physicalDevice = app->getPhysicalDevice();
    ctx.GCT0           = {app->getQueue(0).queue, app->getQueue(0).familyIndex};  // See creation of queues in main()
    ctx.compute        = {app->getQueue(1).queue, app->getQueue(1).familyIndex};
    ctx.transfer       = {app->getQueue(2).queue, app->getQueue(2).familyIndex};

    m_resources.init(ctx);
    m_scene.init(m_resources);

    nvvk::ResourceAllocator* alloc         = m_resources.m_allocator.get();
    uint32_t                 c_queue_index = ctx.compute.familyIndex;

    m_tonemapper.init(ctx.device, alloc);

    m_picker = std::make_unique<nvvk::RayPickerKHR>(ctx.device, ctx.physicalDevice, alloc, c_queue_index);  // RTX Picking utility

    m_emptyRenderer = makeRendererEmpty();
    m_tonemapper.createComputePipeline();

    if(!g_inHdr.empty())
    {
      // Wait for the HDR to be loaded
      m_scene.load(m_resources, g_inHdr);
      m_settings.envSystem    = Settings::eHdr;
      m_settings.maxLuminance = m_scene.m_hdrEnv->getIntegral();
    }
    if(!g_inFilename.empty())
    {
      // Load the glTF file in a separate thread
      onFileDrop(g_inFilename.c_str());
    }
  }

  //--------------------------------------------------------------------------------------------------
  void onDetach() override
  {
    vkDeviceWaitIdle(m_resources.ctx.device);
    m_scene.deinit(m_resources);
    m_emptyRenderer->deinit(m_resources);
    if(m_renderer != nullptr)
      m_renderer->deinit(m_resources);
    m_renderer.reset();
    m_picker->destroy();
  }

  //--------------------------------------------------------------------------------------------------
  void onResize(uint32_t width, uint32_t height) override { m_resources.resizeGbuffers({width, height}); }

  //--------------------------------------------------------------------------------------------------
  void onRender(VkCommandBuffer cmd) override
  {
    if(m_busy.isBusy())
      return;
    if(m_busy.isDone())
    {
      // Post busy work
      m_busy.consumeDone();
    }

    // Handle changes that have happened since last frame
    handleChanges(cmd);

    if(m_renderer && m_scene.isValid())
    {
      // Animate, update Vulkan buffers: scene, frame, acceleration structures
      // It could stop rendering if the scene is not ready or reached max frames
      if(m_scene.processFrame(cmd, m_settings))
      {
        m_renderer->render(cmd, m_resources, m_scene, m_settings, *g_elemProfiler.get());
      }
    }
    else
    {
      m_emptyRenderer->render(cmd, m_resources, m_scene, m_settings, *g_elemProfiler.get());
    }


    // Apply tone mapper to the final image
    {
      auto sec = g_elemProfiler->timeRecurring("Tonemapper", cmd);
      setTonemapperInputOutput();
      m_tonemapper.runCompute(cmd, m_resources.m_finalImage->getSize());
    }
  }

  //--------------------------------------------------------------------------------------------------
  // Set the input and output of the tonemapper
  // Input - the result of the renderer
  // Output - the final image (resources finalImage)
  void setTonemapperInputOutput()
  {
    VkDescriptorImageInfo inputImage  = m_resources.m_finalImage->getDescriptorImageInfo();
    VkDescriptorImageInfo outputImage = inputImage;
    if(m_renderer != nullptr && m_scene.isValid())
    {
      inputImage = m_renderer->getOutputImage();
    }
    m_tonemapper.updateComputeDescriptorSets(inputImage, outputImage);  // To
  }

  //--------------------------------------------------------------------------------------------------
  // Main UI of the application
  // Will display the settings, the viewport and the busy window
  void onUIRender() override
  {
    auto& headerManager = CollapsingHeaderManager::getInstance();

    if(ImGui::Begin("Settings"))
    {
      PE::begin();
      {
        if(PE::Combo("Renderer", (int*)&m_settings.renderSystem, Settings::rendererNames, IM_ARRAYSIZE(Settings::rendererNames)))
        {
          createRenderers();
        }
      }
      PE::end();
      if(m_renderer != nullptr && m_renderer->onUI())
      {
        m_scene.resetFrameCount();
      }
      if(headerManager.beginHeader("Camera"))
      {
        ImGuiH::CameraWidget();
      }
      m_settings.onUI();
      if(headerManager.beginHeader("Tonemapper"))
      {
        m_tonemapper.onUI();
      }

      if(!m_busy.isBusy())
        m_scene.onUI(m_resources, m_settings, m_app->getWindowHandle());
    }
    ImGui::End();  // Settings


    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Pick under mouse cursor
      if((ImGui::IsWindowHovered(ImGuiFocusedFlags_RootWindow) && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
         || ImGui::IsKeyPressed(ImGuiKey_Space))
      {
        screenPicking();
      }

      // Display the G-Buffer image
      ImVec2 imgSize = ImGui::GetContentRegionAvail();
      ImGui::Image(m_resources.m_finalImage->getDescriptorSet(), imgSize);

      // Adding Axis at the bottom left corner of the viewport
      if(m_settings.showAxis)
      {
        float  size        = 25.F;
        ImVec2 window_pos  = ImGui::GetWindowPos();
        ImVec2 window_size = ImGui::GetWindowSize();
        ImVec2 offset      = ImVec2(size * 1.1F, -size * 1.1F) * ImGui::GetWindowDpiScale();
        ImVec2 pos         = ImVec2(window_pos.x, window_pos.y + window_size.y) + offset;
        ImGuiH::Axis(pos, CameraManip.getMatrix(), size);
      }

      ImGui::End();
      ImGui::PopStyleVar();
    }

    if(m_busy.isBusy())
      m_busy.show();
  }

  //--------------------------------------------------------------------------------------------------
  // This is the toolbar at the top of the window
  // Adding menu items for loading, saving, and changing the view
  void onUIMenu() override
  {
    bool v_sync = m_app->isVsync();

    auto getSaveImage = [&]() {
      std::string filename =
          NVPSystem::windowSaveFileDialog(m_app->getWindowHandle(), "Save Image", "PNG(.png),JPG(.jpg)|*.png;*.jpg");
      if(!filename.empty())
      {
        std::filesystem::path ext = std::filesystem::path(filename).extension();
      }
      return filename;
    };

    windowTitle();
    bool loadFile       = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_O);
    bool saveFile       = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S);
    bool saveScreenFile = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiMod_Alt | ImGuiKey_S);
    bool saveImageFile  = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_S);
    bool closeApp       = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Q);
    if(ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_V))
    {
      v_sync = !v_sync;
    }

    if(ImGui::BeginMenu("File"))
    {
      loadFile |= ImGui::MenuItem("Load", "Ctrl+O");
      saveFile |= ImGui::MenuItem("Save As", "Ctrl+S");
      ImGui::Separator();
      saveImageFile |= ImGui::MenuItem("Save Image", "Ctrl+Shift+S");
      saveScreenFile |= ImGui::MenuItem("Save Screen", "Ctrl+Alt+Shift+S");
      ImGui::Separator();
      closeApp |= ImGui::MenuItem("Exit", "Ctrl+Q");
      ImGui::EndMenu();
    }

    // De-selecting the object
    if(ImGui::IsKeyPressed(ImGuiKey_Escape))
    {
      m_scene.selectRenderNode(-1);
    }

    bool fitScene        = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_F);
    bool fitObject       = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_F);
    bool doReloadShaders = ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_R);
    if(ImGui::BeginMenu("View"))
    {
      fitScene |= ImGui::MenuItem("Fit Scene", "Ctrl+Shift+F");
      fitObject |= ImGui::MenuItem("Fit Object", "Ctrl+F");
      ImGui::Separator();
      ImGui::MenuItem("V-Sync", "Ctrl+Shift+V", &v_sync);
      ImGui::EndMenu();
    }
    if(ImGui::BeginMenu("Tools"))
    {
      if(m_resources.hasGlslCompiler())
      {
        doReloadShaders |= ImGui::MenuItem("Reload Shaders", "Ctrl+R");
        ImGui::Separator();
      }
      bool recreate = ImGui::MenuItem("Recreate Tangents - Simple");
      ImGui::SetItemTooltip("This recreate all tangents using MikkTSpace");
      bool mikktspace = ImGui::MenuItem("Recreate Tangents - MikkTSpace");
      ImGui::SetItemTooltip("This fixes NULL tangents");

      if(recreate || mikktspace)
      {
        vkDeviceWaitIdle(m_resources.ctx.device);
        m_busy.start("Recreate Tangents");
        std::thread([&, mikktspace]() {
          m_scene.recreateTangents(mikktspace);
          m_busy.stop();
        }).detach();
      }

      ImGui::EndMenu();
    }

    if(loadFile)
    {
      std::string filename = NVPSystem::windowOpenFileDialog(m_app->getWindowHandle(), "Load glTF | HDR",
                                                             "glTF(.gltf, .glb), OBJ(.obj), "
                                                             "HDR(.hdr)|*.gltf;*.glb;*.obj;*.hdr");
      onFileDrop(filename.c_str());
    }

    if(saveFile)
    {
      m_scene.save(NVPSystem::windowSaveFileDialog(m_app->getWindowHandle(), "Save glTF", "glTF(.gltf, .glb)|*.gltf;*.glb"));
    }

    if(saveScreenFile)
    {
      std::string filename = getSaveImage();
      if(!filename.empty())
      {
        m_app->screenShot(filename, 100);
      }
    }

    if(saveImageFile)
    {

      std::filesystem::path filename = getSaveImage();
      // filename = std::filesystem::path(m_scene.getFilename()).stem().replace_extension(".jpg");
      // filename = std::filesystem::path("C:/temp") / filename;
      if(!filename.empty())
      {
        saveRenderedImage(filename.string());
      }
    }

    if(fitScene)
    {
      m_scene.fitSceneToView();
    }

    if(fitObject)
    {
      m_scene.fitObjectToView();
    }

    if(doReloadShaders)
    {
      reloadShaders();
    }

    if(m_app->isVsync() != v_sync)
    {
      m_app->setVsync(v_sync);
    }

    if(closeApp)
    {
      m_app->close();
    }

#ifndef NDEBUG
    static bool s_showDemo{false};
    static bool s_showDemoPlot{false};
    if(ImGui::BeginMenu("Debug"))
    {
      ImGui::MenuItem("Show ImGui Demo", nullptr, &s_showDemo);
      ImGui::MenuItem("Show ImPlot Demo", nullptr, &s_showDemoPlot);
      ImGui::EndMenu();
    }
    if(s_showDemo)
      ImGui::ShowDemoWindow(&s_showDemo);
    if(s_showDemoPlot)
      ImPlot::ShowDemoWindow(&s_showDemoPlot);
#endif  // !NDEBUG
  }

  void saveRenderedImage(const std::string& filename)
  {
    VkDevice         device         = m_app->getDevice();
    VkPhysicalDevice physicalDevice = m_app->getPhysicalDevice();
    VkImage          srcImage       = m_resources.m_finalImage->getColorImage();
    VkExtent2D       size           = m_resources.m_finalImage->getSize();
    VkImage          dstImage       = {};
    VkDeviceMemory   dstImageMemory = {};

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    imageToRgba8Linear(cmd, device, physicalDevice, srcImage, size, dstImage, dstImageMemory);
    m_app->submitAndWaitTempCmdBuffer(cmd);

    saveImageToFile(device, dstImage, dstImageMemory, size, filename);

    // Clean up resources
    vkUnmapMemory(device, dstImageMemory);
    vkFreeMemory(device, dstImageMemory, nullptr);
    vkDestroyImage(device, dstImage, nullptr);
  }

  //--------------------------------------------------------------------------------------------------
  // Handle the file drop event: glTF, HDR, OBJ
  //
  void onFileDrop(const char* filename) override
  {
    if(m_busy.isBusy())
      return;
    m_busy.start("Loading");
    vkDeviceWaitIdle(m_resources.ctx.device);

    // Loading file in a separate thread
    std::string loadFile = filename;
    std::thread([&, loadFile]() {
      m_scene.load(m_resources, loadFile);
      m_busy.stop();
    }).detach();

    // Visualize the HDR if it is a HDR file
    std::filesystem::path ext = std::filesystem::path(filename).extension();
    if(ext == ".hdr")
    {
      m_settings.envSystem    = Settings::eHdr;
      m_settings.maxLuminance = m_scene.m_hdrEnv->getIntegral();
    }
  }

  //--------------------------------------------------------------------------------------------------
  // Change the window title to display real-time informations
  //
  void windowTitle()
  {
    // Window Title
    static float dirty_timer = 0.0F;
    dirty_timer += ImGui::GetIO().DeltaTime;
    if(dirty_timer > 1.0F)  // Refresh every seconds
    {
      const VkExtent2D&     size = m_app->getViewportSize();
      std::filesystem::path p    = m_scene.getFilename();
      std::string           text =
          fmt::format("{} - {}x{} | {:.0f} FPS / {:.3f}ms | Frame {}", p.filename().string(), size.width, size.height,
                      ImGui::GetIO().Framerate, 1000.F / ImGui::GetIO().Framerate, m_scene.m_sceneFrameInfo.frameCount);

      glfwSetWindowTitle(m_app->getWindowHandle(), text.c_str());
      dirty_timer = 0;
    }
  }

private:
  //--------------------------------------------------------------------------------------------------
  // Invoked when the user double-clicks on the viewport
  // It uses the picking utility to find the object under the mouse cursor
  //
  void screenPicking()
  {
    // Pick under mouse cursor
    if(!m_scene.isValid() || !m_scene.m_gltfSceneRtx || m_scene.m_gltfSceneRtx->tlas() == VK_NULL_HANDLE)
      return;

    // This need to called withing ImGui::Begin("Viewport");
    ImVec2       mouse_pos       = ImGui::GetMousePos();
    const ImVec2 main_size       = ImGui::GetContentRegionAvail();
    const ImVec2 corner          = ImGui::GetCursorScreenPos();  // Corner of the viewport
    const float  aspect_ratio    = main_size.x / main_size.y;
    mouse_pos                    = mouse_pos - corner;
    const ImVec2 local_mouse_pos = mouse_pos / main_size;

    // Finding current camera matrices
    const glm::mat4& view = CameraManip.getMatrix();
    glm::mat4        proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspect_ratio, 0.1F, 1000.0F);
    proj[1][1] *= -1;

    // Setting up the data to do picking
    nvvk::RayPickerKHR::PickInfo pickInfo{
        .modelViewInv   = glm::inverse(view),
        .perspectiveInv = glm::inverse(proj),
        .pickX          = local_mouse_pos.x,
        .pickY          = local_mouse_pos.y,
    };

    // Run and wait for result
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_picker->run(cmd, pickInfo);
    m_app->submitAndWaitTempCmdBuffer(cmd);


    // Retrieving picking information
    const nvvk::RayPickerKHR::PickResult pr = m_picker->getResult();
    if(pr.instanceID == ~0)
    {
      LOGI("Nothing Hit\n");
      m_scene.selectRenderNode(-1);
      return;
    }

    if(pr.hitT <= 0.F)
    {
      LOGI("Hit Distance == 0.0\n");
      return;
    }

    // Find where the hit point is and set the interest position
    const glm::vec3 world_pos = glm::vec3(pr.worldRayOrigin + pr.worldRayDirection * pr.hitT);
    glm::vec3       eye;
    glm::vec3       center;
    glm::vec3       up;
    CameraManip.getLookat(eye, center, up);
    CameraManip.setLookat(eye, world_pos, up, false);

    // Logging picking info.
    const nvh::gltf::RenderNode& renderNode = m_scene.m_gltfScene->getRenderNodes()[pr.instanceID];
    const tinygltf::Node&        node       = m_scene.m_gltfScene->getModel().nodes[renderNode.refNodeID];
    m_scene.selectRenderNode(pr.instanceID);

    LOGI("Node Name: %s\n", node.name.c_str());
    LOGI(" - GLTF: NodeID: %d, MeshID: %d, TriangleId: %d\n", renderNode.refNodeID, node.mesh, pr.primitiveID);
    LOGI(" - Render: RenderNode: %d, RenderPrim: %d\n", pr.instanceID, pr.instanceCustomIndex);
    LOGI("{%3.2f, %3.2f, %3.2f}, Dist: %3.2f\n", world_pos.x, world_pos.y, world_pos.z, pr.hitT);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the renderer based on the settings
  // Note: the tonemapper input/output should be re-adjusted after creating the renderer
  //
  void createRenderers()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    vkDeviceWaitIdle(m_resources.ctx.device);
    m_resources.resetSlangCompiler();  // Resetting the Slang session
    m_renderer.reset();
    if(!m_scene.isValid())
      return;
    switch(m_settings.renderSystem)
    {
      case Settings::ePathtracer:
        m_renderer = makeRendererPathtracer();
        break;
      case Settings::eRaster:
        m_renderer = makeRendererRaster();
        break;
    }
    if(m_renderer)
    {
      bool result = m_renderer->init(m_resources, m_scene);
      if(result == false)
      {
        LOGE("Failed to create renderer: %s\n", Settings::rendererNames[m_settings.renderSystem]);
        m_renderer.reset();
      }
    }
    m_scene.resetFrameCount();
  }

  void reloadShaders()
  {

    nvh::ScopedTimer st(__FUNCTION__);

    vkDeviceWaitIdle(m_resources.ctx.device);
    m_resources.resetSlangCompiler();  // Resetting the Slang session
    if(!m_scene.isValid() || !m_renderer)
      return;
    m_renderer->reloadShaders(m_resources, m_scene);
    m_scene.resetFrameCount();
  }

  //--------------------------------------------------------------------------------------------------
  // Handle changes that have happened since last frame
  // - Scene changes
  // - Resolution changes
  //
  void handleChanges(VkCommandBuffer cmd)
  {
    if(m_scene.hasDirtyFlag(Scene::eHdrEnv))
    {
      m_scene.generateHdrMipmap(cmd, m_resources);
      m_settings.setDefaultLuminance(m_scene.m_hdrEnv->getIntegral());
    }

    // Scene changed (new scene)
    if(m_scene.hasDirtyFlag(Scene::eNewScene))
    {
      vkDeviceWaitIdle(m_resources.ctx.device);
      createRenderers();
      m_picker->setTlas(m_scene.m_gltfSceneRtx->tlas());  // The screen picker is using the new TLAS
    }

    // Letting the renderer handle any changes
    if(m_renderer)
    {
      m_renderer->handleChange(m_resources, m_scene);
    }

    // Clearing flags changed
    m_scene.setDirtyFlag(Scene::eNewScene, false);
    m_scene.setDirtyFlag(Scene::eHdrEnv, false);
    m_scene.setDirtyFlag(Scene::eNodeVisibility, false);
    m_resources.setGBuffersChanged(false);
  }

  // This goes in the .ini file and remember the settings of the application
  void addSettingsHandler()
  {
    m_settingsHandler.setHandlerName("GLTFRenderer");
    m_settingsHandler.setSetting("Renderer", reinterpret_cast<int*>(&m_settings.renderSystem));
    m_settingsHandler.setSetting("MaxFrames", &m_settings.maxFrames);
    m_settingsHandler.setSetting("ShowAxis", &m_settings.showAxis);
    m_settingsHandler.setSetting("SilhouetteColor", &m_settings.silhouetteColor);
    m_settingsHandler.setSetting("BackgrounfColor", &m_settings.solidBackgroundColor);
    m_settingsHandler.setSetting("Tonemapper", &m_tonemapper.settings().method);
    m_settingsHandler.addImGuiHandler();
  }


  nvvkhl::Application*                m_app = nullptr;
  Resources                           m_resources;
  Settings                            m_settings;
  Scene                               m_scene;
  std::unique_ptr<gltfr::Renderer>    m_emptyRenderer{};
  std::unique_ptr<gltfr::Renderer>    m_renderer{};
  nvvkhl::TonemapperPostProcess       m_tonemapper;
  std::unique_ptr<nvvk::RayPickerKHR> m_picker{};
  SettingsHandler                     m_settingsHandler;
  BusyWindow                          m_busy;
};


}  // namespace gltfr

static void setWindowIcon(GLFWwindow* window)
{
  GLFWimage icon{};
  int       channels = 0;
  icon.pixels        = stbi_load_from_memory(app_icon_png, app_icon_png_len, &icon.width, &icon.height, &channels, 4);
  if(icon.pixels)
  {
    glfwSetWindowIcon(window, 1, &icon);
    stbi_image_free(icon.pixels);
  }
}


//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char** argv) -> int
{
#ifdef USE_DGBPRINTF
  g_elemDebugPrintf = std::make_shared<nvvkhl::ElementDbgPrintf>();
#endif

  // Search paths
  g_applicationSearchPaths.push_back(NVPSystem::exePath() + PROJECT_DOWNLOAD_RELDIRECTORY);
  g_applicationSearchPaths.push_back(std::string("GLSL_" PROJECT_NAME));
  // Search path for shaders within the project
  g_applicationSearchPaths.push_back(NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY) + "shaders");
  g_applicationSearchPaths.push_back(NVPRO_CORE_DIR);
  // INSTALL_RELDIRECTORY is defined in CMakeLists.txt
  g_applicationSearchPaths.push_back(NVPSystem::exePath() + std::string("GLSL_" PROJECT_NAME));
  g_applicationSearchPaths.push_back(NVPSystem::exePath() + "media");


  // Create the logger, and redirect the printf to the logger
  g_elemLogger = std::make_shared<nvvkhl::SampleAppLog>();
  nvprintSetCallback([](int level, const char* fmt) { g_elemLogger->addLog(level, "%s", fmt); });
  g_elemLogger->setLogLevel(LOGBITS_INFO);


  // Vulkan Context creation information
  VkContextSettings vkSetup;
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
  vkSetup.instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  // All Vulkan extensions required by the sample
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR baryFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  VkPhysicalDeviceNestedCommandBufferFeaturesEXT nestedCmdFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NESTED_COMMAND_BUFFER_FEATURES_EXT};
  VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV reorderFeature{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV};
  vkSetup.deviceExtensions.emplace_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accel_feature);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rt_pipeline_feature);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_RAY_QUERY_EXTENSION_NAME, &ray_query_features);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  vkSetup.deviceExtensions.emplace_back(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &baryFeatures, false);
  vkSetup.deviceExtensions.emplace_back(VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjFeature);
  vkSetup.deviceExtensions.emplace_back(VK_EXT_NESTED_COMMAND_BUFFER_EXTENSION_NAME, &nestedCmdFeature);
  vkSetup.deviceExtensions.emplace_back(VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, &reorderFeature, false);

#ifdef USE_AFTERMATH
  // #Aftermath - Initialization
  nvvk::GpuCrashTracker gpuCrashTracker;
  static VkDeviceDiagnosticsConfigCreateInfoNV aftermathInfo{VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV};
  if(::isAftermathAvailable())  // Check if the Aftermath SDK is available (See CMake path)
  {
    gpuCrashTracker.initialize();
    aftermathInfo.flags =
        (VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV  // Additional information about the resource related to a GPU virtual address
         | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV  // Automatic checkpoints for all draw calls (ADD OVERHEAD)
         | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV);  // instructs the shader compiler to generate debug information (ADD OVERHEAD)
    vkSetup.deviceExtensions.emplace_back(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
    vkSetup.deviceExtensions.emplace_back(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME, &aftermathInfo);
    LOGW(
        "\n-------------------------------------------------------------------"
        "\nWARNING: Aftermath extensions enabled. This may affect performance."
        "\n-------------------------------------------------------------------\n\n");
  }
#endif  // USE_AFTERMATH


  // Request the creation of all needed queues
  vkSetup.queues = {VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT,  // GTC for rendering
                    VK_QUEUE_COMPUTE_BIT,                                                  // Compute
                    VK_QUEUE_TRANSFER_BIT};                                                // Transfer

  ValidationSettings vvlInfo{};
  // vvlInfo.validate_best_practices = true;
#ifdef USE_DGBPRINTF
  vvlInfo.validate_gpu_based         = {"GPU_BASED_DEBUG_PRINTF"};  // Adding the debug printf extension
  vvlInfo.printf_verbose             = VK_FALSE;
  vvlInfo.printf_to_stdout           = VK_FALSE;
  vvlInfo.printf_buffer_size         = 1024;
  vvlInfo.gpuav_reserve_binding_slot = false;
  vvlInfo.message_id_filter          = {0x76589099};  // Truncate the message when too long
#endif                                                // USE_DGBPRINTF
  ValidationSettings vvl(vvlInfo);
  vkSetup.instanceCreateInfoExt = vvl.buildPNextChain();  // Adding the validation layer settings

  // Creating the Vulkan context
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  if(!vkContext->isValid())
  {
    LOGE("Error in Vulkan context creation\n");
    std::exit(0);
  }

  // Loading Vulkan extension pointers
  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);

  // Setup the application information
  nvvkhl::ApplicationCreateInfo appInfo;
  appInfo.name             = PROJECT_NAME " Sample";
  appInfo.vSync            = false;
  appInfo.instance         = vkContext->getInstance();
  appInfo.device           = vkContext->getDevice();
  appInfo.physicalDevice   = vkContext->getPhysicalDevice();
  appInfo.queues           = vkContext->getQueueInfos();
  appInfo.imguiConfigFlags = ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appInfo);

  setWindowIcon(app->getWindowHandle());

  // Create Elements of the application
  g_elemCamera      = std::make_shared<nvvkhl::ElementCamera>();
  g_elemProfiler    = std::make_shared<nvvkhl::ElementProfiler>(false);
  g_elemBenchmark   = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);
  auto gltfRenderer = std::make_shared<gltfr::GltfRendererElement>();  // This is the main element of the application

  // Parsing arguments
  g_elemBenchmark->parameterLists().addFilename(".gltf|load a file", &g_inFilename);
  g_elemBenchmark->parameterLists().addFilename(".glb|load a file", &g_inFilename);
  g_elemBenchmark->parameterLists().add("hdr|load a HDR", &g_inHdr);
  g_elemBenchmark->setProfiler(g_elemProfiler);  // Linking the profiler to the benchmark parameters

  g_elemBenchmark->parameterLists().add("maxDepth", &gltfr::g_pathtraceSettings.maxDepth);
  g_elemBenchmark->parameterLists().add("maxSamples", &gltfr::g_pathtraceSettings.maxSamples);
  g_elemBenchmark->parameterLists().add("renderMode", (int*)&gltfr::g_pathtraceSettings.renderMode);

  g_elemBenchmark->parameterLists().add("forceExternalShaders", &gltfr::g_forceExternalShaders, true);


  app->addElement(g_elemBenchmark);  // Benchmark/tests and parameters
  app->addElement(gltfRenderer);     // Rendering the glTF scene
  app->addElement(g_elemCamera);     // Controlling the camera movement
  app->addElement(g_elemProfiler);   // GPU Profiler
#ifdef USE_DGBPRINTF
  app->addElement(g_elemDebugPrintf);                                                   // Debug printf
#endif                                                                                  // USE_DGBPRINTF
  app->addElement(std::make_unique<nvvkhl::ElementLogger>(g_elemLogger.get(), false));  // Add logger window
  app->addElement(std::make_unique<nvvkhl::ElementNvml>(false));                        // Add GPU monitor

  g_elemProfiler->setLabelUsage(false);  // Do not use labels for the profiler

  // Start Application: which will loop and call on"Functions" for all Elements
  app->run();

  // Cleanup
  gltfRenderer.reset();
  app.reset();
  vkContext.reset();

  return 0;
}
