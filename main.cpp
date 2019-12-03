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
// This example is creating a scene with many similar objects and a plane. There are a few materials
// and a light direction.
// More details in simple.cpp
//

#include <array>
#include <chrono>
#include <iostream>

#include "nvh/fileoperations.hpp"
#include "nvh/inputparser.h"
#include "nvvkpp/context_vkpp.hpp"
#include "nvvkpp/utilities_vkpp.hpp"
#include "scene.hpp"

int const SAMPLE_SIZE_WIDTH  = 800;
int const SAMPLE_SIZE_HEIGHT = 600;

// Default search path for shaders
std::vector<std::string> defaultSearchPaths{
    "./",
    "../",
    std::string(PROJECT_NAME),
    std::string("SPV_" PROJECT_NAME),
    PROJECT_ABSDIRECTORY,
    NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY),
};

//--------------------------------------------------------------------------------------------------
//
//
int main(int argc, char** argv)
{

  // Parsing the command line: mandatory '-f' for the filename of the scene
  InputParser parser(argc, argv);
  std::string filename;
  if(parser.exist("-f"))
  {
    filename = parser.getString("-f");
  }
  else if(argc == 2 && nvh::endsWith(argv[1], ".gltf"))  // Drag&Drop
  {
    filename = argv[1];
  }
  else
  {
    filename = nvh::findFile("data/FlightHelmet.gltf", defaultSearchPaths);
  }

  std::string hdrFilename = parser.getString("-e");
  if(hdrFilename.empty())
  {
    hdrFilename = nvh::findFile("/data/environment.hdr", defaultSearchPaths);
  }


  // setup some basic things for the sample, logging file for example
  NVPSystem system(argv[0], PROJECT_NAME);

  nvvkpp::ContextCreateInfo deviceInfo;
  deviceInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);
  deviceInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME);
  deviceInfo.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);

  // Creating the Vulkan instance and device
  nvvkpp::Context vkctx{};
  //  vkctx.init(deviceInfo);
  vkctx.initInstance(deviceInfo);

  // Find all compatible devices
  auto compatibleDevices = vkctx.getCompatibleDevices(deviceInfo);
  assert(!compatibleDevices.empty());

  // Use a compatible device
  vkctx.initDevice(compatibleDevices[0], deviceInfo);


  VkScene example;
  example.setScene(filename);
  example.setEnvironmentHdr(hdrFilename);

  // Creating the window
  example.open(0, 0, SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT, PROJECT_NAME);

  // Window need to be opened to get the surface on which to draw
  const vk::SurfaceKHR surface = example.getVkSurface(vkctx.m_instance);
  vkctx.setGCTQueueWithPresent(surface);

  // Printing which GPU we are using
  const vk::PhysicalDevice physicalDevice = vkctx.m_physicalDevice;
  std::cout << "Using " << physicalDevice.getProperties().deviceName << std::endl;

  try
  {
    example.setup(vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
    example.createSurface(surface, SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT);
    example.createDepthBuffer();
    example.createRenderPass();
    example.createFrameBuffers();
    example.initExample();  // Now build the example
    example.initGUI(0);     // Using sub-pass 0
  }
  catch(const std::exception& e)
  {
    const char* what = e.what();
    std::cerr << what << std::endl;
    exit(1);
  }


  // Window system loop
  while(!example.isClosing() && example.pollEvents())
  {
    if(example.isOpen())
    {
      CameraManip.updateAnim();
      example.display();  // infinitely drawing
    }
  }

  example.destroy();
  vkctx.m_instance.destroySurfaceKHR(surface);
  vkctx.deinit();
}
