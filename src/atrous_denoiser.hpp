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

#pragma once

#include "resources.hpp"
#include "nvvk/compute_vk.hpp"
#include "imgui/imgui_helper.h"

#include "_autogen/denoise.comp.glsl.h"

namespace gltfr {

extern bool g_forceExternalShaders;

// This Denoiser class is used to remove noise from an image.
// The implementation is based on the paper "A-Trous Wavelet Transform for Fast Global Illumination Filtering"
// See: https://jo.dreggn.org/home/2010_atrous.pdf
enum AtrousDenoiserImages
{
  eNoisyImage = 0,
  eNormalDepthImage,
  eDenoisedImage,
};

class AtrousDenoiser : public nvvk::PushComputeDispatcher<DH::PushConstantDenoiser, AtrousDenoiserImages>
{
public:
  // images: inColor, normal-depth, outColor
  AtrousDenoiser(Resources& res)
      : PushComputeDispatcher(res.ctx.device)
  {

    VkShaderModuleCreateInfo shaderModuleCreateInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    if(res.hasSlangCompiler() && g_forceExternalShaders)
    {

      shaderc::SpvCompilationResult compilationResult =
          res.compileGlslShader("denoise.comp.glsl", shaderc_shader_kind::shaderc_compute_shader);
      if(!res.createShaderModuleCreateInfo(compilationResult, shaderModuleCreateInfo))
        return;
    }
    else
    {
      // Pre-compiled version
      shaderModuleCreateInfo = {.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                .codeSize = sizeof(denoise_comp_glsl),
                                .pCode    = &denoise_comp_glsl[0]};
    }

    PushComputeDispatcher::getBindings().addBinding(eNoisyImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    PushComputeDispatcher::getBindings().addBinding(eNormalDepthImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    PushComputeDispatcher::getBindings().addBinding(eDenoisedImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    PushComputeDispatcher::setCode(shaderModuleCreateInfo.pCode, shaderModuleCreateInfo.codeSize);
    PushComputeDispatcher::finalizePipeline();

    m_pushConstant.stepWidth = 1;
    m_pushConstant.colorPhi  = 0.5f;
    m_pushConstant.normalPhi = 1.0f;
    m_pushConstant.depthPhi  = 1.0f;
  }

  void render(VkCommandBuffer       cmd,
              VkExtent2D            imgSize,
              VkDescriptorImageInfo colorBuffer,        // Original color buffer (noisy)
              VkDescriptorImageInfo resultBuffer,       // Result of the denoise buffer
              VkDescriptorImageInfo normalDepthBuffer,  // Normal and depth buffer
              VkDescriptorImageInfo tmpBuffer)          // temp buffer for ping-pong
  {


    // Number of denoising iterations
    m_pushConstant.colorPhi  = colorPhi;
    m_pushConstant.normalPhi = normalPhi * normalPhi;
    m_pushConstant.depthPhi  = depthPhi * depthPhi;

    if(numIterations % 2 == 0)
    {  // To make sure the end result is in resultBuffer
      std::swap(resultBuffer, tmpBuffer);
    }

    for(int i = 0; i < numIterations; i++)
    {
      // Update push constants
      m_pushConstant.stepWidth = 1 << i;                            // 1, 2, 4, 8, 16, 32, 64, 128
      m_pushConstant.colorPhi  = (colorPhi * colorPhi) / (1 << i);  // 2^-i

      // Update descriptor sets to point to current input/output
      PushComputeDispatcher::updateBinding(eNoisyImage, colorBuffer.imageView, VK_IMAGE_LAYOUT_GENERAL);
      PushComputeDispatcher::updateBinding(eNormalDepthImage, normalDepthBuffer.imageView, VK_IMAGE_LAYOUT_GENERAL);
      PushComputeDispatcher::updateBinding(eDenoisedImage, resultBuffer.imageView, VK_IMAGE_LAYOUT_GENERAL);

      // Bind pipeline, descriptor sets, push constants
      glm::uvec3 blocks = {PushComputeDispatcher::getBlockCount(imgSize.width, WORKGROUP_SIZE),
                           PushComputeDispatcher::getBlockCount(imgSize.height, WORKGROUP_SIZE), 1};
      PushComputeDispatcher::dispatchBlocks(cmd, blocks, &m_pushConstant);

      // Swap buffers for next iteration
      auto temp    = resultBuffer;
      colorBuffer  = resultBuffer;
      resultBuffer = tmpBuffer;
      tmpBuffer    = temp;
    }
  }

  void onUi()
  {
    namespace PE = ImGuiH::PropertyEditor;
    if(PE::treeNode("Denoiser"))
    {
      PE::Checkbox("Activate", (bool*)&isActive, "Edge-Avoiding À-Trous Wavelet Transform for fast Global Illumination Filtering");
      PE::SliderFloat("Color Phi", &colorPhi, 0.0f, 10.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
      PE::SliderFloat("Normal Phi", &normalPhi, 0.0f, 1.0f, "%.3f");
      PE::SliderFloat("Depth Phi", &depthPhi, 0.0f, 1.0f, "%.3f");
      PE::SliderInt("Iterations", &numIterations, 1, 8);
      PE::treePop();
    }
  }

  bool isActivated() const { return isActive; }

private:
  float colorPhi      = 0.5f;
  float normalPhi     = 1.f;
  float depthPhi      = 0.1f;
  bool  isActive      = false;
  int   numIterations = 1;

  DH::PushConstantDenoiser m_pushConstant{};
};

}  // namespace gltfr
