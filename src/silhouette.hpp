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

#include "nvvk/compute_vk.hpp"

#include "resources.hpp"
#include "slang_compiler.hpp"


extern std::vector<std::string> g_applicationSearchPaths;  // Used by the shader manager

namespace gltfr {


// This Silhouette class, which is used to extract the outline of a 3D object.
// There are two images, one with the information of the silhouette and the other
// which will be composed of the silhouette and the object itself.
enum SilhoutteImages
{
  eObjectID = 0,
  eRGBAIImage,
};
class Silhouette : public nvvk::PushComputeDispatcher<DH::PushConstantSilhouette, SilhoutteImages>
{
public:
  Silhouette(Resources& res)
      : PushComputeDispatcher(res.ctx.device)
  {
    // Slang version
    std::string             filename = nvh::findFile("silhouette.comp.slang", g_applicationSearchPaths, true);
    slang::ICompileRequest* request  = res.m_slangC->createCompileRequest(filename);
    if(SLANG_FAILED(request->compile()))
    {
      LOGE("Error compiling shader %s, %s\n", filename.c_str(), request->getDiagnosticOutput());
      return;
    }
    std::vector<uint32_t> spirvCode;
    res.m_slangC->getSpirvCode(request, spirvCode);
    VkShaderModuleCreateInfo shaderModuleCreateInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirvCode.size() * sizeof(uint32_t),
        .pCode    = spirvCode.data(),
    };

    // GLSL version
    // shaderc::SpvCompilationResult compilationResult =
    //     res.compileGlslShader("silhouette.comp.glsl", shaderc_shader_kind::shaderc_compute_shader);
    // VkShaderModuleCreateInfo shaderModuleCreateInfo;
    // if(!res.createShaderModuleCreateInfo(compilationResult, shaderModuleCreateInfo))
    //   return;


    PushComputeDispatcher::setCode(shaderModuleCreateInfo.pCode, shaderModuleCreateInfo.codeSize);
    PushComputeDispatcher::getBindings().addBinding(eObjectID, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    PushComputeDispatcher::getBindings().addBinding(eRGBAIImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    PushComputeDispatcher::finalizePipeline();

    m_pushConstant.color = glm::vec3(1, 1, 1);
  }

  void dispatch(VkCommandBuffer cmd, const VkExtent2D& imgSize)
  {
    glm::uvec3 blocks = {PushComputeDispatcher::getBlockCount(imgSize.width, WORKGROUP_SIZE),
                         PushComputeDispatcher::getBlockCount(imgSize.height, WORKGROUP_SIZE), 1};
    PushComputeDispatcher::dispatchBlocks(cmd, blocks, &m_pushConstant);
  }

  void setColor(glm::vec3 color) { m_pushConstant.color = color; }

private:
  DH::PushConstantSilhouette m_pushConstant{};
};

}  // namespace gltfr
