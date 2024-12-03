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

/*

This class is used to hold the resources for the renderer.

It holds the Vulkan resources, such as 
- the device
- the physical device
- the queue
- the queue family
- the allocator
- the G-Buffers (just the color final image)
- the temporary command pool
- and the GLSL compiler.

*/

// nvpro-core
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/glsl_compiler.hpp"

// Local to application
#include "slang_compiler.hpp"

namespace gltfr {
struct Queue
{
  VkQueue  queue       = VK_NULL_HANDLE;
  uint32_t familyIndex = ~0U;
};

struct VulkanInfo
{
  VkDevice         device{};
  VkPhysicalDevice physicalDevice{};
  Queue            GCT0;
  Queue            compute;
  Queue            transfer;
};

// Resources for the renderer
class Resources
{
public:
  void init(VulkanInfo& _ctx);
  void resizeGbuffers(const VkExtent2D& size);

  // Create a temporary command buffer
  VkCommandBuffer createTempCmdBuffer();
  void            submitAndWaitTempCmdBuffer(VkCommandBuffer cmd);

  // Shader compilation
  shaderc::SpvCompilationResult compileGlslShader(const std::string& filename, shaderc_shader_kind shaderKind) const;
  VkShaderModule                createShaderModule(shaderc::SpvCompilationResult& compResult) const;
  static bool createShaderModuleCreateInfo(shaderc::SpvCompilationResult& compResult, VkShaderModuleCreateInfo& createInfo);
  void resetSlangCompiler();

  // Did the resolution changed?
  bool hasGBuffersChanged() const { return m_hasGBufferChanged; }
  void setGBuffersChanged(bool changed) { m_hasGBufferChanged = changed; }

  bool hasGlslCompiler() const { return m_glslC != nullptr; }
  bool hasSlangCompiler() const { return m_slangC != nullptr; }

  // Vulkan context resources
  VulkanInfo ctx{};

  std::unique_ptr<nvvk::ResourceAllocatorDma> m_allocator{};
  std::unique_ptr<nvvkhl::GBuffer>            m_finalImage{};  // G-Buffers: color
  std::unique_ptr<nvvk::CommandPool>          m_tempCommandPool{};
  std::unique_ptr<nvvkhl::GlslCompiler>       m_glslC{};
  std::unique_ptr<SlangCompiler>              m_slangC{};

private:
  bool m_hasGBufferChanged{false};
};

}  // namespace gltfr