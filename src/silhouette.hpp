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

#pragma once

#include <glm/glm.hpp>

namespace shaderio {
using namespace glm;
#include "shaders/shaderio.h"  // Shared between host and device
}  // namespace shaderio

#include "resources.hpp"


// This Silhouette class, which is used to extract the outline of a 3D object.
// There are two images, one with the information of the silhouette and the other
// which will be composed of the silhouette and the object itself.
class Silhouette
{
public:
  Silhouette() = default;
  ~Silhouette() { assert(!m_shader && "deinit must be called"); }


  void init(Resources& res);
  void deinit(Resources& res);

  void dispatch(VkCommandBuffer cmd, const VkExtent2D& imgSize, std::vector<VkDescriptorImageInfo>& imageDescriptors);
  void setColor(glm::vec3 color);

private:
  shaderio::SilhouettePushConstant m_pushConstant{};

  nvvk::DescriptorBindings m_bindings;
  VkShaderEXT              m_shader{};
  VkPipelineLayout         m_pipelineLayout{};       // Pipeline layout
  VkDescriptorSetLayout    m_descriptorSetLayout{};  // Descriptor set layout
};
