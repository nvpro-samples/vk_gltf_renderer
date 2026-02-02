/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <nvapp/application.hpp>
#include <nvvk/profiler_vk.hpp>

#include "resources.hpp"

class BaseRenderer
{
public:
  BaseRenderer()          = default;
  virtual ~BaseRenderer() = default;


  virtual void onAttach(Resources& resources, nvvk::ProfilerGpuTimer* profiler) { m_profiler = profiler; }
  virtual void onDetach(Resources& resources) {};
  virtual void onResize(VkCommandBuffer cmd, const VkExtent2D& size, Resources& resources) {};
  virtual void onRender(VkCommandBuffer cmd, Resources& resources) {};
  virtual void onUIMenu() {};

  [[nodiscard]] virtual bool onUIRender(Resources&) { return false; }

  //---
  virtual void compileShader(Resources& resources, bool fromFile = true) {};
  virtual void createPipeline(Resources& resources) {};

protected:
  nvvk::ProfilerGpuTimer* m_profiler{nullptr};
};
