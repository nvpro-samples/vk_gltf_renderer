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

//
// Abstract base class for renderers (path tracer, rasterizer, etc.).
// Defines the virtual interface for attach/detach, resize, render,
// shader compilation, and pipeline creation that each concrete
// renderer implementation must provide.
//

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
  virtual void onSceneInvalidated(Resources& resources) {};

  // A scene edit changed appearance in a way motion vectors can't describe (material/light
  // property write - e.g. a KHR_interactivity pointer/set on a texture-transform offset, or an
  // Inspector material edit). Unlike onSceneInvalidated() (whole-scene swap: frees GPU resources,
  // drops sort/record state), this is a lightweight per-edit DLSS temporal-history discard only -
  // see docs/denoising.md.
  virtual void notifyDlssContentReset(Resources& resources) {};

  [[nodiscard]] virtual bool onUIRender(Resources&) { return false; }

  //---
  // Returns false only when a from-file compile was requested and the Slang compile failed. The
  // renderer then runs the build-time embedded SPIR-V: the previously active shader does not
  // survive a failed reload, so callers that report success must report this.
  virtual bool compileShader(Resources& resources, bool fromFile = true) { return true; };
  virtual void createPipeline(Resources& resources) {};
  virtual void freeRecordCommandBuffer(Resources& resources) {};

protected:
  nvvk::ProfilerGpuTimer* m_profiler{nullptr};
};
