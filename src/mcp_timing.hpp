/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

//
// Optional MCP endpoint (--mcp) scoped to one job: measuring the effect of a shader change.
// It exposes three tools -- recompile, list timers, time a timer -- and relies on nvmcp's own
// generic tools for everything else (settings, logs, screenshots, profiler snapshot).
//

#ifdef USE_NVMCP

#include <cstdint>
#include <memory>

// Full definition, not a forward declaration: callers hand the result to nvapp::Application::
// addElement, which needs to see that nvmcp::Element derives from nvapp::IAppElement.
#include <nvmcp/element_mcp.hpp>

class GltfRenderer;

struct McpTimingCreateInfo
{
  uint16_t port{7671};
};

// Returns the element to add to the application, or nullptr if `renderer` is null.
std::shared_ptr<nvmcp::Element> createTimingMcpServer(const McpTimingCreateInfo& info,
                                                      const std::shared_ptr<GltfRenderer>& renderer);

#endif  // USE_NVMCP
