/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Per-node-operation evaluation: the table-driven dispatcher from InteractivityOp to actual
// behavior (spec chapters "Math Operations" through "Event Operations"). Two entry points:
//   - evaluateNodeOutput: pull-based, for a node's output *value* socket (pure math, variable/get,
//     event/onStart|onTick's synthesized outputs, ...).
//   - executeFlowNode: push-based, for a node whose input *flow* socket was just activated
//     (flow/*, variable/set, ...) - runs its side effect and activates its own output flow(s).
// Ops not yet implemented (see docs/interactivity.md coverage table) safely no-op: an
// unimplemented flow node just never activates an output flow, an unimplemented pure node
// returns its output type's default value - both logged once per op, never a crash.
//

#pragma once

#include <string>

#include "gltf_interactivity_graph.hpp"

namespace nvvkgltf {

class InteractivityGraphInstance;

InteractivityValue evaluateNodeOutput(InteractivityGraphInstance& instance, int nodeIndex, const std::string& socketName);
void executeFlowNode(InteractivityGraphInstance& instance, int nodeIndex, const std::string& enteredSocket);

// Human-readable rendering of any InteractivityValue (debug/log message substitution; also reused
// by the Graphs UI panel's variable/event value display, ui_interactivity.cpp).
std::string stringifyInteractivityValue(const InteractivityValue& v);

// Cubic Bézier easing (spec "variable/interpolate"/"pointer/interpolate": implicit endpoints
// P0(0,0)/P3(1,1), authored control points p1/p2), used by InteractivityGraphInstance's per-tick
// interpolation advance (gltf_interactivity_instance.cpp) to turn a linear time fraction `t` into
// an eased progress value `q`. `t` is clamped to [0,1] first (callers only invoke this once `t` is
// known to be in range - see the op's spec steps for the t<=0 / t>=1 short-circuits).
float cubicBezierEase(const glm::vec2& p1, const glm::vec2& p2, float t);

// Linearly (or, if `slerp` and both are float4, spherically) interpolates two same-typed
// InteractivityValues by coefficient `q` - shared by variable/interpolate and pointer/interpolate.
// Mismatched-type inputs return `a` unchanged (defensive; spec requires the graph to already
// guarantee matching types via the op's own value-socket typing).
InteractivityValue lerpInteractivityValue(const InteractivityValue& a, const InteractivityValue& b, float q, bool slerp);

}  // namespace nvvkgltf
