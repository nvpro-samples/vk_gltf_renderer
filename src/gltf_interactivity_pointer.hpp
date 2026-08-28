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
// JSON-Pointer-Template parsing (KHR_interactivity's `pointer/get`/`set`/`interpolate` addressing
// scheme - the same template family KHR_animation_pointer's *concrete* paths belong to) plus the
// abstract glTF-model read/write surface those ops need.
//
// Kept dependency-light on purpose: this header must NOT pull in gltf_scene.hpp. The core graph
// engine (gltf_interactivity_instance/eval) only needs the abstract InteractivityPointerResolver
// interface to stay unit-testable without a Scene. The concrete Scene-backed implementation lives
// in gltf_interactivity_scene_pointer.hpp/cpp.
//

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "gltf_interactivity_graph.hpp"

namespace nvvkgltf {

class InteractivityGraphInstance;

// One segment of a parsed pointer template, e.g. "/nodes/[nodeIndex]/translation" ->
// {literal "/nodes/"}, {intParam "nodeIndex"}, {literal "/translation"}.
struct PointerTemplateSegment
{
  enum class Kind
  {
    eLiteral,
    eIntParam,  // `[name]` - substituted from an int-typed input value socket named `name`
    eRefParam,  // `{name}` - substituted from a ref-typed input value socket named `name`
  };
  Kind        kind = Kind::eLiteral;
  std::string text;  // literal text, or the parameter's socket name for eIntParam/eRefParam
};

std::vector<PointerTemplateSegment> parsePointerTemplate(const std::string& templateStr);

// Evaluates every parameter segment's named input socket on `node` and substitutes it into the
// template, producing a concrete JSON pointer path (e.g. "/nodes/3/translation"). Returns
// nullopt if any parameter socket is missing or holds the wrong type.
//
// eRefParam is resolved via InteractivityRef::handle as a best-effort integer substitution - the
// spec's "object reference" parameter kind is under-specified enough (relative to `[int]` params,
// which cover the overwhelmingly common "index into an array" case) that this is a documented
// simplification rather than a full implementation. See docs/interactivity.md.
std::optional<std::string> substitutePointerTemplate(const std::vector<PointerTemplateSegment>& segments,
                                                     InteractivityGraphInstance&                instance,
                                                     const InteractivityNode&                   node);

//--------------------------------------------------------------------------------------------------
// Abstract glTF-model read/write surface for pointer/get, pointer/set, pointer/interpolate.
//--------------------------------------------------------------------------------------------------
class InteractivityPointerResolver
{
public:
  virtual ~InteractivityPointerResolver() = default;

  // Reads the property at `concretePath`. Returns nullopt if the path doesn't resolve to a
  // property this resolver knows how to read (spec: pointer/get reports this via its `isValid`
  // output, not an error flow - see gltf_interactivity_eval.cpp).
  virtual std::optional<InteractivityValue> get(const std::string& concretePath) const = 0;

  // Writes `value` to the property at `concretePath`. Returns false if the path is unresolvable,
  // the property isn't writable, or `value`'s type doesn't match the property (spec: pointer/set
  // reports this via its `err` output flow).
  virtual bool set(const std::string& concretePath, const InteractivityValue& value) = 0;
};

}  // namespace nvvkgltf
