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
// Compiled representation of a KHR_interactivity behavior graph.
//
// This header defines the *static* graph model: parsed once from the glTF JSON at load
// time into an immutable, strongly-typed form (op enum instead of strings, pre-resolved
// socket references). It never changes after parse() succeeds. Runtime execution state
// (variable values, flow scheduling, delay timers) lives in InteractivityGraphInstance
// (gltf_interactivity_instance.hpp) - see docs/interactivity.md for the split rationale.
//
// Spec: https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_interactivity
//

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <glm/glm.hpp>
#include <tinygltf/tiny_gltf.h>

namespace nvvkgltf {

#define KHR_INTERACTIVITY_EXTENSION_NAME "KHR_interactivity"

//--------------------------------------------------------------------------------------------------
// Value system - one C++ type per KHR_interactivity `types[].signature` (spec section
// "Value Socket Types"). `eRef` is an opaque handle (e.g. an event or delay instance); `eCustom`
// is a vendor-defined struct type this implementation does not interpret.
//--------------------------------------------------------------------------------------------------
enum class InteractivityValueType : uint8_t
{
  eUnknown,
  eBool,
  eFloat,
  eFloat2,
  eFloat3,
  eFloat4,
  eFloat2x2,
  eFloat3x3,
  eFloat4x4,
  eInt,
  eRef,
  eCustom,
};

// An opaque reference value (spec `ref` type) - an index into an instance-owned handle table
// (event instances, delay instances, ...) or into one of the glTF document's own core arrays
// (nodes, materials, meshes, ...). Distinct from `int` so the variant stays unambiguous.
//
// `category` disambiguates refs whose numeric `handle` alone would collide: a JSON-Pointer literal
// ref (spec "Variables and Types") is parsed from a path like "/nodes/17" or "/materials/17" -
// parseInteractivityLiteral() extracts the trailing index into `handle` and the owning-array name
// into `category`, so two literals with the same index but different owning arrays compare unequal
// (they used to collapse onto the same InteractivityRef and compare equal via `ref/eq`). Refs the
// engine mints internally for a glTF core-array element (ScenePointerResolver's indexRef(),
// Scene::notifyNodeSelected/HoverChanged's node refs) set the matching category so they still
// compare equal to an equivalent literal. Refs with no glTF-array meaning (event/delay occurrence
// handles) leave `category` empty, matching how they were compared before this field existed.
struct InteractivityRef
{
  int32_t     handle = -1;
  std::string category;
  bool        operator==(const InteractivityRef&) const = default;
};

// std::monostate = "no value" (type-default not yet resolved, or an unsupported custom type).
using InteractivityValue =
    std::variant<std::monostate, bool, int32_t, float, glm::vec2, glm::vec3, glm::vec4, glm::mat2, glm::mat3, glm::mat4, InteractivityRef>;

InteractivityValueType interactivityValueType(const InteractivityValue& value);
InteractivityValue     defaultInteractivityValue(InteractivityValueType type);
InteractivityValueType parseTypeSignature(const std::string& signature);
// Parses a `value`/inline-literal JSON array (spec "Inline Values") into a typed InteractivityValue.
InteractivityValue parseInteractivityLiteral(InteractivityValueType type, const tinygltf::Value& jsonArray);

//--------------------------------------------------------------------------------------------------
// Node operation catalog (spec chapters "Math Operations" through "Event Operations").
// Every op the spec defines gets an enum entry so declaration parsing can recognize the full
// catalog even before the evaluator implements it (unimplemented ops are safe no-ops at
// runtime, logged once - see InteractivityGraphInstance). eUnknown covers unrecognized core
// op strings (structurally-invalid declaration, spec ​section 5596) and eExtensionDefined covers
// well-formed `declaration.extension` ops this build doesn't special-case (spec section 339,
// graceful per-node degradation).
//--------------------------------------------------------------------------------------------------
enum class InteractivityOp : uint16_t
{
  eUnknown = 0,
  eExtensionDefined,  // declaration.extension present, op not specially recognized by this build

  // math/* - constants
  eMathE,
  eMathPi,
  eMathTau,
  eMathInf,
  eMathNaN,

  // math/* - float arithmetic (component-wise; also reused, with int/bool-typed sockets, by the
  // integer/boolean catalog below - dispatch keys on op + operand type, not op string alone)
  eMathAbs,
  eMathSign,
  eMathTrunc,
  eMathFloor,
  eMathCeil,
  eMathRound,
  eMathFract,
  eMathNeg,
  eMathAdd,
  eMathSub,
  eMathMul,
  eMathDiv,
  eMathRem,
  eMathMin,
  eMathMax,
  eMathClamp,
  eMathSaturate,
  eMathMix,
  eMathSmoothStep,

  // math/* - comparison / special
  eMathEq,
  eMathLt,
  eMathLe,
  eMathGt,
  eMathGe,
  eMathIsNaN,
  eMathIsInf,
  eMathSelect,
  eMathSwitch,
  eMathRandom,

  // math/* - trig / hyperbolic / exponential
  eMathRad,
  eMathDeg,
  eMathSin,
  eMathCos,
  eMathTan,
  eMathAsin,
  eMathAcos,
  eMathAtan,
  eMathAtan2,
  eMathSinh,
  eMathCosh,
  eMathTanh,
  eMathAsinh,
  eMathAcosh,
  eMathAtanh,
  eMathExp,
  eMathLog,
  eMathLog2,
  eMathLog10,
  eMathSqrt,
  eMathCbrt,
  eMathPow,

  // math/* - vector / matrix / quaternion
  eMathLength,
  eMathNormalize,
  eMathDot,
  eMathCross,
  eMathRotate2D,
  eMathRotate3D,
  eMathTransform,
  eMathSlerp,
  eMathTranspose,
  eMathInverse,
  eMathDeterminant,
  eMathMatMul,
  eMathMatCompose,
  eMathMatDecompose,
  eMathQuatConjugate,
  eMathQuatMul,
  eMathQuatAngleBetween,
  eMathQuatFromAxisAngle,
  eMathQuatToAxisAngle,
  eMathQuatFromDirections,
  eMathQuatFromUpForward,
  eMathQuatFromAngles,
  eMathQuatSlerp,

  // math/* - swizzle
  eMathCombine2,
  eMathCombine3,
  eMathCombine4,
  eMathCombine2x2,
  eMathCombine3x3,
  eMathCombine4x4,
  eMathExtract2,
  eMathExtract3,
  eMathExtract4,
  eMathExtract2x2,
  eMathExtract3x3,
  eMathExtract4x4,

  // math/* - integer-only bitwise
  eMathNot,
  eMathAnd,
  eMathOr,
  eMathXor,
  eMathAsr,
  eMathLsl,
  eMathClz,
  eMathCtz,
  eMathPopcnt,

  // math/* - color
  eMathRgbToOkLCh,
  eMathRgbFromOkLCh,

  // ref/*
  eRefEq,

  // type/* - conversion
  eTypeBoolToInt,
  eTypeBoolToFloat,
  eTypeIntToBool,
  eTypeIntToFloat,
  eTypeFloatToBool,
  eTypeFloatToInt,

  // flow/* - control flow
  eFlowSequence,
  eFlowBranch,
  eFlowSwitch,
  eFlowWhile,
  eFlowFor,
  eFlowDoN,
  eFlowMultiGate,
  eFlowWaitAll,
  eFlowThrottle,
  eFlowSetDelay,
  eFlowCancelDelay,

  // variable/*
  eVariableGet,
  eVariableSet,
  eVariableInterpolate,

  // pointer/*
  ePointerGet,
  ePointerSet,
  ePointerInterpolate,

  // animation/*
  eAnimationStart,
  eAnimationStop,
  eAnimationStopAt,

  // event/*
  eEventOnStart,
  eEventOnTick,
  eEventStopPropagation,
  eEventReceive,
  eEventSend,

  // debug/*
  eDebugLog,

  eCount,
};

// String <-> enum for core (no `declaration.extension`) op ids, e.g. "math/add" <-> eMathAdd.
InteractivityOp    parseCoreOpString(const std::string& op);
const std::string& interactivityOpString(InteractivityOp op);

//--------------------------------------------------------------------------------------------------
// declarations[] - maps a declaration index to an operation. Structurally invalid per spec 5596
// (unknown core op with no `extension`, or a spec op carrying custom sockets) rejects the whole
// graph at parse(); a well-formed but unrecognized `extension` op degrades only its nodes to
// no-ops at runtime (spec 339-345) rather than failing the graph.
//--------------------------------------------------------------------------------------------------
struct InteractivityDeclaration
{
  InteractivityOp op = InteractivityOp::eUnknown;
  std::string     opString;   // raw JSON "op" (diagnostics, and extension-op dispatch key)
  std::string     extension;  // JSON "extension"; empty for core ops
};

//--------------------------------------------------------------------------------------------------
// One input value socket of a node: either a literal (inline `value`+`type`) or a reference to
// another node's output socket (`node`+`socket`, spec "Output Socket References" - the
// referenced node index MUST be less than this node's index, enforced at parse time).
//--------------------------------------------------------------------------------------------------
struct InteractivityValueSocket
{
  bool                   isReference  = false;
  int                    sourceNode   = -1;       // valid iff isReference
  std::string            sourceSocket = "value";  // valid iff isReference (spec default "value")
  InteractivityValue     literal;                 // valid iff !isReference
  InteractivityValueType literalType = InteractivityValueType::eUnknown;
};

// One output flow socket of a node: the node+input-flow-socket it activates (spec default "in").
struct InteractivityFlowSocket
{
  int         targetNode   = -1;
  std::string targetSocket = "in";
};

//--------------------------------------------------------------------------------------------------
// One node in the graph. `configuration` is kept as raw tinygltf::Value (parsed per-op by the
// evaluator, e.g. `variable/get`'s "variable" index or `flow/switch`'s "cases" array) since its
// shape is entirely op-specific. `values`/`flows` are keyed by socket id, matching the JSON.
//--------------------------------------------------------------------------------------------------
struct InteractivityNode
{
  int                                                       declarationIndex = -1;
  std::unordered_map<std::string, tinygltf::Value>          configuration;
  std::unordered_map<std::string, InteractivityValueSocket> values;
  std::unordered_map<std::string, InteractivityFlowSocket>  flows;
};

struct InteractivityTypeInfo
{
  InteractivityValueType signature = InteractivityValueType::eUnknown;
};

struct InteractivityVariableInfo
{
  int                typeIndex = -1;
  InteractivityValue initialValue;
};

struct InteractivityEventValueInfo
{
  int                typeIndex = -1;
  InteractivityValue initialValue;
};

struct InteractivityEventInfo
{
  std::string                                                  id;  // external pub/sub name
  std::unordered_map<std::string, InteractivityEventValueInfo> values;
};

//--------------------------------------------------------------------------------------------------
// InteractivityGraph - the compiled, immutable form of one `KHR_interactivity.graphs[]` entry.
// One instance is owned per graph by nvvkgltf::Scene; runtime state lives in a separate
// InteractivityGraphInstance (see gltf_interactivity_instance.hpp) so Play/Pause/Reset never
// touches this object and multiple concurrent instances of one graph are possible later.
//--------------------------------------------------------------------------------------------------
class InteractivityGraph
{
public:
  // Parses one entry of `KHR_interactivity.graphs[]`. Returns std::nullopt if the graph is
  // structurally invalid per spec (whole-graph rejection) - the caller should log and skip it,
  // not fail the glTF load.
  static std::optional<InteractivityGraph> parse(const tinygltf::Value& graphJson, std::string name, int graphIndex);

  const std::string&                            name() const { return m_name; }
  int                                           graphIndex() const { return m_graphIndex; }
  const std::vector<InteractivityDeclaration>&  declarations() const { return m_declarations; }
  const std::vector<InteractivityNode>&         nodes() const { return m_nodes; }
  const std::vector<InteractivityTypeInfo>&     types() const { return m_types; }
  const std::vector<InteractivityVariableInfo>& variables() const { return m_variables; }
  const std::vector<InteractivityEventInfo>&    events() const { return m_events; }

  // Node indices whose declaration op is eEventOnStart / eEventOnTick, in JSON (ascending index)
  // order - the order the spec requires lifecycle events to activate in.
  const std::vector<int>& onStartNodes() const { return m_onStartNodes; }
  const std::vector<int>& onTickNodes() const { return m_onTickNodes; }

  // Interaction-event handler binding for `event/onHoverIn`/`onHoverOut` (declaration.extension ==
  // "KHR_node_hoverability") and `event/onSelect` (declaration.extension == "KHR_node_selectability").
  // These are ordinary graph nodes, spec-bound to a specific glTF node via that node's own
  // `configuration.nodeIndex` (not positional/implicit) - so unlike onStart/onTick this is a map, not
  // a flat list. Values are graph node indices in ascending JSON order (spec: same-`nodeIndex`
  // handlers activate in that order). A negative or non-integer `nodeIndex` config means "never
  // activates" per spec (5.7 of KHR_node_hoverability/selectability) - such nodes are simply absent
  // from these maps. Out-of-range-but-non-negative `nodeIndex` values (pointing at a glTF node that
  // doesn't exist) are also never looked up in practice, since real events only key on real node
  // indices - this parser intentionally doesn't need the glTF node count to enforce that.
  const std::unordered_map<int, std::vector<int>>& hoverInHandlers() const { return m_hoverInHandlers; }
  const std::unordered_map<int, std::vector<int>>& hoverOutHandlers() const { return m_hoverOutHandlers; }
  const std::unordered_map<int, std::vector<int>>& selectHandlers() const { return m_selectHandlers; }

private:
  std::string                               m_name;
  int                                       m_graphIndex = -1;
  std::vector<InteractivityTypeInfo>        m_types;
  std::vector<InteractivityVariableInfo>    m_variables;
  std::vector<InteractivityEventInfo>       m_events;
  std::vector<InteractivityDeclaration>     m_declarations;
  std::vector<InteractivityNode>            m_nodes;
  std::vector<int>                          m_onStartNodes;
  std::vector<int>                          m_onTickNodes;
  std::unordered_map<int, std::vector<int>> m_hoverInHandlers;
  std::unordered_map<int, std::vector<int>> m_hoverOutHandlers;
  std::unordered_map<int, std::vector<int>> m_selectHandlers;
};

// Parses `model.extensions["KHR_interactivity"]` into one InteractivityGraph per `graphs[]`
// entry. Graphs that fail to parse (structurally invalid, spec 5596) are logged and omitted -
// this never fails the overall glTF load. Returns an empty vector if the extension is absent.
std::vector<InteractivityGraph> parseInteractivityGraphs(const tinygltf::Model& model);

}  // namespace nvvkgltf
