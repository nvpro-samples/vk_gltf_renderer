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
// Compiled representation of a KHR_interactivity behavior graph: the *static* graph model, parsed
// once from the glTF JSON at load time into an immutable, strongly-typed form (op enum instead of
// strings, pre-resolved socket references). Never changes after parse() succeeds -- runtime
// execution state (variable values, flow scheduling, delay timers) lives in
// InteractivityGraphInstance (gltf_interactivity_instance.cpp). See docs/interactivity.md.
//

#include <algorithm>
#include <charconv>
#include <limits>
#include <string>
#include <unordered_map>

#include <nvutils/logger.hpp>

#include "gltf_interactivity_graph.hpp"

namespace nvvkgltf {

//--------------------------------------------------------------------------------------------------
// Value system helpers
//--------------------------------------------------------------------------------------------------

InteractivityValueType interactivityValueType(const InteractivityValue& value)
{
  switch(value.index())
  {
    case 1:
      return InteractivityValueType::eBool;
    case 2:
      return InteractivityValueType::eInt;
    case 3:
      return InteractivityValueType::eFloat;
    case 4:
      return InteractivityValueType::eFloat2;
    case 5:
      return InteractivityValueType::eFloat3;
    case 6:
      return InteractivityValueType::eFloat4;
    case 7:
      return InteractivityValueType::eFloat2x2;
    case 8:
      return InteractivityValueType::eFloat3x3;
    case 9:
      return InteractivityValueType::eFloat4x4;
    case 10:
      return InteractivityValueType::eRef;
    default:
      return InteractivityValueType::eUnknown;
  }
}

InteractivityValue defaultInteractivityValue(InteractivityValueType type)
{
  switch(type)
  {
    case InteractivityValueType::eBool:
      return false;
    case InteractivityValueType::eInt:
      return int32_t{0};
    case InteractivityValueType::eFloat:
      return 0.0f;
    case InteractivityValueType::eFloat2:
      return glm::vec2(0.0f);
    case InteractivityValueType::eFloat3:
      return glm::vec3(0.0f);
    case InteractivityValueType::eFloat4:
      return glm::vec4(0.0f);
    case InteractivityValueType::eFloat2x2:
      return glm::mat2(0.0f);
    case InteractivityValueType::eFloat3x3:
      return glm::mat3(0.0f);
    case InteractivityValueType::eFloat4x4:
      return glm::mat4(0.0f);
    case InteractivityValueType::eRef:
      return InteractivityRef{};
    default:
      return std::monostate{};
  }
}

InteractivityValueType parseTypeSignature(const std::string& signature)
{
  static const std::unordered_map<std::string, InteractivityValueType> table = {
      {"bool", InteractivityValueType::eBool},         {"float", InteractivityValueType::eFloat},
      {"float2", InteractivityValueType::eFloat2},     {"float3", InteractivityValueType::eFloat3},
      {"float4", InteractivityValueType::eFloat4},     {"float2x2", InteractivityValueType::eFloat2x2},
      {"float3x3", InteractivityValueType::eFloat3x3}, {"float4x4", InteractivityValueType::eFloat4x4},
      {"int", InteractivityValueType::eInt},           {"ref", InteractivityValueType::eRef},
      {"custom", InteractivityValueType::eCustom},
  };
  auto it = table.find(signature);
  return it != table.end() ? it->second : InteractivityValueType::eUnknown;
}

namespace {
// Number of scalar components a literal `value` JSON array must have for a given type
// (spec "Inline Values" / variable `value` syntax).
int componentCount(InteractivityValueType type)
{
  switch(type)
  {
    case InteractivityValueType::eBool:
    case InteractivityValueType::eFloat:
    case InteractivityValueType::eInt:
    case InteractivityValueType::eRef:
      return 1;
    case InteractivityValueType::eFloat2:
      return 2;
    case InteractivityValueType::eFloat3:
      return 3;
    case InteractivityValueType::eFloat4:
    case InteractivityValueType::eFloat2x2:
      return 4;
    case InteractivityValueType::eFloat3x3:
      return 9;
    case InteractivityValueType::eFloat4x4:
      return 16;
    default:
      return 0;
  }
}
}  // namespace

InteractivityValue parseInteractivityLiteral(InteractivityValueType type, const tinygltf::Value& jsonArray)
{
  const int need = componentCount(type);
  if(need == 0 || !jsonArray.IsArray() || static_cast<int>(jsonArray.ArrayLen()) < need)
    return std::monostate{};

  // Strict JSON has no numeric token for IEEE-754 Infinity/NaN, so literal float components that
  // need them are authored as quoted string tokens instead (confirmed against Khronos's own
  // conformance suite, e.g. Tests/Interactivity/mathtests.glb's math/length and math/matDecompose
  // cases) - not documented in the core spec text, but real content depends on it.
  // tinygltf::Value::GetNumberAsDouble() on a string value returns 0.0, silently swallowing these.
  bool ok  = true;
  auto num = [&](int i) -> double {
    const tinygltf::Value& v = jsonArray.Get(static_cast<size_t>(i));
    if(v.IsString())
    {
      const std::string& s = v.Get<std::string>();
      if(s == "Infinity")
        return std::numeric_limits<double>::infinity();
      if(s == "-Infinity")
        return -std::numeric_limits<double>::infinity();
      if(s == "NaN")
        return std::numeric_limits<double>::quiet_NaN();
      ok = false;  // unrecognized string token -> fail the whole literal rather than silently coercing to 0
      return 0.0;
    }
    return v.GetNumberAsDouble();
  };

  auto result = [&]() -> InteractivityValue {
    switch(type)
    {
      case InteractivityValueType::eBool: {
        const tinygltf::Value& v = jsonArray.Get(size_t{0});
        return v.IsBool() ? v.Get<bool>() : (v.GetNumberAsDouble() != 0.0);
      }
      case InteractivityValueType::eInt:
        return static_cast<int32_t>(jsonArray.Get(size_t{0}).GetNumberAsInt());
      case InteractivityValueType::eFloat:
        return static_cast<float>(num(0));
      case InteractivityValueType::eRef: {
        // Spec ("Variables and Types"): "Values for the reference type are specified using static
        // JSON Pointers without any template parameters" - e.g. "/nodes/17", NOT a raw integer
        // handle. Resolved here via the same "ref.handle == target's index within its owning array"
        // convention ScenePointerResolver's object-model reads already use, by taking the trailing
        // numeric path segment (glTF's own core-array references, e.g. "/animations/1", are always
        // exactly "/category/index" - this doesn't attempt to resolve arbitrary sub-property paths).
        // An unresolvable/malformed pointer is a null reference (handle -1), matching spec.
        const tinygltf::Value& v = jsonArray.Get(size_t{0});
        if(!v.IsString())
          return InteractivityRef{};
        const std::string& s         = v.Get<std::string>();
        const size_t       lastSlash = s.rfind('/');
        if(lastSlash == std::string::npos)
          return InteractivityRef{};
        const std::string segment = s.substr(lastSlash + 1);
        int32_t           index   = -1;
        if(std::from_chars(segment.data(), segment.data() + segment.size(), index).ec != std::errc{})
          return InteractivityRef{};
        // The owning-array name (spec's "/category/index" core-array reference shape) - e.g.
        // "nodes" for "/nodes/17" - so two literals with the same index but different owning
        // arrays don't collapse onto the same InteractivityRef (see the struct's doc comment).
        const size_t      prevSlash = (lastSlash == 0) ? std::string::npos : s.rfind('/', lastSlash - 1);
        const std::string category =
            (prevSlash == std::string::npos) ? s.substr(0, lastSlash) : s.substr(prevSlash + 1, lastSlash - prevSlash - 1);
        return InteractivityRef{index, category};
      }
      case InteractivityValueType::eFloat2:
        return glm::vec2(static_cast<float>(num(0)), static_cast<float>(num(1)));
      case InteractivityValueType::eFloat3:
        return glm::vec3(static_cast<float>(num(0)), static_cast<float>(num(1)), static_cast<float>(num(2)));
      case InteractivityValueType::eFloat4:
        return glm::vec4(static_cast<float>(num(0)), static_cast<float>(num(1)), static_cast<float>(num(2)),
                         static_cast<float>(num(3)));
      case InteractivityValueType::eFloat2x2:
        // glTF matrices are column-major; the 4/9/16-arg glm constructors fill column-major too.
        return glm::mat2(static_cast<float>(num(0)), static_cast<float>(num(1)), static_cast<float>(num(2)),
                         static_cast<float>(num(3)));
      case InteractivityValueType::eFloat3x3: {
        float c[9];
        for(int i = 0; i < 9; ++i)
          c[i] = static_cast<float>(num(i));
        return glm::mat3(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]);
      }
      case InteractivityValueType::eFloat4x4: {
        float c[16];
        for(int i = 0; i < 16; ++i)
          c[i] = static_cast<float>(num(i));
        return glm::mat4(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11], c[12], c[13], c[14], c[15]);
      }
      default:
        return std::monostate{};
    }
  }();

  return ok ? result : InteractivityValue(std::monostate{});
}

//--------------------------------------------------------------------------------------------------
// Op catalog string table
//--------------------------------------------------------------------------------------------------

namespace {
struct OpEntry
{
  InteractivityOp op;
  const char*     name;
};

// clang-format off
constexpr OpEntry kOpTable[] = {
    {InteractivityOp::eMathE, "math/E"}, {InteractivityOp::eMathPi, "math/Pi"}, {InteractivityOp::eMathTau, "math/Tau"},
    {InteractivityOp::eMathInf, "math/Inf"}, {InteractivityOp::eMathNaN, "math/NaN"},

    {InteractivityOp::eMathAbs, "math/abs"}, {InteractivityOp::eMathSign, "math/sign"}, {InteractivityOp::eMathTrunc, "math/trunc"},
    {InteractivityOp::eMathFloor, "math/floor"}, {InteractivityOp::eMathCeil, "math/ceil"}, {InteractivityOp::eMathRound, "math/round"},
    {InteractivityOp::eMathFract, "math/fract"}, {InteractivityOp::eMathNeg, "math/neg"}, {InteractivityOp::eMathAdd, "math/add"},
    {InteractivityOp::eMathSub, "math/sub"}, {InteractivityOp::eMathMul, "math/mul"}, {InteractivityOp::eMathDiv, "math/div"},
    {InteractivityOp::eMathRem, "math/rem"}, {InteractivityOp::eMathMin, "math/min"}, {InteractivityOp::eMathMax, "math/max"},
    {InteractivityOp::eMathClamp, "math/clamp"}, {InteractivityOp::eMathSaturate, "math/saturate"}, {InteractivityOp::eMathMix, "math/mix"},
    {InteractivityOp::eMathSmoothStep, "math/smoothStep"},

    {InteractivityOp::eMathEq, "math/eq"}, {InteractivityOp::eMathLt, "math/lt"}, {InteractivityOp::eMathLe, "math/le"},
    {InteractivityOp::eMathGt, "math/gt"}, {InteractivityOp::eMathGe, "math/ge"}, {InteractivityOp::eMathIsNaN, "math/isNaN"},
    {InteractivityOp::eMathIsInf, "math/isInf"}, {InteractivityOp::eMathSelect, "math/select"}, {InteractivityOp::eMathSwitch, "math/switch"},
    {InteractivityOp::eMathRandom, "math/random"},

    {InteractivityOp::eMathRad, "math/rad"}, {InteractivityOp::eMathDeg, "math/deg"}, {InteractivityOp::eMathSin, "math/sin"},
    {InteractivityOp::eMathCos, "math/cos"}, {InteractivityOp::eMathTan, "math/tan"}, {InteractivityOp::eMathAsin, "math/asin"},
    {InteractivityOp::eMathAcos, "math/acos"}, {InteractivityOp::eMathAtan, "math/atan"}, {InteractivityOp::eMathAtan2, "math/atan2"},
    {InteractivityOp::eMathSinh, "math/sinh"}, {InteractivityOp::eMathCosh, "math/cosh"}, {InteractivityOp::eMathTanh, "math/tanh"},
    {InteractivityOp::eMathAsinh, "math/asinh"}, {InteractivityOp::eMathAcosh, "math/acosh"}, {InteractivityOp::eMathAtanh, "math/atanh"},
    {InteractivityOp::eMathExp, "math/exp"}, {InteractivityOp::eMathLog, "math/log"}, {InteractivityOp::eMathLog2, "math/log2"},
    {InteractivityOp::eMathLog10, "math/log10"}, {InteractivityOp::eMathSqrt, "math/sqrt"}, {InteractivityOp::eMathCbrt, "math/cbrt"},
    {InteractivityOp::eMathPow, "math/pow"},

    {InteractivityOp::eMathLength, "math/length"}, {InteractivityOp::eMathNormalize, "math/normalize"},
    {InteractivityOp::eMathDot, "math/dot"}, {InteractivityOp::eMathCross, "math/cross"}, {InteractivityOp::eMathRotate2D, "math/rotate2D"},
    {InteractivityOp::eMathRotate3D, "math/rotate3D"}, {InteractivityOp::eMathTransform, "math/transform"},
    {InteractivityOp::eMathSlerp, "math/slerp"}, {InteractivityOp::eMathTranspose, "math/transpose"},
    {InteractivityOp::eMathInverse, "math/inverse"}, {InteractivityOp::eMathDeterminant, "math/determinant"},
    {InteractivityOp::eMathMatMul, "math/matMul"}, {InteractivityOp::eMathMatCompose, "math/matCompose"},
    {InteractivityOp::eMathMatDecompose, "math/matDecompose"}, {InteractivityOp::eMathQuatConjugate, "math/quatConjugate"},
    {InteractivityOp::eMathQuatMul, "math/quatMul"}, {InteractivityOp::eMathQuatAngleBetween, "math/quatAngleBetween"},
    {InteractivityOp::eMathQuatFromAxisAngle, "math/quatFromAxisAngle"}, {InteractivityOp::eMathQuatToAxisAngle, "math/quatToAxisAngle"},
    {InteractivityOp::eMathQuatFromDirections, "math/quatFromDirections"}, {InteractivityOp::eMathQuatFromUpForward, "math/quatFromUpForward"},
    {InteractivityOp::eMathQuatFromAngles, "math/quatFromAngles"}, {InteractivityOp::eMathQuatSlerp, "math/quatSlerp"},

    {InteractivityOp::eMathCombine2, "math/combine2"}, {InteractivityOp::eMathCombine3, "math/combine3"},
    {InteractivityOp::eMathCombine4, "math/combine4"}, {InteractivityOp::eMathCombine2x2, "math/combine2x2"},
    {InteractivityOp::eMathCombine3x3, "math/combine3x3"}, {InteractivityOp::eMathCombine4x4, "math/combine4x4"},
    {InteractivityOp::eMathExtract2, "math/extract2"}, {InteractivityOp::eMathExtract3, "math/extract3"},
    {InteractivityOp::eMathExtract4, "math/extract4"}, {InteractivityOp::eMathExtract2x2, "math/extract2x2"},
    {InteractivityOp::eMathExtract3x3, "math/extract3x3"}, {InteractivityOp::eMathExtract4x4, "math/extract4x4"},

    {InteractivityOp::eMathNot, "math/not"}, {InteractivityOp::eMathAnd, "math/and"}, {InteractivityOp::eMathOr, "math/or"},
    {InteractivityOp::eMathXor, "math/xor"}, {InteractivityOp::eMathAsr, "math/asr"}, {InteractivityOp::eMathLsl, "math/lsl"},
    {InteractivityOp::eMathClz, "math/clz"}, {InteractivityOp::eMathCtz, "math/ctz"}, {InteractivityOp::eMathPopcnt, "math/popcnt"},

    {InteractivityOp::eMathRgbToOkLCh, "math/rgbToOkLCh"}, {InteractivityOp::eMathRgbFromOkLCh, "math/rgbFromOkLCh"},

    {InteractivityOp::eRefEq, "ref/eq"},

    {InteractivityOp::eTypeBoolToInt, "type/boolToInt"}, {InteractivityOp::eTypeBoolToFloat, "type/boolToFloat"},
    {InteractivityOp::eTypeIntToBool, "type/intToBool"}, {InteractivityOp::eTypeIntToFloat, "type/intToFloat"},
    {InteractivityOp::eTypeFloatToBool, "type/floatToBool"}, {InteractivityOp::eTypeFloatToInt, "type/floatToInt"},

    {InteractivityOp::eFlowSequence, "flow/sequence"}, {InteractivityOp::eFlowBranch, "flow/branch"},
    {InteractivityOp::eFlowSwitch, "flow/switch"}, {InteractivityOp::eFlowWhile, "flow/while"}, {InteractivityOp::eFlowFor, "flow/for"},
    {InteractivityOp::eFlowDoN, "flow/doN"}, {InteractivityOp::eFlowMultiGate, "flow/multiGate"},
    {InteractivityOp::eFlowWaitAll, "flow/waitAll"}, {InteractivityOp::eFlowThrottle, "flow/throttle"},
    {InteractivityOp::eFlowSetDelay, "flow/setDelay"}, {InteractivityOp::eFlowCancelDelay, "flow/cancelDelay"},

    {InteractivityOp::eVariableGet, "variable/get"}, {InteractivityOp::eVariableSet, "variable/set"},
    {InteractivityOp::eVariableInterpolate, "variable/interpolate"},

    {InteractivityOp::ePointerGet, "pointer/get"}, {InteractivityOp::ePointerSet, "pointer/set"},
    {InteractivityOp::ePointerInterpolate, "pointer/interpolate"},

    {InteractivityOp::eAnimationStart, "animation/start"}, {InteractivityOp::eAnimationStop, "animation/stop"},
    {InteractivityOp::eAnimationStopAt, "animation/stopAt"},

    {InteractivityOp::eEventOnStart, "event/onStart"}, {InteractivityOp::eEventOnTick, "event/onTick"},
    {InteractivityOp::eEventStopPropagation, "event/stopPropagation"}, {InteractivityOp::eEventReceive, "event/receive"},
    {InteractivityOp::eEventSend, "event/send"},

    {InteractivityOp::eDebugLog, "debug/log"},
};
// clang-format on

const std::unordered_map<std::string, InteractivityOp>& opByName()
{
  static const std::unordered_map<std::string, InteractivityOp> table = [] {
    std::unordered_map<std::string, InteractivityOp> t;
    for(const OpEntry& e : kOpTable)
      t.emplace(e.name, e.op);
    return t;
  }();
  return table;
}
}  // namespace

InteractivityOp parseCoreOpString(const std::string& op)
{
  auto it = opByName().find(op);
  return it != opByName().end() ? it->second : InteractivityOp::eUnknown;
}

const std::string& interactivityOpString(InteractivityOp op)
{
  static const std::string unknown = "<unknown>";
  for(const OpEntry& e : kOpTable)
  {
    if(e.op == op)
    {
      static thread_local std::string cache;  // stable enough for logging call sites
      cache = e.name;
      return cache;
    }
  }
  return unknown;
}

//--------------------------------------------------------------------------------------------------
// Graph parsing
//--------------------------------------------------------------------------------------------------

namespace {

// Parses the `value` property (spec "Inline Values") for a socket/variable/event-value whose type
// is already known, resolving through the graph's `types[]` table.
InteractivityValue parseLiteralWithType(const tinygltf::Value& obj, const std::vector<InteractivityTypeInfo>& types, int typeIndex)
{
  if(typeIndex < 0 || typeIndex >= static_cast<int>(types.size()) || !obj.Has("value"))
    return std::monostate{};
  return parseInteractivityLiteral(types[typeIndex].signature, obj.Get("value"));
}

bool parseDeclarations(const tinygltf::Value& graphJson, std::vector<InteractivityDeclaration>& out)
{
  if(!graphJson.Has("declarations"))
    return true;  // absent is valid (graph with no nodes)
  const tinygltf::Value& arr = graphJson.Get("declarations");
  if(!arr.IsArray())
    return false;

  out.reserve(arr.ArrayLen());
  for(size_t i = 0; i < arr.ArrayLen(); ++i)
  {
    const tinygltf::Value& d = arr.Get(i);
    if(!d.IsObject() || !d.Has("op") || !d.Get("op").IsString())
      return false;  // spec 5596: structurally invalid declaration -> reject whole graph

    InteractivityDeclaration decl;
    decl.opString = d.Get("op").Get<std::string>();
    if(d.Has("extension") && d.Get("extension").IsString())
    {
      decl.extension = d.Get("extension").Get<std::string>();
      decl.op        = InteractivityOp::eExtensionDefined;  // recognized per-(op,extension) in later phases
    }
    else
    {
      decl.op = parseCoreOpString(decl.opString);
      // A core op declared with extension-only fields (inputValueSockets/outputValueSockets) is
      // structurally invalid per the schema's dependentRequired rule.
      if(d.Has("inputValueSockets") || d.Has("outputValueSockets"))
        return false;
    }
    out.push_back(std::move(decl));
  }
  return true;
}

bool parseTypes(const tinygltf::Value& graphJson, std::vector<InteractivityTypeInfo>& out)
{
  if(!graphJson.Has("types"))
    return true;
  const tinygltf::Value& arr = graphJson.Get("types");
  if(!arr.IsArray())
    return false;
  out.reserve(arr.ArrayLen());
  for(size_t i = 0; i < arr.ArrayLen(); ++i)
  {
    const tinygltf::Value& t = arr.Get(i);
    InteractivityTypeInfo  info;
    if(t.IsObject() && t.Has("signature") && t.Get("signature").IsString())
      info.signature = parseTypeSignature(t.Get("signature").Get<std::string>());
    out.push_back(info);
  }
  return true;
}

bool parseVariables(const tinygltf::Value&                    graphJson,
                    const std::vector<InteractivityTypeInfo>& types,
                    std::vector<InteractivityVariableInfo>&   out)
{
  if(!graphJson.Has("variables"))
    return true;
  const tinygltf::Value& arr = graphJson.Get("variables");
  if(!arr.IsArray())
    return false;
  out.reserve(arr.ArrayLen());
  for(size_t i = 0; i < arr.ArrayLen(); ++i)
  {
    const tinygltf::Value&    v = arr.Get(i);
    InteractivityVariableInfo info;
    if(!v.IsObject() || !v.Has("type") || !v.Get("type").IsInt())
      return false;
    info.typeIndex = v.Get("type").GetNumberAsInt();
    if(info.typeIndex < 0 || info.typeIndex >= static_cast<int>(types.size()))
      return false;
    info.initialValue = v.Has("value") ? parseLiteralWithType(v, types, info.typeIndex) :
                                         defaultInteractivityValue(types[info.typeIndex].signature);
    out.push_back(std::move(info));
  }
  return true;
}

bool parseEvents(const tinygltf::Value& graphJson, const std::vector<InteractivityTypeInfo>& types, std::vector<InteractivityEventInfo>& out)
{
  if(!graphJson.Has("events"))
    return true;
  const tinygltf::Value& arr = graphJson.Get("events");
  if(!arr.IsArray())
    return false;
  out.reserve(arr.ArrayLen());
  for(size_t i = 0; i < arr.ArrayLen(); ++i)
  {
    const tinygltf::Value& e = arr.Get(i);
    InteractivityEventInfo info;
    if(e.IsObject() && e.Has("id") && e.Get("id").IsString())
      info.id = e.Get("id").Get<std::string>();
    if(e.IsObject() && e.Has("values") && e.Get("values").IsObject())
    {
      const tinygltf::Value& values = e.Get("values");
      for(const std::string& key : values.Keys())
      {
        const tinygltf::Value&      vv = values.Get(key);
        InteractivityEventValueInfo vInfo;
        if(!vv.IsObject() || !vv.Has("type") || !vv.Get("type").IsInt())
          return false;
        vInfo.typeIndex = vv.Get("type").GetNumberAsInt();
        if(vInfo.typeIndex < 0 || vInfo.typeIndex >= static_cast<int>(types.size()))
          return false;
        vInfo.initialValue = vv.Has("value") ? parseLiteralWithType(vv, types, vInfo.typeIndex) :
                                               defaultInteractivityValue(types[vInfo.typeIndex].signature);
        info.values.emplace(key, std::move(vInfo));
      }
    }
    out.push_back(std::move(info));
  }
  return true;
}

// Parses one node's `values` object (input value sockets). `nodeIndex` is this node's own index,
// used to enforce the spec's backward-reference-only rule for `node` references.
bool parseNodeValues(const tinygltf::Value&                                     valuesJson,
                     const std::vector<InteractivityTypeInfo>&                  types,
                     int                                                        nodeIndex,
                     std::unordered_map<std::string, InteractivityValueSocket>& out)
{
  for(const std::string& key : valuesJson.Keys())
  {
    const tinygltf::Value&   v = valuesJson.Get(key);
    InteractivityValueSocket socket;
    if(v.Has("node") && v.Get("node").IsInt())
    {
      if(v.Has("value"))
        return false;  // spec: node + value together -> invalid node
      const int refNode = v.Get("node").GetNumberAsInt();
      if(refNode < 0 || refNode >= nodeIndex)
        return false;  // spec 5273: value refs must point strictly backward
      socket.isReference = true;
      socket.sourceNode  = refNode;
      socket.sourceSocket = (v.Has("socket") && v.Get("socket").IsString()) ? v.Get("socket").Get<std::string>() : "value";
    }
    else if(v.Has("value") && v.Has("type") && v.Get("type").IsInt())
    {
      const int typeIndex = v.Get("type").GetNumberAsInt();
      if(typeIndex < 0 || typeIndex >= static_cast<int>(types.size()))
        return false;
      socket.isReference = false;
      socket.literalType = types[typeIndex].signature;
      socket.literal     = parseInteractivityLiteral(socket.literalType, v.Get("value"));
    }
    else
    {
      return false;  // neither a valid reference nor a valid literal
    }
    out.emplace(key, std::move(socket));
  }
  return true;
}

bool parseNodeFlows(const tinygltf::Value& flowsJson, std::unordered_map<std::string, InteractivityFlowSocket>& out)
{
  for(const std::string& key : flowsJson.Keys())
  {
    const tinygltf::Value& f = flowsJson.Get(key);
    if(!f.Has("node") || !f.Get("node").IsInt())
      return false;
    InteractivityFlowSocket socket;
    socket.targetNode   = f.Get("node").GetNumberAsInt();
    socket.targetSocket = (f.Has("socket") && f.Get("socket").IsString()) ? f.Get("socket").Get<std::string>() : "in";
    out.emplace(key, socket);
  }
  return true;
}

bool parseNodes(const tinygltf::Value&                       graphJson,
                const std::vector<InteractivityDeclaration>& declarations,
                const std::vector<InteractivityTypeInfo>&    types,
                std::vector<InteractivityNode>&              out)
{
  if(!graphJson.Has("nodes"))
    return true;
  const tinygltf::Value& arr = graphJson.Get("nodes");
  if(!arr.IsArray())
    return false;

  out.resize(arr.ArrayLen());
  for(size_t i = 0; i < arr.ArrayLen(); ++i)
  {
    const tinygltf::Value& n    = arr.Get(i);
    InteractivityNode&     node = out[i];
    if(!n.IsObject() || !n.Has("declaration") || !n.Get("declaration").IsInt())
      return false;
    node.declarationIndex = n.Get("declaration").GetNumberAsInt();
    if(node.declarationIndex < 0 || node.declarationIndex >= static_cast<int>(declarations.size()))
      return false;

    if(n.Has("configuration") && n.Get("configuration").IsObject())
    {
      const tinygltf::Value& cfg = n.Get("configuration");
      for(const std::string& key : cfg.Keys())
      {
        const tinygltf::Value& c = cfg.Get(key);
        if(c.IsObject() && c.Has("value"))
          node.configuration.emplace(key, c.Get("value"));
      }
    }
    if(n.Has("values") && n.Get("values").IsObject())
    {
      if(!parseNodeValues(n.Get("values"), types, static_cast<int>(i), node.values))
        return false;
    }
    if(n.Has("flows") && n.Get("flows").IsObject())
    {
      if(!parseNodeFlows(n.Get("flows"), node.flows))
        return false;
    }
  }
  return true;
}

}  // namespace

std::optional<InteractivityGraph> InteractivityGraph::parse(const tinygltf::Value& graphJson, std::string name, int graphIndex)
{
  if(!graphJson.IsObject())
    return std::nullopt;

  InteractivityGraph g;
  g.m_name       = std::move(name);
  g.m_graphIndex = graphIndex;

  if(!parseTypes(graphJson, g.m_types) || !parseVariables(graphJson, g.m_types, g.m_variables)
     || !parseEvents(graphJson, g.m_types, g.m_events) || !parseDeclarations(graphJson, g.m_declarations)
     || !parseNodes(graphJson, g.m_declarations, g.m_types, g.m_nodes))
  {
    LOGW("KHR_interactivity: graph '%s' is structurally invalid, skipping\n", g.m_name.c_str());
    return std::nullopt;
  }

  // Flow-socket target node indices must resolve to a real node (spec 5362 also requires them to
  // point strictly forward, "so that flow sockets do not form loops" - but every official Khronos
  // showcase scene in glTF-Test-Assets-Interactivity violates that half of the rule, apparently
  // exported by a visual editor that doesn't topologically sort nodes by flow direction. Rejecting
  // the whole graph over it would make this engine unable to run any real-world authored content,
  // so only an unresolvable target (negative or out of bounds) is treated as fatal here; a backward
  // (but in-range) reference is accepted and can form a real cycle -
  // InteractivityGraphInstance::activateFlow()'s call-depth cap is the actual safety net against a
  // graph whose flow chain never terminates, not this check.
  for(size_t i = 0; i < g.m_nodes.size(); ++i)
  {
    for(const auto& [socketName, flow] : g.m_nodes[i].flows)
    {
      if(flow.targetNode < 0 || flow.targetNode >= static_cast<int>(g.m_nodes.size()))
      {
        LOGW("KHR_interactivity: graph '%s' node %zu has a flow reference to nonexistent node %d, skipping graph\n",
             g.m_name.c_str(), i, flow.targetNode);
        return std::nullopt;
      }
    }
  }

  for(int i = 0; i < static_cast<int>(g.m_nodes.size()); ++i)
  {
    const InteractivityDeclaration& decl = g.m_declarations[g.m_nodes[i].declarationIndex];
    if(decl.op == InteractivityOp::eEventOnStart)
    {
      g.m_onStartNodes.push_back(i);
      continue;
    }
    if(decl.op == InteractivityOp::eEventOnTick)
    {
      g.m_onTickNodes.push_back(i);
      continue;
    }
    if(decl.op != InteractivityOp::eExtensionDefined)
      continue;

    // event/onHoverIn, event/onHoverOut (KHR_node_hoverability), event/onSelect
    // (KHR_node_selectability) - bound to a glTF node via THIS node's own configuration.nodeIndex,
    // not implicit/positional (see the accessor comments in the header for the validation rule).
    std::unordered_map<int, std::vector<int>>* handlerMap = nullptr;
    if(decl.extension == "KHR_node_hoverability" && decl.opString == "event/onHoverIn")
      handlerMap = &g.m_hoverInHandlers;
    else if(decl.extension == "KHR_node_hoverability" && decl.opString == "event/onHoverOut")
      handlerMap = &g.m_hoverOutHandlers;
    else if(decl.extension == "KHR_node_selectability" && decl.opString == "event/onSelect")
      handlerMap = &g.m_selectHandlers;
    if(!handlerMap)
      continue;

    auto cfgIt = g.m_nodes[i].configuration.find("nodeIndex");
    if(cfgIt == g.m_nodes[i].configuration.end() || !cfgIt->second.IsArray() || cfgIt->second.ArrayLen() == 0)
      continue;
    const tinygltf::Value& nodeIndexValue = cfgIt->second.Get(size_t{0});
    if(!nodeIndexValue.IsInt())
      continue;
    const int glTFNodeIndex = nodeIndexValue.GetNumberAsInt();
    if(glTFNodeIndex < 0)
      continue;  // Spec: negative nodeIndex -> default configuration -> never activates.

    (*handlerMap)[glTFNodeIndex].push_back(i);
  }

  return g;
}

std::vector<InteractivityGraph> parseInteractivityGraphs(const tinygltf::Model& model)
{
  std::vector<InteractivityGraph> result;

  auto it = model.extensions.find(KHR_INTERACTIVITY_EXTENSION_NAME);
  if(it == model.extensions.end() || !it->second.IsObject() || !it->second.Has("graphs"))
    return result;

  const tinygltf::Value& graphs = it->second.Get("graphs");
  if(!graphs.IsArray())
    return result;

  result.reserve(graphs.ArrayLen());
  for(size_t i = 0; i < graphs.ArrayLen(); ++i)
  {
    const tinygltf::Value& gj = graphs.Get(i);
    std::string name = (gj.IsObject() && gj.Has("name") && gj.Get("name").IsString()) ? gj.Get("name").Get<std::string>() :
                                                                                        ("Graph " + std::to_string(i));
    if(std::optional<InteractivityGraph> parsed = InteractivityGraph::parse(gj, std::move(name), static_cast<int>(i)))
      result.push_back(std::move(*parsed));
  }
  return result;
}

}  // namespace nvvkgltf
