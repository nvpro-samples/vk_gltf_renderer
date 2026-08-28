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
// KHR_interactivity graph-engine unit tests (Phase A - see docs/interactivity.md).
//
// These construct graphs directly as tinygltf::Value trees (no file I/O, no GPU) so they are
// fast and deterministic. Run with: ctest -R Interactivity
//

#include <cmath>

#include <gtest/gtest.h>

#include "gltf_interactivity_animation.hpp"
#include "gltf_interactivity_eval.hpp"
#include "gltf_interactivity_graph.hpp"
#include "gltf_interactivity_instance.hpp"

using namespace nvvkgltf;

namespace {

tinygltf::Value obj(std::initializer_list<tinygltf::Value::Object::value_type> items)
{
  return tinygltf::Value(tinygltf::Value::Object(items));
}
tinygltf::Value arr(std::initializer_list<tinygltf::Value> items)
{
  return tinygltf::Value(tinygltf::Value::Array(items));
}
tinygltf::Value str(const std::string& s)
{
  return tinygltf::Value(s);
}
// Used for declaration/type/node/variable *indices* - the parser requires these to be IsInt(),
// so this must produce an INT_TYPE tinygltf::Value, not REAL_TYPE. Literal socket/variable
// *payload* values (which tolerate either, per GetNumberAsDouble()/GetNumberAsInt()) go through
// literal() below, which builds its own array directly rather than routing through this helper.
tinygltf::Value num(int v)
{
  return tinygltf::Value(v);
}
tinygltf::Value literal(std::initializer_list<double> components, int typeIndex)
{
  tinygltf::Value::Array a;
  for(double v : components)
    a.push_back(tinygltf::Value(v));
  return obj({{"value", tinygltf::Value(a)}, {"type", tinygltf::Value(typeIndex)}});
}
tinygltf::Value ref(int node)
{
  return obj({{"node", tinygltf::Value(node)}});
}
tinygltf::Value ref(int node, const std::string& socket)
{
  return obj({{"node", tinygltf::Value(node)}, {"socket", str(socket)}});
}
tinygltf::Value flow(int node)
{
  return obj({{"node", tinygltf::Value(node)}});
}

}  // namespace

//--------------------------------------------------------------------------------------------------
// Graph parsing validity
//--------------------------------------------------------------------------------------------------

TEST(InteractivityGraph, RejectsDeclarationMissingOp)
{
  tinygltf::Value graphJson = obj({{"declarations", arr({obj({{"extension", str("Foo")}})})}});
  EXPECT_FALSE(InteractivityGraph::parse(graphJson, "bad", 0).has_value());
}

TEST(InteractivityGraph, RejectsForwardValueReference)
{
  // Node 0's input value socket references node 1, which comes *after* it - spec 5273 forbids this.
  tinygltf::Value graphJson =
      obj({{"types", arr({obj({{"signature", str("int")}})})},
           {"declarations", arr({obj({{"op", str("math/add")}}), obj({{"op", str("math/add")}})})},
           {"nodes", arr({obj({{"declaration", num(0)}, {"values", obj({{"a", ref(1)}, {"b", literal({1}, 0)}})}}),
                          obj({{"declaration", num(1)}, {"values", obj({{"a", literal({1}, 0)}, {"b", literal({2}, 0)}})}})})}});
  EXPECT_FALSE(InteractivityGraph::parse(graphJson, "bad", 0).has_value());
}

// Spec 5362 says a flow target index "MUST" point strictly forward, to guarantee the static graph
// has no cycles. Every official scene in Khronos's own glTF-Test-Assets-Interactivity corpus
// violates this (apparently exported by a visual editor that doesn't topologically sort by flow
// direction) - rejecting the whole graph over it would make this engine unable to run any real
// content, so only a genuinely unresolvable target is fatal now (see RejectsOutOfRangeFlowReference
// below); a backward, in-range one - even a literal self-reference, as here - is accepted. The
// runtime safety net for an actual cycle is InteractivityGraphInstance::activateFlow()'s recursion-
// depth cap, exercised by InteractivityInstance.RecursiveActivationCapsRuntimeCycle further down.
TEST(InteractivityGraph, AcceptsBackwardFlowReference)
{
  tinygltf::Value graphJson = obj({{"declarations", arr({obj({{"op", str("flow/sequence")}})})},
                                   {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"a", flow(0)}})}})})}});
  EXPECT_TRUE(InteractivityGraph::parse(graphJson, "selfRef", 0).has_value());
}

TEST(InteractivityGraph, RejectsOutOfRangeFlowReference)
{
  // Only one node (index 0) exists; a flow target of 5 can never resolve to a real node.
  tinygltf::Value graphJson = obj({{"declarations", arr({obj({{"op", str("flow/sequence")}})})},
                                   {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"a", flow(5)}})}})})}});
  EXPECT_FALSE(InteractivityGraph::parse(graphJson, "bad", 0).has_value());
}

TEST(InteractivityGraph, AcceptsExtensionDefinedDeclarationAsGracefulNoOp)
{
  // spec 339-345: a well-formed but unrecognized `extension` op degrades to a no-op, it must not
  // reject the whole graph (e.g. event/onSelect from the companion KHR_node_selectability extension).
  tinygltf::Value graphJson =
      obj({{"declarations", arr({obj({{"op", str("event/onSelect")}, {"extension", str("KHR_node_selectability")}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "ext", 0);
  ASSERT_TRUE(graph.has_value());
  ASSERT_EQ(graph->declarations().size(), 1u);
  EXPECT_EQ(graph->declarations()[0].op, InteractivityOp::eExtensionDefined);
  EXPECT_EQ(graph->declarations()[0].extension, "KHR_node_selectability");
}

//--------------------------------------------------------------------------------------------------
// End-to-end: event/onStart -> math/add -> variable/set, mirrors the spec's own worked examples.
//--------------------------------------------------------------------------------------------------

TEST(InteractivityInstance, OnStartComputesAddAndSetsVariable)
{
  // types[0] = int; variables[0] = int, initial 0
  // declarations: 0=event/onStart, 1=math/add, 2=variable/set
  // node 0 (onStart) --out--> node 2 (variable/set), node 2's socket "0" <- node 1's output
  tinygltf::Value graphJson =
      obj({{"types", arr({obj({{"signature", str("int")}})})},
           {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
           {"declarations",
            arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("math/add")}}), obj({{"op", str("variable/set")}})})},
           {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(2)}})}}),
                          obj({{"declaration", num(1)}, {"values", obj({{"a", literal({1}, 0)}, {"b", literal({2}, 0)}})}}),
                          obj({{"declaration", num(2)},
                               {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                               {"values", obj({{"0", ref(1)}})}})})}});

  auto graph = InteractivityGraph::parse(graphJson, "add", 0);
  ASSERT_TRUE(graph.has_value());
  ASSERT_EQ(graph->onStartNodes().size(), 1u);

  InteractivityGraphInstance instance(*graph);
  EXPECT_TRUE(std::holds_alternative<int32_t>(instance.variable(0)));
  EXPECT_EQ(std::get<int32_t>(instance.variable(0)), 0);

  instance.start();

  ASSERT_TRUE(std::holds_alternative<int32_t>(instance.variable(0)));
  EXPECT_EQ(std::get<int32_t>(instance.variable(0)), 3);
}

TEST(InteractivityInstance, ResetReplaysFromInitialValue)
{
  tinygltf::Value graphJson =
      obj({{"types", arr({obj({{"signature", str("int")}})})},
           {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(5)})}})})},
           {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("variable/set")}})})},
           {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                          obj({{"declaration", num(1)},
                               {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                               {"values", obj({{"0", literal({99}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "reset", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);

  instance.start();
  EXPECT_EQ(std::get<int32_t>(instance.variable(0)), 99);

  instance.reset();
  EXPECT_EQ(std::get<int32_t>(instance.variable(0)), 5) << "reset() must restore the variable's declared initial value";
  EXPECT_FALSE(instance.started());
}

//--------------------------------------------------------------------------------------------------
// flow/branch
//--------------------------------------------------------------------------------------------------

TEST(InteractivityInstance, BranchTakesTrueOrFalsePath)
{
  auto buildAndRun = [](bool condition) {
    tinygltf::Value graphJson =
        obj({{"types", arr({obj({{"signature", str("bool")}}), obj({{"signature", str("int")}})})},
             {"variables", arr({obj({{"type", num(1)}, {"value", arr({num(0)})}})})},
             {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/branch")}}),
                                   obj({{"op", str("variable/set")}}), obj({{"op", str("variable/set")}})})},
             {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                            obj({{"declaration", num(1)},
                                 {"values", obj({{"condition", literal({condition ? 1.0 : 0.0}, 0)}})},
                                 {"flows", obj({{"true", flow(2)}, {"false", flow(3)}})}}),
                            obj({{"declaration", num(2)},
                                 {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                                 {"values", obj({{"0", literal({1}, 1)}})}}),
                            obj({{"declaration", num(3)},
                                 {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                                 {"values", obj({{"0", literal({2}, 1)}})}})})}});
    auto graph = InteractivityGraph::parse(graphJson, "branch", 0);
    EXPECT_TRUE(graph.has_value());
    if(!graph.has_value())
      return -1;
    InteractivityGraphInstance instance(*graph);
    instance.start();
    return std::get<int32_t>(instance.variable(0));
  };

  EXPECT_EQ(buildAndRun(true), 1);
  EXPECT_EQ(buildAndRun(false), 2);
}

//--------------------------------------------------------------------------------------------------
// flow/sequence: activation order follows ascending lexicographic socket id (spec "Socket Order").
//--------------------------------------------------------------------------------------------------

TEST(InteractivityInstance, SequenceActivatesInLexicographicOrder)
{
  // Three variable/set nodes wired to sequence outputs "c", "a", "b" (declared out of order).
  // Each overwrites the same variable; the *last* writer in execution order (lexicographically "c")
  // determines the final value, so this directly observes activation order.
  auto makeSetNode = [](int declIndex, double value) {
    return obj({{"declaration", num(declIndex)},
                {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                {"values", obj({{"0", literal({value}, 0)}})}});
  };

  tinygltf::Value graphJson =
      obj({{"types", arr({obj({{"signature", str("int")}})})},
           {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
           {"declarations",
            arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/sequence")}}), obj({{"op", str("variable/set")}}),
                 obj({{"op", str("variable/set")}}), obj({{"op", str("variable/set")}})})},
           {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                          obj({{"declaration", num(1)}, {"flows", obj({{"c", flow(2)}, {"a", flow(3)}, {"b", flow(4)}})}}),
                          makeSetNode(2, 10.0), makeSetNode(3, 20.0), makeSetNode(4, 30.0)})}});

  auto graph = InteractivityGraph::parse(graphJson, "seq", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();

  // Execution order is a(20) -> b(30) -> c(10); "c" sorts last, so it wins.
  EXPECT_EQ(std::get<int32_t>(instance.variable(0)), 10);
}

//--------------------------------------------------------------------------------------------------
// event/onTick timing (spec: first tick's timeSinceStart == 0, timeSinceLastTick == NaN)
//--------------------------------------------------------------------------------------------------

TEST(InteractivityInstance, FirstTickTimingMatchesSpec)
{
  tinygltf::Value graphJson = obj({{"declarations", arr({obj({{"op", str("event/onTick")}})})}, {"nodes", arr({})}});
  auto            graph     = InteractivityGraph::parse(graphJson, "tick", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);

  instance.tick(0.1f);
  EXPECT_FLOAT_EQ(instance.timeSinceStart(), 0.0f);
  EXPECT_TRUE(std::isnan(instance.timeSinceLastTick()));

  instance.tick(0.1f);
  EXPECT_FLOAT_EQ(instance.timeSinceStart(), 0.1f);
  EXPECT_FLOAT_EQ(instance.timeSinceLastTick(), 0.1f);
}

//--------------------------------------------------------------------------------------------------
// Pure node evaluation (direct evaluateNodeOutput, no flow needed - math/type ops are pull-based).
//--------------------------------------------------------------------------------------------------

TEST(InteractivityEval, TypeConversionsMatchSpec)
{
  auto singleNodeGraph = [](const std::string& op, std::initializer_list<double> aValue, int typeIndex, const std::string& typeName) {
    tinygltf::Value graphJson =
        obj({{"types", arr({obj({{"signature", str(typeName)}})})},
             {"declarations", arr({obj({{"op", str(op)}})})},
             {"nodes", arr({obj({{"declaration", num(0)}, {"values", obj({{"a", literal(aValue, typeIndex)}})}})})}});
    return InteractivityGraph::parse(graphJson, op, 0);
  };

  {
    auto graph = singleNodeGraph("type/floatToInt", {3.9}, 0, "float");
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_EQ(std::get<int32_t>(evaluateNodeOutput(instance, 0, "value")), 3);
  }
  {
    auto graph = singleNodeGraph("type/floatToInt", {-3.9}, 0, "float");
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_EQ(std::get<int32_t>(evaluateNodeOutput(instance, 0, "value")), -3) << "truncates towards zero, not floor";
  }
  {
    auto graph = singleNodeGraph("type/intToFloat", {42}, 0, "int");
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_FLOAT_EQ(std::get<float>(evaluateNodeOutput(instance, 0, "value")), 42.0f);
  }
  {
    auto graph = singleNodeGraph("type/boolToInt", {1}, 0, "bool");
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_EQ(std::get<int32_t>(evaluateNodeOutput(instance, 0, "value")), 1);
  }
}

TEST(InteractivityEval, ComparisonOpsReturnBool)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("float")}})})},
       {"declarations", arr({obj({{"op", str("math/lt")}})})},
       {"nodes", arr({obj({{"declaration", num(0)}, {"values", obj({{"a", literal({1.0}, 0)}, {"b", literal({2.0}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "lt", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  InteractivityValue         result = evaluateNodeOutput(instance, 0, "value");
  ASSERT_TRUE(std::holds_alternative<bool>(result));
  EXPECT_TRUE(std::get<bool>(result));
}

TEST(InteractivityEval, ClampRespectsBounds)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("float")}})})},
       {"declarations", arr({obj({{"op", str("math/clamp")}})})},
       {"nodes", arr({obj({{"declaration", num(0)},
                           {"values", obj({{"a", literal({15.0}, 0)}, {"b", literal({0.0}, 0)}, {"c", literal({10.0}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "clamp", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_FLOAT_EQ(std::get<float>(evaluateNodeOutput(instance, 0, "value")), 10.0f);
}

// Spec formula min(max(a,min(b,c)), max(b,c)) is symmetric in b/c - swapped bounds (b > c) must
// still produce the same result as the normal order, per the spec's own "Authoring Note".
TEST(InteractivityEval, ClampToleratesSwappedBounds)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("float")}})})},
       {"declarations", arr({obj({{"op", str("math/clamp")}})})},
       {"nodes", arr({obj({{"declaration", num(0)},
                           {"values", obj({{"a", literal({15.0}, 0)}, {"b", literal({10.0}, 0)}, {"c", literal({0.0}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "clampSwapped", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_FLOAT_EQ(std::get<float>(evaluateNodeOutput(instance, 0, "value")), 10.0f);
}

// Spec socket ids for math/mix are a, b, c (not "t") - c is the unclamped interpolation coefficient.
TEST(InteractivityEval, MixUsesSpecSocketNames)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("float")}})})},
       {"declarations", arr({obj({{"op", str("math/mix")}})})},
       {"nodes", arr({obj({{"declaration", num(0)},
                           {"values", obj({{"a", literal({0.0}, 0)}, {"b", literal({10.0}, 0)}, {"c", literal({0.25}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "mix", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_FLOAT_EQ(std::get<float>(evaluateNodeOutput(instance, 0, "value")), 2.5f);
}

// Spec: math/div and math/rem are int-only (no float/vecN overload, unlike add/sub/mul/min/max),
// with piecewise b==0 -> 0 and INT_MIN/-1 -> wraps rather than trapping. Worth a dedicated test
// because plain int32_t `/`/`%` for either case is a hardware trap (SIGFPE via the x86 `idiv`
// instruction), found by running real Khronos content that hit math/rem (then unimplemented) and
// would have hit this same trap had a scene divided by a runtime-zero value.
TEST(InteractivityEval, DivHandlesZeroAndIntMinOverflow)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}})})},
       {"declarations", arr({obj({{"op", str("math/div")}})})},
       {"nodes",
        arr({obj({{"declaration", num(0)}, {"values", obj({{"a", literal({10}, 0)}, {"b", literal({3}, 0)}})}}),
             obj({{"declaration", num(0)}, {"values", obj({{"a", literal({-7}, 0)}, {"b", literal({2}, 0)}})}}),
             obj({{"declaration", num(0)}, {"values", obj({{"a", literal({5}, 0)}, {"b", literal({0}, 0)}})}}),
             obj({{"declaration", num(0)}, {"values", obj({{"a", literal({-2147483648.0}, 0)}, {"b", literal({-1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "div", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_EQ(std::get<int32_t>(evaluateNodeOutput(instance, 0, "value")), 3);
  EXPECT_EQ(std::get<int32_t>(evaluateNodeOutput(instance, 1, "value")), -3) << "truncates towards zero, not floor";
  EXPECT_EQ(std::get<int32_t>(evaluateNodeOutput(instance, 2, "value")), 0) << "b==0 must return 0, not trap";
  EXPECT_EQ(std::get<int32_t>(evaluateNodeOutput(instance, 3, "value")), std::numeric_limits<int32_t>::min())
      << "INT_MIN / -1 must wrap, not trap";
}

TEST(InteractivityEval, RemHandlesZeroAndIntMinOverflow)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}})})},
       {"declarations", arr({obj({{"op", str("math/rem")}})})},
       {"nodes",
        arr({obj({{"declaration", num(0)}, {"values", obj({{"a", literal({7}, 0)}, {"b", literal({3}, 0)}})}}),
             obj({{"declaration", num(0)}, {"values", obj({{"a", literal({-7}, 0)}, {"b", literal({3}, 0)}})}}),
             obj({{"declaration", num(0)}, {"values", obj({{"a", literal({5}, 0)}, {"b", literal({0}, 0)}})}}),
             obj({{"declaration", num(0)}, {"values", obj({{"a", literal({-2147483648.0}, 0)}, {"b", literal({-1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "rem", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_EQ(std::get<int32_t>(evaluateNodeOutput(instance, 0, "value")), 1);
  EXPECT_EQ(std::get<int32_t>(evaluateNodeOutput(instance, 1, "value")), -1) << "keeps dividend's sign (truncated division)";
  EXPECT_EQ(std::get<int32_t>(evaluateNodeOutput(instance, 2, "value")), 0) << "b==0 must return 0, not trap";
  EXPECT_EQ(std::get<int32_t>(evaluateNodeOutput(instance, 3, "value")), 0) << "INT_MIN % -1 must wrap to 0, not trap";
}

//--------------------------------------------------------------------------------------------------
// Found by running Khronos's own conformance suite
// (Tests/Interactivity/mathtests.glb) at the user's request: 35 debug/log "Test Failed" lines
// across 17 distinct math/* ops. Root causes, all fixed together since most ops shared one:
// (1) applyUnary/applyBinary/applyComparison excluded matrix types entirely (returning monostate),
//     even though the spec defines most math/* ops as "floatN or floatNxN" (matrix-inclusive) -
//     affected neg/ceil/floor/fract/round/saturate/sign/eq/mix/min/max/clamp/combine4x4 (the
//     latter two only indirectly, via eq/pointer/set comparisons downstream of them);
// (2) math/div and math/rem are ALSO defined for floatN/floatNxN (a second overload, distinct from
//     the int-specific one above) - the int-only fix from the earlier crash-guard pass had made
//     them reject float/vec/mat operands entirely;
// (3) math/eMathMix's "c" (interpolation coefficient) was hard-restricted to a bare scalar float,
//     but spec has it match a/b's type exactly (floatN c or floatNxN c);
// (4) math/smoothStep and math/trunc were simply never implemented (silently no-op'd);
// (5) glm::sign is comparison-based and silently returns 0 for NaN instead of propagating it;
// (6) strict JSON has no numeric token for Infinity/NaN, so literal float components needing them
//     are authored as quoted string tokens ("Infinity"/"-Infinity"/"NaN") - our literal parser
//     only ever called GetNumberAsDouble(), which silently returns 0.0 for a string value.
//--------------------------------------------------------------------------------------------------

TEST(InteractivityEval, DivRemGenericFloatOverloadMatchesSpecFormula)
{
  tinygltf::Value graphJson =
      obj({{"types", arr({obj({{"signature", str("float")}})})},
           {"declarations", arr({obj({{"op", str("math/div")}}), obj({{"op", str("math/rem")}})})},
           {"nodes",
            arr({obj({{"declaration", num(0)}, {"values", obj({{"a", literal({1.0}, 0)}, {"b", literal({0.0}, 0)}})}}),
                 obj({{"declaration", num(0)}, {"values", obj({{"a", literal({-1.0}, 0)}, {"b", literal({0.0}, 0)}})}}),
                 obj({{"declaration", num(0)}, {"values", obj({{"a", literal({0.0}, 0)}, {"b", literal({0.0}, 0)}})}}),
                 obj({{"declaration", num(1)}, {"values", obj({{"a", literal({19.42}, 0)}, {"b", literal({2.23}, 0)}})}}),
                 obj({{"declaration", num(1)}, {"values", obj({{"a", literal({5.0}, 0)}, {"b", literal({0.0}, 0)}})}}),
                 obj({{"declaration", num(1)}, {"values", obj({{"a", literal({-7.0}, 0)}, {"b", literal({3.0}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "divRemFloat", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  const float                div1 = std::get<float>(evaluateNodeOutput(instance, 0, "value"));
  const float                div2 = std::get<float>(evaluateNodeOutput(instance, 1, "value"));
  EXPECT_TRUE(std::isinf(div1) && div1 > 0.0f) << "1/0 = +Infinity";
  EXPECT_TRUE(std::isinf(div2) && div2 < 0.0f) << "-1/0 = -Infinity";
  EXPECT_TRUE(std::isnan(std::get<float>(evaluateNodeOutput(instance, 2, "value")))) << "0/0 = NaN";
  EXPECT_NEAR(std::get<float>(evaluateNodeOutput(instance, 3, "value")), 1.58f, 0.01f)
      << "19.42 rem 2.23 = 19.42 - 2.23*trunc(19.42/2.23)";
  EXPECT_TRUE(std::isnan(std::get<float>(evaluateNodeOutput(instance, 4, "value")))) << "5 rem 0 = NaN (unlike int rem's 0)";
  EXPECT_NEAR(std::get<float>(evaluateNodeOutput(instance, 5, "value")), -1.0f, 1e-5f) << "-7 rem 3 keeps dividend's sign";
}

TEST(InteractivityGraph, ParsesInfinityAndNaNStringLiteralTokens)
{
  tinygltf::Value graphJson =
      obj({{"types", arr({obj({{"signature", str("float3")}})})},
           {"declarations", arr({obj({{"op", str("math/length")}})})},
           {"nodes",
            arr({obj({{"declaration", num(0)},
                      {"values", obj({{"a", obj({{"value", arr({str("Infinity"), num(2), num(3)})}, {"type", num(0)}})}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "infLiteral", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
  ASSERT_TRUE(std::holds_alternative<float>(v));
  EXPECT_TRUE(std::isinf(std::get<float>(v)) && std::get<float>(v) > 0.0f);
}

//--------------------------------------------------------------------------------------------------
// Unimplemented-but-recognized ops (Phase B/C/D catalog) must no-op, not crash.
//--------------------------------------------------------------------------------------------------

TEST(InteractivityEval, UnimplementedOpNoOpsInsteadOfCrashing)
{
  // pointer/get is Phase C territory (data binding) - still recognized, not yet evaluated.
  tinygltf::Value graphJson =
      obj({{"declarations", arr({obj({{"op", str("pointer/get")}})})}, {"nodes", arr({obj({{"declaration", num(0)}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "ptrGet", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_NO_THROW({
    InteractivityValue result = evaluateNodeOutput(instance, 0, "value");
    EXPECT_TRUE(std::holds_alternative<std::monostate>(result));
  });
}

//--------------------------------------------------------------------------------------------------
// Phase B: full math/type catalog (trig/hyperbolic/exponential, vector/matrix/quaternion,
// swizzle, integer/boolean, color, constants/special ops). Helper builds a single-node graph so
// each op can be evaluated directly via evaluateNodeOutput, no flow wiring needed (all pure).
//--------------------------------------------------------------------------------------------------

namespace {
// `types` are declared in order (index 0, 1, 2, ...); `values` is the node's full values object
// (build with literal()/ref()). Node 0 is the only node, using declaration 0 (`op`).
std::optional<InteractivityGraph> singleNodeGraph(const std::string&                 op,
                                                  std::initializer_list<std::string> typeSignatures,
                                                  tinygltf::Value                    valuesObj)
{
  tinygltf::Value::Array types;
  for(const std::string& sig : typeSignatures)
    types.push_back(obj({{"signature", str(sig)}}));
  tinygltf::Value graphJson = obj({{"types", tinygltf::Value(types)},
                                   {"declarations", arr({obj({{"op", str(op)}})})},
                                   {"nodes", arr({obj({{"declaration", num(0)}, {"values", valuesObj}})})}});
  return InteractivityGraph::parse(graphJson, op, 0);
}

// Convenience for the common case: every socket shares type index 0 = "float".
std::optional<InteractivityGraph> floatOpGraph(const std::string& op, tinygltf::Value valuesObj)
{
  return singleNodeGraph(op, {"float"}, std::move(valuesObj));
}

float evalFloat(InteractivityGraphInstance& instance, const std::string& socket = "value")
{
  InteractivityValue v = evaluateNodeOutput(instance, 0, socket);
  return std::holds_alternative<float>(v) ? std::get<float>(v) : std::numeric_limits<float>::quiet_NaN();
}
}  // namespace

TEST(InteractivityEvalTrig, SinCosAtan2Pow)
{
  {
    auto graph = floatOpGraph("math/sin", obj({{"a", literal({0.0}, 0)}}));
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_NEAR(evalFloat(instance), 0.0f, 1e-6f);
  }
  {
    auto graph = floatOpGraph("math/cos", obj({{"a", literal({0.0}, 0)}}));
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_NEAR(evalFloat(instance), 1.0f, 1e-6f);
  }
  {
    // atan2 socket order is a=Y, b=X (spec 1228) - atan2(1,0) = pi/2.
    auto graph = floatOpGraph("math/atan2", obj({{"a", literal({1.0}, 0)}, {"b", literal({0.0}, 0)}}));
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_NEAR(evalFloat(instance), 1.5707964f, 1e-5f);
  }
  {
    auto graph = floatOpGraph("math/pow", obj({{"a", literal({2.0}, 0)}, {"b", literal({10.0}, 0)}}));
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_NEAR(evalFloat(instance), 1024.0f, 1e-3f);
  }
}

TEST(InteractivityEvalTrig, Log10AndCbrtHaveNoGlmBuiltinButWork)
{
  {
    auto graph = floatOpGraph("math/log10", obj({{"a", literal({1000.0}, 0)}}));
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_NEAR(evalFloat(instance), 3.0f, 1e-5f);
  }
  {
    auto graph = floatOpGraph("math/cbrt", obj({{"a", literal({-27.0}, 0)}}));
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_NEAR(evalFloat(instance), -3.0f, 1e-5f);
  }
}

TEST(InteractivityEvalVector, LengthAndNormalize)
{
  auto graph = singleNodeGraph("math/normalize", {"float3"}, obj({{"a", literal({3.0, 4.0, 0.0}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  InteractivityValue         isValid = evaluateNodeOutput(instance, 0, "isValid");
  ASSERT_TRUE(std::holds_alternative<bool>(isValid));
  EXPECT_TRUE(std::get<bool>(isValid));
  InteractivityValue value = evaluateNodeOutput(instance, 0, "value");
  ASSERT_TRUE(std::holds_alternative<glm::vec3>(value));
  const glm::vec3 v = std::get<glm::vec3>(value);
  EXPECT_NEAR(v.x, 0.6f, 1e-5f);
  EXPECT_NEAR(v.y, 0.8f, 1e-5f);
}

TEST(InteractivityEvalVector, NormalizeZeroLengthIsInvalid)
{
  auto graph = singleNodeGraph("math/normalize", {"float3"}, obj({{"a", literal({0.0, 0.0, 0.0}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  InteractivityValue         isValid = evaluateNodeOutput(instance, 0, "isValid");
  ASSERT_TRUE(std::holds_alternative<bool>(isValid));
  EXPECT_FALSE(std::get<bool>(isValid));
}

TEST(InteractivityEvalVector, DotAndCross)
{
  {
    auto graph = singleNodeGraph("math/dot", {"float3"}, obj({{"a", literal({1, 0, 0}, 0)}, {"b", literal({0, 1, 0}, 0)}}));
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_NEAR(evalFloat(instance), 0.0f, 1e-6f);
  }
  {
    auto graph = singleNodeGraph("math/cross", {"float3"}, obj({{"a", literal({1, 0, 0}, 0)}, {"b", literal({0, 1, 0}, 0)}}));
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
    ASSERT_TRUE(std::holds_alternative<glm::vec3>(v));
    EXPECT_NEAR(std::get<glm::vec3>(v).z, 1.0f, 1e-6f);
  }
}

TEST(InteractivityEvalMatrix, TransformMatchesMatMulSemantics)
{
  // types: 0=float3, 1=float3x3
  tinygltf::Value m = literal({2, 0, 0, 0, 3, 0, 0, 0, 4}, 1);  // diag(2,3,4), column-major
  auto graph = singleNodeGraph("math/transform", {"float3", "float3x3"}, obj({{"a", literal({1, 1, 1}, 0)}, {"b", m}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
  ASSERT_TRUE(std::holds_alternative<glm::vec3>(v));
  const glm::vec3 result = std::get<glm::vec3>(v);
  EXPECT_NEAR(result.x, 2.0f, 1e-5f);
  EXPECT_NEAR(result.y, 3.0f, 1e-5f);
  EXPECT_NEAR(result.z, 4.0f, 1e-5f);
}

TEST(InteractivityEvalMatrix, TransposeDeterminantInverse)
{
  // 2x2 matrix [[2,0],[0,4]] (column-major: col0=(2,0), col1=(0,4)) -> det=8, inverse=[[0.5,0],[0,0.25]]
  auto graph = singleNodeGraph("math/determinant", {"float2x2"}, obj({{"a", literal({2, 0, 0, 4}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_NEAR(evalFloat(instance), 8.0f, 1e-5f);

  auto graphInv = singleNodeGraph("math/inverse", {"float2x2"}, obj({{"a", literal({2, 0, 0, 4}, 0)}}));
  ASSERT_TRUE(graphInv.has_value());
  InteractivityGraphInstance instanceInv(*graphInv);
  InteractivityValue         valid = evaluateNodeOutput(instanceInv, 0, "isValid");
  ASSERT_TRUE(std::holds_alternative<bool>(valid));
  EXPECT_TRUE(std::get<bool>(valid));
  InteractivityValue inv = evaluateNodeOutput(instanceInv, 0, "value");
  ASSERT_TRUE(std::holds_alternative<glm::mat2>(inv));
  const glm::mat2 im = std::get<glm::mat2>(inv);
  EXPECT_NEAR(im[0][0], 0.5f, 1e-5f);
  EXPECT_NEAR(im[1][1], 0.25f, 1e-5f);
}

TEST(InteractivityEvalMatrix, SingularMatrixInverseIsInvalid)
{
  auto graph = singleNodeGraph("math/inverse", {"float2x2"}, obj({{"a", literal({0, 0, 0, 0}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  InteractivityValue         valid = evaluateNodeOutput(instance, 0, "isValid");
  ASSERT_TRUE(std::holds_alternative<bool>(valid));
  EXPECT_FALSE(std::get<bool>(valid));
}

// Found by running Khronos's own conformance suite (Tests/Interactivity/
// mathtests.glb) at the user's request: applyUnary/applyBinary/applyComparison previously excluded
// matrix types entirely (returning monostate), even though the spec defines most math/* ops as
// "floatN or floatNxN" (matrix-inclusive) - affected neg/ceil/floor/fract/round/saturate/sign/eq/
// mix/min/max/clamp/combine4x4 (the last two only indirectly, downstream of eq comparisons).
TEST(InteractivityEvalMatrix, NegAndEqSupportMatrices)
{
  auto graph = singleNodeGraph("math/neg", {"float4x4"},
                               obj({{"a", literal({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
  ASSERT_TRUE(std::holds_alternative<glm::mat4>(v));
  const glm::mat4 m = std::get<glm::mat4>(v);
  EXPECT_FLOAT_EQ(m[0][0], -1.0f);
  EXPECT_FLOAT_EQ(m[3][3], -16.0f);

  auto graphEq = singleNodeGraph("math/eq", {"float4x4"},
                                 obj({{"a", literal({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, 0)},
                                      {"b", literal({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, 0)}}));
  ASSERT_TRUE(graphEq.has_value());
  InteractivityGraphInstance instanceEq(*graphEq);
  EXPECT_TRUE(std::get<bool>(evaluateNodeOutput(instanceEq, 0, "value")));
}

// math/eMathMix's "c" (interpolation coefficient) was hard-restricted to a bare scalar float, but
// spec has it match a/b's type exactly (floatN c or floatNxN c) - found the same way.
TEST(InteractivityEval, MixSupportsPerComponentCoefficient)
{
  // c=(2,2): extrapolation past the usual [0,1] range, per-component - not the scalar-only
  // restriction the old implementation forced.
  auto graph = singleNodeGraph("math/mix", {"float2"},
                               obj({{"a", literal({1, 1}, 0)}, {"b", literal({2, 2}, 0)}, {"c", literal({2, 2}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
  ASSERT_TRUE(std::holds_alternative<glm::vec2>(v));
  EXPECT_FLOAT_EQ(std::get<glm::vec2>(v).x, 3.0f);
  EXPECT_FLOAT_EQ(std::get<glm::vec2>(v).y, 3.0f);
}

// math/smoothStep was simply never implemented (silently no-op'd) - found the same way.
TEST(InteractivityEval, SmoothStepMatchesSpecFormula)
{
  // t = saturate((c-min(a,b))/|b-a|); value = t*t*(3-2t). a=0,b=1,c=0.25 -> t=0.25 -> 0.15625.
  auto graph = singleNodeGraph("math/smoothStep", {"float"},
                               obj({{"a", literal({0.0}, 0)}, {"b", literal({1.0}, 0)}, {"c", literal({0.25}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_NEAR(evalFloat(instance), 0.15625f, 1e-5f);
}

// math/trunc was simply never implemented (silently no-op'd) - found the same way.
TEST(InteractivityEval, TruncTruncatesTowardZeroUnlikeFloor)
{
  auto graph = singleNodeGraph("math/trunc", {"float"}, obj({{"a", literal({-9.23}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_FLOAT_EQ(evalFloat(instance), -9.0f) << "trunc rounds toward zero; floor would give -10";
}

// glm::sign is comparison-based and silently returns 0 for NaN instead of propagating it (spec's
// blanket "Arithmetic Operations" rule) - found the same way.
TEST(InteractivityEval, SignPropagatesNaN)
{
  auto graph = singleNodeGraph("math/sign", {"float"}, obj({{"a", literal({std::numeric_limits<double>::quiet_NaN()}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_TRUE(std::isnan(evalFloat(instance)));
}

TEST(InteractivityEvalQuat, FromAxisAngleToAxisAngleRoundTrip)
{
  // types: 0=float3 (axis), 1=float (angle)
  auto graph = singleNodeGraph("math/quatFromAxisAngle", {"float3", "float"},
                               obj({{"axis", literal({0, 1, 0}, 0)}, {"angle", literal({1.0471975512}, 1)}}));  // 60deg
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  InteractivityValue         q = evaluateNodeOutput(instance, 0, "value");
  ASSERT_TRUE(std::holds_alternative<glm::vec4>(q));
  const glm::vec4 quat = std::get<glm::vec4>(q);
  EXPECT_NEAR(quat.w, std::cos(0.5235987756), 1e-5f);  // cos(30deg)

  // Feed the resulting quaternion into quatToAxisAngle and confirm we recover ~60deg.
  auto graph2 = singleNodeGraph("math/quatToAxisAngle", {"float4"}, obj({{"a", literal({quat.x, quat.y, quat.z, quat.w}, 0)}}));
  ASSERT_TRUE(graph2.has_value());
  InteractivityGraphInstance instance2(*graph2);
  EXPECT_NEAR(evalFloat(instance2, "angle"), 1.0471975512f, 1e-4f);
}

TEST(InteractivityEvalQuat, ConjugateNegatesVectorPart)
{
  auto graph = singleNodeGraph("math/quatConjugate", {"float4"}, obj({{"a", literal({0.1, 0.2, 0.3, 0.9}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
  ASSERT_TRUE(std::holds_alternative<glm::vec4>(v));
  const glm::vec4 q = std::get<glm::vec4>(v);
  EXPECT_NEAR(q.x, -0.1f, 1e-6f);
  EXPECT_NEAR(q.w, 0.9f, 1e-6f);
}

TEST(InteractivityEvalQuat, MatComposeMatDecomposeRoundTrip)
{
  // 90-degree rotation about Y = quat(0, sin(45deg), 0, cos(45deg))
  const float half = 0.7853981634f;
  const float s = std::sin(half), c = std::cos(half);
  auto        graph = singleNodeGraph(
      "math/matCompose", {"float3", "float4"},
      obj({{"translation", literal({1, 2, 3}, 0)}, {"rotation", literal({0, s, 0, c}, 1)}, {"scale", literal({2, 2, 2}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  InteractivityValue         m = evaluateNodeOutput(instance, 0, "value");
  ASSERT_TRUE(std::holds_alternative<glm::mat4>(m));

  // Feed the composed matrix's 16 components back in as a literal (types: 0=float4x4).
  const glm::mat4        composed = std::get<glm::mat4>(m);
  tinygltf::Value::Array flat;
  for(int col = 0; col < 4; ++col)
    for(int row = 0; row < 4; ++row)
      flat.push_back(tinygltf::Value(static_cast<double>(composed[col][row])));
  tinygltf::Value matLiteral = obj({{"value", tinygltf::Value(flat)}, {"type", num(0)}});
  auto            graph3     = singleNodeGraph("math/matDecompose", {"float4x4"}, obj({{"a", matLiteral}}));
  ASSERT_TRUE(graph3.has_value());
  InteractivityGraphInstance instance3(*graph3);

  InteractivityValue t = evaluateNodeOutput(instance3, 0, "translation");
  ASSERT_TRUE(std::holds_alternative<glm::vec3>(t));
  EXPECT_NEAR(std::get<glm::vec3>(t).x, 1.0f, 1e-4f);
  EXPECT_NEAR(std::get<glm::vec3>(t).y, 2.0f, 1e-4f);
  EXPECT_NEAR(std::get<glm::vec3>(t).z, 3.0f, 1e-4f);

  InteractivityValue sc = evaluateNodeOutput(instance3, 0, "scale");
  ASSERT_TRUE(std::holds_alternative<glm::vec3>(sc));
  EXPECT_NEAR(std::get<glm::vec3>(sc).x, 2.0f, 1e-4f);
  EXPECT_NEAR(std::get<glm::vec3>(sc).z, 2.0f, 1e-4f);

  InteractivityValue rot = evaluateNodeOutput(instance3, 0, "rotation");
  ASSERT_TRUE(std::holds_alternative<glm::vec4>(rot));
  EXPECT_NEAR(std::get<glm::vec4>(rot).y, s, 1e-4f);
  EXPECT_NEAR(std::get<glm::vec4>(rot).w, c, 1e-4f);
}

TEST(InteractivityEvalQuat, SlerpEndpointsReturnInputs)
{
  auto make = [](float t) {
    return singleNodeGraph("math/quatSlerp", {"float4", "float"},
                           obj({{"a", literal({0, 0, 0, 1}, 0)},
                                {"b", literal({0, 0.7071068, 0, 0.7071068}, 0)},
                                {"c", literal({static_cast<double>(t)}, 1)}}));
  };
  {
    auto graph = make(0.0f);
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
    ASSERT_TRUE(std::holds_alternative<glm::vec4>(v));
    EXPECT_NEAR(std::get<glm::vec4>(v).w, 1.0f, 1e-5f);
  }
  {
    auto graph = make(1.0f);
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
    ASSERT_TRUE(std::holds_alternative<glm::vec4>(v));
    EXPECT_NEAR(std::get<glm::vec4>(v).y, 0.7071068f, 1e-4f);
  }
}

TEST(InteractivityEvalSwizzle, CombineExtractRoundTrip3)
{
  auto graph = singleNodeGraph("math/combine3", {"float"},
                               obj({{"a", literal({1}, 0)}, {"b", literal({2}, 0)}, {"c", literal({3}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  InteractivityValue         combined = evaluateNodeOutput(instance, 0, "value");
  ASSERT_TRUE(std::holds_alternative<glm::vec3>(combined));
  const glm::vec3 v = std::get<glm::vec3>(combined);
  EXPECT_FLOAT_EQ(v.x, 1.0f);
  EXPECT_FLOAT_EQ(v.z, 3.0f);

  auto graphEx = singleNodeGraph("math/extract3", {"float3"}, obj({{"a", literal({4, 5, 6}, 0)}}));
  ASSERT_TRUE(graphEx.has_value());
  InteractivityGraphInstance instanceEx(*graphEx);
  EXPECT_FLOAT_EQ(evalFloat(instanceEx, "0"), 4.0f);
  EXPECT_FLOAT_EQ(evalFloat(instanceEx, "1"), 5.0f);
  EXPECT_FLOAT_EQ(evalFloat(instanceEx, "2"), 6.0f);
}

TEST(InteractivityEvalSwizzle, Combine3x3ExtractRoundTrip)
{
  auto graph = singleNodeGraph("math/combine3x3", {"float"},
                               obj({{"a", literal({1}, 0)},
                                    {"b", literal({2}, 0)},
                                    {"c", literal({3}, 0)},
                                    {"d", literal({4}, 0)},
                                    {"e", literal({5}, 0)},
                                    {"f", literal({6}, 0)},
                                    {"g", literal({7}, 0)},
                                    {"h", literal({8}, 0)},
                                    {"i", literal({9}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  InteractivityValue         m = evaluateNodeOutput(instance, 0, "value");
  ASSERT_TRUE(std::holds_alternative<glm::mat3>(m));
  EXPECT_FLOAT_EQ(std::get<glm::mat3>(m)[0][0], 1.0f);
  EXPECT_FLOAT_EQ(std::get<glm::mat3>(m)[2][2], 9.0f);
}

TEST(InteractivityEvalIntBool, LogicOpsDispatchByType)
{
  auto boolGraph = [](const std::string& op, bool a, bool b) {
    return singleNodeGraph(op, {"bool"}, obj({{"a", literal({a ? 1.0 : 0.0}, 0)}, {"b", literal({b ? 1.0 : 0.0}, 0)}}));
  };
  {
    auto graph = boolGraph("math/and", true, false);
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
    ASSERT_TRUE(std::holds_alternative<bool>(v));
    EXPECT_FALSE(std::get<bool>(v));
  }
  {
    auto graph = boolGraph("math/xor", true, false);
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
    ASSERT_TRUE(std::holds_alternative<bool>(v));
    EXPECT_TRUE(std::get<bool>(v));
  }
  auto intGraph = [](const std::string& op, int a, int b) {
    return singleNodeGraph(op, {"int"},
                           obj({{"a", literal({static_cast<double>(a)}, 0)}, {"b", literal({static_cast<double>(b)}, 0)}}));
  };
  {
    auto graph = intGraph("math/and", 0b1100, 0b1010);
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
    ASSERT_TRUE(std::holds_alternative<int32_t>(v));
    EXPECT_EQ(std::get<int32_t>(v), 0b1000);
  }
}

TEST(InteractivityEvalBitwise, ShiftsClzCtzPopcnt)
{
  auto intOp1 = [](const std::string& op, int a) {
    return singleNodeGraph(op, {"int"}, obj({{"a", literal({static_cast<double>(a)}, 0)}}));
  };
  auto intOp2 = [](const std::string& op, int a, int b) {
    return singleNodeGraph(op, {"int"},
                           obj({{"a", literal({static_cast<double>(a)}, 0)}, {"b", literal({static_cast<double>(b)}, 0)}}));
  };
  auto asInt = [](InteractivityGraphInstance& instance) {
    InteractivityValue v = evaluateNodeOutput(instance, 0, "value");
    return std::holds_alternative<int32_t>(v) ? std::get<int32_t>(v) : -12345;
  };
  {
    auto graph = intOp2("math/lsl", 1, 4);
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_EQ(asInt(instance), 16);
  }
  {
    // Arithmetic shift: sign bit propagates.
    auto graph = intOp2("math/asr", -8, 1);
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_EQ(asInt(instance), -4);
  }
  {
    auto graph = intOp1("math/clz", 1);
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_EQ(asInt(instance), 31);
  }
  {
    auto graph = intOp1("math/popcnt", 0b1011);
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_EQ(asInt(instance), 3);
  }
}

TEST(InteractivityEvalColor, RgbOkLChRoundTrip)
{
  auto graph = singleNodeGraph("math/rgbToOkLCh", {"float"},
                               obj({{"r", literal({0.5}, 0)}, {"g", literal({0.5}, 0)}, {"b", literal({0.5}, 0)}}));
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  const float                l = evalFloat(instance, "l"), c = evalFloat(instance, "c"), h = evalFloat(instance, "h");

  auto graph2 = singleNodeGraph("math/rgbFromOkLCh", {"float"},
                                obj({{"l", literal({static_cast<double>(l)}, 0)},
                                     {"c", literal({static_cast<double>(c)}, 0)},
                                     {"h", literal({static_cast<double>(h)}, 0)}}));
  ASSERT_TRUE(graph2.has_value());
  InteractivityGraphInstance instance2(*graph2);
  EXPECT_NEAR(evalFloat(instance2, "r"), 0.5f, 1e-3f);
  EXPECT_NEAR(evalFloat(instance2, "g"), 0.5f, 1e-3f);
  EXPECT_NEAR(evalFloat(instance2, "b"), 0.5f, 1e-3f);
}

TEST(InteractivityEvalSpecial, ConstantsIsNaNIsInf)
{
  {
    auto graph = floatOpGraph("math/Pi", obj({}));
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_NEAR(evalFloat(instance), 3.14159265f, 1e-5f);
  }
  {
    auto graph = floatOpGraph("math/isNaN", obj({{"a", literal({0.0}, 0)}}));
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
    ASSERT_TRUE(std::holds_alternative<bool>(v));
    EXPECT_FALSE(std::get<bool>(v));
  }
}

TEST(InteractivityEvalSpecial, SelectAndSwitch)
{
  auto selectGraph = [](bool cond) {
    // types: 0=bool, 1=float
    tinygltf::Value condLit = obj({{"value", arr({num(cond ? 1 : 0)})}, {"type", num(0)}});
    return singleNodeGraph("math/select", {"bool", "float"},
                           obj({{"condition", condLit}, {"a", literal({1.0}, 1)}, {"b", literal({2.0}, 1)}}));
  };
  {
    auto graph = selectGraph(true);
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_FLOAT_EQ(evalFloat(instance), 1.0f);
  }
  {
    auto graph = selectGraph(false);
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    EXPECT_FLOAT_EQ(evalFloat(instance), 2.0f);
  }
}

// ref/eq must not collapse two JSON-Pointer literals that share a numeric index but reference
// different owning arrays (e.g. "/nodes/2" vs "/materials/2") - regression test for
// InteractivityRef's `category` field, added because parseInteractivityLiteral() used to build
// InteractivityRef from the trailing index alone.
TEST(InteractivityEvalSpecial, RefEqDistinguishesSameIndexDifferentCategory)
{
  auto refLiteral = [](const char* pointer) { return obj({{"value", arr({str(pointer)})}, {"type", num(0)}}); };
  {
    auto graph = singleNodeGraph("ref/eq", {"ref"}, obj({{"a", refLiteral("/nodes/2")}, {"b", refLiteral("/materials/2")}}));
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
    ASSERT_TRUE(std::holds_alternative<bool>(v));
    EXPECT_FALSE(std::get<bool>(v));
  }
  {
    auto graph = singleNodeGraph("ref/eq", {"ref"}, obj({{"a", refLiteral("/nodes/2")}, {"b", refLiteral("/nodes/2")}}));
    ASSERT_TRUE(graph.has_value());
    InteractivityGraphInstance instance(*graph);
    InteractivityValue         v = evaluateNodeOutput(instance, 0, "value");
    ASSERT_TRUE(std::holds_alternative<bool>(v));
    EXPECT_TRUE(std::get<bool>(v));
  }
}

TEST(InteractivityEvalSpecial, RandomIsStableWithinOneFlowActivationAndFinite)
{
  tinygltf::Value graphJson =
      obj({{"declarations", arr({obj({{"op", str("math/random")}})})}, {"nodes", arr({obj({{"declaration", num(0)}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "rand", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  const float                first  = evalFloat(instance);
  const float                second = evalFloat(instance);  // no flow activation happened in between
  EXPECT_FLOAT_EQ(first, second);
  EXPECT_GE(first, 0.0f);
  EXPECT_LT(first, 1.0f);
}

//--------------------------------------------------------------------------------------------------
// Phase D: remaining flow control, events, debug/log.
//--------------------------------------------------------------------------------------------------

namespace {
// Builds a 2-variable (both int, init 0) graph with `onStart` wired to `firstNodeDecl`, so tests
// can observe flow-control ops by checking which variable(s) end up written.
int32_t varValue(InteractivityGraphInstance& instance, int index)
{
  InteractivityValue v = instance.variable(index);
  return std::holds_alternative<int32_t>(v) ? std::get<int32_t>(v) : -999999;
}
}  // namespace

TEST(InteractivityFlow, SwitchActivatesMatchingCase)
{
  // selection=2, cases=[1,2] -> case "2" fires (writes var0=1), default does not (var1 stays 0).
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}}), obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/switch")}}),
                             obj({{"op", str("variable/set")}}), obj({{"op", str("variable/set")}})})},
       {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                      obj({{"declaration", num(1)},
                           {"configuration", obj({{"cases", obj({{"value", arr({num(1), num(2)})}})}})},
                           {"values", obj({{"selection", literal({2}, 0)}})},
                           {"flows", obj({{"2", flow(2)}, {"default", flow(3)}})}}),
                      obj({{"declaration", num(2)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                           {"values", obj({{"0", literal({1}, 0)}})}}),
                      obj({{"declaration", num(3)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(1)})}})}})},
                           {"values", obj({{"1", literal({1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "switch", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 1);
  EXPECT_EQ(varValue(instance, 1), 0);
}

// for(start=0,end=5): loopBody reads+increments var0 each iteration via variable/get -> math/add
// -> variable/set. Final var0 must be 5, and `for`'s own "index" output must read back as 5.
TEST(InteractivityFlow, ForLoopsCorrectCountAndUpdatesVariable)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/for")}}), obj({{"op", str("variable/get")}}),
                             obj({{"op", str("math/add")}}), obj({{"op", str("variable/set")}})})},
       {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                      obj({{"declaration", num(1)},
                           {"values", obj({{"startIndex", literal({0}, 0)}, {"endIndex", literal({5}, 0)}})},
                           {"flows", obj({{"loopBody", flow(4)}})}}),
                      obj({{"declaration", num(2)}, {"configuration", obj({{"variable", obj({{"value", arr({num(0)})}})}})}}),
                      obj({{"declaration", num(3)}, {"values", obj({{"a", ref(2)}, {"b", literal({1}, 0)}})}}),
                      obj({{"declaration", num(4)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                           {"values", obj({{"0", ref(3)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "for", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 5);
  InteractivityValue index = evaluateNodeOutput(instance, 1, "index");
  ASSERT_TRUE(std::holds_alternative<int32_t>(index));
  EXPECT_EQ(std::get<int32_t>(index), 5);
}

// while(var0 < 3): condition read from a node BEFORE `while` (DAG value-ref rule); increment
// chain lives AFTER `while` (DAG flow-ref rule) but re-reads the variable fresh each iteration.
TEST(InteractivityFlow, WhileLoopsUntilConditionFalse)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("variable/get")}}),
                             obj({{"op", str("math/lt")}}), obj({{"op", str("flow/while")}}), obj({{"op", str("variable/get")}}),
                             obj({{"op", str("math/add")}}), obj({{"op", str("variable/set")}})})},
       {"nodes",
        arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(3)}})}}),
             obj({{"declaration", num(1)}, {"configuration", obj({{"variable", obj({{"value", arr({num(0)})}})}})}}),
             obj({{"declaration", num(2)}, {"values", obj({{"a", ref(1)}, {"b", literal({3}, 0)}})}}),
             obj({{"declaration", num(3)}, {"values", obj({{"condition", ref(2)}})}, {"flows", obj({{"loopBody", flow(6)}})}}),
             obj({{"declaration", num(4)}, {"configuration", obj({{"variable", obj({{"value", arr({num(0)})}})}})}}),
             obj({{"declaration", num(5)}, {"values", obj({{"a", ref(4)}, {"b", literal({1}, 0)}})}}),
             obj({{"declaration", num(6)},
                  {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                  {"values", obj({{"0", ref(5)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "while", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 3);
}

// for(0..5) drives doN(n=3) five times; doN only forwards its first 3 activations, so a counter
// incremented by doN's `out` ends at 3, not 5.
TEST(InteractivityFlow, DoNLimitsActivationsAcrossForLoop)
{
  tinygltf::Value graphJson =
      obj({{"types", arr({obj({{"signature", str("int")}})})},
           {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
           {"declarations",
            arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/for")}}), obj({{"op", str("flow/doN")}}),
                 obj({{"op", str("variable/get")}}), obj({{"op", str("math/add")}}), obj({{"op", str("variable/set")}})})},
           {"nodes",
            arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                 obj({{"declaration", num(1)},
                      {"values", obj({{"startIndex", literal({0}, 0)}, {"endIndex", literal({5}, 0)}})},
                      {"flows", obj({{"loopBody", flow(2)}})}}),
                 obj({{"declaration", num(2)}, {"values", obj({{"n", literal({3}, 0)}})}, {"flows", obj({{"out", flow(5)}})}}),
                 obj({{"declaration", num(3)}, {"configuration", obj({{"variable", obj({{"value", arr({num(0)})}})}})}}),
                 obj({{"declaration", num(4)}, {"values", obj({{"a", ref(3)}, {"b", literal({1}, 0)}})}}),
                 obj({{"declaration", num(5)},
                      {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                      {"values", obj({{"0", ref(4)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "doN", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 3);
  InteractivityValue count = evaluateNodeOutput(instance, 2, "currentCount");
  ASSERT_TRUE(std::holds_alternative<int32_t>(count));
  EXPECT_EQ(std::get<int32_t>(count), 3);
}

// for(0..3) drives multiGate (2 outputs, sequential, no loop) three times: gate "0" fires once,
// gate "1" fires once, the third activation finds everything used and fires nothing.
TEST(InteractivityFlow, MultiGateSequentialStopsAfterAllOutputsUsed)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}}), obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/for")}}), obj({{"op", str("flow/multiGate")}}),
                             obj({{"op", str("variable/set")}}), obj({{"op", str("variable/set")}})})},
       {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                      obj({{"declaration", num(1)},
                           {"values", obj({{"startIndex", literal({0}, 0)}, {"endIndex", literal({3}, 0)}})},
                           {"flows", obj({{"loopBody", flow(2)}})}}),
                      obj({{"declaration", num(2)},
                           {"configuration", obj({{"isRandom", obj({{"value", arr({tinygltf::Value(false)})}})},
                                                  {"isLoop", obj({{"value", arr({tinygltf::Value(false)})}})}})},
                           {"flows", obj({{"0", flow(3)}, {"1", flow(4)}})}}),
                      obj({{"declaration", num(3)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                           {"values", obj({{"0", literal({1}, 0)}})}}),
                      obj({{"declaration", num(4)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(1)})}})}})},
                           {"values", obj({{"1", literal({1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "multiGate", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 1);
  EXPECT_EQ(varValue(instance, 1), 1);
  InteractivityValue lastIndex = evaluateNodeOutput(instance, 2, "lastIndex");
  ASSERT_TRUE(std::holds_alternative<int32_t>(lastIndex));
  EXPECT_EQ(std::get<int32_t>(lastIndex), 1);
}

// A sequence delivers 2 of waitAll's 2 configured inputs in one activation: the first fires
// `out`, the second (completing the set) fires `completed` instead.
TEST(InteractivityFlow, WaitAllFiresCompletedOnlyOnceAllInputsSeen)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}}), obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/sequence")}}), obj({{"op", str("flow/waitAll")}}),
                             obj({{"op", str("variable/set")}}), obj({{"op", str("variable/set")}})})},
       {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                      obj({{"declaration", num(1)}, {"flows", obj({{"a", ref(2, "0")}, {"b", ref(2, "1")}})}}),
                      obj({{"declaration", num(2)},
                           {"configuration", obj({{"inputFlows", obj({{"value", arr({num(2)})}})}})},
                           {"flows", obj({{"out", flow(3)}, {"completed", flow(4)}})}}),
                      obj({{"declaration", num(3)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                           {"values", obj({{"0", literal({1}, 0)}})}}),
                      obj({{"declaration", num(4)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(1)})}})}})},
                           {"values", obj({{"1", literal({1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "waitAll", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 1);  // `out` fired on the first (non-completing) input
  EXPECT_EQ(varValue(instance, 1), 1);  // `completed` fired on the second (completing) input
}

// Two throttle activations in the same instant (duration=10s): the first fires `out`, the
// immediate second does not (elapsed=0 < duration), leaving a counter at 1, not 2.
TEST(InteractivityFlow, ThrottleBlocksImmediateSecondActivation)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}}), obj({{"signature", str("float")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"declarations",
        arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/sequence")}}), obj({{"op", str("flow/throttle")}}),
             obj({{"op", str("variable/get")}}), obj({{"op", str("math/add")}}), obj({{"op", str("variable/set")}})})},
       {"nodes",
        arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
             obj({{"declaration", num(1)}, {"flows", obj({{"a", flow(2)}, {"b", flow(2)}})}}),
             obj({{"declaration", num(2)}, {"values", obj({{"duration", literal({10.0}, 1)}})}, {"flows", obj({{"out", flow(5)}})}}),
             obj({{"declaration", num(3)}, {"configuration", obj({{"variable", obj({{"value", arr({num(0)})}})}})}}),
             obj({{"declaration", num(4)}, {"values", obj({{"a", ref(3)}, {"b", literal({1}, 0)}})}}),
             obj({{"declaration", num(5)},
                  {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                  {"values", obj({{"0", ref(4)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "throttle", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 1);
  InteractivityValue remaining = evaluateNodeOutput(instance, 2, "lastRemainingTime");
  ASSERT_TRUE(std::holds_alternative<float>(remaining));
  EXPECT_NEAR(std::get<float>(remaining), 10.0f, 1e-5f);
}

// setDelay(duration=1s): `out` fires immediately; `done` only fires once enough tick() time has
// elapsed. The first tick() after start() never advances time (spec's own first-tick NaN rule -
// see InteractivityInstance.FirstTickTimingMatchesSpec), so a "priming" tick is needed first.
TEST(InteractivityFlow, SetDelayFiresDoneAfterElapsedTickTime)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}}), obj({{"signature", str("float")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}}), obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/setDelay")}}),
                             obj({{"op", str("variable/set")}}), obj({{"op", str("variable/set")}})})},
       {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                      obj({{"declaration", num(1)},
                           {"values", obj({{"duration", literal({1.0}, 1)}})},
                           {"flows", obj({{"out", flow(2)}, {"done", flow(3)}})}}),
                      obj({{"declaration", num(2)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                           {"values", obj({{"0", literal({1}, 0)}})}}),
                      obj({{"declaration", num(3)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(1)})}})}})},
                           {"values", obj({{"1", literal({1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "setDelay", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 1);  // `out` fired immediately
  EXPECT_EQ(varValue(instance, 1), 0);  // `done` has not fired yet

  instance.tick(0.0f);  // priming tick (first tick never advances time - see comment above)
  instance.tick(0.5f);  // elapsed 0.5s < 1.0s duration
  EXPECT_EQ(varValue(instance, 1), 0);
  instance.tick(0.6f);  // elapsed 1.1s >= 1.0s duration
  EXPECT_EQ(varValue(instance, 1), 1);
}

// onStart -> sequence -> (a) setDelay, then (b) cancelDelay referencing setDelay's `lastDelay`
// output. Since sequence enqueues in order and the queue is FIFO, setDelay has already run (and
// set `lastDelay`) by the time cancelDelay executes - ticking well past the duration afterward
// must never fire `done`.
TEST(InteractivityFlow, CancelDelayPreventsDoneFromFiring)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}}), obj({{"signature", str("ref")}}), obj({{"signature", str("float")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}}), obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/sequence")}}),
                             obj({{"op", str("flow/setDelay")}}), obj({{"op", str("variable/set")}}),
                             obj({{"op", str("flow/cancelDelay")}}), obj({{"op", str("variable/set")}})})},
       {"nodes",
        arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
             obj({{"declaration", num(1)}, {"flows", obj({{"a", flow(2)}, {"b", flow(4)}})}}),
             obj({{"declaration", num(2)}, {"values", obj({{"duration", literal({1.0}, 2)}})}, {"flows", obj({{"done", flow(3)}})}}),
             obj({{"declaration", num(3)},
                  {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                  {"values", obj({{"0", literal({1}, 0)}})}}),
             obj({{"declaration", num(4)}, {"values", obj({{"delay", ref(2, "lastDelay")}})}, {"flows", obj({{"out", flow(5)}})}}),
             obj({{"declaration", num(5)},
                  {"configuration", obj({{"variables", obj({{"value", arr({num(1)})}})}})},
                  {"values", obj({{"1", literal({1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "cancelDelay", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 1), 1);  // cancelDelay's `out` fired

  instance.tick(0.0f);
  instance.tick(5.0f);                  // well past the 1s duration
  EXPECT_EQ(varValue(instance, 0), 0);  // `done` never fired - the delay was cancelled
}

// variable/interpolate (spec 3739-3845): p1=(0,0)/p2=(1,1) is the *linear* Bézier (a straight
// line y=x), so the eased progress q exactly equals the elapsed-time fraction t - making the
// mid-interpolation value trivially predictable (5.0, halfway from 0 to the 10.0 target).
TEST(InteractivityFlow, VariableInterpolateEasesOverTicksAndFiresDone)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}}), obj({{"signature", str("float")}}), obj({{"signature", str("float2")}})})},
       {"variables", arr({obj({{"type", num(1)}, {"value", arr({num(0)})}}), obj({{"type", num(0)}, {"value", arr({num(0)})}}),
                          obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("variable/interpolate")}}),
                             obj({{"op", str("variable/set")}}), obj({{"op", str("variable/set")}})})},
       {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                      obj({{"declaration", num(1)},
                           {"configuration", obj({{"variable", obj({{"value", arr({num(0)})}})},
                                                  {"useSlerp", obj({{"value", arr({tinygltf::Value(false)})}})}})},
                           {"values", obj({{"value", literal({10.0}, 1)},
                                           {"duration", literal({1.0}, 1)},
                                           {"p1", literal({0.0, 0.0}, 2)},
                                           {"p2", literal({1.0, 1.0}, 2)}})},
                           {"flows", obj({{"out", flow(2)}, {"done", flow(3)}})}}),
                      obj({{"declaration", num(2)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(1)})}})}})},
                           {"values", obj({{"1", literal({1}, 0)}})}}),
                      obj({{"declaration", num(3)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(2)})}})}})},
                           {"values", obj({{"2", literal({1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "variableInterpolate", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();

  auto floatVar = [&](int index) { return std::get<float>(instance.variable(index)); };
  EXPECT_NEAR(floatVar(0), 0.0f, 1e-5f);  // t<=0 on the activation tick itself - no write yet
  EXPECT_EQ(varValue(instance, 1), 1);    // `out` fired immediately
  EXPECT_EQ(varValue(instance, 2), 0);    // `done` has not fired yet

  instance.tick(0.0f);  // priming tick (first tick never advances time)
  instance.tick(0.5f);  // elapsed 0.5s / 1.0s duration -> t = q = 0.5 (linear easing)
  EXPECT_NEAR(floatVar(0), 5.0f, 1e-4f);
  EXPECT_EQ(varValue(instance, 2), 0);
  instance.tick(0.6f);  // elapsed 1.1s >= 1.0s duration -> snaps to target, `done` fires
  EXPECT_NEAR(floatVar(0), 10.0f, 1e-4f);
  EXPECT_EQ(varValue(instance, 2), 1);
}

// Out-of-range variable index (spec: "the variable index MUST be a non-negative integer less than
// the total number of custom variables") reports via `err`, never a crash.
TEST(InteractivityFlow, VariableInterpolateInvalidVariableActivatesErr)
{
  tinygltf::Value graphJson =
      obj({{"types", arr({obj({{"signature", str("int")}}), obj({{"signature", str("float")}}), obj({{"signature", str("float2")}})})},
           {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
           {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("variable/interpolate")}}),
                                 obj({{"op", str("variable/set")}})})},
           {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                          obj({{"declaration", num(1)},
                               {"configuration", obj({{"variable", obj({{"value", arr({num(99)})}})}})},
                               {"values", obj({{"value", literal({10.0}, 1)},
                                               {"duration", literal({1.0}, 1)},
                                               {"p1", literal({0.0, 0.0}, 2)},
                                               {"p2", literal({1.0, 1.0}, 2)}})},
                               {"flows", obj({{"err", flow(2)}})}}),
                          obj({{"declaration", num(2)},
                               {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                               {"values", obj({{"0", literal({1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "variableInterpolateInvalid", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_NO_THROW(instance.start());
  EXPECT_EQ(varValue(instance, 0), 1);  // `err` fired, not `out`
}

namespace {
// animation/start|stop|stopAt (spec 4262-4441): stands in for a Scene-backed
// InteractivityAnimationResolver so the timestamp-remap/threshold state machine is exercised
// deterministically (real Khronos conformance content is also verified - see docs/interactivity.md).
// Animation index 0 is "valid" with a duration (spec's T) of 4 seconds; everything else is invalid.
class MockAnimationResolver : public InteractivityAnimationResolver
{
public:
  bool  isValidAnimation(int index) const override { return index == 0; }
  float animationMaxTime(int /*index*/) const override { return 4.0f; }
  // Records every applyPose() call (index, effectiveTime) in call order - tests use this to confirm
  // the pose is applied *before* `done`/stop-`done` fires (spec 4340-4368's own step ordering; the
  // bug this guards against: reading the position via a debug/log reached from that same `done`
  // activation must already see the fresh value, not the previous tick's).
  bool applyPose(int index, float effectiveTime) override
  {
    appliedPoses.push_back({index, effectiveTime});
    return index == 0;
  }
  std::vector<std::pair<int, float>> appliedPoses;
};

// `ref`-typed literal *values* are JSON-Pointer-path strings per spec (e.g. "/animations/0"), not
// raw integer handles - see InteractivityGraph.cpp's eRef literal parsing / docs/interactivity.md.
tinygltf::Value animRef(int animationIndex)
{
  return obj({{"value", arr({str("/animations/" + std::to_string(animationIndex))})}, {"type", num(0)}});
}
}  // namespace

// animation/start: pendingAnimationApplies() surfaces the (animationIndex, effectiveTimestamp) this
// engine computes each tick() - GltfRenderer applies these via AnimationSystem::updateAnimation() in
// real use (see renderer.cpp), but the timestamp math itself needs no Scene/GPU to verify. `done`
// fires exactly once currentTimestamp reaches `endTime`, and applies land at that exact endTime (not
// whatever currentTimestamp happened to overshoot to), matching spec step 6.
TEST(InteractivityFlow, AnimationStartAppliesPosesAndFiresDoneAtEndTime)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("ref")}}), obj({{"signature", str("float")}}), obj({{"signature", str("int")}})})},
       {"variables", arr({obj({{"type", num(2)}, {"value", arr({num(0)})}})})},
       {"declarations",
        arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("animation/start")}}), obj({{"op", str("variable/set")}})})},
       {"nodes",
        arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
             obj({{"declaration", num(1)},
                  {"values", obj({{"animation", animRef(0)}, {"startTime", literal({0.0}, 1)}, {"endTime", literal({4.0}, 1)}, {"speed", literal({1.0}, 1)}})},
                  {"flows", obj({{"done", flow(2)}})}}),
             obj({{"declaration", num(2)},
                  {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                  {"values", obj({{"0", literal({1}, 2)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "animStart", 0);
  ASSERT_TRUE(graph.has_value());

  MockAnimationResolver      resolver;
  InteractivityGraphInstance instance(*graph);
  instance.setAnimationResolver(&resolver);
  instance.start();
  EXPECT_TRUE(instance.pendingAnimationApplies().empty());  // advanceAnimations() only runs on tick()

  instance.tick(0.0f);  // priming tick (first tick never advances time)
  ASSERT_EQ(instance.pendingAnimationApplies().size(), 1u);
  EXPECT_EQ(instance.pendingAnimationApplies()[0].animationIndex, 0);
  EXPECT_NEAR(instance.pendingAnimationApplies()[0].effectiveTime, 0.0f, 1e-5f);
  EXPECT_EQ(varValue(instance, 0), 0);  // `done` not fired yet

  instance.tick(2.0f);  // elapsed 2s / 4s duration
  ASSERT_EQ(instance.pendingAnimationApplies().size(), 1u);
  EXPECT_NEAR(instance.pendingAnimationApplies()[0].effectiveTime, 2.0f, 1e-5f);
  EXPECT_EQ(varValue(instance, 0), 0);

  instance.tick(2.5f);  // elapsed 4.5s >= 4s duration -> snaps to endTime, `done` fires
  ASSERT_EQ(instance.pendingAnimationApplies().size(), 1u);
  EXPECT_NEAR(instance.pendingAnimationApplies()[0].effectiveTime, 4.0f, 1e-5f);
  EXPECT_EQ(varValue(instance, 0), 1);

  instance.tick(1.0f);  // entry was removed on completion - nothing left to (re-)apply
  EXPECT_TRUE(instance.pendingAnimationApplies().empty());
}

// animation/start's err path (spec step 2-4): invalid ref, non-finite start/end times, and
// non-positive/non-finite speed must all activate `err`, never start an entry.
TEST(InteractivityFlow, AnimationStartInvalidInputsActivateErr)
{
  // Strict JSON has no numeric token for Infinity/NaN - authored as quoted string tokens, same as
  // the math/* Infinity/NaN literal tests above.
  auto floatToken = [](const char* token) { return obj({{"value", arr({str(token)})}, {"type", num(1)}}); };

  auto build = [](tinygltf::Value animLit, tinygltf::Value startLit, tinygltf::Value endLit, tinygltf::Value speedLit) {
    return obj(
        {{"types", arr({obj({{"signature", str("ref")}}), obj({{"signature", str("float")}}), obj({{"signature", str("int")}})})},
         {"variables", arr({obj({{"type", num(2)}, {"value", arr({num(0)})}})})},
         {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("animation/start")}}),
                               obj({{"op", str("variable/set")}})})},
         {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                        obj({{"declaration", num(1)},
                             {"values", obj({{"animation", animLit}, {"startTime", startLit}, {"endTime", endLit}, {"speed", speedLit}})},
                             {"flows", obj({{"err", flow(2)}})}}),
                        obj({{"declaration", num(2)},
                             {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                             {"values", obj({{"0", literal({1}, 2)}})}})})}});
  };

  MockAnimationResolver resolver;
  auto                  expectErr = [&](tinygltf::Value graphJson, const char* label) {
    auto graph = InteractivityGraph::parse(graphJson, "animStartInvalid", 0);
    ASSERT_TRUE(graph.has_value()) << label;
    InteractivityGraphInstance instance(*graph);
    instance.setAnimationResolver(&resolver);
    EXPECT_NO_THROW(instance.start()) << label;
    EXPECT_EQ(varValue(instance, 0), 1) << label;
  };

  expectErr(build(animRef(99), literal({0.0}, 1), literal({4.0}, 1), literal({1.0}, 1)), "invalid ref");
  expectErr(build(animRef(0), floatToken("NaN"), literal({4.0}, 1), literal({1.0}, 1)), "startTime NaN");
  expectErr(build(animRef(0), floatToken("Infinity"), literal({4.0}, 1), literal({1.0}, 1)), "startTime Infinity");
  expectErr(build(animRef(0), literal({0.0}, 1), floatToken("NaN"), literal({1.0}, 1)), "endTime NaN");
  expectErr(build(animRef(0), literal({0.0}, 1), literal({4.0}, 1), literal({0.0}, 1)), "speed zero");
  expectErr(build(animRef(0), literal({0.0}, 1), literal({4.0}, 1), literal({-1.0}, 1)), "speed negative");
  expectErr(build(animRef(0), literal({0.0}, 1), literal({4.0}, 1), floatToken("NaN")), "speed NaN");
}

// animation/stop: removes the active entry without firing its `done`; a subsequent tick no longer
// applies anything for that animation. Also verifies the `err` path for an invalid reference.
TEST(InteractivityFlow, AnimationStopRemovesEntryWithoutFiringDone)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("ref")}}), obj({{"signature", str("float")}}), obj({{"signature", str("int")}})})},
       {"variables", arr({obj({{"type", num(2)}, {"value", arr({num(0)})}}), obj({{"type", num(2)}, {"value", arr({num(0)})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/sequence")}}), obj({{"op", str("animation/start")}}),
                             obj({{"op", str("animation/stop")}}), obj({{"op", str("variable/set")}})})},
       {"nodes",
        arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
             obj({{"declaration", num(1)}, {"flows", obj({{"a", flow(2)}, {"b", flow(3)}})}}),
             obj({{"declaration", num(2)},
                  {"values", obj({{"animation", animRef(0)}, {"startTime", literal({0.0}, 1)}, {"endTime", literal({4.0}, 1)}, {"speed", literal({1.0}, 1)}})},
                  {"flows", obj({{"done", flow(4)}})}}),
             obj({{"declaration", num(3)}, {"values", obj({{"animation", animRef(0)}})}}),
             obj({{"declaration", num(4)},
                  {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                  {"values", obj({{"0", literal({1}, 2)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "animStop", 0);
  ASSERT_TRUE(graph.has_value());

  MockAnimationResolver      resolver;
  InteractivityGraphInstance instance(*graph);
  instance.setAnimationResolver(&resolver);
  instance.start();  // animation/start then immediately animation/stop, same activation

  instance.tick(0.0f);
  instance.tick(10.0f);  // well past the 4s duration - `done` must never fire, entry was removed
  EXPECT_TRUE(instance.pendingAnimationApplies().empty());
  EXPECT_EQ(varValue(instance, 0), 0);
}

// animation/stopAt: overrides the entry's stop time; reaching it fires the stop-completion `done`
// (not the original `done` from animation/start) with the pose frozen at that stop time, and the
// entry is removed so the animation never reaches its original endTime.
TEST(InteractivityFlow, AnimationStopAtFiresStopDoneInsteadOfEndDone)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("ref")}}), obj({{"signature", str("float")}}), obj({{"signature", str("int")}})})},
       {"variables", arr({obj({{"type", num(2)}, {"value", arr({num(0)})}}), obj({{"type", num(2)}, {"value", arr({num(0)})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/sequence")}}),
                             obj({{"op", str("animation/start")}}), obj({{"op", str("animation/stopAt")}}),
                             obj({{"op", str("variable/set")}}), obj({{"op", str("variable/set")}})})},
       {"nodes",
        arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
             obj({{"declaration", num(1)}, {"flows", obj({{"a", flow(2)}, {"b", flow(3)}})}}),
             obj({{"declaration", num(2)},
                  {"values", obj({{"animation", animRef(0)}, {"startTime", literal({0.0}, 1)}, {"endTime", literal({4.0}, 1)}, {"speed", literal({1.0}, 1)}})},
                  {"flows", obj({{"done", flow(4)}})}}),
             obj({{"declaration", num(3)},
                  {"values", obj({{"animation", animRef(0)}, {"stopTime", literal({1.0}, 1)}})},
                  {"flows", obj({{"done", flow(5)}})}}),
             obj({{"declaration", num(4)},
                  {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                  {"values", obj({{"0", literal({1}, 2)}})}}),
             obj({{"declaration", num(5)},
                  {"configuration", obj({{"variables", obj({{"value", arr({num(1)})}})}})},
                  {"values", obj({{"1", literal({1}, 2)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "animStopAt", 0);
  ASSERT_TRUE(graph.has_value());

  MockAnimationResolver      resolver;
  InteractivityGraphInstance instance(*graph);
  instance.setAnimationResolver(&resolver);
  instance.start();  // animation/start then immediately animation/stopAt(stopTime=1.0)

  instance.tick(0.0f);
  instance.tick(0.5f);  // elapsed 0.5s < stopTime(1.0)
  EXPECT_EQ(varValue(instance, 0), 0);
  EXPECT_EQ(varValue(instance, 1), 0);
  ASSERT_EQ(instance.pendingAnimationApplies().size(), 1u);
  EXPECT_NEAR(instance.pendingAnimationApplies()[0].effectiveTime, 0.5f, 1e-5f);

  instance.tick(0.6f);                  // elapsed 1.1s >= stopTime(1.0) - stop-completion fires, not end-completion
  EXPECT_EQ(varValue(instance, 0), 0);  // animation/start's own `done` never fires
  EXPECT_EQ(varValue(instance, 1), 1);  // animation/stopAt's `done` fires instead
  ASSERT_EQ(instance.pendingAnimationApplies().size(), 1u);
  EXPECT_NEAR(instance.pendingAnimationApplies()[0].effectiveTime, 1.0f, 1e-5f);  // frozen at stopTime

  instance.tick(5.0f);  // well past the original 4s endTime - entry was removed, nothing to apply
  EXPECT_TRUE(instance.pendingAnimationApplies().empty());
  EXPECT_EQ(varValue(instance, 0), 0);
}

// event/send delivers its wired "payload" value to every event/receive node declared against
// the same event index.
TEST(InteractivityEvent, SendDeliversValueToReceive)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}}), obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"events",
        arr({obj({{"id", str("myEvent")}, {"values", obj({{"payload", obj({{"type", num(0)}, {"value", arr({num(42)})}})}})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("event/send")}}), obj({{"op", str("event/receive")}}),
                             obj({{"op", str("variable/set")}}), obj({{"op", str("variable/set")}})})},
       {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                      obj({{"declaration", num(1)},
                           {"configuration", obj({{"event", obj({{"value", arr({num(0)})}})}})},
                           {"values", obj({{"payload", literal({99}, 0)}})},
                           {"flows", obj({{"out", flow(4)}})}}),
                      obj({{"declaration", num(2)},
                           {"configuration", obj({{"event", obj({{"value", arr({num(0)})}})}})},
                           {"flows", obj({{"out", flow(3)}})}}),
                      obj({{"declaration", num(3)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                           {"values", obj({{"0", ref(2, "payload")}})}}),
                      obj({{"declaration", num(4)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(1)})}})}})},
                           {"values", obj({{"1", literal({1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "eventSend", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 99);  // received the value event/send evaluated for "payload"
  EXPECT_EQ(varValue(instance, 1), 1);   // event/send's own `out` fired too
}

// event/receive's declared "b" value is never wired by event/send in this occurrence, so it must
// read back as the event's declared default (20), not e.g. monostate or a stale value.
TEST(InteractivityEvent, ReceiveUnwiredValueUsesDeclaredDefault)
{
  tinygltf::Value graphJson =
      obj({{"types", arr({obj({{"signature", str("int")}})})},
           {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(-1)})}})})},
           {"events", arr({obj({{"id", str("e")},
                                {"values", obj({{"a", obj({{"type", num(0)}, {"value", arr({num(10)})}})},
                                                {"b", obj({{"type", num(0)}, {"value", arr({num(20)})}})}})}})})},
           {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("event/send")}}),
                                 obj({{"op", str("event/receive")}}), obj({{"op", str("variable/set")}})})},
           {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                          obj({{"declaration", num(1)},
                               {"configuration", obj({{"event", obj({{"value", arr({num(0)})}})}})},
                               {"values", obj({{"a", literal({5}, 0)}})}}),  // "b" intentionally left unwired
                          obj({{"declaration", num(2)},
                               {"configuration", obj({{"event", obj({{"value", arr({num(0)})}})}})},
                               {"flows", obj({{"out", flow(3)}})}}),
                          obj({{"declaration", num(3)},
                               {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                               {"values", obj({{"0", ref(2, "b")}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "eventDefault", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 20);
}

// event/stopPropagation (spec 4492-4530), stopImmediate=true: Receiver A (declared first, node
// index 2) calls stopPropagation with its own occurrence ref (read via its "event" output socket);
// event/send's dispatch loop must skip Receiver B (declared later, node index 5) entirely - matches
// the official Khronos conformance scene's "stopImmediate=true: Receiver B not triggered" sub-test.
TEST(InteractivityEvent, StopPropagationImmediateSkipsRemainingReceivers)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}}), obj({{"signature", str("bool")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}}), obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"events", arr({obj({{"id", str("e")}, {"values", obj({})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("event/send")}}), obj({{"op", str("event/receive")}}),
                             obj({{"op", str("event/stopPropagation")}}), obj({{"op", str("variable/set")}}),
                             obj({{"op", str("event/receive")}}), obj({{"op", str("variable/set")}})})},
       {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                      obj({{"declaration", num(1)}, {"configuration", obj({{"event", obj({{"value", arr({num(0)})}})}})}}),
                      obj({{"declaration", num(2)},
                           {"configuration", obj({{"event", obj({{"value", arr({num(0)})}})}})},
                           {"flows", obj({{"out", flow(3)}})}}),
                      obj({{"declaration", num(3)},
                           {"values", obj({{"event", ref(2, "event")}, {"stopImmediate", literal({1}, 1)}})},
                           {"flows", obj({{"out", flow(4)}})}}),
                      obj({{"declaration", num(4)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                           {"values", obj({{"0", literal({1}, 0)}})}}),
                      obj({{"declaration", num(5)},
                           {"configuration", obj({{"event", obj({{"value", arr({num(0)})}})}})},
                           {"flows", obj({{"out", flow(6)}})}}),
                      obj({{"declaration", num(6)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(1)})}})}})},
                           {"values", obj({{"1", literal({1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "stopPropImmediate", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 1);  // Receiver A fired
  EXPECT_EQ(varValue(instance, 1), 0);  // Receiver B skipped
}

// Same graph shape, but stopImmediate=false: per spec, non-immediate stopPropagation only cancels
// *transitive* activations (scene-graph bubbling - see event/onSelect/onHoverIn/onHoverOut), which
// event/send's sibling event/receive nodes are not - so Receiver B still fires. Matches the
// conformance scene's "stopImmediate=false: Receiver B triggered once" sub-test.
TEST(InteractivityEvent, StopPropagationNonImmediateDoesNotSkipSiblingReceivers)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}}), obj({{"signature", str("bool")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}}), obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"events", arr({obj({{"id", str("e")}, {"values", obj({})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("event/send")}}), obj({{"op", str("event/receive")}}),
                             obj({{"op", str("event/stopPropagation")}}), obj({{"op", str("variable/set")}}),
                             obj({{"op", str("event/receive")}}), obj({{"op", str("variable/set")}})})},
       {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                      obj({{"declaration", num(1)}, {"configuration", obj({{"event", obj({{"value", arr({num(0)})}})}})}}),
                      obj({{"declaration", num(2)},
                           {"configuration", obj({{"event", obj({{"value", arr({num(0)})}})}})},
                           {"flows", obj({{"out", flow(3)}})}}),
                      obj({{"declaration", num(3)},
                           {"values", obj({{"event", ref(2, "event")}, {"stopImmediate", literal({0}, 1)}})},
                           {"flows", obj({{"out", flow(4)}})}}),
                      obj({{"declaration", num(4)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                           {"values", obj({{"0", literal({1}, 0)}})}}),
                      obj({{"declaration", num(5)},
                           {"configuration", obj({{"event", obj({{"value", arr({num(0)})}})}})},
                           {"flows", obj({{"out", flow(6)}})}}),
                      obj({{"declaration", num(6)},
                           {"configuration", obj({{"variables", obj({{"value", arr({num(1)})}})}})},
                           {"values", obj({{"1", literal({1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "stopPropNonImmediate", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 1);  // Receiver A fired
  EXPECT_EQ(varValue(instance, 1), 1);  // Receiver B still fired too
}

// debug/log substitutes `{x}` from an evaluated input socket and must not crash regardless of
// what gets logged; `out` still fires afterward.
TEST(InteractivityDebug, LogSubstitutesAndActivatesOut)
{
  tinygltf::Value graphJson =
      obj({{"types", arr({obj({{"signature", str("float")}}), obj({{"signature", str("int")}})})},
           {"variables", arr({obj({{"type", num(1)}, {"value", arr({num(0)})}})})},
           {"declarations",
            arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("debug/log")}}), obj({{"op", str("variable/set")}})})},
           {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                          obj({{"declaration", num(1)},
                               {"configuration", obj({{"severity", obj({{"value", arr({num(0)})}})},
                                                      {"message", obj({{"value", arr({str("value is {x}")})}})}})},
                               {"values", obj({{"x", literal({3.5}, 0)}})},
                               {"flows", obj({{"out", flow(2)}})}}),
                          obj({{"declaration", num(2)},
                               {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                               {"values", obj({{"0", literal({1}, 1)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "debugLog", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_NO_THROW(instance.start());
  EXPECT_EQ(varValue(instance, 0), 1);
}

// Two variable/set nodes flow into each other (node 2 -> node 1 is backward, now accepted per
// InteractivityGraph.AcceptsBackwardFlowReference above), forming a genuine runtime cycle with no
// natural termination. activateFlow() is now real C++ recursion (spec: flow activation is
// call/return, not a work queue - see docs/interactivity.md's design note), so this must be capped
// by call depth rather than hang or stack-overflow the process. The exact final variable value is
// an internal artifact of the depth cap's parity (alternates 1/2 per level) - deliberately not
// pinned to one specific value here, since that would just be testing kMaxCallDepth's tuning, not
// the actual guarantee (it terminates, safely, with a real value from the cycle).
TEST(InteractivityInstance, RecursiveActivationCapsRuntimeCycle)
{
  tinygltf::Value graphJson =
      obj({{"types", arr({obj({{"signature", str("int")}})})},
           {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
           {"declarations",
            arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("variable/set")}}), obj({{"op", str("variable/set")}})})},
           {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                          obj({{"declaration", num(1)},
                               {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                               {"values", obj({{"0", literal({1}, 0)}})},
                               {"flows", obj({{"out", flow(2)}})}}),
                          obj({{"declaration", num(2)},
                               {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                               {"values", obj({{"0", literal({2}, 0)}})},
                               {"flows", obj({{"out", flow(1)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "cycle", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_NO_THROW(instance.start());  // must return, not stack-overflow or hang
  const int32_t finalValue = varValue(instance, 0);
  EXPECT_TRUE(finalValue == 1 || finalValue == 2) << "got " << finalValue << " - cycle should still land on a value the "
                                                  << "loop body actually wrote, not garbage from an unsafe cutoff";
}

// Regression test for the actual bug found by running Khronos's own conformance suite
// (Tests/Interactivity/event/send_and_receive): a graph with two event/onStart nodes, where the
// FIRST one's chain (in ascending node-index order, i.e. JSON order) is two flow-hops deep and the
// SECOND one's chain is a single hop that mutates a variable the first one reads. Under the old
// breadth-first shared-queue model, the first chain's hop 1 ran, then (because hop 2 had only just
// been enqueued, not executed) the second onStart's single-hop chain ran and mutated the variable,
// and only THEN did the first chain's hop 2 run - reading the now-mutated value. Spec requires each
// onStart's entire chain to complete before the next one starts (Specification.adoc's "function
// pointer"/"method call" flow-socket language), so hop 2 must see the variable's value from BEFORE
// the second onStart ever ran.
TEST(InteractivityInstance, OnStartChainsCompleteBeforeNextOnStartBegins)
{
  // Value-socket refs must still point backward (spec 5273, untouched by the flow-ref relaxation
  // above), so the "variable/get var0" pure node has to sit at a lower index than the hop that
  // reads it - node order here is [pure var0 getter, 1st onStart, 1st chain hop1, hop2, 2nd onStart,
  // 2nd chain], not the more narratively-obvious onStart-then-getter order.
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}}),     // 0: var0, mutated by the 2nd onStart
                          obj({{"type", num(0)}, {"value", arr({num(0)})}}),     // 1: var1, captures var0 mid-1st-chain
                          obj({{"type", num(0)}, {"value", arr({num(0)})}})})},  // 2: dummy, just to add a hop
       {"declarations",
        arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("variable/set")}}), obj({{"op", str("variable/get")}})})},
       {"nodes", arr({obj({{"declaration", num(2)}, {"configuration", obj({{"variable", obj({{"value", arr({num(0)})}})}})}}),  // 0: variable/get var0
                      obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(2)}})}}),  // 1: 1st onStart
                      obj({{"declaration", num(1)},  // 2: 1st chain, hop 1 (dummy)
                           {"configuration", obj({{"variables", obj({{"value", arr({num(2)})}})}})},
                           {"values", obj({{"0", literal({1}, 0)}})},
                           {"flows", obj({{"out", flow(3)}})}}),
                      obj({{"declaration", num(1)},  // 3: 1st chain, hop 2 - captures var0's CURRENT value into var1
                           {"configuration", obj({{"variables", obj({{"value", arr({num(1)})}})}})},
                           {"values", obj({{"0", ref(0)}})}}),
                      obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(5)}})}}),  // 4: 2nd onStart
                      obj({{"declaration", num(1)},  // 5: 2nd chain - mutates var0
                           {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                           {"values", obj({{"0", literal({99}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "onStartOrdering", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 99) << "2nd onStart's chain must still have run";
  EXPECT_EQ(varValue(instance, 1), 0) << "1st onStart's chain must have captured var0 BEFORE the 2nd onStart ran";
}

// Same bug class, but for flow/sequence itself (spec, quoted verbatim: "each output flow is
// activated after the previous output flow completes"). Branch "000" (lexicographically first) is
// two hops deep; branch "001" is a single hop that mutates a variable branch "000"'s second hop
// reads. If "000" doesn't fully complete before "001" starts, its second hop sees "001"'s mutation.
TEST(InteractivityInstance, SequenceBranchCompletesBeforeNextBranchBegins)
{
  // Same backward-value-ref constraint as the test above dictates node order: the "variable/get
  // var0" pure node has to precede the hop that reads it.
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}}), obj({{"type", num(0)}, {"value", arr({num(0)})}}),
                          obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/sequence")}}),
                             obj({{"op", str("variable/set")}}), obj({{"op", str("variable/get")}})})},
       {"nodes", arr({obj({{"declaration", num(3)}, {"configuration", obj({{"variable", obj({{"value", arr({num(0)})}})}})}}),  // 0: variable/get var0
                      obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(2)}})}}),  // 1: onStart
                      obj({{"declaration", num(1)}, {"flows", obj({{"000", flow(3)}, {"001", flow(5)}})}}),  // 2: sequence
                      obj({{"declaration", num(2)},  // 3: branch "000", hop 1 (dummy)
                           {"configuration", obj({{"variables", obj({{"value", arr({num(2)})}})}})},
                           {"values", obj({{"0", literal({1}, 0)}})},
                           {"flows", obj({{"out", flow(4)}})}}),
                      obj({{"declaration", num(2)},  // 4: branch "000", hop 2 - captures var0
                           {"configuration", obj({{"variables", obj({{"value", arr({num(1)})}})}})},
                           {"values", obj({{"0", ref(0)}})}}),
                      obj({{"declaration", num(2)},  // 5: branch "001" - mutates var0
                           {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                           {"values", obj({{"0", literal({99}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "sequenceOrdering", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_EQ(varValue(instance, 0), 99) << "branch \"001\" must still have run";
  EXPECT_EQ(varValue(instance, 1), 0) << "branch \"000\" must have captured var0 BEFORE branch \"001\" ran";
}

//--------------------------------------------------------------------------------------------------
// Event-ref identity and the spec's self-referential object-model pointers (found by running
// Khronos's own conformance suite, Tests/Interactivity/event/Event_Refs): every
// event/onStart node in a graph shares ONE occurrence ref (spec: conceptually a single "start"
// event per run, not one per node), likewise every event/onTick node for a given tick, and
// `/extensions/KHR_interactivity/events/{}` / `.../delays/{}` let a graph check whether a ref is a
// real (event/delay-kind) reference via pointer/get's `isValid`.
//--------------------------------------------------------------------------------------------------

TEST(InteractivityEvent, OnStartNodesShareOneEventRef)
{
  tinygltf::Value graphJson = obj({{"declarations", arr({obj({{"op", str("event/onStart")}})})},
                                   {"nodes", arr({obj({{"declaration", num(0)}}), obj({{"declaration", num(0)}})})}});
  auto            graph     = InteractivityGraph::parse(graphJson, "twoOnStart", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  InteractivityValue a = evaluateNodeOutput(instance, 0, "event");
  InteractivityValue b = evaluateNodeOutput(instance, 1, "event");
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(a));
  EXPECT_NE(std::get<InteractivityRef>(a).handle, -1);
  EXPECT_EQ(a, b);
}

TEST(InteractivityEvent, OnTickNodesShareOneEventRefPerTickButDifferAcrossTicks)
{
  tinygltf::Value graphJson = obj({{"declarations", arr({obj({{"op", str("event/onTick")}})})},
                                   {"nodes", arr({obj({{"declaration", num(0)}}), obj({{"declaration", num(0)}})})}});
  auto            graph     = InteractivityGraph::parse(graphJson, "twoOnTick", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.tick(0.1f);
  InteractivityValue a1 = evaluateNodeOutput(instance, 0, "event");
  InteractivityValue b1 = evaluateNodeOutput(instance, 1, "event");
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(a1));
  EXPECT_EQ(a1, b1) << "both onTick nodes must share the same tick's occurrence ref";

  instance.tick(0.1f);
  InteractivityValue a2 = evaluateNodeOutput(instance, 0, "event");
  EXPECT_NE(a1, a2) << "a new tick must mint a new occurrence ref";
}

TEST(InteractivityEvent, PointerGetEventsValidatesARealEventRef)
{
  tinygltf::Value graphJson =
      obj({{"types", arr({obj({{"signature", str("ref")}})})},
           {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("pointer/get")}})})},
           {"nodes", arr({obj({{"declaration", num(0)}}),
                          obj({{"declaration", num(1)},
                               {"configuration", obj({{"pointer", obj({{"value", arr({str("/extensions/KHR_interactivity/events/{eventRef}")})}})},
                                                      {"type", obj({{"value", arr({num(0)})}})}})},
                               {"values", obj({{"eventRef", ref(0, "event")}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "eventRefPointer", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();
  EXPECT_TRUE(std::get<bool>(evaluateNodeOutput(instance, 1, "isValid")));
  InteractivityValue value = evaluateNodeOutput(instance, 1, "value");
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(value));
  EXPECT_EQ(value, evaluateNodeOutput(instance, 0, "event"));
}

TEST(InteractivityEvent, PointerGetEventsRejectsNullRef)
{
  // "eventRef" is left unwired (no `values` entry -> falls back to the socket's declared type, a
  // type-default null ref per spec 5327), so this must resolve to isValid=false, not crash.
  tinygltf::Value graphJson =
      obj({{"types", arr({obj({{"signature", str("ref")}})})},
           {"declarations", arr({obj({{"op", str("pointer/get")}})})},
           {"nodes", arr({obj({{"declaration", num(0)},
                               {"configuration", obj({{"pointer", obj({{"value", arr({str("/extensions/KHR_interactivity/events/{eventRef}")})}})},
                                                      {"type", obj({{"value", arr({num(0)})}})}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "nullEventRefPointer", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  EXPECT_FALSE(std::get<bool>(evaluateNodeOutput(instance, 0, "isValid")));
}

TEST(InteractivityEvent, PointerGetDelaysValidatesOnlyWhilePending)
{
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("float")}}), obj({{"signature", str("ref")}})})},
       {"declarations",
        arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("flow/setDelay")}}), obj({{"op", str("pointer/get")}})})},
       {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", flow(1)}})}}),
                      obj({{"declaration", num(1)}, {"values", obj({{"duration", literal({10.0}, 0)}})}}),
                      obj({{"declaration", num(2)},
                           {"configuration", obj({{"pointer", obj({{"value", arr({str("/extensions/KHR_interactivity/delays/{delayRef}")})}})},
                                                  {"type", obj({{"value", arr({num(1)})}})}})},
                           {"values", obj({{"delayRef", ref(1, "lastDelay")}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "delayRefPointer", 0);
  ASSERT_TRUE(graph.has_value());
  InteractivityGraphInstance instance(*graph);
  instance.start();  // schedules the 10s delay, doesn't fire it yet
  EXPECT_TRUE(std::get<bool>(evaluateNodeOutput(instance, 2, "isValid"))) << "still pending, must validate";

  instance.tick(1.0f);   // first tick() call: establishes m_hasTicked, contributes 0s (NaN-delta convention)
  instance.tick(20.0f);  // timeSinceStart now 20s, well past the 10s duration - the delay fires and is removed
  EXPECT_FALSE(std::get<bool>(evaluateNodeOutput(instance, 2, "isValid"))) << "no longer pending, must invalidate";
}
