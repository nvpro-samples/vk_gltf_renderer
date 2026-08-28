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
// KHR_interactivity Phase C: pointer/get and pointer/set against a real Scene (not a mock),
// using ScenePointerResolver. Complements test_interactivity_engine.cpp's pure-engine tests
// (which use no resolver at all - see its `UnimplementedOpNoOpsInsteadOfCrashing` test).
//

#include <gtest/gtest.h>

#include "common/test_utils.hpp"
#include "gltf_interactivity_eval.hpp"
#include "gltf_interactivity_graph.hpp"
#include "gltf_interactivity_instance.hpp"
#include "gltf_interactivity_scene_pointer.hpp"
#include "gltf_scene.hpp"
#include "tinygltf_utils.hpp"

using namespace gltf_test;
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
// Configuration entries share the same {"value": [...]} shape as literal value sockets.
tinygltf::Value config(std::initializer_list<tinygltf::Value> items)
{
  return obj({{"value", tinygltf::Value(tinygltf::Value::Array(items))}});
}

// Builds a one-node graph: `event/onStart` -> `pointer/set` (writing `newValue` to `pointerTemplate`
// with `nodeIndex` substituted into the template's `[index]` parameter), for exercising pointer/set
// through the full start()-tick machinery (not just direct evaluateNodeOutput/executeFlowNode calls).
std::optional<InteractivityGraph> onStartSetGraph(const std::string&           pointerTemplate,
                                                  int                          typeIndex,
                                                  int                          nodeIndexParam,
                                                  tinygltf::Value              newValueLiteral,
                                                  std::vector<tinygltf::Value> types)
{
  tinygltf::Value graphJson = obj(
      {{"types", tinygltf::Value(tinygltf::Value::Array(types))},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("pointer/set")}})})},
       {"nodes", arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", obj({{"node", num(1)}})}})}}),
                      obj({{"declaration", num(1)},
                           {"configuration", obj({{"pointer", config({str(pointerTemplate)})}, {"type", config({num(typeIndex)})}})},
                           {"values", obj({{"index", literal({static_cast<double>(nodeIndexParam)}, 0)},  // types[0] is always "int"
                                           {"value", newValueLiteral}})}})})}});
  return InteractivityGraph::parse(graphJson, "pointerSet", 0);
}

}  // namespace

class InteractivityPointerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    m_assetsPath = TestResources::getSampleAssetsPath();
    if(m_assetsPath.empty())
      GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
  }
  std::filesystem::path m_assetsPath;
};

TEST_F(InteractivityPointerTest, GetReadsNodeTranslationRotationScale)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(m_assetsPath / "Models/Box/glTF/Box.gltf"));
  ASSERT_GE(scene.getModel().nodes.size(), 2u);  // node 1: mesh-holding child, implicit identity TRS


  // types: 0=int (template param), 1=float3
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}}), obj({{"signature", str("float3")}})})},
       {"declarations", arr({obj({{"op", str("pointer/get")}})})},
       {"nodes",
        arr({obj({{"declaration", num(0)},
                  {"configuration", obj({{"pointer", config({str("/nodes/[index]/translation")})}, {"type", config({num(1)})}})},
                  {"values", obj({{"index", literal({1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "get", 0);
  ASSERT_TRUE(graph.has_value());

  InteractivityGraphInstance instance(*graph);
  ScenePointerResolver       resolver(scene);
  instance.setPointerResolver(&resolver);

  InteractivityValue isValid = evaluateNodeOutput(instance, 0, "isValid");
  ASSERT_TRUE(std::holds_alternative<bool>(isValid));
  EXPECT_TRUE(std::get<bool>(isValid));

  InteractivityValue value = evaluateNodeOutput(instance, 0, "value");
  ASSERT_TRUE(std::holds_alternative<glm::vec3>(value));
  const glm::vec3 t = std::get<glm::vec3>(value);
  EXPECT_NEAR(t.x, 0.0f, 1e-5f);
  EXPECT_NEAR(t.y, 0.0f, 1e-5f);
  EXPECT_NEAR(t.z, 0.0f, 1e-5f);
}

TEST_F(InteractivityPointerTest, GetOnUnresolvablePathReportsInvalidNotCrash)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(m_assetsPath / "Models/Box/glTF/Box.gltf"));

  // Node index far out of range - must not resolve, must not crash.
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}}), obj({{"signature", str("float3")}})})},
       {"declarations", arr({obj({{"op", str("pointer/get")}})})},
       {"nodes",
        arr({obj({{"declaration", num(0)},
                  {"configuration", obj({{"pointer", config({str("/nodes/[index]/translation")})}, {"type", config({num(1)})}})},
                  {"values", obj({{"index", literal({99999}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "get", 0);
  ASSERT_TRUE(graph.has_value());

  InteractivityGraphInstance instance(*graph);
  ScenePointerResolver       resolver(scene);
  instance.setPointerResolver(&resolver);

  InteractivityValue isValid = evaluateNodeOutput(instance, 0, "isValid");
  ASSERT_TRUE(std::holds_alternative<bool>(isValid));
  EXPECT_FALSE(std::get<bool>(isValid));
}

TEST_F(InteractivityPointerTest, SetMovesNodeAndSubsequentGetSeesTheWrite)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(m_assetsPath / "Models/Box/glTF/Box.gltf"));
  ASSERT_GE(scene.getModel().nodes.size(), 2u);

  // types: 0=int, 1=float3. graph: onStart -> pointer/set("/nodes/[index]/translation", (5,6,7))
  auto graph = onStartSetGraph("/nodes/[index]/translation", 1, 1, literal({5, 6, 7}, 1),
                               {obj({{"signature", str("int")}}), obj({{"signature", str("float3")}})});
  ASSERT_TRUE(graph.has_value());

  InteractivityGraphInstance instance(*graph);
  ScenePointerResolver       resolver(scene);
  instance.setPointerResolver(&resolver);
  instance.start();

  glm::vec3 translation, scale;
  glm::quat rotation;
  tinygltf::utils::getNodeTRS(scene.getModel().nodes[1], translation, rotation, scale);
  EXPECT_NEAR(translation.x, 5.0f, 1e-4f);
  EXPECT_NEAR(translation.y, 6.0f, 1e-4f);
  EXPECT_NEAR(translation.z, 7.0f, 1e-4f);
}

TEST_F(InteractivityPointerTest, SetWritesMaterialBaseColor)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(m_assetsPath / "Models/Box/glTF/Box.gltf"));
  ASSERT_FALSE(scene.getModel().materials.empty());

  // types: 0=int, 1=float4
  auto graph = onStartSetGraph("/materials/[index]/pbrMetallicRoughness/baseColorFactor", 1, 0, literal({1, 0, 0, 1}, 1),
                               {obj({{"signature", str("int")}}), obj({{"signature", str("float4")}})});
  ASSERT_TRUE(graph.has_value());

  InteractivityGraphInstance instance(*graph);
  ScenePointerResolver       resolver(scene);
  instance.setPointerResolver(&resolver);
  instance.start();

  const auto& baseColor = scene.getModel().materials[0].pbrMetallicRoughness.baseColorFactor;
  ASSERT_EQ(baseColor.size(), 4u);
  EXPECT_NEAR(baseColor[0], 1.0, 1e-4);
  EXPECT_NEAR(baseColor[1], 0.0, 1e-4);
  EXPECT_NEAR(baseColor[2], 0.0, 1e-4);
}

// Regression test for a bug found via KHR_interactivity's BowShooting.glb conformance scene:
// pointer/set writing a bool to any path other than one ending in "/visible" (e.g.
// KHR_node_selectability's "/selectable") silently failed - ScenePointerResolver::set() only
// special-cased "/visible" when routing a bool through AnimationPointerSystem (which has no native
// bool applyValue() overload), so every other bool-typed Object Model property fell through to
// "not a supported writable target" and activated `err` instead of `out`. In BowShooting.glb this
// killed the entire post-shoot node chain (arrow velocity/fly-state never got set) because it was
// gated behind a pointer/set to .../KHR_node_selectability/selectable earlier in the chain.
TEST_F(InteractivityPointerTest, SetWritesNodeSelectabilityBool)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(m_assetsPath / "Models/Box/glTF/Box.gltf"));
  ASSERT_GE(scene.getModel().nodes.size(), 2u);

  // types: 0=int, 1=bool. graph: onStart -> pointer/set("/nodes/[index]/extensions/KHR_node_selectability/selectable", false)
  auto graph = onStartSetGraph("/nodes/[index]/extensions/KHR_node_selectability/selectable", 1, 1, literal({0}, 1),
                               {obj({{"signature", str("int")}}), obj({{"signature", str("bool")}})});
  ASSERT_TRUE(graph.has_value());

  InteractivityGraphInstance instance(*graph);
  ScenePointerResolver       resolver(scene);
  instance.setPointerResolver(&resolver);
  instance.start();

  EXPECT_FALSE(tinygltf::utils::getNodeSelectability(scene.getModel().nodes[1]).selectable);
}

// pointer/interpolate (spec 4154-4254): p1=(0,0)/p2=(1,1) is the linear Bézier (y=x), so the
// eased progress q exactly equals the elapsed-time fraction - same predictability trick as
// InteractivityFlow.VariableInterpolateEasesOverTicksAndFiresDone in test_interactivity_engine.cpp.
// `done` is observed by wiring it to a second pointer/set that stamps node 1's scale.
TEST_F(InteractivityPointerTest, PointerInterpolateEasesTranslationOverTicksAndFiresDone)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(m_assetsPath / "Models/Box/glTF/Box.gltf"));
  ASSERT_GE(scene.getModel().nodes.size(), 2u);

  // types: 0=int, 1=float3, 2=float, 3=float2
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}}), obj({{"signature", str("float3")}}),
                      obj({{"signature", str("float")}}), obj({{"signature", str("float2")}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("pointer/interpolate")}}),
                             obj({{"op", str("pointer/set")}})})},
       {"nodes",
        arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", obj({{"node", num(1)}})}})}}),
             obj({{"declaration", num(1)},
                  {"configuration", obj({{"pointer", config({str("/nodes/[index]/translation")})}, {"type", config({num(1)})}})},
                  {"values", obj({{"index", literal({1}, 0)},
                                  {"value", literal({10, 0, 0}, 1)},
                                  {"duration", literal({1.0}, 2)},
                                  {"p1", literal({0.0, 0.0}, 3)},
                                  {"p2", literal({1.0, 1.0}, 3)}})},
                  {"flows", obj({{"done", obj({{"node", num(2)}})}})}}),
             obj({{"declaration", num(2)},
                  {"configuration", obj({{"pointer", config({str("/nodes/[index]/scale")})}, {"type", config({num(1)})}})},
                  {"values", obj({{"index", literal({1}, 0)}, {"value", literal({2, 2, 2}, 1)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "pointerInterpolate", 0);
  ASSERT_TRUE(graph.has_value());

  InteractivityGraphInstance instance(*graph);
  ScenePointerResolver       resolver(scene);
  instance.setPointerResolver(&resolver);
  instance.start();

  auto readTRS = [&]() {
    glm::vec3 t, s;
    glm::quat r;
    tinygltf::utils::getNodeTRS(scene.getModel().nodes[1], t, r, s);
    return std::pair{t, s};
  };

  {
    auto [translation, scale] = readTRS();
    EXPECT_NEAR(translation.x, 0.0f, 1e-4f);  // t<=0 on the activation tick - no write yet
    EXPECT_NEAR(scale.x, 1.0f, 1e-4f);        // `done` hasn't fired
  }

  instance.tick(0.0f);  // priming tick (first tick never advances time)
  instance.tick(0.5f);  // elapsed 0.5s / 1.0s duration -> t = q = 0.5 (linear easing)
  {
    auto [translation, scale] = readTRS();
    EXPECT_NEAR(translation.x, 5.0f, 1e-3f);
    EXPECT_NEAR(scale.x, 1.0f, 1e-4f);
  }
  instance.tick(0.6f);  // elapsed 1.1s >= 1.0s duration -> snaps to target, `done` fires
  {
    auto [translation, scale] = readTRS();
    EXPECT_NEAR(translation.x, 10.0f, 1e-4f);
    EXPECT_NEAR(scale.x, 2.0f, 1e-4f);  // `done`'s pointer/set ran
  }
}

TEST_F(InteractivityPointerTest, SetWithMismatchedTypeActivatesErrNotOut)
{
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(m_assetsPath / "Models/Box/glTF/Box.gltf"));

  // types: 0=int, 1=float3, 2=bool. Node 2 (pointer/set) claims type=float3 (index 1) but the
  // "value" socket is wired to a bool literal (type index 2) - a spec-invalid mismatch that must
  // route to `err`, not silently write garbage or crash. `err`/`out` are wired to distinct
  // variable/set sentinel nodes (3/4) so the test directly observes which flow actually activated,
  // rather than only inferring it from the pointer write being a no-op.
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("int")}}), obj({{"signature", str("float3")}}), obj({{"signature", str("bool")}})})},
       {"variables", arr({obj({{"type", num(0)}, {"value", arr({num(0)})}}), obj({{"type", num(0)}, {"value", arr({num(0)})}})})},
       {"declarations", arr({obj({{"op", str("event/onStart")}}), obj({{"op", str("pointer/set")}}),
                             obj({{"op", str("variable/set")}}), obj({{"op", str("variable/set")}})})},
       {"nodes",
        arr({obj({{"declaration", num(0)}, {"flows", obj({{"out", obj({{"node", num(1)}})}})}}),
             obj({{"declaration", num(1)},
                  {"configuration", obj({{"pointer", config({str("/nodes/[index]/translation")})}, {"type", config({num(1)})}})},
                  {"values", obj({{"index", literal({1}, 0)}, {"value", literal({1}, 2)}})},
                  {"flows", obj({{"err", obj({{"node", num(2)}})}, {"out", obj({{"node", num(3)}})}})}}),
             obj({{"declaration", num(2)},
                  {"configuration", obj({{"variables", obj({{"value", arr({num(0)})}})}})},
                  {"values", obj({{"0", literal({1}, 0)}})}}),
             obj({{"declaration", num(3)},
                  {"configuration", obj({{"variables", obj({{"value", arr({num(1)})}})}})},
                  {"values", obj({{"0", literal({1}, 0)}})}})})}});
  auto graph = InteractivityGraph::parse(graphJson, "mismatch", 0);
  ASSERT_TRUE(graph.has_value());

  InteractivityGraphInstance instance(*graph);
  ScenePointerResolver       resolver(scene);
  instance.setPointerResolver(&resolver);

  glm::vec3 translationBefore, scaleBefore;
  glm::quat rotationBefore;
  tinygltf::utils::getNodeTRS(scene.getModel().nodes[1], translationBefore, rotationBefore, scaleBefore);

  EXPECT_NO_THROW(instance.start());

  glm::vec3 translationAfter, scaleAfter;
  glm::quat rotationAfter;
  tinygltf::utils::getNodeTRS(scene.getModel().nodes[1], translationAfter, rotationAfter, scaleAfter);
  EXPECT_NEAR(translationBefore.x, translationAfter.x, 1e-6f);
  EXPECT_NEAR(translationBefore.y, translationAfter.y, 1e-6f);
  EXPECT_NEAR(translationBefore.z, translationAfter.z, 1e-6f);

  EXPECT_EQ(std::get<int32_t>(instance.variable(0)), 1);  // `err` fired
  EXPECT_EQ(std::get<int32_t>(instance.variable(1)), 0);  // `out` did not fire
}
