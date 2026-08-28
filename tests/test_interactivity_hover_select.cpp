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
// KHR_interactivity Phase E: event/onSelect (KHR_node_selectability) and event/onHoverIn/
// event/onHoverOut (KHR_node_hoverability) - ancestor-chain bubbling via Scene::notifyNodeSelected/
// notifyNodeHoverChanged, exercised against a real Scene with a hand-built node hierarchy (Box.gltf
// as a base, extra nodes added via SceneEditor for full control over parent/child/sibling shape).
//

#include <gtest/gtest.h>

#include "common/test_utils.hpp"
#include "gltf_interactivity_eval.hpp"
#include "gltf_interactivity_graph.hpp"
#include "gltf_interactivity_instance.hpp"
#include "gltf_scene.hpp"
#include "gltf_scene_editor.hpp"

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
// Configuration entries share the {"value": [...]} shape node/graph literal sockets use.
tinygltf::Value config(std::initializer_list<tinygltf::Value> items)
{
  return obj({{"value", tinygltf::Value(tinygltf::Value::Array(items))}});
}
// event/onSelect or event/onHoverIn/onHoverOut declaration bound to a specific glTF node.
tinygltf::Value handlerDeclaration(const std::string& op, const std::string& extension)
{
  return obj({{"op", str(op)}, {"extension", str(extension)}});
}
tinygltf::Value handlerNode(int declarationIndex, int boundGltfNodeIndex)
{
  return obj({{"declaration", num(declarationIndex)}, {"configuration", obj({{"nodeIndex", config({num(boundGltfNodeIndex)})}})}});
}

InteractivityValue evalOut(InteractivityGraphInstance& instance, int nodeIndex, const std::string& socket)
{
  return evaluateNodeOutput(instance, nodeIndex, socket);
}

// Scene::notifyNodeSelected/notifyNodeHoverChanged operate on the Scene's OWN internal graph
// instance (populated by parseInteractivityGraphs(), keyed off model.extensions), never on a
// caller-constructed InteractivityGraphInstance - so tests must inject the graph through the
// model and re-parse, then fetch the instance the Scene itself owns, rather than constructing one
// standalone (which would silently never be touched by any notify*() call).
InteractivityGraphInstance* loadGraph(nvvkgltf::Scene& scene, tinygltf::Value graphJson)
{
  scene.getModel().extensions[KHR_INTERACTIVITY_EXTENSION_NAME] = obj({{"graphs", arr({std::move(graphJson)})}});
  scene.parseInteractivityGraphs();
  return scene.getInteractivityInstance(scene.getDefaultInteractivityGraph());
}

}  // namespace

class InteractivityHoverSelectTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    m_assetsPath = TestResources::getSampleAssetsPath();
    if(m_assetsPath.empty())
      GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;
    ASSERT_TRUE(m_scene.load(m_assetsPath / "Models/Box/glTF/Box.gltf"));

    // Build root -> {childA -> grandchild, childB} on top of Box's own 2 nodes, for full control
    // over the hierarchy shape independent of Box.gltf's own structure.
    root       = m_scene.editor().addNode("root");
    childA     = m_scene.editor().addNode("childA", root);
    childB     = m_scene.editor().addNode("childB", root);
    grandchild = m_scene.editor().addNode("grandchild", childA);
  }

  std::filesystem::path m_assetsPath;
  nvvkgltf::Scene       m_scene;
  int                   root = -1, childA = -1, childB = -1, grandchild = -1;
};

TEST_F(InteractivityHoverSelectTest, SelectBubblesToUnboundAncestorHandler)
{
  // event/onSelect bound to `root` only; selecting `grandchild` (no handler of its own) must still
  // fire root's handler via bubbling.
  tinygltf::Value graphJson = obj({{"declarations", arr({handlerDeclaration("event/onSelect", "KHR_node_selectability")})},
                                   {"nodes", arr({handlerNode(0, root)})}});
  InteractivityGraphInstance* instance = loadGraph(m_scene, graphJson);
  ASSERT_NE(instance, nullptr);

  // Before any selection: spec-defined defaults, not monostate.
  InteractivityValue beforeSelected = evalOut(*instance, 0, "selectedNode");
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(beforeSelected));
  EXPECT_EQ(std::get<InteractivityRef>(beforeSelected).handle, -1);
  InteractivityValue beforeController = evalOut(*instance, 0, "controllerIndex");
  ASSERT_TRUE(std::holds_alternative<int32_t>(beforeController));
  EXPECT_EQ(std::get<int32_t>(beforeController), -1);

  m_scene.notifyNodeSelected(grandchild, glm::vec3(1, 2, 3), glm::vec3(4, 5, 6));

  InteractivityValue selectedNode = evalOut(*instance, 0, "selectedNode");
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(selectedNode));
  EXPECT_EQ(std::get<InteractivityRef>(selectedNode).handle, grandchild);
  InteractivityValue controllerIndex = evalOut(*instance, 0, "controllerIndex");
  ASSERT_TRUE(std::holds_alternative<int32_t>(controllerIndex));
  EXPECT_EQ(std::get<int32_t>(controllerIndex), 0);
  InteractivityValue selectionPoint = evalOut(*instance, 0, "selectionPoint");
  ASSERT_TRUE(std::holds_alternative<glm::vec3>(selectionPoint));
  EXPECT_FLOAT_EQ(std::get<glm::vec3>(selectionPoint).x, 1.0f);
  InteractivityValue eventRef = evalOut(*instance, 0, "event");
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(eventRef));
  EXPECT_NE(std::get<InteractivityRef>(eventRef).handle, -1);  // a real occurrence was minted
}

TEST_F(InteractivityHoverSelectTest, SelectFiresEveryBoundAncestorPropagationContinues)
{
  // event/onSelect bound to BOTH `grandchild` itself and `root` - spec: propagation continues past
  // the first match, so selecting grandchild must fire both, not just the nearest one.
  tinygltf::Value graphJson = obj({{"declarations", arr({handlerDeclaration("event/onSelect", "KHR_node_selectability"),
                                                         handlerDeclaration("event/onSelect", "KHR_node_selectability")})},
                                   {"nodes", arr({handlerNode(0, grandchild), handlerNode(1, root)})}});
  InteractivityGraphInstance* instance = loadGraph(m_scene, graphJson);
  ASSERT_NE(instance, nullptr);

  m_scene.notifyNodeSelected(grandchild, glm::vec3(0), glm::vec3(0));

  InteractivityValue leafSelected = evalOut(*instance, 0, "selectedNode");
  InteractivityValue rootSelected = evalOut(*instance, 1, "selectedNode");
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(leafSelected));
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(rootSelected));
  EXPECT_EQ(std::get<InteractivityRef>(leafSelected).handle, grandchild);
  EXPECT_EQ(std::get<InteractivityRef>(rootSelected).handle, grandchild);
}

TEST_F(InteractivityHoverSelectTest, SelectStopPropagationCancelsFurtherBubbling)
{
  // grandchild's onSelect handler calls event/stopPropagation with its own occurrence ref (read via
  // its "event" output socket). Per spec 4492-4530 this cancels further *transitive* (scene-graph
  // bubble) activations regardless of stopImmediate - so childA's and root's onSelect handlers
  // (further up the bubble) must never fire, even though ordinary propagation would otherwise reach
  // them (see SelectFiresEveryBoundAncestorPropagationContinues).
  tinygltf::Value graphJson = obj(
      {{"types", arr({obj({{"signature", str("bool")}})})},
       {"declarations",
        arr({handlerDeclaration("event/onSelect", "KHR_node_selectability"), handlerDeclaration("event/onSelect", "KHR_node_selectability"),
             handlerDeclaration("event/onSelect", "KHR_node_selectability"), obj({{"op", str("event/stopPropagation")}})})},
       {"nodes",
        arr({obj({{"declaration", num(0)},
                  {"configuration", obj({{"nodeIndex", config({num(grandchild)})}})},
                  {"flows", obj({{"out", obj({{"node", num(3)}})}})}}),
             handlerNode(1, childA), handlerNode(2, root),
             obj({{"declaration", num(3)},
                  {"values", obj({{"event", obj({{"node", num(0)}, {"socket", str("event")}})},
                                  {"stopImmediate", obj({{"type", num(0)}, {"value", arr({tinygltf::Value(false)})}})}})}})})}});
  InteractivityGraphInstance* instance = loadGraph(m_scene, graphJson);
  ASSERT_NE(instance, nullptr);

  m_scene.notifyNodeSelected(grandchild, glm::vec3(0), glm::vec3(0));

  EXPECT_EQ(std::get<InteractivityRef>(evalOut(*instance, 0, "selectedNode")).handle, grandchild);  // grandchild's own handler fired
  EXPECT_EQ(std::get<InteractivityRef>(evalOut(*instance, 1, "selectedNode")).handle, -1);  // childA never fired
  EXPECT_EQ(std::get<InteractivityRef>(evalOut(*instance, 2, "selectedNode")).handle, -1);  // root never fired
}

TEST_F(InteractivityHoverSelectTest, HoverInFiresAllAncestorsOnFirstHover)
{
  // No previous hover (-1): hovering `grandchild` must fire onHoverIn bound anywhere on its full
  // ancestor chain (grandchild, childA, root), all the way to the root.
  tinygltf::Value graphJson = obj({{"declarations", arr({handlerDeclaration("event/onHoverIn", "KHR_node_hoverability"),
                                                         handlerDeclaration("event/onHoverIn", "KHR_node_hoverability")})},
                                   {"nodes", arr({handlerNode(0, childA), handlerNode(1, root)})}});
  InteractivityGraphInstance* instance = loadGraph(m_scene, graphJson);
  ASSERT_NE(instance, nullptr);

  m_scene.notifyNodeHoverChanged(-1, grandchild);

  InteractivityValue childAHovered = evalOut(*instance, 0, "hoveredNode");
  InteractivityValue rootHovered   = evalOut(*instance, 1, "hoveredNode");
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(childAHovered));
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(rootHovered));
  EXPECT_EQ(std::get<InteractivityRef>(childAHovered).handle, grandchild);
  EXPECT_EQ(std::get<InteractivityRef>(rootHovered).handle, grandchild);
}

TEST_F(InteractivityHoverSelectTest, HoverTransitionBetweenSiblingsDoesNotRefireSharedParent)
{
  // root has onHoverIn+onHoverOut bound. Hover childA (child of root) first - root fires once.
  // Then move the hover to childB (childA's sibling, same parent root) - root's handlers must NOT
  // refire, since the hover never left root's subtree.
  tinygltf::Value graphJson = obj({{"declarations", arr({handlerDeclaration("event/onHoverIn", "KHR_node_hoverability"),
                                                         handlerDeclaration("event/onHoverOut", "KHR_node_hoverability")})},
                                   {"nodes", arr({handlerNode(0, root), handlerNode(1, root)})}});
  InteractivityGraphInstance* instance = loadGraph(m_scene, graphJson);
  ASSERT_NE(instance, nullptr);

  m_scene.notifyNodeHoverChanged(-1, childA);
  InteractivityValue firstHoverIn = evalOut(*instance, 0, "hoveredNode");
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(firstHoverIn));
  EXPECT_EQ(std::get<InteractivityRef>(firstHoverIn).handle, childA);
  InteractivityValue eventBeforeSiblingMove = evalOut(*instance, 0, "event");

  m_scene.notifyNodeHoverChanged(childA, childB);

  // root's onHoverIn must NOT have refired (still shows childA, not childB) - same event ref.
  InteractivityValue hoverInAfter = evalOut(*instance, 0, "hoveredNode");
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(hoverInAfter));
  EXPECT_EQ(std::get<InteractivityRef>(hoverInAfter).handle, childA);
  EXPECT_EQ(evalOut(*instance, 0, "event"), eventBeforeSiblingMove);

  // root's onHoverOut must NOT have fired at all (still at its pre-occurrence default: null ref).
  InteractivityValue hoverOut = evalOut(*instance, 1, "hoveredNode");
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(hoverOut));
  EXPECT_EQ(std::get<InteractivityRef>(hoverOut).handle, -1);
}

TEST_F(InteractivityHoverSelectTest, HoverIntoOwnChildDoesNotFireParentHoverOut)
{
  // Hover childA, then move into grandchild (childA's own child). childA's onHoverOut must not
  // fire (the hover never left childA's subtree); grandchild's onHoverIn (if bound) does fire.
  tinygltf::Value graphJson = obj({{"declarations", arr({handlerDeclaration("event/onHoverOut", "KHR_node_hoverability"),
                                                         handlerDeclaration("event/onHoverIn", "KHR_node_hoverability")})},
                                   {"nodes", arr({handlerNode(0, childA), handlerNode(1, grandchild)})}});
  InteractivityGraphInstance* instance = loadGraph(m_scene, graphJson);
  ASSERT_NE(instance, nullptr);

  m_scene.notifyNodeHoverChanged(-1, childA);
  m_scene.notifyNodeHoverChanged(childA, grandchild);

  InteractivityValue childAHoverOut = evalOut(*instance, 0, "hoveredNode");
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(childAHoverOut));
  EXPECT_EQ(std::get<InteractivityRef>(childAHoverOut).handle, -1);  // never fired

  InteractivityValue grandchildHoverIn = evalOut(*instance, 1, "hoveredNode");
  ASSERT_TRUE(std::holds_alternative<InteractivityRef>(grandchildHoverIn));
  EXPECT_EQ(std::get<InteractivityRef>(grandchildHoverIn).handle, grandchild);
}
