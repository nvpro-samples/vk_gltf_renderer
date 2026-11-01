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

// The live scene's answer to "which light is the sky's sun".
//
// The scene resolves it when its lights are rebuilt and when SetSkySunCommand moves the marker, and
// the frame only reads it. These tests pin both ends: every way the marker changes is followed,
// and a marker on a node without a light never wins.

#include <gtest/gtest.h>

#include "gltf_environment_sky.hpp"
#include "gltf_scene.hpp"
#include "undo_redo.hpp"

namespace {

// Node 0: marker, no light (stale). Nodes 1 and 2: directional lights 0 and 1.
tinygltf::Model makeModel(int markedLightNode)
{
  tinygltf::Model model;
  model.asset.version = "2.0";
  model.extensionsUsed.push_back("KHR_lights_punctual");
  for(int i = 0; i < 2; ++i)
  {
    tinygltf::Light light;
    light.type = "directional";
    model.lights.push_back(light);
  }

  model.nodes.resize(3);
  gltf_environment_sky::setSkySunMarker(model.nodes[0], gltf_environment_sky::kSkySunOwnerScene);
  model.nodes[1].light = 0;
  model.nodes[2].light = 1;
  if(markedLightNode >= 0)
    gltf_environment_sky::setSkySunMarker(model.nodes[markedLightNode], gltf_environment_sky::kSkySunOwnerScene);

  tinygltf::Scene scene;
  scene.nodes = {0, 1, 2};
  model.scenes.push_back(scene);
  model.defaultScene = 0;
  return model;
}

// Node index of the scene's resolved sun, or -1.
int sunNode(const nvvkgltf::Scene& scene)
{
  const int i = scene.getSkySunLight();
  return i < 0 ? -1 : scene.getRenderLights()[i].nodeID;
}

}  // namespace

TEST(SkySunScene, LoadResolvesTheMarkedLightAndIgnoresAStaleMarker)
{
  nvvkgltf::Scene scene;
  scene.takeModel(makeModel(2));
  EXPECT_EQ(sunNode(scene), 2);
}

TEST(SkySunScene, NoMarkedLightMeansNoSun)
{
  nvvkgltf::Scene scene;
  scene.takeModel(makeModel(-1));
  EXPECT_EQ(sunNode(scene), -1);
}

TEST(SkySunScene, SetSkySunCommandIsFollowedThroughUndoAndRedo)
{
  nvvkgltf::Scene scene;
  scene.takeModel(makeModel(1));
  ASSERT_EQ(sunNode(scene), 1);

  SetSkySunCommand pick(scene, 1, 2);
  pick.execute();
  EXPECT_EQ(sunNode(scene), 2);
  pick.undo();
  EXPECT_EQ(sunNode(scene), 1);
  pick.execute();
  EXPECT_EQ(sunNode(scene), 2);

  SetSkySunCommand clear(scene, 2, -1);
  clear.execute();
  EXPECT_EQ(sunNode(scene), -1);
  clear.undo();
  EXPECT_EQ(sunNode(scene), 2);
}

TEST(SkySunScene, RebuildingTheLightsKeepsTheSun)
{
  nvvkgltf::Scene scene;
  scene.takeModel(makeModel(2));
  scene.rebuildRenderNodes();
  EXPECT_EQ(sunNode(scene), 2);
}
