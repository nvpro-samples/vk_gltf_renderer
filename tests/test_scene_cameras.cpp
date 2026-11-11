/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include "gltf_camera_utils.hpp"
#include "gltf_scene.hpp"
#include "gltf_scene_editor.hpp"
#include "undo_redo.hpp"
#include "common/test_utils.hpp"
#include <filesystem>
#include <fstream>

using namespace gltf_test;

// Scene cameras: the viewport view saved with a scene, and glTF cameras created from a view.

namespace {
nvvkgltf::RenderCamera makeView(glm::dvec3 eye, glm::dvec3 center)
{
  nvvkgltf::RenderCamera view;
  view.type   = nvvkgltf::RenderCamera::CameraType::ePerspective;
  view.eye    = eye;
  view.center = center;
  view.up     = {0.0, 1.0, 0.0};
  view.yfov   = 0.8;
  view.znear  = 0.05;
  view.zfar   = 500.0;
  return view;
}
}  // namespace

class SceneCamerasTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    tempDir = std::filesystem::temp_directory_path() / "gltf_scene_cameras_test";
    std::filesystem::create_directories(tempDir);
  }
  void TearDown() override { std::filesystem::remove_all(tempDir); }

  std::filesystem::path tempDir;
};

TEST_F(SceneCamerasTest, SavedViewRoundTripsWithoutTouchingCameras)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Cameras/glTF/Cameras.gltf"));
  const size_t nodeCount   = scene.getModel().nodes.size();
  const size_t cameraCount = scene.getModel().cameras.size();
  EXPECT_FALSE(scene.getSavedView().has_value());

  const nvvkgltf::RenderCamera view = makeView({3.0, 2.0, 5.0}, {0.5, 0.0, -1.0});
  scene.setSavedView(view);
  // Same camera:: keys as a camera node, on the scene's extras.
  const tinygltf::Value& extras = scene.getModel().scenes[0].extras;
  EXPECT_TRUE(extras.Has("camera::eye"));
  EXPECT_TRUE(extras.Has("camera::center"));
  EXPECT_TRUE(extras.Has("camera::up"));
  EXPECT_TRUE(extras.Has("camera::yfov"));
  ASSERT_TRUE(scene.save(tempDir / "cameras.gltf"));

  nvvkgltf::Scene loaded;
  ASSERT_TRUE(loaded.load(tempDir / "cameras.gltf"));
  // The saved view is not a glTF camera: no node, no camera added, the authored ones kept as they were.
  EXPECT_EQ(loaded.getModel().nodes.size(), nodeCount);
  EXPECT_EQ(loaded.getModel().cameras.size(), cameraCount);
  EXPECT_EQ(loaded.getRenderCameras().size(), scene.getRenderCameras().size());

  const auto saved = loaded.getSavedView();
  ASSERT_TRUE(saved.has_value());
  EXPECT_NEAR(saved->eye.x, 3.0, 1e-9);
  EXPECT_NEAR(saved->eye.y, 2.0, 1e-9);
  EXPECT_NEAR(saved->center.z, -1.0, 1e-9);
  EXPECT_NEAR(saved->yfov, 0.8, 1e-9);
  EXPECT_NEAR(saved->zfar, 500.0, 1e-9);
}

TEST_F(SceneCamerasTest, SavedViewKeepsOrthographicLens)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  nvvkgltf::RenderCamera view = makeView({0.0, 0.0, 5.0}, {0.0, 0.0, 0.0});
  view.type                   = nvvkgltf::RenderCamera::CameraType::eOrthographic;
  view.xmag                   = 4.0;
  view.ymag                   = 3.0;
  scene.setSavedView(view);
  ASSERT_TRUE(scene.save(tempDir / "box.gltf"));

  nvvkgltf::Scene loaded;
  ASSERT_TRUE(loaded.load(tempDir / "box.gltf"));
  const auto saved = loaded.getSavedView();
  ASSERT_TRUE(saved.has_value());
  EXPECT_EQ(saved->type, nvvkgltf::RenderCamera::CameraType::eOrthographic);
  EXPECT_NEAR(saved->xmag, 4.0, 1e-9);
  EXPECT_NEAR(saved->ymag, 3.0, 1e-9);
}

TEST_F(SceneCamerasTest, AddCameraNodeLooksThroughTheView)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  const int before = static_cast<int>(scene.getModel().cameras.size());

  const nvvkgltf::RenderCamera view = makeView({4.0, 3.0, 6.0}, {0.0, 0.0, 0.0});
  const int                    node = scene.editor().addCameraNode("Camera X", view);
  ASSERT_GE(node, 0);

  const tinygltf::Model& model = scene.getModel();
  ASSERT_EQ(static_cast<int>(model.cameras.size()), before + 1);
  EXPECT_EQ(model.nodes[node].camera, before);
  EXPECT_EQ(model.cameras[before].name, "Camera X");
  EXPECT_NEAR(model.cameras[before].perspective.yfov, 0.8, 1e-9);
  EXPECT_EQ(scene.getCameraNode(before), node);

  const nvvkgltf::RenderCamera* added   = nullptr;
  const auto&                   rebuilt = scene.getRenderCameras(true);
  for(size_t i = 0; i < rebuilt.size(); i++)
  {
    if(rebuilt[i].cameraIndex == before)
      added = &rebuilt[i];
  }
  ASSERT_NE(added, nullptr);
  EXPECT_NEAR(added->eye.x, 4.0, 1e-6);

  // A save and reload keeps the camera: one more than before, not duplicated.
  ASSERT_TRUE(scene.save(tempDir / "box_cam.gltf"));
  nvvkgltf::Scene loaded;
  ASSERT_TRUE(loaded.load(tempDir / "box_cam.gltf"));
  ASSERT_EQ(static_cast<int>(loaded.getRenderCameras().size()), before + 1);
  bool        found      = false;
  const auto& loadedCams = loaded.getRenderCameras();
  for(size_t i = 0; i < loadedCams.size(); i++)
  {
    if(std::abs(loadedCams[i].eye.y - 3.0) < 1e-5 && std::abs(loadedCams[i].eye.x - 4.0) < 1e-5)
      found = true;
  }
  EXPECT_TRUE(found);
}

TEST_F(SceneCamerasTest, AddCameraUnderTransformedParentKeepsWorldPose)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));

  const int parent = scene.editor().addNode("Rig");
  ASSERT_GE(parent, 0);
  scene.editor().setNodeTRS(parent, glm::vec3(10.0f, 0.0f, -2.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3(1.0f));

  const int node = scene.editor().addCameraNode("Camera 0", makeView({4.0, 3.0, 6.0}, {0.0, 0.0, 0.0}), parent);
  ASSERT_GE(node, 0);

  // Local translation is the world eye seen from the parent.
  const auto& t = scene.getModel().nodes[node].translation;
  EXPECT_NEAR(t[0], 4.0 - 10.0, 1e-9);
  EXPECT_NEAR(t[1], 3.0, 1e-9);
  EXPECT_NEAR(t[2], 6.0 + 2.0, 1e-9);
}

TEST_F(SceneCamerasTest, SetCameraNodeFromViewMovesAnExistingCamera)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Cameras/glTF/Cameras.gltf"));
  const size_t cameraCount = scene.getModel().cameras.size();
  ASSERT_GT(cameraCount, 0u);
  const int node = scene.getCameraNode(0);
  ASSERT_GE(node, 0);

  scene.editor().setCameraNodeFromView(node, makeView({7.0, 1.0, 2.0}, {0.0, 1.0, 0.0}));
  EXPECT_EQ(scene.getModel().cameras.size(), cameraCount);
  bool        found = false;
  const auto& cams  = scene.getRenderCameras();
  for(size_t i = 0; i < cams.size(); i++)
  {
    if(cams[i].cameraIndex == 0)
    {
      found = true;
      EXPECT_NEAR(cams[i].eye.x, 7.0, 1e-6);
    }
  }
  EXPECT_TRUE(found);
}

TEST_F(SceneCamerasTest, AddCameraCommandUndoRestoresTheScene)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  const size_t nodes   = scene.getModel().nodes.size();
  const size_t cameras = scene.getModel().cameras.size();

  AddCameraCommand cmd(scene, "Camera X", makeView({4.0, 3.0, 6.0}, {0.0, 0.0, 0.0}), -1, nullptr);
  cmd.execute();
  EXPECT_EQ(scene.getModel().nodes.size(), nodes + 1);
  EXPECT_EQ(scene.getModel().cameras.size(), cameras + 1);

  cmd.undo();
  EXPECT_EQ(scene.getModel().nodes.size(), nodes);
  EXPECT_EQ(scene.getModel().cameras.size(), cameras);
  EXPECT_EQ(scene.getRenderCameras(true).size(), cameras);

  cmd.execute();  // redo
  EXPECT_EQ(scene.getModel().cameras.size(), cameras + 1);
}

TEST_F(SceneCamerasTest, RemoveCameraDeletesItsNodeOrOnlyDetachesIt)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  const size_t nodes   = scene.getModel().nodes.size();
  const size_t cameras = scene.getModel().cameras.size();

  // A node that only carries the camera is deleted with it.
  ASSERT_GE(scene.editor().addCameraNode("Alone", makeView({1.0, 1.0, 5.0}, {0.0, 0.0, 0.0})), 0);
  ASSERT_EQ(scene.getModel().cameras.size(), cameras + 1);
  scene.editor().removeCamera(static_cast<int>(cameras));
  EXPECT_EQ(scene.getModel().nodes.size(), nodes);
  EXPECT_EQ(scene.getModel().cameras.size(), cameras);

  // A node with children keeps its place in the graph and loses only the camera.
  const int rig = scene.editor().addCameraNode("Rig", makeView({1.0, 1.0, 5.0}, {0.0, 0.0, 0.0}));
  ASSERT_GE(rig, 0);
  ASSERT_GE(scene.editor().addNode("Child", rig), 0);
  scene.editor().removeCamera(static_cast<int>(cameras));
  EXPECT_EQ(scene.getModel().nodes.size(), nodes + 2);
  EXPECT_EQ(scene.getModel().nodes[rig].camera, -1);
  EXPECT_EQ(scene.getModel().cameras.size(), cameras);
}

TEST_F(SceneCamerasTest, RemoveCameraCommandUndoRestoresTheCamera)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Cameras/glTF/Cameras.gltf"));
  const size_t nodes   = scene.getModel().nodes.size();
  const size_t cameras = scene.getModel().cameras.size();
  ASSERT_GT(cameras, 1u);
  const std::string name = scene.getModel().cameras[0].name;

  RemoveCameraCommand cmd(scene, 0, nullptr);
  cmd.execute();
  EXPECT_EQ(scene.getModel().cameras.size(), cameras - 1);
  EXPECT_LE(scene.getModel().nodes.size(), nodes);

  cmd.undo();
  EXPECT_EQ(scene.getModel().nodes.size(), nodes);
  ASSERT_EQ(scene.getModel().cameras.size(), cameras);
  EXPECT_EQ(scene.getModel().cameras[0].name, name);
  EXPECT_GE(scene.getCameraNode(0), 0);
}

// A scene is not given a camera behind the user's back: loading one without cameras adds no node, and
// removing the last camera leaves none (the viewer fits the view to the scene instead).
TEST_F(SceneCamerasTest, ASceneWithoutCamerasStaysThatWay)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  EXPECT_TRUE(scene.getModel().cameras.empty());
  EXPECT_TRUE(scene.getRenderCameras().empty());

  ASSERT_GE(scene.editor().addCameraNode("Only", makeView({4.0, 3.0, 6.0}, {0.0, 0.0, 0.0})), 0);
  ASSERT_EQ(scene.getModel().cameras.size(), 1u);
  scene.editor().removeCamera(0);
  EXPECT_TRUE(scene.getModel().cameras.empty());
  EXPECT_TRUE(scene.getRenderCameras(true).empty());
}

// A saved view needs its lens too: with only the point of interest there is nothing usable to start on.
TEST_F(SceneCamerasTest, SavedViewWithoutItsLensIsIgnored)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  scene.setSavedView(makeView({3.0, 2.0, 5.0}, {0.0, 0.0, 0.0}));
  ASSERT_TRUE(scene.getSavedView().has_value());

  tinygltf::Value::Object extras = scene.getModel().scenes[0].extras.Get<tinygltf::Value::Object>();
  extras.erase("camera::yfov");  // a perspective view without its field of view
  scene.getModel().scenes[0].extras = tinygltf::Value(extras);
  EXPECT_FALSE(scene.getSavedView().has_value());

  extras["camera::yfov"]            = tinygltf::Value(0.8);
  extras["camera::type"]            = tinygltf::Value(std::string("orthographic"));  // needs xmag and ymag
  scene.getModel().scenes[0].extras = tinygltf::Value(extras);
  EXPECT_FALSE(scene.getSavedView().has_value());
}

// A parent whose scale or mirror cannot hold the view in a node's TRS is not used: the camera is exact at the root.
TEST_F(SceneCamerasTest, CameraUnderAParentThatCannotHoldItsPoseGoesToTheRoot)
{
  auto assetsPath = TestResources::getSampleAssetsPath();
  if(assetsPath.empty())
    GTEST_SKIP() << "glTF-Sample-Assets not found at: " << GLTF_SAMPLE_ASSETS_PATH;

  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(assetsPath / "Models/Box/glTF/Box.gltf"));
  const nvvkgltf::RenderCamera view = makeView({4.0, 3.0, 6.0}, {0.0, 0.0, 0.0});

  const glm::vec3 scales[3] = {glm::vec3(2.0f, 1.0f, 1.0f), glm::vec3(-1.0f, 1.0f, 1.0f), glm::vec3(0.0f, 1.0f, 1.0f)};  // shear, mirror, zero
  for(size_t i = 0; i < 3; i++)
  {
    const int parent = scene.editor().addNode("Rig");
    ASSERT_GE(parent, 0);
    scene.editor().setNodeTRS(parent, glm::vec3(0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), scales[i]);
    const int node = scene.editor().addCameraNode("Camera", view, parent);
    ASSERT_GE(node, 0);
    EXPECT_EQ(scene.getNodeParents()[node], -1) << "case " << i;
    EXPECT_NEAR(scene.getModel().nodes[node].translation[0], 4.0, 1e-9) << "case " << i;
  }

  // A uniformly scaled parent can hold it: the camera stays a child.
  const int parent = scene.editor().addNode("Uniform");
  scene.editor().setNodeTRS(parent, glm::vec3(0.0f), glm::quat(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3(2.0f));
  const int node = scene.editor().addCameraNode("Camera", view, parent);
  ASSERT_GE(node, 0);
  EXPECT_EQ(scene.getNodeParents()[node], parent);
  EXPECT_NEAR(scene.getModel().nodes[node].translation[0], 2.0, 1e-9);
}

namespace {
// Two cameras and an animation whose KHR_animation_pointer channel animates /cameras/<animatedCamera>/perspective/yfov.
std::string cameraAnimationGltf(int animatedCamera)
{
  std::string gltf = R"({
  "asset": {"version": "2.0"},
  "extensionsUsed": ["KHR_animation_pointer"],
  "scene": 0,
  "scenes": [{"nodes": [0, 1]}],
  "nodes": [{"name": "A", "camera": 0}, {"name": "B", "camera": 1}],
  "cameras": [
    {"name": "A", "type": "perspective", "perspective": {"yfov": 0.7, "znear": 0.1, "zfar": 100.0}},
    {"name": "B", "type": "perspective", "perspective": {"yfov": 0.9, "znear": 0.1, "zfar": 100.0}}],
  "buffers": [{"byteLength": 16, "uri": "data:application/octet-stream;base64,AAAAAAAAgD8AAAA/ZmZmPw=="}],
  "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 8}, {"buffer": 0, "byteOffset": 8, "byteLength": 8}],
  "accessors": [
    {"bufferView": 0, "componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [1.0]},
    {"bufferView": 1, "componentType": 5126, "count": 2, "type": "SCALAR"}],
  "animations": [{
    "samplers": [{"input": 0, "output": 1, "interpolation": "LINEAR"}],
    "channels": [{"sampler": 0, "target": {"path": "pointer", "extensions": {"KHR_animation_pointer": {"pointer": "/cameras/@/perspective/yfov"}}}}]}]
})";
  gltf.replace(gltf.find('@'), 1, std::to_string(animatedCamera));
  return gltf;
}

std::string firstChannelPointer(const tinygltf::Model& model)
{
  return model.animations[0].channels[0].target_extensions.at("KHR_animation_pointer").Get("pointer").Get<std::string>();
}
}  // namespace

// Removing a camera renumbers the cameras after it; an animation pointer must follow its camera, and the
// animation of the removed camera goes with it.
TEST_F(SceneCamerasTest, RemoveCameraRenumbersAnimationPointers)
{
  const std::filesystem::path file = tempDir / "camera_anim.gltf";
  {
    std::ofstream out(file);
    out << cameraAnimationGltf(1);
  }
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(file));
  ASSERT_EQ(scene.getModel().cameras.size(), 2u);
  ASSERT_EQ(scene.getModel().animations.size(), 1u);

  scene.editor().removeCamera(0);  // camera B becomes camera 0
  ASSERT_EQ(scene.getModel().cameras.size(), 1u);
  EXPECT_EQ(scene.getModel().cameras[0].name, "B");
  ASSERT_EQ(scene.getModel().animations.size(), 1u);
  EXPECT_EQ(firstChannelPointer(scene.getModel()), "/cameras/0/perspective/yfov");
}

TEST_F(SceneCamerasTest, RemoveCameraDropsItsOwnAnimation)
{
  const std::filesystem::path file = tempDir / "camera_anim0.gltf";
  {
    std::ofstream out(file);
    out << cameraAnimationGltf(0);
  }
  nvvkgltf::Scene scene;
  ASSERT_TRUE(scene.load(file));
  ASSERT_EQ(scene.getModel().animations.size(), 1u);

  scene.editor().removeCamera(0);
  EXPECT_EQ(scene.getModel().cameras.size(), 1u);
  EXPECT_TRUE(scene.getModel().animations.empty());
}

// A bookmark equal to a glTF camera is a duplicate; one that differs only in orthographic zoom is not, nor one
// a few units away at large coordinates.
TEST_F(SceneCamerasTest, SameViewIsNotTooLooseFarFromTheOrigin)
{
  nvvkgltf::RenderCamera farView  = makeView({1.0e6, 3.0, 6.0}, {1.0e6, 0.0, 0.0});
  nvvkgltf::RenderCamera nearView = farView;
  nearView.eye.x += 5.0;
  nearView.center.x += 5.0;
  EXPECT_FALSE(nvvkgltf::isSameView(nvvkgltf::toManipulatorCamera(farView), nvvkgltf::toManipulatorCamera(nearView)));
  EXPECT_TRUE(nvvkgltf::isSameView(nvvkgltf::toManipulatorCamera(farView), nvvkgltf::toManipulatorCamera(farView)));
}

// A bookmark equal to a glTF camera is a duplicate; one that differs only in orthographic zoom is not.
TEST_F(SceneCamerasTest, SameViewComparesTheOrthographicZoom)
{
  nvvkgltf::RenderCamera ortho  = makeView({4.0, 3.0, 6.0}, {0.0, 0.0, 0.0});
  ortho.type                    = nvvkgltf::RenderCamera::CameraType::eOrthographic;
  ortho.xmag                    = 4.0;
  ortho.ymag                    = 3.0;
  nvvkgltf::RenderCamera zoomed = ortho;
  zoomed.xmag                   = 8.0;
  zoomed.ymag                   = 6.0;

  const nvutils::CameraManipulator::Camera a = nvvkgltf::toManipulatorCamera(ortho);
  EXPECT_TRUE(nvvkgltf::isSameView(a, nvvkgltf::toManipulatorCamera(ortho)));
  EXPECT_FALSE(nvvkgltf::isSameView(a, nvvkgltf::toManipulatorCamera(zoomed)));
}
