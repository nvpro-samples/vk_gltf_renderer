/*
 * Copyright (c) 2022--2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022--2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//
// Camera conversion utilities for glTF scenes. Extracts camera parameters
// (eye, center, up, FOV, clip planes) from glTF camera nodes and converts
// between the glTF RenderCamera representation and the application's
// CameraManipulator system. Also handles viewport-dependent adjustments.
//

#pragma once

#include <filesystem>
#include <limits>
#include "nvgui/camera.hpp"
#include "gltf_scene.hpp"


namespace nvvkgltf {

// Helper: Convert RenderCamera to CameraManipulator::Camera
inline nvutils::CameraManipulator::Camera toManipulatorCamera(const nvvkgltf::RenderCamera& cam)
{
  nvutils::CameraManipulator::Camera uiCam;
  uiCam.eye     = cam.eye;
  uiCam.ctr     = cam.center;
  uiCam.up      = cam.up;
  uiCam.nearFar = {cam.znear, cam.zfar};
  if(cam.type == nvvkgltf::RenderCamera::CameraType::eOrthographic)
  {
    uiCam.projectionType = nvutils::CameraManipulator::ProjectionType::Orthographic;
    uiCam.orthMag        = {cam.xmag, cam.ymag};
    uiCam.fov            = 45.0;
  }
  else
  {
    uiCam.projectionType = nvutils::CameraManipulator::ProjectionType::Perspective;
    uiCam.fov            = glm::degrees(cam.yfov);
  }
  return uiCam;
}

// Same viewpoint up to round-off (glTF stores float32, the sidecar JSON doubles). The clip planes are
// not part of the comparison: a bookmark is a place to look from.
// Relative tolerance of the "same view" test: a few float32 rounding steps, which is all a glTF store
// of a camera loses. Anything wider could take a distinct bookmark for a duplicate and delete it.
constexpr double kViewTolerance = 4.0 * std::numeric_limits<float>::epsilon();

inline bool isClosePoint(const glm::dvec3& x, const glm::dvec3& y)
{
  const glm::dvec3 epsilon = kViewTolerance * glm::max(glm::dvec3(1.0), glm::max(glm::abs(x), glm::abs(y)));
  return glm::all(glm::lessThanEqual(glm::abs(x - y), epsilon));
}

inline bool isSameView(const nvutils::CameraManipulator::Camera& a, const nvutils::CameraManipulator::Camera& b)
{
  if(a.projectionType != b.projectionType || !isClosePoint(a.eye, b.eye) || !isClosePoint(a.ctr, b.ctr)
     || !isClosePoint(a.up, b.up))
    return false;
  if(a.projectionType == nvutils::CameraManipulator::ProjectionType::Orthographic)
    return isClosePoint(glm::dvec3(a.orthMag, 0.0), glm::dvec3(b.orthMag, 0.0));  // same zoom
  return std::abs(a.fov - b.fov) <= kViewTolerance * std::max(1.0, std::abs(a.fov));
}

// Starts the viewport on a freshly loaded scene and fills the camera widget.
//  - Starting view: the one saved with the file, else the first glTF camera, else a fit to the scene.
//    It is also the widget's HOME.
//  - The widget then holds only the user's view bookmarks of this scene (the .json next to the
//    executable). The scene's glTF cameras are not copied into it: they live in the model (Elements >
//    Cameras), so editing, adding or deleting one never leaves a stale copy.
//  - Bookmarks that duplicate a glTF camera (older versions copied the cameras into that file) are dropped.
inline void setupSceneView(std::shared_ptr<nvutils::CameraManipulator> cameraManip,
                           const std::filesystem::path&                filename,
                           nvvkgltf::Scene&                            scene)
{
  const std::vector<nvvkgltf::RenderCamera>&  cameras = scene.getRenderCameras();
  const std::optional<nvvkgltf::RenderCamera> saved   = scene.getSavedView();

  if(saved || !cameras.empty())
  {
    const nvutils::CameraManipulator::Camera cam = toManipulatorCamera(saved ? *saved : cameras[0]);
    cameraManip->setCamera(cam);
    nvgui::SetHomeCamera(cam);
  }
  else
  {
    // Re-adjusting camera to fit the new scene
    const nvutils::Bbox& sceneBbox = scene.getSceneBounds();
    cameraManip->fit(sceneBbox.min(), sceneBbox.max(), true);
    // Planes follow the framing distance fit() just chose (see defaultClipPlanes)
    cameraManip->setClipPlanes(
        defaultClipPlanes(glm::length(cameraManip->getEye() - glm::dvec3(sceneBbox.center())), sceneBbox.radius()));
    nvgui::SetHomeCamera(cameraManip->getCamera());
  }

  nvgui::SetCameraJsonFile(filename.stem(), cameraManip);  // loads the bookmarks now; HOME stays at [0]

  std::vector<nvutils::CameraManipulator::Camera>& widgetCameras = nvgui::GetCameras();
  for(size_t n = widgetCameras.size(); n-- > 1;)
  {
    bool isSceneCamera = false;
    for(size_t c = 0; c < cameras.size() && !isSceneCamera; c++)
      isSceneCamera = isSameView(widgetCameras[n], toManipulatorCamera(cameras[c]));
    if(isSceneCamera)
      widgetCameras.erase(widgetCameras.begin() + n);
  }
}

}  // namespace nvvkgltf
