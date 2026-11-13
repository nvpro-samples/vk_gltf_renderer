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

// Starts the viewport on a freshly loaded scene and fills the camera widget.
//  - Starting view: the one saved with the file, else the first glTF camera, else a fit to the scene.
//    It is also the widget's HOME.
//  - The widget then holds only the user's view bookmarks of this scene (the .json next to the
//    executable), exactly as saved. The scene's glTF cameras are not copied into it on load: they live in
//    the model (Elements > Cameras), so editing, adding or deleting one never leaves a stale copy. A user
//    can import them as bookmarks on purpose (Camera window); those are kept like any other bookmark.
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
}

}  // namespace nvvkgltf
