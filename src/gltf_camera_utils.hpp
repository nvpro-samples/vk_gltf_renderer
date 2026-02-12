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

// Various Application utilities
// - Display a menu with File/Quit
// - Display basic information in the window title

#pragma once

#include <filesystem>
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
  uiCam.nearFar = {static_cast<float>(cam.znear), static_cast<float>(cam.zfar)};
  if(cam.type == nvvkgltf::RenderCamera::CameraType::eOrthographic)
  {
    uiCam.projectionType = nvutils::CameraManipulator::ProjectionType::Orthographic;
    uiCam.orthMag.x      = static_cast<float>(cam.xmag);
    uiCam.orthMag.y      = static_cast<float>(cam.ymag);
    uiCam.fov            = 45.0f;
  }
  else
  {
    uiCam.projectionType = nvutils::CameraManipulator::ProjectionType::Perspective;
    uiCam.fov            = static_cast<float>(glm::degrees(cam.yfov));
  }
  return uiCam;
}

// This function adds the camera to the camera manipulator
// It also sets the camera to the first camera in the list
// If there is no camera, it fits the camera to the scene
inline void addSceneCamerasToWidget(std::shared_ptr<nvutils::CameraManipulator> cameraManip,
                                    const std::filesystem::path&                filename,
                                    const std::vector<nvvkgltf::RenderCamera>&  cameras,
                                    const nvutils::Bbox&                        sceneBbox)
{
  nvgui::SetCameraJsonFile(filename.stem());
  if(!cameras.empty())
  {
    const auto& camera = cameras[0];
    auto        cam    = toManipulatorCamera(camera);

    cameraManip->setCamera(cam);
    nvgui::SetHomeCamera(cam);

    for(const auto& cam : cameras)
    {
      nvgui::AddCamera(toManipulatorCamera(cam));
    }
  }
  else
  {
    // Re-adjusting camera to fit the new scene
    cameraManip->fit(sceneBbox.min(), sceneBbox.max(), true);
    cameraManip->setClipPlanes(glm::vec2(0.001F * sceneBbox.radius(), 100.0F * sceneBbox.radius()));
    nvgui::SetHomeCamera(cameraManip->getCamera());
  }
}

// Convert widget cameras to RenderCamera list (HOME included at index 0)
inline std::vector<nvvkgltf::RenderCamera> getCamerasFromWidget()
{
  std::vector<nvvkgltf::RenderCamera> renderCameras;
  const auto                          widgetCameras = nvgui::GetCameras();
  renderCameras.reserve(widgetCameras.size());

  for(const auto& cam : widgetCameras)
  {
    nvvkgltf::RenderCamera renderCam;
    renderCam.eye    = cam.eye;
    renderCam.center = cam.ctr;
    renderCam.up     = cam.up;
    renderCam.znear  = static_cast<double>(cam.nearFar.x);
    renderCam.zfar   = static_cast<double>(cam.nearFar.y);

    if(cam.projectionType == nvutils::CameraManipulator::ProjectionType::Orthographic)
    {
      renderCam.type = nvvkgltf::RenderCamera::CameraType::eOrthographic;
      renderCam.xmag = static_cast<double>(cam.orthMag.x);
      renderCam.ymag = static_cast<double>(cam.orthMag.y);
    }
    else
    {
      renderCam.type = nvvkgltf::RenderCamera::CameraType::ePerspective;
      renderCam.yfov = static_cast<double>(glm::radians(cam.fov));
    }

    renderCameras.push_back(renderCam);
  }

  return renderCameras;
}


}  // namespace nvvkgltf
