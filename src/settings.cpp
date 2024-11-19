/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2024 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "imgui.h"
#include "imgui/imgui_helper.h"

#include "settings.hpp"
#include <glm/gtc/type_ptr.hpp>
#include "collapsing_header_manager.h"

//-----------------------------------------------------------------------------
// Rendering the UI for the settings
//
void gltfr::Settings::onUI()
{
  auto& headerManager = CollapsingHeaderManager::getInstance();
  namespace PE = ImGuiH::PropertyEditor;
  if(headerManager.beginHeader("Settings"))
  {
    PE::begin("gltfr::Settings::onUI");
    PE::SliderInt("Max Frames", &maxFrames, 1, 1000000);
    PE::Checkbox("Show Axis", &showAxis);
    PE::SliderFloat("Max Luminance", &maxLuminance, 0.0f, 10000.0f);
    PE::ColorEdit3("Silhouette Color", glm::value_ptr(silhouetteColor),
                   ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | ImGuiColorEditFlags_Float);
    PE::end();
  }
}

// The HDR intensity is the luminance of the environment when using HDR.
// See: HdrEnv::getIntegral()
void gltfr::Settings::setDefaultLuminance(float hdrEnvIntensity)
{
  maxLuminance = (envSystem == Settings::eSky) ? 10.0f : hdrEnvIntensity;
}
