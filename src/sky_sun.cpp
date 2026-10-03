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

#include <algorithm>
#include <cmath>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include <nvutils/logger.hpp>

#include "gltf_environment_sky.hpp"
#include "gltf_scene.hpp"
#include "resources.hpp"
#include "sky_sun.hpp"
#include "sun_position.hpp"
#include "undo_redo.hpp"

//--------------------------------------------------------------------------------------------------
// The sun's direction as a unit vector, and the inverse. Azimuth is measured from +X toward +Z --
// nvgui::azimuthElevationSliders' own convention -- and elevation up from the horizon.
//
namespace {
glm::vec3 sunDirectionFromAngles(float azimuthDegrees, float elevationDegrees, bool yIsUp)
{
  const float az  = glm::radians(azimuthDegrees);
  const float el  = glm::radians(elevationDegrees);
  const float cel = std::cos(el);
  return yIsUp ? glm::vec3(cel * std::cos(az), std::sin(el), cel * std::sin(az)) :
                 glm::vec3(cel * std::cos(az), cel * std::sin(az), std::sin(el));
}

void anglesFromSunDirection(const glm::vec3& direction, bool yIsUp, float& azimuthDegrees, float& elevationDegrees)
{
  const glm::vec3 d  = glm::normalize(direction);
  const float     up = yIsUp ? d.y : d.z;
  elevationDegrees   = glm::degrees(std::asin(std::clamp(up, -1.0F, 1.0F)));
  azimuthDegrees     = glm::degrees(yIsUp ? std::atan2(d.z, d.x) : std::atan2(d.y, d.x));
}
}  // namespace

void SkySun::init(Resources* resources, UndoStack* undoStack, Actions actions)
{
  m_resources = resources;
  m_undoStack = undoStack;
  m_actions   = std::move(actions);
}

//--------------------------------------------------------------------------------------------------
// Adopt the scene's marked sun light, if it has one.
//
// The sky is a medium and the sun is a light -- that is OMI_environment_sky's own model, and it is
// why a saved sky carries a directional light beside it. This is the read half: whichever light
// the marker names becomes the sun, and the sky follows it wherever it is moved.
//
// Unmarked lights are never considered. A scene may hold any number of directional lights and
// none of them is a sun by virtue of being directional; guessing at one was the mistake this
// marker exists to retire.
//
void SkySun::syncFromMarkedLight()
{
  m_lightIndex = -1;
  m_lightNode  = -1;

  const nvvkgltf::Scene* scene = m_resources->getScene();
  if(scene == nullptr || !scene->valid())
    return;

  const tinygltf::Model& model   = scene->getModel();
  const int              nodeIdx = gltf_environment_sky::findSkySunNode(model);
  if(nodeIdx < 0)
    return;

  const int lightIdx = gltf_environment_sky::nodeLightIndex(model.nodes[nodeIdx]);
  if(lightIdx < 0)
    return;  // marker on a node that carries no light: stale, ignore it

  // The render-light list is what the shaders index, and it carries the world matrix -- which is
  // what a nested or animated light needs. Matched on the node as well as the light: one light may
  // be instanced by several nodes, and only the marked one is the sun.
  const std::vector<nvvkgltf::RenderLight>& rlights = scene->getRenderLights();
  for(size_t i = 0; i < rlights.size(); ++i)
  {
    if(rlights[i].light != lightIdx || rlights[i].nodeID != nodeIdx)
      continue;

    // glTF aims a light down its node's -Z, so +Z points back at the sun.
    const glm::vec3 toSun = glm::vec3(rlights[i].worldMatrix[2]);
    const float     len   = glm::length(toSun);
    if(len < 1e-6F)
      return;  // degenerate node scale; no usable direction

    m_lightIndex              = static_cast<int>(i);
    m_lightNode               = nodeIdx;
    const glm::vec3 dir       = toSun / len;
    const bool      moved     = !glm::all(glm::epsilonEqual(dir, m_resources->sunDirection, 1e-6F));
    m_resources->sunDirection = dir;
    syncAngles();

    if(moved)
    {
      // Only the physical sky's baked image depends on where the sun is; the authored skies keep
      // it out of their bake entirely. A preview, not a commit -- the light moves for as long as
      // someone drags its gizmo.
      if(m_resources->settings.envSystem == shaderio::EnvSystem::eSky)
        if(m_actions.environmentPreview)
          m_actions.environmentPreview();
      if(m_actions.resetFrame)
        m_actions.resetFrame();
    }
    return;
  }
}

//--------------------------------------------------------------------------------------------------
// Candidate suns: every directional light in the scene.
//
std::vector<std::pair<int, std::string>> SkySun::candidateLights() const
{
  std::vector<std::pair<int, std::string>> out;
  const nvvkgltf::Scene*                   scene = m_resources->getScene();
  if(scene == nullptr || !scene->valid())
    return out;

  const tinygltf::Model& model = scene->getModel();
  for(size_t i = 0; i < model.nodes.size(); ++i)
  {
    const tinygltf::Node& node     = model.nodes[i];
    int                   lightIdx = node.light;
    if(lightIdx < 0)
    {
      const auto ext = node.extensions.find("KHR_lights_punctual");
      if(ext != node.extensions.end() && ext->second.Has("light"))
        lightIdx = ext->second.Get("light").GetNumberAsInt();
    }
    if(lightIdx < 0 || lightIdx >= static_cast<int>(model.lights.size()))
      continue;
    if(model.lights[lightIdx].type != "directional")
      continue;

    std::string name = node.name.empty() ? model.lights[lightIdx].name : node.name;
    if(name.empty())
      name = "Directional light " + std::to_string(lightIdx);
    out.emplace_back(static_cast<int>(i), std::move(name));
  }
  return out;
}

//--------------------------------------------------------------------------------------------------
// Choose which light is the sky's sun.
//
void SkySun::setSunNode(int nodeIndex)
{
  nvvkgltf::Scene* scene = m_resources->getScene();
  if(scene == nullptr || !scene->valid() || nodeIndex == m_lightNode)
    return;

  m_undoStack->executeCommand(std::make_unique<SetSkySunCommand>(*scene, m_lightNode, nodeIndex));

  // The sun's direction changes with the choice, and for the physical sky its image depends on
  // that direction. syncSunFromMarkedLight() picks the new one up at the next frame top; the bake
  // has to be told separately because nothing else here marks it stale.
  if(m_resources->settings.envSystem == shaderio::EnvSystem::eSky)
    if(m_actions.environmentPreview)
      m_actions.environmentPreview();
  if(m_actions.resetFrame)
    m_actions.resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Aim the marked sun light, undoably. The write half of syncSunFromMarkedLight().
//
bool SkySun::setMarkedDirection(const glm::vec3& toSun)
{
  nvvkgltf::Scene* scene = m_resources->getScene();
  if(m_lightNode < 0 || scene == nullptr || !scene->valid())
    return false;

  const tinygltf::Model& model = scene->getModel();
  if(m_lightNode >= static_cast<int>(model.nodes.size()))
    return false;

  const tinygltf::Node& node = model.nodes[m_lightNode];

  // Local TRS, which is what a node stores and what the undo command records. The direction is a
  // world one, so a sun parented under a rotated rig has the parent's rotation taken back out --
  // otherwise syncFromMarkedLight() would read the parent's turn on top at the next frame.
  // getNodeTRS decomposes a `matrix` node too; reading the TRS fields directly would see identity
  // there and move the light to the origin on the setNodeTRS below.
  glm::vec3 translation{0.0F};
  glm::quat oldRotation{1.0F, 0.0F, 0.0F, 0.0F};
  glm::vec3 scale{1.0F};
  tinygltf::utils::getNodeTRS(node, translation, oldRotation, scale);

  // glTF aims a light down -Z, so +Z is what points at the sun.
  const glm::quat worldRotation = glm::rotation(glm::vec3(0.0F, 0.0F, 1.0F), glm::normalize(toSun));
  // The scene keeps the parent table, so no hierarchy walk: this runs every frame of a slider or
  // Time of Day drag.
  const std::vector<int>& parents     = scene->getNodeParents();
  const int               parent      = m_lightNode < static_cast<int>(parents.size()) ? parents[m_lightNode] : -1;
  const glm::mat4         parentWorld = parent >= 0 ? scene->computeNodeWorldMatrix(parent) : glm::mat4(1.0F);
  const glm::quat         newRotation = gltf_environment_sky::localRotationForWorld(parentWorld, worldRotation);

  scene->editor().setNodeTRS(m_lightNode, translation, newRotation, scale);
  m_undoStack->pushExecuted(std::make_unique<SetTransformCommand>(*scene, m_lightNode, translation, oldRotation, scale,
                                                                  translation, newRotation, scale));
  return true;
}

//--------------------------------------------------------------------------------------------------
// Apply a new sun direction, through whatever owns the sun.
//
// The UI paths that move the sun -- the azimuth/elevation sliders, the Time of Day widget, and a
// preset load -- each used to answer the same three questions for itself: who owns the sun, do the
// reported angles still match it, and does the image have to be re-baked. Answering them here is
// what keeps a new caller from getting one of them wrong.
//
void SkySun::aim(const glm::vec3& toSun)
{
  if(!setMarkedDirection(toSun))
    m_resources->sunDirection = glm::normalize(toSun);

  syncAngles();  // keep the reported and persisted angles equal to the direction just set

  // Only the physical sky's baked image depends on where the sun is. The authored skies keep the
  // sun out of their lighting field entirely -- it is a directional light that next-event
  // estimation samples -- so a gradient sky costs nothing here.
  if(m_resources->settings.envSystem == shaderio::EnvSystem::eSky)
    if(m_actions.environmentPreview)
      m_actions.environmentPreview();
  if(m_actions.resetFrame)
    m_actions.resetFrame();
}

//--------------------------------------------------------------------------------------------------
// A compass bearing, in the azimuth the renderer measures.
//
// The two disagree by a quarter turn, and the reason is glTF: a camera looks down -Z, so -Z is the
// direction a scene faces and the natural place to put north. The renderer's own azimuth is measured
// from +X toward +Z (nvgui::azimuthElevationSliders), which puts +X at east.
//
//     north -Z    east +X    south +Z    west -X
//
// That is the compass of the environment's own frame. A model built facing some other way turns
// the environment (Settings::envRotation), and the compass turns with it -- see applyTimeOfDay.
//
static float sceneAzimuthFromCompass(float bearingDeg)
{
  return bearingDeg - 90.0F;
}

//--------------------------------------------------------------------------------------------------
// Put the sun where it actually was, at a place and a moment.
//
// The Time of Day settings are a clock reading; sun_position wants UTC, so the offset comes off
// here. The result is a compass bearing, which sceneAzimuthFromCompass turns into the renderer's own
// azimuth -- measured from +X toward +Z -- in the environment's frame. The environment's rotation
// then carries it into the world, so north is wherever the sky has been turned to put it, and the
// sun and the sky it sits in can never disagree about which way that is.
//
void SkySun::applyTimeOfDay()
{
  const Settings& st = m_resources->settings;

  int year = 0, month = 0, day = 0;
  if(!sun_position::parseIsoDate(st.todDate, year, month, day))
  {
    LOGW("Time of Day: could not read '%s' as a yyyy-mm-dd date; the sun was left where it was.\n", st.todDate.c_str());
    return;
  }

  // Hours outside 0..24 are fine and are not clamped: a UTC offset pushes a local midnight into
  // the neighbouring day, and the Julian-day arithmetic carries that for us.
  const sun_position::AzimuthElevation sun =
      sun_position::computeSunPosition({st.todLatitude, st.todLongitude}, {year, month, day, st.todHour - st.todUtcOffset});

  const glm::vec3 inSky = sunDirectionFromAngles(sceneAzimuthFromCompass(sun.azimuthDeg), sun.elevationDeg, m_resources->sunYIsUp);
  aim(st.envRotationQuat() * inSky);
}

//--------------------------------------------------------------------------------------------------
// Drive the shared sun from angles. The inverse of syncAngles(), and the single place the
// conversion happens on the way in -- the settings callback and the Environment panel's reset both
// land here rather than each calling the free function with their own yIsUp argument. Through
// aim(), like every other move, so a marked sun light follows instead of being overwritten by
// syncFromMarkedLight() on the next frame.
//
void SkySun::setAngles(float azimuthDegrees, float elevationDegrees)
{
  aim(sunDirectionFromAngles(azimuthDegrees, elevationDegrees, m_resources->sunYIsUp));
}

//--------------------------------------------------------------------------------------------------
// Mirror Resources::sunDirection back into the reported azimuth/elevation. Called after the sky UI
// moves the sun, so reading sunAzimuth/sunElevation tells the truth.
void SkySun::syncAngles()
{
  anglesFromSunDirection(m_resources->sunDirection, m_resources->sunYIsUp, m_resources->settings.sunAzimuth,
                         m_resources->settings.sunElevation);
}
