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

#pragma once

#include <optional>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <tinygltf/tiny_gltf.h>

//--------------------------------------------------------------------------------------------------
// Host model and serializer for the glTF `OMI_environment_sky` extension -- the authored sky of a
// scene.
//
// The *document* side of the environment: what a .gltf says the sky is, kept separate from the
// renderer's sky state (Settings / SkyOmiParameters) because the two answer different questions.
// This struct answers "what did the author write?"; Settings answers "what are we drawing now?",
// and covers viewer-only controls the extension has no field for.
//
// This renderer authors one sky. A file carrying several is honoured through `scenes[i].sky`; the
// rest are kept in the document and logged.
//
// Member defaults are the *extension's* defaults, so a descriptor built from a file that omits a
// property matches what a spec-compliant reader would assume. They are not the renderer's starting
// values -- those live in Settings and are chosen for a usable empty scene.
//--------------------------------------------------------------------------------------------------
struct SkyDescriptor
{
  // Values match the extension's `type` string. ePanorama resolves to a loaded lat-long image;
  // the other three are evaluated by the renderer.
  enum class Type
  {
    ePlain,
    eGradient,
    ePanorama,
    ePhysical,
  };

  Type type{Type::ePlain};

  // OMI's `ambientLightColor` and `ambientSkyContribution` are deliberately not modelled here.
  // This renderer lights entirely from the environment it bakes; it has no separate ambient term
  // for them to scale, and a field parsed into a struct nobody reads is worse than an absent one --
  // it reads like support. A file carrying them keeps them verbatim through `raw`, so nothing is
  // lost on a round trip; they simply do not change the image. See docs/sky.md.

  // OMI's sky orientation. The renderer turns the environment about +Y only, so only that
  // component survives a load -- a tilted sky arrives upright, with a warning, rather than being
  // silently accepted and then drawn wrong. The quaternion is stored whole regardless, so a file
  // that carries a tilt still round-trips it.
  glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};  // w, x, y, z (glTF writes x, y, z, w)

  // False when the file carried no `rotation`. The distinction matters on load: the renderer's
  // rotation control is viewer state a user, the command line or MCP may have set, and a scene that
  // says nothing about orientation must not reset it. Only a rotation the file actually states
  // overrides it -- the same "said nothing" vs "said identity" rule the atmosphere block follows.
  bool rotationAuthored{false};

  // Exactly one of the blocks below is meaningful, selected by `type`. The others keep their
  // defaults, and any value the file carried for them survives in `raw`.
  struct Plain
  {
    glm::vec3 color{0.0f, 0.0f, 0.0f};
  } plain;

  // The extension requires the three colors and defaults the four curves. The color defaults here
  // are a plausible daylight sky, used only when a file violates that requirement.
  struct Gradient
  {
    glm::vec3 bottomColor{0.2f, 0.169f, 0.133f};
    glm::vec3 horizonColor{0.646f, 0.656f, 0.67f};
    glm::vec3 topColor{0.385f, 0.454f, 0.55f};
    float     bottomCurve{0.02f};
    float     topCurve{0.15f};
    float     sunAngleMax{0.5f};
    float     sunCurve{0.15f};
  } gradient;

  // The extension references glTF textures, not URIs. `cubemap` is never written; a file carrying
  // one keeps it through `raw`.
  struct Panorama
  {
    // OMI's own field, an index into textures[]. Preserved but never produced: an .hdr/.exr is not
    // a glTF image mime type, and inlining one to satisfy the schema would bloat every scene.
    int equirectangular{-1};

    // Where this renderer stores the panorama instead: a URI relative to the glTF, carried in a
    // sibling NV_environment_sky_panorama block. A reader that only knows OMI sees `panorama` with
    // no source and falls back to its own default -- a visible gap rather than a wrong answer.
    std::string uri;
  } panorama;

  struct Physical
  {
    glm::vec3 groundColor{0.3f, 0.2f, 0.1f};
    glm::vec3 mieColor{1.0f, 1.0f, 1.0f};
    glm::vec3 rayleighColor{0.3f, 0.5f, 1.0f};
    float     mieAnisotropy{0.8f};
    float     mieCoefficient{0.000005f};
    float     rayleighCoefficient{0.00003f};
  } physical;

  // The rest of the atmosphere, in a sibling NV_environment_sky_atmosphere block.
  //
  // OMI's `physical` describes what the air scatters but not the world it surrounds -- no star
  // spectrum, planet size, scale heights, aerosol absorption or ozone layer. Without them a saved
  // scene reopens as Earth whatever it was.
  //
  // A sibling block rather than extra keys inside `physical`, for the same reason the panorama URI
  // is one: a reader that knows only OMI still gets a coherent sky instead of meeting keys the
  // schema forbids.
  //
  // Metres and m^-1, matching OMI's units so the two blocks agree inside one file; the model works
  // in kilometres and converts at the file boundary.
  struct Atmosphere
  {
    // False when the file carried no such block. A load leaves every field below untouched then,
    // so an OMI-only scene keeps whatever the renderer already had rather than being forced to
    // Earth -- the distinction between "said nothing" and "said Earth".
    bool present{false};

    glm::vec3 solarIrradiance{1.474f, 1.8504f, 1.91198f};                // W/m^2 at the top of the atmosphere
    glm::vec3 ozoneExtinction{6.49717e-07f, 1.8809e-06f, 8.50167e-08f};  // m^-1 at the peak
    float     sunAngularRadius{0.004675f};                               // radians
    float     rayleighScaleHeight{8000.0f};                              // m
    float     mieScaleHeight{1200.0f};                                   // m
    float     mieAlbedo{0.9f};                                           // scattering / extinction
    float     ozoneCenter{25000.0f};                                     // m
    float     ozoneWidth{30000.0f};                                      // m
    float     planetRadius{6360000.0f};                                  // m
    float     atmosphereThickness{60000.0f};                             // m
  } atmosphere;

  // The sky entry exactly as it was parsed, including every key this build does not model.
  //
  // This is what makes a load/save round trip non-destructive. A typed struct cannot hold what
  // it has no fields for: a scene authored by a newer build, by another vendor's exporter, or
  // carrying a sky type this phase does not implement would otherwise be silently stripped the
  // first time it is opened and saved here. On save the typed members are written *over a copy
  // of this value*, so unmodelled keys pass through untouched -- the same preserve-and-merge
  // discipline gltf_scene.cpp already applies to node `extras`.
  tinygltf::Value raw;

  // The file's `type` string when this build does not recognize it; empty otherwise. `type` then
  // reads ePlain (the fallback the renderer shows), and this is what keeps the save from rewriting
  // the sky as plain: toValue() writes `raw` back as it came while this is set. Cleared as soon as
  // the renderer assigns a type of its own.
  std::string unknownType;
};

// A scene either authors a sky or it does not; `eNone` is a renderer mode, not an authored type,
// so the absence of a sky is modelled as an empty optional rather than an enum value.
using EnvironmentState = std::optional<SkyDescriptor>;

namespace gltf_environment_sky {

// Marker stamped into the `extras` of the node holding the sky's sun.
//
// OMI_environment_sky describes a *medium*, not a light source -- its own overview says to add
// suns with KHR_lights_punctual. So a sky that has a sun needs a directional light beside it, and
// the renderer needs to know which light that is. "The first directional light" is a guess, and a
// wrong one as soon as a scene has two; this marker makes it a fact.
//
// In `extras` rather than a new extension because it degrades correctly: any other renderer reads
// a perfectly ordinary directional light and lights the sky with it, which is exactly what the
// specification asks for. Follows the precedent of Scene::kExternalAssetContentKey.
constexpr const char* kSkySunMarkerKey = "NV_sky_sun";

// The marker's value records *provenance*, which decides who may remove the light again:
//
//   "renderer" -- this renderer created the node and its light, when a save needed a sun the
//                 scene did not have. Ours to withdraw when a later save no longer needs one.
//   "scene"    -- the light was already there and someone chose it. Never ours to remove; the
//                 most we may do is stop calling it the sun.
//
// Without the distinction, unmarking and deleting look the same, and switching a saved scene from
// Sky to HDR quietly unpicks a light the user authored.
constexpr const char* kSkySunOwnerRenderer = "renderer";
constexpr const char* kSkySunOwnerScene    = "scene";

inline void setSkySunMarker(tinygltf::Node& node, const char* owner = kSkySunOwnerScene)
{
  tinygltf::Value::Object obj = node.extras.IsObject() ? node.extras.Get<tinygltf::Value::Object>() : tinygltf::Value::Object{};
  obj[kSkySunMarkerKey] = tinygltf::Value(std::string(owner));
  node.extras           = tinygltf::Value(std::move(obj));
}

// True when this renderer created the node, and may therefore take it away again.
[[nodiscard]] inline bool isRendererOwnedSun(const tinygltf::Node& node)
{
  if(!node.extras.Has(kSkySunMarkerKey))
    return false;
  const tinygltf::Value& v = node.extras.Get(kSkySunMarkerKey);
  return v.IsString() && v.Get<std::string>() == kSkySunOwnerRenderer;
}

[[nodiscard]] inline bool hasSkySunMarker(const tinygltf::Node& node)
{
  return node.extras.Has(kSkySunMarkerKey);
}

// The light a node instances, or -1. `node.light` is the canonical reference; the extension map is
// the fallback for JSON edited by hand, mirroring what gltf_compact_scene.cpp does.
[[nodiscard]] inline int nodeLightIndex(const tinygltf::Node& node)
{
  if(node.light >= 0)
    return node.light;
  const auto ext = node.extensions.find("KHR_lights_punctual");
  if(ext != node.extensions.end() && ext->second.Has("light"))
    return ext->second.Get("light").GetNumberAsInt();
  return -1;
}

// Index of the node carrying the marker, or -1. First match wins: the writer only ever creates
// one, and a file with several is malformed rather than meaningful.
[[nodiscard]] int findSkySunNode(const tinygltf::Model& model);

// The local rotation that gives a node under `parentWorld` the world rotation `worldRotation`, with
// the parent's rotation taken back out. A sun direction is a world direction, but a node stores a
// local one; writing the world value straight into a parented light would have the parent's turn
// applied on top on the next read.
//
// The live scene already tracks parents (nvvkgltf::Scene::getNodeParents / computeNodeWorldMatrix),
// so a caller holding one passes that parent's world matrix here.
[[nodiscard]] glm::quat localRotationForWorld(const glm::mat4& parentWorld, const glm::quat& worldRotation);

// The same, for a bare model with no parent table -- the save hook's `outModel`, which for a scene
// with external assets is a transformed copy whose node indices need not match the live scene's.
// Finds the ancestors by searching `children`, so it is for a one-off save, not a per-frame path.
[[nodiscard]] glm::quat localRotationForWorld(const tinygltf::Model& model, int nodeIndex, const glm::quat& worldRotation);

// The extension this module reads and writes. Also the key registered in the scene's supported
// extension list.
constexpr const char* kExtensionName = "OMI_environment_sky";

// One sky, as the extension's `skies[]` entry -- the whole serializer, without the document around
// it.
//
// A `.sky.json` preset *is* this object: the plan's file format is exactly one entry of that array,
// no wrapper. Splitting it out is what lets a preset and a glTF share one implementation rather than
// two that drift; everything below is expressed in terms of these two.
[[nodiscard]] tinygltf::Value toValue(const SkyDescriptor& sky);

// The inverse. nullopt when `entry` is not an object; unknown keys are preserved in `raw`.
[[nodiscard]] EnvironmentState fromValue(const tinygltf::Value& entry);

// Reads the sky the given scene references, or nullopt when the model authors none.
//
// Follows `scenes[sceneIndex].extensions.OMI_environment_sky.sky` into the document-level
// `skies` array, defaulting that index to 0. Additional skies are dropped with an info log --
// the renderer supports exactly one per scene by design.
EnvironmentState parse(const tinygltf::Model& model, int sceneIndex);

// Writes `sky` as the single entry of `outModel.extensions.OMI_environment_sky.skies`, and points
// `outModel.scenes[sceneIndex]` at it.
//
// Must be handed the model that is actually serialized. Scene::save builds a copy for the
// external-asset transform and writes *that*, so anything written to the live model instead never
// reaches the file -- and does so silently.
//
// The typed members are written over a copy of `sky.raw`, so keys this build does not model
// survive the round trip.
void write(tinygltf::Model& outModel, int sceneIndex, const SkyDescriptor& sky);

// Removes the extension from the document and from every scene. Used when the save toggle is off,
// so that a scene which once carried a sky does not keep a stale one.
void strip(tinygltf::Model& outModel);

}  // namespace gltf_environment_sky
