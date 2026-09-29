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

//
// OMI_environment_sky serialization -- see gltf_environment_sky.hpp for the data model.
//
// The whole file is preserve-and-merge: every read falls back to the extension's own default, and
// every write starts from the verbatim JSON the file carried. That is what lets a scene authored
// by a newer build, another vendor's exporter, or with a sky type this build does not render
// survive a load/save round trip instead of being quietly reduced to the fields we happen to model.
//
// Nothing here knows about the renderer. Mapping a descriptor onto the live sky (and back) is the
// renderer's job, so this stays a pure document layer that a test can exercise without a device.
//

#include <algorithm>

#include <nvutils/logger.hpp>

#include "gltf_environment_sky.hpp"
#include "tinygltf_utils.hpp"

namespace gltf_environment_sky {

namespace {

// Property names, verbatim from the extension. Named once so a typo cannot differ between the
// reader and the writer -- the failure mode there is a silent round-trip loss, not a build error.
constexpr const char* kSkies    = "skies";
constexpr const char* kSky      = "sky";
constexpr const char* kType     = "type";
constexpr const char* kRotation = "rotation";

// Sibling extension carrying the panorama URI. Separate from OMI_environment_sky because it is our
// addition, not theirs: a reader must be able to reject it by name without rejecting the sky.
constexpr const char* kPanoramaExtension   = "NV_environment_sky_panorama";
constexpr const char* kAtmosphereExtension = "NV_environment_sky_atmosphere";

// NV_environment_sky_atmosphere keys. Metres and m^-1 throughout -- see SkyDescriptor::Atmosphere.
constexpr const char* kSolarIrradiance     = "solarIrradiance";
constexpr const char* kSunAngularRadius    = "sunAngularRadius";
constexpr const char* kRayleighScaleHeight = "rayleighScaleHeight";
constexpr const char* kMieScaleHeight      = "mieScaleHeight";
constexpr const char* kMieAlbedo           = "mieAlbedo";
constexpr const char* kOzoneExtinction     = "ozoneExtinction";
constexpr const char* kOzoneCenter         = "ozoneCenter";
constexpr const char* kOzoneWidth          = "ozoneWidth";
constexpr const char* kPlanetRadius        = "planetRadius";
constexpr const char* kAtmosphereThickness = "atmosphereThickness";
constexpr const char* kExtensions          = "extensions";
constexpr const char* kUri                 = "uri";

constexpr const char* kTypePlain    = "plain";
constexpr const char* kTypeGradient = "gradient";
constexpr const char* kTypePanorama = "panorama";
constexpr const char* kTypePhysical = "physical";

constexpr const char* kColor               = "color";
constexpr const char* kBottomColor         = "bottomColor";
constexpr const char* kHorizonColor        = "horizonColor";
constexpr const char* kTopColor            = "topColor";
constexpr const char* kBottomCurve         = "bottomCurve";
constexpr const char* kTopCurve            = "topCurve";
constexpr const char* kSunAngleMax         = "sunAngleMax";
constexpr const char* kSunCurve            = "sunCurve";
constexpr const char* kEquirectangular     = "equirectangular";
constexpr const char* kGroundColor         = "groundColor";
constexpr const char* kMieColor            = "mieColor";
constexpr const char* kRayleighColor       = "rayleighColor";
constexpr const char* kMieAnisotropy       = "mieAnisotropy";
constexpr const char* kMieCoefficient      = "mieCoefficient";
constexpr const char* kRayleighCoefficient = "rayleighCoefficient";

//--------------------------------------------------------------------------------------------------
// The per-type sub-object is keyed by the type name itself ("plain": { ... }).
//
const char* typeName(SkyDescriptor::Type type)
{
  switch(type)
  {
    case SkyDescriptor::Type::ePlain:
      return kTypePlain;
    case SkyDescriptor::Type::eGradient:
      return kTypeGradient;
    case SkyDescriptor::Type::ePanorama:
      return kTypePanorama;
    case SkyDescriptor::Type::ePhysical:
      return kTypePhysical;
    default:
      assert(false && "unhandled SkyDescriptor::Type");
      return kTypePlain;
  }
}

//--------------------------------------------------------------------------------------------------
// Unknown or missing type strings fall back to plain rather than failing the load: a viewer that
// refuses to open a scene because it does not recognize one sky type is worse than one that shows
// a default sky. fromValue() records an unknown string in `unknownType`, which is what keeps the
// original JSON intact for the next save.
//
SkyDescriptor::Type parseType(const std::string& name)
{
  if(name == kTypeGradient)
    return SkyDescriptor::Type::eGradient;
  if(name == kTypePanorama)
    return SkyDescriptor::Type::ePanorama;
  if(name == kTypePhysical)
    return SkyDescriptor::Type::ePhysical;
  if(name != kTypePlain && !name.empty())
    LOGW("OMI_environment_sky: unknown sky type \"%s\"; treating it as plain\n", name.c_str());
  return SkyDescriptor::Type::ePlain;
}

//--------------------------------------------------------------------------------------------------
// Ensures `parent[key]` is an object and returns it, so the caller can merge fields into whatever
// the file already had there instead of replacing the whole sub-object.
//
tinygltf::Value& ensureObject(tinygltf::Value& parent, const std::string& key)
{
  tinygltf::Value::Object& obj = parent.Get<tinygltf::Value::Object>();
  auto                     it  = obj.find(key);
  if(it == obj.end() || !it->second.IsObject())
    it = obj.insert_or_assign(key, tinygltf::Value(tinygltf::Value::Object())).first;
  return it->second;
}

}  // namespace

//--------------------------------------------------------------------------------------------------
//
EnvironmentState parse(const tinygltf::Model& model, int sceneIndex)
{
  const tinygltf::Value* root = tinygltf::utils::findExtension(model.extensions, kExtensionName);
  if(root == nullptr || !root->Has(kSkies))
    return std::nullopt;

  const tinygltf::Value& skiesValue = root->Get(kSkies);
  if(!skiesValue.IsArray() || skiesValue.ArrayLen() == 0)
    return std::nullopt;

  // Which sky this scene uses. The per-scene reference is optional and defaults to 0.
  int skyIndex = 0;
  if(sceneIndex >= 0 && sceneIndex < static_cast<int>(model.scenes.size()))
  {
    if(const tinygltf::Value* sceneExt = tinygltf::utils::findExtension(model.scenes[sceneIndex].extensions, kExtensionName))
      tinygltf::utils::getValue(*sceneExt, kSky, skyIndex);
  }

  const int skyCount = static_cast<int>(skiesValue.ArrayLen());
  if(skyIndex < 0 || skyIndex >= skyCount)
  {
    LOGW("OMI_environment_sky: scene references sky %d of %d; using sky 0\n", skyIndex, skyCount);
    skyIndex = 0;
  }
  if(skyCount > 1)
  {
    LOGI("OMI_environment_sky: file carries %d skies; using %d. The others are kept in the file only if it is saved without the sky toggle.\n",
         skyCount, skyIndex);
  }

  return fromValue(skiesValue.Get(skyIndex));
}

//--------------------------------------------------------------------------------------------------
//
EnvironmentState fromValue(const tinygltf::Value& entry)
{
  if(!entry.IsObject())
    return std::nullopt;

  SkyDescriptor sky;
  sky.raw = entry;  // Everything, including what we do not model. Written back out on save.

  std::string typeString;
  tinygltf::utils::getValue(entry, kType, typeString);
  sky.type = parseType(typeString);
  if(!typeString.empty() && typeString != typeName(sky.type))
    sky.unknownType = typeString;

  // glTF writes a quaternion x, y, z, w; glm's constructor takes w first. A zero-length quaternion
  // is not a rotation, so it is treated as if the key had been absent rather than normalised into
  // a division by zero.
  if(entry.Has(kRotation))
  {
    glm::vec4 q{0.0f, 0.0f, 0.0f, 1.0f};
    tinygltf::utils::getArrayValue(entry, kRotation, q);
    const float len = glm::length(q);
    if(len > 0.0f)
    {
      sky.rotation         = glm::quat(q.w / len, q.x / len, q.y / len, q.z / len);
      sky.rotationAuthored = true;
    }
    else
    {
      LOGW("OMI_environment_sky: `rotation` is a zero-length quaternion; ignoring it.\n");
    }
  }

  if(sky.type == SkyDescriptor::Type::ePanorama && entry.Has(kExtensions))
  {
    const tinygltf::Value& extensions = entry.Get(kExtensions);
    if(extensions.Has(kPanoramaExtension))
      tinygltf::utils::getValue(extensions.Get(kPanoramaExtension), kUri, sky.panorama.uri);
  }

  // The atmosphere overlay is read after the OMI block below sets the scattering, and describes
  // only what OMI has no field for -- the two are disjoint, so there is nothing to win or lose on
  // overlap. Each key is optional: a block naming three of them leaves the other seven alone.
  if(sky.type == SkyDescriptor::Type::ePhysical && entry.Has(kExtensions))
  {
    const tinygltf::Value& extensions = entry.Get(kExtensions);
    if(extensions.Has(kAtmosphereExtension))
    {
      const tinygltf::Value& atmo = extensions.Get(kAtmosphereExtension);
      sky.atmosphere.present      = true;
      tinygltf::utils::getArrayValue(atmo, kSolarIrradiance, sky.atmosphere.solarIrradiance);
      tinygltf::utils::getArrayValue(atmo, kOzoneExtinction, sky.atmosphere.ozoneExtinction);
      tinygltf::utils::getValue(atmo, kSunAngularRadius, sky.atmosphere.sunAngularRadius);
      tinygltf::utils::getValue(atmo, kRayleighScaleHeight, sky.atmosphere.rayleighScaleHeight);
      tinygltf::utils::getValue(atmo, kMieScaleHeight, sky.atmosphere.mieScaleHeight);
      tinygltf::utils::getValue(atmo, kMieAlbedo, sky.atmosphere.mieAlbedo);
      tinygltf::utils::getValue(atmo, kOzoneCenter, sky.atmosphere.ozoneCenter);
      tinygltf::utils::getValue(atmo, kOzoneWidth, sky.atmosphere.ozoneWidth);
      tinygltf::utils::getValue(atmo, kPlanetRadius, sky.atmosphere.planetRadius);
      tinygltf::utils::getValue(atmo, kAtmosphereThickness, sky.atmosphere.atmosphereThickness);
    }
  }

  // Only the block matching the declared type is read. A file may carry several (an exporter that
  // remembers the user's last gradient while the active type is plain); those stay in `raw`.
  if(entry.Has(typeName(sky.type)))
  {
    const tinygltf::Value& params = entry.Get(typeName(sky.type));
    switch(sky.type)
    {
      case SkyDescriptor::Type::ePlain:
        tinygltf::utils::getArrayValue(params, kColor, sky.plain.color);
        break;
      case SkyDescriptor::Type::eGradient:
        tinygltf::utils::getArrayValue(params, kBottomColor, sky.gradient.bottomColor);
        tinygltf::utils::getArrayValue(params, kHorizonColor, sky.gradient.horizonColor);
        tinygltf::utils::getArrayValue(params, kTopColor, sky.gradient.topColor);
        tinygltf::utils::getValue(params, kBottomCurve, sky.gradient.bottomCurve);
        tinygltf::utils::getValue(params, kTopCurve, sky.gradient.topCurve);
        tinygltf::utils::getValue(params, kSunAngleMax, sky.gradient.sunAngleMax);
        tinygltf::utils::getValue(params, kSunCurve, sky.gradient.sunCurve);
        break;
      case SkyDescriptor::Type::ePanorama:
        tinygltf::utils::getValue(params, kEquirectangular, sky.panorama.equirectangular);
        break;
      case SkyDescriptor::Type::ePhysical:
        tinygltf::utils::getArrayValue(params, kGroundColor, sky.physical.groundColor);
        tinygltf::utils::getArrayValue(params, kMieColor, sky.physical.mieColor);
        tinygltf::utils::getArrayValue(params, kRayleighColor, sky.physical.rayleighColor);
        tinygltf::utils::getValue(params, kMieAnisotropy, sky.physical.mieAnisotropy);
        tinygltf::utils::getValue(params, kMieCoefficient, sky.physical.mieCoefficient);
        tinygltf::utils::getValue(params, kRayleighCoefficient, sky.physical.rayleighCoefficient);
        break;
      default:
        assert(false && "unhandled SkyDescriptor::Type");
        break;
    }
  }

  return sky;
}

//--------------------------------------------------------------------------------------------------
//
int findSkySunNode(const tinygltf::Model& model)
{
  for(size_t i = 0; i < model.nodes.size(); ++i)
  {
    if(hasSkySunMarker(model.nodes[i]))
      return static_cast<int>(i);
  }
  return -1;
}

//--------------------------------------------------------------------------------------------------
//
glm::quat localRotationForWorld(const glm::mat4& parentWorld, const glm::quat& worldRotation)
{
  // The rotation part alone: scale is divided out of each axis. A degenerate (zero-scale) parent
  // has no rotation to undo, so the world value is the best there is.
  glm::mat3 axes(parentWorld);
  for(int c = 0; c < 3; ++c)
  {
    const float len = glm::length(axes[c]);
    if(len < 1e-6F)
      return worldRotation;
    axes[c] /= len;
  }
  return glm::inverse(glm::quat_cast(axes)) * worldRotation;
}

//--------------------------------------------------------------------------------------------------
//
glm::quat localRotationForWorld(const tinygltf::Model& model, int nodeIndex, const glm::quat& worldRotation)
{
  // Walk up by searching `children`: a bare model stores no parent link. Runs once per save.
  // Bounded by the node count so a malformed cycle cannot spin forever.
  glm::mat4 parentWorld(1.0F);
  int       child = nodeIndex;
  for(size_t depth = 0; depth < model.nodes.size(); ++depth)
  {
    int parent = -1;
    for(size_t i = 0; i < model.nodes.size() && parent < 0; ++i)
    {
      const std::vector<int>& children = model.nodes[i].children;
      if(std::find(children.begin(), children.end(), child) != children.end())
        parent = static_cast<int>(i);
    }
    if(parent < 0)
      break;
    parentWorld = tinygltf::utils::getNodeMatrix(model.nodes[parent]) * parentWorld;
    child       = parent;
  }
  return localRotationForWorld(parentWorld, worldRotation);
}

//--------------------------------------------------------------------------------------------------
//
tinygltf::Value toValue(const SkyDescriptor& sky)
{
  // Start from what the file carried so unmodelled keys survive, and overwrite only what we model.
  tinygltf::Value entry = sky.raw.IsObject() ? sky.raw : tinygltf::Value(tinygltf::Value::Object());

  // The orientation applies to any type, recognized or not. Only written when stated: a sky that
  // said nothing about it must not come back saying "identity" (see rotationAuthored).
  // `raw` can still hold one the descriptor does not (a zero-length quaternion the loader ignored),
  // so an unstated rotation is removed rather than passed through.
  if(sky.rotationAuthored)
  {
    const glm::vec4 q{sky.rotation.x, sky.rotation.y, sky.rotation.z, sky.rotation.w};
    tinygltf::utils::setArrayValue(entry, kRotation, 4, glm::value_ptr(q));
  }
  else if(entry.IsObject())
  {
    entry.Get<tinygltf::Value::Object>().erase(kRotation);
  }

  // A type this build cannot render is not one it can describe either: the typed members hold only
  // the plain fallback shown in its place. Everything else is written back exactly as it was read.
  if(!sky.unknownType.empty())
    return entry;

  tinygltf::utils::setValue(entry, kType, std::string(typeName(sky.type)));
  // ambientLightColor / ambientSkyContribution are not written: this build does not implement them,
  // and a value it never applied is not a value it should claim. One the file arrived with is still
  // here, carried in `entry` from `sky.raw`.

  // Our two sibling blocks belong to one sky type each. `raw` carries whichever the file had, so a
  // sky loaded as physical and saved as a gradient would otherwise keep an atmosphere it no longer
  // has -- and advertise the extension for it. Dropped here; the matching type re-adds its own below
  // (the atmosphere only when `present`, so the block never outlives the flag that owns it).
  if(entry.Has(kExtensions) && entry.Get(kExtensions).IsObject())
  {
    tinygltf::Value::Object& extensions = entry.Get<tinygltf::Value::Object>()[kExtensions].Get<tinygltf::Value::Object>();
    if(sky.type != SkyDescriptor::Type::ePanorama)
      extensions.erase(kPanoramaExtension);
    if(sky.type != SkyDescriptor::Type::ePhysical || !sky.atmosphere.present)
      extensions.erase(kAtmosphereExtension);
    if(extensions.empty())
      entry.Get<tinygltf::Value::Object>().erase(kExtensions);
  }

  if(sky.type == SkyDescriptor::Type::ePanorama && !sky.panorama.uri.empty())
  {
    tinygltf::Value& extensions = ensureObject(entry, kExtensions);
    tinygltf::Value& panorama   = ensureObject(extensions, kPanoramaExtension);
    tinygltf::utils::setValue(panorama, kUri, sky.panorama.uri);
  }

  // Written beside a physical sky that has one, never for any other type: the block describes an
  // atmosphere, and a scene whose sky is a gradient has none. A physical sky loaded without the
  // block and saved unchanged stays without it -- materializing Earth here would make the next
  // load force Earth over whatever the renderer had. Written in full rather than only where it
  // differs from Earth -- a partial block would make "absent" and "default" indistinguishable on
  // the next load, which is the one thing `present` exists to tell apart.
  if(sky.type == SkyDescriptor::Type::ePhysical && sky.atmosphere.present)
  {
    tinygltf::Value& extensions = ensureObject(entry, kExtensions);
    tinygltf::Value& atmo       = ensureObject(extensions, kAtmosphereExtension);
    tinygltf::utils::setArrayValue(atmo, kSolarIrradiance, 3, glm::value_ptr(sky.atmosphere.solarIrradiance));
    tinygltf::utils::setArrayValue(atmo, kOzoneExtinction, 3, glm::value_ptr(sky.atmosphere.ozoneExtinction));
    tinygltf::utils::setValue(atmo, kSunAngularRadius, sky.atmosphere.sunAngularRadius);
    tinygltf::utils::setValue(atmo, kRayleighScaleHeight, sky.atmosphere.rayleighScaleHeight);
    tinygltf::utils::setValue(atmo, kMieScaleHeight, sky.atmosphere.mieScaleHeight);
    tinygltf::utils::setValue(atmo, kMieAlbedo, sky.atmosphere.mieAlbedo);
    tinygltf::utils::setValue(atmo, kOzoneCenter, sky.atmosphere.ozoneCenter);
    tinygltf::utils::setValue(atmo, kOzoneWidth, sky.atmosphere.ozoneWidth);
    tinygltf::utils::setValue(atmo, kPlanetRadius, sky.atmosphere.planetRadius);
    tinygltf::utils::setValue(atmo, kAtmosphereThickness, sky.atmosphere.atmosphereThickness);
  }

  tinygltf::Value& params = ensureObject(entry, typeName(sky.type));
  switch(sky.type)
  {
    case SkyDescriptor::Type::ePlain:
      tinygltf::utils::setArrayValue(params, kColor, 3, glm::value_ptr(sky.plain.color));
      break;
    case SkyDescriptor::Type::eGradient:
      tinygltf::utils::setArrayValue(params, kBottomColor, 3, glm::value_ptr(sky.gradient.bottomColor));
      tinygltf::utils::setArrayValue(params, kHorizonColor, 3, glm::value_ptr(sky.gradient.horizonColor));
      tinygltf::utils::setArrayValue(params, kTopColor, 3, glm::value_ptr(sky.gradient.topColor));
      tinygltf::utils::setValue(params, kBottomCurve, sky.gradient.bottomCurve);
      tinygltf::utils::setValue(params, kTopCurve, sky.gradient.topCurve);
      tinygltf::utils::setValue(params, kSunAngleMax, sky.gradient.sunAngleMax);
      tinygltf::utils::setValue(params, kSunCurve, sky.gradient.sunCurve);
      break;
    case SkyDescriptor::Type::ePanorama:
      // Only written when the file genuinely carried a texture index; this renderer never makes one.
      if(sky.panorama.equirectangular >= 0)
        tinygltf::utils::setValue(params, kEquirectangular, sky.panorama.equirectangular);
      break;
    case SkyDescriptor::Type::ePhysical:
      tinygltf::utils::setArrayValue(params, kGroundColor, 3, glm::value_ptr(sky.physical.groundColor));
      tinygltf::utils::setArrayValue(params, kMieColor, 3, glm::value_ptr(sky.physical.mieColor));
      tinygltf::utils::setArrayValue(params, kRayleighColor, 3, glm::value_ptr(sky.physical.rayleighColor));
      tinygltf::utils::setValue(params, kMieAnisotropy, sky.physical.mieAnisotropy);
      tinygltf::utils::setValue(params, kMieCoefficient, sky.physical.mieCoefficient);
      tinygltf::utils::setValue(params, kRayleighCoefficient, sky.physical.rayleighCoefficient);
      break;
    default:
      assert(false && "unhandled SkyDescriptor::Type");
      break;
  }

  return entry;
}

//--------------------------------------------------------------------------------------------------
//
void write(tinygltf::Model& outModel, int sceneIndex, const SkyDescriptor& sky)
{
  // Exactly one authored sky, at index 0 -- see the single-sky decision. Any additional skies the
  // file had are not re-emitted; that loss is deliberate and documented.
  tinygltf::Value::Array skies{toValue(sky)};
  tinygltf::Value&       root = tinygltf::utils::ensureExtension(outModel.extensions, kExtensionName);
  tinygltf::utils::setValue(root, kSkies, tinygltf::Value(skies));

  // Every other scene's reference is dropped rather than kept: with one sky left, an index it held
  // may now point past the end. Absent means sky 0, which is the only sky there is.
  for(size_t i = 0; i < outModel.scenes.size(); ++i)
  {
    if(static_cast<int>(i) == sceneIndex)
    {
      tinygltf::Value& sceneExt = tinygltf::utils::ensureExtension(outModel.scenes[i].extensions, kExtensionName);
      tinygltf::utils::setValue(sceneExt, kSky, 0);
    }
    else
    {
      outModel.scenes[i].extensions.erase(kExtensionName);
    }
  }

  // extensionsUsed maintains itself: syncExtensionsUsed already walks scenes[i].extensions.
}

//--------------------------------------------------------------------------------------------------
//
void strip(tinygltf::Model& outModel)
{
  outModel.extensions.erase(kExtensionName);
  for(tinygltf::Scene& scene : outModel.scenes)
    scene.extensions.erase(kExtensionName);
}

}  // namespace gltf_environment_sky
