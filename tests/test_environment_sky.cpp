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

// CPU-only round-trip tests for the OMI_environment_sky serializer.
//
// The class of bug these exist to catch is silent: a property that is read under one name and
// written under another, or a sky type this build does not render being quietly dropped, both
// produce a file that loads without error and has simply lost the author's data. Nothing in the
// rendering path would notice. No Vulkan device is needed, so this runs under the normal test
// target rather than the GPU benchmark harness.

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>

#include "gltf_environment_sky.hpp"

namespace {

// Wraps a sky entry in the document/scene structure the parser expects.
tinygltf::Model modelWithSky(const tinygltf::Value& skyEntry)
{
  tinygltf::Model model;
  model.scenes.emplace_back();
  model.defaultScene = 0;

  tinygltf::Value::Array  skies{skyEntry};
  tinygltf::Value::Object root;
  root["skies"]                                          = tinygltf::Value(skies);
  model.extensions[gltf_environment_sky::kExtensionName] = tinygltf::Value(root);

  tinygltf::Value::Object sceneExt;
  sceneExt["sky"]                                                  = tinygltf::Value(0);
  model.scenes[0].extensions[gltf_environment_sky::kExtensionName] = tinygltf::Value(sceneExt);
  return model;
}

// Save `sky` into a fresh model and read it straight back, which is exactly what a save/reopen
// cycle does to it.
SkyDescriptor roundTrip(const SkyDescriptor& sky)
{
  tinygltf::Model outModel;
  outModel.scenes.emplace_back();
  outModel.defaultScene = 0;

  gltf_environment_sky::write(outModel, 0, sky);
  EnvironmentState reloaded = gltf_environment_sky::parse(outModel, 0);
  EXPECT_TRUE(reloaded.has_value());
  return reloaded.value_or(SkyDescriptor{});
}

constexpr float kEps = 1e-6f;

}  // namespace

TEST(EnvironmentSky, AbsentExtensionParsesToNullopt)
{
  tinygltf::Model model;
  model.scenes.emplace_back();
  model.defaultScene = 0;
  EXPECT_FALSE(gltf_environment_sky::parse(model, 0).has_value());
}

TEST(EnvironmentSky, PlainRoundTripsEveryField)
{
  SkyDescriptor sky;
  sky.type        = SkyDescriptor::Type::ePlain;
  sky.plain.color = {0.25f, 0.5f, 0.75f};

  const SkyDescriptor back = roundTrip(sky);
  EXPECT_EQ(back.type, SkyDescriptor::Type::ePlain);
  EXPECT_NEAR(back.plain.color.x, 0.25f, kEps);
  EXPECT_NEAR(back.plain.color.y, 0.5f, kEps);
  EXPECT_NEAR(back.plain.color.z, 0.75f, kEps);
}

// The ambient terms are not implemented (see gltf_environment_sky.hpp), and this is the contract
// that replaces implementing them: a file that carries them still carries them afterwards. They
// ride the `raw` passthrough like any other key this build does not model, so authoring them in a
// tool that does support them survives a trip through this renderer.
TEST(EnvironmentSky, UnsupportedAmbientTermsSurviveUntouched)
{
  tinygltf::Value::Array  ambient{tinygltf::Value(0.1), tinygltf::Value(0.2), tinygltf::Value(0.3)};
  tinygltf::Value::Object entry;
  entry["type"]                   = tinygltf::Value(std::string("plain"));
  entry["ambientLightColor"]      = tinygltf::Value(ambient);
  entry["ambientSkyContribution"] = tinygltf::Value(0.6);

  const EnvironmentState parsed = gltf_environment_sky::fromValue(tinygltf::Value(entry));
  ASSERT_TRUE(parsed.has_value());

  const tinygltf::Value written = gltf_environment_sky::toValue(*parsed);
  ASSERT_TRUE(written.Has("ambientSkyContribution"));
  EXPECT_NEAR(written.Get("ambientSkyContribution").GetNumberAsDouble(), 0.6, 1e-9);
  ASSERT_TRUE(written.Has("ambientLightColor"));
  ASSERT_EQ(written.Get("ambientLightColor").ArrayLen(), 3u);
  EXPECT_NEAR(written.Get("ambientLightColor").Get(2).GetNumberAsDouble(), 0.3, 1e-9);
}

// A scene that says nothing about orientation must not be read as saying "identity": the renderer's
// rotation control is viewer state, and applySkyDescriptor only overrides it when the file states
// one. See the finding this test pins.
TEST(EnvironmentSky, AbsentRotationIsNotAuthored)
{
  tinygltf::Value::Object entry;
  entry["type"] = tinygltf::Value(std::string("plain"));

  const EnvironmentState parsed = gltf_environment_sky::fromValue(tinygltf::Value(entry));
  ASSERT_TRUE(parsed.has_value());
  EXPECT_FALSE(parsed->rotationAuthored);
}

TEST(EnvironmentSky, AuthoredRotationIsFlagged)
{
  tinygltf::Value::Array  rot{tinygltf::Value(0.0), tinygltf::Value(1.0), tinygltf::Value(0.0), tinygltf::Value(0.0)};
  tinygltf::Value::Object entry;
  entry["type"]     = tinygltf::Value(std::string("plain"));
  entry["rotation"] = tinygltf::Value(rot);

  const EnvironmentState parsed = gltf_environment_sky::fromValue(tinygltf::Value(entry));
  ASSERT_TRUE(parsed.has_value());
  EXPECT_TRUE(parsed->rotationAuthored);
  EXPECT_NEAR(parsed->rotation.y, 1.0f, kEps);
}

// A zero-length quaternion is not a rotation. Treated as absent rather than normalised, which would
// be a division by zero.
TEST(EnvironmentSky, ZeroLengthRotationIsIgnored)
{
  tinygltf::Value::Array  rot{tinygltf::Value(0.0), tinygltf::Value(0.0), tinygltf::Value(0.0), tinygltf::Value(0.0)};
  tinygltf::Value::Object entry;
  entry["type"]     = tinygltf::Value(std::string("plain"));
  entry["rotation"] = tinygltf::Value(rot);

  const EnvironmentState parsed = gltf_environment_sky::fromValue(tinygltf::Value(entry));
  ASSERT_TRUE(parsed.has_value());
  EXPECT_FALSE(parsed->rotationAuthored);
  EXPECT_NEAR(parsed->rotation.w, 1.0f, kEps);
}

TEST(EnvironmentSky, RotationRoundTripsAsAQuaternion)
{
  SkyDescriptor sky;
  sky.type             = SkyDescriptor::Type::ePlain;
  sky.rotation         = glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
  sky.rotationAuthored = true;

  const SkyDescriptor back = roundTrip(sky);
  EXPECT_NEAR(back.rotation.w, sky.rotation.w, kEps);
  EXPECT_NEAR(back.rotation.x, sky.rotation.x, kEps);
  EXPECT_NEAR(back.rotation.y, sky.rotation.y, kEps);
  EXPECT_NEAR(back.rotation.z, sky.rotation.z, kEps);
}

// A file that omits `rotation` means "not turned", not "undefined": the identity quaternion is
// what a spec-compliant reader assumes.
TEST(EnvironmentSky, AbsentRotationIsIdentity)
{
  tinygltf::Value::Object entry;
  entry["type"] = tinygltf::Value(std::string("plain"));

  const EnvironmentState parsed = gltf_environment_sky::fromValue(tinygltf::Value(entry));
  ASSERT_TRUE(parsed.has_value());
  EXPECT_NEAR(parsed->rotation.w, 1.0f, kEps);
  EXPECT_NEAR(parsed->rotation.x, 0.0f, kEps);
  EXPECT_NEAR(parsed->rotation.y, 0.0f, kEps);
  EXPECT_NEAR(parsed->rotation.z, 0.0f, kEps);
}

TEST(EnvironmentSky, GradientRoundTripsEveryField)
{
  SkyDescriptor sky;
  sky.type                  = SkyDescriptor::Type::eGradient;
  sky.gradient.bottomColor  = {0.01f, 0.02f, 0.03f};
  sky.gradient.horizonColor = {0.11f, 0.12f, 0.13f};
  sky.gradient.topColor     = {0.21f, 0.22f, 0.23f};
  sky.gradient.bottomCurve  = 0.031f;
  sky.gradient.topCurve     = 0.161f;
  sky.gradient.sunAngleMax  = 0.71f;
  sky.gradient.sunCurve     = 0.081f;

  const SkyDescriptor back = roundTrip(sky);
  EXPECT_EQ(back.type, SkyDescriptor::Type::eGradient);
  EXPECT_NEAR(back.gradient.bottomColor.x, 0.01f, kEps);
  EXPECT_NEAR(back.gradient.horizonColor.y, 0.12f, kEps);
  EXPECT_NEAR(back.gradient.topColor.z, 0.23f, kEps);
  EXPECT_NEAR(back.gradient.bottomCurve, 0.031f, kEps);
  EXPECT_NEAR(back.gradient.topCurve, 0.161f, kEps);
  EXPECT_NEAR(back.gradient.sunAngleMax, 0.71f, kEps);
  EXPECT_NEAR(back.gradient.sunCurve, 0.081f, kEps);
}

TEST(EnvironmentSky, PanoramaRoundTripsTheUri)
{
  SkyDescriptor sky;
  sky.type         = SkyDescriptor::Type::ePanorama;
  sky.panorama.uri = "../env/studio.exr";

  const SkyDescriptor back = roundTrip(sky);
  EXPECT_EQ(back.type, SkyDescriptor::Type::ePanorama);
  EXPECT_EQ(back.panorama.uri, "../env/studio.exr");
  // OMI's own field is absent, and must stay absent rather than being invented as 0 -- index 0 is
  // a valid texture, so a reader would load the wrong image rather than fall back.
  EXPECT_LT(back.panorama.equirectangular, 0);
}

TEST(EnvironmentSky, PanoramaKeepsAnAuthoredTextureIndex)
{
  // A file that legitimately uses OMI's texture index must not lose it just because this renderer
  // prefers the URI.
  SkyDescriptor sky;
  sky.type                     = SkyDescriptor::Type::ePanorama;
  sky.panorama.equirectangular = 3;
  sky.panorama.uri             = "sky.hdr";

  const SkyDescriptor back = roundTrip(sky);
  EXPECT_EQ(back.panorama.equirectangular, 3);
  EXPECT_EQ(back.panorama.uri, "sky.hdr");
}

TEST(EnvironmentSky, PanoramaWithoutUriWritesNoExtensionBlock)
{
  // An empty URI must not produce an NV_environment_sky_panorama block with an empty string in it:
  // a reader would treat that as "the author said the image is at ''" rather than "unspecified".
  SkyDescriptor sky;
  sky.type = SkyDescriptor::Type::ePanorama;

  tinygltf::Model model;
  model.scenes.resize(1);
  model.defaultScene = 0;
  gltf_environment_sky::write(model, 0, sky);

  const auto it = model.extensions.find(gltf_environment_sky::kExtensionName);
  ASSERT_NE(it, model.extensions.end());
  const tinygltf::Value& entry = it->second.Get("skies").Get(0);
  if(entry.Has("extensions"))
    EXPECT_FALSE(entry.Get("extensions").Has("NV_environment_sky_panorama"));
}

TEST(EnvironmentSky, PhysicalRoundTripsEveryField)
{
  SkyDescriptor sky;
  sky.type                         = SkyDescriptor::Type::ePhysical;
  sky.physical.groundColor         = {0.11f, 0.22f, 0.33f};
  sky.physical.mieColor            = {0.9f, 0.8f, 0.7f};
  sky.physical.rayleighColor       = {0.31f, 0.51f, 0.99f};
  sky.physical.mieAnisotropy       = 0.76f;
  sky.physical.mieCoefficient      = 0.0000051f;
  sky.physical.rayleighCoefficient = 0.000031f;

  const SkyDescriptor back = roundTrip(sky);
  EXPECT_EQ(back.type, SkyDescriptor::Type::ePhysical);
  EXPECT_NEAR(back.physical.groundColor.z, 0.33f, kEps);
  EXPECT_NEAR(back.physical.mieColor.x, 0.9f, kEps);
  EXPECT_NEAR(back.physical.rayleighColor.y, 0.51f, kEps);
  EXPECT_NEAR(back.physical.mieAnisotropy, 0.76f, kEps);
  EXPECT_NEAR(back.physical.mieCoefficient, 0.0000051f, kEps);
  EXPECT_NEAR(back.physical.rayleighCoefficient, 0.000031f, kEps);
}

// "Absent" must survive a save as absent. A sky that stated no rotation and no atmosphere block is
// written without them, so the next load leaves the renderer's own orientation and planet alone
// instead of forcing identity and Earth.
TEST(EnvironmentSky, AbsentRotationAndAtmosphereAreNotMaterialized)
{
  tinygltf::Value::Object entry;
  entry["type"] = tinygltf::Value(std::string("physical"));

  const EnvironmentState parsed = gltf_environment_sky::fromValue(tinygltf::Value(entry));
  ASSERT_TRUE(parsed.has_value());
  ASSERT_FALSE(parsed->rotationAuthored);
  ASSERT_FALSE(parsed->atmosphere.present);

  const tinygltf::Value written = gltf_environment_sky::toValue(*parsed);
  EXPECT_FALSE(written.Has("rotation"));
  EXPECT_FALSE(written.Has("extensions") && written.Get("extensions").Has("NV_environment_sky_atmosphere"));

  // Stated, they are written.
  SkyDescriptor stated      = *parsed;
  stated.rotationAuthored   = true;
  stated.atmosphere.present = true;
  const tinygltf::Value out = gltf_environment_sky::toValue(stated);
  EXPECT_TRUE(out.Has("rotation"));
  ASSERT_TRUE(out.Has("extensions"));
  EXPECT_TRUE(out.Get("extensions").Has("NV_environment_sky_atmosphere"));
}

// `raw` keeps what the file carried, including a key the descriptor chose not to honor. A
// zero-length rotation is ignored on load (rotationAuthored stays false), so it must not reappear on
// save -- nor may an atmosphere block survive a descriptor whose `present` says there is none.
TEST(EnvironmentSky, StaleRawRotationAndAtmosphereAreDropped)
{
  tinygltf::Value::Array  zero{tinygltf::Value(0.0), tinygltf::Value(0.0), tinygltf::Value(0.0), tinygltf::Value(0.0)};
  tinygltf::Value::Object atmo;
  atmo["planetRadius"] = tinygltf::Value(3389500.0);
  tinygltf::Value::Object extensions;
  extensions["NV_environment_sky_atmosphere"] = tinygltf::Value(atmo);

  tinygltf::Value::Object entry;
  entry["type"]       = tinygltf::Value(std::string("physical"));
  entry["rotation"]   = tinygltf::Value(zero);
  entry["extensions"] = tinygltf::Value(extensions);

  EnvironmentState parsed = gltf_environment_sky::fromValue(tinygltf::Value(entry));
  ASSERT_TRUE(parsed.has_value());
  ASSERT_FALSE(parsed->rotationAuthored);
  EXPECT_FALSE(gltf_environment_sky::toValue(*parsed).Has("rotation"));

  parsed->atmosphere.present    = false;
  const tinygltf::Value written = gltf_environment_sky::toValue(*parsed);
  EXPECT_FALSE(written.Has("extensions") && written.Get("extensions").Has("NV_environment_sky_atmosphere"));
}

// The gate on the `raw` passthrough. A sky carrying a vendor sub-extension and a property this
// build has no field for must come back byte-identical; without the passthrough it is stripped
// the first time the scene is opened and saved, with nothing to indicate the loss.
TEST(EnvironmentSky, UnmodelledKeysSurviveRoundTrip)
{
  tinygltf::Value::Object vendor;
  vendor["atmoGroundRadius"] = tinygltf::Value(6360.5);
  vendor["vendorOnlyBlob"]   = tinygltf::Value(std::string("must-survive-round-trip"));

  tinygltf::Value::Object extensions;
  extensions["NV_environment_sky_atmosphere"] = tinygltf::Value(vendor);

  tinygltf::Value::Object physical;
  physical["mieAnisotropy"] = tinygltf::Value(0.76);

  tinygltf::Value::Object entry;
  entry["type"]                      = tinygltf::Value(std::string("physical"));
  entry["physical"]                  = tinygltf::Value(physical);
  entry["extensions"]                = tinygltf::Value(extensions);
  entry["someFutureKeyWeDoNotModel"] = tinygltf::Value(std::string("keep me"));

  const tinygltf::Model  model  = modelWithSky(tinygltf::Value(entry));
  const EnvironmentState parsed = gltf_environment_sky::parse(model, 0);
  ASSERT_TRUE(parsed.has_value());

  tinygltf::Model outModel;
  outModel.scenes.emplace_back();
  outModel.defaultScene = 0;
  gltf_environment_sky::write(outModel, 0, *parsed);

  const tinygltf::Value& written = outModel.extensions.at(gltf_environment_sky::kExtensionName).Get("skies").Get(0);

  ASSERT_TRUE(written.Has("someFutureKeyWeDoNotModel"));
  EXPECT_EQ(written.Get("someFutureKeyWeDoNotModel").Get<std::string>(), "keep me");

  ASSERT_TRUE(written.Has("extensions"));
  const tinygltf::Value& vendorBack = written.Get("extensions").Get("NV_environment_sky_atmosphere");
  ASSERT_TRUE(vendorBack.Has("vendorOnlyBlob"));
  EXPECT_EQ(vendorBack.Get("vendorOnlyBlob").Get<std::string>(), "must-survive-round-trip");
  EXPECT_NEAR(vendorBack.Get("atmoGroundRadius").Get<double>(), 6360.5, 1e-9);
}

// Switching the authored type must not leave the previous type's block claiming to be current:
// `type` selects which block a reader honors, and the stale block stays only as preserved data.
TEST(EnvironmentSky, ChangingTypeRewritesTypeAndKeepsPriorBlock)
{
  SkyDescriptor sky;
  sky.type              = SkyDescriptor::Type::eGradient;
  sky.gradient.topColor = {0.5f, 0.6f, 0.7f};

  tinygltf::Model outModel;
  outModel.scenes.emplace_back();
  outModel.defaultScene = 0;
  gltf_environment_sky::write(outModel, 0, sky);

  EnvironmentState reloaded = gltf_environment_sky::parse(outModel, 0);
  ASSERT_TRUE(reloaded.has_value());

  // Author now switches to plain; the gradient block came back in `raw` and rides along.
  SkyDescriptor asPlain = *reloaded;
  asPlain.type          = SkyDescriptor::Type::ePlain;
  asPlain.plain.color   = {1.0f, 0.0f, 0.0f};

  tinygltf::Model outModel2;
  outModel2.scenes.emplace_back();
  outModel2.defaultScene = 0;
  gltf_environment_sky::write(outModel2, 0, asPlain);

  const tinygltf::Value& written = outModel2.extensions.at(gltf_environment_sky::kExtensionName).Get("skies").Get(0);
  EXPECT_EQ(written.Get("type").Get<std::string>(), "plain");
  EXPECT_TRUE(written.Has("gradient"));  // preserved, but no longer selected

  const EnvironmentState back = gltf_environment_sky::parse(outModel2, 0);
  ASSERT_TRUE(back.has_value());
  EXPECT_EQ(back->type, SkyDescriptor::Type::ePlain);
  EXPECT_NEAR(back->plain.color.x, 1.0f, kEps);
}

// The per-scene reference is what selects the sky; index 0 is only the default.
// A sky type this build does not render loads as its plain fallback, but must not be saved as one:
// the file's own `type` and its block go back out exactly as they came in.
TEST(EnvironmentSky, UnknownTypeSurvivesRoundTrip)
{
  tinygltf::Value::Object future;
  future["density"] = tinygltf::Value(0.25);

  tinygltf::Value::Object entry;
  entry["type"]       = tinygltf::Value(std::string("volumetric"));
  entry["volumetric"] = tinygltf::Value(future);

  const EnvironmentState parsed = gltf_environment_sky::fromValue(tinygltf::Value(entry));
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->type, SkyDescriptor::Type::ePlain);
  EXPECT_EQ(parsed->unknownType, "volumetric");

  const tinygltf::Value written = gltf_environment_sky::toValue(*parsed);
  EXPECT_EQ(written.Get("type").Get<std::string>(), "volumetric");
  EXPECT_FALSE(written.Has("plain"));
  ASSERT_TRUE(written.Has("volumetric"));
  EXPECT_NEAR(written.Get("volumetric").Get("density").GetNumberAsDouble(), 0.25, 1e-9);

  // Once a supported type is chosen (the renderer clears unknownType), it is written as such.
  SkyDescriptor chosen = *parsed;
  chosen.unknownType.clear();
  chosen.type = SkyDescriptor::Type::eGradient;
  EXPECT_EQ(gltf_environment_sky::toValue(chosen).Get("type").Get<std::string>(), "gradient");
}

TEST(EnvironmentSky, SceneReferenceSelectsTheSky)
{
  tinygltf::Value::Object first;
  first["type"] = tinygltf::Value(std::string("plain"));

  tinygltf::Value::Object secondPlain;
  secondPlain["color"] =
      tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(1.0), tinygltf::Value(0.0), tinygltf::Value(0.0)});
  tinygltf::Value::Object second;
  second["type"]  = tinygltf::Value(std::string("plain"));
  second["plain"] = tinygltf::Value(secondPlain);

  tinygltf::Model model;
  model.scenes.emplace_back();
  model.defaultScene = 0;

  tinygltf::Value::Array  skies{tinygltf::Value(first), tinygltf::Value(second)};
  tinygltf::Value::Object root;
  root["skies"]                                          = tinygltf::Value(skies);
  model.extensions[gltf_environment_sky::kExtensionName] = tinygltf::Value(root);

  tinygltf::Value::Object sceneExt;
  sceneExt["sky"]                                                  = tinygltf::Value(1);
  model.scenes[0].extensions[gltf_environment_sky::kExtensionName] = tinygltf::Value(sceneExt);

  const EnvironmentState parsed = gltf_environment_sky::parse(model, 0);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_NEAR(parsed->plain.color.x, 1.0f, kEps);
}

// An out-of-range reference is a broken file, not a crash: fall back to sky 0 and keep going.
TEST(EnvironmentSky, OutOfRangeSceneReferenceFallsBackToFirstSky)
{
  tinygltf::Value::Object plain;
  plain["color"] = tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(0.0), tinygltf::Value(1.0), tinygltf::Value(0.0)});
  tinygltf::Value::Object entry;
  entry["type"]  = tinygltf::Value(std::string("plain"));
  entry["plain"] = tinygltf::Value(plain);

  tinygltf::Model model = modelWithSky(tinygltf::Value(entry));
  // Point the scene at a sky that does not exist.
  tinygltf::Value::Object sceneExt;
  sceneExt["sky"]                                                  = tinygltf::Value(7);
  model.scenes[0].extensions[gltf_environment_sky::kExtensionName] = tinygltf::Value(sceneExt);

  const EnvironmentState parsed = gltf_environment_sky::parse(model, 0);
  ASSERT_TRUE(parsed.has_value());
  EXPECT_NEAR(parsed->plain.color.y, 1.0f, kEps);
}

// Strip has to clear both the document-level skies and every scene reference, or a re-save leaves
// a scene pointing at an extension that is no longer there.
TEST(EnvironmentSky, StripRemovesDocumentAndSceneEntries)
{
  tinygltf::Value::Object entry;
  entry["type"] = tinygltf::Value(std::string("plain"));

  tinygltf::Model model = modelWithSky(tinygltf::Value(entry));
  gltf_environment_sky::strip(model);

  EXPECT_EQ(model.extensions.count(gltf_environment_sky::kExtensionName), 0u);
  EXPECT_EQ(model.scenes[0].extensions.count(gltf_environment_sky::kExtensionName), 0u);
  EXPECT_FALSE(gltf_environment_sky::parse(model, 0).has_value());
}

// A sun under a rotated, scaled parent: the local rotation has the parent's turn taken out (and
// only its turn -- scale must not leak into a rotation). The model-walking overload, used by the
// save hook, must agree with the parent-matrix one the live scene uses.
TEST(EnvironmentSky, LocalRotationForWorldUndoesParentRotationOnly)
{
  const glm::quat parentRot = glm::angleAxis(glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
  const glm::quat world     = glm::angleAxis(glm::radians(30.0f), glm::vec3(1.0f, 0.0f, 0.0f));

  tinygltf::Model model;
  model.nodes.resize(2);
  model.nodes[0].rotation = {parentRot.x, parentRot.y, parentRot.z, parentRot.w};
  model.nodes[0].scale    = {2.0, 2.0, 2.0};
  model.nodes[0].children = {1};

  const glm::mat4 parentWorld = glm::mat4_cast(parentRot) * glm::scale(glm::mat4(1.0f), glm::vec3(2.0f));
  const glm::quat fromMatrix  = gltf_environment_sky::localRotationForWorld(parentWorld, world);
  const glm::quat fromModel   = gltf_environment_sky::localRotationForWorld(model, 1, world);

  const glm::quat expected = glm::inverse(parentRot) * world;
  EXPECT_NEAR(std::abs(glm::dot(fromMatrix, expected)), 1.0f, 1e-5f);
  EXPECT_NEAR(std::abs(glm::dot(fromModel, expected)), 1.0f, 1e-5f);

  // A root node has nothing to undo.
  EXPECT_NEAR(std::abs(glm::dot(gltf_environment_sky::localRotationForWorld(model, 0, world), world)), 1.0f, 1e-5f);
}
