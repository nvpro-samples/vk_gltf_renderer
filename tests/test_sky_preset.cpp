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

// `.sky.json` presets: does a sky survive leaving the scene and coming back?
//
// A preset is one entry of the OMI `skies[]` array written as its own file, so it shares the glTF
// serializer entirely and this file tests the layer around it -- the JSON text, the sun angle OMI
// has no field for, and the promise that keys this build does not model are carried through rather
// than dropped. That last one is the whole reason the round trip is worth a test: it is invisible
// until someone else's exporter writes a key we have never heard of.
//
// Phase 4b earned exact round-tripping through NV_environment_sky_atmosphere by bit-comparing 31
// leaf values. The same standard applies here: floats go out and come back as themselves, not
// approximately.

#include <filesystem>
#include <fstream>

#include <gtest/gtest.h>

#include "gltf_environment_sky.hpp"
#include "sky_preset.hpp"

namespace {

// A sky with every modelled block filled in with values that are not defaults, so a field the
// writer forgets shows up as a difference rather than as a coincidence.
SkyDescriptor makePhysicalSky()
{
  SkyDescriptor sky;
  sky.type = SkyDescriptor::Type::ePhysical;

  sky.physical.groundColor         = {0.4F, 0.35F, 0.3F};
  sky.physical.rayleighColor       = {0.3F, 0.5F, 1.0F};
  sky.physical.rayleighCoefficient = 3.1e-5F;
  sky.physical.mieColor            = {0.9F, 0.95F, 1.0F};
  sky.physical.mieCoefficient      = 5.2e-6F;
  sky.physical.mieAnisotropy       = 0.76F;

  sky.atmosphere.present             = true;
  sky.atmosphere.solarIrradiance     = {1.5F, 1.6F, 1.7F};
  sky.atmosphere.ozoneExtinction     = {6.5e-4F, 1.88e-3F, 8.5e-5F};
  sky.atmosphere.sunAngularRadius    = 0.004675F;
  sky.atmosphere.rayleighScaleHeight = 8.1F;
  sky.atmosphere.mieScaleHeight      = 1.25F;
  sky.atmosphere.mieAlbedo           = 0.91F;
  sky.atmosphere.ozoneCenter         = 25.3F;
  sky.atmosphere.ozoneWidth          = 15.7F;
  sky.atmosphere.planetRadius        = 6361.0F;
  sky.atmosphere.atmosphereThickness = 99.5F;
  return sky;
}

std::filesystem::path tempPresetPath(const char* name)
{
  return std::filesystem::temp_directory_path() / (std::string("vk_gltf_renderer_") + name + ".sky.json");
}

}  // namespace

// The whole point: a sky written out and read back is the same sky, exactly.
TEST(SkyPreset, PhysicalRoundTripsExactly)
{
  const SkyDescriptor original = makePhysicalSky();
  const auto          path     = tempPresetPath("physical");

  ASSERT_TRUE(sky_preset::save(path, {.sky = original}));
  const std::optional<sky_preset::Preset> loaded = sky_preset::load(path);
  ASSERT_TRUE(loaded.has_value());

  const SkyDescriptor& back = loaded->sky;
  EXPECT_EQ(int(back.type), int(original.type));

  EXPECT_EQ(back.physical.groundColor, original.physical.groundColor);
  EXPECT_EQ(back.physical.rayleighColor, original.physical.rayleighColor);
  EXPECT_EQ(back.physical.rayleighCoefficient, original.physical.rayleighCoefficient);
  EXPECT_EQ(back.physical.mieColor, original.physical.mieColor);
  EXPECT_EQ(back.physical.mieCoefficient, original.physical.mieCoefficient);
  EXPECT_EQ(back.physical.mieAnisotropy, original.physical.mieAnisotropy);

  ASSERT_TRUE(back.atmosphere.present);
  EXPECT_EQ(back.atmosphere.solarIrradiance, original.atmosphere.solarIrradiance);
  EXPECT_EQ(back.atmosphere.ozoneExtinction, original.atmosphere.ozoneExtinction);
  EXPECT_EQ(back.atmosphere.sunAngularRadius, original.atmosphere.sunAngularRadius);
  EXPECT_EQ(back.atmosphere.rayleighScaleHeight, original.atmosphere.rayleighScaleHeight);
  EXPECT_EQ(back.atmosphere.mieScaleHeight, original.atmosphere.mieScaleHeight);
  EXPECT_EQ(back.atmosphere.mieAlbedo, original.atmosphere.mieAlbedo);
  EXPECT_EQ(back.atmosphere.ozoneCenter, original.atmosphere.ozoneCenter);
  EXPECT_EQ(back.atmosphere.ozoneWidth, original.atmosphere.ozoneWidth);
  EXPECT_EQ(back.atmosphere.planetRadius, original.atmosphere.planetRadius);
  EXPECT_EQ(back.atmosphere.atmosphereThickness, original.atmosphere.atmosphereThickness);

  std::filesystem::remove(path);
}

// A preset without the sun angle is half the look it captured, so the angle travels -- and OMI has
// no field for it, which is why this is the one key the format adds.
TEST(SkyPreset, SunRotationSurvives)
{
  const glm::quat sun{0.9238795F, -0.3826834F, 0.0F, 0.0F};  // 45 degrees about X
  const auto      path = tempPresetPath("sun");

  ASSERT_TRUE(sky_preset::save(path, {.sky = makePhysicalSky(), .sunRotation = sun}));
  const std::optional<sky_preset::Preset> loaded = sky_preset::load(path);
  ASSERT_TRUE(loaded.has_value());
  ASSERT_TRUE(loaded->sunRotation.has_value());

  EXPECT_EQ(loaded->sunRotation->x, sun.x);
  EXPECT_EQ(loaded->sunRotation->y, sun.y);
  EXPECT_EQ(loaded->sunRotation->z, sun.z);
  EXPECT_EQ(loaded->sunRotation->w, sun.w);

  std::filesystem::remove(path);
}

// Absent is not zero. A gradient sky has no sun rotation worth writing, and a reader must be able
// to tell "the file said nothing" from "the file said identity" -- otherwise loading a preset
// silently swings the sun to a default nobody chose.
TEST(SkyPreset, AbsentSunRotationStaysAbsent)
{
  const auto path = tempPresetPath("nosun");
  ASSERT_TRUE(sky_preset::save(path, {.sky = makePhysicalSky()}));

  const std::optional<sky_preset::Preset> loaded = sky_preset::load(path);
  ASSERT_TRUE(loaded.has_value());
  EXPECT_FALSE(loaded->sunRotation.has_value());

  std::filesystem::remove(path);
}

// Someone else's exporter writes a key this build has never heard of. It has to come back out, or
// a preset is a lossy format for everyone but us.
TEST(SkyPreset, UnmodelledKeysSurvive)
{
  const auto path = tempPresetPath("unmodelled");

  {
    std::ofstream file(path);
    file << R"({
      "type": "gradient",
      "gradient": { "topColor": [0.1, 0.2, 0.3], "sunCurve": 0.25 },
      "somebodyElsesKey": { "nested": [1, 2, 3], "flag": true },
      "ambientSkyContribution": 0.5
    })";
  }

  const std::optional<sky_preset::Preset> loaded = sky_preset::load(path);
  ASSERT_TRUE(loaded.has_value());
  EXPECT_EQ(int(loaded->sky.type), int(SkyDescriptor::Type::eGradient));

  ASSERT_TRUE(sky_preset::save(path, *loaded));
  const std::optional<sky_preset::Preset> again = sky_preset::load(path);
  ASSERT_TRUE(again.has_value());

  // The stranger's key is still there, with its structure intact.
  const tinygltf::Value& raw = again->sky.raw;
  ASSERT_TRUE(raw.IsObject());
  ASSERT_TRUE(raw.Has("somebodyElsesKey"));
  const tinygltf::Value& theirs = raw.Get("somebodyElsesKey");
  ASSERT_TRUE(theirs.IsObject());
  ASSERT_TRUE(theirs.Has("nested"));
  EXPECT_EQ(theirs.Get("nested").ArrayLen(), 3u);
  ASSERT_TRUE(theirs.Has("flag"));
  EXPECT_TRUE(theirs.Get("flag").Get<bool>());

  // And what we do model came through unharmed alongside it.
  EXPECT_FLOAT_EQ(again->sky.gradient.sunCurve, 0.25F);
  // ambientSkyContribution is unmodelled now, so it is checked the way any stranger's key is:
  // it must still be in the document the preset round-tripped.
  ASSERT_TRUE(again->sky.raw.Has("ambientSkyContribution"));
  EXPECT_NEAR(again->sky.raw.Get("ambientSkyContribution").GetNumberAsDouble(), 0.5, 1e-9);

  std::filesystem::remove(path);
}

// A preset that silently does nothing is worse than one that fails, so every bad input is a
// nullopt rather than a default-constructed sky.
TEST(SkyPreset, BadInputFailsRatherThanGuessing)
{
  EXPECT_FALSE(sky_preset::load(tempPresetPath("does_not_exist_at_all")).has_value());

  const auto path = tempPresetPath("broken");
  {
    std::ofstream file(path);
    file << "{ this is not json";
  }
  EXPECT_FALSE(sky_preset::load(path).has_value());

  {
    std::ofstream file(path);
    file << "[1, 2, 3]";  // valid JSON, but not a sky
  }
  EXPECT_FALSE(sky_preset::load(path).has_value());

  std::filesystem::remove(path);
}

// The full `.sky.json`, because the viewport's drop handler already claims `.scene.json`. A rule
// of "any .json is a sky" would take those over, and only on the day someone drops one.
TEST(SkyPreset, RecognisesItsOwnFilesAndNotTheNeighbours)
{
  EXPECT_TRUE(sky_preset::isPresetPath("sunset.sky.json"));
  EXPECT_TRUE(sky_preset::isPresetPath("SUNSET.SKY.JSON"));
  EXPECT_TRUE(sky_preset::isPresetPath("/some/where/blue hour.sky.json"));

  EXPECT_FALSE(sky_preset::isPresetPath("city.scene.json"));  // the collision this guards
  EXPECT_FALSE(sky_preset::isPresetPath("anything.json"));
  EXPECT_FALSE(sky_preset::isPresetPath("scene.gltf"));
  EXPECT_FALSE(sky_preset::isPresetPath("studio.hdr"));
}
