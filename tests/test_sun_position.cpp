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

// Where the sun is: NOAA solcalc, checked against astronomy rather than against itself.
//
// This is deliberately not a regression test of previously-recorded output. Every gate this feature
// had through phase 4b compared the renderer with itself and passed while three calibration defects
// were live (docs/sky.md, tasks/plans/omi_environment_sky.md § Calibration). So each case here is a
// fact about the Earth that holds independently of this code: the sun's noon altitude is 90 minus
// latitude at equinox, it rises due east at the equator, it does not set at Tromso in June, it does
// not rise there in December, and it is in the *north* at Sydney noon.
//
// Tolerances are ~1 degree, which is looser than the algorithm (NOAA claims about an arc-minute) and
// looser than the refraction correction this port deliberately omits (up to 0.57 degrees at the
// horizon). That is the point: these assert the astronomy, not the last digit.

#include <cmath>

#include <gtest/gtest.h>

#include "sun_position.hpp"

using sun_position::AzimuthElevation;
using sun_position::computeSunPosition;
using sun_position::DateTimeUTC;
using sun_position::Location;

namespace {

// Signed smallest angle between two compass bearings, so 359 and 1 are 2 degrees apart.
float bearingDelta(float a, float b)
{
  float d = std::fmod(a - b + 540.0F, 360.0F) - 180.0F;
  return std::fabs(d);
}

}  // namespace

// Zurich, March equinox, *solar* noon. At equinox the declination is zero, so the zenith angle at
// noon is exactly the latitude and the altitude is 90 - latitude. Due south, by definition of noon.
//
// The times throughout this file are solar, not clock: a place is rarely on its time zone's
// meridian (Zurich's solar noon is ~34 minutes before CET noon) and the equation of time moves the
// rest. Converting the two is the widget's job, not this function's -- it takes UTC.
TEST(SunPosition, NoonZurichEquinox)
{
  const Location    zurich{47.37F, 8.54F};
  const DateTimeUTC solarNoon{2026, 3, 20, 11.554F};

  const AzimuthElevation sun = computeSunPosition(zurich, solarNoon);

  EXPECT_NEAR(sun.elevationDeg, 90.0F - zurich.latitudeDeg, 1.0F);
  EXPECT_LT(bearingDelta(sun.azimuthDeg, 180.0F), 1.0F);  // due south
}

// Nairobi, March equinox, sunrise. On an equinox the sun rises due east everywhere on Earth, and
// at the equator it does so six hours before solar noon -- the one moment whose geometry is simple
// enough to assert both numbers at once.
//
// Nairobi is 36.8 degrees east, so this is 03:40 UTC and not the 03:00 that its clock's 06:00
// would suggest. Kenya's zone meridian is 45 E; the city is not on it.
TEST(SunPosition, SunriseNairobiEquinox)
{
  const Location    nairobi{-1.29F, 36.82F};
  const DateTimeUTC solarSunrise{2026, 3, 20, 3.669F};

  const AzimuthElevation sun = computeSunPosition(nairobi, solarSunrise);

  EXPECT_NEAR(sun.elevationDeg, 0.0F, 1.0F);
  EXPECT_LT(bearingDelta(sun.azimuthDeg, 90.0F), 2.0F);  // due east
}

// Tromso, June solstice, solar midnight. Above the Arctic Circle the sun does not set: its altitude
// at its lowest point in the day is still positive. This is the case that catches an hour-angle
// wrap bug, since midnight is where the angle crosses +/-180.
TEST(SunPosition, MidnightSunTromsoSummerSolstice)
{
  const Location tromso{69.65F, 18.96F};
  // Solar midnight, 18.96 degrees east: 22:46 UTC, the low point of the 21st's day.
  const DateTimeUTC solarMidnight{2026, 6, 21, 22.766F};

  const AzimuthElevation sun = computeSunPosition(tromso, solarMidnight);

  EXPECT_GT(sun.elevationDeg, 0.0F);
  // 69.65 N is 3.1 degrees inside the Arctic Circle, so the midnight sun clears the horizon by
  // about that much; the declination cannot lift it further.
  EXPECT_LT(sun.elevationDeg, 6.0F);
  EXPECT_LT(bearingDelta(sun.azimuthDeg, 0.0F), 20.0F);  // north, where the midnight sun is
}

// Tromso, December solstice, solar noon. The other half of the same fact: inside the Arctic Circle
// the polar night means the sun stays below the horizon even at its daily high point.
TEST(SunPosition, PolarNightTromsoWinterSolstice)
{
  const Location    tromso{69.65F, 18.96F};
  const DateTimeUTC solarNoon{2026, 12, 21, 10.77F};

  const AzimuthElevation sun = computeSunPosition(tromso, solarNoon);

  EXPECT_LT(sun.elevationDeg, 0.0F);
  // Twilight, not deep night: the solstice sun sits a few degrees under the horizon at noon here.
  EXPECT_GT(sun.elevationDeg, -8.0F);
}

// Sydney, December solstice, solar noon. Southern hemisphere: the sun is high *and* in the north.
// A northern-hemisphere-only azimuth convention passes every test above and fails this one, which
// is exactly why it is here.
TEST(SunPosition, SouthernHemisphereSydneySummer)
{
  const Location sydney{-33.87F, 151.21F};
  // Solar noon: 151.21 degrees east is 10h05m ahead of Greenwich, so 01:53 UTC -- not the 01:00
  // that AEDT's clock noon gives, since the zone's meridian is 165 E.
  const DateTimeUTC solarNoon{2026, 12, 21, 1.891F};

  const AzimuthElevation sun = computeSunPosition(sydney, solarNoon);

  // Declination is -23.44 at the December solstice, so the zenith angle is |lat - decl| = 10.4.
  EXPECT_NEAR(sun.elevationDeg, 79.6F, 1.0F);
  EXPECT_LT(bearingDelta(sun.azimuthDeg, 0.0F), 1.0F);  // due north
}

// The city list is data, and the two things that can go wrong with data are a typo in an entry and
// a lookup that does not find it. Both are cheap to pin.
TEST(SunPosition, CityListIsWellFormed)
{
  ASSERT_GT(sun_position::kCityCount, 0);

  for(int i = 0; i < sun_position::kCityCount; ++i)
  {
    const sun_position::City& city = sun_position::kCities[i];
    EXPECT_NE(city.name, nullptr);
    EXPECT_GE(city.latitudeDeg, -90.0F) << city.name;
    EXPECT_LE(city.latitudeDeg, 90.0F) << city.name;
    EXPECT_GE(city.longitudeDeg, -180.0F) << city.name;
    EXPECT_LE(city.longitudeDeg, 180.0F) << city.name;
    // Real offsets run -12..+14, and a transposed sign is the likeliest typo here.
    EXPECT_GE(city.utcOffsetHours, -12.0F) << city.name;
    EXPECT_LE(city.utcOffsetHours, 14.0F) << city.name;

    // The list claims to be ordered north to south, and the UI leans on that.
    if(i > 0)
      EXPECT_LE(city.latitudeDeg, sun_position::kCities[i - 1].latitudeDeg) << city.name;

    // Every entry has to be findable by its own name and by its own coordinates, or the panel
    // shows "Custom" for a city the user just picked.
    EXPECT_EQ(sun_position::findCity(city.name), i) << city.name;
    EXPECT_EQ(sun_position::cityAt(city.latitudeDeg, city.longitudeDeg), i) << city.name;
  }
}

// The command line is the caller that cannot see the list, so the lookup forgives case, spaces and
// punctuation -- and still refuses a place that is not on it.
TEST(SunPosition, CityLookupIgnoresCaseAndSpacing)
{
  const int newYork = sun_position::findCity("New York");
  ASSERT_GE(newYork, 0);
  EXPECT_EQ(sun_position::findCity("new york"), newYork);
  EXPECT_EQ(sun_position::findCity("NEWYORK"), newYork);
  EXPECT_EQ(sun_position::findCity("new_york"), newYork);
  EXPECT_EQ(sun_position::findCity("  New   York  "), newYork);

  EXPECT_LT(sun_position::findCity("Atlantis"), 0);
  EXPECT_LT(sun_position::findCity(""), 0);
  // A prefix is not a match: "New" must not silently become New York.
  EXPECT_LT(sun_position::findCity("New"), 0);
}

TEST(SunPosition, ParseIsoDateRejectsImpossibleDates)
{
  int y = 0, m = 0, d = 0;
  EXPECT_TRUE(sun_position::parseIsoDate("2026-06-21", y, m, d));
  EXPECT_EQ(y, 2026);
  EXPECT_EQ(m, 6);
  EXPECT_EQ(d, 21);
  EXPECT_TRUE(sun_position::parseIsoDate("2024-02-29", y, m, d));   // leap year
  EXPECT_TRUE(sun_position::parseIsoDate("2000-02-29", y, m, d));   // divisible by 400
  EXPECT_FALSE(sun_position::parseIsoDate("2026-02-29", y, m, d));  // not a leap year
  EXPECT_FALSE(sun_position::parseIsoDate("1900-02-29", y, m, d));  // century, not by 400
  EXPECT_FALSE(sun_position::parseIsoDate("2026-04-31", y, m, d));
  EXPECT_FALSE(sun_position::parseIsoDate("2026-13-01", y, m, d));
  EXPECT_FALSE(sun_position::parseIsoDate("2026-06-21x", y, m, d));  // trailing characters
  EXPECT_FALSE(sun_position::parseIsoDate("2026-06-21 ", y, m, d));
}
