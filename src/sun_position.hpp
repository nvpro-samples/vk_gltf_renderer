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

//==================================================================================================
// WHERE THE SUN IS, FOR A PLACE AND A MOMENT
//==================================================================================================
//
// One pure function, so that "the sun at 17:47 on 15 June in Berlin" is a calculation rather than a
// pair of sliders somebody nudged until it looked right.
//
// The algorithm is NOAA's solar calculator (solcalc), accurate to about an arc-minute for the years
// 1800-2200: https://gml.noaa.gov/grad/solcalc/calcdetails.html. NREL's SPA is three orders of
// magnitude more accurate and two orders of magnitude more code; a sky dome cannot show the
// difference, so it is not worth carrying.
//
// Deliberately self-contained: no Vulkan, no glTF, no ImGui, no settings, no allocation, no state.
// Nothing else in the renderer links against it, so the Time of Day widget can be removed by
// deleting these two files and the panel section that calls them.
//
// Conventions:
//  - Azimuth is a compass bearing: degrees clockwise from North, so 0 = N, 90 = E, 180 = S, 270 = W.
//    Mapping that onto the renderer's own azimuth (measured from +X toward +Z) is the caller's job,
//    and is what the panel's North offset exists for.
//  - Elevation is the *geometric* altitude above the horizon, negative when the sun is down. NOAA's
//    web calculator additionally reports a refraction-corrected altitude, up to ~0.57 degrees higher
//    near the horizon. That correction is not applied here: it describes how an observer inside an
//    atmosphere sees the sun, which is the physical sky's own job, and applying it here would bend
//    the light twice.
//

#include <string>

namespace sun_position {

struct AzimuthElevation
{
  float azimuthDeg{};    // compass bearing, clockwise from North
  float elevationDeg{};  // geometric altitude above the horizon, negative below
};

struct Location
{
  float latitudeDeg{};   // +north
  float longitudeDeg{};  // +east
};

struct DateTimeUTC
{
  int   year{};
  int   month{};  // 1-12
  int   day{};    // 1-31
  float hour{};   // hours past UTC midnight: h + m/60 + s/3600
};

/// Sun position for a place and an instant. Pure: same inputs, same answer, no side effects.
AzimuthElevation computeSunPosition(Location loc, DateTimeUTC utc);

/// A named place, so the widget does not ask people to look up their own latitude.
///
/// Chosen for spread rather than for population: two inside the Arctic Circle, three on or near the
/// equator, five in the southern hemisphere, and one on a half-hour offset. That spread is the point
/// -- the sun behaves visibly differently at each, which is what a sky widget is for.
///
/// `utcOffsetHours` is **standard** time. There is no daylight-saving rule here and no time-zone
/// database in this renderer; a summer date at most of these is an hour off until the UTC Offset
/// slider beside the list is nudged, which is a smaller lie than shipping a zone table that goes
/// stale.
struct City
{
  const char* name;
  float       latitudeDeg;
  float       longitudeDeg;
  float       utcOffsetHours;
};

/// The list, ordered north to south. Ordering is itself information here: it is the axis the sun
/// cares about, so scrolling the list walks the sun's arc from the midnight sun to the far south.
extern const City kCities[];
extern const int  kCityCount;

/// Index of the city named `name`, or -1. Case- and space-insensitive, so `--todCity "new york"`,
/// `NewYork` and `new_york` all land: the command line is the caller that cannot see the list.
int findCity(const std::string& name);

/// Index of the city at this location, or -1 for somewhere else. Within about a kilometre.
///
/// Deliberately ignores the UTC offset, so adjusting it for daylight saving does not make the panel
/// forget which city you picked.
int cityAt(float latitudeDeg, float longitudeDeg);

/// What time it is on this machine, in the shape the fields above want.
///
/// Not astronomy, and the one thing here that touches the host: it lives beside the calculation
/// because "which moment" and "where is the sun at that moment" are the same question asked twice,
/// and because a widget that opens on an arbitrary epoch is a widget somebody has to correct before
/// it says anything true. `valid` is false if the platform refused to break the clock down, in which
/// case nothing else has been written.
struct LocalClock
{
  bool        valid{false};
  std::string date;              // yyyy-mm-dd, local
  float       hour{};            // local clock hours past midnight
  float       utcOffsetHours{};  // how far the local clock runs ahead of UTC, DST included
};
LocalClock systemLocalClock();

/// Read a "yyyy-mm-dd" date. False -- and the outputs untouched -- if it is not one.
///
/// Here rather than at the two call sites because both the panel and the settings callback need it
/// and they must agree about what counts as a date; the format is also what the command line and the
/// ini file carry.
bool parseIsoDate(const std::string& text, int& year, int& month, int& day);

}  // namespace sun_position
