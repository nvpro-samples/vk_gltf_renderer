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

//--------------------------------------------------------------------------------------------------
// NOAA solar position (solcalc). See sun_position.hpp for why this algorithm and what the
// conventions are.
//
// The variable names and the order of the steps follow NOAA's published spreadsheet so the two can
// be read side by side; the magic constants are that document's, not tuning. Everything is in
// double precision because the Julian century is a small difference between large numbers and the
// polynomials below amplify it.
//

#include <cctype>
#include <cmath>
#include <ctime>
#include <sstream>

#include "sun_position.hpp"

namespace {

constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;
constexpr double kRad2Deg = 180.0 / 3.14159265358979323846;

// Julian day for a Gregorian calendar date at 00:00 UTC. Fliegel & Van Flandern, which is exact in
// integer arithmetic for any proleptic Gregorian date -- no month-length table, no leap-year cases.
double julianDayAtMidnight(int year, int month, int day)
{
  // January and February are counted as months 13 and 14 of the previous year, which is what makes
  // the leap day fall at the end of the shifted year and lets the formula be uniform.
  if(month <= 2)
  {
    year -= 1;
    month += 12;
  }
  const double a = std::floor(year / 100.0);
  const double b = 2.0 - a + std::floor(a / 4.0);
  return std::floor(365.25 * (year + 4716)) + std::floor(30.6001 * (month + 1)) + day + b - 1524.5;
}

// Name comparison for a value typed on a command line: case is noise, and so is whether the user
// wrote "New York", "new_york" or "newyork".
bool equalsLoosely(const char* candidate, const std::string& typed)
{
  const auto significant = [](unsigned char c) { return std::isalnum(c) != 0; };
  const auto fold        = [](unsigned char c) { return static_cast<char>(std::tolower(c)); };

  size_t a = 0, b = 0;
  while(true)
  {
    while(candidate[a] != '\0' && !significant(static_cast<unsigned char>(candidate[a])))
      ++a;
    while(b < typed.size() && !significant(static_cast<unsigned char>(typed[b])))
      ++b;

    const bool endA = candidate[a] == '\0';
    const bool endB = b >= typed.size();
    if(endA || endB)
      return endA && endB;
    if(fold(static_cast<unsigned char>(candidate[a])) != fold(static_cast<unsigned char>(typed[b])))
      return false;
    ++a;
    ++b;
  }
}

}  // namespace

namespace sun_position {

// Ordered north to south. Names are plain ASCII on purpose -- Tromso and Reykjavik lose a diacritic
// each, which costs less than depending on what glyphs the UI font happens to carry.
const City kCities[] = {
    {"Tromso", 69.65F, 18.96F, 1.0F},      {"Reykjavik", 64.15F, -21.94F, 0.0F},
    {"Moscow", 55.76F, 37.62F, 3.0F},      {"Berlin", 52.52F, 13.40F, 1.0F},
    {"London", 51.51F, -0.13F, 0.0F},      {"Paris", 48.86F, 2.35F, 1.0F},
    {"Zurich", 47.37F, 8.54F, 1.0F},       {"New York", 40.71F, -74.01F, -5.0F},
    {"Beijing", 39.90F, 116.41F, 8.0F},    {"San Francisco", 37.77F, -122.42F, -8.0F},
    {"Tokyo", 35.68F, 139.69F, 9.0F},      {"Cairo", 30.04F, 31.24F, 2.0F},
    {"Dubai", 25.20F, 55.27F, 4.0F},       {"Mexico City", 19.43F, -99.13F, -6.0F},
    {"Mumbai", 19.08F, 72.88F, 5.5F},      {"Singapore", 1.35F, 103.82F, 8.0F},
    {"Nairobi", -1.29F, 36.82F, 3.0F},     {"Rio de Janeiro", -22.91F, -43.17F, -3.0F},
    {"Sydney", -33.87F, 151.21F, 10.0F},   {"Cape Town", -33.92F, 18.42F, 2.0F},
    {"Auckland", -36.85F, 174.76F, 12.0F}, {"Ushuaia", -54.80F, -68.30F, -3.0F},
};

const int kCityCount = static_cast<int>(sizeof(kCities) / sizeof(kCities[0]));

int findCity(const std::string& name)
{
  for(int i = 0; i < kCityCount; ++i)
  {
    if(equalsLoosely(kCities[i].name, name))
      return i;
  }
  return -1;
}

int cityAt(float latitudeDeg, float longitudeDeg)
{
  // A hundredth of a degree is about a kilometre, which is far below anything the sun's position
  // can distinguish and far above the rounding a float round trip costs.
  constexpr float kEpsilon = 0.01F;
  for(int i = 0; i < kCityCount; ++i)
  {
    if(std::fabs(kCities[i].latitudeDeg - latitudeDeg) < kEpsilon && std::fabs(kCities[i].longitudeDeg - longitudeDeg) < kEpsilon)
      return i;
  }
  return -1;
}

LocalClock systemLocalClock()
{
  LocalClock        out;
  const std::time_t now = std::time(nullptr);

  std::tm local{};
  std::tm utc{};
#if defined(_WIN32)
  if(localtime_s(&local, &now) != 0 || gmtime_s(&utc, &now) != 0)
    return out;
#else
  if(localtime_r(&now, &local) == nullptr || gmtime_r(&now, &utc) == nullptr)
    return out;
#endif

  char buffer[16] = {};
  if(std::strftime(buffer, sizeof(buffer), "%Y-%m-%d", &local) == 0)
    return out;

  // The offset as the difference of the two broken-down times rather than a platform time-zone
  // query: mktime re-reads a tm as local, so feeding it the UTC one yields an instant shifted by
  // exactly the offset -- daylight saving included, because it is today's offset being measured
  // and not a rule being looked up.
  std::tm localCopy             = local;
  std::tm utcCopy               = utc;
  localCopy.tm_isdst            = -1;
  utcCopy.tm_isdst              = -1;
  const std::time_t utcAsLocal  = std::mktime(&utcCopy);
  const std::time_t localAsItIs = std::mktime(&localCopy);
  if(utcAsLocal == static_cast<std::time_t>(-1) || localAsItIs == static_cast<std::time_t>(-1))
    return out;

  out.valid = true;
  out.date  = buffer;
  out.hour = static_cast<float>(local.tm_hour) + static_cast<float>(local.tm_min) / 60.0F + static_cast<float>(local.tm_sec) / 3600.0F;
  out.utcOffsetHours = static_cast<float>(std::difftime(localAsItIs, utcAsLocal) / 3600.0);
  return out;
}

bool parseIsoDate(const std::string& text, int& year, int& month, int& day)
{
  int                y = 0, m = 0, d = 0;
  char               sep1 = 0, sep2 = 0;
  std::istringstream in(text);
  if(!(in >> y >> sep1 >> m >> sep2 >> d) || sep1 != '-' || sep2 != '-')
    return false;
  // Nothing may follow the day ("2026-06-21x" is not a date).
  if(in.peek() != std::char_traits<char>::eof())
    return false;
  if(m < 1 || m > 12)
    return false;
  const bool leap          = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
  const int  daysInMonth[] = {31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  if(d < 1 || d > daysInMonth[m - 1])
    return false;

  year  = y;
  month = m;
  day   = d;
  return true;
}

AzimuthElevation computeSunPosition(Location loc, DateTimeUTC utc)
{
  // Julian centuries since J2000.0. The hour goes in here rather than being applied later: every
  // orbital term below drifts with it, and at solstice latitudes the difference over a single day
  // is visible on the horizon.
  const double julianDay = julianDayAtMidnight(utc.year, utc.month, utc.day) + utc.hour / 24.0;
  const double t         = (julianDay - 2451545.0) / 36525.0;

  const double geomMeanLongSun = std::fmod(280.46646 + t * (36000.76983 + t * 0.0003032), 360.0);
  const double geomMeanAnomSun = 357.52911 + t * (35999.05029 - 0.0001537 * t);
  const double eccentricity    = 0.016708634 - t * (0.000042037 + 0.0000001267 * t);

  // Equation of the centre: the difference between the real elliptical orbit and the uniform one
  // the mean anomaly describes.
  const double sunEqOfCentre = std::sin(kDeg2Rad * geomMeanAnomSun) * (1.914602 - t * (0.004817 + 0.000014 * t))
                               + std::sin(kDeg2Rad * 2.0 * geomMeanAnomSun) * (0.019993 - 0.000101 * t)
                               + std::sin(kDeg2Rad * 3.0 * geomMeanAnomSun) * 0.000289;

  const double sunTrueLong = geomMeanLongSun + sunEqOfCentre;
  // Apparent longitude: true longitude corrected for aberration and the largest nutation term.
  const double sunAppLong = sunTrueLong - 0.00569 - 0.00478 * std::sin(kDeg2Rad * (125.04 - 1934.136 * t));

  const double meanObliqEcliptic = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0;
  const double obliqCorr         = meanObliqEcliptic + 0.00256 * std::cos(kDeg2Rad * (125.04 - 1934.136 * t));

  const double sunDeclin = kRad2Deg * std::asin(std::sin(kDeg2Rad * obliqCorr) * std::sin(kDeg2Rad * sunAppLong));

  // Equation of time, in minutes: how far the real sun runs ahead of or behind the clock.
  const double varY = std::tan(kDeg2Rad * obliqCorr / 2.0) * std::tan(kDeg2Rad * obliqCorr / 2.0);
  const double eqOfTime =
      4.0 * kRad2Deg
      * (varY * std::sin(2.0 * kDeg2Rad * geomMeanLongSun) - 2.0 * eccentricity * std::sin(kDeg2Rad * geomMeanAnomSun)
         + 4.0 * eccentricity * varY * std::sin(kDeg2Rad * geomMeanAnomSun) * std::cos(2.0 * kDeg2Rad * geomMeanLongSun)
         - 0.5 * varY * varY * std::sin(4.0 * kDeg2Rad * geomMeanLongSun)
         - 1.25 * eccentricity * eccentricity * std::sin(2.0 * kDeg2Rad * geomMeanAnomSun));

  // Apparent solar time at the observer's meridian, in minutes past its solar midnight. The input
  // is UTC, so the only longitude term is the observer's own -- there is no time-zone offset to
  // subtract here; the caller applies that before building the DateTimeUTC.
  double trueSolarTime = std::fmod(utc.hour * 60.0 + eqOfTime + 4.0 * loc.longitudeDeg, 1440.0);
  if(trueSolarTime < 0.0)
    trueSolarTime += 1440.0;

  // Hour angle: 0 at local solar noon, -180..180, four minutes to the degree.
  double hourAngle = trueSolarTime / 4.0 - 180.0;
  if(hourAngle < -180.0)
    hourAngle += 360.0;

  const double latRad  = kDeg2Rad * loc.latitudeDeg;
  const double declRad = kDeg2Rad * sunDeclin;
  const double haRad   = kDeg2Rad * hourAngle;
  double cosZenith     = std::sin(latRad) * std::sin(declRad) + std::cos(latRad) * std::cos(declRad) * std::cos(haRad);
  cosZenith            = cosZenith < -1.0 ? -1.0 : (cosZenith > 1.0 ? 1.0 : cosZenith);
  const double zenithRad = std::acos(cosZenith);
  const double elevation = 90.0 - kRad2Deg * zenithRad;

  // Azimuth from the spherical law of cosines, measured from North. NOAA writes it as an acos
  // against the meridian, which spans only half the compass, plus a branch on the hour angle for
  // which half of the day it is.
  //
  // The denominator vanishes only at the poles and with the sun exactly overhead, where the bearing
  // is genuinely undefined; report the value the afternoon branch approaches rather than a NaN.
  double       azimuth   = (loc.latitudeDeg >= 0.0) ? 180.0 : 0.0;
  const double sinZenith = std::sin(zenithRad);
  const double cosLat    = std::cos(latRad);
  if(sinZenith > 1e-9 && std::fabs(cosLat) > 1e-9)
  {
    double cosAz   = (std::sin(latRad) * cosZenith - std::sin(declRad)) / (cosLat * sinZenith);
    cosAz          = cosAz < -1.0 ? -1.0 : (cosAz > 1.0 ? 1.0 : cosAz);
    const double a = kRad2Deg * std::acos(cosAz);
    azimuth        = (hourAngle > 0.0) ? (a + 180.0) : (540.0 - a);
  }
  azimuth = std::fmod(azimuth, 360.0);
  if(azimuth < 0.0)
    azimuth += 360.0;

  return {static_cast<float>(azimuth), static_cast<float>(elevation)};
}

}  // namespace sun_position
