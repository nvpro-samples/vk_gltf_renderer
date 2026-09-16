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
// Parses an IESNA LM-63 (.ies) photometric-web file (EXT_lights_ies) into a normalized,
// azimuthally-averaged candela distribution: relative intensity per degree of vertical angle off
// the light's photometric (nadir) axis. Multiple horizontal (azimuth) planes are averaged into one
// 1D curve -- genuinely asymmetric fixtures lose their azimuthal variation, matching how most
// real-time engines treat IES profiles for axially-symmetric luminaires.
//

#include "ies_profile.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <string_view>

namespace {

// Splits the remainder of an IES file into whitespace-separated tokens. IES data (everything
// after the TILT= line) is free-form whitespace/newline separated, not fixed to one value per
// line, so consumers must tokenize rather than split lines.
class Tokenizer
{
public:
  Tokenizer(std::string_view text, size_t pos)
      : m_text(text)
      , m_pos(pos)
  {
  }

  bool nextToken(std::string_view& token)
  {
    while(m_pos < m_text.size() && std::isspace(static_cast<unsigned char>(m_text[m_pos])))
      m_pos++;
    if(m_pos >= m_text.size())
      return false;
    size_t start = m_pos;
    while(m_pos < m_text.size() && !std::isspace(static_cast<unsigned char>(m_text[m_pos])))
      m_pos++;
    token = m_text.substr(start, m_pos - start);
    return true;
  }

  bool nextFloat(float& value)
  {
    std::string_view token;
    if(!nextToken(token))
      return false;
    auto result = std::from_chars(token.data(), token.data() + token.size(), value);
    return result.ec == std::errc();
  }

  bool nextInt(int& value)
  {
    std::string_view token;
    if(!nextToken(token))
      return false;
    auto result = std::from_chars(token.data(), token.data() + token.size(), value);
    return result.ec == std::errc();
  }

private:
  std::string_view m_text;
  size_t           m_pos;
};

}  // namespace

//--------------------------------------------------------------------------------------------------
// Parses one IESNA LM-63 (.ies) photometric-web file (EXT_lights_ies) into the normalized, fixed-
// size candela curve described in ies_profile.hpp (see there for the exact output contract and its
// azimuthal-averaging / TILT-data simplifications).
//
// Downstream usage: SceneVk::loadIesProfiles() (gltf_scene_vk.cpp) calls this once per
// EXT_lights_ies profile the scene references and copies outSamples into a flat GPU buffer laid
// out as [profile][sample], one kIesProfileSampleCount-wide row per profile. GltfLight::iesProfile
// indexes a row; gltf_light_ies.h.slang's sampleIesProfile()/evalIesFactor() linearly interpolate
// along it at shading time, producing a [0,1] multiplier that reshapes a KHR_lights_punctual
// light's angular falloff -- the light's own intensity/color/range come from KHR_lights_punctual
// unchanged.
//--------------------------------------------------------------------------------------------------
bool nvvkgltf::parseIesProfile(std::span<const uint8_t> fileBytes, std::vector<float>& outSamples)
{
  std::string_view text(reinterpret_cast<const char*>(fileBytes.data()), fileBytes.size());

  // Find the TILT= line -- marks the end of the free-form header/keyword lines.
  size_t tiltPos = text.find("TILT=");
  if(tiltPos == std::string_view::npos)
    return false;

  size_t lineEnd = text.find_first_of("\r\n", tiltPos);
  if(lineEnd == std::string_view::npos)
    return false;
  std::string_view tiltValue = text.substr(tiltPos + 5, lineEnd - (tiltPos + 5));
  while(!tiltValue.empty() && std::isspace(static_cast<unsigned char>(tiltValue.back())))
    tiltValue.remove_suffix(1);

  Tokenizer tok(text, lineEnd);

  if(tiltValue == "INCLUDE")
  {
    // TILT=INCLUDE: inline block is lampToLuminaireGeometry (int), numPairs (int),
    // then numPairs angle values followed by numPairs multiplier values (floats).
    // Consume to stay positioned; tilt correction is not applied (see ies_profile.hpp).
    int lampToLuminaireGeometry = 0, numPairs = 0;
    if(!tok.nextInt(lampToLuminaireGeometry) || !tok.nextInt(numPairs) || numPairs < 0 || numPairs > 1024)
      return false;
    float dummy = 0.0f;
    for(int i = 0; i < numPairs * 2; i++)
    {
      if(!tok.nextFloat(dummy))
        return false;
    }
  }
  else if(tiltValue != "NONE")
  {
    // TILT=<filename>: tilt data lives in the named external file, not inline here.
    // Nothing to consume; photometric data follows immediately in the current stream.
  }

  int   numLamps = 0, numVerticalAngles = 0, numHorizontalAngles = 0, photometricType = 0, unitsType = 0;
  float lumensPerLamp = 0, candelaMultiplier = 0, width = 0, length = 0, height = 0;
  if(!tok.nextInt(numLamps) || !tok.nextFloat(lumensPerLamp) || !tok.nextFloat(candelaMultiplier)
     || !tok.nextInt(numVerticalAngles) || !tok.nextInt(numHorizontalAngles) || !tok.nextInt(photometricType)
     || !tok.nextInt(unitsType) || !tok.nextFloat(width) || !tok.nextFloat(length) || !tok.nextFloat(height))
    return false;

  // 1024 per axis is far beyond any real IES file (181 vertical × 361 horizontal is the physical
  // limit for 1-degree grids) and keeps the worst-case candela allocation under 4 MB.
  if(numVerticalAngles <= 0 || numHorizontalAngles <= 0 || numVerticalAngles > 1024 || numHorizontalAngles > 1024)
    return false;

  // Ballast factor, [ballast-lamp photometric factor | future use], input watts.
  float ballastFactor = 1.0f, unused = 0.0f, inputWatts = 0.0f;
  if(!tok.nextFloat(ballastFactor) || !tok.nextFloat(unused) || !tok.nextFloat(inputWatts))
    return false;

  std::vector<float> verticalAngles(numVerticalAngles);
  for(float& a : verticalAngles)
    if(!tok.nextFloat(a))
      return false;

  std::vector<float> horizontalAngles(numHorizontalAngles);
  for(float& a : horizontalAngles)
    if(!tok.nextFloat(a))
      return false;

  std::vector<float> candela(size_t(numHorizontalAngles) * size_t(numVerticalAngles));
  for(float& c : candela)
  {
    if(!tok.nextFloat(c))
      return false;
    c *= candelaMultiplier * ballastFactor;
  }

  // Resample each horizontal-angle row to a common 1-degree vertical grid, then average the rows
  // (azimuthal averaging -- see the "Simplification" note in ies_profile.hpp).
  outSamples.assign(nvvkgltf::kIesProfileSampleCount, 0.0f);
  for(int h = 0; h < numHorizontalAngles; h++)
  {
    const float* row = &candela[size_t(h) * numVerticalAngles];
    for(int i = 0; i < nvvkgltf::kIesProfileSampleCount; i++)
    {
      float thetaDeg = float(i);  // 0..180 degrees, 1-degree steps
      float value;
      if(thetaDeg <= verticalAngles.front())
      {
        value = row[0];
      }
      else if(thetaDeg >= verticalAngles.back())
      {
        value = row[numVerticalAngles - 1];
      }
      else
      {
        auto   it   = std::upper_bound(verticalAngles.begin(), verticalAngles.end(), thetaDeg);
        size_t hi   = size_t(it - verticalAngles.begin());
        size_t lo   = hi - 1;
        float  span = verticalAngles[hi] - verticalAngles[lo];
        float  t    = (span > 0.0f) ? (thetaDeg - verticalAngles[lo]) / span : 0.0f;
        value       = row[lo] + t * (row[hi] - row[lo]);
      }
      outSamples[i] += value;
    }
  }
  for(float& s : outSamples)
    s /= float(numHorizontalAngles);

  float peak = *std::max_element(outSamples.begin(), outSamples.end());
  if(peak > 0.0f)
  {
    for(float& s : outSamples)
      s /= peak;
  }

  return true;
}
