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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iterator>

// Drag speed for a PE::DragFloat that should feel logarithmic: the step is always ~10% of the
// current order of magnitude, so small values change slowly and large values change quickly.
// Use as the v_speed argument:
//   PE::DragFloat("MyField", &val, logarithmicStep(val), 0.0f, FLT_MAX, "%.4g");
inline float logarithmicStep(float value)
{
  if(value <= 0.0f)
    return 0.001f;
  return 0.1f * std::pow(10.0f, std::floor(std::log10(value)));
}

// Number formatting for UI text. Each formatter returns a NumberString whose text lives in an inline
// buffer: no heap allocation, so it is cheap to call per row, per frame. The temporary lives until
// the end of the full expression:
//   ImGui::TextUnformatted(formatThousands(count).c_str());
struct NumberString
{
  char    buf[32]{};  // fits UINT64_MAX with separators: 20 digits + 6 commas + NUL
  uint8_t start = 0;  // an offset, not a pointer, so the object stays safely copyable

  const char* c_str() const { return buf + start; }
};

// Exact integer with thousands separators: 951130816 -> "951,130,816".
inline NumberString formatThousands(uint64_t value)
{
  NumberString s;
  char*        p = s.buf + sizeof(s.buf) - 1;  // buf is zero-initialized, so the last byte is the NUL
  for(int digits = 0;; ++digits)
  {
    if(digits > 0 && digits % 3 == 0)
      *--p = ',';
    *--p = char('0' + value % 10);
    value /= 10;
    if(value == 0)
      break;
  }
  s.start = uint8_t(p - s.buf);
  return s;
}

// Shared by formatCompact / formatBytes: divide by `base` until the value fits one unit. A unit
// switch happens when the value would *print* as `base` (999950 -> "1.0M", never "1000.0K").
// Unscaled values print as plain integers ("999", "512 B"). `decimals` is clamped to 1..3.
inline NumberString formatScaled(uint64_t value, double base, const char* const* units, int unitCount, const char* separator, int decimals)
{
  decimals            = std::clamp(decimals, 1, 3);
  const double carry  = base - 0.5 * std::pow(10.0, -decimals);  // smallest value that rounds to `base`
  double       scaled = double(value);
  int          unit   = 0;
  while(unit + 1 < unitCount && scaled >= carry)
  {
    scaled /= base;
    ++unit;
  }
  NumberString s;
  if(unit == 0)
    std::snprintf(s.buf, sizeof(s.buf), "%llu%s%s", static_cast<unsigned long long>(value), separator, units[0]);
  else
    std::snprintf(s.buf, sizeof(s.buf), "%.*f%s%s", decimals, scaled, separator, units[unit]);
  return s;
}

// Compact count for narrow cells and summaries: 1234 -> "1.2K", 12345678 -> "12.3M".
inline NumberString formatCompact(uint64_t value, int decimals = 1)
{
  static const char* const units[] = {"", "K", "M", "B", "T"};
  return formatScaled(value, 1000.0, units, int(std::size(units)), "", decimals);
}

// Memory size in binary units: 3565158 -> "3.4 MB".
inline NumberString formatBytes(uint64_t bytes, int decimals = 1)
{
  static const char* const units[] = {"B", "KB", "MB", "GB", "TB"};
  return formatScaled(bytes, 1024.0, units, int(std::size(units)), " ", decimals);
}
