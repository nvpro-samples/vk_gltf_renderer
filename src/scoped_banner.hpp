/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdio>
#include <cstring>
#include <string>

#include <nvutils/logger.hpp>
#include <nvutils/timers.hpp>

// RAII class for printing banners around major operations.
// Prints an opening banner with the operation name at construction,
// and a closing banner with elapsed time at destruction.
//
// Usage:
//   {
//     ScopedBanner banner("Tangent Generation");
//     // ... operation logs appear here ...
//   }  // Prints closing banner with elapsed time
//
// Output:
//   >>>>>>>>>>>>>>>>>>> Tangent Generation >>>>>>>>>>>>>>>>>>>
//   ... operation logs ...
//   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 1.234 ms <<<
class ScopedBanner
{
public:
  explicit ScopedBanner(const std::string& operationName)
  {
    // Create a banner like: >>>>>>>>>>>>>>>>>>> Operation Name >>>>>>>>>>>>>>>>>>>
    constexpr size_t totalWidth   = 70;
    constexpr char   openChar     = '>';
    const size_t     nameLen      = operationName.length() + 2;  // +2 for spaces around name
    const size_t     arrowsNeeded = (totalWidth > nameLen) ? (totalWidth - nameLen) : 4;
    const size_t     leftArrows   = arrowsNeeded / 2;
    const size_t     rightArrows  = arrowsNeeded - leftArrows;

    m_bannerWidth = leftArrows + nameLen + rightArrows;

    std::string banner(leftArrows, openChar);
    banner += ' ';
    banner += operationName;
    banner += ' ';
    banner += std::string(rightArrows, openChar);

    LOGI("%s\n", banner.c_str());
  }

  ~ScopedBanner()
  {
    // Create a closing banner like: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 1.234 ms <<<
    constexpr char closeChar = '<';

    char timeStr[32];
    snprintf(timeStr, sizeof(timeStr), " %.3f ms ", m_timer.getMilliseconds());
    const size_t timeLen     = strlen(timeStr);
    const size_t arrowsTotal = (m_bannerWidth > timeLen) ? (m_bannerWidth - timeLen) : 4;
    const size_t rightArrows = 3;
    const size_t leftArrows  = (arrowsTotal > rightArrows) ? (arrowsTotal - rightArrows) : 1;

    std::string banner(leftArrows, closeChar);
    banner += timeStr;
    banner += std::string(rightArrows, closeChar);

    LOGI("%s\n", banner.c_str());
  }

  // Non-copyable, non-movable
  ScopedBanner(const ScopedBanner&)            = delete;
  ScopedBanner& operator=(const ScopedBanner&) = delete;
  ScopedBanner(ScopedBanner&&)                 = delete;
  ScopedBanner& operator=(ScopedBanner&&)      = delete;

private:
  nvutils::PerformanceTimer m_timer;
  size_t                    m_bannerWidth = 0;
};

// Macro for consistent naming in scope
#define SCOPED_BANNER(name) ScopedBanner scopedBanner##__LINE__(name)
