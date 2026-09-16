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

#include <cmath>

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
