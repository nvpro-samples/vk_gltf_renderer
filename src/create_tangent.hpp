/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <tinygltf/tiny_gltf.h>

//--------------------------------------------------------------------------------------------------
// Recomputes tangent space vectors for all mesh primitives in a glTF model.
//
// Two methods are available:
//   - Simple (UV gradient): Fast, modifies tangent buffer in-place, no vertex splitting
//   - MikkTSpace: High-quality, may split vertices at UV seams or mirrored UVs
//
// @param model         The glTF model to modify (in-place)
// @param forceCreation If true, creates tangents even for primitives that already have them
// @param mikktspace    If true, uses MikkTSpace algorithm; otherwise uses simple UV gradient
//
// @return true if vertex splitting occurred (buffers grew), requiring full scene recreation:
//         - Destroy SceneVk/SceneRtx
//         - Call scene.setCurrentScene() to re-parse
//         - Call createVulkanScene()
//         - Update UI scene graph
//--------------------------------------------------------------------------------------------------
bool recomputeTangents(tinygltf::Model& model, bool forceCreation, bool mikktspace);
