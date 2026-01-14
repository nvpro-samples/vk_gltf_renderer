/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <tinygltf/tiny_gltf.h>

/*-------------------------------------------------------------------------------------------------
## Function `compactModel`

Removes orphaned data from a glTF model, reducing memory usage and file size.

After operations that create new accessors/buffer views (like MikkTSpace tangent generation
with vertex splitting), old data becomes orphaned - still present in buffers but no longer
referenced.

Returns true if any data was removed, false if model was already compact.
-------------------------------------------------------------------------------------------------*/
bool compactModel(tinygltf::Model& model);
