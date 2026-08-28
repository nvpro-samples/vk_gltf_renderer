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

#include <gtest/gtest.h>
#include "gltf_animation_pointer.hpp"
#include "tinygltf_utils.hpp"

// Regression test for a bug found via KHR_interactivity's Calculator.glb conformance scene:
// pointer/set writing a single field of a texture-info extension (e.g. KHR_texture_transform's
// "offset") used to silently drop sibling fields (scale, texCoord) that came from the original
// glTF and were never themselves written, because the texture-slot merge path replaced the whole
// extension entry instead of merging into it - unlike the material-level extensions merge just
// below it in the same function, which already merged correctly.
TEST(InteractivityAnimationPointerSystem, PartialTextureTransformWritePreservesSiblingFields)
{
  tinygltf::Model model;
  model.materials.emplace_back();
  tinygltf::Material& mat = model.materials[0];

  KHR_texture_transform tt;
  tt.offset = {0.1f, 0.0f};
  tt.scale  = {0.1f, 1.0f};
  tinygltf::utils::setTextureTransform(mat.pbrMetallicRoughness.baseColorTexture, tt);
  // setTextureTransform() intentionally never writes texCoord (see its doc comment) - author it
  // directly on the raw extension JSON to simulate a texCoord that came from the original glTF,
  // the same sibling-field-survival scenario as scale above.
  tinygltf::utils::setValue(tinygltf::utils::ensureExtension(mat.pbrMetallicRoughness.baseColorTexture.extensions,
                                                             KHR_TEXTURE_TRANSFORM_EXTENSION_NAME),
                            "texCoord", 1);

  nvvkgltf::AnimationPointerSystem pointer(model);
  ASSERT_TRUE(pointer.applyValue("/materials/0/pbrMetallicRoughness/baseColorTexture/extensions/KHR_texture_transform/offset",
                                 glm::vec2(0.8f, 0.0f)));
  pointer.syncToModel();

  KHR_texture_transform result = tinygltf::utils::getTextureTransform(mat.pbrMetallicRoughness.baseColorTexture);
  EXPECT_FLOAT_EQ(result.offset.x, 0.8f);
  EXPECT_FLOAT_EQ(result.offset.y, 0.0f);
  EXPECT_FLOAT_EQ(result.scale.x, 0.1f);  // must survive the offset-only write
  EXPECT_FLOAT_EQ(result.scale.y, 1.0f);
  EXPECT_EQ(result.texCoord, 1);  // must survive the offset-only write
}
