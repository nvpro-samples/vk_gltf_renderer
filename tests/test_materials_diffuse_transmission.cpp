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

// Host-side contract of KHR_materials_diffuse_transmission: the color factor is read from and
// written to the spec's "diffuseTransmissionColorFactor" (earlier versions mistyped it as
// "diffuseTransmissionColor").
// Specification: https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_diffuse_transmission

#include <gtest/gtest.h>

#include "tinygltf_utils.hpp"

namespace {

tinygltf::Value vec3Value(float x, float y, float z)
{
  return tinygltf::Value(
      tinygltf::Value::Array{tinygltf::Value(double(x)), tinygltf::Value(double(y)), tinygltf::Value(double(z))});
}

}  // namespace

TEST(MaterialsDiffuseTransmission, ReadsSpecColorFactor)
{
  tinygltf::Material      mat;
  tinygltf::Value::Object ext;
  ext["diffuseTransmissionFactor"]                                  = tinygltf::Value(1.0);
  ext["diffuseTransmissionColorFactor"]                             = vec3Value(0.84f, 0.8f, 0.74f);
  mat.extensions[KHR_MATERIALS_DIFFUSE_TRANSMISSION_EXTENSION_NAME] = tinygltf::Value(std::move(ext));

  const KHR_materials_diffuse_transmission dt = tinygltf::utils::getDiffuseTransmission(mat);
  EXPECT_FLOAT_EQ(dt.diffuseTransmissionFactor, 1.0f);
  EXPECT_FLOAT_EQ(dt.diffuseTransmissionColorFactor.x, 0.84f);
  EXPECT_FLOAT_EQ(dt.diffuseTransmissionColorFactor.y, 0.8f);
  EXPECT_FLOAT_EQ(dt.diffuseTransmissionColorFactor.z, 0.74f);
}

TEST(MaterialsDiffuseTransmission, WritesSpecColorFactor)
{
  tinygltf::Material                 mat;
  KHR_materials_diffuse_transmission dt;
  dt.diffuseTransmissionColorFactor = glm::vec3(0.1f, 0.2f, 0.3f);
  tinygltf::utils::setDiffuseTransmission(mat, dt);

  const tinygltf::Value& written = mat.extensions[KHR_MATERIALS_DIFFUSE_TRANSMISSION_EXTENSION_NAME];
  EXPECT_TRUE(written.Has("diffuseTransmissionColorFactor"));
  EXPECT_FALSE(written.Has("diffuseTransmissionColor"));
  EXPECT_EQ(tinygltf::utils::getDiffuseTransmission(mat).diffuseTransmissionColorFactor, glm::vec3(0.1f, 0.2f, 0.3f));
}
