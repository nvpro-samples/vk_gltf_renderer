/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


float3 debugValue(PbrMaterial pbrMat, HitState hit, DebugMethod dbgMethod)
{
    switch (dbgMethod)
    {
    case DebugMethod::eBaseColor:
        return float3(pbrMat.baseColor);
    case DebugMethod::eMetallic:
        return float3(pbrMat.metallic);
    case DebugMethod::eRoughness:
        return float3(pbrMat.roughness.xyx);
    case DebugMethod::eNormal:
        return float3(pbrMat.N * .5 + .5);
    case DebugMethod::eTangent:
        return pbrMat.T * .5 + .5;
    case DebugMethod::eBitangent:
        return pbrMat.B * .5 + .5;
    case DebugMethod::eEmissive:
        return float3(pbrMat.emissive);
    case DebugMethod::eOpacity:
        return float3(pbrMat.opacity * (1.0 - pbrMat.transmission));
    case DebugMethod::eTexCoord0:
        return float3(hit.uv[0], 0);
    case DebugMethod::eTexCoord1:
        return float3(hit.uv[1], 0);
    }
    return float3(0);
}