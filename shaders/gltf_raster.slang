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

#include "gltf_scene_io.h.slang"
#include "gltf_vertex_access.h.slang"
// Raster runs in a fragment shader, where implicit-LOD .Sample() has valid screen-space
// derivatives and gives correct hardware mipmapping. Opt into that path in getTexture();
// the path tracer (compute/raygen) has no such derivatives and uses explicit mip 0 instead.
#define RASTER_PIPELINE 1
#include "gltf_material_eval.h.slang"
#include "shaderio.h"
#ifdef HAS_DLSS_MOTION
#include "dlss_util.h"  // calculateMotionVector (clip-pos overload) - only when DLAA motion attachment is built in
#endif
#include "get_hit.h.slang"
#include "common.h.slang"

// NVSHADERS are under: https://github.com/nvpro-samples/nvpro_core2/tree/main/nvshaders
#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/bsdf_types.h.slang"
#include "nvshaders/functions.h.slang"
#include "nvshaders/hdr_env_sampling.h.slang"
#include "nvshaders/light_contrib.h.slang"
#include "nvshaders/pbr_ggx_microfacet.h.slang"
#include "nvshaders/pbr_direct_eval.h.slang"
#include "nvshaders/pbr_ibl_eval.h.slang"
#include "nvshaders/ray_utils.h.slang"
#include "nvshaders/sample_blur.h.slang"


// Bindings
// clang-format off
[[vk::push_constant]]                                       ConstantBuffer<RasterPushConstant>  pushConst;
[[vk::binding(BindingPoints::eTextures, 0)]]                Sampler2D                           allTextures[];
[[vk::binding(BindingPoints::eTexturesHdr, 0)]]             Sampler2D                           texturesHdr[];
[[vk::binding(BindingPoints::eTexturesCube, 0)]]            SamplerCube                         texturesCube[];
// clang-format on


// Define the vertex attributes that will be provided by the vertex buffer
// Note: In Slang, we use semantic annotations (: POSITION, etc.) to specify
// how these fields map to vertex buffer data
struct VertexInput
{
  [[vk::binding(0)]]
  float3 position : POSITION;  // Vertex position in model space
};

// Define the data that will be interpolated and passed from vertex to fragment shader
// Note: SV_Position is a special semantic in Slang/HLSL for the clip-space position
//
// currClipUnjit/prevClipUnjit are clip-space positions WITHOUT the DLSS sub-pixel jitter.
// They are used in the fragment shader to compute screen-space motion vectors that the
// DLSS network expects (NGX wants un-jittered MVs and de-jitters them itself via
// InJitterOffsetX/Y). When DLAA is disabled `position` and `currClipUnjit` are identical
// because frameInfo.jitter defaults to (0,0). The unjittered fields are only emitted in
// builds that ship the motion attachment (USE_DLSS=ON); USE_DLSS=OFF builds drop them so
// Slang's DCE removes the prevMVP multiply and the extra interpolators.
struct VertexOutput
{
  float4 position : SV_Position;  // Clip space position (jittered when DLAA on) - required
  float3 worldPos;
#ifdef HAS_DLSS_MOTION
  float4 currClipUnjit;  // Current-frame unjittered clip-space position
  float4 prevClipUnjit;  // Previous-frame unjittered clip-space position
#endif
};

// Define the final output of the fragment shader. Color attachments:
//   0 = shaded color (or DLAA input when enabled)
//   1 = render-node ID for silhouette/picking
//   2 = screen-space motion vector ([prev - curr] in normalized 0..1 screen UV) for DLAA -
//       ONLY in USE_DLSS=ON builds. USE_DLSS=OFF binaries drop this attachment entirely so
//       the rasterizer pipeline has 2 color outputs and the rasterizer's Dlss (Kind::SR)
//       instance never allocates a motion image. HAS_DLSS_MOTION is passed by CMake when USE_DLSS=ON.
struct PixelOutput
{
  float4 color : SV_TARGET0;
  float4 selection : SV_TARGET1;
#ifdef HAS_DLSS_MOTION
  float2 motion : SV_TARGET2;
#endif
};


// ---------- Screen-space transmission (KHR_materials_transmission, ibl.glsl:58-114) ----------
//
// `texturesHdr[HDR_OPAQUE_INDEX]` was filled by the rasterizer's first pass with the resolved
// opaque image downsized to OPAQUE_COLOR_SIZE x OPAQUE_COLOR_SIZE plus a full mip chain.
// Roughness drives the mip LOD via `applyIorToRoughness(roughness, ior) * log2(size)`, the same
// mapping the Khronos reference uses.

float3 getTransmissionSample(float2 uv, float roughness, float ior)
{
  float framebufferLod = log2(float(OPAQUE_COLOR_SIZE)) * applyIorToRoughness(roughness, ior);
  return texturesHdr[HDR_OPAQUE_INDEX].SampleLevel(uv, framebufferLod).rgb;
}

// Volume scatter (KHR_materials_volume_scatter) approximation: analytic single-fragment estimate
// of the disk-filter SSS when no scatter pre-pass is available. Mask matches Khronos pbr.frag:355
// `(1-metallic)(1-iridescence)(1-transmission)`; the clearcoat mask is applied by the caller.
float3 volumeScatterApprox(PbrMaterial mat, float3 hemisphericalIrradiance)
{
  if(max3(mat.scatterCoefficient) <= 0.0)
    return float3(0.0);
  float3 scattered = hemisphericalIrradiance * mat.scatterCoefficient * mat.diffuseTransmissionColor;
  float  mask      = (1.0 - mat.metallic) * (1.0 - mat.iridescence) * (1.0 - mat.transmission);
  return scattered * mask;
}

#if GLTF_USE_RETROREFLECTION
PbrMaterial disableBtdf(PbrMaterial mat)
{
  mat.transmission              = 0.0f;
  mat.diffuseTransmissionFactor = 0.0f;
  return mat;
}

// MRM (Portsmouth et al. 2026): blend reflection lobes only; keep regular BTDF (Portsmouth Sec. 5).
float3 bsdfEvalPunctualKhronosMRM(PbrMaterial mat, float3 V, float3 pointToLight, float3 lightIntensity, float3 transmissionRay, float dttThickness, Sampler2D sheenLUT)
{
  float3 cf = bsdfEvaluatePunctualKhronos(mat, V, pointToLight, lightIntensity, transmissionRay, dttThickness, sheenLUT);
  if(mat.retroreflection <= 0.0f)
    return cf;

  PbrMaterial matBrdf = disableBtdf(mat);
  float3 cfBrdf = bsdfEvaluatePunctualKhronos(matBrdf, V, pointToLight, lightIntensity, transmissionRay, dttThickness, sheenLUT);

  // V is outgoing (toward camera); MRM uses the incident view ray: reflect(-V, N).
  float3 Vp = reflect(-V, mat.N);
  float3 crBrdf = bsdfEvaluatePunctualKhronos(matBrdf, Vp, pointToLight, lightIntensity, transmissionRay, dttThickness, sheenLUT);
  return (cf - cfBrdf) + lerp(cfBrdf, crBrdf, mat.retroreflection);
}

float3 evalKhronosIBLMRM(PbrMaterial mat,
                         float3      V,
                         SamplerCube diffuseCube,
                         SamplerCube glossyCube,
                         Sampler2D   lut,
                         Sampler2D   sheenLUT,
                         float       envRotation,
                         float       envIntensity,
                         float       dttThickness,
                         float3      f_specular_transmission,
                         bool        haveTransmissionSample)
{
  float3 cf = evalKhronosIBL(mat, V, diffuseCube, glossyCube, lut, sheenLUT, envRotation, envIntensity, dttThickness,
                             f_specular_transmission, haveTransmissionSample);
  if(mat.retroreflection <= 0.0f)
    return cf;

  PbrMaterial matBrdf = disableBtdf(mat);
  float3      cfBrdf  = evalKhronosIBL(matBrdf, V, diffuseCube, glossyCube, lut, sheenLUT, envRotation, envIntensity,
                                       dttThickness, float3(0.0f), false);

  float3 Vp     = reflect(-V, mat.N);
  float3 crBrdf = evalKhronosIBL(matBrdf, Vp, diffuseCube, glossyCube, lut, sheenLUT, envRotation, envIntensity,
                                 dttThickness, float3(0.0f), false);
  return (cf - cfBrdf) + lerp(cfBrdf, crBrdf, mat.retroreflection);
}
#endif

float3 getIBLVolumeRefraction(float3   n,
                              float3   v,
                              float    perceptualRoughness,
                              float3   baseColor,
                              float3   worldPos,
                              float3x3 modelMatrix3,
                              float    ior,
                              float    thickness,
                              float3   attenuationColor,
                              float    attenuationDistance,
                              float    dispersion)
{
  // Khronos uses a 4x4 model matrix; we only need column lengths for thickness scaling. We pass
  // a mat3 of the upper-left of objectToWorld which is sufficient.
  // Emulate getVolumeTransmissionRay() inline using mat3.
  // (Keeps signatures simple and avoids passing full mat4.)

  float3 modelScale;
  modelScale.x = length(modelMatrix3[0]);
  modelScale.y = length(modelMatrix3[1]);
  modelScale.z = length(modelMatrix3[2]);

  float3 transmittedLight;
  float  transmissionRayLength = 0.0;

  if(dispersion > 0.0)
  {
    // Per-channel IOR for chromatic dispersion (ibl.glsl:73-95).
    float halfSpread = (ior - 1.0) * 0.025 * dispersion;
    float iors[3]    = { ior - halfSpread, ior, ior + halfSpread };

    [unroll]
    for(int i = 0; i < 3; i++)
    {
      float3 refractionVector = refract(-v, normalize(n), 1.0 / iors[i]);
      float3 transmissionRay  = normalize(refractionVector) * thickness * modelScale;
      transmissionRayLength   = length(transmissionRay);  // overwritten by blue ray; matches reference
      float3 refractedRayExit = worldPos + transmissionRay;

      float4 ndcPos           = mul(float4(refractedRayExit, 1.0), pushConst.frameInfo.viewProjMatrix);
      float2 refractionCoords = ndcPos.xy / max(abs(ndcPos.w), 1e-6);
      refractionCoords        = refractionCoords * 0.5 + 0.5;

      transmittedLight[i] = getTransmissionSample(refractionCoords, perceptualRoughness, iors[i])[i];
    }
  }
  else
  {
    float3 refractionVector = refract(-v, normalize(n), 1.0 / ior);
    float3 transmissionRay  = normalize(refractionVector) * thickness * modelScale;
    transmissionRayLength   = length(transmissionRay);
    float3 refractedRayExit = worldPos + transmissionRay;

    float4 ndcPos           = mul(float4(refractedRayExit, 1.0), pushConst.frameInfo.viewProjMatrix);
    float2 refractionCoords = ndcPos.xy / max(abs(ndcPos.w), 1e-6);
    refractionCoords        = refractionCoords * 0.5 + 0.5;

    transmittedLight = getTransmissionSample(refractionCoords, perceptualRoughness, ior);
  }

  float3 attenuatedColor = applyVolumeAttenuation(transmittedLight, transmissionRayLength, attenuationColor, attenuationDistance);
  return attenuatedColor * baseColor;
}

//------------------------------------------------------------------------------
// Vertex Shader
//
// SV_Position is the UN-JITTERED clip-space position with a post-projection NDC
// shift applied from frameInfo.jitter so the rasterizer samples sub-pixel offsets
// for DLAA. The shift mirrors the standard "translate viewProj in clip-space"
// trick (`clip.xy -= jitter * 2/imageSize * clip.w`)
//
// NOTE: per-object animation/skinning is NOT yet propagated into prevMVP, so
// MVs only reflect camera motion. Animated meshes will still antialias correctly
// from the screen-space MV at the camera level; deferring object-level MVs.
//------------------------------------------------------------------------------
[shader("vertex")]
VertexOutput vertexMain(VertexInput input)
{
  GltfRenderNode renderNode = pushConst.gltfScene.renderNodes[pushConst.renderNodeID];

  float3 pos = mul(float4(input.position, 1.0), renderNode.objectToWorld).xyz;

  VertexOutput output;
  output.worldPos = pos;

#ifdef HAS_DLSS_MOTION
  // DLSS path: emit unjittered clip-space positions for fragment-side MV calc, then
  // post-multiply by the per-frame jitter shift to produce the rasterizer's sample positions.
  output.currClipUnjit = mul(float4(pos, 1.0), pushConst.frameInfo.viewProjMatrix);
  output.prevClipUnjit = mul(float4(pos, 1.0), pushConst.frameInfo.prevMVP);

  // Apply DLAA sub-pixel jitter as a post-projection clip-space shift. Equivalent to multiplying
  // by `translate(-jitter*(2/imgSize, 2/imgSize, 0)) * viewProj` on the host. When DLAA is
  // disabled at runtime the host writes jitter=(0,0) so output.position == currClipUnjit.
  float4 jittered = output.currClipUnjit;
  jittered.xy -= pushConst.frameInfo.jitter * (2.0 / pushConst.frameInfo.imageSize) * jittered.w;
  output.position = jittered;
#else
  // Non-DLSS build: no jitter, no MV - the standard rasterizer clip-space position is enough.
  output.position = mul(float4(pos, 1.0), pushConst.frameInfo.viewProjMatrix);
#endif

  return output;
}

//------------------------------------------------------------------------------
// Fragment/Pixel Shader
//------------------------------------------------------------------------------
// Pixel-space motion vector shared with the path tracer (dlss_util.h::calculateMotionVector).
// We pass UN-JITTERED clip-space positions before the perspective divide (interpolated across
// the triangle by the rasterizer) so the projective interpolation is correct, then the helper
// performs the /w per fragment. NGX's InMVScaleX/Y stays at 1.0, matching DLSS-RR.
[shader("fragment")]
PixelOutput fragmentMain(VertexOutput input, uint primitiveID: SV_PrimitiveID, float3 baryWeights: SV_Barycentrics)
{
  PixelOutput output;
  output.color.a = 1.0;
#ifdef HAS_DLSS_MOTION
  output.motion = calculateMotionVector(input.currClipUnjit, input.prevClipUnjit, pushConst.frameInfo.imageSize);
#endif

  // Setting up scene info
  GltfShadeMaterial material   = pushConst.gltfScene->materials[pushConst.materialID];      // Buffer of materials
  GltfRenderNode    renderNode = pushConst.gltfScene->renderNodes[pushConst.renderNodeID];  // Buffer of render nodes
  GltfRenderPrimitive renderPrimitive = pushConst.gltfScene->renderPrimitives[pushConst.renderPrimID];  // Buffer of meshes

  float3 cameraPos =
      float3(pushConst.frameInfo.viewInv[3].x, pushConst.frameInfo.viewInv[3].y, pushConst.frameInfo.viewInv[3].z);
  float3 worldRayDirection;
  if(hasFlag(pushConst.frameInfo.flags, SceneFrameInfoFlags::eSceneIsOrthographic))
  {
    worldRayDirection = normalize(mul(float4(0, 0, -1, 0), pushConst.frameInfo.viewInv).xyz);
  }
  else
  {
    worldRayDirection = normalize(input.worldPos - cameraPos);
  }

  HitState hit = getHitState(renderPrimitive, baryWeights, renderNode.worldToObject, renderNode.objectToWorld,
                             primitiveID, worldRayDirection);

  // Evaluate the material at the hit point
  MeshState mesh          = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, false, 0.0f);
  mesh.baseColorVertexMul = hit.color;
  PbrMaterial pbrMat      = evaluateMaterial(material, mesh, allTextures, pushConst.gltfScene->textureInfos);

  output.color.xyz = pbrMat.baseColor;
  output.color.a   = pbrMat.opacity * (1.0 - pbrMat.transmission);

  // Write ObjectID for silhouette (0 = no hit, N+1 = render node N); silhouette uses bitmask for selection
  output.selection = float4(asfloat(pushConst.renderNodeID + 1), 0, 0, 0);


  if(pushConst.frameInfo.visualization != Visualization::eRendered)
  {
    float3              vizColor;
    VisualizationResult vizResult =
        applyVisualization(pbrMat, hit, pushConst.frameInfo.visualization, pushConst.renderPrimID, primitiveID, vizColor);
    if(vizResult == VisualizationResult::eColorOverride)
    {
      output.color.xyz = vizColor;
      output.color.a   = 1.0;
      return output;
    }
  }

  if(material.alphaMode == AlphaMode::eAlphaModeMask)
  {
    if(pbrMat.opacity < material.alphaCutoff)
      discard;
  }

#if GLTF_USE_UNLIT
  // KHR_materials_unlit: skip lighting entirely. Match Khronos pbr.frag (color = baseColor + emissive).
  if(material.unlit > 0)
  {
    float finalA = (material.alphaMode == AlphaMode::eAlphaModeBlend) ? pbrMat.opacity : 1.0;
    output.color = float4(pbrMat.baseColor + pbrMat.emissive, finalA);
    return output;
  }
#endif

  const float3 V = -worldRayDirection;

  // Result
  float3 contribution = float3(0);

  // Effective IOR for transmission ray and screen-space refraction (path tracer uses ior2/ior1
  // already; we mirror that to keep raster geometry identical).
  float effectiveIor        = pbrMat.ior2 / max(pbrMat.ior1, 1e-5);
  float perceptualRoughness = sqrt(pbrMat.roughness.y);

  // Model scale (length of each objectToWorld column) is needed by the volume transmission ray
  // and by the diffuse-transmission thickness scaling. Compute once.
  float3 modelScale = float3(length(renderNode.objectToWorld[0].xyz), length(renderNode.objectToWorld[1].xyz),
                             length(renderNode.objectToWorld[2].xyz));

  // World-space thickness for KHR_materials_diffuse_transmission's volume attenuation
  // (`pbr.frag:145`). Hoisted here so it's available to the punctual evaluator regardless of
  // the env mode.
  float dttThickness = (pbrMat.thickness > 0.0) ? (pbrMat.thickness * (modelScale.x + modelScale.y + modelScale.z) / 3.0) : 0.0;

  // Pre-compute the volume transmission ray (object-space refraction direction * thickness *
  // model scale). Used by both the screen-space sample and the punctual BTDF.
  float3 transmissionRay = float3(0.0);
#if GLTF_USE_TRANSMISSION
  if(pbrMat.transmission > 0.0 && pbrMat.thickness > 0.0)
  {
    float3 refractionVector = refract(-V, normalize(pbrMat.N), 1.0 / max(effectiveIor, 1e-5));
    transmissionRay         = normalize(refractionVector) * pbrMat.thickness * modelScale;
  }
#endif

  // Screen-space transmitted radiance (KHR_materials_transmission) computed once for both
  // env-mode branches when the opaque-color framebuffer was captured this frame.
  bool   haveTransmissionSample  = false;
  float3 f_specular_transmission = float3(0.0);
#if GLTF_USE_TRANSMISSION
  if(pbrMat.transmission > 0.0 && pushConst.opaqueColorReady != 0)
  {
    float3x3 m3 = float3x3(renderNode.objectToWorld[0].xyz, renderNode.objectToWorld[1].xyz, renderNode.objectToWorld[2].xyz);
    f_specular_transmission =
        getIBLVolumeRefraction(pbrMat.N, V, perceptualRoughness, pbrMat.baseColor, hit.pos, m3, effectiveIor,
                               pbrMat.thickness, pbrMat.attenuationColor, pbrMat.attenuationDistance, pbrMat.dispersion);
    haveTransmissionSample = true;
  }
#endif

  // Indirect (IBL or sky-fallback) lighting.
  if(!hasFlag(pushConst.frameInfo.flags, SceneFrameInfoFlags::eSceneUseHdrEnvironment))
  {
    // Sky mode: cheap hemispherical ambient + sun-as-directional. The sun is fed through the
    // Khronos-faithful punctual evaluator so the BRDF math matches the IBL composite layer.
    float3 ambientColor = lerp(float3(0.4F), float3(0.17F, 0.37F, 0.65F), pbrMat.N.y * 0.5 + 0.5) * 0.3;
    // Hemisphere ambient drives both the diffuse term (via baseColor) and a tiny specular
    // floor for metals so they don't go pitch black under sun-only lighting.
    float3 metalFloor = lerp(float3(0.04), pbrMat.baseColor, pbrMat.metallic);
    contribution += ambientColor * lerp(pbrMat.baseColor, metalFloor, pbrMat.metallic);

    // Sun direction is the unit vector "toward sun" (directional-light convention). Pass
    // unshifted; the evaluator applies `transmissionRay` internally for the BTDF lobe only.
#if GLTF_USE_RETROREFLECTION
    contribution += bsdfEvalPunctualKhronosMRM(pbrMat, V, pushConst.skyParams.sunDirection, pushConst.skyParams.sunDiskIntensity,
                                               transmissionRay, dttThickness, texturesHdr[HDR_SHEEN_INDEX]);
#else
    contribution += bsdfEvaluatePunctualKhronos(pbrMat, V, pushConst.skyParams.sunDirection, pushConst.skyParams.sunDiskIntensity,
                                                transmissionRay, dttThickness, texturesHdr[HDR_SHEEN_INDEX]);
#endif

#if GLTF_USE_TRANSMISSION
    if(haveTransmissionSample)
      contribution = lerp(contribution, f_specular_transmission, pbrMat.transmission);
#endif
  }
  else
  {
    // HDR IBL mode: single-call Khronos split-sum evaluator (matches `pbr.frag:162-228`).
    // Handles diffuse, specular (isotropic/anisotropic), sheen, clearcoat, iridescence,
    // diffuse-transmission volume attenuation, and optional screen-space transmission mixing.
#if GLTF_USE_RETROREFLECTION
    contribution += evalKhronosIBLMRM(pbrMat, V, texturesCube[HDR_DIFFUSE_INDEX], texturesCube[HDR_GLOSSY_INDEX],
                                      texturesHdr[HDR_LUT_INDEX], texturesHdr[HDR_SHEEN_INDEX],
                                      pushConst.frameInfo.envRotation, pushConst.frameInfo.envIntensity, dttThickness,
                                      f_specular_transmission, haveTransmissionSample);
#else
    contribution +=
        evalKhronosIBL(pbrMat, V, texturesCube[HDR_DIFFUSE_INDEX], texturesCube[HDR_GLOSSY_INDEX],
                       texturesHdr[HDR_LUT_INDEX], texturesHdr[HDR_SHEEN_INDEX], pushConst.frameInfo.envRotation,
                       pushConst.frameInfo.envIntensity, dttThickness, f_specular_transmission, haveTransmissionSample);
#endif

#if GLTF_USE_VOLUME_SCATTER
    // KHR_materials_volume_scatter contribution is separate from the IBL composite (matches
    // `pbr.frag:352-356`). The clearcoat mask `(1 - clearcoat * F_coat)` is angle-dependent.
    float3 ccFmask   = float3(1.0) - pbrMat.clearcoat * khronosClearcoatFresnel(pbrMat, V);
    float3 scatterIr = getIBLDiffuse(texturesCube[HDR_DIFFUSE_INDEX], pbrMat.N, pushConst.frameInfo.envRotation,
                                     pushConst.frameInfo.envIntensity);
    contribution += volumeScatterApprox(pbrMat, scatterIr) * ccFmask;
#endif
  }

  // Apply ambient occlusion to indirect lighting only (matches Khronos pbr.frag).
  contribution *= pbrMat.occlusion;

  // Emissive under the clearcoat layer (`pbr.frag:373`).
  contribution += pbrMat.emissive * (float3(1.0) - pbrMat.clearcoat * khronosClearcoatFresnel(pbrMat, V));

  // Punctual lights — Khronos-faithful direct lighting: spec-correct KHR_lights_punctual
  // attenuation via `singleLightContribution`, then per-light layered BSDF via
  // `bsdfEvaluatePunctualKhronos` (VdotH Schlick + height-correlated Smith + Khronos composite).
  // Intensity is computed once per light at the front-surface position; only the BTDF lobe gets
  // the transmission-ray shift applied internally (intentional fix vs. `pbr.frag:316`).
  for(int i = 0; i < pushConst.gltfScene.numLights; i++)
  {
    GltfLight    light   = pushConst.gltfScene.lights[i];
    LightContrib contrib = singleLightContribution(light, hit.pos, pbrMat.N);
    if(all(contrib.intensity == 0.0))
      continue;

    // contrib.incidentVector points from the light toward the surface (= propagation direction).
    // The evaluator wants the unnormalized surface-to-light vector (length matters for the
    // BTDF transmission-ray shift to match `pbr.frag:303`).
    float3 pointToLight;
    if(light.type == LightType::eLightTypeDirectional)
      pointToLight = -contrib.incidentVector;
    else
      pointToLight = -contrib.incidentVector * contrib.distance;

#if GLTF_USE_RETROREFLECTION
    contribution += bsdfEvalPunctualKhronosMRM(pbrMat, V, pointToLight, contrib.intensity, transmissionRay,
                                               dttThickness, texturesHdr[HDR_SHEEN_INDEX]);
#else
    contribution += bsdfEvaluatePunctualKhronos(pbrMat, V, pointToLight, contrib.intensity, transmissionRay,
                                                dttThickness, texturesHdr[HDR_SHEEN_INDEX]);
#endif
  }


  // Final alpha. Khronos pbr.frag writes a=1.0 for opaque/mask and `baseColor.a` for blend, and
  // relies on the screen-space transmission sample to make transmissive surfaces see-through.
  //
  // When the transmission framebuffer sample IS available (`haveTransmissionSample`), the
  // fragment color already contains the refracted backdrop, so alpha stays at the authored
  // opacity -- reducing it by `(1 - transmission)` would multiply SRC_ALPHA=0 and the surface
  // would vanish entirely (SRC_ALPHA blending: `0*src + 1*dst` = background kept as-is).
  //
  // When we do NOT have a sample (sky mode with no capture, or transmissive material in a
  // scene that never triggered the opaque-color pass), we fall back to alpha-based see-through
  // so the surface still lets the background show by reducing alpha by transmission.
  float finalAlpha = 1.0;
  if(material.alphaMode == AlphaMode::eAlphaModeBlend)
    finalAlpha = pbrMat.opacity;
#if GLTF_USE_TRANSMISSION
  if(pbrMat.transmission > 0.0 && !haveTransmissionSample)
    finalAlpha *= (1.0 - pbrMat.transmission);
#endif

  output.color = float4(contribution, finalAlpha);


  return output;
}

//------------------------------------------------------------------------------
// Wireframe Shader
// Return a constant color for wireframe rendering. Selection/motion are zero - the
// wireframe pass is a visual debug overlay and doesn't participate in DLAA.
[shader("fragment")]
PixelOutput fragmentWireframeMain(VertexOutput input)
{
  PixelOutput output;
  output.color     = float4(0.4, 0.01, 0.01, 1);
  output.selection = float4(0.0);
#ifdef HAS_DLSS_MOTION
  output.motion = float2(0.0);
#endif
  return output;
}
