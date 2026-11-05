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

#ifndef CAUSTICS_H
#define CAUSTICS_H 1

#include "caustics_grid.h.slang"

//-----------------------------------------------------------------------
// Beta caustics by light tracing and photon mapping. See docs/caustics.md.
//
// Three estimators split the paths that carry a caustic, so no energy is counted twice:
//
//   light tracing  L S+ D E   photons leave the emitters, follow the smooth casters (S) and, at the
//                             first surface that is not one (D), connect to the camera. D has to be
//                             what the camera sees directly.
//   photon map     L S+ D ... E  the same photons, kept for the frame. A camera path that reaches a surface D
//                             other than the one the camera sees directly, through casters or not, gathers
//                             them there: the caustic light that D receives, direct or indirect.
//   path tracing   everything else. At such a surface D, it gives up the paths that start with a
//                             caster: the shadow ray of the light sampler treats casters as opaque
//                             (otherwise glass would let light through twice), and a path that leaves D
//                             through casters and then finds a light adds nothing.
//
// The partition is by material: isCausticCaster() is the single definition, read by both sides.
//
// Everything here is compiled only with USE_CAUSTICS, so a build without it carries none of it.
//-----------------------------------------------------------------------

// A caster is a surface that only has delta-like lobes: smooth metal and smooth, fully transmissive
// glass. Anything with a diffuse, sheen, coat, scattering or stochastic-opacity component is a
// receiver (or ordinary geometry) for this purpose, and stays with path tracing.
static const float CAUSTIC_MAX_ROUGHNESS = 0.1F;
static const int   CAUSTIC_MAX_DEPTH     = 16;  // Casters a photon follows before it is given up

bool isCausticCaster(PbrMaterial m)
{
  // Diffuse reflection weight in the glTF layering: what is neither metal nor transmitted
  const float diffuse = (1.0F - m.metallic) * (1.0F - m.transmission);
  // Thin-walled transmission (no thickness: a window pane) bends nothing: light goes straight through, which the
  // path tracer's light samples handle without noise. Making it a caster would hand all the light seen through
  // every window to the photons.
  const bool thinWalled = m.transmission > 0.0F && m.thickness <= 0.0F;
  return !thinWalled && max(m.roughness.x, m.roughness.y) <= CAUSTIC_MAX_ROUGHNESS && diffuse <= 1e-3F
         && m.clearcoat <= 0.0F && m.diffuseTransmissionFactor <= 0.0F && all(m.sheenColor <= float3(0.0F))
         && all(m.scatterCoefficient <= float3(0.0F)) && m.opacity >= 1.0F;
}

// A thin-walled pane (transmission, no thickness, smooth): not a caster, since it bends nothing, but a photon goes
// through it on its way to the casters behind (a window between the sun and a glass ball) instead of landing on it.
bool isCausticThinPane(PbrMaterial m)
{
  return m.transmission > 0.0F && m.thickness <= 0.0F && max(m.roughness.x, m.roughness.y) <= CAUSTIC_MAX_ROUGHNESS
         && m.opacity >= 1.0F;
}

// The same question for a surface a shadow ray crosses: evaluate the material there, at texture
// level 0 like every shadow-ray texture lookup.
bool isCausticCasterAtHit(GltfRenderNode node, GltfRenderPrimitive prim, int triangleID, float3 barycentrics, float3 rayDirection)
{
  GltfShadeMaterial* materials = pushConst.gltfScene->materials;
  GltfTextureInfo*   texInfos  = pushConst.gltfScene->textureInfos;

  HitState  hit  = getHitState(prim, barycentrics, node.worldToObject, node.objectToWorld, triangleID, rayDirection);
  MeshState mesh = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, false, 0.0F);
  mesh.baseColorVertexMul = hit.color;
  return isCausticCaster(evaluateMaterial(max(0, node.materialID), mesh, allImages, allSamplers, texInfos, materials));
}

//-----------------------------------------------------------------------
// Path-tracing side of the partition
//-----------------------------------------------------------------------

// A path reached a surface: a caster extends the chain of casters since the last receiver, anything
// else ends it. A receiver whose light the photons carry is the camera's primary hit for the splats, and with
// a photon map the first CAUSTIC_MAP_RECEIVERS surfaces of the path that are not casters. Returns true for the
// latter: the caller gathers the photons there.
static const int CAUSTIC_MAP_RECEIVERS = 3;  // Receivers of a path that gather; after them the path tracer has the light

bool causticAdvance(inout PathTracerState pt, bool caster)
{
  if(caster)
  {
    pt.causticChain++;
    return false;
  }
  pt.causticChain = 0;
  if(pushConst.caustics == nullptr)  // This frame traces no photons (the setting is changing, an orthographic camera): nothing is covered
  {
    pt.causticReceiver = false;
    return false;
  }
  const bool hasMap  = pushConst.caustics->photons != nullptr;
  const bool primary = pt.surfaceDepth == 0;
  const bool gathers = hasMap && !primary && pt.causticReceivers < CAUSTIC_MAP_RECEIVERS;
  pt.causticReceivers++;
  pt.causticReceiver = primary || gathers;
  return gathers;
}

// What the path found came through casters from that receiver: the photons have it already.
bool causticCovered(PathTracerState pt)
{
  return pushConst.caustics != nullptr && pt.causticChain > 0 && pt.causticReceiver;
}

//-----------------------------------------------------------------------
// Light tracing
//-----------------------------------------------------------------------

struct PhotonEmission
{
  float3 origin    = float3(0.0F);
  float3 direction = float3(0.0F, 0.0F, 1.0F);
  float3 flux      = float3(0.0F);  // Power of the photon, divided by the pdf and the number of photons
  bool   valid     = false;
};

// How much of the sphere of directions the sphere `s` covers from `origin`, as 1 - cos of its half angle (2: all
// of them, from inside), without the cancellation of a far, small sphere.
float causticOneMinusCos(float3 origin, CausticSphere s)
{
  const float3 offset          = s.center - origin;
  const float  distanceSquared = dot(offset, offset);
  const float  radiusSquared   = s.radius * s.radius;
  if(distanceSquared <= radiusSquared * 1.0001F)
    return 2.0F;
  const float ratio = radiusSquared / distanceSquared;
  return ratio / (1.0F + sqrt(max(0.0F, 1.0F - ratio)));
}

// A direction from `origin` toward the caster spheres: one sphere is chosen with a probability in proportion to
// the part of the directions it covers times its weight, and the direction is uniform in its cone. Where the
// cones overlap, every sphere that holds the direction adds its density (pdf is of the mixture).
float3 causticSampleDirection(float3 origin, float2 u, inout uint seed, out float pdf)
{
  CausticParams* causticParams = pushConst.caustics;
  float          total         = 0.0F;
  for(uint i = 0; i < causticParams->casterCount; i++)
    total += causticOneMinusCos(origin, causticParams->casters[i]) * causticParams->casters[i].weight;

  float pick     = rand(seed) * total;
  uint  selected = causticParams->casterCount - 1;
  for(uint i = 0; i < causticParams->casterCount; i++)
  {
    pick -= causticOneMinusCos(origin, causticParams->casters[i]) * causticParams->casters[i].weight;
    if(pick <= 0.0F)
    {
      selected = i;
      break;
    }
  }

  const CausticSphere sphere      = causticParams->casters[selected];
  const float3        offset      = sphere.center - origin;
  const float         oneMinusCos = causticOneMinusCos(origin, sphere);
  const float3        axis        = dot(offset, offset) > 1e-20F ? normalize(offset) : float3(0.0F, 0.0F, 1.0F);
  const float         cosT        = 1.0F - u.x * oneMinusCos;
  const float         sinT        = sqrt(max(0.0F, 1.0F - cosT * cosT));
  const float         phi         = M_TWO_PI * u.y;
  float3              tangent, bitangent;
  orthonormalBasis(axis, tangent, bitangent);
  const float3 direction = normalize(axis * cosT + (tangent * cos(phi) + bitangent * sin(phi)) * sinT);

  float overlaps = 0.0F;
  for(uint i = 0; i < causticParams->casterCount; i++)
  {
    const CausticSphere other = causticParams->casters[i];
    const float3        d     = other.center - origin;
    const float         omc   = causticOneMinusCos(origin, other);
    if(omc >= 2.0F || dot(normalize(d), direction) >= 1.0F - omc)
      overlaps += other.weight;
  }
  pdf = overlaps / max(M_TWO_PI * total, 1e-20F);
  return direction;
}

// A point of a disk, perpendicular to the light's direction, that covers one of the caster spheres and lies
// outside the scene: what a light at infinity (sun, sky) sends its photons through. They start outside so that
// anything between the light and the casters shadows them, as it should. The disk is chosen with a probability
// in proportion to its area times the sphere's weight; `diskPdf` is the density of the mixture over the plane.
float3 causticDiskOrigin(float3 toLight, float2 u, inout uint seed, out float diskPdf)
{
  CausticParams* causticParams = pushConst.caustics;
  float          area          = 0.0F;
  for(uint i = 0; i < causticParams->casterCount; i++)
    area += causticParams->casters[i].radius * causticParams->casters[i].radius * causticParams->casters[i].weight;

  float pick     = rand(seed) * area;
  uint  selected = causticParams->casterCount - 1;
  for(uint i = 0; i < causticParams->casterCount; i++)
  {
    pick -= causticParams->casters[i].radius * causticParams->casters[i].radius * causticParams->casters[i].weight;
    if(pick <= 0.0F)
    {
      selected = i;
      break;
    }
  }

  const CausticSphere sphere = causticParams->casters[selected];
  const float reach = length(sphere.center - causticParams->sceneCenter) + causticParams->sceneRadius + sphere.radius;
  float3      tangent, bitangent;
  orthonormalBasis(toLight, tangent, bitangent);
  const float  r      = sphere.radius * sqrt(u.x);
  const float  phi    = M_TWO_PI * u.y;
  const float3 origin = sphere.center + toLight * reach + (tangent * cos(phi) + bitangent * sin(phi)) * r;

  float overlaps = 0.0F;
  for(uint i = 0; i < causticParams->casterCount; i++)
  {
    const CausticSphere other  = causticParams->casters[i];
    const float3        offset = origin - other.center;
    const float3        across = offset - toLight * dot(offset, toLight);
    if(dot(across, across) <= other.radius * other.radius)
      overlaps += other.weight;
  }
  diskPdf = overlaps / max(M_PI * area, 1e-20F);
  return origin;
}

// Pick an emitter as sampleLights() would -- the same technique and sub-technique probabilities --
// and send a photon from it toward the casters. Textured emissive meshes are not in the light list.
PhotonEmission causticEmit(inout uint seed, float2 aim)
{
  PhotonEmission photon        = {};
  CausticParams* causticParams = pushConst.caustics;
  const float    count         = float(causticParams->photonCount);

  LightingTechniqueProbabilities probs = getDirectLightingTechniqueProbabilities();
  if(probs.lightWeight <= 0.0F && probs.envWeight <= 0.0F)
    return photon;

  const bool sampleLight = rand(seed) < probs.lightWeight;
  if(sampleLight)
  {
    bool pickEmissive;
    if(probs.emissiveWeight <= 0.0F)
      pickEmissive = false;
    else if(probs.punctualWeight <= 0.0F)
      pickEmissive = true;
    else
      pickEmissive = (rand(seed) < probs.emissiveWeight);

    if(!pickEmissive)
    {
      //-- Punctual light
      const int       numPunctual = punctualLightCount();
      const float     selection   = probs.lightWeight * probs.punctualWeight / float(numPunctual);
      const GltfLight light       = getPunctualLight(min(int(rand(seed) * numPunctual), numPunctual - 1));
      const float3    intensity   = light.intensity * light.color;

      if(light.type == LightType::eLightTypeDirectional)
      {
        // The sun is a cone of directions; the photons' flux does not depend on how it is sampled
        float3 toLight = -normalize(light.direction);
        if(light.angularSizeOrInvRange > 0.0F)
        {
          const float oneMinusCos = 1.0F - cos(0.5F * light.angularSizeOrInvRange);
          const float cosT        = 1.0F - rand(seed) * oneMinusCos;
          const float sinT        = sqrt(max(0.0F, 1.0F - cosT * cosT));
          const float phi         = M_TWO_PI * rand(seed);
          float3      tangent, bitangent;
          orthonormalBasis(toLight, tangent, bitangent);
          toLight = normalize(toLight * cosT + (tangent * cos(phi) + bitangent * sin(phi)) * sinT);
        }
        photon.direction = -toLight;
        float diskPdf;
        photon.origin = causticDiskOrigin(toLight, aim, seed, diskPdf);
        photon.flux   = intensity / (selection * diskPdf * count);
        photon.valid  = diskPdf > 0.0F;
      }
      else
      {
        float3 position = light.position;
        float  pdf;
        float  emission = 1.0F;  // what the surface of a sphere light adds: 4 cos for its Lambertian radiance
        if(light.radius > 0.0F)
        {
          const float  z      = 1.0F - 2.0F * rand(seed);
          const float  s      = sqrt(max(0.0F, 1.0F - z * z));
          const float  phi    = M_TWO_PI * rand(seed);
          const float3 normal = float3(s * cos(phi), s * sin(phi), z);
          position += light.radius * normal;
          photon.direction = causticSampleDirection(position, aim, seed, pdf);
          emission         = 4.0F * dot(normal, photon.direction);
          if(emission <= 0.0F)
            return photon;  // toward the inside of the sphere
        }
        else
        {
          photon.direction = causticSampleDirection(position, aim, seed, pdf);
        }
        photon.origin = position;
        photon.flux   = intensity * emission / (selection * pdf * count);
        if(light.type == LightType::eLightTypeSpot)
          photon.flux *= getSpotAttenuation(-photon.direction, light.direction, cos(light.outerAngle), cos(light.innerAngle));
        photon.flux *= evalIesFactor(pushConst.gltfScene, light, position + photon.direction);  // EXT_lights_ies
        photon.valid = any(photon.flux > 0.0F);
      }
    }
    else
    {
      //-- Emissive triangle: power-weighted like the light sampler, then uniform over its area
      EmissiveTriangle emissiveTriangle = pushConst.gltfScene.emissiveTriangles[selectEmissiveTriangle(rand(seed))];
      float3           p0, p1, p2;
      getEmissiveTriangleWorld(emissiveTriangle, p0, p1, p2);
      const float  sqrtU       = sqrt(rand(seed));
      const float  b1          = 1.0F - sqrtU;
      const float  b2          = rand(seed) * sqrtU;
      const float3 normal      = cross(p1 - p0, p2 - p0);
      const float  twoArea     = length(normal);
      const float  totalWeight = pushConst.gltfScene.emissiveTotalWeight;
      if(twoArea <= 1e-12F || totalWeight <= 0.0F)
        return photon;

      float pdf;
      photon.origin    = p0 * (1.0F - b1 - b2) + p1 * b1 + p2 * b2;
      photon.direction = causticSampleDirection(photon.origin, aim, seed, pdf);
      float cosL       = dot(normal / twoArea, photon.direction);
      if(emissiveTriangle.doubleSided != 0)
        cosL = abs(cosL);
      if(cosL <= 1e-6F)
        return photon;  // one-sided emitter facing away
      // Leave the emitter's plane: a double-sided one would be hit again at once, and the photon lost
      const float3 emitterNormal = normal / twoArea;
      photon.origin += (dot(emitterNormal, photon.direction) >= 0.0F ? emitterNormal : -emitterNormal)
                       * (1e-5F * causticParams->sceneRadius);

      const float areaPdf = emitterDefensiveLuminance(emissiveTriangle.radiance) / totalWeight;  // selection pmf over the area
      photon.flux = emissiveTriangle.radiance * cosL / (probs.lightWeight * probs.emissiveWeight * areaPdf * pdf * count);
      photon.valid = true;
    }
  }
  else if(probs.envWeight > 0.0F)
  {
    //-- Environment: the same lat-long sampler as the light sampler, through a disk over the casters
    float3       toLight;
    const uint2  grid = uint2(pushConst.frameInfo->envSamplingGridWidth, pushConst.frameInfo->envSamplingGridHeight);
    const float4 radiance_pdf = environmentSample(texturesHdr[HDR_IMAGE_INDEX], envSamplingData, grid,
                                                  float3(rand(seed), rand(seed), rand(seed)), toLight);
    if(radiance_pdf.w <= 0.0F)
      return photon;
    toLight          = quatRotate(pushConst.frameInfo->envRotation, toLight);
    photon.direction = -toLight;
    float diskPdf;
    photon.origin = causticDiskOrigin(toLight, aim, seed, diskPdf);
    photon.flux = radiance_pdf.xyz * pushConst.frameInfo->envIntensity / (radiance_pdf.w * probs.envWeight * diskPdf * count);
    photon.valid = diskPdf > 0.0F;
  }
  return photon;
}

// Veach's correction for a light path that scatters with a shading normal: what the adjoint BSDF
// needs for the path to carry the same energy as the camera path along it. `view` points back
// toward where the photon came from, `outgoing` where it goes next; both away from the surface.
float causticAdjoint(float3 view, float3 outgoing, float3 shading, float3 geometric)
{
  const float denominator = abs(dot(view, geometric) * dot(outgoing, shading));
  if(denominator <= 1e-12F)
    return 0.0F;
  return min(abs(dot(view, shading) * dot(outgoing, geometric)) / denominator, 4.0F);
}

// Float add on a pixel's channel, through the integer the buffer holds: no float atomics needed.
void causticAtomicAdd(uint* address, float value)
{
  if(!(value > 0.0F) || !isfinite(value))
    return;
  Atomic<uint>* a    = (Atomic<uint>*)address;
  uint          prev = a->load();
  // Lock-free: whenever this thread loses, another one has added its own value, so it cannot starve
  for(;;)
  {
    const uint seen = a->compareExchange(prev, asuint(asfloat(prev) + value));
    if(seen == prev)
      return;
    prev = seen;
  }
}

// The photon reached a receiver `hit` through casters: connect it to the camera and add what the
// pixel there sees to the frame's splat buffer. The receiver has to be what the camera sees
// directly -- the partition above gave the other paths to path tracing.
void causticSplat(IRaytracer raytracer, HitState hit, PbrMaterial mat, float3 photonDirection, float3 flux, inout uint seed)
{
  CausticParams*  causticParams = pushConst.caustics;
  SceneFrameInfo* frameInfo     = pushConst.frameInfo;
  const float2    imageSize     = float2(pushConst.renderSize);
  const float3    eye           = frameInfo->viewInv[3].xyz;
  const float3    right         = frameInfo->viewInv[0].xyz;
  const float3    up            = frameInfo->viewInv[1].xyz;
  const float3    forward       = -normalize(frameInfo->viewInv[2].xyz);

  // A point of the lens; the pinhole when there is no aperture. The camera samples the lens as
  // `sqrt(aperture * u)` of a disk (see samplePixel).
  float3 lens = eye;
  if(pushConst.aperture > 0.0F)
  {
    const float r   = sqrt(pushConst.aperture * rand(seed));
    const float phi = M_TWO_PI * rand(seed);
    lens += (cos(phi) * right + sin(phi) * up) * r;
  }

  const float3 toLens = lens - hit.pos;
  const float  dist   = length(toLens);
  if(dist <= 1e-6F)
    return;
  const float3 toCamera = toLens / dist;

  // Which pixel does this lens point see the receiver in? The path tracer's rays cross the focal
  // sphere around the eye at the focus distance, so intersect the line lens -> point with it, and
  // project the point it crosses along the same direction from the eye.
  float3 projected = hit.pos;
  if(pushConst.aperture > 0.0F)
  {
    const float3 u = -toCamera;  // lens -> point
    const float3 o = lens - eye;
    const float  b = dot(o, u);
    const float  c = dot(o, o) - pushConst.focalDistance * pushConst.focalDistance;
    const float  h = b * b - c;
    if(h < 0.0F)
      return;
    projected = eye + normalize(o + (sqrt(h) - b) * u);
  }
  if(dot(hit.pos - eye, forward) <= 0.0F)
    return;
  const float4 clip = mul(float4(projected, 1.0F), frameInfo->viewProjMatrix);
  if(clip.w <= 0.0F)
    return;
  float2 film = clip.xy / clip.w * 0.5F + 0.5F;
  // DLSS shifts every camera ray by its jitter: the pixel that sees this point is shifted back
  if(hasFlag(pushConst.flags, PathtracerFlags::ePtUseDlss))
    film -= frameInfo->jitter / imageSize;
  if(any(film < float2(0.0F)) || any(film >= float2(1.0F)))
    return;

  // Visible from the camera? Anything in between hides it, glass included: a receiver behind a
  // caster is not a primary hit, and belongs to path tracing.
  const float3 offsetDir = dot(toCamera, hit.geonrm) > 0.0F ? hit.geonrm : -hit.geonrm;
  RayDesc      visibility;
  visibility.Origin    = safeOffsetRay(hit.pos, offsetDir, hit.offsetDistance);
  visibility.Direction = toCamera;
  visibility.TMin      = 0.0F;
  visibility.TMax      = dist;
  HitPayload payload   = {};
  raytracer.Trace(visibility, payload, seed);
  HitState planeHit = {};
  checkInfinitePlaneIntersection(visibility, payload, planeHit, frameInfo);
  if(payload.hitT < dist)
    return;

  // The material toward the camera. k1 plays the view, k2 the light: the BSDF is reciprocal, and
  // the cosine it returns is then the one at the receiver toward the camera, which the measure needs.
  BsdfEvaluateData evalData;
  evalData.k1 = -photonDirection;
  evalData.k2 = toCamera;
  evalData.xi = float3(rand(seed), rand(seed), rand(seed));
  bsdfEvaluate(evalData, mat);
  const float3 response = evalData.bsdf_diffuse + evalData.bsdf_glossy;
  if(!any(response > 0.0F))
    return;

  // Pixels per steradian: the film's area at unit distance is 4 tan(fovx/2) tan(fovy/2)
  const float cosCamera = dot(-toCamera, forward);
  if(cosCamera <= 0.0F)
    return;
  const float filmArea   = 4.0F * abs(frameInfo->projInv[0][0] * frameInfo->projInv[1][1]);
  const float importance = imageSize.x * imageSize.y / (filmArea * dist * dist * cosCamera * cosCamera * cosCamera);

  const float3 value = flux * response * importance * causticAdjoint(-photonDirection, toCamera, mat.N, hit.geonrm);
  const uint2  pixel = min(uint2(film * imageSize), pushConst.renderSize - 1);
  uint*        splat = causticParams->splats + (pixel.y * pushConst.renderSize.x + pixel.x) * 3;
  causticAtomicAdd(splat + 0, value.x);
  causticAtomicAdd(splat + 1, value.y);
  causticAtomicAdd(splat + 2, value.z);
}

//-----------------------------------------------------------------------
// The photon map: a slot for each photon of the frame, and a grid over the ones that landed,
// built by a counting sort (count per hash cell, prefix sum, scatter) so that the photons of a cell are
// contiguous in memory. A cell is several gather radii wide (each cell a gather reads is a cache miss); a
// gather reads the cells its disk overlaps.
//-----------------------------------------------------------------------
static const float CAUSTIC_SHINY_ROUGHNESS  = 0.35F;  // A receiver this smooth evaluates its BSDF for each photon
static const float CAUSTIC_MAX_RADIUS_CELLS = 2.0F;   // The widest gather disk, in cells of the grid
static const uint CAUSTIC_CELL_READS = 1024;  // Photons of a cell a gather reads, at most: a more crowded cell is read with a stride

// The flux of a photon in a word: 8 bits per channel and a shared exponent. The rounding is to the nearest,
// so it adds noise, not bias. 0 is no flux; a flux that is not 0 never packs to it (its largest channel is
// at least 128).
uint causticPackFlux(float3 flux)
{
  const float peak = max(max(flux.x, flux.y), flux.z);
  if(!(peak > 0.0F))
    return 0u;
  const int   exponent = clamp(int(floor(log2(peak))) + 1, -126, 127);
  const uint3 mantissa = uint3(clamp(round(flux * exp2(-float(exponent)) * 255.0F), 0.0F, 255.0F));
  return mantissa.x | (mantissa.y << 8) | (mantissa.z << 16) | (uint(exponent + 127) << 24);
}

float3 causticUnpackFlux(uint packed)
{
  if(packed == 0u)
    return float3(0.0F);
  return float3(float(packed & 255u), float((packed >> 8) & 255u), float((packed >> 16) & 255u))
         * (exp2(float(int(packed >> 24) - 127)) / 255.0F);
}

uint causticPackDirection(float3 direction)
{
  float2 p = direction.xy / (abs(direction.x) + abs(direction.y) + abs(direction.z));
  if(direction.z < 0.0F)
    p = (1.0F - abs(p.yx)) * float2(p.x >= 0.0F ? 1.0F : -1.0F, p.y >= 0.0F ? 1.0F : -1.0F);
  const uint2 q = uint2(saturate(p * 0.5F + 0.5F) * 65535.0F + 0.5F);
  return q.x | (q.y << 16);
}

float3 causticUnpackDirection(uint packed)
{
  float2 p = float2(float(packed & 0xFFFFu), float(packed >> 16)) * (2.0F / 65535.0F) - 1.0F;
  float3 d = float3(p, 1.0F - abs(p.x) - abs(p.y));
  if(d.z < 0.0F)
    d.xy = (1.0F - abs(d.yx)) * float2(d.x >= 0.0F ? 1.0F : -1.0F, d.y >= 0.0F ? 1.0F : -1.0F);
  return normalize(d);
}

// The photon landed at `position`, travelling along `travel`, carrying `flux` (times the frame's photons)
void causticStore(uint slot, float3 position, float3 travel, float3 flux)
{
  if(!all(isfinite(flux)) || !any(flux > 0.0F))
    return;
  CausticPhoton* photon = pushConst.caustics->photons + slot;
  photon->position      = position;
  photon->direction     = causticPackDirection(travel);
  photon->flux          = causticPackFlux(flux);  // Last: a slot with flux is one the grid passes use
}

// Where the photons of hash cell `slot` are in the sorted array: [begin, end)
void causticCellRun(uint slot, out uint begin, out uint end)
{
  begin = pushConst.caustics->cellStarts[slot];
  end   = pushConst.caustics->cellStarts[slot + 1u];
}

// A photon of the map lands within `radius` of `position`, on its surface: a photon of the wall behind a
// corner is not this floor's
bool causticOnDisk(CausticPhoton photon, float3 position, float3 normal, float radius, out float distanceSquared)
{
  const float3 offset = photon.position - position;
  distanceSquared     = dot(offset, offset);
  return distanceSquared < radius * radius && abs(dot(offset, normal)) <= 0.25F * radius;
}

// The grid's cells that can hold such photons: the box around the disk, which is thin along the normal
void causticCellRange(float3 position, float3 normal, float radius, out int3 lowest, out int3 highest)
{
  const float3 extent = radius * (sqrt(max(1.0F - normal * normal, 0.0F)) + 0.25F * abs(normal));
  lowest              = causticCell(pushConst.caustics, position - extent);
  highest             = causticCell(pushConst.caustics, position + extent);
}

// The radius of the disk a receiver gathers its photons from: a few pixels of the camera at the receiver
// (grazing receivers, whose pixels are long, get a longer one), which shrinks as the accumulation grows, and
// at most a world radius that shrinks with it. Where the camera sees the receiver through casters, the
// distance is the straight one: a good enough footprint for what the casters bend.
float causticGatherRadius(float3 position, float3 normal)
{
  CausticParams* causticParams = pushConst.caustics;
  const float3   toReceiver    = position - pushConst.frameInfo->viewInv[3].xyz;
  const float    distance      = max(length(toReceiver), 1e-6F);
  const float    cosView       = abs(dot(normal, toReceiver / distance));
  const float    footprint     = pushConst.pixelAngle * distance / sqrt(max(0.0625F, cosView));
  return max(min(causticParams->gatherWorld, causticParams->gatherPixels * footprint), 1e-7F);
}

// What the photons around a receiver add up to, in the disk of `radius`: their flux in an Epanechnikov
// kernel, and for a rough receiver the directions the flux came from. A shiny receiver evaluates its BSDF for
// each photon instead, since the highlight depends on where each came from. Returns false, early, when the
// disk holds more than gatherMax photons: the caller shrinks it. `unlimited` (the last try) never gives up: a
// disk that is never small enough must not lose its energy.
bool causticSum<let SHINY : int>(HitState hit, PbrMaterial mat, float3 outgoing, float radius, bool unlimited, inout uint seed, out float3 sum, out float3 fromSum)
{
  CausticParams* causticParams = pushConst.caustics;
  const float    radiusSquared = radius * radius;
  sum                          = float3(0.0F);  // Shiny: the response of each photon. Rough: the flux.
  fromSum                      = float3(0.0F);  // Rough: the directions the flux came from, weighted by it
  int3 lowest, highest;
  causticCellRange(hit.pos, hit.geonrm, radius, lowest, highest);
  const bool  transmissive = mat.diffuseTransmissionFactor > 0.0F || mat.transmission > 0.0F;
  const float strideShift  = rand(seed);  // Where a crowded cell's stride starts, the same for every cell of this sum
  int         count        = 0;
  for(int z = lowest.z; z <= highest.z; z++)
  {
    for(int y = lowest.y; y <= highest.y; y++)
    {
      for(int x = lowest.x; x <= highest.x; x++)
      {
        const int3 cell = int3(x, y, z);
        uint       begin, end;
        causticCellRun(causticCellSlot(pushConst.caustics, cell), begin, end);
        // A cell with many photons (the core of a caustic, in a cell several radii wide) is read with a stride, each
        // photon read standing for `stride` of them: the same expectation, and the cost of a cell is bounded
        const uint  stride    = max(1u, (end - begin + CAUSTIC_CELL_READS - 1u) / CAUSTIC_CELL_READS);
        const float standsFor = float(stride);
        for(uint index = begin + min(uint(strideShift * standsFor), end - begin); index < end; index += stride)
        {
          const CausticPhoton photon = causticParams->sorted[index];
          float               distanceSquared;
          // A cell of the same hash holds photons of another place: each cell counts for its own
          if(any(causticCell(pushConst.caustics, photon.position) != cell)
             || !causticOnDisk(photon, hit.pos, hit.geonrm, radius, distanceSquared))
            continue;
          const float3 from = -causticUnpackDirection(photon.direction);  // Toward where the photon came from
          // hit.geonrm faces the path: a photon that came from behind the surface lit its other side (a thin
          // quad, a wall seen across a corner), unless the surface passes light through
          if(!transmissive && dot(from, hit.geonrm) <= 0.0F)
            continue;
          count += int(stride);
          if(count > int(causticParams->gatherMax) && !unlimited)
            return false;

          const float  weight = standsFor * (1.0F - distanceSquared / radiusSquared);
          const float3 flux   = causticUnpackFlux(photon.flux);
          if(SHINY != 0)
          {
            BsdfEvaluateData evalData;
            evalData.k1 = from;
            evalData.k2 = outgoing;
            evalData.xi = float3(rand(seed), rand(seed), rand(seed));
            bsdfEvaluate(evalData, mat);
            const float3 response = evalData.bsdf_diffuse + evalData.bsdf_glossy;  // BSDF times the cosine toward `outgoing`
            sum += flux * response * (weight * causticAdjoint(from, outgoing, mat.N, hit.geonrm));
          }
          else
          {
            sum += flux * weight;
            fromSum += from * (weight * dot(flux, float3(1.0F / 3.0F)));
          }
        }
      }
    }
  }
  return true;
}

// The radiance a receiver `hit` sends toward `outgoing` (away from the surface) from the photons around it
// that followed casters, normalized by the photons of the frame. Where the caustic is bright the disk holds
// many photons: it shrinks, by halves of its area, until it holds no more than gatherMax, so the cores of a
// caustic keep their sharpness and the cost of a gather is bounded.
float3 causticGather(HitState hit, PbrMaterial mat, float3 outgoing, inout uint seed)
{
  CausticParams* causticParams = pushConst.caustics;
  const float    cosOutgoing   = dot(mat.N, outgoing);
  if(causticParams->photonCount == 0 || cosOutgoing <= 1e-4F)
    return float3(0.0F);

  // A transmissive receiver (a leaf) gets photons from both sides: the mean direction of those means nothing, so it
  // evaluates its BSDF for each photon, like a shiny one
  const bool transmissive = mat.diffuseTransmissionFactor > 0.0F || mat.transmission > 0.0F;
  const bool shiny        = transmissive || max(mat.roughness.x, mat.roughness.y) < CAUSTIC_SHINY_ROUGHNESS;
  // The disk covers a few cells at most (a receiver far from the casters, in a grid sized for them, would read
  // hundreds): the cost of a gather is bounded, and a farther receiver is noisier instead
  float  radius = min(causticGatherRadius(hit.pos, hit.geonrm), CAUSTIC_MAX_RADIUS_CELLS * causticParams->cellSize);
  float3 sum, fromSum;
  for(int halvings = 0; halvings < 12; halvings++)
  {
    // Two loops, not one with a branch: a shiny receiver's loop evaluates a BSDF per photon, and the rough one's is lean
    const bool done = shiny ? causticSum<1>(hit, mat, outgoing, radius, halvings == 11, seed, sum, fromSum) :
                              causticSum<0>(hit, mat, outgoing, radius, halvings == 11, seed, sum, fromSum);
    if(done)
      break;
    radius *= 0.70710678F;
    sum     = float3(0.0F);
    fromSum = float3(0.0F);
  }

  if(!shiny)  // A rough receiver evaluates its BSDF once, for the direction the photons' flux comes from on average
  {
    const float fromLength = length(fromSum);
    if(fromLength <= 0.0F)
      return float3(0.0F);
    const float3     from = fromSum / fromLength;
    BsdfEvaluateData evalData;
    evalData.k1 = from;
    evalData.k2 = outgoing;
    evalData.xi = float3(rand(seed), rand(seed), rand(seed));
    bsdfEvaluate(evalData, mat);
    sum *= (evalData.bsdf_diffuse + evalData.bsdf_glossy) * causticAdjoint(from, outgoing, mat.N, hit.geonrm);
  }
  // 2 / (pi r^2) normalizes the kernel. The response carries the shading cosine toward `outgoing` and the adjoint
  // the ratio of the geometric one to it: a radiance has the geometric one in neither, so divide by it
  const float kept = float(min(causticParams->photonCount, causticParams->mapSlots));  // The map holds the first of the photons
  return sum * (2.0F / (M_PI * radius * radius * kept * max(abs(dot(hit.geonrm, outgoing)), 1e-4F)));
}

// The photon landed on a receiver after crossing casters: keep it in the map (one slot per photon: a photon that
// goes on through a pane and lands again keeps the later place), and splat it to the camera
void causticLand(IRaytracer raytracer, uint slot, bool hasMap, HitState hit, PbrMaterial mat, float3 travel, float3 flux, inout uint seed)
{
  if(hasMap)
    causticStore(slot, hit.pos, travel, flux * float(pushConst.caustics->photonCount));
  causticSplat(raytracer, hit, mat, travel, flux, seed);
}

// One photon: from an emitter, through the casters, to the first surface that is not one.
void tracePhoton(IRaytracer raytracer, inout uint seed, float2 aim, uint slot)
{
  SceneFrameInfo*      frameInfo        = pushConst.frameInfo;
  GltfRenderNode*      renderNodes      = pushConst.gltfScene->renderNodes;
  GltfRenderPrimitive* renderPrimitives = pushConst.gltfScene->renderPrimitives;
  GltfShadeMaterial*   materials        = pushConst.gltfScene->materials;
  GltfTextureInfo*     texInfos         = pushConst.gltfScene->textureInfos;

  // This photon's slot of the map starts empty: it holds what the photon of the last frame left
  CausticParams* causticParams = pushConst.caustics;
  const bool     hasMap        = causticParams->photons != nullptr && slot < causticParams->mapSlots;
  if(hasMap)
    causticParams->photons[slot].flux = 0u;

  PhotonEmission emission = causticEmit(seed, aim);
  if(!emission.valid)
    return;

  RayDesc ray;
  ray.Origin    = emission.origin;
  ray.Direction = emission.direction;
  ray.TMin      = 0.0F;
  ray.TMax      = INFINITE;
  float3 flux   = emission.flux;
  int    chain  = 0;  // Casters crossed so far

#if GLTF_USE_TRANSMISSION || GLTF_USE_VOLUME
  bool isInside = false;
#endif
#if GLTF_USE_VOLUME
  VolumeMedium medium = {};
#endif

  for(int depth = 0; depth < CAUSTIC_MAX_DEPTH; depth++)
  {
    HitPayload payload = {};
    raytracer.Trace(ray, payload, seed);

    HitState            hit        = {};
    GltfRenderNode      renderNode = {};
    GltfRenderPrimitive renderPrim = {};
    if(payload.hitT != INFINITE)
    {
      renderNode   = renderNodes[payload.rnodeID];
      renderPrim   = renderPrimitives[payload.rprimID];
      float3 barys = float3(1.0 - payload.bary.x - payload.bary.y, payload.bary.x, payload.bary.y);
      hit = getHitState(renderPrim, barys, renderNode.worldToObject, renderNode.objectToWorld, payload.primitiveID, ray.Direction);
    }
    const bool hitPlane = checkInfinitePlaneIntersection(ray, payload, hit, frameInfo);
    if(payload.hitT == INFINITE)
      return;

    PbrMaterial mat;
    if(hitPlane)
    {
      if(hasFlag(frameInfo->flags, SceneFrameInfoFlags::eSceneInfinitePlaneShadowCatcher))
        return;  // a catcher shows shadows, not light
      mat = defaultPbrMaterial(frameInfo->infinitePlaneBaseColor, frameInfo->infinitePlaneMetallic,
                               frameInfo->infinitePlaneRoughness, hit.nrm, hit.nrm);
    }
    else
    {
      const int materialIndex = max(0, renderNode.materialID);
#if GLTF_USE_UNLIT
      if(materials[materialIndex].unlit > 0)
        return;  // unlit surfaces take no light
#endif
#if GLTF_USE_TRANSMISSION || GLTF_USE_VOLUME
      const bool inside = isInside;
#else
      const bool inside = false;
#endif
      MeshState mesh          = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, inside, 0.0F);
      mesh.baseColorVertexMul = hit.color;
      mat                     = evaluateMaterial(materialIndex, mesh, allImages, allSamplers, texInfos, materials);
    }

    // The medium the photon travelled through to get here
#if GLTF_USE_VOLUME
    if(isInside && hasVolumeMedium(medium))
      flux *= exp(-volumeExtinction(medium) * payload.hitT);
#endif

    // A thin pane lets the photon through: transmitted, it goes on straight, without making the chain longer
    // (it is not a caster). A photon that came through casters also lights the pane itself, like any receiver;
    // one the pane reflects ends here.
    if(!hitPlane && isCausticThinPane(mat))
    {
      if(chain > 0)
        causticLand(raytracer, slot, hasMap, hit, mat, ray.Direction, flux, seed);
      BsdfSampleData paneSample;
      paneSample.k1 = -ray.Direction;
      paneSample.xi = float3(rand(seed), rand(seed), rand(seed));
      bsdfSample(paneSample, mat);
      if((paneSample.event_type & BSDF_EVENT_TRANSMISSION) == 0)
        return;
      flux *= paneSample.bsdf_over_pdf;
      const float3 paneOffset = dot(paneSample.k2, hit.geonrm) > 0.0F ? hit.geonrm : -hit.geonrm;
      ray.Origin              = safeOffsetRay(hit.pos, paneOffset, hit.offsetDistance);
      ray.Direction           = paneSample.k2;
      continue;
    }

    if(hitPlane || !isCausticCaster(mat))
    {
      if(chain > 0)
        causticLand(raytracer, slot, hasMap, hit, mat, ray.Direction, flux, seed);
      return;  // a photon that has not met a caster is not a caustic: the other estimator has it
    }

    // A caster: scatter like the path tracer would, in the direction the photon travels
    BsdfSampleData sampleData;
    sampleData.k1 = -ray.Direction;
    sampleData.xi = float3(rand(seed), rand(seed), rand(seed));
    bsdfSample(sampleData, mat);
    if(sampleData.event_type == BSDF_EVENT_ABSORB)
      return;
    flux *= sampleData.bsdf_over_pdf * causticAdjoint(-ray.Direction, sampleData.k2, mat.N, hit.geonrm);

#if GLTF_USE_TRANSMISSION
    if((sampleData.event_type & BSDF_EVENT_TRANSMISSION) != 0 && mat.thickness > 0.0F)
    {
      isInside = !isInside;
#if GLTF_USE_VOLUME
      if(isInside)
        medium = makeVolumeMedium(mat);
#endif
    }
#endif

    const float3 offsetDir = dot(sampleData.k2, hit.geonrm) > 0.0F ? hit.geonrm : -hit.geonrm;
    ray.Origin             = safeOffsetRay(hit.pos, offsetDir, hit.offsetDistance);
    ray.Direction          = sampleData.k2;
    chain++;
  }
}

//-----------------------------------------------------------------------
// Where photon `index` of this frame aims: the 2D R2 low-discrepancy sequence (fixed point, so it stays
// exact for any count), shifted by a random offset that is the same for the whole frame. The photons of
// one frame then cover the aim disk or cone evenly instead of in clumps, which is what a frame with few
// photons, e.g. under DLSS, needs; the offset keeps one frame independent of the next, so the average
// over frames is still unbiased. Every photon is uniform on its own, so the estimate does not change.
//-----------------------------------------------------------------------
float2 causticAim(uint index)
{
  const uint2 shift =
      uint2(xxhash32(uint3(uint(pushConst.seed), 0xA511E9B3u, 1u)), xxhash32(uint3(uint(pushConst.seed), 0x63D83595u, 2u)));
  const uint2 r2 = uint2(index + 1u) * uint2(3242174889u, 2447445414u) + shift;  // 2^32 / (the plastic number, its square)
  return float2(r2 >> 8) * (1.0F / 16777216.0F);
}

//-----------------------------------------------------------------------
// The photon pass: one thread, one photon. It runs before the frame's paths, which add the splats
// it leaves to their pixels (processPixel).
//-----------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void causticMain(uint3 threadIdx: SV_DispatchThreadID)
{
  RAYQUERY_USE_OPACITY_MICROMAP(causticMain);
  if(threadIdx.x >= pushConst.caustics->photonCount)
    return;

  // The index of the photon among all the app traced, not the frame's: the photons of a frame the camera
  // reset the accumulation at are new ones, which the map keeps with the others
  const uint        emitIndex = pushConst.caustics->photonFirst + threadIdx.x;
  uint              seed = xxhash32(uint3(emitIndex, 0x7F4A7C15u, uint(pushConst.seed) * 0x9E3779B9u + 0x51ED270Bu));
  RayQueryRaytracer raytracer;
  tracePhoton(raytracer, seed, causticAim(emitIndex), threadIdx.x);
}

#endif  // CAUSTICS_H
