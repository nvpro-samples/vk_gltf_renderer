// --------------------------------------------------------------------
// Forwarded declarations
void        traceRay(Ray r, inout uint seed);
vec3        traceShadow(Ray r, float maxDist, inout uint seed);
const float MIN_TRANSMISSION = 0.01;  // Minimum transmission factor to continue tracing


struct SampleResult
{
  vec4  radiance;
  vec3  normal;
  float depth;
};


// --------------------------------------------------------------------
// Sampling the Sun or the HDR
// - Returns
//      The contribution divided by PDF
//      The direction to the light source
//      The PDF
//
vec3 sampleLights(in vec3 pos, vec3 normal, in vec3 worldRayDirection, inout uint seed, out vec3 dirToLight, out float lightPdf, out float lightDist)
{
  vec3 radiance = vec3(0);
  lightPdf      = 0.0;
  float envPdf  = 0.0;
  lightDist     = INFINITE;

  // We use the one-sample model to perform multiple importance sampling
  // between point lights and environment lights.
  // See section 9.2.4 of https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf .

  // Probability we'll select each sampling scheme (TODO: adjust these based on
  // scene characteristics: light intensity vs environment intensity)
  float lightWeight = (sceneDesc.numLights > 0) ? 0.5 : 0.0;
  float envWeight   = (TEST_FLAG(frameInfo.flags, USE_SKY_FLAG) || frameInfo.envIntensity.x > 0.0) ? 0.5 : 0.0;

  // Normalize weights
  float totalWeight = lightWeight + envWeight;
  if(totalWeight == 0.0f)
    return vec3(0.);

  lightWeight /= totalWeight;
  envWeight /= totalWeight;

  // Decide whether to sample the light or the environment.
  bool sampleLights = (rand(seed) <= lightWeight);

  // We'll choose a direction from one technique, but for MIS we need the
  // PDFs of each technique in the direction we chose. That's why we always get
  // the light PDF (which is constant), but only sample a light if
  // `sampleLights` is true.

  // Lights
  if(lightWeight > 0)
  {
    lightPdf = 1.0 / sceneDesc.numLights;
    if(sampleLights)  // Use this technique for MIS?
    {
      int          lightIndex = min(int(rand(seed) * sceneDesc.numLights), sceneDesc.numLights - 1);
      Light        light      = RenderLightBuf(sceneDesc.lightAddress)._[lightIndex];
      LightContrib contrib    = singleLightContribution(light, pos, normal, vec2(rand(seed), rand(seed)));
      dirToLight              = -contrib.incidentVector;
      radiance                = contrib.intensity / (lightPdf * lightWeight);
      lightDist               = contrib.distance;
    }
  }

  // Environment
  if(envWeight > 0)
  {
    if(TEST_FLAG(frameInfo.flags, USE_SKY_FLAG))
    {
      if(!sampleLights)  // Use this technique for MIS?
      {
        vec2              random_sample = vec2(rand(seed), rand(seed));
        SkySamplingResult skySample     = samplePhysicalSky(skyInfo, random_sample);
        dirToLight                      = skySample.direction;
        envPdf                          = skySample.pdf;
        radiance                        = skySample.radiance / (envPdf * envWeight);
      }
      else
      {
        envPdf = samplePhysicalSkyPDF(skyInfo, dirToLight);
      }
    }
    else
    {
      if(!sampleLights)  // Use this technique for MIS?
      {
        vec3 rand_val     = vec3(rand(seed), rand(seed), rand(seed));
        vec4 radiance_pdf = environmentSample(hdrTexture, rand_val, dirToLight);
        envPdf            = radiance_pdf.w;
        radiance          = radiance_pdf.xyz * frameInfo.envIntensity.xyz / (envPdf * envWeight);
        dirToLight        = rotate(dirToLight, vec3(0, 1, 0), frameInfo.envRotation);
      }
      else
      {
        vec3 dir          = rotate(dirToLight, vec3(0, 1, 0), -frameInfo.envRotation);
        vec2 uv           = getSphericalUv(dir);
        vec4 radiance_pdf = textureLod(hdrTexture, uv, 0);
        envPdf            = radiance_pdf.w;
      }
    }
  }

  // MIS weight calculation
  float misWeight = (sampleLights ? lightPdf : envPdf) / (lightPdf + envPdf);
  radiance *= misWeight;
  // Update the total PDF
  lightPdf = lightWeight * lightPdf + envWeight * envPdf;

  return radiance;
}


//----------------------------------------------------------
// Testing if the hit is opaque or alpha-transparent
// Return true is opaque
//----------------------------------------------------------
float getOpacity(RenderNode renderNode, RenderPrimitive renderPrim, int triangleID, vec3 barycentrics)
{
  // Scene materials
  uint              matIndex = max(0, renderNode.materialID);
  GltfShadeMaterial mat      = GltfMaterialBuf(sceneDesc.materialAddress).m[matIndex];

  if(mat.alphaMode == ALPHA_OPAQUE)
    return 1.0;

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = getTriangleIndices(renderPrim, triangleID);

  float baseColorAlpha = 1;
  if(mat.usePbrSpecularGlossiness == 0)
  {
    baseColorAlpha = mat.pbrBaseColorFactor.a;
    if(isTexturePresent(mat.pbrBaseColorTexture))
    {
      // Retrieve the interpolated texture coordinate from the vertex
      vec2 uv = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);

      baseColorAlpha *= texture(texturesMap[nonuniformEXT(mat.pbrBaseColorTexture.index)], uv).a;
    }
  }
  else
  {
    baseColorAlpha = mat.pbrDiffuseFactor.a;
    if(isTexturePresent(mat.pbrDiffuseTexture))
    {
      vec2 uv = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);

      baseColorAlpha *= texture(texturesMap[nonuniformEXT(mat.pbrDiffuseTexture.index)], uv).a;
    }
  }

  baseColorAlpha *= getInterpolatedVertexColor(renderPrim, triangleIndex, barycentrics).a;

  if(mat.alphaMode == ALPHA_MASK)
  {
    return  baseColorAlpha >= mat.alphaCutoff ? 1.0 : 0.0;
  }
 
  return baseColorAlpha;
}

vec3 getShadowTransmission(RenderNode      renderNode,
                           RenderPrimitive renderPrim,
                           int             triangleID,
                           vec3            barycentrics,
                           float           hitT,
                           mat4x3          worldToObject,
                           vec3            rayDirection,
                           inout bool      isInside)
{
  uint              matIndex = max(0, renderNode.materialID);
  GltfShadeMaterial mat      = GltfMaterialBuf(sceneDesc.materialAddress).m[matIndex];

  // If hit a non-transmissive surface, terminate with full shadow
  if(mat.transmissionFactor <= MIN_TRANSMISSION)
  {
    return vec3(0.0);
  }


  // Get triangle indices and compute normal
  uvec3 indices = getTriangleIndices(renderPrim, triangleID);


  vec3 normal;
  {
    // Compute geometric normal
    vec3 v0 = getVertexPosition(renderPrim, indices.x);
    vec3 v1 = getVertexPosition(renderPrim, indices.y);
    vec3 v2 = getVertexPosition(renderPrim, indices.z);
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    normal  = normalize(cross(e1, e2));
    normal  = normalize(vec3(normal * worldToObject));
  }

  // Transmission calculation
  vec3 currentTransmission = vec3(mat.transmissionFactor);

  // Regular transmission with Fresnel using normal
  float cosTheta = abs(dot(rayDirection, normal));
  float fresnel  = schlickFresnel(mat.ior, cosTheta);
  currentTransmission *= vec3((1.0 - fresnel));

  // Apply material color tint to transmission
  currentTransmission *= mat.pbrBaseColorFactor.rgb;

  // Volume attenuation (Beer's law)
  if(mat.thicknessFactor > 0.0)
  {
    if(isInside)
    {
      // Calculate per-channel attenuation with improved color preservation
      vec3 absorbance = -log(max(mat.attenuationColor, vec3(0.001))) / max(mat.attenuationDistance, 0.001);
      vec3 attenuation;
      attenuation.r = exp(-hitT * absorbance.r);
      attenuation.g = exp(-hitT * absorbance.g);
      attenuation.b = exp(-hitT * absorbance.b);

      currentTransmission *= attenuation;
    }
    isInside = !isInside;
  }

  // Attenuation due to roughness and metallic
  float transmissionAttenuation = 1.0;
  {
    float roughness = mat.pbrRoughnessFactor;
    float metallic  = mat.pbrMetallicFactor;
    if(isTexturePresent(mat.pbrMetallicRoughnessTexture))
    {
      vec2 tc[2];
      tc[0]          = getInterpolatedVertexTexCoord0(renderPrim, indices, barycentrics);
      tc[1]          = getInterpolatedVertexTexCoord1(renderPrim, indices, barycentrics);
      vec4 mr_sample = getTexture(mat.pbrMetallicRoughnessTexture, tc);
      roughness *= mr_sample.g;
      metallic *= mr_sample.b;
    }

    // Metallic completely blocks transmission
    transmissionAttenuation *= (1.0 - metallic);

    // Roughness reduces transmission non-linearly
    float roughnessEffect = 1.0 - (roughness * roughness);
    transmissionAttenuation *= mix(0.65, 1.0, roughnessEffect);
  }
  currentTransmission *= transmissionAttenuation;

  return currentTransmission;
}

//-----------------------------------------------------------------------
// Samples a 2D Gaussian distribution with a standard distribution of 1,
// using the Box-Muller algorithm.
// The input must be two random numbers in the range [0,1].
//-----------------------------------------------------------------------
vec2 sampleGaussian(vec2 u)
{
  const float r     = sqrt(-2.0f * log(max(1e-38f, u.x)));  // Radius
  const float theta = 2.0f * M_PI * u.y;                    // Angle
  return r * vec2(cos(theta), sin(theta));
}

//-----------------------------------------------------------------------
// Standard deviation of the Gaussian filter used for antialiasing,
// in units of pixels.
// This value of 1 / sqrt(8 ln(2)) makes it so that a Gaussian centered
// on a pixel is at exactly 1/2 its maximum at the midpoints between
// orthogonally adjacent pixels, and 1/4 its maximum at the "corners"
// of pixels. It also empirically looks nice: larger values are
// too blurry, and smaller values make thin lines look jagged.
//-----------------------------------------------------------------------
#define ANTIALIASING_STANDARD_DEVIATION 0.4246609F

Ray getRay(vec2 samplePos, vec2 offset, vec2 imageSize, mat4 projMatrixI, mat4 viewMatrixI)
{
  const vec2 pixelCenter = vec2(samplePos.xy) + offset;
  const vec2 inUV        = pixelCenter / vec2(imageSize.xy);
  const vec2 d           = inUV * 2.0 - 1.0;
  const vec4 origin      = viewMatrixI * vec4(0.0F, 0.0F, 0.0F, 1.0F);
  const vec4 target      = projMatrixI * vec4(d.x, d.y, 0.01F, 1.0F);
  const vec4 direction   = viewMatrixI * vec4(normalize(target.xyz), 0.0F);

  return Ray(origin.xyz, direction.xyz);
}

//-----------------------------------------------------------------------
//
//-----------------------------------------------------------------------
SampleResult pathTrace(Ray r, inout uint seed)
{
  vec3 radiance     = vec3(0.0F);
  vec3 throughput   = vec3(1.0F);
  bool isInside     = false;
  vec2 maxRoughness = vec2(0.0);

  SampleResult sampleResult;
  sampleResult.depth    = 0;
  sampleResult.normal   = vec3(0, 0, 0);
  sampleResult.radiance = vec4(0, 0, 0, 1);

  GltfMaterialBuf materials = GltfMaterialBuf(sceneDesc.materialAddress);

  float lastSamplePdf = DIRAC;

  for(int depth = 0; depth < pc.maxDepth; depth++)
  {
    bool firstRay = (depth == 0);
    traceRay(r, seed);

    HitState hit = hitPayload.hit;

    // Hitting the environment, then exit
    if(hitPayload.hitT == INFINITE)
    {
      if(firstRay)  // If we come in here, the first ray didn't hit anything
      {
        sampleResult.radiance.a = 0.0;  // Set it to transparent

        // Solid color background and blurred HDR environment, aren't part of the
        // lighting equation (backplate), so we can return them directly.
        if(TEST_FLAG(frameInfo.flags, USE_SOLID_BACKGROUND_FLAG))
        {
          sampleResult.radiance.xyz = frameInfo.backgroundColor;
          return sampleResult;
        }
        else if(TEST_FLAG(frameInfo.flags, USE_HDR_FLAG) && frameInfo.envBlur > 0)
        {
          vec3 dir                  = rotate(r.direction, vec3(0, 1, 0), -frameInfo.envRotation);
          vec2 uv                   = getSphericalUv(dir);  // See sampling.glsl
          sampleResult.radiance.xyz = smoothHDRBlur(hdrTexture, uv, frameInfo.envBlur).xyz;
          return sampleResult;
        }
      }

      // Add sky or HDR texture
      vec3  envColor;
      float envPdf;
      if(TEST_FLAG(frameInfo.flags, USE_SKY_FLAG))
      {
        envColor = evalPhysicalSky(skyInfo, r.direction);
        envPdf   = samplePhysicalSkyPDF(skyInfo, r.direction);
      }
      else
      {
        // Adding HDR lookup
        vec3 dir = rotate(r.direction, vec3(0, 1, 0), -frameInfo.envRotation);
        vec2 uv  = getSphericalUv(dir);  // See sampling.glsl
        vec4 env = textureLod(hdrTexture, uv, 0);
        envColor = env.rgb * frameInfo.envIntensity.xyz;
        envPdf   = env.w;
      }

      // We may hit the environment twice: once via sampleLights() and once when hitting the sky while probing
      // for more indirect hits. This is the counter part of the MIS weighting in sampleLights()
      float misWeight = (lastSamplePdf == DIRAC) ? 1.0 : (lastSamplePdf / (lastSamplePdf + envPdf));
      radiance += throughput * misWeight * envColor;

      sampleResult.radiance.xyz = radiance;
      return sampleResult;
    }

    if(depth == 0)
    {
      sampleResult.normal = hit.nrm;
      sampleResult.depth  = hitPayload.hitT;
    }

    // Retrieve the Instance buffer information
    RenderNode      renderNode = RenderNodeBuf(sceneDesc.renderNodeAddress)._[hitPayload.rnodeID];
    RenderPrimitive renderPrim = RenderPrimitiveBuf(sceneDesc.renderPrimitiveAddress)._[hitPayload.rprimID];

    // Setting up the material
    GltfShadeMaterial material = materials.m[renderNode.materialID];  // Material of the hit object
    material.pbrBaseColorFactor *= hit.color;                         // Color at vertices
    MeshState   mesh   = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, isInside);
    PbrMaterial pbrMat = evaluateMaterial(material, mesh);

    // Keep track of the maximum roughness to prevent firefly artifacts
    // by forcing subsequent bounces to be at least as rough
    maxRoughness     = max(pbrMat.roughness, maxRoughness);
    pbrMat.roughness = maxRoughness;

    // Adding emissive
    radiance += pbrMat.emissive * throughput;

    // Unlit
    if(material.unlit > 0)
    {
      radiance += pbrMat.baseColor;
      sampleResult.radiance.xyz = radiance;
      sampleResult.radiance.a   = 1.0;
      return sampleResult;
    }

    // Apply volume attenuation
    if(isInside && !pbrMat.isThinWalled)
    {
      const vec3 abs_coeff = absorptionCoefficient(pbrMat);
      throughput *= exp(-hitPayload.hitT * abs_coeff);
    }

    // Light contribution; can be environment or punctual lights
    vec3  contribution         = vec3(0);
    vec3  dirToLight           = vec3(0);
    float lightPdf             = 0.F;
    float lightDist            = 0.F;
    vec3  lightRadianceOverPdf = sampleLights(hit.pos, pbrMat.N, r.direction, seed, dirToLight, lightPdf, lightDist);

    // Do not next event estimation (but delay the adding of contribution)
    bool nextEventValid = (dot(dirToLight, hit.geonrm) > 0.0f) && lightPdf != 0.0f;

    // Evaluate BSDF for Light
    if(nextEventValid)
    {
      BsdfEvaluateData evalData;
      evalData.k1 = -r.direction;
      evalData.k2 = dirToLight;
      evalData.xi = vec3(rand(seed), rand(seed), rand(seed));

      bsdfEvaluate(evalData, pbrMat);

      if(evalData.pdf > 0.0)
      {
        const float mis_weight = (lightPdf == DIRAC) ? 1.0f : lightPdf / (lightPdf + evalData.pdf);

        // sample weight
        const vec3 w = throughput * lightRadianceOverPdf * mis_weight;
        contribution += w * evalData.bsdf_diffuse;
        contribution += w * evalData.bsdf_glossy;
      }
    }

    // Sample BSDF
    {
      BsdfSampleData sampleData;
      sampleData.k1            = -r.direction;  // outgoing direction
      sampleData.xi            = vec3(rand(seed), rand(seed), rand(seed));
      sampleData.event_type    = 0;                    ///< output: the type of event for the generated sample
      sampleData.pdf           = 0.0;                  // output: pdf (non-projected hemisphere)
      sampleData.bsdf_over_pdf = vec3(0.0, 0.0, 0.0);  ///< output: bsdf * dot(normal, k2) / pdf

      bsdfSample(sampleData, pbrMat);

      throughput *= sampleData.bsdf_over_pdf;
      r.direction   = sampleData.k2;
      lastSamplePdf = sampleData.pdf;

      if(sampleData.event_type == BSDF_EVENT_ABSORB)
      {
        // Exit tracing rays, but still finish this iteration; in particular the visibility test
        // for the light that we may have hit.
        depth = pc.maxDepth;
      }
      else
      {
        // Continue path
        bool isSpecular     = (sampleData.event_type & BSDF_EVENT_IMPULSE) != 0;
        bool isTransmission = (sampleData.event_type & BSDF_EVENT_TRANSMISSION) != 0;

        vec3 offsetDir = dot(r.direction, hit.geonrm) > 0 ? hit.geonrm : -hit.geonrm;
        r.origin       = offsetRay(hit.pos, offsetDir);

        // Flip the information if we are inside the object, but only if it is a solid object
        // The doubleSided flag is used to know if the object is solid or thin-walled.
        // This is not a glTF specification, but works in many cases.
        if(isTransmission)
        {
          isInside = !isInside;
        }
      }
    }

    // We are adding the contribution to the radiance only if the ray is not occluded by an object.
    if(nextEventValid)
    {
      vec3 shadowRayOrigin = offsetRay(hit.pos, hit.geonrm);
      Ray  shadowRay       = Ray(shadowRayOrigin, dirToLight);
      vec3 shadowFactor    = traceShadow(shadowRay, lightDist, seed);
      // We are adding the contribution to the radiance only if the ray is not occluded by an object.
      radiance += contribution * shadowFactor;
    }

#if USE_RUSIAN_ROULETTE
    // Russian-Roulette (minimizing live state)
    float rrPcont = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001F, 0.95F);
    if(rand(seed) >= rrPcont)
      break;                // paths with low throughput that won't contribute
    throughput /= rrPcont;  // boost the energy of the non-terminated paths
#endif
  }

  sampleResult.radiance.xyz = radiance;
  return sampleResult;
}


//-----------------------------------------------------------------------
// Sampling the pixel
//-----------------------------------------------------------------------
SampleResult samplePixel(inout uint seed, vec2 samplePos, vec2 subpixelJitter, vec2 imageSize, mat4 projMatrixI, mat4 viewMatrixI, float focalDist, float aperture)
{
  Ray ray = getRay(samplePos, subpixelJitter, imageSize, projMatrixI, viewMatrixI);

  // Depth-of-Field
  vec3  focalPoint        = focalDist * ray.direction;
  float cam_r1            = rand(seed) * M_TWO_PI;
  float cam_r2            = rand(seed) * aperture;
  vec4  cam_right         = viewMatrixI * vec4(1, 0, 0, 0);
  vec4  cam_up            = viewMatrixI * vec4(0, 1, 0, 0);
  vec3  randomAperturePos = (cos(cam_r1) * cam_right.xyz + sin(cam_r1) * cam_up.xyz) * sqrt(cam_r2);
  vec3  finalRayDir       = normalize(focalPoint - randomAperturePos);

  ray = Ray(ray.origin + randomAperturePos, finalRayDir);

  SampleResult sampleResult = pathTrace(ray, seed);

// Removing fireflies
#if USE_FIREFLY_FILTER
  float lum = dot(sampleResult.radiance.xyz, vec3(1.0F / 3.0F));
  if(lum > pc.maxLuminance)
  {
    sampleResult.radiance *= pc.maxLuminance / lum;
  }
#endif

  return sampleResult;
}


//---
vec3 debugRendering(vec2 samplePos, vec2 imageSize)
{
  uint seed = 0;

  Ray ray = getRay(samplePos, vec2(0, 0), imageSize, frameInfo.projMatrixI, frameInfo.viewMatrixI);
  traceRay(ray, seed);

  HitState hit = hitPayload.hit;

  if(hitPayload.hitT == INFINITE)
    return vec3(0);

  // Retrieve the Instance buffer information
  RenderNode      renderNode = RenderNodeBuf(sceneDesc.renderNodeAddress)._[hitPayload.rnodeID];
  RenderPrimitive renderPrim = RenderPrimitiveBuf(sceneDesc.renderPrimitiveAddress)._[hitPayload.rprimID];

  // Setting up the material
  GltfMaterialBuf   materials = GltfMaterialBuf(sceneDesc.materialAddress);  // Buffer of materials
  GltfShadeMaterial material  = materials.m[renderNode.materialID];          // Material of the hit object
  MeshState         mesh      = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, false);
  PbrMaterial       pbrMat    = evaluateMaterial(material, mesh);

  switch(pc.dbgMethod)
  {
    case eDbgMethod_metallic:
      return vec3(pbrMat.metallic);
    case eDbgMethod_roughness:
      return vec3(pbrMat.roughness.xy, 0);
    case eDbgMethod_normal:
      return vec3(pbrMat.N * .5 + .5);
    case eDbgMethod_tangent:
      return pbrMat.T * .5 + .5;
    case eDbgMethod_bitangent:
      return pbrMat.B * .5 + .5;
    case eDbgMethod_baseColor:
      return vec3(pbrMat.baseColor);
    case eDbgMethod_emissive:
      return vec3(pbrMat.emissive);
    case eDbgMethod_opacity:
      return vec3(pbrMat.opacity * (1.0 - pbrMat.transmission));
    case eDbgMethod_texCoord0:
      return vec3(hit.uv[0], 0);
    case eDbgMethod_texCoord1:
      return vec3(hit.uv[1], 0);
  }

  return vec3(0);
}
