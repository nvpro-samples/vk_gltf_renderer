// --------------------------------------------------------------------
// Forwarded declarations
void traceRay(Ray r, inout uint seed);
bool traceShadow(Ray r, float maxDist, inout uint seed);


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

  // Weight for MIS (TODO: adjust these based on scene characteristics: light intensity vs environment intensity)
  float lightWeight = (sceneDesc.numLights > 0) ? 0.5 : 0.0;
  float envWeight   = (TEST_FLAG(frameInfo.flags, USE_SKY_FLAG) || frameInfo.envIntensity.x > 0.0) ? 0.5 : 0.0;

  // Normalize weights
  float totalWeight = lightWeight + envWeight;
  if(totalWeight == 0.0f)
    return vec3(0.);

  lightWeight /= totalWeight;
  envWeight /= totalWeight;

  // Sample lights
  vec3 lightRadiance = vec3(0);
  vec3 lightDir;
  if(sceneDesc.numLights > 0 && lightWeight > 0)
  {
    int          lightIndex = min(int(rand(seed) * sceneDesc.numLights), sceneDesc.numLights - 1);
    Light        light      = RenderLightBuf(sceneDesc.lightAddress)._[lightIndex];
    LightContrib contrib    = singleLightContribution(light, pos, normal, vec2(rand(seed), rand(seed)));
    lightDir                = -contrib.incidentVector;
    lightPdf                = (1.0 / sceneDesc.numLights) * lightWeight;
    lightRadiance           = contrib.intensity / lightPdf;
    lightDist               = contrib.distance;
  }

  // Sample environment
  vec3 envRadiance = vec3(0);
  vec3 envDir;
  if(envWeight > 0)
  {
    if(TEST_FLAG(frameInfo.flags, USE_SKY_FLAG))
    {
      vec2              random_sample = vec2(rand(seed), rand(seed));
      SkySamplingResult skySample     = samplePhysicalSky(skyInfo, random_sample);
      envDir                          = skySample.direction;
      envPdf                          = skySample.pdf * envWeight;
      envRadiance                     = skySample.radiance / envPdf;
    }
    else
    {
      vec3 rand_val     = vec3(rand(seed), rand(seed), rand(seed));
      vec4 radiance_pdf = environmentSample(hdrTexture, rand_val, envDir);
      envRadiance       = radiance_pdf.xyz;
      envPdf            = radiance_pdf.w * envWeight;
      envDir            = rotate(envDir, vec3(0, 1, 0), frameInfo.envRotation);
      envRadiance *= frameInfo.envIntensity.xyz;
      envRadiance /= envPdf;
    }
  }

  // Choose between light and environment using MIS
  float rnd = rand(seed);
  if(rnd < lightWeight && lightWeight > 0)
  {
    dirToLight = lightDir;
    radiance   = lightRadiance;
    // MIS weight calculation
    float misWeight = lightPdf / (lightPdf + envPdf);
    radiance *= misWeight;
  }
  else if(envWeight > 0)
  {
    dirToLight = envDir;
    radiance   = envRadiance;
    // MIS weight calculation
    float misWeight = envPdf / (lightPdf + envPdf);
    radiance *= misWeight;
  }

  // Update the total PDF
  lightPdf = lightPdf + envPdf;

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

  float baseColorAlpha = mat.pbrBaseColorFactor.a;
  if(isTexturePresent(mat.pbrBaseColorTexture))
  {

    // Getting the 3 indices of the triangle (local)
    uvec3 triangleIndex = getTriangleIndices(renderPrim, triangleID);

    // Retrieve the interpolated texture coordinate from the vertex
    vec2 uv = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);

    baseColorAlpha *= texture(texturesMap[nonuniformEXT(mat.pbrBaseColorTexture.index)], uv).a;
  }

  float opacity;
  if(mat.alphaMode == ALPHA_MASK)
  {
    opacity = baseColorAlpha > mat.alphaCutoff ? 1.0 : 0.0;
  }
  else
  {
    opacity = baseColorAlpha;
  }

  return opacity;
}

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
  vec3 radiance   = vec3(0.0F);
  vec3 throughput = vec3(1.0F);
  bool isInside   = false;

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

      if(TEST_FLAG(frameInfo.flags, USE_SKY_FLAG))
      {
        radiance.xyz += throughput * evalPhysicalSky(skyInfo, r.direction);
      }
      else
      {
        // Adding HDR lookup
        vec3 dir = rotate(r.direction, vec3(0, 1, 0), -frameInfo.envRotation);
        vec2 uv  = getSphericalUv(dir);  // See sampling.glsl
        vec4 env = textureLod(hdrTexture, uv, 0);

        // We may hit the environment twice: once via sampleLights() and once when hitting the sky while probing
        // for more indirect hits. This is the counter part of the MIS weighting in sampleLights()
        float misWeight = (lastSamplePdf == DIRAC) ? 1.0 : (lastSamplePdf / (lastSamplePdf + env.w));
        radiance.xyz += throughput * misWeight * env.rgb * frameInfo.envIntensity.xyz;
      }

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
      bool inShadow        = traceShadow(shadowRay, lightDist, seed);
      // We are adding the contribution to the radiance only if the ray is not occluded by an object.
      if(!inShadow)
      {
        radiance += contribution;
      }
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
