
// Forward declarations
float getOpacity(RenderNode renderNode, RenderPrimitive renderPrim, int triangleID, vec3 barycentrics);
vec3  getShadowTransmission(RenderNode      renderNode,
                            RenderPrimitive renderPrim,
                            int             triangleID,
                            vec3            barycentrics,
                            float           hitT,
                            mat4x3          worldToObject,
                            vec3            rayDirection,
                            inout bool      isInside);
Ray   getRay(vec2 samplePos, vec2 offset, vec2 imageSize, mat4 projMatrixI, mat4 viewMatrixI);


//-----------------------------------------------------------------------
// Shoot a ray an return the information of the closest hit, in the
// PtPayload structure (PRD)
//
void traceRay(Ray ray, inout uint seed)
{
  rayQueryEXT rayQuery;

  hitPayload.hitT = 0.0F;
  uint rayFlags   = gl_RayFlagsNoneEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
  rayQueryInitializeEXT(rayQuery, topLevelAS, rayFlags, 0xFF, ray.origin, 0.0, ray.direction, INFINITE);

  while(rayQueryProceedEXT(rayQuery))
  {
    if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionTriangleEXT)
    {
      int  instanceID   = rayQueryGetIntersectionInstanceIdEXT(rayQuery, false);
      int  renderPrimID = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false);
      int  triangleID   = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);
      vec2 bary         = rayQueryGetIntersectionBarycentricsEXT(rayQuery, false);

      // Barycentric coordinate on the triangle
      const vec3 barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);

      RenderNode      renderNode = RenderNodeBuf(sceneDesc.renderNodeAddress)._[instanceID];
      RenderPrimitive renderPrim = RenderPrimitiveBuf(sceneDesc.renderPrimitiveAddress)._[renderPrimID];

      float opacity = getOpacity(renderNode, renderPrim, triangleID, barycentrics);

      // do alpha blending the stochastically way
      if(rand(seed) <= opacity)
        rayQueryConfirmIntersectionEXT(rayQuery);
    }
  }

  if(rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT)
  {
    vec2   bary              = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    int    instanceID        = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
    int    renderPrimID      = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    int    triangleID        = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    mat4x3 worldToObject     = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
    mat4x3 objectToWorld     = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
    float  hitT              = rayQueryGetIntersectionTEXT(rayQuery, true);
    vec3   worldRayDirection = rayQueryGetWorldRayDirectionEXT(rayQuery);
    vec3   worldRayOrigin    = rayQueryGetWorldRayOriginEXT(rayQuery);

    // Retrieve the Primitive mesh buffer information
    RenderNode      renderNode = RenderNodeBuf(sceneDesc.renderNodeAddress)._[instanceID];
    RenderPrimitive renderPrim = RenderPrimitiveBuf(sceneDesc.renderPrimitiveAddress)._[renderPrimID];

    hitPayload.hitT    = hitT;
    hitPayload.rnodeID = instanceID;
    hitPayload.rprimID = renderPrimID;  // Should be equal to renderNode.rprimID

    const vec3 barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);

    hitPayload.hit = getHitState(renderPrim, barycentrics, triangleID, worldRayOrigin, objectToWorld, worldToObject);
  }
  else
  {
    hitPayload.hitT = INFINITE;
  }
}

//-----------------------------------------------------------------------
// Shadow ray - return true if a ray hits anything
//
vec3 traceShadow(Ray ray, float maxDist, inout uint seed)
{
  const float MIN_TRANSMISSION  = 0.01;  // Minimum transmission factor to continue tracing
  vec3        totalTransmission = vec3(1.0);
  bool        isInside          = false;
  float       approxHitT        = 0;

  rayQueryEXT rayQuery;
  uint        rayFlags = gl_RayFlagsNoneEXT;  // Start with no flags

  rayQueryInitializeEXT(rayQuery, topLevelAS, rayFlags, 0xFF, ray.origin, 0.0, ray.direction, maxDist);

  // Collect all potential hits first
  while(rayQueryProceedEXT(rayQuery))
  {
    float  hitT          = rayQueryGetIntersectionTEXT(rayQuery, false);
    int    instanceID    = rayQueryGetIntersectionInstanceIdEXT(rayQuery, false);
    int    renderPrimID  = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false);
    int    triangleID    = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);
    vec2   bary          = rayQueryGetIntersectionBarycentricsEXT(rayQuery, false);
    mat4x3 worldToObject = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, false);

    RenderNode      renderNode   = RenderNodeBuf(sceneDesc.renderNodeAddress)._[instanceID];
    RenderPrimitive renderPrim   = RenderPrimitiveBuf(sceneDesc.renderPrimitiveAddress)._[renderPrimID];
    vec3            barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    float           opacity      = getOpacity(renderNode, renderPrim, triangleID, barycentrics);

    float r = rand(seed);
    if(r < opacity)
    {
      approxHitT               = abs(hitT - approxHitT);
      vec3 currentTransmission = getShadowTransmission(renderNode, renderPrim, triangleID, barycentrics, approxHitT,
                                                       worldToObject, ray.direction, isInside);

      totalTransmission *= currentTransmission;

      if(max(max(totalTransmission.r, totalTransmission.g), totalTransmission.b) <= MIN_TRANSMISSION)
      {
        return vec3(0.0);
      }
    }
  }

  // It is possible that we didn't get any candidate, because the object had the OPAQUE flag, which than
  // means it would not have entered any hit information.
  if(rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT)
  {
    return vec3(0.0);
  }

  return totalTransmission;
}


//-----------------------------------------------------------------------
// Shoot a ray and store 1 if the ray hits the selected object
void selectObject(vec2 samplePos, vec2 imageSize)
{
  if(pc.selectedRenderNode <= -1)
    return;

  float g_selectedObject = 0.0;

  Ray ray            = getRay(samplePos, vec2(0, 0), imageSize, frameInfo.projMatrixI, frameInfo.viewMatrixI);
  hitPayload.rnodeID = -1;
  uint rayFlags      = gl_RayFlagsOpaqueEXT;
  {
    rayQueryEXT rayQuery;
    rayQueryInitializeEXT(rayQuery, topLevelAS, rayFlags, 0xFF, ray.origin, 0.0, ray.direction, INFINITE);
    while(rayQueryProceedEXT(rayQuery))
    {
      if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionTriangleEXT)
      {
        rayQueryConfirmIntersectionEXT(rayQuery);
      }
    }
    if(rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT)
    {
      hitPayload.rnodeID = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
    }
  }

  if(hitPayload.rnodeID != -1 && hitPayload.rnodeID == pc.selectedRenderNode)
    g_selectedObject = 1.0f;

  imageStore(selectImage, ivec2(samplePos.xy), vec4(g_selectedObject, 0, 0, 1));
}
