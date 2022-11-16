// This function returns the geometric information at hit point
// Note: depends on the buffer layout PrimMeshInfo

#ifndef GETHIT_GLSL
#define GETHIT_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "nvvkhl/shaders/ray_util.glsl"

precision highp float;

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  vec3  pos;
  vec3  nrm;
  vec3  geonrm;
  vec2  uv;
  vec3  tangent;
  vec3  bitangent;
  float bitangentSign;
};


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
HitState getHitState(uint64_t vertexAddress, uint64_t indexAddress)
{
  HitState hit;

  // Vextex and indices of the primitive
  Vertices vertices = Vertices(vertexAddress);
  Indices  indices  = Indices(indexAddress);

  // Barycentric coordinate on the triangle
  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = indices.i[gl_PrimitiveID];

  // All vertex attributes of the triangle.
  Vertex v0 = vertices.v[triangleIndex.x];
  Vertex v1 = vertices.v[triangleIndex.y];
  Vertex v2 = vertices.v[triangleIndex.z];

  // Triangle info
  const vec3 pos0 = v0.position.xyz;
  const vec3 pos1 = v1.position.xyz;
  const vec3 pos2 = v2.position.xyz;
  const vec3 nrm0 = v0.normal.xyz;
  const vec3 nrm1 = v1.normal.xyz;
  const vec3 nrm2 = v2.normal.xyz;
  const vec2 uv0  = vec2(v0.position.w, v0.normal.w);
  const vec2 uv1  = vec2(v1.position.w, v1.normal.w);
  const vec2 uv2  = vec2(v2.position.w, v2.normal.w);
  const vec4 tng0 = vec4(v0.tangent);
  const vec4 tng1 = vec4(v1.tangent);
  const vec4 tng2 = vec4(v2.tangent);

  // Position
  hit.pos = mixBary(pos0, pos1, pos2, barycentrics);
  hit.pos = pointOffset(hit.pos, pos0, pos1, pos2, nrm0, nrm1, nrm2, barycentrics);  // Shadow offset position - hacking shadow terminator
  hit.pos = vec3(gl_ObjectToWorldEXT * vec4(hit.pos, 1.0));

  // Normal
  hit.nrm    = normalize(mixBary(nrm0, nrm1, nrm2, barycentrics));
  hit.nrm    = normalize(vec3(hit.nrm * gl_WorldToObjectEXT));
  hit.geonrm = normalize(cross(pos1 - pos0, pos2 - pos0));
  hit.geonrm = normalize(vec3(hit.geonrm * gl_WorldToObjectEXT));

  // TexCoord
  hit.uv = mixBary(uv0, uv1, uv2, barycentrics);

  // Tangent - Bitangent
  hit.tangent       = normalize(mixBary(tng0.xyz, tng1.xyz, tng2.xyz, barycentrics));
  hit.tangent       = vec3(gl_ObjectToWorldEXT * vec4(hit.tangent, 0.0));
  hit.tangent       = normalize(hit.tangent - hit.nrm * dot(hit.nrm, hit.tangent));
  hit.bitangent     = cross(hit.nrm, hit.tangent) * tng0.w;
  hit.bitangentSign = tng0.w;

  // Adjusting normal
  const vec3 V = -gl_WorldRayDirectionEXT;
  if(dot(hit.geonrm, V) < 0)  // Flip if back facing
    hit.geonrm = -hit.geonrm;

  // If backface
  if(dot(hit.geonrm, hit.nrm) < 0)  // Make Normal and GeoNormal on the same side
  {
    hit.nrm       = -hit.nrm;
    hit.tangent   = -hit.tangent;
    hit.bitangent = -hit.bitangent;
  }

  return hit;
}


#endif
