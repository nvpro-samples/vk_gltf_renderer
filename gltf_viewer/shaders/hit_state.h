
#ifndef HIT_STATE_H
#define HIT_STATE_H

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  vec3 pos;
  vec3 shadowpos;
  vec3 nrm;
  vec3 geonrm;
  vec2 uv;
  vec3 tangent;
  vec3 bitangent;
};

#endif // HIT_STATE_H