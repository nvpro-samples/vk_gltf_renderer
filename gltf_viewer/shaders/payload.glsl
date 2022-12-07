#ifndef PAYLOAD_H
#define PAYLOAD_H

precision highp float;

#include "hit_state.h"

struct HitPayload
{
  uint  seed;
  float hitT;
  int   instanceIndex;
  HitState hit;
//  vec3  pos;
//  vec3  nrm;
//  vec3  tangent;
//  vec3  bitangent;
//  vec2  uv;
};

#endif  // PAYLOAD_H