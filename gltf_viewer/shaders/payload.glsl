#ifndef PAYLOAD_H
#define PAYLOAD_H

precision highp float;

#define MISS_DEPTH 1000


struct HitPayload
{
  uint  seed;
  float hitT;
  vec3  contrib;
  vec3  weight;
  vec3  rayOrigin;
  vec3  rayDirection;
};

HitPayload initPayload()
{
  HitPayload p;
  p.seed         = 0U;
  p.hitT         = 0.F;
  p.contrib      = vec3(0.F);
  p.weight       = vec3(1.F);
  p.rayOrigin    = vec3(0.F);
  p.rayDirection = vec3(0.F, 0.F, -1.F);
  return p;
}

#endif  // PAYLOAD_H