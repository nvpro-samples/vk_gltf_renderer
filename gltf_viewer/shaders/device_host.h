#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#ifdef __cplusplus
using mat4  = glm::mat4;
using vec4  = glm::vec4;
using vec3  = glm::vec3;
using Light = nvvkhl_shaders::Light;
#endif  // __cplusplus

#include "nvvkhl/shaders/dh_lighting.h"

struct PushConstant
{
  int frame;       // For RTX
  int maxDepth;    // For RTX
  int maxSamples;  // For RTX
  int materialID;  // For raster
  int instanceID;  // Instance nodel ID
  int meshID;      // Mesh ID of the node
};


#define MAX_NB_LIGHTS 1
#define WORKGROUP_SIZE 16

const int eDbgMethod_none      = 0;
const int eDbgMethod_metallic  = 1;
const int eDbgMethod_roughness = 2;
const int eDbgMethod_normal    = 3;
const int eDbgMethod_basecolor = 4;
const int eDbgMethod_emissive  = 5;

struct FrameInfo
{
  mat4  proj;
  mat4  view;
  mat4  projInv;
  mat4  viewInv;
  Light light[MAX_NB_LIGHTS];
  vec4  envColor;
  vec3  camPos;
  int   useSky;
  int   nbLights;
  float envRotation;
  float maxLuminance;
  int   dbgMethod;
};


#endif  // HOST_DEVICE_H
