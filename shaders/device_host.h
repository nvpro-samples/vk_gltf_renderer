#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#include "nvvkhl/shaders/dh_lighting.h"

#ifdef __cplusplus
using mat4  = glm::mat4;
using vec4  = glm::vec4;
using vec3  = glm::vec3;
using vec2  = glm::vec2;
using Light = nvvkhl_shaders::Light;
// clang-format off
#define ENUM_START(name) enum name {
#define ENUM_ENTRY(name, value) name = value,
#define ENUM_END() };                                                                                                      
//clang-format on
#else
bool useDebug = false;
#define ENUM_START(name)
#define ENUM_ENTRY(name, value) const int name = value;
#define ENUM_END()
#endif  // __cplusplus

ENUM_START(EDebugMethod)
ENUM_ENTRY(eDbgMethod_none, 0)
ENUM_ENTRY(eDbgMethod_metallic, 1)
ENUM_ENTRY(eDbgMethod_roughness, 2)
ENUM_ENTRY(eDbgMethod_normal, 3)
ENUM_ENTRY(eDbgMethod_tangent, 4)
ENUM_ENTRY(eDbgMethod_bitangent, 5)
ENUM_ENTRY(eDbgMethod_baseColor, 6)
ENUM_ENTRY(eDbgMethod_emissive, 7)
ENUM_ENTRY(eDbgMethod_opacity, 8)
ENUM_ENTRY(eDbgMethod_texCoord0, 9)
ENUM_ENTRY(eDbgMethod_texCoord1, 10)
ENUM_END()

// Define bit flags for useSky and useSolidBackground
#define USE_SKY_FLAG				(1 << 0)
#define USE_HDR_FLAG				(1 << 1)
#define USE_SOLID_BACKGROUND_FLAG	(1 << 2)

// Macros to set and test the flags
#define SET_FLAG(flags, flag) ((flags) |= (flag))
#define CLEAR_FLAG(flags, flag) ((flags) &= ~(flag))
#define TEST_FLAG(flags, flag) bool((flags) & (flag))


struct PushConstantPathtracer
{
  int   frame;
  int   maxDepth;
  int   maxSamples;
  float maxLuminance;
  int   dbgMethod;
  int   selectedRenderNode;
  float focalDistance;
  float aperture;
  vec2  mouseCoord;  // Debugging (printf) mouse coordinates
};

struct PushConstantRaster
{
  int materialID;
  int renderNodeID;
  int renderPrimID;
  int dbgMethod;
  int selectedRenderNode;
};

struct PushConstantSilhouette
{
  vec3 color;
};

struct PushConstantDenoiser
{
  int   stepWidth;
  float colorPhi;
  float normalPhi;
  float depthPhi;
};

#define MAX_NB_LIGHTS 1
#define WORKGROUP_SIZE 16


struct SceneFrameInfo
{
  mat4  projMatrix;
  mat4  projMatrixI;
  mat4  viewMatrix;
  mat4  viewMatrixI;
  Light light[MAX_NB_LIGHTS];
  vec4  envIntensity;
  vec3  camPos;  // camera position
  int   flags;   // Use flag bits instead of separate useSky and useSolidBackground
  int   nbLights;
  float envRotation;
  int   frameCount;
  float envBlur;
  int   useSolidBackground;
  vec3  backgroundColor;
};

struct Ray
{
  vec3 origin;
  vec3 direction;
};

#ifndef EPSILON
#define EPSILON 0.0  // 1.19209e-07
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38
#endif

#endif  // HOST_DEVICE_H
