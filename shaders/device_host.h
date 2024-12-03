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
// clang-format on
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
#define USE_SKY_FLAG (1 << 0)
#define USE_HDR_FLAG (1 << 1)
#define USE_SOLID_BACKGROUND_FLAG (1 << 2)

// Macros to set and test the flags
#define SET_FLAG(flags, flag) ((flags) |= (flag))
#define CLEAR_FLAG(flags, flag) ((flags) &= ~(flag))
#define TEST_FLAG(flags, flag) bool((flags) & (flag))


struct PushConstantPathtracer
{
  int   maxDepth;            // Maximum ray depth
  int   maxSamples;          // Number of samples per frame
  float maxLuminance;        // Maximum luminance value used by the firefly clamping
  int   dbgMethod;           // Various debug informations
  int   selectedRenderNode;  // The node that is selected, used to create silhouette
  float focalDistance;       // Focal distance for depth of field
  float aperture;            // Aperture for depth of field
  vec2  mouseCoord;          // Debugging (printf) mouse coordinates
  int   useRTDenoiser;       // Use the RTX denoiser?
};

struct PushConstantRaster
{
  int materialID;          // Material used by the rendering instance
  int renderNodeID;        // Node used by the rendering instance
  int renderPrimID;        // Primitive used by the rendering instance
  int dbgMethod;           // Debugging method
  int selectedRenderNode;  // The node that is selected, used to create silhouette
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
  mat4  projMatrix;            // projection matrix
  mat4  projMatrixI;           // inverse projection matrix
  mat4  viewMatrix;            // view matrix (world to camera)
  mat4  viewMatrixI;           // inverse view matrix (camera to world)
  Light light[MAX_NB_LIGHTS];  // Light information
  vec4  envIntensity;          // Environment intensity
  vec3  camPos;                // Camera position
  int   flags;                 // Use flag bits instead of separate useSky and useSolidBackground
  int   nbLights;              // Number of lights
  float envRotation;           // Environment rotation (used for the HDR)
  int   frameCount;            // Current render frame [0, ... [
  float envBlur;               // Level of blur for the environment map (0.0: no blur, 1.0: full blur)
  int   useSolidBackground;    // Use solid background color (0==false, 1==true)
  vec3  backgroundColor;       // Background color when using solid background
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
