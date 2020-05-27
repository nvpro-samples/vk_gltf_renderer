#version 450
#extension GL_ARB_separate_shader_objects : enable

// Scene UBO
layout(set = 0, binding = 0) uniform UBOscene
{
  mat4 projection;
  mat4 modelView;
  vec4 camPos;
  vec4 lightDir;
  float lightIntensity;
  float exposure;
}
uboScene;

// Object Dynamic storage buffer
layout(set = 1, binding = 0) readonly buffer UBOinstance
{
  mat4 matrix;
  mat4 matrixIT;
}
uboInstance;


// Input
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec2 inUV;


// Output
layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outColor;
layout(location = 3) out vec2 outUV;


out gl_PerVertex
{
  vec4 gl_Position;
};


void main()
{
  vec4 worldPos = uboInstance.matrix * vec4(inPos, 1.0);

  outUV    = inUV;
  outColor = inColor;
  outNormal = vec3(uboInstance.matrixIT * vec4(inNormal, 0.0));
  outWorldPos = worldPos.xyz;

  gl_Position   = uboScene.projection * uboScene.modelView * worldPos;
}
