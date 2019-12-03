#version 450
#extension GL_ARB_separate_shader_objects : enable

// Scene UBO
layout(set = 0, binding = 0) uniform UBOscene
{
  mat4  projection;
  mat4  modelView;
  vec4  camPos;
  vec4  lightDir;
  float lightIntensity;
  float exposure;
}
uboScene;


// Input
layout(location = 0) in vec3 inPos;
//layout(location = 1) in vec3 inNormal;
//layout(location = 2) in vec3 inColor;
//layout(location = 3) in vec2 inUV;


// Output
layout(location = 0) out vec3 outWorldPos;
//layout(location = 1) out vec3 outNormal;
//layout(location = 2) out vec3 outColor;
//layout(location = 3) out vec2 outUV;

out gl_PerVertex
{
  vec4 gl_Position;
};


void main()
{
  vec4 pos      = vec4(inPos.xyz, 1.0);
  gl_Position   = uboScene.projection * pos;
  gl_Position.z = 0.5; // always draw
  mat4 m        = inverse(uboScene.modelView);
  m[3][0]       = 0.0;
  m[3][1]       = 0.0;
  m[3][2]       = 0.0;
  outWorldPos   = vec3(m * pos).xyz;
}
