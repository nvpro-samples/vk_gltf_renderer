#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable


//
// This fragment shader defines a reference implementation for Physically Based Shading of
// a microfacet surface material defined by a glTF model.
// See https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/master/src/shaders/metallic-roughness.frag
//
// References:
// [1] Real Shading in Unreal Engine 4
//     http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
// [2] Physically Based Shading at Disney
//     http://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
// [3] README.md - Environment Maps
//     https://github.com/KhronosGroup/glTF-WebGL-PBR/#environment-maps
// [4] "An Inexpensive BRDF Model for Physically based Rendering" by Christophe Schlick
//     https://www.cs.virginia.edu/~jdl/bib/appearance/analytic%20models/schlick94b.pdf


precision highp float;

layout(set = 0, binding = 0) uniform UBOscene
{
  mat4  projection;
  mat4  model;
  vec4  camPos;
  vec4  lightDir;
  float lightRadiance;
  float exposure;
  float gamma;
  int   materialMode;
  int   tonemap;
  float envIntensity;
}
ubo;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec2 inUV0;

#define USE_HDR 1

#define NO_DEBUG_OUTPUT 0
#define DEBUG_METALLIC 1
#define DEBUG_NORMAL 2
#define DEBUG_BASECOLOR 3
#define DEBUG_OCCLUSION 4
#define DEBUG_EMISSIVE 5
#define DEBUG_F0 6
#define DEBUG_ALPHA 7
#define DEBUG_ROUGHNESS 8

#include "gltf.glsl"
layout(push_constant) uniform shaderInformation
{
  GltfShadeMaterial material;
};


layout(set = 2, binding = 0) uniform sampler2D texturesMap[];  // All textures

layout(set = 3, binding = 2) uniform samplerCube samplerIrradiance;
layout(set = 3, binding = 1) uniform sampler2D samplerBRDFLUT;
layout(set = 3, binding = 0) uniform samplerCube prefilteredMap;


layout(location = 0) out vec4 outColor;

// Dealing with All samples above
#include "textures.glsl"
// General functions
#include "functions.glsl"
// Tonemapping functions
#include "tonemapping.glsl"

// KHR_lights_punctual extension.
// see https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_lights_punctual
struct Light
{
  vec3  direction;
  float range;
  vec3  color;
  float intensity;
  vec3  position;
  float innerConeCos;
  float outerConeCos;
  int   type;
  vec2  padding;
};

const int LightType_Directional = 0;
const int LightType_Point       = 1;
const int LightType_Spot        = 2;

#ifdef USE_PUNCTUAL
uniform Light u_Lights[LIGHT_COUNT];
#endif


struct MaterialInfo
{
  float perceptualRoughness;  // roughness value, as authored by the model creator (input to shader)
  vec3  reflectance0;         // full reflectance color (normal incidence angle)
  float alphaRoughness;       // roughness mapped to a more linear change in the roughness (proposed by [2])
  vec3  diffuseColor;         // color contribution from diffuse lighting
  vec3  reflectance90;        // reflectance color at grazing angle
  vec3  specularColor;        // color contribution from specular lighting
};

// Calculation of the lighting contribution from an Image Based Light source.
vec3 getIBLContribution(MaterialInfo materialInfo, vec3 n, vec3 v)
{
  float NdotV = clamp(dot(n, v), 0.0, 1.0);

  float lod        = clamp(materialInfo.perceptualRoughness * float(10.0), 0.0, float(10.0));
  vec3  reflection = normalize(reflect(-v, n));

  vec2 brdfSamplePoint = clamp(vec2(NdotV, materialInfo.perceptualRoughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
  // retrieve a scale and bias to F0. See [1], Figure 3
  vec2 brdf = texture(samplerBRDFLUT, brdfSamplePoint).rg;

  vec4 diffuseSample = texture(samplerIrradiance, n) * ubo.envIntensity;

#ifdef USE_TEX_LOD
  vec4 specularSample = textureCubeLodEXT(u_SpecularEnvSampler, reflection, lod);
#else
  vec4 specularSample = textureLod(prefilteredMap, reflection, lod);
#endif

#ifdef USE_HDR
  // Already linear.
  vec3 diffuseLight  = diffuseSample.rgb;
  vec3 specularLight = specularSample.rgb;
#else
  vec3 diffuseLight   = SRGBtoLINEAR(diffuseSample).rgb;
  vec3 specularLight  = SRGBtoLINEAR(specularSample).rgb;
#endif

  vec3 diffuse  = diffuseLight * materialInfo.diffuseColor;
  vec3 specular = specularLight * (materialInfo.specularColor * brdf.x + brdf.y);

  return diffuse + specular;
}
//#endif

// Lambert lighting
// see https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
vec3 diffuse(MaterialInfo materialInfo)
{
  return materialInfo.diffuseColor / M_PI;
}

// The following equation models the Fresnel reflectance term of the spec equation (aka F())
// Implementation of fresnel from [4], Equation 15
vec3 specularReflection(MaterialInfo materialInfo, AngularInfo angularInfo)
{
  return materialInfo.reflectance0
         + (materialInfo.reflectance90 - materialInfo.reflectance0) * pow(clamp(1.0 - angularInfo.VdotH, 0.0, 1.0), 5.0);
}

// Smith Joint GGX
// Note: Vis = G / (4 * NdotL * NdotV)
// see Eric Heitz. 2014. Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs. Journal of Computer Graphics Techniques, 3
// see Real-Time Rendering. Page 331 to 336.
// see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
float visibilityOcclusion(MaterialInfo materialInfo, AngularInfo angularInfo)
{
  float NdotL            = angularInfo.NdotL;
  float NdotV            = angularInfo.NdotV;
  float alphaRoughnessSq = materialInfo.alphaRoughness * materialInfo.alphaRoughness;

  float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);
  float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);

  float GGX = GGXV + GGXL;
  if(GGX > 0.0)
  {
    return 0.5 / GGX;
  }
  return 0.0;
}

// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
float microfacetDistribution(MaterialInfo materialInfo, AngularInfo angularInfo)
{
  float alphaRoughnessSq = materialInfo.alphaRoughness * materialInfo.alphaRoughness;
  float f                = (angularInfo.NdotH * alphaRoughnessSq - angularInfo.NdotH) * angularInfo.NdotH + 1.0;
  return alphaRoughnessSq / (M_PI * f * f);
}

vec3 getPointShade(vec3 pointToLight, MaterialInfo materialInfo, vec3 normal, vec3 view)
{
  AngularInfo angularInfo = getAngularInfo(pointToLight, normal, view);

  // If one of the dot products is larger than zero, no division by zero can happen. Avoids black borders.
  if(angularInfo.NdotL > 0.0 || angularInfo.NdotV > 0.0)
  {
    // Calculate the shading terms for the microfacet specular shading model
    vec3  F   = specularReflection(materialInfo, angularInfo);
    float Vis = visibilityOcclusion(materialInfo, angularInfo);
    float D   = microfacetDistribution(materialInfo, angularInfo);

    // Calculation of analytical lighting contribution
    vec3 diffuseContrib = (1.0 - F) * diffuse(materialInfo);
    vec3 specContrib    = F * Vis * D;

    // Obtain final intensity as reflectance (BRDF) scaled by the energy of the light (cosine law)
    return angularInfo.NdotL * (diffuseContrib + specContrib);
  }

  return vec3(0.0, 0.0, 0.0);
}

// https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md#range-property
float getRangeAttenuation(float range, float distance)
{
  if(range < 0.0)
  {
    // negative range means unlimited
    return 1.0;
  }
  return max(min(1.0 - pow(distance / range, 4.0), 1.0), 0.0) / pow(distance, 2.0);
}

// https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md#inner-and-outer-cone-angles
float getSpotAttenuation(vec3 pointToLight, vec3 spotDirection, float outerConeCos, float innerConeCos)
{
  float actualCos = dot(normalize(spotDirection), normalize(-pointToLight));
  if(actualCos > outerConeCos)
  {
    if(actualCos < innerConeCos)
    {
      return smoothstep(outerConeCos, innerConeCos, actualCos);
    }
    return 1.0;
  }
  return 0.0;
}

vec3 applyDirectionalLight(Light light, MaterialInfo materialInfo, vec3 normal, vec3 view)
{
  vec3 pointToLight = -light.direction;
  vec3 shade        = getPointShade(pointToLight, materialInfo, normal, view);
  return light.intensity * light.color * shade;
}

vec3 applyPointLight(Light light, MaterialInfo materialInfo, vec3 normal, vec3 view)
{
  vec3  pointToLight = light.position - inWorldPos;
  float distance     = length(pointToLight);
  float attenuation  = getRangeAttenuation(light.range, distance);
  vec3  shade        = getPointShade(pointToLight, materialInfo, normal, view);
  return attenuation * light.intensity * light.color * shade;
}

vec3 applySpotLight(Light light, MaterialInfo materialInfo, vec3 normal, vec3 view)
{
  vec3  pointToLight     = light.position - inWorldPos;
  float distance         = length(pointToLight);
  float rangeAttenuation = getRangeAttenuation(light.range, distance);
  float spotAttenuation  = getSpotAttenuation(pointToLight, light.direction, light.outerConeCos, light.innerConeCos);
  vec3  shade            = getPointShade(pointToLight, materialInfo, normal, view);
  return rangeAttenuation * spotAttenuation * light.intensity * light.color * shade;
}


void main()
{
  // Metallic and Roughness material properties are packed together
  // In glTF, these factors can be specified by fixed scalar values
  // or from a metallic-roughness map
  float perceptualRoughness = material.pbrRoughnessFactor;
  float metallic            = material.pbrMetallicFactor;
  vec4  baseColor           = vec4(0.0, 0.0, 0.0, 1.0);
  vec3  diffuseColor        = vec3(0.0);
  vec3  specularColor       = vec3(0.0);
  vec3  f0                  = vec3(0.04);

  // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
  // This layout intentionally reserves the 'r' channel for (optional) occlusion map data
  if(material.pbrMetallicRoughnessTexture > -1)
  {
    vec4 mrSample = texture(texturesMap[nonuniformEXT(material.pbrMetallicRoughnessTexture)], inUV0);
    perceptualRoughness *= mrSample.g;
    metallic *= mrSample.b;
  }

  // The albedo may be defined from a base texture or a flat color
  baseColor = material.pbrBaseColorFactor;
  if(material.pbrBaseColorTexture > -1)
    baseColor *= SRGBtoLINEAR(texture(texturesMap[nonuniformEXT(material.pbrBaseColorTexture)], inUV0), 2.2);
  baseColor *= getVertexColor();

  diffuseColor = baseColor.rgb * (vec3(1.0) - f0) * (1.0 - metallic);

  specularColor = mix(f0, baseColor.rgb, metallic);

  if(material.alphaMode > 0 && baseColor.a < material.alphaCutoff)
  {
    discard;
  }
  baseColor.a = 1.0;


#ifdef MATERIAL_UNLIT
  outColor = vec4(gammaCorrection(baseColor.rgb), baseColor.a);
  return;
#endif


  perceptualRoughness = clamp(perceptualRoughness, 0.0, 1.0);
  metallic            = clamp(metallic, 0.0, 1.0);

  // Roughness is authored as perceptual roughness; as is convention,
  // convert to material roughness by squaring the perceptual roughness [2].
  float alphaRoughness = perceptualRoughness * perceptualRoughness;

  // Compute reflectance.
  float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);

  vec3 specularEnvironmentR0 = specularColor.rgb;
  // Anything less than 2% is physically impossible and is instead considered to be shadowing. Compare to "Real-Time-Rendering" 4th editon on page 325.
  vec3 specularEnvironmentR90 = vec3(clamp(reflectance * 50.0, 0.0, 1.0));

  MaterialInfo materialInfo = MaterialInfo(perceptualRoughness, specularEnvironmentR0, alphaRoughness, diffuseColor,
                                           specularEnvironmentR90, specularColor);

  // LIGHTING

  vec3 color  = vec3(0.0, 0.0, 0.0);
  vec3 normal = getNormal(material.normalTexture);
  vec3 view   = normalize(ubo.camPos.xyz - inWorldPos);

#ifdef USE_PUNCTUAL
  for(int i = 0; i < LIGHT_COUNT; ++i)
  {
    Light light = u_Lights[i];
    if(light.type == LightType_Directional)
    {
      color += applyDirectionalLight(light, materialInfo, normal, view);
    }
    else if(light.type == LightType_Point)
    {
      color += applyPointLight(light, materialInfo, normal, view);
    }
    else if(light.type == LightType_Spot)
    {
      color += applySpotLight(light, materialInfo, normal, view);
    }
  }
#else
  // Using the default directional light
  Light light;
  light.type      = LightType_Directional;
  light.direction = ubo.lightDir.xyz;
  light.intensity = ubo.lightRadiance;
  light.color     = vec3(1, 1, 1);
  color += applyDirectionalLight(light, materialInfo, normal, view);
#endif

  // Calculate lighting contribution from image based lighting source (IBL)
  color += getIBLContribution(materialInfo, normal, view);

  // Ambient occulsion
  float ao = 1.0;
  if(material.occlusionTexture > -1)
    ao = texture(texturesMap[nonuniformEXT(material.occlusionTexture)], inUV0).r;
  color = mix(color, color * ao, 1.0 /*u_OcclusionStrength*/);

  // Emissive term
  vec3 emissive = material.emissiveFactor;
  if(material.emissiveTexture > -1)
    emissive = SRGBtoLINEAR(texture(texturesMap[nonuniformEXT(material.emissiveTexture)], inUV0), 2.2).rgb * material.emissiveFactor;
  color += emissive;


  switch(ubo.materialMode)
  {
    case NO_DEBUG_OUTPUT:
      outColor = vec4(toneMap(color, ubo.tonemap, ubo.gamma, ubo.exposure), baseColor.a);
      break;
    case DEBUG_METALLIC:
      outColor.rgb = vec3(metallic);
      break;
    case DEBUG_ROUGHNESS:
      outColor.rgb = vec3(perceptualRoughness);
      break;

    case DEBUG_NORMAL:
      outColor.rgb = getNormal(material.normalTexture);
      break;

    case DEBUG_BASECOLOR:
      outColor.rgb = gammaCorrection(baseColor.rgb, ubo.gamma);
      break;

    case DEBUG_OCCLUSION:
      if(material.occlusionTexture > -1)
        outColor.rgb = texture(texturesMap[nonuniformEXT(material.occlusionTexture)], inUV0).rrr;
      else
        outColor.rgb = vec3(1);
      break;

    case DEBUG_EMISSIVE:
      if(material.emissiveTexture > -1)
        outColor.rgb = texture(texturesMap[nonuniformEXT(material.emissiveTexture)], inUV0).rrr;
      else
        outColor.rgb = vec3(0);
      break;

    case DEBUG_F0:
      outColor.rgb = vec3(f0);
      break;

    case DEBUG_ALPHA:
      outColor.rgb = vec3(baseColor.a);
      break;
    default:
      outColor = vec4(toneMap(color, ubo.tonemap, ubo.gamma, ubo.exposure), baseColor.a);
  }

  outColor.a = 1.0;
}
