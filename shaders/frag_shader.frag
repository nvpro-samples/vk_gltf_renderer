#version 450
#extension GL_ARB_separate_shader_objects : enable
//
//// PBR shader based on the Khronos WebGL PBR implementation
//// See https://github.com/KhronosGroup/glTF-WebGL-PBR
//// Supports both metallic roughness and specular glossiness inputs
//
//#define MANUAL_SRGB 1
//#define TONEMAP_UNCHARTED 1
//// #define DEBUG_OUTPUT 1
//// #define DEBUG_METALLIC 1
////#define MATERIAL_SPECULARGLOSSINESS 1
//#define MATERIAL_METALLICROUGHNESS 1
//#define HAS_METALLIC_ROUGHNESS_MAP 1
//#define HAS_OCCLUSION_MAP 1
//#define HAS_EMISSIVE_MAP 1
////#define MATERIAL_UNLIT 1
//
//const int LightType_Directional = 0;
//const int LightType_Point = 1;
//const int LightType_Spot = 2;
//
//
//layout(location = 0) in vec3 inWorldPos;
//layout(location = 1) in vec3 inNormal;
//layout(location = 2) in vec3 inColor;
//layout(location = 3) in vec2 inUV0;
//
//
//layout(push_constant) uniform shaderInformation
//{
//  int shadingModel;  // 0: metallic-roughness, 1: specular-glossiness
//
//  // PbrMetallicRoughness
//  vec4  pbrBaseColorFactor;
//  int   pbrBaseColorTexture;
//  float pbrMetallicFactor;
//  float pbrRoughnessFactor;
//  int   pbrMetallicRoughnessTexture;
//
//  // KHR_materials_pbrSpecularGlossiness
//  vec4  khrDiffuseFactor;
//  int   khrDiffuseTexture;
//  vec3  khrSpecularFactor;
//  float khrGlossinessFactor;
//  int   khrSpecularGlossinessTexture;
//
//  int   emissiveTexture;
//  vec3  emissiveFactor;
//  int   alphaMode;
//  float alphaCutoff;
//  bool  doubleSided;
//
//  int   normalTexture;
//  float normalTextureScale;
//  int   occlusionTexture;
//  float occlusionTextureStrength;
//}
//material;
//
//layout(set = 0, binding = 0) uniform UBOscene
//{
//  mat4  projection;
//  mat4  model;
//  vec4  camPos;
//  vec4  lightDir;
//  float lightRadiance;
//  float exposure;
//  float gamma;
//}
//ubo;
//
//layout(set = 2, binding = 0) uniform sampler2D texturesMap;
//
//
layout(location = 0) out vec4 outColor;
//
//
//const float M_PI           = 3.141592653589793;
//const float c_MinRoughness = 0.04;
//
//// Encapsulate the various inputs used by the various functions in the shading equation
//// We store values in this struct to simplify the integration of alternative implementations
//// of the shading terms, outlined in the Readme.MD Appendix.
//struct PBRInfo
//{
//  float NdotL;                // cos angle between normal and light direction
//  float NdotV;                // cos angle between normal and view direction
//  float NdotH;                // cos angle between normal and half vector
//  float LdotH;                // cos angle between light direction and half vector
//  float VdotH;                // cos angle between view direction and half vector
//  float perceptualRoughness;  // roughness value, as authored by the model creator (input to shader)
//  float metalness;            // metallic value at the surface
//  vec3  reflectance0;         // full reflectance color (normal incidence angle)
//  vec3  reflectance90;        // reflectance color at grazing angle
//  float alphaRoughness;       // roughness mapped to a more linear change in the roughness (proposed by [2])
//  vec3  diffuseColor;         // color contribution from diffuse lighting
//  vec3  specularColor;        // color contribution from specular lighting
//};
//
//struct AngularInfo
//{
//  float NdotL;  // cos angle between normal and light direction
//  float NdotV;  // cos angle between normal and view direction
//  float NdotH;  // cos angle between normal and half vector
//  float LdotH;  // cos angle between light direction and half vector
//  float VdotH;  // cos angle between view direction and half vector
//  vec3  padding;
//};
//
//struct MaterialInfo
//{
//  float perceptualRoughness;  // roughness value, as authored by the model creator (input to shader)
//  vec3  reflectance0;         // full reflectance color (normal incidence angle)
//
//  float alphaRoughness;  // roughness mapped to a more linear change in the roughness (proposed by [2])
//  vec3  diffuseColor;    // color contribution from diffuse lighting
//
//  vec3 reflectance90;  // reflectance color at grazing angle
//  vec3 specularColor;  // color contribution from specular lighting
//};
//
//
//struct Light
//{
//    vec3 direction;
//    float range;
//
//    vec3 color;
//    float intensity;
//
//    vec3 position;
//    float innerConeCos;
//
//    float outerConeCos;
//    int type;
//
//    vec2 padding;
//};
//
//AngularInfo getAngularInfo(vec3 pointToLight, vec3 normal, vec3 view)
//{
//  // Standard one-letter names
//  vec3 n = normalize(normal);        // Outward direction of surface point
//  vec3 v = normalize(view);          // Direction from surface point to view
//  vec3 l = normalize(pointToLight);  // Direction from surface point to light
//  vec3 h = normalize(l + v);         // Direction of the vector between l and v
//
//  float NdotL = clamp(dot(n, l), 0.01, 1.0);
//  float NdotV = clamp(dot(n, v), 0.0, 1.0);
//  float NdotH = clamp(dot(n, h), 0.0, 1.0);
//  float LdotH = clamp(dot(l, h), 0.0, 1.0);
//  float VdotH = clamp(dot(v, h), 0.0, 1.0);
//
//  return AngularInfo(NdotL, NdotV, NdotH, LdotH, VdotH, vec3(0, 0, 0));
//}
//
//
//// From http://filmicgames.com/archives/75
//vec3 Uncharted2Tonemap(vec3 x)
//{
//  float A = 0.15;
//  float B = 0.50;
//  float C = 0.10;
//  float D = 0.20;
//  float E = 0.02;
//  float F = 0.30;
//  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
//}
//
//vec4 SRGBtoLINEAR(vec4 srgbIn)
//{
//return vec4(pow(srgbIn.xyz, vec3(ubo.gamma)), srgbIn.w);
//}
//
//// Gamma Correction in Computer Graphics
//// see https://www.teamten.com/lawrence/graphics/gamma/
//vec3 gammaCorrection(vec3 color)
//{
//    return pow(color, vec3(1.0 / ubo.gamma));
//}
//
//
//// Uncharted 2 tone map
//// see: http://filmicworlds.com/blog/filmic-tonemapping-operators/
//vec3 toneMapUncharted2Impl(vec3 color)
//{
//    const float A = 0.15;
//    const float B = 0.50;
//    const float C = 0.10;
//    const float D = 0.20;
//    const float E = 0.02;
//    const float F = 0.30;
//    return ((color*(A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-E/F;
//}
//
//vec3 toneMapUncharted(vec3 color)
//{
//    const float W = 11.2;
//    color = toneMapUncharted2Impl(color * 2.0);
//    vec3 whiteScale = 1.0 / toneMapUncharted2Impl(vec3(W));
//    return gammaCorrection(color * whiteScale);
//}
//
//// Hejl Richard tone map
//// see: http://filmicworlds.com/blog/filmic-tonemapping-operators/
//vec3 toneMapHejlRichard(vec3 color)
//{
//    color = max(vec3(0.0), color - vec3(0.004));
//    return (color*(6.2*color+.5))/(color*(6.2*color+1.7)+0.06);
//}
//
//// ACES tone map
//// see: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
//vec3 toneMapACES(vec3 color)
//{
//    const float A = 2.51;
//    const float B = 0.03;
//    const float C = 2.43;
//    const float D = 0.59;
//    const float E = 0.14;
//    return gammaCorrection(clamp((color * (A * color + B)) / (color * (C * color + D) + E), 0.0, 1.0));
//}
//
//vec3 toneMap(vec3 color)
//{
//    color *= ubo.exposure;
//
//#ifdef TONEMAP_UNCHARTED
//    return toneMapUncharted(color);
//#endif
//
//#ifdef TONEMAP_HEJLRICHARD
//    return toneMapHejlRichard(color);
//#endif
//
//#ifdef TONEMAP_ACES
//    return toneMapACES(color);
//#endif
//
//    return gammaCorrection(color);
//}
//
//// Calculation of the lighting contribution from an optional Image Based Light source.
//// Precomputed Environment Maps are required uniform inputs and are computed as outlined in [1].
//// See our README.md on Environment Maps [3] for additional discussion.
//vec3 getIBLContribution(PBRInfo pbrInputs, vec3 n, vec3 reflection)
//{
//  //  float lod = (pbrInputs.perceptualRoughness * uboParams.prefilteredCubeMipLevels);
//  //  // retrieve a scale and bias to F0. See [1], Figure 3
//  //  vec3 brdf         = (texture(samplerBRDFLUT, vec2(pbrInputs.NdotV, 1.0 - pbrInputs.perceptualRoughness))).rgb;
//  //  vec3 diffuseLight = SRGBtoLINEAR(tonemap(texture(samplerIrradiance, n))).rgb;
//  //
//  //  vec3 specularLight = SRGBtoLINEAR(tonemap(textureLod(prefilteredMap, reflection, lod))).rgb;
//  //
//  //  vec3 diffuse  = diffuseLight * pbrInputs.diffuseColor;
//  //  vec3 specular = specularLight * (pbrInputs.specularColor * brdf.x + brdf.y);
//  //
//  //  // For presentation, this allows us to disable IBL terms
//  //  diffuse *= uboParams.scaleIBLAmbient;
//  //  specular *= uboParams.scaleIBLAmbient;
//  //
//  //  return diffuse + specular;
//  return vec3(1.f);
//}
//
//// See http://www.thetenthplanet.de/archives/1180
//vec3 getNormal()
//{
//  vec3 tangentNormal = texture(normalMap, inUV0).xyz;
//  if(length(tangentNormal) == 0.0)
//    return inNormal;
//  tangentNormal = tangentNormal * 2.0 - 1.0;
//  vec3 q1       = dFdx(inWorldPos);
//  vec3 q2       = dFdy(inWorldPos);
//  vec2 st1      = dFdx(inUV0);
//  vec2 st2      = dFdy(inUV0);
//
//  vec3 N   = normalize(inNormal);
//  vec3 T   = normalize(q1 * st2.t - q2 * st1.t);
//  vec3 B   = -normalize(cross(N, T));
//  mat3 TBN = mat3(T, B, N);
//
//  return normalize(TBN * tangentNormal);
//}
//
//struct UBOParams
//{
//  float exposure;
//  float gamma;
//} uboParams;
//
//
//// Lambert lighting
//// see https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
//vec3 diffuse(MaterialInfo materialInfo)
//{
//    return materialInfo.diffuseColor / M_PI;
//}
//
//// The following equation models the Fresnel reflectance term of the spec equation (aka F())
//// Implementation of fresnel from [4], Equation 15
//vec3 specularReflection(MaterialInfo materialInfo, AngularInfo angularInfo)
//{
//    return materialInfo.reflectance0 + (materialInfo.reflectance90 - materialInfo.reflectance0) * pow(clamp(1.0 - angularInfo.VdotH, 0.0, 1.0), 5.0);
//}
//
//
//// This calculates the specular geometric attenuation (aka G()),
//// where rougher material will reflect less light back to the viewer.
//// This implementation is based on [1] Equation 4, and we adopt their modifications to
//// alphaRoughness as input as originally proposed in [2].
//float geometricOcclusion(PBRInfo pbrInputs)
//{
//  float NdotL = pbrInputs.NdotL;
//  float NdotV = pbrInputs.NdotV;
//  float r     = pbrInputs.alphaRoughness;
//
//  float attenuationL = 2.0 * NdotL / (NdotL + sqrt(r * r + (1.0 - r * r) * (NdotL * NdotL)));
//  float attenuationV = 2.0 * NdotV / (NdotV + sqrt(r * r + (1.0 - r * r) * (NdotV * NdotV)));
//  return attenuationL * attenuationV;
//}
//
//// Smith Joint GGX
//// Note: Vis = G / (4 * NdotL * NdotV)
//// see Eric Heitz. 2014. Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs. Journal of Computer Graphics Techniques, 3
//// see Real-Time Rendering. Page 331 to 336.
//// see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
//float visibilityOcclusion(MaterialInfo materialInfo, AngularInfo angularInfo)
//{
//    float NdotL = angularInfo.NdotL;
//    float NdotV = angularInfo.NdotV;
//    float alphaRoughnessSq = materialInfo.alphaRoughness * materialInfo.alphaRoughness;
//
//    float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);
//    float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);
//
//    return 0.5 / (GGXV + GGXL);
//}
//
//
//// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
//// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
//// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
//float microfacetDistribution(MaterialInfo materialInfo, AngularInfo angularInfo)
//{
//    float alphaRoughnessSq = materialInfo.alphaRoughness * materialInfo.alphaRoughness;
//    float f = (angularInfo.NdotH * alphaRoughnessSq - angularInfo.NdotH) * angularInfo.NdotH + 1.0;
//    return alphaRoughnessSq / (M_PI * f * f);
//}
//
//vec3 getPointShade(vec3 pointToLight, MaterialInfo materialInfo, vec3 normal, vec3 view)
//{
//    AngularInfo angularInfo = getAngularInfo(pointToLight, normal, view);
//
//    // If one of the dot products is larger than zero, no division by zero can happen. Avoids black borders.
//    if (angularInfo.NdotL > 0.0 || angularInfo.NdotV > 0.0)
//    {
//        // Calculate the shading terms for the microfacet specular shading model
//        vec3 F = specularReflection(materialInfo, angularInfo);
//        float Vis = visibilityOcclusion(materialInfo, angularInfo);
//        float D = microfacetDistribution(materialInfo, angularInfo);
//
//        // Calculation of analytical lighting contribution
//        vec3 diffuseContrib = (1.0 - F) * diffuse(materialInfo);
//        vec3 specContrib = F * Vis * D;
//
//        // Obtain final intensity as reflectance (BRDF) scaled by the energy of the light (cosine law)
//        return angularInfo.NdotL * (diffuseContrib + specContrib);
//    }
//
//    return vec3(0.0, 0.0, 0.0);
//}
//
//vec3 applyDirectionalLight(Light light, MaterialInfo materialInfo, vec3 normal, vec3 view)
//{
//    vec3 pointToLight = -light.direction;
//    vec3 shade = getPointShade(pointToLight, materialInfo, normal, view);
//    return max(vec3(0.0),light.intensity * light.color * shade);
//}
//
//
//
//
void main()
{
outColor = vec4(1,0,0,0);
}
