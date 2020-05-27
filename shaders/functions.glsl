// textures.glsl needs to be included

const float M_PI             = 3.141592653589793;
const float c_MinReflectance = 0.04;

#define HAS_NORMALS 1
//#define HAS_VERTEX_COLOR_VEC3 1
#define HAS_VERTEX_COLOR_VEC4 1
#define HAS_NORMAL_MAP 1

#ifdef HAS_NORMALS
#ifdef HAS_TANGENTS
varying mat3 v_TBN;
#endif
#endif

#ifdef HAS_VERTEX_COLOR_VEC4
//varying vec4 inColor;
#endif

#ifdef HAS_VERTEX_COLOR_VEC3
//vec3 inColor;
#endif

struct AngularInfo
{
  float NdotL;  // cos angle between normal and light direction
  float NdotV;  // cos angle between normal and view direction
  float NdotH;  // cos angle between normal and half vector
  float LdotH;  // cos angle between light direction and half vector

  float VdotH;  // cos angle between view direction and half vector

  vec3 padding;
};

vec4 getVertexColor()
{
  vec4 color = vec4(1.0, 1.0, 1.0, 1.0);

#ifdef HAS_VERTEX_COLOR_VEC3
  color.rgb = inColor;
#endif
#ifdef HAS_VERTEX_COLOR_VEC4
  color = inColor;
#endif

  return color;
}

// Find the normal for this fragment, pulling either from a predefined normal map
// or from the interpolated mesh normal and tangent attributes.
// See http://www.thetenthplanet.de/archives/1180
vec3 getNormal(int normalTexture)
{
  if(normalTexture > -1)
  {
    vec3 tangentNormal = texture(texturesMap[normalTexture], inUV0).xyz;
    if(length(tangentNormal) <= 0.01)
      return inNormal;
    tangentNormal = tangentNormal * 2.0 - 1.0;
    vec3 q1       = dFdx(inWorldPos);
    vec3 q2       = dFdy(inWorldPos);
    vec2 st1      = dFdx(inUV0);
    vec2 st2      = dFdy(inUV0);

    vec3 N   = normalize(inNormal);
    vec3 T   = normalize(q1 * st2.t - q2 * st1.t);
    vec3 B   = -normalize(cross(N, T));
    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
  }
  else
    return normalize(inNormal);
}


float getPerceivedBrightness(vec3 vector)
{
  return sqrt(0.299 * vector.r * vector.r + 0.587 * vector.g * vector.g + 0.114 * vector.b * vector.b);
}

// https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness/examples/convert-between-workflows/js/three.pbrUtilities.js#L34
float solveMetallic(vec3 diffuse, vec3 specular, float oneMinusSpecularStrength)
{
  float specularBrightness = getPerceivedBrightness(specular);

  if(specularBrightness < c_MinReflectance)
  {
    return 0.0;
  }

  float diffuseBrightness = getPerceivedBrightness(diffuse);

  float a = c_MinReflectance;
  float b = diffuseBrightness * oneMinusSpecularStrength / (1.0 - c_MinReflectance) + specularBrightness - 2.0 * c_MinReflectance;
  float c = c_MinReflectance - specularBrightness;
  float D = b * b - 4.0 * a * c;

  return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
}

AngularInfo getAngularInfo(vec3 pointToLight, vec3 normal, vec3 view)
{
  // Standard one-letter names
  vec3 n = normalize(normal);        // Outward direction of surface point
  vec3 v = normalize(view);          // Direction from surface point to view
  vec3 l = normalize(pointToLight);  // Direction from surface point to light
  vec3 h = normalize(l + v);         // Direction of the vector between l and v

  float NdotL = clamp(dot(n, l), 0.05, 1.0);
  float NdotV = clamp(dot(n, v), 0.0, 1.0);
  float NdotH = clamp(dot(n, h), 0.0, 1.0);
  float LdotH = clamp(dot(l, h), 0.0, 1.0);
  float VdotH = clamp(dot(v, h), 0.0, 1.0);

  return AngularInfo(NdotL, NdotV, NdotH, LdotH, VdotH, vec3(0, 0, 0));
}
