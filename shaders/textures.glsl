
vec2 getNormalUV()
{
    vec3 uv = vec3(inUV0, 1.0);
#ifdef HAS_NORMAL_MAP
    uv.xy = u_NormalUVSet < 1 ? inUV0 : v_UVCoord2;
    #ifdef HAS_NORMAL_UV_TRANSFORM
    uv *= u_NormalUVTransform;
    #endif
#endif
    return uv.xy;
}

vec2 getEmissiveUV()
{
    vec3 uv = vec3(inUV0, 1.0);
#ifdef HAS_EMISSIVE_MAP
    uv.xy = u_EmissiveUVSet < 1 ? inUV0 : v_UVCoord2;
    #ifdef HAS_EMISSIVE_UV_TRANSFORM
    uv *= u_EmissiveUVTransform;
    #endif
#endif

    return uv.xy;
}

vec2 getOcclusionUV()
{
    vec3 uv = vec3(inUV0, 1.0);
#ifdef HAS_OCCLUSION_MAP
    uv.xy = u_OcclusionUVSet < 1 ? inUV0 : v_UVCoord2;
    #ifdef HAS_OCCLSION_UV_TRANSFORM
    uv *= u_OcclusionUVTransform;
    #endif
#endif
    return uv.xy;
}

vec2 getBaseColorUV()
{
    vec3 uv = vec3(inUV0, 1.0);
#ifdef HAS_BASE_COLOR_MAP
    uv.xy = u_BaseColorUVSet < 1 ? inUV0 : v_UVCoord2;
    #ifdef HAS_BASECOLOR_UV_TRANSFORM
    uv *= u_BaseColorUVTransform;
    #endif
#endif
    return uv.xy;
}

vec2 getMetallicRoughnessUV()
{
    vec3 uv = vec3(inUV0, 1.0);
#ifdef HAS_METALLIC_ROUGHNESS_MAP
    uv.xy = u_MetallicRoughnessUVSet < 1 ? inUV0 : v_UVCoord2;
    #ifdef HAS_METALLICROUGHNESS_UV_TRANSFORM
    uv *= u_MetallicRoughnessUVTransform;
    #endif
#endif
    return uv.xy;
}

vec2 getSpecularGlossinessUV()
{
    vec3 uv = vec3(inUV0, 1.0);
#ifdef HAS_SPECULAR_GLOSSINESS_MAP
    uv.xy = u_SpecularGlossinessUVSet < 1 ? inUV0 : v_UVCoord2;
    #ifdef HAS_SPECULARGLOSSINESS_UV_TRANSFORM
    uv *= u_SpecularGlossinessUVTransform;
    #endif
#endif
    return uv.xy;
}

vec2 getDiffuseUV()
{
    vec3 uv = vec3(inUV0, 1.0);
#ifdef HAS_DIFFUSE_MAP
    uv.xy = u_DiffuseUVSet < 1 ? inUV0 : v_UVCoord2;
    #ifdef HAS_DIFFUSE_UV_TRANSFORM
    uv *= u_DiffuseUVTransform;
    #endif
#endif
    return uv.xy;
}
