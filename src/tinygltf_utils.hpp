/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*-------------------------------------------------------------------------------------------------
# namespace `tinygltf::utils`
> Utility functions for extracting structs from tinygltf's representation of glTF.
-------------------------------------------------------------------------------------------------*/

#include <algorithm>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include <tinygltf/tiny_gltf.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtc/type_ptr.hpp>

#define KHR_MATERIALS_VARIANTS_EXTENSION_NAME "KHR_materials_variants"
#define EXT_MESH_GPU_INSTANCING_EXTENSION_NAME "EXT_mesh_gpu_instancing"
#define EXT_TEXTURE_WEBP_EXTENSION_NAME "EXT_texture_webp"
#define EXTENSION_ATTRIB_IRAY "NV_attributes_iray"
#define MSFT_TEXTURE_DDS_NAME "MSFT_texture_dds"
#define KHR_LIGHTS_PUNCTUAL_EXTENSION_NAME "KHR_lights_punctual"
#define KHR_ANIMATION_POINTER "KHR_animation_pointer"

// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_specular/README.md
#define KHR_MATERIALS_SPECULAR_EXTENSION_NAME "KHR_materials_specular"
struct KHR_materials_specular
{
  float                 specularFactor       = 1.0f;
  tinygltf::TextureInfo specularTexture      = {};
  glm::vec3             specularColorFactor  = {1.0f, 1.0f, 1.0f};
  tinygltf::TextureInfo specularColorTexture = {};
};

// https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_texture_transform
#define KHR_TEXTURE_TRANSFORM_EXTENSION_NAME "KHR_texture_transform"
struct KHR_texture_transform
{
  glm::vec2 offset      = {0.0f, 0.0f};
  float     rotation    = 0.0f;
  glm::vec2 scale       = {1.0f, 1.0f};
  int       texCoord    = 0;
  glm::mat3 uvTransform = glm::mat3(1);  // Computed transform of offset, rotation, scale
  void      updateTransform()
  {
    // Compute combined transformation matrix
    float cosR  = cos(rotation);
    float sinR  = sin(rotation);
    float tx    = offset.x;
    float ty    = offset.y;
    float sx    = scale.x;
    float sy    = scale.y;
    uvTransform = glm::mat3(sx * cosR, sx * sinR, tx, -sy * sinR, sy * cosR, ty, 0, 0, 1);
  }
};

// https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_clearcoat/README.md
#define KHR_MATERIALS_CLEARCOAT_EXTENSION_NAME "KHR_materials_clearcoat"
struct KHR_materials_clearcoat
{
  float                 factor           = 0.0f;
  tinygltf::TextureInfo texture          = {};
  float                 roughnessFactor  = 0.0f;
  tinygltf::TextureInfo roughnessTexture = {};
  tinygltf::TextureInfo normalTexture    = {};
};

// https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_sheen/README.md
#define KHR_MATERIALS_SHEEN_EXTENSION_NAME "KHR_materials_sheen"
struct KHR_materials_sheen
{
  glm::vec3             sheenColorFactor      = {0.0f, 0.0f, 0.0f};
  tinygltf::TextureInfo sheenColorTexture     = {};
  float                 sheenRoughnessFactor  = 0.0f;
  tinygltf::TextureInfo sheenRoughnessTexture = {};
};

// https://github.com/DassaultSystemes-Technology/glTF/tree/KHR_materials_volume/extensions/2.0/Khronos/KHR_materials_transmission
#define KHR_MATERIALS_TRANSMISSION_EXTENSION_NAME "KHR_materials_transmission"
struct KHR_materials_transmission
{
  float                 factor  = 0.0f;
  tinygltf::TextureInfo texture = {};
};

// https://github.com/KhronosGroup/glTF/tree/master/extensions/2.0/Khronos/KHR_materials_unlit
#define KHR_MATERIALS_UNLIT_EXTENSION_NAME "KHR_materials_unlit"
struct KHR_materials_unlit
{
  int active = 0;
};

// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_anisotropy/README.md
#define KHR_MATERIALS_ANISOTROPY_EXTENSION_NAME "KHR_materials_anisotropy"
struct KHR_materials_anisotropy
{
  float                 anisotropyStrength = 0.0f;
  float                 anisotropyRotation = 0.0f;
  tinygltf::TextureInfo anisotropyTexture  = {};
};


// https://github.com/DassaultSystemes-Technology/glTF/tree/KHR_materials_ior/extensions/2.0/Khronos/KHR_materials_ior
#define KHR_MATERIALS_IOR_EXTENSION_NAME "KHR_materials_ior"
struct KHR_materials_ior
{
  float ior = 1.5f;
};

// https://github.com/DassaultSystemes-Technology/glTF/tree/KHR_materials_volume/extensions/2.0/Khronos/KHR_materials_volume
#define KHR_MATERIALS_VOLUME_EXTENSION_NAME "KHR_materials_volume"
struct KHR_materials_volume
{
  float                 thicknessFactor     = 0;
  tinygltf::TextureInfo thicknessTexture    = {};
  float                 attenuationDistance = std::numeric_limits<float>::max();
  glm::vec3             attenuationColor    = {1.0f, 1.0f, 1.0f};
};

// https://github.com/KhronosGroup/glTF/blob/e17468db6fd9ae3ce73504a9f317bd853af01a30/extensions/2.0/Khronos/KHR_materials_volume_scatter/README.md
#define KHR_MATERIALS_VOLUME_SCATTER_EXTENSION_NAME "KHR_materials_volume_scatter"
struct KHR_materials_volume_scatter
{
  glm::vec3 multiscatterColor = {0.0f, 0.0f, 0.0f};
  float     scatterAnisotropy = 0.0f;
};


// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_texture_basisu/README.md
#define KHR_TEXTURE_BASISU_EXTENSION_NAME "KHR_texture_basisu"
struct KHR_texture_basisu
{
  tinygltf::TextureInfo source;
};

// https://github.com/KhronosGroup/glTF/issues/948
#define KHR_MATERIALS_DISPLACEMENT_EXTENSION_NAME "KHR_materials_displacement"
struct KHR_materials_displacement
{
  float                 displacementGeometryFactor  = 1.0f;
  float                 displacementGeometryOffset  = 0.0f;
  tinygltf::TextureInfo displacementGeometryTexture = {};
};


// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_emissive_strength/README.md
#define KHR_MATERIALS_EMISSIVE_STRENGTH_EXTENSION_NAME "KHR_materials_emissive_strength"
struct KHR_materials_emissive_strength
{
  float emissiveStrength = 1.0;
};

// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_iridescence/README.md
#define KHR_MATERIALS_IRIDESCENCE_EXTENSION_NAME "KHR_materials_iridescence"
struct KHR_materials_iridescence
{
  float                 iridescenceFactor           = 0.0f;
  tinygltf::TextureInfo iridescenceTexture          = {};
  float                 iridescenceIor              = 1.3f;
  float                 iridescenceThicknessMinimum = 100.f;
  float                 iridescenceThicknessMaximum = 400.f;
  tinygltf::TextureInfo iridescenceThicknessTexture = {};
};

// https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_dispersion
#define KHR_MATERIALS_DISPERSION_EXTENSION_NAME "KHR_materials_dispersion"
struct KHR_materials_dispersion
{
  float dispersion = 0.0f;
};

// https://github.com/KhronosGroup/glTF/pull/2410
#define KHR_NODE_VISIBILITY_EXTENSION_NAME "KHR_node_visibility"
struct KHR_node_visibility
{
  bool visible = true;
};

#define KHR_MATERIALS_PBR_SPECULAR_GLOSSINESS_EXTENSION_NAME "KHR_materials_pbrSpecularGlossiness"
struct KHR_materials_pbrSpecularGlossiness
{
  glm::vec4             diffuseFactor             = glm::vec4(1.0f);
  glm::vec3             specularFactor            = glm::vec3(1.0f);
  float                 glossinessFactor          = 1.0f;
  tinygltf::TextureInfo diffuseTexture            = {};
  tinygltf::TextureInfo specularGlossinessTexture = {};
};


// https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_diffuse_transmission
#define KHR_MATERIALS_DIFFUSE_TRANSMISSION_EXTENSION_NAME "KHR_materials_diffuse_transmission"
struct KHR_materials_diffuse_transmission
{
  float                 diffuseTransmissionFactor       = 0.0f;
  tinygltf::TextureInfo diffuseTransmissionTexture      = {};
  glm::vec3             diffuseTransmissionColor        = {1.0f, 1.0f, 1.0f};
  tinygltf::TextureInfo diffuseTransmissionColorTexture = {};
};

// https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Vendor/EXT_meshopt_compression
#define EXT_MESHOPT_COMPRESSION_EXTENSION_NAME "EXT_meshopt_compression"
struct EXT_meshopt_compression
{
  enum EXT_meshopt_compression_mode
  {
    MESHOPT_COMPRESSION_MODE_INVALID,
    MESHOPT_COMPRESSION_MODE_ATTRIBUTES,
    MESHOPT_COMPRESSION_MODE_TRIANGLES,
    MESHOPT_COMPRESSION_MODE_INDICES,
  };
  enum EXT_meshopt_compression_filter
  {
    MESHOPT_COMPRESSION_FILTER_NONE,
    MESHOPT_COMPRESSION_FILTER_OCTAHEDRAL,
    MESHOPT_COMPRESSION_FILTER_QUATERNION,
    MESHOPT_COMPRESSION_FILTER_EXPONENTIAL,
  };
  int                            buffer{-1};
  size_t                         byteOffset{0};
  size_t                         byteLength{0};
  size_t                         byteStride{0};
  size_t                         count{0};
  EXT_meshopt_compression_mode   compressionMode   = MESHOPT_COMPRESSION_MODE_INVALID;
  EXT_meshopt_compression_filter compressionFilter = MESHOPT_COMPRESSION_FILTER_NONE;
};

namespace tinygltf {

namespace utils {

/*-------------------------------------------------------------------------------------------------
## Function `getValue<T>`
> Gets the value of type T for the attribute `name`.

This function retrieves the value of the specified attribute from a tinygltf::Value
and stores it in the provided result variable.

Parameters:
- value: The `tinygltf::Value` from which to retrieve the attribute.
- name: The name of the attribute to retrieve.
- result: The variable to store the retrieved value in.
-------------------------------------------------------------------------------------------------*/
template <typename T>
inline void getValue(const tinygltf::Value& value, const std::string& name, T& result)
{
  if(value.Has(name))
  {
    result = value.Get(name).Get<T>();
  }
}

/*-------------------------------------------------------------------------------------------------
## Function `getValue(..., float& result)`
> Specialization of `getValue()` for float type.

Retrieves the value of the specified attribute as a float and stores it in the result variable.
-------------------------------------------------------------------------------------------------*/
template <>
inline void getValue(const tinygltf::Value& value, const std::string& name, float& result)
{
  if(value.Has(name))
  {
    result = static_cast<float>(value.Get(name).Get<double>());
  }
}

/*-------------------------------------------------------------------------------------------------
## Function `getValue(..., tinygltf::TextureInfo& result)`
> Specialization of `getValue()` for `gltf::Texture` type.

Retrieves the texture attribute values and stores them in the result variable.
-------------------------------------------------------------------------------------------------*/
template <>
inline void getValue(const tinygltf::Value& value, const std::string& name, tinygltf::TextureInfo& result)
{
  if(value.Has(name))
  {
    const auto& t = value.Get(name);
    getValue(t, "index", result.index);
    getValue(t, "texCoord", result.texCoord);
    getValue(t, "extensions", result.extensions);
  }
}


/*-------------------------------------------------------------------------------------------------
## Function `setValue<T>`
> Sets attribute `key` to value `val`.
-------------------------------------------------------------------------------------------------*/
template <typename T>
inline void setValue(tinygltf::Value& value, const std::string& key, const T& val)
{
  value.Get<tinygltf::Value::Object>()[key] = tinygltf::Value(val);
}


/*-------------------------------------------------------------------------------------------------
## Function `setValue(... tinygltf::TextureInfo)`
> Sets attribute `key` to a JSON object with an `index` and `texCoord` set from
> the members of `textureInfo`.
-------------------------------------------------------------------------------------------------*/
inline void setValue(tinygltf::Value& value, const std::string& key, const tinygltf::TextureInfo& textureInfo)
{
  auto& t                                      = value.Get<tinygltf::Value::Object>()[key];
  t.Get<tinygltf::Value::Object>()["index"]    = tinygltf::Value(textureInfo.index);
  t.Get<tinygltf::Value::Object>()["texCoord"] = tinygltf::Value(textureInfo.texCoord);
  value.Get<tinygltf::Value::Object>()[key]    = t;
}


/*-------------------------------------------------------------------------------------------------
## Function `getArrayValue<T>`
> Gets the value of type T for the attribute `name`.

This function retrieves the array value of the specified attribute from a `tinygltf::Value`
and stores it in the provided result variable. It is used for types such as `glm::vec3`, `glm::vec4`, `glm::mat4`, etc.

Parameters:
- value: The `tinygltf::Value` from which to retrieve the attribute.
- name: The name of the attribute to retrieve.
- result: The variable to store the retrieved array value in.
-------------------------------------------------------------------------------------------------*/
template <class T>
inline void getArrayValue(const tinygltf::Value& value, const std::string& name, T& result)
{
  if(value.Has(name))
  {
    const auto& v = value.Get(name).Get<tinygltf::Value::Array>();
    std::transform(v.begin(), v.end(), glm::value_ptr(result),
                   [](const tinygltf::Value& v) { return static_cast<float>(v.Get<double>()); });
  }
}

/*-------------------------------------------------------------------------------------------------
## Function `setArrayValue<T>`
> Sets attribute `name` of the given `value` to an array with the first
> `numElements` elements from the `array` pointer.
-------------------------------------------------------------------------------------------------*/
template <typename T>
inline void setArrayValue(tinygltf::Value& value, const std::string& name, uint32_t numElem, T* array)
{
  tinygltf::Value::Array arrayValue(numElem);
  for(uint32_t n = 0; n < numElem; n++)
  {
    arrayValue[n] = tinygltf::Value(array[n]);
  }
  value.Get<tinygltf::Value::Object>()[name] = tinygltf::Value(arrayValue);
}


/*-------------------------------------------------------------------------------------------------
## Function `convertToTinygltfValue`
> Converts a vector of elements to a `tinygltf::Value`.

This function converts a given array of float elements into a `tinygltf::Value::Array`,
suitable for use within the tinygltf library.

Parameters:
- numElements: The number of elements in the array.
- elements: A pointer to the array of float elements.

Returns:
- A `tinygltf::Value` representing the array of elements.
-------------------------------------------------------------------------------------------------*/
tinygltf::Value convertToTinygltfValue(int numElements, const float* elements);


/*-------------------------------------------------------------------------------------------------
## Function `getNodeTRS`
> Retrieves the translation, rotation, and scale of a GLTF node.

This function extracts the translation, rotation, and scale (TRS) properties
from the given GLTF node. If the node has a matrix defined, it decomposes
the matrix to obtain these properties. Otherwise, it directly retrieves
the TRS values from the node's properties.

Parameters:
- node: The GLTF node from which to extract the TRS properties.
- translation: Output parameter for the translation vector.
- rotation: Output parameter for the rotation quaternion.
- scale: Output parameter for the scale vector.
-------------------------------------------------------------------------------------------------*/
void getNodeTRS(const tinygltf::Node& node, glm::vec3& translation, glm::quat& rotation, glm::vec3& scale);


/*-------------------------------------------------------------------------------------------------
* ## Function `setNodeTRS`
> Sets the translation, rotation, and scale of a GLTF node.

This function sets the translation, rotation, and scale (TRS) properties of
the given GLTF node using the provided values.

Parameters:
- node: The GLTF node to modify.
- translation: The translation vector to set.
- rotation: The rotation quaternion to set.
- scale: The scale vector to set.
-------------------------------------------------------------------------------------------------*/
void setNodeTRS(tinygltf::Node& node, const glm::vec3& translation, const glm::quat& rotation, const glm::vec3& scale);


/*-------------------------------------------------------------------------------------------------
* ## Function `getNodeMatrix`
> Retrieves the transformation matrix of a GLTF node.

This function computes the transformation matrix for the given GLTF node.
If the node has a direct matrix defined, it returns that matrix as defined in
the specification. Otherwise, it computes the matrix from the node's translation,
rotation, and scale (TRS) properties.

Parameters:
- node: The GLTF node for which to retrieve the transformation matrix.

Returns:
- The transformation matrix of the node.
-------------------------------------------------------------------------------------------------*/
glm::mat4 getNodeMatrix(const tinygltf::Node& node);


/*-------------------------------------------------------------------------------------------------
## Function `generatePrimitiveKey`
> Generates a unique key for a GLTF primitive based on its attributes.

This function creates a unique string key for the given GLTF primitive by
concatenating its attribute keys and values. This is useful for caching
the primitive data, thereby avoiding redundancy.

Parameters:
- primitive: The GLTF primitive for which to generate the key.

Returns:
- A unique string key representing the primitive's attributes.
-------------------------------------------------------------------------------------------------*/
std::string generatePrimitiveKey(const tinygltf::Primitive& primitive);


/*-------------------------------------------------------------------------------------------------
## Function `traverseSceneGraph`
> Traverses the scene graph and calls the provided functions for each element.

This utility function recursively traverses the scene graph starting from the
specified node ID. It calls the provided functions for cameras, lights, and
meshes when encountered. The traversal can be stopped early if any function
returns `true`.

Parameters:
- model: The GLTF model containing the scene graph.
- nodeID: The ID of the node to start traversal from.
- parentMat: The transformation matrix of the parent node.
- fctCam: Function to call when a camera is encountered. Can be `nullptr`.
- fctLight: Function to call when a light is encountered. Can be `nullptr`.
- fctMesh: Function to call when a mesh is encountered. Can be `nullptr`.
-------------------------------------------------------------------------------------------------*/
void traverseSceneGraph(const tinygltf::Model&                            model,
                        int                                               nodeID,
                        const glm::mat4&                                  parentMat,
                        const std::function<bool(int, const glm::mat4&)>& fctCam   = nullptr,
                        const std::function<bool(int, const glm::mat4&)>& fctLight = nullptr,
                        const std::function<bool(int, const glm::mat4&)>& fctMesh  = nullptr,
                        const std::function<bool(int, const glm::mat4&)>& anyNode  = nullptr);


/*-------------------------------------------------------------------------------------------------
## Function `getVertexCount`
> Returns the number of vertices in a primitive.

This function retrieves the number of vertices for the given GLTF primitive
by accessing the "POSITION" attribute in the model's accessors.

Parameters:
- model: The GLTF model containing the primitive data.
- primitive: The GLTF primitive for which to retrieve the vertex count.

Returns:
- The number of vertices in the primitive.
-------------------------------------------------------------------------------------------------*/
size_t getVertexCount(const tinygltf::Model& model, const tinygltf::Primitive& primitive);

/*-------------------------------------------------------------------------------------------------
## Function `getIndexCount`
> Returns the number of indices in a primitive.

This function retrieves the number of indices for the given GLTF primitive
by accessing the indices in the model's accessors. If no indices are present,
it returns the number of vertices instead.

Parameters:
- model: The GLTF model containing the primitive data.
- primitive: The GLTF primitive for which to retrieve the index count.

Returns:
- The number of indices in the primitive, or the number of vertices if no indices are present.
-------------------------------------------------------------------------------------------------*/
size_t getIndexCount(const tinygltf::Model& model, const tinygltf::Primitive& primitive);

/*-------------------------------------------------------------------------------------------------
## Function `hasElementName<MapType>`
> Check if the map has the specified element.

Can be used for extensions, extras, or any other map.
Returns `true` if the map has the specified element, `false` otherwise.
-------------------------------------------------------------------------------------------------*/
template <typename MapType>
bool hasElementName(const MapType& map, const std::string& key)
{
  return map.find(key) != map.end();
}


/*-------------------------------------------------------------------------------------------------
## Function `getElementValue<MapType>`
> Get the value of the specified element from the map.

Can be `extensions`, `extras`, or any other map.
Returns the value of the element.
-------------------------------------------------------------------------------------------------*/
template <typename MapType>
const typename MapType::mapped_type& getElementValue(const MapType& map, const std::string& key)
{
  return map.at(key);
}


// ScalarTypeGetter<T> returns the type of a single element (scalar) of T if
// T is a GLM vector, or T itself otherwise.
// We use this template specialization trick here, because T::value_type can
// only be constructed if T is a GLM vector.
template <typename T, bool isVector = std::is_class_v<T>>
struct ScalarTypeGetter
{
  using type = T::value_type;
};

template <typename T>
struct ScalarTypeGetter<T, false>
{
  using type = T;
};

/*-------------------------------------------------------------------------------------------------
## Function `isAccessorSimple<T>`
Returns whether the data an accessor points to can be directly used as an array of T.

Specifically, an accessor is simple if it:
* has type T (i.e. doesn't need type conversion)
* and is closely packed
* and is not a sparse accessor.

`T` must be `float`, `uint32_t`, `int32_t`, or a GLM vector or matrix type.
-----------------------------------------------------------------------------------------------*/
template <typename T>
bool isAccessorSimple(const tinygltf::Model& tmodel, const tinygltf::Accessor& accessor)
{
  // Sparse-only accessors (bufferView == -1) are never simple
  if(accessor.bufferView < 0)
  {
    return false;
  }
  const auto& bufferView          = tmodel.bufferViews[accessor.bufferView];
  using ScalarType                = ScalarTypeGetter<T>::type;
  constexpr int gltfComponentType = (std::is_same_v<ScalarType, float> ? TINYGLTF_COMPONENT_TYPE_FLOAT :  //
                                         (std::is_same_v<ScalarType, uint32_t> ? TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT :  //
                                              TINYGLTF_COMPONENT_TYPE_INT));
  return (accessor.componentType == gltfComponentType)                          // Must not require conversion
         && (bufferView.byteStride == 0 || bufferView.byteStride == sizeof(T))  // Must not require re-packing
         && !accessor.sparse.isSparse;                                          // Must not be sparse
}

/*-------------------------------------------------------------------------------------------------
## Function `forEachSparseValue<T>`
> Calls a function (such as a lambda function) for each `(index, value)` pair in
> a sparse accessor.

It's only potentially called for indices from
`accessorFirstElement` through `accessorFirstElement + numElementsToProcess - 1`.
-------------------------------------------------------------------------------------------------*/
template <class T>
void forEachSparseValue(const tinygltf::Model&                            tmodel,
                        const tinygltf::Accessor&                         accessor,
                        size_t                                            accessorFirstElement,
                        size_t                                            numElementsToProcess,
                        std::function<void(size_t index, const T* value)> fn)
{
  if(!accessor.sparse.isSparse)
  {
    return;  // Nothing to do
  }

  const auto& idxs = accessor.sparse.indices;
  if(!(idxs.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE      //
       || idxs.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT  //
       || idxs.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT))
  {
    assert(!"Unsupported sparse accessor index type.");
    return;
  }

  const tinygltf::BufferView& idxBufferView = tmodel.bufferViews[idxs.bufferView];
  const unsigned char*        idxBuffer     = &tmodel.buffers[idxBufferView.buffer].data[idxBufferView.byteOffset];
  const size_t                idxBufferByteStride =
      idxBufferView.byteStride ? idxBufferView.byteStride : tinygltf::GetComponentSizeInBytes(idxs.componentType);
  if(idxBufferByteStride == size_t(-1))
    return;  // Invalid

  const auto&                 vals          = accessor.sparse.values;
  const tinygltf::BufferView& valBufferView = tmodel.bufferViews[vals.bufferView];
  const unsigned char*        valBuffer     = &tmodel.buffers[valBufferView.buffer].data[valBufferView.byteOffset];
  const size_t                valBufferByteStride = accessor.ByteStride(valBufferView);
  if(valBufferByteStride == size_t(-1))
    return;  // Invalid

  // Note that this could be faster for lots of small copies, since we could
  // binary search for the first sparse accessor index to use (since the
  // glTF specification requires the indices be sorted)!
  for(size_t pairIdx = 0; pairIdx < accessor.sparse.count; pairIdx++)
  {
    // Read the index from the index buffer, converting its type
    size_t               index = 0;
    const unsigned char* pIdx  = idxBuffer + idxBufferByteStride * pairIdx;
    switch(idxs.componentType)
    {
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        index = *reinterpret_cast<const uint8_t*>(pIdx);
        break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        index = *reinterpret_cast<const uint16_t*>(pIdx);
        break;
      case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        index = *reinterpret_cast<const uint32_t*>(pIdx);
        break;
    }

    // If it's not in range, skip it
    if(index < accessorFirstElement || (index - accessorFirstElement) >= numElementsToProcess)
    {
      continue;
    }

    fn(index, reinterpret_cast<const T*>(valBuffer + valBufferByteStride * pairIdx));
  }
}

/*-------------------------------------------------------------------------------------------------
The following 4 functions access accessor data. They all handle all types of accessors.

The `accessor` functions take an accessor as an argument; the `attribute`
functions load an accessor with a given name from an object.

The `get` functions return a pointer to the data and have a fast-path.
the `copy` functions always copy and output to a vector.

If the accessor is complex, the `get` functions can use an `std::vector`
for storage to unpack values; this vector must remain alive as long as the
span is in use.
They return a span with nullptr data and 0 length on error, or if the accessor
was complex and no temporary storage was provided.

The `copy` functions return true on success. This distinguishes between an
empty accessor and a failed copy.

`T` must be `float`, `uint32_t`, `int32_t`, or a GLM vector or matrix type.
-------------------------------------------------------------------------------------------------*/

/*-------------------------------------------------------------------------------------------------
## Function `getAccessorData<T>`
> Returns a span with all the values of `accessor`.

Examples:
```
// Get a glTF primitive's indices.
const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
std::vector<uint32_t>     indexStorage;
std::span<const uint32_t> indices = getAccessorData<uint32_t>(model, indexAccessor, &indexStorage);

// The same, but returns null if the accessor is complex. Saves one line of code.
std::span<const uint32_t> indices = getAccessorData<uint32_t>(model, indexAccessor, nullptr);
```
-------------------------------------------------------------------------------------------------*/
template <typename T>
inline std::span<T> getAccessorData(tinygltf::Model& model, const tinygltf::Accessor& accessor, std::vector<T>* storageIfComplex)
{
  // The following block of code figures out how to access T.
  using ScalarType                 = ScalarTypeGetter<T>::type;
  constexpr bool toFloat           = std::is_same_v<ScalarType, float>;
  constexpr int  gltfComponentType = (toFloat ? TINYGLTF_COMPONENT_TYPE_FLOAT :  //
                                         (std::is_same_v<ScalarType, uint32_t> ? TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT :  //
                                              TINYGLTF_COMPONENT_TYPE_INT));
  // 1, 2, 3, 4, 16 for scalar, VEC2, VEC3, VEC4, MAT4, etc.
  constexpr int nbComponents = sizeof(T) / sizeof(ScalarType);

  // Make sure the input and output have the same number of components
  if(nbComponents != tinygltf::GetNumComponentsInType(accessor.type))
  {
    return {};  // Invalid
  }

  // Handle sparse-only accessors (bufferView == -1): per glTF spec, all elements
  // are zero-initialized, then sparse values are applied on top.
  if(accessor.bufferView < 0)
  {
    if(!storageIfComplex || !storageIfComplex->empty())
    {
      return {};
    }
    std::vector<T>& storageRef = *storageIfComplex;
    storageRef.resize(accessor.count, T{});  // Zero-initialize all elements

    // Apply sparse values on top
    forEachSparseValue<T>(model, accessor, 0, accessor.count,
                          [&storageRef](size_t index, const T* value) { storageRef[index] = *value; });

    return std::span<T>(storageRef.data(), storageRef.size());
  }

  tinygltf::BufferView& view        = model.bufferViews[accessor.bufferView];
  tinygltf::Buffer&     buffer      = model.buffers[view.buffer];
  unsigned char*        bufferBytes = &buffer.data[accessor.byteOffset + view.byteOffset];

  // Fast path: Can we return a pointer to the data directly?
  if(isAccessorSimple<T>(model, accessor))
  {
    T* bufferData = reinterpret_cast<T*>(bufferBytes);
    return std::span<T>(bufferData, accessor.count);
  }

  // The accessor is complex, so we need to unpack to storage.
  // It must exist, and be empty.
  if(!storageIfComplex || !storageIfComplex->empty())
  {
    return {};
  }

  std::vector<T>& storageRef = *storageIfComplex;
  storageRef.resize(accessor.count);


  const size_t byteStride = accessor.ByteStride(view);
  if(byteStride == size_t(-1))
  {
    return {};  // Invalid
  }

  // Copying the attributes
  if(accessor.componentType == gltfComponentType)
  {
    // No type conversion necessary.
    // Can we memcpy?
    assert(0 != byteStride);
    if(sizeof(T) == byteStride)
    {
      memcpy(storageRef.data(), bufferBytes, accessor.count * sizeof(T));
    }
    else
    {
      // Must copy one-by-one
      for(size_t i = 0; i < accessor.count; i++)
      {
        storageRef[i] = *reinterpret_cast<const T*>(bufferBytes + byteStride * i);
      }
    }

    // Handle sparse accessors by overwriting already copied elements.
    forEachSparseValue<T>(model, accessor, 0, accessor.count,
                          [&storageRef](size_t index, const T* value) { storageRef[index] = *value; });
  }
  else
  {
    // The component is smaller than 32 bits and needs to be converted
    if(!(accessor.componentType == TINYGLTF_COMPONENT_TYPE_BYTE || accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE
         || accessor.componentType == TINYGLTF_COMPONENT_TYPE_SHORT || accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT))
    {
      assert(!"Unhandled tinygltf component type!");
      return {};
    }

    const auto& copyElementFn = [&](size_t elementIdx, const unsigned char* pElement) {
      T vecValue{};

      for(int c = 0; c < nbComponents; c++)
      {
        ScalarType v{};

        switch(accessor.componentType)
        {
          case TINYGLTF_COMPONENT_TYPE_BYTE:
            v = static_cast<ScalarType>(*(reinterpret_cast<const char*>(pElement) + c));
            if constexpr(toFloat)
            {
              if(accessor.normalized)
              {
                v = std::max(v / 127.f, -1.f);
              }
            }
            break;
          case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            v = static_cast<ScalarType>(*(reinterpret_cast<const unsigned char*>(pElement) + c));
            if constexpr(toFloat)
            {
              if(accessor.normalized)
              {
                v = v / 255.f;
              }
            }
            break;
          case TINYGLTF_COMPONENT_TYPE_SHORT:
            v = static_cast<ScalarType>(*(reinterpret_cast<const short*>(pElement) + c));
            if constexpr(toFloat)
            {
              if(accessor.normalized)
              {
                v = std::max(v / 32767.f, -1.f);
              }
            }
            break;
          case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
            v = static_cast<ScalarType>(*(reinterpret_cast<const unsigned short*>(pElement) + c));
            if constexpr(toFloat)
            {
              if(accessor.normalized)
              {
                v = v / 65535.f;
              }
            }
            break;
        }

        if constexpr(nbComponents == 1)
        {
          vecValue = v;
        }
        else
        {
          glm::value_ptr(vecValue)[c] = v;
        }
      }

      storageRef[elementIdx] = vecValue;
    };

    for(size_t i = 0; i < accessor.count; i++)
    {
      copyElementFn(i, bufferBytes + byteStride * i);
    }

    forEachSparseValue<unsigned char>(model, accessor, 0, accessor.count, copyElementFn);
  }

  return std::span<T>(storageRef.data(), storageRef.size());
}

template <typename T>
inline std::span<const T> getAccessorData(const tinygltf::Model& model, const tinygltf::Accessor& accessor, std::vector<T>* storageIfComplex)
{
  return getAccessorData(const_cast<tinygltf::Model&>(model), accessor, storageIfComplex);
}

/*-------------------------------------------------------------------------------------------------
## Function `copyAccessorData<T>`
> Copies all the values of `accessor` to a vector. Returns true on success.

Example:
```
// Get a glTF primitive's indices.
std::vector<uint32_t> indices;
if(!copyAccessorData<uint32_t>(model, model.accessors[primitive.indices], indices))
{
  LOGE("Accessor was not valid!\n");
}
```
-------------------------------------------------------------------------------------------------*/
template <typename T>
inline bool copyAccessorData(const tinygltf::Model& model, const tinygltf::Accessor& accessor, std::vector<T>& output)
{
  if(!output.empty())
  {
    assert(!"Output must be empty!\n");
    return false;
  }

  std::span<const T> dataPointer = getAccessorData(model, accessor, &output);
  if(nullptr == dataPointer.data())
  {
    // Retrieving the accessor failed.
    return false;
  }

  if(output.empty())
  {
    // getAccessorData returned a raw pointer to the data. Make a copy:
    output.assign(dataPointer.begin(), dataPointer.end());
  }
  return true;
}

/*-------------------------------------------------------------------------------------------------
## Function `getAttributeData<T>`
> Returns a span with all the values of the primitive's attribute with the given name.

Examples:
```
// Get a glTF primitive's vertex positions.
std::vector<glm::vec3>     vertexStorage;
std::span<const glm::vec3> vertices = getAttributeData<glm::vec3>(model, primitive, "POSITION", &indexStorage);

// The same, but returns null if the accessor is complex. Saves one line of code.
std::span<const glm::vec3> vertices = getAttributeData<glm::vec3>(model, primitive, "POSITION", nullptr);
```
-------------------------------------------------------------------------------------------------*/
template <typename T>
inline std::span<T> getAttributeData3(tinygltf::Model&           model,
                                      const tinygltf::Primitive& primitive,
                                      const std::string&         attributeName,
                                      std::vector<T>*            storageIfComplex)
{
  const auto& it = primitive.attributes.find(attributeName);
  if(primitive.attributes.end() == it)
    return {};

  const tinygltf::Accessor& accessor = model.accessors.at(it->second);
  return getAccessorData(model, accessor, storageIfComplex);
}

template <typename T>
inline std::span<const T> getAttributeData3(const tinygltf::Model&     model,
                                            const tinygltf::Primitive& primitive,
                                            const std::string&         attributeName,
                                            std::vector<T>*            storageIfComplex)
{
  return getAttributeData3<T>(const_cast<tinygltf::Model&>(model), primitive, attributeName, storageIfComplex);
}

template <typename T>
inline std::span<T> getAttributeData3(tinygltf::Model&       model,
                                      const tinygltf::Value& attributes,
                                      const std::string&     attributeName,
                                      std::vector<T>*        storageIfComplex)
{
  if(!attributes.Has(attributeName))
    return {};

  const tinygltf::Accessor& accessor = model.accessors.at(attributes.Get(attributeName).GetNumberAsInt());
  return getAccessorData(model, accessor, storageIfComplex);
}

template <typename T>
inline std::span<const T> getAttributeData3(const tinygltf::Model& model,
                                            const tinygltf::Value& attributes,
                                            const std::string&     attributeName,
                                            std::vector<T>*        storageIfComplex)
{
  return getAttributeData3<T>(const_cast<tinygltf::Model&>(model), attributes, attributeName, storageIfComplex);
}

/*-------------------------------------------------------------------------------------------------
## Function `copyAttributeData<T>`
> Copies all the values of the primitive's attribute to a vector. Returns true on success.

Example:
```
// Get a glTF primitive's vertex positions.
std::vector<glm::vec3> vertices;
if(!copyAttributeData<glm::vec3>(model, primitive, "POSITION", vertices))
{
  LOGE("Accessor was not valid or attribute did not exist!\n");
}
```
-------------------------------------------------------------------------------------------------*/
template <typename T>
inline bool copyAttributeData(const tinygltf::Model&     model,
                              const tinygltf::Primitive& primitive,
                              const std::string&         attributeName,
                              std::vector<T>&            output)
{
  const auto& it = primitive.attributes.find(attributeName);
  if(primitive.attributes.end() == it)
  {
    return false;
  }

  const tinygltf::Accessor& accessor = model.accessors.at(it->second);
  return copyAccessorData<T>(model, accessor, output);
}

template <typename T>
inline bool copyAttributeData3(const tinygltf::Model& model,
                               const tinygltf::Value& attributes,
                               const std::string&     attributeName,
                               std::vector<T>&        output)
{
  if(!attributes.Has(attributeName))
  {
    return false;
  }

  const tinygltf::Accessor& accessor = model.accessors.at(attributes.Get(attributeName).GetNumberAsInt());
  return copyAccessorData<T>(model, accessor, output);
}


/*-------------------------------------------------------------------------------------------------
## Function `appendData<T>`
> Appends data from `inData` to the binary buffer `buffer` and returns the number
> of bytes of data added.

`T` should be a type like `std::vector`.
-------------------------------------------------------------------------------------------------*/
template <class T>
uint32_t appendData(tinygltf::Buffer& buffer, const T& inData)
{
  auto*    pData = reinterpret_cast<const char*>(inData.data());
  uint32_t len   = static_cast<uint32_t>(sizeof(inData[0]) * inData.size());
  buffer.data.insert(buffer.data.end(), pData, pData + len);
  return len;
}


//--------------------------------------------------------------------------------------------------
// Materials
//--------------------------------------------------------------------------------------------------
KHR_materials_unlit          getUnlit(const tinygltf::Material& tmat);
void                         setUnlit(tinygltf::Material& tmat, const KHR_materials_unlit& unlit);
KHR_materials_specular       getSpecular(const tinygltf::Material& tmat);
void                         setSpecular(tinygltf::Material& tmat, const KHR_materials_specular& specular);
KHR_materials_clearcoat      getClearcoat(const tinygltf::Material& tmat);
void                         setClearcoat(tinygltf::Material& tmat, const KHR_materials_clearcoat& clearcoat);
KHR_materials_sheen          getSheen(const tinygltf::Material& tmat);
void                         setSheen(tinygltf::Material& tmat, const KHR_materials_sheen& sheen);
KHR_materials_transmission   getTransmission(const tinygltf::Material& tmat);
void                         setTransmission(tinygltf::Material& tmat, const KHR_materials_transmission& transmission);
KHR_materials_anisotropy     getAnisotropy(const tinygltf::Material& tmat);
void                         setAnisotropy(tinygltf::Material& tmat, const KHR_materials_anisotropy& anisotropy);
KHR_materials_ior            getIor(const tinygltf::Material& tmat);
void                         setIor(tinygltf::Material& tmat, const KHR_materials_ior& ior);
KHR_materials_volume         getVolume(const tinygltf::Material& tmat);
void                         setVolume(tinygltf::Material& tmat, const KHR_materials_volume& volume);
KHR_materials_volume_scatter getVolumeScatter(const tinygltf::Material& tmat);
void                         setVolumeScatter(tinygltf::Material& tmat, const KHR_materials_volume_scatter& scatter);
KHR_materials_displacement   getDisplacement(const tinygltf::Material& tmat);
void                         setDisplacement(tinygltf::Material& tmat, const KHR_materials_displacement& displacement);
KHR_materials_emissive_strength getEmissiveStrength(const tinygltf::Material& tmat);
void setEmissiveStrength(tinygltf::Material& tmat, const KHR_materials_emissive_strength& emissiveStrength);
KHR_materials_iridescence getIridescence(const tinygltf::Material& tmat);
void                      setIridescence(tinygltf::Material& tmat, const KHR_materials_iridescence& iridescence);
KHR_materials_dispersion  getDispersion(const tinygltf::Material& tmat);
void                      setDispersion(tinygltf::Material& tmat, const KHR_materials_dispersion& dispersion);
KHR_materials_pbrSpecularGlossiness getPbrSpecularGlossiness(const tinygltf::Material& tmat);
void setPbrSpecularGlossiness(tinygltf::Material& tmat, const KHR_materials_pbrSpecularGlossiness& dispersion);
KHR_materials_diffuse_transmission getDiffuseTransmission(const tinygltf::Material& tmat);
void setDiffuseTransmission(tinygltf::Material& tmat, const KHR_materials_diffuse_transmission& diffuseTransmission);


template <typename T>
inline KHR_texture_transform getTextureTransform(const T& tinfo)
{
  KHR_texture_transform gmat;
  if(tinygltf::utils::hasElementName(tinfo.extensions, KHR_TEXTURE_TRANSFORM_EXTENSION_NAME))
  {
    const tinygltf::Value& ext = tinygltf::utils::getElementValue(tinfo.extensions, KHR_TEXTURE_TRANSFORM_EXTENSION_NAME);
    tinygltf::utils::getArrayValue(ext, "offset", gmat.offset);
    tinygltf::utils::getArrayValue(ext, "scale", gmat.scale);
    tinygltf::utils::getValue(ext, "rotation", gmat.rotation);
    tinygltf::utils::getValue(ext, "texCoord", gmat.texCoord);

    gmat.updateTransform();
  }
  return gmat;
}

/*-------------------------------------------------------------------------------------------------
## Function `getTextureImageIndex`
> Retrieves the image index of a texture, accounting for extensions such as
> `MSFT_texture_dds` and `KHR_texture_basisu`.
-------------------------------------------------------------------------------------------------*/
int getTextureImageIndex(const tinygltf::Texture& texture);

/*-------------------------------------------------------------------------------------------------
> Retrieves the visibility of a node using `KHR_node_visibility`.

Does not search up the node hierarchy, so e.g. if node A points to node B and
node A is set to invisible and node B is set to visible, then
`getNodeVisibility(B)` will return `KHR_node_visibility{true}` even though
node B would not be visible due to node A.
-------------------------------------------------------------------------------------------------*/
KHR_node_visibility getNodeVisibility(const tinygltf::Node& node);
void                setNodeVisibility(tinygltf::Node& node, const KHR_node_visibility& visibility);

/*-------------------------------------------------------------------------------------------------
## Function `createTangentAttribute`
 Create a tangent attribute for the primitive
-------------------------------------------------------------------------------------------------*/
void createTangentAttribute(tinygltf::Model& model, tinygltf::Primitive& primitive);

/*-------------------------------------------------------------------------------------------------
## Function `simpleCreateTangents`
Compute tangents based on the texture coordinates, using also position and normal attributes
-------------------------------------------------------------------------------------------------*/
void simpleCreateTangents(tinygltf::Model& model, tinygltf::Primitive& primitive);

/*------------------------------------------------------------------------------------------------*/
bool getMeshoptCompression(const tinygltf::BufferView& bview, EXT_meshopt_compression& mcomp);

}  // namespace utils

}  // namespace tinygltf
