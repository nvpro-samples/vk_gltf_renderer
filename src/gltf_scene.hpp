/*
 * Copyright (c) 2019-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>
#include <tinygltf/tiny_gltf.h>
#include <nvutils/bounding_box.hpp>

#include "tinygltf_utils.hpp"
#include "gltf_animation_pointer.hpp"


namespace nvvkgltf {

// The render node is the instance of a primitive in the scene that will be rendered
struct RenderNode
{
  glm::mat4 worldMatrix  = glm::mat4(1.0f);
  int       materialID   = 0;   // Reference to the material
  int       renderPrimID = -1;  // Reference to the unique primitive
  int       refNodeID    = -1;  // Reference to the tinygltf::Node
  int       skinID       = -1;  // Reference to the skin, if the node is skinned, -1 if not skinned
  bool      visible      = true;
};

// The RenderPrimitive is a unique primitive in the scene
struct RenderPrimitive
{
  tinygltf::Primitive* pPrimitive  = nullptr;
  int                  vertexCount = 0;
  int                  indexCount  = 0;
  int                  meshID      = 0;
};

struct RenderCamera
{
  enum CameraType
  {
    ePerspective,
    eOrthographic
  };

  CameraType type   = ePerspective;
  glm::vec3  eye    = {0.0f, 0.0f, 0.0f};
  glm::vec3  center = {0.0f, 0.0f, 0.0f};
  glm::vec3  up     = {0.0f, 1.0f, 0.0f};

  // Perspective
  double yfov = {0.0};  // in radians

  // Orthographic
  double xmag = {0.0};
  double ymag = {0.0};

  double znear = {0.0};
  double zfar  = {0.0};
};

// See: https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md
struct RenderLight
{
  glm::mat4 worldMatrix = glm::mat4(1.0f);
  int       light       = 0;
};

// Animation data
struct AnimationInfo
{
  std::string name;
  float       start       = std::numeric_limits<float>::max();
  float       end         = std::numeric_limits<float>::min();
  float       currentTime = 0.0f;
  float       reset() { return currentTime = start; }
  float       incrementTime(float deltaTime, bool loop = true)
  {
    currentTime += deltaTime;
    if(loop)
    {
      float duration = end - start;
      // Wrap currentTime around using modulo arithmetic
      float wrapped = std::fmod(currentTime - start, duration);
      // fmod can return negative values if (currentTime - start) < 0, so fix that.
      if(wrapped < 0.0f)
        wrapped += duration;

      currentTime = start + wrapped;
    }
    else
    {
      if(currentTime > end)
      {
        currentTime = end;
      }
    }
    return currentTime;
  }
};


/*-------------------------------------------------------------------------------------------------

# nvh::nvvkgltf::Scene 

The Scene class is responsible for loading and managing a glTF scene.
- It is used to load a glTF file and parse it into a scene representation.
- It can be used to save the scene back to a glTF file.
- It can be used to manage the animations of the scene.
- What it returns is a list of RenderNodes, RenderPrimitives, RenderCameras, and RenderLights.
-  RenderNodes are the instances of the primitives in the scene that will be rendered.
-  RenderPrimitives are the unique primitives in the scene.

Note: The Scene class is a more advanced and light weight version of the GltfScene class.
      But it is to the user to retrieve the primitive data from the RenderPrimitives.
      Check the tinygltf_utils.hpp for more information on how to extract the primitive data.

-------------------------------------------------------------------------------------------------*/


class Scene
{
public:
  Scene();

  // Used to specify the type of pipeline to be used
  enum PipelineType
  {
    eRasterSolid,
    eRasterSolidDoubleSided,
    eRasterBlend,
    eRasterAll
  };

  // File Management
  bool load(const std::filesystem::path& filename);  // Load the glTF file, .gltf or .glb
  bool save(const std::filesystem::path& filename);  // Save the glTF file, .gltf or .glb
  // Read or modify supported extensions before loading (for e.g. custom image formats)
  std::unordered_set<std::string>& supportedExtensions() { return m_supportedExtensions; }
  const std::filesystem::path&     getFilename() const { return m_filename; }
  void                             takeModel(tinygltf::Model&& model);  // Use a model that has been loaded

  // Getters
  const tinygltf::Model& getModel() const { return m_model; }
  tinygltf::Model&       getModel() { return m_model; }
  bool                   valid() const { return m_validSceneParsed; }

  // Update render node world matrices.
  // If dirtyNodeIds is empty, performs a full update.
  void updateNodeWorldMatrices(const std::unordered_set<int>& dirtyNodeIds = {});
  void updateVisibility(int nodeID);

  // Animation Management
  std::unordered_set<int>  updateAnimation(uint32_t animationIndex);
  int                      getNumAnimations() const { return static_cast<int>(m_animations.size()); }
  bool                     hasAnimation() const { return !m_animations.empty(); }
  nvvkgltf::AnimationInfo& getAnimationInfo(int index) { return m_animations[index].info; }

  // KHR_animation_pointer - Get dirty resources (for GPU updates)
  nvvkgltf::AnimationPointerSystem& getAnimationPointer() { return m_animationPointer; }

  // Resource Management
  void destroy();  // Destroy the loaded resources

  // Light Management
  const std::vector<nvvkgltf::RenderLight>& getRenderLights() const { return m_lights; }

  // Camera Management
  const std::vector<nvvkgltf::RenderCamera>& getRenderCameras(bool force = false);
  void                                       setSceneCamera(const nvvkgltf::RenderCamera& camera);
  void                                       setSceneCameras(const std::vector<nvvkgltf::RenderCamera>& cameras);

  // Render Node Management
  const std::vector<nvvkgltf::RenderNode>& getRenderNodes() const { return m_renderNodes; }
  // Collect render node indices affected by node changes (optionally including descendants).
  // Returns true if a full update is recommended (e.g., root node changed or too many nodes affected).
  // Returns false if a partial update is recommended or if no render nodes are affected.
  bool collectRenderNodeIndices(const std::unordered_set<int>& nodeIds,
                                std::unordered_set<int>&       outRenderNodeIndices,
                                bool                           includeDescendants = true,
                                float                          fullUpdateRatio    = 0.5f) const;
  // Map (nodeID, primitiveIndex) to RenderNode index using the scene-owned mapping.
  int getRenderNodeForPrimitive(int nodeId, int primitiveIndex) const;
  // Map RenderNode index back to primitive index within its node.
  int getPrimitiveIndexForRenderNode(int renderNodeIndex) const;

  // Render Primitive Management
  const std::vector<nvvkgltf::RenderPrimitive>& getRenderPrimitives() const { return m_renderPrimitives; }
  const nvvkgltf::RenderPrimitive&              getRenderPrimitive(size_t ID) const { return m_renderPrimitives[ID]; }
  size_t                                        getNumRenderPrimitives() const { return m_renderPrimitives.size(); }
  const std::vector<uint32_t>&                  getMorphPrimitives() const { return m_morphPrimitives; }
  const std::vector<uint32_t>&                  getSkinNodes() const { return m_skinNodes; }

  // Scene Management
  void setCurrentScene(int sceneID);  // Parse the scene and create the render nodes, call when changing scene
  int  getCurrentScene() const { return m_currentScene; }
  const std::vector<glm::mat4>& getNodesWorldMatrices() const { return m_nodesWorldMatrices; }

  // Variant Management
  // Set the variant to be used and return the list of modified renderNodes
  void                            setCurrentVariant(int variant, std::unordered_set<int>& dirtyRenderNodes);
  const std::vector<std::string>& getVariants() const { return m_variants; }
  int                             getCurrentVariant() const { return m_currentVariant; }

  // Return the list of render nodes that are affected by the material variant change
  std::unordered_set<int> getMaterialRenderNodes(const std::unordered_set<int>& materialVariantNodeIDs) const;

  // Shading Management
  std::vector<uint32_t> getShadedNodes(PipelineType type);  // Get the nodes that will be shaded by the pipeline type

  // Statistics
  int           getNumTriangles() const { return m_numTriangles; }
  nvutils::Bbox getSceneBounds();


private:
  struct AnimationChannel
  {
    enum PathType
    {
      eTranslation,
      eRotation,
      eScale,
      eWeights,
      ePointer
    };
    PathType path         = eTranslation;
    int      node         = -1;
    uint32_t samplerIndex = 0;

    std::string pointerPath;  // JSON pointer string (e.g., "/materials/0/pbrMetallicRoughness/baseColorFactor")
  };

  struct AnimationSampler
  {
    enum InterpolationType
    {
      eLinear,
      eStep,
      eCubicSpline
    };
    InterpolationType               interpolation = eLinear;
    std::vector<float>              inputs;
    std::vector<glm::vec2>          outputsVec2;
    std::vector<glm::vec3>          outputsVec3;
    std::vector<glm::vec4>          outputsVec4;
    std::vector<std::vector<float>> outputsFloat;
  };

  struct Animation
  {
    AnimationInfo                 info;
    std::vector<AnimationSampler> samplers;
    std::vector<AnimationChannel> channels;
  };


  void parseScene();                    // Parse the scene and create the render nodes
  void clearParsedData();               // Clear the parsed data
  void parseAnimations();               // Parse the animations
  void parseVariants();                 // Parse the variants
  void setSceneElementsDefaultNames();  // Set a default name for the scene elements
  void createSceneCamera();             // Create a camera for the scene

  int getUniqueRenderPrimitive(tinygltf::Primitive& primitive, int meshID);
  int getMaterialVariantIndex(const tinygltf::Primitive& primitive, int currentVariant);

  bool   handleRenderNode(int nodeID, glm::mat4 worldMatrix);
  size_t handleGpuInstancing(const tinygltf::Value& attributes, nvvkgltf::RenderNode renderNode, glm::mat4 worldMatrix);
  bool   handleCameraTraversal(int nodeID, const glm::mat4& worldMatrix);
  bool   handleLightTraversal(int nodeID, const glm::mat4& worldMatrix);
  void   createMissingTangents();
  bool processAnimationChannel(tinygltf::Node* gltfNode, AnimationSampler& sampler, const AnimationChannel& channel, float time);
  float calculateInterpolationFactor(float inputStart, float inputEnd, float time);
  void handleLinearInterpolation(tinygltf::Node* gltfNode, AnimationSampler& sampler, const AnimationChannel& channel, float t, size_t index);
  void handleStepInterpolation(tinygltf::Node* gltfNode, AnimationSampler& sampler, const AnimationChannel& channel, size_t index);
  void handleCubicSplineInterpolation(tinygltf::Node*         gltfNode,
                                      AnimationSampler&       sampler,
                                      const AnimationChannel& channel,
                                      float                   t,
                                      float                   keyDelta,
                                      size_t                  index);
  void updateRenderNodesFull();


  tinygltf::Model                        m_model;                 // The glTF model
  std::filesystem::path                  m_filename;              // Filename of the glTF
  std::vector<nvvkgltf::RenderNode>      m_renderNodes;           // Render nodes
  std::vector<nvvkgltf::RenderPrimitive> m_renderPrimitives;      // Unique primitives from key
  std::vector<nvvkgltf::RenderCamera>    m_cameras;               // Cameras
  std::vector<nvvkgltf::RenderLight>     m_lights;                // Lights
  std::vector<Animation>                 m_animations;            // Animations
  std::vector<std::string>               m_variants;              // KHR_materials_variants
  std::unordered_map<std::string, int>   m_uniquePrimitiveIndex;  // Key: primitive, Value: renderPrimID
  std::vector<uint32_t>                  m_morphPrimitives;       // All the primitives that are animated
  std::vector<uint32_t>                  m_skinNodes;             // All the primitives that are animated


  std::vector<glm::mat4>        m_nodesLocalMatrices;
  std::vector<glm::mat4>        m_nodesWorldMatrices;
  std::vector<std::vector<int>> m_nodeToRenderNodes;  // nodeID -> renderNode indices in primitive order
  std::vector<int>              m_nodeParents;        // Parent node index for each node (cached)

  nvvkgltf::AnimationPointerSystem m_animationPointer;     // Unified animation pointer system (direct member)
  std::unordered_set<std::string>  m_supportedExtensions;  // Modifiable list of extensions

  int           m_numTriangles    = 0;   // Stat - Number of triangles
  int           m_currentScene    = 0;   // Scene index
  int           m_currentVariant  = 0;   // Variant index
  int           m_sceneCameraNode = -1;  // Node index of the camera
  nvutils::Bbox m_sceneBounds;           // Scene bounds

  bool m_validSceneParsed = false;
};

}  // namespace nvvkgltf
