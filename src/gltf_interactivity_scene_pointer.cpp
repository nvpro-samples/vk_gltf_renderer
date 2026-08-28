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

#include <algorithm>
#include <charconv>
#include <vector>

#include "gltf_interactivity_scene_pointer.hpp"
#include "gltf_scene_animation.hpp"
#include "gltf_scene_editor.hpp"
#include "tinygltf_utils.hpp"

namespace nvvkgltf {

namespace {
// Shared with AnimationPointerSystem::parseResourceInfo() (same "/prefix/<index>/rest..."
// parsing) - see tinygltf::utils::parsePointerIndexAndRest()'s doc comment.
using tinygltf::utils::parsePointerIndexAndRest;
// Shared with AnimationPointerSystem::applyValue(float) - see its doc comment.
using tinygltf::utils::isBoolAnimationPointerPath;

std::vector<std::string> splitPath(const std::string& path)
{
  std::vector<std::string> segs;
  size_t                   i = 0;
  while(i < path.size())
  {
    if(path[i] == '/')
    {
      ++i;
      continue;
    }
    size_t end = path.find('/', i);
    if(end == std::string::npos)
      end = path.size();
    segs.push_back(path.substr(i, end - i));
    i = end;
  }
  return segs;
}

// Converts a generic tinygltf::Value leaf to an InteractivityValue, inferring the shape from the
// JSON value itself (a bool, a bare number, or a 2/3/4-element number array) rather than from a
// separately-declared expected type - callers (pointer/get's evaluator) already re-validate the
// result against the node's own configured type, so a wrong guess here just reports "invalid",
// not a silent misread.
std::optional<InteractivityValue> tinygltfValueToInteractivity(const tinygltf::Value& v)
{
  if(v.IsBool())
    return InteractivityValue(v.Get<bool>());
  if(v.IsNumber())
    return InteractivityValue(static_cast<float>(v.GetNumberAsDouble()));
  if(v.IsArray())
  {
    auto comp = [&](size_t i) { return static_cast<float>(v.Get(i).GetNumberAsDouble()); };
    switch(v.ArrayLen())
    {
      case 2:
        return InteractivityValue(glm::vec2(comp(0), comp(1)));
      case 3:
        return InteractivityValue(glm::vec3(comp(0), comp(1), comp(2)));
      case 4:
        return InteractivityValue(glm::vec4(comp(0), comp(1), comp(2), comp(3)));
      default:
        return std::nullopt;
    }
  }
  return std::nullopt;
}

// Walks `segs[startIdx..]` (which MUST begin with "extensions", "<EXTENSION_NAME>") through a
// tinygltf::ExtensionMap and however many nested object keys follow - the KHR_interactivity
// pointer/get path syntax for extension properties (spec's own "/extensions/<NAME>/prop" pattern),
// including properties nested arbitrarily deep inside an extension's own JSON (e.g. a texture-info
// sub-object's own "extensions/KHR_texture_transform/offset"), since tinygltf parses everything
// below an ExtensionMap entry as a plain recursive Value tree with no further special-casing.
const tinygltf::Value* walkExtensionsPath(const tinygltf::ExtensionMap& rootExt, const std::vector<std::string>& segs, size_t startIdx)
{
  if(startIdx + 1 >= segs.size() || segs[startIdx] != "extensions")
    return nullptr;
  auto it = rootExt.find(segs[startIdx + 1]);
  if(it == rootExt.end())
    return nullptr;
  const tinygltf::Value* cur = &it->second;
  for(size_t i = startIdx + 2; i < segs.size(); ++i)
  {
    if(!cur->IsObject() || !cur->Has(segs[i]))
      return nullptr;
    cur = &cur->Get(segs[i]);
  }
  return cur;
}

// Locates the ExtensionMap of a material's own (non-extension-nested) named texture-info slot, so
// `/materials/N/<slot>/extensions/KHR_texture_transform/...` can reuse walkExtensionsPath() above.
// Textures nested inside a material EXTENSION (e.g. KHR_materials_anisotropy's anisotropyTexture)
// don't need this - they're already plain Value-tree navigation once inside that extension's entry.
const tinygltf::ExtensionMap* namedTextureExtensions(const tinygltf::Material& mat, const std::string& slot)
{
  if(slot == "normalTexture")
    return &mat.normalTexture.extensions;
  if(slot == "occlusionTexture")
    return &mat.occlusionTexture.extensions;
  if(slot == "emissiveTexture")
    return &mat.emissiveTexture.extensions;
  if(slot == "pbrMetallicRoughness/baseColorTexture")
    return &mat.pbrMetallicRoughness.baseColorTexture.extensions;
  if(slot == "pbrMetallicRoughness/metallicRoughnessTexture")
    return &mat.pbrMetallicRoughness.metallicRoughnessTexture.extensions;
  return nullptr;
}

// InteractivityRef convention for object-model cross-references (spec type `ref`): the ref's
// handle is the target's own index within its owning array - same convention already used
// elsewhere in this engine for hoveredNode/selectedNode refs. A negative core-glTF index (e.g.
// node.mesh == -1, "no mesh") maps to the default null ref (handle == -1). `category` is the
// owning array's glTF name (e.g. "nodes", "materials") - it must match the array-name segment
// parseInteractivityLiteral() extracts from an authored "/category/index" JSON-Pointer literal, so
// a ref this resolver vends compares equal (via `ref/eq`) to an equivalent authored literal.
InteractivityValue indexRef(int index, const char* category)
{
  return InteractivityValue(InteractivityRef{index, category});
}

// Computes nodeIndex's world matrix fresh from the live tinygltf::Model, walking up the parent
// chain - deliberately NOT Scene::getNodesWorldMatrices()'s cache, which a same-tick pointer/set
// write (e.g. a graph moving this node then immediately reading /globalMatrix back) can leave
// stale until the next explicit updateNodeWorldMatrices() call. tinygltf::utils::getNodeMatrix()
// (the same function Scene itself uses to refresh that cache) already reads the Node's current
// TRS/matrix fields directly, so this stays correct across an arbitrary same-tick write.
glm::mat4 computeFreshWorldMatrix(const tinygltf::Model& model, const std::vector<int>& parents, int nodeIndex)
{
  const glm::mat4 local  = tinygltf::utils::getNodeMatrix(model.nodes[nodeIndex]);
  const int       parent = nodeIndex < static_cast<int>(parents.size()) ? parents[nodeIndex] : -1;
  if(parent < 0 || parent >= static_cast<int>(model.nodes.size()))
    return local;
  return computeFreshWorldMatrix(model, parents, parent) * local;
}
}  // namespace

std::optional<InteractivityValue> ScenePointerResolver::get(const std::string& concretePath) const
{
  const tinygltf::Model& model = m_scene.getModel();

  if(concretePath.rfind("/nodes/", 0) == 0)
  {
    auto [index, sub] = parsePointerIndexAndRest(concretePath, 7);
    if(index < 0 || index >= static_cast<int>(model.nodes.size()))
      return std::nullopt;
    const tinygltf::Node& node = model.nodes[index];
    if(sub == "/translation" || sub == "/rotation" || sub == "/scale")
    {
      glm::vec3 translation, scale;
      glm::quat rotation;
      tinygltf::utils::getNodeTRS(node, translation, rotation, scale);
      if(sub == "/translation")
        return InteractivityValue(translation);
      if(sub == "/scale")
        return InteractivityValue(scale);
      return InteractivityValue(glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w));
    }
    if(sub == "/matrix")
      return InteractivityValue(tinygltf::utils::getNodeMatrix(node));
    if(sub == "/globalMatrix")
      return InteractivityValue(computeFreshWorldMatrix(model, m_scene.getNodeParents(), index));
    if(sub == "/mesh")
      return indexRef(node.mesh, "meshes");
    if(sub == "/camera")
      return indexRef(node.camera, "cameras");
    if(sub == "/skin")
      return indexRef(node.skin, "skins");
    if(sub == "/parent")
    {
      const auto& parents = m_scene.getNodeParents();
      if(index >= static_cast<int>(parents.size()))
        return std::nullopt;
      return indexRef(parents[index], "nodes");
    }
    if(sub == "/children.length")
      return InteractivityValue(static_cast<int32_t>(node.children.size()));
    if(sub.rfind("/children/", 0) == 0)
    {
      auto [childPos, rest] = parsePointerIndexAndRest(sub, 10);
      if(!rest.empty() || childPos < 0 || childPos >= static_cast<int>(node.children.size()))
        return std::nullopt;
      return indexRef(node.children[childPos], "nodes");
    }
    if(sub == "/weights.length")
    {
      // A node without a mesh has no morph targets to weight at all - the property doesn't exist,
      // not "exists with length 0" (a real node.weights.size()==0 vs. "no mesh" distinction the
      // conformance suite checks for explicitly).
      if(node.mesh < 0)
        return std::nullopt;
      return InteractivityValue(static_cast<int32_t>(node.weights.size()));
    }
    if(sub.rfind("/weights/", 0) == 0)
    {
      auto [weightPos, rest] = parsePointerIndexAndRest(sub, 9);
      if(!rest.empty() || weightPos < 0 || weightPos >= static_cast<int>(node.weights.size()))
        return std::nullopt;
      return InteractivityValue(static_cast<float>(node.weights[weightPos]));
    }
    if(sub == "/extensions/KHR_node_visibility/visible")
      return InteractivityValue(tinygltf::utils::getNodeVisibility(node).visible);
    return std::nullopt;
  }

  if(concretePath.rfind("/materials/", 0) == 0)
  {
    auto [index, sub] = parsePointerIndexAndRest(concretePath, 11);
    if(index < 0 || index >= static_cast<int>(model.materials.size()))
      return std::nullopt;
    const tinygltf::Material& mat = model.materials[index];
    if(sub == "/pbrMetallicRoughness/baseColorFactor" && mat.pbrMetallicRoughness.baseColorFactor.size() >= 4)
    {
      const auto& c = mat.pbrMetallicRoughness.baseColorFactor;
      return InteractivityValue(glm::vec4(static_cast<float>(c[0]), static_cast<float>(c[1]), static_cast<float>(c[2]),
                                          static_cast<float>(c[3])));
    }
    if(sub == "/pbrMetallicRoughness/metallicFactor")
      return InteractivityValue(static_cast<float>(mat.pbrMetallicRoughness.metallicFactor));
    if(sub == "/pbrMetallicRoughness/roughnessFactor")
      return InteractivityValue(static_cast<float>(mat.pbrMetallicRoughness.roughnessFactor));
    if(sub == "/emissiveFactor" && mat.emissiveFactor.size() >= 3)
    {
      const auto& e = mat.emissiveFactor;
      return InteractivityValue(glm::vec3(static_cast<float>(e[0]), static_cast<float>(e[1]), static_cast<float>(e[2])));
    }
    if(sub == "/alphaCutoff")
      return InteractivityValue(static_cast<float>(mat.alphaCutoff));
    if(sub == "/doubleSided")
      return InteractivityValue(mat.doubleSided);
    if(sub == "/normalTexture/scale")
      return InteractivityValue(static_cast<float>(mat.normalTexture.scale));
    if(sub == "/occlusionTexture/strength")
      return InteractivityValue(static_cast<float>(mat.occlusionTexture.strength));

    // Generic fallback for anything not covered above: any material-level "/extensions/<NAME>/..."
    // property (including one nested inside e.g. a texture-info sub-object's own "extensions/
    // KHR_texture_transform/..."), or "/<textureSlot>/extensions/KHR_texture_transform/..." for
    // one of the material's own named texture slots.
    std::vector<std::string> segs = splitPath(sub);
    if(!segs.empty())
    {
      const tinygltf::Value* leaf = nullptr;
      if(segs[0] == "extensions")
        leaf = walkExtensionsPath(mat.extensions, segs, 0);
      else if(const tinygltf::ExtensionMap* texExt = namedTextureExtensions(mat, segs[0]);
              texExt && segs.size() > 1 && segs[1] == "extensions")
        leaf = walkExtensionsPath(*texExt, segs, 1);
      else if(segs.size() > 2 && segs[0] == "pbrMetallicRoughness" && segs[2] == "extensions")
      {
        if(const tinygltf::ExtensionMap* pbrTexExt = namedTextureExtensions(mat, segs[0] + "/" + segs[1]))
          leaf = walkExtensionsPath(*pbrTexExt, segs, 2);
      }
      if(leaf)
        return tinygltfValueToInteractivity(*leaf);
    }
    return std::nullopt;
  }

  static const std::string kLightPrefix = "/extensions/KHR_lights_punctual/lights/";
  if(concretePath.rfind(kLightPrefix, 0) == 0)
  {
    auto [index, sub] = parsePointerIndexAndRest(concretePath, kLightPrefix.size());
    if(index < 0 || index >= static_cast<int>(model.lights.size()))
      return std::nullopt;
    const tinygltf::Light& light = model.lights[index];
    if(sub == "/color" && light.color.size() >= 3)
      return InteractivityValue(glm::vec3(static_cast<float>(light.color[0]), static_cast<float>(light.color[1]),
                                          static_cast<float>(light.color[2])));
    if(sub == "/intensity")
      return InteractivityValue(static_cast<float>(light.intensity));
    if(sub == "/range")
      return InteractivityValue(static_cast<float>(light.range));
    if(sub == "/spot/innerConeAngle")
      return InteractivityValue(static_cast<float>(light.spot.innerConeAngle));
    if(sub == "/spot/outerConeAngle")
      return InteractivityValue(static_cast<float>(light.spot.outerConeAngle));
    return std::nullopt;
  }

  if(concretePath.rfind("/cameras/", 0) == 0)
  {
    auto [index, sub] = parsePointerIndexAndRest(concretePath, 9);
    if(index < 0 || index >= static_cast<int>(model.cameras.size()))
      return std::nullopt;
    const tinygltf::Camera& cam = model.cameras[index];
    if(cam.type != "perspective")
      return std::nullopt;
    if(sub == "/perspective/yfov")
      return InteractivityValue(static_cast<float>(cam.perspective.yfov));
    if(sub == "/perspective/znear")
      return InteractivityValue(static_cast<float>(cam.perspective.znear));
    if(sub == "/perspective/zfar")
      return InteractivityValue(static_cast<float>(cam.perspective.zfar));
    if(sub == "/perspective/aspectRatio")
      return InteractivityValue(static_cast<float>(cam.perspective.aspectRatio));
    return std::nullopt;
  }

  if(concretePath.rfind("/meshes/", 0) == 0)
  {
    auto [index, sub] = parsePointerIndexAndRest(concretePath, 8);
    if(index < 0 || index >= static_cast<int>(model.meshes.size()))
      return std::nullopt;
    const tinygltf::Mesh& mesh = model.meshes[index];
    if(sub == "/primitives.length")
      return InteractivityValue(static_cast<int32_t>(mesh.primitives.size()));
    if(sub.rfind("/primitives/", 0) == 0)
    {
      auto [primPos, rest] = parsePointerIndexAndRest(sub, 12);
      if(primPos < 0 || primPos >= static_cast<int>(mesh.primitives.size()))
        return std::nullopt;
      if(rest == "/material")
        return indexRef(mesh.primitives[primPos].material, "materials");
      return std::nullopt;
    }
    if(sub == "/weights.length")
      return InteractivityValue(static_cast<int32_t>(mesh.weights.size()));
    return std::nullopt;
  }

  if(concretePath.rfind("/animations/", 0) == 0)
  {
    auto [index, sub] = parsePointerIndexAndRest(concretePath, 12);
    if(index < 0 || index >= m_scene.animation().getNumAnimations())
      return std::nullopt;
    if(sub.empty())
      return indexRef(index, "animations");
    // KHR_interactivity's own asset-object-model addition (spec "Animation Object" section) - the
    // animation's own duration, needed since core glTF has no such property. Reuses the same
    // AnimationInfo::end already computed at load by AnimationSystem (keyframe max-time scan across
    // all of the animation's samplers), rather than re-deriving it here.
    if(sub == "/extensions/KHR_interactivity/maxTime")
      return InteractivityValue(m_scene.animation().getAnimationInfo(index).end);
    return std::nullopt;
  }

  if(concretePath.rfind("/skins/", 0) == 0)
  {
    auto [index, sub] = parsePointerIndexAndRest(concretePath, 7);
    if(index < 0 || index >= static_cast<int>(model.skins.size()))
      return std::nullopt;
    const tinygltf::Skin& skin = model.skins[index];
    if(sub == "/joints.length")
      return InteractivityValue(static_cast<int32_t>(skin.joints.size()));
    if(sub.rfind("/joints/", 0) == 0)
    {
      auto [jointPos, rest] = parsePointerIndexAndRest(sub, 8);
      if(!rest.empty() || jointPos < 0 || jointPos >= static_cast<int>(skin.joints.size()))
        return std::nullopt;
      return indexRef(skin.joints[jointPos], "nodes");
    }
    if(sub == "/skeleton")
      return indexRef(skin.skeleton, "nodes");
    return std::nullopt;
  }

  if(concretePath.rfind("/scenes/", 0) == 0)
  {
    auto [index, sub] = parsePointerIndexAndRest(concretePath, 8);
    if(index < 0 || index >= static_cast<int>(model.scenes.size()))
      return std::nullopt;
    const tinygltf::Scene& scene = model.scenes[index];
    if(sub == "/nodes.length")
      return InteractivityValue(static_cast<int32_t>(scene.nodes.size()));
    if(sub.rfind("/nodes/", 0) == 0)
    {
      auto [nodePos, rest] = parsePointerIndexAndRest(sub, 7);
      if(!rest.empty() || nodePos < 0 || nodePos >= static_cast<int>(scene.nodes.size()))
        return std::nullopt;
      return indexRef(scene.nodes[nodePos], "nodes");
    }
    return std::nullopt;
  }

  // Core read-only top-level array lengths and the default-scene pointer.
  if(concretePath == "/animations.length")
    return InteractivityValue(static_cast<int32_t>(model.animations.size()));
  if(concretePath == "/cameras.length")
    return InteractivityValue(static_cast<int32_t>(model.cameras.size()));
  if(concretePath == "/materials.length")
    return InteractivityValue(static_cast<int32_t>(model.materials.size()));
  if(concretePath == "/meshes.length")
    return InteractivityValue(static_cast<int32_t>(model.meshes.size()));
  if(concretePath == "/nodes.length")
    return InteractivityValue(static_cast<int32_t>(model.nodes.size()));
  if(concretePath == "/scenes.length")
    return InteractivityValue(static_cast<int32_t>(model.scenes.size()));
  if(concretePath == "/skins.length")
    return InteractivityValue(static_cast<int32_t>(model.skins.size()));
  if(concretePath == "/scene")
    return indexRef(model.defaultScene, "scenes");

  // KHR_interactivity's own self-description pointers (spec "Asset Capabilities" section) - asset
  // version, per-extension "enabled" (true only if BOTH listed in extensionsUsed AND actually
  // supported by this implementation - spec: "Each glTF extension that is listed in the glTF
  // asset's extensionsUsed array and supported by the implementation MUST create a corresponding
  // virtual object..."), and the implementation-defined concurrency limits. This engine doesn't
  // enforce those limits anywhere yet, so the values reported here are informational, not load-bearing.
  static const std::string kAssetPrefix = "/extensions/KHR_interactivity/asset/";
  if(concretePath.rfind(kAssetPrefix, 0) == 0)
  {
    const std::string sub = concretePath.substr(kAssetPrefix.size());
    if(sub == "majorVersion" || sub == "minorVersion")
    {
      const size_t dot = model.asset.version.find('.');
      if(dot == std::string::npos)
        return std::nullopt;
      const std::string part = sub == "majorVersion" ? model.asset.version.substr(0, dot) : model.asset.version.substr(dot + 1);
      int value = 0;
      if(std::from_chars(part.data(), part.data() + part.size(), value).ec != std::errc{})
        return std::nullopt;
      return InteractivityValue(static_cast<int32_t>(value));
    }
    static const std::string kExtPrefix = "extensions/";
    if(sub.rfind(kExtPrefix, 0) == 0 && sub.ends_with("/enabled"))
    {
      // Always a valid bool, even for a name nobody declared (confirmed against Khronos's own
      // conformance suite: querying a made-up extension's "enabled" expects isValid==true,
      // value==false - not "unresolvable" the way an out-of-range array index would be).
      const std::string extName = sub.substr(kExtPrefix.size(), sub.size() - kExtPrefix.size() - std::string("/enabled").size());
      const bool used =
          std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(), extName) != model.extensionsUsed.end();
      return InteractivityValue(used && m_scene.supportedExtensions().contains(extName));
    }
    return std::nullopt;
  }
  static const std::string kLimitsPrefix = "/extensions/KHR_interactivity/limits/";
  if(concretePath.rfind(kLimitsPrefix, 0) == 0)
  {
    const std::string sub = concretePath.substr(kLimitsPrefix.size());
    if(sub == "maxActiveAnimations" || sub == "maxActiveDelays" || sub == "maxActivePropertyInterpolations"
       || sub == "maxActiveVariableInterpolations")
      return InteractivityValue(int32_t{64});
    return std::nullopt;
  }

  return std::nullopt;
}

bool ScenePointerResolver::set(const std::string& concretePath, const InteractivityValue& value)
{
  AnimationPointerSystem& pointer = m_scene.animation().getAnimationPointer();

  bool applied = false;
  if(isBoolAnimationPointerPath(concretePath) && std::holds_alternative<bool>(value))
    applied = pointer.applyValue(concretePath, std::get<bool>(value) ? 1.0f : 0.0f);
  else if(std::holds_alternative<float>(value))
    applied = pointer.applyValue(concretePath, std::get<float>(value));
  else if(std::holds_alternative<glm::vec2>(value))
    applied = pointer.applyValue(concretePath, std::get<glm::vec2>(value));
  else if(std::holds_alternative<glm::vec3>(value))
    applied = pointer.applyValue(concretePath, std::get<glm::vec3>(value));
  else if(std::holds_alternative<glm::vec4>(value))
    applied = pointer.applyValue(concretePath, std::get<glm::vec4>(value));
  else
    return false;  // int / other bool / matrix / ref: not a supported writable target yet

  if(!applied)
    return false;

  // Flush immediately (not batched across the whole tick) so a pointer/get later in the same
  // tick sees this write - AnimationPointerSystem::syncToModel() only touches currently-dirty
  // resources, so this is cheap when (as usual) only one property changed.
  pointer.syncToModel();
  for(int nodeIndex : pointer.getDirtyNodes())
  {
    m_scene.markNodeDirty(nodeIndex);
    m_scene.editor().updateVisibility(nodeIndex);
  }
  for(int materialIndex : pointer.getDirtyMaterials())
    m_scene.markMaterialDirty(materialIndex);
  for(int lightIndex : pointer.getDirtyLights())
    m_scene.markLightDirty(lightIndex);
  pointer.clearDirty();
  return true;
}

}  // namespace nvvkgltf
