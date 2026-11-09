/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//--------------------------------------------------------------------------------------------------
// Viewport icons: draws the lights and cameras as wireframe glyphs over the viewport image.
//
// Every glyph is built in world space around the node, with its size derived from the node's distance
// so that it covers the same number of pixels anywhere (Painter::unitsPerPixel), and is then projected
// to the screen one segment at a time (Painter::segment), cut where it passes behind the camera.
//
//   Painter      collects the projected segments of one glyph, draws them with a dark halo, or measures
//                how far the mouse is from them.
//   Frame        the node's position and axes, and the world size of a pixel there.
//   add*()       one function per glyph: point, spot and directional light, camera.
//   visitGlyphs  builds the glyph of each light and camera to show and hands it to the caller, which
//                draws it (drawIcons) or tests it (pickIcon): one list of glyphs for both.
//
// The lights come from the scene's render lights, the active scene's lights with their world matrices
// already kept up to date; the cameras from the scene's usage index, those the active scene reaches. Both
// lists can be huge, so each is read only up to kMaxIconsPerKind entries per frame.
//--------------------------------------------------------------------------------------------------

#include "ui_viewport_icons.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <glm/gtc/constants.hpp>

#include "gltf_environment_sky.hpp"  // gltf_environment_sky::nodeLightIndex
#include "gltf_scene.hpp"

namespace ui::viewport_icons {

namespace {

// Clip-space w below which a point counts as behind the camera.
constexpr float kMinClipW = 1e-3F;

// Glyph sizes in pixels at 100% DPI scale; the painter multiplies them by the window's DPI scale.
constexpr float kDiscRadius   = 7.0F;   // the disc every light glyph is built around
constexpr float kRingRadius   = 13.0F;  // point light: the ring around the disc
constexpr float kRayInner     = 10.0F;  // directional light: rays start here...
constexpr float kRayOuter     = 15.0F;  // ...and end here
constexpr float kArrowLength  = 70.0F;  // directional light: the arrow along the light's direction
constexpr float kConeLength   = 60.0F;  // spot light: the cone's length
constexpr float kCameraLength = 60.0F;  // camera: the frustum's depth
constexpr float kLineWidth    = 1.5F;
constexpr float kHaloWidth    = 3.5F;  // the dark outline under the lines, so they read on any background

constexpr int kRayCount    = 8;
constexpr int kCircleSides = 24;
constexpr int kConeEdges   = 4;  // spot cone: edges drawn from the apex to the base circle

constexpr float kMaxHalfAngle  = 1.2F;          // radians; keeps a very wide cone or field of view a drawable size
constexpr float kDefaultAspect = 16.0F / 9.0F;  // glTF leaves a perspective camera's aspect ratio optional

constexpr size_t kMaxIconsPerKind = 256;     // lights, and cameras, considered per frame
constexpr float  kCullMargin      = 200.0F;  // pixels outside the image within which a glyph can still reach in
constexpr float  kPickDistance    = 6.0F;    // pixels from a glyph's lines at which a click picks it

// How a glyph is drawn: the selected one stands out, the others stay quiet.
struct Style
{
  ImU32 line      = 0;
  ImU32 halo      = 0;  // the dark outline under the lines, so they read on any background
  float lineWidth = 1.0F;
  float haloWidth = 1.0F;
};
constexpr Style kSelectedStyle{IM_COL32(255, 160, 40, 255), IM_COL32(0, 0, 0, 150), 1.5F, 3.5F};
constexpr Style kOtherStyle{IM_COL32(225, 225, 225, 210), IM_COL32(0, 0, 0, 130), 1.25F, 3.0F};

// Distance from point `p` to the segment a-b.
float distanceToSegment(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b)
{
  const glm::vec2 ab = b - a;
  const float     t  = std::clamp(glm::dot(p - a, ab) / std::max(glm::dot(ab, ab), 1e-6F), 0.0F, 1.0F);
  return glm::distance(p, a + ab * t);
}

// The screen-space pieces of one glyph, collected while it is built: line segments, and the discs filled
// with a light's color. flush() draws them, distanceTo() measures them, clear() starts the next glyph.
class Painter
{
public:
  // Starts an empty glyph for `view`; `dpiScale` is the window's DPI scale.
  Painter(const View& view, float dpiScale)
      : m_view(view)
      , m_viewProj(view.projection * view.view)
      , m_dpi(dpiScale)
  {
    m_segments.reserve(128);
  }

  // The directions the viewer's right and up point to in world space: a circle built from them faces the viewer.
  glm::vec3 viewRight() const { return {m_view.view[0][0], m_view.view[1][0], m_view.view[2][0]}; }
  glm::vec3 viewUp() const { return {m_view.view[0][1], m_view.view[1][1], m_view.view[2][1]}; }

  // World units covered by one screen pixel at `p`, or a negative value when `p` is behind the camera.
  // One formula for both projections: a perspective w is the view depth, an orthographic one is 1.
  float unitsPerPixel(const glm::vec3& p) const
  {
    const float w = (m_viewProj * glm::vec4(p, 1.0F)).w;
    if(w <= kMinClipW)
      return -1.0F;
    return 2.0F * w / (std::abs(m_view.projection[1][1]) * m_view.imageSize.y);
  }

  // The window's DPI scale, which the pixel sizes of the glyphs are multiplied by.
  float dpi() const { return m_dpi; }

  // Adds the segment a-b, cut where it passes behind the camera.
  void segment(const glm::vec3& a, const glm::vec3& b)
  {
    glm::vec4 ca = m_viewProj * glm::vec4(a, 1.0F);
    glm::vec4 cb = m_viewProj * glm::vec4(b, 1.0F);
    if(ca.w <= kMinClipW && cb.w <= kMinClipW)
      return;
    // Clip coordinates interpolate linearly, so cutting at the w threshold is exact.
    if(ca.w <= kMinClipW)
      ca = glm::mix(ca, cb, (kMinClipW - ca.w) / (cb.w - ca.w));
    else if(cb.w <= kMinClipW)
      cb = glm::mix(cb, ca, (kMinClipW - cb.w) / (ca.w - cb.w));

    const glm::vec2 pa = toScreen(ca);
    const glm::vec2 pb = toScreen(cb);
    if(glm::dot(pb - pa, pb - pa) < 0.25F)  // under half a pixel: nothing to see
      return;
    m_segments.push_back({pa, pb});
  }

  // Adds a circle of `radius` around `center`, in the plane spanned by the unit vectors `u` and `v`.
  void circle(const glm::vec3& center, const glm::vec3& u, const glm::vec3& v, float radius)
  {
    glm::vec3 previous = center + u * radius;
    for(int i = 1; i <= kCircleSides; ++i)
    {
      const float     angle = glm::two_pi<float>() * float(i) / float(kCircleSides);
      const glm::vec3 next  = center + (u * std::cos(angle) + v * std::sin(angle)) * radius;
      segment(previous, next);
      previous = next;
    }
  }

  // Adds the disc of a light at `center`, filled with the light's color (glTF: linear RGB, 1 when absent).
  void fillDisc(const glm::vec3& center, const std::vector<double>& color)
  {
    glm::vec2 screen;
    if(!project(center, screen))
      return;
    const auto channel = [&color](size_t i) {
      return color.size() > i ? std::clamp(static_cast<float>(color[i]), 0.0F, 1.0F) : 1.0F;
    };
    m_discs.push_back({screen, ImGui::ColorConvertFloat4ToU32({channel(0), channel(1), channel(2), 1.0F})});
  }

  // Draws everything collected in `style`: the discs, the halo under the lines, then the lines. The halo
  // goes first for every segment, so it never covers a neighbouring line.
  void flush(ImDrawList* drawList, const Style& style) const
  {
    if(drawList == nullptr)
      return;
    for(size_t i = 0; i < m_discs.size(); ++i)
      drawList->AddCircleFilled({m_discs[i].center.x, m_discs[i].center.y}, discRadius(), m_discs[i].color);
    for(size_t i = 0; i < m_segments.size(); ++i)
      drawList->AddLine({m_segments[i].a.x, m_segments[i].a.y}, {m_segments[i].b.x, m_segments[i].b.y}, style.halo,
                        style.haloWidth * m_dpi);
    for(size_t i = 0; i < m_segments.size(); ++i)
      drawList->AddLine({m_segments[i].a.x, m_segments[i].a.y}, {m_segments[i].b.x, m_segments[i].b.y}, style.line,
                        style.lineWidth * m_dpi);
  }

  // Distance in pixels from `p` to the collected glyph: to its nearest line, or to its disc's edge, so a
  // point inside a disc is at distance 0. Large when the glyph is empty.
  float distanceTo(const glm::vec2& p) const
  {
    float best = std::numeric_limits<float>::max();
    for(size_t i = 0; i < m_segments.size(); ++i)
      best = std::min(best, distanceToSegment(p, m_segments[i].a, m_segments[i].b));
    for(size_t i = 0; i < m_discs.size(); ++i)
      best = std::min(best, std::max(0.0F, glm::distance(p, m_discs[i].center) - discRadius()));
    return best;
  }

  // Forgets the collected glyph.
  void clear()
  {
    m_segments.clear();
    m_discs.clear();
  }

  // True when world point `p` is in front of the camera and not further than the cull margin outside the
  // image: a glyph anchored there can reach into the image.
  bool isNearImage(const glm::vec3& p) const
  {
    glm::vec2 s;
    if(!project(p, s))
      return false;
    const glm::vec2 margin(kCullMargin * m_dpi);
    return glm::all(glm::greaterThanEqual(s, m_view.imageOrigin - margin))
           && glm::all(glm::lessThanEqual(s, m_view.imageOrigin + m_view.imageSize + margin));
  }

private:
  struct Segment
  {
    glm::vec2 a = glm::vec2(0.0F);
    glm::vec2 b = glm::vec2(0.0F);
  };
  struct Disc
  {
    glm::vec2 center = glm::vec2(0.0F);
    ImU32     color  = 0;
  };

  // The ImGui screen position of a world point, or false when it is behind the camera.
  bool project(const glm::vec3& p, glm::vec2& screen) const
  {
    const glm::vec4 clip = m_viewProj * glm::vec4(p, 1.0F);
    if(clip.w <= kMinClipW)
      return false;
    screen = toScreen(clip);
    return true;
  }

  // The radius in pixels of the colored disc, a little inside the ring that outlines it.
  float discRadius() const { return (kDiscRadius - 2.0F) * m_dpi; }

  // Clip space to ImGui screen pixels. The projection already carries Vulkan's flipped Y, so +Y is down.
  glm::vec2 toScreen(const glm::vec4& clip) const
  {
    const glm::vec2 ndc = glm::vec2(clip) / clip.w;
    return m_view.imageOrigin + (ndc * 0.5F + 0.5F) * m_view.imageSize;
  }

  View                 m_view;
  glm::mat4            m_viewProj = glm::mat4(1.0F);
  float                m_dpi      = 1.0F;
  std::vector<Segment> m_segments;
  std::vector<Disc>    m_discs;
};

// A glTF light or camera node's placement: its position and the axes it looks along. glTF lights and
// cameras look down -Z with +Y up. `pixel` is the world size of one screen pixel at the node, so a glyph
// sized in pixels is `pixel` times larger in world units.
struct Frame
{
  glm::vec3 position = glm::vec3(0.0F);
  glm::vec3 right    = glm::vec3(1.0F, 0.0F, 0.0F);
  glm::vec3 up       = glm::vec3(0.0F, 1.0F, 0.0F);
  glm::vec3 forward  = glm::vec3(0.0F, 0.0F, -1.0F);
  float     pixel    = 0.0F;
};

// The Frame of a node's world matrix, without its scale. False when the node is behind the camera or too
// far outside the image to show, or its matrix has no usable axes.
bool makeFrame(const Painter& painter, const glm::mat4& world, Frame& frame)
{
  const glm::vec3 x = glm::vec3(world[0]);
  const glm::vec3 y = glm::vec3(world[1]);
  const glm::vec3 z = glm::vec3(world[2]);
  if(glm::dot(x, x) < 1e-12F || glm::dot(y, y) < 1e-12F || glm::dot(z, z) < 1e-12F)
    return false;

  frame.position = glm::vec3(world[3]);
  frame.right    = glm::normalize(x);
  frame.up       = glm::normalize(y);
  frame.forward  = -glm::normalize(z);
  frame.pixel    = painter.unitsPerPixel(frame.position) * painter.dpi();
  return frame.pixel > 0.0F && painter.isNearImage(frame.position);
}

// The disc every light glyph is built around, facing the viewer. Filled with the light's color.
void addDisc(Painter& painter, const Frame& frame, const std::vector<double>& color)
{
  painter.circle(frame.position, painter.viewRight(), painter.viewUp(), kDiscRadius * frame.pixel);
  painter.fillDisc(frame.position, color);
}

// Point light: the disc inside a ring.
void addPointLight(Painter& painter, const Frame& frame)
{
  painter.circle(frame.position, painter.viewRight(), painter.viewUp(), kRingRadius * frame.pixel);
}

// Spot light: a cone from the light, opening by the light's real outer cone angle.
void addSpotLight(Painter& painter, const Frame& frame, const tinygltf::Light& light)
{
  const float     length = kConeLength * frame.pixel;
  const float     radius = length * std::tan(std::min(static_cast<float>(light.spot.outerConeAngle), kMaxHalfAngle));
  const glm::vec3 base   = frame.position + frame.forward * length;
  painter.circle(base, frame.right, frame.up, radius);
  for(int i = 0; i < kConeEdges; ++i)
  {
    const float     angle = glm::two_pi<float>() * float(i) / float(kConeEdges);
    const glm::vec3 edge  = base + (frame.right * std::cos(angle) + frame.up * std::sin(angle)) * radius;
    painter.segment(frame.position, edge);
  }
}

// Directional light: rays around the disc, and an arrow along the direction the light travels.
void addDirectionalLight(Painter& painter, const Frame& frame)
{
  const glm::vec3 right = painter.viewRight();
  const glm::vec3 up    = painter.viewUp();
  for(int i = 0; i < kRayCount; ++i)
  {
    const float     angle = glm::two_pi<float>() * float(i) / float(kRayCount);
    const glm::vec3 dir   = right * std::cos(angle) + up * std::sin(angle);
    painter.segment(frame.position + dir * kRayInner * frame.pixel, frame.position + dir * kRayOuter * frame.pixel);
  }

  const float     length = kArrowLength * frame.pixel;
  const glm::vec3 tip    = frame.position + frame.forward * length;
  const glm::vec3 back   = frame.forward * (-0.2F * length);
  const float     wing   = 0.1F * length;
  painter.segment(frame.position + frame.forward * (kRayOuter * frame.pixel), tip);
  painter.segment(tip, tip + back + frame.right * wing);
  painter.segment(tip, tip + back - frame.right * wing);
  painter.segment(tip, tip + back + frame.up * wing);
  painter.segment(tip, tip + back - frame.up * wing);
}

// Draws the glyph of the light type. Unknown types get the disc alone.
void addLight(Painter& painter, const Frame& frame, const tinygltf::Light& light)
{
  addDisc(painter, frame, light.color);
  if(light.type == "point")
    addPointLight(painter, frame);
  else if(light.type == "spot")
    addSpotLight(painter, frame, light);
  else if(light.type == "directional")
    addDirectionalLight(painter, frame);
}

// Camera: the frustum the camera sees, closed by a rectangle, with a triangle above it for "up". A
// perspective camera widens from its position by its field of view; an orthographic one keeps its
// section, shown at a fixed size since its real size has no screen meaning.
void addCamera(Painter& painter, const Frame& frame, const tinygltf::Camera& camera)
{
  const float depth     = kCameraLength * frame.pixel;
  float       nearHalfW = 0.0F, nearHalfH = 0.0F, farHalfW = 0.0F, farHalfH = 0.0F;
  if(camera.type == "orthographic")
  {
    const double ratio = camera.orthographic.ymag != 0.0 ? std::abs(camera.orthographic.xmag / camera.orthographic.ymag) : 1.0;
    farHalfH  = 0.3F * depth;
    farHalfW  = farHalfH * static_cast<float>(ratio);
    nearHalfW = farHalfW;
    nearHalfH = farHalfH;
  }
  else
  {
    const float aspect = camera.perspective.aspectRatio > 0.0 ? static_cast<float>(camera.perspective.aspectRatio) : kDefaultAspect;
    farHalfH = depth * std::tan(std::min(0.5F * static_cast<float>(camera.perspective.yfov), kMaxHalfAngle));
    farHalfW = farHalfH * aspect;
  }

  const auto corner = [&frame](float distance, float x, float y) {
    return frame.position + frame.forward * distance + frame.right * x + frame.up * y;
  };
  const std::array<glm::vec3, 4> nearCorners = {corner(0, -nearHalfW, -nearHalfH), corner(0, nearHalfW, -nearHalfH),
                                                corner(0, nearHalfW, nearHalfH), corner(0, -nearHalfW, nearHalfH)};
  const std::array<glm::vec3, 4> farCorners  = {corner(depth, -farHalfW, -farHalfH), corner(depth, farHalfW, -farHalfH),
                                                corner(depth, farHalfW, farHalfH), corner(depth, -farHalfW, farHalfH)};
  for(size_t i = 0; i < 4; ++i)
  {
    const size_t next = (i + 1) % 4;
    painter.segment(nearCorners[i], nearCorners[next]);
    painter.segment(farCorners[i], farCorners[next]);
    painter.segment(nearCorners[i], farCorners[i]);
  }

  // The "up" triangle, above the far rectangle's top edge.
  const glm::vec3 apex = corner(depth, 0.0F, 1.5F * farHalfH);
  painter.segment(corner(depth, -0.4F * farHalfW, farHalfH), apex);
  painter.segment(corner(depth, 0.4F * farHalfW, farHalfH), apex);
}

// The light and the camera a node carries, each -1 when it carries none or the index is out of range.
struct NodeIcons
{
  int light  = -1;
  int camera = -1;
};

NodeIcons findNodeIcons(const tinygltf::Model& model, int node)
{
  NodeIcons icons;
  if(node < 0 || node >= static_cast<int>(model.nodes.size()))
    return icons;
  const tinygltf::Node& gltfNode = model.nodes[static_cast<size_t>(node)];
  const int             light    = gltf_environment_sky::nodeLightIndex(gltfNode);
  if(light >= 0 && light < static_cast<int>(model.lights.size()))
    icons.light = light;
  if(gltfNode.camera >= 0 && gltfNode.camera < static_cast<int>(model.cameras.size()))
    icons.camera = gltfNode.camera;
  return icons;
}

// Builds the glyph of each light and camera to show, one at a time in `painter`, and calls
// visit(node, isSelected) for each: all of them when `request.showAll`, then the selected node's, last so
// that it ends up on top. `visit` draws or measures what the painter holds. A glyph that cannot be placed
// (behind the camera, far outside the image) is skipped. `request.scene` must not be null.
template <class Visit>
void visitGlyphs(Painter& painter, const Request& request, Visit&& visit)
{
  const nvvkgltf::Scene& scene = *request.scene;
  const tinygltf::Model& model = scene.getModel();
  auto show = [&painter, &model, &visit](int node, bool isSelected, int light, int camera, const glm::mat4& world) {
    painter.clear();
    Frame frame;
    if(!makeFrame(painter, world, frame))
      return;
    if(light >= 0)
      addLight(painter, frame, model.lights[static_cast<size_t>(light)]);
    if(camera >= 0)
      addCamera(painter, frame, model.cameras[static_cast<size_t>(camera)]);
    visit(node, isSelected);
  };

  if(request.showAll)
  {
    const std::vector<nvvkgltf::RenderLight>& lights = scene.getRenderLights();
    for(size_t i = 0; i < std::min(lights.size(), kMaxIconsPerKind); ++i)
    {
      const nvvkgltf::RenderLight& light = lights[i];
      if(light.nodeID != request.selectedNode && light.light >= 0 && light.light < static_cast<int>(model.lights.size()))
        show(light.nodeID, false, light.light, -1, light.worldMatrix);
    }
    const std::span<const int> cameraNodes = scene.getActiveCameraNodes();
    for(size_t i = 0; i < std::min(cameraNodes.size(), kMaxIconsPerKind); ++i)
    {
      const int node = cameraNodes[i];
      if(node != request.selectedNode)
        show(node, false, -1, model.nodes[static_cast<size_t>(node)].camera, scene.computeNodeWorldMatrix(node));
    }
  }

  const NodeIcons selected = findNodeIcons(model, request.selectedNode);
  if(selected.light >= 0 || selected.camera >= 0)
    show(request.selectedNode, true, selected.light, selected.camera, scene.computeNodeWorldMatrix(request.selectedNode));
}

}  // namespace

void drawIcons(ImDrawList* drawList, const Request& request)
{
  if(drawList == nullptr || request.scene == nullptr)
    return;
  Painter painter(request.view, ImGui::GetWindowDpiScale());
  visitGlyphs(painter, request, [&painter, drawList](int, bool isSelected) {
    painter.flush(drawList, isSelected ? kSelectedStyle : kOtherStyle);
  });
}

int pickIcon(const Request& request, const glm::vec2& mousePos)
{
  if(request.scene == nullptr)
    return -1;
  Painter painter(request.view, ImGui::GetWindowDpiScale());
  int     picked  = -1;
  float   nearest = kPickDistance * painter.dpi();
  visitGlyphs(painter, request, [&painter, &picked, &nearest, &mousePos](int node, bool) {
    // The selected glyph comes last, so on a tie it wins.
    const float distance = painter.distanceTo(mousePos);
    if(distance <= nearest)
    {
      nearest = distance;
      picked  = node;
    }
  });
  return picked;
}

}  // namespace ui::viewport_icons
