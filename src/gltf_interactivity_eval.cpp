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
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <unordered_set>

#include <glm/glm.hpp>
#include <nvutils/logger.hpp>

#include "gltf_interactivity_animation.hpp"
#include "gltf_interactivity_eval.hpp"
#include "gltf_interactivity_instance.hpp"
#include "gltf_interactivity_pointer.hpp"

namespace nvvkgltf {

namespace {

// Logs once per (graph, op) the first time an unimplemented-but-recognized op is hit, instead of
// spamming every tick. Phase B/C/D fill in the rest of the catalog (see docs/interactivity.md).
void warnUnimplementedOnce(InteractivityOp op)
{
  static std::unordered_set<InteractivityOp> warned;
  if(warned.insert(op).second)
    LOGW("KHR_interactivity: op '%s' is recognized but not yet implemented (no-op)\n", interactivityOpString(op).c_str());
}

const InteractivityNode& nodeAt(InteractivityGraphInstance& instance, int nodeIndex)
{
  return instance.graph().nodes()[nodeIndex];
}

InteractivityOp opOf(InteractivityGraphInstance& instance, const InteractivityNode& node)
{
  return instance.graph().declarations()[node.declarationIndex].op;
}

// True if `v` holds one of the scalar alternatives applyUnary/applyBinary/applyComponentwise
// bottom out at once they've decomposed a vecN/matNxN down to individual components.
template <typename T>
constexpr bool kIsScalar = std::is_same_v<T, float> || std::is_same_v<T, int32_t>;

template <typename T>
constexpr bool kIsMatrix = std::is_same_v<T, glm::mat2> || std::is_same_v<T, glm::mat3> || std::is_same_v<T, glm::mat4>;

// True if `v` holds one of the arithmetic-capable scalar/vector/matrix alternatives (float, int,
// floatN, or floatNxN) most math/* ops accept per spec (e.g. math/neg, math/eq, math/clamp).
template <typename T>
constexpr bool kIsArithmetic = std::is_same_v<T, float> || std::is_same_v<T, int32_t> || std::is_same_v<T, glm::vec2>
                               || std::is_same_v<T, glm::vec3> || std::is_same_v<T, glm::vec4> || kIsMatrix<T>;

// float/vecN only, deliberately NOT including matrices or int - for the handful of ops the spec
// restricts to floatN with no floatNxN row (math/normalize, math/length, math/smoothStep).
template <typename T>
constexpr bool kIsFloatArithmetic =
    std::is_same_v<T, float> || std::is_same_v<T, glm::vec2> || std::is_same_v<T, glm::vec3> || std::is_same_v<T, glm::vec4>;

// float/vecN/matNxN, no int - the "floatN or floatNxN" family most unary float ops use per spec
// (abs, sign, floor, ceil, round, fract, trunc, saturate) - broader than kIsFloatArithmetic (adds
// matrices) but narrower than kIsArithmetic (excludes int, which these ops' spec text never lists).
template <typename T>
constexpr bool kIsFloatOrMatrixArithmetic = kIsFloatArithmetic<T> || kIsMatrix<T>;

// Applies `f` (a plain scalar function) to every component of `v`, all the way down to individual
// floats/ints - needed because glm's free functions (floor, min, ...) don't accept matrix
// arguments the way this spec still wants componentwise semantics for them (a mat4's "component"
// is a vec4 column, whose own components are floats, hence the two-level recursion for matrices).
template <typename T, typename F>
T applyComponentwise(const T& v, F&& f)
{
  if constexpr(kIsScalar<T>)
    return f(v);
  else
  {
    T out{};
    for(glm::length_t i = 0; i < T::length(); ++i)
      out[i] = applyComponentwise(v[i], f);
    return out;
  }
}

template <typename T, typename F>
T applyComponentwiseBinary(const T& a, const T& b, F&& f)
{
  if constexpr(kIsScalar<T>)
    return f(a, b);
  else
  {
    T out{};
    for(glm::length_t i = 0; i < T::length(); ++i)
      out[i] = applyComponentwiseBinary(a[i], b[i], f);
    return out;
  }
}

// Ternary sibling for math/clamp and math/mix (both take a/b/c of the same type).
template <typename T, typename F>
T applyComponentwiseTernary(const T& a, const T& b, const T& c, F&& f)
{
  if constexpr(kIsScalar<T>)
    return f(a, b, c);
  else
  {
    T out{};
    for(glm::length_t i = 0; i < T::length(); ++i)
      out[i] = applyComponentwiseTernary(a[i], b[i], c[i], f);
    return out;
  }
}

// Applies `f` to a single arithmetic-typed value (spec math/abs, math/neg, math/floor, ...).
// `FloatOnly` gates ops that GLSL/the spec only define for floating-point types (floor/ceil/...).
template <bool FloatOnly, typename F>
InteractivityValue applyUnary(const InteractivityValue& a, F&& f)
{
  return std::visit(
      [&](auto&& av) -> InteractivityValue {
        using T = std::decay_t<decltype(av)>;
        if constexpr((FloatOnly && kIsFloatOrMatrixArithmetic<T>) || (!FloatOnly && kIsArithmetic<T>))
          return InteractivityValue(applyComponentwise(av, f));
        else
          return std::monostate{};
      },
      a);
}

template <bool FloatOnly, typename F>
InteractivityValue applyBinary(const InteractivityValue& a, const InteractivityValue& b, F&& f)
{
  if(a.index() != b.index())
    return std::monostate{};  // spec: no implicit coercion, mismatched operand types are invalid
  return std::visit(
      [&](auto&& av) -> InteractivityValue {
        using T = std::decay_t<decltype(av)>;
        if constexpr((FloatOnly && kIsFloatOrMatrixArithmetic<T>) || (!FloatOnly && kIsArithmetic<T>))
          return InteractivityValue(applyComponentwiseBinary(av, std::get<T>(b), f));
        else
          return std::monostate{};
      },
      a);
}

// Comparison ops always produce a bool, regardless of the (matching) operand type. math/eq is
// defined for bool too (spec 2922, same op id as the arithmetic-type overloads - see 5227's
// overload-by-socket-type rule), so this also accepts bool, unlike applyUnary/applyBinary.
template <typename F>
InteractivityValue applyComparison(const InteractivityValue& a, const InteractivityValue& b, F&& f)
{
  if(a.index() != b.index())
    return std::monostate{};
  return std::visit(
      [&](auto&& av) -> InteractivityValue {
        using T = std::decay_t<decltype(av)>;
        if constexpr(kIsArithmetic<T> || std::is_same_v<T, bool>)
          return InteractivityValue(f(av, std::get<T>(b)));
        else
          return std::monostate{};
      },
      a);
}

// Quaternions are represented as float4 = (x,y,z,w) with w the scalar/real part (spec 3860, same
// convention as a glTF node's `rotation`) - never glm::quat, whose (w,x,y,z) constructor-argument
// order is a frequent source of silent bugs when mixed with vec4-as-quaternion code.
glm::vec4 quatMulRaw(const glm::vec4& a, const glm::vec4& b)
{
  return glm::vec4(a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
                   a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

// Shared by math/quatSlerp and lerpInteractivityValue's slerp path (variable/interpolate,
// pointer/interpolate). Same x,y,z,w = vec4 convention as quatMulRaw.
glm::vec4 quatSlerpRaw(glm::vec4 qa, glm::vec4 qb, float t)
{
  float d = glm::dot(qa, qb);
  if(d < 0.0f)
  {
    d  = -d;
    qb = -qb;
  }
  float ka, kb;
  if(d > 1.0f - 1e-6f)
  {
    ka = 1.0f - t;
    kb = t;
  }
  else
  {
    const float omega    = std::acos(d);
    const float sinOmega = std::sin(omega);
    ka                   = std::sin(omega * (1.0f - t)) / sinOmega;
    kb                   = std::sin(omega * t) / sinOmega;
  }
  return qa * ka + qb * kb;
}

// variable/interpolate + pointer/interpolate (spec: "any component of p1/p2 is NaN or infinite, or
// the first component is negative or greater than 1" is an err condition, checked before starting
// the interpolation).
bool isValidBezierControlPoint(const glm::vec2& p)
{
  return !std::isnan(p.x) && !std::isnan(p.y) && !std::isinf(p.x) && !std::isinf(p.y) && p.x >= 0.0f && p.x <= 1.0f;
}

InteractivityValue evaluatePureMath(InteractivityGraphInstance& instance, const InteractivityNode& node, InteractivityOp op)
{
  const InteractivityValue a = instance.evaluateInput(node, "a");
  switch(op)
  {
    // Unary
    case InteractivityOp::eMathAbs:
      return applyUnary<false>(a, [](auto v) { return glm::abs(v); });
    case InteractivityOp::eMathNeg:
      return applyUnary<false>(a, [](auto v) { return -v; });
    case InteractivityOp::eMathSign:
      // glm::sign is comparison-based ((v>0)-(v<0)-ish), which silently gives 0 for NaN instead of
      // propagating it (spec's blanket "Arithmetic Operations" rule: any NaN component in -> NaN
      // component out) - applyUnary decomposes to scalar, so v is always float or int32_t here.
      return applyUnary<false>(a, [](auto v) {
        if constexpr(std::is_same_v<std::decay_t<decltype(v)>, float>)
          return std::isnan(v) ? v : glm::sign(v);
        else
          return glm::sign(v);
      });
    case InteractivityOp::eMathFloor:
      return applyUnary<true>(a, [](auto v) { return glm::floor(v); });
    case InteractivityOp::eMathCeil:
      return applyUnary<true>(a, [](auto v) { return glm::ceil(v); });
    case InteractivityOp::eMathRound:
      // Spec: half away from zero (glm::round is round-half-to-even on some platforms), so implement
      // explicitly. applyUnary always decomposes down to scalar float before calling this lambda.
      return applyUnary<true>(a, [](float v) { return v < 0.0f ? -std::floor(-v + 0.5f) : std::floor(v + 0.5f); });
    case InteractivityOp::eMathFract:
      return applyUnary<true>(a, [](auto v) { return glm::fract(v); });
    case InteractivityOp::eMathTrunc:
      // Spec: "if the argument is infinity, it is returned unchanged" - std::trunc already does
      // this natively (IEEE-754 trunc(inf) == inf).
      return applyUnary<true>(a, [](float v) { return std::trunc(v); });
    case InteractivityOp::eMathSaturate:
      return applyUnary<true>(a, [](auto v) { return glm::clamp(v, decltype(v)(0), decltype(v)(1)); });

    // Binary
    case InteractivityOp::eMathAdd:
      return applyBinary<false>(a, instance.evaluateInput(node, "b"), [](auto x, auto y) { return x + y; });
    case InteractivityOp::eMathSub:
      return applyBinary<false>(a, instance.evaluateInput(node, "b"), [](auto x, auto y) { return x - y; });
    case InteractivityOp::eMathMul:
      return applyBinary<false>(a, instance.evaluateInput(node, "b"), [](auto x, auto y) { return x * y; });
    // math/div and math/rem (spec: int-only, no float/vecN overload - unlike add/sub/mul/min/max
    // above, which apply generically via kIsArithmetic) need explicit zero and INT_MIN/-1 guards:
    // both are hardware traps (SIGFPE via the x86 `idiv` instruction) if computed with plain `/`/`%`,
    // not just UB-on-paper like signed add/sub/mul overflow (which merely relies on the platform's
    // actual, universally two's-complement behavior and never traps). Spec mandates b==0 -> 0 and
    // INT_MIN/-1 wraps to INT_MIN (the same overflow-wrap convention as math/mul).
    // Both are overloaded per spec: an int-specific overload (piecewise-safe, above) AND a generic
    // floatN/floatNxN overload with its own, different edge-case semantics (plain IEEE-754 division
    // for div; a 3-way NaN/Infinity/trunc piecewise formula for rem) - dispatch on operand type.
    case InteractivityOp::eMathDiv: {
      const InteractivityValue b = instance.evaluateInput(node, "b");
      if(std::holds_alternative<int32_t>(a) && std::holds_alternative<int32_t>(b))
      {
        const int32_t ai = std::get<int32_t>(a);
        const int32_t bi = std::get<int32_t>(b);
        if(bi == 0)
          return int32_t(0);
        if(ai == std::numeric_limits<int32_t>::min() && bi == -1)
          return ai;
        return static_cast<int32_t>(ai / bi);
      }
      // Generic overload: plain a/b - IEEE-754 already gives +-Infinity/NaN for the b==0 cases
      // spec calls out, no special-casing needed.
      return applyBinary<true>(a, b, [](auto x, auto y) { return x / y; });
    }
    case InteractivityOp::eMathRem: {
      const InteractivityValue b = instance.evaluateInput(node, "b");
      if(std::holds_alternative<int32_t>(a) && std::holds_alternative<int32_t>(b))
      {
        const int32_t ai = std::get<int32_t>(a);
        const int32_t bi = std::get<int32_t>(b);
        if(bi == 0)
          return int32_t(0);
        if(ai == std::numeric_limits<int32_t>::min() && bi == -1)
          return int32_t(0);  // a - b*trunc(a/b), trunc(a/b) wrapped to INT_MIN per eMathDiv above
        return static_cast<int32_t>(ai - bi * (ai / bi));
      }
      // Generic overload's own piecewise formula (spec): NaN if a=+-Inf or b=+-0; a if a is finite
      // and b=+-Inf; otherwise a - b*trunc(a/b).
      return applyBinary<true>(a, b, [](float x, float y) {
        if(std::isinf(x) || y == 0.0f)
          return std::numeric_limits<float>::quiet_NaN();
        if(std::isinf(y))
          return x;
        return x - y * std::trunc(x / y);
      });
    }
    case InteractivityOp::eMathMin:
      return applyBinary<false>(a, instance.evaluateInput(node, "b"), [](auto x, auto y) { return glm::min(x, y); });
    case InteractivityOp::eMathMax:
      return applyBinary<false>(a, instance.evaluateInput(node, "b"), [](auto x, auto y) { return glm::max(x, y); });

    case InteractivityOp::eMathClamp: {
      // Spec formula is min(max(a, min(b,c)), max(b,c)) - symmetric in b/c, unlike glm::clamp
      // (which assumes lo<=hi), so boundaries given in either order produce the spec-defined result.
      // Decomposed to scalar (applyComponentwiseTernary) rather than calling glm::min/max on the
      // whole T directly, since glm doesn't define those for matrix arguments the way this spec's
      // floatNxN overload still wants componentwise semantics for them.
      const InteractivityValue b = instance.evaluateInput(node, "b");
      const InteractivityValue c = instance.evaluateInput(node, "c");
      if(a.index() != b.index() || a.index() != c.index())
        return std::monostate{};
      return std::visit(
          [&](auto&& av) -> InteractivityValue {
            using T = std::decay_t<decltype(av)>;
            if constexpr(kIsArithmetic<T>)
              return InteractivityValue(applyComponentwiseTernary(av, std::get<T>(b), std::get<T>(c), [](auto x, auto y, auto z) {
                const auto lo = std::min(y, z);
                const auto hi = std::max(y, z);
                return std::min(std::max(x, lo), hi);
              }));
            else
              return std::monostate{};
          },
          a);
    }
    case InteractivityOp::eMathMix: {
      // Spec socket ids are a (t=0 value), b (t=1 value), c (unclamped interpolation coefficient) -
      // not "t". c matches a/b's type exactly (spec: "floatN c or floatNxN c", the same N as a/b) -
      // NOT forced to a bare scalar, so a vecN/matNxN mix can have a per-component coefficient.
      // Decomposed to scalar rather than calling glm::mix on the whole T: for matrices, `c * b`
      // would otherwise invoke glm's matrix-multiply operator instead of the per-element multiply
      // the spec's componentwise formula actually wants.
      const InteractivityValue b = instance.evaluateInput(node, "b");
      const InteractivityValue c = instance.evaluateInput(node, "c");
      if(a.index() != b.index() || a.index() != c.index())
        return std::monostate{};
      return std::visit(
          [&](auto&& av) -> InteractivityValue {
            using T = std::decay_t<decltype(av)>;
            if constexpr(kIsFloatOrMatrixArithmetic<T>)
              return InteractivityValue(applyComponentwiseTernary(
                  av, std::get<T>(b), std::get<T>(c), [](float x, float y, float z) { return (1.0f - z) * x + z * y; }));
            else
              return std::monostate{};
          },
          a);
    }
    case InteractivityOp::eMathSmoothStep: {
      // Spec: floatN only (no floatNxN row, unlike most of the ops above), t = saturate((c -
      // min(a,b)) / |b - a|), value = t*t*(3-2t) - "defined in terms of math/min and math/saturate".
      const InteractivityValue b = instance.evaluateInput(node, "b");
      const InteractivityValue c = instance.evaluateInput(node, "c");
      if(a.index() != b.index() || a.index() != c.index())
        return std::monostate{};
      return std::visit(
          [&](auto&& av) -> InteractivityValue {
            using T = std::decay_t<decltype(av)>;
            if constexpr(kIsFloatArithmetic<T>)
              return InteractivityValue(applyComponentwiseTernary(av, std::get<T>(b), std::get<T>(c), [](float x, float y, float z) {
                const float t = std::clamp((z - std::min(x, y)) / std::fabs(y - x), 0.0f, 1.0f);
                return t * t * (3.0f - 2.0f * t);
              }));
            else
              return std::monostate{};
          },
          a);
    }

    // Comparison
    case InteractivityOp::eMathEq:
      return applyComparison(a, instance.evaluateInput(node, "b"), [](auto x, auto y) { return x == y; });
    case InteractivityOp::eMathLt: {
      const InteractivityValue b = instance.evaluateInput(node, "b");
      if(std::holds_alternative<float>(a) && std::holds_alternative<float>(b))
        return InteractivityValue(std::get<float>(a) < std::get<float>(b));
      if(std::holds_alternative<int32_t>(a) && std::holds_alternative<int32_t>(b))
        return InteractivityValue(std::get<int32_t>(a) < std::get<int32_t>(b));
      return InteractivityValue(std::monostate{});
    }
    case InteractivityOp::eMathLe: {
      const InteractivityValue b = instance.evaluateInput(node, "b");
      if(std::holds_alternative<float>(a) && std::holds_alternative<float>(b))
        return InteractivityValue(std::get<float>(a) <= std::get<float>(b));
      if(std::holds_alternative<int32_t>(a) && std::holds_alternative<int32_t>(b))
        return InteractivityValue(std::get<int32_t>(a) <= std::get<int32_t>(b));
      return InteractivityValue(std::monostate{});
    }
    case InteractivityOp::eMathGt: {
      const InteractivityValue b = instance.evaluateInput(node, "b");
      if(std::holds_alternative<float>(a) && std::holds_alternative<float>(b))
        return InteractivityValue(std::get<float>(a) > std::get<float>(b));
      if(std::holds_alternative<int32_t>(a) && std::holds_alternative<int32_t>(b))
        return InteractivityValue(std::get<int32_t>(a) > std::get<int32_t>(b));
      return InteractivityValue(std::monostate{});
    }
    case InteractivityOp::eMathGe: {
      const InteractivityValue b = instance.evaluateInput(node, "b");
      if(std::holds_alternative<float>(a) && std::holds_alternative<float>(b))
        return InteractivityValue(std::get<float>(a) >= std::get<float>(b));
      if(std::holds_alternative<int32_t>(a) && std::holds_alternative<int32_t>(b))
        return InteractivityValue(std::get<int32_t>(a) >= std::get<int32_t>(b));
      return InteractivityValue(std::monostate{});
    }

    // Trigonometric / hyperbolic / exponential - all floatN-only (spec 1099-1476), unary unless noted.
    case InteractivityOp::eMathRad:
      return applyUnary<true>(a, [](auto v) { return glm::radians(v); });
    case InteractivityOp::eMathDeg:
      return applyUnary<true>(a, [](auto v) { return glm::degrees(v); });
    case InteractivityOp::eMathSin:
      return applyUnary<true>(a, [](auto v) { return glm::sin(v); });
    case InteractivityOp::eMathCos:
      return applyUnary<true>(a, [](auto v) { return glm::cos(v); });
    case InteractivityOp::eMathTan:
      return applyUnary<true>(a, [](auto v) { return glm::tan(v); });
    case InteractivityOp::eMathAsin:
      return applyUnary<true>(a, [](auto v) { return glm::asin(v); });
    case InteractivityOp::eMathAcos:
      return applyUnary<true>(a, [](auto v) { return glm::acos(v); });
    case InteractivityOp::eMathAtan:
      return applyUnary<true>(a, [](auto v) { return glm::atan(v); });
    case InteractivityOp::eMathAtan2:
      // Spec socket names: a = Y, b = X (spec 1228), matching glm::atan(y, x).
      return applyBinary<true>(a, instance.evaluateInput(node, "b"), [](auto y, auto x) { return glm::atan(y, x); });
    case InteractivityOp::eMathSinh:
      return applyUnary<true>(a, [](auto v) { return glm::sinh(v); });
    case InteractivityOp::eMathCosh:
      return applyUnary<true>(a, [](auto v) { return glm::cosh(v); });
    case InteractivityOp::eMathTanh:
      return applyUnary<true>(a, [](auto v) { return glm::tanh(v); });
    case InteractivityOp::eMathAsinh:
      return applyUnary<true>(a, [](auto v) { return glm::asinh(v); });
    case InteractivityOp::eMathAcosh:
      return applyUnary<true>(a, [](auto v) { return glm::acosh(v); });
    case InteractivityOp::eMathAtanh:
      return applyUnary<true>(a, [](auto v) { return glm::atanh(v); });
    case InteractivityOp::eMathExp:
      return applyUnary<true>(a, [](auto v) { return glm::exp(v); });
    case InteractivityOp::eMathLog:
      return applyUnary<true>(a, [](auto v) { return glm::log(v); });
    case InteractivityOp::eMathLog2:
      return applyUnary<true>(a, [](auto v) { return glm::log2(v); });
    case InteractivityOp::eMathLog10:
      // No glm/GLSL builtin for log10; applyUnary already decomposes down to scalar float.
      return applyUnary<true>(a, [](float v) { return std::log10(v); });
    case InteractivityOp::eMathSqrt:
      return applyUnary<true>(a, [](auto v) { return glm::sqrt(v); });
    case InteractivityOp::eMathCbrt:
      return applyUnary<true>(a, [](float v) { return std::cbrt(v); });
    case InteractivityOp::eMathPow:
      return applyBinary<true>(a, instance.evaluateInput(node, "b"), [](auto x, auto y) { return glm::pow(x, y); });

    // Constants (spec 422-469): zero input value sockets, output "value".
    case InteractivityOp::eMathE:
      return 2.718281828459045f;
    case InteractivityOp::eMathPi:
      return 3.141592653589793f;
    case InteractivityOp::eMathTau:
      return 6.283185307179586f;
    case InteractivityOp::eMathInf:
      return std::numeric_limits<float>::infinity();
    case InteractivityOp::eMathNaN:
      return std::numeric_limits<float>::quiet_NaN();

    case InteractivityOp::eMathIsNaN:
      return std::holds_alternative<float>(a) ? InteractivityValue(std::isnan(std::get<float>(a))) :
                                                InteractivityValue(std::monostate{});
    case InteractivityOp::eMathIsInf:
      return std::holds_alternative<float>(a) ? InteractivityValue(std::isinf(std::get<float>(a))) :
                                                InteractivityValue(std::monostate{});

    case InteractivityOp::eMathSelect: {
      const InteractivityValue cond = instance.evaluateInput(node, "condition");
      const InteractivityValue bv   = instance.evaluateInput(node, "b");
      if(!std::holds_alternative<bool>(cond) || a.index() != bv.index())
        return std::monostate{};
      return std::get<bool>(cond) ? a : bv;
    }

    case InteractivityOp::eMathSwitch: {
      // Same dynamic-socket generation procedure as flow/switch, but for a value (spec 972-980):
      // config `cases` (int[]) names which "<case>" (decimal-string) input sockets exist; the
      // `selection` input picks one by value, falling back to `default` if unmatched.
      const InteractivityValue selection = instance.evaluateInput(node, "selection");
      if(!std::holds_alternative<int32_t>(selection))
        return std::monostate{};
      auto cfgIt = node.configuration.find("cases");
      if(cfgIt != node.configuration.end() && cfgIt->second.IsArray())
      {
        const int32_t sel = std::get<int32_t>(selection);
        for(size_t i = 0; i < cfgIt->second.ArrayLen(); ++i)
        {
          if(cfgIt->second.Get(i).GetNumberAsInt() == sel)
            return instance.evaluateInput(node, std::to_string(sel));
        }
      }
      return instance.evaluateInput(node, "default");
    }

      // eMathRandom is handled directly in evaluateNodeOutput (needs the node's own index to cache
      // by - not available here).

    case InteractivityOp::eRefEq: {
      const InteractivityValue b = instance.evaluateInput(node, "b");
      if(!std::holds_alternative<InteractivityRef>(a) || !std::holds_alternative<InteractivityRef>(b))
        return std::monostate{};
      return std::get<InteractivityRef>(a) == std::get<InteractivityRef>(b);
    }

    default:
      warnUnimplementedOnce(op);
      return std::monostate{};
  }
}

// spec "Float to Integer": truncate towards zero, then two's-complement-wrap into int32 range.
int32_t floatToIntTruncateWrap(float a)
{
  if(a == 0.0f || !std::isfinite(a))
    return 0;
  const double t       = std::trunc(static_cast<double>(a));
  const double k       = std::fmod(t, 4294967296.0);  // wrap into (-2^32, 2^32)
  const double wrapped = (k >= 2147483648.0) ? k - 4294967296.0 : (k < -2147483648.0 ? k + 4294967296.0 : k);
  return static_cast<int32_t>(wrapped);
}

InteractivityValue evaluateTypeConversion(InteractivityGraphInstance& instance, const InteractivityNode& node, InteractivityOp op)
{
  const InteractivityValue a = instance.evaluateInput(node, "a");
  switch(op)
  {
    case InteractivityOp::eTypeBoolToInt:
      return std::holds_alternative<bool>(a) ? InteractivityValue(int32_t(std::get<bool>(a) ? 1 : 0)) :
                                               InteractivityValue(std::monostate{});
    case InteractivityOp::eTypeBoolToFloat:
      return std::holds_alternative<bool>(a) ? InteractivityValue(std::get<bool>(a) ? 1.0f : 0.0f) :
                                               InteractivityValue(std::monostate{});
    case InteractivityOp::eTypeIntToBool:
      return std::holds_alternative<int32_t>(a) ? InteractivityValue(std::get<int32_t>(a) != 0) :
                                                  InteractivityValue(std::monostate{});
    case InteractivityOp::eTypeIntToFloat:
      return std::holds_alternative<int32_t>(a) ? InteractivityValue(static_cast<float>(std::get<int32_t>(a))) :
                                                  InteractivityValue(std::monostate{});
    case InteractivityOp::eTypeFloatToBool:
      return std::holds_alternative<float>(a) ?
                 InteractivityValue(!std::isnan(std::get<float>(a)) && std::get<float>(a) != 0.0f) :
                 InteractivityValue(std::monostate{});
    case InteractivityOp::eTypeFloatToInt:
      return std::holds_alternative<float>(a) ? InteractivityValue(floatToIntTruncateWrap(std::get<float>(a))) :
                                                InteractivityValue(std::monostate{});
    default:
      return std::monostate{};
  }
}

//--------------------------------------------------------------------------------------------------
// Vector / matrix / quaternion operations (spec 1481-2274). Quaternions are float4 = (x,y,z,w),
// w = scalar part (spec 3860) - see quatMulRaw's comment. Ops with multiple output sockets
// (normalize, inverse, quatToAxisAngle, matDecompose) branch on `socketName`.
//--------------------------------------------------------------------------------------------------
InteractivityValue evaluateVectorMatrixQuat(InteractivityGraphInstance& instance,
                                            const InteractivityNode&    node,
                                            InteractivityOp             op,
                                            const std::string&          socketName)
{
  auto in = [&](const char* name) { return instance.evaluateInput(node, name); };

  // Shared by matDecompose and quatFromUpForward: standard rotation-matrix -> quaternion,
  // branching on trace (the common numerically-stable algorithm; spec leaves the exact algorithm
  // implementation-defined for matDecompose's shear/negative-determinant corner, so this is a
  // reasonable, documented choice rather than a spec-mandated one).
  auto matrixToQuat = [](const glm::mat3& B) -> glm::vec4 {
    const float m00 = B[0][0], m11 = B[1][1], m22 = B[2][2];
    const float trace = m00 + m11 + m22;
    float       qx, qy, qz, qw;
    if(trace > 0.0f)
    {
      const float s = std::sqrt(trace + 1.0f) * 2.0f;
      qw            = 0.25f * s;
      qx            = (B[1][2] - B[2][1]) / s;
      qy            = (B[2][0] - B[0][2]) / s;
      qz            = (B[0][1] - B[1][0]) / s;
    }
    else if(m00 > m11 && m00 > m22)
    {
      const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
      qw            = (B[1][2] - B[2][1]) / s;
      qx            = 0.25f * s;
      qy            = (B[0][1] + B[1][0]) / s;
      qz            = (B[2][0] + B[0][2]) / s;
    }
    else if(m11 > m22)
    {
      const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
      qw            = (B[2][0] - B[0][2]) / s;
      qx            = (B[0][1] + B[1][0]) / s;
      qy            = 0.25f * s;
      qz            = (B[1][2] + B[2][1]) / s;
    }
    else
    {
      const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
      qw            = (B[0][1] - B[1][0]) / s;
      qx            = (B[2][0] + B[0][2]) / s;
      qy            = (B[1][2] + B[2][1]) / s;
      qz            = 0.25f * s;
    }
    return glm::vec4(qx, qy, qz, qw);
  };

  switch(op)
  {
    case InteractivityOp::eMathLength: {
      InteractivityValue a = in("a");
      return std::visit(
          [](auto&& v) -> InteractivityValue {
            using T = std::decay_t<decltype(v)>;
            if constexpr(std::is_same_v<T, float>)
              return std::fabs(v);
            else if constexpr(kIsFloatArithmetic<T>)
              return glm::length(v);
            else
              return std::monostate{};
          },
          a);
    }
    case InteractivityOp::eMathNormalize: {
      InteractivityValue a = in("a");
      return std::visit(
          [&](auto&& v) -> InteractivityValue {
            using T = std::decay_t<decltype(v)>;
            if constexpr(kIsFloatArithmetic<T>)
            {
              if constexpr(std::is_same_v<T, float>)
              {
                const float len   = std::fabs(v);
                const bool  valid = std::isfinite(len) && len > 0.0f;
                if(socketName == "isValid")
                  return valid;
                return valid ? InteractivityValue(v / len) : InteractivityValue(0.0f);
              }
              else
              {
                const float len   = glm::length(v);
                const bool  valid = std::isfinite(len) && len > 0.0f;
                if(socketName == "isValid")
                  return valid;
                return valid ? InteractivityValue(v / len) : InteractivityValue(T(0.0f));
              }
            }
            else
              return std::monostate{};
          },
          a);
    }
    case InteractivityOp::eMathDot: {
      InteractivityValue a = in("a"), b = in("b");
      if(a.index() != b.index())
        return std::monostate{};
      return std::visit(
          [&](auto&& av) -> InteractivityValue {
            using T = std::decay_t<decltype(av)>;
            if constexpr(std::is_same_v<T, float>)
              return av * std::get<T>(b);
            else if constexpr(kIsFloatArithmetic<T>)
              return glm::dot(av, std::get<T>(b));
            else
              return std::monostate{};
          },
          a);
    }
    case InteractivityOp::eMathCross: {
      InteractivityValue a = in("a"), b = in("b");
      if(!std::holds_alternative<glm::vec3>(a) || !std::holds_alternative<glm::vec3>(b))
        return std::monostate{};
      return glm::cross(std::get<glm::vec3>(a), std::get<glm::vec3>(b));
    }
    case InteractivityOp::eMathRotate2D: {
      InteractivityValue a = in("a"), ang = in("angle");
      if(!std::holds_alternative<glm::vec2>(a) || !std::holds_alternative<float>(ang))
        return std::monostate{};
      const glm::vec2 v = std::get<glm::vec2>(a);
      const float     t = std::get<float>(ang);
      const float     c = std::cos(t), s = std::sin(t);
      return glm::vec2(v.x * c - v.y * s, v.x * s + v.y * c);
    }
    case InteractivityOp::eMathRotate3D: {
      InteractivityValue a = in("a"), rot = in("rotation");
      if(!std::holds_alternative<glm::vec3>(a) || !std::holds_alternative<glm::vec4>(rot))
        return std::monostate{};
      const glm::vec3 v   = std::get<glm::vec3>(a);
      const glm::vec4 q   = std::get<glm::vec4>(rot);
      const glm::vec3 r   = glm::vec3(q);
      const glm::vec3 rxv = glm::cross(r, v);
      return v + 2.0f * (glm::cross(r, rxv) + q.w * rxv);
    }
    case InteractivityOp::eMathTransform: {
      InteractivityValue a = in("a"), b = in("b");
      if(std::holds_alternative<glm::vec2>(a) && std::holds_alternative<glm::mat2>(b))
        return std::get<glm::mat2>(b) * std::get<glm::vec2>(a);
      if(std::holds_alternative<glm::vec3>(a) && std::holds_alternative<glm::mat3>(b))
        return std::get<glm::mat3>(b) * std::get<glm::vec3>(a);
      if(std::holds_alternative<glm::vec4>(a) && std::holds_alternative<glm::mat4>(b))
        return std::get<glm::mat4>(b) * std::get<glm::vec4>(a);
      return std::monostate{};
    }
    case InteractivityOp::eMathSlerp: {
      // float2/float3 vector slerp - NOT quaternion slerp (that's math/quatSlerp). Falls back to
      // linear interpolation when either input is near-zero length (spec: matches math/mix there).
      InteractivityValue a = in("a"), b = in("b"), c = in("c");
      if(a.index() != b.index() || !std::holds_alternative<float>(c))
        return std::monostate{};
      const float t = std::get<float>(c);
      if(std::holds_alternative<glm::vec2>(a))
      {
        const glm::vec2 av = std::get<glm::vec2>(a), bv = std::get<glm::vec2>(b);
        const float     la = glm::length(av), lb = glm::length(bv);
        if(la <= 1e-6f || lb <= 1e-6f)
          return av * (1.0f - t) + bv * t;
        const glm::vec2 an = av / la, bn = bv / lb;
        float           theta = std::acos(glm::clamp(glm::dot(an, bn), -1.0f, 1.0f));
        if(an.x * bn.y - an.y * bn.x < 0.0f)
          theta = -theta;
        const float L  = (1.0f - t) * la + t * lb;
        const float ct = std::cos(t * theta), st = std::sin(t * theta);
        return glm::vec2(an.x * ct - an.y * st, an.x * st + an.y * ct) * L;
      }
      if(std::holds_alternative<glm::vec3>(a))
      {
        const glm::vec3 av = std::get<glm::vec3>(a), bv = std::get<glm::vec3>(b);
        const float     la = glm::length(av), lb = glm::length(bv);
        if(la <= 1e-6f || lb <= 1e-6f)
          return av * (1.0f - t) + bv * t;
        const glm::vec3 an = av / la, bn = bv / lb;
        const float     d = glm::clamp(glm::dot(an, bn), -1.0f, 1.0f);
        const float     L = (1.0f - t) * la + t * lb;
        if(d > 1.0f - 1e-6f)
          return av * (1.0f - t) + bv * t;
        const glm::vec3 axis =
            (d < -1.0f + 1e-6f) ?
                glm::normalize(glm::cross(an, std::fabs(an.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0))) :
                glm::normalize(glm::cross(an, bn));
        const float     half  = t * std::acos(d) * 0.5f;
        const glm::vec3 rAxis = axis * std::sin(half);
        const float     w     = std::cos(half);
        const glm::vec3 rxv   = glm::cross(rAxis, an);
        return (an + 2.0f * (glm::cross(rAxis, rxv) + w * rxv)) * L;
      }
      return std::monostate{};
    }
    case InteractivityOp::eMathTranspose: {
      InteractivityValue a = in("a");
      return std::visit(
          [](auto&& v) -> InteractivityValue {
            using T = std::decay_t<decltype(v)>;
            if constexpr(std::is_same_v<T, glm::mat2> || std::is_same_v<T, glm::mat3> || std::is_same_v<T, glm::mat4>)
              return glm::transpose(v);
            else
              return std::monostate{};
          },
          a);
    }
    case InteractivityOp::eMathDeterminant: {
      InteractivityValue a = in("a");
      return std::visit(
          [](auto&& v) -> InteractivityValue {
            using T = std::decay_t<decltype(v)>;
            if constexpr(std::is_same_v<T, glm::mat2> || std::is_same_v<T, glm::mat3> || std::is_same_v<T, glm::mat4>)
              return glm::determinant(v);
            else
              return std::monostate{};
          },
          a);
    }
    case InteractivityOp::eMathInverse: {
      InteractivityValue a = in("a");
      return std::visit(
          [&](auto&& v) -> InteractivityValue {
            using T = std::decay_t<decltype(v)>;
            if constexpr(std::is_same_v<T, glm::mat2> || std::is_same_v<T, glm::mat3> || std::is_same_v<T, glm::mat4>)
            {
              const float det   = glm::determinant(v);
              const bool  valid = std::isfinite(det) && det != 0.0f;
              if(socketName == "isValid")
                return valid;
              return valid ? InteractivityValue(glm::inverse(v)) : InteractivityValue(T(0.0f));
            }
            else
              return std::monostate{};
          },
          a);
    }
    case InteractivityOp::eMathMatMul: {
      InteractivityValue a = in("a"), b = in("b");
      if(a.index() != b.index())
        return std::monostate{};
      return std::visit(
          [&](auto&& av) -> InteractivityValue {
            using T = std::decay_t<decltype(av)>;
            if constexpr(std::is_same_v<T, glm::mat2> || std::is_same_v<T, glm::mat3> || std::is_same_v<T, glm::mat4>)
              return av * std::get<T>(b);
            else
              return std::monostate{};
          },
          a);
    }
    case InteractivityOp::eMathMatCompose: {
      // Order is translation, rotation, scale (spec 1829) - composes as T * R * S.
      InteractivityValue t = in("translation"), r = in("rotation"), s = in("scale");
      if(!std::holds_alternative<glm::vec3>(t) || !std::holds_alternative<glm::vec4>(r) || !std::holds_alternative<glm::vec3>(s))
        return std::monostate{};
      const glm::vec3 T = std::get<glm::vec3>(t), S = std::get<glm::vec3>(s);
      const glm::vec4 Q  = std::get<glm::vec4>(r);
      const float     xx = Q.x * Q.x, yy = Q.y * Q.y, zz = Q.z * Q.z, xy = Q.x * Q.y, xz = Q.x * Q.z, yz = Q.y * Q.z,
                  wx = Q.w * Q.x, wy = Q.w * Q.y, wz = Q.w * Q.z;
      const glm::vec3 col0(1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz), 2.0f * (xz - wy));
      const glm::vec3 col1(2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx));
      const glm::vec3 col2(2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (xx + yy));
      glm::mat4       m(1.0f);
      m[0] = glm::vec4(col0 * S.x, 0.0f);
      m[1] = glm::vec4(col1 * S.y, 0.0f);
      m[2] = glm::vec4(col2 * S.z, 0.0f);
      m[3] = glm::vec4(T, 1.0f);
      return m;
    }
    case InteractivityOp::eMathMatDecompose: {
      // Inverse of matCompose; intended only for matCompose-produced matrices (spec CAUTION).
      // Negative-determinant (mirrored) input negates X per spec's "implementation picks one of
      // 4 options" allowance; shear is not detected/corrected (spec leaves this implementation-defined).
      InteractivityValue a = in("a");
      if(!std::holds_alternative<glm::mat4>(a))
        return std::monostate{};
      const glm::mat4 m = std::get<glm::mat4>(a);
      const glm::vec3 translation(m[3]);
      glm::vec3       col0(m[0]), col1(m[1]), col2(m[2]);
      const float     sx = glm::length(col0), sy = glm::length(col1), sz = glm::length(col2);
      glm::vec3       scale(sx, sy, sz);
      glm::vec4       rotation(0, 0, 0, 1);
      if(sx > 0.0f && sy > 0.0f && sz > 0.0f && std::isfinite(sx) && std::isfinite(sy) && std::isfinite(sz))
      {
        glm::mat3 B(col0 / sx, col1 / sy, col2 / sz);
        if(glm::determinant(B) < 0.0f)
        {
          scale.x = -scale.x;
          B[0]    = -B[0];
        }
        rotation = matrixToQuat(B);
      }
      if(socketName == "translation")
        return translation;
      if(socketName == "rotation")
        return rotation;
      if(socketName == "scale")
        return scale;
      return std::monostate{};
    }
    case InteractivityOp::eMathQuatConjugate: {
      InteractivityValue a = in("a");
      if(!std::holds_alternative<glm::vec4>(a))
        return std::monostate{};
      const glm::vec4 q = std::get<glm::vec4>(a);
      return glm::vec4(-q.x, -q.y, -q.z, q.w);
    }
    case InteractivityOp::eMathQuatMul: {
      InteractivityValue a = in("a"), b = in("b");
      if(!std::holds_alternative<glm::vec4>(a) || !std::holds_alternative<glm::vec4>(b))
        return std::monostate{};
      return quatMulRaw(std::get<glm::vec4>(a), std::get<glm::vec4>(b));
    }
    case InteractivityOp::eMathQuatAngleBetween: {
      InteractivityValue a = in("a"), b = in("b");
      if(!std::holds_alternative<glm::vec4>(a) || !std::holds_alternative<glm::vec4>(b))
        return std::monostate{};
      const float d = glm::clamp(glm::dot(std::get<glm::vec4>(a), std::get<glm::vec4>(b)), -1.0f, 1.0f);
      return 2.0f * std::acos(d);
    }
    case InteractivityOp::eMathQuatFromAxisAngle: {
      InteractivityValue axis = in("axis"), angle = in("angle");
      if(!std::holds_alternative<glm::vec3>(axis) || !std::holds_alternative<float>(angle))
        return std::monostate{};
      const glm::vec3 ax = std::get<glm::vec3>(axis);
      const float     h = std::get<float>(angle) * 0.5f, s = std::sin(h);
      return glm::vec4(ax.x * s, ax.y * s, ax.z * s, std::cos(h));
    }
    case InteractivityOp::eMathQuatToAxisAngle: {
      InteractivityValue a = in("a");
      if(!std::holds_alternative<glm::vec4>(a))
        return std::monostate{};
      const glm::vec4 q = std::get<glm::vec4>(a);
      const float     w = glm::clamp(q.w, -1.0f, 1.0f);
      glm::vec3       axis;
      float           angle;
      if(std::fabs(w) > 1.0f - 1e-6f)
      {
        axis  = glm::vec3(1, 0, 0);
        angle = 0.0f;
      }
      else
      {
        const float denom = std::sqrt(1.0f - w * w);
        axis              = glm::vec3(q) / denom;
        angle             = 2.0f * std::acos(w);
      }
      if(socketName == "axis")
        return axis;
      if(socketName == "angle")
        return angle;
      return std::monostate{};
    }
    case InteractivityOp::eMathQuatFromDirections: {
      InteractivityValue a = in("a"), b = in("b");
      if(!std::holds_alternative<glm::vec3>(a) || !std::holds_alternative<glm::vec3>(b))
        return std::monostate{};
      const glm::vec3 av = std::get<glm::vec3>(a), bv = std::get<glm::vec3>(b);
      const float     c = glm::clamp(glm::dot(av, bv), -1.0f, 1.0f);
      if(c > 1.0f - 1e-6f)
        return glm::vec4(0, 0, 0, 1);
      if(c < -1.0f + 1e-6f)
      {
        const glm::vec3 perp = glm::normalize(glm::cross(av, std::fabs(av.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0)));
        return glm::vec4(perp.x, perp.y, perp.z, 0.0f);
      }
      const glm::vec3 r     = glm::normalize(glm::cross(av, bv));
      const float     sHalf = std::sqrt(0.5f - 0.5f * c), cHalf = std::sqrt(0.5f + 0.5f * c);
      return glm::vec4(r.x * sHalf, r.y * sHalf, r.z * sHalf, cHalf);
    }
    case InteractivityOp::eMathQuatFromUpForward: {
      InteractivityValue up = in("up"), fwd = in("forward");
      if(!std::holds_alternative<glm::vec3>(up) || !std::holds_alternative<glm::vec3>(fwd))
        return std::monostate{};
      const glm::vec3 u = std::get<glm::vec3>(up), f = std::get<glm::vec3>(fwd);
      glm::vec3       s = glm::cross(u, f);
      if(glm::length(s) < 1e-6f)
        s = glm::cross(std::fabs(f.x) < 0.9f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0), f);
      s                 = glm::normalize(s);
      const glm::vec3 t = glm::cross(f, s);
      return matrixToQuat(glm::mat3(s, t, f));
    }
    case InteractivityOp::eMathQuatFromAngles: {
      // Config `order` (default "yxz"; invalid/missing falls back silently per spec 2205-2229).
      // Composition convention (intrinsic axis rotations, left-multiplied in declared order) is
      // a documented choice for a corner the spec leaves to composing per-axis quaternions.
      InteractivityValue xv = in("x"), yv = in("y"), zv = in("z");
      if(!std::holds_alternative<float>(xv) || !std::holds_alternative<float>(yv) || !std::holds_alternative<float>(zv))
        return std::monostate{};
      std::string order = "yxz";
      auto        cfgIt = node.configuration.find("order");
      if(cfgIt != node.configuration.end() && cfgIt->second.IsArray() && cfgIt->second.ArrayLen() > 0)
      {
        const tinygltf::Value&                       first        = cfgIt->second.Get(size_t{0});
        static const std::unordered_set<std::string> kValidOrders = {"xyz", "xzy", "yxz", "yzx", "zxy", "zyx"};
        if(first.IsString() && kValidOrders.count(first.Get<std::string>()))
          order = first.Get<std::string>();
      }
      const float X = std::get<float>(xv), Y = std::get<float>(yv), Z = std::get<float>(zv);
      auto        axisQuat = [](char c, float angle) -> glm::vec4 {
        const float h = angle * 0.5f, s = std::sin(h), cw = std::cos(h);
        if(c == 'x')
          return glm::vec4(s, 0, 0, cw);
        if(c == 'y')
          return glm::vec4(0, s, 0, cw);
        return glm::vec4(0, 0, s, cw);
      };
      auto      angleFor = [&](char c) { return c == 'x' ? X : (c == 'y' ? Y : Z); };
      glm::vec4 q(0, 0, 0, 1);
      for(char c : order)
        q = quatMulRaw(q, axisQuat(c, angleFor(c)));
      return q;
    }
    case InteractivityOp::eMathQuatSlerp: {
      InteractivityValue a = in("a"), b = in("b"), c = in("c");
      if(!std::holds_alternative<glm::vec4>(a) || !std::holds_alternative<glm::vec4>(b) || !std::holds_alternative<float>(c))
        return std::monostate{};
      return quatSlerpRaw(std::get<glm::vec4>(a), std::get<glm::vec4>(b), std::get<float>(c));
    }
    default:
      warnUnimplementedOnce(op);
      return std::monostate{};
  }
}

//--------------------------------------------------------------------------------------------------
// Swizzle operations (spec 2278-2452): combine (named a,b,c,... inputs -> one vector/matrix
// output) and extract (one vector/matrix input -> numbered "0","1",... outputs), both in
// column-major element order for matrices.
//--------------------------------------------------------------------------------------------------
InteractivityValue evaluateSwizzle(InteractivityGraphInstance& instance,
                                   const InteractivityNode&    node,
                                   InteractivityOp             op,
                                   const std::string&          socketName)
{
  auto inF = [&](const char* name) -> float {
    InteractivityValue v = instance.evaluateInput(node, name);
    return std::holds_alternative<float>(v) ? std::get<float>(v) : 0.0f;
  };

  switch(op)
  {
    case InteractivityOp::eMathCombine2:
      return glm::vec2(inF("a"), inF("b"));
    case InteractivityOp::eMathCombine3:
      return glm::vec3(inF("a"), inF("b"), inF("c"));
    case InteractivityOp::eMathCombine4:
      return glm::vec4(inF("a"), inF("b"), inF("c"), inF("d"));
    case InteractivityOp::eMathCombine2x2:
      return glm::mat2(inF("a"), inF("b"), inF("c"), inF("d"));
    case InteractivityOp::eMathCombine3x3:
      return glm::mat3(inF("a"), inF("b"), inF("c"), inF("d"), inF("e"), inF("f"), inF("g"), inF("h"), inF("i"));
    case InteractivityOp::eMathCombine4x4:
      return glm::mat4(inF("a"), inF("b"), inF("c"), inF("d"), inF("e"), inF("f"), inF("g"), inF("h"), inF("i"),
                       inF("j"), inF("k"), inF("l"), inF("m"), inF("n"), inF("o"), inF("p"));

    case InteractivityOp::eMathExtract2:
    case InteractivityOp::eMathExtract3:
    case InteractivityOp::eMathExtract4: {
      InteractivityValue a   = instance.evaluateInput(node, "a");
      const int          idx = std::atoi(socketName.c_str());
      return std::visit(
          [&](auto&& v) -> InteractivityValue {
            using T = std::decay_t<decltype(v)>;
            if constexpr(std::is_same_v<T, glm::vec2> || std::is_same_v<T, glm::vec3> || std::is_same_v<T, glm::vec4>)
              return (idx >= 0 && idx < T::length()) ? InteractivityValue(v[idx]) : InteractivityValue(std::monostate{});
            else
              return std::monostate{};
          },
          a);
    }
    case InteractivityOp::eMathExtract2x2:
    case InteractivityOp::eMathExtract3x3:
    case InteractivityOp::eMathExtract4x4: {
      InteractivityValue a   = instance.evaluateInput(node, "a");
      const int          idx = std::atoi(socketName.c_str());
      return std::visit(
          [&](auto&& v) -> InteractivityValue {
            using T = std::decay_t<decltype(v)>;
            if constexpr(std::is_same_v<T, glm::mat2> || std::is_same_v<T, glm::mat3> || std::is_same_v<T, glm::mat4>)
            {
              const int n = T::length();
              return (idx >= 0 && idx < n * n) ? InteractivityValue(v[idx / n][idx % n]) : InteractivityValue(std::monostate{});
            }
            else
              return std::monostate{};
          },
          a);
    }
    default:
      warnUnimplementedOnce(op);
      return std::monostate{};
  }
}

//--------------------------------------------------------------------------------------------------
// math/not, math/and, math/or, math/xor share one op id across int (bitwise) and bool (logical)
// socket types (spec 5227's type-overload rule) - dispatch on the actual operand type held.
//--------------------------------------------------------------------------------------------------
InteractivityValue evaluateIntBoolLogic(InteractivityGraphInstance& instance, const InteractivityNode& node, InteractivityOp op)
{
  const InteractivityValue a = instance.evaluateInput(node, "a");
  if(op == InteractivityOp::eMathNot)
  {
    if(std::holds_alternative<bool>(a))
      return !std::get<bool>(a);
    if(std::holds_alternative<int32_t>(a))
      return ~std::get<int32_t>(a);
    return std::monostate{};
  }

  const InteractivityValue b = instance.evaluateInput(node, "b");
  if(std::holds_alternative<bool>(a) && std::holds_alternative<bool>(b))
  {
    const bool ba = std::get<bool>(a), bb = std::get<bool>(b);
    switch(op)
    {
      case InteractivityOp::eMathAnd:
        return ba && bb;
      case InteractivityOp::eMathOr:
        return ba || bb;
      case InteractivityOp::eMathXor:
        return ba != bb;
      default:
        return std::monostate{};
    }
  }
  if(std::holds_alternative<int32_t>(a) && std::holds_alternative<int32_t>(b))
  {
    const int32_t ia = std::get<int32_t>(a), ib = std::get<int32_t>(b);
    switch(op)
    {
      case InteractivityOp::eMathAnd:
        return ia & ib;
      case InteractivityOp::eMathOr:
        return ia | ib;
      case InteractivityOp::eMathXor:
        return ia ^ ib;
      default:
        return std::monostate{};
    }
  }
  return std::monostate{};
}

//--------------------------------------------------------------------------------------------------
// Integer-only bit-manipulation ops (spec 2833-2926). C++20 guarantees `>>` on a signed integer
// is an arithmetic (sign-propagating) shift, matching the spec's asr requirement without UB.
//--------------------------------------------------------------------------------------------------
InteractivityValue evaluateBitwise(InteractivityGraphInstance& instance, const InteractivityNode& node, InteractivityOp op)
{
  const InteractivityValue a = instance.evaluateInput(node, "a");
  if(!std::holds_alternative<int32_t>(a))
    return std::monostate{};
  const int32_t ia = std::get<int32_t>(a);

  switch(op)
  {
    case InteractivityOp::eMathAsr:
    case InteractivityOp::eMathLsl: {
      const InteractivityValue b = instance.evaluateInput(node, "b");
      if(!std::holds_alternative<int32_t>(b))
        return std::monostate{};
      const uint32_t shift = static_cast<uint32_t>(std::get<int32_t>(b)) & 31u;  // spec: only low 5 bits used
      return op == InteractivityOp::eMathAsr ? InteractivityValue(static_cast<int32_t>(ia >> shift)) :
                                               InteractivityValue(static_cast<int32_t>(static_cast<uint32_t>(ia) << shift));
    }
    case InteractivityOp::eMathClz: {
      uint32_t u = static_cast<uint32_t>(ia);
      if(u == 0)
        return int32_t(32);
      int32_t n = 0;
      while(!(u & 0x80000000u))
      {
        u <<= 1;
        ++n;
      }
      return n;
    }
    case InteractivityOp::eMathCtz: {
      uint32_t u = static_cast<uint32_t>(ia);
      if(u == 0)
        return int32_t(32);
      int32_t n = 0;
      while(!(u & 1u))
      {
        u >>= 1;
        ++n;
      }
      return n;
    }
    case InteractivityOp::eMathPopcnt: {
      uint32_t u = static_cast<uint32_t>(ia);
      int32_t  n = 0;
      while(u)
      {
        n += static_cast<int32_t>(u & 1u);
        u >>= 1;
      }
      return n;
    }
    default:
      return std::monostate{};
  }
}

//--------------------------------------------------------------------------------------------------
// Color operations (spec 2977-3096): linear-sRGB <-> OkLCh, via Bjorn Ottosson's published OkLab
// transform. Neither direction clamps (spec CAUTION) - chain math/saturate if clamping is wanted.
//--------------------------------------------------------------------------------------------------
InteractivityValue evaluateColor(InteractivityGraphInstance& instance, const InteractivityNode& node, InteractivityOp op, const std::string& socketName)
{
  auto inF = [&](const char* name) -> float {
    InteractivityValue v = instance.evaluateInput(node, name);
    return std::holds_alternative<float>(v) ? std::get<float>(v) : 0.0f;
  };

  if(op == InteractivityOp::eMathRgbToOkLCh)
  {
    const float r = inF("r"), g = inF("g"), b = inF("b");
    const float l  = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
    const float m  = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
    const float s  = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;
    const float l_ = std::cbrt(l), m_ = std::cbrt(m), s_ = std::cbrt(s);
    const float L = 0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_;
    const float A = 1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_;
    const float B = 0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_;
    if(socketName == "l")
      return L;
    if(socketName == "c")
      return std::sqrt(A * A + B * B);
    if(socketName == "h")
      return std::atan2(B, A);
    return std::monostate{};
  }
  if(op == InteractivityOp::eMathRgbFromOkLCh)
  {
    const float L = inF("l"), C = inF("c"), H = inF("h");
    const float A = C * std::cos(H), B = C * std::sin(H);
    const float l_ = L + 0.3963377774f * A + 0.2158037573f * B;
    const float m_ = L - 0.1055613458f * A - 0.0638541728f * B;
    const float s_ = L - 0.0894841775f * A - 1.2914855480f * B;
    const float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    if(socketName == "r")
      return 4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    if(socketName == "g")
      return -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
    if(socketName == "b")
      return -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
    return std::monostate{};
  }
  return std::monostate{};
}

// Reads `node`'s "type" configuration (an index into the graph's types[]) as an InteractivityValueType.
InteractivityValueType configuredType(InteractivityGraphInstance& instance, const InteractivityNode& node)
{
  auto typeIt = node.configuration.find("type");
  if(typeIt == node.configuration.end() || !typeIt->second.IsArray() || typeIt->second.ArrayLen() == 0)
    return InteractivityValueType::eUnknown;
  const int typeIndex = typeIt->second.Get(size_t{0}).GetNumberAsInt();
  if(typeIndex < 0 || typeIndex >= static_cast<int>(instance.graph().types().size()))
    return InteractivityValueType::eUnknown;
  return instance.graph().types()[typeIndex].signature;
}

// Reads `node`'s "pointer" configuration (a JSON-Pointer-Template string), already tokenized.
// Returns nullopt if the configuration is missing or malformed.
std::optional<std::vector<PointerTemplateSegment>> configuredPointerTemplate(const InteractivityNode& node)
{
  auto ptrIt = node.configuration.find("pointer");
  if(ptrIt == node.configuration.end() || !ptrIt->second.IsArray() || ptrIt->second.ArrayLen() == 0
     || !ptrIt->second.Get(size_t{0}).IsString())
    return std::nullopt;
  return parsePointerTemplate(ptrIt->second.Get(size_t{0}).Get<std::string>());
}

//--------------------------------------------------------------------------------------------------
// pointer/get (spec 4012-4081): pure/0-flow, 2 outputs (value, isValid). Unresolvable paths,
// missing resolver, or a resolved-value type mismatch all report via isValid=false + type-default
// value - never a crash, matching the spec's "not an error" framing for this op.
//--------------------------------------------------------------------------------------------------
// The two self-referential "object model" pointers KHR_interactivity itself defines (spec's "Delay
// References" / "Event References" sections) query the running instance's own ref-tracking state,
// not the glTF model - so they're resolved here directly rather than routed through
// InteractivityPointerResolver, which is deliberately Scene/model-only (see
// gltf_interactivity_pointer.hpp's header comment: it must stay usable without a Scene). Recognized
// as exactly [literal prefix, single eRefParam] with nothing else, matching how the spec defines
// each as a single implicit ref parameter with no further path ("{}").
std::optional<InteractivityValue> resolveSelfReferentialRefPointer(InteractivityGraphInstance&                instance,
                                                                   const InteractivityNode&                   node,
                                                                   const std::vector<PointerTemplateSegment>& segments)
{
  if(segments.size() != 2 || segments[0].kind != PointerTemplateSegment::Kind::eLiteral
     || segments[1].kind != PointerTemplateSegment::Kind::eRefParam)
    return std::nullopt;
  const bool isEvents = segments[0].text == "/extensions/KHR_interactivity/events/";
  const bool isDelays = segments[0].text == "/extensions/KHR_interactivity/delays/";
  if(!isEvents && !isDelays)
    return std::nullopt;

  const InteractivityValue paramValue = instance.evaluateInput(node, segments[1].text);
  if(!std::holds_alternative<InteractivityRef>(paramValue))
    return std::nullopt;
  const InteractivityRef ref   = std::get<InteractivityRef>(paramValue);
  const bool             valid = isEvents ? instance.isEventRef(ref) : instance.isPendingDelayRef(ref);
  // Null (never-null-checked-away, see isEventRef/isPendingDelayRef) or otherwise-unknown refs fall
  // through as "unresolvable" (nullopt), same as any other pointer/get miss - not a distinct case.
  return valid ? std::optional<InteractivityValue>(ref) : std::nullopt;
}

InteractivityValue evaluatePointerGet(InteractivityGraphInstance& instance, const InteractivityNode& node, const std::string& socketName)
{
  const InteractivityValueType expectedType = configuredType(instance, node);
  const auto                   segments     = configuredPointerTemplate(node);

  std::optional<InteractivityValue> resolved;
  if(segments)
    resolved = resolveSelfReferentialRefPointer(instance, node, *segments);
  if(!resolved && segments)
  {
    if(InteractivityPointerResolver* resolver = instance.pointerResolver())
    {
      if(std::optional<std::string> concretePath = substitutePointerTemplate(*segments, instance, node))
        resolved = resolver->get(*concretePath);
    }
  }

  const bool valid = resolved.has_value() && interactivityValueType(*resolved) == expectedType;
  if(socketName == "isValid")
    return valid;
  return valid ? *resolved : defaultInteractivityValue(expectedType);
}

//--------------------------------------------------------------------------------------------------
// pointer/set (spec 4082-4148): flow-triggered, activates `out` on success or `err` if the type
// doesn't match the configured type, the path is unresolvable, or the property isn't writable.
//--------------------------------------------------------------------------------------------------
void executePointerSet(InteractivityGraphInstance& instance, const InteractivityNode& node)
{
  auto activate = [&](const char* socketName) {
    auto it = node.flows.find(socketName);
    if(it != node.flows.end())
      instance.activateFlow(it->second.targetNode, it->second.targetSocket);
  };

  const InteractivityValueType expectedType = configuredType(instance, node);
  const InteractivityValue     newValue     = instance.evaluateInput(node, "value");
  if(interactivityValueType(newValue) != expectedType)
  {
    activate("err");
    return;
  }

  const auto                    segments = configuredPointerTemplate(node);
  InteractivityPointerResolver* resolver = instance.pointerResolver();
  std::optional<std::string>    concretePath;
  if(segments)
    concretePath = substitutePointerTemplate(*segments, instance, node);

  if(!resolver || !concretePath || !resolver->set(*concretePath, newValue))
  {
    activate("err");
    return;
  }
  activate("out");
}

//--------------------------------------------------------------------------------------------------
// variable/interpolate (spec 3739-3845): eases a custom variable to `value` over `duration` seconds
// using a cubic Bézier easing curve (p1/p2 control points). Registers an entry in the instance's
// interpolation state array; the actual per-tick value writes happen in
// InteractivityGraphInstance::advanceVariableInterpolations() (gltf_interactivity_instance.cpp) -
// this function only validates inputs and starts the interpolation, per spec's "When the `in` input
// flow is activated" steps.
//--------------------------------------------------------------------------------------------------
void executeVariableInterpolate(InteractivityGraphInstance& instance, const InteractivityNode& node)
{
  auto activate = [&](const char* socketName) {
    auto it = node.flows.find(socketName);
    if(it != node.flows.end())
      instance.activateFlow(it->second.targetNode, it->second.targetSocket);
  };

  int32_t variableIndex = -1;
  if(auto it = node.configuration.find("variable");
     it != node.configuration.end() && it->second.IsArray() && it->second.ArrayLen() > 0)
    variableIndex = it->second.Get(size_t{0}).GetNumberAsInt();
  if(variableIndex < 0 || variableIndex >= static_cast<int>(instance.graph().variables().size()))
  {
    activate("err");
    return;
  }

  bool useSlerp = false;
  if(auto it = node.configuration.find("useSlerp");
     it != node.configuration.end() && it->second.IsArray() && it->second.ArrayLen() > 0)
    useSlerp = it->second.Get(size_t{0}).Get<bool>();

  const InteractivityValue targetValue = instance.evaluateInput(node, "value");
  const InteractivityValue durationV   = instance.evaluateInput(node, "duration");
  if(!std::holds_alternative<float>(durationV))
  {
    activate("err");
    return;
  }
  const float duration = std::get<float>(durationV);
  if(std::isnan(duration) || std::isinf(duration) || duration < 0.0f)
  {
    activate("err");
    return;
  }

  const InteractivityValue p1V = instance.evaluateInput(node, "p1");
  const InteractivityValue p2V = instance.evaluateInput(node, "p2");
  if(!std::holds_alternative<glm::vec2>(p1V) || !std::holds_alternative<glm::vec2>(p2V))
  {
    activate("err");
    return;
  }
  const glm::vec2 p1 = std::get<glm::vec2>(p1V), p2 = std::get<glm::vec2>(p2V);
  if(!isValidBezierControlPoint(p1) || !isValidBezierControlPoint(p2))
  {
    activate("err");
    return;
  }

  auto doneIt = node.flows.find("done");
  instance.startVariableInterpolation({.variableIndex  = variableIndex,
                                       .startTime      = instance.timeSinceStart(),
                                       .duration       = duration,
                                       .startValue     = instance.variable(variableIndex),
                                       .targetValue    = targetValue,
                                       .p1             = p1,
                                       .p2             = p2,
                                       .useSlerp       = useSlerp,
                                       .doneTargetNode = doneIt != node.flows.end() ? doneIt->second.targetNode : -1,
                                       .doneTargetSocket = doneIt != node.flows.end() ? doneIt->second.targetSocket : std::string{}});
  activate("out");
}

//--------------------------------------------------------------------------------------------------
// pointer/interpolate (spec 4154-4254): same easing model as variable/interpolate, targeting an
// Object Model property instead of a custom variable. Quaternion (spherical) interpolation is used
// when the resolved type is float4 and the concrete path ends in "/rotation" - the only float4
// property this app's pointer surface (gltf_interactivity_scene_pointer.cpp) exposes that's
// actually a quaternion; other float4 properties (e.g. base color) get plain linear interpolation.
//--------------------------------------------------------------------------------------------------
void executePointerInterpolate(InteractivityGraphInstance& instance, const InteractivityNode& node)
{
  auto activate = [&](const char* socketName) {
    auto it = node.flows.find(socketName);
    if(it != node.flows.end())
      instance.activateFlow(it->second.targetNode, it->second.targetSocket);
  };

  const InteractivityValueType expectedType = configuredType(instance, node);
  if(expectedType == InteractivityValueType::eUnknown || expectedType == InteractivityValueType::eBool
     || expectedType == InteractivityValueType::eInt)
  {
    activate("err");
    return;
  }

  const auto                    segments = configuredPointerTemplate(node);
  InteractivityPointerResolver* resolver = instance.pointerResolver();
  std::optional<std::string>    concretePath;
  if(segments)
    concretePath = substitutePointerTemplate(*segments, instance, node);

  std::optional<InteractivityValue> currentValue;
  if(resolver && concretePath)
    currentValue = resolver->get(*concretePath);
  if(!currentValue || interactivityValueType(*currentValue) != expectedType)
  {
    activate("err");
    return;
  }

  const InteractivityValue targetValue = instance.evaluateInput(node, "value");
  if(interactivityValueType(targetValue) != expectedType)
  {
    activate("err");
    return;
  }

  const InteractivityValue durationV = instance.evaluateInput(node, "duration");
  if(!std::holds_alternative<float>(durationV))
  {
    activate("err");
    return;
  }
  const float duration = std::get<float>(durationV);
  if(std::isnan(duration) || std::isinf(duration) || duration < 0.0f)
  {
    activate("err");
    return;
  }

  const InteractivityValue p1V = instance.evaluateInput(node, "p1");
  const InteractivityValue p2V = instance.evaluateInput(node, "p2");
  if(!std::holds_alternative<glm::vec2>(p1V) || !std::holds_alternative<glm::vec2>(p2V))
  {
    activate("err");
    return;
  }
  const glm::vec2 p1 = std::get<glm::vec2>(p1V), p2 = std::get<glm::vec2>(p2V);
  if(!isValidBezierControlPoint(p1) || !isValidBezierControlPoint(p2))
  {
    activate("err");
    return;
  }

  const bool useSlerp = expectedType == InteractivityValueType::eFloat4 && concretePath->ends_with("/rotation");

  auto doneIt = node.flows.find("done");
  instance.startPointerInterpolation({.path           = *concretePath,
                                      .startTime      = instance.timeSinceStart(),
                                      .duration       = duration,
                                      .startValue     = *currentValue,
                                      .targetValue    = targetValue,
                                      .p1             = p1,
                                      .p2             = p2,
                                      .useSlerp       = useSlerp,
                                      .doneTargetNode = doneIt != node.flows.end() ? doneIt->second.targetNode : -1,
                                      .doneTargetSocket = doneIt != node.flows.end() ? doneIt->second.targetSocket : std::string{}});
  activate("out");
}

//--------------------------------------------------------------------------------------------------
// animation/start, animation/stop, animation/stopAt (spec 4262-4441). All three validate the
// `animation` ref via InteractivityAnimationResolver::isValidAnimation() before touching any
// instance state - this engine does not itself apply the resulting pose (see
// gltf_interactivity_animation.hpp); InteractivityGraphInstance::advanceAnimations() only computes
// *what* to apply, and GltfRenderer drains InteractivityGraphInstance::pendingAnimationApplies()
// each frame to actually do it.
//--------------------------------------------------------------------------------------------------
void executeAnimationStart(InteractivityGraphInstance& instance, const InteractivityNode& node)
{
  auto activate = [&](const char* socketName) {
    auto it = node.flows.find(socketName);
    if(it != node.flows.end())
      instance.activateFlow(it->second.targetNode, it->second.targetSocket);
  };

  InteractivityAnimationResolver* resolver = instance.animationResolver();
  const InteractivityValue        animV    = instance.evaluateInput(node, "animation");
  if(!std::holds_alternative<InteractivityRef>(animV) || !resolver
     || !resolver->isValidAnimation(std::get<InteractivityRef>(animV).handle))
  {
    activate("err");
    return;
  }
  const InteractivityRef animRef = std::get<InteractivityRef>(animV);

  const InteractivityValue startTimeV = instance.evaluateInput(node, "startTime");
  const InteractivityValue endTimeV   = instance.evaluateInput(node, "endTime");
  if(!std::holds_alternative<float>(startTimeV) || !std::holds_alternative<float>(endTimeV))
  {
    activate("err");
    return;
  }
  const float startTime = std::get<float>(startTimeV), endTime = std::get<float>(endTimeV);
  if(!std::isfinite(startTime) || !std::isfinite(endTime))
  {
    activate("err");
    return;
  }

  const InteractivityValue speedV = instance.evaluateInput(node, "speed");
  if(!std::holds_alternative<float>(speedV))
  {
    activate("err");
    return;
  }
  const float speed = std::get<float>(speedV);
  if(std::isnan(speed) || std::isinf(speed) || speed <= 0.0f)
  {
    activate("err");
    return;
  }

  auto doneIt = node.flows.find("done");
  instance.startAnimation({.animation         = animRef,
                           .startTime         = startTime,
                           .endTime           = endTime,
                           .stopTime          = endTime,
                           .speed             = speed,
                           .entryCreationTime = instance.timeSinceStart(),
                           .doneTargetNode    = doneIt != node.flows.end() ? doneIt->second.targetNode : -1,
                           .doneTargetSocket = doneIt != node.flows.end() ? doneIt->second.targetSocket : std::string{}});
  activate("out");
}

void executeAnimationStop(InteractivityGraphInstance& instance, const InteractivityNode& node)
{
  auto activate = [&](const char* socketName) {
    auto it = node.flows.find(socketName);
    if(it != node.flows.end())
      instance.activateFlow(it->second.targetNode, it->second.targetSocket);
  };

  InteractivityAnimationResolver* resolver = instance.animationResolver();
  const InteractivityValue        animV    = instance.evaluateInput(node, "animation");
  if(!std::holds_alternative<InteractivityRef>(animV) || !resolver
     || !resolver->isValidAnimation(std::get<InteractivityRef>(animV).handle))
  {
    activate("err");
    return;
  }
  instance.stopAnimation(std::get<InteractivityRef>(animV));
  activate("out");
}

void executeAnimationStopAt(InteractivityGraphInstance& instance, const InteractivityNode& node)
{
  auto activate = [&](const char* socketName) {
    auto it = node.flows.find(socketName);
    if(it != node.flows.end())
      instance.activateFlow(it->second.targetNode, it->second.targetSocket);
  };

  InteractivityAnimationResolver* resolver = instance.animationResolver();
  const InteractivityValue        animV    = instance.evaluateInput(node, "animation");
  if(!std::holds_alternative<InteractivityRef>(animV) || !resolver
     || !resolver->isValidAnimation(std::get<InteractivityRef>(animV).handle))
  {
    activate("err");
    return;
  }

  const InteractivityValue stopTimeV = instance.evaluateInput(node, "stopTime");
  if(!std::holds_alternative<float>(stopTimeV) || !std::isfinite(std::get<float>(stopTimeV)))
  {
    activate("err");
    return;
  }

  auto doneIt = node.flows.find("done");
  instance.scheduleAnimationStop(std::get<InteractivityRef>(animV), std::get<float>(stopTimeV),
                                 doneIt != node.flows.end() ? doneIt->second.targetNode : -1,
                                 doneIt != node.flows.end() ? doneIt->second.targetSocket : std::string{});
  activate("out");
}

// Finds `socketName` in `node.flows` and activates it if present (spec: an unconnected output
// flow socket is a documented no-op, never an error) - the common tail of nearly every flow op.
void activateNamed(InteractivityGraphInstance& instance, const InteractivityNode& node, const char* socketName)
{
  auto it = node.flows.find(socketName);
  if(it != node.flows.end())
    instance.activateFlow(it->second.targetNode, it->second.targetSocket);
}

//--------------------------------------------------------------------------------------------------
// flow/multiGate (spec 3456-3509): stateful gate cycling through its output flows, either in
// ascending id order or randomly, optionally looping once all outputs have fired.
//--------------------------------------------------------------------------------------------------
void executeMultiGate(InteractivityGraphInstance& instance, const InteractivityNode& node, int nodeIndex, const std::string& enteredSocket)
{
  // Output ids in Socket Order (ascending lexicographic - spec) - multiGate's `used` bitmap is
  // indexed by this order.
  std::vector<std::string> outputIds;
  outputIds.reserve(node.flows.size());
  for(const auto& [id, flow] : node.flows)
    outputIds.push_back(id);
  std::sort(outputIds.begin(), outputIds.end());

  InteractivityGraphInstance::MultiGateState& state = instance.multiGateState(nodeIndex);
  if(state.used.size() != outputIds.size())
    state.used.assign(outputIds.size(), false);

  if(enteredSocket == "reset")
  {
    state.lastIndex = -1;
    std::fill(state.used.begin(), state.used.end(), false);
    return;
  }

  bool isRandom = false, isLoop = false;
  if(auto it = node.configuration.find("isRandom");
     it != node.configuration.end() && it->second.IsArray() && it->second.ArrayLen() > 0)
    isRandom = it->second.Get(size_t{0}).Get<bool>();
  if(auto it = node.configuration.find("isLoop");
     it != node.configuration.end() && it->second.IsArray() && it->second.ArrayLen() > 0)
    isLoop = it->second.Get(size_t{0}).Get<bool>();

  auto pickUnused = [&]() -> int {
    std::vector<size_t> candidates;
    for(size_t i = 0; i < state.used.size(); ++i)
      if(!state.used[i])
        candidates.push_back(i);
    if(candidates.empty())
      return -1;
    if(!isRandom)
      return static_cast<int>(candidates.front());
    // Reuses the per-node random cache math/random uses (keyed by this multiGate node's own
    // index, so no collision with any actual math/random node) rather than exposing the RNG.
    const size_t pick =
        static_cast<size_t>(instance.randomValue(nodeIndex) * static_cast<float>(candidates.size())) % candidates.size();
    return static_cast<int>(candidates[pick]);
  };

  int picked = pickUnused();
  if(picked == -1 && isLoop)
  {
    std::fill(state.used.begin(), state.used.end(), false);
    picked = pickUnused();
  }
  if(picked == -1)
    return;

  state.used[static_cast<size_t>(picked)] = true;
  state.lastIndex                         = picked;
  activateNamed(instance, node, outputIds[static_cast<size_t>(picked)].c_str());
}

//--------------------------------------------------------------------------------------------------
// flow/waitAll (spec 3510-3561): fires `out` on each new (not-yet-seen) indexed input, `completed`
// once every configured input has fired at least once since the last reset.
//--------------------------------------------------------------------------------------------------
void executeWaitAll(InteractivityGraphInstance& instance, const InteractivityNode& node, int nodeIndex, const std::string& enteredSocket)
{
  int32_t inputFlows = 0;
  if(auto it = node.configuration.find("inputFlows");
     it != node.configuration.end() && it->second.IsArray() && it->second.ArrayLen() > 0)
    inputFlows = it->second.Get(size_t{0}).GetNumberAsInt();
  if(inputFlows < 0 || inputFlows > 64)
    inputFlows = 0;

  InteractivityGraphInstance::WaitAllState& state = instance.waitAllState(nodeIndex);
  if(state.remaining < 0)  // never initialized
  {
    state.remaining = inputFlows;
    state.used.assign(static_cast<size_t>(inputFlows), false);
  }

  if(enteredSocket == "reset")
  {
    state.remaining = inputFlows;
    std::fill(state.used.begin(), state.used.end(), false);
    return;
  }

  char*      endPtr = nullptr;
  const long idx    = std::strtol(enteredSocket.c_str(), &endPtr, 10);
  if(endPtr != enteredSocket.c_str() + enteredSocket.size() || idx < 0 || idx >= static_cast<long>(state.used.size()))
    return;  // not one of this node's indexed input sockets
  if(!state.used[static_cast<size_t>(idx)])
  {
    state.used[static_cast<size_t>(idx)] = true;
    --state.remaining;
  }

  activateNamed(instance, node, state.remaining == 0 ? "completed" : "out");
}

// debug/log's `message` template: `{name}` substitutes the evaluated `name` input socket;
// `{{`/`}}` are literal-brace escapes (spec 4619-4709). Processed left-to-right in one pass
// (rather than the spec's descending-offset in-place substitution) - equivalent result since
// stringified values here never themselves need re-escaping.
std::string formatDebugMessage(InteractivityGraphInstance& instance, const InteractivityNode& node, const std::string& messageTemplate)
{
  std::string result;
  size_t      i = 0;
  while(i < messageTemplate.size())
  {
    if(messageTemplate[i] == '{' && i + 1 < messageTemplate.size() && messageTemplate[i + 1] == '{')
    {
      result += '{';
      i += 2;
      continue;
    }
    if(messageTemplate[i] == '}' && i + 1 < messageTemplate.size() && messageTemplate[i + 1] == '}')
    {
      result += '}';
      i += 2;
      continue;
    }
    if(messageTemplate[i] == '{')
    {
      const size_t end = messageTemplate.find('}', i + 1);
      if(end == std::string::npos)
      {
        result += messageTemplate.substr(i);
        break;
      }
      const std::string paramName = messageTemplate.substr(i + 1, end - i - 1);
      result += stringifyInteractivityValue(instance.evaluateInput(node, paramName));
      i = end + 1;
      continue;
    }
    result += messageTemplate[i];
    ++i;
  }
  return result;
}

}  // namespace

// Stringifies any InteractivityValue for debug/log's message template substitution (above) and the
// Graphs UI panel's variable/event value display (ui_interactivity.cpp) - declared in the header so
// both can call it.
std::string stringifyInteractivityValue(const InteractivityValue& v)
{
  return std::visit(
      [](auto&& x) -> std::string {
        using T = std::decay_t<decltype(x)>;
        if constexpr(std::is_same_v<T, std::monostate>)
          return "<undefined>";
        else if constexpr(std::is_same_v<T, bool>)
          return x ? "true" : "false";
        else if constexpr(std::is_same_v<T, int32_t>)
          return std::to_string(x);
        else if constexpr(std::is_same_v<T, float>)
          return std::to_string(x);
        else if constexpr(std::is_same_v<T, InteractivityRef>)
          return "ref#" + std::to_string(x.handle);
        else if constexpr(std::is_same_v<T, glm::vec2> || std::is_same_v<T, glm::vec3> || std::is_same_v<T, glm::vec4>)
        {
          std::string s = "(";
          for(glm::length_t i = 0; i < T::length(); ++i)
            s += (i ? ", " : "") + std::to_string(x[i]);
          return s + ")";
        }
        else  // mat2/mat3/mat4: column-major flatten
        {
          std::string s     = "(";
          bool        first = true;
          for(glm::length_t c = 0; c < T::length(); ++c)
            for(glm::length_t r = 0; r < T::length(); ++r)
            {
              if(!first)
                s += ", ";
              first = false;
              s += std::to_string(x[c][r]);
            }
          return s + ")";
        }
      },
      v);
}

// Standard "UnitBezier" solve: Newton-Raphson for the common case, falling back to bisection when
// the derivative is near zero (a flat/vertical stretch of the curve) - same algorithm CSS
// `cubic-bezier()` timing functions use. p1/p2 are pre-validated by isValidBezierControlPoint
// (x components in [0,1]) before this is ever called, so a solution always exists in [0,1].
float cubicBezierEase(const glm::vec2& p1, const glm::vec2& p2, float t)
{
  if(t <= 0.0f)
    return 0.0f;
  if(t >= 1.0f)
    return 1.0f;

  auto sampleX  = [&](float s) { return 3 * (1 - s) * (1 - s) * s * p1.x + 3 * (1 - s) * s * s * p2.x + s * s * s; };
  auto sampleY  = [&](float s) { return 3 * (1 - s) * (1 - s) * s * p1.y + 3 * (1 - s) * s * s * p2.y + s * s * s; };
  auto sampleDX = [&](float s) {
    return 3 * (1 - s) * (1 - s) * p1.x + 6 * (1 - s) * s * (p2.x - p1.x) + 3 * s * s * (1 - p2.x);
  };

  float s = t;
  for(int i = 0; i < 8; ++i)
  {
    const float x = sampleX(s) - t;
    if(std::abs(x) < 1e-6f)
      return sampleY(s);
    const float d = sampleDX(s);
    if(std::abs(d) < 1e-6f)
      break;
    s -= x / d;
  }
  float lo = 0.0f, hi = 1.0f;
  s = t;
  for(int i = 0; i < 20 && std::abs(sampleX(s) - t) >= 1e-6f; ++i)
  {
    if(sampleX(s) < t)
      lo = s;
    else
      hi = s;
    s = (lo + hi) * 0.5f;
  }
  return sampleY(s);
}

InteractivityValue lerpInteractivityValue(const InteractivityValue& a, const InteractivityValue& b, float q, bool slerp)
{
  if(a.index() != b.index())
    return a;
  if(slerp && std::holds_alternative<glm::vec4>(a))
    return quatSlerpRaw(std::get<glm::vec4>(a), std::get<glm::vec4>(b), q);
  return std::visit(
      [&](auto&& av) -> InteractivityValue {
        using T = std::decay_t<decltype(av)>;
        if constexpr(kIsFloatOrMatrixArithmetic<T>)
          return InteractivityValue(
              applyComponentwiseBinary(av, std::get<T>(b), [q](float x, float y) { return x + (y - x) * q; }));
        else
          return a;  // bool/int/ref: not interpolatable (spec keeps these out of variable/interpolate's
                     // and pointer/interpolate's valid-type sets)
      },
      a);
}

InteractivityValue evaluateNodeOutput(InteractivityGraphInstance& instance, int nodeIndex, const std::string& socketName)
{
  if(nodeIndex < 0 || nodeIndex >= static_cast<int>(instance.graph().nodes().size()))
    return std::monostate{};

  const InteractivityNode& node = nodeAt(instance, nodeIndex);
  const InteractivityOp    op   = opOf(instance, node);

  switch(op)
  {
    case InteractivityOp::eEventOnStart:
      // Every event/onStart node in the graph shares ONE occurrence ref (spec: conceptually a
      // single "start" event per run, not one per node) - minted once in start(), not per-node.
      return socketName == "event" ? InteractivityValue(instance.currentStartRef()) : InteractivityValue(std::monostate{});

    case InteractivityOp::eEventOnTick:
      if(socketName == "timeSinceStart")
        return instance.timeSinceStart();
      if(socketName == "timeSinceLastTick")
        return instance.timeSinceLastTick();
      if(socketName == "event")
        // Shared by every event/onTick node for THIS tick, same rationale as onStart above.
        return instance.currentTickRef();
      return std::monostate{};

    case InteractivityOp::eVariableGet: {
      auto it = node.configuration.find("variable");
      if(it == node.configuration.end() || !it->second.IsArray() || it->second.ArrayLen() == 0)
        return std::monostate{};
      const int index = it->second.Get(size_t{0}).GetNumberAsInt();
      if(index < 0 || index >= static_cast<int>(instance.graph().variables().size()))
        return std::monostate{};
      return instance.variable(index);
    }

    case InteractivityOp::eMathAbs:
    case InteractivityOp::eMathNeg:
    case InteractivityOp::eMathSign:
    case InteractivityOp::eMathFloor:
    case InteractivityOp::eMathCeil:
    case InteractivityOp::eMathRound:
    case InteractivityOp::eMathFract:
    case InteractivityOp::eMathTrunc:
    case InteractivityOp::eMathSaturate:
    case InteractivityOp::eMathAdd:
    case InteractivityOp::eMathSub:
    case InteractivityOp::eMathMul:
    case InteractivityOp::eMathDiv:
    case InteractivityOp::eMathRem:
    case InteractivityOp::eMathMin:
    case InteractivityOp::eMathMax:
    case InteractivityOp::eMathClamp:
    case InteractivityOp::eMathMix:
    case InteractivityOp::eMathSmoothStep:
    case InteractivityOp::eMathEq:
    case InteractivityOp::eMathLt:
    case InteractivityOp::eMathLe:
    case InteractivityOp::eMathGt:
    case InteractivityOp::eMathGe:
    case InteractivityOp::eMathRad:
    case InteractivityOp::eMathDeg:
    case InteractivityOp::eMathSin:
    case InteractivityOp::eMathCos:
    case InteractivityOp::eMathTan:
    case InteractivityOp::eMathAsin:
    case InteractivityOp::eMathAcos:
    case InteractivityOp::eMathAtan:
    case InteractivityOp::eMathAtan2:
    case InteractivityOp::eMathSinh:
    case InteractivityOp::eMathCosh:
    case InteractivityOp::eMathTanh:
    case InteractivityOp::eMathAsinh:
    case InteractivityOp::eMathAcosh:
    case InteractivityOp::eMathAtanh:
    case InteractivityOp::eMathExp:
    case InteractivityOp::eMathLog:
    case InteractivityOp::eMathLog2:
    case InteractivityOp::eMathLog10:
    case InteractivityOp::eMathSqrt:
    case InteractivityOp::eMathCbrt:
    case InteractivityOp::eMathPow:
    case InteractivityOp::eMathE:
    case InteractivityOp::eMathPi:
    case InteractivityOp::eMathTau:
    case InteractivityOp::eMathInf:
    case InteractivityOp::eMathNaN:
    case InteractivityOp::eMathIsNaN:
    case InteractivityOp::eMathIsInf:
    case InteractivityOp::eMathSelect:
    case InteractivityOp::eMathSwitch:
    case InteractivityOp::eRefEq:
      return evaluatePureMath(instance, node, op);

    case InteractivityOp::eMathRandom:
      // Per-node cached (spec 1083: stable within one flow activation, refreshed on the next).
      return instance.randomValue(nodeIndex);

    case InteractivityOp::eTypeBoolToInt:
    case InteractivityOp::eTypeBoolToFloat:
    case InteractivityOp::eTypeIntToBool:
    case InteractivityOp::eTypeIntToFloat:
    case InteractivityOp::eTypeFloatToBool:
    case InteractivityOp::eTypeFloatToInt:
      return evaluateTypeConversion(instance, node, op);

    case InteractivityOp::eMathLength:
    case InteractivityOp::eMathNormalize:
    case InteractivityOp::eMathDot:
    case InteractivityOp::eMathCross:
    case InteractivityOp::eMathRotate2D:
    case InteractivityOp::eMathRotate3D:
    case InteractivityOp::eMathTransform:
    case InteractivityOp::eMathSlerp:
    case InteractivityOp::eMathTranspose:
    case InteractivityOp::eMathDeterminant:
    case InteractivityOp::eMathInverse:
    case InteractivityOp::eMathMatMul:
    case InteractivityOp::eMathMatCompose:
    case InteractivityOp::eMathMatDecompose:
    case InteractivityOp::eMathQuatConjugate:
    case InteractivityOp::eMathQuatMul:
    case InteractivityOp::eMathQuatAngleBetween:
    case InteractivityOp::eMathQuatFromAxisAngle:
    case InteractivityOp::eMathQuatToAxisAngle:
    case InteractivityOp::eMathQuatFromDirections:
    case InteractivityOp::eMathQuatFromUpForward:
    case InteractivityOp::eMathQuatFromAngles:
    case InteractivityOp::eMathQuatSlerp:
      return evaluateVectorMatrixQuat(instance, node, op, socketName);

    case InteractivityOp::eMathCombine2:
    case InteractivityOp::eMathCombine3:
    case InteractivityOp::eMathCombine4:
    case InteractivityOp::eMathCombine2x2:
    case InteractivityOp::eMathCombine3x3:
    case InteractivityOp::eMathCombine4x4:
    case InteractivityOp::eMathExtract2:
    case InteractivityOp::eMathExtract3:
    case InteractivityOp::eMathExtract4:
    case InteractivityOp::eMathExtract2x2:
    case InteractivityOp::eMathExtract3x3:
    case InteractivityOp::eMathExtract4x4:
      return evaluateSwizzle(instance, node, op, socketName);

    case InteractivityOp::eMathNot:
    case InteractivityOp::eMathAnd:
    case InteractivityOp::eMathOr:
    case InteractivityOp::eMathXor:
      return evaluateIntBoolLogic(instance, node, op);

    case InteractivityOp::eMathAsr:
    case InteractivityOp::eMathLsl:
    case InteractivityOp::eMathClz:
    case InteractivityOp::eMathCtz:
    case InteractivityOp::eMathPopcnt:
      return evaluateBitwise(instance, node, op);

    case InteractivityOp::eMathRgbToOkLCh:
    case InteractivityOp::eMathRgbFromOkLCh:
      return evaluateColor(instance, node, op, socketName);

    case InteractivityOp::ePointerGet:
      return evaluatePointerGet(instance, node, socketName);

    case InteractivityOp::eFlowFor: {
      if(socketName != "index")
        return std::monostate{};
      if(instance.hasNodeIntState(nodeIndex))
        return instance.nodeIntState(nodeIndex);
      auto it = node.configuration.find("initialIndex");
      if(it != node.configuration.end() && it->second.IsArray() && it->second.ArrayLen() > 0)
        return it->second.Get(size_t{0}).GetNumberAsInt();
      return int32_t(0);
    }

    case InteractivityOp::eFlowDoN:
      return socketName == "currentCount" ? InteractivityValue(instance.nodeIntState(nodeIndex)) :
                                            InteractivityValue(std::monostate{});

    case InteractivityOp::eFlowMultiGate:
      return socketName == "lastIndex" ? InteractivityValue(instance.multiGateState(nodeIndex).lastIndex) :
                                         InteractivityValue(std::monostate{});

    case InteractivityOp::eFlowWaitAll: {
      if(socketName != "remainingInputs")
        return std::monostate{};
      const auto& state = instance.waitAllState(nodeIndex);
      if(state.remaining >= 0)
        return state.remaining;
      auto it = node.configuration.find("inputFlows");
      if(it != node.configuration.end() && it->second.IsArray() && it->second.ArrayLen() > 0)
        return it->second.Get(size_t{0}).GetNumberAsInt();
      return int32_t(0);
    }

    case InteractivityOp::eFlowThrottle:
      return socketName == "lastRemainingTime" ? InteractivityValue(instance.throttleState(nodeIndex).lastRemainingTime) :
                                                 InteractivityValue(std::monostate{});

    case InteractivityOp::eFlowSetDelay:
      return socketName == "lastDelay" ? InteractivityValue(instance.nodeRefState(nodeIndex)) :
                                         InteractivityValue(std::monostate{});

    case InteractivityOp::eEventReceive: {
      if(socketName == "event")
        return instance.nodeOccurrenceRef(nodeIndex);
      if(const InteractivityValue* v = instance.nodeOccurrenceValue(nodeIndex, socketName))
        return *v;
      // No occurrence yet (or this socket isn't one of the event's declared values): the event's
      // declared initial value for it, if any (spec: outputs init to type-default/declared initial).
      auto cfgIt = node.configuration.find("event");
      if(cfgIt != node.configuration.end() && cfgIt->second.IsArray() && cfgIt->second.ArrayLen() > 0)
      {
        const int eventIndex = cfgIt->second.Get(size_t{0}).GetNumberAsInt();
        if(eventIndex >= 0 && eventIndex < static_cast<int>(instance.graph().events().size()))
        {
          const auto& values = instance.graph().events()[eventIndex].values;
          auto        valIt  = values.find(socketName);
          if(valIt != values.end())
            return valIt->second.initialValue;
        }
      }
      return std::monostate{};
    }

    // event/onHoverIn, event/onHoverOut (KHR_node_hoverability), event/onSelect
    // (KHR_node_selectability) - all three currently collapse into eExtensionDefined; distinguish
    // by the declaration's raw op string. Anything else extension-defined stays an unimplemented no-op.
    case InteractivityOp::eExtensionDefined: {
      const InteractivityDeclaration& decl    = instance.graph().declarations()[node.declarationIndex];
      const bool                      isHover = decl.extension == "KHR_node_hoverability"
                           && (decl.opString == "event/onHoverIn" || decl.opString == "event/onHoverOut");
      const bool isSelect = decl.extension == "KHR_node_selectability" && decl.opString == "event/onSelect";
      if(!isHover && !isSelect)
      {
        warnUnimplementedOnce(op);
        return std::monostate{};
      }
      if(socketName == "event")
        return instance.nodeOccurrenceRef(nodeIndex);
      if(const InteractivityValue* v = instance.nodeOccurrenceValue(nodeIndex, socketName))
        return *v;
      // No occurrence yet: spec-defined initial values (KHR_node_hoverability README:108,
      // KHR_node_selectability README:106), not a bare monostate.
      if(socketName == "hoveredNode" || socketName == "selectedNode")
        return InteractivityRef{};
      if(socketName == "controllerIndex")
        return int32_t(-1);
      if(isSelect && (socketName == "selectionPoint" || socketName == "selectionRayOrigin"))
        return glm::vec3(std::numeric_limits<float>::quiet_NaN());
      return std::monostate{};
    }

    default:
      warnUnimplementedOnce(op);
      return std::monostate{};
  }
}

void executeFlowNode(InteractivityGraphInstance& instance, int nodeIndex, const std::string& enteredSocket)
{
  if(nodeIndex < 0 || nodeIndex >= static_cast<int>(instance.graph().nodes().size()))
    return;

  const InteractivityNode& node = nodeAt(instance, nodeIndex);
  const InteractivityOp    op   = opOf(instance, node);

  switch(op)
  {
    case InteractivityOp::eFlowSequence: {
      // Spec "Socket Order": activate all output flows in ascending lexicographic order of id.
      std::map<std::string, InteractivityFlowSocket> ordered(node.flows.begin(), node.flows.end());
      for(const auto& [socketId, flow] : ordered)
        instance.activateFlow(flow.targetNode, flow.targetSocket);
      break;
    }

    case InteractivityOp::eFlowBranch: {
      const InteractivityValue cond  = instance.evaluateInput(node, "condition");
      const bool               taken = std::holds_alternative<bool>(cond) && std::get<bool>(cond);
      auto                     it    = node.flows.find(taken ? "true" : "false");
      if(it != node.flows.end())
        instance.activateFlow(it->second.targetNode, it->second.targetSocket);
      break;
    }

    case InteractivityOp::eVariableSet: {
      auto cfgIt = node.configuration.find("variables");
      if(cfgIt == node.configuration.end() || !cfgIt->second.IsArray())
        break;
      const tinygltf::Value& variables = cfgIt->second;
      for(size_t i = 0; i < variables.ArrayLen(); ++i)
      {
        const int varIndex = variables.Get(i).GetNumberAsInt();
        if(varIndex < 0 || varIndex >= static_cast<int>(instance.graph().variables().size()))
          continue;
        const InteractivityValue newValue = instance.evaluateInput(node, std::to_string(varIndex));
        if(!std::holds_alternative<std::monostate>(newValue))
          instance.variable(varIndex) = newValue;
      }
      auto it = node.flows.find("out");
      if(it != node.flows.end())
        instance.activateFlow(it->second.targetNode, it->second.targetSocket);
      break;
    }

    case InteractivityOp::ePointerSet:
      executePointerSet(instance, node);
      break;

    case InteractivityOp::eVariableInterpolate:
      executeVariableInterpolate(instance, node);
      break;

    case InteractivityOp::ePointerInterpolate:
      executePointerInterpolate(instance, node);
      break;

    case InteractivityOp::eAnimationStart:
      executeAnimationStart(instance, node);
      break;

    case InteractivityOp::eAnimationStop:
      executeAnimationStop(instance, node);
      break;

    case InteractivityOp::eAnimationStopAt:
      executeAnimationStopAt(instance, node);
      break;

    case InteractivityOp::eFlowSwitch: {
      const InteractivityValue sel = instance.evaluateInput(node, "selection");
      if(!std::holds_alternative<int32_t>(sel))
        break;
      const int32_t selValue = std::get<int32_t>(sel);
      bool          inCases  = false;
      auto          cfgIt    = node.configuration.find("cases");
      if(cfgIt != node.configuration.end() && cfgIt->second.IsArray())
      {
        for(size_t i = 0; i < cfgIt->second.ArrayLen(); ++i)
        {
          if(cfgIt->second.Get(i).GetNumberAsInt() == selValue)
          {
            inCases = true;
            break;
          }
        }
      }
      activateNamed(instance, node, inCases ? std::to_string(selValue).c_str() : "default");
      break;
    }

    case InteractivityOp::eFlowFor: {
      const InteractivityValue startV = instance.evaluateInput(node, "startIndex");
      if(!std::holds_alternative<int32_t>(startV))
        break;
      int32_t index = std::get<int32_t>(startV);
      while(true)
      {
        // Re-evaluated every iteration (spec: only endIndex is re-read on re-entry, not startIndex).
        const InteractivityValue endV = instance.evaluateInput(node, "endIndex");
        if(!std::holds_alternative<int32_t>(endV) || index >= std::get<int32_t>(endV))
          break;
        instance.nodeIntState(nodeIndex) = index;  // visible to loopBody's chain via evaluateNodeOutput
        auto it                          = node.flows.find("loopBody");
        if(it != node.flows.end())
          instance.activateFlow(it->second.targetNode, it->second.targetSocket);  // synchronous: runs to completion
        ++index;
      }
      instance.nodeIntState(nodeIndex) = index;
      activateNamed(instance, node, "completed");
      break;
    }

    case InteractivityOp::eFlowWhile: {
      while(true)
      {
        const InteractivityValue cond = instance.evaluateInput(node, "condition");
        if(!(std::holds_alternative<bool>(cond) && std::get<bool>(cond)))
          break;
        auto it = node.flows.find("loopBody");
        if(it != node.flows.end())
          instance.activateFlow(it->second.targetNode, it->second.targetSocket);  // synchronous: runs to completion
      }
      activateNamed(instance, node, "completed");
      break;
    }

    case InteractivityOp::eFlowDoN: {
      if(enteredSocket == "reset")
      {
        instance.nodeIntState(nodeIndex) = 0;
        break;
      }
      const InteractivityValue nVal = instance.evaluateInput(node, "n");
      if(!std::holds_alternative<int32_t>(nVal))
        break;
      int32_t& count = instance.nodeIntState(nodeIndex);
      if(count < std::get<int32_t>(nVal))
      {
        ++count;
        activateNamed(instance, node, "out");
      }
      break;
    }

    case InteractivityOp::eFlowMultiGate:
      executeMultiGate(instance, node, nodeIndex, enteredSocket);
      break;

    case InteractivityOp::eFlowWaitAll:
      executeWaitAll(instance, node, nodeIndex, enteredSocket);
      break;

    case InteractivityOp::eFlowThrottle: {
      InteractivityGraphInstance::ThrottleState& state = instance.throttleState(nodeIndex);
      if(enteredSocket == "reset")
      {
        state.hasFired          = false;
        state.lastRemainingTime = 0.0f;
        break;
      }
      const InteractivityValue durV = instance.evaluateInput(node, "duration");
      if(!std::holds_alternative<float>(durV))
      {
        activateNamed(instance, node, "err");
        break;
      }
      const float duration = std::get<float>(durV);
      if(!std::isfinite(duration) || duration < 0.0f)
      {
        activateNamed(instance, node, "err");
        break;
      }
      const float now = instance.timeSinceStart();
      if(!state.hasFired)
      {
        state.hasFired          = true;
        state.lastFireTimestamp = now;
        state.lastRemainingTime = 0.0f;
        activateNamed(instance, node, "out");
        break;
      }
      const float elapsed = now - state.lastFireTimestamp;
      if(duration <= elapsed)
      {
        state.lastFireTimestamp = now;
        state.lastRemainingTime = 0.0f;
        activateNamed(instance, node, "out");
      }
      else
      {
        state.lastRemainingTime = duration - elapsed;
      }
      break;
    }

    case InteractivityOp::eFlowSetDelay: {
      if(enteredSocket == "cancel")
      {
        instance.cancelDelaysForNode(nodeIndex);
        instance.nodeRefState(nodeIndex) = InteractivityRef{};
        break;
      }
      const InteractivityValue durV = instance.evaluateInput(node, "duration");
      if(!std::holds_alternative<float>(durV))
      {
        activateNamed(instance, node, "err");
        break;
      }
      const float duration = std::get<float>(durV);
      if(!std::isfinite(duration) || duration < 0.0f)
      {
        activateNamed(instance, node, "err");
        break;
      }
      const InteractivityRef ref{instance.allocateRefHandle()};
      instance.scheduleDelay(nodeIndex, ref, duration);
      instance.nodeRefState(nodeIndex) = ref;
      activateNamed(instance, node, "out");
      break;
    }

    case InteractivityOp::eFlowCancelDelay: {
      const InteractivityValue delayV = instance.evaluateInput(node, "delay");
      if(std::holds_alternative<InteractivityRef>(delayV))
        instance.cancelDelay(std::get<InteractivityRef>(delayV));
      activateNamed(instance, node, "out");  // always fires - spec: no `err` socket on this op
      break;
    }

    case InteractivityOp::eEventSend: {
      int32_t eventIndex = -1;
      if(auto it = node.configuration.find("event");
         it != node.configuration.end() && it->second.IsArray() && it->second.ArrayLen() > 0)
        eventIndex = it->second.Get(size_t{0}).GetNumberAsInt();
      if(eventIndex < 0 || eventIndex >= static_cast<int>(instance.graph().events().size()))
        break;  // op has no `err` socket; an invalid config should already have failed graph parse

      const InteractivityEventInfo& eventInfo = instance.graph().events()[eventIndex];

      std::unordered_map<std::string, InteractivityValue> values;
      for(const auto& [name, valueInfo] : eventInfo.values)
      {
        const InteractivityValue v = instance.evaluateInput(node, name);
        values[name]               = std::holds_alternative<std::monostate>(v) ? valueInfo.initialValue : v;
      }
      instance.sendEvent(eventIndex, std::move(values));
      activateNamed(instance, node, "out");
      break;
    }

    case InteractivityOp::eEventStopPropagation: {
      // Spec 4492-4530: an invalid/non-ref `event` input still activates `out` (step 2), it just
      // has nothing to cancel - InteractivityGraphInstance::stopEventPropagation() itself no-ops
      // for a ref that isn't a real event occurrence (isEventRef() false), so no separate check
      // is needed here.
      const InteractivityValue eventV         = instance.evaluateInput(node, "event");
      const InteractivityValue stopImmediateV = instance.evaluateInput(node, "stopImmediate");
      if(std::holds_alternative<InteractivityRef>(eventV))
      {
        const bool stopImmediate = std::holds_alternative<bool>(stopImmediateV) && std::get<bool>(stopImmediateV);
        instance.stopEventPropagation(std::get<InteractivityRef>(eventV), stopImmediate);
      }
      activateNamed(instance, node, "out");
      break;
    }

    case InteractivityOp::eDebugLog: {
      std::string messageTemplate;
      if(auto it = node.configuration.find("message"); it != node.configuration.end() && it->second.IsArray()
                                                       && it->second.ArrayLen() > 0 && it->second.Get(size_t{0}).IsString())
        messageTemplate = it->second.Get(size_t{0}).Get<std::string>();
      const std::string message = formatDebugMessage(instance, node, messageTemplate);
      LOGI("KHR_interactivity debug/log: %s\n", message.c_str());
      instance.appendLogEntry(0, message);
      activateNamed(instance, node, "out");
      break;
    }

    default:
      warnUnimplementedOnce(op);
      break;
  }
}

}  // namespace nvvkgltf
