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

//
// Runtime state for one playing instance of a KHR_interactivity behavior graph: variable values,
// the flow-execution queue, and per-tick timing state. InteractivityGraph
// (gltf_interactivity_graph.cpp) is the compiled, immutable graph this reads; Play/Pause/Reset only
// ever touch an instance, so re-running a graph from scratch is just constructing a fresh one. See
// docs/interactivity.md.
//

#include <algorithm>
#include <cmath>
#include <limits>

#include <nvutils/logger.hpp>

#include "gltf_interactivity_animation.hpp"
#include "gltf_interactivity_eval.hpp"
#include "gltf_interactivity_instance.hpp"
#include "gltf_interactivity_pointer.hpp"

namespace {
// Spec's requested-to-effective-timestamp remap (animation/start ~4284-4297): wraps an arbitrary
// (possibly negative, possibly beyond T) requested timestamp `r` into the [0, T] range the stored
// animation data actually covers, supporting both forward and backward looping.
float effectiveAnimationTimestamp(float r, float T)
{
  if(T == 0.0f)
    return 0.0f;
  const float s = (r > 0.0f) ? std::ceil((r - T) / T) : std::floor(r / T);
  return r - s * T;
}
}  // namespace

namespace nvvkgltf {

InteractivityGraphInstance::InteractivityGraphInstance(const InteractivityGraph& graph)
    : m_graph(graph)
{
  reset();
}

void InteractivityGraphInstance::reset()
{
  m_variables.clear();
  m_variables.reserve(m_graph.variables().size());
  for(const InteractivityVariableInfo& v : m_graph.variables())
    m_variables.push_back(v.initialValue);

  m_callDepth         = 0;
  m_started           = false;
  m_hasTicked         = false;
  m_timeSinceStart    = 0.0f;
  m_timeSinceLastTick = std::numeric_limits<float>::quiet_NaN();

  m_nodeIntState.clear();
  m_nodeRefState.clear();
  m_multiGateState.clear();
  m_waitAllState.clear();
  m_throttleState.clear();
  m_pendingDelays.clear();
  m_variableInterpolations.clear();
  m_pointerInterpolations.clear();
  m_animationPlaybacks.clear();
  m_animationApplies.clear();
  m_nodeOccurrences.clear();
  m_nextRefHandle = 1;
  m_eventRefHandles.clear();
  m_transitivePropagationStopped.clear();
  m_immediatePropagationStopped.clear();
  m_currentStartRef = InteractivityRef{};
  m_currentTickRef  = InteractivityRef{};
  m_logEntries.clear();
}

void InteractivityGraphInstance::activateFlow(int nodeIndex, const std::string& socketName)
{
  if(nodeIndex < 0 || nodeIndex >= static_cast<int>(m_graph.nodes().size()))
    return;
  if(m_callDepth >= kMaxCallDepth)
  {
    LOGW("KHR_interactivity: graph '%s' exceeded %d nested flow activations, treating node %d's activation as a no-op\n",
         m_graph.name().c_str(), kMaxCallDepth, nodeIndex);
    return;
  }
  ++m_flowActivationCounter;  // math/random's per-activation cache invalidation (spec 1083)
  ++m_callDepth;
  executeFlowNode(*this, nodeIndex, socketName);
  --m_callDepth;
}

float InteractivityGraphInstance::randomValue(int nodeIndex)
{
  auto it = m_randomCache.find(nodeIndex);
  if(it != m_randomCache.end() && it->second.second == m_flowActivationCounter)
    return it->second.first;
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float                           v = dist(m_rng);
  m_randomCache[nodeIndex]                = {v, m_flowActivationCounter};
  return v;
}

void InteractivityGraphInstance::scheduleDelay(int nodeIndex, InteractivityRef ref, float delaySeconds)
{
  m_pendingDelays.push_back({nodeIndex, ref, m_timeSinceStart + delaySeconds});
}

void InteractivityGraphInstance::cancelDelay(InteractivityRef ref)
{
  std::erase_if(m_pendingDelays, [&](const PendingDelay& d) { return d.ref == ref; });
}

bool InteractivityGraphInstance::isPendingDelayRef(InteractivityRef ref) const
{
  return std::any_of(m_pendingDelays.begin(), m_pendingDelays.end(), [&](const PendingDelay& d) { return d.ref == ref; });
}

void InteractivityGraphInstance::cancelDelaysForNode(int nodeIndex)
{
  std::erase_if(m_pendingDelays, [&](const PendingDelay& d) { return d.nodeIndex == nodeIndex; });
}

void InteractivityGraphInstance::advancePendingDelays()
{
  // Partition instead of erasing one-by-one so a `done` handler that itself calls setDelay/
  // cancelDelay (mutating m_pendingDelays) doesn't invalidate this loop's iterators.
  std::vector<PendingDelay> fired;
  std::erase_if(m_pendingDelays, [&](const PendingDelay& d) {
    if(m_timeSinceStart < d.activationTime)
      return false;
    fired.push_back(d);
    return true;
  });
  for(const PendingDelay& d : fired)
  {
    const InteractivityNode& node = m_graph.nodes()[d.nodeIndex];
    auto                     it   = node.flows.find("done");
    if(it != node.flows.end())
      activateFlow(it->second.targetNode, it->second.targetSocket);
  }
}

void InteractivityGraphInstance::startVariableInterpolation(VariableInterpolationState state)
{
  std::erase_if(m_variableInterpolations,
                [&](const VariableInterpolationState& s) { return s.variableIndex == state.variableIndex; });
  m_variableInterpolations.push_back(std::move(state));
}

void InteractivityGraphInstance::startPointerInterpolation(PointerInterpolationState state)
{
  std::erase_if(m_pointerInterpolations, [&](const PointerInterpolationState& s) { return s.path == state.path; });
  m_pointerInterpolations.push_back(std::move(state));
}

// variable/interpolate (spec 3739-3845): "on each tick, for each entry ..." - t<=0 skips (the
// activation tick itself never applies a value), t NaN/>=1 snaps to the target and fires `done`,
// otherwise the value is eased via cubicBezierEase()/lerpInteractivityValue(). Two-pass (collect
// `fired` during erase_if, activate afterward) for the same reason advancePendingDelays() is:
// a `done` handler could itself start a new interpolation, mutating the vector mid-iteration.
void InteractivityGraphInstance::advanceVariableInterpolations()
{
  std::vector<VariableInterpolationState> fired;
  std::erase_if(m_variableInterpolations, [&](VariableInterpolationState& s) {
    const float t = (m_timeSinceStart - s.startTime) / s.duration;
    if(t <= 0.0f)
      return false;
    if(std::isnan(t) || t >= 1.0f)
    {
      m_variables[s.variableIndex] = s.targetValue;
      fired.push_back(s);
      return true;
    }
    const float q                = cubicBezierEase(s.p1, s.p2, t);
    m_variables[s.variableIndex] = lerpInteractivityValue(s.startValue, s.targetValue, q, s.useSlerp);
    return false;
  });
  for(const VariableInterpolationState& s : fired)
    if(s.doneTargetNode >= 0)
      activateFlow(s.doneTargetNode, s.doneTargetSocket);
}

// pointer/interpolate (spec 4154-4254): same timing model as variable/interpolate above, writing
// through InteractivityPointerResolver::set() instead of a variable slot. A resolver that's missing,
// or a write that fails (e.g. the target node was deleted), drops the entry immediately instead of
// letting it ride to t>=1.0 - `done` only fires for an interpolation whose write actually succeeded.
void InteractivityGraphInstance::advancePointerInterpolations()
{
  std::vector<PointerInterpolationState> fired;
  std::erase_if(m_pointerInterpolations, [&](PointerInterpolationState& s) {
    const float t = (m_timeSinceStart - s.startTime) / s.duration;
    if(t <= 0.0f)
      return false;
    if(std::isnan(t) || t >= 1.0f)
    {
      if(!m_pointerResolver || !m_pointerResolver->set(s.path, s.targetValue))
        return true;  // failed write - drop silently, do not fire `done`
      fired.push_back(s);
      return true;
    }
    if(!m_pointerResolver
       || !m_pointerResolver->set(s.path, lerpInteractivityValue(s.startValue, s.targetValue,
                                                                 cubicBezierEase(s.p1, s.p2, t), s.useSlerp)))
      return true;  // failed write - drop silently, do not fire `done`
    return false;
  });
  for(const PointerInterpolationState& s : fired)
    if(s.doneTargetNode >= 0)
      activateFlow(s.doneTargetNode, s.doneTargetSocket);
}

void InteractivityGraphInstance::startAnimation(AnimationPlaybackState state)
{
  std::erase_if(m_animationPlaybacks, [&](const AnimationPlaybackState& s) { return s.animation == state.animation; });
  m_animationPlaybacks.push_back(std::move(state));
}

void InteractivityGraphInstance::stopAnimation(InteractivityRef animRef)
{
  std::erase_if(m_animationPlaybacks, [&](const AnimationPlaybackState& s) { return s.animation == animRef; });
}

void InteractivityGraphInstance::scheduleAnimationStop(InteractivityRef animRef, float stopTime, int doneTargetNode, std::string doneTargetSocket)
{
  for(AnimationPlaybackState& s : m_animationPlaybacks)
  {
    if(s.animation == animRef)
    {
      s.stopTime             = stopTime;
      s.stopDoneTargetNode   = doneTargetNode;
      s.stopDoneTargetSocket = std::move(doneTargetSocket);
      return;
    }
  }
}

// animation/start (spec ~4340-4368): "on each asset animation update, for each entry..." - computes
// the current timestamp from elapsed real time * speed (direction-aware), checks it against the
// stop time (animation/stopAt) and end time in that order, and otherwise applies the ongoing pose.
// The spec applies the pose to the asset BEFORE activating `done`/stop-`done` - a pointer/get or
// debug/log reached transitively from that same activation must already see the fresh value - so
// applyPose() is called synchronously here, inside the erase_if lambda, strictly before the
// `fired` flows are activated in the loop below (not merged into that loop: a done/stopDone handler
// could itself start a new animation, mutating m_animationPlaybacks mid-iteration, same reason
// advancePendingDelays() two-passes).
void InteractivityGraphInstance::advanceAnimations()
{
  m_animationApplies.clear();
  if(!m_animationResolver)
    return;

  struct FiredDone
  {
    int         targetNode;
    std::string targetSocket;
  };
  std::vector<FiredDone> fired;

  auto applyAndRecord = [&](const AnimationPlaybackState& s, float requestedTime, float T) {
    const float t = effectiveAnimationTimestamp(requestedTime, T);
    m_animationResolver->applyPose(s.animation.handle, t);
    m_animationApplies.push_back({s.animation.handle, t});
  };

  std::erase_if(m_animationPlaybacks, [&](AnimationPlaybackState& s) {
    const float T = m_animationResolver->animationMaxTime(s.animation.handle);

    if(s.startTime == s.endTime)
    {
      applyAndRecord(s, s.startTime, T);
      fired.push_back({s.doneTargetNode, s.doneTargetSocket});
      return true;
    }

    const float elapsed       = m_timeSinceStart - s.entryCreationTime;
    const bool  forward       = s.startTime < s.endTime;
    float       scaledElapsed = elapsed * s.speed;
    if(!forward)
      scaledElapsed = -scaledElapsed;
    const float currentTimestamp = s.startTime + scaledElapsed;

    const bool reachedStop = forward ?
                                 (currentTimestamp >= s.stopTime && s.stopTime >= s.startTime && s.stopTime < s.endTime) :
                                 (currentTimestamp <= s.stopTime && s.stopTime <= s.startTime && s.stopTime > s.endTime);
    if(reachedStop)
    {
      applyAndRecord(s, s.stopTime, T);
      fired.push_back({s.stopDoneTargetNode, s.stopDoneTargetSocket});
      return true;
    }

    const bool reachedEnd = forward ? (currentTimestamp >= s.endTime) : (currentTimestamp <= s.endTime);
    if(reachedEnd)
    {
      applyAndRecord(s, s.endTime, T);
      fired.push_back({s.doneTargetNode, s.doneTargetSocket});
      return true;
    }

    applyAndRecord(s, currentTimestamp, T);
    return false;
  });

  for(const FiredDone& f : fired)
    if(f.targetNode >= 0)
      activateFlow(f.targetNode, f.targetSocket);
}

void InteractivityGraphInstance::setNodeOccurrence(int nodeIndex, std::unordered_map<std::string, InteractivityValue> values, InteractivityRef eventRef)
{
  m_nodeOccurrences[nodeIndex] = {std::move(values), eventRef};
}

const InteractivityValue* InteractivityGraphInstance::nodeOccurrenceValue(int nodeIndex, const std::string& socketName) const
{
  auto occIt = m_nodeOccurrences.find(nodeIndex);
  if(occIt == m_nodeOccurrences.end())
    return nullptr;
  auto valIt = occIt->second.values.find(socketName);
  return valIt != occIt->second.values.end() ? &valIt->second : nullptr;
}

InteractivityRef InteractivityGraphInstance::nodeOccurrenceRef(int nodeIndex) const
{
  auto it = m_nodeOccurrences.find(nodeIndex);
  return it != m_nodeOccurrences.end() ? it->second.eventRef : InteractivityRef{};
}

void InteractivityGraphInstance::activateBoundHandlers(const std::unordered_map<int, std::vector<int>>& handlerMap,
                                                       int                                              glTFNodeIndex,
                                                       std::unordered_map<std::string, InteractivityValue> values,
                                                       InteractivityRef occurrenceRef)
{
  auto it = handlerMap.find(glTFNodeIndex);
  if(it == handlerMap.end())
    return;

  // "event" is carried via NodeOccurrence's dedicated eventRef field (read through
  // nodeOccurrenceRef()), matching event/receive's existing convention - not duplicated into `values`.
  for(int handlerNodeIndex : it->second)
  {
    if(isImmediatePropagationStopped(occurrenceRef))
      return;  // event/stopPropagation(stopImmediate=true) - rest of this occurrence is cancelled
    setNodeOccurrence(handlerNodeIndex, values, occurrenceRef);
    const InteractivityNode& handlerNode = m_graph.nodes()[handlerNodeIndex];
    auto                     flowIt      = handlerNode.flows.find("out");
    if(flowIt != handlerNode.flows.end())
      activateFlow(flowIt->second.targetNode, flowIt->second.targetSocket);
  }
}

bool InteractivityGraphInstance::sendEvent(int eventIndex, std::unordered_map<std::string, InteractivityValue> values)
{
  if(eventIndex < 0 || eventIndex >= static_cast<int>(m_graph.events().size()))
    return false;

  const InteractivityEventInfo& eventInfo = m_graph.events()[eventIndex];
  for(const auto& [name, valueInfo] : eventInfo.values)
    if(!values.contains(name))
      values[name] = valueInfo.initialValue;

  const InteractivityRef occurrenceRef = allocateEventRef();

  // Deliver to every event/receive node declared against this event index, in ascending
  // node-index (JSON declaration) order (spec: activate in JSON order).
  const auto& nodes = m_graph.nodes();
  const auto& decls = m_graph.declarations();
  for(size_t i = 0; i < nodes.size(); ++i)
  {
    if(isImmediatePropagationStopped(occurrenceRef))
      break;  // event/stopPropagation(stopImmediate=true) - remaining event/receive nodes skipped
    const InteractivityNode& candidate = nodes[i];
    if(decls[candidate.declarationIndex].op != InteractivityOp::eEventReceive)
      continue;
    auto cfgIt = candidate.configuration.find("event");
    if(cfgIt == candidate.configuration.end() || !cfgIt->second.IsArray() || cfgIt->second.ArrayLen() == 0
       || cfgIt->second.Get(size_t{0}).GetNumberAsInt() != eventIndex)
      continue;
    setNodeOccurrence(static_cast<int>(i), values, occurrenceRef);
    auto flowIt = candidate.flows.find("out");
    if(flowIt != candidate.flows.end())
      activateFlow(flowIt->second.targetNode, flowIt->second.targetSocket);
  }
  return true;
}

void InteractivityGraphInstance::stopEventPropagation(InteractivityRef ref, bool stopImmediate)
{
  if(!isEventRef(ref))
    return;
  m_transitivePropagationStopped.insert(ref.handle);
  if(stopImmediate)
    m_immediatePropagationStopped.insert(ref.handle);
}

bool InteractivityGraphInstance::isTransitivePropagationStopped(InteractivityRef ref) const
{
  return m_transitivePropagationStopped.contains(ref.handle);
}

bool InteractivityGraphInstance::isImmediatePropagationStopped(InteractivityRef ref) const
{
  return m_immediatePropagationStopped.contains(ref.handle);
}

void InteractivityGraphInstance::appendLogEntry(int32_t severity, std::string message)
{
  m_logEntries.push_back({severity, std::move(message)});
  if(m_logEntries.size() > kMaxLogEntries)
    m_logEntries.pop_front();
}

void InteractivityGraphInstance::start()
{
  if(m_started)
    return;
  m_started = true;

  // One shared occurrence ref for every event/onStart node in this run (spec: conceptually a
  // single "start" event, not one per node - see docs/interactivity.md's design note).
  m_currentStartRef = allocateEventRef();

  // Each onStart node's entire chain runs to completion (activateFlow is synchronous) before the
  // next onStart node begins - spec: "function pointer"/"method call" semantics, not concurrent or
  // interleaved (Specification.adoc ~219).
  for(int nodeIndex : m_graph.onStartNodes())
  {
    if(isImmediatePropagationStopped(m_currentStartRef))
      break;  // event/stopPropagation(stopImmediate=true) - remaining onStart nodes skipped
    const InteractivityNode& node = m_graph.nodes()[nodeIndex];
    auto                     it   = node.flows.find("out");
    if(it != node.flows.end())
      activateFlow(it->second.targetNode, it->second.targetSocket);
  }
}

void InteractivityGraphInstance::tick(float deltaSeconds)
{
  if(!m_started)
    start();

  m_timeSinceLastTick = m_hasTicked ? deltaSeconds : std::numeric_limits<float>::quiet_NaN();
  m_timeSinceStart += m_hasTicked ? deltaSeconds : 0.0f;
  m_hasTicked = true;

  advancePendingDelays();  // flow/setDelay callbacks whose activationTime has now passed
  advanceVariableInterpolations();
  advancePointerInterpolations();
  advanceAnimations();

  // One shared occurrence ref for every event/onTick node THIS tick (changes tick to tick, same
  // sharing rationale as m_currentStartRef above).
  m_currentTickRef = allocateEventRef();

  // Same run-to-completion-before-the-next-one ordering as start() above.
  for(int nodeIndex : m_graph.onTickNodes())
  {
    if(isImmediatePropagationStopped(m_currentTickRef))
      break;  // event/stopPropagation(stopImmediate=true) - remaining onTick nodes skipped
    const InteractivityNode& node = m_graph.nodes()[nodeIndex];
    auto                     it   = node.flows.find("out");
    if(it != node.flows.end())
      activateFlow(it->second.targetNode, it->second.targetSocket);
  }
}

InteractivityValue InteractivityGraphInstance::evaluateInput(const InteractivityNode& node, const std::string& socketName) const
{
  auto it = node.values.find(socketName);
  if(it == node.values.end())
    return std::monostate{};

  const InteractivityValueSocket& socket = it->second;
  if(!socket.isReference)
    return socket.literal;

  // const_cast is safe here: evaluateNodeOutput only mutates instance-owned runtime state
  // (nothing reachable from `node`, which belongs to the immutable compiled graph).
  return evaluateNodeOutput(const_cast<InteractivityGraphInstance&>(*this), socket.sourceNode, socket.sourceSocket);
}

}  // namespace nvvkgltf
