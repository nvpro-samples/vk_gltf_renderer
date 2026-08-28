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
// Runtime state for one playing instance of a KHR_interactivity behavior graph.
//
// InteractivityGraph (gltf_interactivity_graph.hpp) is the compiled, immutable graph; this class
// is the *runtime* half - variable values, the flow-execution queue, and per-tick timing state.
// Play/Pause/Reset only ever touch an instance, never the compiled graph, so re-running a graph
// from scratch is just constructing a fresh instance. See docs/interactivity.md.
//

#pragma once

#include <deque>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gltf_interactivity_graph.hpp"

namespace nvvkgltf {

class InteractivityPointerResolver;
class InteractivityAnimationResolver;

class InteractivityGraphInstance
{
public:
  explicit InteractivityGraphInstance(const InteractivityGraph& graph);

  const InteractivityGraph& graph() const { return m_graph; }

  // Optional glTF-model read/write surface for pointer/get, pointer/set, pointer/interpolate.
  // Not owned; nullptr (the default) means those ops safely resolve to "unresolvable" - a graph
  // with no scene attached (e.g. a unit test exercising pure math/flow logic) still runs.
  void setPointerResolver(InteractivityPointerResolver* resolver) { m_pointerResolver = resolver; }
  InteractivityPointerResolver* pointerResolver() const { return m_pointerResolver; }

  // Optional glTF-animation validity/duration query surface for animation/start/stop/stopAt. Same
  // "nullptr is safe, ops just resolve to err/no-op" contract as the pointer resolver above.
  void setAnimationResolver(InteractivityAnimationResolver* resolver) { m_animationResolver = resolver; }
  InteractivityAnimationResolver* animationResolver() const { return m_animationResolver; }

  // Activates every `event/onStart` node (JSON order, spec 4461) and drains the resulting flow
  // queue. Idempotent-safe to call only once per instance lifetime; reset() before calling again.
  void start();

  // Advances one tick: fires any flow/setDelay callbacks whose time has come, activates every
  // `event/onTick` node (spec requires this only after all onStart activations - start() must
  // have been called first), then drains the resulting queue.
  void tick(float deltaSeconds);

  // Discards all runtime state (variables reset to initial values, queue cleared, timers reset,
  // all flow-op and delay/event state cleared). The next start() replays the graph from scratch.
  void reset();

  bool  started() const { return m_started; }
  bool  ticked() const { return m_hasTicked; }
  float timeSinceStart() const { return m_timeSinceStart; }
  float timeSinceLastTick() const { return m_timeSinceLastTick; }

  // The single event/onStart occurrence's ref, shared by every event/onStart node's "event" output
  // socket in the graph (spec: there is conceptually one "start" event per run, not one per node -
  // see docs/interactivity.md). Minted once in start(), before any onStart node activates.
  InteractivityRef currentStartRef() const { return m_currentStartRef; }
  // The current tick's event/onTick occurrence ref, shared by every event/onTick node's "event"
  // output socket for THIS tick (changes every tick() call, same sharing rationale as above).
  InteractivityRef currentTickRef() const { return m_currentTickRef; }

  InteractivityValue&       variable(int index) { return m_variables.at(index); }
  const InteractivityValue& variable(int index) const { return m_variables.at(index); }

  // Evaluates one of `node`'s input value sockets: pulls (recursively evaluates) a referenced
  // node's output, or returns the inline literal. Returns the socket's type default if the
  // socket is absent from the node's JSON (defensive - a spec-valid graph always has it wired
  // for sockets a supported op actually reads).
  InteractivityValue evaluateInput(const InteractivityNode& node, const std::string& socketName) const;

  // Activates `socketName` (an input flow socket) on `nodeIndex`: executes it immediately and
  // synchronously, and everything it transitively triggers, before returning - matching the spec's
  // own "output flow sockets represent 'function pointers' the node will call" / "method" language
  // (Specification.adoc ~219) and flow/sequence's explicit "each output flow is activated after the
  // previous one completes" rule. This is real C++ recursion (call depth == transitive flow-chain
  // depth), guarded by a conservative call-depth cap (see .cpp) rather than the unbounded-stack risk
  // a spec-conformant graph with backward flow references (see docs/interactivity.md) could pose -
  // a graph doesn't have to be malicious to hit this, just deep, so this fails soft (warns, no-ops)
  // rather than crashing. There used to be a separate runFlowChainToCompletion() for flow/for and
  // flow/while's "self-activate only after loopBody fully completes" needs, worked around via a
  // temporary shared-queue swap; with activateFlow() itself now synchronous, plain activateFlow()
  // already gives them exactly that, so the workaround (and the shared queue it swapped) is gone.
  void activateFlow(int nodeIndex, const std::string& socketName);

  // math/random (spec 1066-1088): the value of a given random node MUST stay stable across
  // repeated pure-value reads within one flow-node execution, and change on the next flow
  // activation (including self-activations). Cached per node index, keyed to a counter bumped once
  // per activateFlow() call.
  float randomValue(int nodeIndex);

  // Persistent per-node int state: flow/for's `index`, flow/doN's `currentCount`. Default-
  // constructs to 0 on first access; never collides across ops since keyed by node index.
  int32_t& nodeIntState(int nodeIndex) { return m_nodeIntState[nodeIndex]; }
  bool     hasNodeIntState(int nodeIndex) const { return m_nodeIntState.find(nodeIndex) != m_nodeIntState.end(); }

  // Persistent per-node ref state: flow/setDelay's `lastDelay`. Default-constructs to a null ref.
  InteractivityRef& nodeRefState(int nodeIndex) { return m_nodeRefState[nodeIndex]; }

  struct MultiGateState
  {
    int32_t           lastIndex = -1;
    std::vector<bool> used;  // sized to the node's output flow count on first use
  };
  MultiGateState& multiGateState(int nodeIndex) { return m_multiGateState[nodeIndex]; }

  struct WaitAllState
  {
    int32_t           remaining = -1;  // -1 = not yet initialized from the node's `inputFlows` config
    std::vector<bool> used;
  };
  WaitAllState& waitAllState(int nodeIndex) { return m_waitAllState[nodeIndex]; }

  struct ThrottleState
  {
    bool  hasFired          = false;
    float lastFireTimestamp = 0.0f;  // instance.timeSinceStart() at the last successful `in`
    float lastRemainingTime = 0.0f;
  };
  ThrottleState& throttleState(int nodeIndex) { return m_throttleState[nodeIndex]; }

  // Mints a fresh, instance-unique ref handle (flow/setDelay's `lastDelay`). Prefer allocateEventRef()
  // for anything that is conceptually an *event* occurrence (event/send, hover/select) - it also
  // registers the handle so the spec's `/extensions/KHR_interactivity/events/{}` self-referential
  // pointer (see isEventRef()) can recognize it later, even after the occurrence itself is long gone.
  int32_t allocateRefHandle() { return m_nextRefHandle++; }

  // Mints an event-kind ref (spec "Event References" section) and remembers it permanently, so
  // isEventRef() keeps reporting true for it indefinitely - the spec explicitly says the ref's
  // validity here does NOT depend on the event object's current internal state.
  InteractivityRef allocateEventRef()
  {
    const InteractivityRef ref{allocateRefHandle()};
    m_eventRefHandles.insert(ref.handle);
    return ref;
  }
  // `/extensions/KHR_interactivity/events/{}`: true iff `ref` was ever produced by an event
  // operation (event/onStart, event/onTick, event/send, or - Phase E - hover/select), regardless of
  // whether that specific occurrence is still "current".
  bool isEventRef(InteractivityRef ref) const { return m_eventRefHandles.contains(ref.handle); }
  // `/extensions/KHR_interactivity/delays/{}`: true iff `ref` is CURRENTLY in the dynamic array of
  // pending flow/setDelay activations (unlike isEventRef(), this one does depend on live state).
  bool isPendingDelayRef(InteractivityRef ref) const;

  // flow/setDelay / flow/cancelDelay support. Fired delays activate `nodeIndex`'s "done" flow
  // socket from advancePendingDelays() (called once per tick(), before that tick's own onTick nodes).
  void scheduleDelay(int nodeIndex, InteractivityRef ref, float delaySeconds);
  void cancelDelay(InteractivityRef ref);   // silently no-ops if not found (spec: no `err` here)
  void cancelDelaysForNode(int nodeIndex);  // flow/setDelay's own `cancel` flow: all of ITS pending refs

  // variable/interpolate / pointer/interpolate support (spec "... interpolation state dynamic
  // array"): one entry per in-flight interpolation, advanced once per tick() (before that tick's
  // onTick nodes, same as delay handling - see advanceVariableInterpolations()/
  // advancePointerInterpolations() in the .cpp). `doneTargetNode`/`doneTargetSocket` are the
  // *resolved* flow target the spec describes as "an implementation-specific pointer to the done
  // output flow" - resolved once at start time by gltf_interactivity_eval.cpp's
  // executeVariableInterpolate()/executePointerInterpolate() rather than re-looked-up by node index.
  struct VariableInterpolationState
  {
    int                variableIndex;
    float              startTime;
    float              duration;
    InteractivityValue startValue;
    InteractivityValue targetValue;
    glm::vec2          p1, p2;
    bool               useSlerp       = false;
    int                doneTargetNode = -1;
    std::string        doneTargetSocket;
  };
  struct PointerInterpolationState
  {
    std::string        path;
    float              startTime;
    float              duration;
    InteractivityValue startValue;
    InteractivityValue targetValue;
    glm::vec2          p1, p2;
    bool               useSlerp       = false;
    int                doneTargetNode = -1;
    std::string        doneTargetSocket;
  };
  // Starts a new interpolation, replacing any existing one for the same variable/path first (spec:
  // "If the ... state dynamic array contains an entry with the same variable reference/effective
  // JSON Pointer, remove it from the array").
  void startVariableInterpolation(VariableInterpolationState state);
  void startPointerInterpolation(PointerInterpolationState state);

  // animation/start / animation/stop / animation/stopAt support (spec 4262-4441): one entry per
  // active glTF animation, advanced once per tick() (advanceAnimations(), same slot as the delay/
  // interpolation advances above). Unlike pointer/variable interpolation, this engine does NOT apply
  // the resulting pose itself - see gltf_interactivity_animation.hpp for why (it's a GPU-pipeline
  // operation, not a single field write) - it only computes *what* to apply, surfaced via
  // pendingAnimationApplies() for GltfRenderer to drain each frame. `doneTargetNode`/
  // `doneTargetSocket` (fired when `endTime` is reached) and `stopDoneTargetNode`/
  // `stopDoneTargetSocket` (fired when `stopTime` is reached, via animation/stopAt) are resolved
  // flow targets, same convention as the interpolation states above.
  struct AnimationPlaybackState
  {
    InteractivityRef animation;
    float            startTime         = 0.0f;
    float            endTime           = 0.0f;
    float            stopTime          = 0.0f;  // == endTime until animation/stopAt overrides it
    float            speed             = 1.0f;
    float            entryCreationTime = 0.0f;  // instance.timeSinceStart() when this entry was added
    int              doneTargetNode    = -1;
    std::string      doneTargetSocket;
    int              stopDoneTargetNode = -1;
    std::string      stopDoneTargetSocket;
  };
  // animation/start: replaces any existing entry for the same animation reference first, without
  // firing its `done` (spec step 6 - "the previously set done flows MUST NOT be activated").
  void startAnimation(AnimationPlaybackState state);
  // animation/stop: removes the entry for `animRef` if present; its `done` is NOT fired (spec step
  // 3). No-ops (still spec-legal) if nothing is currently playing that reference.
  void stopAnimation(InteractivityRef animRef);
  // animation/stopAt: updates the stop time/stop-completion target of the entry for `animRef` if one
  // exists; no-ops (spec step 4's "if... contains an entry... update" - silently skipped otherwise).
  void scheduleAnimationStop(InteractivityRef animRef, float stopTime, int doneTargetNode, std::string doneTargetSocket);

  // (animationIndex, effectiveTimestamp) pairs that changed this tick and need their pose
  // (re-)applied - rebuilt from scratch by advanceAnimations() every tick(), not accumulated.
  // GltfRenderer drains this right after Scene::tickInteractivityGraphs().
  struct AnimationApply
  {
    int   animationIndex;
    float effectiveTime;
  };
  const std::vector<AnimationApply>& pendingAnimationApplies() const { return m_animationApplies; }

  // Per-node "last activation occurrence" output-value storage, shared by every op whose output
  // value sockets are populated once per external activation rather than computed from inputs:
  // event/receive, and (Phase E) event/onHoverIn/onHoverOut/onSelect. Values are wholesale-replaced
  // (not merged) on every occurrence, per spec: a value not set on a given occurrence reverts to
  // its declared default, it isn't sticky.
  void setNodeOccurrence(int nodeIndex, std::unordered_map<std::string, InteractivityValue> values, InteractivityRef eventRef);
  const InteractivityValue* nodeOccurrenceValue(int nodeIndex, const std::string& socketName) const;
  InteractivityRef          nodeOccurrenceRef(int nodeIndex) const;

  // Activates every event/onHoverIn/onHoverOut/onSelect handler node bound (via that handler
  // node's own `configuration.nodeIndex`) to `glTFNodeIndex`, in ascending JSON order (spec:
  // same-config handlers activate in declaration order). `handlerMap` selects which binding table
  // to use - pass `graph().hoverInHandlers()`/`hoverOutHandlers()`/`selectHandlers()`. `values`
  // supplies every *other* output value socket (e.g. `hoveredNode`/`controllerIndex` for hover,
  // `selectedNode`/`controllerIndex`/`selectionPoint`/`selectionRayOrigin` for select). `occurrenceRef`
  // is the shared event ref for this whole notification (e.g. one whole ancestor-bubble walk, not one
  // per ancestor - see Scene::notifyNodeSelected/notifyNodeHoverChanged, which mint it once and pass
  // it to every activateBoundHandlers() call in that bubble) - this is what lets event/stopPropagation
  // recognize "the rest of this same occurrence" and cancel it. Stops partway through `glTFNodeIndex`'s
  // own handler list if an earlier handler in this call already stopped immediate propagation for
  // `occurrenceRef`. No-ops if nothing is bound to `glTFNodeIndex`.
  void activateBoundHandlers(const std::unordered_map<int, std::vector<int>>&    handlerMap,
                             int                                                 glTFNodeIndex,
                             std::unordered_map<std::string, InteractivityValue> values,
                             InteractivityRef                                    occurrenceRef);

  // Sends a custom event (spec event/send + event/receive): mints an occurrence ref and delivers
  // `values` to every event/receive node declared against `eventIndex`, in ascending JSON order
  // (spec: activate in that order), stopping early if a receive node's handler chain calls
  // event/stopPropagation with stopImmediate=true for this occurrence. Any of eventIndex's declared
  // values missing from `values` falls back to that value's own declared default. Shared by
  // event/send's own node execution (gltf_interactivity_eval.cpp) and the Graphs panel's custom
  // event sender (ui_interactivity.cpp) - both are "something external delivers this event," just
  // triggered differently. Returns false (no-op) if eventIndex is out of range.
  bool sendEvent(int eventIndex, std::unordered_map<std::string, InteractivityValue> values = {});

  // event/stopPropagation (spec 4492-4530): marks `ref`'s occurrence as having had its propagation
  // stopped. "Transitive" activations (scene-graph bubbling - hover/select) beyond the current node
  // are always cancelled; if `stopImmediate` is also true, any remaining not-yet-fired handlers
  // sharing this exact occurrence ref (whether later in the same node's handler list, later in the
  // scene-graph bubble, or - for event/onStart/onTick/event/send - later in that dispatch's handler
  // list) are cancelled too. No-ops for a ref that was never a real event occurrence (spec: an
  // invalid ref is not an error, just nothing to cancel).
  void stopEventPropagation(InteractivityRef ref, bool stopImmediate);
  // True once stopEventPropagation() has cancelled `ref`'s remaining transitive (bubble)
  // activations - checked by the scene-graph ancestor-walking loops
  // (Scene::notifyNodeSelected/notifyNodeHoverChanged) before processing the next ancestor.
  // (isImmediatePropagationStopped() implies this too, since stopImmediate always also sets it.)
  bool isTransitivePropagationStopped(InteractivityRef ref) const;
  // True once stopEventPropagation() has cancelled `ref`'s remaining immediate activations -
  // checked before firing each handler in a same-occurrence dispatch loop (activateBoundHandlers,
  // sendEvent, start()/tick()'s onStart/onTick loops).
  bool isImmediatePropagationStopped(InteractivityRef ref) const;

  // debug/log's captured history (spec severity value, formatted message), oldest first, capped at
  // kMaxLogEntries so a runaway per-tick debug/log doesn't grow this unbounded. The Graphs panel's
  // log surface is the primary consumer; gltf_interactivity_eval.cpp's eDebugLog case still also
  // logs via LOGI for console visibility - this is an additional, UI-facing record, not a replacement.
  struct LogEntry
  {
    int32_t     severity = 0;
    std::string message;
  };
  static constexpr size_t     kMaxLogEntries = 200;
  const std::deque<LogEntry>& logEntries() const { return m_logEntries; }
  void                        appendLogEntry(int32_t severity, std::string message);
  void                        clearLogEntries() { m_logEntries.clear(); }

private:
  void advancePendingDelays();
  void advanceVariableInterpolations();
  void advancePointerInterpolations();
  void advanceAnimations();

  const InteractivityGraph&       m_graph;
  std::vector<InteractivityValue> m_variables;

  // Recursion-depth guard for activateFlow() (see its doc comment): a conservative cap far below
  // an actual stack overflow, since backward flow references (see docs/interactivity.md) are
  // accepted and can form a real cycle - a spec-conformant graph doesn't have to be malicious to
  // recurse deeply, just long or cyclic, so this fails soft (warns once, no-ops) rather than
  // crashing the process. Not tuned to a measured stack budget; deliberately small relative to the
  // old queue-based step cap (100'000) because C++ stack frames cost far more than a heap queue
  // entry did.
  static constexpr int kMaxCallDepth = 512;
  int                  m_callDepth   = 0;

  bool  m_started           = false;
  bool  m_hasTicked         = false;
  float m_timeSinceStart    = 0.0f;
  float m_timeSinceLastTick = 0.0f;

  InteractivityRef m_currentStartRef;  // minted once in start(), shared by every onStart node
  InteractivityRef m_currentTickRef;   // minted once per tick(), shared by every onTick node

  uint64_t                                            m_flowActivationCounter = 0;
  std::unordered_map<int, std::pair<float, uint64_t>> m_randomCache;
  std::mt19937                                        m_rng{std::random_device{}()};

  InteractivityPointerResolver*   m_pointerResolver   = nullptr;
  InteractivityAnimationResolver* m_animationResolver = nullptr;

  std::unordered_map<int, int32_t>          m_nodeIntState;
  std::unordered_map<int, InteractivityRef> m_nodeRefState;
  std::unordered_map<int, MultiGateState>   m_multiGateState;
  std::unordered_map<int, WaitAllState>     m_waitAllState;
  std::unordered_map<int, ThrottleState>    m_throttleState;

  int32_t                     m_nextRefHandle = 1;
  std::unordered_set<int32_t> m_eventRefHandles;

  // event/stopPropagation state, keyed by event ref handle (see stopEventPropagation()).
  std::unordered_set<int32_t> m_transitivePropagationStopped;
  std::unordered_set<int32_t> m_immediatePropagationStopped;

  struct PendingDelay
  {
    int              nodeIndex;
    InteractivityRef ref;
    float            activationTime;  // absolute instance.timeSinceStart() value
  };
  std::vector<PendingDelay> m_pendingDelays;

  std::vector<VariableInterpolationState> m_variableInterpolations;
  std::vector<PointerInterpolationState>  m_pointerInterpolations;
  std::vector<AnimationPlaybackState>     m_animationPlaybacks;
  std::vector<AnimationApply>             m_animationApplies;

  struct NodeOccurrence
  {
    std::unordered_map<std::string, InteractivityValue> values;
    InteractivityRef                                    eventRef;
  };
  std::unordered_map<int, NodeOccurrence> m_nodeOccurrences;

  std::deque<LogEntry> m_logEntries;
};

}  // namespace nvvkgltf
