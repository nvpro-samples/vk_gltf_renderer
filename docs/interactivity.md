# KHR_interactivity (behavior graphs)

> **For contributors and agents.** Architecture-level; the symbols cited in `src/` are the
> source of truth for exact behavior. This doc tracks a phased rollout — see the coverage table
> below for what is actually implemented today. Never assume a node op works because it's listed
> in the spec; check the table.

`KHR_interactivity` embeds a Turing-complete, node-based "behavior graph" (visual scripting) in a
glTF asset — 135 spec-defined node operations across events, control flow, math, type conversion,
variables, pointers, and animation control. vk_gltf_renderer implements a from-scratch C++
execution engine for it (not a port of the Khronos JS reference implementation).

## Architecture

Two-object split, mirroring the project's general `Scene`/`SceneVk` (data vs. GPU-instance)
pattern:

- **`InteractivityGraph`** ([gltf_interactivity_graph.hpp](../src/gltf_interactivity_graph.hpp)) —
  the *compiled* graph: parsed once from `model.extensions.KHR_interactivity.graphs[]` into an
  immutable, strongly-typed form (`InteractivityOp` enum instead of raw strings, pre-resolved
  socket references). One instance per glTF `graphs[]` entry, owned by `nvvkgltf::Scene`.
- **`InteractivityGraphInstance`** ([gltf_interactivity_instance.hpp](../src/gltf_interactivity_instance.hpp))
  — the *runtime* state for one playing graph: variable values, per-node op state, and per-tick
  timing. Play/Pause/Reset only ever touch an instance; the compiled graph never changes.

Node evaluation is table-driven in [gltf_interactivity_eval.cpp](../src/gltf_interactivity_eval.cpp):
`evaluateNodeOutput` (pull-based, for value sockets) and `executeFlowNode` (push-based, for flow
activations). Values are a strongly-typed `std::variant` (`InteractivityValue`), not the loosely-typed
values a JS engine would use — a node with mismatched operand types simply fails to evaluate rather
than silently coercing. Flow activation (`InteractivityGraphInstance::activateFlow()`) is real C++
recursion, not a work queue — see the design note below; a flow chain's C++ call depth equals its
transitive length, guarded by a conservative cap rather than an unbounded-stack risk.

`Scene::tickInteractivityGraphs()` advances the default graph's instance once per frame
(`GltfRenderer::updateInteractivityGraphs`, called from `onRender` alongside `updateAnimation`).
An op the parser recognizes but the evaluator hasn't implemented yet safely no-ops (logged once),
rather than crashing — see the coverage table below for what that currently means in practice.

## Coverage (update this table as phases land)

| Area | Status | Notes |
|---|---|---|
| Graph parsing/validation | ✅ | Full `declarations`/`types`/`variables`/`events`/`nodes` schema, structural-validity checks per spec. Flow-socket targets are only required to resolve to a real node, not to point strictly forward — see design note below. |
| `event/onStart`, `event/onTick` | ✅ | Lifecycle timing (`timeSinceStart`/`timeSinceLastTick`) matches spec, including the first-tick NaN rule. Every `event/onStart` node shares one occurrence ref per run (likewise every `event/onTick` node per tick) — see design note below. |
| `flow/sequence`, `flow/branch`, `flow/switch` | ✅ | Lexicographic socket-order rule implemented for `sequence`. |
| `flow/for`, `flow/while` | ✅ | Implemented as a direct C++ loop calling `InteractivityGraphInstance::activateFlow()` per iteration, which is itself synchronous/recursive — see design note below — not fake self-activation through a work queue. |
| `flow/doN`, `flow/multiGate`, `flow/waitAll`, `flow/throttle` | ✅ | Persistent per-node state on the instance (`nodeIntState`/`multiGateState`/`waitAllState`/`throttleState`), keyed by node index. |
| `flow/setDelay`, `flow/cancelDelay` | ✅ | The one genuinely tick-spanning flow op in this batch: pending delays are timestamp-based (not decremented-per-tick), checked once per `tick()` in `advancePendingDelays()`. |
| `event/send`, `event/receive` | ✅ | `event/send` scans the graph for matching `event/receive` declarations and delivers values directly (no queue-level pub/sub bus). |
| `event/stopPropagation` | ✅ | Full cancellation, not a no-op: `InteractivityGraphInstance::stopEventPropagation()` marks the target event ref's remaining activations cancelled — "transitive" (further scene-graph bubbling, `event/onSelect`/`onHoverIn`/`onHoverOut`) is always cancelled; "immediate" (any not-yet-fired handler sharing this exact occurrence — remaining `event/receive` nodes in the same `event/send` dispatch, remaining `event/onStart`/`onTick` nodes, or remaining ancestors in the same bubble) is additionally cancelled when `stopImmediate` is true — see design note below. |
| `debug/log` | ✅ | `{param}` template substitution + `{{`/`}}` escaping; logged via `LOGI`, severity value included but its exact spec-defined level mapping is unverified. |
| `type/*` (6 conversions) | ✅ | Includes the exact `floatToInt` truncate/wrap procedure. |
| `variable/get`, `variable/set` | ✅ | |
| `math/*` full catalog (arithmetic, comparison, trig/hyperbolic/exponential, vector/matrix/quaternion, swizzle, integer/boolean, color, constants, `select`/`switch`/`random`) | ✅ | All ~130 `math/*` + `ref/eq` ops, including `math/smoothStep`/`math/trunc` and matrix (`floatNxN`) support for the ops the spec defines that way (abs/sign/floor/ceil/round/fract/trunc/saturate/neg/add/sub/mul/div/min/max/clamp/mix/eq) — see design note below. Quaternions are `float4` = (x,y,z,w), w = scalar part. `math/random` implements the spec's per-flow-activation value-caching rule. `math/div`/`math/rem` are two-overload ops (int-specific piecewise-safe, and a separate generic `floatN`/`floatNxN` IEEE-754 overload) — dispatched by operand type. |
| `pointer/get`, `pointer/set` | ✅ (broad but still partial property surface) | Reads/writes the live `tinygltf::Model` via `ScenePointerResolver` ([gltf_interactivity_scene_pointer.hpp](../src/gltf_interactivity_scene_pointer.hpp)). Covers node TRS/matrix/globalMatrix/mesh/camera/skin/parent/children/weights + `KHR_node_visibility`, core + extension material properties (any `/extensions/<NAME>/...` path, including `KHR_texture_transform` under any texture slot) via a generic value walk, punctual-light color/intensity/range/spot cone angles, perspective-camera basics, core read-only array/length/ref pointers (animations/cameras/materials/meshes/nodes/scenes/skins), the spec's own asset/limits self-description pointers, and the animation asset's `/extensions/KHR_interactivity/maxTime` — not the full glTF Object Model (no accessor/buffer/image/sampler/texture-index-level pointers, no skin `inverseBindMatrices`). Grep that file's `get()`/`set()` for the exact path list. `ref`-typed literal *values* (as opposed to `{name}` template parameters, substituted via the ref's raw handle) are JSON-Pointer-path strings per spec (e.g. `"/nodes/17"`) — see design note below. The spec's own two self-referential pointers, `/extensions/KHR_interactivity/events/{}` and `.../delays/{}` (ref-validity checks), ARE implemented — but resolved directly against the instance in `gltf_interactivity_eval.cpp`, not through `ScenePointerResolver`. The per-animation-object `isPlaying`/`minTime`/`playhead`/`virtualPlayhead` pointers are not yet implemented (`maxTime` is — see the design note below). |
| `pointer/interpolate`, `variable/interpolate` | ✅ | Multi-tick interpolation state, structurally similar to `flow/setDelay`'s pending-delay list but with per-tick cubic-Bézier-eased value writes (`InteractivityGraphInstance::advanceVariableInterpolations()`/`advancePointerInterpolations()`). Quaternion (spherical) interpolation is used automatically when the target is `float4` and (for `pointer/interpolate`) the resolved path ends in `/rotation` — see design note below. |
| `animation/start`/`stop`/`stopAt` | ✅ | Concurrent multi-clip playback: `InteractivityGraphInstance` computes the spec's requested-to-effective-timestamp remap and start/stop/end threshold logic purely (no Scene needed), applying each active clip's pose *synchronously* via `InteractivityAnimationResolver::applyPose()` before firing that entry's `done`/stop-`done` (spec step order — see design note below). `GltfRenderer::updateInteractivityGraphs()` only handles the remaining GPU-side reconciliation tail (`reconcileAnimationGpuState()`, factored out of the UI-driven single-clip `updateAnimation()` so both paths share one pipeline), run once per tick via `pendingAnimationApplies()`. Per spec, presence of a `KHR_interactivity` graph also disables the UI's animation autoplay default (`AnimationControl::play` starts `false`) at scene load. |
| `event/onHoverIn`/`onHoverOut` (`KHR_node_hoverability`), `event/onSelect` (`KHR_node_selectability`) | ✅ (approximated ray semantics) | Full spec-literal ancestor-chain bubbling: every bound ancestor fires for `onSelect` (propagation continues past the first match); `onHoverIn`/`onHoverOut` additionally stop at the lowest common ancestor of the previous/new hover target (DOM `mouseenter`/`mouseleave`-style truncation) so a transition between siblings doesn't refire their shared parent. See `Scene::notifyNodeSelected`/`notifyNodeHoverChanged` ([gltf_scene.cpp](../src/gltf_scene.cpp)). Hover detection reads `Resources::eImgSelection` (the same per-pixel render-node G-buffer the silhouette pass already produces) via an async, non-blocking readback (`HoverPicker`, [hover_picker.hpp](../src/hover_picker.hpp)) instead of casting extra rays — see the design note below for why. Both ops approximate the spec's "skip non-hoverable/non-selectable geometry, continue the ray" termination rule as "redirect to the nearest hoverable/selectable ancestor" (mirroring how click-selection already redirects for `KHR_node_selectability`) — a single G-buffer sample (or the existing `RayPicker` for clicks) can't skip to occluded geometry behind an opted-out object. `event/onSelect`'s `selectionPoint`/`selectionRayOrigin` output sockets report the real ray-pick hit point/origin for an actual 3D-viewport click (plumbed from `RayPicker::PickResult` through `SceneSelection::Event`) — NaN (the spec-sanctioned "no ray info" fallback, `KHR_node_selectability` README ~110) only for non-ray selections. See design note below; `tests/test_interactivity_hover_select.cpp` exercises this against a real `Scene`. |
| UI (Graphs window, play/pause/reset, custom events) | ✅ | A standalone "Interactivity" window ([ui_interactivity.hpp](../src/ui_interactivity.hpp)/[.cpp](../src/ui_interactivity.cpp)), closed by default (opened via View > Windows > Interactivity, or F8) - developer-facing content (live node/variable/event counts, started/ticked/time stats, a per-index variable value list, a custom event sender, a debug/log history surface) a user never has to see. A separate viewport-toolbar Play/Pause indicator (`ui::interactivity::renderToolbarIndicator`) only exists when the loaded scene has a graph, toggles `InteractivityControl::play` directly, and does not open the window - the two are independent controls over the same shared state. Visual node/wire graph debugging is explicit future work, not in scope here. End-to-end UI coverage previously lived in `tests/ui/interactivity_panel.txt`, driven via a scripted UI-automation integration that has since been removed pending reinstatement — see the [Tests](#tests) section below. |

A `KHR_interactivity` graph can now move/color/toggle scene objects at runtime via `pointer/set`,
ease values smoothly via `pointer/interpolate`/`variable/interpolate`, play/stop glTF animation
clips via `animation/start`/`stop`/`stopAt`, react to the user hovering or selecting scene objects
via `event/onHoverIn`/`onHoverOut`/`onSelect` (with real `stopPropagation` cancellation), and be
driven/inspected live from the standalone Interactivity window. See the coverage table above for
per-area status and known gaps (e.g. `pointer/get`/`set`'s still-partial Object Model property surface).

## Design notes worth knowing before extending this

- **Flow activation is real C++ recursion (call/return), not a work queue.** The spec is explicit
  that output flow sockets are "function pointers" the node "will call" and input flow sockets are
  "methods" (`Specification.adoc`), and `flow/sequence` requires "each output flow is activated
  after the previous output flow completes" — i.e. depth-first, synchronous. A breadth-first-queue
  execution model (closer to a JS event loop) is spec-incorrect: it lets independent chains
  interleave instead of each running to completion, which breaks ordering guarantees ops like
  `event/send`/`event/receive` depend on. `InteractivityGraphInstance::activateFlow()` calls
  `executeFlowNode()` directly and returns only once that node's entire transitive downstream chain
  has finished, guarded by a conservative call-depth counter (`kMaxCallDepth`) that fails soft
  (logs once, no-ops) rather than crashing. `flow/for`/`flow/while` are a plain C++ loop calling
  `activateFlow()` per iteration — no separate queue-draining mechanism needed. Regression coverage:
  `InteractivityInstance.OnStartChainsCompleteBeforeNextOnStartBegins` and
  `.SequenceBranchCompletesBeforeNextBranchBegins` in `tests/test_interactivity_engine.cpp`.
- **Flow-socket targets are allowed to point backward, deliberately deviating from spec text.**
  The spec (`Specification.adoc`, "Output Flow Socket Pointers") requires a flow target's node
  index to be strictly greater than the current node's, specifically to guarantee the static graph
  has no cycles. Real authored content (visual-editor exports, which serialize nodes in
  creation/layout order rather than topological flow order) routinely violates this. Rejecting the
  whole graph over it would make this engine unable to run real content, so
  `InteractivityGraph::parse()` only rejects a flow target that's genuinely unresolvable (negative
  or `>= nodes().size()`); an in-range backward reference is accepted and can form a real runtime
  cycle. Combined with flow activation being real recursion (previous bullet), a spec-conformant
  graph doesn't have to be malicious to recurse deeply — just long or cyclic — so the actual safety
  net is `activateFlow()`'s `kMaxCallDepth` guard (fails soft), not the parse-time check.
- **`event/onStart`/`event/onTick` occurrence refs are shared per run/tick, not minted per node** —
  spec: conceptually one "start" event per graph run, not one per node (likewise one "tick" event
  per tick), so two `event/onStart` nodes' `event` outputs must compare equal via `ref/eq`.
  `InteractivityGraphInstance::start()`/`tick()` mint exactly one ref each (`allocateEventRef()`,
  stored as `m_currentStartRef`/`m_currentTickRef`) before activating any `onStart`/`onTick` node.
- **`/extensions/KHR_interactivity/events/{}` and `.../delays/{}` are the spec's own "is this ref
  real" pointers** (its "Event References"/"Delay References" sections). They query the *instance's*
  live ref-tracking state, not the glTF model, so `evaluatePointerGet()` resolves them directly
  (`resolveSelfReferentialRefPointer()` in `gltf_interactivity_eval.cpp`) rather than routing
  through `ScenePointerResolver`, which is deliberately kept model-only/Scene-free so the core
  engine stays unit-testable without a `Scene`. `events/{}` validity is permanent —
  `InteractivityGraphInstance::isEventRef()` checks a ref against every handle `allocateEventRef()`
  has *ever* minted, per spec ("the internal state of the event object has no effect on this
  operation") — while `delays/{}` is live: `isPendingDelayRef()` checks the *current*
  `m_pendingDelays` array, so a fired or cancelled delay's ref stops validating. `flow/setDelay`'s
  ref stays on plain `allocateRefHandle()` (not `allocateEventRef()`): it's a delay-kind ref, not an
  event-kind one, and the two pointers must not cross-validate each other's refs.
- **`math/div`/`math/rem` are each two overloads, not one.** The spec defines an `int`-specific
  overload (piecewise `b==0 → 0`, `INT_MIN / -1` wraps rather than traps — naive `int32_t`
  `/`/`%` is a hardware trap for those cases) *and*, separately, a generic `floatN`/`floatNxN`
  overload with its own edge-case formula (plain IEEE-754 `a/b` for div; a 3-way
  NaN/Infinity/`trunc` piecewise formula for rem). `evaluatePureMath()`'s `eMathDiv`/`eMathRem`
  cases dispatch on operand type: both `int32_t` → the piecewise-safe path; anything else matching
  → `applyBinary<true>()` with the generic formula. Don't collapse these back into one code path.
- **Most `math/*` unary/binary ops are spec'd "`floatN` or `floatNxN`", including matrices.**
  `applyUnary`/`applyBinary`/`applyComparison` decompose a matrix operand down to scalar components
  (`applyComponentwise`/`applyComponentwiseBinary`/`applyComponentwiseTernary` in
  `gltf_interactivity_eval.cpp`, recursing matrix → column vector → float) *before* calling each
  op's lambda, rather than calling glm's free functions (`glm::floor`, `glm::min`, ...) on a whole
  `matN` directly — glm doesn't define those for matrix arguments with the componentwise semantics
  the spec wants. This also gets `math/mul`'s spec-required *per-element* matrix multiplication for
  free (via elementwise `vec*vec` on columns), as opposed to a real matrix product. `math/eq` stays
  a single, non-decomposed `av == bv` — glm's `operator==` already reduces a whole vec/mat to one
  bool, matching spec. `math/eMathMix`'s `c` (interpolation coefficient) matches `a`/`b`'s type
  exactly (`floatN c` or `floatNxN c`, same `N`), not a bare scalar. `math/sign`'s `glm::sign` is
  comparison-based and silently returns `0` for `NaN` instead of propagating it (spec's blanket
  "any NaN component in → NaN component out" rule) — guarded with an explicit `std::isnan` check.
- **Strict JSON has no numeric token for IEEE-754 Infinity/NaN — conformance content authors these
  as quoted string tokens instead**, e.g. `"value": ["Infinity", 2, 3]`. Not documented in the core
  spec text, but real content depends on it. `InteractivityGraph::parseInteractivityLiteral()`'s
  `num()` helper checks `IsString()` first and recognizes `"Infinity"`/`"-Infinity"`/`"NaN"` —
  `tinygltf::Value::GetNumberAsDouble()` on a string value silently returns `0.0`.
- **`ref`-typed literal *values* are JSON-Pointer-path strings, not raw integer handles.** Spec
  ("Variables and Types"): a `ref` literal's `value` array holds `["/nodes/17"]`, not `[17]`. Don't
  confuse this with `{name}` *template* parameters in a pointer path string
  (`/nodes/{nodeRef}/matrix`), which substitute a ref's `.handle` as a decimal integer via
  `substitutePointerTemplate()` — a different mechanism. `parseInteractivityLiteral()`'s `eRef` case
  parses the trailing `/`-delimited numeric segment (e.g. `17` from `"/nodes/17"`) as the ref's
  handle — consistent with this engine's "ref.handle == target's index within its owning array"
  convention; a malformed or unresolvable pointer becomes a null ref, per spec.
  `InteractivityRef` also carries the owning-array name as `category` (e.g. `"nodes"` vs.
  `"materials"`), since two literals with the same index but different owning arrays would
  otherwise collapse onto the same handle and compare equal via `ref/eq`. Every ref this engine
  mints internally for a glTF core-array element (`ScenePointerResolver`'s `indexRef()`,
  `Scene::notifyNodeSelected`/`notifyNodeHoverChanged`'s node refs) tags the matching category, so
  it still compares equal to an equivalent authored literal.
- **`/nodes/N/matrix` and `/nodes/N/globalMatrix` compute fresh from the live `tinygltf::Node`
  on every read, deliberately not `Scene::getNodesLocalMatrices()`/`getNodesWorldMatrices()`'s
  cache.** Those caches are only refreshed by an explicit `updateNodeWorldMatrices()` call
  elsewhere in the render loop — a graph that `pointer/set`s a node's translation and then
  `pointer/get`s its `matrix`/`globalMatrix` back *in the same tick* would see stale, pre-write data
  if `get()` read the cache. `tinygltf::utils::getNodeMatrix()` (the same function `Scene` itself
  calls to populate that cache) reads the `Node`'s current TRS/matrix fields directly, so using it
  here stays correct across an arbitrary same-tick write; `globalMatrix` walks the parent chain
  (`Scene::getNodeParents()`) doing the same fresh per-node computation at each level.
- **Hover detection reads a G-buffer instead of casting rays, deliberately.** Click-selection uses
  `nvvk::RayPicker`: one ray, submitted and *synchronously waited* on every click — fine once per
  click, but doing that every frame for continuous hover would stall the pipeline on every
  mouse-move and fight the path tracer's progressive accumulation. `Resources::eImgSelection`
  already holds a per-pixel render-node ID and is already valid exactly when hover needs it: the
  path tracer only rewrites it on `frameCount==0` (i.e. exactly on reset), and during pure
  accumulation frames it's unchanged but still correct, since nothing moved. The rasterizer writes
  it unconditionally every frame it runs. `HoverPicker` ([hover_picker.hpp](../src/hover_picker.hpp))
  copies the pixel under the cursor into a small pool of host-visible buffers each frame
  (`requestReadback`, recorded *after* the silhouette pass's own read of the same image, so it's a
  read-after-read needing no extra barrier) and polls for completion non-blocking
  (`nvvk::SemaphoreState::testSignaled`, never `vkDeviceWaitIdle`/fence-wait) —`pollResult()` in
  `GltfRenderer::updateHoverState()` ([renderer.cpp](../src/renderer.cpp)). The G-buffer's render-node
  ID is bit-cast (not numerically converted — GLSL/Slang `asfloat`) into the float channel, offset
  by +1 so 0 can mean "nothing hit" (`traceSelectionRay`, [pathtrace_functions.h.slang](../shaders/pathtrace_functions.h.slang);
  `gltf_raster.slang` writes the identical encoding) — decoding this wrong (e.g. a numeric cast
  instead of a bit-cast) silently produces garbage node indices, not a crash, so don't "simplify" it.
- **Hover/select event propagation is real ancestor-chain bubbling, not exact-node matching.**
  `event/onSelect`/`onHoverIn`/`onHoverOut` bind to a glTF node via that *handler* node's own
  `configuration.nodeIndex` (spec: `KHR_node_hoverability`/`KHR_node_selectability` READMEs) — a
  click/hover on a leaf with no handler of its own must still reach a handler on an ancestor.
  `Scene::notifyNodeSelected`/`notifyNodeHoverChanged` ([gltf_scene.cpp](../src/gltf_scene.cpp)) walk
  the ancestor chain (`m_nodeParents`) firing every bound node — for `onSelect`, all the way to the
  root (spec: propagation continues past the first match); for hover, up to but excluding the
  lowest common ancestor of the previous and new hover target, so a transition between two children
  of the same parent doesn't refire the parent's own handler (DOM `mouseenter`/`mouseleave`-style
  common-ancestor truncation). Both share one occurrence ref across the whole ancestor walk (see
  the `event/stopPropagation` bullet below) rather than minting a fresh one per ancestor. Test
  coverage: [tests/test_interactivity_hover_select.cpp](../tests/test_interactivity_hover_select.cpp).
- **Quaternion convention.** `float4` quaternions are `(x, y, z, w)` with `w` the scalar/real part
  (spec, same as a glTF node's `rotation`) — never `glm::quat`, whose `(w,x,y,z)` constructor order
  is a frequent source of silent bugs when mixed with vec4-as-quaternion code. All quaternion math
  in [gltf_interactivity_eval.cpp](../src/gltf_interactivity_eval.cpp) is implemented directly on
  `glm::vec4` (see `quatMulRaw`) for exactly this reason.
- **A few corners are genuinely implementation-defined** (the spec says so explicitly): `math/matDecompose`'s
  handling of sheared or negative-determinant input, and `math/quatFromAngles`'s exact per-axis
  composition order. Both have a documented, tested (round-trip) choice in `gltf_interactivity_eval.cpp`
  — don't treat a different-but-plausible result on these as a bug without checking the spec text first.
- **Template parsing vs. concrete-path writing are two different problems, deliberately not merged.**
  `pointer/get`/`set`'s JSON-Pointer *Templates* (`[int]`/`{ref}` placeholders, spec-only to
  KHR_interactivity) are parsed by `parsePointerTemplate`/`substitutePointerTemplate`
  ([gltf_interactivity_pointer.hpp](../src/gltf_interactivity_pointer.hpp)) into a concrete path
  string like `/nodes/3/translation`. That concrete string is then handed to the *existing*
  `AnimationPointerSystem::applyValue()` ([gltf_animation_pointer.hpp](../src/gltf_animation_pointer.hpp))
  for the actual write — `ScenePointerResolver::set()` is a thin adapter, not a second write path.
  `AnimationPointerSystem` itself has no template syntax to parse (animation channels always target
  a fixed, already-concrete path), so there is exactly one template parser and one model-write
  implementation, each doing the one job it's suited for.
- **`pointer/get` has no existing analog to reuse for actually reading values.** `AnimationPointerSystem`
  is write-only (a JSON shadow tree plus per-resource-type sync-back). `ScenePointerResolver::get()`
  reads directly from the live `tinygltf::Model` instead (always current, no shadow needed for
  reads). Both still need to parse a concrete path's leading `"/prefix/<index>/"` segment
  (`AnimationPointerSystem::parseResourceInfo()` to route a write to the right dirty-set;
  `ScenePointerResolver::get()` to route a read to the right resource array) — that piece,
  `tinygltf::utils::parsePointerIndexAndRest()` ([tinygltf_utils.hpp](../src/tinygltf_utils.hpp)),
  is shared so there's exactly one `std::from_chars`-based path-segment parser in the codebase.
- **Runtime graph writes bypass undo/redo.** `pointer/set` mutates via `ScenePointerResolver` →
  `AnimationPointerSystem::applyValue()` → `Scene::markNodeDirty`/`markMaterialDirty`/`markLightDirty`
  — the same direct entry points the continuous gizmo-drag path already uses — never through
  `UndoStack`. A graph re-writing a node's transform every tick is not a user edit.
- **Declaration failure modes are spec-mandated, not implementation choices.** A structurally
  invalid declaration (bad/missing `op`, or a core op carrying extension-only fields) rejects the
  whole graph at parse time; a well-formed but unrecognized `extension` op degrades only the nodes
  using it to no-ops. Don't "fix" the latter into a hard failure — it's intentional graceful
  degradation for third-party authoring-tool graphs using nodes this build doesn't know yet. This
  extends to malformed/unresolvable *values* (an unrecognized `types[].signature`, a literal with
  too few components): resolving to a type-default or `monostate` and letting the affected node fail
  to evaluate is the deliberate choice, not a gap — `InteractivityGraph::parse()` failure currently
  has whole-graph blast radius (`parseInteractivityGraphs()` silently drops any graph that fails to
  parse), so a hard reject on one bad value takes out graphs that never touch it. Don't add one
  without first narrowing that blast radius.
- **Op dispatch keys on the declaration, not a bare string switch on the node.** Several op ids are
  reused across type-overloads (e.g. `math/add` for float/int) and some ops are only meaningful
  paired with a specific `extension` (e.g. `event/onSelect` + `KHR_node_selectability`) — always go
  through `InteractivityDeclaration::op`/`extension`, never re-parse `node` JSON directly.
- **`variable/interpolate`/`pointer/interpolate`'s easing curve is a real cubic-Bézier solve, not an
  approximation.** The spec defines the eased progress `q` as the *y* value on a cubic Bézier with
  implicit endpoints `P0(0,0)`/`P3(1,1)` and authored control points `p1`/`p2`, sampled at the *x*
  position equal to the elapsed-time fraction `t` — which requires solving `x(s) = t` for the curve
  parameter `s` first, then evaluating `y(s)`. `cubicBezierEase()` (`gltf_interactivity_eval.cpp`)
  uses the same Newton-Raphson-with-bisection-fallback algorithm CSS `cubic-bezier()` timing
  functions use, not a linear/lookup-table shortcut. Quaternion (`float4`) targets get spherical
  interpolation (`quatSlerpRaw()`, shared with `math/quatSlerp`) instead of a plain componentwise
  lerp when `useSlerp` is set (`variable/interpolate`, author's choice) or — for
  `pointer/interpolate`, which has no `useSlerp` socket — when the resolved path ends in `/rotation`
  (spec requires slerp for quaternion-typed properties; detecting *which* `float4` properties are
  quaternions is otherwise implementation-defined, and `/rotation` is the only quaternion-typed
  property this app's pointer surface exposes today). Both ops share one per-tick advance path
  (`InteractivityGraphInstance::advanceVariableInterpolations()`/`advancePointerInterpolations()`,
  called from `tick()` alongside `advancePendingDelays()`), storing the `done` flow's *resolved*
  target (node+socket) at start time rather than re-looking it up later.
- **`event/stopPropagation` requires hover/select bubbling to share one occurrence ref across the
  whole ancestor walk, not mint a fresh one per ancestor.** `activateBoundHandlers()` takes the
  occurrence ref as a parameter; the two `Scene` call sites mint it once before the ancestor loop
  and pass it through, checking `isTransitivePropagationStopped()` after each ancestor to decide
  whether to keep walking. Because flow execution is synchronous (see the first design note above),
  "cancel pending activations" doesn't need real queue entries — the loop position *is* the queue,
  so cancellation is just a flag (`isImmediatePropagationStopped()`) checked before each iteration:
  within `activateBoundHandlers()`'s own handler list, in `sendEvent()`'s `event/receive` scan, and
  in `start()`/`tick()`'s `onStart`/`onTick` loops.
- **`animation/start` reuses `AnimationSystem` as-is; it's an orchestration layer, not a rewrite.**
  `AnimationSystem::updateAnimation(animationIndex)` is already parameterized per-clip — the
  "single clip" limitation lives only in `AnimationControl`/the UI scrubber, one layer up. So
  `InteractivityGraphInstance` computes the timestamp math purely and, via
  `InteractivityAnimationResolver::applyPose()`, calls straight into
  `AnimationSystem::updateAnimation()` **synchronously inside `tick()`** (the pose must be written
  before that entry's `done`/stop-`done` fires — spec step order; getting this backward means a
  `pointer/get`/`debug/log` reached transitively from `done` reads a stale, one-tick-old position).
  `GltfRenderer::updateInteractivityGraphs()` only handles what's left: the downstream GPU pipeline
  (world matrices, GPU sync, morph/skin compute, BLAS/TLAS update), run once per tick via a shared
  `reconcileAnimationGpuState()` factored out of the UI-driven `updateAnimation()` so neither path
  duplicates Vulkan command recording. `updateInteractivityGraphs()` must report back whether it
  applied a pose this tick, OR'd into `onRender()`'s dirty/`changed` check — otherwise a
  graph-applied pose never resets the path tracer's progressive accumulation (it only clears the
  scene's dirty flags itself, before `onRender()`'s normal dirty-flag path would have seen them).
- **`event/onSelect`'s `selectionPoint`/`selectionRayOrigin` need the real ray-pick hit
  point/origin threaded through, not the spec's NaN "no ray info" fallback, whenever the selection
  came from an actual viewport click.** `RayPicker::PickResult` is threaded through
  `updateSelectionFromPick()` → `SceneSelection::selectPrimitive()` → `Event` →
  `notifyNodeSelected()`, with NaN staying the default for every other selection path (Scene
  Browser row click, Inspector, undo/redo). A selection that gets *redirected* mid-pick (e.g.
  `KHR_node_selectability`'s nearest-selectable-ancestor fallback, which reissues the selection via
  `SceneSelection::selectNode()`) must still carry the original ray data through — see
  `selectNode()`'s optional `selectionPoint`/`selectionRayOrigin` parameters.
- **`pointer/set` writing a single field of an extension object must merge into the existing
  value, never replace it wholesale.** A texture-info extension (e.g. `KHR_texture_transform`)
  commonly has sibling fields (`scale`, `texCoord`) baked in from the original glTF that the graph
  never itself writes; replacing `target.extensions[extName]` outright from the write's shadow JSON
  silently drops them. `AnimationPointerSystem::mergeJsonIntoMaterial()`'s texture-slot merge path
  now merges into the existing tinygltf value, matching how the material-level extensions merge in
  the same function already worked. Regression coverage:
  `InteractivityAnimationPointerSystem.PartialTextureTransformWritePreservesSiblingFields`
  (`tests/test_animation_pointer.cpp`).
- **`pointer/set` writing a `bool` only works for Object Model paths the write path explicitly
  recognizes as boolean-typed.** `AnimationPointerSystem` has no native bool `applyValue()`
  overload; every bool write must route through a shared suffix check
  (`tinygltf::utils::isBoolAnimationPointerPath()`, [tinygltf_utils.hpp](../src/tinygltf_utils.hpp))
  used by both `ScenePointerResolver::set()` and `AnimationPointerSystem::applyValue(float)`, which
  stores a JSON *boolean* rather than a raw number so the later `Get<bool>()` read-back sees the
  right type. Adding a new bool-typed Object Model path means adding its suffix here, not just to
  `ScenePointerResolver`. Regression coverage: `InteractivityPointerTest.SetWritesNodeSelectabilityBool`.
- **Viewport click-to-deselect must be suppressed while a graph is playing.** `GltfRenderer::
  updateSelectionFromPick()` ([ui_renderer.cpp](../src/ui_renderer.cpp)) normally treats clicking the
  currently-selected node again as a deselect — correct for ordinary editing, but wrong once a
  graph is driving the scene: a click-through state machine on one node (e.g. aim → draw → release,
  all bound to the same node's `event/onSelect`) needs *every* click to re-select and re-fire the
  event. `isInteractivityPlaying()` gates the toggle off whenever `Scene::getInteractivityGraphs()`
  is non-empty and `InteractivityControl::play` is true; click-to-deselect is unaffected on scenes
  with no graph, and returns the moment the graph is paused.
- **Viewport click-picking is asynchronous, not a synchronous GPU submit-and-wait, mirroring
  `HoverPicker`'s split.** `mouseClickedInViewport()` ([ui_renderer.cpp](../src/ui_renderer.cpp))
  only captures a `PendingClickPick` (camera state, normalized cursor position, single/double-click
  classification, all snapshotted *at click time* so a camera move before the pick runs doesn't
  skew the ray) with no GPU work; `onRender(cmd)` records `m_rayPicker.run(cmd, ...)` on the frame's
  own command buffer right where `HoverPicker::requestReadback` already runs, then hands off a
  frame-timeline semaphore to poll instead of waiting; `updateClickPickState()` polls it
  non-blocking each frame and, once signaled, reads `m_rayPicker.getResult()` and applies it via
  `applyClickPickResult()`. `isInteractivityPlaying()` also selects between the normal debounced
  single/double-click classification and `UiMouseState::isMouseReleasedNoDrag()` (no debounce) —
  a graph-driven scene wants every click to register instantly since double-click-to-recenter isn't
  a feature it needs, while ordinary editing keeps the debounce so a double-click can still mean
  "recenter camera" instead of two separate selects.
- **TLAS rebuild-vs-update must key on whether *any* instance's active state flipped, not on the
  net visible-instance count.** `SceneRtx::rebuildTopLevelAS()` ([gltf_scene_rtx.cpp](../src/gltf_scene_rtx.cpp))
  chooses between an in-place TLAS *update* (refits already-active instances' transforms) and a
  fresh *build* (needed to bring a previously-inactive instance's geometry into the traversable
  structure at all). A net-count check misses the case where one instance becomes visible while a
  different one becomes invisible in the same sync (net delta zero, e.g. `KHR_node_visibility`
  toggling between two mutually-exclusive nodes) — the newly-active instance still needs a BUILD,
  but a net-count check would pick UPDATE and silently leave it un-rendered (and can device-lost the
  TLAS builder outright). `anyActiveStateChanged`, computed per-instance during the existing
  dirty-node loop, is the correct signal; there is no legitimate use for `m_numVisibleElement`-style
  count-only bookkeeping here.
- **DLSS instance-motion-vector detection must run after the interactivity tick, not before.**
  `GltfRenderer::onRender()` ([renderer.cpp](../src/renderer.cpp)) decides whether to snapshot
  previous-frame transforms for DLSS by checking `Scene::DirtyFlags::nodes` — that check must come
  *after* `updateInteractivityGraphs(cmd)` (the call that ticks the graph and populates those dirty
  flags via `pointer/set`), or it only ever sees the previous frame's already-cleared flags. Gizmo/
  editor edits don't hit this (their dirty flags are set during the UI phase, before `onRender`
  starts) and `animation/*` clip playback doesn't either (detected proactively via
  `hasPlayableAnimation()`, not dirty flags) — only interactivity's per-tick `pointer/set` writes
  need the ordering. See [docs/denoising.md](denoising.md) for the instance-motion-vector pipeline.
- **DLSS's temporal history needs an explicit reset on appearance-only changes, separate from the
  path tracer's own accumulation reset.** A `pointer/set` write that changes *appearance* at a
  stationary surface (e.g. a `KHR_texture_transform` offset swapping which texture region shows) is
  not a motion event — `resetFrame()` only restarts the path tracer's own accumulation, it doesn't
  touch DLSS's separate internal temporal-history buffer (`Dlss::notifyReset()`, see
  [docs/denoising.md](denoising.md)). `BaseRenderer::notifyDlssContentReset()` (implemented by both
  `PathTracer`/`Rasterizer`) is invoked from `GltfRenderer::updateSceneChanges()` whenever
  `DirtyFlags::materials`/`lights` is non-empty; node-transform-only changes are deliberately
  excluded since those are already handled by instance motion vectors.
- **`pointer/get` on `/animations/[index]` and its `KHR_interactivity/maxTime` extension pointer
  are real Object Model paths some content depends on for hover/select-driven clip selection.**
  `/animations/[index]` resolves to a `ref` (the `indexRef()` pattern also used for `/mesh`,
  `/skin`, etc.); the `maxTime` pointer reads the already-computed `AnimationInfo::end` from
  `Scene::animation()` rather than re-deriving a clip's duration. A graph that reads an animation
  index via `pointer/get` and feeds it to `animation/start` silently takes the `err` path if either
  is missing, with no obvious symptom short of "nothing plays."

## Tests

`tests/test_interactivity_engine.cpp` — engine-level GoogleTest coverage (graph parsing/validation,
flow scheduling, node evaluation), no GPU or scene file required.
`tests/test_interactivity_pointer.cpp`/`test_interactivity_hover_select.cpp` — the same, but against
a real `nvvkgltf::Scene` (`ScenePointerResolver`, `Scene::notifyNodeSelected`/`notifyNodeHoverChanged`)
for the parts that need one. Run with `ctest -R Interactivity` or see
[tests/README.md](../tests/README.md#interactivity-tests).

A scripted UI-automation integration previously drove end-to-end coverage for the Graphs panel,
`animation/start`/`stop`/`stopAt` playback, and several viewport-selection scenarios. It has been
pulled out of the app and is tracked as future work to reinstate — see
[tests/README.md](../tests/README.md#ui-scenario-scripts-tests-ui--currently-dormant). The scenario
scripts still live under `tests/ui/` as reference but are not runnable today.
