# Rendering Architecture -- Vulkan Ray Tracing Data Flow

> **For contributors and agents.** The scene data-flow model (glTF model → RenderNodes → GPU SSBO / BLAS / TLAS) explained conceptually, with pointers to the owning code. Exact struct fields, thresholds, and function signatures live in the cited source files, not here.

**Purpose:** Document the complete data flow from glTF model through the Vulkan ray tracing and rasterization pipeline -- from scene graph to BLAS/TLAS acceleration structures and GPU render nodes.  
**Critical for:** Understanding how scene editing (duplicate/delete) affects the ray tracing and rasterization paths.

---

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│ CPU: tinygltf::Model (Scene Hierarchy)                                 │
│   • nodes[i] - Scene graph nodes                                       │
│   • meshes[j] - Mesh definitions                                       │
│   • materials[k] - Material definitions                                │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ parseScene()
                                 │ (traverses hierarchy, flattens to instances)
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ CPU: Flat Render Arrays (Derived, Regenerated)                                  │
│   • RenderNode[] - One per primitive instance (N:1 with leaf nodes/primitives)  │
│   • RenderPrimitive[] - Deduplicated geometry / BLAS                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
         uploadRenderNodes/syncFromScene    cmdCreateBuildTopLevelAS()
                    │                         │
                    ▼                         ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│ GPU: SSBO (Shader Access)    │  │ Ray Tracing: TLAS            │
│  GltfRenderNode nodes[N]     │  │  VkASInstance instances[N]   │
│  • objectToWorld             │  │  • transform matrix          │
│  • materialID                │  │  • BLAS reference            │
│  • renderPrimID              │  │  • instanceCustomIndex       │
└──────────────────────────────┘  └──────────────────────────────┘
```

**Key Insight:** RenderNodes are **derived structures**, rebuilt from Model whenever hierarchy changes.

### BLAS ↔ Primitive Index Contract (Critical)

There is a **direct correlation between BLAS and primitive index**:

- **BLAS** (Bottom-Level Acceleration Structures) are built from `RenderPrimitive[]` in order: `m_blasAccel[renderPrimID]`.
- **TLAS** (Top-Level AS) instances reference BLAS via `object.renderPrimID`: `blasAddress = m_blasAccel[object.renderPrimID].address`.

**When are BLAS actually built?** Only when the Vulkan scene is (re)created:

- **Once at load:** `GltfRenderer::createVulkanScene()` → `buildAccelerationStructures()` → `createBottomLevelAccelerationStructure()` + `cmdBuildBottomLevelAccelerationStructure()`.
- **Again only on full geometry rebuild:** `GltfRenderer::rebuildVulkanSceneInternal()` → `buildAccelerationStructures()` (e.g. when `primitivesChanged` is set in dirty flags).

BLAS are **not** rebuilt every frame or on every `parseScene()`. Hierarchy-only edits (reparent, duplicate, delete) that do not change the set of meshes/primitives only update CPU render nodes and TLAS (transforms/visibility); the existing BLAS array is reused. So for a given geometry state, BLAS are built once.

If the **order** of primitives (and thus `renderPrimID`) ever changed **without** rebuilding the BLAS, the TLAS would reference the **wrong** BLAS. Therefore:

1. **RenderPrimitive list** must be built in **deterministic order** (by mesh index, then primitive index), not in traversal order. See `parseScene()`: the unique-primitive list is filled by iterating `m_model.meshes` and their primitives **before** any scene-graph traversal.
2. **BLAS** are built from that ordered list; `renderPrimID` is stable across hierarchy edits (reparent, duplicate, delete) as long as the mesh set is unchanged.
3. Do **not** clear and repopulate `m_renderPrimitives` / `m_uniquePrimitiveIndex` in a way that depends on traversal order, or BLAS and TLAS will go out of sync.

### Opacity micromaps (optional)

When a scene uses `EXT_mesh_opacity_micromap` and the device exposes `VK_EXT_opacity_micromap`,
`SceneOmm` (owned by `SceneVk`, see `gltf_scene_omm.*`) builds one `VkMicromapEXT` per root
`micromaps[]` entry and uploads the per-primitive micromap index buffers alongside the geometry.
Results are keyed by `renderPrimID` so they follow the same BLAS ordering contract above.
`SceneRtx::createBottomLevelAccelerationStructure()` then attaches a
`VkAccelerationStructureTrianglesOpacityMicromapEXT` to each primitive's triangle geometry
(`triangles.pNext`), and the path-tracer RT pipeline is created with the OMM flag
(`VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT`). This only accelerates alpha-tested
traversal — microtriangles flagged "unknown" still run the existing any-hit alpha logic — so the
image is identical to the non-OMM path. When the extension is unsupported (or
`--useOpacityMicromap 0`), the whole subsystem is skipped and rendering falls back to the regular
alpha path.

The `eOpacityMicromap` entry of `enum Visualization` (see `shaders/shaderio.h`) adds a debug view
that colors the primary surface by how OMM traversal resolved it, using the fact that alpha
evaluation only happens for "unknown" micro-triangles: OMM-opaque micro-triangles are committed by
the hardware without any alpha work (green); "unknown" ones invoke the any-hit shader, which sets
`payload.ommUnknown` (yellow); transparent ones are culled so those rays miss the mesh and show the
environment behind. No baked micromap buffers are read by the shaders. The RT-pipeline technique
consults the micromap (via the pipeline flag above); the inline RayQuery technique does not opt in,
so there every alpha-tested triangle reads as "unknown".

This debug view (and its `payload.ommUnknown` field) is compiled out of the normal render path
behind the `USE_VISUALIZE` macro — the host sets it (see `PathTracer::compileShader`) only when the
selected `Visualization` mode is not `eRendered`, so shipping renders carry no extra payload field
or visualization branches. Because it is a compile gate, switching between normal rendering and any
visualization mode triggers a shader recompile (cached per `VariantKey`, so repeat switches are
fast); switching *among* visualization modes does not, since they share `USE_VISUALIZE == 1` and
select the mode at runtime via `frameInfo->visualization`.

---

## Stage 1: parseScene() - Build Render Structures

### Function: `void Scene::parseScene()`

**Location:** `gltf_scene.cpp`

**Purpose:** Traverse scene hierarchy and flatten into renderable instances.

### What It Does:

```
parseScene():
  1. Snapshot current render-node state (worldMatrix, materialID, renderPrimID, visible)
  2. clearParsedData()                    -- wipe render nodes, primitives, lights
  3. createMissingTangentsForModel()      -- ensure tangent attributes exist (stabilizes primitive keys)
  4. primMap = buildPrimitiveKeyMap()     -- register unique primitives in deterministic order
                                            (by mesh index, then primitive index — not traversal order)
  5. For each root node in current scene:
       traverseSceneGraph(depth-first):
         - collect lights via handleLightTraversal()
         - collect render nodes via handleRenderNode(nodeID, worldMatrix, primMap)
  6. updateRenderNodesFull()             -- apply animations, skinning, morph, visibility
  7. Diff against snapshot → set dirty flags:
       renderNodesVk / renderNodesRtx    (indices that changed)
       allRenderNodesDirty               (count changed or ≥ kFullUpdateRatio dirty)
       materials, primitivesChanged, lights
```

### How Hierarchy → Flat Array Works:

```
handleRenderNode(nodeID, worldMatrix, primMap):
  if node has no mesh → skip, continue traversal
  createRenderNodesForNode(nodeID, worldMatrix, visible=true, primMap):
    for each primitive in mesh:
      renderPrimID = primMap[primitiveKey]       -- deduplicated primitive index
      build RenderNode { worldMatrix, materialID, renderPrimID, refNodeID, skinID, visible }
      if node has EXT_mesh_gpu_instancing:
        handleGpuInstancing() → create N RenderNodes (one per instance transform)
      else:
        register single RenderNode in m_renderNodeRegistry
```

### Example Scene Graph → RenderNodes:

```
tinygltf::Model:                    RenderNodes[] (flat, contiguous):
  Scene.nodes = [0]                   
  Node[0] "Car"                     (No RenderNode - empty transform)
    ├─ children = [1, 2, 3]
    │
    ├─ Node[1] "Body"               → RenderNode[0]: {worldMat, matID=5, primID=10, refNode=1}
    │    mesh = 5 (1 primitive)
    │
    ├─ Node[2] "Wheels"             (No RenderNode - empty transform)
    │    ├─ children = [4, 5]
    │    │
    │    ├─ Node[4] "WheelFL"       → RenderNode[1]: {worldMat, matID=7, primID=12, refNode=4}
    │    │    mesh = 7 (1 primitive)
    │    │
    │    └─ Node[5] "WheelFR"       → RenderNode[2]: {worldMat, matID=7, primID=12, refNode=5}
    │         mesh = 7 (1 primitive)     ↑ SAME primID (instanced!)
    │
    └─ Node[3] "Engine"             → RenderNode[3]: {worldMat, matID=9, primID=15, refNode=3}
                                    → RenderNode[4]: {worldMat, matID=9, primID=16, refNode=3}
         mesh = 8 (2 primitives)         ↑ SAME node, multiple RenderNodes!
```

**Observations:**
- **Empty nodes** (Car, Wheels) → No RenderNodes (just hierarchy)
- **Multi-primitive mesh** (Engine) → Multiple RenderNodes (one per primitive)
- **Instancing** (WheelFL, WheelFR) → Different RenderNodes, same `primID`
- **Hierarchy depth irrelevant** → Flat RenderNodes array

---

## Stage 2: uploadRenderNodes() / syncFromScene() - Upload to GPU

### Function: `void SceneVk::uploadRenderNodes(staging, scene, dirtyIndices)` or `syncFromScene(staging, scene)`

**Location:** `gltf_scene_vk.cpp`

**Purpose:** Upload RenderNode data to GPU SSBO for shader access.

### GPU Data Structure:

Each render node uploads to the GPU as a `GltfRenderNode` (object↔world transforms plus
`materialID` and `renderPrimID`). The authoritative layout lives in
`shaders/gltf_scene_io.h.slang` and is consumed by `gltf_raster.slang` /
`gltf_pathtrace.slang` — read the header for the exact fields.

### Upload Logic:

```
uploadRenderNodes(staging, scene, dirtyIndices):
  renderNodes = scene.getRenderNodes()
  ensureRenderNodeBuffer(renderNodes.size)         -- recreate GPU buffer if size changed

  if buffer was recreated OR dirtyIndices is empty:
    Full upload: convert all RenderNodes → GltfRenderNode[], stage entire buffer
  else:
    Surgical upload: for each index in dirtyIndices,
      convert RenderNode[index] → GltfRenderNode, stage at byte offset
```

### Shader Access:

```
Rasterization (vertex shader):
  instance = renderNodes[pushConst.renderNodeID]    -- one draw per render node; index via push constant
  worldMatrix = instance.objectToWorld
  materialID  = instance.materialID

Ray Tracing (closest hit / any hit):
  instance = renderNodes[InstanceIndex()]           -- Slang: TLAS instance index
  worldMatrix = instance.objectToWorld
  materialID  = instance.materialID
```

### Emissive area lights (`SceneVk::uploadEmissiveTriangles`)

The path tracer treats emissive triangle **meshes** as area lights it samples directly (next-event
estimation + MIS in `sampleLights` / `gltf_pathtrace.slang`), instead of only finding them by chance
when a bounce ray happens to hit one. Punctual lights and the environment are sampled the same way.
Path tracer only — the rasterizer is unaffected.

**How it works, step by step**

Host — build the emitter list (`SceneVk::uploadEmissiveTriangles`):

1. Walk every **render node** (an instance of a primitive) and look at its material.
2. Keep it only if it is a **constant emitter**: emissive (`emissiveFactor × emissiveStrength`) is
   non-black **and** it has no emissive texture. (Textured emitters stay on the BSDF-only path.)
3. For each triangle of that primitive, store a **reference** — render node, primitive, triangle
   index — plus the **radiance** (not baked vertices, so the shader reads live geometry/transforms).
4. Compute the triangle's **world-space area** (verts × node transform) and give it a **selection
   weight = area × luminance** (see the defensive tweak below). Build a **Vose alias table** from the
   weights (`aliasProb` / `aliasIndex` per triangle) for O(1) GPU selection, plus the totals
   (`numEmissiveTriangles`, `emissiveTotalWeight`, `emissiveMeanLum`).
5. Upload the flat list + those scalars to the scene descriptor.

Device — sample a light at each surface hit (`sampleLights`):

6. See what light sources exist: **environment**, **punctual lights**, **emissive triangles**.
7. **Choose a category**: first "a light" vs "the environment", then within lights "a punctual light"
   vs "an emissive triangle" (the probabilities balance the categories).
8. **For an emissive triangle**: pick one in **O(1) via the alias table** (bright/large ones win more)
   → read its live world vertices → **sample a random point** on it → form direction/distance and the
   sample's **pdf** → evaluate the surface BSDF toward it → **cast a shadow ray**; if unblocked, add
   the light, **MIS-weighted** against BSDF sampling.
9. Separately, when a normal bounce ray **hits an emitter directly** (`gltf_pathtrace.slang`, the
   emissive add), its glow is added but **MIS-weighted**, so the "sampled via NEE" and "hit by chance"
   paths combine without double-counting.

**Details / why it's shaped this way**

The list is derived data, regenerated wholesale (never patched). It stores references, but each
emitter's **selection weight is its world-space area × luminance**, so it also depends on the
emitter's transform. `syncFromScene` rebuilds it when materials, the emitter set, or an emitter's
transform change; a `DirtyFlags::emissiveDirty` latch is consumed independently of the sync mask so
the GPU-transform path (which syncs only materials + lights) rebuilds it too. The shader samples the
emitter at its **live** GPU transform, and the pdf cancels the world area (below), so the host table
and the live geometry must agree on that area — a stale table biases the estimate, it does not merely
add noise. Because the GPU-transform path leaves the render-node CPU mirror lazily stale,
`Scene::updateLocalMatricesAndLights` refreshes emitters' world matrices on edits (as it does light
world matrices) and flags a rebuild when one moved.

Selection is **power-weighted** (an O(1) **Vose alias table**, not a CDF binary search), with a
**defensive floor** so a dim emitter next to a bright one is never starved: each emitter's luminance
is pulled toward the scene's area-weighted mean, `lum' = (1−f)·lum + f·meanLum`
(`EMISSIVE_DEFENSIVE_FRACTION`, kept in sync between host and shader). The alias table samples the
same distribution `P(i) = weightᵢ / emissiveTotalWeight`, so the pdf is unchanged.
The world area **cancels** in the solid-angle pdf (`pdf = lum' · dist² / (emissiveTotalWeight · cos)`),
so the BSDF-hit MIS weight is recoverable from local hit geometry (the emitter's radiance +
`emissiveTotalWeight` + `emissiveMeanLum`) with **no per-triangle lookup**; and because `meanLum = W/A`
the total weight `W` is unchanged. The weight uses world area, so the list is rebuilt whenever an
emitter's transform changes: interactive edits via `Scene::updateLocalMatricesAndLights`, and
animated/rigid moves via the world-matrix update (`updateWorldMatrices*` latch `emissiveDirty` when a
moving render node is an emitter). Morph/skin-deformed emitters are the one remaining case that falls
back to last-rebuilt-pose sampling. The scene-descriptor scalars refresh whenever the count
**or** the weights change, so radiance-only edits (e.g. dragging emissive strength) take effect
immediately.

---

## Stage 3: cmdCreateBuildTopLevelAccelerationStructure() - Build TLAS

### Function: `void SceneRtx::cmdCreateBuildTopLevelAccelerationStructure(cmd, staging, scene)`

**Location:** `gltf_scene_rtx.cpp`

**Purpose:** Build Top-Level Acceleration Structure (TLAS) for ray tracing.

### What It Does:

```
cmdCreateBuildTopLevelAccelerationStructure(cmd, staging, scene):
  drawObjects = scene.getRenderNodes()

  for each RenderNode in drawObjects:
    blasAddress = m_blasAccel[renderNode.renderPrimID].address
    if not renderNode.visible:
      blasAddress = 0                                   -- hide from ray traversal

    create VkAccelerationStructureInstanceKHR:
      transform              = renderNode.worldMatrix   -- 3x4 row-major
      instanceCustomIndex    = renderNode.renderPrimID  -- read in shader via InstanceID()
      accelerationStructureReference = blasAddress       -- which BLAS to use
      mask                   = 0x01
      flags                  = getInstanceFlag(material) -- cull mode from material

    append to m_tlasInstances[]

  upload m_tlasInstances[] to GPU via staging
  build TLAS acceleration structure on device
```

### TLAS Structure:

```
TLAS (Top-Level Acceleration Structure)
├─ Instance[0] → BLAS[primID=10] @ transform[worldMatrix]
├─ Instance[1] → BLAS[primID=12] @ transform[worldMatrix]  ← Instanced (same BLAS)
├─ Instance[2] → BLAS[primID=12] @ transform[worldMatrix]  ← Instanced (same BLAS)
├─ Instance[3] → BLAS[primID=15] @ transform[worldMatrix]
└─ Instance[4] → BLAS[primID=16] @ transform[worldMatrix]

Each BLAS (Bottom-Level AS):
  BLAS[primID] = acceleration structure for RenderPrimitive[primID]
    • Built from vertex/index buffers
    • Shared across multiple instances (e.g., wheels)
```

### Ray Tracing Hit Correlation:

```
When a ray hits TLAS instance[i] (Slang RT / ray query builtins):
  InstanceIndex()  = i             -- index into renderNodes SSBO
  InstanceID()     = renderPrimID  -- index into renderPrimitives / BLAS (instanceCustomIndex)
  Shader reads:
    renderNodes[InstanceIndex()]        → transform, materialID
    renderPrimitives[InstanceID()]      → vertex/index buffer info
```

---

## Data Structure Relationships

### RenderNode (CPU)

`struct RenderNode` (defined in `src/gltf_scene.hpp`) is one flattened primitive instance:
a world matrix computed during `parseScene()` traversal plus back-references
(`materialID` → model materials, `renderPrimID` → `m_renderPrimitives`, `refNodeID` →
model node, `skinID`) and a `visible` flag. See the header for the exact fields.

**Relationships:**
- **N:1 with Node** - Multiple RenderNodes per node (if mesh has multiple primitives)
- **N:1 with Primitive** - Multiple RenderNodes share same primitive (instancing)
- **N:1 with Material** - Many RenderNodes use same material

### RenderPrimitive (CPU)

`struct RenderPrimitive` (in `src/gltf_scene.hpp`) is a unique, deduplicated piece of
geometry: a pointer back to the `tinygltf::Primitive`, vertex/index counts, and the
owning `meshID`.

**Purpose:** Deduplicated geometry - if two nodes use same mesh, they share RenderPrimitives.

**Indexing:** `renderNode.renderPrimID` → `m_renderPrimitives[renderPrimID]`

### Mapping (CPU) – RenderNodeRegistry

```cpp
// RenderNodeRegistry: flat vector + bidirectional maps
// - getRenderNodes() → vector<RenderNode> (flat array for GPU upload)
// - getRenderNodesForNode(nodeID) → nodeID → list of RenderNode indices (unordered_map internally)
// - getNodeAndPrim(renderNodeID) → (nodeID, primIndex)

// Example:
// getRenderNodesForNode(3) = {5, 6}  // Node 3 has RenderNodes 5 and 6
//   → Node 3 has a mesh with 2 primitives
//   → RenderNode[5] and RenderNode[6] both have refNodeID = 3
```

---

## Critical Flows

### Flow 1: Initial Load



```
1. scene.load("file.gltf")
     → load tinygltf::Model from disk
     → parseScene()
        → build m_renderPrimitives[] (deduplicated, deterministic order)
        → build m_renderNodes[] (flat instances from hierarchy traversal)

2. sceneGpu.create(cmd, scene)   (drives SceneVk internally)
     → upload vertex/index buffers for all primitives
     → uploadRenderNodes() → create GPU SSBO, upload all GltfRenderNode[]

3. buildAccelerationStructures()
     → createBottomLevelAccelerationStructure() → prepare BLAS build data
     → cmdBuildBottomLevelAccelerationStructure() → GPU BLAS build (budgeted)
     → cmdCreateBuildTopLevelAccelerationStructure() → GPU TLAS build
```

---

### Flow 2: Transform Update (Animation, User Edit)



```
User edits a node transform (gizmo, inspector):
  scene.editor().setNodeTRS(nodeIdx, translation, rotation, scale)
    → modifies m_model.nodes[nodeIdx] TRS values
    → calls markNodeDirty(nodeIdx) → adds to m_dirtyFlags.nodes

On next frame, updateSceneChanges(cmd):
  1. updateSceneChanges_NodeTransforms():
       scene.updateNodeWorldMatrices()
         → recomputes world matrices for dirty nodes and descendants
         → updates RenderNode.worldMatrix in the registry
         (markNodeDirty() only queues node indices; updateNodeWorldMatrices()
          marks the affected render-node indices as it walks the hierarchy)
  2. sceneVk.syncFromScene(staging, scene)
       → reads dirty flags → uploads ONLY changed RenderNodes (surgical)
  3. sceneRtx.syncTopLevelAS(cmd, staging, scene)
       → updates TLAS instances with new transforms (rebuild or update)
```

**Note:** Animation updates follow a similar but separate path inline in the animation
processing block (not via `updateSceneChanges`). The same functions are called
(`updateNodeWorldMatrices`, `syncFromScene`, `syncTopLevelAS`)
but within the animation frame section, which also handles morph/skin GPU uploads.

**Optimization:** Only changed RenderNodes uploaded, not entire buffer.

---

### Flow 3: Hierarchy Change (Add/Delete/Duplicate Node)



```
User duplicates a node:
  scene.editor().duplicateNode(nodeIdx)
    → duplicateNodeRecursive() deep-copies nodes + subtree in m_model
    → links new subtree into parent's children (or scene roots)
    → calls parseScene() internally:
        snapshot → clearParsedData → buildPrimitiveKeyMap → traverse → diff
        → dirty flags: allRenderNodesDirty = true (count changed)

On next frame, updateSceneChanges(cmd):
  1. sceneVk.syncFromScene(staging, scene)
       → buffer size mismatch detected → recreate buffer → full upload
  2. sceneRtx.syncTopLevelAS(cmd, staging, scene)
       → instance count changed → full TLAS rebuild
```

**Key:** Hierarchy changes trigger **full rebuild** (not surgical update).

---

## Why This Design Works for Editing

### When you call `deleteNode(idx)`:

Note: `deleteNode` and `duplicateNode` are methods on `SceneEditor`, accessed via `scene.editor()`.

```
SceneEditor::deleteNode(nodeIndex):
  1. deleteNodeRecursive(nodeIndex):
       for each child (deepest first):
         removeNodeFromParent / removeNodeFromSceneRoots
         erase from m_model.nodes[]
         remapIndicesAfterNodeDeletion()        -- fix all node/animation/skin references
  2. parseScene()                              -- full rebuild, sets dirty flags

On next frame, updateSceneChanges(cmd):
  sceneVk.syncFromScene(...)                   -- resize + upload render nodes
  sceneRtx.syncTopLevelAS(...)                 -- rebuild TLAS
```

**Result:** Deleted node's RenderNodes disappear automatically (not in traversal anymore).

---

### When you call `duplicateNode(idx)`:

```
SceneEditor::duplicateNode(originalIndex):
  1. newIdx = duplicateNodeRecursive(originalIndex, originalParent)
       deep-copies node + all descendants in m_model.nodes[]
  2. Link new subtree under same parent (or as scene root)
  3. parseScene()                              -- full rebuild, sets dirty flags
  return newIdx

On next frame, updateSceneChanges(cmd):
  sceneVk.syncFromScene(...)                   -- resize + upload render nodes
  sceneRtx.syncTopLevelAS(...)                 -- rebuild TLAS
```

**Result:** Duplicated node's RenderNodes appear automatically (in traversal now).

---

## Performance Characteristics

The millisecond figures below are rough order-of-magnitude estimates for orientation, not
measured benchmarks — profile with the built-in tools (see `docs/benchmarking.md`) for real numbers.

### Surgical Update (Transform Only):
- **CPU:** O(N) where N = number of dirty nodes (typically 1-10)
- **GPU Upload:** Only changed RenderNodes (typically < 1KB)
- **Cost:** ~0.1 ms

### Full Rebuild (Hierarchy Change):
- **CPU:** O(N) where N = total nodes in scene (parseScene traversal)
- **GPU Upload:** All RenderNodes (typically 1-100 KB)
- **TLAS Rebuild:** O(N) instances
- **Cost:** ~1-5 ms (depends on scene complexity)

### Typical Scene Sizes:
- **Simple:** 10-50 nodes → 20-100 RenderNodes
- **Complex:** 100-500 nodes → 200-1000 RenderNodes
- **Very Complex:** 1000+ nodes → 2000+ RenderNodes

**Implication:** Full rebuilds are cheap enough to do on every hierarchy edit (< 5ms).

---

## Memory Layout

### CPU Side (Contiguous Vectors):

```
m_renderNodeRegistry.getRenderNodes()   -- contiguous vector<RenderNode>
m_renderPrimitives[]                    -- contiguous vector<RenderPrimitive>

Both are directly uploadable to GPU via staging (single memcpy-style transfer).
```

### GPU Side (SSBO):

```
┌────────────────────────────────────────┐
│ m_bRenderNode (SSBO)                   │
│ ┌────────────────────────────────────┐ │
│ │ GltfRenderNode[0]                  │ │ ← Instance 0
│ │ GltfRenderNode[1]                  │ │ ← Instance 1
│ │ GltfRenderNode[2]                  │ │ ← Instance 2
│ │ ...                                │ │
│ │ GltfRenderNode[N-1]                │ │ ← Instance N-1
│ └────────────────────────────────────┘ │
└────────────────────────────────────────┘
         ↑ Shader indexing: raster → nodes[pushConst.renderNodeID];
                            ray tracing → nodes[InstanceIndex()]
```

**Access Pattern:** Direct indexing (O(1)) in shaders.

---

## Important Invariants

### 1. RenderNodes Always Derived

Never modify render nodes in the registry directly.
Always modify the tinygltf Model, then call the appropriate rebuild:
- Transform change: `setNodeTRS()` + `updateNodeWorldMatrices()`
- Hierarchy change: editor operation (which calls `parseScene()` internally)

### 2. RenderNodes Index = TLAS Instance Index

`RenderNodes[i]` corresponds to `TLAS Instance[i]`.
When a ray hits instance `i`, `InstanceIndex() = i`, and the shader reads `renderNodes[i]` for transform and material.

### 3. RenderPrimitive Index = BLAS Array Index

`m_renderPrimitives[p]` corresponds to `m_blasAccel[p]`.
The BLAS array is stable across hierarchy-only edits (reparent, duplicate, delete of
existing meshes). It is rebuilt when the deduplicated primitive set changes
(`primitivesChanged`), and individual BLAS may be *updated* in place for animated
geometry (skinning / morph via `SceneRtx::updateBottomLevelAS()`). The TLAS is rebuilt
or updated when instances move, appear, or disappear.

Animated scenes build their BLAS with `ALLOW_UPDATE` and skip compaction; only static scenes
compact (see `GltfRenderer::buildAccelerationStructures()`). Vulkan permits compacting an
updatable BLAS and refitting it afterward, but this app's build+compact+per-frame-update path
triggers a GPU device-lost, so compaction is skipped for animated scenes as a workaround.

---

## When Structures Are Rebuilt

| Operation | Model | RenderNodes | GPU SSBO | TLAS |
|-----------|-------|-------------|----------|------|
| **Load scene** | ✅ New | ✅ Full rebuild | ✅ Create & upload | ✅ Create & build |
| **Merge scene** | ✅ New | ✅ Full rebuild | ✅ Full GPU recreation | ✅ Full rebuild |
| **Transform change** | ✅ Modified | ✅ Partial update | ✅ Surgical upload | ✅ Update instances |
| **Add empty node** | ✅ Modified | No change (no `parseScene()`; empty node has no mesh) | No GPU work | No GPU work |
| **Delete node** | ✅ Modified | ✅ Full rebuild | ✅ Resize & full upload | ✅ Rebuild |
| **Duplicate node** | ✅ Modified | ✅ Full rebuild | ✅ Resize & upload new | ✅ Rebuild |
| **Reparent** | ✅ Modified | ✅ Transforms only | ✅ Surgical upload (transforms) | ✅ Update instances |
| **Split mesh** | ✅ Modified | ✅ Full rebuild | ✅ Surgical upload (changed indices) | ✅ Update instances |
| **Merge mesh** | ✅ Modified | ✅ Full rebuild | ✅ Surgical upload (changed indices) | ✅ Update instances |
| **Material change** | ✅ Modified | ✅ Partial update | ✅ Surgical upload | Usually none; RTX instance flags may update for alpha-mode / double-sided edits |

**Key:** `parseScene()` always does a full CPU rebuild, but its internal diff sets precise dirty flags
so the GPU sync is surgical -- only changed render node indices, new materials, etc. are uploaded.

### Dirty Flag System

`parseScene()` snapshots the full render node state `(worldMatrix, materialID, renderPrimID, visible)` before
clearing, then compares after rebuild. This sets precise dirty flags:

- `renderNodesVk` / `renderNodesRtx`: indices where any field differs (surgical upload)
- `allRenderNodesDirty`: set when count changes or the dirty fraction reaches `kFullUpdateRatio` (`gltf_scene.hpp`) — full upload, avoids hash-set overhead
- `materials`: new material indices
- `primitivesChanged`: primitive count changed (BLAS rebuild needed)
- `lights`: all lights dirty if light count changed

The renderer has **one unified sync path** (`syncFromScene` + `syncTopLevelAS`) that processes these flags.
Buffer resize is handled automatically by size-mismatch detection in `uploadRenderNodes` and `rebuildTopLevelAS`.

#### Texture-set changes

Image/texture edits are consumed at frame top (before any panel records an `ImGui::Image`), not through
`syncFromScene`, and come in three forms:

- `texturesChanged` — a **structural** change (replace-in-place, reload, remove/insert at an arbitrary
  index, or a texture's sampler *reference*). `GltfRenderer::applyPendingTextureRebuild()` does a full
  `rebuildVulkanSceneFull()`: it stalls the queue, frees and recreates every scene image (re-reading them
  from disk) and rebuilds the acceleration structures.
- `texturesTailChanged` — a **tail-only** change: image(s)/texture(s) were appended to, or removed from,
  the end of the model with every lower index untouched (importing, and undoing/redoing an imported
  texture — see `SceneEditor::importImageAsTexture`). `GltfRenderer::applyPendingTextureTailSync()` calls
  `SceneVk::syncTextureTail()`, which loads/creates only the new tail images (or deferred-frees removed
  ones) and appends their views; the renderer then writes only the new bindless descriptor slots
  (`writeTextureDescriptorRange`). No queue stall, no re-read of existing images, no AS work. A first
  import into a scene that has no images/textures falls back to `texturesChanged`, because the empty scene
  carries 1×1 dummy defaults on the GPU that do not match the model sizes.
- `samplers` — an **in-place property** edit of an existing `model.samplers[i]` (wrap/filter, from the
  Inspector). Images, texture views and sampler *slot assignments* are untouched, so
  `GltfRenderer::applyPendingSamplerUpdate()` only recreates that one `VkSampler`
  (`SceneVk::updateSampler()`) and rewrites its single `eSamplers` descriptor slot. No queue stall, no
  image touch, no AS work.

`texturesChanged` and `texturesTailChanged` gate the per-frame material sync in `updateSceneChanges()`
until the frame-top reconcile has run, so the material buffer never references a texture index the
descriptor array does not yet contain. `samplers` needs no such gate — a sampler's slot index never
changes, only its `VkSampler` properties — but it is likewise preserved across `updateSceneChanges()`'s
end-of-frame `clearDirtyFlags()` so it survives until `applyPendingSamplerUpdate()` consumes it.

Merging or referencing a scene reuses the same tail idea from the (threaded) rebuild path rather than the
frame-top flags: because `SceneMerger` only appends (existing image/texture/material indices never move),
`rebuildVulkanSceneInternal(RebuildMode::eMergeAppend)` rebuilds geometry, render nodes and materials in
full (cheap, no disk) but calls `SceneVk::syncTextureTail()` to keep the resident textures and load only
the new tail images — so a merge no longer re-reads every image. A merge into a scene with no textures
still uses `eFull` (the empty-scene dummy defaults do not match the model, same reason as the import
fallback above).

### Debug Validation (debug builds only)

When the `m_validateGpuSync` flag is enabled (default: `true`), after every `updateSceneChanges()`, `validateGpuSync()` compares:
- Shadow copy of last-uploaded render nodes (materialID, renderPrimID) against current CPU state
- TLAS `instanceCustomIndex` against CPU `renderPrimID`
- Material buffer size against CPU material count

Any mismatch is logged as a warning with a descriptive error message.

---

## Code Locations Reference

### Scene (CPU)
- `parseScene()` - `gltf_scene.cpp`
- `handleRenderNode()` - `gltf_scene.cpp`
- `createRenderNodesForNode()` - `gltf_scene.cpp`
- `buildPrimitiveKeyMap()` - `gltf_scene.cpp`
- `updateNodeWorldMatrices()` - `gltf_scene.cpp`
- `updateRenderNodeDirtyFromNodes()` - `gltf_scene.cpp` (helper; not in the default transform/animation path — dirty flags are maintained incrementally)
- `markNodeDirty()` - `gltf_scene.cpp`

### SceneEditor (Editing)
- `duplicateNode()` / `duplicateNodeRecursive()` - `gltf_scene_editor.cpp`
- `deleteNode()` / `deleteNodeRecursive()` / `deleteNodeSingle()` - `gltf_scene_editor.cpp`
- `remapIndicesAfterNodeDeletion()` - `gltf_scene_editor.cpp`

### SceneVk (GPU Upload)
- `syncFromScene()` - `gltf_scene_vk.cpp`
- `uploadRenderNodes()` - `gltf_scene_vk.cpp`
- `uploadPrimitives()` - `gltf_scene_vk.cpp`
- `uploadMaterials()` - `gltf_scene_vk.cpp` (uploads the `shaderio::GltfShadeMaterial` array
  produced by `nvvkgltf::MaterialCache` in `gltf_material_cache.cpp`; the material struct
  is locally forked, see [developer.md → Material System](developer.md#material-system))
- `createVertexBuffers()` - `gltf_scene_vk.cpp`

### SceneRtx (Ray Tracing)
- `cmdCreateBuildTopLevelAccelerationStructure()` - `gltf_scene_rtx.cpp`
- `createBottomLevelAccelerationStructure()` - `gltf_scene_rtx.cpp`
- `syncTopLevelAS()` / `rebuildTopLevelAS()` - `gltf_scene_rtx.cpp`

### Renderer (Orchestration)
- `updateSceneChanges()` - `renderer.cpp` (unified sync path, processes all dirty flags)

---

## Debugging Tips

### To verify RenderNodes are correct:

After `parseScene()`, log each render node's `refNodeID`, `materialID`, and `renderPrimID`.
The count should match the expected number of mesh-primitive instances in the scene.

### To verify TLAS instances match:

After `cmdCreateBuildTopLevelAccelerationStructure()`, check that `m_tlasInstances.size()` equals `scene.getRenderNodes().size()`.
The built-in `validateGpuSync()` does this automatically when enabled.

### Common Issues:

1. **RenderNodes count wrong** → parseScene() not called after structural edit
2. **Invisible objects** → Check `renderNode.visible` flag
3. **Wrong transforms** → Check if `updateNodeWorldMatrices()` called
4. **Missing instances** → Check if node was added to scene.nodes[] (not just created)
5. **Wrong materials after structural edit** → Check `validateGpuSync()` output in debug build

---

## Summary

**Data is ONE-WAY:**
```
Model (authoritative) → parseScene() → RenderNodes (derived) → GPU
```

**Never modify RenderNodes directly** - always modify Model and rebuild.

**Hierarchy changes are cheap** - parseScene() + GPU upload ~1-5ms.

**This architecture makes editing simple:**
- Modify vectors in `m_model` directly
- Call `parseScene()` to regenerate RenderNodes
- Upload to GPU via `syncFromScene()` or `uploadRenderNodes()`
- Rebuild TLAS via `syncTopLevelAS()` or `cmdCreateBuildTopLevelAccelerationStructure()`

**No complex synchronization needed - derived data is always regenerated from source of truth.**

---

## Future Considerations

### If Performance Becomes an Issue:

**Option 1:** Split `parseScene()` into targeted rebuilds
- `rebuildPrimitivesAndRenderNodes()` for split/merge mesh
- `rebuildRenderNodesAndLights()` already exists and is used for adding lights;
  node delete/duplicate still go through a full `parseScene()`
- Only rebuild what the operation requires
- Medium complexity

**Option 2:** Incremental RenderNode updates
- Add/remove RenderNodes without full rebuild
- Remap TLAS instances manually
- Very complex, high bug risk

**Current Approach (Full CPU Rebuild + Surgical GPU Sync):**
- ✅ Simple and correct
- ✅ CPU rebuild < 5ms for typical scenes (~10-20ms for 1M nodes)
- ✅ GPU sync is surgical (only changed indices uploaded)
- ✅ Debug validation catches any drift
- ✅ One unified sync path in the renderer

**Recommendation:** Keep full CPU rebuild + surgical GPU sync. Split `parseScene()` later if profiling shows CPU cost matters at 1M+ node scale.
