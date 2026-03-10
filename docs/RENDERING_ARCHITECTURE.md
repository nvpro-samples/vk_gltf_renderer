# Rendering Architecture -- Vulkan Ray Tracing Data Flow

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
       allRenderNodesDirty               (count changed or >50% dirty)
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

```cpp
// GPU format (shaders/gltf_raster.slang, gltf_pathtrace.slang)
struct GltfRenderNode {
  mat4 objectToWorld;  // 64 bytes - Transform to world space
  mat4 worldToObject;  // 64 bytes - Inverse transform (for normals)
  int  materialID;     // Material array index
  int  renderPrimID;   // Primitive array index (vertex/index data)
};
```

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
  instance = renderNodes[gl_InstanceIndex]          -- direct indexing into SSBO
  worldMatrix = instance.objectToWorld
  materialID  = instance.materialID

Ray Tracing (closest hit / any hit):
  instance = renderNodes[gl_InstanceID]             -- TLAS instance index
  worldMatrix = instance.objectToWorld
  materialID  = instance.materialID
```

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
      instanceCustomIndex    = renderNode.renderPrimID  -- accessible in shader as gl_InstanceCustomIndexEXT
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
When a ray hits TLAS instance[i]:
  gl_InstanceID              = i                        -- index into renderNodes SSBO
  gl_InstanceCustomIndexEXT  = renderPrimID             -- index into renderPrimitives / BLAS
  Shader reads:
    renderNodes[gl_InstanceID]              → transform, materialID
    renderPrimitives[gl_InstanceCustomIndexEXT] → vertex/index buffer info
```

---

## Data Structure Relationships

### RenderNode (CPU)

```cpp
struct RenderNode {
  mat4 worldMatrix;   // Computed during parseScene() traversal
  int  materialID;    // → m_model.materials[materialID]
  int  renderPrimID;  // → m_renderPrimitives[renderPrimID]
  int  refNodeID;     // → m_model.nodes[refNodeID] (back-reference)
  int  skinID;        // → m_model.skins[skinID] (or -1)
  bool visible;       // Visibility flag
};
```

**Relationships:**
- **N:1 with Node** - Multiple RenderNodes per node (if mesh has multiple primitives)
- **N:1 with Primitive** - Multiple RenderNodes share same primitive (instancing)
- **N:1 with Material** - Many RenderNodes use same material

### RenderPrimitive (CPU)

```cpp
struct RenderPrimitive {
  tinygltf::Primitive* pPrimitive;  // → mesh.primitives[i] (pointer!) (speed up for Skin)
  int vertexCount;
  int indexCount;
  int meshID;                       // Which mesh this came from
};
```

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

2. sceneVk.create(cmd, staging, scene)
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
       scene.updateRenderNodeDirtyFromNodes(true)
         → converts dirty node indices to dirty render-node indices
       scene.updateNodeWorldMatrices()
         → recomputes world matrices for dirty nodes and descendants
         → updates RenderNode.worldMatrix in the registry
  2. sceneVk.syncFromScene(staging, scene)
       → reads dirty flags → uploads ONLY changed RenderNodes (surgical)
  3. sceneRtx.syncTopLevelAS(cmd, staging, scene)
       → updates TLAS instances with new transforms (rebuild or update)
```

**Note:** Animation updates follow a similar but separate path inline in the animation
processing block (not via `updateSceneChanges`). The same functions are called
(`updateRenderNodeDirtyFromNodes`, `updateNodeWorldMatrices`, `syncFromScene`, `syncTopLevelAS`)
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
         ↑ Shader indexing: nodes[gl_InstanceIndex]
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
When a ray hits instance `i`, `gl_InstanceID = i`, and the shader reads `renderNodes[i]` for transform and material.

### 3. RenderPrimitive Index = BLAS Array Index

`m_renderPrimitives[p]` corresponds to `m_blasAccel[p]`.
BLAS are built once during load (geometry does not change at runtime).
Only the TLAS is rebuilt when instances move, appear, or disappear.

---

## When Structures Are Rebuilt

| Operation | Model | RenderNodes | GPU SSBO | TLAS |
|-----------|-------|-------------|----------|------|
| **Load scene** | ✅ New | ✅ Full rebuild | ✅ Create & upload | ✅ Create & build |
| **Merge scene** | ✅ New | ✅ Full rebuild | ✅ Full GPU recreation | ✅ Full rebuild |
| **Transform change** | ✅ Modified | ✅ Partial update | ✅ Surgical upload | ✅ Update instances |
| **Add empty node** | ✅ Modified | ✅ Full rebuild | No GPU work | No GPU work |
| **Delete node** | ✅ Modified | ✅ Full rebuild | ✅ Resize & full upload | ✅ Rebuild |
| **Duplicate node** | ✅ Modified | ✅ Full rebuild | ✅ Resize & upload new | ✅ Rebuild |
| **Reparent** | ✅ Modified | ✅ Transforms only | ✅ Surgical upload (transforms) | ✅ Update instances |
| **Split mesh** | ✅ Modified | ✅ Full rebuild | ✅ Surgical upload (changed indices) | ✅ Update instances |
| **Merge mesh** | ✅ Modified | ✅ Full rebuild | ✅ Surgical upload (changed indices) | ✅ Update instances |
| **Material change** | ✅ Modified | ✅ Partial update | ✅ Surgical upload | No change |

**Key:** `parseScene()` always does a full CPU rebuild, but its internal diff sets precise dirty flags
so the GPU sync is surgical -- only changed render node indices, new materials, etc. are uploaded.

### Dirty Flag System

`parseScene()` snapshots the full render node state `(worldMatrix, materialID, renderPrimID, visible)` before
clearing, then compares after rebuild. This sets precise dirty flags:

- `renderNodesVk` / `renderNodesRtx`: indices where any field differs (surgical upload)
- `allRenderNodesDirty`: set when count changes or >50% of indices differ (full upload, avoids hash-set overhead)
- `materials`: new material indices
- `primitivesChanged`: primitive count changed (BLAS rebuild needed)
- `lights`: all lights dirty if light count changed

The renderer has **one unified sync path** (`syncFromScene` + `syncTopLevelAS`) that processes these flags.
Buffer resize is handled automatically by size-mismatch detection in `uploadRenderNodes` and `rebuildTopLevelAS`.

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
- `updateRenderNodeDirtyFromNodes()` - `gltf_scene.cpp`
- `markNodeDirty()` - `gltf_scene.cpp`

### SceneEditor (Editing)
- `duplicateNode()` / `duplicateNodeRecursive()` - `gltf_scene_editor.cpp`
- `deleteNode()` / `deleteNodeRecursive()` / `deleteNodeSingle()` - `gltf_scene_editor.cpp`
- `remapIndicesAfterNodeDeletion()` - `gltf_scene_editor.cpp`

### SceneVk (GPU Upload)
- `syncFromScene()` - `gltf_scene_vk.cpp`
- `uploadRenderNodes()` - `gltf_scene_vk.cpp`
- `uploadPrimitives()` - `gltf_scene_vk.cpp`
- `uploadMaterials()` - `gltf_scene_vk.cpp`
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
- `rebuildRenderNodesAndLights()` for node add/delete (already exists)
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
