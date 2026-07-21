# External Assets (glTF 2.1 complex scenes)

> **For contributors and agents.** Authoritative design of glTF 2.1 external-asset load / merge / edit / save. Concept-level; the symbols cited in `src/` are the source of truth for the details.

How `vk_gltf_renderer` loads, renders, edits, and saves glTF **2.1 external assets** — the
feature that lets one glTF file reference and instantiate *other* glTF files. This document is the
authoritative description of the mechanism; read it before touching the code paths below.

> **TL;DR** — A referencing (`externalAsset`) node's target file is loaded and **merged** into the
> single runtime `tinygltf::Model`, attached as children of that node. Merged-in nodes are flagged
> **read-only** by a marker written into each node's `extras`. On save, that marker is used to
> either strip the merged content back out (**re-externalize**, the default) or strip only the
> references and keep the geometry (**flatten**).

---

## 1. The glTF 2.1 data model

Three additions to core glTF, parsed/serialized by our fork of tinygltf
(`nvpro_core2/third_party/tinygltf/tiny_gltf.h`, branch `mkl/externalAssets`):

| JSON | tinygltf | Meaning |
|------|----------|---------|
| top-level `files[]`          | `Model::files` (`tinygltf::File`)           | A file reference: external `uri` **or** a `bufferView`, plus a required `mimeType`, plus optional `aliases`. |
| top-level `externalAssets[]` | `Model::externalAssets` (`tinygltf::ExternalAsset`) | An external glTF asset; its `file` indexes `files[]`. |
| `node.externalAsset`         | `Node::externalAsset` (int, `-1` = none)    | A node instantiating `externalAssets[externalAsset]` at its transform. |

A file with `node.externalAsset` set is a **complex scene**. The node carrying it is the
**instance node** (editable); everything loaded from the referenced file is **merged content**
(read-only).

> Note: tinygltf here is the **v2** header. v2 is in maintenance mode (sunset after mid-2026), but
> the whole renderer is built on v2, so external-asset support lives in v2. The runtime-only
> read-only marker is deliberately **not** a `Node` struct field (see §4) so the shared header is
> not polluted for a single application.

---

## 2. End-to-end flow

```text
LOAD  (Scene::load, gltf_scene.cpp)
  loadGltfFile(top.gltf) ────────────────► m_model (has files/externalAssets, node.externalAsset)
  resolveExternalAssets()  ◄── the merge step (§3)
      for each unique file:
          load child.gltf
          mergeIntoNode(child, firstRefNode)         → append child, attach roots as children
          instanceSubtree(...) for extra ref nodes   → share geometry (§3)
          recordReferencedAsset(...) → record provenance + stamp read-only marker into each merged node's extras (§4)
  parseScene()  → build RenderNodes / RenderPrimitives (dedup → shared BLAS)  (§6)

EDIT  (SceneEditor + UI)
  every mutator checks isNodeReadOnly() and refuses to touch merged content (§5)

SAVE  (Scene::save(filename, selfContained), gltf_scene.cpp)
  work on a COPY of m_model (live scene untouched):
    selfContained == false  → removeExternalAssetContent()  (re-externalize, default) (§7)
    selfContained == true   → flattenExternalAssets()       (bake, portable)          (§7)
```

The runtime `m_model` always holds the fully-merged scene; the two save transforms run on a
throwaway copy.

---

## 3. Merging references (`resolveExternalAssets` + `SceneMerger`)

`Scene::resolveExternalAssets()` (`gltf_scene.cpp`) runs inside `load()`, **after** `loadGltfFile`
and **before** `parseScene`. Algorithm:

1. **Group** referencing nodes by the `files[]` index they resolve to (via `externalAssets[]`).
   This is captured up front, before any merge appends nodes.
2. For each unique file: resolve its `uri` relative to the parent asset, load it with
   `loadGltfFile`, and validate.
3. **Merge once** under the *first* referencing node with `SceneMerger::mergeIntoNode`.
4. **Instance** the merged subtree under each *additional* referencing node with
   `SceneMerger::instanceSubtree`.
5. **Tag** every merged node read-only via `recordReferencedAsset` (which stamps the marker through `setExternalAssetMarker`) (§4).

The instance node's own `mesh`/`camera` are cleared (spec: `externalAsset` takes precedence).

### `SceneMerger` (`gltf_scene_merger.{hpp,cpp}`)

- **`mergeIntoNode(base, imported, targetNode)`** — appends *all* of `imported`'s resources
  (buffers, accessors, meshes, materials, skins, animations, nodes, …) into `base` with every
  index rebased by an `IndexRemapping` offset table, then attaches the imported default-scene
  **root nodes as children of `targetNode`** (no synthetic wrapper node — this matches the spec:
  *"append the externalAsset root node(s) to the array of children for the node"*). Returns a
  `MergeResult { firstNode, lastNode, roots }`.
- **`instanceSubtree(base, source, targetNode)`** — creates another instance of an
  already-merged subtree **sharing geometry**: it copies the node hierarchy (and duplicates skins
  and animation channels so instances animate independently) but leaves `mesh`/`material`/
  `accessor` indices pointing at the shared, already-appended resources. Because the copies
  reference the same accessors, `parseScene` deduplicates them to the same `RenderPrimitive` and
  hence the **same BLAS** — instancing, not duplication.

> **Why merge into one model?** The renderer is hard-bound to a single `tinygltf::Model`: all GPU
> arrays (materials/textures/vertex/index buffers) are flat and indexed into that one model, and
> ray tracing already instances a shared BLAS via TLAS transforms. Keeping external assets as
> separate coexisting models would require per-model index offsets threaded through the CPU build,
> the `shaderio` structs, the shaders, and the editor — a much larger change for no runtime win.

> **Index remapping is exhaustive — extend it when you add an extension.** Because everything is
> appended into one model, *every* index the imported model stores must be rebased by the matching
> `IndexRemapping` field (see `computeOffsets` / the `remapAndAppend*` helpers in
> `gltf_scene_merger.cpp`). This includes indices buried inside **extensions**, both in the core
> arrays they point at (accessors, bufferViews, textures, …) and in any per-model array the
> extension defines itself. When an extension owns a root-level array (e.g. `KHR_lights_punctual`
> `lights[]`, `EXT_mesh_opacity_micromap` `micromaps[]`), that array must be concatenated onto the
> base's, a new `IndexRemapping` offset added for it, and the per-node / per-primitive references to
> it (and the bufferView/accessor indices it stores) remapped. A missed remap is silent: it either
> drops the extension (merged into an empty base, offset 0, so the un-copied root array simply never
> appears) or cross-links it to the wrong resource (non-empty base). **If you add a glTF extension
> that stores any index, add its remapping here in the same change.**

### Nested references & cycle detection

A referenced file may itself reference other files. Before a child is merged, it is made
self-contained by `Scene::flattenReferencedModel` (`gltf_scene.cpp`), which recursively resolves the
child's own `externalAsset`s (loading, alias-applying, and merging them in place) so only one level
of merging ever happens against the runtime `m_model`.

Cycle detection uses a **DFS ancestry stack** — the canonical paths of the files currently on the
resolution chain:

1. `resolveExternalAssets` seeds the `ancestry` list with the canonicalized top-level scene file, so
   a child that references back to the root is caught.
2. Before loading each child, its path is canonicalized (`weakly_canonical(absolute(path))`) and
   checked against `ancestry`. A match is a back-edge → the reference is logged
   (`reference cycle detected`) and **skipped**.
3. The canonical key is pushed before recursing and popped after, so it is a true path stack. As a
   result **diamonds are not false positives** (`A→B→D`, `A→C→D` is fine) — only genuine cycles
   (`A↔B`, `A→B→C→A`, self-reference, child→root) are flagged.
4. A hard depth backstop (`kMaxExternalAssetDepth = 16`) stops runaway recursion even if the path
   check is defeated (see caveat in §9).

---

## 4. Read-only flagging (the `extras` marker)

Merged nodes must be recognisable as read-only *for the lifetime of the runtime model*, surviving
node renumbering (delete) and copying (duplicate). The mechanism:

- **Marker:** `Scene::kExternalAssetContentKey` = `"NV_external_asset_content"`, written into each
  merged node's **`extras`** (`gltf_scene.cpp`, `recordReferencedAsset`/`setExternalAssetMarker`). Its value is
  the index into `Scene::m_referencedAssets` (provenance: source URI + `file`/`externalAsset`
  indices).
- **Query:** `Scene::isNodeReadOnly(nodeIndex)` returns `node.extras.Has(marker)`;
  `Scene::isMeshReadOnly(meshIndex)` returns true if any read-only node uses that mesh.
- The **instance node itself is NOT marked** — it stays editable (you can move/rotate/scale the
  whole reference). Only the merged descendants are marked.

### Why `extras` and not a set or a struct field

- A `std::unordered_set<int>` of node indices **goes stale** the moment `deleteNode` renumbers the
  node array — the previous implementation had exactly this bug.
- A field on `tinygltf::Node` would pollute the **shared** tinygltf header for one app.
- `extras` lives *inside* the node, so it is copied by `duplicateNode` and travels through
  `deleteNode` renumbering automatically — zero bookkeeping. It is standard glTF app-data.

> ⚠️ **Invariant — never replace a node's `extras` wholesale.** Any code that does
> `node.extras = tinygltf::Value(newObject)` will wipe the marker, un-tagging merged content. Such
> a node then leaks into saves and **duplicates on reload**. Always *merge* keys into the existing
> `extras` object. (`applyRenderCameraToNode` in `gltf_scene.cpp` had this bug — it now preserves
> existing extras.)

---

## 5. Editing guards

The marker is enforced so referenced content cannot be edited, while the instance node stays free.

- **Editor** (`gltf_scene_editor.cpp`): `blockIfNodeReadOnly(nodeIndex, op)` early-returns (with a
  log) from every mutator that could change merged content — `setNodeTRS`, `renameNode`,
  `duplicateNode`, `deleteNode`, `setNodeParent`, `setNode*`/`clearNode*` (mesh/camera/skin),
  `duplicateMeshForNode`, `split`/`mergePrimitiveMaterial`. `setPrimitiveMaterial` is mesh-keyed so
  it guards via `isMeshReadOnly`. Undo commands and drag-drop route through these methods, so they
  are covered too.
- **UI**:
  - `ui_inspector.cpp` shows a `ICON_MS_LOCK Referenced asset (read-only)` badge and wraps the transform /
    light / camera / material controls in `ImGui::BeginDisabled(readOnly)`.
  - `ui_scene_browser.cpp` skips read-only roots in the **scene-transform** apply loop
    (`applySceneTransform`, which edits nodes directly, bypassing the editor) and disables the node
    context-menu mutations (duplicate/delete/add-child/rename) for read-only nodes. The per-node
    viewport gizmo instead routes through `SceneEditor::setNodeTRS`, which is guarded by
    `blockIfNodeReadOnly`.

---

## 6. Rendering / instancing

`parseScene` builds one **`RenderPrimitive`** per unique geometry (dedup key = attribute + index
accessor indices) and one **`RenderNode`** per (node, primitive) instance. Ray tracing builds one
BLAS per `RenderPrimitive` and one TLAS instance per `RenderNode`. Because `instanceSubtree` shares
accessors across references, N references to the same file yield **one BLAS, N TLAS instances** —
verify with the BLAS count when a file is referenced more than once.

---

## 7. Saving

`Scene::save(filename, selfContained)` (`gltf_scene.cpp`) transforms a **copy** of `m_model` when
`m_referencedAssets` is non-empty, then serializes the copy. Two forms:

### Re-externalize — `removeExternalAssetContent(model)` (default, `selfContained == false`)

Writes the **complex-scene** form (small top file that references the external files):

1. Remove every node with the marker, remapping all node-index references (children, scene roots,
   skin joints/skeleton, animation channel targets; empty animations erased).
2. Run the resource-compaction cascade (`collectReferencedResources` → remap tables →
   `extractUsedElements` → `updateAllReferences` → `::compactModel`) to drop the now-orphaned
   meshes/materials/skins/animations/accessors/etc.
3. Drop any buffer no `bufferView` references (a geometry-less complex scene has **no** buffer; an
   empty `byteLength: 0` buffer is invalid glTF and fails to load).

Result: instance nodes (keeping `externalAsset`) + the `files`/`externalAssets` tables, no inlined
geometry, no markers.

### Flatten — `flattenExternalAssets(model)` (`selfContained == true`)

Produces a **portable** self-contained file (shareable without the referenced assets): keeps the
merged geometry inline and instead strips every *reference* — clears `node.externalAsset`, removes
the marker from `extras`, and clears the `files`/`externalAssets` tables. The former instance nodes
become ordinary transform nodes.

Both functions live in `gltf_compact_scene.cpp` (declared in `gltf_compact_model.hpp`) and run on a
copy, so the live runtime scene is never disturbed.

### Image relocation (`.gltf` only)

When writing a text `.gltf` (not `.glb`), `Scene::save` relocates referenced `image.uri` files next
to the saved file. It resolves each URI through the load-time image search paths, then: **keeps in
place** any image that already resolves to the destination (a save-in-place, or a save-as into a
folder that already holds the referenced files, does not duplicate into `images/`); **deduplicates**
by canonical source path so a file shared by several entries is copied at most once; and otherwise
**copies** into `images/` with a uniquified name and rewrites the URI. See the `uriBySource` loop in
`Scene::save` (`gltf_scene.cpp`) for the exact rules.

---

## 8. File / symbol map

| Concern | Location |
|---|---|
| Parse/serialize `files`/`externalAssets`/`node.externalAsset` | `tiny_gltf.h` (nvpro_core2, branch `mkl/externalAssets`) |
| Resolve + merge on load | `Scene::resolveExternalAssets`, `gltf_scene.cpp` |
| Merge / instance primitives | `SceneMerger::mergeIntoNode`, `SceneMerger::instanceSubtree`, `MergeResult`, `gltf_scene_merger.{hpp,cpp}` |
| Read-only marker + queries | `Scene::kExternalAssetContentKey`, `isNodeReadOnly`, `isMeshReadOnly`, `ReferencedAsset`, `gltf_scene.{hpp,cpp}` |
| Editor guards | `SceneEditor::blockIfNodeReadOnly` + guarded mutators, `gltf_scene_editor.cpp` |
| UI (badge / disabled controls / gizmo skip) | `ui_inspector.cpp`, `ui_scene_browser.cpp` |
| Save (re-externalize / flatten) | `Scene::save`, `removeExternalAssetContent`, `flattenExternalAssets`, `gltf_scene.cpp` + `gltf_compact_scene.cpp` |

---

## 9. Known limitations / deferred work

- **Embedded external assets** (`file.bufferView` / `data:` URI) are not yet resolved — only
  external file `uri`s. This also applies to `FileAlias` targets: an alias whose `file` is stored
  in a `bufferView`/`data:` URI is skipped (only `uri`-backed alias targets are honored).
- **Save to a different folder** does not relocate/copy the referenced files or rewrite relative
  `file.uri`s (fine for save-in-place).
- **Cycle detection is path-string based.** The ancestry check (§3) compares canonical paths with a
  case-sensitive string compare, and `weakly_canonical` does not normalize a path to its on-disk
  casing. On case-insensitive filesystems (Windows) a cycle reached through differently-cased paths
  (`a.gltf` ↔ `A.gltf`) or via hardlinks can slip past the check. In that case the
  `kMaxExternalAssetDepth = 16` backstop still terminates recursion (with redundant merges and a
  warning) rather than hanging.

Implemented (glTF 2.1): nested external assets are resolved recursively with a cycle guard,
`FileAlias` inner-URI redirection is applied at load time, and `file.uri` is percent-decoded.
