# glTF 2.1 support

> **For contributors and agents.** Where glTF 2.1 stands in this renderer, how we work against a spec
> that is not final yet, and where 2.1 parsing comes from. The code is authoritative for behavior;
> this page is the map.

glTF 2.1 is a backward-compatible revision of glTF 2.0
([KhronosGroup/glTF#2585](https://github.com/KhronosGroup/glTF/issues/2585)). Its normative text and
schemas are still being written in the Khronos 3D Formats working group. We implement it ahead of
ratification on purpose: the renderer is a place to try the features out and to report problems back
to the working group. So parts of our implementation rest on draft text that may still change.

---

## 1. Status

| Feature | Khronos issue | State | Where |
|---|---|---|---|
| Load / save 2.1 assets (version policy, merger remapping) | [#2585](https://github.com/KhronosGroup/glTF/issues/2585) | Loads and round-trips the new properties; version policy done (§2). Merger remapping not done. | `src/gltf_version.cpp`, `Scene::load` / `Scene::save` |
| External assets, unified `files` | [#2586](https://github.com/KhronosGroup/glTF/issues/2586), [#2590](https://github.com/KhronosGroup/glTF/issues/2590) | Mostly done: each file loaded once per scene, static content drawn as instance sets | [external_assets.md](external_assets.md) |
| Packaging (`files` aliases, embedded assets) | [#2589](https://github.com/KhronosGroup/glTF/issues/2589) | `uri` aliases only | [external_assets.md](external_assets.md) §9 |
| Promoted extensions (WebP, emissive strength, quantization, node visibility) | [#2592](https://github.com/KhronosGroup/glTF/issues/2592) | Read as extensions only; core properties wait on the spec | — |
| GLB v3 (64-bit lengths, several BIN chunks) | [#2594](https://github.com/KhronosGroup/glTF/issues/2594), [#2611](https://github.com/KhronosGroup/glTF/issues/2611) | tinygltf reads single-chunk v3 (no test file yet); the renderer always writes v2 | `tiny_gltf.h` |
| New accessor component types, integer attributes | [#2596](https://github.com/KhronosGroup/glTF/issues/2596), [#2606](https://github.com/KhronosGroup/glTF/issues/2606) | Never crash: positions or indices the renderer can't read skip their primitive with a warning; other attributes it can't read are treated as absent; 32-bit integer custom attributes load. | `tinygltf::utils::canReadAccessor`, `primitiveGeometryError` (`src/gltf_scene.cpp`) |
| Non-sequential `TEXCOORD_n` / `COLOR_n` | [#2595](https://github.com/KhronosGroup/glTF/issues/2595) | Not started | `src/gltf_material_cache.cpp`, `src/gltf_create_tangent.cpp` |
| `uid` | [#2597](https://github.com/KhronosGroup/glTF/issues/2597) | Passed through: loaded and saved as is, never checked or rewritten (merges keep every uid) while the draft settles what a uid must satisfy. Only a copy made by the editor drops it, so a save doesn't write one twice. Not used for geometry identity: a uid is unique within its file only ([external_assets.md](external_assets.md) §6). | `tests/test_uid.cpp` |
| Multiple scenes deprecation | [#2591](https://github.com/KhronosGroup/glTF/issues/2591) | Done: multi-scene files load and honor `scene`; saves keep every scene, with a warning when saved as 2.1 | `Scene::save`; test `RoundTripTest.MultipleScenes` |
| Shapes, bounding volumes | [#2588](https://github.com/KhronosGroup/glTF/issues/2588), [#2587](https://github.com/KhronosGroup/glTF/issues/2587) | Preserved on load/save; not used | — |
| Thumbnails | [#2593](https://github.com/KhronosGroup/glTF/issues/2593) | Preserved on load/save; not shown | — |

"Preserved" means the regression tests in `tests/test_gltf_2_1.cpp` check that the property comes
back unchanged after a save. Detailed task lists live in the project's issue tracker
(label `glTF-2.1`).

---

## 2. Versions: what loads, what a save declares

`src/gltf_version.{hpp,cpp}` owns both rules; every load (scene, external asset, merge) goes through
`loadGltfFile`, and `Scene::save` sets the version.

- **Loading** follows the 2.0 spec's "Versioning" and "Asset" sections: a file loads if its major
  version is ours, or, when it has a `minVersion`, if we support that minor version too. A newer
  minor version (2.x) loads with a warning: what we don't understand is ignored, and so lost on save.
  Malformed version strings, and a `minVersion` above `version`, are refused.
- **Saving** declares the lowest version the content needs: `"2.1"` when the model uses a 2.1
  feature (`findGltf21Feature`) or its `minVersion` asks for 2.1, otherwise `"2.0"`, so the file
  stays readable by 2.0-only clients. The loaded version is not kept: a newer minor version's unknown
  content is gone, and a 2.1 file whose 2.1 content was removed (e.g. by flattening external assets)
  is honestly 2.0. `minVersion` is kept.
- The 2.1 feature list is `PROVISIONAL`: it follows the current draft and grows as tinygltf learns
  the remaining properties (the promoted extensions of Khronos MR 19, in particular).

---

## 3. Working against a draft spec: `PROVISIONAL` markers

We don't wait for the spec to be final. Where the draft is silent, ambiguous, or still under review,
pick the most likely reading, implement it, and **mark it**:

```cpp
// glTF 2.1 PROVISIONAL(Khronos MR 19): core node.visible wins over KHR_node_visibility when both are set.
```

- **Format:** `PROVISIONAL(<reference>): <the assumption>`. The reference is the Khronos issue or
  merge request whose outcome decides it, so the marker says when to come back.
- **Everywhere:** the same marker goes in C++/Slang comments, in `tests/gltf_2_1/generate_fixtures.py`,
  and in docs (as plain text or `<!-- PROVISIONAL(...) -->`).
- **Tested:** every provisional behavior gets a fixture and a test. When the spec settles the other
  way, a test fails instead of the behavior silently staying wrong.
- **Revisiting:** when a Khronos change merges, grep for its reference and resolve each hit. Delete
  the marker once the code matches the ratified text.

```bash
git grep -n "PROVISIONAL("
```

Plain `glTF 2.1:` comments mark 2.1 code that follows settled text; `PROVISIONAL` is only for
choices that may change.

---

## 4. tinygltf fork

2.1 parsing and serialization live in **our fork of tinygltf**,
[nvpro-samples/tinygltf](https://github.com/nvpro-samples/tinygltf) branch `nv/all-fixes`. It is
copied into `nvpro_core2/third_party/tinygltf` by the third-party sync rules
(`nvpro_core2/third_party/repro_rules.json`, which also record the fork commit currently vendored).

**Why a fork, and for how long.** Upstream ([syoyo/tinygltf](https://github.com/syoyo/tinygltf)) has
not reviewed our changes, and its successor, the pure-C "v3" API, is still experimental and changes
its API. The renderer is built on the C++ v2 header (`tiny_gltf.h`), so 2.1 support goes there. We
keep the fork until v3 is stable enough to port to, or upstream takes the changes.

**Rules for changes to the fork:**
- Additive only: new fields and parse/serialize code, with existing 2.0 behavior unchanged.
- Each 2.1 addition carries a `glTF 2.1` comment (or `PROVISIONAL(...)`), so a later port to v3 or
  upstream can find every one of them.
- Land the change on `nv/all-fixes`, then update the vendored copy in nvpro_core2 in a separate
  commit. Add a row to the patch log below.

**Patch log** (fork-only commits on `nv/all-fixes`):

| Commit | Change | Spec |
|---|---|---|
| `03e8c36` | `files`, `externalAssets`, `node.externalAsset`, file aliases | Khronos MR 18 |
| `580f925` | `uid`, `shapes`, new component types, `node.boundingVolume` | Khronos MR 6, 11, 17 |
| `1bdf3ca` | GLB v3 read and write (64-bit lengths, chunk encoding) | [#2594](https://github.com/KhronosGroup/glTF/issues/2594) |
| `107dbb9` | Shape / bounding-volume defaults, `operator==`, no `matrix` on bounding volumes | Khronos MR 17 |
| `0173613` | `plane` shape | Khronos MR 17 |
| `5bd5c48` | Bounding volume without `shape` fails to parse | Khronos MR 17 |
| `2314b0e` | Comments and trailing commas in JSON (not 2.1) | — |
| `f85f9ff` | CMake / MSVC warning fixes (not 2.1) | — |
| `830ab62` | Write `asset.minVersion` (it was parsed but never serialized) | glTF 2.0 "Asset" |
| `a76edf3` | Accept float16 / int64 / uint64 accessors (any file using one failed to load) | Khronos MR 6 |

**Known gaps in the fork:**
- `uid` is parsed on images, textures, meshes, nodes, scenes, materials, animations, samplers,
  skins, cameras and lights, but not on accessors, buffers, buffer views, files, external assets or
  shapes. <!-- PROVISIONAL(Khronos MR 11): confirm which objects may carry a uid. -->
- Several BIN chunks (`buffer.chunk`), and a JSON chunk that is not first, are not read.
- `convexMesh` and `mesh` shapes from the current draft are not parsed.
- The core properties promoted by Khronos MR 19 (`material.emissiveStrength`, `node.visible`, WebP
  without the extension) are not parsed.

---

## 5. Test fixtures

`tests/gltf_2_1/generate_fixtures.py` describes small `.gltf` files, one per feature area, that
`tests/test_gltf_2_1.cpp` loads and round-trips: `core_properties.gltf` groups the small core
properties (thumbnail, shapes, bounding volumes, `uid`), and external assets get two folders
(`external/` for references, files and images; `placement/` for how placed content is drawn). None of the official sample assets use 2.1 yet, so
these are the only regression coverage.

- **Generated at build time, never committed.** Only the script is in the repository. The test
  build runs it into `<build>/tests/gltf_2_1/` whenever it changes (`tests/CMakeLists.txt`), so the
  fixtures can't drift from their source. The build needs Python 3 for this; without it the 2.1
  tests skip. Buffers and images are `data:` URIs, so a file stands alone unless the feature is
  about referencing other files.
- **Minimal on purpose.** A fixture checks one feature area and is small enough to read. Polished
  showcase models belong in [glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets),
  where we intend to contribute them once the spec is ratified.
- **Adding one:** add a function to the script, add a test that loads the result, and name the
  feature in the test's comment. Building the tests regenerates the fixtures. To look at them in the
  renderer, generate them anywhere:

```bash
python tests/gltf_2_1/generate_fixtures.py out/gltf_2_1
```
