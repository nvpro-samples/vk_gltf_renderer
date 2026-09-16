# CLAUDE.md — Agent guide for vk_gltf_renderer

Read this first. It is a **router**, not a manual: it points you at the right
doc and states the rules to follow. Keep it thin — deep knowledge lives in
`docs/`. If you change an architectural invariant, update the relevant `docs/`
file **and** this file in the same commit.

## What this is

A Vulkan **ray tracing** renderer for glTF 2.0 scenes, built on
[nvpro_core2](https://github.com/nvpro-samples/nvpro_core2) with
[Slang](https://github.com/shader-slang/slang) shaders. Two render paths share
the same Vulkan resources and scene data:

- **RTX path tracer** (`renderer_pathtracer`) — the authoritative, high-quality
  reference for glTF PBR material evaluation and new extensions.
- **Rasterizer** (`renderer_rasterizer`) — a fast preview path for interaction
  and editing. Not the material reference.

C++20. Host and device share structs via `shaders/shaderio.h`.

## Where to read before you code (do this instead of grepping the whole tree)

| Your task | Read |
|---|---|
| Understand overall architecture, app lifecycle, source map | [docs/developer.md](docs/developer.md) |
| Scene data flow: Model → RenderNodes → GPU SSBO / BLAS / TLAS | [docs/RENDERING_ARCHITECTURE.md](docs/RENDERING_ARCHITECTURE.md) |
| glTF 2.1 multi-file scenes, read-only flagging, re-externalize on save | [docs/external_assets.md](docs/external_assets.md) |
| KHR_interactivity behavior graphs: node execution, event wiring, coverage status | [docs/interactivity.md](docs/interactivity.md) |
| MCP shader timing: enabling `--mcp`, the tool surface, measuring a change | [docs/mcp.md](docs/mcp.md) |
| Runtime behavior, editor workflows, features | [docs/user-guide.md](docs/user-guide.md) |
| DLSS / OptiX denoising, motion vectors, jitter/reset (incl. why animated meshes ghost) | [docs/denoising.md](docs/denoising.md) |
| Headless timing / scripted GPU benchmarks | [docs/benchmarking.md](docs/benchmarking.md) |
| Capture the UI (panels + viewport) or script-drive selection to review UI/UX | [docs/benchmarking.md](docs/benchmarking.md) (§ UI inspection) |
| Test suite: running (CTest vs. direct), adding tests, benchmarks | [tests/README.md](tests/README.md) |
| Where scene/model assets come from | [docs/external_assets.md](docs/external_assets.md), [docs/resources.md](docs/resources.md) |

`docs/developer.md` contains the full annotated `src/` and `shaders/` file map —
use it to locate a subsystem instead of fanning out readers across the tree.

## Source of truth — read the code, not the doc

For anything the compiler or loader **enumerates or enforces**, the code is authoritative and
docs drift. Grep the symbol below instead of trusting a prose list, and when a doc disagrees
with the code, the code wins (fix the doc — see "Keep the docs true").

| You need… | Grep / read this (authoritative) |
|---|---|
| Supported glTF extensions | `m_supportedExtensions` in `src/gltf_scene.cpp` |
| Visualization / debug modes | `enum Visualization` in `shaders/shaderio.h` |
| Command-line / MCP / persisted settings (names, defaults, whether they persist) | `m_settings.add(...)` in `src/renderer*.cpp`, plus `parameterRegistry.add(...)` in `src/main.cpp` and `src/benchmarking.cpp` |
| Menu labels, keyboard shortcuts, UI panels | menu/UI builders in `src/ui_renderer.cpp` (and other `src/ui_*`) |
| Elements-list categories, columns, CRUD | `ensureElementRegistry()` (the `ElementTypeDesc` array) in `src/ui_scene_browser_elements.cpp` |
| Material extension gates | `MAT_EXT_*` in `shaders/gltf_material_config.h`, `GLTF_USE_*` in `shaders/gltf_eval_config.h` |
| Host/device structs & layout | `shaders/shaderio.h` + the `*_io.h.slang` / `*_shaderio.h.slang` headers |
| Which files are tests/benchmarks | `tests/CMakeLists.txt` |
| MCP tool names, schemas, and annotations | `registerTool(...)` in `src/mcp_timing.cpp` |
| CMake options & defaults | root `CMakeLists.txt` |

## Source layout (one line each; full map in developer.md)

- `src/renderer*.{cpp,hpp}` — render backends (`BaseRenderer` → `PathTracer` /
  `Rasterizer`), plus a separate `Silhouette` compute pass, all orchestrated by
  `renderer.cpp` (`GltfRenderer`).
- `src/gltf_scene*.{cpp,hpp}` — scene loading, GPU upload (`SceneVk`),
  acceleration structures (`SceneRtx`), editing, animation, merge, compaction.
- `src/ui_*` — ImGui panels (inspector, scene browser, animation); viewport and
  menus live in `ui_renderer.cpp`. The Scene Browser's **Elements** tab is
  data-driven: one `ElementTypeDesc` per glTF collection
  (`ui_scene_browser_elements.cpp`) feeds a single generic list renderer.
- `src/mcp_timing.{cpp,hpp}` — the optional MCP endpoint (`--mcp`): recompile
  shaders and time a GPU pass, so an agent can measure a shader change. Loading
  and every other setting go through the parameter registry instead. Built only
  when `NVPRO2_ENABLE_nvmcp` is on. See [docs/mcp.md](docs/mcp.md).
- `src/dlss*`, `src/optix_denoiser*`, `src/vk_cuda*` — AI denoisers + CUDA interop.
- `src/gizmo_*` — transform gizmo, grid, overlays.
- `shaders/*.slang` + `shaders/shaderio.h` — GPU code and host/device structs.

## Invariants — do not break these

- **The two render paths share resources.** Scene buffers, textures, and
  descriptors are built once and consumed by both. A change for one path must
  not fork or duplicate that shared state — if you touch shared resources,
  verify both paths still work.
- **RenderNode / RenderPrimitive arrays are derived and regenerated** from the
  glTF model (see RENDERING_ARCHITECTURE.md). Don't hand-mutate them as if they
  were source of truth; drive changes from the model / editor path so GPU SSBO
  and TLAS stay in sync.
- **The path tracer is the material reference.** New glTF material extensions
  land in the path tracer's Slang material eval first
  (`shaders/gltf_material_eval.h.slang`, gated by `MAT_EXT_*` flags in
  `shaders/gltf_material_config.h`). The rasterizer follows, it does not lead.
- **Host/device structs live in `shaders/shaderio.h`** (and the `*_io.h.slang` /
  `*_shaderio.h.slang` headers). Change both sides together; never let CPU and
  GPU layouts drift.
- **Every user-settable value is declared once**, through `SettingsRegistry`
  (`src/settings_registry.hpp`). One call feeds the command line, benchmark
  sequences, MCP, and the persisted `.ini`; `Persist` says whether it survives a
  restart. Never register a setting in two places again — that is what let the
  two old lists drift.
- **Scene editing is non-destructive** and fully undoable — route mutations
  through the editor + `undo_redo` system, not ad-hoc.
- **Scene, model, and Vulkan state is application-thread-owned.** An MCP tool
  that touches them sets `runOnApplicationThread` and stays short. A tool that
  waits on rendered frames must not, since the application thread is the one
  producing them — `GltfRenderer::measureTimer` is the deliberate exception and
  reads only the profiler's mutex-guarded snapshots, never the scene.

## Fix what you find — auto-fix on discovery

When you find a bug, drift, gap, or clear defect while working — **even if it's outside the scope
of the current task** — fix it in the same change instead of only reporting it. Don't leave
known-bad code or docs behind. ("Keep the docs true" below is the docs-specific instance of this.)

- **Fix it now:** typos, stale/incorrect docs, dead code, obvious bugs, missing null checks, wrong
  comments. Root-cause it (no symptom patches), prefer the elegant fix, and verify (build / test /
  inspect) before claiming done.
- **Ask first:** scope-expanding or architectural changes, public API / on-disk-format / behavior
  changes, anything destructive (deleting files, migrations, history rewrites), or anything that
  would break an **Invariant** above.
- **Can't fix it now?** Leave a visible marker and call it out in your summary so the next reader
  (or a grep) finds it — e.g. `// TODO:` or `<!-- DRIFT? doc says X; code does Y -->`. Never
  silently leave known-bad text, and never guess a fix you can't verify.

## Keep the docs true — every agent maintains this

The **code is the source of truth.** The docs (`docs/`, this file, `README.md`)
exist to explain *how things work at a high level* and to point at *where* they
live. They must never restate what the code already says — no copied signatures,
no exhaustive enumerations the code defines, no line-by-line narration that
drifts the moment someone edits.

This is a public, NVIDIA-facing repository read by many developers. Hold the
docs to the same bar as the code: accurate, concise, professional. As you work,
you are expected to keep them that way:

- **Fix drift when you see it.** If a doc is wrong, stale, or contradicts the
  code — even if it's outside your task — correct it or, if you can't verify the
  intended behavior, flag it clearly rather than leaving known-bad text.
- **Update docs in the same change as the code.** When you alter behavior that a
  doc describes (an invariant, a data-flow step, a workflow, a build option),
  update that doc and this file together. A code change that invalidates a doc
  is not complete until the doc is fixed.
- **Complete what's missing.** When you add a feature or glTF extension, add its
  high-level explanation and a pointer to the right file — enough for the next
  reader to find and understand it, not a re-implementation in prose.
- **Verify before you trust.** A doc may lag the code. Before relying on a doc claim that
  names a symbol, flag, value, or path, grep it (see "Source of truth"). If it no longer
  exists or disagrees with the code, treat the code as correct and repair the doc.
- **Flag what you can't verify.** If you spot likely drift but can't confirm the intended
  behavior, leave a visible marker instead of silent bad text — e.g.
  `<!-- DRIFT? doc says X; src/foo.cpp seems to do Y -->` — so the next reader (or a grep)
  finds it.

**Never hard-code in docs** (these are exactly what drifts — link to the symbol instead):

- Enumerations the code defines (extensions, visualization modes, CLI flags, enum values).
- Exact numeric values the code owns (thresholds, sizes, offsets, frame counts).
- Verbatim struct layouts or function signatures — describe the role, point to the header.
- Source line numbers — reference a symbol or section anchor, never `around line N`.

**When you change code, update its doc (reverse of the routing table):**

| You changed… | Update |
|---|---|
| Scene sync / dirty-flag / BLAS-TLAS data flow | [docs/RENDERING_ARCHITECTURE.md](docs/RENDERING_ARCHITECTURE.md) |
| App lifecycle, source layout, material fork, build/test | [docs/developer.md](docs/developer.md) + this file |
| A user-facing feature, menu, or shortcut | [docs/user-guide.md](docs/user-guide.md) |
| DLSS/OptiX, motion-vector / jitter / reset behavior | [docs/denoising.md](docs/denoising.md) |
| glTF 2.1 external-asset load/save/edit behavior | [docs/external_assets.md](docs/external_assets.md) |
| KHR_interactivity node coverage, execution model, event wiring | [docs/interactivity.md](docs/interactivity.md) |
| MCP tools or the timing measurement | [docs/mcp.md](docs/mcp.md) |
| Headless/benchmark flags or output format | [docs/benchmarking.md](docs/benchmarking.md), `utils/benchmark/README.md` |
| Added a glTF material extension | path tracer eval first, then README extension list + developer.md checklist |

## Conventions

- **Reuse before you write.** Prefer existing nvpro_core2 helpers (`nvvk`,
  `nvapp`, `nvutils`, `nvgui`) and the project's own utilities over new code.
  Check `docs/developer.md` and the sibling `nvpro_core2/` repo first. Keep it
  DRY — this codebase is meant to grow cleanly, not accrete duplicates.
- Match the naming and file-pairing already in `src/` (`foo.cpp` / `foo.hpp`,
  `*_vk` for Vulkan-resource owners, `ui_*` for panels, `gltf_scene_*` for
  scene subsystems).
- Slang shaders, not GLSL. Shader hot-reload is **Ctrl+Shift+R** at runtime.

## Build / run / test

```bash
# Sibling clone recommended: nvpro_core2 next to vk_gltf_renderer
# (CMake auto-downloads nvpro_core2 into build/_deps if not found locally)
# Windows
cmake -B build -S . -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
./_bin/Release/vk_gltf_renderer.exe

# Linux
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./_bin/Release/vk_gltf_renderer
```

Key CMake options: `USE_DLSS`, `USE_OPTIX_DENOISER`, `USE_DRACO`,
`NVPRO2_ENABLE_nvmcp` (all `ON`; the last one adds the `--mcp` endpoint and fetches
cpp-mcp at configure time — see [docs/mcp.md](docs/mcp.md)),
`BUILD_TESTING` (`OFF` — turn on to build unit tests + benchmarks; `ctest` runs the
unit tests, run the benchmark executable directly). See `tests/README.md` and
`docs/benchmarking.md`.
