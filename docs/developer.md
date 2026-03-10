# Developer Guide -- Vulkan Ray Tracing glTF Renderer

Architecture and source code reference for contributors to the Vulkan glTF Renderer. For runtime usage, see the [User Guide](user-guide.md).

This project is developer-oriented: the **RTX path tracer** is the primary high-quality reference for ray tracing glTF PBR materials, while the **rasterizer** is a practical fallback for fast interaction and iteration. Both renderers share Vulkan resources and Slang shaders.

## Quick Navigation

- [Architecture](#architecture)
- [Scene Graph](#scene-graph)
- [Source Code Structure](#source-code-structure)
- [Common Development Workflows](#common-development-workflows)
- [Testing](#testing)
- [Contributing](#contributing)

For the internal data-flow walkthrough (Model -> RenderNodes -> GPU/TLAS), including how ray tracing acceleration structures are built, see [Rendering Architecture](RENDERING_ARCHITECTURE.md).

---

## Architecture

### Application Framework

`nvapp::Application` provides the Vulkan application lifecycle. You attach `nvapp::IAppElement` instances, and each is called for different states:

| Callback | Purpose |
|---|---|
| `onAttach()` | Create Vulkan resources, compile shaders, set up descriptor sets |
| `onDetach()` | Clean up all GPU resources |
| `onRender(cmd)` | Record rendering commands (dispatch to path tracer or rasterizer, then tone map) |
| `onResize()` | Recreate G-Buffer and notify renderers |
| `onUIRender()` | Render ImGui UI and handle input |
| `onUIMenu()` | Create menu bar (File, View, Tools, Debug) |
| `onFileDrop()` | Load dropped .gltf/.glb/.obj as scenes, .hdr as environment maps |

In `main()` we attach:
- `GltfRenderer` — the main element (scene loading, rendering, UI)
- `ElementProfiler` — GPU execution timing
- `ElementLogger` — log output in a UI window
- `ElementGpuMonitor` — NVML-based GPU status monitoring

---

## Scene Graph

The glTF scene is loaded via TinyGLTF and converted to a GPU-friendly representation shared by both renderers.

````mermaid
---
config:
  layout: elk
  theme: neutral
---
flowchart TB
 subgraph RenderNodes["RenderNodes"]
        rn0["rn0"]
        rn1["rn1"]
        rn2["rn2"]
        rn3["rn3"]
        rn4["rn4"]
  end
 subgraph s1["RenderPrimitives"]
        n1["rp0"]
        n2["rp1"]
  end
 subgraph s2["RenderMaterials"]
        n3["m1"]
        n4["m2"]
        n5["m3"]
  end
    sceneRoot["sceneRoot"] --> node0["node0"]
    node0 --> node1["node1"] & node2["node2"] & node3["node3"]
    node3 --> mesh1["mesh1"]
    mesh0["mesh0"] --> prim0["prim0"] & prim1["prim1"]
    mesh1 --> prim2["prim1'"]
    node1 --> mesh0
    node2 --> mesh0
    prim0 --> rn0 & rn2 & n1 & n3
    prim1 --> rn1 & rn3 & n2 & n4
    prim2 --> rn4 & n2 & n5

    style node0 fill:#BBDEFB
    style node1 fill:#BBDEFB
    style node2 fill:#BBDEFB
    style node3 fill:#BBDEFB
    style prim0 fill:#C8E6C9
    style prim1 fill:#C8E6C9
    style prim2 fill:#C8E6C9
    style RenderNodes fill:#BBDEFB
    style s1 fill:#C8E6C9
    style s2 fill:#E1BEE7
````

**RenderNodes** are the flattened instances with world matrices and material references. **RenderPrimitives** are the unique geometry (vertex/index buffers). Shared primitives create one BLAS for multiple render nodes.

> RN0 == node0 -> node1 -> prim0
> RN1 == node0 -> node1 -> prim1
> RN2 == node0 -> node2 -> prim0
> RN3 == node0 -> node2 -> prim1
> RN4 == node0 -> node3 -> prim1'

---

## Source Code Structure

```text
src/
├── main.cpp                    # Entry point, Vulkan context, CLI parsing
├── renderer.cpp/hpp            # Main renderer orchestrator (GltfRenderer)
├── renderer_base.hpp           # Abstract base for rendering backends
├── renderer_pathtracer.cpp/hpp # Monte Carlo path tracer (Vulkan ray tracing + ray query)
├── renderer_rasterizer.cpp/hpp # Forward PBR rasterizer
├── renderer_silhouette.cpp/hpp # Selection highlight (compute shader)
├── resources.hpp               # Shared Vulkan resources and settings
│
├── gltf_scene.cpp/hpp          # Core scene loading and management
├── gltf_scene_vk.cpp/hpp       # GPU buffer/texture upload (SceneVk)
├── gltf_scene_rtx.cpp/hpp      # BLAS/TLAS acceleration structures (SceneRtx)
├── gltf_scene_editor.cpp/hpp   # Scene editing (add/delete/duplicate nodes)
├── gltf_scene_animation.cpp/hpp# Animation playback and interpolation
├── gltf_scene_merger.cpp/hpp   # Merge two glTF scenes
├── gltf_scene_validator.cpp/hpp# Scene validation
├── gltf_compact_*.cpp/hpp      # Orphan resource removal / buffer compaction
├── gltf_create_tangent.cpp/hpp # Tangent space generation (UV gradient + MikkTSpace)
├── gltf_material_cache.cpp/hpp # CPU-side material cache
├── gltf_image_loader.cpp/hpp   # Image decoding (DDS, KTX, STB, WebP)
├── gltf_animation_pointer.*    # KHR_animation_pointer support
│
├── ui_inspector.cpp/hpp        # Property inspector (materials, transforms, etc.)
├── ui_scene_browser.cpp/hpp    # Scene graph browser (tree + flat views)
├── ui_renderer.cpp             # Viewport UI and mouse interaction
├── ui_xmp.cpp/hpp              # KHR_xmp_json_ld metadata display
├── ui_animation_control.hpp    # Animation playback UI widget
├── ui_mouse_state.hpp          # Mouse state tracking for viewport
├── ui_busy_window.hpp          # Loading/busy indicator overlay
│
├── dlss_denoiser.cpp/hpp       # DLSS Ray Reconstruction integration
├── dlss_wrapper.cpp/hpp        # NGX / DLSS SDK wrapper
├── optix_denoiser.cpp/hpp      # OptiX AI Denoiser integration
├── vk_cuda.cpp/hpp             # Vulkan-CUDA interop (shared memory/semaphores)
│
├── gizmo_transform_vk.cpp/hpp  # 3D translate/rotate/scale gizmo
├── gizmo_grid_vk.cpp/hpp       # Infinite procedural grid
├── gizmo_visuals_vk.cpp/hpp    # Helper overlay manager
│
├── undo_redo.cpp/hpp           # Full undo/redo system for scene editing
├── tinygltf_utils.cpp/hpp      # TinyGLTF material extension extraction
├── tinygltf_converter.cpp/hpp  # OBJ to glTF converter
├── tiny_stb_implementation.cpp # STB library implementation (single compilation unit)
├── pipeline_cache_util.cpp/hpp # Vulkan pipeline cache persistence
├── scene_selection.cpp/hpp     # Selection state and events
├── gltf_camera_utils.hpp       # Camera utility functions
├── gpu_memory_tracker.hpp      # GPU memory category tracking
├── scoped_banner.hpp           # Scoped UI banner helper
└── utils.hpp                   # Miscellaneous utilities

shaders/
├── gltf_pathtrace.slang        # Path tracer (compute + ray generation)
├── gltf_raster.slang           # PBR rasterizer (vertex + fragment)
├── silhouette.comp.slang       # Selection edge detection
├── shaderio.h                  # Host/device shared structures
├── common.h.slang              # Shared shader utilities
├── get_hit.h.slang             # Hit state interpolation
├── raytracer_interface.h.slang # RayQuery vs traditional RT abstraction
├── dlss_util.h                 # DLSS motion vector / guide buffer utilities
├── gizmo_grid.slang            # Infinite grid rendering
├── gizmo_grid_shaderio.h.slang # Grid shader I/O
├── gizmo_visuals.slang         # Transform gizmo overlays
├── gizmo_visuals_shaderio.h.slang # Gizmo shader I/O
└── optix_image_to_buffer.slang # OptiX denoiser buffer conversion
```

---

## Common Development Workflows

### Add a command-line parameter

1. Register it in `main.cpp` through `nvutils::ParameterRegistry`.
2. Wire it to renderer or app settings.
3. Document it in [user-guide.md](user-guide.md#command-line-reference).

### Add a renderer setting

1. Add state in shared settings/resources.
2. Bind and consume it in `renderer_pathtracer.cpp` or `renderer_rasterizer.cpp`.
3. Expose UI controls in the corresponding UI module.
4. Persist with the settings handler when needed.

### Add or update glTF extension support

1. Parse and serialize extension fields in `tinygltf_utils.*`.
2. Integrate runtime behavior in scene/material/shader code as needed.
3. Update extension documentation in `README.md` and usage notes in `user-guide.md`.

### Modify sync/data-flow behavior

Use [RENDERING_ARCHITECTURE.md](RENDERING_ARCHITECTURE.md) as the source of truth for:
- parse/rebuild flow
- render-node dirty propagation
- GPU buffer synchronization
- TLAS/BLAS update invariants

---

## Testing

Tests are disabled by default. Enable with `BUILD_TESTING=ON`:

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build --target vk_gltf_renderer_tests
ctest --test-dir build -C Release --output-on-failure
```

See [tests/README.md](../tests/README.md) for details.

---

## Contributing

This project uses the [Developer Certificate of Origin](https://developercertificate.org/) (DCO). By contributing, you certify that you have the right to submit the work under the project's open source license.

Please open issues for bug reports and feature requests. Pull requests should target the `main` branch and include a clear description of the change.
