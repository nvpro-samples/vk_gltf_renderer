# Vulkan glTF Renderer — RTX Ray Tracing & PBR Path Tracer

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-green.svg)](#requirements)
[![C++](https://img.shields.io/badge/C%2B%2B-20-orange.svg)](#build-and-run)
[![Vulkan](https://img.shields.io/badge/Vulkan-1.4%2B-red.svg)](#requirements)

> Open-source **Vulkan ray tracing** renderer for **glTF 2.0** scenes — Reference for glTF PBR materials, RTX path tracing, physically based rendering, AI denoising, and a built-in scene editor.

| glTF Renderer |
|---|
| ![](docs/gltf_renderer.jpg) |

A production-quality **Vulkan ray tracing** renderer for **glTF 2.0** scenes. Includes a high-fidelity **RTX path tracer** with full **PBR material** support, **DLSS Ray Reconstruction** and **OptiX AI Denoiser**, a rasterizer preview path, and a complete scene editor — built for graphics developers who want a reference they can study, profile, and extend.

Built in C++ on [nvpro_core2](https://github.com/nvpro-samples/nvpro_core2) with [Slang](https://github.com/shader-slang/slang) shaders. Successor to [vk_raytrace](https://github.com/nvpro-samples/vk_raytrace).

## Highlights

- **Ray tracing reference** — Monte Carlo path tracer with NEE, MIS, and adaptive sampling for physically accurate glTF PBR results.
- **AI denoising** — DLSS Ray Reconstruction and OptiX AI Denoiser produce clean images at interactive rates.
- **Rasterizer preview** — Fast PBR rasterizer shares scene resources for instant iteration.
- **Scene editor** — Hierarchy manipulation, transform gizmo, material editing, undo/redo, save back to glTF.
- **31 glTF extensions** — Anisotropy, clearcoat, transmission, volume, sheen, iridescence, dispersion, retroreflection, Draco, interactivity, and more.
- **glTF 2.1 complex scenes (preview)** — Compose multi-file scenes with External Assets: reference glTF/GLB files as instances that share geometry, with nested references, cycle detection, and file aliases — re-externalized on save.
- **Developer tools** — GPU profiler, memory tracker, shader hot-reload (Ctrl+Shift+R), headless batch mode.

## Quick Tour

![](docs/hero_demo.gif)

The demo shows a short end-to-end workflow: switching renderer modes, tuning settings, and inspecting scene content in the integrated editor.

## Build and Run

### Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10 / Linux | Windows 11 / Ubuntu 22.04+ |
| **GPU** | NVIDIA RTX 20-series (Turing) | NVIDIA RTX 40-series (Ada) |
| **Driver** | 535+ | Latest Game Ready / Studio |
| **CMake** | 3.22 | 3.28+ |
| **C++ Compiler** | C++20 (MSVC 2022 / GCC 12 / Clang 15) | MSVC 2022 17.8+ |
| **Vulkan SDK** | 1.4 | [Latest](https://vulkan.lunarg.com/sdk/home) |

### Quick start

```bash
# Clone (recommended: siblings; CMake auto-downloads nvpro_core2 if missing)
git clone https://github.com/nvpro-samples/nvpro_core2.git
git clone https://github.com/nvpro-samples/vk_gltf_renderer.git
cd vk_gltf_renderer
```

```bash
# Windows
cmake -B build -S . -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
.\_bin\Release\vk_gltf_renderer.exe
```

```bash
# Linux
cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./_bin/Release/vk_gltf_renderer
```

### Common CMake options

| Option | Default | Description |
|---|---|---|
| `USE_DLSS` | `ON` | Enable DLSS Ray Reconstruction integration |
| `USE_OPTIX_DENOISER` | `ON` | Enable OptiX AI Denoiser (requires CUDA Toolkit) |
| `USE_DRACO` | `ON` | Enable Draco mesh compression support |
| `BUILD_TESTING` | `OFF` | Build unit tests and benchmarks |

## Features

- **Ray tracing**: High-quality path tracing reference for glTF PBR materials — Monte Carlo global illumination, next event estimation, multiple importance sampling, and adaptive sampling.
- Profiler, GPU monitor, GPU memory tracking, and statistics.
- Shader hot-reload (Ctrl+Shift+R) for developers who want to experiment.
- AI denoisers: DLSS Ray Reconstruction and OptiX AI Denoiser.
- Rasterizer fallback for fast scene interaction and editing.
- A scene asset editor with hierarchy operations, a transform gizmo, material editing, merging, and saving back to glTF (non-destructive).
- glTF 2.1 complex-scene composition (preview): reference external glTF/GLB assets, instance them (shared geometry), resolve nested references with cycle detection, and re-externalize on save.
- Support for 31 glTF extensions, including anisotropy, clearcoat, transmission, volume, sheen, iridescence, dispersion, diffuse transmission, retroreflection, material variant and scattering.
- HDR environments, a physical sun and sky model, depth of field, and multiple tone mappers.
- Animation support includes skeletal, morph targets, and KHR_animation_pointer.
- GPU compute accelerates both skinning/morphing and per-level world-matrix propagation.

**More features** --> [User Guide](docs/user-guide.md)
 
## Showcase

| Feature | Preview |
|---|---|
| Showcase | ![](docs/ABeautifulGame.jpg) ![](docs/ToyCar.jpg) ![](docs/DamagedHelmet.jpg) ![](docs/Sponza.jpg) |
| Material features | ![](docs/SunglassesKhronos.jpg) ![](docs/SheenCloth.jpg) ![](docs/TransmissionTest.jpg) ![](docs/volume.png) ![](docs/volume_scatter.png) ![](docs/IridescenceAbalone.jpg) |
| Lighting and camera | ![](docs/sky_1.jpg) ![](docs/hdr_1.jpg) ![](docs/dof_1.jpg) ![](docs/light.jpg) |

For a **full walkthrough** of rendering modes, editor workflows, and feature screenshots, see the [User Guide](docs/user-guide.md).

For **headless timing** and optional scripted GPU benchmarks, see [Benchmarking](docs/benchmarking.md) (`utils/benchmark/`).

## glTF Support

**Reference scope:** The list below reflects what this renderer loads and displays. The **path tracer** is the authoritative PBR implementation — especially for ray-traced material evaluation, sampling, and new extensions (e.g. [KHR_materials_retroreflection](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_materials_retroreflection/README.md)). The **rasterizer** is a preview path for interaction, not the primary material reference.

### Core

- ✅ glTF 2.0 (.gltf/.glb)
- ✅ Images (HDR, PNG, JPEG, KTX, KTX2, DDS, WebP)
- ✅ Buffers (geometry, animation, skinning)
- ✅ Textures and samplers
- ✅ Materials (PBR metallic-roughness and specular-glossiness)
- ✅ Animations (keyframe, skeletal)
- ✅ Skins
- ✅ Morph targets
- ✅ Cameras (perspective and orthographic)
- ✅ Punctual lights (directional, point, spot)
- ✅ Nodes and scene hierarchy
- ✅ Multiple scenes

### Extensions

- ✅ KHR_animation_pointer
- ✅ KHR_draco_mesh_compression
- 🚧 KHR_interactivity — behavior graph is parsed and preserved on save; not yet executed
- ✅ KHR_lights_punctual
- ✅ KHR_materials_anisotropy
- ✅ KHR_materials_clearcoat
- ✅ KHR_materials_diffuse_transmission
- ✅ KHR_materials_dispersion
- ✅ KHR_materials_emissive_strength
- ✅ KHR_materials_ior
- ✅ KHR_materials_iridescence
- ✅ KHR_materials_pbrSpecularGlossiness
- ✅ KHR_materials_retroreflection
- ✅ KHR_materials_sheen
- ✅ KHR_materials_specular
- ✅ KHR_materials_transmission
- ✅ KHR_materials_unlit
- ✅ KHR_materials_variants
- ✅ KHR_materials_volume
- ✅ KHR_materials_volume_scatter
- ✅ KHR_mesh_quantization
- ✅ KHR_meshopt_compression
- ✅ KHR_node_hoverability
- ✅ KHR_node_selectability
- ✅ KHR_node_visibility
- ✅ KHR_texture_basisu
- ✅ KHR_texture_transform
- ✅ KHR_xmp_json_ld
- ✅ EXT_mesh_gpu_instancing
- ✅ EXT_texture_webp
- ✅ MSFT_texture_dds

### glTF 2.1 (Complex Scenes — preview)

Early support for the [glTF 2.1 complex-scene](https://www.khronos.org/blog/introducing-gltf-2.1-with-complex-scenes) composition features — the standardized successor to the earlier glTFX / External Reference proposals:

- ✅ External Assets — reference other glTF/GLB files from scene nodes (`node.externalAsset`) and instantiate them at load time
- ✅ Multiple instances of the same asset share geometry (meshes / BLAS)
- ✅ Nested external assets — recursive resolution with cycle detection
- ✅ Unified file references — top-level `files` array (external `uri`)
- ✅ File aliases — inner-URI redirection for shared or overridden resources
- ✅ Re-externalized on save (references preserved); "Make Editable" embeds an asset inline
- 🚧 Packaging — embedded external assets (`bufferView` / `data:` URI) not yet resolved
- 🚧 Shapes and scene-level bounding volumes

See [External Assets](docs/external_assets.md) for design details.

## Documentation

**For users**

- [User Guide](docs/user-guide.md) — renderer settings, PBR materials, scene editor, camera, environment, tone mapping, common CLI flags, and troubleshooting.
- [glTF Resources](docs/resources.md) — curated collection of glTF models, HDR environments, specifications, and tools.

**For contributors**

- [Developer Guide](docs/developer.md) — architecture overview, source map, material system, and testing.
- [Rendering Architecture](docs/RENDERING_ARCHITECTURE.md) — data flow from glTF model to GPU, BLAS/TLAS acceleration structures, and render nodes.
- [External Assets](docs/external_assets.md) — glTF 2.1 complex scenes: reference / merge / edit / save mechanics.
- [Benchmarking](docs/benchmarking.md) — headless timing and scripted GPU benchmarks.

These docs explain concepts and workflows and point at *where* things live; enumerated facts (extension lists, CLI flags, enum values) are owned by the code.

## License

[Apache License 2.0](LICENSE) - Copyright (c) 2023-2026, NVIDIA CORPORATION.
