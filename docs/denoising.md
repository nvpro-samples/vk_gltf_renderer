# Denoising & Upscaling — DLSS, OptiX, and Motion Vectors

> **For contributors and agents.** Conceptual map of the AI denoise/upscale backends and the
> guide-buffer + motion-vector pipeline they consume. Exact formats, flags, and struct fields live
> in the cited source — grep the symbols, don't trust values copied here.

## Backends

- **DLSS** — Ray Reconstruction (RR) and Super Resolution (SR): `src/dlss.{cpp,hpp}` (app adapter,
  jitter, guide buffers, UI) over `src/dlss_wrapper.{cpp,hpp}` (project-independent NGX/Vulkan wrapper).
- **OptiX AI Denoiser** — `src/optix_denoiser.{cpp,hpp}` (+ CUDA interop in `src/vk_cuda*`).
- The **path tracer** (`renderer_pathtracer`) is the reference producer of the guide buffers; the
  **rasterizer** produces a subset for DLSS-SR.

## DLSS startup

`Dlss::init` starts a nonblocking NGX prewarm as soon as the renderer is attached. This does not
require a loaded scene: the worker only initializes NGX and probes feature availability. The actual
DLSS feature handle and sized render targets are still created later from `Dlss::updateSize`, where
a command buffer and display extent are available. Disabled DLSS-RR therefore reports **Ready** once
NGX is warmed, but its heavier guide buffers stay unsized until the user enables it.

## Guide buffers

The path tracer writes all guide buffers in `processPixel` (`shaders/gltf_pathtrace.slang`, first-hit
captures around the `USE_DLSS_SHADER` / `USE_GUIDE_SHADER` blocks). They are declared as the
`OutputImage` enum (`shaders/shaderio.h`), bound to NGX inputs in `Dlss::setResourcesRr`
(`src/dlss.cpp`), and consumed in `DlssFeature::cmdEvaluate` (`src/dlss_wrapper.cpp`). Roles (grep
the enum for the authoritative list and formats):

| Guide (`OutputImage`) | Role | NGX input |
|---|---|---|
| `eResultImage` | Noisy 1-spp path-traced radiance (the image being denoised/upscaled) | `pInColor` |
| `eDlssAlbedo` | Diffuse albedo (base color; clear-glass refinement applied) | `pInDiffuseAlbedo` |
| `eDlssSpecAlbedo` | Specular albedo (`EnvBRDFApprox2` in `shaders/dlss_util.h`) | `pInSpecularAlbedo` |
| `eDlssNormalRoughness` | World-space normal + roughness, packed | `pInNormals` (and `pInRoughness` in packed mode) |
| `eDlssMotion` | Pixel-space motion vectors (see below) | `pInMotionVectors` |
| `eDlssDepth` | NDC depth (`z/w`, `1.0` on miss) | `pInDepth` |
| `eDlssSpecularHitDist` | Specular hit distance, captured at the first reflection bounce | `pInSpecularHitDistance` (optional) |

- **DLSS-SR** uses only color + depth + motion — the material guides above are RR-only.
- **OptiX** uses a single combined guide, `eOptixAlbedoNormal` (albedo + camera-space encoded
  normal, written on the OptiX branch of `processPixel`). It deliberately **aliases enum index 2**
  with `eDlssAlbedo`: DLSS and OptiX are mutually exclusive, so they share the slot.
- Guides are produced at DLSS **render** resolution every frame while the denoiser is active; the
  main-GBuffer depth/selection are only written on frame 0 / after a reset.

## Motion vectors — what is and isn't captured

Motion vectors are pixel-space (`InMVScale = 1`, low-res) and computed by `calculateMotionVector`
(`shaders/dlss_util.h`): `MV = (prevNDC - currNDC) * 0.5 * resolution`. Two sources of motion are
combined:

- **Camera motion** — reprojection through `SceneFrameInfo::prevMVP` vs `viewProjMatrix`
  (`shaders/shaderio.h`; both are unjittered and updated per frame in `GltfRenderer::onRender`,
  `src/renderer.cpp`). `prevMVP` holds the previous rendered frame's view-projection and is *not*
  cleared by `resetFrame()`, so a continuous camera move keeps correct camera MVs.
- **Instance (node) motion** — previous render-node transforms snapshotted by
  `snapshot_prev_transforms.comp.slang` (host: `TransformComputeVk::cmdSnapshotPrevObjectToWorld`)
  and applied via `prevRenderNodeObjectToWorld` in `shaders/gltf_pathtrace.slang`. Gated by
  `Resources::dlssInstanceMotionActive` (bound only on frames where transforms actually change) -
  this gate must see `Scene::DirtyFlags::nodes` as populated by *this* frame's transform writes,
  which is why the gate check in `GltfRenderer::onRender()` runs after `updateInteractivityGraphs()`
  (see [docs/interactivity.md](interactivity.md) for a real bug this ordering fixed - a graph-driven
  per-tick move ghosted for its entire duration when the check ran before the tick instead of after).

**Not captured: per-vertex skin/morph deformation.** Skinning/morph overwrite the position buffer
in place each frame (`SceneVk::uploadPrimitives`, `src/gltf_scene_vk.cpp`) with no previous-frame
vertex copy, so a deforming surface only ever receives camera + instance-transform motion. The
result is ghosting/smearing on animated (skinned/morphed) meshes under motion — most visible during
a camera dolly, where the whole surface also moves in screen space. The rasterizer shares this
limitation (see the note in `shaders/gltf_raster.slang`). Closing the gap requires a previous-frame
position buffer per deforming primitive, reprojected the same way as instance motion.

## Sky / background

Primary misses use a point-at-infinity model (`w = 0`) so camera translation cancels and only
rotation moves the environment. See the `calculateMotionVector` overload comment in
`shaders/dlss_util.h`.

## Jitter & reset

- Halton jitter is applied as a ray subpixel offset (path tracer) or clip-space shift (raster),
  never baked into `viewProjMatrix` / `prevMVP`; NGX de-jitters via `InJitterOffset` (negated in
  `src/dlss_wrapper.cpp`).
- Temporal history is **intentionally not reset on camera motion.** `InReset` is only for
  discontinuities (resize, preset/quality change, re-enable, and — since the fix documented in
  [docs/interactivity.md](interactivity.md) — a material/light property write via
  `BaseRenderer::notifyDlssContentReset()`, called from `GltfRenderer::updateSceneChanges()`
  whenever `Scene::DirtyFlags::materials`/`lights` is non-empty) via `Dlss::notifyReset` /
  `m_forceResetUntilFrame`. Do not reset on dolly/pan — it discards accumulation and adds noise
  without fixing MV correctness. Node-transform-only changes are deliberately excluded from this
  reset — those are already correctly handled by instance motion vectors (see "Instance (node)
  motion" above); only *appearance* discontinuities (texture/material content changing at a
  stationary surface, which motion vectors can't describe) need a full history discard.

## Debugging motion vectors

In the DLSS panel, expand **Developer Guide Buffers** and select the **Motion** thumbnail
(`Dlss::buildGuideEntries`, `src/dlss.cpp`) to visualize the MV buffer while moving the camera.

---

## DLSS-NR (Neural Rendering)

DLSS-NR (`NVSDK_NGX_Feature_DLSSNR`) is a **display-resolution image enhancer** that runs
**after** DLSS-RR or DLSS-SR and **after tonemapping**. It is orthogonal to RR/SR: any instance
(path-tracer RR or rasterizer SR) can optionally run NR on top. Gated by the `USE_DLSSNR` CMake
option and the `#if defined(USE_DLSSNR)` guards in `src/dlss.{cpp,hpp}`.

### Why post-tonemap?

NR expects **LDR input** (0–1 range). Running it on the pre-tonemap HDR image
(`eImgRendered`, linear float) causes visible corruption on bright light sources — the
values blow out the internal normalisation. The correct pipeline is:

```text
path tracer / rasterizer
  → DLSS-RR or DLSS-SR evaluate    (→ eImgRendered)
  → tonemap()                       (→ eImgTonemapped, R8G8B8A8_UNORM)
  → Dlss::evaluateNr()              (reads & writes eImgTonemapped)
```

`evaluateNr()` is called from `GltfRenderer::onRender` (`src/renderer.cpp`) immediately after
`tonemap()`. Call sites: the path tracer and rasterizer both call `setNrImage(eImgTonemapped,…)`
at resize time (`renderer_pathtracer.cpp`, `renderer_rasterizer.cpp`).

### NR GBuffer

NR needs a **SFLOAT scratch** image to avoid format conversion artefacts inside the NGX temporal
buffers. `m_nrGBuffer` (R32G32B32A32_SFLOAT, display resolution) is allocated in
`Dlss::updateSizeNr`. After NR evaluates, `evaluateNr()` blits the SFLOAT scratch back to
`eImgTonemapped` (R8G8B8A8_UNORM) using `vkCmdBlitImage2KHR`, which handles the format
conversion automatically.

### Temporal stability — depth and motion vectors

Without depth and motion, NR re-computes local structure detection independently each frame,
producing a characteristic **flicker** between two local-structure states. `evaluateNr()` provides
`pInDepth` and `pInMVec` so NGX can stabilise across frames.

| Kind | Depth source | Motion source |
|---|---|---|
| RR (path tracer) | `eDlssDepth` colour attachment, R16_SFLOAT linearised ViewZ | `eDlssMotion` colour attachment, R16G16_SFLOAT |
| SR (rasterizer) | hardware depth attachment (reversed-Z), `VK_IMAGE_ASPECT_DEPTH_BIT` | `getSrImage(SrSlot::eMotion)` |

MV are in render-pixel space; NR operates at display resolution. `InMVecScaleX/Y` is set to
`displayWidth / renderWidth` (and `…Y`). `InDepthInverted = 0` for RR (linearised), `1` for SR
(reversed-Z).

### NR control parameters

`NrSettings` (nested in `Dlss`, `src/dlss.hpp`) exposes the per-session global knobs:

| Field | NGX param | Range |
|---|---|---|
| `intensity` | `InIntensity` | [0, 1] |
| `localToneStrength` | `InLocalToneStrength` | [0, 1] |
| `localStructureStrength` | `InLocalStructureStrength` | [0, 1] |
| `globalToneStrength` | `InGlobalToneStrength` | [0, 1] |
| `skinStructureStrength` | `InSkinStructureStrength` | [0, 1] |
| `style` | `InStyle` | SDK enum |
| `useAutoMask` | `InUseAutoMask` | bool |

These are edited in the DLSS panel's NR settings pane and apply globally to the entire image.

### Per-material NR control mask — `EXT_DLSS_NR`

`pInControlMask` is a 4-channel image passed to NGX that **multiplies** the global strength
values per pixel:

| Channel | Meaning | Default |
|---|---|---|
| R | intensity mask | 1.0 |
| G | local tone strength mask | 1.0 |
| B | local structure strength mask | 1.0 |
| A | global tone strength mask | 1.0 |

`(1, 1, 1, 1)` is a no-op (global values pass through unchanged). Setting a channel to 0 disables
that effect for the covered pixels. This enables per-character / per-object NR tuning — e.g.
suppress local structure on a background material while keeping it on a foreground character.

This is exposed as the `EXT_DLSS_NR` glTF material extension. The full pipeline:

1. **Material data** — `EXT_DLSS_NR.nrMask` (vec4, default (1,1,1,1)) stored via
   `getExtDlssNr` / `setExtDlssNr` (`src/tinygltf_utils.{hpp,cpp}`). Loaded into
   `GltfShadeMaterial.nrMask` by `gltf_material_cache.cpp` under `#if MAT_EXT_DLSS_NR`.
   Gated by `MAT_EXT_DLSS_NR` in `shaders/gltf_material_config.h` (defaults to 1).

2. **GBuffer slot** — `OutputImage::eNrMask` (index 8, R16G16B16A16_SFLOAT) is the 9th colour
   attachment in the RR inner GBuffer. The path tracer writes it in `processPixel`
   (`shaders/gltf_pathtrace.slang`) on the first-hit block under `USE_DLSS_SHADER &&
   MAT_EXT_DLSS_NR`, reading directly from `materials[materialIndex].nrMask`. The slot is
   registered in `kRrSlots[]` in `renderer_pathtracer.cpp` so it is wired into the `outImages`
   bindless array. The format is declared in `kRrInnerFormats[]` (`src/dlss.cpp`).

3. **Temporal accumulation** — the mask is captured in `GuideScratch.nrMask` (first-hit only,
   no per-bounce accumulation) and copied to `GuideOutput.nrMask` by `packSampleResult`
   (`shaders/pathtrace_functions.h.slang`). Both structs are guarded by `USE_DLSS_SHADER`.

4. **NGX wiring** — `Dlss::evaluateNr()` reads `eNrMask` directly from `m_innerGBuffer` (Kind::RR
   only) and passes it as `pInControlMask` with `InControlMaskSubrectSize` at render resolution.
   When a mask is present, `InUseAutoMask = 0`; when absent (Kind::SR or path tracer not active),
   it falls back to the `useAutoMask` setting.

5. **UI** — `UiInspector::materialDlssNr()` (`src/ui_inspector.cpp`) renders four `SliderFloat`
   controls (Intensity / Local Tone / Local Structure / Global Tone) via
   `renderMaterialExtensionSection()`, which provides the Add / Remove scaffolding automatically.
   The extension appears in the **Material Extensions** section of the inspector.

6. **glTF support** — `"EXT_DLSS_NR"` is declared in `m_supportedExtensions`
   (`src/gltf_scene.cpp`) so the loader preserves the extension data on round-trip save/load.

> **Rasterizer note:** The rasterizer does not write `eNrMask` (no 4th framebuffer attachment in
> the SR pass). For Kind::SR, `pInControlMask = nullptr` and `InUseAutoMask` follows the global
> `useAutoMask` setting. Per-material masking from `EXT_DLSS_NR` is a path-tracer-only feature.

### Adding a new NR parameter

If a future NGX SDK exposes a new per-pixel input (e.g. skin mask, emissive mask):

1. Add a field to `GltfShadeMaterial` (gated by a new `MAT_EXT_*` flag in
   `gltf_material_config.h`).
2. Add a `GBuffer` slot to `OutputImage` (`shaders/shaderio.h`) and the matching format to
   `kRrInnerFormats[]` (`src/dlss.cpp`).
3. Write the slot in the path tracer first-hit block (`shaders/gltf_pathtrace.slang`), propagate
   through `GuideScratch` → `GuideOutput` → `packSampleResult` if needed.
4. Register the slot in `kRrSlots[]` (`src/renderer_pathtracer.cpp`).
5. Wire the NGX input in `Dlss::evaluateNr()` (`src/dlss.cpp`), mirroring the `pInControlMask`
   pattern.
6. Add tinygltf helpers, material cache loading, glTF extension registration, and inspector UI
   following the `EXT_DLSS_NR` pattern.
