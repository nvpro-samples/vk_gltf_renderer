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
  `Resources::dlssInstanceMotionActive` (bound only on frames where transforms actually change).

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
  discontinuities (resize, preset/quality change, re-enable) via `Dlss::notifyReset` /
  `m_forceResetUntilFrame`. Do not reset on dolly/pan — it discards accumulation and adds noise
  without fixing MV correctness.

## Debugging motion vectors

In the DLSS panel, expand **Guide Images** and select the **Motion** thumbnail
(`Dlss::buildGuideEntries`, `src/dlss.cpp`) to visualize the MV buffer while moving the camera.
