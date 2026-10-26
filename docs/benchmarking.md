# Benchmarking vk_gltf_renderer

> **For contributors and agents.** How to measure performance — GPU timing, headless timing and the scripted sequencer. The exact flags, log fields, and comparison rules are owned by the code (`src/benchmarking.cpp`, `utils/benchmark/`); this doc explains the workflow.

Two performance workflows, and they answer different questions:

1. **[GPU time at a fixed sample count](#gpu-time-at-a-fixed-sample-count-recommended-for-renderer-changes)** — the profiler's `Path Trace` GPU timer with the work per frame pinned. Use this to judge a change to the renderer: it is the only one that isolates GPU render cost from shader compilation, denoiser CPU work and the rest of the loop.
2. **[Headless timing](#headless-timing-simple)** — post-warmup wall-clock throughput for N frames; convenient for a quick end-to-end number or comparing whole builds, but it is CPU wall-clock and will not resolve a small renderer delta.

The GPU workflow drives the scripted sequencer described further down; headless timing runs from plain command-line flags. A third, non-performance workflow reuses the same sequencer to **capture and review the UI** — see [UI inspection](#ui-inspection-windowed-panel-capture) below.

## Headless timing (simple)

Render **500 frames** (or any count) with path tracing. The path tracer's progress is counted in
samples per pixel (spp), not frames: `--ptSamples` is the spp traced each frame and `--ptMaxSamples`
the spp at which the accumulation stops.

### One-off CLI

```bash
./vk_gltf_renderer --headless --size 1920 1080 \
  --scenefile shader_ball.gltf \
  --hdrfile std_env.hdr \
  --frames 500 \
  --ptSamples 1 --ptAdaptiveSampling 0 \
  --renderSystem 0 --envSystem 1
```

While rendering, periodic progress lines confirm the run is not stuck (every 50 frames or 5 seconds):

```text
HEADLESS_START frames=500 maxSamples=500 ptSamples=1
HEADLESS_PROGRESS spp 50/500 app_frame 50/500 (10%) elapsed_ms=1234.5 ms_per_frame=24.69
...
HEADLESS_SUMMARY frames=500 maxSamples=500 ptSamples=1 effective_spp=500 measured_effective_spp=499 resolution=1920x1080 wall_ms=12320.987 ms_per_frame=24.691 total_wall_ms=12345.678 total_ms_per_frame=24.691 warmup_frames=1 measured_frames=499 throughput_MSps=84.0 spp_per_sec=40.49
BENCHMARK_JSON {"schema":1,"type":"headless_summary",...}
```

- **spp** — samples per pixel accumulated, out of the cap; **app_frame** — headless loop index
  (`--frames`). The percentage follows whichever ends the run first.
- Without `--ptMaxSamples`, a headless run raises the cap to `--frames` × the samples per frame (the
  Auto SPP ceiling when `--ptAdaptiveSampling` is on), so every app frame accumulates samples during
  timing runs. An explicit `--ptMaxSamples` is kept and **ends the run**: once the image reaches it
  (and is denoised, see [denoising.md](denoising.md)), the app saves the output and exits, so
  `--frames` is only an upper bound. `--frames 0` renders until the cap.
- Animations don't play in headless mode unless `--animPlay 1` is given: an advancing animation resets the path tracer's accumulation every frame, leaving the output at about one sample per pixel. A scene with a `KHR_interactivity` behavior graph never autoplays, whatever `--animPlay` says.
- **wall_ms** — measured post-warmup render time. The first completed frame is excluded so one-time setup such as shader specialization is not charged to throughput.
- **total_wall_ms** — full headless render-loop wall time, including warmup and any synchronous setup.
- **ms_per_frame** — `wall_ms / measured_frames`.
- **effective_spp** — samples per pixel the final image holds (`Resources::accumulatedSamples`). It
  already accounts for the cap, adaptive sampling, and the restarts that can happen during the first
  frames while the scene loads.
- **measured_effective_spp** — samples accumulated during the measured post-warmup window.
- **throughput_MSps** — measured mega pixel-samples per second (`resolution × measured_effective_spp / wall_s / 10⁶`; higher is faster).
- **spp_per_sec** — measured `measured_effective_spp / wall_s` at this resolution (how fast quality accumulates; higher is faster).

Repeat with `--ptSamples 5` for 5 spp per frame (`effective_spp=2500`).

### Fastest render to a sample count (Auto SPP)

To get an image of N samples per pixel as fast as possible, let the path tracer pick the samples per
frame and stop at N:

```bash
./vk_gltf_renderer --headless --scenefile scene.gltf --frames 0 --ptMaxSamples 4096 --output out.png
```

Several samples per frame share the per-frame work (tonemapper, submit, the CPU loop), which makes
most scenes faster than one sample per frame, up to 2.6x on a light scene. Some scenes are the
opposite: a sample costs more in a longer dispatch, and one sample per frame is the fastest. So a
headless run uses Auto SPP's **Max Throughput** target (`--ptPerformanceTarget 4`, the headless
default while `--ptAdaptiveSampling` is on, i.e. unless `--ptSamples` is given): it tries 1, 4, 16,
64, ... samples per frame, each over a quarter second of wall time, up to frames of 100 ms, and keeps
the count with the lowest wall time per sample (`sample_schedule::ThroughputSearch`,
`src/sample_schedule.hpp`). Auto SPP may go above the Samples per Frame slider's range
(`PathTracer::MAX_ADAPTIVE_SAMPLES_PER_PIXEL`), which small or simple images need to fill a frame.

Measured at 1920x1080 on an RTX PRO 4500 Blackwell, time to the given count, search included
(`--frames 0 --ptMaxSamples N`):

| Scene | spp | 1 spp per frame | Quality target (66 ms) | Max Throughput (settled on) |
|---|---|---|---|---|
| shader_ball | 5000 | 12.6 s | 4.7 s | 4.8 s (64) |
| ABeautifulGame | 4000 | 5.8 s | 5.4 s | 5.4 s (64) |
| Sponza | 4000 | 4.8 s | 4.4 s | 4.5 s (64) |
| USDShaderBallForGltf | 2500 | 5.6 s | 6.0 s | 5.7 s (1) |
| complex_scene | 400 | 5.4 s | 6.1 s | 5.4 s (1) |

Each count tried has to be measured, so a short run spends a larger share of its time searching;
for a timing comparison pin the count instead (`--ptSamples N`).

The frame-rate targets (Interactive to Max Quality, the interactive default) do not search
(`sample_schedule::FrameRateControl`): after each reset the count climbs by one per frame until the
path-trace GPU time reaches the target, then holds while the view is still. Comparing the cost per
sample across counts, as the throughput search does, proved unreliable on a live GPU (clock changes,
thermal throttling, sub-millisecond frames).

### Convergence series in one run (`--outputSamples`)

To study convergence (for example a denoiser against a converged reference), save the image at
several sample counts from a single accumulation instead of one run per count:

```bash
./vk_gltf_renderer --headless --size 1000 800 --scenefile scene.gltf   --frames 0 --ptMaxSamples 1024   --oidnEnable 1 --oidnAutoDenoiseSpp 1   --outputSamples 1,2,4,8,16,32,64,128,256,512,1024 --output series/oidn.png
```

Each time the image shown reaches one of the counts, it is saved as `<output>_<spp>spp<ext>`
(`series/oidn_64spp.png`). The count is the samples per pixel of the image shown: a denoised image
counts the samples it was denoised from, so OptiX's one-frame lag is accounted for. Each count is
also a stop for the path tracer: a frame ends exactly on it, whatever the samples per frame (Auto
SPP included), so the raw image, and an image denoised every frame, has exactly that count. A count
the image shown skips over (a denoise interval above 1) is saved at the first image past it, under
its actual count.
When the accumulation restarts while the scene loads, the counts are saved again from the new
accumulation. Set `--ptMaxSamples` to the largest count and `--frames 0`: the run ends once that
count is saved. A count that was not reached (a `--frames` limit hit first) is reported at the end. Implementation: `GltfRenderer::updateSampleCaptures`
(`src/renderer.cpp`).

For error measurements against a converged reference:

- Save as **`.hdr`** to get the linear rendered image (before the tonemapper, float); any other
  extension saves the tonemapped 8-bit image. The same rule applies to `--output` and to the
  interactive File > Save Image (`GltfRenderer::saveViewportImage`). The rendered image is the path
  tracer's accumulation (DLSS-RR's output when DLSS is on), never the OptiX or OIDN result, and
  `--outputSamples` names a `.hdr` capture after its own sample count.
- Render the reference with **`--ptSeed 1`**. The random sequence otherwise depends only on the pixel
  and the frame, so frame *k* of the reference would start with the very same sample as frame *k* of
  the test render, and their noise would partly cancel in the difference.
- Save the reference at two counts, *N* and *N*/2 (`--outputSamples 32768,65536`): the difference
  of the two estimates the reference's own noise, which can then be subtracted from the measured MSE.
- Use **one fixed exposure** (`--tmAutoExposure 0 --tmExposure <value>`) and `--tmDither 0`. Auto
  exposure exposes each image from its own histogram, which noise skews at low sample counts, so
  images would differ in brightness on top of their noise.

### Denoiser cost (`DENOISER_TIMING`)

At the end of a headless run, each AI denoiser that produced an image prints one line. OIDN runs in
the frame's command buffer, so its cost is the GPU time of its last denoise, from its profiler
section. Those timestamps resolve a few frames later, so a run whose only denoise is the final image
(`--oidnAutoDenoiseSpp` above `--ptMaxSamples`) prints no OIDN line. OptiX runs outside
it, so its line breaks down the host wall time of one denoise (averaged), plus the GPU time of the
OptiX invoke from CUDA events:

```text
DENOISER_TIMING oidn gpu_ms=2.766
DENOISER_TIMING optix total_ms=6.714 drain_ms=3.195 copy_in_ms=0.159 invoke_ms=3.103 invoke_gpu_ms=3.015 copy_out_ms=0.256
```

`drain_ms` is the wait for the frames in flight (`vkQueueWaitIdle`) that OptiX needs before CUDA
can read the image: not denoiser work, but a stall the frame pays. Use `--oidnAutoDenoiseSpp 1`
/ `--optixAutoDenoiseSpp 1` so every frame is denoised.

### Batch helper (1 spp and 5 spp)

```bash
python utils/benchmark/benchmark.py headless --scene resources/shader_ball.gltf --frames 500 --spp 1 5
```

Logs: `utils/benchmark/output/headless_<scene>_spp<N>.log`
CSV: `utils/benchmark/output/headless_results.csv`

Compare two builds (same scene/spp, different executable):

```bash
python utils/benchmark/benchmark.py headless-compare \
  utils/benchmark/output/headless_shader_ball_spp1_baseline.log \
  utils/benchmark/output/headless_shader_ball_spp1_candidate.log
```

(Use one log from build A and one from build B.)

---

## GPU time at a fixed sample count (recommended for renderer changes)

`--headless`'s `ms_per_frame` is **CPU wall-clock around the render loop**. It includes shader
compilation on the first frame, denoiser CPU work (the OptiX denoiser can cost tens of milliseconds
of CPU while reporting ~0 GPU), and everything else in the loop. For judging a change to the path
tracer it is the wrong instrument -- it has been observed to disagree with itself by more than 10%
and to reverse the sign of a real 12% difference.

Use the profiler instead: run the sequencer at log level `eSTATS` and read the `Path Trace (RQ)` /
`Path Trace (RTX)` GPU timer, with the sample count pinned so every frame does identical work.

```bash
./vk_gltf_renderer --benchmark 1 --logLevel 1 --sequencefile gpu.cfg \
  --size 1280 720 --scenefile scene.gltf --hdrfile std_env.hdr \
  --optixEnable 0 --optixAutoDenoiseEnabled 0 --oidnEnable 0 --dlssEnable 0
```

The script must do two things. It has to pin the work per frame -- `--ptSamples N`,
`--ptAdaptiveSampling 0` (adaptive sampling varies the samples per frame, so ms/frame stops being
comparable), and `--ptMaxSamples` large enough that accumulation never stops mid-sequence (otherwise
the frame being timed is just the tonemapper). And it has to **open with a throwaway `load`
sequence**, because scene loading and pipeline creation consume the frames of whichever sequence is
running at the time -- see the note below.

```text
SEQUENCE "load"
--sequenceframes 240
--renderSystem 0
--ptSamples 2
--ptAdaptiveSampling 0
--ptMaxSamples 1000000
--ptTechnique 1
--ptOptimalShader 1
--ptUseSER 1
--fitScene

SEQUENCE "warmup"
--sequenceframes 120
--sequenceaverages 32
--updateData

SEQUENCE "measure"
--sequenceframes 400
--sequenceaverages 200
--updateData
```

The knobs that change path-tracing cost the most, and which therefore have to be stated whenever a
timing is quoted. Their magnitudes depend on the content: below, *scatter* means the scatter
sample scenes (volumetric, highly divergent) and *surfaces* means opaque/transmissive interiors and
outdoor terrain (a furnished room with ~100 materials, a forested mountain, FlightHelmet), all at
1920x1080 on an RTX PRO 4500 Blackwell. Neither is universal; measure your own scene:

| Parameter | Effect |
|---|---|
| `--ptTechnique` | Ray query (compute) vs the ray tracing pipeline. Scatter: the pipeline was ~2x faster. Surfaces: ray query was 10-30% faster than the pipeline *with* SER, more without |
| `--ptUseSER` | Shader Execution Reordering, ray tracing pipeline only. Scatter: ~2.3-2.8x. Surfaces: ~20-25%. Silently ignored when the device does not support it |
| `--ptOptimalShader` | On by default: compiles with only the scene's feature gates (pass `0` to time the all-features shader). Scatter: 14-20% with SER on, ~32% with SER off. Surfaces: ~20% on the pipeline, more on ray query, where the compiled-out features mostly free registers |
| `--dlssInputSize` | DLSS-RR's input resolution. The default (Optimal) path-traces below native resolution, so a DLSS-on vs DLSS-off comparison at defaults compares different pixel counts: on one scene the path trace was ~2.1x cheaper with DLSS on. Set `--dlssInputSize 2` (Max, native) to compare at equal resolution; see [denoising.md](denoising.md#dlss-rr-input-resolution) |

Three traps worth knowing, each of which produces confident-looking numbers that mean nothing:

- **A starved first sequence.** Scene load and pipeline creation eat the frames of the sequence
  that happens to be running, so a script whose first sequence is `measure` can burn all of them
  before a single frame is path traced. The symptom is an empty profiler block
  (`ParameterSequence 0 "warmup" = { }`) or a low `samples` count, and a `--screenshot` taken in
  that window is **solid black** because the framebuffer has not accumulated anything yet. Always
  check `samples` in the block you quote: it should be close to the `--sequenceaverages` you asked
  for. This is the same rule the [UI inspection](#ui-inspection-windowed-panel-capture) workflow
  states, and it applies just as much to timing.
- **Persisted settings (no longer a trap).** `_bin/<config>/vk_gltf_renderer.ini` stores
  `ptOptimalShader`, `ptTechnique`, `ptAdaptiveSampling`, the tonemapper (including auto-exposure)
  and the layout between interactive sessions. Headless, scripted (`--sequencefile`) and benchmark
  runs neither read nor write it, so they start from the built-in defaults plus the command line and
  the same command gives the same result on any machine. To reuse an interactive setup on purpose,
  pass it explicitly with `--iniFile path/to/settings.ini`; a path that cannot be read aborts the
  run rather than silently falling back to the defaults. Still state every knob that matters in the
  script: defaults can change between builds.

  (Scripted runs used to write that file back, leaving a measurement run's window size and hidden
  side panels behind as the next interactive layout. They used to read it too, so a previous
  session silently changed what you measured.)
- **Stale shader search paths.** The path tracer compiles Slang at runtime from the directories baked
  in at build time. If those point somewhere stale, it falls back to the SPIR-V embedded at build
  time and a shader-swap A/B silently measures nothing. Verify a swap actually took effect (via a
  change you can see) before trusting it.

---

## UI inspection (windowed panel capture)

The headless workflow captures the **viewport render only** — it never composites the ImGui panels. To review the Scene Browser, Inspector, and other panels (e.g. when iterating on UI/UX), drive the app with a sequencer script **without** `--benchmark`: the side panels stay visible, the sequencer steps the script, and it closes the app when finished.

```bash
./vk_gltf_renderer --sequencefile utils/benchmark/ui_inspect.cfg \
  --scenefile shader_ball.gltf --hdrfile std_env.hdr --size 1600 900
```

Two script parameters support this (registered alongside the other sequencer commands — see `BenchmarkController::registerParameters` in `src/benchmarking.cpp`):

- **`--uiScreenshot <file>`** — capture the full composited window (panels + viewport). This reads the swapchain PRESENT image, so it works only in a windowed run; it is ignored in `--headless` mode (no swapchain). Contrast with `--screenshot`, which saves the viewport render (gbuffer) only.
- **`--selectNode <index>`** — select a scene-graph node by glTF node index (as listed in the Scene Browser tree), driving the Inspector contents and the Scene Browser highlight, exactly as a click would. A negative or out-of-range index clears the selection.

Two ordering rules matter (both illustrated in [`utils/benchmark/ui_inspect.cfg`](../utils/benchmark/ui_inspect.cfg)):

1. Give the `load` sequence enough frames — scene load and ray-query pipeline creation are asynchronous in a non-benchmark run, and a first-time pipeline build (cache miss) can take several seconds.
2. Put a UI action (e.g. `--selectNode`) in one `SEQUENCE` and its `--uiScreenshot` in the **next** one. The capture is scheduled a couple of frames after the parameters apply; a separate sequence guarantees the selection has propagated (selection → Scene Browser → Inspector) before the shot is taken.

---

## Scripted sequencer (optional)

Scripted benchmarks measure GPU frame time and VRAM usage across scenes, cameras, and renderer settings. Output is designed for regression testing when comparing builds or branches.

## Quick start (sequencer)

Build the sample, then from the project directory:

```bash
# Fast smoke benchmark (one scene)
python utils/benchmark/benchmark.py run quick.cfg --scene resources/shader_ball.gltf --hdr std_env.hdr

# Full matrix (edit utils/benchmark/scenes.example.txt for your asset paths)
python utils/benchmark/benchmark.py run matrix.cfg \
  --scenes-file utils/benchmark/scenes.example.txt \
  --scenes-root . \
  --csv-name benchmark_results.csv
```

Results land in `utils/benchmark/output/` (logs per scene + CSV). See also [utils/benchmark/README.md](../utils/benchmark/README.md).

## How it works

1. **`--benchmark 1`** turns off vsync, hides side panels (keeps a fullscreen viewport with the tonemapped image), and drains the scene load pipeline synchronously each frame. Scripted sequencer runs are **interactive-window** runs (minimal viewport) — they do *not* pass `--headless`; use the headless workflow above for windowless timing.
2. **`ElementSequencer`** steps through a `.cfg` script (`SEQUENCE "name"` blocks).
3. After each sequence, **`ProfilerManager`** logs `ParameterSequence` blocks at log level `eSTATS` (GPU/CPU timer averages).
4. **`benchmarkAdvance()`** records Scene and PathTracer/Rasterizer VRAM stats.
5. When the script finishes, the app closes automatically.

## Benchmark script format

Example — one sequence from `utils/benchmark/quick.cfg` (the file defines several: `Warmup`,
`PT 1spp`, `Rasterizer`; see `quick.cfg` and `matrix.cfg` for the full scripts):

```
SEQUENCE "PT 1spp"
--sequenceframes 256
--sequenceaverages 64
--sequenceresetframes 8
--renderSystem 0
--ptSamples 1
--ptMaxSamples 1000000
--ptAdaptiveSampling 0
--gltfCamera 0
--updateData
```

| Token | Meaning |
|---|---|
| `SEQUENCE "..."` | Starts a new measured step |
| `--sequenceframes` | Frames to run this step |
| `--sequenceaverages` | Frames averaged for profiler report |
| `--sequenceresetframes` | Warmup frames after parameter changes (0 = measure immediately) |
| Other `--flags` | Any registered CLI parameter (renderer, path tracer, tonemapper, etc.) |

**Path tracer note:** Pin `--ptSamples` with `--ptAdaptiveSampling 0` and keep `--ptMaxSamples` above what the sequence can accumulate (the scripts use `1000000`): once the accumulation reaches the cap the path tracer stops, and the frames timed after that are only the tonemapper.

## Comparing versions

```bash
# Baseline build
python utils/benchmark/benchmark.py run matrix.cfg --scene my_scene.gltf --csv-name baseline.csv

# Candidate build (rebuild executable first)
python utils/benchmark/benchmark.py run matrix.cfg --scene my_scene.gltf --csv-name candidate.csv

python utils/benchmark/benchmark.py compare baseline.csv candidate.csv --output diff.csv --regression-threshold-pct 5
```

`compare` marks **Regression** when candidate GPU time is more than N% slower than baseline **or** when the candidate's Scene VRAM peak exceeds the baseline by more than the VRAM threshold (default 64 MB). Negative delta % means faster. See `compare_csv` in `utils/benchmark/benchmark_results.py` for the exact rules.

## Log parsing

`utils/benchmark/benchmark.py` reads stable `BENCHMARK_JSON` records first, with legacy text parsing as a fallback:

- `BENCHMARK_JSON {"schema":1,"type":"headless_summary",...}`
- `BENCHMARK_JSON {"schema":1,"type":"sequence_memory",...}`
- `ParameterSequence N "name" = { Timer "..."; GPU; avg ...; CPU; avg ...; }`
- `BENCHMARK_ADV N { Memory Scene; ... Memory PathTracer; ... }`

Auto-generated log: `log_<executable>.txt` next to the binary (Logger behavior).

## Tips

- Use fixed resolution (`--size 1920 1080`, set in `utils/benchmark/benchmark_runner.py`) for comparable numbers.
- Disable validation layers for performance runs (`--vvl` off by default in Release).
- Add scenes to `utils/benchmark/scenes.example.txt` (name + relative path per line).
- Multi-camera scenes: add sequences with `--gltfCamera 0`, `--gltfCamera 1`, etc., or
  `--cameraString "{eye}, {center}, {up}, {fov}, {near, far}"` for views not in the file (also
  headless: it is re-applied after the scene loads).
- After large setting changes, use `--updateData` or `--resetFrame` (no value; bool triggers) and non-zero `--sequenceresetframes`.
