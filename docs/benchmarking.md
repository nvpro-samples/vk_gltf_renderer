# Benchmarking vk_gltf_renderer

> **For contributors and agents.** How to measure performance — GPU timing, headless timing and the scripted sequencer. The exact flags, log fields, and comparison rules are owned by the code (`src/benchmarking.cpp`, `utils/benchmark/`); this doc explains the workflow.

Two performance workflows, and they answer different questions:

1. **[GPU time at a fixed sample count](#gpu-time-at-a-fixed-sample-count-recommended-for-renderer-changes)** — the profiler's `Path Trace` GPU timer with the work per frame pinned. Use this to judge a change to the renderer: it is the only one that isolates GPU render cost from shader compilation, denoiser CPU work and the rest of the loop.
2. **[Headless timing](#headless-timing-simple)** — post-warmup wall-clock throughput for N frames; convenient for a quick end-to-end number or comparing whole builds, but it is CPU wall-clock and will not resolve a small renderer delta.

The GPU workflow drives the scripted sequencer described further down; headless timing runs from plain command-line flags. A third, non-performance workflow reuses the same sequencer to **capture and review the UI** — see [UI inspection](#ui-inspection-windowed-panel-capture) below.

## Headless timing (simple)

Render **500 frames** (or any count) with path tracing. Set **`--frames` and `--ptMaxFrames` to the same value** so every app frame accumulates samples (default `ptMaxFrames` is already 500).

### One-off CLI

```bash
./vk_gltf_renderer --headless --size 1920 1080 \
  --scenefile shader_ball.gltf \
  --hdrfile std_env.hdr \
  --frames 500 --ptMaxFrames 500 \
  --ptSamples 1 --ptAdaptiveSampling 0 \
  --renderSystem 0 --envSystem 1
```

While rendering, periodic progress lines confirm the run is not stuck (every 50 frames or 5 seconds):

```text
HEADLESS_START frames=500 maxFrames=500 ptSamples=1
HEADLESS_PROGRESS app_frame 50/500 (10%) elapsed_ms=1234.5 ms_per_frame=24.69
...
HEADLESS_SUMMARY frames=500 maxFrames=500 ptSamples=1 effective_spp=500 measured_effective_spp=499 resolution=1920x1080 wall_ms=12320.987 ms_per_frame=24.691 total_wall_ms=12345.678 total_ms_per_frame=24.691 warmup_frames=1 measured_frames=499 throughput_MSps=84.0 spp_per_sec=40.49
BENCHMARK_JSON {"schema":1,"type":"headless_summary",...}
```

- **app_frame** — headless loop index (`--frames`).
- In headless mode, `main()` raises `--ptMaxFrames` to at least `--frames` if you set it lower, so every app frame can accumulate samples during timing runs.
- **wall_ms** — measured post-warmup render time. The first completed frame is excluded so one-time setup such as shader specialization is not charged to throughput.
- **total_wall_ms** — full headless render-loop wall time, including warmup and any synchronous setup.
- **ms_per_frame** — `wall_ms / measured_frames`.
- **effective_spp** — total final-image accumulation: `min(frames, maxFrames) × ptSamples`.
- **measured_effective_spp** — accumulation covered by the measured post-warmup window.
- **throughput_MSps** — measured mega pixel-samples per second (`resolution × measured_effective_spp / wall_s / 10⁶`; higher is faster).
- **spp_per_sec** — measured `measured_effective_spp / wall_s` at this resolution (how fast quality accumulates; higher is faster).

Repeat with `--ptSamples 5` for 5 spp per frame (`effective_spp=2500`).

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
  --optixEnable 0 --optixAutoDenoiseEnabled 0 --dlssEnable 0
```

The script must do two things. It has to pin the work per frame -- `--ptSamples N`,
`--ptAdaptiveSampling 0` (adaptive sampling varies the samples per frame, so ms/frame stops being
comparable), and `--ptMaxFrames` large enough that accumulation never stops mid-sequence (otherwise
the frame being timed is just the tonemapper). And it has to **open with a throwaway `load`
sequence**, because scene loading and pipeline creation consume the frames of whichever sequence is
running at the time -- see the note below.

```text
SEQUENCE "load"
--sequenceframes 240
--renderSystem 0
--ptSamples 2
--ptAdaptiveSampling 0
--ptMaxFrames 1000000
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
timing is quoted. The magnitudes below come from the scatter sample scenes, which are volumetric
and highly divergent; treat them as indicative of that class of content rather than universal:

| Parameter | Effect |
|---|---|
| `--ptTechnique` | Ray query (compute) vs the ray tracing pipeline; the pipeline was ~2x faster |
| `--ptUseSER` | Shader Execution Reordering; ~2.3-2.8x on a divergent scattering workload. Silently ignored when the device does not support it |
| `--ptOptimalShader` | Recompiles with only the scene's feature gates; 14-20% across the scatter sample scenes with SER on, ~32% with SER off |

Three traps worth knowing, each of which produces confident-looking numbers that mean nothing:

- **A starved first sequence.** Scene load and pipeline creation eat the frames of the sequence
  that happens to be running, so a script whose first sequence is `measure` can burn all of them
  before a single frame is path traced. The symptom is an empty profiler block
  (`ParameterSequence 0 "warmup" = { }`) or a low `samples` count, and a `--screenshot` taken in
  that window is **solid black** because the framebuffer has not accumulated anything yet. Always
  check `samples` in the block you quote: it should be close to the `--sequenceaverages` you asked
  for. This is the same rule the [UI inspection](#ui-inspection-windowed-panel-capture) workflow
  states, and it applies just as much to timing.
- **Persisted settings.** `_bin/<config>/vk_gltf_renderer.ini` stores `ptOptimalShader`, `ptTechnique`
  and `ptAdaptiveSampling` between runs, so a previous session silently changes what you measure.
  Set every knob that matters explicitly in the script rather than relying on defaults.
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
--ptMaxFrames 1
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

**Path tracer note:** Set `--ptMaxFrames` to match `--ptSamples` when measuring convergence cost. Use `--ptMaxFrames 1` with `--ptSamples 1` for per-frame interactive GPU time.

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
- Multi-camera scenes: add sequences with `--gltfCamera 0`, `--gltfCamera 1`, etc.
- After large setting changes, use `--updateData` or `--resetFrame` (no value; bool triggers) and non-zero `--sequenceresetframes`.
