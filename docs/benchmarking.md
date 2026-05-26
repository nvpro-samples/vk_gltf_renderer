# Benchmarking vk_gltf_renderer

Two workflows:

1. **Headless timing (recommended)** — total wall time for N frames at 1 or 5 samples per pixel; compare logs between builds.
2. **Scripted sequencer** — multi-step `.cfg` matrix with per-stage GPU profiler stats (optional, heavier).

## Headless timing (simple)

Render **500 frames** (or any count) with path tracing. Set **`--frames` and `--maxFrames` to the same value** so every app frame accumulates samples (default `maxFrames` is already 500).

### One-off CLI

```bash
./vk_gltf_renderer --headless --size 1920 1080 \
  --scenefile shader_ball.gltf \
  --hdrfile std_env.hdr \
  --frames 500 --maxFrames 500 \
  --ptSamples 1 --ptAdaptiveSampling 0 \
  --renderSystem 0 --envSystem 1
```

While rendering, periodic progress lines confirm the run is not stuck (every 50 frames or 5 seconds):

```text
HEADLESS_START frames=500 maxFrames=500 ptSamples=1
HEADLESS_PROGRESS app_frame 50/500 (10%) elapsed_ms=1234.5 ms_per_frame=24.69
...
HEADLESS_SUMMARY frames=500 maxFrames=500 ptSamples=1 effective_spp=500 resolution=1920x1080 wall_ms=12345.678 ms_per_frame=24.691 throughput_MSps=84.0 spp_per_sec=40.49
BENCHMARK_JSON {"schema":1,"type":"headless_summary",...}
```

- **app_frame** — headless loop index (`--frames`).
- In headless mode, `main()` raises `--maxFrames` to at least `--frames` if you set it lower, so every app frame can accumulate samples during timing runs.
- **wall_ms** — wall-clock time for the headless render loop (what you want for A/B builds).
- **ms_per_frame** — `wall_ms / frames`.
- **effective_spp** — `min(frames, maxFrames) × ptSamples` (actual accumulated samples per pixel).
- **throughput_MSps** — mega pixel-samples per second (`resolution × effective_spp / wall_s / 10⁶`; higher is faster).
- **spp_per_sec** — `effective_spp / wall_s` at this resolution (how fast quality accumulates; higher is faster).

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

1. **`--benchmark 1`** turns off vsync, hides side panels (keeps a fullscreen viewport with the tonemapped image), and drains the scene load pipeline synchronously each frame.
2. **`ElementSequencer`** steps through a `.cfg` script (`SEQUENCE "name"` blocks).
3. After each sequence, **`ProfilerManager`** logs `ParameterSequence` blocks at log level `eSTATS` (GPU/CPU timer averages).
4. **`benchmarkAdvance()`** records Scene and PathTracer/Rasterizer VRAM stats.
5. When the script finishes, the app closes automatically.

## Benchmark script format

Example (`utils/benchmark/quick.cfg`):

```
SEQUENCE "Path tracer - 1 spp"
--sequenceframes 512
--sequenceaverages 128
--sequenceresetframes 16
--renderSystem 0
--ptSamples 1
--maxFrames 1
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

**Path tracer note:** Set `--maxFrames` to match `--ptSamples` when measuring convergence cost. Use `--maxFrames 1` with `--ptSamples 1` for per-frame interactive GPU time.

## Comparing versions

```bash
# Baseline build
python utils/benchmark/benchmark.py run matrix.cfg --scene my_scene.gltf --csv-name baseline.csv

# Candidate build (rebuild executable first)
python utils/benchmark/benchmark.py run matrix.cfg --scene my_scene.gltf --csv-name candidate.csv

python utils/benchmark/benchmark.py compare baseline.csv candidate.csv --output diff.csv --regression-threshold-pct 5
```

`compare` marks **Regression** when candidate GPU time is more than N% slower than baseline. Negative delta % means faster.

## Log parsing

`utils/benchmark/benchmark.py` reads stable `BENCHMARK_JSON` records first, with legacy text parsing as a fallback:

- `BENCHMARK_JSON {"schema":1,"type":"headless_summary",...}`
- `BENCHMARK_JSON {"schema":1,"type":"sequence_memory",...}`
- `ParameterSequence N "name" = { Timer "..."; GPU; avg ...; CPU; avg ...; }`
- `BENCHMARK_ADV N { Memory Scene; ... Memory PathTracer; ... }`

Auto-generated log: `log_<executable>.txt` next to the binary (Logger behavior).

## Tips

- Use fixed resolution (`--size 1920 1080` in `utils/benchmark/benchmark.py`) for comparable numbers.
- Disable validation layers for performance runs (`--vvl` off by default in Release).
- Add scenes to `utils/benchmark/scenes.example.txt` (name + relative path per line).
- Multi-camera scenes: add sequences with `--gltfCamera 0`, `--gltfCamera 1`, etc.
- After large setting changes, use `--updateData` or `--resetFrame` (no value; bool triggers) and non-zero `--sequenceresetframes`.
