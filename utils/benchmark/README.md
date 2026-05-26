# Benchmark utilities

Scripts and configs for performance regression checks. Full documentation: [docs/benchmarking.md](../../docs/benchmarking.md).

## Layout

| File | Purpose |
|------|---------|
| `benchmark.py` | CLI entry point |
| `benchmark_paths.py` | Path, executable, and scene-list resolution |
| `benchmark_runner.py` | Subprocess orchestration for headless and sequencer runs |
| `benchmark_results.py` | JSON/log parsing, CSV output, and comparisons |
| `quick.cfg` / `matrix.cfg` | Sequencer scripts (optional, heavier) |
| `tests/` | Parser and comparison fixture tests |
| `output/` | Generated logs and CSV (gitignored) |

Run from the **repository root**:

```bash
# Recommended: headless wall time for 500 frames at 1 and 5 spp
python utils/benchmark/benchmark.py headless --scene resources/shader_ball.gltf --frames 500

# Optional: scripted GPU profiler matrix
python utils/benchmark/benchmark.py run quick.cfg --scene resources/shader_ball.gltf
```

Compare two builds:

```bash
python utils/benchmark/benchmark.py headless-compare \
  utils/benchmark/output/headless_shader_ball_spp1.log \
  utils/benchmark/output/headless_shader_ball_spp1_candidate.log
```
