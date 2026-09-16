# MCP shader timing

An optional [Model Context Protocol](https://modelcontextprotocol.io) endpoint that lets an AI
agent measure the cost of a shader or render-configuration change against the running renderer.

It is scoped to that one job: load what you want to measure, change the configuration, measure.
There is no scene editing and no asset authoring — the whole feature is
`src/mcp_timing.{cpp,hpp}` plus a handful of lines in `main.cpp`.

## Enable it

Built by default (`NVPRO2_ENABLE_nvmcp`, see the root `CMakeLists.txt`). Enabling it makes
nvpro_core2 fetch its pinned cpp-mcp dependency at configure time; turn the option off to avoid
that download and drop the feature entirely.

```bash
./_bin/Release/vk_gltf_renderer.exe --mcp --scenefile <scene.gltf>
```

The server listens on `http://127.0.0.1:7671/mcp` (Streamable HTTP). Use `--mcpPort` for a
different port. Point your agent's MCP client at that URL.

## What it exposes

Three tools of its own — see the `registerTool(...)` calls in `src/mcp_timing.cpp` for the
authoritative names, schemas and descriptions:

| Tool | Purpose |
|---|---|
| `vk_gltf_reload_shaders` | Recompile Slang shaders from source and swap in the new pipelines |
| `vk_gltf_list_timers` | GPU profiler timer names available right now |
| `vk_gltf_measure` | Time one timer over a controlled window and return the distribution |

Loading is not a tool: set `scenefile` or `hdrfile` and it loads. Those parameters carry a
`callbackSuccess`, so the same write works from the command line, a benchmark sequence, and
`nvpro_set_parameters` alike. Both loads run synchronously on the application thread — the parse
and the Vulkan resource creation complete before the write returns. On failure the load leaves an
empty scene / no HDR (rather than silently continuing on the previous asset) and the callback in
`main.cpp` logs a warning naming the offending file, so a benchmark or agent cannot mistake stale
output for a successful swap. `nvpro_set_parameters` still reports the write itself as successful.

Everything else comes from nvmcp for free: reading and writing parameters, logs, application
state, screenshots, and shutdown.

**Every registered command-line parameter is settable at runtime.** That is
`NVPRO2_ENABLE_MCP_AUTO_PARAMETER_REGISTRY`, on by default here, so the parameter registry is the
single place to add a setting — register it for the command line and the agent can reach it.

Names are grouped by prefix: `pt*` path tracer, `tm*` tonemapper, `hdr*` and `sky*` environment,
`dlss*`/`optix*` denoisers, `dbg*` debug views, `ui*` panels and gizmo.

A parameter that names *state* the renderer reads each frame takes effect immediately. A parameter
that names an *action* needs a `callbackSuccess` to do anything — `scenefile` and `hdrfile` have
one, so they load on write. Anything consumed once at start-up and without a callback (`size`,
`device`, `frames`) still accepts a write and changes nothing.

## Measuring a change

```jsonc
// Pin a fixed-cost frame: no adaptive sampling, no denoiser, a fixed sample count.
nvpro_set_parameters {"values": {"ptAdaptiveSampling": false, "ptSamples": 10,
                                 "dlssEnable": false, "ptOptimalShader": true}}

vk_gltf_list_timers {}                                        // -> "Path Trace (RTX)", ...
vk_gltf_measure {"timer": "Path Trace (RTX)", "frames": 100, "warmup": 30}

// ... edit shaders/gltf_pathtrace.slang ...
vk_gltf_reload_shaders {}                                     // -> compile time
vk_gltf_measure {"timer": "Path Trace (RTX)", "frames": 100, "warmup": 30}
```

**Compare the `median`.** It reproduces to roughly 1% here. The `mean` and `standardDeviation` are
inflated by occasional scheduling outliers and will hide a real change.

The window is sampled per frame after a reset, so it describes only what is running now — unlike
the profiler's own running average, which spans the whole session.

## Limitations and traps

- **A parameter can be set and have no effect.** Automatic exposure cannot tell which values are
  read only at start-up (`size`, `device`, `frames`, …). Those accept a write and change nothing.
  The server instructions list the settings that do take effect live.
- **An idle renderer records no timers.** The profiler rebuilds its snapshot each frame from the
  sections that actually ran, so once accumulation stops at `ptMaxFrames` the timer list goes empty
  and `vk_gltf_measure` times out. Raise `ptMaxFrames` to keep drawing.
- **The scene keeps warming up after the write returns.** `scenefile` completes CPU-side and
  installs the Vulkan resources before the write returns, but BLAS/TLAS builds and the initial
  data uploads are queued into a background pipeline that drains one submission per frame in
  interactive mode (headless/benchmark modes drain fully each frame). Frames drawn during that
  warm-up do record timers — for a partial scene — so sampling then gives a wrong number rather
  than an error (measured on Sponza: a 4x understatement). `vk_gltf_measure` sidesteps this by
  refusing to start until the requested timer produces a non-zero sample: the target section
  does not run until the pipeline it needs is live, so the first sample is also the first
  correct sample. Anything else reading timers straight after a load is on its own — insert a
  short delay or call `vk_gltf_list_timers` in a loop until it lists what you plan to measure.
- **Timings are per view and per environment.** The cost depends on what the camera frames, at
  what resolution, and under which HDR, so a number is only comparable against another taken under
  the same conditions.
- **A playing animation never converges** — it resets accumulation every frame. Pause it before
  measuring.
- **Compiler diagnostics are not returned.** `vk_gltf_reload_shaders` reports wall-clock compile
  time and fails the call when the Slang compile fails, but not *why*. Read `nvpro_get_logs` for the
  errors. A failed compile does **not** leave the previous shader running: the renderer falls back
  to the build-time embedded SPIR-V, so measuring after a failed reload times that fallback.

## Adding a tool

One `registerTool(...)` call in `src/mcp_timing.cpp`. The important decisions:

- **`runOnApplicationThread`** — scene, model and Vulkan state are application-thread-owned, so a
  tool that touches them must set this true and stay short. A tool that *waits on rendered frames*
  must leave it false: the application thread is the one producing those frames, and blocking it
  would deadlock. `vk_gltf_reload_shaders` is the first kind, `vk_gltf_measure` the second.
- **Validate what you read.** nvmcp does not enforce `inputSchema`; the schema documents the tool
  for the agent, but the handler is what rejects bad input.

Renderer-side helpers live in the same file, defined as `GltfRenderer` members so the whole
optional feature stays in one translation unit.
