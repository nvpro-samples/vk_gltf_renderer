# Agentic Workflow

Architecture and protocol for optional external image generation in the Vulkan glTF Renderer. For ComfyUI install, models, and day-to-day commands, see [ComfyUI Agentic setup](comfyui-agentic-setup.md).

## Status

| Layer | State |
|-------|--------|
| Filesystem bridge (`agentic_bridge.*`) | Implemented |
| Renderer controller (`agentic::Controller`, `ui_agentic.cpp`) | Implemented — **HDRI** and **Beautify** only |
| ComfyUI adapter (`utils/comfy_bridge/comfy_bridge.py`) | Implemented |
| MCP-style control plane | Planned (manifest notes `controlPlane.status: planned`) |
| Agent feedback / benchmark loop | Planned |

The sample builds and runs with no ComfyUI install. Generation is opt-in via the **Agentic** window or CLI bridge init.

## Architecture

```text
┌─────────────────────┐     requests/*.json      ┌──────────────────────┐
│ vk_gltf_renderer    │ ───────────────────────► │ comfy_bridge.py      │
│ agentic::Controller │                          │ (poll + Comfy HTTP)  │
│                     │ ◄─────────────────────── │                      │
└─────────────────────┘     responses/*.json     └──────────┬───────────┘
         │ reads assets/                                      │ WebSocket
         │ polls .job_progress/                               ▼
         │ reads .adapter_heartbeat.json              ┌──────────────┐
         └────────────────────────────────────────────│ ComfyUI      │
                                                      └──────────────┘
```

**Code map**

| Responsibility | Location |
|----------------|----------|
| JSON schemas, bridge layout, parse/write | `src/agentic_bridge.cpp`, `agentic_bridge.hpp` |
| Job queue, poll, apply HDR / beautified image | `src/agentic.cpp`, `agentic.hpp` |
| ImGui | `src/ui_agentic.cpp` |
| ComfyUI adapter | `utils/comfy_bridge/comfy_bridge.py` |
| Workflow templates (API format) | `utils/comfy_bridge/workflows/*.json` |
| Prompt presets (UI dropdowns) | `utils/comfy_bridge/prompts.json` (loaded by `agentic::Controller::loadPromptPresets`; `<bridge>/prompts.json` overrides) |

On build, CMake **POST_BUILD** copies `utils/comfy_bridge/` (adapter + workflows) and `utils/png_to_hdr.py` next to the executable (`_bin/<Config>/utils/...`), so the PNG→HDR converter resolves at runtime and the UI’s “copy adapter command” points at real paths. Gated by the `USE_AGENTIC` CMake option (ON by default).

## Bridge directory

Default root: `<exe_dir>/agentic_bridge` (overridable in the Agentic window or `--agenticBridgeRoot`).

```text
agentic_bridge/
├── manifest.json              # written on init; informational (not read at runtime)
├── requests/                  # one JSON per job from the renderer
├── responses/                 # one JSON per job from the adapter
├── assets/                    # inputs + generated outputs (bridge-relative paths)
├── .adapter_heartbeat.json    # adapter liveness + Comfy reachability (optional read)
└── .job_progress/             # live ComfyUI progress per active job (optional read)
    └── <job_id>.json
```

Initialize from the app (the **Agentic** window (F7) creates the layout on first generation) or CLI:

```bash
vk_gltf_renderer --agenticBridgeInit --agenticBridgeRoot path/to/agentic_bridge
```

Omit `--agenticBridgeRoot` to create `agentic_bridge` next to the executable.

## Generation tasks

The manifest lists three task kinds for discoverability. The **in-app UI only queues two of them**:

| Task kind | Comfy workflow (default) | Used in UI |
|-----------|---------------------------|------------|
| `hdri_from_prompt` | `hdri_from_prompt.json` — 1024×512 gen, bicubic 4× → 4096×2048 | **Generate HDRI from prompt** |
| `hdri_from_prompt` | `hdri_from_prompt_4x.json` — PixelDiT 4× → 4096×2048, when **High-res 4× upscale** is on | same button |
| `image_to_image` | `image_beautifier.json` | **Beautify last render** |
| `text_to_image` | *(no workflow shipped — provide your own)* | not exposed |

The `text_to_image` task is advertised in the manifest for discoverability but the UI does not queue it and no `text_to_image.json` template ships; supplying one under `--workflow-dir` is enough to use it. The workflow file for each task is decided by the renderer (`defaultWorkflowFile` in `src/agentic_bridge.cpp`) and the Agentic window.

### HDRI job behavior

- Prompt from the multiline HDRI field; **Sampling** `steps` / `seed` passed as `parameters` (`steps`, `seed`, `noise_seed`).
- Output preferred path: `assets/<job_id>.hdr`.
- On success, the controller loads the HDR as the active environment (via renderer `applyHdri` callback).
- Comfy may emit PNG; the adapter runs `png_to_hdr.py` when the preferred extension is `.hdr` / `.exr`.

### Beautify job behavior

- Task kind is `image_to_image` (img2img), workflow `image_beautifier.json`.
- Input: tonemapped viewport saved as `assets/<job_id>_input.jpg` at **full G-buffer size** (no alignment crop).
- If the viewport is showing the beautified overlay, the controller re-tonemaps from the **live render** before capture so Comfy conditions on the scene, not the previous beautify.
- Parameters include `width` / `height` from the viewport, `match_input_size`, `format=png`, plus shared `steps` / `seed` / `noise_seed`.
- The adapter sizes the workflow's generation resolution to the viewport (`ImageScaleToTotalPixels.megapixels`, clamped ~0.5–2.5 MP) so the result is produced at roughly the display size. This avoids a large internal up/down-scale of the diffusion output, whose fine VAE decode grid would otherwise beat into visible moiré. Input resampling uses `lanczos`, not `nearest`.
- Output: `assets/<job_id>_beautified.png`; displayed in-viewport when dimensions match the current framebuffer.

## Request envelope

File: `requests/<job_id>.json`

```json
{
  "schema": "vk_gltf_renderer.external_generation.request",
  "schemaVersion": 1,
  "job": {
    "id": "hdri-1730000000000-0",
    "kind": "hdri_from_prompt",
    "workflow": "hdri_from_prompt_4x.json",
    "prompt": "…",
    "parameters": {
      "width": "4096",
      "height": "2048",
      "format": "hdr",
      "steps": "20",
      "seed": "12345",
      "noise_seed": "12345"
    },
    "inputs": {},
    "outputs": {
      "preferredPath": "assets/hdri-1730000000000-0.hdr"
    }
  }
}
```

For beautify, `inputs.image` points at the saved JPG (bridge-relative). The adapter patches the named Comfy workflow (prompt, sizes, seeds, load image path) and submits to ComfyUI `/prompt`. Bridge-relative paths in requests and responses are confined to the bridge root on both sides — absolute paths and `..` traversal are rejected — because the bridge directory is shared with a separate process.

## Response envelope

File: `responses/<job_id>.json` (atomic write via `.tmp` sibling)

```json
{
  "schema": "vk_gltf_renderer.external_generation.response",
  "schemaVersion": 1,
  "jobId": "hdri-1730000000000-0",
  "status": "succeeded",
  "outputs": {
    "image": "assets/hdri-1730000000000-0.hdr"
  },
  "message": "ComfyUI prompt … completed"
}
```

`status`: `queued` | `running` | `succeeded` | `failed`. The renderer auto-polls (or **Poll now**) while a job is active; parse failures leave the job active for retry. Successful HDRI/beautify paths are resolved relative to the bridge root.

## Sidecar: job progress

While ComfyUI runs a prompt, the adapter writes:

`.job_progress/<job_id>.json`

```json
{
  "schema": "vk_gltf_renderer.agentic_bridge.job_progress",
  "schemaVersion": 1,
  "jobId": "hdri-…",
  "promptId": "…",
  "phase": "progress",
  "value": 3,
  "max": 10,
  "node": "7",
  "message": "ComfyUI 3/10"
}
```

The Agentic window shows a progress bar when `max > 0`, otherwise the `message` / phase text. The file is removed when the job finishes.

## Sidecar: adapter heartbeat

`.adapter_heartbeat.json` — updated each adapter poll tick (and periodically during a long generation so a busy adapter is not reported dead). The UI shows green / yellow / red from the file age; the thresholds are `kAdapterStaleThreshold` / `kAdapterDeadThreshold` in `src/agentic_bridge.hpp`. Includes `comfyReachable` and adapter metadata.

## Manifest

`manifest.json` is written on bridge init and describes capabilities and planned MCP control plane. **The renderer does not read it at runtime**; request/response schemas are enforced in C++ and the adapter.

## Polling and adapter process

Typical loop:

1. Enable bridge, **Auto Poll** on.
2. Run `comfy_bridge.py` with `--bridge-root`, `--workflow-dir`, `--comfy-url`, and `--converter-python` for HDR conversion (see setup doc).
3. Queue HDRI or Beautify; adapter picks up new `requests/*.json`, skips jobs that already have `responses/*.json`.

The adapter is tool-neutral: any process that honors the same JSON files and writes assets can replace ComfyUI.

## Phase 2: MCP-style control plane (planned)

Expose renderer operations as inspectable tools — scene inspection/editing, renderer and camera controls, capture, profiler telemetry — without coupling them to the generation queue. The manifest already reserves `controlPlane.style: mcp`.

## Phase 3: Agent feedback loop (planned)

Build on benchmark JSON output: generate or modify workflows, run headless/benchmark modes, compare screenshots and timings, iterate. Depends on stable generation + control surfaces from phases 1–2.
