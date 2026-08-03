# ComfyUI Bridge Adapter

Optional adapter that connects the renderer's filesystem generation bridge to a
local ComfyUI server. The renderer works fully without it.

**Start here:**

- [ComfyUI Agentic setup](https://github.com/nvpro-samples/vk_gltf_renderer/blob/main/docs/comfyui-agentic-setup.md) — install ComfyUI, download models, and run all three processes.
- [Agentic workflow](https://github.com/nvpro-samples/vk_gltf_renderer/blob/main/docs/agentic-workflow.md) — the request/response protocol and bridge layout.

This folder holds only what a developer editing the adapter needs:

- `comfy_bridge.py` — the adapter (Python standard library only; requires Python 3.10+).
- `workflows/*.json` — ComfyUI **API-format** workflow templates the renderer names in each request.

## Run it

The Agentic window's **Copy command** button emits this line pre-filled with your
paths. From the repo root (adjust the bridge root to your build config):

```bash
python utils/comfy_bridge/comfy_bridge.py --bridge-root _bin/Release/agentic_bridge --workflow-dir utils/comfy_bridge/workflows --comfy-url http://127.0.0.1:8188
```

Add `--converter-python <python-with-numpy+Pillow>` to enable PNG→HDR conversion
for HDRI jobs (see the setup guide). The adapter loads the workflow named in
`job.workflow` from `--workflow-dir` with no renaming, patches the prompt / seed /
steps / input image into it, and submits to ComfyUI. Any process that honors the
same JSON files can replace it — see the protocol doc.
