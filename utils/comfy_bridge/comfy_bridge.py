#!/usr/bin/env python3
# Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""ComfyUI adapter for vk_gltf_renderer's filesystem generation bridge.

The renderer does not link to ComfyUI directly. Instead, it speaks a small
tool-neutral file protocol:

* requests/<job_id>.json  - renderer asks an external generator to make image data
* responses/<job_id>.json - adapter reports success/failure for that exact job
* assets/                 - shared image inputs and generated outputs
* .adapter_heartbeat.json - best-effort liveness/ComfyUI health signal

This script is the ComfyUI implementation of that protocol. Each polling tick
looks for request JSON files, ignores jobs that already have a response, loads a
ComfyUI API-format workflow named in job["workflow"], patches renderer job data
into that workflow, submits it to ComfyUI's HTTP API, waits for the first output
record in ComfyUI history, downloads the file, and writes the renderer response.

The live bridge path is Python-standard-library only so it can be copied next to
the sample executable without installing packages. The optional PNG-to-HDR path
is deliberately isolated in utils/png_to_hdr.py and invoked as a subprocess only
when ComfyUI produced PNG bytes for an .hdr/.exr preferred output.
"""

from __future__ import annotations

import argparse
import base64
import copy
import datetime
import hashlib
import json
import mimetypes
import os
import re
import secrets
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse


REQUEST_SCHEMA = "vk_gltf_renderer.external_generation.request"
RESPONSE_SCHEMA = "vk_gltf_renderer.external_generation.response"
HEARTBEAT_SCHEMA = "vk_gltf_renderer.agentic_bridge.heartbeat"
PROGRESS_SCHEMA = "vk_gltf_renderer.agentic_bridge.job_progress"
ADAPTER_NAME = "comfy_bridge.py"
ADAPTER_VERSION = "1.0"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_TO_HDR_SCRIPT = Path(__file__).resolve().parent.parent / "png_to_hdr.py"

def confine_to_root(bridge_root: Path, candidate: str) -> Path:
    """Resolve a request-supplied, bridge-relative path safely under bridge_root.

    The bridge directory is a shared drop-box: anything able to write a request
    JSON controls these strings. Absolute paths and any ".." traversal are
    rejected so a request can neither read an input from, nor write an output to,
    a location outside the bridge. Raises ValueError on violation (run_job turns
    that into a failed response).
    """
    raw = Path(candidate)
    if raw.is_absolute() or any(part == ".." for part in raw.parts):
        raise ValueError(f"path escapes bridge root: {candidate!r}")
    root = bridge_root.resolve()
    resolved = (root / raw).resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"path escapes bridge root: {candidate!r}")
    return resolved


def read_json(path: Path) -> dict:
    """Read one JSON object from disk.

    Bridge request/response files are small enough that loading them whole keeps
    the adapter simple and makes malformed JSON fail at the job boundary.
    """
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON through a sibling temp file, then atomically publish it.

    The renderer polls response and heartbeat files. Replacing the final path in
    one step prevents it from observing partially written JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def post_json(base_url: str, route: str, payload: dict) -> dict:
    """POST JSON to a ComfyUI endpoint and decode its JSON response.

    On an HTTP error, ComfyUI's body carries the useful diagnosis (for /prompt a
    ``node_errors`` map naming the missing model or invalid input). Surface it in
    the raised message so the renderer's response says *what* failed instead of a
    bare "HTTP Error 400: Bad Request".
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{route}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            node_errors = parsed.get("node_errors") or parsed.get("error")
            detail = f": {json.dumps(node_errors)}" if node_errors else f": {body[:500]}"
        except Exception:  # noqa: BLE001 - fall back to the bare status if the body is unreadable
            detail = ""
        raise RuntimeError(f"ComfyUI {route} failed ({exc.code} {exc.reason}){detail}") from exc


def get_json(base_url: str, route: str) -> dict:
    """GET a JSON object from a ComfyUI endpoint."""
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{route}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def upload_image(base_url: str, image_path: Path) -> str:
    """Upload a renderer-provided image to ComfyUI and return its server name.

    Image-to-image jobs put their source image in the bridge assets directory.
    ComfyUI LoadImage nodes cannot read that path directly, so the adapter first
    uploads the file through /upload/image and then patches workflow nodes to
    refer to the returned ComfyUI-side filename.
    """
    boundary = f"----vk-gltf-renderer-{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
    )
    body.extend(image_path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/upload/image",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("name") or image_path.name


def patch_workflow(workflow: dict, job: dict, bridge_root: Path, base_url: str) -> dict:
    """Return a job-specific copy of an exported ComfyUI API workflow.

    Workflows are stored as reusable templates under utils/comfy_bridge/workflows.
    At runtime the renderer supplies the prompt, optional input image, preferred
    output path, and task kind in the bridge request. The current patching rules
    intentionally stay conservative:

    * every non-negative CLIPTextEncode node with a string `text` input receives job["prompt"]
    * PrimitiveStringMultiline `value` receives job["prompt"] (Flux image_beautifier)
    * every LoadImage node receives the ComfyUI upload name for job.inputs.image
    * job.parameters["steps"] patches Flux2Scheduler and KSampler
    * job.parameters["seed"] / ["noise_seed"] patch KSampler.seed and RandomNoise.noise_seed

    More advanced parameter patching can be added here without changing the
    renderer-side filesystem protocol.
    """
    patched = copy.deepcopy(workflow)
    prompt = job.get("prompt", "")
    parameters = job.get("parameters", {})

    for node in patched.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        title = str(node.get("_meta", {}).get("title", "")).lower()
        if class_type == "CLIPTextEncode" and isinstance(inputs.get("text"), str) and "negative" not in title:
            inputs["text"] = prompt
        if class_type in ("PrimitiveStringMultiline", "PrimitiveString") and prompt:
            if isinstance(inputs.get("value"), str):
                inputs["value"] = prompt
            elif isinstance(inputs.get("text"), str):
                inputs["text"] = prompt
        if prompt and isinstance(inputs.get("prompt"), str) and "negative" not in title:
            inputs["prompt"] = prompt

    input_image = job.get("inputs", {}).get("image")
    if input_image:
        image_path = confine_to_root(bridge_root, input_image)
        uploaded_name = upload_image(base_url, image_path)
        for node in patched.values():
            if not isinstance(node, dict):
                continue
            if node.get("class_type") == "LoadImage":
                node.setdefault("inputs", {})["image"] = uploaded_name

    if "steps" in parameters:
        try:
            steps = int(parameters["steps"])
        except (TypeError, ValueError):
            steps = None
        if steps is not None:
            for node in patched.values():
                if not isinstance(node, dict):
                    continue
                class_type = node.get("class_type", "")
                if class_type in ("Flux2Scheduler", "KSampler"):
                    node.setdefault("inputs", {})["steps"] = steps

    # Match the working resolution to the requested viewport size so a beautify
    # result can be displayed ~1:1 instead of stretched. The renderer sends the
    # viewport width/height; the template otherwise pins ImageScaleToTotalPixels
    # to 1 MP, and stretching that lower-res output over the viewport beats the
    # VAE's fine decode grid into visible moire. Clamped for VRAM / model range.
    try:
        req_w = int(parameters.get("width", 0))
        req_h = int(parameters.get("height", 0))
    except (TypeError, ValueError):
        req_w = req_h = 0
    if req_w > 0 and req_h > 0:
        megapixels = max(0.5, min((req_w * req_h) / 1_000_000.0, 2.5))
        for node in patched.values():
            if isinstance(node, dict) and node.get("class_type") == "ImageScaleToTotalPixels":
                node.setdefault("inputs", {})["megapixels"] = round(megapixels, 3)

    seed_value = None
    for key in ("seed", "noise_seed"):
        if key in parameters:
            try:
                seed_value = int(parameters[key])
            except (TypeError, ValueError):
                seed_value = None
            if seed_value is not None:
                break
    if seed_value is not None:
        for node in patched.values():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type", "")
            inputs = node.setdefault("inputs", {})
            if class_type == "KSampler" and "seed" in inputs:
                inputs["seed"] = seed_value
            if class_type == "RandomNoise":
                inputs["noise_seed"] = seed_value
            if class_type == "SamplerCustom" and "noise_seed" in inputs:
                inputs["noise_seed"] = seed_value

    return patched


def first_output_file(history: dict) -> dict | None:
    """Pick the first media file ComfyUI exposes for a completed prompt.

    ComfyUI history is keyed by node id and each node can expose different media
    buckets. The renderer only needs one generated image/video artifact per
    phase-one job, so the adapter accepts the first image, gif, or video listed.
    """
    for entry in history.values():
        outputs = entry.get("outputs", {})
        for output in outputs.values():
            for key in ("images", "gifs", "videos"):
                files = output.get(key)
                if files:
                    return files[0]
    return None


def download_output(base_url: str, file_info: dict, output_path: Path) -> None:
    """Download a ComfyUI output descriptor from /view into the bridge assets."""
    query = urllib.parse.urlencode(
        {
            "filename": file_info.get("filename", ""),
            "subfolder": file_info.get("subfolder", ""),
            "type": file_info.get("type", "output"),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/view?{query}", timeout=120) as response:
        output_path.write_bytes(response.read())


def preferred_output_path(job: dict, bridge_root: Path, file_info: dict) -> Path:
    """Resolve the renderer-requested output path, falling back to assets/<name>.

    The renderer may ask for a specific relative path such as assets/foo.hdr so
    it can load the generated file after parsing the response. If no preference
    exists, keep ComfyUI's filename and place it in the shared assets directory.
    """
    preferred = job.get("outputs", {}).get("preferredPath")
    if preferred:
        # Confine to the bridge root: an adapter must never let a request write
        # its output to an arbitrary absolute path or via "../" traversal.
        return confine_to_root(bridge_root, preferred)
    # Fallback: ComfyUI's own filename. Strip any directory components and confine
    # to assets/ so a compromised ComfyUI can't steer the write out of the bridge.
    filename = file_info.get("filename") or f"{job.get('id', 'output')}.png"
    return confine_to_root(bridge_root, f"assets/{Path(filename).name}")


def _iso_now() -> str:
    """Return the UTC timestamp format used by bridge metadata files."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def job_progress_path(bridge_root: Path, job_id: str) -> Path:
    """Sidecar file the renderer reads while a ComfyUI prompt is running."""
    return bridge_root / ".job_progress" / f"{job_id}.json"


def write_job_progress(
    bridge_root: Path,
    job_id: str,
    *,
    prompt_id: str = "",
    phase: str = "running",
    value: int = 0,
    max_value: int = 0,
    node: str = "",
    message: str = "",
) -> None:
    """Publish ComfyUI progress for the Agentic window (best-effort)."""
    if max_value > 0 and not message:
        message = f"ComfyUI {value}/{max_value}"
    elif not message:
        message = phase.replace("_", " ").title() if phase else "Running"
    try:
        write_json_atomic(
            job_progress_path(bridge_root, job_id),
            {
                "schema": PROGRESS_SCHEMA,
                "schemaVersion": 1,
                "jobId": job_id,
                "promptId": prompt_id,
                "phase": phase,
                "value": value,
                "max": max_value,
                "node": node,
                "message": message,
                "updatedAt": _iso_now(),
            },
        )
    except Exception as exc:  # noqa: BLE001 - progress must not fail the job
        print(f"[progress] write failed: {exc}", file=sys.stderr)


def clear_job_progress(bridge_root: Path, job_id: str) -> None:
    """Remove progress sidecar when a job finishes or fails."""
    try:
        job_progress_path(bridge_root, job_id).unlink(missing_ok=True)
    except OSError:
        pass


def _http_to_ws_url(http_url: str, client_id: str) -> str:
    parsed = urlparse(http_url.rstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    path = "/ws"
    query = urllib.parse.urlencode({"clientId": client_id})
    return urlunparse((scheme, netloc, path, "", query, ""))


def _ws_connect(ws_url: str, timeout: float = 10.0) -> socket.socket:
    """Minimal blocking WebSocket client (stdlib only) for ComfyUI progress events."""
    parsed = urlparse(ws_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    path = parsed.path or "/ws"
    if parsed.query:
        path = f"{path}?{parsed.query}"

    sock = socket.create_connection((host, port), timeout=timeout)
    if parsed.scheme == "wss":
        ctx = ssl.create_default_context()
        sock = ctx.wrap_socket(sock, server_hostname=host)

    key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
    accept = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()).decode(
        "ascii"
    )
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    sock.sendall(request.encode("utf-8"))
    sock.settimeout(timeout)

    header = bytearray()
    while b"\r\n\r\n" not in header:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("WebSocket handshake closed")
        header.extend(chunk)
    if b"101" not in header.split(b"\r\n", 1)[0]:
        raise ConnectionError(f"WebSocket handshake failed: {header[:200]!r}")
    if accept.encode("ascii") not in header:
        raise ConnectionError("WebSocket handshake missing Sec-WebSocket-Accept")
    sock.settimeout(1.0)
    return sock


def _ws_recv_text(sock: socket.socket) -> str | None:
    """Read one server text frame. Returns None on timeout or non-text frame."""
    try:
        header = sock.recv(2)
    except TimeoutError:
        return None
    if len(header) < 2:
        return None

    b1, b2 = header[0], header[1]
    opcode = b1 & 0x0F
    masked = (b2 & 0x80) != 0
    length = b2 & 0x7F
    if length == 126:
        length = struct.unpack("!H", sock.recv(2))[0]
    elif length == 127:
        length = struct.unpack("!Q", sock.recv(8))[0]

    mask_key = sock.recv(4) if masked else b""
    payload = bytearray()
    while len(payload) < length:
        chunk = sock.recv(length - len(payload))
        if not chunk:
            break
        payload.extend(chunk)
    if masked and mask_key:
        payload = bytearray(b ^ mask_key[i % 4] for i, b in enumerate(payload))

    if opcode == 0x8:  # close
        return None
    if opcode == 0x9:  # ping -> pong
        sock.sendall(bytes([0x8A, 0x00]))
        return None
    if opcode != 0x1:  # text only
        return None
    return payload.decode("utf-8", errors="replace")


def _handle_comfy_ws_message(
    message: dict,
    *,
    prompt_id: str,
    bridge_root: Path,
    job_id: str,
) -> None:
    msg_type = message.get("type", "")
    data = message.get("data") or {}
    if not isinstance(data, dict):
        return
    if data.get("prompt_id") and data.get("prompt_id") != prompt_id:
        return

    if msg_type == "progress":
        value = int(data.get("value", 0))
        max_value = int(data.get("max", 0))
        node = str(data.get("node", ""))
        write_job_progress(
            bridge_root,
            job_id,
            prompt_id=prompt_id,
            phase="progress",
            value=value,
            max_value=max_value,
            node=node,
        )
    elif msg_type == "executing":
        node = str(data.get("node") or "")
        if node:
            write_job_progress(
                bridge_root,
                job_id,
                prompt_id=prompt_id,
                phase="executing",
                node=node,
                message=f"Running node {node}",
            )
        else:
            write_job_progress(
                bridge_root,
                job_id,
                prompt_id=prompt_id,
                phase="executing",
                message="Finishing execution",
            )
    elif msg_type == "execution_start":
        write_job_progress(
            bridge_root,
            job_id,
            prompt_id=prompt_id,
            phase="execution_start",
            message="ComfyUI started",
        )
    elif msg_type in ("execution_success", "execution_cached"):
        write_job_progress(
            bridge_root,
            job_id,
            prompt_id=prompt_id,
            phase=msg_type,
            message="ComfyUI execution complete",
        )


def monitor_comfy_progress(
    comfy_url: str,
    client_id: str,
    prompt_id: str,
    bridge_root: Path,
    job_id: str,
    stop_event: threading.Event,
) -> None:
    """Background thread: subscribe to ComfyUI /ws and update .job_progress/<job_id>.json."""
    write_job_progress(bridge_root, job_id, prompt_id=prompt_id, phase="queued", message="Waiting for ComfyUI")
    ws_url = _http_to_ws_url(comfy_url, client_id)
    sock = None
    try:
        sock = _ws_connect(ws_url, timeout=10.0)
        while not stop_event.is_set():
            raw = _ws_recv_text(sock)
            if raw is None:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                _handle_comfy_ws_message(payload, prompt_id=prompt_id, bridge_root=bridge_root, job_id=job_id)
    except Exception as exc:  # noqa: BLE001 - progress is optional telemetry
        write_job_progress(
            bridge_root,
            job_id,
            prompt_id=prompt_id,
            phase="ws_error",
            message=f"Progress stream: {exc}",
        )
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def _probe_comfy(base_url: str, timeout: float = 2.0) -> bool:
    """Check whether the ComfyUI process is reachable without blocking the loop."""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/system_stats", timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except Exception:  # noqa: BLE001 - any error means "not reachable right now"
        return False


def write_heartbeat(bridge_root: Path, started_at_iso: str, comfy_url: str) -> None:
    """Publish adapter and ComfyUI liveness for the renderer UI.

    The heartbeat answers two separate questions for the Agentic window:

    * is this Python adapter still polling the bridge directory?
    * was the adapter recently able to reach the configured ComfyUI server?

    Heartbeat writes are best-effort. A permission blip or transient filesystem
    error should not stop the adapter from consuming jobs.
    """
    try:
        write_json_atomic(
            bridge_root / ".adapter_heartbeat.json",
            {
                "schema": HEARTBEAT_SCHEMA,
                "schemaVersion": 1,
                "adapter": ADAPTER_NAME,
                "adapterVersion": ADAPTER_VERSION,
                "pid": os.getpid(),
                "startedAt": started_at_iso,
                "lastPollAt": _iso_now(),
                "comfyUrl": comfy_url,
                "comfyReachable": _probe_comfy(comfy_url),
            },
        )
    except Exception as exc:  # noqa: BLE001 - heartbeat must not crash the adapter
        print(f"[heartbeat] write failed: {exc}", file=sys.stderr)


def maybe_convert_png_to_hdr(output_path: Path, python_exe: str) -> str | None:
    """Convert PNG bytes to Radiance HDR when an HDRI workflow emitted PNG.

    ComfyUI image-save nodes (SaveImage / PreviewImage) write PNG, but the
    renderer expects a real HDR-compatible file when the requested path ends in
    .hdr/.exr. If that mismatch is detected by the PNG signature, this helper
    temporarily renames the file to .png and asks utils/png_to_hdr.py to rewrite
    the original output path as Radiance RGBE. This lets an HDRI workflow use
    ComfyUI's ordinary PNG output and still deliver a loadable environment map.

    Returns a status note on success, or None if no conversion was needed. Raises
    RuntimeError if a conversion IS needed but fails — the caller then fails the
    job rather than reporting success for an unloadable file.

    Only .hdr is handled: png_to_hdr.py writes Radiance RGBE, which is not a valid
    .exr, so an .exr request is left untouched (no fake EXR).
    """
    if output_path.suffix.lower() != ".hdr":
        return None
    try:
        with output_path.open("rb") as f:
            head = f.read(len(PNG_SIGNATURE))
    except OSError:
        return None
    if head != PNG_SIGNATURE:
        return None
    if not PNG_TO_HDR_SCRIPT.exists():
        raise RuntimeError(
            f"output is PNG bytes but {PNG_TO_HDR_SCRIPT.name} is missing next to the adapter; "
            f"cannot deliver a loadable .hdr"
        )

    tmp_png = output_path.with_suffix(output_path.suffix + ".png")
    output_path.replace(tmp_png)
    try:
        result = subprocess.run(
            [python_exe, str(PNG_TO_HDR_SCRIPT), str(tmp_png), str(output_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:  # noqa: BLE001 - restore, then fail the job
        if tmp_png.exists() and not output_path.exists():
            tmp_png.replace(output_path)
        raise RuntimeError(f"png_to_hdr crashed: {exc}") from exc

    if result.returncode != 0:
        if tmp_png.exists() and not output_path.exists():
            tmp_png.replace(output_path)
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        last = detail[-1] if detail else "no output"
        hint = ""
        if "No module named" in last and ("PIL" in last or "numpy" in last):
            hint = (
                " (the converter needs numpy + Pillow; pass "
                "--converter-python pointing at a Python that has them, "
                "e.g. ComfyUI portable's python_embeded\\python.exe)"
            )
        raise RuntimeError(f"png_to_hdr failed (exit {result.returncode}): {last}{hint}")

    tmp_png.unlink(missing_ok=True)
    return f"converted PNG -> Radiance HDR via {PNG_TO_HDR_SCRIPT.name}"


def run_job(request_path: Path, args: argparse.Namespace) -> None:
    """Process one bridge request file from validation through response write.

    The response filename mirrors the request stem. That makes the adapter
    idempotent: if a response already exists, the job is considered complete and
    can be skipped on later polling ticks or after adapter restarts.
    """
    bridge_root = Path(args.bridge_root)
    response_path = bridge_root / "responses" / f"{request_path.stem}.json"
    if response_path.exists():
        return

    # Parse defensively: a malformed request must fail this one job (with a
    # response so it is never retried), never crash the polling daemon. Without
    # this, a single bad JSON file would take down the adapter on every restart.
    try:
        request = read_json(request_path)
        if not isinstance(request, dict):
            raise ValueError("request is not a JSON object")
        if request.get("schema") != REQUEST_SCHEMA:
            raise ValueError(f"unexpected schema: {request.get('schema')!r}")
        job = request["job"]
        if not isinstance(job, dict) or "id" not in job:
            raise ValueError("request is missing job.id")
        # job.id becomes the .job_progress/<id>.json sidecar filename, so restrict
        # it to a safe token (the renderer only ever generates ids like
        # "beautify-<ms>-<n>"). Rejects path separators and "..".
        if not isinstance(job["id"], str) or not re.fullmatch(r"[A-Za-z0-9._-]+", job["id"]):
            raise ValueError(f"job.id is not a safe token: {job.get('id')!r}")
    except Exception as exc:  # noqa: BLE001 - reject malformed requests, don't crash
        write_json_atomic(
            response_path,
            {
                "schema": RESPONSE_SCHEMA,
                "schemaVersion": 1,
                "jobId": request_path.stem,
                "status": "failed",
                "message": f"Malformed request: {exc}",
            },
        )
        return

    workflow_file = job.get("workflow")
    if not workflow_file:
        write_json_atomic(
            response_path,
            {
                "schema": RESPONSE_SCHEMA,
                "schemaVersion": 1,
                "jobId": job.get("id", request_path.stem),
                "status": "failed",
                "message": "Request job is missing required field: workflow",
            },
        )
        return

    # Confine the request-supplied workflow name to --workflow-dir: it must be a
    # bare filename, never an absolute path or one with directory components / "..".
    if not isinstance(workflow_file, str) or Path(workflow_file).name != workflow_file:
        write_json_atomic(
            response_path,
            {
                "schema": RESPONSE_SCHEMA,
                "schemaVersion": 1,
                "jobId": job["id"],
                "status": "failed",
                "message": f"Invalid workflow name: {workflow_file!r}",
            },
        )
        return

    workflow_path = Path(args.workflow_dir) / workflow_file
    if not workflow_path.exists():
        write_json_atomic(
            response_path,
            {
                "schema": RESPONSE_SCHEMA,
                "schemaVersion": 1,
                "jobId": job["id"],
                "status": "failed",
                "message": f"Missing workflow: {workflow_path}",
            },
        )
        return

    try:
        # Build and queue a ComfyUI prompt from the workflow template selected
        # by the renderer's task kind.
        workflow = patch_workflow(read_json(workflow_path), job, bridge_root, args.comfy_url)
        submit = post_json(args.comfy_url, "/prompt", {"prompt": workflow, "client_id": args.client_id})
        prompt_id = submit.get("prompt_id")
        if not prompt_id:
            node_errors = submit.get("node_errors")
            raise RuntimeError(f"ComfyUI accepted no prompt_id (node_errors: {json.dumps(node_errors)})")
        job_id = job["id"]

        stop_progress = threading.Event()
        progress_thread = threading.Thread(
            target=monitor_comfy_progress,
            args=(args.comfy_url, args.client_id, prompt_id, bridge_root, job_id, stop_progress),
            daemon=True,
        )
        progress_thread.start()

        # ComfyUI reports completion through /history/<prompt_id>. Poll until at
        # least one output media descriptor appears or the job-level timeout hits.
        deadline = time.monotonic() + args.timeout
        file_info = None
        next_heartbeat = 0.0
        try:
            while time.monotonic() < deadline:
                # A long generation blocks the outer poll loop, so refresh the
                # heartbeat here too; otherwise the renderer would read a stale
                # file and report the (busy) adapter as dead.
                now = time.monotonic()
                if now >= next_heartbeat:
                    write_heartbeat(bridge_root, getattr(args, "started_at", ""), args.comfy_url)
                    next_heartbeat = now + 2.0
                history = get_json(args.comfy_url, f"/history/{prompt_id}")
                file_info = first_output_file(history)
                if file_info:
                    break
                time.sleep(args.poll_interval)
        finally:
            stop_progress.set()
            progress_thread.join(timeout=2.0)
            clear_job_progress(bridge_root, job_id)

        if not file_info:
            raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")

        output_path = preferred_output_path(job, bridge_root, file_info)
        download_output(args.comfy_url, file_info, output_path)
        conversion_note = maybe_convert_png_to_hdr(output_path, args.converter_python)

        # Responses use bridge-relative paths when possible so the renderer can
        # move with the bridge root and still load the generated asset.
        rel_output = output_path.relative_to(bridge_root) if output_path.is_relative_to(bridge_root) else output_path
        message = f"ComfyUI prompt {prompt_id} completed"
        if conversion_note:
            message += f"; {conversion_note}"
        write_json_atomic(
            response_path,
            {
                "schema": RESPONSE_SCHEMA,
                "schemaVersion": 1,
                "jobId": job["id"],
                "status": "succeeded",
                "outputs": {"image": rel_output.as_posix()},
                "message": message,
            },
        )
    except Exception as exc:  # noqa: BLE001 - adapter should report all failures to the bridge
        clear_job_progress(bridge_root, job.get("id", request_path.stem))
        write_json_atomic(
            response_path,
            {
                "schema": RESPONSE_SCHEMA,
                "schemaVersion": 1,
                "jobId": job.get("id", request_path.stem),
                "status": "failed",
                "message": str(exc),
            },
        )


def main() -> int:
    """Parse CLI options, prepare the bridge layout, and run the polling loop."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-root", required=True)
    parser.add_argument("--workflow-dir", required=True)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--client-id", default=f"vk-gltf-renderer-{uuid.uuid4()}")
    parser.add_argument(
        "--converter-python",
        default=sys.executable,
        help=(
            "Python interpreter used to run the optional utils/png_to_hdr.py "
            "converter when an .hdr/.exr output came back as PNG bytes. "
            "Defaults to the current interpreter; set this to ComfyUI portable's "
            "python_embeded\\python.exe if numpy and Pillow live there instead."
        ),
    )
    args = parser.parse_args()

    bridge_root = Path(args.bridge_root)
    (bridge_root / "requests").mkdir(parents=True, exist_ok=True)
    (bridge_root / "responses").mkdir(parents=True, exist_ok=True)
    (bridge_root / "assets").mkdir(parents=True, exist_ok=True)
    (bridge_root / ".job_progress").mkdir(parents=True, exist_ok=True)

    started_at = _iso_now()
    args.started_at = started_at
    try:
        while True:
            write_heartbeat(bridge_root, started_at, args.comfy_url)
            for request_path in sorted((bridge_root / "requests").glob("*.json")):
                run_job(request_path, args)
            if args.once:
                return 0
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("comfy_bridge: stopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
