"""Convert one leaf: what a unit draws itself, without the assets it places. Runs in worker
processes, so it takes and returns plain data and opens only its own unit."""

from __future__ import annotations

import contextlib
import io
import time
import traceback
from pathlib import Path

import numpy as np

from pxr import Usd, UsdGeom

from .. import convert as converter
from ..convert import Converter, convert_camera, usd_diagnostics
from .cache import layer_inputs, make_key
from .merge import merge_leaf
from .scan import open_unit, placed, separates_changed

GENERATOR = "usd-world.py (vk_gltf_renderer)"


def convert(task: dict) -> dict:
    """Convert a unit's own geometry. ``task``: id, root, layer (absolute), prim, selections,
    variants, out, identity, options, inputs (the unit's, from its scan). Returns the leaf's record,
    or {"id", "error"} when the conversion failed (as text: USD errors can't be sent between
    processes)."""
    try:
        return _convert(task)
    except Exception:  # noqa: BLE001 - one broken asset must not stop the others
        return {"id": task["id"], "error": traceback.format_exc()}


def _convert(task: dict) -> dict:
    t0 = time.time()
    out = Path(task["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    text = io.StringIO()
    converter.FILES_READ.clear()  # a worker converts many leaves
    with contextlib.redirect_stdout(text), usd_diagnostics():
        stage, root = open_unit(task["layer"], task["prim"], task["selections"], task["variants"])
        o, cameras = task["options"], []
        conv = Converter(stage, out, o["materials"], o["exposure"] is None, o["single_sided"],
                         skip=lambda prim: placed(prim, separates_changed(task["selections"])),
                         default_sky=False, exposure=o["exposure"] or 0.0, shared_textures=Path(task["textures"]),
                         scene_cameras=cameras)
        conv.run(root)
        scene = _scene_parts(conv, root, cameras, out)
        merged = merge_leaf(conv.gltf, root.GetName()) if o["merge"] == "material" else {}
    # The layers the scan found, and the textures, MDL and MaterialX files the conversion read: a
    # change to any of them converts the leaf again.
    inputs = {**task["inputs"], **layer_inputs(Path(task["root"]), sorted(converter.FILES_READ))}
    rec = {"inputs": inputs, "key": make_key(task["identity"], inputs),
           "triangles": conv.stats["triangles"], "meshes": conv.stats["meshes"],
           "nodes": conv.stats["mesh instances"], "point_instances": conv.stats["point instances"],
           "lights": conv.stats["lights"], "cameras": conv.stats["cameras"],
           "materials": len(conv.gltf.doc.get("materials", [])), "images": len(conv.gltf.doc.get("images", []))}
    if merged:
        rec["merged"] = dict(merged)
    if scene:
        rec["scene"] = scene
    rec["empty"] = not conv.gltf.doc["scenes"][0]["nodes"]
    if not rec["empty"]:
        rec["bounds"] = bounds(conv.gltf)
        conv.gltf.doc["asset"]["generator"] = GENERATOR
        conv.gltf.doc["asset"]["extras"] = {"usd_world": {"id": task["id"], "key": rec["key"]}}
        conv.gltf.write()
    elif out.exists():  # was converted with content before
        out.unlink()
    rec["seconds"] = round(time.time() - t0, 3)
    rec["id"], rec["log"] = task["id"], text.getvalue()  # for the caller, not stored
    return rec


def _scene_parts(conv: Converter, root: Usd.Prim, cameras: list, out: Path) -> dict:
    """Cameras and the environment belong to the scene, not to an asset placed in it: they are kept
    out of the leaf, in its record, and the assembly that has this unit at its top places them.
    Camera matrices are relative to the unit's root; the sky's image is an absolute path."""
    parts = {}
    xforms = UsdGeom.XformCache(conv.time)
    up = None if np.allclose(conv.frame.rot, np.eye(3)) else np.eye(4)
    if up is not None:  # cameras look down their own -Z with +Y up, a frame not rotated with the geometry
        up[:3, :3] = conv.frame.rot
    for prim in cameras:
        cam = {"name": prim.GetName(), "camera": convert_camera(prim, conv.time, conv.frame),
               "matrix": conv.frame.matrix(xforms.ComputeRelativeTransform(prim, root)[0])}
        if up is not None:
            cam["up"] = [float(x) for x in up.T.flatten()]
        parts.setdefault("cameras", []).append(cam)
    sky, conv.gltf.sky = conv.gltf.sky, None
    if sky:
        pano = sky.get("extensions", {}).get("NV_environment_sky_panorama")
        if pano and pano.get("uri"):
            pano["uri"] = (out.parent / pano["uri"]).resolve().as_posix()
        parts["sky"] = sky
    return parts


# ---- bounds ----

def _read(doc: dict, blob: bytes, accessor: int) -> np.ndarray:
    """An accessor written by ``Gltf.accessor`` (float32 VEC3/VEC4), from the unwritten buffer."""
    acc = doc["accessors"][accessor]
    view = doc["bufferViews"][acc["bufferView"]]
    n = {"VEC3": 3, "VEC4": 4}[acc["type"]]
    return np.frombuffer(blob, np.float32, acc["count"] * n, view["byteOffset"]).reshape(-1, n).astype(np.float64)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def corners(lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return np.array([[(hi if i & 1 else lo)[0], (hi if i & 2 else lo)[1], (hi if i & 4 else lo)[2]] for i in range(8)])


def node_matrix(node: dict) -> np.ndarray:
    return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T if "matrix" in node else np.eye(4)


def bounds(gltf) -> list[list[float]] | None:
    """World box [min, max] of a converted document's default scene (glTF space, metres)."""
    doc, blob = gltf.doc, None
    mesh_box = {}
    for i, mesh in enumerate(doc["meshes"]):
        accs = [doc["accessors"][p["attributes"]["POSITION"]] for p in mesh["primitives"]]
        mesh_box[i] = corners(np.min([a["min"] for a in accs], 0), np.max([a["max"] for a in accs], 0))
    points = []

    def visit(index: int, parent: np.ndarray):
        nonlocal blob
        node = doc["nodes"][index]
        world = parent @ node_matrix(node)
        if "mesh" in node:
            box = mesh_box[node["mesh"]]
            inst = node.get("extensions", {}).get("EXT_mesh_gpu_instancing")
            if inst:
                a = inst["attributes"]
                blob = blob if blob is not None else b"".join(gltf.chunks)
                t, s = _read(doc, blob, a["TRANSLATION"]), _read(doc, blob, a["SCALE"])
                r = _quat_to_matrix(_read(doc, blob, a["ROTATION"]))
                local = np.einsum("nij,kj->nki", r * s[:, None, :], box) + t[:, None, :]  # (n, 8, 3)
                pts = local.reshape(-1, 3)
            else:
                pts = box
            points.append(pts @ world[:3, :3].T + world[:3, 3])
        for child in node.get("children", []):
            visit(child, world)

    for root in doc["scenes"][0]["nodes"]:
        visit(root, np.eye(4))
    if not points:
        return None
    pts = np.concatenate(points)
    return [pts.min(0).tolist(), pts.max(0).tolist()]
