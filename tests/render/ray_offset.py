#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: a ray leaving a surface neither hits it again nor skips what lies just above it.

A bounce or shadow ray starts a small step off the surface it leaves (safeOffsetRay). Too small a
step, and the ray hits that surface again: the surface shadows itself in speckles ("acne"). Too
large a step, and the ray starts beyond geometry that lies just above the surface: light leaks
under an object resting on it.

The scene is a large ground quad (two triangles, 100 m across), lit by a directional light at 40
degrees of elevation (so its shadow rays are not aligned with the floor's axis), seen from above.
A small plate floats 0.5 mm over the floor. The plate is clear to the camera (thin, transmissive,
IOR 1) and opaque to shadow rays (--ptShadowTransmission 0), so the camera sees the floor through
it, and the floor under it must be in the plate's shadow. The floor also glows faintly, so a floor
in shadow still shows, and is told apart from a floor the camera ray went through. The script checks:
  - contact: the floor under the plate shows its glow alone -- no light (the shadow rays start
    under the plate), and no missing floor (the ray through the plate starts above the floor),
  - no acne: the open floor is as bright in every placement of the same scene -- near the origin,
    1 km from it, tilted 30 degrees, scaled down 1000 times, and with the floor's size in its
    vertices rather than its node's scale, at both sizes,
with both trace techniques.

Needs a GPU and a built renderer; uses only the Python standard library.

  python tests/render/ray_offset.py --exe _bin/Release/vk_gltf_renderer.exe
"""

import argparse
import base64
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from volume_scatter_punctual import IMAGE_SIZE, TECHNIQUES, read_hdr

SAMPLES = 64

FLOOR_HALF   = 50.0    # m: the floor quad is 100 m across, two triangles
PLATE_HALF   = 0.1     # m
PLATE_GAP    = 0.0005  # m: the plate floats 0.5 mm over the floor
CAMERA_Y     = 1.0     # m, looking straight down
LIGHT_ANGLE  = 40.0    # degrees of elevation
FLOOR_GLOW   = 0.05    # emissive radiance of the floor, well under what the light gives it


# ---------------------------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------------------------

def quad_buffer(half=1.0):
    """Quad [-half,half]^2 in the XZ plane, facing +Y: 4 vertices, 6 indices, as one glTF buffer."""
    positions = [(-half, 0.0, half), (half, 0.0, half), (half, 0.0, -half), (-half, 0.0, -half)]
    indices   = (0, 1, 2, 0, 2, 3)
    pos_bytes = b"".join(struct.pack("<3f", *p) for p in positions)
    nrm_bytes = b"".join(struct.pack("<3f", 0.0, 1.0, 0.0) for _ in positions)
    idx_bytes = b"".join(struct.pack("<H", i) for i in indices)
    return pos_bytes + nrm_bytes + idx_bytes, len(positions), len(indices)


def quat_axis_angle(axis, degrees):
    half = math.radians(degrees) / 2.0
    s    = math.sin(half)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half)]


def quat_rotating_neg_z_to(d):
    """Quaternion turning -Z (where glTF cameras and lights point) to the unit vector d."""
    a = (0.0, 0.0, -1.0)
    axis = (a[1] * d[2] - a[2] * d[1], a[2] * d[0] - a[0] * d[2], a[0] * d[1] - a[1] * d[0])
    length = math.sqrt(sum(c * c for c in axis))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(a, d))))))
    return quat_axis_angle([c / length for c in axis], angle)


def make_scene(scale=1.0, translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0), baked=False):
    """The floor, the plate, the camera and the light under one root node.

    `scale` multiplies every length; `translation` and `rotation` move the root, so they reach the
    floor and the plate as instance transforms. `baked` puts the floor's size in its vertices instead
    of its node's scale: the hardware tests a ray against a triangle in object space, so the floor's
    100 m then shows in the coordinates it computes with."""
    unit, vertex_count, index_count = quad_buffer()
    floor_half = FLOOR_HALF * scale
    floor_data = quad_buffer(floor_half)[0] if baked else unit
    floor_scale = [1.0, 1.0, 1.0] if baked else [floor_half, 1.0, floor_half]
    data = unit + floor_data
    pos_len = vertex_count * 12
    mesh_len = len(unit)
    light_dir = (-math.cos(math.radians(LIGHT_ANGLE)), -math.sin(math.radians(LIGHT_ANGLE)), 0.0)
    children = [1, 2, 3, 4]
    gltf = {
        "asset": {"version": "2.0", "generator": "ray_offset.py"},
        "extensionsUsed": ["KHR_lights_punctual", "KHR_materials_transmission", "KHR_materials_ior"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "Root", "translation": list(translation), "rotation": list(rotation), "children": children},
            {"name": "Camera", "camera": 0, "translation": [0.0, CAMERA_Y * scale, 0.0],
             "rotation": quat_axis_angle((1.0, 0.0, 0.0), -90.0)},
            {"name": "Light", "rotation": quat_rotating_neg_z_to(light_dir),
             "extensions": {"KHR_lights_punctual": {"light": 0}}},
            {"name": "Floor", "mesh": 0, "scale": floor_scale},
            {"name": "Plate", "mesh": 1, "translation": [0.0, PLATE_GAP * scale, 0.0],
             "scale": [PLATE_HALF * scale, 1.0, PLATE_HALF * scale]},
        ],
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.75, "znear": 0.01 * scale,
                                                            "zfar": 100.0 * scale, "aspectRatio": 1.0}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 3, "NORMAL": 4}, "indices": 5, "material": 0}]},
                   {"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 1}]}],
        "materials": [
            {"name": "Floor", "emissiveFactor": [FLOOR_GLOW] * 3,
             "pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.5, 0.5, 1.0], "metallicFactor": 0.0,
                                      "roughnessFactor": 1.0}},
            # Clear to the camera; --ptShadowTransmission 0 makes it opaque to shadow rays.
            {"name": "Plate", "doubleSided": True,
             "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0,
                                      "roughnessFactor": 0.0},
             "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0},
                            "KHR_materials_ior": {"ior": 1.0}}},
        ],
        "buffers": [{"byteLength": len(data),
                     "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": pos_len, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": 2 * pos_len, "byteLength": index_count * 2, "target": 34963},
            {"buffer": 0, "byteOffset": mesh_len, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": mesh_len + pos_len, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": mesh_len + 2 * pos_len, "byteLength": index_count * 2, "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": vertex_count, "type": "VEC3",
             "min": [-1.0, 0.0, -1.0], "max": [1.0, 0.0, 1.0]},
            {"bufferView": 1, "componentType": 5126, "count": vertex_count, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5123, "count": index_count, "type": "SCALAR"},
            {"bufferView": 3, "componentType": 5126, "count": vertex_count, "type": "VEC3",
             "min": [-floor_half if baked else -1.0, 0.0, -floor_half if baked else -1.0],
             "max": [floor_half if baked else 1.0, 0.0, floor_half if baked else 1.0]},
            {"bufferView": 4, "componentType": 5126, "count": vertex_count, "type": "VEC3"},
            {"bufferView": 5, "componentType": 5123, "count": index_count, "type": "SCALAR"},
        ],
        "extensions": {"KHR_lights_punctual": {"lights": [{"type": "directional", "intensity": 3.0}]}},
    }
    return gltf


PLACEMENTS = {
    "origin": make_scene(),
    "far":    make_scene(translation=(1000.0, 0.0, 1000.0)),
    "tilted": make_scene(rotation=quat_axis_angle((0.0, 0.0, 1.0), 30.0)),
    "small":  make_scene(scale=0.001),
    "baked":  make_scene(baked=True),
    "baked_small": make_scene(scale=0.001, baked=True),
}


# ---------------------------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------------------------

def render(exe, scene_path, output_path, technique):
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE),
           "--scenefile", str(scene_path), "--envSystem", "2", "--renderSystem", "0",
           "--ptTechnique", str(technique), "--ptShadowTransmission", "0",
           "--frames", "0", "--ptMaxSamples", str(SAMPLES), "--output", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        sys.exit(f"render failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def mean_luminance(pixels, width, x0, y0, x1, y1):
    total = 0.0
    for y in range(y0, y1):
        for x in range(x0, x1):
            i = 3 * (y * width + x)
            total += 0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2]
    return total / ((x1 - x0) * (y1 - y0))


# The plate spans the central quarter of the image (half extent 0.1 m seen from 1 m with a 0.75 rad
# field of view): the floor under it is a square well inside that, the open floor a corner square.
UNDER = (IMAGE_SIZE // 2 - 16, IMAGE_SIZE // 2 - 16, IMAGE_SIZE // 2 + 16, IMAGE_SIZE // 2 + 16)
OPEN  = (8, 8, 48, 48)


def measure(path):
    width, _, px = read_hdr(path)
    return mean_luminance(px, width, *UNDER), mean_luminance(px, width, *OPEN)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "ray_offset",
                        help="directory for the generated scenes and images")
    args = parser.parse_args()
    exe = args.exe.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, gltf in PLACEMENTS.items():
        (args.out / f"{name}.gltf").write_text(json.dumps(gltf, indent=1))

    failures = []

    def check(ok, message):
        print(("PASS  " if ok else "FAIL  ") + message)
        if not ok:
            failures.append(message)

    for technique_name, technique in TECHNIQUES.items():
        under, open_floor = {}, {}
        for name in PLACEMENTS:
            image_path = args.out / f"{name}_{technique_name}.hdr"
            image_path.unlink(missing_ok=True)
            render(exe, (args.out / f"{name}.gltf").resolve(), image_path.resolve(), technique)
            under[name], open_floor[name] = measure(image_path)
            print(f"{technique_name:10s} {name:8s} under the plate {under[name]:.6f}  open floor {open_floor[name]:.6f}")

        for name in PLACEMENTS:
            check(abs(under[name] - FLOOR_GLOW) <= 0.1 * FLOOR_GLOW,
                  f"{technique_name}: {name}: the floor 0.5 mm under the plate shows, in the plate's shadow "
                  f"({under[name]:.6f}, its glow alone is {FLOOR_GLOW}; open floor {open_floor[name]:.6f})")
        reference = open_floor["origin"]
        for name in PLACEMENTS:
            check(abs(open_floor[name] - reference) <= 0.01 * reference,
                  f"{technique_name}: {name}: the open floor is as bright as at the origin, no self-shadowing "
                  f"({open_floor[name]:.6f} vs {reference:.6f})")

    print(f"\n{len(failures)} failure(s); images in {args.out.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
