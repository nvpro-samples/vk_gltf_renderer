#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: punctual lights must light a scattering medium.

Next-event estimation at a scatter point is the only technique that can reach a punctual light
(point, spot, directional): a phase-sampled ray never hits one. sampleLights() marks those samples
with pdf == DIRAC (-1), and volumeScatterNEE() once dropped them with a `pdf <= 0` guard, so a fog
lit only by a spot light rendered black (GitHub issue nvpro-samples/vk_gltf_renderer#46).

The script writes small glTF scenes (a KHR_materials_scatter fog cube, no other geometry, no
environment), renders each one headless to a linear .hdr, and checks:
  - a spot, a directional and a point light shining into the fog light it,
  - the same spot turned away from the fog leaves it dark,
  - two coincident point lights of half the intensity give the same image as one light,
    which checks the selection pdf the NEE estimate divides by.

Needs a GPU and a built renderer; uses only the Python standard library.

  python tests/render/volume_scatter_punctual.py --exe _bin/Release/vk_gltf_renderer.exe
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

IMAGE_SIZE = 256
SAMPLES    = 256

# --ptTechnique values. TraceShadow has one implementation per technique (ray query, and a
# traditional any-hit shader), so every check runs with both.
TECHNIQUES = {"rayquery": 0, "raytracing": 1}

# Fog cube [-1,1]^3 at the origin, camera on +Z looking at it, lights above.
CAMERA_Z = 4.5
LIGHT_Y  = 3.0


# ---------------------------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------------------------

def cube_buffer():
    """Unit cube with per-face normals: 24 vertices, 36 indices, as one glTF buffer."""
    positions, normals, indices = [], [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            n = [0.0, 0.0, 0.0]
            n[axis] = sign
            u, v = [(axis + 1) % 3, (axis + 2) % 3]
            base = len(positions)
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis], p[u], p[v] = sign, du, dv
                positions.append(p)
                normals.append(n)
            # Counter-clockwise seen from outside.
            quad = (0, 1, 2, 0, 2, 3) if sign > 0 else (0, 2, 1, 0, 3, 2)
            indices.extend(base + i for i in quad)

    pos_bytes = b"".join(struct.pack("<3f", *p) for p in positions)
    nrm_bytes = b"".join(struct.pack("<3f", *n) for n in normals)
    idx_bytes = b"".join(struct.pack("<H", i) for i in indices)
    return pos_bytes + nrm_bytes + idx_bytes, len(positions), len(indices)


def make_scene(lights):
    """glTF with the fog cube, a camera, and one node per (light, translation, rotation)."""
    data, vertex_count, index_count = cube_buffer()
    pos_len = vertex_count * 12
    gltf = {
        "asset": {"version": "2.0", "generator": "volume_scatter_punctual.py"},
        "extensionsUsed": ["KHR_lights_punctual", "KHR_materials_transmission", "KHR_materials_volume",
                           "KHR_materials_scatter"],
        "scene": 0,
        "scenes": [{"nodes": list(range(2 + len(lights)))}],
        "nodes": [
            {"name": "Fog", "mesh": 0},
            {"name": "Camera", "camera": 0, "translation": [0.0, 0.0, CAMERA_Z]},
        ],
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.75, "znear": 0.1, "zfar": 100.0,
                                                            "aspectRatio": 1.0}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0}]}],
        # A clear, mostly scattering medium: no surface color, so all light seen comes from the volume.
        "materials": [{
            "name": "Fog",
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0,
                                     "roughnessFactor": 0.0},
            "extensions": {
                "KHR_materials_transmission": {"transmissionFactor": 1.0},
                "KHR_materials_volume": {"thicknessFactor": 1.0, "attenuationDistance": 1.0,
                                         "attenuationColor": [0.5, 0.5, 0.5]},
                "KHR_materials_scatter": {"scatterStrengthFactor": 1.0, "multiscatterColorFactor": [0.9, 0.9, 0.9],
                                          "scatterAnisotropy": 0.0},
            },
        }],
        "buffers": [{"byteLength": len(data),
                     "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": pos_len, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": 2 * pos_len, "byteLength": index_count * 2, "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": vertex_count, "type": "VEC3",
             "min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            {"bufferView": 1, "componentType": 5126, "count": vertex_count, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5123, "count": index_count, "type": "SCALAR"},
        ],
        "extensions": {"KHR_lights_punctual": {"lights": []}},
    }
    for i, (light, translation, rotation) in enumerate(lights):
        gltf["extensions"]["KHR_lights_punctual"]["lights"].append(light)
        gltf["nodes"].append({"name": f"Light{i}", "translation": translation, "rotation": rotation,
                              "extensions": {"KHR_lights_punctual": {"light": i}}})
    return gltf


# Lights shine along their node's -Z: a -90 degree turn about X points them down (-Y), +90 up.
POINT_DOWN = [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
POINT_UP   = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
ABOVE      = [0.0, LIGHT_Y, 0.0]


def spot(intensity):
    # Narrower than the cube, so the beam shows and most scatter points lie outside the cone.
    return {"type": "spot", "intensity": intensity, "spot": {"innerConeAngle": 0.15, "outerConeAngle": 0.25}}


def point(intensity):
    return {"type": "point", "intensity": intensity}


SCENES = {
    "spot_down":       [(spot(50.0), ABOVE, POINT_DOWN)],
    "spot_away":       [(spot(50.0), ABOVE, POINT_UP)],
    "directional":     [({"type": "directional", "intensity": 5.0}, ABOVE, POINT_DOWN)],
    "point_single":    [(point(50.0), ABOVE, POINT_DOWN)],
    "point_split":     [(point(25.0), ABOVE, POINT_DOWN), (point(25.0), ABOVE, POINT_DOWN)],
}


# ---------------------------------------------------------------------------------------------
# Radiance .hdr reader (the renderer writes linear RGBE, run-length encoded scanlines)
# ---------------------------------------------------------------------------------------------

def read_hdr(path):
    """Returns (width, height, [r, g, b, r, g, b, ...]) in linear float."""
    data = Path(path).read_bytes()
    header_end = data.index(b"\n\n") + 2
    line_end   = data.index(b"\n", header_end)
    dims       = data[header_end:line_end].split()  # b"-Y", h, b"+X", w
    height, width = int(dims[1]), int(dims[3])
    pos = line_end + 1

    pixels = []
    for _ in range(height):
        if width >= 8 and width < 32768 and data[pos] == 2 and data[pos + 1] == 2 and data[pos + 2] < 128:
            pos += 4
            channels = []
            for _ in range(4):
                channel = bytearray()
                while len(channel) < width:
                    count = data[pos]
                    pos += 1
                    if count > 128:
                        channel.extend(bytes([data[pos]]) * (count - 128))
                        pos += 1
                    else:
                        channel.extend(data[pos:pos + count])
                        pos += count
                channels.append(channel)
            rgbe = zip(*channels)
        else:
            rgbe = [tuple(data[pos + 4 * x:pos + 4 * x + 4]) for x in range(width)]
            pos += 4 * width
        for r, g, b, e in rgbe:
            scale = math.ldexp(1.0, e - 136) if e else 0.0
            pixels.extend((r * scale, g * scale, b * scale))
    return width, height, pixels


def center_luminance(path):
    """Mean luminance over the central half of the image, which the fog cube fills."""
    width, height, px = read_hdr(path)
    total, count = 0.0, 0
    for y in range(height // 4, 3 * height // 4):
        for x in range(width // 4, 3 * width // 4):
            i = 3 * (y * width + x)
            total += 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]
            count += 1
    return total / count


# ---------------------------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------------------------

def render(exe, scene_path, output_path, technique):
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE),
           "--scenefile", str(scene_path), "--envSystem", "2", "--renderSystem", "0",
           "--ptTechnique", str(technique),
           # Without it the shadow ray stops at the fog cube's own transmissive boundary: the light
           # then reaches the medium only through a refracting interface, which NEE cannot sample.
           "--ptShadowTransmission", "1",
           "--frames", "0", "--ptMaxSamples", str(SAMPLES), "--output", str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        sys.exit(f"render failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "volume_scatter_punctual",
                        help="directory for the generated scenes and images")
    args = parser.parse_args()
    exe = args.exe.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, lights in SCENES.items():
        (args.out / f"{name}.gltf").write_text(json.dumps(make_scene(lights), indent=1))

    failures = []

    def check(ok, message):
        print(("PASS  " if ok else "FAIL  ") + message)
        if not ok:
            failures.append(message)

    for technique_name, technique in TECHNIQUES.items():
        lum = {}
        for name in SCENES:
            image_path = args.out / f"{name}_{technique_name}.hdr"
            image_path.unlink(missing_ok=True)
            render(exe, (args.out / f"{name}.gltf").resolve(), image_path.resolve(), technique)
            lum[name] = center_luminance(image_path)
            print(f"{technique_name:10s} {name:14s} mean luminance {lum[name]:.6f}")

        # The light shining away bounds what reaches the fog without punctual NEE: with nothing but
        # the fog in the scene, that is nothing at all.
        dark = max(lum["spot_away"], 1e-6)
        for name in ("spot_down", "directional", "point_single"):
            check(lum[name] > 100.0 * dark,
                  f"{technique_name}: {name} lights the fog ({lum[name]:.6f} vs dark {lum['spot_away']:.6f})")

        single, split = lum["point_single"], lum["point_split"]
        check(abs(split - single) <= 0.03 * single,
              f"{technique_name}: two half-intensity point lights match one ({split:.6f} vs {single:.6f})")

    print(f"\n{len(failures)} failure(s); images in {args.out.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
