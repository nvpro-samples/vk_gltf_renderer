#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test for the beta caustics (docs/caustics.md): a glass ball on a diffuse floor.

The script writes small glTF scenes, renders each one headless to a linear .hdr, and checks that
  - off is the default, and a scene with nothing that can cast a caustic renders the same with the
    setting on,
  - a point light, which path tracing cannot make a caustic of at all, lights the floor behind the
    ball with the caustics on, and leaves it in the ball's black shadow with them off,
  - the estimator carries the right energy: a clear pane of glass (no refraction, no reflection) over
    the floor, seen from below it, lights the floor exactly like the scene without the pane, for a
    point light and an area light alike -- the photons replace the light the pane hides from the paths.

The images land in --out; open them to see the caustic. Needs a GPU and a built renderer; uses only
the Python standard library.

  python tests/render/caustics.py --exe _bin/Release/vk_gltf_renderer.exe [--windowed]
"""

import argparse
import base64
import json
import math
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

IMAGE_SIZE = 384
TECHNIQUE  = 0
BALL       = (0.0, 0.5, 0.0)  # centre of the glass ball, radius 0.5, resting on the floor
WINDOW_AT  = 2.0  # height of the thin pane `window` adds, between the light and the ball
COVER_AT   = 1.4  # height of the pane `cover` adds, above the ball
LIGHT_AT   = (-1.6, 3.0, -0.8)  # above and behind the ball: the caustic falls on the camera's side of it


# ---------------------------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------------------------

def sphere(center, radius, rings=48, segments=96):
    """UV sphere with smooth normals, counter-clockwise seen from outside."""
    positions, normals, indices = [], [], []
    for i in range(rings + 1):
        theta = math.pi * i / rings
        for j in range(segments + 1):
            phi = 2.0 * math.pi * j / segments
            n = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
            positions.append(tuple(center[k] + radius * n[k] for k in range(3)))
            normals.append(n)
    for i in range(rings):
        for j in range(segments):
            a = i * (segments + 1) + j
            b = a + segments + 1
            indices.extend((a, a + 1, b, a + 1, b + 1, b))
    return positions, normals, indices


def quad(y, half, up):
    """Horizontal square at height y, facing +Y (up) or -Y."""
    p = [(-half, y, -half), (-half, y, half), (half, y, half), (half, y, -half)]
    n = [(0.0, 1.0 if up else -1.0, 0.0)] * 4
    idx = (0, 1, 2, 0, 2, 3) if up else (0, 2, 1, 0, 3, 2)
    return p, n, list(idx)


def wall(z, half, height):
    """Vertical rectangle at depth z, from the floor up, facing +Z."""
    p = [(-half, 0.0, z), (half, 0.0, z), (half, height, z), (-half, height, z)]
    return p, [(0.0, 0.0, 1.0)] * 4, [0, 1, 2, 0, 2, 3]


def merge(a, b):
    """Two meshes (positions, normals, indices) as one."""
    return (a[0] + b[0], a[1] + b[1], a[2] + [i + len(a[0]) for i in b[2]])


# A pane has a thickness, so the path tracer takes it for glass that a ray enters and leaves (a thin-walled sheet bends
# nothing, and is not a caster); its index of refraction is 1, so nothing is bent anyway.
PANE_THICKNESS = 0.02
PANE_VOLUME = {"KHR_materials_volume": {"thicknessFactor": PANE_THICKNESS}}


def slab(y, half):
    """A horizontal pane: its top and its bottom face."""
    return merge(quad(y + 0.5 * PANE_THICKNESS, half, True), quad(y - 0.5 * PANE_THICKNESS, half, False))


def vslab(z, half, height):
    """A pane standing across Z: its front and its back face."""
    front, back = wall(z + 0.5 * PANE_THICKNESS, half, height), wall(z - 0.5 * PANE_THICKNESS, half, height)
    return merge(front, ([p for p in back[0]], [(0.0, 0.0, -1.0)] * 4, [0, 2, 1, 0, 3, 2]))


def look_at_rotation(eye, target):
    """Quaternion (x, y, z, w) of a camera at eye looking at target, up = +Y (glTF cameras look down -Z)."""
    f = [target[i] - eye[i] for i in range(3)]
    l = math.sqrt(sum(c * c for c in f))
    f = [c / l for c in f]
    r = [f[1] * 0.0 - f[2] * 1.0, f[2] * 0.0 - f[0] * 0.0, f[0] * 1.0 - f[1] * 0.0]  # f x up
    l = math.sqrt(sum(c * c for c in r))
    r = [c / l for c in r]
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]  # r x f
    # Rotation matrix with columns right, up, -forward
    m = [[r[0], u[0], -f[0]], [r[1], u[1], -f[1]], [r[2], u[2], -f[2]]]
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return [(m[2][1] - m[1][2]) * s, (m[0][2] - m[2][0]) * s, (m[1][0] - m[0][1]) * s, 0.25 / s]
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        return [0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s]
    if m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        return [(m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s]
    s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
    return [(m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s]


def make_scene(light, kind, above=False, cover=False, floor_roughness=1.0, window=False, low=False, thin_cover=False, light_half=0.3):
    """Floor, a camera and a light, and what `kind` says: a "glass" or "matte" ball, a "pane" (a clear sheet of
    glass over the floor, invisible but for being a caster) or "open" (the scene without the pane).
    `light` is "area" or "point". The camera of "pane" and "open" is under the pane, or above it with `above`:
    it then sees the floor through the pane. `cover` adds a pane of that kind over the scene, between the camera
    (above) and everything else, which then is seen through a caster. `window` adds a thin-walled pane (no thickness)
    between the light and the ball: no caster, and the photons must go through it. `low` puts the camera under that
    window (it then sees the scene without looking through it); `thin_cover` makes the `cover` thin-walled, so
    that the camera looks through a pane that is no caster. `light_half` is the half-width of the area light (its power
    stays the same)."""
    ball_material = kind if kind in ("glass", "matte") else None
    meshes = [quad(0.0, 6.0, True), sphere(BALL, 0.5) if ball_material else (slab(1.0, 6.0) if kind == "pane" else quad(1.0, 6.0, True))]
    materials = [
        {"name": "Floor", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1.0], "metallicFactor": 0.0,
                                                    "roughnessFactor": floor_roughness}},
        {"name": "Ball", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0,
                                                   "roughnessFactor": 0.0},
         "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0}, "KHR_materials_ior": {"ior": 1.5},
                        "KHR_materials_volume": {"thicknessFactor": 1.0}}},
    ]
    if kind == "pane":  # index of refraction 1: light goes straight through, nothing is reflected
        materials[1] = {"name": "Pane", "doubleSided": True,
                        "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0,
                                                 "roughnessFactor": 0.0},
                        "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0},
                                       "KHR_materials_ior": {"ior": 1.0}, **PANE_VOLUME}}
    if ball_material == "matte":
        materials[1] = {"name": "Ball", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                                                                   "metallicFactor": 0.0, "roughnessFactor": 1.0}}
    if cover:  # a clear sheet over the scene, which the camera looks through
        meshes.append(merge(quad(COVER_AT, 6.0, True), quad(COVER_AT, 6.0, False)) if thin_cover else slab(COVER_AT, 6.0))
        materials.append({"name": "Cover", "doubleSided": True,
                          "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.0},
                          "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0}, "KHR_materials_ior": {"ior": 1.0},
                                         **({} if thin_cover else PANE_VOLUME)}})
    window_index = None
    if window:
        window_index = len(meshes)
        meshes.append(merge(quad(WINDOW_AT, 6.0, True), quad(WINDOW_AT, 6.0, False)))
        materials.append({"name": "Window", "doubleSided": True,
                          "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.0},
                          "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0}, "KHR_materials_ior": {"ior": 1.0}}})
    if light == "area":
        meshes.append(quad(LIGHT_AT[1], light_half, False))
        meshes[-1] = ([(x + LIGHT_AT[0], y, z + LIGHT_AT[2]) for x, y, z in meshes[-1][0]], meshes[-1][1], meshes[-1][2])
        materials.append({"name": "Light", "emissiveFactor": [1.0, 1.0, 1.0],
                          "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": 150.0 * (0.3 / light_half) ** 2}}})

    eye = (1.0, 2.4, 3.4) if ball_material or above else (1.0, 0.6, 3.4)
    if low:
        eye = (1.0, 1.2, 3.4)
    nodes = [{"name": "Floor", "mesh": 0}] + ([] if kind == "open" else [{"name": "Ball", "mesh": 1}]) + (
             [{"name": "Cover", "mesh": 2}] if cover else []) + (
             [{"name": "Window", "mesh": window_index}] if window else []) + [
             {"name": "Camera", "camera": 0, "translation": list(eye), "rotation": look_at_rotation(eye, (0.3, 0.0, 0.2))}]
    extensions = None
    if light == "area":
        nodes.append({"name": "Light", "mesh": len(meshes) - 1})
    else:
        extensions = {"KHR_lights_punctual": {"lights": [{"type": "point", "intensity": 60.0}]}}
        nodes.append({"name": "Light", "translation": list(LIGHT_AT), "extensions": {"KHR_lights_punctual": {"light": 0}}})
    return assemble(meshes, materials, nodes, extensions)


def assemble(meshes, materials, nodes, extensions=None):
    """A glTF of `meshes` (positions, normals, indices; mesh i uses material i), `materials` and `nodes`, one camera."""
    data = b""
    buffer_views, accessors, gltf_meshes = [], [], []
    for index, (p, n, idx) in enumerate(meshes):
        pos = b"".join(struct.pack("<3f", *v) for v in p)
        nrm = b"".join(struct.pack("<3f", *v) for v in n)
        ind = b"".join(struct.pack("<I", i) for i in idx)
        base = len(accessors)
        for blob, target in ((pos, 34962), (nrm, 34962), (ind, 34963)):
            while len(data) % 4:
                data += b"\0"
            buffer_views.append({"buffer": 0, "byteOffset": len(data), "byteLength": len(blob), "target": target})
            data += blob
        lo = [min(v[k] for v in p) for k in range(3)]
        hi = [max(v[k] for v in p) for k in range(3)]
        accessors += [
            {"bufferView": base, "componentType": 5126, "count": len(p), "type": "VEC3", "min": lo, "max": hi},
            {"bufferView": base + 1, "componentType": 5126, "count": len(n), "type": "VEC3"},
            {"bufferView": base + 2, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
        ]
        gltf_meshes.append({"primitives": [{"attributes": {"POSITION": base, "NORMAL": base + 1}, "indices": base + 2,
                                            "material": index}]})
    gltf = {
        "asset": {"version": "2.0", "generator": "caustics.py"},
        "extensionsUsed": ["KHR_materials_transmission", "KHR_materials_ior", "KHR_materials_volume",
                           "KHR_materials_emissive_strength", "KHR_lights_punctual"],
        "scene": 0,
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.7, "znear": 0.1, "zfar": 100.0, "aspectRatio": 1.0}}],
        "meshes": gltf_meshes,
        "materials": materials,
        "buffers": [{"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode()}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "nodes": nodes,
        "scenes": [{"nodes": list(range(len(nodes)))}],
    }
    if extensions:
        gltf["extensions"] = extensions
    return gltf


def make_far_balls():
    """Two glass balls far apart on a wide floor, lit by a point light between them, and a camera that looks at the first:
    photons aimed at one sphere around both would mostly miss them, and few would land on the one in view."""
    centers = [(-25.0, 0.5, 0.0), (25.0, 0.5, 0.0)]
    meshes = [quad(0.0, 80.0, True)] + [sphere(c, 0.5) for c in centers]
    glass = {"name": "Ball", "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.0},
             "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0}, "KHR_materials_ior": {"ior": 1.5},
                            "KHR_materials_volume": {"thicknessFactor": 1.0}}}
    materials = [{"name": "Floor", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}},
                 dict(glass), dict(glass)]
    eye = (-21.5, 1.4, 3.5)
    nodes = [{"name": "Floor", "mesh": 0}, {"name": "BallA", "mesh": 1}, {"name": "BallB", "mesh": 2},
             {"name": "Camera", "camera": 0, "translation": list(eye), "rotation": look_at_rotation(eye, (-26.5, 0.0, 0.0))},
             {"name": "Light", "translation": [0.0, 7.0, 0.0], "extensions": {"KHR_lights_punctual": {"light": 0}}}]
    return assemble(meshes, materials, nodes, {"KHR_lights_punctual": {"lights": [{"type": "point", "intensity": 3000.0}]}})


def make_spot_room(pane):
    """Light that reaches the camera's floor by a second bounce only: a narrow spot lights the back wall, which lights
    the floor; with `pane` a clear sheet stands in front of the wall, between it and the spot. The spot does not touch
    the floor, so what the camera sees there is the wall's light, which the pane's photons carry."""
    matte = {"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}}
    meshes = [quad(0.0, 6.0, True), wall(-2.5, 6.0, 5.0)]
    materials = [dict(matte, name="Floor"), dict(matte, name="Wall")]
    nodes = [{"name": "Floor", "mesh": 0}, {"name": "Wall", "mesh": 1}]
    if pane:  # index of refraction 1: invisible but for being a caster
        meshes.append(vslab(-1.5, 6.0, 5.0))
        materials.append({"name": "Pane", "doubleSided": True,
                          "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.0},
                          "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0}, "KHR_materials_ior": {"ior": 1.0}, **PANE_VOLUME}})
        nodes.append({"name": "Pane", "mesh": 2})
    eye = (1.0, 0.6, 3.4)
    nodes.append({"name": "Camera", "camera": 0, "translation": list(eye), "rotation": look_at_rotation(eye, (0.3, 0.4, -0.5))})
    nodes.append({"name": "Light", "translation": [0.0, 2.0, 2.0], "extensions": {"KHR_lights_punctual": {"light": 0}}})
    spot = {"type": "spot", "intensity": 400.0, "spot": {"innerConeAngle": 0.02, "outerConeAngle": 0.1}}
    return assemble(meshes, materials, nodes, {"KHR_lights_punctual": {"lights": [spot]}})


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


def mean_luminance(path, box=None):
    """Mean luminance over the image, or over box = (x0, y0, x1, y1) as fractions of its size."""
    width, height, px = read_hdr(path)
    x0, y0, x1, y1 = box or (0.0, 0.0, 1.0, 1.0)
    total, count = 0.0, 0
    for y in range(int(y0 * height), int(y1 * height)):
        for x in range(int(x0 * width), int(x1 * width)):
            i = 3 * (y * width + x)
            total += 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]
            count += 1
    return total / count


def luminances(path, box):
    """Luminance of every pixel of box = (x0, y0, x1, y1), as fractions of the image size."""
    width, height, px = read_hdr(path)
    x0, y0, x1, y1 = box
    return [0.2126 * px[3 * (y * width + x)] + 0.7152 * px[3 * (y * width + x) + 1] + 0.0722 * px[3 * (y * width + x) + 2]
            for y in range(int(y0 * height), int(y1 * height)) for x in range(int(x0 * width), int(x1 * width))]


def block_means(path, blocks=16):
    """The mean luminance of each of blocks x blocks tiles of the image: the picture without its noise."""
    width, height, px = read_hdr(path)
    bw, bh = width // blocks, height // blocks
    return [sum(0.2126 * px[3 * ((by * bh + y) * width + bx * bw + x)] + 0.7152 * px[3 * ((by * bh + y) * width + bx * bw + x) + 1]
                + 0.0722 * px[3 * ((by * bh + y) * width + bx * bw + x) + 2] for y in range(bh) for x in range(bw)) / (bw * bh)
            for by in range(blocks) for bx in range(blocks)]


def relative_error(image, reference):
    """Root mean square difference of two luminance lists, over the mean of the reference."""
    rms = math.sqrt(sum((a - b) ** 2 for a, b in zip(image, reference)) / len(image))
    return rms / (sum(reference) / len(reference))


def png_mean(path):
    """Mean of the colour channels of an 8-bit RGB(A) PNG."""
    data, pos, idat = Path(path).read_bytes(), 8, b""
    while pos < len(data):
        size, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + size]
        if kind == b"IHDR":
            width, height, _, color_type = struct.unpack(">IIBB", body[:10])
        elif kind == b"IDAT":
            idat += body
        pos += 12 + size
    raw, bpp = zlib.decompress(idat), {2: 3, 6: 4}[color_type]
    stride, previous, total, k = width * bpp, bytearray(width * bpp), 0, 0
    for _ in range(height):
        kind, line = raw[k], bytearray(raw[k + 1:k + 1 + stride])
        k += 1 + stride
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = previous[i]
            c = previous[i - bpp] if i >= bpp else 0
            if kind == 1:
                line[i] = (line[i] + a) & 255
            elif kind == 2:
                line[i] = (line[i] + b) & 255
            elif kind == 3:
                line[i] = (line[i] + (a + b) // 2) & 255
            elif kind == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        total += sum(line[i] for i in range(stride) if i % bpp < 3)
        previous = line
    return total / (width * height * 3)


# ---------------------------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------------------------

def render(exe, scene, out, name, samples, *extra):
    """Uses the global TECHNIQUE (--technique): 0 ray query, 1 ray tracing pipeline (its own any-hit shadow code)."""
    output = out / f"{name}.hdr"
    output.unlink(missing_ok=True)
    cmd = [str(exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE), "--scenefile", str(scene), "--envSystem", "2",
           "--renderSystem", "0", "--ptAdaptiveSampling", "0", "--ptSamples", "1", "--frames", "0",
           "--ptTechnique", str(TECHNIQUE), "--ptMaxSamples", str(samples), "--ptFireflyClamp", "0", "--output", str(output), *map(str, extra)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not output.exists():
        sys.exit(f"render failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return output


def scripted_screenshot(exe, scene, out, name, intensities):
    """Runs the windowed sequencer (headless ignores sequence files): 60 frames at each environment intensity of
    `intensities`, then a screenshot with a fixed exposure. Returns its mean value."""
    cfg, shot = out / f"{name}.cfg", out / f"{name}.png"
    shot.unlink(missing_ok=True)
    text = "".join(f'SEQUENCE "{i}"\n--sequenceframes 60\n--hdrIntensity {v}\n--updateData\n' for i, v in enumerate(intensities))
    cfg.write_text(text + f'SEQUENCE "shot"\n--sequenceframes 30\n--screenshot {shot.as_posix()}\n')
    cmd = [str(exe), "--scenefile", str(scene), "--envSystem", "1", "--renderSystem", "0", "--ptAdaptiveSampling", "0", "--ptSamples", "1",
           "--ptMaxSamples", "1000000", "--ptFireflyClamp", "0", "--benchmark", "1", "--tmAutoExposure", "0", "--ptShadowTransmission", "0",
           "--ptCaustics", "1", "--ptTechnique", str(TECHNIQUE), "--sequencefile", str(cfg)]
    subprocess.run(cmd, capture_output=True, text=True)
    if not shot.exists():
        sys.exit(f"no screenshot from {' '.join(cmd)}")
    return png_mean(shot)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "caustics", help="directory for scenes and images")
    parser.add_argument("--samples", type=int, default=256, help="samples per pixel of the caustic renders")
    parser.add_argument("--windowed", action="store_true",
                        help="also run the scene-change check, which needs the sequencer and so opens a window for a few seconds")
    parser.add_argument("--technique", type=int, default=0, choices=(0, 1), help="--ptTechnique: 0 ray query, 1 ray tracing pipeline")
    args = parser.parse_args()
    global TECHNIQUE
    TECHNIQUE = args.technique
    exe = args.exe.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = {}
    for name, (light, ball) in {"point_glass": ("point", "glass"), "point_matte": ("point", "matte"),
                                "point_pane": ("point", "pane"), "point_open": ("point", "open"),
                                "area_pane": ("area", "pane"), "area_open": ("area", "open"),
                                "point_pane_above": ("point", "pane"), "point_open_above": ("point", "open"),
                                "area_pane_above": ("area", "pane"), "area_open_above": ("area", "open"),
                                "area_ball_above": ("area", "glass"), "area_ball_covered": ("area", "glass")}.items():
        scenes[name] = (args.out / f"{name}.gltf").resolve()
        scenes[name].write_text(json.dumps(make_scene(light, ball, name.endswith("_above") or name.startswith("area_ball"),
                                                      name.endswith("_covered")), indent=1))
    # A shiny floor (a receiver that evaluates its BSDF for each photon): the same ball, seen directly and through a pane
    for name, cover in (("area_ball_shiny_above", False), ("area_ball_shiny_covered", True)):
        scenes[name] = (args.out / f"{name}.gltf").resolve()
        scenes[name].write_text(json.dumps(make_scene("area", "glass", True, cover, floor_roughness=0.25), indent=1))
    # A thin window between the light and the ball, and the same scene without it
    for name, window in (("point_window", True), ("point_nowindow", False)):
        scenes[name] = (args.out / f"{name}.gltf").resolve()
        scenes[name].write_text(json.dumps(make_scene("point", "glass", window=window), indent=1))
    # An area light, which paths can hit, behind a thin window; the camera under the window, and the same scene without it
    for name, window in (("area_window", True), ("area_nowindow", False)):
        scenes[name] = (args.out / f"{name}.gltf").resolve()
        scenes[name].write_text(json.dumps(make_scene("area", "glass", window=window, low=True), indent=1))
    # The glass ball seen through a thin pane, and seen directly
    scenes["area_ball_thincover"] = (args.out / "area_ball_thincover.gltf").resolve()
    scenes["area_ball_thincover"].write_text(json.dumps(make_scene("area", "glass", True, True, thin_cover=True), indent=1))
    scenes["far_balls"] = (args.out / "far_balls.gltf").resolve()
    scenes["far_balls"].write_text(json.dumps(make_far_balls(), indent=1))
    scenes["spot_room_pane"] = (args.out / "spot_room_pane.gltf").resolve()
    scenes["spot_room_pane"].write_text(json.dumps(make_spot_room(True), indent=1))

    failures = []

    def check(ok, message):
        print(("PASS  " if ok else "FAIL  ") + message)
        if not ok:
            failures.append(message)

    # The floor in the lower middle of the image, where the ball's shadow and caustic fall
    region = (0.25, 0.45, 0.95, 0.95)
    s = args.samples

    # Nothing can cast a caustic: the setting changes nothing
    off = mean_luminance(render(exe, scenes["point_matte"], args.out, "point_matte_off", s, "--ptCaustics", 0), region)
    on  = mean_luminance(render(exe, scenes["point_matte"], args.out, "point_matte_on", s, "--ptCaustics", 1), region)
    check(abs(on - off) <= 0.01 * off, f"no caster, no change ({on:.6f} vs {off:.6f})")

    # A point light: no path can reach it through glass, so only the photons make the caustic
    # (Shadow Transmission off in all three, so only the caustics differ.) Off is the default: a render
    # without the setting equals the explicit off, on a scene where turning it on changes the image.
    base = ("--ptShadowTransmission", 0)
    off     = mean_luminance(render(exe, scenes["point_glass"], args.out, "point_glass_off", s, *base, "--ptCaustics", 0), region)
    default = mean_luminance(render(exe, scenes["point_glass"], args.out, "point_glass_default", s, *base), region)
    on      = mean_luminance(render(exe, scenes["point_glass"], args.out, "point_glass_on", s, *base, "--ptCaustics", 1), region)
    check(abs(default - off) <= 0.005 * off, f"off is the default ({default:.6f} vs {off:.6f})")
    check(on > 1.02 * off, f"point light: the caustic adds light behind the ball ({on:.6f} vs {off:.6f} without)")

    # A pane with no thickness is thin-walled: it bends nothing, and light samples see through it. It is not a caster,
    # so the setting leaves the picture as it was, pixel for pixel (a window must not hand its light to the photons).
    thin = json.loads(scenes["area_pane"].read_text())
    for material in thin["materials"]:
        material.get("extensions", {}).pop("KHR_materials_volume", None) if material.get("name") == "Pane" else None
    thin_path = (args.out / "area_pane_thin.gltf").resolve()
    thin_path.write_text(json.dumps(thin))
    thin_off = luminances(render(exe, thin_path, args.out, "area_pane_thin_off", 64, "--ptCaustics", 0), (0.0, 0.5, 1.0, 1.0))
    thin_on = luminances(render(exe, thin_path, args.out, "area_pane_thin_on", 64, "--ptCaustics", 1), (0.0, 0.5, 1.0, 1.0))
    error = relative_error(thin_on, thin_off)
    check(error < 0.01, f"a thin-walled pane is not a caster: the setting changes nothing (pixel error {error:.4f})")

    # A thin window between the light and the ball: no caster, so the photons go through it. The caustic is the one the
    # open scene has (light samples see through the window as a filter, which Shadow Transmission turns on).
    through = ("--ptShadowTransmission", 1)
    window_off = mean_luminance(render(exe, scenes["point_window"], args.out, "point_window_off", s, *through, "--ptCaustics", 0), region)
    window_on  = mean_luminance(render(exe, scenes["point_window"], args.out, "point_window_on", s, *through, "--ptCaustics", 1), region)
    open_on    = mean_luminance(render(exe, scenes["point_nowindow"], args.out, "point_nowindow_on", s, *through, "--ptCaustics", 1), region)
    check(window_on > 1.02 * window_off, f"a window before the light: the ball still casts its caustic ({window_on:.6f} vs {window_off:.6f} without)")
    check(abs(window_on - open_on) <= 0.05 * open_on, f"...and it is the one of the open scene ({window_on:.6f} vs {open_on:.6f})")

    # The same with an area light, which a path can hit. The light of a camera path floor -> ball -> window -> light is
    # the caustic the photons carry, so the window must not let the paths add it a second time.
    window_on = mean_luminance(render(exe, scenes["area_window"], args.out, "area_window_on", s, *through, "--ptCaustics", 1), region)
    open_on = mean_luminance(render(exe, scenes["area_nowindow"], args.out, "area_nowindow_on", s, *through, "--ptCaustics", 1), region)
    check(abs(window_on - open_on) <= 0.05 * open_on,
          f"an area light behind a window: the caustic is counted once ({window_on:.6f} vs {open_on:.6f} without the window)")

    # The ball seen through a thin pane, as the pane were not there: the light the ball shows (the area light, refracted
    # by it) is the paths' business, and the floor's caustic the photons'.
    direct = render(exe, scenes["area_ball_above"], args.out, "area_ball_above_through", s, *through, "--ptCaustics", 1)
    seen = render(exe, scenes["area_ball_thincover"], args.out, "area_ball_thincover_on", s, *through, "--ptCaustics", 1)
    direct_mean, seen_mean = mean_luminance(direct), mean_luminance(seen)
    check(abs(seen_mean - direct_mean) <= 0.03 * direct_mean,
          f"a ball seen through a thin pane: the same energy as seen directly ({seen_mean:.6f} vs {direct_mean:.6f})")
    error = relative_error(block_means(seen), block_means(direct))
    check(error < 0.08, f"...and the same picture (relative error of 16 x 16 blocks {error:.3f})")

    # The photon map must not disturb what it does not light: on the floor in front of the ball, which the camera sees
    # directly (the splats' business), the map on and off are the same picture pixel for pixel. A pass of the map that
    # clobbered the path tracer's state once left a fixed pepper of dark and bright pixels there, which means hide.
    strip = (0.0, 0.8, 1.0, 1.0)
    map_off = luminances(render(exe, scenes["point_glass"], args.out, "point_glass_map_off", 64, *base, "--ptCaustics", 1, "--ptCausticGather", 0), strip)
    map_on = luminances(render(exe, scenes["point_glass"], args.out, "point_glass_map_on", 64, *base, "--ptCaustics", 1, "--ptCausticGather", 1), strip)
    error = relative_error(map_on, map_off)
    check(error < 0.05, f"the photon map leaves the floor it does not light alone (pixel error {error:.4f})")

    # A frame with few photons, as DLSS renders: the photons are spread evenly (stratified), so 4 samples
    # land close to the converged image. Independent random photons measure about 0.5 here, stratified about 0.3.
    splats = ("--ptCaustics", 1, "--ptCausticGather", 0)
    reference = luminances(render(exe, scenes["point_glass"], args.out, "point_glass_reference", 512, *base, *splats), region)
    few = luminances(render(exe, scenes["point_glass"], args.out, "point_glass_few", 4, *base, *splats), region)
    error = relative_error(few, reference)
    check(error < 0.42, f"few photons: the caustic is close to the converged one (relative error {error:.3f})")

    # Two balls 50 apart, the camera at one: the photons aim at spheres around each ball, not one around both. A
    # sphere around both takes about one photon in 100 to a ball (error 4.6 here, 0.11 now).
    far = scenes["far_balls"]
    far_args = (*base, *splats)
    reference = luminances(render(exe, far, args.out, "far_balls_reference", 256, *far_args, "--ptCausticPhotonsAuto", 0, "--ptCausticPhotonsPerFrame", 1048576), (0.0, 0.4, 1.0, 1.0))
    few = luminances(render(exe, far, args.out, "far_balls_few", 4, *far_args, "--ptCausticPhotonsAuto", 0, "--ptCausticPhotonsPerFrame", 16384), (0.0, 0.4, 1.0, 1.0))
    error = relative_error(few, reference)
    check(error < 0.25, f"two casters far apart: the photons find them ({error:.3f})")

    # The estimator's energy: a pane that changes nothing but who carries the light, seen from under it
    floor = (0.0, 0.5, 1.0, 1.0)
    for light in ("point", "area"):
        pane = mean_luminance(render(exe, scenes[f"{light}_pane"], args.out, f"{light}_pane_on", s, "--ptCaustics", 1), floor)
        open_ = mean_luminance(render(exe, scenes[f"{light}_open"], args.out, f"{light}_open", s), floor)
        check(abs(pane - open_) <= 0.05 * open_, f"{light} light: the pane's photons match the open floor ({pane:.6f} vs {open_:.6f})")

    # The same pane seen from above: the floor is behind a caster, so the photon map carries its light
    for light in ("point", "area"):
        pane = mean_luminance(render(exe, scenes[f"{light}_pane_above"], args.out, f"{light}_pane_above_on", s, "--ptCaustics", 1), (0, 0, 1, 1))
        open_ = mean_luminance(render(exe, scenes[f"{light}_open_above"], args.out, f"{light}_open_above", s), (0, 0, 1, 1))
        check(abs(pane - open_) <= 0.05 * open_, f"{light} light, from above: the photon map matches the open floor ({pane:.6f} vs {open_:.6f})")

    # A second bounce: a spot lights the back wall through a clear pane, and the wall lights the floor in turn. The
    # photons carry the light of the wall at the floor's bounce, where Shadow Transmission is off: the image has to be
    # the one the path tracer alone makes with it on (clear glass passes the light), which it is only if the photon
    # map also serves the surfaces a path reaches after the first. Without that, the pane blocks the wall's light.
    pane = mean_luminance(render(exe, scenes["spot_room_pane"], args.out, "spot_room_pane_on", s, *base, "--ptCaustics", 1))
    plain = mean_luminance(render(exe, scenes["spot_room_pane"], args.out, "spot_room_pane_plain", s, "--ptCaustics", 0))
    check(abs(pane - plain) <= 0.02 * plain, f"a room behind the pane: the second bounce is covered too ({pane:.6f} vs {plain:.6f})")

    # A shiny floor lights per photon, not once for their mean direction: the map and the splats still agree
    shiny_direct = mean_luminance(render(exe, scenes["area_ball_shiny_above"], args.out, "area_ball_shiny_above_on", s, *base, "--ptCaustics", 1))
    shiny_covered = mean_luminance(render(exe, scenes["area_ball_shiny_covered"], args.out, "area_ball_shiny_covered_on", s, *base, "--ptCaustics", 1))
    check(abs(shiny_covered - shiny_direct) <= 0.05 * shiny_direct,
          f"a shiny floor: the photon map agrees with the splats ({shiny_covered:.6f} vs {shiny_direct:.6f})")

    # A real caustic seen through a caster: a clear pane over the glass ball changes nothing but who carries the
    # light of the floor (the photon map instead of the splats), so the image has to stay. The pane is
    # opaque to shadow rays with Shadow Transmission off, as in the glass tests: without the map it would be black.
    direct = render(exe, scenes["area_ball_above"], args.out, "area_ball_above_on", s, *base, "--ptCaustics", 1)
    covered = render(exe, scenes["area_ball_covered"], args.out, "area_ball_covered_on", s, *base, "--ptCaustics", 1)
    direct_mean, covered_mean = mean_luminance(direct), mean_luminance(covered)
    check(abs(covered_mean - direct_mean) <= 0.03 * direct_mean,
          f"the caustic seen through a pane carries the same energy ({covered_mean:.6f} vs {direct_mean:.6f})")
    error = relative_error(block_means(covered), block_means(direct))
    check(error < 0.08, f"...and the same picture (relative error of 16 x 16 blocks {error:.3f})")

    # The photon map is the frame's photons, so a changed scene cannot show the photons of before. The environment
    # alone lights this scene (the ball under a pane), so its intensity is the change; after 1.0 -> 0.25 the image
    # has to be the one a run at 0.25 gives, not a mix.
    if args.windowed:
        env_scene = make_scene("point", "glass", True, True)
        env_scene["nodes"] = [n for n in env_scene["nodes"] if n["name"] != "Light"]
        env_scene.pop("extensions", None)
        env_scene["scenes"] = [{"nodes": list(range(len(env_scene["nodes"])))}]
        env_path = (args.out / "env_ball_covered.gltf").resolve()
        env_path.write_text(json.dumps(env_scene))
        fresh = scripted_screenshot(exe, env_path, args.out, "env_fresh", [0.25])
        changed = scripted_screenshot(exe, env_path, args.out, "env_changed", [1.0, 0.25])
        bright = scripted_screenshot(exe, env_path, args.out, "env_bright", [1.0])
        check(bright > 1.1 * fresh, f"the two intensities differ ({bright:.1f} vs {fresh:.1f})")
        check(abs(changed - fresh) <= 0.03 * fresh, f"a scene change shows at once, with no photons of before ({changed:.1f} after 1.0 -> 0.25, {fresh:.1f} at 0.25)")

    print(f"\n{len(failures)} failure(s); images in {args.out.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
