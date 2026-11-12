#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: emissive textures light the scene as area lights, like flat emitters.

A material with an emissive texture is in the emissive-triangle list the path tracer samples
(docs/RENDERING_ARCHITECTURE.md): the pick uses the constant factor, and the shader multiplies the
radiance by the texture at the sampled point (the hit-side MIS uses the same constant factor, and the
caustic photons carry the texture like the light sampler does).

The script writes small glTF scenes (a diffuse floor under a downward-facing emissive quad), renders
each one headless to a linear .hdr, and checks, for both path-tracing techniques:
  - energy: a quad textured with a white and gray checker, or a white and black one, lights the floor
    like a flat quad of the mean radiance (the light sample and the BSDF hit weigh the textured
    emitter the same way, black texels included),
  - a uniformly white textured quad is as noisy as the same flat quad: alone in the scene, beside a flat
    emitter, and under the default sky and its sun,
  - the beta caustics treat a textured quad like the flat quad of the same radiance: the floor under a
    clear pane gets the same light with them on (the photons carry the texture).
Noise is the RMS difference between a low-sample image and the same scene's high-sample image over
the floor, relative to its mean.

The images land in --out. Needs a GPU and a built renderer; uses only the Python standard library.

  python tests/render/emissive_texture.py --exe _bin/Release/vk_gltf_renderer.exe
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

IMAGE_SIZE = 256
LOW_SAMPLES, HIGH_SAMPLES = 64, 1024
ENERGY_SAMPLES = 1024

# --ptTechnique values. TraceShadow has one implementation per technique (ray query, and a
# traditional any-hit shader), so every check runs with both.
TECHNIQUES = {"rayquery": 0, "raytracing": 1}

# --envSystem values (enum EnvSystem in shaders/shaderio.h)
ENV_SKY, ENV_HDR, ENV_NONE, ENV_PLAIN = 0, 1, 2, 3

# The floor as the camera sees it, in fractions of the image: left, top, right, bottom.
FLOOR = (0.12, 0.45, 0.88, 0.98)

CHECKER_LEVELS = (255, 64)  # the two emissive texel values of the checker (sRGB)
EMISSIVE_FACTOR = 4.0
NOISE_LIMIT = 1.15  # allowed noise ratio between two scenes that must sample alike
ENERGY_TOLERANCE = 0.015
CAUSTICS_TOLERANCE = 0.02

# A clear pane (index of refraction 1, so it bends and reflects nothing) between the emitter and the
# floor: with the caustics on it is a caster, and the light that reaches the floor through it is
# the photons' business. (The paths alone get less light through it than there is, and that is not
# what the check is about: it compares the caustics on and off.)
PANE_AT = 1.2
PANE_THICKNESS = 0.02


# ---------------------------------------------------------------------------------------------
# Textures and environments
# ---------------------------------------------------------------------------------------------

def srgb_to_linear(value):
    x = value / 255.0
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


TEXTURES = {
    # Fine cells (about 6 cm on the quad): the floor sees the mean of the two levels.
    "checker": lambda x, y: (CHECKER_LEVELS[((x // 8) + (y // 8)) % 2],) * 3,
    # The same cells, white and black: half of the light samples land on a black texel
    "checker_bw": lambda x, y: (255 if ((x // 8) + (y // 8)) % 2 == 0 else 0,) * 3,
    "white": lambda x, y: (255,) * 3,
}
TEXTURE_SIZE = 256


def write_png(path, size, pixel):
    raw = b"".join(b"\x00" + bytes(c for x in range(size) for c in pixel(x, y)) for y in range(size))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


HDR_WIDTH, HDR_HEIGHT = 1024, 512


# ---------------------------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------------------------

def emitter(texture, factor, half, at):
    """A downward-facing square emitter of half-size `half` centered at `at`; texture is a TEXTURES key or None."""
    return {"texture": texture, "factor": factor, "half": half, "at": at}


def make_scene(emitters, pane=False):
    """glTF: a 10 m diffuse floor, a camera looking at it from the side, and the emitters above it,
    and with `pane` a clear sheet of glass between them (the camera is under it)."""
    positions = [(-1.0, 0.0, -1.0), (1.0, 0.0, -1.0), (1.0, 0.0, 1.0), (-1.0, 0.0, 1.0)]
    uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    chunks = [
        (b"".join(struct.pack("<3f", *p) for p in positions), 34962, 5126, 4, "VEC3"),
        (b"".join(struct.pack("<2f", *t) for t in uvs), 34962, 5126, 4, "VEC2"),
        (b"".join(struct.pack("<3f", 0.0, 1.0, 0.0) for _ in positions), 34962, 5126, 4, "VEC3"),   # normals, floor
        (b"".join(struct.pack("<3f", 0.0, -1.0, 0.0) for _ in positions), 34962, 5126, 4, "VEC3"),  # normals, emitters
        (b"".join(struct.pack("<H", i) for i in (0, 2, 1, 0, 3, 2)), 34963, 5123, 6, "SCALAR"),    # floor, CCW from above
        (b"".join(struct.pack("<H", i) for i in (0, 1, 2, 0, 2, 3)), 34963, 5123, 6, "SCALAR"),    # emitters, CCW from below
    ]
    data, views, accessors = bytearray(), [], []
    for blob, target, component, count, kind in chunks:
        views.append({"buffer": 0, "byteOffset": len(data), "byteLength": len(blob), "target": target})
        data.extend(blob)
        accessors.append({"bufferView": len(views) - 1, "componentType": component, "count": count, "type": kind})
    accessors[0]["min"], accessors[0]["max"] = [-1.0, 0.0, -1.0], [1.0, 0.0, 1.0]

    textures = sorted({e["texture"] for e in emitters if e["texture"]})
    materials = [{
        "name": "Floor",
        "pbrMetallicRoughness": {"baseColorFactor": [0.6, 0.6, 0.6, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0},
        "extensions": {"KHR_materials_ior": {"ior": 1.0}},  # purely diffuse: nothing but the lights' own light shows
    }]
    meshes = [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 2, "TEXCOORD_0": 1}, "indices": 4, "material": 0}]}]
    nodes = [{"name": "Floor", "mesh": 0, "scale": [5.0, 1.0, 5.0]},
             {"name": "Camera", "camera": 0, "translation": [0.0, 1.0, 4.5], "rotation": [-0.1, 0.0, 0.0, 0.995]}]
    for i, e in enumerate(emitters):
        material = {"name": f"Emitter{i}",
                    "pbrMetallicRoughness": {"baseColorFactor": [0.0, 0.0, 0.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0},
                    "emissiveFactor": [e["factor"]] * 3}
        if e["texture"]:
            material["emissiveTexture"] = {"index": textures.index(e["texture"])}
        materials.append(material)
        meshes.append({"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 3, "TEXCOORD_0": 1}, "indices": 5,
                                       "material": len(materials) - 1}]})
        nodes.append({"name": f"Emitter{i}", "mesh": len(meshes) - 1, "translation": list(e["at"]),
                      "scale": [e["half"], 1.0, e["half"]]})
    if pane:  # a slab of two faces, with a thickness so that it is glass a ray enters and leaves
        materials.append({"name": "Pane", "doubleSided": True,
                          "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.0},
                          "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0}, "KHR_materials_ior": {"ior": 1.0},
                                         "KHR_materials_volume": {"thicknessFactor": PANE_THICKNESS}}})
        for normals, indices, y in ((2, 4, PANE_AT + 0.5 * PANE_THICKNESS), (3, 5, PANE_AT - 0.5 * PANE_THICKNESS)):  # top, bottom
            meshes.append({"primitives": [{"attributes": {"POSITION": 0, "NORMAL": normals, "TEXCOORD_0": 1}, "indices": indices,
                                           "material": len(materials) - 1}]})
            nodes.append({"name": "Pane", "mesh": len(meshes) - 1, "translation": [0.0, y, 0.0], "scale": [6.0, 1.0, 6.0]})

    return {
        "asset": {"version": "2.0", "generator": "emissive_texture.py"},
        "extensionsUsed": ["KHR_materials_ior", "KHR_materials_transmission", "KHR_materials_volume"],
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.9, "znear": 0.05, "zfar": 100.0, "aspectRatio": 1.0}}],
        "meshes": meshes,
        "materials": materials,
        # Nearest filtering: the checker's two levels cover exactly half of the quad each
        "samplers": [{"magFilter": 9728, "minFilter": 9728}],
        "images": [{"uri": f"{name}.png"} for name in textures],
        "textures": [{"source": i, "sampler": 0} for i in range(len(textures))],
        "buffers": [{"byteLength": len(data), "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(data)).decode()}],
        "bufferViews": views,
        "accessors": accessors,
    }


# Emitters of the scenes below. All hang above the origin, facing down.
AT = (0.0, 1.5, 0.0)
SCENES = {
    # Energy: the checker against a flat quad of its mean radiance
    "checker": [emitter("checker", EMISSIVE_FACTOR, 1.0, AT)],
    "checker_twin": [emitter(None, EMISSIVE_FACTOR * sum(srgb_to_linear(v) for v in CHECKER_LEVELS) / len(CHECKER_LEVELS), 1.0, AT)],
    "checker_bw": [emitter("checker_bw", EMISSIVE_FACTOR, 1.0, AT)],
    "checker_bw_twin": [emitter(None, EMISSIVE_FACTOR / 2.0, 1.0, AT)],
    # Sampled like a flat emitter: a small white-textured quad against the same flat quad
    "small": [emitter("white", 8.0, 0.3, AT)],
    "small_twin": [emitter(None, 8.0, 0.3, AT)],
    # ... beside a flat emitter (a tiny dim quad off to the side)
    "beside": [emitter("white", 8.0, 0.3, AT), emitter(None, 4.0, 0.1, (2.5, 1.0, -1.0))],
    "beside_twin": [emitter(None, 8.0, 0.3, AT), emitter(None, 4.0, 0.1, (2.5, 1.0, -1.0))],
    # Caustics: the white-textured quad over the floor, with a clear pane between them
    "pane": [emitter("white", 8.0, 0.3, AT)],
    "pane_twin": [emitter(None, 8.0, 0.3, AT)],
}


# ---------------------------------------------------------------------------------------------
# Radiance .hdr reader (the renderer writes linear RGBE, run-length encoded scanlines)
# ---------------------------------------------------------------------------------------------

def read_hdr(path):
    """Returns (width, height, [r, g, b, r, g, b, ...]) in linear float."""
    data = Path(path).read_bytes()
    header_end = data.index(b"\n\n") + 2
    line_end = data.index(b"\n", header_end)
    dims = data[header_end:line_end].split()  # b"-Y", h, b"+X", w
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


def floor_luminance(path):
    """Luminance of every pixel of the floor region, row by row."""
    width, height, px = read_hdr(path)
    left, top, right, bottom = FLOOR
    values = []
    for y in range(int(top * height), int(bottom * height)):
        for x in range(int(left * width), int(right * width)):
            i = 3 * (y * width + x)
            values.append(0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2])
    return values


def noise(low, high):
    mean = sum(high) / len(high)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(low, high)) / len(high)) / mean


# ---------------------------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------------------------

class Renderer:
    def __init__(self, exe, out):
        self.exe, self.out = exe, out

    def render(self, scene, technique_name, env, samples, extra=()):
        """Renders SCENES[scene]; returns the floor's luminance."""
        technique = TECHNIQUES[technique_name]
        image = self.out / f"{scene}_{technique_name}_env{env}_{samples}.hdr"
        image.unlink(missing_ok=True)
        cmd = [str(self.exe), "--headless", "--size", str(IMAGE_SIZE), str(IMAGE_SIZE),
               "--scenefile", str((self.out / f"{scene}.gltf").resolve()), "--envSystem", str(env), "--renderSystem", "0",
               "--ptTechnique", str(technique), "--frames", "0", "--ptMaxSamples", str(samples),
               # The raw running average: one sample per frame, no clamp, no denoiser
               "--ptAdaptiveSampling", "0", "--ptSamples", "1", "--ptFireflyClamp", "0",
               "--dlssEnable", "0", "--optixEnable", "0", "--oidnEnable", "0",
               "--output", str(image.resolve()), *extra]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not image.exists():
            sys.exit(f"render failed ({result.returncode}): {' '.join(cmd)}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
        return floor_luminance(image)

    def scene_noise(self, scene, technique_name, env, extra=()):
        high = self.render(scene, technique_name, env, HIGH_SAMPLES, extra)
        return noise(self.render(scene, technique_name, env, LOW_SAMPLES, extra), high)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "emissive_texture",
                        help="directory for the generated scenes and images")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, pixel in TEXTURES.items():
        write_png(args.out / f"{name}.png", TEXTURE_SIZE, pixel)
    for name, emitters in SCENES.items():
        (args.out / f"{name}.gltf").write_text(json.dumps(make_scene(emitters, pane=name.startswith("pane")), indent=1))

    renderer = Renderer(args.exe.resolve(), args.out)
    failures = []

    def check(ok, message):
        print(("PASS  " if ok else "FAIL  ") + message, flush=True)
        if not ok:
            failures.append(message)

    def same_noise(technique_name, label, scene, twin, env, extra=()):
        ratio = renderer.scene_noise(scene, technique_name, env, extra) / renderer.scene_noise(twin, technique_name, env, extra)
        return ratio, f"{technique_name}: {label} (noise {ratio:.2f}x of {twin}, limit {NOISE_LIMIT})"

    for technique_name in TECHNIQUES:
        # Energy, no environment: the textured quad is the only light, so it is light-sampled and hit by BSDF rays.
        for scene, twin, levels in (("checker", "checker_twin", "white and gray"), ("checker_bw", "checker_bw_twin", "white and black")):
            textured = sum(renderer.render(scene, technique_name, ENV_NONE, ENERGY_SAMPLES))
            flat = sum(renderer.render(twin, technique_name, ENV_NONE, ENERGY_SAMPLES))
            check(abs(textured / flat - 1.0) <= ENERGY_TOLERANCE,
                  f"{technique_name}: a {levels} checker-textured quad lights the floor like a flat quad of its mean radiance "
                  f"({textured / flat:.4f}x, tolerance {ENERGY_TOLERANCE})")

        # As noisy as the same flat quad
        ratio, message = same_noise(technique_name, "a white textured quad alone is sampled like a flat one",
                                    "small", "small_twin", ENV_NONE)
        check(ratio <= NOISE_LIMIT, message)
        ratio, message = same_noise(technique_name, "a white textured quad beside a flat emitter is sampled like a flat one",
                                    "beside", "beside_twin", ENV_PLAIN, ("--plainColor", "0.05", "0.05", "0.05"))
        check(ratio <= NOISE_LIMIT, message)
        ratio, message = same_noise(technique_name, "a white textured quad under the default sky and its sun is sampled like a flat one",
                                    "small", "small_twin", ENV_SKY)
        check(ratio <= NOISE_LIMIT, message)

        # Caustics: the textured quad sends photons like the flat quad, so the floor under a clear pane gets the same light
        black = ("--plainColor", "0", "0", "0", "--ptCaustics", "1")
        textured = sum(renderer.render("pane", technique_name, ENV_PLAIN, ENERGY_SAMPLES, black))
        flat = sum(renderer.render("pane_twin", technique_name, ENV_PLAIN, ENERGY_SAMPLES, black))
        check(abs(textured / flat - 1.0) <= CAUSTICS_TOLERANCE,
              f"{technique_name}: with the caustics on, a textured quad lights the floor under a clear pane like "
              f"the flat quad ({textured / flat:.3f}x, tolerance {CAUSTICS_TOLERANCE})")

    print(f"\n{len(failures)} failure(s); images in {args.out.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
