#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Render test: light that reaches the camera through a medium is attenuated by it.

Three ways light used to cross a KHR_materials_volume medium for free:
  - A shadow ray attenuated a segment only when it ended on a boundary, so the part from the last
    boundary (or from the origin, for a ray starting inside) to a light inside the medium was never
    attenuated, and a surface's shadow ray did not know it started inside one.
  - An emitter was added before the medium segment in front of it was processed, so it was seen at
    full strength through any medium.
  - A camera path started in clear air wherever the camera was, so from inside a medium nothing in
    front of the camera was attenuated, and the parity of every boundary it crossed was inverted.

The script writes small glTF scenes (boxes of medium, no environment), renders each one headless to
a linear .hdr and checks:
  - an emitter inside an absorbing medium is dimmed by exp(-sigma_t * d), against the same emitter
    with no medium around it, both with the camera outside the medium and with the camera inside it
    (where the camera ray meets the emitter before any boundary of the medium),
  - a diffuse surface lit by a point light, both inside an absorbing medium, is dimmed by
    exp(-sigma_t * (d_eye + d_light)): the camera segment and the shadow ray,
  - a point light just inside a scattering fog lights it like the same light just outside: moving
    the light across the boundary must not remove the medium between it and the fog,
  - a fog whose extinction is given with an attenuation distance of 0.2 mm renders like the same
    extinction given over 1 m: the distance only parameterizes sigma_t = -log(color) / distance, so
    nothing may clamp it (a 1 mm floor in one of two places once dropped all absorption of media
    described over a shorter distance),
  - a lit fog scaled down 5000 times renders like the original: light transport has no length of its
    own, so with every length scaled by s, sigma_t by 1/s and the light's intensity by s^2, the
    image must not change. A fixed-size offset anywhere on the path (a ray origin pushed off a
    surface by an absolute step) shows up here as a medium that grows brighter as the scene shrinks.

Every check runs with both trace techniques, each of which has its own TraceShadow.

The medium boxes have an IOR of 1, so there is no refraction and no Fresnel term at normal
incidence, and the analytic ratios hold for the pixels at the center of the image.

Needs a GPU and a built renderer; uses only the Python standard library.

  python tests/render/volume_transmittance.py --exe _bin/Release/vk_gltf_renderer.exe
"""

import argparse
import base64
import json
import math
import struct
import sys
import tempfile
from pathlib import Path

from volume_scatter_punctual import CAMERA_Z, IMAGE_SIZE, TECHNIQUES, cube_buffer, read_hdr, render

SIGMA_T = 1.0  # extinction of the absorbing medium, 1/m


# ---------------------------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------------------------

def medium(sigma_t, scatter=None, distance=1.0):
    """Clear, transmissive box filled with a medium of extinction sigma_t (per channel)."""
    exts = {
        "KHR_materials_transmission": {"transmissionFactor": 1.0},
        "KHR_materials_ior": {"ior": 1.0},
        # attenuationColor is what is left after attenuationDistance: exp(-sigma_t * distance).
        "KHR_materials_volume": {"thicknessFactor": 1.0, "attenuationDistance": distance,
                                 "attenuationColor": [math.exp(-sigma_t * distance)] * 3},
    }
    if scatter is not None:
        exts["KHR_materials_scatter"] = {"scatterStrengthFactor": 1.0, "multiscatterColorFactor": [scatter] * 3,
                                         "scatterAnisotropy": 0.0}
    return {"pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0,
                                     "roughnessFactor": 0.0},
            "extensions": exts}


def diffuse(albedo):
    return {"pbrMetallicRoughness": {"baseColorFactor": [albedo, albedo, albedo, 1.0], "metallicFactor": 0.0,
                                     "roughnessFactor": 1.0}}


def emitter(strength):
    return {"pbrMetallicRoughness": {"baseColorFactor": [0.0, 0.0, 0.0, 1.0], "metallicFactor": 0.0},
            "emissiveFactor": [1.0, 1.0, 1.0],
            "extensions": {"KHR_materials_emissive_strength": {"emissiveStrength": strength}}}


def make_scene(boxes, lights=(), scale=1.0):
    """glTF with one box per (material, center, half extents), a camera on +Z, and point lights.

    `scale` multiplies every position and size, the camera's included; materials and light
    intensities are the caller's to match."""
    data, vertex_count, index_count = cube_buffer()
    pos_len = vertex_count * 12
    materials = [material for material, _, _ in boxes]
    gltf = {
        "asset": {"version": "2.0", "generator": "volume_transmittance.py"},
        "extensionsUsed": ["KHR_lights_punctual", "KHR_materials_transmission", "KHR_materials_volume",
                           "KHR_materials_ior", "KHR_materials_scatter", "KHR_materials_emissive_strength"],
        "scene": 0,
        "scenes": [{"nodes": list(range(1 + len(boxes) + len(lights)))}],
        "nodes": [{"name": "Camera", "camera": 0, "translation": [0.0, 0.0, CAMERA_Z * scale]}],
        "cameras": [{"type": "perspective", "perspective": {"yfov": 0.75, "znear": 0.1 * scale,
                                                            "zfar": 100.0 * scale, "aspectRatio": 1.0}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": i}]}
                   for i in range(len(materials))],
        "materials": materials,
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
    for i, (_, center, half) in enumerate(boxes):
        gltf["nodes"].append({"name": f"Box{i}", "mesh": i, "translation": [scale * c for c in center],
                              "scale": [scale * h for h in half]})
    for i, (intensity, position) in enumerate(lights):
        gltf["extensions"]["KHR_lights_punctual"]["lights"].append({"type": "point", "intensity": intensity})
        gltf["nodes"].append({"name": f"Light{i}", "translation": [scale * p for p in position],
                              "extensions": {"KHR_lights_punctual": {"light": i}}})
    return gltf


MEDIUM_BOX = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

# An emitter cube at the center of the medium box: the camera sees its front face (z = 0.25) through
# 0.75 of medium.
EMITTER    = (emitter(1.0), [0.0, 0.0, 0.0], [0.25, 0.25, 0.25])
EMITTER_D  = 1.0 - 0.25

# The same emitter, the camera and it both inside a large box of medium: the camera sees the front
# face through CAMERA_Z - 0.25 of medium. A thinner medium than SIGMA_T, over that longer distance.
CAMERA_SIGMA_T = 0.2
CAMERA_BOX     = ([0.0, 0.0, 0.0], [6.0, 6.0, 6.0])
CAMERA_D       = CAMERA_Z - 0.25

# A diffuse panel behind a point light, both in the medium box: the camera ray crosses 1.48 of
# medium to the panel's front face (z = -0.48), and the light is 0.48 in front of it.
PANEL      = (diffuse(0.5), [0.0, 0.0, -0.5], [0.6, 0.6, 0.02])
PANEL_LIGHT = (2.0, [0.0, 0.0, 0.0])
PANEL_D    = (1.0 + 0.48) + 0.48

# A dense, bright fog, lit by a point light just under or just over its top face.
FOG        = (medium(3.0, scatter=0.9),) + MEDIUM_BOX
FOG_LIGHT_OFFSET = 0.02

FOG_LIGHT        = (5.0, [0.0, 1.5, 0.0])
SHORT_DISTANCE   = 0.0002  # attenuationDistance 0.2 mm, the same sigma_t as FOG

# The FOG_LIGHT scene, every length times SMALL_SCALE: sigma_t is 1/SMALL_SCALE times larger, over an
# attenuationDistance that much shorter (the same attenuationColor), and the light's intensity is
# SMALL_SCALE^2 times weaker, so it gives the fog the same irradiance at the same relative distance.
SMALL_SCALE      = 0.0002

SCENES = {
    "emitter_alone":     make_scene([EMITTER]),
    "emitter_in_medium": make_scene([(medium(SIGMA_T),) + MEDIUM_BOX, EMITTER]),
    "emitter_camera_in_medium": make_scene([(medium(CAMERA_SIGMA_T),) + CAMERA_BOX, EMITTER]),
    "panel_clear":       make_scene([(medium(0.0),) + MEDIUM_BOX, PANEL], [PANEL_LIGHT]),
    "panel_in_medium":   make_scene([(medium(SIGMA_T),) + MEDIUM_BOX, PANEL], [PANEL_LIGHT]),
    "fog_light_inside":  make_scene([FOG], [(5.0, [0.0, 1.0 - FOG_LIGHT_OFFSET, 0.0])]),
    "fog_light_outside": make_scene([FOG], [(5.0, [0.0, 1.0 + FOG_LIGHT_OFFSET, 0.0])]),
    "distance_one":      make_scene([FOG], [FOG_LIGHT]),
    "distance_short":    make_scene([(medium(3.0, scatter=0.9, distance=SHORT_DISTANCE),) + MEDIUM_BOX], [FOG_LIGHT]),
    "scale_small":       make_scene([(medium(3.0 / SMALL_SCALE, scatter=0.9, distance=SMALL_SCALE),) + MEDIUM_BOX],
                                    [(FOG_LIGHT[0] * SMALL_SCALE ** 2, FOG_LIGHT[1])], scale=SMALL_SCALE),
}


# ---------------------------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------------------------

def mean_luminance(path, half_size):
    """Mean luminance over a centered square of 2 * half_size pixels."""
    width, height, px = read_hdr(path)
    total, count = 0.0, 0
    for y in range(height // 2 - half_size, height // 2 + half_size):
        for x in range(width // 2 - half_size, width // 2 + half_size):
            i = 3 * (y * width + x)
            total += 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2]
            count += 1
    return total / count


# The emitter's and the panel's front faces cover more than these squares, and over them the path
# lengths differ from the on-axis ones by well under 1%.
REGION = {"emitter": 8, "panel": 8, "fog": IMAGE_SIZE // 4, "distance": IMAGE_SIZE // 4, "scale": IMAGE_SIZE // 4}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", required=True, type=Path, help="vk_gltf_renderer executable")
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "volume_transmittance",
                        help="directory for the generated scenes and images")
    args = parser.parse_args()
    exe = args.exe.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    for name, gltf in SCENES.items():
        (args.out / f"{name}.gltf").write_text(json.dumps(gltf, indent=1))

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
            lum[name] = mean_luminance(image_path, REGION[name.split("_")[0]])
            print(f"{technique_name:10s} {name:24s} mean luminance {lum[name]:.6f}")

        def check_ratio(name, measured, expected, tolerance):
            check(abs(measured - expected) <= tolerance * expected,
                  f"{technique_name}: {name}: ratio {measured:.4f}, expected {expected:.4f} (+-{100 * tolerance:.0f}%)")

        check_ratio("emitter seen through the medium", lum["emitter_in_medium"] / lum["emitter_alone"],
                    math.exp(-SIGMA_T * EMITTER_D), 0.03)
        check_ratio("emitter seen from a camera inside the medium",
                    lum["emitter_camera_in_medium"] / lum["emitter_alone"], math.exp(-CAMERA_SIGMA_T * CAMERA_D), 0.03)
        check_ratio("panel lit and seen through the medium", lum["panel_in_medium"] / lum["panel_clear"],
                    math.exp(-SIGMA_T * PANEL_D), 0.05)

        # Outside, the shadow rays also cross the top face, and the many that leave the fog near its
        # top do so at grazing angles, where Schlick's Fresnel reflects much of the light even at IOR 1:
        # the light inside comes out about 15% brighter. Without the medium on its shadow rays, it was 8x.
        inside, outside = lum["fog_light_inside"], lum["fog_light_outside"]
        check(0.97 * outside <= inside <= 1.3 * outside,
              f"{technique_name}: fog lit from just inside matches just outside, up to the top face's Fresnel "
              f"({inside:.6f} vs {outside:.6f})")

        check_ratio("fog with sigma_t given over 0.2 mm matches the same sigma_t over 1 m",
                    lum["distance_short"] / lum["distance_one"], 1.0, 0.02)
        # distance_one is the same scene at scale 1.
        check_ratio(f"fog scaled by {SMALL_SCALE} matches the fog at scale 1",
                    lum["scale_small"] / lum["distance_one"], 1.0, 0.02)

    print(f"\n{len(failures)} failure(s); images in {args.out.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
