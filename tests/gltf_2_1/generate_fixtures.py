#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Generate the glTF 2.1 test fixtures in this folder.

Each fixture is a tiny, hand-checkable .gltf exercising one 2.1 feature area (core_properties.gltf
groups the small core properties; external assets get a folder of their own). Buffers and images are
data: URIs, so every file is self-contained except where the feature is about referencing other
files (external assets). Output is deterministic: running the script twice yields identical files,
so a diff after a spec change shows exactly what moved.

Only this script is committed. The build runs it into the build tree (tests/CMakeLists.txt) whenever
it changes, and the tests read the result from there. Choices that rely on a draft spec are tagged
`PROVISIONAL` below, the same marker used in the C++ code (see docs/gltf_2_1.md).

Usage:  python generate_fixtures.py <output folder>
"""

from __future__ import annotations  # `str | None` annotations on Python < 3.10

import base64
import json
import struct
import sys
import zlib
from pathlib import Path

STAMP = ".generated"  # written last; the build depends on it
GLTF_MIME = "model/gltf+json"  # PROVISIONAL(Khronos MR 18): media type of a referenced .gltf file


# --------------------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------------------


def data_uri(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def quad_geometry(model: dict, size: float = 1.0) -> dict:
    """Append a quad (POSITION, NORMAL, indices) in one data: URI buffer. Returns the primitive
    (attributes + indices), without a material."""
    h = size * 0.5
    positions = [(-h, -h, 0.0), (h, -h, 0.0), (h, h, 0.0), (-h, h, 0.0)]
    normals = [(0.0, 0.0, 1.0)] * 4
    indices = [0, 1, 2, 0, 2, 3]

    pos_bytes = b"".join(struct.pack("<3f", *p) for p in positions)
    nrm_bytes = b"".join(struct.pack("<3f", *n) for n in normals)
    idx_bytes = b"".join(struct.pack("<H", i) for i in indices)
    idx_bytes += b"\0" * (-len(idx_bytes) % 4)
    blob = pos_bytes + nrm_bytes + idx_bytes

    buffer = len(model.setdefault("buffers", []))
    model["buffers"].append({"byteLength": len(blob), "uri": data_uri("application/octet-stream", blob)})

    views = model.setdefault("bufferViews", [])
    accessors = model.setdefault("accessors", [])
    first_view = len(views)
    views += [
        {"buffer": buffer, "byteOffset": 0, "byteLength": len(pos_bytes), "target": 34962},
        {"buffer": buffer, "byteOffset": len(pos_bytes), "byteLength": len(nrm_bytes), "target": 34962},
        {"buffer": buffer, "byteOffset": len(pos_bytes) + len(nrm_bytes), "byteLength": 12, "target": 34963},
    ]
    first_accessor = len(accessors)
    accessors += [
        {"bufferView": first_view, "componentType": 5126, "count": 4, "type": "VEC3",
         "min": [-h, -h, 0.0], "max": [h, h, 0.0]},
        {"bufferView": first_view + 1, "componentType": 5126, "count": 4, "type": "VEC3"},
        {"bufferView": first_view + 2, "componentType": 5123, "count": 6, "type": "SCALAR"},
    ]
    return {"attributes": {"POSITION": first_accessor, "NORMAL": first_accessor + 1},
            "indices": first_accessor + 2}


def quad_mesh(model: dict, name: str, color, uid: str | None = None) -> int:
    """Append a material + quad mesh. Returns the mesh index."""
    materials = model.setdefault("materials", [])
    material = {"name": f"{name}_material",
                "pbrMetallicRoughness": {"baseColorFactor": list(color), "metallicFactor": 0.0}}
    if uid:
        material["uid"] = f"{uid}-material"
    materials.append(material)

    primitive = quad_geometry(model)
    primitive["material"] = len(materials) - 1
    mesh = {"name": name, "primitives": [primitive]}
    if uid:
        mesh["uid"] = uid
    model.setdefault("meshes", []).append(mesh)
    return len(model["meshes"]) - 1


def png_1x1(rgba) -> bytes:
    """Smallest valid RGBA PNG, built by hand so the script needs no imaging library."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    pixels = zlib.compress(b"\0" + bytes(rgba), 9)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")


def new_model(version: str = "2.1") -> dict:
    return {"asset": {"version": version, "generator": "vk_gltf_renderer tests/gltf_2_1/generate_fixtures.py"}}


def write(out_dir: Path, relpath: str, model: dict) -> None:
    path = out_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(model, indent=2) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"  {relpath}")


def external_reference(model: dict, uri: str) -> int:
    """Append a `files` entry + `externalAssets` entry for a referenced glTF. Returns the external
    asset index for `node.externalAsset`."""
    files = model.setdefault("files", [])
    files.append({"uri": uri, "mimeType": GLTF_MIME})
    model.setdefault("externalAssets", []).append({"name": Path(uri).stem, "file": len(files) - 1})
    return len(model["externalAssets"]) - 1


# --------------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------------


def core_properties(out_dir: Path) -> None:
    """#26/#32/#36/#37/#38: every new core property tinygltf parses, on one small scene.

    Loading must keep them, and a save + reload must reproduce them unchanged."""
    m = new_model()
    quad_mesh(m, "quad", (0.8, 0.3, 0.2, 1.0), uid="quad-mesh")

    # #38 thumbnail: an image referenced only by asset.thumbnail, not by any texture.
    m["images"] = [{"name": "thumbnail", "uid": "thumbnail-image", "mimeType": "image/png",
                    "uri": data_uri("image/png", png_1x1((204, 77, 51, 255)))}]
    m["asset"]["thumbnail"] = 0

    # #36 shapes: one of each type the current draft defines.
    # PROVISIONAL(Khronos MR 17): the draft also lists convexMesh and mesh, which tinygltf does not parse yet.
    m["shapes"] = [
        {"type": "box", "name": "bounds", "box": {"size": [2.0, 2.0, 1.0]}},
        {"type": "sphere", "sphere": {"radius": 0.75}},
        {"type": "capsule", "capsule": {"height": 1.0, "radiusTop": 0.25, "radiusBottom": 0.5}},
        {"type": "cylinder", "cylinder": {"height": 2.0, "radiusTop": 0.5, "radiusBottom": 0.5}},
        {"type": "plane", "plane": {"sizeX": 4.0, "sizeZ": 3.0}},
    ]

    # #37 bounding volumes: a two-level BVH (root box encloses the child sphere).
    m["nodes"] = [
        {"name": "root", "uid": "root-node", "children": [1],
         "boundingVolume": {"shape": 0, "translation": [0.5, 0.0, 0.0]}},
        {"name": "quad", "uid": "quad-node", "mesh": 0, "translation": [0.5, 0.0, 0.0],
         "boundingVolume": {"shape": 1}},
    ]
    m["scenes"] = [{"name": "scene", "uid": "main-scene", "nodes": [0]}]
    m["scene"] = 0
    write(out_dir, "core_properties.gltf", m)


def external_assets(out_dir: Path) -> None:
    """#33: a root instancing one part three times, plus a two-level nested reference whose inner
    URI is relative to the nested file, not to the root."""
    # Leaf, two folders down: parts/leaf/leaf.gltf
    leaf = new_model()
    quad_mesh(leaf, "leaf", (0.2, 0.8, 0.3, 1.0))
    leaf["nodes"] = [{"name": "leaf", "mesh": 0}]
    leaf["scenes"] = [{"nodes": [0]}]
    leaf["scene"] = 0
    write(out_dir, "external/parts/leaf/leaf.gltf", leaf)

    # Nested: parts/nested.gltf references "leaf/leaf.gltf" (relative to parts/).
    nested = new_model()
    nested_ref = external_reference(nested, "leaf/leaf.gltf")
    nested["nodes"] = [{"name": "nested_instance", "externalAsset": nested_ref, "translation": [0.0, 0.0, -1.0]}]
    nested["scenes"] = [{"nodes": [0]}]
    nested["scene"] = 0
    write(out_dir, "external/parts/nested.gltf", nested)

    # Part instanced three times by the root.
    part = new_model()
    quad_mesh(part, "part", (0.2, 0.3, 0.8, 1.0))
    part["nodes"] = [{"name": "part", "mesh": 0}]
    part["scenes"] = [{"nodes": [0]}]
    part["scene"] = 0
    write(out_dir, "external/parts/part.gltf", part)

    root = new_model()
    part_ref = external_reference(root, "parts/part.gltf")
    nested_ref = external_reference(root, "parts/nested.gltf")
    root["nodes"] = [
        {"name": "part_a", "externalAsset": part_ref, "translation": [-1.5, 0.0, 0.0]},
        {"name": "part_b", "externalAsset": part_ref, "translation": [0.0, 0.0, 0.0]},
        {"name": "part_c", "externalAsset": part_ref, "translation": [1.5, 0.0, 0.0]},
        {"name": "nested", "externalAsset": nested_ref, "translation": [0.0, 1.5, 0.0]},
    ]
    root["scenes"] = [{"nodes": [0, 1, 2, 3]}]
    root["scene"] = 0
    write(out_dir, "external/root.gltf", root)

    # Cycle: a -> b -> a. Each file also has a quad of its own, so what survives is countable.
    for name, other, color in (("cycle_a", "cycle_b", (0.9, 0.9, 0.2, 1.0)), ("cycle_b", "cycle_a", (0.9, 0.2, 0.9, 1.0))):
        m = new_model()
        quad_mesh(m, name, color)
        ref = external_reference(m, f"{other}.gltf")
        m["nodes"] = [{"name": name, "mesh": 0}, {"name": f"{name}_to_{other}", "externalAsset": ref}]
        m["scenes"] = [{"nodes": [0, 1]}]
        m["scene"] = 0
        write(out_dir, f"external/{name}.gltf", m)

    shared_images(out_dir)
    placement(out_dir)
    library(out_dir)
    shared_deformers(out_dir)


def single_scene(model: dict, roots) -> None:
    model["scenes"] = [{"nodes": list(roots)}]
    model["scene"] = 0


def textured_quad(model: dict, name: str, texture: int) -> None:
    """A material sampling `texture` on a quad, as the file's only node."""
    model["materials"] = [{"name": f"{name}_material", "pbrMetallicRoughness": {"baseColorTexture": {"index": texture}}}]
    model["meshes"] = [{"name": name, "primitives": [dict(quad_geometry(model), material=0)]}]
    model["nodes"] = [{"name": name, "mesh": 0}]
    single_scene(model, [0])


def shared_images(out_dir: Path) -> None:
    """Images of referenced files that read the same file are one scene image. kit/textured_a.gltf
    and kit/sub/textured_b.gltf both read kit/Textures/colormap.png, b through "../" and both from
    `source` and from an MSFT_texture_dds `source`, after a data: image of its own that stays
    separate. other/textured_c.gltf reads its own Textures/colormap.png, another file with the same
    relative name: not shared. Expected in the scene: 3 images, 5 textures."""
    base = out_dir / "external/shared_images"
    for folder, rgba in (("kit", (200, 30, 30, 255)), ("other", (30, 200, 30, 255))):
        (base / folder / "Textures").mkdir(parents=True, exist_ok=True)
        (base / folder / "Textures/colormap.png").write_bytes(png_1x1(rgba))

    a = new_model()
    a["images"] = [{"uri": "Textures/colormap.png"}]
    a["textures"] = [{"name": "a_tex", "source": 0}]
    textured_quad(a, "a", 0)
    write(out_dir, "external/shared_images/kit/textured_a.gltf", a)

    b = new_model()
    b["extensionsUsed"] = ["MSFT_texture_dds"]
    b["images"] = [{"uri": data_uri("image/png", png_1x1((30, 30, 200, 255)))}, {"uri": "../Textures/colormap.png"}]
    b["textures"] = [{"name": "b_tex", "source": 1},
                     {"name": "b_dds", "source": 0, "extensions": {"MSFT_texture_dds": {"source": 1}}},
                     {"name": "b_data", "source": 0}]
    textured_quad(b, "b", 1)
    b["materials"][0]["emissiveTexture"] = {"index": 2}
    b["materials"][0]["occlusionTexture"] = {"index": 0}
    write(out_dir, "external/shared_images/kit/sub/textured_b.gltf", b)

    c = new_model()
    c["images"] = [{"uri": "Textures/colormap.png"}]
    c["textures"] = [{"name": "c_tex", "source": 0}]
    textured_quad(c, "c", 0)
    write(out_dir, "external/shared_images/other/textured_c.gltf", c)

    m = new_model()
    refs = [external_reference(m, uri) for uri in ("kit/textured_a.gltf", "kit/sub/textured_b.gltf", "other/textured_c.gltf")]
    m["nodes"] = [{"name": f"textured_{i}", "externalAsset": ref, "translation": [1.5 * i, 0.0, 0.0]}
                  for i, ref in enumerate(refs)]
    single_scene(m, range(len(refs)))
    write(out_dir, "external/shared_images/root.gltf", m)


def placement(out_dir: Path) -> None:
    """How the loader places referenced content: static trees as instance sets, a file shared by two
    placing files loaded once, animated content as nodes, and the runtime rules. Quads are 1 x 1
    and 0.5 x 0.5, so geometry is countable."""
    # kit/piece.gltf: one node with two primitives, "piece_paint" and "piece_trim". The root has a
    # transform (an object offset), which the placement keeps.
    piece = new_model()
    piece["materials"] = [{"name": "piece_paint", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.1, 0.1, 1.0]}},
                          {"name": "piece_trim", "pbrMetallicRoughness": {"baseColorFactor": [0.5, 0.5, 0.5, 1.0]}}]
    a = quad_geometry(piece, 1.0)
    a["material"] = 0
    b = quad_geometry(piece, 0.5)
    b["material"] = 1
    piece["meshes"] = [{"name": "piece", "primitives": [a, b]}]
    piece["nodes"] = [{"name": "piece", "mesh": 0, "translation": [0.0, 0.5, 0.0]}]
    single_scene(piece, [0])
    write(out_dir, "placement/kit/piece.gltf", piece)

    # kit/shelf.gltf: a static two-level tree, a shelf at (0, 1, 0) holding a box 1 above it. The box
    # carries extras, which a self-contained save must keep.
    shelf = new_model()
    shelf["materials"] = [{"name": "shelf_material"}]
    shelf["meshes"] = [{"name": "shelf", "primitives": [dict(quad_geometry(shelf, 1.0), material=0)]},
                       {"name": "box", "primitives": [dict(quad_geometry(shelf, 0.5), material=0)]}]
    shelf["nodes"] = [{"name": "shelf", "mesh": 0, "translation": [0.0, 1.0, 0.0], "children": [1]},
                      {"name": "box", "mesh": 1, "translation": [0.0, 1.0, 0.0], "extras": {"sku": "box-1"}}]
    single_scene(shelf, [0])
    write(out_dir, "placement/kit/shelf.gltf", shelf)

    # static.gltf: the piece placed three times, the shelf twice (the first with a local child,
    # attached once and kept editable; the second scaled by 2). Every placement is static: an
    # instance set on the placing node.
    m = new_model()
    piece_ref = external_reference(m, "kit/piece.gltf")
    shelf_ref = external_reference(m, "kit/shelf.gltf")
    local = quad_mesh(m, "local", (0.2, 0.8, 0.2, 1.0))
    m["nodes"] = [{"name": f"piece_{i}", "externalAsset": piece_ref, "translation": [2.0 * i, 0.0, 0.0]} for i in range(3)]
    m["nodes"] += [{"name": "shelf_0", "externalAsset": shelf_ref, "translation": [0.0, 0.0, 5.0], "children": [5]},
                   {"name": "shelf_1", "externalAsset": shelf_ref, "translation": [4.0, 0.0, 5.0], "scale": [2.0, 2.0, 2.0]},
                   {"name": "local_child", "mesh": local, "translation": [0.0, 3.0, 0.0]}]
    single_scene(m, range(5))
    write(out_dir, "placement/static.gltf", m)

    # shared_kit.gltf places district_a and district_b, which both place kit/piece.gltf twice: the
    # piece is loaded once, so its two primitives are two RenderPrimitives, not four.
    for name, x in (("district_a", 0.0), ("district_b", 10.0)):
        d = new_model()
        ref = external_reference(d, "kit/piece.gltf")
        d["nodes"] = [{"name": f"piece_{i}", "externalAsset": ref, "translation": [x + 2.0 * i, 0.0, 0.0]} for i in range(2)]
        single_scene(d, [0, 1])
        write(out_dir, f"placement/{name}.gltf", d)
    m = new_model()
    a_ref = external_reference(m, "district_a.gltf")
    b_ref = external_reference(m, "district_b.gltf")
    m["nodes"] = [{"name": "district_a", "externalAsset": a_ref}, {"name": "district_b", "externalAsset": b_ref}]
    single_scene(m, [0, 1])
    write(out_dir, "placement/shared_kit.gltf", m)

    # animated.gltf: kit/walker.gltf moves its body from y = 0 to y = 2 over one second (one
    # channel), placed twice. Animated content is not an instance set: each placement is nodes with
    # a channel of its own.
    walker = new_model()
    walker["meshes"] = [{"name": "walker_body", "primitives": [quad_geometry(walker, 1.0)]}]
    walker["nodes"] = [{"name": "walker", "children": [1]}, {"name": "walker_body", "mesh": 0}]
    times = add_accessor(walker, [0.0, 1.0], 5126, "SCALAR", False)
    walker["accessors"][times].update({"min": [0.0], "max": [1.0]})
    moves = add_accessor(walker, [(0.0, 0.0, 0.0), (0.0, 2.0, 0.0)], 5126, "VEC3", False)
    for accessor in (times, moves):
        walker["bufferViews"][walker["accessors"][accessor]["bufferView"]].pop("target")
    walker["animations"] = [{"name": "walk", "samplers": [{"input": times, "output": moves}],
                             "channels": [{"sampler": 0, "target": {"node": 1, "path": "translation"}}]}]
    single_scene(walker, [0])
    write(out_dir, "placement/kit/walker.gltf", walker)

    m = new_model()
    walker_ref = external_reference(m, "kit/walker.gltf")
    m["nodes"] = [{"name": f"walker_{i}", "externalAsset": walker_ref, "translation": [2.0 * i, 0.0, 0.0]} for i in range(2)]
    single_scene(m, [0, 1])
    write(out_dir, "placement/animated.gltf", m)

    # pointer.gltf: the piece placed by node 0, after the scene's own quad (node 1), which is
    # animated through KHR_animation_pointer. A default save drops the placed content and must keep
    # that channel, still pointing at the quad.
    m = new_model()
    m["extensionsUsed"] = ["KHR_animation_pointer"]
    piece_ref = external_reference(m, "kit/piece.gltf")
    quad_mesh(m, "own", (0.8, 0.8, 0.2, 1.0))
    m["nodes"] = [{"name": "placed", "externalAsset": piece_ref, "translation": [0.0, 0.0, 3.0]},
                  {"name": "own", "mesh": 0}]
    times = add_accessor(m, [0.0, 1.0], 5126, "SCALAR", False)
    m["accessors"][times].update({"min": [0.0], "max": [1.0]})
    moves = add_accessor(m, [(0.0, 0.0, 0.0), (0.0, 2.0, 0.0)], 5126, "VEC3", False)
    for accessor in (times, moves):
        m["bufferViews"][m["accessors"][accessor]["bufferView"]].pop("target")
    m["animations"] = [{"name": "lift", "samplers": [{"input": times, "output": moves}],
                        "channels": [{"sampler": 0, "target": {"path": "pointer", "extensions": {
                            "KHR_animation_pointer": {"pointer": "/nodes/1/translation"}}}}]}]
    single_scene(m, [0, 1])
    write(out_dir, "placement/pointer.gltf", m)

    # runtime_rules.gltf: what can't be resolved is skipped, the load goes on. A missing file, a file
    # that places itself (each loaded copy draws its own quad once), and a chain of files nested
    # deeper than the loader goes. One valid placement remains.
    m = new_model()
    refs = [external_reference(m, uri) for uri in ("kit/missing.gltf", "self_placing.gltf", "deep/deep_0.gltf", "kit/piece.gltf")]
    m["nodes"] = [{"name": name, "externalAsset": ref} for name, ref in zip(("missing", "self", "deep", "valid"), refs)]
    single_scene(m, range(len(refs)))
    write(out_dir, "placement/runtime_rules.gltf", m)

    s = new_model()
    quad_mesh(s, "self_quad", (0.9, 0.9, 0.2, 1.0))
    ref = external_reference(s, "self_placing.gltf")
    s["nodes"] = [{"name": "self_quad", "mesh": 0}, {"name": "again", "externalAsset": ref, "translation": [1.0, 0.0, 0.0]}]
    single_scene(s, [0, 1])
    write(out_dir, "placement/self_placing.gltf", s)

    for level in range(DEEP_CHAIN):
        d = new_model()
        quad_mesh(d, f"deep_{level}", (0.5, 0.5, 0.5, 1.0))
        d["nodes"] = [{"name": f"deep_{level}", "mesh": 0}]
        if level + 1 < DEEP_CHAIN:
            d["nodes"].append({"name": "next", "externalAsset": external_reference(d, f"deep_{level + 1}.gltf")})
        single_scene(d, range(len(d["nodes"])))
        write(out_dir, f"placement/deep/deep_{level}.gltf", d)


DEEP_CHAIN = 20  # files in placement/deep; more than the renderer's nesting limit


def library(out_dir: Path) -> None:
    """A library of parts: one parts.bin that several files read, each through accessors of its own
    (the layout of a glTF 2.0 library: no file places another, they share bytes). Two files reading
    the same bytes in the same format draw the same geometry, so the loader shares it, except where
    a render primitive is per mesh or per skin (morph targets, skins).

    library.gltf places district_a, district_b (static parts and a morphed flag, the two declaring
    their accessors differently), rig_a and rig_b (a skinned quad each). Expected: parts.bin held
    once; 6 RenderPrimitives (the 2 static parts shared, the flag once per district, the rig once per
    file) for 12 render nodes. aliased.gltf (no external asset) has two primitives with their own
    accessors over one range: 2 RenderPrimitives, as before."""
    base = "placement/library"
    blob = bytearray()

    def put(data: bytes) -> int:
        offset = len(blob)
        blob.extend(data + b"\0" * (-len(data) % 4))
        return offset

    quads = {}
    for name, h in (("big", 0.5), ("small", 0.25)):
        quads[name] = {
            "POSITION": put(b"".join(struct.pack("<3f", *p) for p in [(-h, -h, 0.0), (h, -h, 0.0), (h, h, 0.0), (-h, h, 0.0)])),
            "NORMAL": put(struct.pack("<3f", 0.0, 0.0, 1.0) * 4),
            "indices": put(b"".join(struct.pack("<H", i) for i in [0, 1, 2, 0, 2, 3])),
            "h": h,
        }
    displacement = put(struct.pack("<3f", 0.0, 0.0, 0.5) * 4)
    joints = put(bytes([0, 0, 0, 0]) * 4)
    weights = put(struct.pack("<4f", 1.0, 0.0, 0.0, 0.0) * 4)
    (out_dir / base).mkdir(parents=True, exist_ok=True)
    (out_dir / base / "parts.bin").write_bytes(bytes(blob))
    print(f"  {base}/parts.bin")

    def with_parts(model: dict) -> None:
        model["buffers"] = [{"uri": "parts.bin", "byteLength": len(blob)}]

    def per_accessor_views(model: dict) -> dict:
        """One bufferView per accessor (district_a, the rigs). Returns name -> accessor maker."""
        views = model.setdefault("bufferViews", [])
        accessors = model.setdefault("accessors", [])

        def accessor(offset, length, component, type_, count, target, **extra):
            views.append({"buffer": 0, "byteOffset": offset, "byteLength": length, "target": target})
            accessors.append(dict({"bufferView": len(views) - 1, "componentType": component, "count": count, "type": type_}, **extra))
            return len(accessors) - 1
        return accessor

    def quad_primitive(accessor, q) -> dict:
        h = q["h"]
        return {"attributes": {"POSITION": accessor(q["POSITION"], 48, 5126, "VEC3", 4, 34962, min=[-h, -h, 0.0], max=[h, h, 0.0]),
                               "NORMAL": accessor(q["NORMAL"], 48, 5126, "VEC3", 4, 34962)},
                "indices": accessor(q["indices"], 12, 5123, "SCALAR", 6, 34963)}

    def district(name: str, shared_view: bool) -> None:
        d = new_model("2.0")
        with_parts(d)
        if shared_view:
            # district_b: every vec3 through one strided view, in another order, so the accessors,
            # their indices and their bufferViews all differ from district_a's; the bytes are the same.
            d["bufferViews"] = [{"buffer": 0, "byteOffset": 0, "byteLength": len(blob), "byteStride": 12, "target": 34962}]
            d["accessors"] = []

            def accessor(offset, length, component, type_, count, target, **extra):
                if target == 34962:
                    d["accessors"].append(dict({"bufferView": 0, "byteOffset": offset, "componentType": component, "count": count, "type": type_}, **extra))
                else:
                    d["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": length, "target": target})
                    d["accessors"].append({"bufferView": len(d["bufferViews"]) - 1, "componentType": component, "count": count, "type": type_})
                return len(d["accessors"]) - 1
            order = ("small", "big")
        else:
            accessor = per_accessor_views(d)
            order = ("big", "small")
        d["materials"] = [{"name": "part_material"}, {"name": "flag_material"}]
        prims = {key: dict(quad_primitive(accessor, quads[key]), material=0) for key in order}
        flag = dict(quad_primitive(accessor, quads["small"]), material=1)
        flag["targets"] = [{"POSITION": accessor(displacement, 48, 5126, "VEC3", 4, 34962, min=[0.0, 0.0, 0.5], max=[0.0, 0.0, 0.5])}]
        d["meshes"] = [{"name": "part", "primitives": [prims["big"], prims["small"]]},
                       {"name": "flag", "primitives": [flag], "weights": [1.0]}]
        d["nodes"] = [{"name": "part_0", "mesh": 0}, {"name": "part_1", "mesh": 0, "translation": [2.0, 0.0, 0.0]},
                      {"name": "flag", "mesh": 1, "translation": [0.0, 2.0, 0.0]}]
        single_scene(d, [0, 1, 2])
        write(out_dir, f"{base}/{name}.gltf", d)

    district("district_a", shared_view=False)
    district("district_b", shared_view=True)

    for name in ("rig_a", "rig_b"):
        r = new_model("2.0")
        with_parts(r)
        accessor = per_accessor_views(r)
        prim = quad_primitive(accessor, quads["big"])
        prim["attributes"]["JOINTS_0"] = accessor(joints, 16, 5121, "VEC4", 4, 34962)
        prim["attributes"]["WEIGHTS_0"] = accessor(weights, 64, 5126, "VEC4", 4, 34962)
        prim["material"] = 0
        r["materials"] = [{"name": f"{name}_material"}]
        r["meshes"] = [{"name": f"{name}_body", "primitives": [prim]}]
        r["skins"] = [{"joints": [1]}]
        r["nodes"] = [{"name": f"{name}_body", "mesh": 0, "skin": 0, "children": [1]}, {"name": f"{name}_joint"}]
        single_scene(r, [0])
        write(out_dir, f"{base}/{name}.gltf", r)

    m = new_model()
    m["nodes"] = []
    for i, name in enumerate(("district_a", "district_b", "rig_a", "rig_b")):
        m["nodes"].append({"name": name, "externalAsset": external_reference(m, f"{name}.gltf"), "translation": [0.0, 0.0, 4.0 * i]})
    single_scene(m, range(4))
    write(out_dir, f"{base}/library.gltf", m)

    # aliased.gltf: no external asset; two meshes whose accessors read the same range.
    a = new_model("2.0")
    with_parts(a)
    accessor = per_accessor_views(a)
    a["materials"] = [{"name": "part_material"}]
    a["meshes"] = [{"name": f"alias_{i}", "primitives": [dict(quad_primitive(accessor, quads["big"]), material=0)]} for i in range(2)]
    a["nodes"] = [{"name": "alias_0", "mesh": 0}, {"name": "alias_1", "mesh": 1, "translation": [2.0, 0.0, 0.0]}]
    single_scene(a, [0, 1])
    write(out_dir, f"{base}/aliased.gltf", a)


def shared_deformers(out_dir: Path) -> None:
    """One skinned or morphed mesh drawn by several nodes, each with a pose of its own. Skinning and
    morphing write a render primitive's vertices, so every node that deforms the mesh needs one.
    Quads are 1 x 1, centred on the origin of their node."""
    def skinned_quad(model: dict) -> dict:
        prim = quad_geometry(model)
        prim["attributes"]["JOINTS_0"] = add_accessor(model, [(0, 0, 0, 0)] * 4, 5121, "VEC4", True)
        prim["attributes"]["WEIGHTS_0"] = add_accessor(model, [(1.0, 0.0, 0.0, 0.0)] * 4, 5126, "VEC4", True)
        return prim

    # deform/two_skins.gltf: one skinned mesh, drawn by body_a (skin 0, its joint at (0, 1, 0)),
    # body_b (skin 1, its joint at (4, 3, 0)) and unskinned by still (at (8, 0, 0)). A skin without
    # inverse bind matrices puts the quad at its joint, wherever the drawing node is. Expected quad
    # centres: (0, 1, 0), (4, 3, 0), (8, 0, 0).
    m = new_model("2.0")
    m["meshes"] = [{"name": "skinned_quad", "primitives": [skinned_quad(m)]}]
    m["skins"] = [{"joints": [1]}, {"joints": [3]}]
    m["nodes"] = [{"name": "body_a", "mesh": 0, "skin": 0},
                  {"name": "joint_a", "translation": [0.0, 1.0, 0.0]},
                  {"name": "body_b", "mesh": 0, "skin": 1, "translation": [4.0, 0.0, 0.0]},
                  {"name": "joint_b", "translation": [4.0, 3.0, 0.0]},
                  {"name": "still", "mesh": 0, "translation": [8.0, 0.0, 0.0]}]
    single_scene(m, range(5))
    write(out_dir, "deform/two_skins.gltf", m)

    # deform/two_weights.gltf: one morphed mesh (one target moving the quad up by 1, mesh weight 0)
    # drawn by three nodes: rest (the mesh's weights), raised (node weight 1, animated from 1 to 0
    # over one second) and half (node weight 0.5). Expected quad centres at y = 0, 1, 0.5; at
    # t = 0.5 of the animation, raised is at 0.5 and the others don't move.
    m = new_model("2.0")
    prim = quad_geometry(m)
    prim["targets"] = [{"POSITION": add_accessor(m, [(0.0, 1.0, 0.0)] * 4, 5126, "VEC3", True)}]
    m["meshes"] = [{"name": "morphed_quad", "primitives": [prim], "weights": [0.0]}]
    m["nodes"] = [{"name": "rest", "mesh": 0},
                  {"name": "raised", "mesh": 0, "weights": [1.0], "translation": [2.0, 0.0, 0.0]},
                  {"name": "half", "mesh": 0, "weights": [0.5], "translation": [4.0, 0.0, 0.0]}]
    times = add_accessor(m, [0.0, 1.0], 5126, "SCALAR", False)
    m["accessors"][times].update({"min": [0.0], "max": [1.0]})
    values = add_accessor(m, [1.0, 0.0], 5126, "SCALAR", False)
    for accessor in (times, values):
        m["bufferViews"][m["accessors"][accessor]["bufferView"]].pop("target")
    m["animations"] = [{"name": "lower", "samplers": [{"input": times, "output": values}],
                        "channels": [{"sampler": 0, "target": {"node": 1, "path": "weights"}}]}]
    single_scene(m, range(3))
    write(out_dir, "deform/two_weights.gltf", m)

    # deform/mesh_weights.gltf: three meshes over the same morphed primitive (same accessors): zero
    # (weight 0), one (weight 1) and unweighted (no weights: 0 by default), drawn by one node each.
    # unweighted's node is animated from weight 0 to 1 over one second. Expected quad centres at
    # y = 0, 1, 0; at t = 0.5 of the animation, unweighted is at 0.5.
    m = new_model("2.0")
    prim = quad_geometry(m)
    prim["targets"] = [{"POSITION": add_accessor(m, [(0.0, 1.0, 0.0)] * 4, 5126, "VEC3", True)}]
    m["meshes"] = [{"name": "zero", "primitives": [prim], "weights": [0.0]},
                   {"name": "one", "primitives": [prim], "weights": [1.0]},
                   {"name": "unweighted", "primitives": [prim]}]
    m["nodes"] = [{"name": name, "mesh": i, "translation": [2.0 * i, 0.0, 0.0]} for i, name in enumerate(("zero", "one", "unweighted"))]
    times = add_accessor(m, [0.0, 1.0], 5126, "SCALAR", False)
    m["accessors"][times].update({"min": [0.0], "max": [1.0]})
    values = add_accessor(m, [0.0, 1.0], 5126, "SCALAR", False)
    for accessor in (times, values):
        m["bufferViews"][m["accessors"][accessor]["bufferView"]].pop("target")
    m["animations"] = [{"name": "raise", "samplers": [{"input": times, "output": values}],
                        "channels": [{"sampler": 0, "target": {"node": 2, "path": "weights"}}]}]
    single_scene(m, range(3))
    write(out_dir, "deform/mesh_weights.gltf", m)

    # deform/two_targets.gltf: two meshes over the same base accessors, each with a morph target of
    # its own (up by 1, up by 2) at weight 1. Different targets are different geometry: two render
    # primitives. Expected quad centres at y = 1 and y = 2.
    m = new_model("2.0")
    base = quad_geometry(m)
    m["meshes"] = [{"name": f"up_{d}", "weights": [1.0],
                    "primitives": [dict(base, targets=[{"POSITION": add_accessor(m, [(0.0, float(d), 0.0)] * 4, 5126, "VEC3", True)}])]}
                   for d in (1, 2)]
    m["nodes"] = [{"name": f"up_{d}", "mesh": i, "translation": [2.0 * i, 0.0, 0.0]} for i, d in enumerate((1, 2))]
    single_scene(m, range(2))
    write(out_dir, "deform/two_targets.gltf", m)

    # placement/skinned.gltf: kit/rig.gltf (a skinned quad and its joint, at (0, 1, 0) in the file)
    # placed twice. The placements share the file's mesh; each has nodes and a skin of its own.
    rig = new_model()
    rig["meshes"] = [{"name": "rig_quad", "primitives": [skinned_quad(rig)]}]
    rig["skins"] = [{"joints": [2]}]
    rig["nodes"] = [{"name": "rig", "children": [1, 2]}, {"name": "rig_body", "mesh": 0, "skin": 0},
                    {"name": "rig_joint", "translation": [0.0, 1.0, 0.0]}]
    single_scene(rig, [0])
    write(out_dir, "placement/kit/rig.gltf", rig)

    m = new_model()
    rig_ref = external_reference(m, "kit/rig.gltf")
    m["nodes"] = [{"name": f"rig_{i}", "externalAsset": rig_ref, "translation": [4.0 * i, 0.0, 0.0]} for i in range(2)]
    single_scene(m, [0, 1])
    write(out_dir, "placement/skinned.gltf", m)


def versions(out_dir: Path) -> None:
    """#26: which versions load, and which version a save writes. Each file is a single quad; only the
    asset block (and, for the newer minor version, some unknown content) differs."""

    def quad_scene(version: str, min_version: str | None = None) -> dict:
        m = new_model(version)
        if min_version:
            m["asset"]["minVersion"] = min_version
        quad_mesh(m, "quad", (0.6, 0.6, 0.6, 1.0))
        m["nodes"] = [{"name": "quad", "mesh": 0}]
        m["scenes"] = [{"nodes": [0]}]
        m["scene"] = 0
        return m

    # Loads: no 2.1 feature, so a save writes "2.0".
    write(out_dir, "version/plain_2_1.gltf", quad_scene("2.1"))
    # Loads: minVersion asks for 2.1, so a save keeps "2.1" and the minVersion.
    write(out_dir, "version/min_version_2_1.gltf", quad_scene("2.1", "2.1"))

    # Loads: a newer minor version must load, ignoring what we don't know (spec "Versioning").
    future = quad_scene("2.7")
    future["futureCollection"] = [{"name": "unknown to 2.1"}]
    future["nodes"][0]["futureProperty"] = {"value": 1}
    write(out_dir, "version/future_minor.gltf", future)

    # Refused: another major version.
    write(out_dir, "version/future_major.gltf", quad_scene("3.0"))
    # Refused: minVersion above what we support.
    write(out_dir, "version/min_version_too_new.gltf", quad_scene("2.7", "2.2"))
    # Refused: contradictory, minVersion must not exceed version.
    write(out_dir, "version/min_version_above_version.gltf", quad_scene("2.0", "2.1"))
    # Refused: not a valid version string (leading zero).
    write(out_dir, "version/invalid_version.gltf", quad_scene("2.00"))


# Component type -> (struct format, byte size). glTF 2.1 adds int32 (5124), float64 (5130),
# float16 (5131), int64 (5134) and uint64 (5135).
COMPONENT_FORMATS = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5124: ("i", 4),
                     5125: ("I", 4), 5126: ("f", 4), 5130: ("d", 8), 5131: ("e", 2), 5134: ("q", 8),
                     5135: ("Q", 8)}
TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def add_accessor(model: dict, values: list, component_type: int, type_: str, vertex: bool) -> int:
    """Append one accessor (own buffer + bufferView) holding `values` (a list of tuples, or of
    scalars for SCALAR) encoded as `component_type`. Vertex attributes get a 4-byte aligned stride, as
    the spec requires."""
    fmt, size = COMPONENT_FORMATS[component_type]
    n = TYPE_COMPONENTS[type_]
    element = n * size
    stride = (element + 3) // 4 * 4 if vertex else element
    blob = b""
    for v in values:
        comps = v if isinstance(v, tuple) else (v,)
        cast = int if fmt not in "fde" else float
        packed = struct.pack("<" + fmt * n, *(cast(c) for c in comps))
        blob += packed + b"\0" * (stride - element)
    blob += b"\0" * (-len(blob) % 4)

    model.setdefault("buffers", []).append({"byteLength": len(blob), "uri": data_uri("application/octet-stream", blob)})
    view = {"buffer": len(model["buffers"]) - 1, "byteLength": len(blob), "target": 34962 if vertex else 34963}
    if vertex and stride != element:
        view["byteStride"] = stride
    model.setdefault("bufferViews", []).append(view)
    accessor = {"bufferView": len(model["bufferViews"]) - 1, "componentType": component_type,
                "count": len(values), "type": type_}
    if type_ == "VEC3" and vertex:
        accessor["min"] = [min(v[i] for v in values) for i in range(3)]
        accessor["max"] = [max(v[i] for v in values) for i in range(3)]
    model.setdefault("accessors", []).append(accessor)
    return len(model["accessors"]) - 1


def typed_quad(position=5126, normal=5126, texcoord=5126, indices=5123, color=None, extra=None,
               normal_map=False, index_type="SCALAR") -> dict:
    """A one-quad scene whose accessors use the given component types. `indices=None` writes the two
    triangles unindexed; `color` adds a VEC4 COLOR_0 of that type; `extra` maps additional attribute
    names to a component type (scalar per vertex, value = vertex index); `normal_map` gives the
    material a normal texture, which makes the renderer generate tangents."""
    m = new_model()
    corners = [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)]
    uvs = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    order = [0, 1, 2, 0, 2, 3] if indices is None else [0, 1, 2, 3]
    attributes = {
        "POSITION": add_accessor(m, [corners[i] for i in order], position, "VEC3", True),
        "NORMAL": add_accessor(m, [(0.0, 0.0, 1.0)] * len(order), normal, "VEC3", True),
        "TEXCOORD_0": add_accessor(m, [uvs[i] for i in order], texcoord, "VEC2", True),
    }
    if color is not None:
        attributes["COLOR_0"] = add_accessor(m, [(1, 1, 1, 1)] * len(order), color, "VEC4", True)
    for name, component_type in (extra or {}).items():
        attributes[name] = add_accessor(m, list(range(len(order))), component_type, "SCALAR", True)
    material = {"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1.0]}}
    if normal_map:
        m["images"] = [{"mimeType": "image/png", "uri": data_uri("image/png", png_1x1((128, 128, 255, 255)))}]
        m["textures"] = [{"source": 0}]
        material["normalTexture"] = {"index": 0}
    m["materials"] = [material]
    primitive = {"attributes": attributes, "material": 0}
    if indices is not None:
        tris = [0, 1, 2, 0, 2, 3] if index_type == "SCALAR" else [(0, 1, 2), (0, 2, 3)]
        primitive["indices"] = add_accessor(m, tris, indices, index_type, False)
    m["meshes"] = [{"primitives": [primitive]}]
    m["nodes"] = [{"name": "quad", "mesh": 0}]
    m["scenes"] = [{"nodes": [0]}]
    m["scene"] = 0
    return m


def component_types(out_dir: Path) -> None:
    """#29/#30: the new accessor component types, where core allows them and where it doesn't. Loading
    must never crash or read garbage: supported uses render, unsupported ones are skipped with a
    warning."""
    # PROVISIONAL(Khronos MR 6): core keeps each semantic's 2.0 component types; extensions opt in.
    cases = {
        # Not allowed by core for these semantics.
        "position_float16": dict(position=5131),
        "position_float64": dict(position=5130),
        "position_int32": dict(position=5124),
        "normal_float16": dict(normal=5131),
        "texcoord_float16": dict(texcoord=5131),
        "texcoord_int64": dict(texcoord=5134),
        "indices_int32": dict(indices=5124),
        "indices_uint64": dict(indices=5135),
        "indices_vec3": dict(indices=5125, index_type="VEC3"),  # allowed component type, but not SCALAR
        # Allowed in 2.1 (#30): 32-bit integer application-specific attributes.
        "feature_id_uint32": dict(extra={"_FEATURE_ID_0": 5125}),
        "feature_id_int32": dict(extra={"_FEATURE_ID_0": 5124}),
        # Allowed accessor types in a custom attribute the renderer doesn't use.
        "custom_uint64": dict(extra={"_CUSTOM": 5135}),
        # An optional attribute we can't read is ignored; the quad still renders.
        "color_int32": dict(color=5124),
        # Not 2.1-specific: tangents generated for a non-indexed primitive with a normal map.
        "unindexed_normal_map": dict(indices=None, normal_map=True),
    }
    for name, kwargs in cases.items():
        write(out_dir, f"component_types/{name}.gltf", typed_quad(**kwargs))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    out_dir = Path(sys.argv[1]).resolve()
    print(f"Writing glTF 2.1 fixtures to {out_dir}")
    core_properties(out_dir)
    external_assets(out_dir)
    versions(out_dir)
    component_types(out_dir)
    (out_dir / STAMP).write_text("", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
