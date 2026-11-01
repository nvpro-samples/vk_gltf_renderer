"""Merge a leaf by material: what an asset draws itself becomes one node and one mesh, with one
primitive per (material, texture coordinates, shadow casting), its USD meshes moved into the
asset's frame and concatenated. A large scene places hundreds of thousands of assets made of a few
small meshes each: one mesh per asset is what keeps loading (vertex buffers, BLAS builds) and the
scene graph proportional to the assets, not to their parts.

Each merged primitive names its parts in ``extras.usd_parts``: [[USD mesh name, first index, index
count], ...], contiguous and in order, so a pick or a look rule can still tell them apart.

Kept as their own nodes, directly under the leaf's node with their transform flattened: meshes
drawn with EXT_mesh_gpu_instancing, meshes placed several times when their copies would be large,
cameras and lights. A leaf with skins or animations is left as it is."""

from __future__ import annotations

from collections import Counter

import numpy as np

COPY_LIMIT = 65_536  # vertices: a mesh placed several times is copied into the merge up to this, all copies counted

_COMPONENTS = {5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _node_matrix(node: dict) -> np.ndarray:
    return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T if "matrix" in node else np.eye(4)


def _flat(m: np.ndarray) -> list[float] | None:
    return None if np.allclose(m, np.eye(4)) else [float(x) for x in m.T.flatten()]


class _Reader:
    """The document's accessors as they were before the merge rewrites them."""

    def __init__(self, doc: dict, blob: bytes):
        self.accessors, self.views, self.blob = doc["accessors"], doc["bufferViews"], blob

    def __call__(self, accessor: int) -> np.ndarray:
        acc = self.accessors[accessor]
        view = self.views[acc["bufferView"]]
        n = _WIDTH[acc["type"]]
        a = np.frombuffer(self.blob, _COMPONENTS[acc["componentType"]], acc["count"] * n,
                          view["byteOffset"] + acc.get("byteOffset", 0))
        return a if n == 1 else a.reshape(-1, n)


def merge_leaf(gltf, name: str) -> Counter:
    """Merge the converted, not yet written document of ``gltf`` (a ``convert.Gltf``) in place; its
    one node is called ``name``. Returns what it did."""
    doc, stats = gltf.doc, Counter()
    if not doc["meshes"] or doc.get("skins") or doc.get("animations"):
        return stats
    read = _Reader(doc, b"".join(gltf.chunks))
    nodes, meshes = doc["nodes"], doc["meshes"]
    uses = Counter(n["mesh"] for n in nodes if "mesh" in n)
    vertices = [sum(doc["accessors"][p["attributes"]["POSITION"]]["count"] for p in m["primitives"]) for m in meshes]

    baked: list[tuple[int, np.ndarray, bool]] = []  # (mesh, world, casts shadows)
    kept: list[dict] = []  # nodes of their own, under the leaf's node

    def visit(index: int, parent: np.ndarray) -> None:
        node = nodes[index]
        world = parent @ _node_matrix(node)
        ext = node.get("extensions", {})
        if "mesh" in node:
            m = node["mesh"]
            if "EXT_mesh_gpu_instancing" not in ext and (uses[m] == 1 or vertices[m] * uses[m] <= COPY_LIMIT):
                shadows = ext.get("NV_node_shadow_casting", {}).get("castShadows", True)
                baked.append((m, world, shadows))
            else:
                kept.append({"name": node["name"], "mesh": m, **({"extensions": ext} if ext else {})})
                kept[-1]["matrix"] = _flat(world)
        elif "camera" in node or "KHR_lights_punctual" in ext:
            kept.append({k: v for k, v in node.items() if k not in ("children", "matrix")})
            kept[-1]["matrix"] = _flat(world)
        for child in node.get("children", []):
            visit(child, world)

    for root in doc["scenes"][0]["nodes"]:
        visit(root, np.eye(4))
    if len(baked) < 2 and not kept:  # one mesh at most: nothing to gain
        return stats

    # Group the baked primitives.
    groups: dict[tuple, list] = {}
    for m, world, shadows in baked:
        for prim in meshes[m]["primitives"]:
            key = (prim.get("material"), "TEXCOORD_0" in prim["attributes"], shadows)
            groups.setdefault(key, []).append((meshes[m]["name"], prim, world))

    # Every accessor is written again: kept meshes and instancing data copied, merged ones new.
    old_accessors = doc["accessors"]
    gltf.chunks, gltf.offset, doc["bufferViews"], doc["accessors"] = [], 0, [], []

    def copy(accessor: int) -> int:
        acc = old_accessors[accessor]
        arr = read(accessor)
        if acc["type"] == "SCALAR":
            return gltf.accessor(arr.astype(np.int64), "SCALAR")
        return gltf.accessor(arr, acc["type"], minmax="min" in acc)

    new_meshes, mesh_remap = [], {}
    for node in kept:
        if "mesh" in node:
            m = node["mesh"]
            if m not in mesh_remap:
                prims = [{**p, "attributes": {k: copy(a) for k, a in p["attributes"].items()},
                          **({"indices": copy(p["indices"])} if "indices" in p else {})} for p in meshes[m]["primitives"]]
                new_meshes.append({**meshes[m], "primitives": prims})
                mesh_remap[m] = len(new_meshes) - 1
            node["mesh"] = mesh_remap[m]
        inst = node.get("extensions", {}).get("EXT_mesh_gpu_instancing")
        if inst:
            inst["attributes"] = {k: copy(a) for k, a in inst["attributes"].items()}

    merged = {True: [], False: []}
    for (material, uv, shadows), parts in groups.items():
        pos, nrm, tex, idx, names, base, first = [], [], [], [], [], 0, 0
        for mesh_name, prim, world in parts:
            a = prim["attributes"]
            lin, t = world[:3, :3], world[:3, 3]
            p = read(a["POSITION"]).astype(np.float64) @ lin.T + t
            n = read(a["NORMAL"]).astype(np.float64) @ np.linalg.inv(lin)  # inverse transpose, row vectors
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
            i = read(prim["indices"]).astype(np.int64).reshape(-1, 3)
            if np.linalg.det(lin) < 0:  # a mirrored placement turns the triangles around
                i = i[:, [0, 2, 1]]
            pos.append(p), nrm.append(n), idx.append(i.reshape(-1) + base)
            if uv:
                tex.append(read(a["TEXCOORD_0"]))
            names.append([mesh_name, first, int(i.size)])
            base, first = base + len(p), first + int(i.size)
        attrs = {"POSITION": gltf.accessor(np.concatenate(pos), "VEC3", minmax=True),
                 "NORMAL": gltf.accessor(np.concatenate(nrm), "VEC3")}
        if uv:
            attrs["TEXCOORD_0"] = gltf.accessor(np.concatenate(tex), "VEC2")
        prim = {"attributes": attrs, "indices": gltf.accessor(np.concatenate(idx), "SCALAR"),
                "extras": {"usd_parts": names}}
        if material is not None:
            prim["material"] = material
        merged[shadows].append(prim)
        stats["primitives merged"] += len(parts)
        stats["primitives written"] += 1

    # Materials no primitive uses any more (a displayColor fallback) are dropped.
    all_prims = [p for m in new_meshes for p in m["primitives"]] + merged[True] + merged[False]
    used = sorted({p["material"] for p in all_prims if "material" in p})
    remap = {old: new for new, old in enumerate(used)}
    for p in all_prims:
        if "material" in p:
            p["material"] = remap[p["material"]]
    stats["materials dropped"] += len(doc["materials"]) - len(used)
    doc["materials"] = [doc["materials"][i] for i in used]

    root: dict = {"name": name}
    children = []
    if merged[True]:
        new_meshes.append({"name": name, "primitives": merged[True]})
        root["mesh"] = len(new_meshes) - 1
    if merged[False]:
        new_meshes.append({"name": f"{name}_no_shadows", "primitives": merged[False]})
        shade = {"NV_node_shadow_casting": {"castShadows": False}}
        if "mesh" in root:
            children.append({"name": f"{name}_no_shadows", "mesh": len(new_meshes) - 1, "extensions": shade})
        else:
            root.update({"mesh": len(new_meshes) - 1, "extensions": shade})
    children += kept
    stats["nodes before"] += len(nodes)
    doc["meshes"], doc["nodes"] = new_meshes, [root]
    for child in children:
        if child.get("matrix") is None:
            child.pop("matrix", None)
        doc["nodes"].append(child)
    if children:
        root["children"] = list(range(1, len(doc["nodes"])))
    doc["scenes"][0]["nodes"] = [0]
    stats["nodes after"] += len(doc["nodes"])
    return stats
