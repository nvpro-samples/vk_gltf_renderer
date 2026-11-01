"""Assemble: write a unit as a glTF 2.1 file that places the converted leaves (core ``files``,
``externalAssets`` and ``node.externalAsset``), from the scan records alone: no USD is opened.

- One node per placement, with the placement's transform. A unit that only draws itself is placed
  straight from its leaf; one that places others is written as group nodes (``--nesting inline``),
  or as its own assembly file (``--nesting files``).
- Switches (a variant set of alternatives: levels of detail, a full model and its proxy): the option
  ``--variant`` names is placed, else the one ``--lod`` picks: by triangle count, by what a camera
  sees (``camera``: per placement, see ``LodView``), or the authored selection. Every option is an ``externalAssets`` entry, and each lists them all, most triangles
  first, in ``EXT_external_asset_lod``. No node references the others, so a loader that doesn't
  know the extension draws the picked one; a level-of-detail tool rewrites ``node.externalAsset``.
- Bounding volumes: every node gets a core ``boundingVolume`` (a box shape) enclosing everything
  under it, in its own space, so the tree is a bounding volume hierarchy of the scene.
- ``--hierarchy compact`` (``auto``: on large scenes) writes fewer, smaller nodes: a group with one
  child is folded into it (the outer, placing prim's name is kept), transforms are translation,
  rotation and scale, bounding volumes stay on switches and the document's roots only, and a
  placement repeated at the same place under the same parent is written once.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from collections import Counter
from pathlib import Path

import numpy as np

from .leaf import GENERATOR, corners

# PROVISIONAL(EXT_external_asset_lod draft): working name and form of the level-of-detail chain;
# nothing is specified yet. Same prefix question as EXT_external_asset_placement.
LOD_EXTENSION = "EXT_external_asset_lod"
GLTF_MIME = "model/gltf+json"  # PROVISIONAL(Khronos MR 18): media type of a referenced .gltf file


# GPU memory per unique triangle placed: vertices, indices and compacted BLAS, measured in
# vk_gltf_renderer (2026-10: full Caldera and Da Vinci's workshop, about 80 bytes; a little kept
# spare). Textures are not counted. Too high, and "auto" texture sizes shrink images for nothing.
BYTES_PER_TRIANGLE = 90


def check_lod(policy: str) -> str:
    """``authored``, ``finest``, ``coarsest``, ``levelN`` (the N-th option by triangle count,
    0 = most triangles; the coarsest when there are fewer) or ``camera`` (``LodView``)."""
    if policy in ("authored", "finest", "coarsest", "camera") or re.fullmatch(r"level\d+", policy):
        return policy
    raise ValueError(f"unknown level-of-detail policy {policy!r}")


def geo_leaf(unit_id: str) -> str:
    return f"units/{unit_id}/geo"


def assembly_file(unit_id: str, lod: str, look: str | None, nesting: str, variants: dict[str, dict[str, str]],
                  compact: bool = False) -> str:
    """Assemblies made with other options sit side by side: the defaults give ``assembly``, others
    add a tag each (``assembly.coarsest.caldera.files.districtLod-proxy``). Choices for named
    switches add their count and a hash (``districtLod-full-2-1a2b``)."""
    tags = [t for t in (lod if lod != "authored" else None, look, nesting if nesting != "inline" else None,
                        "compact" if compact else None) if t]
    for name, choices in sorted(variants.items()):
        if "*" in choices:
            tags.append(f"{name}-{choices['*']}")
        named = sorted((k, v) for k, v in choices.items() if k != "*")
        for value in sorted({v for _, v in named}):
            names = [k for k, v in named if v == value]
            tags.append(f"{name}-{value}-{len(names)}-{hashlib.sha1(','.join(names).encode()).hexdigest()[:4]}")
    return ".".join([f"units/{unit_id}/assembly", *tags])


def short_name(unit_id: str) -> str:
    """A readable name for a unit: its layer's file name without extensions, and its selections."""
    layer = re.split(r"\.usd[ac]?(?=/|$)", unit_id, maxsplit=1)[0]
    name = layer.rsplit("/", 1)[-1].split(".")[0]
    sel = re.search(r"\{.*\}$", unit_id)
    return name + (sel.group(0) if sel else "")


def transform_box(matrix: list[float] | None, box):
    if box is None or matrix is None:
        return box
    m = np.array(matrix, dtype=np.float64).reshape(4, 4).T
    pts = corners(*box) @ m[:3, :3].T + m[:3, 3]
    return [pts.min(0).tolist(), pts.max(0).tolist()]


def union(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return [np.min([b[0] for b in boxes], 0).tolist(), np.max([b[1] for b in boxes], 0).tolist()]


class LodView:
    """Levels of detail by what cameras see, the same rule for every scene: an option's projected
    size in pixels (its box's diagonal over its distance to the nearest eye, for a vertical field of
    view ``fov`` degrees and ``pixels`` rows) asks for ``density`` triangles per square pixel; the
    coarsest option with that many is placed, else the finest. Distances are in the scene's metres."""

    def __init__(self, eyes: list, fov: float = 60.0, pixels: int = 1080, density: float = 1.0):
        self.eyes = [np.asarray(e, dtype=np.float64) for e in eyes]
        self.tan_half, self.pixels, self.density = np.tan(np.radians(fov) / 2), pixels, density

    def target(self, box) -> float:
        """Triangles wanted for something in ``box`` (scene space)."""
        lo, hi = np.asarray(box[0]), np.asarray(box[1])
        size = float(np.linalg.norm(hi - lo))
        distance = min(float(np.linalg.norm(np.maximum(np.maximum(lo - e, 0.0), e - hi))) for e in self.eyes)
        # Within half its size of the box (or inside it), an object fills the view.
        px = size / max(distance, size / 2, 1e-6) / self.tan_half * self.pixels / 2
        return self.density * min(px, 4 * self.pixels) ** 2


SWITCH_MARK = "_switch"  # set while assembling on the nodes that place a switch's option


def _matrix(node: dict) -> np.ndarray:
    return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T if "matrix" in node else np.eye(4)


def _set_transform(node: dict, m: np.ndarray) -> None:
    """``m`` as translation, rotation and scale when it is one (rounded to what a scene needs: 10 um,
    1e-7 of a unit quaternion), else as a matrix; nothing for the identity."""
    for key in ("matrix", "translation", "rotation", "scale"):
        node.pop(key, None)
    basis = m[:3, :3]
    scale = np.linalg.norm(basis, axis=0)
    if np.linalg.det(basis) < 0:
        scale[0] = -scale[0]
    rot = basis / np.where(np.abs(scale) > 1e-12, scale, 1.0)
    if np.abs(m[3] - [0, 0, 0, 1]).max() > 1e-9 or np.abs(rot.T @ rot - np.eye(3)).max() > 1e-5:
        node["matrix"] = [float(x) for x in m.T.reshape(-1)]  # sheared or projective: kept as is
        return
    t = [round(float(x), 5) for x in m[:3, 3]]
    if any(t):
        node["translation"] = t
    q = _quaternion(rot)
    if abs(q[3]) < 1.0 - 1e-12:
        node["rotation"] = [round(float(x), 7) for x in q]
    s = [round(float(x), 7) for x in scale]
    if s != [1.0, 1.0, 1.0]:
        node["scale"] = s


def _quaternion(r: np.ndarray) -> list[float]:
    """Unit quaternion (x, y, z, w) of a rotation matrix."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        k = 0.5 / np.sqrt(trace + 1.0)
        q = np.array([(r[2, 1] - r[1, 2]) * k, (r[0, 2] - r[2, 0]) * k, (r[1, 0] - r[0, 1]) * k, 0.25 / k])
    else:
        i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        f = np.sqrt(r[i, i] - r[j, j] - r[k, k] + 1.0) * 2
        q = np.zeros(4)
        q[i], q[j], q[k], q[3] = 0.25 * f, (r[j, i] + r[i, j]) / f, (r[k, i] + r[i, k]) / f, (r[k, j] - r[j, k]) / f
    q /= np.linalg.norm(q)
    return (q if q[3] >= 0 else -q).tolist()


def compact_document(doc: dict) -> Counter:
    """Fewer, smaller nodes for the same scene (see ``--hierarchy`` above), in place."""
    nodes, stats = doc["nodes"], Counter()
    out: list[dict] = []

    def foldable(node: dict) -> bool:  # a plain group of one: only a name, a transform and a child
        return set(node) <= {"name", "matrix", "children", "boundingVolume"} and len(node.get("children", ())) == 1

    def emit(index: int, root: bool) -> int:
        node = dict(nodes[index])
        m = _matrix(node)
        while foldable(node):
            child = nodes[node["children"][0]]
            if child.get(SWITCH_MARK) or "camera" in child:
                break  # its bounding volume or its camera's frame is in its own space
            m = m @ _matrix(child)
            # The node now has the child's space, where the child's bounding volume is.
            node = {"name": node.get("name", child.get("name")),
                    **{k: v for k, v in child.items() if k not in ("name", "matrix")}}
            stats["nodes folded"] += 1
        if not (root or node.get(SWITCH_MARK)):
            node.pop("boundingVolume", None)
        if "children" in node:
            seen, children = set(), []
            for c in node["children"]:
                child = nodes[c]
                if "externalAsset" in child and "children" not in child:
                    key = (child["externalAsset"], tuple(child.get("matrix", ())))
                    if key in seen:
                        stats["duplicate placements dropped"] += 1
                        continue
                    seen.add(key)
                children.append(emit(c, False))
            node["children"] = children
        _set_transform(node, m)
        out.append(node)
        return len(out) - 1

    scene = doc["scenes"][0]
    scene["nodes"] = [emit(r, True) for r in scene["nodes"]]
    stats["nodes before"], stats["nodes after"] = len(nodes), len(out)
    doc["nodes"] = out
    used = sorted({n["boundingVolume"]["shape"] for n in out if "boundingVolume" in n})
    if len(used) < len(doc.get("shapes", ())):  # only the shapes still used, renumbered
        remap = {old: new for new, old in enumerate(used)}
        doc["shapes"] = [doc["shapes"][old] for old in used]
        for n in out:
            if "boundingVolume" in n:
                n["boundingVolume"]["shape"] = remap[n["boundingVolume"]["shape"]]
    return stats


class _Doc:
    """One glTF 2.1 assembly being written."""

    def __init__(self, path: Path, extras: dict, compact: bool = False):
        self.path, self.compact = path, compact
        self.doc = {"asset": {"version": "2.1", "generator": GENERATOR, "extras": {"usd_world": extras}},
                    "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "shapes": [], "files": [], "externalAssets": []}
        self._assets: dict[Path, int] = {}
        self._shapes: dict[tuple, int] = {}
        self.extensions: set[str] = set()

    def external(self, file: Path, name: str) -> int:
        if file not in self._assets:
            uri = Path(os.path.relpath(file, self.path.parent)).as_posix()
            self.doc["files"].append({"uri": urllib.parse.quote(uri), "mimeType": GLTF_MIME})
            self.doc["externalAssets"].append({"name": name, "file": len(self.doc["files"]) - 1})
            self._assets[file] = len(self.doc["externalAssets"]) - 1
        return self._assets[file]

    def bounding_volume(self, box) -> dict:
        lo, hi = np.array(box[0]), np.array(box[1])
        size = tuple(round(max(float(x), 1e-4), 4) for x in hi - lo)
        if size not in self._shapes:
            self.doc["shapes"].append({"type": "box", "box": {"size": list(size)}})
            self._shapes[size] = len(self.doc["shapes"]) - 1
        bv = {"shape": self._shapes[size]}
        center = (lo + hi) / 2
        if np.any(np.abs(center) > 1e-6):
            bv["translation"] = [round(float(x), 5) for x in center]
        return bv

    def add(self, node: dict) -> int:
        self.doc["nodes"].append(node)
        return len(self.doc["nodes"]) - 1

    def write(self) -> bool:
        """Write when the content changed; True if written."""
        if self.compact:
            self.stats = compact_document(self.doc)
        for node in self.doc["nodes"]:
            node.pop(SWITCH_MARK, None)
        for key in ("shapes", "files", "externalAssets"):
            if not self.doc[key]:
                del self.doc[key]
        if self.extensions:
            self.doc["extensionsUsed"] = sorted(self.extensions)
        text = json.dumps(self.doc, separators=(",", ":"))  # generated, and large: no indentation
        if self.path.exists() and self.path.read_text(encoding="utf-8") == text:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text, encoding="utf-8")
        return True


class _DryDoc(_Doc):
    """A document a dry run assembles: nothing is kept but what is counted (Caldera's levels are
    millions of placements, assembled a dozen times by the budget's search)."""

    def external(self, file: Path, name: str) -> int:
        return 0

    def bounding_volume(self, box) -> None:
        return None

    def add(self, node: dict) -> int:
        return 0

    def write(self) -> bool:
        return False


class Assembler:
    def __init__(self, cache, units: dict, lod: str, nesting: str, look: str | None,
                 variants: dict[str, dict[str, str]], compact: bool = False, exclude: list[str] = ()):
        """``exclude``: regexes; a placement whose prim name starts with a match is left out (the
        look's ``exclude``: a game's occluder volumes and editor markers)."""
        self.cache, self.units, self.compact = cache, units, compact
        self.exclude = re.compile("|".join(f"(?:{p})" for p in exclude), re.IGNORECASE) if exclude else None
        self.lod, self.nesting, self.look, self.variants = check_lod(lod), nesting, look, variants
        self.tick = None  # called now and then while assembling: the job's progress (and its Stop)
        self._ticks = 0
        self._all_leaves: dict[str, dict] | None = None  # leaf id -> record, read once (see leaf())
        self._leaf_paths: dict[str, Path] = {}
        self.written: dict[str, tuple[Path | None, list | None]] = {}  # unit -> (assembly or None if empty, box)
        self.written_drawn: dict[str, int] = {}  # unit -> triangles its assembly draws
        self._triangles: dict[str, int] = {}
        self.top: str | None = None  # the unit whose cameras and environment the scene gets
        self.top_path: Path | None = None  # where the scene is written (default: with the unit's other assemblies)
        self.view: LodView | None = None  # for lod "camera"
        self.dry = False  # plan only: nothing is written (the memory budget's search)
        self.file_triangles: dict[Path, int] = {}  # leaf file -> its triangles
        self.files: set[Path] = set()  # the leaf files the document being written reaches
        self.written_files: dict[str, set[Path]] = {}  # unit -> the leaf files its assembly reaches
        self.drawn = 0  # triangles drawn by the document being written
        self.stats = Counter()
        # Decide from the scan's estimates (triangles, box) rather than the converted leaves: a policy
        # that compares levels then chooses before anything is converted, and the same after.
        self.estimate = False
        self.placed_units: set[str] = set()  # the units whose leaf the document being written places

    # ---- leaves ----
    def leaf(self, unit_id: str) -> dict | None:
        """The record of a unit's geo leaf, when it draws something. The scan decides: a leaf left
        from an older scan of the unit is not used. With ``estimate``, the triangles and box are the
        scan's, and a dry run also places the leaves not converted yet (what the conversion is for)."""
        unit = self.units.get(unit_id, {})
        if not unit.get("geo"):
            return None
        if self._all_leaves is None:  # every converted leaf in one query, not one per unit
            self._all_leaves = {i: json.loads(r) for i, r in self.cache.db.execute("SELECT id, record FROM leaves")}
        rec = self._all_leaves.get(geo_leaf(unit_id))
        rec = rec if rec and not rec.get("empty") and "error" not in rec else None
        est = unit.get("estimate") if self.estimate else None
        if est is None:
            return rec
        if rec is None:  # not converted yet: placed by a dry run when it draws something
            draws = est["triangles"] or any(k in ("light", "Camera") for k in unit["geo"])
            return dict(est) if self.dry and draws else None
        return {**rec, **est}

    def own_triangles(self, unit_id: str) -> int:
        """Triangles of a unit's own leaf: the scan's estimate when deciding from it (a level not
        converted still competes), else the converted leaf's."""
        unit = self.units.get(unit_id, {})
        if self.estimate and unit.get("geo") and "estimate" in unit:
            return unit["estimate"]["triangles"]
        leaf = self.leaf(unit_id)
        return leaf["triangles"] if leaf else 0

    def leaf_path(self, unit_id: str) -> Path:
        if unit_id not in self._leaf_paths:
            leaf_id = geo_leaf(unit_id)
            if self.look and self.look in self.leaf(unit_id).get("looks", {}):
                self._leaf_paths[unit_id] = self.cache.path(f"{leaf_id}.{self.look}")
            else:
                self._leaf_paths[unit_id] = self.cache.path(leaf_id)
        return self._leaf_paths[unit_id]

    def is_leaf(self, unit_id: str) -> bool:
        """A unit that only draws itself: placed straight from its leaf."""
        return unit_id in self.units and not self.units[unit_id]["children"] and self.leaf(unit_id) is not None

    # ---- assemblies ----
    def memory(self) -> int:
        """Estimated GPU bytes of the geometry the scene places (each leaf file once). Options of a
        switch that are not placed are written too, but don't count."""
        return sum(self.file_triangles[f] for f in self.written_files.get(self.top, ())) * BYTES_PER_TRIANGLE

    def _use_leaf(self, unit_id: str) -> Path:
        """A leaf placed once more: counted, and its file."""
        path = self.leaf_path(unit_id)
        self.placed_units.add(unit_id)
        self.file_triangles[path] = self.leaf(unit_id)["triangles"]
        self.files.add(path)
        self.drawn += self.leaf(unit_id)["triangles"]
        return path

    def write(self, unit_id: str, world: np.ndarray | None = None) -> tuple[Path | None, list | None]:
        """Write a unit's assembly, placed at ``world`` (to the scene; the first placement's, for a
        unit placed from a file several times: its switches are chosen for that one)."""
        if unit_id in self.written:
            self.drawn += self.written_drawn[unit_id]
            return self.written[unit_id]
        if unit_id == self.top and self.top_path:
            path = self.top_path
        else:
            path = self.cache.path(assembly_file(unit_id, self.lod, self.look, self.nesting, self.variants,
                                                 self.compact))
        d = (_DryDoc if self.dry else _Doc)(path, {"id": unit_id, "lod": self.lod, "nesting": self.nesting,
                                                  "variants": self.variants, **({"compact": True} if self.compact else {})},
                                            self.compact)
        drawn, self.drawn = self.drawn, 0
        files, self.files = self.files, set()
        roots, box = self._content(d, unit_id, np.eye(4) if world is None else world)
        if unit_id == self.top:
            roots += self._scene_parts(d, unit_id)
        d.doc["scenes"][0]["nodes"] = roots
        if not self.dry:
            self.stats["assemblies written" if d.write() else "assemblies unchanged"] += 1
            self.stats.update(getattr(d, "stats", {}))
        self.written[unit_id] = (path if roots else None, box)
        self.written_drawn[unit_id], self.drawn = self.drawn, drawn + self.drawn
        self.written_files[unit_id], self.files = self.files, files
        return self.written[unit_id]

    def _scene_parts(self, d: _Doc, unit_id: str) -> list[int]:
        """The cameras and environment of the top unit (kept in its leaf's record, see leaf.py)."""
        rec = self.cache.get("leaves", geo_leaf(unit_id)) or {}
        parts, roots = rec.get("scene", {}), []
        for cam in parts.get("cameras", []):
            d.doc.setdefault("cameras", []).append(cam["camera"])
            camera = {"camera": len(d.doc["cameras"]) - 1}
            node = {"name": cam["name"], **({"matrix": cam["matrix"]} if cam.get("matrix") else {})}
            if cam.get("up"):
                node["children"] = [d.add({**camera, "matrix": cam["up"]})]
            else:
                node.update(camera)
            roots.append(d.add(node))
        sky = parts.get("sky")
        if sky:
            pano = sky.get("extensions", {}).get("NV_environment_sky_panorama")
            if pano and pano.get("uri"):
                pano["uri"] = urllib.parse.quote(Path(os.path.relpath(pano["uri"], d.path.parent)).as_posix())
                d.extensions.add("NV_environment_sky_panorama")
            d.doc.setdefault("extensions", {})["OMI_environment_sky"] = {"skies": [sky]}
            d.doc["scenes"][0]["extensions"] = {"OMI_environment_sky": {"sky": 0}}
            d.extensions.add("OMI_environment_sky")
        return roots

    def unit_file(self, unit_id: str, world: np.ndarray | None = None) -> tuple[Path | None, list | None]:
        """The file that draws a unit (its leaf, or its own assembly; None when it draws nothing),
        and its box (None for lights and cameras only). Not counted as placed: see ``place_file``."""
        if self.is_leaf(unit_id):
            return self.leaf_path(unit_id), self.leaf(unit_id).get("bounds")
        drawn = self.drawn
        found = self.write(unit_id, world)
        self.drawn = drawn  # written, not placed yet
        return found

    def _place_file(self, unit_id: str, world: np.ndarray | None) -> tuple[Path | None, list | None]:
        """``unit_file``, counted as placed once more."""
        if self.is_leaf(unit_id):
            return self._use_leaf(unit_id), self.leaf(unit_id).get("bounds")
        found = self.unit_file(unit_id, world)
        self.drawn += self.written_drawn.get(unit_id, 0)
        self.files |= self.written_files.get(unit_id, set())
        return found

    def _content(self, d: _Doc, unit_id: str, world: np.ndarray) -> tuple[list[int], list | None]:
        """Nodes for what ``unit_id`` (placed at ``world``) draws and places, and their box in the
        unit's space."""
        if unit_id not in self.units:  # failed to scan
            return [], None
        nodes, boxes = [], []
        if self.leaf(unit_id):
            box = self.leaf(unit_id).get("bounds")
            node = {"name": "geo", "externalAsset": d.external(self._use_leaf(unit_id), f"{short_name(unit_id)} geo")}
            if box is not None:
                node["boundingVolume"] = d.bounding_volume(box)
            nodes.append(d.add(node))
            boxes.append(box)
        for entry in self.units[unit_id]["children"]:
            index, box = self._entry(d, entry, world)
            if index is not None:
                nodes.append(index)
                boxes.append(box)
        return nodes, union(boxes)

    def _entry(self, d: _Doc, entry: dict, parent: np.ndarray) -> tuple[int | None, list | None]:
        if self.tick:
            self._ticks += 1
            if self._ticks % 20000 == 0:
                self.tick()
        if self.exclude and self.exclude.match(entry["name"]):
            self.stats["placements excluded"] += 1
            return None, None
        node: dict = {"name": entry["name"]}
        world = parent
        if "matrix" in entry:
            node["matrix"] = entry["matrix"]
            world = parent @ np.array(entry["matrix"], dtype=np.float64).reshape(4, 4).T
        unit = entry.get("unit")
        if "switch" in entry:
            placed = self._switch(d, entry, world)
            if placed is None:
                return None, None
            node["externalAsset"], box = placed
            node[SWITCH_MARK] = True  # its bounding volume is what a level-of-detail tool reads
        elif unit and (self.is_leaf(unit) or self.nesting == "files"):
            path, box = self._place_file(unit, world)
            if path is None:
                return None, None
            node["externalAsset"] = d.external(path, short_name(unit))
            self.stats["placements"] += 1
        else:
            if unit:
                children, box = self._content(d, unit, world)
                self.stats["placements"] += 1
            else:
                results = [self._entry(d, e, world) for e in entry["children"]]
                children = [i for i, _ in results if i is not None]
                box = union(b for _, b in results)
            if not children:
                return None, None
            node["children"] = children
        if box is not None:
            node["boundingVolume"] = d.bounding_volume(box)
        return d.add(node), transform_box(node.get("matrix"), box)

    # ---- switches ----
    def _ranked(self, entry: dict) -> list[str]:
        """The options of a switch that draw something, most triangles first."""
        options = [v for v, u in entry["options"].items() if self.triangles(u) > 0]
        return sorted(options, key=lambda v: -self.triangles(entry["options"][v]))

    def _choice(self, entry: dict, box=None) -> str | None:
        """The option placed: named by ``--variant``, else picked by the policy (``camera`` needs the
        placement's ``box`` in the scene; without one it places the authored selection)."""
        ranked = self._ranked(entry)
        if not ranked:
            return None
        choices = self.variants.get(entry["switch"], {})
        choice = choices.get(entry["name"], choices.get("*"))
        if choice in ranked:
            return choice
        if choice in entry["options"]:  # named, and draws nothing: nothing placed
            return None
        if self.lod == "camera" and self.view and box is not None:
            wanted = self.view.target(box)
            enough = [v for v in ranked if self.triangles(entry["options"][v]) >= wanted]
            return enough[-1] if enough else ranked[0]
        if self.lod == "finest":
            return ranked[0]
        if self.lod == "coarsest":
            return ranked[-1]
        if self.lod.startswith("level"):
            return ranked[min(int(self.lod[5:]), len(ranked) - 1)]
        if self.lod == "authored" and entry["selected"] in entry["options"]:
            return entry["selected"] if entry["selected"] in ranked else None  # every option placed is converted
        return entry["selected"] if entry["selected"] in ranked else ranked[0]

    def _switch(self, d: _Doc, entry: dict, world: np.ndarray) -> tuple[int, list] | None:
        """Every option as a file, chained by triangle count; the chosen one placed."""
        assets = {}
        for variant in self._ranked(entry):
            unit = entry["options"][variant]
            path, box = self.unit_file(unit, world)
            if path is not None:
                assets[variant] = (d.external(path, f"{short_name(unit)} {variant}"), box)
        boxes = [b for _, b in assets.values() if b is not None]
        choice = self._choice(entry, transform_box([float(x) for x in world.T.flatten()], union(boxes)) if boxes else None)
        if choice not in assets:
            return None
        self._place_file(entry["options"][choice], world)
        chain = [a for a, _ in assets.values()]
        if len(chain) > 1 and not self.dry:
            for a in chain:
                d.doc["externalAssets"][a].setdefault("extensions", {})[LOD_EXTENSION] = {"lods": chain}
            d.extensions.add(LOD_EXTENSION)
        self.stats["switch placements"] += 1
        return assets[choice]

    # ---- statistics ----
    def triangles(self, unit_id: str) -> int:
        """Triangles a unit draws as assembled (each placement counted), from the leaf records."""
        if unit_id not in self._triangles:
            self._triangles[unit_id] = 0  # a cycle counts nothing
            children = self.units[unit_id]["children"] if unit_id in self.units else []
            self._triangles[unit_id] = self.own_triangles(unit_id) + sum(self._entry_triangles(e) for e in children)
        return self._triangles[unit_id]

    def _entry_triangles(self, entry: dict) -> int:
        if "switch" in entry:
            choice = self._choice(entry)
            return self.triangles(entry["options"][choice]) if choice else 0
        if "unit" in entry:
            return self.triangles(entry["unit"])
        return sum(self._entry_triangles(e) for e in entry["children"])
