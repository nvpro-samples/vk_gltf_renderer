"""Looks: glTF PBR materials made from material *names*, for scenes that ship names and no materials
(Caldera names every surface, plaster_stucco_beige_01 or marble_black_01, and defines none).

A recipe is a JSON file in this folder (``names.json``, ``caldera.json``) or anywhere else, edited
without touching code:

    extends          another recipe this one adds to (its own rules come first)
    families         [{name, match (regex), color (sRGB), roughness, metallic, extension?, emissive?,
                     emissive_strength?, variation?}], first match wins; emissive (sRGB) makes the surface
                     glow, variation replaces the recipe's (vehicle paint: any hue)
    colours          {word: sRGB}: a colour word in the name tints the family ("darker"/"lighter" scale it)
    finishes         {regex: {roughness: +-delta, value: factor, metallic: value}}: worn, polished, painted...
    variation        {value, hue}: a small deterministic jitter per name, so that 40 woods aren't one
    unknown          {color, roughness, metallic}: for names no family matches (jittered too)
    remove           [word]: surfaces to drop (game-only geometry, decals that need their texture)
    exclude          [regex]: placements usd-world leaves out of the scene, by prim name (start of it):
                     a game's occluder volumes and editor markers
    primitive_rules  [{material?, mesh?, min_size?, max_size?, family}]: primitives whose material and
                     mesh names match (regex) and whose size fits (metres, per axis, null = any) get that
                     family: Caldera's ocean proxy -> water, a shared trim material on glass_outsideShape -> glass.
                     A primitive merged from several USD meshes (``extras.usd_parts``, usd-world's
                     --leaf-merge) is matched part by part, and split where they differ
    extensions       {name: {glTF material extensions}}: what a family's ``extension`` adds

Only the materials a converter made up are changed: by default, those usd-to-gltf.py writes for a
surface without a material (no texture, no extension, metallic 0, roughness 0.5); ``everything``
changes all of them. ``utils/gltf-look.py`` applies a recipe to any glTF file; usd-world applies it
to every leaf (``--look``), next to the leaf, so changing a recipe never converts geometry again.
"""

from __future__ import annotations

import colorsys
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent


def available() -> list[str]:
    return sorted(p.stem for p in HERE.glob("*.json"))


def load(recipe: str | Path) -> dict:
    """A recipe by name (a file of this folder) or path, with what it extends merged in."""
    path = Path(recipe) if str(recipe).lower().endswith(".json") else HERE / f"{recipe}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    base = load(data["extends"]) if data.get("extends") else {}
    merged = {**base, **{k: v for k, v in data.items() if k not in ("extends", "description")}}
    for key in ("families", "primitive_rules"):
        merged[key] = data.get(key, []) + base.get(key, [])
    for key in ("remove", "exclude"):
        merged[key] = base.get(key, []) + data.get(key, [])
    for key in ("colours", "finishes", "extensions"):
        merged[key] = {**base.get(key, {}), **data.get(key, {})}
    merged["name"] = path.stem
    merged["hash"] = hashlib.sha1(json.dumps({k: v for k, v in merged.items() if k != "hash"}, sort_keys=True).encode()).hexdigest()
    return merged


def made_up(material: dict) -> bool:
    """A material usd-to-gltf.py writes for a surface that has none (its displayColor, or a grey)."""
    pbr = material.get("pbrMetallicRoughness", {})
    return (set(material) <= {"name", "doubleSided", "pbrMetallicRoughness", "extras"}
            and set(pbr) <= {"baseColorFactor", "metallicFactor", "roughnessFactor"}
            and pbr.get("metallicFactor", 1.0) == 0.0 and pbr.get("roughnessFactor", 1.0) == 0.5)


MIN_ALBEDO = 0.03  # darkest linear luminance a made-up non-metal gets (charcoal, black rubber)


def _linear(c):
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]


def _jitter(name: str, colour, amount: dict):
    """The colour moved a little, the same way for the same name."""
    h = hashlib.sha1(name.encode()).digest()
    hue, light, sat = colorsys.rgb_to_hls(*colour)
    hue = (hue + (h[0] / 255 - 0.5) * 2 * amount.get("hue", 0.0)) % 1.0
    light = min(1.0, max(0.0, light * (1 + (h[1] / 255 - 0.5) * 2 * amount.get("value", 0.0))))
    return colorsys.hls_to_rgb(hue, light, sat)


def material(name: str, recipe: dict, family: str | None = None) -> tuple[dict, str | None]:
    """(glTF material, family name or None when nothing matched) for a material name."""
    n = name.lower()
    fam = None
    for f in recipe["families"]:
        if (family and f["name"] == family) or (not family and re.search(f["match"], n)):
            fam = f
            break
    if fam is None:
        base = recipe.get("unknown", {"color": [0.5, 0.5, 0.5], "roughness": 0.6, "metallic": 0.0})
        colour, rough, metal, ext = tuple(base["color"]), base["roughness"], base.get("metallic", 0.0), None
    else:
        colour, rough, metal, ext = tuple(fam["color"]), fam["roughness"], fam.get("metallic", 0.0), fam.get("extension")
    words = re.split(r"[_\d\s.-]+", n)
    variation = (fam or {}).get("variation", recipe.get("variation", {}))
    for w in words:
        tint = recipe["colours"].get(w)
        if isinstance(tint, list) and ext not in ("glass", "water") and not (metal and w in ("black", "gray", "grey")):
            colour, variation = tuple(tint), recipe.get("variation", {})  # the name's colour, not a random one
    if "dark" in words or "darker" in words:
        colour = tuple(c * 0.7 for c in colour)
    if "light" in words or "lighter" in words:
        colour = tuple(min(1.0, c * 1.25) for c in colour)
    for pattern, finish in recipe["finishes"].items():
        if re.search(pattern, n):
            rough = min(1.0, max(0.0, rough + finish.get("roughness", 0.0)))
            colour = tuple(min(1.0, c * finish.get("value", 1.0)) for c in colour)
            metal = finish.get("metallic", metal)
    if ext not in ("glass", "water"):
        colour = _jitter(n, colour, variation)
    linear = _linear(colour)
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    if not metal and 0.0 < luminance < MIN_ALBEDO:  # colour words compound: keep it above charcoal
        linear = [min(1.0, x * MIN_ALBEDO / luminance) for x in linear]
    mat = {"name": name, "doubleSided": True,
           "pbrMetallicRoughness": {"baseColorFactor": [round(x, 4) for x in linear] + [1.0],
                                    "metallicFactor": metal, "roughnessFactor": round(rough, 3)}}
    if ext and ext in recipe["extensions"]:
        mat["extensions"] = json.loads(json.dumps(recipe["extensions"][ext]))
    if fam and fam.get("emissive"):
        mat["emissiveFactor"] = [round(x, 4) for x in _linear(fam["emissive"])]
        if fam.get("emissive_strength", 1.0) != 1.0:
            mat.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {"emissiveStrength": fam["emissive_strength"]}
    return mat, fam["name"] if fam else None


def _runs(parts: list, hit) -> list[tuple[bool, list]]:
    """Consecutive ``usd_parts`` grouped by whether ``hit(name)``: [(hit, [part, ...]), ...]."""
    runs: list[tuple[bool, list]] = []
    for part in parts:
        h = hit(part[0])
        if runs and runs[-1][0] == h:
            runs[-1][1].append(part)
        else:
            runs.append((h, [part]))
    return runs


def _sub_primitive(doc: dict, prim: dict, run: list) -> dict:
    """The parts ``run`` of a merged primitive: the same vertices, its index range alone (a new
    accessor into the same buffer view, so the look writes no binary data)."""
    first, count = run[0][1], sum(p[2] for p in run)
    indices = doc["accessors"][prim["indices"]]
    size = 4 if indices["componentType"] == 5125 else 2
    doc["accessors"].append({k: v for k, v in indices.items() if k not in ("min", "max")}
                            | {"byteOffset": indices.get("byteOffset", 0) + first * size, "count": count})
    return {**prim, "indices": len(doc["accessors"]) - 1,
            "extras": {**prim.get("extras", {}), "usd_parts": [[n, f - first, c] for n, f, c in run]}}


def apply(doc: dict, recipe: dict, everything: bool = False) -> Counter:
    """Give the glTF document ``doc`` the recipe's look, in place. The counts it returns include
    ``family <name>`` and ``unmatched <name>`` per material, for improving the recipe."""
    stats = Counter()
    materials = doc.get("materials", [])
    remove = re.compile(r"(^|_)(" + "|".join(recipe["remove"]) + r")(_|$)") if recipe["remove"] else None
    targets = {i for i, m in enumerate(materials) if everything or made_up(m)}
    removed = {i for i in targets if remove and remove.search(materials[i].get("name", "").lower())}
    for i in targets - removed:
        name = materials[i].get("name", "")
        materials[i], fam = material(name, recipe)
        stats[f"family {fam}" if fam else f"unmatched {name}"] += 1
    stats["materials changed"] += len(targets - removed)
    stats["materials kept"] += len(materials) - len(targets)

    for rule in recipe["primitive_rules"]:  # e.g. a flat sheet kilometres across is the sea
        pattern, made = re.compile(rule.get("material", "")), None
        mesh_pattern = re.compile(rule.get("mesh", ""))
        for mesh in doc.get("meshes", []):
            out = []
            for prim in mesh["primitives"]:
                m = prim.get("material")
                a = doc["accessors"][prim["attributes"]["POSITION"]]
                size = [h - l for l, h in zip(a["min"], a["max"])]
                if m not in targets or m in removed or not pattern.search(materials[m].get("name", "").lower()) or \
                        not all(lo is None or s >= lo for s, lo in zip(size, rule.get("min_size", [None] * 3))) or \
                        not all(hi is None or s <= hi for s, hi in zip(size, rule.get("max_size", [None] * 3))):
                    out.append(prim)
                    continue
                parts = prim.get("extras", {}).get("usd_parts") or [[mesh.get("name", ""), 0, None]]
                runs = _runs(parts, lambda name: bool(mesh_pattern.search(name.lower())))
                if not any(hit for hit, _ in runs):
                    out.append(prim)
                    continue
                if made is None:
                    materials.append(material(rule["family"], recipe, rule["family"])[0])
                    made = len(materials) - 1
                for hit, run in runs:
                    piece = prim if len(runs) == 1 else _sub_primitive(doc, prim, run)
                    if hit:
                        piece["material"] = made
                        stats[f"rule {rule['family']}"] += len(run)
                    out.append(piece)
            mesh["primitives"] = out

    for mesh in doc.get("meshes", []):
        keep = [p for p in mesh["primitives"] if p.get("material") not in removed]
        stats["primitives removed"] += len(mesh["primitives"]) - len(keep)
        mesh["primitives"] = keep
    # glTF meshes need a primitive: drop the emptied ones and renumber the rest
    meshes = doc.get("meshes", [])
    empty = {i for i, m in enumerate(meshes) if not m["primitives"]}
    remap, kept = {}, []
    for i, m in enumerate(meshes):
        if i not in empty:
            remap[i] = len(kept)
            kept.append(m)
    if kept:
        doc["meshes"] = kept
    else:
        doc.pop("meshes", None)  # glTF: present means at least one
    for node in doc.get("nodes", []):
        if "mesh" in node:
            if node["mesh"] in empty:
                del node["mesh"]
                node.get("extensions", {}).pop("EXT_mesh_gpu_instancing", None)
            else:
                node["mesh"] = remap[node["mesh"]]
    stats["meshes emptied"] += len(empty)
    used = {e for m in materials for e in m.get("extensions", {})}
    if used or doc.get("extensionsUsed"):
        doc["extensionsUsed"] = sorted(set(doc.get("extensionsUsed", [])) | used)
    return stats
