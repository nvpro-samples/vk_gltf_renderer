"""A project: one cache, the stage it converts and the settings it is built with, and the steps.

Each step can run alone, on the whole scene or on some units and everything under them, from the
command line (cli.py) or the UI (ui.py); every step reads what the previous ones recorded in the
index and keeps its own results there:

    scan      the asset graph (scan.py)
    convert   what each unit draws itself, as its own glTF leaf (leaf.py)
    ktx       the leaves' images as KTX2 (ktx.py)
    look      a material recipe applied to the leaves (looks/)
    assemble  the glTF 2.1 scene placing the leaves (assemble.py)

The settings are stored in the cache, so a later command or the UI only needs the cache folder.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
from pxr import Usd

from .. import looks
from ..convert import log as print_log, stage_exposure, usd_diagnostics
from .assemble import Assembler, LodView, assembly_file, check_lod, geo_leaf
from .cache import CONVERTER_VERSION, Cache, make_key
from .ktx import ktx
from .leaf import convert
from .scan import Scanner, top_prim, unit_id
from .textures import auto_size, images, scene_bytes

STEPS = ("scan", "convert", "ktx", "look", "assemble")
# Workers decode full-resolution textures: all cores of a big machine can run out of memory.
DEFAULT_JOBS = min(8, os.cpu_count() or 1)
DEFAULTS = {"input": None, "prim": None, "output": None, "jobs": DEFAULT_JOBS, "materials": "auto",
            "single_sided": False, "keep_exposure": False, "ktx": True, "look": None, "lod": "authored",
            # The largest KTX2 texture (pixels; 0: as authored; "auto": the largest that fits the
            # budget beside the geometry, chosen when the images are encoded).
            "ktx_max_size": "auto",
            "variants": [], "nesting": "inline",
            # Assembly nodes: ``usd`` one per prim, ``compact`` fewer and smaller (see assemble.py),
            # ``auto`` compact when the scan found more than Project.COMPACT_UNITS units.
            "hierarchy": "auto",
            # Leaves: ``none`` one mesh per USD mesh, as authored; ``material`` each asset's meshes
            # merged into one, a primitive per material (merge.py): faster to load, less memory.
            "leaf_merge": "none",
            # Variants chosen for one asset (the Selected asset panel): {unit id: {set: value}}.
            "asset_variants": {},
            # lod "camera": where from (``scene``: its cameras; ``x,y,z;...``; or vk_gltf_renderer
            # --cameraString values separated by ``;``), the view, and the GPU budget for geometry
            # and textures (also what "auto" texture sizes fit; empty: 70 % of the GPU's memory).
            "lod_camera": "scene", "lod_fov": 60.0, "lod_pixels": 1080, "lod_density": 1.0, "lod_budget_gb": None}


def available_looks() -> list[str]:
    return looks.available()


def parse_variants(specs: list[str]) -> dict[str, dict[str, str]]:
    """``SET=VALUE`` (everywhere) and ``SET=VALUE:name,name`` (the switches with those prim names,
    such as districts) -> {set: {"*" or name: value}}."""
    out: dict[str, dict[str, str]] = {}
    for spec in specs:
        name, _, value = spec.partition("=")
        value, _, names = value.partition(":")
        for target in names.split(",") if names else ["*"]:
            out.setdefault(name, {})[target] = value
    return out


def look_leaf(task: dict) -> Counter:
    """Apply recipe ``task["look"]`` to one leaf file, writing ``task["out"]``. Runs in worker processes."""
    doc = json.loads(Path(task["leaf"]).read_text(encoding="utf-8"))
    stats = looks.apply(doc, looks.load(task["look"]))
    Path(task["out"]).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return stats


class Project:
    def __init__(self, cache: Cache, settings: dict, log=print_log, progress=None):
        """``log(text)`` reports, ``progress(step, done, total)`` advances a step."""
        self.cache, self.settings = cache, {**DEFAULTS, **settings}
        self.log, self.progress = log, progress or (lambda step, done, total: None)
        self.scanner: Scanner | None = None
        self.top: str | None = None
        self._needed: set[str] | None = None  # see needed_units()
        self._level_try = 0  # the budget search's step, for the progress
        self.stats: dict[str, Counter] = {}

    # ---- settings ----
    @classmethod
    def open(cls, folder: Path, root: Path | None = None, changes: dict | None = None, **kw) -> Project:
        """The project of a cache folder, with ``changes`` to its stored settings (None values are
        left alone) saved."""
        cache = Cache(folder, root)
        row = cache.db.execute("SELECT value FROM meta WHERE name='project'").fetchone()
        settings = {**DEFAULTS, **(json.loads(row[0]) if row else {})}
        settings.update({k: v for k, v in (changes or {}).items() if v is not None})
        check_lod(settings["lod"])
        if settings["input"]:
            settings["input"] = os.path.abspath(settings["input"])
        project = cls(cache, settings, **kw)
        project.save()
        return project

    def save(self) -> None:
        self.cache.db.execute("INSERT OR REPLACE INTO meta VALUES ('project', ?)", (json.dumps(self.settings),))
        self.cache.commit()

    @property
    def variants(self) -> dict[str, dict[str, str]]:
        return parse_variants(self.settings["variants"])

    @property
    def everywhere(self) -> dict[str, str]:
        """Selections made in every unit while composing: part of every id and key. A switch's set
        (districtLod, modelLod) is not one: the assembly picks its option, and its choice never
        changes how an asset composes, so choosing it for the whole scene reuses every asset
        converted (Caldera: 38,000 of them) instead of making them all new."""
        switches = set(self.get_meta("switch_sets", []))
        return {name: v["*"] for name, v in sorted(self.variants.items()) if "*" in v and name not in switches}

    COMPACT_UNITS = 10_000  # Caldera has 140,000 units; the sample scenes tried have under 2,000

    def compact(self) -> bool:
        """Whether assemblies are written compact (the ``hierarchy`` setting)."""
        h = self.settings["hierarchy"]
        if h == "auto":
            return bool(self.scanner) and len(self.scanner.units) > self.COMPACT_UNITS
        return h == "compact"

    def scene_path(self) -> Path:
        s = self.settings
        if s["output"]:
            return Path(s["output"]).resolve()
        tags = assembly_file("", s["lod"], s["look"], s["nesting"], self.variants,
                             self.compact()).removeprefix("units//assembly")
        return self.cache.dir / f"{Path(s['input']).stem}{tags}.gltf"

    # ---- selection ----
    def subtree(self, patterns: list[str]) -> set[str] | None:
        """The units whose id matches a glob of ``patterns``, and every unit under them (None: all)."""
        if not patterns:
            return None
        units = self.scanner.units
        todo = [u for u in units if any(fnmatch(u, p) for p in patterns)]
        found: set[str] = set()
        while todo:
            u = todo.pop()
            if u in found or u not in units:
                continue
            found.add(u)
            todo += _placed(units[u]["children"])
        return found

    def leaf_ids(self, units: set[str] | None = None, compute: bool = True) -> list[str]:
        needed = self.needed_units(compute)
        return [geo_leaf(u) for u, rec in self.scanner.units.items()
                if rec["geo"] and (units is None or u in units) and (needed is None or u in needed)]

    def needed_units(self, compute: bool = True) -> set[str] | None:
        """The units whose leaves the scene places when the level-of-detail policy compares levels
        (None: as authored, every unit scanned is placed). Chosen from the scan's estimates, so only
        the levels the policy picks are converted; the top unit first, its cameras drive the choice.
        Finding them assembles the whole graph (minutes for Caldera's): jobs do it and remember the
        result; the page (``compute`` False) only reads what a job found, else just the top."""
        if self.settings["lod"] == "authored" or not self.top:
            return None
        if self._needed is None:
            key = self._choice_key()
            known = self.get_meta("needed")
            if known and known["key"] == key:
                self._needed = set(known["units"])
            elif not compute:
                return {self.top}
            else:
                try:
                    a = self.dry_assembly(self.density())
                except SystemExit:  # camera policy before the top leaf (the cameras) is converted
                    return {self.top}
                # The leaves the scene places, not every level written into a switch's chain.
                placed = a.written_files.get(self.top, set())
                self._needed = {u for u in a.placed_units if a.leaf_path(u) in placed} | {self.top}
                self.set_meta("needed", {"key": key, "units": sorted(self._needed)})
        return self._needed

    def exclude(self) -> list[str]:
        """The placements the look leaves out (its ``exclude``), by prim name."""
        return looks.load(self.settings["look"]).get("exclude", []) if self.settings["look"] else []

    def _choice_key(self) -> str:
        """What the levels chosen depend on: the assembly's settings, the asset choices, the scan."""
        s = self.settings
        return make_key({"assembly": self.assembly_settings(), "asset_variants": s["asset_variants"],
                         "exclude": self.exclude(), "scan": self._scan_key()}, {})

    def _scan_key(self) -> str:
        """Every unit's scan key: an edit that changes what a unit holds, not only how many units
        there are, chooses the levels again."""
        h = hashlib.sha1()
        for id, rec in sorted(self.scanner.units.items()):
            h.update(f"{id}={rec.get('key')}\n".encode())
        return h.hexdigest()

    def density(self) -> float | None:
        """The camera policy's density within the budget (fit_budget), remembered for the settings
        and scan it was found with: the page asks for it on every refresh."""
        s = self.settings
        if s["lod"] != "camera" or not s["lod_budget_gb"]:
            return None
        key = self._choice_key()
        known = self.get_meta("fitted")
        if known and known["key"] == key:
            return known["density"]
        density = self.fit_budget()
        self.set_meta("fitted", {"key": key, "density": density})
        return density

    # ---- steps ----
    def _report(self, step: str, stats: Counter, t0: float) -> None:
        self.stats[step] = stats
        self.log(f"{step}: {time.time() - t0:.1f} s")
        for k, v in stats.items():
            self.log(f"  {k}: {v:,.1f}" if isinstance(v, float) else f"  {k}: {v:,}")

    def scan(self, pool, force: list[str] = ()) -> str | None:
        if not self.settings["input"]:
            raise SystemExit("error: no input stage: give one (it is remembered by the cache)")
        t0 = time.time()
        self.scanner = Scanner(self.cache, lambda id: any(fnmatch(id, p) for p in force), self.everywhere,
                               self.settings["asset_variants"], self.settings["lod"], self.variants)
        self.top = self.scanner.run(self.settings["input"], self.settings["prim"], pool,
                                    progress=lambda done, found: self.progress("scan", done, found))
        self.cache.commit()
        # Remember which sets are switches: a scene-wide choice of one is the assembly's (everywhere).
        found = Counter()
        for rec in self.scanner.units.values():
            _switch_sets(rec["children"], found)
        known = set(self.get_meta("switch_sets", []))
        if not set(found) <= known:
            before = self.everywhere
            self.set_meta("switch_sets", sorted(known | set(found)))
            if self.everywhere != before:  # a switch was chosen scene-wide before it was known: scan as it should be
                return self.scan(pool, force)
        if self.top in self.scanner.errors:  # nothing to convert: say why rather than write nothing
            raise RuntimeError(f"the stage couldn't be scanned: {self.scanner.errors[self.top].strip().splitlines()[-1]}")
        stats = self.scanner.stats
        stats["units"] = len(self.scanner.units)
        stats["switches"] = sum(_count_switches(rec["children"]) for rec in self.scanner.units.values())
        self._report("scan", stats, t0)
        return self.top

    def load(self) -> str | None:
        """The asset graph as the last scan recorded it, without opening USD (the UI's view)."""
        if not self.settings["input"]:
            return None
        self.scanner = Scanner(self.cache, lambda id: False, self.everywhere, self.settings["asset_variants"],
                               self.settings["lod"], self.variants)
        prim = self.settings["prim"] or top_prim(self.settings["input"])
        self.top = self.scanner.choose(unit_id(self.cache, self.settings["input"], prim), self.settings["input"],
                                       prim, {})[0] if prim else None
        todo = [self.top] if self.top else []
        while todo:
            id = todo.pop()
            if id in self.scanner.units:
                continue
            rec = self.cache.get("units", id)
            if rec:
                rec = self.scanner.adopt(id, rec)
                wanted = self.scanner.wanted(rec)
                todo += [u for u, *_ in rec["units"] if u in wanted]
            else:
                self.scanner.missing.add(id)
        return self.top if self.top in self.scanner.units else None

    def get_meta(self, name: str, default=None):
        row = self.cache.db.execute("SELECT value FROM meta WHERE name=?", (name,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, name: str, value) -> None:
        self.cache.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (name, json.dumps(value)))
        self.cache.commit()

    ASSEMBLY_SETTINGS = ("look", "lod", "variants", "nesting", "hierarchy", "output", "lod_camera", "lod_fov", "lod_pixels",
                         "lod_density", "lod_budget_gb")

    def assembly_settings(self) -> dict:
        return {k: self.settings[k] for k in self.ASSEMBLY_SETTINGS}

    def estimate(self) -> dict:
        """The scene as it would be assembled, without writing it: triangles drawn and geometry memory."""
        s = self.settings
        density = self.density()
        a = self.dry_assembly(density)
        result = {"drawn": a.written_drawn.get(self.top, 0), "gb": a.memory() / 2**30,
                  "textures_gb": self.texture_bytes(a) / 2**30,
                  "files": len(a.written_files.get(self.top, ())), "density": density}
        self.log(f"estimate: {result['drawn']:,} triangles drawn, about {result['gb']:.2f} GB of geometry"
                 f" and {result['textures_gb']:.2f} GB of textures" + (f" (density {density:.3g})" if density else ""))
        return result

    def dry_assembly(self, density: float | None = None) -> Assembler:
        """The scene assembled in memory, nothing written: what it places, and how much."""
        a = self.assembler(density)
        a.dry = True
        # Caldera's levels are millions of placements: report progress now and then (and so let Stop in).
        self._level_try += 1
        a.tick = lambda n=self._level_try: self.progress("levels", n, max(n, 12))
        a.write(self.top)
        return a

    def texture_bytes(self, a: Assembler) -> int:
        """GPU bytes of the textures of an assembly: as encoded, or as they will be."""
        size = self.settings["ktx_max_size"]
        return scene_bytes(self.image_files(a), self.settings["ktx"], 0 if size == "auto" else int(size))

    @staticmethod
    def image_files(a: Assembler) -> list[Path]:
        """The leaf files an assembly places that have images: the conversion counted them, so a
        scene of thousands of leaves without textures (Caldera) reads none."""
        unit_of = {a.leaf_path(u): u for u in a.placed_units}
        return [f for f in a.written_files.get(a.top, ())
                if f not in unit_of or (a.leaf(unit_of[f]) or {}).get("images", 1)]

    def budget_gb(self) -> float:
        gpu = gpu_memory_gb()
        return self.settings["lod_budget_gb"] or (round(0.7 * gpu, 1) if gpu else 8.0)

    def ktx_size(self) -> int:
        """The largest texture the KTX2 step makes. "auto": the largest that fits what the geometry
        leaves of the budget; with camera levels of detail, whose geometry takes what is left, half."""
        size = self.settings["ktx_max_size"]
        if size != "auto":
            return int(size)
        budget = self.budget_gb() * 2**30
        a = self.dry_assembly()
        allowance = budget / 2 if self.settings["lod"] == "camera" else max(0.0, budget - a.memory())
        size = auto_size(self.image_files(a), allowance, progress=lambda done, total: self.progress("ktx-size", done, total))
        self.log(f"  texture size: {f'at most {size} pixels' if size else 'as authored'} "
                 f"(textures within {allowance / 2**30:.1f} GB of the {budget / 2**30:.1f} GB budget)")
        return size

    def recommend(self, summary: dict) -> dict:
        """The options the page proposes first, from what the scan and the conversion found."""
        rec = self.cache.get("leaves", geo_leaf(self.top)) if self.top else None
        cameras = len((rec or {}).get("scene", {}).get("cameras", []))
        names, materials = [], 0
        for leaf_id in self.leaf_ids(compute=False)[:200]:  # a sample is enough to tell a scene of names from one of materials
            path = self.cache.path(leaf_id)
            if path.exists():
                doc = json.loads(path.read_text(encoding="utf-8"))
                # A surface bound to nothing gets the converter's "displayColor": no name to go by.
                names += [m.get("name", "") for m in doc.get("materials", [])
                          if looks.made_up(m) and m.get("name") != "displayColor"]
                materials += len(doc.get("materials", []))
        look = None
        if materials and len(names) / materials >= 0.5:  # mostly names without materials: a recipe helps
            # The recipe that knows the most names; on a tie, the most general (fewest rules), since a
            # data set's own recipe also drops what that data set doesn't draw.
            scored = []
            for name in looks.available():
                recipe = looks.load(name)
                known = sum(looks.material(n, recipe)[1] is not None for n in names)
                scored.append((known, -len(recipe["families"]), name))
            best = max(scored)
            look = best[2] if best[0] >= 0.5 * len(names) else None  # a recipe that knows most of the names
        gpu = gpu_memory_gb()
        return {"look": look, "made_up_share": len(names) / materials if materials else None, "ktx_max_size": "auto",
                "lod": "camera" if summary["switches"] and cameras else "authored",
                "cameras": cameras, "gpu_gb": gpu, "budget_gb": round(0.7 * gpu, 1) if gpu else 8.0}

    def summary(self) -> dict:
        """What the page shows after a scan: the asset graph in numbers, the choices it offers."""
        units = self.scanner.units if self.scanner else {}
        out = {"units": len(units), "to_scan": len(self.scanner.missing) if self.scanner else 0,
               "scan_failed": len(self.scanner.errors) if self.scanner else 0, "leaves": 0, "converted": 0, "failed": 0, "to_convert": 0, "ktx": 0,
               "out_of_date": 0, "empty": 0, "switches": Counter(), "variant_sets": {}, "drawn": None}
        options = self.leaf_options() if units else None
        needed = self.needed_units(compute=False) if units else None  # a policy comparing levels converts only those placed
        out["levels_unknown"] = needed is not None and self._needed is None  # chosen by the next conversion
        lists: dict[str, dict[tuple, list[str]]] = {}  # variant set -> its values -> the assets offering them
        for unit_id, rec in units.items():
            for name, values in rec.get("variant_sets", {}).items():
                lists.setdefault(name, {}).setdefault(tuple(sorted(values)), []).append(Path(rec["layer"]).stem)
            _switch_sets(rec["children"], out["switches"])
            if rec["geo"] and (needed is None or unit_id in needed):
                out["leaves"] += 1
                leaf = self.cache.get("leaves", geo_leaf(unit_id))
                if leaf is None:
                    out["to_convert"] += 1
                elif "error" in leaf:
                    out["failed"] += 1
                elif not self.leaf_up_to_date(geo_leaf(unit_id), options):
                    out["to_convert"] += 1
                    out["out_of_date"] += 1
                else:
                    out["converted"] += 1
                    out["ktx"] += bool(leaf.get("ktx")) and not leaf.get("empty")
                    out["empty"] += bool(leaf.get("empty"))
        # A set is a scene-wide choice when its assets share values (Da Vinci's material: render,
        # renderLow...); a set whose every asset has its own (Kitchen_set's modelingVariant: BowlA
        # for the bowl, CerealBoxA for the box) is chosen per asset.
        for name, by_values in sorted(lists.items()):
            common = set.intersection(*(set(v) for v in by_values))
            out["variant_sets"][name] = {
                "values": sorted({x for v in by_values for x in v}), "common": sorted(common),
                "assets": sum(len(a) for a in by_values.values()),
                "examples": [f"{assets[0]}: {', '.join(values[:4])}{'...' if len(values) > 4 else ''}"
                             # the shortest lists: an asset's own, not a stage holding them all
                             for values, assets in sorted(by_values.items(), key=lambda kv: len(kv[0]))[:3]]}
        out["switches"] = dict(out["switches"])
        if self.top and self.top in units:
            out["drawn"] = self.assembler().triangles(self.top)
        # The stages: each done when what it makes is there and up to date.
        assembled = self.get_meta("assembled")
        scene = self.scene_path() if self.settings["input"] else None
        converted = bool(units) and out["to_convert"] == 0 and out["failed"] == 0 and not out["levels_unknown"] \
            and (not self.settings["ktx"] or out["ktx"] + out["empty"] >= out["converted"])
        out["stage"] = {
            "scan": bool(units) and not self.scanner.missing,  # a choice can need assets not scanned yet
            "convert": converted,
            # Each stage needs the ones before it: a scene of out-of-date leaves is not done.
            "assemble": converted and bool(assembled and scene and scene.exists()
                                           and assembled["with"] == self.assembly_settings()
                                           and assembled["at"] >= self.get_meta("leaves_changed", 0)),
        }
        out["assembled"] = assembled
        return out

    def exposure(self) -> float | None:
        """The input stage's exposure, baked into every leaf as usd-to-gltf.py does (None: physical values)."""
        if self.settings["keep_exposure"]:
            return None
        with usd_diagnostics(lambda note: None):  # read for its cameras only: the scan reports the rest
            stage = Usd.Stage.Open(self.settings["input"], Usd.Stage.LoadNone)
        return stage_exposure(stage, Usd.TimeCode.EarliestTime())[0]

    def leaf_options(self) -> dict:
        s = self.settings
        if not hasattr(self, "_exposure"):
            self._exposure = self.exposure()
        return {"materials": s["materials"], "single_sided": s["single_sided"], "exposure": self._exposure,
                "merge": s["leaf_merge"]}

    def leaf_identity(self, leaf_id: str, options: dict) -> dict:
        unit_id = leaf_id.removeprefix("units/").removesuffix("/geo")
        # The scan decides what the leaf holds (what is placed elsewhere): its key is part of the leaf's.
        return {"converter": CONVERTER_VERSION, "options": options, "variants": self.everywhere,
                "unit": unit_id, "scan": self.scanner.units[unit_id]["key"]}

    def leaf_up_to_date(self, leaf_id: str, options: dict) -> bool:
        leaf = self.cache.get("leaves", leaf_id)
        return self.cache.fresh(leaf, self.leaf_identity(leaf_id, options)) and \
            (leaf.get("empty") or self.cache.path(leaf_id).exists()) and not self._ktx_resize(leaf_id, leaf)

    def _ktx_resize(self, leaf_id: str, leaf: dict) -> bool:
        """Whether the leaf's images were encoded at another size than the one asked for: the
        originals are gone, so the leaf is converted again (one without images never is)."""
        size = self.settings["ktx_max_size"]
        if not (self.settings["ktx"] and leaf.get("ktx")) or size == "auto" or leaf.get("ktx_max", 0) == int(size):
            return False
        if "images" not in leaf:  # encoded before the size was recorded: count them once
            leaf["images"] = len(images(self.cache.path(leaf_id)))
            self.cache.put("leaves", leaf_id, leaf)
            self.cache.commit()
        return leaf["images"] > 0

    def convert(self, pool, units: set[str] | None = None, force: list[str] = (), choose: bool = True) -> Counter:
        """``choose`` False: convert ``units`` without choosing the levels first (the top leaf, whose
        cameras the choice needs)."""
        t0, options = time.time(), self.leaf_options()
        tasks, leaf_ids = [], self.leaf_ids(units, compute=choose)
        for leaf_id in leaf_ids:
            unit_id = leaf_id.removeprefix("units/").removesuffix("/geo")
            rec = self.scanner.units[unit_id]
            if any(fnmatch(leaf_id, p) for p in force) or not self.leaf_up_to_date(leaf_id, options):
                tasks.append({"id": leaf_id, "root": str(self.cache.root), "layer": self.cache.abs(rec["layer"]),
                              "prim": rec["prim"], "selections": rec.get("selections", {}),
                              "variants": self.scanner.variants_for(unit_id),
                              "out": str(self.cache.path(leaf_id)), "textures": str(self.cache.dir / "textures"),
                              "identity": self.leaf_identity(leaf_id, options), "options": options,
                              "inputs": rec["inputs"]})
        stats, committed = Counter(), time.time()
        logs = self.cache.dir / "logs"
        logs.mkdir(exist_ok=True)
        with open(logs / "convert.log", "a", encoding="utf-8") as warnings:
            for n, rec in enumerate(pool.imap_unordered(convert, tasks), 1):
                leaf_id = rec.pop("id")
                if "error" in rec:  # recorded without a key: converted again next time
                    warnings.write(f"== {leaf_id} FAILED\n{rec['error']}")
                    self.cache.put("leaves", leaf_id, {"error": rec["error"].strip().splitlines()[-1]})
                    stats["failed"] += 1
                    if stats["failed"] <= 3:
                        self.log(f"  failed: {leaf_id}: {rec['error'].strip().splitlines()[-1]}")
                else:
                    text = rec.pop("log")
                    if text.strip():
                        warnings.write(f"== {leaf_id}\n{text}")
                    self.cache.put("leaves", leaf_id, rec)
                    stats["converted"] += 1
                    stats["empty"] += rec["empty"]
                    stats["triangles"] += rec["triangles"]
                    stats["worker seconds"] += rec["seconds"]
                self.progress("convert", n, len(tasks))
                # Recorded within seconds: a run stopped midway converts only what's left next time.
                if time.time() - committed > 2 or n == len(tasks):
                    self.cache.commit()
                    committed = time.time()
                if n % max(1, len(tasks) // 20) == 0 or n == len(tasks):
                    self.log(f"  {n}/{len(tasks)} leaves, {time.time() - t0:.1f} s")
        stats["up to date"] = len(leaf_ids) - len(tasks)
        if tasks:
            self.set_meta("leaves_changed", time.time())
        self._report("convert", stats, t0)
        return stats

    def ktx(self, units: set[str] | None = None) -> Counter:
        t0 = time.time()
        stats = ktx(self.cache, self.leaf_ids(units), self.settings["jobs"],
                    progress=lambda done, total, step: self.progress(step, done, total), max_size=self.ktx_size(),
                    delete_originals=units is None and self.needed_units() is None)
        if stats["leaves rewritten"]:
            self.set_meta("leaves_changed", time.time())
        self._report("ktx", stats, t0)
        return stats

    def look(self, pool, units: set[str] | None = None) -> Counter:
        t0, look = time.time(), self.settings["look"]
        recipe = looks.load(look)["hash"]  # with what it extends: editing either redoes the look
        stats, todo = Counter(), {}
        for leaf_id in self.leaf_ids(units):
            rec = self.cache.get("leaves", leaf_id)
            if not rec or rec.get("empty") or "error" in rec:
                continue
            key = make_key({"look": look, "recipe": recipe}, {"leaf": rec["key"], "ktx": rec.get("ktx", False)})
            out = self.cache.path(f"{leaf_id}.{look}")
            if rec.get("looks", {}).get(look) == key and out.exists():
                stats["unchanged"] += 1
                continue
            todo[leaf_id] = (rec, key, {"look": look, "leaf": str(self.cache.path(leaf_id)), "out": str(out)})
        tasks = [t for _, _, t in todo.values()]
        for n, ((leaf_id, (rec, key, _)), result) in enumerate(zip(todo.items(), pool.imap(look_leaf, tasks, chunksize=16)), 1):
            stats.update(result)
            rec.setdefault("looks", {})[look] = key
            self.cache.put("leaves", leaf_id, rec)
            stats["written"] += 1
            self.progress("look", n, len(tasks))
        self.cache.commit()
        # Per-material counts would be a line per name: sum them, and name the most used misses.
        unmatched = Counter({k[len("unmatched "):]: v for k, v in stats.items() if k.startswith("unmatched ")})
        families = sum(v for k, v in stats.items() if k.startswith("family "))
        stats = Counter({k: v for k, v in stats.items() if not k.startswith(("unmatched ", "family "))})
        stats["materials matched by a family"], stats["materials matched by none"] = families, sum(unmatched.values())
        self._report(f"look {look}", stats, t0)
        if unmatched:
            self.log("  names no family matches (add them to the recipe): "
                     + ", ".join(f"{n} ({c})" for n, c in unmatched.most_common(12)))
        return stats

    def lod_view(self, density: float | None = None) -> LodView | None:
        """The cameras of lod ``camera`` (see ``LodView``)."""
        s = self.settings
        if s["lod"] != "camera":
            return None
        eyes, fov = [], s["lod_fov"]
        if not s["lod_camera"] or s["lod_camera"] == "scene":
            rec = self.cache.get("leaves", geo_leaf(self.top)) or {}
            for cam in rec.get("scene", {}).get("cameras", []):
                eyes.append(np.array(cam["matrix"]).reshape(4, 4).T[:3, 3] if cam.get("matrix") else np.zeros(3))
                if cam["camera"].get("perspective", {}).get("yfov"):
                    fov = math.degrees(cam["camera"]["perspective"]["yfov"])
            if not eyes:
                raise SystemExit("error: lod camera: the scene has no camera; give positions (--lod-camera x,y,z)")
        else:
            for part in s["lod_camera"].split(";"):
                v = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", part)]
                eyes.append(v[:3])
                if "{" in part and len(v) >= 10:  # --cameraString: {eye}, {center}, {up}, {fov}, ...
                    fov = v[9]
        return LodView(eyes, fov, s["lod_pixels"], density if density is not None else s["lod_density"])

    def assembler(self, density: float | None = None) -> Assembler:
        s = self.settings
        a = Assembler(self.cache, self.scanner.units, s["lod"], s["nesting"], s["look"], self.variants,
                      self.compact(), self.exclude())
        a.top, a.top_path, a.view = self.top, self.scene_path(), self.lod_view(density)
        a.estimate = s["lod"] != "authored"  # levels compared by the scan's estimates, converted or not
        return a

    def fit_budget(self) -> float:
        """The highest density (up to lod_density) whose scene fits lod_budget_gb, by bisection of its
        logarithm: each try assembles the scene in memory. Memory doesn't follow the density in
        proportion (what has no levels, and the coarsest levels, are always there), hence no guess."""
        s = self.settings
        budget = s["lod_budget_gb"]

        def gb(density: float) -> float:
            a = self.dry_assembly(density)
            used = (a.memory() + self.texture_bytes(a)) / 2**30
            self.log(f"  budget {budget} GB: density {density:.3g} -> {used:.3f} GB")
            return used

        hi, lo = s["lod_density"], s["lod_density"] * 1e-6
        if gb(hi) <= budget:
            return hi
        if gb(lo) > budget:
            self.log("  the budget can't be met: even the coarsest levels need more (placed at density "
                     f"{lo:.3g})")
            return lo
        for _ in range(10):  # a factor of 1e6 in 10 halvings of its logarithm: within 4 %
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if gb(mid) <= budget else (lo, mid)
        return lo

    def assemble(self) -> Path:
        t0 = time.time()
        s = self.settings
        density = self.density()
        assembler = self.assembler(density)
        path, box = assembler.write(self.top)
        assembler.stats["triangles drawn"] = assembler.written_drawn.get(self.top, 0)
        assembler.stats["leaf files placed"] = len(assembler.written_files.get(self.top, ()))
        self._report("assemble", assembler.stats, t0)
        textures = self.texture_bytes(assembler) / 2**30
        self.log(f"  on the GPU: about {assembler.memory() / 2**30:.2f} GB of geometry and {textures:.2f} GB of textures")
        self.set_meta("assembled", {"at": time.time(), "with": self.assembly_settings(),
                                    "drawn": assembler.written_drawn.get(self.top, 0), "gb": assembler.memory() / 2**30,
                                    "textures_gb": textures})
        if box:
            self.log(f"  bounds (m): {[round(x, 1) for x in box[0]]} .. {[round(x, 1) for x in box[1]]}")
        self.log(f"Wrote {path}")
        if not self.settings["keep_exposure"]:
            self.log("  the input's camera exposure is baked into lights, emission and sky (the renderer's "
                     "auto-exposure works as is; exposure 1 shows the stage's own)")
        return path

    def run(self, steps: list[str], pool, units: list[str] = (), force: list[str] = ()) -> dict | None:
        """The steps, in order, on the whole scene or on ``units`` (globs on ids) and what they place.
        The scan always runs first (the asset graph is read from it, and it is cached)."""
        if self.scan(pool, force if "scan" in steps else ()) is None:
            return
        selected = self.subtree(list(units))
        if "convert" in steps:
            if self.settings["lod"] != "authored" and self.top:
                # The levels to convert are chosen from the top leaf's cameras: convert it first.
                self.convert(pool, {self.top}, force, choose=False)
                self._needed = None
            self.convert(pool, selected, force)
        if "ktx" in steps:
            self.ktx(selected)
        if "look" in steps and self.settings["look"]:
            self.look(pool, selected)
        if "assemble" in steps:
            self.assemble()
        if "estimate" in steps:
            return self.estimate()
        return None


@functools.lru_cache(maxsize=None)
def gpu_memory_gb() -> float | None:
    """The first GPU's memory (nvidia-smi, asked once), for a default budget; None without one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return float(out[0]) / 1024 if out else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def guess_root(stage: str, max_layers: int = 5000, seconds: float = 20.0) -> Path:
    """The folder holding every layer ``stage`` uses (as far as ``max_layers`` / ``seconds`` of
    following its references go): unit ids are paths relative to it. Any root works; a file outside
    it only gets a longer id."""
    from pxr import Sdf, Tf

    seen: set[str] = set()
    todo, t0 = [os.path.normpath(os.path.abspath(stage))], time.time()
    while todo and len(seen) < max_layers and time.time() - t0 < seconds:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        try:
            layer = Sdf.Layer.FindOrOpen(path)
        except Tf.ErrorException:  # a file USD can't read (.mtlx without usdMtlx): part of the data set still
            continue
        if not layer:
            continue
        for dep in layer.GetCompositionAssetDependencies():
            found = layer.ComputeAbsolutePath(dep)
            if found and os.path.isfile(found):
                todo.append(os.path.normpath(found))
    try:
        return Path(os.path.commonpath(sorted(seen)))
    except ValueError:  # on several drives
        return Path(stage).resolve().parent


def _placed(entries: list[dict]) -> list[str]:
    """The units placed by these entries (and by the switches' options)."""
    out = []
    for e in entries:
        if "unit" in e:
            out.append(e["unit"])
        elif "switch" in e:
            out += e["options"].values()
        else:
            out += _placed(e.get("children", []))
    return out


def _switch_sets(entries: list[dict], counts: Counter) -> None:
    for e in entries:
        if "switch" in e:
            counts[e["switch"]] += 1
        else:
            _switch_sets(e.get("children", []), counts)


def _count_switches(entries: list[dict]) -> int:
    return sum(1 if "switch" in e else _count_switches(e.get("children", [])) for e in entries)
