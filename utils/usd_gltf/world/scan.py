"""Scan: the asset graph of a stage, from its composition alone, with no knowledge of the scene.

A *unit* is an asset: a prim of a layer (with variant selections, when its placement makes some)
that the stage places by reference or payload, or the input stage itself. Its record is the tree
of what it places, each with its transform:

    {"name", "matrix"?, "children": [...]}      a group on the way to placements
    {"name", "matrix"?, "unit": id}             another unit
    {"name", "matrix"?, "switch": set, "selected": variant, "options": {variant: unit id}}
                                                 alternatives: a variant set whose every variant
                                                 places one asset (levels of detail, full or proxy)

plus ``geo``, what the unit draws itself (meshes, lights, cameras, point instancers), converted
as the unit's ``geo`` leaf. Matrices are glTF's (column-major, metres, Y-up), relative to the
parent entry.

Every reference or payload to another layer is a placement, at any depth of the composition. When
the opinions stronger than it add nothing but a transform (and variant selections, part of the unit's id), the unit is the
asset itself, shared by every such placement. When it changes the asset (overrides below it, a
material binding, primvars), the unit is the asset as changed: the placing prim of this layer,
converted where it is composed. Stages are opened without payloads; those a unit composes into
itself are loaded, the ones it places are not.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from pxr import Sdf, Tf, Usd, UsdGeom, UsdLux, UsdShade

from ..convert import Frame, log, usd_diagnostics
from .cache import SCAN_VERSION, Cache, Paths, make_key

MASKED = True  # the scan composes only each unit's subtree

# Payloads are not all loaded, so the default predicate (which wants loaded prims) would hide placements.
PREDICATE = Usd.TraverseInstanceProxies(Usd.PrimIsActive & Usd.PrimIsDefined & ~Usd.PrimIsAbstract)
# What a placing prim may author and still place the asset unchanged.
TRANSFORM_ONLY = re.compile(r"^(xformOp:.*|xformOpOrder|extentsHint|extent|visibility|purpose|proxyPrim|kind)$")


def _arc_items(spec: Sdf.PrimSpec):
    return [i for i in list(spec.referenceList.GetAddedOrExplicitItems()) + list(spec.payloadList.GetAddedOrExplicitItems())
            if i.assetPath]


def _target(spec: Sdf.PrimSpec, item) -> tuple[str, str | None]:
    return (os.path.normpath(spec.layer.ComputeAbsolutePath(item.assetPath)),
            None if item.primPath.isEmpty else str(item.primPath))


def _changes(spec: Sdf.PrimSpec) -> bool:
    """Whether an opinion does more than move what it places."""
    return bool(spec.nameChildren) or any(not TRANSFORM_ONLY.match(p.name) for p in spec.properties)


def placement(prim: Usd.Prim) -> tuple[str, str | None, dict, bool] | None:
    """(layer, prim or None for its default prim, variant selections, unchanged) of the asset
    ``prim`` places, or None. The strongest opinion that references or loads another layer places
    it; only the opinions stronger than it can change it (``unchanged`` is False when one adds more
    than a transform), and the selections they make for variant sets they don't define are made in
    the asset."""
    specs = prim.GetPrimStack()
    for i, spec in enumerate(specs):
        items = _arc_items(spec)
        if not items:
            continue
        own = specs[:i + 1]
        defined = {name for s in own for name in s.variantSets.keys()}
        selections = {}
        for s in own:
            for name, value in s.variantSelections.items():
                if name not in defined:
                    selections.setdefault(name, value)
        return (*_target(spec, items[0]), selections, not any(_changes(s) for s in own))
    return None


def change_key(prim: Usd.Prim) -> str:
    """What the opinions stronger than a placement change in the asset, as a short hash: their
    properties other than the transform, with values (asset paths resolved, targets below the
    placing prim made relative to it), and the overrides below the placing prim. Placements that
    change an asset the same way share one unit."""
    base = str(prim.GetPath())
    h = hashlib.sha1()

    def value(spec: Sdf.PropertySpec):
        if isinstance(spec, Sdf.RelationshipSpec):
            return sorted(str(t).replace(base, ".", 1) for t in spec.targetPathList.GetAddedOrExplicitItems())
        v = spec.default
        if isinstance(v, Sdf.AssetPath):
            return spec.layer.ComputeAbsolutePath(v.path) if v.path else ""
        return repr(v)

    def arcs(spec: Sdf.PrimSpec) -> list[str]:
        """What a child composes in: references and payloads (files resolved), inherits, specializes."""
        out = []
        for name, items in (("ref", spec.referenceList), ("payload", spec.payloadList)):
            for i in (*items.GetAddedOrExplicitItems(), *items.prependedItems, *items.appendedItems):
                file = spec.layer.ComputeAbsolutePath(i.assetPath) if i.assetPath else ""
                out.append(f"{name}:{os.path.normcase(file)}:{i.primPath}")
        for name, items in (("inherit", spec.inheritPathList), ("specialize", spec.specializesList)):
            out += [f"{name}:{p}" for p in (*items.GetAddedOrExplicitItems(), *items.prependedItems, *items.appendedItems)]
        return out

    def visit(spec: Sdf.PrimSpec, rel: str) -> None:
        for prop in sorted(spec.properties, key=lambda p: p.name):
            if rel or not TRANSFORM_ONLY.match(prop.name):
                h.update(f"{rel}.{prop.name}={value(prop)}\n".encode())
        if rel:  # the placing prim's own arcs are the placement, already in the unit's id
            h.update(f"{rel}:{spec.specifier}:{spec.typeName}:{sorted(spec.variantSelections.items())}:"
                     f"{arcs(spec)}\n".encode())
        for name, child in spec.nameChildren.items():
            visit(child, f"{rel}/{name}")

    for spec in own_specs(prim):
        visit(spec, "")
    return h.hexdigest()[:10]


def own_specs(prim: Usd.Prim) -> list[Sdf.PrimSpec]:
    """The opinions a placement depends on: down to the one that places the asset (all of them when
    the prim places nothing). The asset's own opinions are its unit's."""
    specs = prim.GetPrimStack()
    for i, spec in enumerate(specs):
        if _arc_items(spec):
            return specs[:i + 1]
    return specs


def variant_switch(prim: Usd.Prim) -> tuple[str, dict] | None:
    """(variant set, {variant: (layer, prim)}) when ``prim`` has a variant set whose every variant
    only places one asset: alternatives of one thing, such as levels of detail, or a full model and
    its proxy. Read from the layers, since the stage composes the selected variant only. The prim's
    other opinions apply to whichever option is placed; only its transform is kept. A prim that also
    gets content from arcs outside its variants is assembled from parts (a prop referencing its
    geometry, rig and material layers, with a variant set choosing one material): not a switch."""
    specs = prim.GetPrimStack()
    if any(_arc_items(spec) for spec in specs if not spec.path.ContainsPrimVariantSelection()):
        return None
    for spec in specs:
        if spec.path.ContainsPrimVariantSelection():
            continue
        for name, vset in spec.variantSets.items():
            options = {}
            for variant, v in vset.variants.items():
                items = _arc_items(v.primSpec)
                if len(items) != 1 or _changes(v.primSpec):
                    break
                options[variant] = _target(spec, items[0])
            else:
                if len(options) > 1:
                    return name, options
    return None


def placed(prim: Usd.Prim, separate_changed: bool) -> bool:
    """Whether ``prim`` is converted elsewhere: a switch or a placement. A changed asset is its own
    unit only when ``separate_changed`` (see ``separates_changed``)."""
    if variant_switch(prim):
        return True
    target = placement(prim)
    return bool(target) and (target[3] or separate_changed)


def separates_changed(selections: dict) -> bool:
    """A changed asset becomes a unit of its own (this layer, at the placing prim) unless the unit
    was placed with variant selections, which that unit's id could not carry: then it is composed
    into this unit's own geometry."""
    return not selections


def default_prim(layer_path: str) -> str | None:
    """The layer's default prim, or its only root prim when it names none (a stage that holds one
    prim and doesn't say it is the default). None for a file USD can't read (a MaterialX
    reference without the usdMtlx plugin): USD skips that arc too."""
    try:
        stamp = os.stat(layer_path).st_mtime_ns
    except OSError:
        stamp = None
    return _default_prim(layer_path, stamp)


@lru_cache(maxsize=None)
def _default_prim(layer_path: str, stamp) -> str | None:
    # Asked for every asset a unit places (thousands of times for a common prop): the layer's
    # metadata is enough, once per file; the whole layer only when it names no default prim.
    try:
        header = Sdf.Layer.OpenAsAnonymous(layer_path, metadataOnly=True)
        if header and header.defaultPrim:
            return f"/{header.defaultPrim}"
        layer = Sdf.Layer.FindOrOpen(layer_path)
    except Tf.ErrorException:
        return None
    if not layer:
        return None
    if layer.defaultPrim:
        return f"/{layer.defaultPrim}"
    roots = [p for p in layer.rootPrims if p.specifier == Sdf.SpecifierDef]
    return str(roots[0].path) if len(roots) == 1 else None


def top_prim(layer_path: str) -> str:
    """What a stage converts without a prim given: its default prim (or its only root prim), else
    the whole stage ("/": a stage that holds several roots, such as Caldera's world, players and
    cameras, and names none of them)."""
    return default_prim(layer_path) or "/"


def unit_id(paths: Paths, layer_path: str, prim: str, selections: dict | None = None) -> str:
    """``<layer relative to the root>``, plus ``<prim>`` unless it is the layer's default prim, plus
    ``{set=variant,...}`` for variant selections, plus ``@set=variant,...`` for the selections made
    everywhere (``paths.variants``). Ids are cache paths: kept short for Windows' 260 characters,
    which a layer file named after its prim would otherwise double."""
    sel = ",".join(f"{k}={v}" for k, v in sorted((selections or {}).items()))
    everywhere = ",".join(f"{k}={v}" for k, v in sorted(paths.variants.items()))
    own = "" if prim in ("/", default_prim(layer_path)) else prim  # "/": the whole stage
    return paths.rel(layer_path) + own + (f"{{{sel}}}" if sel else "") + (f"@{everywhere}" if everywhere else "")


def open_unit(layer_path: str, prim_path: str, selections: dict, variants: dict,
              masked: bool = False) -> tuple[Usd.Stage, Usd.Prim]:
    """The unit's stage and root: variant selections applied (the placement's on the root,
    ``variants`` -- {set: value}, from the command line -- everywhere), and the payloads the unit
    composes into itself loaded. ``masked``: only the unit's subtree is composed (the scan, which
    reads nothing outside it; a conversion needs the materials bound from elsewhere)."""
    if masked and prim_path != "/":
        stage = Usd.Stage.OpenMasked(layer_path, Usd.StagePopulationMask([prim_path]), Usd.Stage.LoadNone)
    else:
        stage = Usd.Stage.Open(layer_path, Usd.Stage.LoadNone)
    root = stage.GetPrimAtPath(prim_path)
    if not root:  # a changed asset inside a payload: load the payloads on the way, not below
        stage.Load(prim_path, Usd.LoadWithoutDescendants)
        root = stage.GetPrimAtPath(prim_path)
    if not root:
        raise RuntimeError(f"{layer_path} has no prim {prim_path}")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name, value in selections.items():
            if root.GetVariantSets().HasVariantSet(name):
                root.GetVariantSet(name).SetVariantSelection(value)
        if variants:
            _select_everywhere(stage, root, variants)
        root = stage.GetPrimAtPath(prim_path)  # an ancestor's selection recomposes it
    _load_inline(stage, root, separates_changed(selections))
    return stage, root


def _select_everywhere(stage: Usd.Stage, root: Usd.Prim, variants: dict) -> None:
    """Select ``variants`` on every prim of the unit that has the set, and on its ancestors (a unit
    opened at a placing prim sits under prims whose variants decide what it is: the shader ball's
    example_material binds the ball's material), except on switches (the assembly chooses those).
    Repeated until stable: a selection may bring in more variant sets."""
    path = root.GetPath()
    while True:
        root = stage.GetPrimAtPath(path)
        if not root:
            return
        ancestors, up = [], root.GetParent()
        while up and not up.IsPseudoRoot():
            ancestors.append(up)
            up = up.GetParent()
        todo = []
        for prim in [*ancestors, *Usd.PrimRange(root, PREDICATE)]:
            for name, value in variants.items():
                vset = prim.GetVariantSets().GetVariantSet(name) if prim.GetVariantSets().HasVariantSet(name) else None
                if vset and value in vset.GetVariantNames() and vset.GetVariantSelection() != value \
                        and not variant_switch(prim):
                    todo.append((prim.GetPath(), name, value))
        if not todo:
            return
        for path, name, value in todo:
            stage.GetPrimAtPath(path).GetVariantSet(name).SetVariantSelection(value)


def _load_inline(stage: Usd.Stage, root: Usd.Prim, separate_changed: bool) -> None:
    """Load the payloads the unit composes into itself, but not the assets it places. A point
    instancer's prototypes are drawn by the instancer, so they are loaded whole."""
    loaded: set = set()
    while True:
        todo = []
        it = iter(Usd.PrimRange(root, PREDICATE))
        for prim in it:
            if prim != root and placed(prim, separate_changed):
                it.PruneChildren()
            elif prim.IsA(UsdGeom.PointInstancer):
                for target in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets():
                    proto = stage.GetPrimAtPath(target)
                    if proto and proto.IsActive() and target not in loaded:
                        todo.append(target)
                it.PruneChildren()
            elif not prim.IsLoaded():
                todo.append(prim.GetPath())
                it.PruneChildren()
        if not todo:
            return
        loaded.update(todo)
        stage.LoadAndUnload(todo, [])


def _pruned(prim: Usd.Prim) -> bool:
    if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader) or prim.IsA(UsdShade.NodeGraph):
        return True
    img = UsdGeom.Imageable(prim)
    if img:
        if img.GetVisibilityAttr().Get() == UsdGeom.Tokens.invisible:
            return True
        if img.GetPurposeAttr().Get() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            return True
    return False


def _estimate(prim: Usd.Prim, root: Usd.Prim, bbox: UsdGeom.BBoxCache, frame: Frame) -> tuple[int, list | None]:
    """(triangles, box in the unit's glTF space) of what ``prim`` draws, read from the topology and
    the extents without converting: a polygon of n corners is n - 2 triangles, as the converter
    triangulates it. What the level-of-detail policies compare before anything is converted."""
    triangles = 0
    it = iter(Usd.PrimRange(prim, PREDICATE))
    for p in it:
        if _pruned(p):
            it.PruneChildren()
            continue
        if p.IsA(UsdGeom.Mesh):
            counts = np.asarray(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
            triangles += int((counts[counts >= 3] - 2).sum())
        elif p.IsA(UsdGeom.PointInstancer):
            pi = UsdGeom.PointInstancer(p)
            ids = np.asarray(pi.GetProtoIndicesAttr().Get() or [], dtype=np.int64)
            per = np.bincount(ids) if len(ids) else np.zeros(0, dtype=np.int64)
            for index, target in enumerate(pi.GetPrototypesRel().GetTargets()):
                proto = p.GetStage().GetPrimAtPath(target)
                if proto and index < len(per) and per[index]:
                    triangles += int(per[index]) * _estimate(proto, proto, bbox, frame)[0]
            it.PruneChildren()  # the prototypes, counted per instance above
    box = bbox.ComputeRelativeBound(prim, root).ComputeAlignedRange()
    if box.IsEmpty():
        return triangles, None
    lo, hi = np.array(box.GetMin(), dtype=np.float64), np.array(box.GetMax(), dtype=np.float64)
    corners = frame.points(np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]))
    return triangles, [corners.min(0).tolist(), corners.max(0).tolist()]


def _draws(prim: Usd.Prim) -> str | None:
    """What the converter would make of ``prim`` itself, if anything."""
    if prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.PointInstancer) or prim.IsA(UsdGeom.Camera):
        return prim.GetTypeName()
    if prim.HasAPI(UsdLux.LightAPI) or prim.IsA(UsdLux.BoundableLightBase) or prim.IsA(UsdLux.NonboundableLightBase):
        return "light"
    return None


def asset_variant_sets(layer_path: str, prim_path: str, selections: dict) -> dict[str, dict]:
    """The variant sets of a unit's root prim (a switch's aside: the assembly chooses those), with
    the values it offers and the one selected as the unit is placed."""
    with usd_diagnostics(lambda note: None):  # the page's question: the warnings belong to the scan
        stage = Usd.Stage.Open(layer_path, Usd.Stage.LoadNone)
    root = stage.GetPrimAtPath(prim_path)
    if not root:
        stage.Load(prim_path, Usd.LoadWithoutDescendants)
        root = stage.GetPrimAtPath(prim_path)
    if not root:
        return {}
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for name, value in selections.items():
            if root.GetVariantSets().HasVariantSet(name):
                root.GetVariantSet(name).SetVariantSelection(value)
    switch = variant_switch(root)
    return {name: {"values": root.GetVariantSet(name).GetVariantNames(),
                   "selected": root.GetVariantSet(name).GetVariantSelection()}
            for name in root.GetVariantSets().GetNames() if not (switch and switch[0] == name)}


def _rename(entries: list, rename: dict) -> None:
    for e in entries:
        if "unit" in e:
            e["unit"] = rename.get(e["unit"], e["unit"])
        if "options" in e:
            e["options"] = {v: rename.get(u, u) for v, u in e["options"].items()}
        _rename(e.get("children", []), rename)


class Scanner:
    def __init__(self, cache: Cache, forced, variants: dict, choices: dict | None = None,
                 lod: str = "finest", switches: dict | None = None):
        self.cache, self.forced, self.variants = cache, forced, variants
        cache.variants = variants  # part of every unit id
        self.units: dict[str, dict] = {}
        self.errors: dict[str, str] = {}  # units that failed to scan
        self.missing: set[str] = set()  # wanted, not scanned yet (load: the scan has work to do)
        self.stats = Counter()
        # Which options of a switch are followed: with the "authored" policy, the one the assembly
        # places (``switches``: --variant choices by switch name, as parse_variants gives them);
        # with a policy that compares the options, all of them. A district's full version or a
        # model's other levels of detail are then scanned (and converted) only when asked for.
        self.lod, self.switches = lod, switches or {}
        # Variants chosen for one asset: {unit id: {set: value}}. The unit is placed with those
        # selections added, which makes it another unit; the records stay as scanned (their keys,
        # and the leaves', don't change), only the scanner's view of them follows the choice.
        self.choices = choices or {}
        self.chosen_from: dict[str, tuple[str, dict]] = {}  # chosen unit -> (unit it replaces, its selections)

    def wanted(self, rec: dict) -> set[str]:
        """The units ``rec`` places that the assembly may use (Assembler._choice picks the same)."""
        out: set[str] = set()

        def walk(entries):
            for e in entries:
                if "unit" in e:
                    out.add(e["unit"])
                elif "switch" in e:
                    options = e["options"]
                    if self.lod != "authored":
                        out.update(options.values())
                        continue
                    named = self.switches.get(e["switch"], {})
                    pick = named.get(e["name"], named.get("*"))
                    pick = pick if pick in options else e["selected"]
                    out.update([options[pick]] if pick in options else options.values())
                else:
                    walk(e.get("children", []))
        walk(rec["children"])
        return out

    def choose(self, id: str, layer_path: str, prim: str, selections: dict) -> tuple[str, dict]:
        """A unit as placed, or the one the variants chosen for it make."""
        if id not in self.choices:
            return id, selections
        chosen = {**selections, **self.choices[id]}
        new = unit_id(self.cache, layer_path, prim, chosen)
        self.chosen_from[new] = (id, selections)
        return new, chosen

    def variants_for(self, id: str) -> dict:
        """The variants selected everywhere in a unit: the scene-wide ones, but those of a set chosen
        for this asset, which win."""
        if id not in self.chosen_from:
            return self.variants
        mine = self.choices.get(self.chosen_from[id][0], {})
        return {k: v for k, v in self.variants.items() if k not in mine}

    def adopt(self, id: str, rec: dict) -> dict:
        """``rec``, placing the units chosen in place of those it records."""
        rename, units = {}, []
        for unit, layer, prim, selections in rec["units"]:
            new, chosen = self.choose(unit, self.cache.abs(layer), prim, selections)
            if new != unit:
                rename[unit] = new
            units.append([new, layer, prim, chosen])
        if rename:
            rec["units"] = units
            _rename(rec["children"], rename)
        self.units[id] = rec
        return rec

    def run(self, layer_path: str, prim: str | None, pool, progress=None) -> str | None:
        """Scan the unit at ``layer_path``/``prim`` and every unit it places, breadth first: the
        units of one level that need scanning are scanned in parallel by ``pool``.
        ``progress(units done, units found so far)`` follows it."""
        prim = prim or top_prim(layer_path)
        top, selections = self.choose(unit_id(self.cache, layer_path, prim), layer_path, prim, {})
        frontier = [(top, layer_path, prim, selections)]
        while frontier:
            found, todo = [], {}
            for id, layer, target, selections in frontier:
                if id in self.units or id in todo:
                    continue
                identity = {"scan": SCAN_VERSION, "unit": id, "variants": self.variants}
                rec = self.cache.get("units", id)
                # A unit scanned before the scan estimated triangles is scanned again when a policy
                # compares levels (same key: its leaf stays valid).
                stale = rec is not None and rec.get("geo") and "estimate" not in rec and self.lod != "authored"
                if self.forced(id) or stale or not self.cache.fresh(rec, identity):
                    todo[id] = {"root": str(self.cache.root), "layer": layer, "prim": target,
                                "selections": selections, "variants": self.variants_for(id)}
                else:
                    self.adopt(id, rec)
                    self.stats["units cached"] += 1
                found.append(id)
            base = len(self.units)  # done so far; this level's todo is what's left that is known
            for k, (id, rec) in enumerate(zip(todo, pool.imap(scan_unit, todo.values(), chunksize=4)), 1):
                if progress:
                    progress(base + k, base + len(todo))
                if "error" in rec:
                    log(f"  failed: {id}: {rec['error']}")
                    self.errors[id] = rec["error"]
                    self.stats["units failed"] += 1
                    found.remove(id)
                    continue
                rec["key"] = make_key({"scan": SCAN_VERSION, "unit": id, "variants": self.variants}, rec["inputs"])
                self.cache.put("units", id, rec)
                self.adopt(id, rec)
                self.stats["units scanned"] += 1
            self.cache.commit()
            frontier = []
            for id in found:
                wanted = self.wanted(self.units[id])
                frontier += [(unit, self.cache.abs(layer), target, selections)
                             for unit, layer, target, selections in self.units[id]["units"] if unit in wanted]
        return top


def scan_unit(task: dict) -> dict:
    """Scan one unit (``task``: root, layer, prim, selections, variants). Runs in worker processes;
    a failure comes back as {"error": text}."""
    try:
        with usd_diagnostics(lambda note: None if "MaterialX" in note else log(note)):
            return _scan_unit(task)
    except Exception as e:  # noqa: BLE001 - one broken asset must not stop the others
        return {"error": f"{type(e).__name__}: {e}"}


def _scan_unit(task: dict) -> dict:
    paths, layer_path = Paths(task["root"], task["variants"]), task["layer"]
    stage, root = open_unit(layer_path, task["prim"], task["selections"], task["variants"], masked=MASKED)
    frame, xforms = Frame(stage), UsdGeom.XformCache()
    geo, layers = Counter(), set()
    separate = separates_changed(task["selections"])
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.EarliestTime(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                             useExtentsHint=True)
    estimate = {"triangles": 0, "bounds": None}

    def note_geo(prim: Usd.Prim) -> None:
        triangles, box = _estimate(prim, root, bbox, frame)
        estimate["triangles"] += triangles
        if box:
            b = estimate["bounds"]
            estimate["bounds"] = box if b is None else [np.minimum(b[0], box[0]).tolist(), np.maximum(b[1], box[1]).tolist()]

    units: dict[str, tuple] = {}  # id -> where to open it: (layer, prim, selections)
    outside: list = []  # point instancer prototypes outside the masked subtree
    variant_sets: dict[str, set] = {}  # every variant set seen, with its variants: the choices to offer

    def note_variant_sets(prim: Usd.Prim) -> None:
        sets = prim.GetVariantSets()
        for name in sets.GetNames():
            variant_sets.setdefault(name, set()).update(sets.GetVariantSet(name).GetVariantNames())

    def asset_id(prim: Usd.Prim, target: str, target_prim: str | None, selections: dict) -> tuple[str, str] | None:
        if not os.path.isfile(target):
            log(f"  warning: {prim.GetPath()}: missing {target}")
            return None
        if not Sdf.FileFormat.FindByExtension(os.path.splitext(target)[1].lstrip(".")):
            return None  # a format this USD can't read (.mtlx without usdMtlx): USD skips the arc too
        target_prim = target_prim or default_prim(target)
        return (unit_id(paths, target, target_prim, selections), target_prim) if target_prim else None

    def place(prim: Usd.Prim, target: str, target_prim: str | None, selections: dict) -> str | None:
        """The id of the unit placed at ``prim`` (``target``, as is), noted to be scanned."""
        found = asset_id(prim, target, target_prim, selections)
        if found:
            units.setdefault(found[0], (paths.rel(target), found[1], selections))
        return found[0] if found else None

    def place_changed(prim: Usd.Prim, target: str, target_prim: str | None, selections: dict) -> str | None:
        """The asset as this unit changes it: identified by the asset and the changes, opened at the
        first placing prim (the changes are the same at every other)."""
        found = asset_id(prim, target, target_prim, selections)
        if not found:
            return None
        id = f"{found[0]}~{change_key(prim)}"
        units.setdefault(id, (paths.rel(layer_path), str(prim.GetPath()), {}))
        return id

    def walk(prim: Usd.Prim, parent: Usd.Prim) -> dict | None:
        if _pruned(prim):
            return None
        entry: dict = {"name": prim.GetName()}
        matrix = frame.matrix(xforms.ComputeRelativeTransform(prim, parent)[0])
        if matrix:
            entry["matrix"] = matrix
        note_variant_sets(prim)
        switch, target = variant_switch(prim), placement(prim)
        # A placement depends on the opinions down to the one placing it; the asset's are its own unit's.
        layers.update(s.layer for s in (prim.GetPrimStack() if switch else own_specs(prim)))
        if switch:
            name, options = switch
            entry["switch"], entry["selected"] = name, prim.GetVariantSet(name).GetVariantSelection()
            entry["options"] = {v: u for v, u in ((v, place(prim, *arc, {})) for v, arc in options.items()) if u}
            return entry if entry["options"] else None
        if target and (target[3] or separate):
            entry["unit"] = (place if target[3] else place_changed)(prim, *target[:3])
            return entry if entry["unit"] else None
        kind = _draws(prim)
        if kind:
            geo[kind] += 1
            note_geo(prim)
            if prim.IsA(UsdGeom.PointInstancer):  # its prototypes are converted with it
                for t in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets():
                    if not stage.GetPrimAtPath(t):  # outside the unit's subtree: not in the masked stage
                        outside.append(t)
                    for p in Usd.PrimRange(stage.GetPrimAtPath(t), PREDICATE) if stage.GetPrimAtPath(t) else []:
                        layers.update(s.layer for s in p.GetPrimStack())
            return None  # the unit's geo leaf converts it; meshes have no placements below
        children = [c for c in (walk(ch, prim) for ch in prim.GetFilteredChildren(PREDICATE)) if c]
        if not children:
            return None
        entry["children"] = children
        return entry

    # A unit whose root is itself a switch (a district choosing between its full prefab and its
    # proxy) is that one entry.
    note_variant_sets(root)
    if variant_switch(root):
        children = [c for c in [walk(root, root)] if c]
    else:
        if root.IsA(UsdGeom.Gprim) or root.IsA(UsdGeom.PointInstancer):  # an asset that is one mesh
            geo[_draws(root)] += 1
            note_geo(root)
        children = [c for c in (walk(ch, root) for ch in root.GetFilteredChildren(PREDICATE)) if c]
    layers.update(spec.layer for spec in root.GetPrimStack())
    if outside:  # prototypes elsewhere in the stage: their layers, from the whole stage
        full, _ = open_unit(layer_path, task["prim"], task["selections"], task["variants"])
        for t in outside:
            for p in Usd.PrimRange(full.GetPrimAtPath(t), PREDICATE) if full.GetPrimAtPath(t) else []:
                layers.update(s.layer for s in p.GetPrimStack())
    return {"layer": paths.rel(layer_path), "prim": task["prim"], "selections": task["selections"],
            "children": children, "geo": dict(geo), "estimate": estimate,
            "variant_sets": {name: sorted(values) for name, values in variant_sets.items()},
            "units": sorted([id, layer, prim, sel] for id, (layer, prim, sel) in units.items()),
            "inputs": paths.inputs(l.realPath for l in layers if l.realPath)}
