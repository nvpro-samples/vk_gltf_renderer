"""usd-world: convert each asset of a USD stage to its own glTF file, then assemble them with glTF 2.1.

    py -3.13 utils/usd-world.py build <stage.usd> --cache <dir> --root <data set root> [--ktx] [--look <recipe>]
    py -3.13 utils/usd-world.py ui --cache <dir>

Commands (project.py describes the steps; each is cached, none is specific to a scene):

    build     every step: scan, convert, ktx (with --ktx), look (with --look), assemble
    scan      the asset graph
    convert   what each asset draws itself, as its own glTF leaf, in parallel
    ktx       the leaves' images as KTX2, in place, with gltf-ktx2-converter.py's defaults
    look      a material recipe applied to the leaves
    assemble  the glTF 2.1 scene placing the leaves: <cache>/<input name>[.<options>].gltf, or -o
    status    what the cache holds
    ui        a local web page to look at the scene graph and run all of the above

The stage and the settings are remembered by the cache: after the first command, ``--cache`` is
enough, and a setting given again changes it. ``--unit`` runs a step on some units (globs on ids)
and everything under them instead of the whole scene. A leaf is converted again only when its
layers changed, CONVERTER_VERSION changed, or ``--force`` names it; assemblies are rewritten only
when their content changes.
"""

from __future__ import annotations

import argparse
import multiprocessing
from collections import Counter
from pathlib import Path

from ..convert import log
from .assemble import check_lod
from .project import DEFAULT_JOBS, Project, available_looks

COMMANDS = ("build", "scan", "convert", "ktx", "look", "assemble")


def run(args) -> int:
    changes = {"input": args.input, "prim": args.prim, "output": args.output, "jobs": args.jobs,
               "materials": args.materials, "lod": args.lod, "nesting": args.nesting, "look": args.look,
               "hierarchy": args.hierarchy, "leaf_merge": args.leaf_merge,
               "variants": args.variant or None, "single_sided": args.single_sided or None,
               "keep_exposure": args.keep_exposure or None, "ktx": args.ktx or None, "ktx_max_size": args.ktx_max_size,
               "lod_camera": args.lod_camera, "lod_fov": args.lod_fov, "lod_pixels": args.lod_pixels,
               "lod_density": args.lod_density, "lod_budget_gb": args.lod_budget}
    project = Project.open(Path(args.cache), Path(args.root) if args.root else None, changes)
    s = project.settings
    if args.command == "build":
        steps = ["scan", "convert", *(["ktx"] if s["ktx"] else []), *(["look"] if s["look"] else []), "assemble"]
    else:
        steps = [args.command]
    with multiprocessing.Pool(s["jobs"]) as pool:
        project.run(steps, pool, args.unit, args.force)
    # Failures don't stop the rest of the scene, but automation must see them.
    failed = sum(project.stats.get(step, {}).get(key, 0) for step, key in
                 (("scan", "units failed"), ("convert", "failed"), ("ktx", "images failed")))
    return 1 if failed else 0


def status(args) -> int:
    project = Project.open(Path(args.cache))
    stats = Counter()
    for _, rec in project.cache.records("leaves"):
        stats["leaves"] += 1
        stats["empty leaves"] += bool(rec.get("empty"))
        stats["failed leaves"] += "error" in rec
        stats["leaves with KTX2"] += bool(rec.get("ktx"))
        stats["unique triangles"] += rec.get("triangles", 0)
    log(f"{project.cache.dir}: data set {project.cache.root}")
    log(f"  settings: {project.settings}")
    log(f"  units {project.cache.count('units'):,}")
    for k, v in stats.items():
        log(f"  {k}: {v:,}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="usd-world.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        p = sub.add_parser(name)
        p.add_argument("input", nargs="?", help="USD stage or prefab layer (remembered by the cache)")
        p.add_argument("--cache", required=True, help="Cache folder (created if missing)")
        p.add_argument("--root", help="Data set root; ids are paths relative to it. Needed for a new cache.")
        p.add_argument("--unit", action="append", default=[], metavar="GLOB",
                       help="Run on the units whose id matches, and everything they place (repeatable; default: all)")
        p.add_argument("--force", action="append", default=[], metavar="GLOB",
                       help="Convert (or scan) again the leaves / units whose id matches, e.g. 'units/*curtain_rod*'")
        p.add_argument("-o", "--output", help="The scene file (default: <cache>/<input name>[.<options>].gltf)")
        p.add_argument("--prim", help="Prim of the input to convert (default: its default prim, or its only root prim, else the whole stage)")
        p.add_argument("--jobs", type=int, help=f"Worker processes (default: {DEFAULT_JOBS}; each may decode large "
                                                "textures, so all cores of a big machine can run out of memory)")
        p.add_argument("--materials", choices=("auto", "preview"), help="As usd-to-gltf.py --materials")
        p.add_argument("--no-force-double-sided", dest="single_sided", action="store_true",
                       help="Each mesh as its USD doubleSided says (default: every surface double-sided, as "
                            "Omniverse RTX renders USD)")
        p.add_argument("--keep-exposure", action="store_true",
                       help="Physical light values (default: pre-exposed, the input stage's camera exposure baked "
                            "into lights, emission and sky, as usd-to-gltf.py does)")
        p.add_argument("--ktx", action="store_true",
                       help="build: convert the leaves' images to KTX2 (BC7, mipmaps, gltf-ktx2-converter.py's "
                            "defaults), in place; the originals are deleted. Large scenes need it to fit in GPU memory.")
        p.add_argument("--ktx-max-size", type=lambda v: v if v == "auto" else int(v), metavar="PIXELS|auto",
                       help="Largest KTX2 texture: larger images are halved until they fit (0: as authored; "
                            "default auto: the largest that fits the GPU budget beside the geometry)")
        p.add_argument("--look", choices=available_looks(), help="Material recipe from usd_gltf/looks")
        p.add_argument("--lod", type=check_lod,
                       help="Option placed at each switch (a variant set of alternatives) that --variant doesn't "
                            "choose: authored (the layer's selection, the default), finest or coarsest (by triangle "
                            "count), levelN (the N-th by triangle count, 0 = finest), or camera (by what the "
                            "--lod-camera views see, within --lod-budget)")
        p.add_argument("--lod-camera", help="lod camera: 'scene' (its cameras, the default), 'x,y,z' positions in "
                                            "metres, or vk_gltf_renderer --cameraString values, separated by ';'")
        p.add_argument("--lod-fov", type=float, help="lod camera: vertical field of view, degrees (default: the "
                                                     "scene camera's, else 60)")
        p.add_argument("--lod-pixels", type=int, help="lod camera: image height in pixels (default 1080)")
        p.add_argument("--lod-density", type=float, help="lod camera: triangles wanted per square pixel (default 1)")
        p.add_argument("--lod-budget", type=float, metavar="GB",
                       help="lod camera: GPU memory for geometry; the density is lowered until the scene fits")
        p.add_argument("--variant", action="append", metavar="SET=VALUE",
                       help="Select VALUE in variant set SET: at a switch, the option placed; anywhere else, the "
                            "selection the assets are composed with (e.g. material=renderLow). SET=VALUE:name,name "
                            "applies to the switches with those prim names only, e.g. districtLod=full:map_capital. "
                            "Replaces the remembered list (repeatable)")
        p.add_argument("--hierarchy", choices=("auto", "usd", "compact"),
                       help="usd: one node per prim; compact: wrapper groups folded, translation/rotation/scale, "
                            "bounding volumes on switches and roots only; auto (default): compact on large scenes")
        p.add_argument("--leaf-merge", choices=("none", "material"),
                       help="none (default): one mesh per USD mesh, as authored; material: each asset's meshes "
                            "merged into one mesh, a primitive per material (faster to load, less memory, same "
                            "frame time; the parts' names stay in extras)")
        p.add_argument("--nesting", choices=("inline", "files"),
                       help="inline: units that place others become group nodes and only leaves are placed "
                            "(default); files: they are placed as their own assembly files")
    p = sub.add_parser("status")
    p.add_argument("--cache", required=True)
    p = sub.add_parser("ui")
    p.add_argument("--cache", help="A project's cache folder to open (default: the start page, to open a USD stage)")
    p.add_argument("--root", help="Data set root, for a new cache")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="Don't open the page in the browser")
    p.add_argument("--renderer", help="vk_gltf_renderer executable for 'Open in renderer' (default: this repository's build)")
    args = ap.parse_args(argv)
    if args.command == "status":
        return status(args)
    if args.command == "ui":
        from .ui import serve
        return serve(args)
    return run(args)

