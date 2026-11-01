"""Give a glTF file materials made from its material names, with a recipe of ``usd_gltf/looks/``
(``names``, ``caldera``) or any recipe JSON file. See ``usd_gltf/looks/__init__.py`` for the format.

    py utils/gltf-look.py scene.gltf scene_look.gltf --recipe names
    py utils/gltf-look.py scene.gltf --in-place --recipe my_recipe.json --report

Only the materials a converter made up (usd-to-gltf.py's for surfaces without a material) are
changed, unless ``--everything``. ``--report`` lists the names no family matched, to improve the
recipe with. Needs Python 3 only.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from usd_gltf import looks  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", help="glTF file (.gltf)")
    ap.add_argument("output", nargs="?", help="Output .gltf (default: <input>_look.gltf, next to the input: "
                                             "it shares the input's .bin and images)")
    ap.add_argument("--in-place", action="store_true", help="Rewrite the input")
    ap.add_argument("--recipe", default="names", help=f"A recipe of usd_gltf/looks ({', '.join(looks.available())}) "
                                                      "or a .json file (default: names)")
    ap.add_argument("--everything", action="store_true", help="Change every material, not only the made-up ones")
    ap.add_argument("--report", action="store_true", help="List the families used and the names no family matched")
    args = ap.parse_args(argv)

    src = Path(args.input)
    out = src if args.in_place else Path(args.output) if args.output else src.with_name(f"{src.stem}_look.gltf")
    if out.parent.resolve() != src.parent.resolve():
        ap.error("the output must be next to the input: it shares the input's .bin and images")
    doc = json.loads(src.read_text(encoding="utf-8"))
    stats = looks.apply(doc, looks.load(args.recipe), args.everything)
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"Wrote {out}: {stats['materials changed']} materials changed, {stats['materials kept']} kept, "
          f"{stats['primitives removed']} primitives removed")
    if args.report:
        for key, n in sorted(stats.items(), key=lambda kv: (not kv[0].startswith("family"), -kv[1])):
            if key.startswith(("family ", "unmatched ", "rule ")):
                print(f"  {n:5d} {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
