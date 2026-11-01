"""Convert a USD stage to one glTF 2.0 file. The converter lives in ``usd_gltf/convert.py`` (see its
docstring, or ``--help``); ``usd-world.py`` uses the same code to convert per asset.

    py -3.13 utils/usd-to-gltf.py World.usd out/World.gltf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from usd_gltf.convert import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
