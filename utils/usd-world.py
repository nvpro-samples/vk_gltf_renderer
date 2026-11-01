"""Convert each asset of a USD stage to its own glTF file and assemble them with glTF 2.1. See
``usd_gltf/world/cli.py`` or ``--help``.

    py -3.13 utils/usd-world.py build <stage.usd> --cache <dir> --root <data set root>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # also for the worker processes, which import this file

from usd_gltf.world.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
