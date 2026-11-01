"""GPU memory of a scene's textures, read from the files without USD: what "Estimate" adds to the
geometry, and what the KTX2 step's automatic size (``ktx_max_size`` "auto") fits in the budget.

A KTX2 file is counted from its header (BC7 and the other block formats as such). An image not
encoded yet is counted as the KTX2 step would make it (BC7 with mipmaps, at most ``max_size``)
when KTX2 is on, else as the renderer uploads it (RGBA8 with mipmaps).
"""

from __future__ import annotations

import json
import os
import struct
import urllib.parse
from pathlib import Path

from PIL import Image

SIZES = (8192, 4096, 2048, 1024, 512, 256)  # what "auto" chooses from, largest first; 0 is full size
MIPS = 4 / 3  # a full mip chain adds a third

# VkFormat -> bytes per 4x4 block, for the block-compressed formats a KTX2 file may hold.
_BLOCK = {**{f: 8 for f in (131, 132, 133, 134, 139, 140)}, **{f: 16 for f in range(135, 139)},
          **{f: 16 for f in range(141, 147)}}

_docs: dict[tuple, list[Path]] = {}  # (leaf file, mtime) -> its image files
_sizes: dict[tuple, tuple] = {}  # (image file, mtime) -> (width, height, KTX2 vkFormat or None, levels)


def _stamp(path: Path) -> tuple | None:
    try:
        return (path, os.stat(path).st_mtime_ns)
    except OSError:
        return None


def images(leaf_file: Path) -> list[Path]:
    """The image files a glTF file uses (data URIs aside)."""
    key = _stamp(leaf_file)
    if key is None:
        return []
    if key not in _docs:
        doc = json.loads(leaf_file.read_text(encoding="utf-8"))
        _docs[key] = [(leaf_file.parent / urllib.parse.unquote(i["uri"])).resolve()
                      for i in doc.get("images", []) if i.get("uri") and not i["uri"].startswith("data:")]
    return _docs[key]


def size(image: Path) -> tuple | None:
    """(width, height, vkFormat or None, mip levels or None) of an image file, from its header."""
    key = _stamp(image)
    if key is None:
        return None
    if key not in _sizes:
        if image.suffix.lower() == ".ktx2":
            with open(image, "rb") as f:
                head = f.read(44)
            fmt, _, w, h, _, _, _, levels = struct.unpack_from("<8I", head, 12)
            _sizes[key] = (w, h, fmt, max(1, levels))
        else:
            try:
                with Image.open(image) as im:
                    _sizes[key] = (*im.size, None, None)
            except OSError:
                _sizes[key] = (0, 0, None, None)
    return _sizes[key]


def fitted(w: int, h: int, max_size: int) -> tuple[int, int, int]:
    """(width, height, halvings) of an image halved until it fits ``max_size`` (0: as it is)."""
    n = 0
    while max_size and max(w, h) > max_size:
        w, h, n = max(1, w // 2), max(1, h // 2), n + 1
    return w, h, n


def gpu_bytes(image: Path, ktx: bool, max_size: int) -> int:
    s = size(image)
    if not s or not s[0]:
        return 0
    w, h, fmt, levels = s
    if fmt is not None:  # encoded: exactly what it holds
        block = _BLOCK.get(fmt)
        total = 0
        for level in range(levels):
            lw, lh = max(1, w >> level), max(1, h >> level)
            total += ((lw + 3) // 4) * ((lh + 3) // 4) * block if block else lw * lh * 4
        return total
    if ktx:  # as the KTX2 step will encode it: BC7, a byte per texel
        w, h, _ = fitted(w, h, max_size)
        return int(w * h * MIPS)
    return int(w * h * 4 * MIPS)


def scene_bytes(leaf_files, ktx: bool, max_size: int, progress=None) -> int:
    """GPU bytes of the textures the files use, each image once. ``progress(done, total)`` follows
    the reading of the files (each is read once: later calls are fast)."""
    files = list(leaf_files)
    unique = set()
    for n, f in enumerate(files, 1):
        unique.update(images(Path(f)))
        if progress and (n % 200 == 0 or n == len(files)):
            progress(n, len(files))
    return sum(gpu_bytes(i, ktx, max_size) for i in unique)


def auto_size(leaf_files, allowance: int, progress=None) -> int:
    """The largest size (0: full) at which the not yet encoded images of ``leaf_files``, with the
    encoded ones, fit in ``allowance`` bytes; the smallest of SIZES when none does."""
    scene_bytes(leaf_files, True, 0, progress)  # reads every file once, with progress
    for max_size in (0, *SIZES):
        if scene_bytes(leaf_files, True, max_size) <= allowance:
            return max_size
    return SIZES[-1]
