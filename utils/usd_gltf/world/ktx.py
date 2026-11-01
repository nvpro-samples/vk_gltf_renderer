"""KTX2: the leaves' images as GPU-compressed KTX2 textures (``KHR_texture_basisu``), so large scenes
fit in GPU memory. The encoding is ``utils/gltf-ktx2-converter.py``'s, with its defaults; this step
only organizes it for a cache of many leaves sharing one texture folder:

- every image is encoded once, however many leaves use it (``<image>.ktx2`` next to it);
- an image larger than ``max_size`` is halved until it fits first (``<image>.<max_size>.ktx2``):
  nvtt_export doesn't resize;
- each leaf is rewritten in place (its name doesn't change), and records that it was;
- the PNG / JPEG originals are deleted once encoded: nothing needs them any more. A leaf converted
  again writes its image again; this step then reuses the ``.ktx2`` and deletes it again.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import logging
import os
import urllib.parse
from collections import Counter
from functools import lru_cache
from pathlib import Path

from PIL import Image

from ..convert import log
from .textures import fitted, size


@lru_cache(None)
def converter():
    """``utils/gltf-ktx2-converter.py`` as a module (its file name is not importable)."""
    path = Path(__file__).resolve().parents[2] / "gltf-ktx2-converter.py"
    spec = importlib.util.spec_from_file_location("gltf_ktx2_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def target(src: Path, max_size: int) -> tuple[Path, tuple | None]:
    """The KTX2 file of an image, and the (width, height) to shrink it to first, if any."""
    w, h, halvings = fitted(*(size(src) or (0, 0))[:2], max_size)
    if not halvings:
        return src.with_suffix(".ktx2"), None
    return src.with_name(f"{src.stem}.{max_size}.ktx2"), (w, h)


def shrink(src: Path, dims: tuple, out: Path) -> None:
    with Image.open(src) as im:
        if im.mode not in ("L", "LA", "RGB", "RGBA", "I", "F"):
            im = im.convert("RGBA")
        im.resize(dims, Image.Resampling.LANCZOS).save(out)


def ktx(cache, leaf_ids: list[str], jobs: int, progress=None, max_size: int = 0,
        delete_originals: bool = True) -> Counter:
    """Encode the images of ``leaf_ids``, none larger than ``max_size`` (0: as they are);
    ``progress(done, total)`` follows the encoding. Leaves share their textures: the originals are
    deleted only when ``leaf_ids`` are every leaf converted, else another may still use them."""
    k = converter()
    args = k.parse_args(["leaf.gltf"])  # the script's defaults
    stats = Counter()
    tool = k.resolve_tool(args)
    sources: dict[Path, list[bool]] = {}  # image -> [normal map, linear data], over every leaf using it
    plans = []
    for n, leaf_id in enumerate(leaf_ids, 1):
        if progress and (n % 200 == 0 or n == len(leaf_ids)):  # reading thousands of leaves takes a while
            progress(n, len(leaf_ids), "ktx-read")
        rec = cache.get("leaves", leaf_id)
        if not rec or rec.get("empty") or rec.get("ktx") or "error" in rec:
            continue
        if rec.get("images") == 0:  # the conversion counted none: done without reading the file
            rec["ktx"], rec["ktx_max"] = True, max_size
            cache.put("leaves", leaf_id, rec)
            continue
        path = cache.path(leaf_id)
        doc = json.loads(path.read_text(encoding="utf-8"))
        normal, linear = k.classify_texture_images(doc)
        images = []
        for i, image in enumerate(doc.get("images", [])):
            uri = image.get("uri", "")
            if not uri or k.is_data_uri(uri) or uri.lower().endswith(".ktx2"):
                continue
            src = k.uri_to_path(path.parent, uri).resolve()
            flags = sources.setdefault(src, [False, False])
            flags[0] |= i in normal
            flags[1] |= i in linear
            images.append((i, src))
        plans.append((leaf_id, rec, path, doc, images))

    targets = {src: target(src, max_size) for src in sources if src.exists()}
    todo = {src: flags for src, flags in sources.items() if src in targets and not targets[src][0].exists()}
    failed: set[Path] = set()
    if todo:
        logging.getLogger().setLevel(logging.WARNING)  # the script logs every image at INFO

        def encode(item):
            # Written under another name first: a run stopped midway leaves no truncated .ktx2 that
            # the next run would take as done.
            src, (is_normal, is_linear) = item
            out, dims = targets[src]
            part = out.with_name(out.stem + ".part.ktx2")
            small = src.with_name(src.stem + ".part.png") if dims else None
            try:
                if small:
                    shrink(src, dims, small)
                k.run_encoder(args, tool, small or src, part, is_normal, is_linear)
            finally:
                if small:
                    small.unlink(missing_ok=True)
            os.replace(part, out)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
        try:
            futures = {pool.submit(encode, item): item[0] for item in todo.items()}
            for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
                if future.exception():
                    failed.add(futures[future])
                    if len(failed) <= 3:
                        log(f"  failed: {futures[future].name}: {future.exception()}")
                if progress:
                    progress(n, len(todo), "ktx")
                if n % max(1, len(todo) // 20) == 0 or n == len(todo):
                    log(f"  {n}/{len(todo)} images")
        finally:  # stopped (progress raised): the images being encoded finish, the others don't start
            pool.shutdown(cancel_futures=True)
    stats["images encoded"] = len(todo) - len(failed)
    stats["images failed"] = len(failed)
    stats["images reused"] = len(sources) - len(todo)
    stats["images shrunk"] = sum(1 for src in todo if targets[src][1] and src not in failed)

    for leaf_id, rec, path, doc, images in plans:
        converted = set()
        for i, src in images:
            if src not in failed and src in targets and targets[src][0].exists():
                image = doc["images"][i]
                image["uri"] = urllib.parse.quote(Path(urllib.parse.unquote(image["uri"])).with_name(targets[src][0].name).as_posix())
                image["mimeType"] = "image/ktx2"
                converted.add(i)
        if converted:
            k.update_textures_for_basisu(doc, converted, keep_fallback_source=args.keep_fallback_source)
            k.add_unique_extension(doc, "extensionsUsed", k.KHR_TEXTURE_BASISU)
            if args.extension_required:
                k.add_unique_extension(doc, "extensionsRequired", k.KHR_TEXTURE_BASISU)
            part = path.with_name(path.name + ".part")  # a run stopped midway leaves the leaf whole
            part.write_text(json.dumps(doc, indent=1), encoding="utf-8")
            os.replace(part, path)
            stats["leaves rewritten"] += 1
        if len(converted) == len(images):
            # The size is the leaf's: changing it needs the originals, so the leaf converted again.
            rec["ktx"], rec["ktx_max"], rec["images"] = True, max_size, len(images)
            cache.put("leaves", leaf_id, rec)
    cache.commit()

    for src in sources if delete_originals else ():
        if src not in failed and src in targets and targets[src][0].exists():
            src.unlink()
            stats["originals deleted"] += 1
    return stats
