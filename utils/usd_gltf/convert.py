"""Convert a USD stage to glTF 2.0, keeping as much of its look as glTF can express.

The renderer does not load USD itself (OpenUSD and MDL are heavy dependencies), so this script
converts offline with the ``usd-core`` Python package:

    py -3.13 -m pip install usd-core numpy pillow [OpenEXR]
    py -3.13 utils/usd-to-gltf.py World.usd out/World.gltf

What is converted:
    Stage           - Composition (references, payloads, instancing) is resolved by OpenUSD;
                      ``--variant SET=VALUE`` selects variants first (e.g. material=renderLow).
                      Z-up stages are rotated to glTF's Y-up and metersPerUnit is baked in, so the
                      output is in metres. Identical meshes and materials are written once, even
                      when they are copies rather than USD instances. A PointInstancer becomes
                      EXT_mesh_gpu_instancing. USDZ package members are read in place. Images
                      with identical content are written once.
    Meshes          - Triangulated; split into one primitive per GeomSubset material; holes, left-
                      handed orientation, indexed and every-interpolation primvars handled. Meshes
                      without normals get smooth (or, for subdivisionScheme "none", flat) normals,
                      and so do authored normals that are zero or not finite.
                      Skinned meshes (UsdSkel) are posed once, at their skeleton's rest pose (or
                      its animation at the converted time): static geometry, no glTF skin.
                      primvars:doNotCastShadows becomes the vendor NV_node_shadow_casting. Every
                      material is double-sided, as Omniverse RTX renders USD (it ignores doubleSided
                      and culls nothing by default); ``--single-sided`` honors the attribute instead.
    Materials       - MDL first: OmniPBR (and the Base library materials that wrap it), OmniSurface
                      and the glass shaders map to glTF PBR and its material extensions, using the
                      original full-resolution textures. Then MaterialX (materialx.py): OpenPBR and
                      Standard Surface, in the stage or in a referenced .mtlx document, with their
                      medium (volume, scatter, subsurface), dispersion, anisotropy and thin film;
                      EXR textures are read with OpenEXR (ACEScg converted). Anything else falls back
                      to the UsdPreviewSurface network (Omniverse bakes one next to every MDL), then to
                      displayColor. A
                      GeomSubset bound to no material keeps its name on its displayColor material,
                      so files that only name their materials keep those names for later editing.
                      An input wired to a shader graph (a texture blended with grunge layers) takes
                      the graph's base texture, since glTF has one texture per slot.
                      Texture tiling and UVW projection are baked into TEXCOORD_0. UDIM textures
                      (``<UDIM>``) split each mesh per UV tile, one material per tile. A normal map
                      of one flat colour is left out: it has no detail, and when it is not the
                      neutral (0.5, 0.5, 1) it is a placeholder that tilts the whole surface.
    Lights          - Sphere -> point, disk/rect -> spot, distant -> directional (KHR_lights_punctual),
                      with radiance converted to candela from the light's area and color temperature
                      applied. A dome light becomes an OMI_environment_sky: a plain color, or a
                      panorama when it has a texture. A backdrop mesh with OmniProject_Environment
                      becomes the panorama sky instead (scaled and rotated; needs OpenEXR). A stage
                      lit only by its own lights gets a black sky, not the viewer's default.
    Cameras         - Perspective and orthographic. glTF has no camera exposure, so the cameras'
                      exposure (Omniverse scenes use physical light values and expose them down) is
                      baked into every light, emissive and the sky: the output is meant to be viewed
                      at exposure 1 (``--keep-exposure`` keeps the physical values instead). Without
                      a camera exposure, Omniverse's photographic settings (f-number, ISO, shutter)
                      give it, as the RTX tonemapper would.

Use ``--materials preview`` to ignore MDL and use the UsdPreviewSurface networks only.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import glob
import hashlib
import json
import math
import io
import os
import posixpath
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("PXR_USDC_EMIT_DEPRECATION_WARNINGS", "0")

import numpy as np
from PIL import Image
from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdSkel, Vt

from . import materialx

# --------------------------------------------------------------------------------------
# Coordinate conversion
# --------------------------------------------------------------------------------------

Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)  # (x, y, z) -> (x, z, -y)


class Frame:
    """Maps stage space to glTF space: p' = scale * R * p. Rotation-plus-uniform-scale, so a local
    matrix L becomes C L C^-1 and every node keeps a rigid, unscaled conversion."""

    def __init__(self, stage: Usd.Stage):
        self.scale = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
        self.rot = Z_UP_TO_Y_UP if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else np.eye(3)

    def points(self, p: np.ndarray) -> np.ndarray:
        return (p @ self.rot.T) * self.scale

    def vectors(self, v: np.ndarray) -> np.ndarray:
        return v @ self.rot.T

    def matrix(self, usd_m: Gf.Matrix4d) -> list[float] | None:
        """USD row-vector local matrix -> glTF column-major list, or None for identity."""
        m = np.array(usd_m, dtype=np.float64).T  # column-vector convention
        out = np.eye(4)
        out[:3, :3] = self.rot @ m[:3, :3] @ self.rot.T
        out[:3, 3] = self.scale * (self.rot @ m[:3, 3])
        if np.allclose(out, np.eye(4), atol=1e-9):
            return None
        # Below float precision anyway, and keeps the JSON short: 1.2e-16 is 0, translations to the micrometre.
        out[:3, :3] = np.round(out[:3, :3], 9)
        out[:3, 3] = np.round(out[:3, 3], 6)
        return [float(x) + 0.0 for x in out.T.flatten()]  # + 0.0: no -0.0


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


@contextlib.contextmanager
def usd_diagnostics(note=log):
    """USD's warnings while composing, given to ``note`` instead of printed by USD: a reference
    to a .mtlx document (usd-core has no MaterialX plugin; materialx.py reads the document itself)
    becomes one line naming the files, rather than an error-looking warning per arc."""
    from pxr import UsdUtils
    delegate = UsdUtils.CoalescingDiagnosticDelegate()
    try:
        yield
    finally:
        mtlx = set()
        for d in delegate.TakeUncoalescedDiagnostics():
            found = re.search(r"@([^@]+?\.mtlx)", d.commentary)
            if found and "Could not open asset" in d.commentary:
                mtlx.add(os.path.basename(found.group(1)))
            else:
                note(f"  USD: {d.commentary}")
        if mtlx:
            note(f"  MaterialX documents read directly (usd-core has no MaterialX plugin): {', '.join(sorted(mtlx))}")


def _package_parts(path: str) -> tuple[str, str] | None:
    """(package, member) of a USDZ member path such as ``scene.usdz[textures/a.png]``."""
    if path.endswith("]") and "[" in path:
        package, member = path[:-1].split("[", 1)
        return package, member
    return None


# Files other than USD layers that a conversion read (textures, MDL wrappers, MaterialX documents),
# for usd-world's cache keys: a USDZ member counts as its package.
FILES_READ: set[str] = set()


def note_read(path: str) -> None:
    parts = _package_parts(path)
    FILES_READ.add(os.path.abspath(parts[0] if parts else path))


def read_asset(path: str) -> bytes | None:
    """Bytes of a file on disk or of a USDZ package member, through OpenUSD's resolver."""
    note_read(path)
    if os.path.isfile(path):
        return Path(path).read_bytes()
    if _package_parts(path):
        asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(path))
        return asset.GetBuffer() if asset else None
    return None


def asset_exists(path: str) -> bool:
    return os.path.isfile(path) or (_package_parts(path) is not None and read_asset(path) is not None)


def asset_sibling(base: str, relative: str) -> str:
    """``relative`` resolved against the directory of ``base``, inside the same package if any."""
    parts = _package_parts(base)
    if parts:
        member = posixpath.normpath(posixpath.join(posixpath.dirname(parts[1]), relative.lstrip("/")))
        return f"{parts[0]}[{member}]"
    return os.path.normpath(os.path.join(os.path.dirname(base), relative.lstrip("/")))


def asset_basename(path: str) -> str:
    parts = _package_parts(path)
    return posixpath.basename(parts[1]) if parts else os.path.basename(path)


UDIM = "<UDIM>"


def udim_tiles(pattern: str) -> list[int]:
    """The UDIM tiles (1001, 1002, ...) present on disk for a ``<UDIM>`` texture path."""
    head, tail = pattern.split(UDIM, 1)
    found = []
    for f in glob.glob(glob.escape(head) + "[0-9][0-9][0-9][0-9]" + glob.escape(tail)):
        tile = f[len(head):len(head) + 4]
        if tile.isdigit():
            found.append(int(tile))
    return sorted(found)


def asset_path(value, attr: Usd.Attribute | None = None) -> str | None:
    """Absolute path of an Sdf.AssetPath (a file, or a USDZ member), resolved against the layer
    that authored it. A UDIM texture keeps its ``<UDIM>`` token when at least one tile exists."""
    if not isinstance(value, Sdf.AssetPath) or not value.path:
        return None
    if value.resolvedPath and asset_exists(value.resolvedPath):
        return value.resolvedPath if _package_parts(value.resolvedPath) else os.path.normpath(value.resolvedPath)
    if attr is not None:
        for spec in attr.GetPropertyStack(Usd.TimeCode.Default()):
            candidate = spec.layer.ComputeAbsolutePath(value.path)
            if not candidate:
                continue
            candidate = os.path.normpath(candidate)
            if os.path.isfile(candidate) or (UDIM in candidate and udim_tiles(candidate)):
                return candidate
    return None


def to_rgb(v, default=(1.0, 1.0, 1.0)) -> list[float]:
    if v is None:
        return list(default)
    if isinstance(v, (int, float)):
        return [float(v)] * 3
    return [float(x) for x in list(v)[:3]]


# --------------------------------------------------------------------------------------
# MDL parsing: enough to read the arguments of a wrapper material (e.g. Base library Oak.mdl is
# ``export material Oak(*) = ::OmniPBR::OmniPBR(diffuse_texture: texture_2d("./Oak/..."), ...)``).
# --------------------------------------------------------------------------------------

OMNIPBR_DEFAULTS = {
    "diffuse_color_constant": (0.2, 0.2, 0.2), "diffuse_texture": None, "albedo_desaturation": 0.0,
    "albedo_add": 0.0, "albedo_brightness": 1.0, "diffuse_tint": (1.0, 1.0, 1.0),
    "reflection_roughness_constant": 0.5, "reflection_roughness_texture_influence": 0.0,
    "reflectionroughness_texture": None, "metallic_constant": 0.0, "metallic_texture_influence": 0.0,
    "metallic_texture": None, "specular_level": 0.5, "enable_ORM_texture": False, "ORM_texture": None,
    "ao_to_diffuse": 0.0, "ao_texture": None, "enable_emission": False, "emissive_color": (1.0, 0.1, 0.1),
    "emissive_color_texture": None, "emissive_mask_texture": None, "emissive_intensity": 40.0,
    "bump_factor": 1.0, "normalmap_texture": None, "enable_opacity": False, "opacity_constant": 1.0,
    "enable_opacity_texture": False, "opacity_texture": None, "opacity_mode": 1, "opacity_threshold": 0.0,
    "project_uvw": False, "world_or_object": False, "uv_space_index": 0, "texture_translate": (0.0, 0.0),
    "texture_rotate": 0.0, "texture_scale": (1.0, 1.0),
}
GLASS_FAMILIES = ("OmniGlass", "GlassWithVolume", "Tinted_Glass", "Clear_Glass")
_MDL_CALL = re.compile(r"=\s*((?:::)?[A-Za-z_][\w:]*)\s*\(")
_MDL_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?f?"


def _split_top_level(s: str) -> list[str]:
    parts, depth, cur, in_str = [], 0, [], False
    for ch in s:
        if ch == '"':
            in_str = not in_str
        if not in_str:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == "," and depth == 0:
                parts.append("".join(cur))
                cur = []
                continue
        cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def _mdl_value(text: str, mdl_file: str):
    text = re.sub(r"/\*.*?\*/", "", text).strip()
    if text.startswith("texture_2d"):
        m = re.search(r'"([^"]*)"', text)
        if not m or not m.group(1):
            return None
        path = asset_sibling(mdl_file, m.group(1))
        return path if asset_exists(path) or (UDIM in path and udim_tiles(path)) else None
    if text in ("true", "false"):
        return text == "true"
    # Numbers come from inside the constructor only: the "2" of "float2(0.f)" is not a component.
    args = text[text.index("(") + 1:text.rindex(")")] if "(" in text and ")" in text else text
    nums = [float(x.rstrip("f")) for x in re.findall(_MDL_NUM, args)]
    if text.startswith(("color", "float3")):
        return tuple(nums * 3 if len(nums) == 1 else nums[:3])
    if text.startswith("float2"):
        return tuple(nums * 2 if len(nums) == 1 else nums[:2])
    if len(nums) == 1 and re.fullmatch(_MDL_NUM, text):
        return nums[0]
    if "::" in text and text.split("::")[-1].isdigit() is False:  # enum such as ::base::mono_average
        return text.split("::")[-1]
    return None


def parse_mdl_wrapper(path: str, name: str) -> tuple[str, dict]:
    note_read(path)  # the parse is cached per process, the reads are counted per conversion
    return _parse_mdl_wrapper(path, name)


@functools.lru_cache(maxsize=None)
def _parse_mdl_wrapper(path: str, name: str) -> tuple[str, dict]:
    """Return (family, arguments) of ``export material <name>(...) = ::a::b::Family(args)``.
    Memoized: the result is shared, so callers copy the arguments before changing them."""
    family, args = "", {}
    try:
        src = (read_asset(path) or b"").decode("utf-8", errors="replace")
    except OSError:
        return family, args
    src = re.sub(r"//[^\n]*", "", src)
    for m in re.finditer(r"export\s+material\s+(\w+)\s*\(", src):
        if m.group(1) != name:
            continue
        call = _MDL_CALL.search(src, m.end())
        if not call:
            continue
        family = call.group(1).split("::")[-1]
        depth, i = 1, call.end()
        while i < len(src) and depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        for part in _split_top_level(src[call.end():i - 1]):
            if ":" in part and not part.strip().startswith("::"):
                k, v = part.split(":", 1)
                val = _mdl_value(v, path)
                if val is not None or "texture_2d" in v:
                    args[k.strip()] = val
        break
    return family, args


# --------------------------------------------------------------------------------------
# glTF document builder
# --------------------------------------------------------------------------------------

class Gltf:
    """``shared_textures``: a folder images are written to by content (``<name>-<hash>``), shared by
    every document written there, instead of ``<name>_textures/`` next to this one."""

    def __init__(self, out_path: Path, shared_textures: Path | None = None):
        self.out_path = out_path
        self.shared_textures = shared_textures is not None
        self.tex_dir = shared_textures or out_path.parent / f"{out_path.stem}_textures"
        self.doc = {"asset": {"version": "2.0", "generator": "usd-to-gltf.py (vk_gltf_renderer)"},
                    "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "materials": [],
                    "accessors": [], "bufferViews": [], "buffers": [], "images": [], "textures": [],
                    "samplers": [], "cameras": []}
        self.extensions: set[str] = set()
        self.chunks: list[bytes] = []
        self.offset = 0
        self._images: dict[str, int] = {}
        self._image_by_content: dict[str, int] = {}  # sha1 of the pixels' source -> image index
        self._textures: dict[tuple, int] = {}
        self._samplers: dict[tuple, int] = {}
        self._names: set[str] = set()
        self.lights: list[dict] = []
        self.sky: dict | None = None

    # ---- binary data ----
    def _view(self, data: bytes, target: int | None) -> int:
        pad = (-self.offset) % 4
        if pad:
            self.chunks.append(b"\0" * pad)
            self.offset += pad
        view = {"buffer": 0, "byteOffset": self.offset, "byteLength": len(data)}
        if target:
            view["target"] = target
        self.chunks.append(data)
        self.offset += len(data)
        self.doc["bufferViews"].append(view)
        return len(self.doc["bufferViews"]) - 1

    def accessor(self, arr: np.ndarray, kind: str, minmax: bool = False) -> int:
        if kind == "SCALAR":
            ctype = 5125 if arr.max(initial=0) > 65535 else 5123
            arr = arr.astype(np.uint32 if ctype == 5125 else np.uint16)
            view = self._view(arr.tobytes(), 34963)
        else:
            ctype = 5126
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            view = self._view(arr.tobytes(), 34962)
        acc = {"bufferView": view, "componentType": ctype, "count": int(arr.shape[0]), "type": kind}
        if minmax:
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    # ---- textures ----
    def _unique_name(self, stem: str, ext: str) -> str:
        stem = re.sub(r"[^\w.-]+", "_", stem)
        name, n = f"{stem}{ext}", 1
        while name.lower() in self._names:
            name, n = f"{stem}_{n}{ext}", n + 1
        self._names.add(name.lower())
        return name

    def image_from_file(self, src: str) -> int | None:
        if src in self._images:
            return self._images[src]
        stem, ext = os.path.splitext(asset_basename(src))
        data = read_asset(src)
        if not data:
            log(f"  warning: cannot read texture {src}")
            self._images[src] = None
            return None
        if ext.lower() in (".png", ".jpg", ".jpeg"):
            write = lambda path: path.write_bytes(data)  # noqa: E731
        elif ext.lower() == ".exr":
            img = _exr_as_srgb(src, data)
            if img is None:
                log(f"  warning: cannot read texture {src} (an EXR texture needs the OpenEXR package)")
                self._images[src] = None
                return None
            ext, write = ".png", img.save
        else:
            try:
                img = Image.open(io.BytesIO(data))
            except Exception as e:  # noqa: BLE001 - any decoder failure means "no texture"
                log(f"  warning: cannot read texture {src}: {e}")
                self._images[src] = None
                return None
            ext, write = ".png", img.save
        self._images[src] = self._add_image(hashlib.sha1(data).hexdigest(), stem, ext.lower(), write)
        return self._images[src]

    def _add_image(self, content: str, stem: str, ext: str, write) -> int:
        """Index of the image whose pixels hash to ``content``, written once: copies of an asset
        in different folders carry byte-identical textures, which would otherwise upload twice."""
        if content not in self._image_by_content:
            self.tex_dir.mkdir(parents=True, exist_ok=True)
            if self.shared_textures:  # one file per content for every document: named by it alone
                name = f"{re.sub(r'[^0-9a-f]', '', content)[:16]}{ext}"
                if not (self.tex_dir / name).exists():
                    tmp = self.tex_dir / f".{name}.{os.getpid()}{ext}"  # other processes may write it too
                    write(tmp)
                    try:
                        os.replace(tmp, self.tex_dir / name)
                    except PermissionError:  # another process wrote it first (Windows refuses the swap)
                        if not (self.tex_dir / name).exists():
                            raise
                        tmp.unlink()
            else:
                name = self._unique_name(stem, ext)
                write(self.tex_dir / name)
            uri = Path(os.path.relpath(self.tex_dir / name, self.out_path.parent)).as_posix()
            # A file named by content keeps its readable name in the document.
            self.doc["images"].append({"uri": uri, "name": stem} if self.shared_textures else {"uri": uri})
            self._image_by_content[content] = len(self.doc["images"]) - 1
        return self._image_by_content[content]

    def image_from_array(self, key: tuple, stem: str, make) -> int | None:
        """Image produced by ``make()`` (an HxWx{3,4} uint8 array), cached by ``key``."""
        k = "op:" + hashlib.sha1(repr(key).encode()).hexdigest()
        if k in self._images:
            return self._images[k]
        arr = make()
        if arr is None:
            self._images[k] = None
            return None
        content = hashlib.sha1(repr(arr.shape).encode() + arr.tobytes()).hexdigest()
        self._images[k] = self._add_image("px:" + content, stem, ".png", lambda path: Image.fromarray(arr).save(path))
        return self._images[k]

    def panorama_from_exr(self, src: str, scale: list[float], roll: float) -> str | None:
        """``src`` as an EXR whose pixels are times ``scale`` (RGB) and whose columns are shifted by
        ``roll`` of the width; its URI, or None when the OpenEXR package is not installed."""
        try:
            import OpenEXR
        except ImportError:
            return None
        note_read(src)
        with OpenEXR.File(io.BytesIO(read_asset(src) or b"") if _package_parts(src) else src) as exr:
            channels = exr.channels()
            name = next(iter(channels))
            pixels = np.asarray(channels[name].pixels, dtype=np.float32)
        rgb = pixels[..., :3] * np.asarray(scale, dtype=np.float32)
        rgb = np.roll(rgb, -int(round(roll * rgb.shape[1])), axis=1)
        self.tex_dir.mkdir(parents=True, exist_ok=True)
        half_ok = float(np.max(rgb)) <= float(np.finfo(np.float16).max)  # else float32, not inf
        rgb = rgb.astype(np.float16 if half_ok else np.float32)
        stem = re.sub(r"[^\w.-]+", "_", os.path.splitext(asset_basename(src))[0])
        if self.shared_textures:  # every document's: named by its pixels, as _add_image does
            out = f"{stem}.{hashlib.sha1(rgb.tobytes()).hexdigest()[:16]}.exr"
        else:
            out = self._unique_name(stem, ".exr")
        if not (self.shared_textures and (self.tex_dir / out).exists()):
            tmp = self.tex_dir / f".{out}.{os.getpid()}.exr"  # other processes may write it too
            header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
            with OpenEXR.File(header, {"RGB": rgb}) as exr:
                exr.write(str(tmp))
            os.replace(tmp, self.tex_dir / out)
        return Path(os.path.relpath(self.tex_dir / out, self.out_path.parent)).as_posix()

    def texture(self, image: int | None, wrap=(10497, 10497)) -> dict | None:
        if image is None:
            return None
        if wrap not in self._samplers:
            self.doc["samplers"].append({"wrapS": wrap[0], "wrapT": wrap[1], "magFilter": 9729, "minFilter": 9987})
            self._samplers[wrap] = len(self.doc["samplers"]) - 1
        key = (image, wrap)
        if key not in self._textures:
            self.doc["textures"].append({"source": image, "sampler": self._samplers[wrap]})
            self._textures[key] = len(self.doc["textures"]) - 1
        return {"index": self._textures[key]}

    # ---- output ----
    def write(self) -> None:
        bin_name = self.out_path.with_suffix(".bin").name
        # A buffer must not be empty: a stage of cameras and lights only has none.
        self.doc["buffers"] = [{"uri": bin_name, "byteLength": self.offset}] if self.offset else []
        if self.lights:
            self.doc.setdefault("extensions", {})["KHR_lights_punctual"] = {"lights": self.lights}
            self.extensions.add("KHR_lights_punctual")
        if self.sky:
            self.doc.setdefault("extensions", {})["OMI_environment_sky"] = {"skies": [self.sky]}
            self.doc["scenes"][0]["extensions"] = {"OMI_environment_sky": {"sky": 0}}
            self.extensions.add("OMI_environment_sky")
            if "NV_environment_sky_panorama" in self.sky.get("extensions", {}):
                self.extensions.add("NV_environment_sky_panorama")
        if self.extensions:
            self.doc["extensionsUsed"] = sorted(self.extensions)
        for key in ("images", "textures", "samplers", "cameras", "materials", "meshes", "accessors", "bufferViews", "buffers"):
            if not self.doc[key]:
                del self.doc[key]
        if self.offset:
            with open(self.out_path.with_suffix(".bin"), "wb") as f:
                for c in self.chunks:
                    f.write(c)
        self.out_path.write_text(json.dumps(self.doc, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Image operations (numpy over Pillow), for what glTF needs packed differently than the source
# --------------------------------------------------------------------------------------

def _load(path: str) -> np.ndarray | None:
    """RGBA in 0..1. Single-channel 16-bit and 32-bit images are scaled by their own range:
    Pillow's convert("RGBA") would clip them to 0..255, turning nearly every pixel white."""
    try:
        img = Image.open(io.BytesIO(read_asset(path) or b""))
        if img.mode.startswith("I;16") or img.mode in ("I", "F"):
            grey = np.asarray(img, dtype=np.float32)
            if img.mode != "F":
                grey = grey / 65535.0  # 8-bit grey opens as "L", so an integer mode is 16-bit
            grey = np.clip(grey, 0.0, 1.0)
            return np.dstack([grey, grey, grey, np.ones_like(grey)])
        return np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    except Exception as e:  # noqa: BLE001
        log(f"  warning: cannot read texture {path}: {e}")
        return None


@functools.lru_cache(maxsize=None)
def _flat_image(path: str) -> bool:
    """True when every pixel of ``path`` has the same value (False when Pillow cannot read it)."""
    try:
        bands = Image.open(io.BytesIO(read_asset(path) or b"")).getextrema()
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(bands[0], tuple):  # a single band
        bands = (bands,)
    return all(lo == hi for lo, hi in bands)


_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _channel(img: np.ndarray, ch: str) -> np.ndarray:
    """A scalar from an RGBA image: a channel letter, or an MDL mono mode."""
    if len(ch) == 1 and ch in "rgba":
        return img[..., "rgba".index(ch)]
    return {"alpha": img[..., 3], "average": img[..., :3].mean(-1), "maximum": img[..., :3].max(-1),
            "luminance": img[..., :3] @ _LUMA}.get(ch, img[..., 0])


def _to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


# ACEScg (AP1, D60) to linear Rec.709 (D65, Bradford adapted).
_ACESCG_TO_REC709 = np.array([[1.70505, -0.62179, -0.08326],
                              [-0.13026, 1.14080, -0.01055],
                              [-0.02400, -0.12897, 1.15297]], dtype=np.float32)


def _exr_as_srgb(src: str, data: bytes) -> Image.Image | None:
    """An EXR texture as an 8-bit sRGB image (glTF color textures are sRGB). Its primaries are
    ACEScg when the name says so (``name.ACEScg.exr``, as asset kits name them), else linear
    Rec.709, MaterialX's default. None without the OpenEXR package."""
    try:
        import OpenEXR
    except ImportError:
        return None
    with OpenEXR.File(io.BytesIO(data)) as exr:
        channels = exr.channels()
        name = "RGBA" if "RGBA" in channels else "RGB" if "RGB" in channels else next(iter(channels))
        pixels = np.asarray(channels[name].pixels, dtype=np.float32)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=-1)
    rgb = pixels[..., :3]
    if "acescg" in os.path.basename(src).lower():
        rgb = rgb @ _ACESCG_TO_REC709.T
    out = (_to_srgb(rgb) * 255.0 + 0.5).astype(np.uint8)
    if pixels.shape[-1] == 4:
        alpha = (np.clip(pixels[..., 3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(np.dstack([out, alpha]), "RGBA")
    return Image.fromarray(out, "RGB")


def _resize(img: np.ndarray, shape) -> np.ndarray:
    if img.shape[:2] == tuple(shape):
        return img
    pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return np.asarray(pil.resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0


def _u8(a: np.ndarray) -> np.ndarray:
    return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def make_metal_rough(rough: tuple | None, metal: tuple | None):
    """rough/metal = (path, channel, constant, influence[, scale, bias]):
    value = lerp(constant, tex * scale + bias, influence)."""
    def make():
        srcs = [(s, _load(s[0])) for s in (rough, metal) if s]
        srcs = [(s, img) for s, img in srcs if img is not None]
        if not srcs:
            return None
        shape = max((img.shape[:2] for _, img in srcs), key=lambda s: s[0] * s[1])
        out = np.ones(shape + (3,), dtype=np.float32)
        for idx, src in ((1, rough), (2, metal)):
            if not src:
                continue
            img = _load(src[0])
            if img is None:
                continue
            tex = _channel(_resize(img, shape), src[1])
            if len(src) > 4:
                tex = tex * src[4] + src[5]
            out[..., idx] = src[2] * (1.0 - src[3]) + tex * src[3]
        return _u8(out)
    return make


def make_base_color(base: str | None, scale: float, add: float, desaturation: float,
                    ao: tuple | None, opacity: tuple | None):
    """OmniPBR's albedo, computed in linear space like the MDL: lerp(t, luminance(t), desaturation)
    with t = texture * scale + add, darkened by ``ao`` = (path, mode, weight). Alpha comes from
    ``opacity`` = (path, mode). Modes are channel letters or alpha/average/luminance/maximum."""
    def make():
        img = _load(base) if base else None
        alpha_img = _load(opacity[0]) if opacity else None
        if img is None and alpha_img is None:
            return None
        shape = (img if img is not None else alpha_img).shape[:2]
        rgb = _to_linear(img[..., :3]) if img is not None else np.ones(shape + (3,), dtype=np.float32)
        rgb = rgb * scale + add
        if desaturation:
            rgb = rgb * (1.0 - desaturation) + (rgb @ _LUMA)[..., None] * desaturation
        if ao:
            ao_img = _load(ao[0])
            if ao_img is not None:
                occ = _channel(_resize(ao_img, shape), ao[1])
                rgb = rgb * (1.0 - ao[2] + ao[2] * occ)[..., None]
        alpha = img[..., 3] if img is not None else np.ones(shape, dtype=np.float32)
        if alpha_img is not None:
            alpha = _channel(_resize(alpha_img, shape), opacity[1])
        return _u8(np.concatenate([_to_srgb(rgb), alpha[..., None]], axis=-1))
    return make


def make_single_channel(path: str, ch: str):
    def make():
        img = _load(path)
        return None if img is None else _u8(np.repeat(_channel(img, ch)[..., None], 3, axis=-1))
    return make


# --------------------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------------------

class UvSpec:
    """How the mesh must build TEXCOORD_0 for a material: which primvar (or a projection), then
    uv' = scale * R(-rotate) * uv + translate (OmniPBR's order), before the glTF V flip."""

    def __init__(self, primvar="st", index=0, scale=(1.0, 1.0), rotate=0.0, translate=(0.0, 0.0), project=None):
        self.primvar, self.index, self.project = primvar, index, project
        self.udim = False  # textures are UDIM tiles: the mesh splits its triangles per tile
        self.scale = (float(scale[0]), float(scale[1]))
        self.rotate, self.translate = float(rotate), (float(translate[0]), float(translate[1]))

    def key(self):
        return (self.primvar, self.index, self.scale, self.rotate, self.translate, self.project)


OPACITY_MODES = {0: "alpha", 1: "average", 2: "luminance", 3: "maximum",
                 "mono_alpha": "alpha", "mono_average": "average", "mono_luminance": "luminance", "mono_maximum": "maximum"}


# Inputs that carry a texture node's file or the base layer of a blend or adjust node, tried in order.
_GRAPH_BASE_INPUTS = ("file", "texture", "texture_1", "color_1", "base", "a", "color", "tint", "layer")


# A graph wired to a constant input feeds the texture slot of that value.
_GRAPH_TEXTURE_SLOTS = {"diffuse_color_constant": "diffuse_texture",
                        "reflection_roughness_constant": "reflectionroughness_texture",
                        "metallic_constant": "metallic_texture"}
_GRAPH_TEXTURE_INFLUENCE = {"reflectionroughness_texture": "reflection_roughness_texture_influence",
                            "metallic_texture": "metallic_texture_influence"}


def _graph_base_texture(inp: UsdShade.Input, depth: int = 0) -> str | None:
    """The texture at the bottom of the graph connected to ``inp``: through node graphs, into their
    outputs, and down each node's base input until a texture file. None when there is none."""
    if depth > 16 or not inp.HasConnectedSource():
        return None
    for source in inp.GetConnectedSources()[0]:
        prim = source.source.GetPrim()
        if prim.IsA(UsdShade.NodeGraph) and not prim.IsA(UsdShade.Shader):
            # a node graph's output is wired to a node inside it
            output = UsdShade.NodeGraph(prim).GetOutput(source.sourceName)
            if output and output.HasConnectedSource():
                inner = output.GetConnectedSources()[0][0]
                prim = inner.source.GetPrim()
        node = UsdShade.ConnectableAPI(prim)
        for name in _GRAPH_BASE_INPUTS:
            node_input = node.GetInput(name)
            if not node_input:
                continue
            value = node_input.Get()
            if isinstance(value, Sdf.AssetPath) and value.path:
                path = asset_path(value, node_input.GetAttr())
                if path:
                    return path
            found = _graph_base_texture(node_input, depth + 1)
            if found:
                return found
    return None


def bound_material(prim: Usd.Prim):
    """The material bound to ``prim``: a UsdShade.Material, a (``.mtlx`` file, material name) when the
    binding names a material of a MaterialX document USD couldn't read, or None."""
    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    return material if material else materialx.missing_binding(prim)


class MaterialConverter:
    def __init__(self, gltf: Gltf, mode: str, radiance_scale: float):
        self.gltf, self.mode, self.radiance_scale = gltf, mode, radiance_scale
        self.meters_per_unit = 1.0  # the stage's (set by the Converter): glTF volume distances are in metres
        self.cache: dict[tuple, tuple[int, UvSpec]] = {}
        self._tile: int | None = None
        self._udim_seen = False
        self._by_content: dict[str, int] = {}
        self.report: Counter = Counter()

    # ---- entry point ----
    def get(self, material: UsdShade.Material | str | None, double_sided: bool, display_color,
            tile: int | None = None) -> tuple[int, UvSpec]:
        """glTF material index and UV spec. A UDIM material is converted once per ``tile``; without a
        tile it only reports ``spec.udim`` (index -1), so the mesh can split its triangles first.
        ``material`` may also be the name of an unbound GeomSubset: a displayColor material by that
        name, so the material slots a file names without authoring them survive for later editing."""
        if material is None or isinstance(material, str):
            color = tuple(round(c, 4) for c in to_rgb(display_color, (0.18, 0.18, 0.18)))
            name = material or "displayColor"
            key = ("displayColor", name, color, double_sided)
            if key not in self.cache:
                self.report["displayColor" if name == "displayColor" else "unbound GeomSubset (named displayColor)"] += 1
                mat = {"name": name, "doubleSided": double_sided,
                       "pbrMetallicRoughness": {"baseColorFactor": [*color, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.5}}
                self.cache[key] = (self._add(mat), UvSpec())
            return self.cache[key]
        if isinstance(material, tuple):  # a material of a .mtlx document (bound_material)
            key = (material, double_sided)
            if key not in self.cache:
                family, inputs = materialx.document_material(*material)
                mat, uv = self._materialx(material[1], family, inputs)
                mat["doubleSided"] = double_sided
                self.report[f"MaterialX {family} (.mtlx)"] += 1
                self.cache[key] = (self._add(mat), uv)
            return self.cache[key]
        key = (str(material.GetPath()), double_sided, tile)
        if key not in self.cache:
            self._tile, self._udim_seen = tile, False
            mat, uv, source = self._convert(material)
            uv.udim = self._udim_seen
            if uv.udim and tile is None:
                self.cache[key] = (-1, uv)
                return self.cache[key]
            mat["doubleSided"] = double_sided
            if tile is not None:
                mat["name"] = f"{mat['name']}_{tile}"
            self.report[source + (" (UDIM tile)" if tile is not None else "")] += 1
            self.cache[key] = (self._add(mat), uv)
        return self.cache[key]

    def _udim(self, path):
        """``path`` for the tile being converted: a ``<UDIM>`` token becomes the tile number, or None
        (no texture) during the detection pass, or when that tile is missing."""
        if not isinstance(path, str) or UDIM not in path:
            return path
        self._udim_seen = True
        if self._tile is None:
            return None
        concrete = path.replace(UDIM, str(self._tile))
        return concrete if asset_exists(concrete) else None

    def _add(self, mat: dict) -> int:
        """Index of ``mat``, shared with an identical material already written (copies of one asset
        bind their own material prims, which convert to the same glTF material)."""
        key = json.dumps(mat, sort_keys=True)
        if key not in self._by_content:
            self.gltf.doc["materials"].append(mat)
            self._by_content[key] = len(self.gltf.doc["materials"]) - 1
        return self._by_content[key]

    def _convert(self, material: UsdShade.Material):
        name = material.GetPrim().GetName()
        if self.mode == "auto":
            mdl = self._surface_shader(material, "mdl")
            if mdl:
                family, params = self._mdl_params(mdl)
                params = {k: self._udim(v) for k, v in params.items()}
                if family.startswith("OmniPBR"):
                    mat, uv = self._omnipbr(name, params)
                    return mat, uv, f"MDL {family}"
                if family in GLASS_FAMILIES or "glass" in family.lower():
                    mat, uv = self._glass(name, params)
                    return mat, uv, f"MDL {family} (glass)"
                if family.startswith("OmniSurface"):
                    mat, uv = self._omnisurface(name, params)
                    return mat, uv, f"MDL {family}"
        if self.mode == "auto":
            mtlx = materialx.surface_shader(material)
            if mtlx:
                family = materialx.family_of(mtlx)
                inputs = materialx.stage_inputs(mtlx, lambda v, a: self._udim(asset_path(v, a)))
                if "base_color" not in inputs:
                    # A MaterialX network left without its base color (the StandardShaderBall's
                    # neutral has an image node it never connects) takes the one of the
                    # UsdPreviewSurface next to it, when that one has it.
                    preview = self._surface_shader(material, "")
                    if preview and preview.GetIdAttr().Get() == "UsdPreviewSurface":
                        value, tex = self._preview_input(preview, "diffuseColor")
                        if tex and tex["file"]:
                            inputs["base_color"] = ("texture", tex["file"], "average")
                        elif value is not None and preview.GetInput("diffuseColor").GetAttr().HasAuthoredValue():
                            inputs["base_color"] = tuple(float(x) for x in value)
                mat, uv = self._materialx(name, family, inputs)
                return mat, uv, f"MaterialX {family}"
        preview = self._surface_shader(material, "")
        if preview and preview.GetIdAttr().Get() == "UsdPreviewSurface":
            mat, uv = self._preview(name, preview)
            return mat, uv, "UsdPreviewSurface"
        return {"name": name, "pbrMetallicRoughness": {"metallicFactor": 0.0}}, UvSpec(), "unsupported (default)"

    @staticmethod
    def _surface_shader(material: UsdShade.Material, context: str) -> UsdShade.Shader | None:
        """The surface shader of a render context ("" the universal one), through node graphs."""
        if context and not material.GetSurfaceOutput(context):
            return None  # ComputeSurfaceSource would fall back to the universal output
        shader = material.ComputeSurfaceSource(context)[0]
        return shader if shader else None

    # ---- MDL ----
    @staticmethod
    def _mdl_params(shader: UsdShade.Shader) -> tuple[str, dict]:
        prim = shader.GetPrim()
        attr = prim.GetAttribute("info:mdl:sourceAsset")
        src = attr.Get() if attr else None
        sub = prim.GetAttribute("info:mdl:sourceAsset:subIdentifier").Get() or ""
        base = os.path.splitext(os.path.basename(src.path))[0] if src else ""
        params: dict = {}
        family = base
        if base in ("OmniPBR", "OmniPBR_ClearCoat", "OmniGlass", "OmniSurface", "OmniSurfaceLite"):
            family = base
        else:
            path = asset_path(src, attr) if src else None
            if path:
                family, wrapper_args = parse_mdl_wrapper(path, sub)
                params = dict(wrapper_args)
                family = family or base
        if family.startswith("OmniPBR"):
            params = {**OMNIPBR_DEFAULTS, **params}
        from_graphs = {}
        for inp in shader.GetInputs():  # USD inputs override the MDL file's arguments
            if inp.HasConnectedSource():
                # A shader graph feeds this input (e.g. a texture blended with grunge layers): glTF has
                # one texture per slot, so the graph's base layer stands in for it.
                name = inp.GetBaseName()
                slot = _GRAPH_TEXTURE_SLOTS.get(name, name if name.endswith("_texture") else None)
                texture = _graph_base_texture(inp) if slot else None
                if texture:
                    from_graphs[slot] = texture
                continue
            if not inp.GetAttr().HasAuthoredValue():
                continue
            v = inp.Get()
            if isinstance(v, Sdf.AssetPath):
                v = asset_path(v, inp.GetAttr())
            elif hasattr(v, "__len__") and not isinstance(v, str):
                v = tuple(float(x) for x in v)
            params[inp.GetBaseName()] = v
        for name, texture in from_graphs.items():  # after the loop: an empty authored slot must not win
            if not params.get(name):
                params[name] = texture
                influence = _GRAPH_TEXTURE_INFLUENCE.get(name)
                if influence:  # the graph drives the value alone, as the texture at full influence
                    params[influence] = 1.0
        return family, params

    def _omnipbr(self, name: str, p: dict) -> tuple[dict, UvSpec]:
        g = self.gltf
        pbr: dict = {}
        mat: dict = {"name": name, "pbrMetallicRoughness": pbr}
        ext: dict = {}

        # Base color, as OmniPBR computes it: the texture gets brightness, add and desaturation (a
        # constant color gets none of them), then the tint, then ao_to_diffuse darkening.
        tint = np.array(to_rgb(p.get("diffuse_tint")))
        brightness = float(p.get("albedo_brightness", 1.0))
        desat, add = float(p.get("albedo_desaturation", 0.0)), float(p.get("albedo_add", 0.0))
        base_tex = p.get("diffuse_texture")
        orm = p.get("ORM_texture") if p.get("enable_ORM_texture") else None
        ao_weight = float(p.get("ao_to_diffuse", 0.0))
        ao_src = (orm, "r") if orm else ((p["ao_texture"], "average") if p.get("ao_texture") else None)
        ao = (*ao_src, ao_weight) if ao_src and ao_weight > 0.0 else None
        opacity = None
        alpha_const = 1.0
        if p.get("enable_opacity"):
            mode = OPACITY_MODES.get(p.get("opacity_mode", 1), "average")
            if p.get("enable_opacity_texture") and p.get("opacity_texture"):
                opacity = (p["opacity_texture"], mode)
            else:
                alpha_const = float(p.get("opacity_constant", 1.0))
            threshold = float(p.get("opacity_threshold", 0.0))
            if threshold > 0.0:
                mat["alphaMode"], mat["alphaCutoff"] = "MASK", threshold
            elif opacity or alpha_const < 1.0:
                mat["alphaMode"] = "BLEND"
        if base_tex or opacity or ao:
            # Scale and desaturation are linear in the texture, so without an offset the brightness
            # stays in the factor and the 8-bit texture keeps its precision.
            bake_scale = brightness if (add and base_tex) else 1.0
            if add or desat or opacity or ao:
                img = g.image_from_array(("base", base_tex, bake_scale, add, desat, ao, opacity), f"{name}_baseColor",
                                         make_base_color(base_tex, bake_scale, add if base_tex else 0.0,
                                                         desat if base_tex else 0.0, ao, opacity))
            else:
                img = g.image_from_file(base_tex)
            if img is not None:
                pbr["baseColorTexture"] = g.texture(img)
            if base_tex:
                color = tint * (brightness / bake_scale)
            else:
                color = np.array(to_rgb(p.get("diffuse_color_constant"))) * tint
        else:
            color = np.array(to_rgb(p.get("diffuse_color_constant"))) * tint
        pbr["baseColorFactor"] = [float(x) for x in np.clip(color, 0, 1)] + [alpha_const]

        # Metallic / roughness: lerp(constant, ORM channel or texture, influence). An ORM texture
        # used at full influence is already glTF's layout; anything else is packed.
        rough_c, metal_c = float(p.get("reflection_roughness_constant", 0.5)), float(p.get("metallic_constant", 0.0))
        rough_src = (orm, "g") if orm else ((p["reflectionroughness_texture"], "average") if p.get("reflectionroughness_texture") else None)
        metal_src = (orm, "b") if orm else ((p["metallic_texture"], "average") if p.get("metallic_texture") else None)
        r_inf = float(p.get("reflection_roughness_texture_influence", 0.0)) if rough_src else 0.0
        m_inf = float(p.get("metallic_texture_influence", 0.0)) if metal_src else 0.0
        rough = (*rough_src, rough_c, r_inf) if r_inf > 0 else None
        metal = (*metal_src, metal_c, m_inf) if m_inf > 0 else None
        if orm and r_inf == 1.0 and m_inf == 1.0:
            pbr["metallicRoughnessTexture"] = self._texref(orm)
        elif rough or metal:
            img = g.image_from_array(("mr", rough, metal), f"{name}_metalRough", make_metal_rough(rough, metal))
            if img is not None:
                pbr["metallicRoughnessTexture"] = g.texture(img)
        pbr["roughnessFactor"] = 1.0 if rough else rough_c
        pbr["metallicFactor"] = 1.0 if metal else metal_c

        # Specular: OmniPBR's F0 is 0.08 * specular_level, glTF's dielectric default is 0.04.
        level = float(p.get("specular_level", 0.5))
        if abs(level - 0.5) > 1e-4:
            ext["KHR_materials_specular"] = {"specularColorFactor": [level / 0.5] * 3} if level > 0.5 else {"specularFactor": level / 0.5}

        if p.get("normalmap_texture"):
            mat["normalTexture"] = self._normal_texref(p["normalmap_texture"], scale=float(p.get("bump_factor", 1.0)))

        if p.get("enable_emission"):
            emissive_tex = p.get("emissive_color_texture") or p.get("emissive_mask_texture")
            color = [1.0, 1.0, 1.0] if p.get("emissive_color_texture") else to_rgb(p.get("emissive_color"))
            # MDL's default emission mode is radiant exitance; glTF emission is radiance.
            self._emission(mat, ext, color, float(p.get("emissive_intensity", 40.0)) / math.pi, emissive_tex)

        self._finish(mat, ext)
        uv = UvSpec(index=int(p.get("uv_space_index", 0)), scale=p.get("texture_scale", (1, 1)),
                    rotate=p.get("texture_rotate", 0.0), translate=p.get("texture_translate", (0, 0)),
                    project=("world" if p.get("world_or_object") else "object") if p.get("project_uvw") else None)
        return mat, uv

    def _omnisurface(self, name: str, p: dict) -> tuple[dict, UvSpec]:
        """OmniSurface (Autodesk Standard Surface): each lobe has a glTF extension, and an "_image"
        input replaces its constant."""
        g = self.gltf
        f = lambda k, d=0.0: float(p.get(k, d))  # noqa: E731
        pbr: dict = {}
        mat: dict = {"name": name, "pbrMetallicRoughness": pbr}
        ext: dict = {}

        color = np.array(to_rgb(p.get("diffuse_reflection_color", (0.8, 0.8, 0.8)))) * f("diffuse_reflection_weight", 0.8)
        alpha = f("geometry_opacity", 1.0) if p.get("enable_opacity") else 1.0
        opacity = (p["geometry_opacity_image"], p.get("geometry_opacity_image_mode", "average")) \
            if p.get("enable_opacity") and p.get("geometry_opacity_image") else None
        base_img = p.get("diffuse_reflection_color_image")
        if opacity:
            img = g.image_from_array(("base", base_img, 1.0, 0.0, 0.0, None, opacity), f"{name}_baseColor",
                                     make_base_color(base_img, 1.0, 0.0, 0.0, None, opacity))
            pbr["baseColorTexture"] = g.texture(img)
        elif base_img:
            pbr["baseColorTexture"] = self._texref(base_img)
        if opacity or alpha < 1.0:
            threshold = f("geometry_opacity_threshold")
            mat["alphaMode"] = "MASK" if threshold > 0 else "BLEND"
            if threshold > 0:
                mat["alphaCutoff"] = threshold

        rough_img, metal_img = p.get("specular_reflection_roughness_image"), p.get("metalness_image")
        rough = (rough_img, p.get("specular_reflection_roughness_image_mode", "average"), 0.0, 1.0) if rough_img else None
        metal = (metal_img, p.get("metalness_image_mode", "average"), 0.0, 1.0) if metal_img else None
        if rough or metal:
            img = g.image_from_array(("mr", rough, metal), f"{name}_metalRough", make_metal_rough(rough, metal))
            if img is not None:
                pbr["metallicRoughnessTexture"] = g.texture(img)
        pbr["roughnessFactor"] = 1.0 if rough else f("specular_reflection_roughness", 0.2)
        pbr["metallicFactor"] = 1.0 if metal else f("metalness")

        spec_w = f("specular_reflection_weight", 1.0)
        spec_c = to_rgb(p.get("specular_reflection_color"))
        if abs(spec_w - 1.0) > 1e-4 or spec_c != [1.0, 1.0, 1.0]:
            ext["KHR_materials_specular"] = {"specularFactor": spec_w, "specularColorFactor": spec_c}
        ior = f("specular_reflection_ior", 1.5)
        if abs(ior - 1.5) > 1e-4:
            ext["KHR_materials_ior"] = {"ior": ior}

        if f("coat_weight") > 0:
            ext["KHR_materials_clearcoat"] = {"clearcoatFactor": f("coat_weight"),
                                              "clearcoatRoughnessFactor": f("coat_roughness", 0.1)}
        if f("sheen_weight") > 0:
            sheen = np.array(to_rgb(p.get("sheen_color"))) * f("sheen_weight")
            ext["KHR_materials_sheen"] = {"sheenColorFactor": [float(x) for x in sheen],
                                          "sheenRoughnessFactor": f("sheen_roughness", 0.3)}
        if p.get("enable_specular_transmission") and f("specular_transmission_weight") > 0:
            ext["KHR_materials_transmission"] = {"transmissionFactor": f("specular_transmission_weight")}
            color = color * (1.0 - f("specular_transmission_weight")) + \
                np.array(to_rgb(p.get("specular_transmission_color"))) * f("specular_transmission_weight")
            if not p.get("thin_walled"):
                ext["KHR_materials_volume"] = {"thicknessFactor": 0.01}
        if f("subsurface_weight") > 0:
            # With enable_diffuse_transmission the subsurface lobe is rendered as thin translucency.
            if p.get("enable_diffuse_transmission") or p.get("thin_walled"):
                ext["KHR_materials_diffuse_transmission"] = {
                    "diffuseTransmissionFactor": f("subsurface_weight"),
                    "diffuseTransmissionColorFactor": to_rgb(p.get("subsurface_transmission_color"))}
            else:
                self.report["note: OmniSurface volumetric subsurface approximated as diffuse"] += 1
        pbr["baseColorFactor"] = [float(x) for x in np.clip(color, 0, 1)] + [alpha]

        if p.get("geometry_normal_image"):
            mat["normalTexture"] = self._normal_texref(p["geometry_normal_image"], scale=f("geometry_normal_strength", 1.0))
        if f("emission_weight") > 0:
            self._emission(mat, ext, to_rgb(p.get("emission_color")), f("emission_weight") * f("emission_intensity", 1.0),
                           p.get("emission_color_image"))
        self._finish(mat, ext)
        uv = UvSpec(index=int(p.get("uvw_uv_set", 0)),
                    scale=(f("uvw_s_scale", 1.0), f("uvw_t_scale", 1.0)))
        return mat, uv

    def _materialx(self, name: str, family: str, inputs: dict) -> tuple[dict, UvSpec]:
        """OpenPBR / Standard Surface: the surface as OmniSurface (the same model), then what that
        translation leaves out (materialx.extra_extensions: the medium, dispersion, anisotropy, thin film)."""
        before = Counter(self.report)
        mat, uv = self._omnisurface(name, materialx.omnisurface_params(family, inputs))
        extra = materialx.extra_extensions(family, inputs, self.meters_per_unit)
        if extra:
            if "KHR_materials_diffuse_transmission" in extra:  # subsurface is a volume here, not approximated
                self.report = before
            mat.setdefault("extensions", {}).update(extra)
            self.gltf.extensions.update(extra)
        return mat, uv

    def _glass(self, name: str, p: dict) -> tuple[dict, UvSpec]:
        color = to_rgb(p.get("glass_color", p.get("transmission_color", (1, 1, 1))))
        ior = float(p.get("glass_ior", p.get("ior", 1.5)))
        rough = float(p.get("frosting_roughness", p.get("roughness", 0.0)))
        thin = bool(p.get("thin_walled", False))
        mat = {"name": name, "pbrMetallicRoughness": {"baseColorFactor": [*color, 1.0], "metallicFactor": 0.0,
                                                        "roughnessFactor": rough}}
        ext = {"KHR_materials_transmission": {"transmissionFactor": 1.0}}
        if abs(ior - 1.5) > 1e-4 and ior >= 1.0:
            ext["KHR_materials_ior"] = {"ior": ior}
        if not thin:
            # A solid glass body: let the path tracer refract through the real geometry.
            ext["KHR_materials_volume"] = {"thicknessFactor": 0.01}
        if p.get("normal_map_texture"):
            mat["normalTexture"] = self._normal_texref(p["normal_map_texture"], scale=float(p.get("normal_map_strength", 1.0)))
        self._finish(mat, ext)
        return mat, UvSpec(scale=p.get("texture_scale", (1, 1)), rotate=p.get("texture_rotate", 0.0),
                           translate=p.get("texture_translate", (0, 0)))

    def _emission(self, mat: dict, ext: dict, color, intensity: float, tex_path: str | None) -> None:
        """Emission = color * intensity: the factor keeps the color within [0, 1] and the rest goes to
        KHR_materials_emissive_strength."""
        radiance = [c * intensity * self.radiance_scale for c in color]
        peak = max(radiance)
        if peak <= 0.0:
            return
        strength = max(peak, 1.0)
        mat["emissiveFactor"] = [c / strength for c in radiance]
        if tex_path:
            mat["emissiveTexture"] = self._texref(tex_path)
        if strength > 1.0:
            ext["KHR_materials_emissive_strength"] = {"emissiveStrength": strength}

    def _texref(self, path: str | None, wrap=(10497, 10497), **extra) -> dict | None:
        """textureInfo for ``path``, or None when it is missing or unreadable."""
        ref = self.gltf.texture(self.gltf.image_from_file(path), wrap) if path else None
        return {**ref, **extra} if ref else None

    def _normal_texref(self, path: str | None, wrap=(10497, 10497), scale: float = 1.0) -> dict | None:
        """normalTexture for ``path``, or None for a map of one flat colour: it has no detail, and one
        that is not (0.5, 0.5, 1) is a placeholder that tilts the whole surface (Davinci's PaperScroll
        ships lime green, a normal pointing into the paper)."""
        if path and _flat_image(path):
            self.report["note: normal map of one flat colour, skipped"] += 1
            return None
        return self._texref(path, wrap, scale=scale)

    def _finish(self, mat: dict, ext: dict) -> None:
        for d in (mat, mat.get("pbrMetallicRoughness", {})):
            for k in [k for k, v in d.items() if v is None]:
                del d[k]
        if ext:
            mat["extensions"] = ext
            self.gltf.extensions.update(ext)

    # ---- UsdPreviewSurface ----
    def _preview_input(self, shader: UsdShade.Shader, name: str):
        """(value, texture) where texture = dict(file, channel, scale, bias, wrap, uv) if connected."""
        inp = shader.GetInput(name)
        if not inp:
            return None, None
        if inp.HasConnectedSource():
            src, out_name, _ = inp.GetConnectedSource()
            tex = UsdShade.Shader(src.GetPrim())
            if tex.GetIdAttr().Get() == "UsdUVTexture":
                file_in = tex.GetInput("file")
                path = self._udim(asset_path(file_in.Get(), file_in.GetAttr())) if file_in else None
                wrap = {"repeat": 10497, "mirror": 33648, "clamp": 33071, "black": 33071}
                uv = UvSpec()
                st = tex.GetInput("st")
                if st and st.HasConnectedSource():
                    reader = UsdShade.Shader(st.GetConnectedSource()[0].GetPrim())
                    if reader.GetIdAttr().Get() == "UsdTransform2d":
                        g = lambda n, d: reader.GetInput(n).Get() if reader.GetInput(n) else d  # noqa: E731
                        uv = UvSpec(scale=g("scale", (1, 1)), rotate=g("rotation", 0.0), translate=g("translation", (0, 0)))
                        inner = reader.GetInput("in")
                        reader = UsdShade.Shader(inner.GetConnectedSource()[0].GetPrim()) if inner and inner.HasConnectedSource() else None
                    if reader and reader.GetInput("varname"):
                        uv.primvar = str(reader.GetInput("varname").Get() or "st")
                def get(n, d):
                    i = tex.GetInput(n)
                    v = i.Get() if i else None
                    return d if v is None else (tuple(v) if hasattr(v, "__len__") and not isinstance(v, str) else v)
                return inp.Get(), {"file": path, "channel": str(out_name), "scale": get("scale", (1, 1, 1, 1)),
                                   "bias": get("bias", (0, 0, 0, 0)), "uv": uv,
                                   "wrap": (wrap.get(str(get("wrapS", "repeat")), 10497), wrap.get(str(get("wrapT", "repeat")), 10497))}
        return inp.Get(), None

    def _preview(self, name: str, s: UsdShade.Shader) -> tuple[dict, UvSpec]:
        g = self.gltf
        pbr: dict = {}
        mat: dict = {"name": name, "pbrMetallicRoughness": pbr}
        ext: dict = {}
        uv = None

        def tex_info(t):
            nonlocal uv
            uv = uv or t["uv"]
            return t

        diffuse, dtex = self._preview_input(s, "diffuseColor")
        opacity, otex = self._preview_input(s, "opacity")
        threshold, _ = self._preview_input(s, "opacityThreshold")
        alpha = 1.0 if otex else float(opacity if opacity is not None else 1.0)
        ior_in = s.GetInput("ior")
        if alpha < 1.0 and not (threshold and float(threshold) > 0.0) and ior_in and ior_in.GetAttr().HasAuthoredValue() \
                and float(ior_in.Get() or 1.0) > 1.0:
            # Opacity below 1 with an index of refraction is glass, as renderers read UsdPreviewSurface
            # (the StandardShaderBall's usdpreview_glass: opacity 0, ior 1.52): the light goes through,
            # refracted, rather than the surface fading out. Unauthored, the color is clear, not the
            # default 0.18 grey.
            ext["KHR_materials_transmission"] = {"transmissionFactor": 1.0 - alpha}
            ext["KHR_materials_volume"] = {"thicknessFactor": 0.01}
            if not (s.GetInput("diffuseColor") and s.GetInput("diffuseColor").GetAttr().HasAuthoredValue()) and not dtex:
                diffuse = (1.0, 1.0, 1.0)
            alpha = 1.0
        if dtex and dtex["file"]:
            tex_info(dtex)
            pbr["baseColorFactor"] = [float(x) for x in dtex["scale"][:3]] + [alpha]
            if otex and otex["file"] and not (otex["file"] == dtex["file"] and otex["channel"] == "a"):
                img = g.image_from_array(("pbase", dtex["file"], otex["file"], otex["channel"]), f"{name}_baseColor",
                                         make_base_color(dtex["file"], 1.0, 0.0, 0.0, None, (otex["file"], otex["channel"])))
            else:
                img = g.image_from_file(dtex["file"])
            pbr["baseColorTexture"] = g.texture(img, dtex["wrap"])
        else:
            pbr["baseColorFactor"] = to_rgb(diffuse, (0.18, 0.18, 0.18)) + [alpha]
            if otex and otex["file"]:
                tex_info(otex)
                img = g.image_from_array(("palpha", otex["file"], otex["channel"]), f"{name}_opacity",
                                         make_base_color(None, 1.0, 0.0, 0.0, None, (otex["file"], otex["channel"])))
                pbr["baseColorTexture"] = g.texture(img, otex["wrap"])
        if otex or alpha < 1.0:
            if threshold and float(threshold) > 0.0:
                mat["alphaMode"], mat["alphaCutoff"] = "MASK", float(threshold)
            else:
                mat["alphaMode"] = "BLEND"

        rough, rtex = self._preview_input(s, "roughness")
        metal, mtex = self._preview_input(s, "metallic")
        def scalar(t):  # the channel's UsdUVTexture scale and bias come along
            c = "rgba".index(t["channel"]) if t["channel"] in ("r", "g", "b", "a") else 0
            return (t["file"], t["channel"], 0.0, 1.0, float(t["scale"][c]), float(t["bias"][c]))

        r = scalar(rtex) if rtex and rtex["file"] else None
        m = scalar(mtex) if mtex and mtex["file"] else None
        if r or m:
            tex_info(rtex if r else mtex)
            img = g.image_from_array(("mr", r, m), f"{name}_metalRough", make_metal_rough(r, m))
            pbr["metallicRoughnessTexture"] = g.texture(img, (rtex if r else mtex)["wrap"])
        pbr["roughnessFactor"] = 1.0 if r else float(rough if rough is not None else 0.5)
        pbr["metallicFactor"] = 1.0 if m else float(metal if metal is not None else 0.0)

        _, ntex = self._preview_input(s, "normal")
        if ntex and ntex["file"]:
            tex_info(ntex)
            sc, bias = ntex["scale"], ntex["bias"]
            # A standard map decodes as 2c-1 (scale 2, bias -1). A map stored raw with a large Z
            # scale is a flattened normal: approximate it with a weaker glTF normal scale.
            strength = 1.0 if abs(bias[0] + 1) < 1e-3 else min(1.0, 2.0 / max(abs(sc[2]), 1e-6))
            mat["normalTexture"] = self._normal_texref(ntex["file"], ntex["wrap"], scale=strength)

        _, aotex = self._preview_input(s, "occlusion")
        if aotex and aotex["file"]:
            tex_info(aotex)
            img = (g.image_from_file(aotex["file"]) if aotex["channel"] == "r" else
                   g.image_from_array(("ao", aotex["file"], aotex["channel"]), f"{name}_occlusion",
                                      make_single_channel(aotex["file"], aotex["channel"])))
            mat["occlusionTexture"] = g.texture(img, aotex["wrap"])

        emissive, etex = self._preview_input(s, "emissiveColor")
        ecolor = [float(x) for x in (etex["scale"][:3] if etex else to_rgb(emissive, (0, 0, 0)))]
        if etex and etex["file"]:
            tex_info(etex)
        if (etex and etex["file"]) or max(ecolor) > 0:
            self._emission(mat, ext, ecolor, 1.0, etex["file"] if etex else None)

        ior, _ = self._preview_input(s, "ior")
        if ior is not None and float(ior) > 1.0 and abs(float(ior) - 1.5) > 1e-4:
            ext["KHR_materials_ior"] = {"ior": float(ior)}
        cc, _ = self._preview_input(s, "clearcoat")
        if cc and float(cc) > 0:
            ccr, _ = self._preview_input(s, "clearcoatRoughness")
            ext["KHR_materials_clearcoat"] = {"clearcoatFactor": float(cc), "clearcoatRoughnessFactor": float(ccr or 0.01)}
        spec_wf, _ = self._preview_input(s, "useSpecularWorkflow")
        if spec_wf:
            spec, _ = self._preview_input(s, "specularColor")
            ext["KHR_materials_specular"] = {"specularColorFactor": [c / 0.04 for c in to_rgb(spec, (0.04, 0.04, 0.04))]}
            pbr["metallicFactor"] = 0.0
        self._finish(mat, ext)
        return mat, uv or UvSpec()


# --------------------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------------------

def _flatten_primvar(pv: UsdGeom.Primvar, t) -> np.ndarray | None:
    v = pv.ComputeFlattened(t)
    return None if v is None or len(v) == 0 else np.asarray(v, dtype=np.float64)


def _per_corner(values: np.ndarray | None, interp: str, corner: np.ndarray, point: np.ndarray, face: np.ndarray):
    """Gather a primvar for each triangle corner. ``corner`` indexes faceVertexIndices, ``point`` the
    points and ``face`` the source polygon of each corner."""
    if values is None:
        return None
    if interp == UsdGeom.Tokens.constant:
        return np.broadcast_to(values[0], (len(corner),) + values.shape[1:])
    if interp == UsdGeom.Tokens.uniform:
        return values[face] if values.shape[0] > face.max(initial=0) else None
    if interp in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
        return values[point] if values.shape[0] > point.max(initial=0) else None
    return values[corner] if values.shape[0] > corner.max(initial=0) else None  # faceVarying


def _texcoord_primvars(api: UsdGeom.PrimvarsAPI) -> list[UsdGeom.Primvar]:
    out = [pv for pv in api.GetPrimvars()
           if pv.GetTypeName().role == Sdf.ValueRoleNames.TextureCoordinate or pv.GetPrimvarName() in ("st", "uv")]
    return sorted(out, key=lambda pv: (pv.GetPrimvarName() != "st", pv.GetPrimvarName()))


def _box_project(pos: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Cubic projection: each triangle takes the plane its geometric normal faces most."""
    axis = np.abs(normal).argmax(axis=1)
    uv = np.where(axis[:, None] == 0, pos[:, [1, 2]], np.where(axis[:, None] == 1, pos[:, [0, 2]], pos[:, [0, 1]]))
    return uv


def _matrix_to_quaternion(m: np.ndarray) -> np.ndarray:
    """(n, 3, 3) rotation matrices (column vectors) to (n, 4) unit quaternions x, y, z, w. Shepperd's
    method: the largest of 4w^2, 4x^2, 4y^2, 4z^2 gives one component, the others follow from
    sums and differences of off-diagonal terms, so 180-degree rotations (w = 0) stay exact."""
    d0, d1, d2 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    diff = np.stack([m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]], axis=1)
    sum_xy, sum_xz, sum_yz = m[:, 1, 0] + m[:, 0, 1], m[:, 0, 2] + m[:, 2, 0], m[:, 2, 1] + m[:, 1, 2]
    four_sq = np.stack([1 + d0 + d1 + d2, 1 + d0 - d1 - d2, 1 - d0 + d1 - d2, 1 - d0 - d1 + d2], axis=1)
    pick = four_sq.argmax(axis=1)
    s = 2 * np.sqrt(np.maximum(four_sq[np.arange(len(m)), pick], 1e-30))  # 4 * the picked component
    q = np.empty((len(m), 4))
    for k, (x, y, z, w) in enumerate([(diff[:, 0], diff[:, 1], diff[:, 2], s * s / 4),
                                      (s * s / 4, sum_xy, sum_xz, diff[:, 0]),
                                      (sum_xy, s * s / 4, sum_yz, diff[:, 1]),
                                      (sum_xz, sum_yz, s * s / 4, diff[:, 2])]):
        rows = pick == k
        q[rows] = np.stack([x, y, z, w], axis=1)[rows] / s[rows, None]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _ear_clip(poly: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangles (local vertex indices) of one simple polygon, projected on its dominant plane."""
    n = len(poly)
    normal = np.sum(np.cross(poly, np.roll(poly, -1, axis=0)), axis=0)  # Newell
    drop = int(np.abs(normal).argmax())
    p2 = np.delete(poly, drop, axis=1)
    if (normal[drop] < 0) != (drop == 1):  # dropping Y swaps the (x, z) plane's handedness
        p2 = p2[:, ::-1]  # keep the polygon counter-clockwise in 2D
    cross = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])  # noqa: E731
    remaining, out = list(range(n)), []
    while len(remaining) > 3:
        for i in range(len(remaining)):
            a, b, c = remaining[i - 1], remaining[i], remaining[(i + 1) % len(remaining)]
            if cross(p2[a], p2[b], p2[c]) <= 0:
                continue  # reflex corner
            if any(cross(p2[a], p2[b], p2[v]) >= 0 and cross(p2[b], p2[c], p2[v]) >= 0 and cross(p2[c], p2[a], p2[v]) >= 0
                   for v in remaining if v not in (a, b, c)):
                continue  # another vertex inside: not an ear
            out.append((a, b, c))
            remaining.pop(i)
            break
        else:  # degenerate polygon: fan the rest rather than loop forever
            out += [(remaining[0], remaining[j], remaining[j + 1]) for j in range(1, len(remaining) - 1)]
            return out
    out.append(tuple(remaining))
    return out


def _triangulate(pts: np.ndarray, counts: np.ndarray, indices: np.ndarray, holes: np.ndarray | None):
    """Triangles as corner indices into faceVertexIndices, and the face of each. Convex faces are
    fanned; a face whose corners do not all turn the same way is ear-clipped."""
    ntri = np.maximum(counts - 2, 0)
    if holes is not None:
        ntri[holes[holes < len(counts)]] = 0
    starts = np.cumsum(counts) - counts
    concave = np.zeros(len(counts), dtype=bool)
    for k in np.unique(counts[counts >= 4]):
        faces = np.nonzero((counts == k) & (ntri > 0))[0]
        poly = pts[indices[starts[faces, None] + np.arange(k)]]  # (faces, k, 3)
        normal = np.sum(np.cross(poly, np.roll(poly, -1, axis=1)), axis=1)
        turn = np.cross(np.roll(poly, -1, axis=1) - poly, np.roll(poly, -2, axis=1) - np.roll(poly, -1, axis=1))
        concave[faces] = (np.einsum("fkc,fc->fk", turn, normal) < -1e-12 * np.linalg.norm(normal, axis=1)[:, None] ** 2).any(1)
    fan = ntri.copy()
    fan[concave] = 0
    tri_face = np.repeat(np.arange(len(counts)), fan)
    k = np.arange(len(tri_face)) - np.repeat(np.cumsum(fan) - fan, fan)
    c0 = starts[tri_face]
    tris = np.stack([c0, c0 + k + 1, c0 + k + 2], axis=1)
    if concave.any():
        extra, extra_face = [], []
        for f in np.nonzero(concave)[0]:
            for tri in _ear_clip(pts[indices[starts[f] + np.arange(counts[f])]]):
                extra.append([starts[f] + v for v in tri])
                extra_face.append(f)
        tris = np.concatenate([tris, np.asarray(extra, dtype=np.int64)])
        tri_face = np.concatenate([tri_face, np.asarray(extra_face, dtype=np.int64)])
    return tris, tri_face


class SkelPose:
    """Meshes bound to a UsdSkel skeleton store their points in the bind pose; drawn as static
    geometry they belong at the skeleton's rest pose (or its animation's pose at the converted time).
    Rigid game props are built this way: each piece of a thatch roof is bound to its own joint."""

    def __init__(self):
        self.cache = UsdSkel.Cache()
        self.roots: set[str] = set()

    def key(self, prim: Usd.Prim, t) -> bytes:
        """What the pose of ``prim`` depends on, for mesh deduplication (empty when not skinned)."""
        skel = UsdSkel.BindingAPI(prim).GetInheritedSkeleton() if prim.HasAPI(UsdSkel.BindingAPI) else None
        if not skel:
            return b""
        query = self._query(prim, skel)
        xforms = query[1].ComputeSkinningTransforms(t) if query else None
        return str(skel.GetPath()).encode() + (np.asarray(xforms).tobytes() if xforms else b"")

    def _query(self, prim: Usd.Prim, skel: UsdSkel.Skeleton):
        root = UsdSkel.Root.Find(prim)
        if not root:
            return None
        if str(root.GetPath()) not in self.roots:
            self.cache.Populate(root, Usd.TraverseInstanceProxies())
            self.roots.add(str(root.GetPath()))
        skinning, skel_query = self.cache.GetSkinningQuery(prim), self.cache.GetSkelQuery(skel)
        if not skinning or not skinning.HasJointInfluences() or not skel_query:
            return None
        return skinning, skel_query

    def deform(self, prim: Usd.Prim, t, pts: np.ndarray):
        """(points posed, in the mesh's space; per-point 3x3 normal matrices), or None when the mesh
        is not skinned."""
        skel = UsdSkel.BindingAPI(prim).GetInheritedSkeleton() if prim.HasAPI(UsdSkel.BindingAPI) else None
        query = self._query(prim, skel) if skel else None
        if not query:
            return None
        skinning, skel_query = query
        xforms = skel_query.ComputeSkinningTransforms(t)
        if not xforms:
            return None

        def skin(p: np.ndarray) -> np.ndarray | None:
            arr = Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(p, dtype=np.float32))
            return np.asarray(arr, dtype=np.float64) if skinning.ComputeSkinnedPoints(xforms, arr, t) else None

        posed = skin(pts)
        if posed is None:
            return None
        # Skinning is linear per point: skinning unit offsets gives each point's matrix, for normals.
        lin = np.stack([skin(pts + axis) - posed for axis in np.eye(3)], axis=2)  # (n, 3, 3), columns
        # Skinned points are in the skeleton's space; the mesh node carries the mesh's own transform.
        xf = UsdGeom.XformCache(t)
        to_mesh = np.array(xf.GetLocalToWorldTransform(skel.GetPrim()) * xf.GetLocalToWorldTransform(prim).GetInverse())
        posed = posed @ to_mesh[:3, :3] + to_mesh[3, :3]
        lin = np.einsum("ji,njk->nik", to_mesh[:3, :3], lin)  # row-vector to_mesh: column form is its transpose
        return posed, np.linalg.inv(lin).transpose(0, 2, 1)


def convert_mesh(prim: Usd.Prim, t, frame: Frame, gltf: Gltf, materials: MaterialConverter, world: Gf.Matrix4d,
                 single_sided: bool, skel: SkelPose | None = None):
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get(t)
    counts = mesh.GetFaceVertexCountsAttr().Get(t)
    indices = mesh.GetFaceVertexIndicesAttr().Get(t)
    if not pts or not counts or not indices:
        return None
    pts = np.asarray(pts, dtype=np.float64)
    posed = skel.deform(prim, t, pts) if skel else None
    normal_matrices = None
    if posed is not None:
        pts, normal_matrices = posed
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if counts.sum() != len(indices) or indices.max() >= len(pts):
        log(f"  warning: {prim.GetPath()} has inconsistent topology, skipped")
        return None

    # Triangulate; holes and degenerate faces produce no triangles.
    nface = len(counts)
    holes = mesh.GetHoleIndicesAttr().Get(t)
    tris, tri_face = _triangulate(pts, counts, indices, np.asarray(holes, dtype=np.int64) if holes else None)
    if mesh.GetOrientationAttr().Get(t) == UsdGeom.Tokens.leftHanded:
        tris = tris[:, [0, 2, 1]]
    if not len(tris):
        return None

    # Material per face: the mesh binding, overridden by material-bind GeomSubsets.
    bind = UsdShade.MaterialBindingAPI(prim)
    face_mat = np.zeros(nface, dtype=np.int64)
    mat_list = [bound_material(prim)]
    for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
        mat_list.append(bound_material(subset.GetPrim()) or subset.GetPrim().GetName())
        ids = subset.GetIndicesAttr().Get(t)
        if ids:
            ids = np.asarray(ids, dtype=np.int64)
            face_mat[ids[ids < nface]] = len(mat_list) - 1
    api = UsdGeom.PrimvarsAPI(prim)
    dc = api.GetPrimvar("displayColor")
    display_color = dc.ComputeFlattened(t)[0] if dc and dc.ComputeFlattened(t) else None
    # Omniverse RTX ignores USD doubleSided and culls nothing by default, so content made there can
    # rely on back faces (an inside-out mesh still shows). Only --single-sided honors the attribute.
    double_sided = bool(mesh.GetDoubleSidedAttr().Get(t)) if single_sided else True

    # Normals: primvars:normals wins over the normals attribute; compute when missing.
    corner = tris.reshape(-1)
    point = indices[corner]
    face = np.repeat(tri_face, 3)
    normals = None
    npv = api.GetPrimvar("normals")
    if npv and npv.HasValue():
        normals = _per_corner(_flatten_primvar(npv, t), npv.GetInterpolation(), corner, point, face)
    elif mesh.GetNormalsAttr().HasValue():
        n = mesh.GetNormalsAttr().Get(t)
        if n:
            normals = _per_corner(np.asarray(n, dtype=np.float64), mesh.GetNormalsInterpolation(), corner, point, face)
    if normals is not None and normal_matrices is not None:  # authored in the bind pose
        normals = np.einsum("nij,nj->ni", normal_matrices[point], normals)
    p_tri = pts[indices[tris]]
    face_n = np.cross(p_tri[:, 1] - p_tri[:, 0], p_tri[:, 2] - p_tri[:, 0])

    def computed_normals():
        # face_n comes from the already reordered triangles, so it faces the front for either orientation
        if mesh.GetSubdivisionSchemeAttr().Get(t) == UsdGeom.Tokens.none:
            return np.repeat(face_n, 3, axis=0)
        acc = np.zeros_like(pts)
        for i in range(3):
            np.add.at(acc, indices[tris[:, i]], face_n)
        return acc[point]

    if normals is None:
        normals = computed_normals()
    else:
        # Authored normals can be zero or not finite (Davinci's PaperScroll edges are all (0, 0, 0)):
        # glTF requires unit normals, and a zero one is NaN once a renderer normalizes it.
        normals = np.array(normals, dtype=np.float64)  # writable
        bad = ~np.isfinite(normals).all(axis=1) | (np.linalg.norm(normals, axis=1) < 1e-12)
        if bad.any():
            normals[bad] = computed_normals()[bad]
            log(f"  warning: {prim.GetPath()} has {int(bad.sum()):,} zero or invalid normals, computed instead")
    normals = frame.vectors(np.asarray(normals, dtype=np.float64))
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)

    positions = frame.points(pts[point])
    uv_sets = _texcoord_primvars(api)
    uv_cache: dict[str, np.ndarray | None] = {}

    def uv_source(spec: UvSpec, sel: np.ndarray) -> np.ndarray | None:
        if spec.project:
            src = pts[point[sel]]
            if spec.project == "world":
                m = np.array(world, dtype=np.float64)
                src = src @ m[:3, :3] + m[3, :3]
            return _box_project(src, np.repeat(face_n, 3, axis=0)[sel])
        name = spec.primvar
        if spec.index and spec.index < len(uv_sets):
            name = uv_sets[spec.index].GetPrimvarName()
        if name not in uv_cache:
            pv = api.GetPrimvar(name)
            if not pv or not pv.HasValue():
                pv = uv_sets[0] if uv_sets else None
            uv_cache[name] = (_per_corner(_flatten_primvar(pv, t), pv.GetInterpolation(), corner, point, face)
                              if pv else None)
        full = uv_cache[name]
        return None if full is None else full[sel]

    prims = []
    world_uvs = False  # UVs projected from world positions differ per instance
    emit = lambda sel, uv, mat_index: _emit_primitive(gltf, prims, positions, normals, sel, uv, mat_index)  # noqa: E731
    tri_mat = face_mat[tri_face]
    for mi, material in enumerate(mat_list):
        tri_sel = np.nonzero(tri_mat == mi)[0]
        if not len(tri_sel):
            continue
        mat_index, spec = materials.get(material, double_sided, display_color)
        world_uvs |= spec.project == "world"
        sel = (tri_sel[:, None] * 3 + np.arange(3)).reshape(-1)
        uv = uv_source(spec, sel)
        if uv is not None:
            uv = uv[:, :2]
            if spec.rotate:  # OmniPBR's rotation_translation_scale: rotate (clockwise), scale, translate
                a = math.radians(spec.rotate)
                uv = uv @ np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            uv = uv * np.array(spec.scale) + np.array(spec.translate)
        if not (spec.udim and uv is not None):
            emit(sel, uv, mat_index if mat_index >= 0 else materials.get(material, double_sided, display_color, 1001)[0])
            continue
        # UDIM: tile 1001 + u + 10 v of each triangle's centroid; its UVs move back into 0..1.
        centroid = uv.reshape(-1, 3, 2).mean(axis=1)
        tu = np.clip(np.floor(centroid[:, 0]), 0, 9).astype(np.int64)
        tv = np.maximum(np.floor(centroid[:, 1]), 0).astype(np.int64)
        tri_tile = 1001 + tu + 10 * tv
        for tile in np.unique(tri_tile):
            in_tile = np.nonzero(tri_tile == tile)[0]
            corners = (in_tile[:, None] * 3 + np.arange(3)).reshape(-1)
            offset = np.array([(tile - 1001) % 10, (tile - 1001) // 10], dtype=np.float64)
            emit(sel[corners], uv[corners] - offset, materials.get(material, double_sided, display_color, int(tile))[0])
    if not prims:
        return None
    gltf.doc["meshes"].append({"name": prim.GetName(), "primitives": prims})
    return len(gltf.doc["meshes"]) - 1, len(tris), world_uvs


def _emit_primitive(gltf: Gltf, prims: list, positions, normals, sel, uv, mat_index) -> None:
    """One primitive from the corners ``sel``: corners deduplicated into indexed vertices; ``uv`` is
    in USD convention (V up) and flipped here for glTF."""
    cols = [positions[sel], normals[sel]]
    if uv is not None:
        cols.append(np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1))
    rows = np.ascontiguousarray(np.concatenate(cols, axis=1), dtype=np.float32)
    unique, inverse = np.unique(rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel(),
                                return_inverse=True)
    verts = unique.view(np.float32).reshape(-1, rows.shape[1])
    attrs = {"POSITION": gltf.accessor(verts[:, 0:3], "VEC3", minmax=True),
             "NORMAL": gltf.accessor(verts[:, 3:6], "VEC3")}
    if uv is not None:
        attrs["TEXCOORD_0"] = gltf.accessor(verts[:, 6:8], "VEC2")
    prims.append({"attributes": attrs, "indices": gltf.accessor(inverse.astype(np.int64).ravel(), "SCALAR"),
                  "material": mat_index})


# --------------------------------------------------------------------------------------
# Lights and cameras
# --------------------------------------------------------------------------------------

def light_radiance(prim: Usd.Prim, t) -> float:
    light = UsdLux.LightAPI(prim)
    get = lambda a, d: (a.Get(t) if a and a.HasValue() else d)  # noqa: E731
    return float(get(light.GetIntensityAttr(), 1.0)) * 2.0 ** float(get(light.GetExposureAttr(), 0.0))


def light_color(prim: Usd.Prim, t) -> list[float]:
    """inputs:color, times the blackbody color of inputs:colorTemperature when enabled (UsdLux's own
    conversion, normalized to unit luminance so the temperature does not change the brightness)."""
    light = UsdLux.LightAPI(prim)
    color_attr = light.GetColorAttr()
    color = to_rgb(color_attr.Get(t) if color_attr and color_attr.HasValue() else None)
    enable = light.GetEnableColorTemperatureAttr()
    if enable and enable.Get(t):
        kelvin = light.GetColorTemperatureAttr().Get(t) or 6500.0
        color = [c * b for c, b in zip(color, UsdLux.BlackbodyTemperatureAsRgb(float(kelvin)))]
    return color


def convert_light(prim: Usd.Prim, t, frame: Frame, radiance_scale: float, world: Gf.Matrix4d) -> dict | None:
    """UsdLux radiance (intensity * 2^exposure * color) to KHR_lights_punctual candela / lux. A light's
    size follows its transform, so its area takes the world scale (glTF lights ignore node scale)."""
    light = UsdLux.LightAPI(prim)
    get = lambda a, d: (a.Get(t) if a and a.HasValue() else d)  # noqa: E731
    radiance = light_radiance(prim, t) * radiance_scale
    # glTF light colors stay in [0, 1]: a brighter channel (a warm temperature) moves into the intensity.
    color = light_color(prim, t)
    peak = max(max(color), 1e-6)
    color = [c / peak for c in color]
    radiance *= peak
    normalize = bool(get(light.GetNormalizeAttr(), False))
    sx, sy, sz = (Gf.Vec3d(world.GetRow3(i)).GetLength() * frame.scale for i in range(3))
    cone = float(get(UsdLux.ShapingAPI(prim).GetShapingConeAngleAttr(), 180.0))
    out = {"name": prim.GetName(), "color": color}
    if prim.IsA(UsdLux.SphereLight):
        r = float(get(UsdLux.SphereLight(prim).GetRadiusAttr(), 0.5))
        out["type"], area = ("spot" if cone < 90 else "point"), math.pi * r * r * (sx * sy + sy * sz + sx * sz) / 3.0
    elif prim.IsA(UsdLux.DiskLight):
        r = float(get(UsdLux.DiskLight(prim).GetRadiusAttr(), 0.5))
        out["type"], area = "spot", math.pi * r * r * sx * sy
    elif prim.IsA(UsdLux.RectLight):
        rl = UsdLux.RectLight(prim)
        out["type"], area = "spot", float(get(rl.GetWidthAttr(), 1.0)) * float(get(rl.GetHeightAttr(), 1.0)) * sx * sy
    elif prim.IsA(UsdLux.DistantLight):
        out["type"], out["intensity"] = "directional", radiance
        return out
    else:
        return None
    out["intensity"] = radiance if normalize else radiance * area
    if out["type"] == "spot":
        out["spot"] = {"innerConeAngle": 0.0, "outerConeAngle": math.radians(min(cone, 90.0))}
    return out


def convert_camera(prim: Usd.Prim, t, frame: Frame) -> dict:
    cam = UsdGeom.Camera(prim).GetCamera(t)
    near, far = cam.clippingRange.min * frame.scale, cam.clippingRange.max * frame.scale
    if cam.projection == Gf.Camera.Orthographic:
        return {"name": prim.GetName(), "type": "orthographic", "orthographic": {
            "xmag": cam.horizontalAperture * 0.05 * frame.scale, "ymag": cam.verticalAperture * 0.05 * frame.scale,
            "znear": near, "zfar": far}}
    return {"name": prim.GetName(), "type": "perspective", "perspective": {
        "yfov": math.radians(cam.GetFieldOfView(Gf.Camera.FOVVertical)),
        "aspectRatio": cam.horizontalAperture / cam.verticalAperture, "znear": max(near, 1e-4), "zfar": far}}


# --------------------------------------------------------------------------------------
# Scene traversal
# --------------------------------------------------------------------------------------

def camera_exposure(stage: Usd.Stage, t) -> tuple[float, Counter]:
    """The most common camera exposure (stops) and every value seen."""
    seen: Counter = Counter()
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            attr = UsdGeom.Camera(prim).GetExposureAttr()
            seen[float(attr.Get(t)) if attr and attr.HasValue() else 0.0] += 1
    return (seen.most_common(1)[0][0] if seen else 0.0), seen


def omniverse_exposure(stage: Usd.Stage) -> float | None:
    """Exposure (stops) of Omniverse RTX's photographic tonemapper, from the render settings Kit saves
    in the root layer: EV100 = log2(N^2 / t) - log2(ISO / 100), exposure 1 / (1.2 * 2^EV100). None
    when the stage carries none of those settings. Missing ones take Kit's defaults."""
    settings = stage.GetRootLayer().customLayerData.get("renderSettings", {})
    keys = ("rtx:post:tonemap:fNumber", "rtx:post:tonemap:filmIso", "rtx:post:tonemap:cameraShutter")
    if not any(k in settings for k in keys):
        return None
    f_number = float(settings.get(keys[0], 5.0))
    iso = float(settings.get(keys[1], 100.0))
    shutter = float(settings.get(keys[2], 50.0))  # 1 / seconds
    ev100 = math.log2(f_number * f_number * shutter) - math.log2(iso / 100.0)
    return -(ev100 + math.log2(1.2))


def stage_exposure(stage: Usd.Stage, t) -> tuple[float, Counter, str]:
    """(stops, the cameras' exposures, where it comes from): the cameras' exposure, else, without
    one, Omniverse's photographic settings, as the RTX tonemapper would."""
    exposure, exposures = camera_exposure(stage, t)
    if exposure == 0.0:
        photographic = omniverse_exposure(stage)
        if photographic is not None:
            return photographic, exposures, "Omniverse tonemap f-number/ISO/shutter"
    return exposure, exposures, "camera exposure"


class Converter:
    """``skip(prim)`` prunes a prim and its subtree (usd-world converts what an assembly places
    elsewhere). ``default_sky`` gives a stage lit only by its own lights a black sky; a single asset
    meant to be placed in a world has no sky."""

    def __init__(self, stage: Usd.Stage, out: Path, material_mode: str, keep_exposure: bool, single_sided: bool,
                 skip=None, default_sky: bool = True, exposure: float | None = None, shared_textures: Path | None = None,
                 scene_cameras: list | None = None):
        """``exposure``: stops baked into lights, emission and the sky, instead of the stage's own
        (a part of a world takes the world's); ``shared_textures``: see ``Gltf``; ``scene_cameras``:
        a list that collects the camera prims instead of converting them (a world places them)."""
        self.stage, self.frame = stage, Frame(stage)
        self.skip, self.default_sky, self.scene_cameras = skip, default_sky, scene_cameras
        self.single_sided = single_sided
        self.time = Usd.TimeCode.EarliestTime()
        if exposure is None:
            self.exposure, self.exposures, self.exposure_source = stage_exposure(stage, self.time)
        else:
            self.exposure, self.exposures, self.exposure_source = exposure, Counter(), "given"
        self.radiance_scale = 1.0 if keep_exposure else 2.0 ** self.exposure
        self.gltf = Gltf(out, shared_textures)
        self.materials = MaterialConverter(self.gltf, material_mode, self.radiance_scale)
        self.materials.meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        self.xform_cache = UsdGeom.XformCache(self.time)
        self.mesh_cache: dict[tuple, int | None] = {}
        self._proto_keys: dict[str, tuple] = {}
        self.stats: Counter = Counter()
        self.skipped: Counter = Counter()
        self.dome_lights: list[str] = []
        self.predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        self.skel = SkelPose()

    def run(self, root: Usd.Prim | None = None) -> None:
        """Convert the children of ``root`` (the whole stage by default), relative to it, and
        ``root`` itself when it draws (an asset that is one mesh), without its own transform."""
        root = root or self.stage.GetPseudoRoot()
        if not root.IsPseudoRoot() and (root.IsA(UsdGeom.Gprim) or root.IsA(UsdGeom.PointInstancer)):
            roots = [n for n in [self._node(root, own=True)] if n is not None]
        else:
            roots = [n for n in (self._node(p) for p in root.GetFilteredChildren(self.predicate)) if n is not None]
        self.gltf.doc["scenes"][0]["nodes"] = roots
        if self.default_sky and self.gltf.sky is None and self.gltf.lights:
            # USD has no environment unless a dome light says so; without this the viewer's own default
            # sky would light an interior that its authors lit only with its lights.
            self.gltf.sky = {"type": "plain", "plain": {"color": [0.0, 0.0, 0.0]}}
            self.dome_lights.append("none: black plain sky (the stage is lit by its own lights only)")

    def _point_instancer(self, prim: Usd.Prim, node: dict) -> bool:
        """Each mesh of each prototype becomes a child node drawn once per instance with
        EXT_mesh_gpu_instancing (translation, rotation, scale per instance, in the node's frame)."""
        instancer = UsdGeom.PointInstancer(prim)
        proto_ids = np.asarray(instancer.GetProtoIndicesAttr().Get(self.time) or [], dtype=np.int64)
        if not len(proto_ids):
            return False
        xforms = np.asarray(instancer.ComputeInstanceTransformsAtTime(
            self.time, self.time, UsdGeom.PointInstancer.IncludeProtoXform, UsdGeom.PointInstancer.IgnoreMask), dtype=np.float64)
        mask = np.asarray(instancer.ComputeMaskAtTime(self.time) or [True] * len(proto_ids), dtype=bool)
        proto_ids, xforms = proto_ids[mask], xforms[mask]  # inactive / invisible instances are dropped
        instancer_world = np.array(self.xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        protos = instancer.GetPrototypesRel().GetTargets()
        for index, proto_path in enumerate(protos):
            mine = xforms[proto_ids == index]
            proto = self.stage.GetPrimAtPath(proto_path)
            if not len(mine) or not proto:
                continue
            for mesh_prim in Usd.PrimRange(proto, self.predicate):
                if not mesh_prim.IsA(UsdGeom.Mesh):
                    continue
                relative = np.array(self.xform_cache.ComputeRelativeTransform(mesh_prim, proto)[0], dtype=np.float64)
                world = relative[None] @ mine  # row-vector matrices: mesh in prototype, then instance
                key = self._mesh_key(mesh_prim)
                mesh_index = self.mesh_cache.get(key, -1)
                if mesh_index == -1:
                    res = self._instance_mesh(mesh_prim, world[0] @ instancer_world)
                    if res and res[2]:  # UVs projected from world positions: one mesh per instance
                        self._instanced_node(node, mesh_prim, res[0], world[:1])
                        for one in world[1:]:
                            res = self._instance_mesh(mesh_prim, one @ instancer_world)
                            if res:
                                self._instanced_node(node, mesh_prim, res[0], one[None])
                        continue
                    mesh_index = self.mesh_cache.setdefault(key, res[0] if res else None)
                if mesh_index is not None:
                    self._instanced_node(node, mesh_prim, mesh_index, world)
        return bool(node.get("_content"))

    def _instance_mesh(self, mesh_prim: Usd.Prim, world: np.ndarray):
        """``convert_mesh`` for a point instance's mesh at the row-vector world matrix ``world``."""
        res = convert_mesh(mesh_prim, self.time, self.frame, self.gltf, self.materials,
                           Gf.Matrix4d(world.tolist()), self.single_sided, self.skel)
        if res:
            self.stats["triangles"] += res[1]
            self.stats["meshes"] += 1
        return res

    def _instanced_node(self, node: dict, mesh_prim: Usd.Prim, mesh_index: int, world: np.ndarray) -> None:
        """A child of ``node`` drawing ``mesh_index`` at each of the (n, 4, 4) instancer-space
        matrices ``world`` with EXT_mesh_gpu_instancing."""
        # To glTF's frame (C L C^-1, column vectors), then translation / rotation / scale.
        cols = np.transpose(world, (0, 2, 1))
        lin = self.frame.rot @ cols[:, :3, :3] @ self.frame.rot.T
        trans = self.frame.scale * (cols[:, :3, 3] @ self.frame.rot.T)
        sizes = np.linalg.norm(lin, axis=1)
        mirrored = np.linalg.det(lin) < 0  # a reflection is no rotation: carry it in the X scale
        sizes[mirrored, 0] *= -1
        quat = _matrix_to_quaternion(lin / np.where(np.abs(sizes) > 1e-20, sizes, 1e-20)[:, None, :])
        attributes = {"TRANSLATION": self.gltf.accessor(trans, "VEC3"),
                      "ROTATION": self.gltf.accessor(quat, "VEC4"),
                      "SCALE": self.gltf.accessor(sizes, "VEC3")}
        self.gltf.doc["nodes"].append({"name": mesh_prim.GetName(), "mesh": mesh_index,
                                       "extensions": {"EXT_mesh_gpu_instancing": {"attributes": attributes}}})
        node.setdefault("_content", []).append(len(self.gltf.doc["nodes"]) - 1)
        self.gltf.extensions.add("EXT_mesh_gpu_instancing")
        self.stats["point instances"] += len(world)

    def _mesh_key(self, prim: Usd.Prim) -> tuple:
        """What makes two meshes convert identically: their data and their converted materials. Copies
        of one asset that are not USD instances (the same prop placed four times) share one glTF mesh."""
        if prim.IsInstanceProxy():
            proto = str(prim.GetPrimInPrototype().GetPath())
            if proto in self._proto_keys:
                return self._proto_keys[proto]
        mesh = UsdGeom.Mesh(prim)
        h = hashlib.sha1()
        for attr in (mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr(),
                     mesh.GetHoleIndicesAttr(), mesh.GetNormalsAttr()):
            v = attr.Get(self.time)
            h.update(np.asarray(v).tobytes() if v else b"-")
        for token in (mesh.GetOrientationAttr(), mesh.GetSubdivisionSchemeAttr(), mesh.GetDoubleSidedAttr()):
            h.update(str(token.Get(self.time)).encode())
        h.update(self.skel.key(prim, self.time))  # a skinned mesh is drawn in its skeleton's pose
        for pv in sorted(UsdGeom.PrimvarsAPI(prim).GetPrimvars(), key=lambda pv: pv.GetPrimvarName()):
            v = pv.ComputeFlattened(self.time) if pv.HasValue() else None
            h.update(pv.GetPrimvarName().encode() + pv.GetInterpolation().encode())
            if v is None or isinstance(v, str) or not hasattr(v, "__len__"):
                h.update(repr(v).encode())  # scalar primvar (e.g. doNotCastShadows)
            elif len(v) and not isinstance(v[0], str):
                h.update(np.asarray(v).tobytes())
            else:
                h.update(repr(list(v)).encode())
        bindings = [(None, bound_material(prim))]
        for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
            ids = subset.GetIndicesAttr().Get(self.time)
            bindings.append((np.asarray(ids).tobytes() if ids else b"",
                             bound_material(subset.GetPrim()) or subset.GetPrim().GetName()))
        materials = []
        for ids, material in bindings:
            # A UDIM material is represented by its first tile, enough to tell its content apart.
            index, spec = self.materials.get(material or None, True, None)
            if index < 0:
                index = self.materials.get(material, True, None, 1001)[0]
            materials.append((ids, index, spec.key(), str(material.GetPath()) if spec.project and hasattr(material, "GetPath") else ""))
        key = (h.hexdigest(), tuple(materials))
        if prim.IsInstanceProxy():
            self._proto_keys[str(prim.GetPrimInPrototype().GetPath())] = key
        return key

    def _projected_environment(self, prim: Usd.Prim) -> bool:
        """A backdrop dome with Omniverse's OmniProject_Environment material is an environment map
        projected onto geometry: u = (atan2(z, x) + rotation) / 2pi, v = acos(-y) / pi, emitting
        2^exposure * tint. It becomes the scene's panorama sky -- columns rolled to this renderer's
        u = atan2(z, x) / 2pi + 0.5, pixels scaled by the emission -- and the dome is not drawn."""
        material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        shader = MaterialConverter._surface_shader(material, "mdl") if material else None
        source = shader.GetPrim().GetAttribute("info:mdl:sourceAsset") if shader else None
        if not source or not source.Get() or "OmniProject_Environment" not in source.Get().path:
            return False
        inputs = {i.GetBaseName(): i for i in shader.GetInputs()}
        get = lambda n, d: inputs[n].Get() if n in inputs and inputs[n].Get() is not None else d  # noqa: E731
        path = asset_path(get("skydome_map", None), inputs["skydome_map"].GetAttr()) if "skydome_map" in inputs else None
        if not path:
            return False
        if self.gltf.sky is not None:
            self.dome_lights.append(f"{prim.GetPath()}: projected sky skipped, the scene already has one")
            return True
        if UsdGeom.GetStageUpAxis(self.stage) != UsdGeom.Tokens.y:
            log(f"  warning: {prim.GetPath()}: projected sky assumes a Y-up stage")
        scale = 2.0 ** float(get("exposure", 1.0)) * self.radiance_scale
        tint = to_rgb(get("skydome_color_tint", (1, 1, 1)))
        roll = float(get("rotation", 0.0)) / 360.0 - 0.5  # this renderer's u minus the material's
        uri = self.gltf.panorama_from_exr(path, [scale * t for t in tint], roll)
        if uri is None:  # no OpenEXR: the file as is, the brightness reported
            uri = f"{self.gltf.tex_dir.name}/{os.path.basename(path)}"
            self.gltf.tex_dir.mkdir(parents=True, exist_ok=True)
            (self.gltf.tex_dir / os.path.basename(path)).write_bytes(read_asset(path) or b"")
            self.dome_lights.append(f"{prim.GetPath()}: OpenEXR missing, panorama copied unscaled and unrotated: "
                                    f"use --hdrIntensity {scale:g}")
        self.gltf.sky = {"type": "panorama", "panorama": {"equirectangular": True},
                         "extensions": {"NV_environment_sky_panorama": {"uri": uri}}}
        self.dome_lights.append(f"{prim.GetPath()}: projected sky -> panorama {uri} (x{scale:.3g})")
        return True

    def _dome(self, prim: Usd.Prim) -> None:
        """The first dome light becomes the scene's OMI_environment_sky."""
        radiance = light_radiance(prim, self.time) * self.radiance_scale
        color = [c * radiance for c in light_color(prim, self.time)]
        tex = UsdLux.DomeLight(prim).GetTextureFileAttr()
        path = asset_path(tex.Get(self.time), tex) if tex else None
        if self.gltf.sky is not None:
            self.dome_lights.append(f"{prim.GetPath()}: only the first dome light is converted")
            return
        uri = None
        if path and path.lower().endswith(".exr"):  # kept HDR
            uri = self.gltf.panorama_from_exr(path, [1.0, 1.0, 1.0], 0.0)
        if path and uri is None:
            image = self.gltf.image_from_file(path)
            uri = self.gltf.doc["images"][image]["uri"] if image is not None else None
        if path and uri is None:
            self.dome_lights.append(f"{prim.GetPath()}: {os.path.basename(path)} could not be read")
        if uri:
            self.gltf.sky = {"type": "panorama", "panorama": {"equirectangular": True},
                             "extensions": {"NV_environment_sky_panorama": {"uri": uri}}}
            self.dome_lights.append(f"{prim.GetPath()}: panorama {os.path.basename(path)}; its intensity "
                                    f"{radiance:g} has no glTF field: use --hdrIntensity {radiance:g}")
        else:
            self.gltf.sky = {"type": "plain", "plain": {"color": color}}
            self.dome_lights.append(f"{prim.GetPath()}: plain sky, radiance {radiance:g}")

    def _attach(self, node: dict, content: dict) -> None:
        """Cameras and lights look down their own -Z with +Y up, a frame that was not rotated with the
        geometry: under a Z-up stage they hang from a child node carrying the up-axis rotation, named
        like its parent so the scene graph shows the prim's name on both."""
        rot = self.frame.rot
        if np.allclose(rot, np.eye(3)):
            node.update(content)
            return
        m = np.eye(4)
        m[:3, :3] = rot
        self.gltf.doc["nodes"].append({"name": node["name"], **content, "matrix": [float(x) for x in m.T.flatten()]})
        node.setdefault("_content", []).append(len(self.gltf.doc["nodes"]) - 1)

    def _node(self, prim: Usd.Prim, own: bool = False) -> int | None:
        """``own``: ``prim`` is the root of what is converted, placed by someone else (not skipped,
        and its transform is the placement's)."""
        if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader) or prim.IsA(UsdShade.NodeGraph):
            return None
        if self.skip and not own and self.skip(prim):
            return None
        img = UsdGeom.Imageable(prim)
        if img:
            if img.GetVisibilityAttr().Get(self.time) == UsdGeom.Tokens.invisible:
                self.skipped["invisible"] += 1
                return None
            if img.GetPurposeAttr().Get(self.time) in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                self.skipped["guide/proxy purpose"] += 1
                return None
        node: dict = {"name": prim.GetName()}
        xf = UsdGeom.Xformable(prim)
        if xf and not own:
            local = xf.GetLocalTransformation(self.time)
            if xf.GetResetXformStack():  # world-space transform: express it relative to the parent's world matrix
                parent = self.xform_cache.GetLocalToWorldTransform(prim.GetParent())
                local = local * parent.GetInverse()
            m = self.frame.matrix(local)
            if m:
                node["matrix"] = m
        content = False
        if prim.IsA(UsdGeom.Mesh) and self._projected_environment(prim):
            pass
        elif prim.IsA(UsdGeom.Mesh):
            key = self._mesh_key(prim)
            mesh_index = self.mesh_cache.get(key, -1)
            if mesh_index == -1:
                res = convert_mesh(prim, self.time, self.frame, self.gltf, self.materials,
                                   self.xform_cache.GetLocalToWorldTransform(prim), self.single_sided, self.skel)
                mesh_index = res[0] if res else None
                if not (res and res[2]):  # world-projected UVs belong to this instance alone
                    self.mesh_cache[key] = mesh_index
                if res:
                    self.stats["triangles"] += res[1]
                    self.stats["meshes"] += 1
            if mesh_index is not None:
                node["mesh"] = mesh_index
                self.stats["mesh instances"] += 1
                content = True
                # primvars inherit down the namespace, so a flag on an ancestor counts too
                no_shadow = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance("doNotCastShadows")
                if no_shadow and no_shadow.Get(self.time):
                    node["extensions"] = {"NV_node_shadow_casting": {"castShadows": False}}
                    self.gltf.extensions.add("NV_node_shadow_casting")
                    self.stats["meshes not casting shadows"] += 1
        elif prim.IsA(UsdGeom.PointInstancer):
            content = self._point_instancer(prim, node)
        elif prim.IsA(UsdGeom.Gprim):
            self.skipped[f"{prim.GetTypeName()} (not a mesh)"] += 1
        elif prim.IsA(UsdGeom.Camera) and self.scene_cameras is not None:
            self.scene_cameras.append(prim)  # the caller places it in the scene
            self.stats["cameras"] += 1
        elif prim.IsA(UsdGeom.Camera):
            self.gltf.doc["cameras"].append(convert_camera(prim, self.time, self.frame))
            self._attach(node, {"camera": len(self.gltf.doc["cameras"]) - 1})
            self.stats["cameras"] += 1
            content = True
        elif prim.HasAPI(UsdLux.LightAPI) or prim.IsA(UsdLux.BoundableLightBase) or prim.IsA(UsdLux.NonboundableLightBase):
            if prim.IsA(UsdLux.DomeLight):
                self._dome(prim)
            else:
                light = convert_light(prim, self.time, self.frame, self.radiance_scale,
                                      self.xform_cache.GetLocalToWorldTransform(prim))
                if light:
                    self.gltf.lights.append(light)
                    self._attach(node, {"extensions": {"KHR_lights_punctual": {"light": len(self.gltf.lights) - 1}}})
                    self.stats["lights"] += 1
                    content = True
                else:
                    self.skipped[f"{prim.GetTypeName()} light"] += 1
        if prim.IsA(UsdGeom.PointInstancer):
            prototypes = {str(t) for t in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets()}
            kids = [ch for ch in prim.GetFilteredChildren(self.predicate)
                    if not any(str(ch.GetPath()) == t or t.startswith(str(ch.GetPath()) + "/") for t in prototypes)]
        else:
            kids = list(prim.GetFilteredChildren(self.predicate))
        children = node.pop("_content", []) + [c for c in (self._node(ch) for ch in kids) if c is not None]
        if children:
            node["children"] = children
        if not content and not children:
            return None
        self.gltf.doc["nodes"].append(node)
        return len(self.gltf.doc["nodes"]) - 1


def select_variant(stage: Usd.Stage, name: str, value: str) -> int:
    """Select ``value`` in every variant set called ``name``, in the session layer (the files are not
    touched). Repeats until stable, since a selection can bring in more prims carrying the set.
    Prims inside instances cannot be edited; their selection comes from the instance itself."""
    total = 0
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        while True:
            todo = [prim.GetPath() for prim in stage.Traverse()
                    if prim.GetVariantSets().HasVariantSet(name)
                    and value in prim.GetVariantSet(name).GetVariantNames()
                    and prim.GetVariantSet(name).GetVariantSelection() != value]
            if not todo:
                return total
            with Sdf.ChangeBlock():
                for path in todo:
                    stage.GetPrimAtPath(path).GetVariantSet(name).SetVariantSelection(value)
            total += len(todo)


def main() -> int:
    with usd_diagnostics():  # USD's warnings while composing, the .mtlx ones as one note
        return _main()


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", help="USD stage (.usd, .usda, .usdc, .usdz)")
    ap.add_argument("output", help="Output .gltf (a .bin and a <name>_textures/ folder are written next to it)")
    ap.add_argument("--materials", choices=("auto", "preview"), default="auto",
                    help="auto: MDL when understood, else UsdPreviewSurface; preview: UsdPreviewSurface only")
    ap.add_argument("--keep-exposure", action="store_true",
                    help="Keep physical light values instead of baking the cameras' exposure into them")
    ap.add_argument("--variant", action="append", default=[], metavar="SET=VALUE",
                    help="Select VALUE in every variant set named SET before converting, e.g. material=renderLow "
                         "(repeatable)")
    ap.add_argument("--single-sided", action="store_true",
                    help="Honor USD doubleSided (back faces culled); by default every mesh is double-sided, "
                         "as Omniverse RTX renders it")
    args = ap.parse_args()

    out = Path(args.output).resolve()
    if out.suffix.lower() != ".gltf":
        ap.error("output must be a .gltf file")
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    stage = Usd.Stage.Open(args.input)
    if not stage:
        log(f"error: cannot open {args.input}")
        return 1
    log(f"Opened {args.input}: up axis {UsdGeom.GetStageUpAxis(stage)}, {UsdGeom.GetStageMetersPerUnit(stage)} m/unit")
    for spec in args.variant:
        name, _, value = spec.partition("=")
        log(f"  variant {name}={value}: {select_variant(stage, name, value)} prims")
    conv = Converter(stage, out, args.materials, args.keep_exposure, args.single_sided)
    conv.run()
    conv.gltf.write()

    log(f"Wrote {out} in {time.time() - t0:.1f} s")
    for k, v in conv.stats.items():
        log(f"  {k}: {v:,}")
    log(f"  materials: {len(conv.gltf.doc.get('materials', []))}, images: {len(conv.gltf.doc.get('images', []))}")
    for k, v in sorted(conv.materials.report.items(), key=lambda x: -x[1]):
        log(f"    {v:4d} from {k}")
    for k, v in conv.skipped.items():
        log(f"  skipped {k}: {v}")
    for d in conv.dome_lights:
        log(f"  dome light {d}")
    if len(conv.exposures) > 1:
        log(f"  cameras disagree on exposure {dict(conv.exposures)}; used {conv.exposure:g}")
    if conv.radiance_scale != 1.0:
        log(f"  {conv.exposure_source} ({conv.exposure:.2f} stops) baked into lights, emission and sky "
            f"(the renderer's auto-exposure works as is; exposure 1 shows the stage's own)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
