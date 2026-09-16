#!/usr/bin/env python3
#
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import argparse
import concurrent.futures
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path

try:
    import winreg
except ImportError:
    winreg = None


KHR_TEXTURE_BASISU = "KHR_texture_basisu"
KTX2_MAGIC = b"\xABKTX 20\xBB\r\n\x1A\n"
DDS_MAGIC = b"DDS "
TEXTURE_SOURCE_EXTENSIONS = (
    "EXT_texture_webp",
    "MSFT_texture_dds",
    KHR_TEXTURE_BASISU,
)


def decode_uri(uri):
    return urllib.parse.unquote(uri)


def encode_uri(path):
    return urllib.parse.quote(path.as_posix(), safe="/-._~")


def uri_to_path(gltf_dir, uri):
    decoded = decode_uri(uri)
    path = Path(decoded)
    if path.is_absolute():
        return path
    return gltf_dir / path


def path_to_uri(path, base_dir, allow_absolute):
    try:
        relative = Path(os.path.relpath(path.resolve(), base_dir.resolve()))
    except ValueError as exc:
        if not allow_absolute:
            raise ValueError(
                f"Cannot make '{path}' relative to '{base_dir}'. "
                "Use --allow-absolute-uris or choose an output directory under the glTF output folder."
            ) from exc
        return encode_uri(path.resolve())
    return encode_uri(relative)


def get_list(obj, key):
    value = obj.get(key)
    return value if isinstance(value, list) else []


def add_unique_extension(gltf, key, extension_name):
    extensions = gltf.setdefault(key, [])
    if extension_name not in extensions:
        extensions.append(extension_name)


def remove_unique_extension(gltf, key, extension_name):
    extensions = gltf.get(key)
    if isinstance(extensions, list):
        gltf[key] = [ext for ext in extensions if ext != extension_name]
        if not gltf[key]:
            del gltf[key]


def is_data_uri(uri):
    return uri.lower().startswith("data:")


def image_name(image, index):
    return image.get("name") or f"image_{index}"


def texture_sources(texture):
    sources = []
    source = texture.get("source")
    if isinstance(source, int):
        sources.append(("source", source))

    extensions = texture.get("extensions", {})
    if not isinstance(extensions, dict):
        return sources

    for extension_name in TEXTURE_SOURCE_EXTENSIONS:
        extension = extensions.get(extension_name)
        if isinstance(extension, dict) and isinstance(extension.get("source"), int):
            sources.append((extension_name, extension["source"]))
    return sources


def classify_texture_images(gltf):
    """Return (normal_image_indices, linear_image_indices).

    ``normal_image_indices`` are images used as tangent-space normal maps. They get
    nvcompress ``-normal``, and their encoder format comes from ``encoder_format()``,
    which keeps an already-two-channel source two-channel.

    ``linear_image_indices`` are images that carry non-color data per the glTF
    2.0 spec and its official Khronos material extensions — normal, occlusion,
    metallic-roughness, transmission, thickness, sheen roughness, clearcoat,
    clearcoat roughness, iridescence, iridescence thickness, anisotropy, and
    the scalar specular / diffuse-transmission channels. These must be tagged
    with a linear transfer function in the KTX2 output; otherwise Vulkan will
    apply sRGB→linear on sample and crush the data (a normal-map (128,128,255)
    becomes near-zero, roughness values shift, etc.). ``normal_image_indices``
    is a subset of ``linear_image_indices``.

    Color textures (baseColor, emissive, sheen color, specular color,
    diffuse-transmission color) are intentionally left off — the encoder's
    default sRGB tag is correct for them.
    """
    normal_texture_indices = set()
    linear_texture_indices = set()

    def add_linear(tex_ref):
        if isinstance(tex_ref, dict) and isinstance(tex_ref.get("index"), int):
            linear_texture_indices.add(tex_ref["index"])

    def add_normal(tex_ref):
        if isinstance(tex_ref, dict) and isinstance(tex_ref.get("index"), int):
            normal_texture_indices.add(tex_ref["index"])
            linear_texture_indices.add(tex_ref["index"])

    for material in get_list(gltf, "materials"):
        add_normal(material.get("normalTexture"))
        add_linear(material.get("occlusionTexture"))
        pbr = material.get("pbrMetallicRoughness")
        if isinstance(pbr, dict):
            add_linear(pbr.get("metallicRoughnessTexture"))

        extensions = material.get("extensions", {})
        if not isinstance(extensions, dict):
            continue

        clearcoat = extensions.get("KHR_materials_clearcoat")
        if isinstance(clearcoat, dict):
            add_normal(clearcoat.get("clearcoatNormalTexture"))
            add_linear(clearcoat.get("clearcoatTexture"))
            add_linear(clearcoat.get("clearcoatRoughnessTexture"))

        transmission = extensions.get("KHR_materials_transmission")
        if isinstance(transmission, dict):
            add_linear(transmission.get("transmissionTexture"))

        volume = extensions.get("KHR_materials_volume")
        if isinstance(volume, dict):
            add_linear(volume.get("thicknessTexture"))

        specular = extensions.get("KHR_materials_specular")
        if isinstance(specular, dict):
            add_linear(specular.get("specularTexture"))

        sheen = extensions.get("KHR_materials_sheen")
        if isinstance(sheen, dict):
            add_linear(sheen.get("sheenRoughnessTexture"))

        iridescence = extensions.get("KHR_materials_iridescence")
        if isinstance(iridescence, dict):
            add_linear(iridescence.get("iridescenceTexture"))
            add_linear(iridescence.get("iridescenceThicknessTexture"))

        anisotropy = extensions.get("KHR_materials_anisotropy")
        if isinstance(anisotropy, dict):
            add_linear(anisotropy.get("anisotropyTexture"))

        diffuse_transmission = extensions.get("KHR_materials_diffuse_transmission")
        if isinstance(diffuse_transmission, dict):
            add_linear(diffuse_transmission.get("diffuseTransmissionTexture"))

    textures = get_list(gltf, "textures")

    def texture_indices_to_image_indices(texture_indices):
        image_indices = set()
        for texture_index in texture_indices:
            if not 0 <= texture_index < len(textures):
                continue
            for _, image_index in texture_sources(textures[texture_index]):
                image_indices.add(image_index)
        return image_indices

    normal_image_indices = texture_indices_to_image_indices(normal_texture_indices)
    linear_image_indices = texture_indices_to_image_indices(linear_texture_indices)
    return normal_image_indices, linear_image_indices


def discover_windows_nvtt_tools():
    if os.name != "nt":
        return {}

    candidates = []
    env_program_files = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
    ]
    for base in env_program_files:
        if base:
            candidates.append(Path(base) / "NVIDIA Corporation" / "NVIDIA Texture Tools")

    if winreg is not None:
        roots = (
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        )
        for root, uninstall_key in roots:
            try:
                with winreg.OpenKey(root, uninstall_key) as key:
                    for index in range(winreg.QueryInfoKey(key)[0]):
                        try:
                            subkey_name = winreg.EnumKey(key, index)
                            with winreg.OpenKey(key, subkey_name) as subkey:
                                display_name = read_registry_value(subkey, "DisplayName")
                                if "NVIDIA Texture Tools" not in display_name:
                                    continue
                                install_location = read_registry_value(subkey, "InstallLocation")
                                if install_location:
                                    candidates.append(Path(install_location))
                        except OSError:
                            continue
            except OSError:
                continue

    tools = {}
    for tool_name in ("nvtt_export.exe", "nvcompress.exe"):
        path = shutil.which(tool_name)
        if path:
            tools[tool_name] = Path(path)

    for directory in candidates:
        for tool_name in ("nvtt_export.exe", "nvcompress.exe"):
            candidate = directory / tool_name
            if candidate.exists():
                tools.setdefault(tool_name, candidate)
    return tools


def read_registry_value(key, name):
    try:
        value, _ = winreg.QueryValueEx(key, name)
    except OSError:
        return ""
    return str(value)


def resolve_tool(args):
    if args.tool == "custom":
        if not args.command_template:
            raise SystemExit("--tool custom requires --command-template")
        return None

    exe_name = "nvcompress.exe" if args.tool == "nvcompress" else "nvtt_export.exe"

    if args.tool_path:
        path = Path(args.tool_path)
        if not path.exists():
            raise SystemExit(f"Tool path does not exist: {path}")
        return path

    path = shutil.which(exe_name)
    if path:
        return Path(path)

    tools = discover_windows_nvtt_tools()
    path = tools.get(exe_name)
    if path:
        return path

    raise SystemExit(f"Could not find {exe_name}. Pass --tool-path.")


def print_discovered_tools():
    tools = discover_windows_nvtt_tools()
    for tool in ("nvtt_export.exe", "nvcompress.exe"):
        path = tools.get(tool)
        if path:
            print(f"{tool}: {path}")
        else:
            found = shutil.which(tool)
            print(f"{tool}: {found or '(not found)'}")


# VkFormat / DXGI_FORMAT values that store exactly two channels. Mirrors the two-channel arm of
# vkFormatColorComponentCount() in src/gltf_scene_vk.cpp -- that function is what decides whether the
# renderer rebuilds a normal map's Z, so the two lists have to agree.
_TWO_CHANNEL_VK_FORMATS = frozenset(
    {
        16, 17, 22,  # R8G8 UNORM / SNORM / SRGB
        77, 78, 83,  # R16G16 UNORM / SNORM / SFLOAT
        103,  # R32G32_SFLOAT
        141, 142,  # BC5 UNORM / SNORM
        155, 156,  # EAC_R11G11 UNORM / SNORM
    }
)
_TWO_CHANNEL_DXGI_FORMATS = frozenset(
    {
        15, 16, 17, 18,  # R32G32 typeless / float / uint / sint
        33, 34, 35, 36, 37, 38,  # R16G16 family
        48, 49, 50, 51, 52,  # R8G8 family
        82, 83, 84,  # BC5 typeless / UNORM / SNORM
    }
)
# Pre-DX10 FourCC codes for BC5. ATI2 is the original 3Dc name; nvidia tools also emit BC5U/BC5S.
_TWO_CHANNEL_FOURCC = frozenset({b"ATI2", b"BC5U", b"BC5S", b"A2XY"})


def source_is_two_channel(path):
    """True when ``path`` is an image whose pixel format stores exactly two channels.

    Only DDS and KTX2 can be two-channel in practice; PNG and JPEG always decode to grayscale or
    RGB(A), so they return False and keep the regular --format. Anything unreadable or unrecognized
    also returns False, which leaves the existing behaviour untouched rather than guessing.
    """
    try:
        with path.open("rb") as stream:
            header = stream.read(152)
    except OSError:
        return False

    if header.startswith(DDS_MAGIC) and len(header) >= 128:
        fourcc = header[84:88]
        if fourcc == b"DX10":
            if len(header) < 132:
                return False
            return int.from_bytes(header[128:132], "little") in _TWO_CHANNEL_DXGI_FORMATS
        return fourcc in _TWO_CHANNEL_FOURCC

    if header.startswith(KTX2_MAGIC) and len(header) >= 16:
        return int.from_bytes(header[12:16], "little") in _TWO_CHANNEL_VK_FORMATS

    return False


def encoder_format(args, input_path, is_normal_map):
    """Pick the encoder format for one image.

    A tangent-space normal map whose source is already two-channel (a BC5 / RG DDS or KTX2) must stay
    two-channel. Such a source has no blue channel; expanding it to a three-channel format like BC7
    makes the encoder bake the synthetic blue -- a real 0 -- into the data, which decodes to Z = -1
    and renders black. BC5 keeps it two-channel, and the renderer rebuilds Z from X/Y.

    Normal maps that already carry a real blue channel (PNG, JPEG, an RGB DDS) are left on --format;
    BC7 handles them correctly. An explicit --normal-format overrides both cases.
    """
    if not is_normal_map:
        return args.format
    if args.normal_format:
        return args.normal_format
    if source_is_two_channel(input_path):
        logging.info("%s is a two-channel normal map; encoding as bc5 to keep Z reconstructable", input_path)
        return "bc5"
    return args.format


def build_command(args, tool_path, input_path, output_path, is_normal_map, is_linear_data):
    if args.tool == "custom":
        values = {
            "input": str(input_path),
            "output": str(output_path),
            "format": encoder_format(args, input_path, is_normal_map),
        }
        try:
            command = args.command_template.format(**values)
        except KeyError as exc:
            raise SystemExit(f"Unknown placeholder in --command-template: {exc}") from exc
        if os.name == "nt":
            return command
        return shlex.split(command)

    if args.tool == "nvcompress":
        fmt = encoder_format(args, input_path, is_normal_map)
        command = [str(tool_path), "-silent"]
        nvcompress_quality = "fast" if args.quality == "fastest" else args.quality
        if nvcompress_quality in ("fast", "production", "highest"):
            command.append(f"-{nvcompress_quality}")
        if is_normal_map:
            command.append("-normal")
        if not args.mips:
            command.append("-nomips")
        command.extend(args.encoder_arg)
        command.extend([f"-{fmt}", str(input_path), str(output_path)])
        return command

    fmt = encoder_format(args, input_path, is_normal_map)
    nvtt_quality = "fastest" if args.quality == "fast" else args.quality
    command = [
        str(tool_path),
        "--format",
        fmt,
        "--quality",
        nvtt_quality,
    ]
    if args.mips:
        command.append("--mips")
    else:
        command.append("--no-mips")
    if args.zcmp is not None:
        command.extend(["--zcmp", str(args.zcmp)])
    if args.no_cuda:
        command.append("--no-cuda")
    # Force a linear transfer function for non-color data (normal, occlusion,
    # metallic-roughness, transmission, thickness, etc.). Without this, nvtt_export
    # tags BC7/BC1/BC3 output as *_SRGB, Vulkan applies sRGB→linear on sample,
    # and normal / roughness / metallic values are silently crushed. Color
    # textures (baseColor, emissive, ...) keep the default (sRGB) tag.
    if is_linear_data:
        command.extend(["--export-transfer-function", "linear"])
    command.extend(args.encoder_arg)
    command.extend(["--output-file", str(output_path), str(input_path)])
    return command


def apply_fast_preset(args):
    if not args.fast:
        return
    if args.quality == "normal":
        args.quality = "fastest"
    if args.zcmp == 5:
        args.zcmp = None


def run_encoder(args, tool_path, input_path, output_path, is_normal_map, is_linear_data):
    command = build_command(args, tool_path, input_path, output_path, is_normal_map, is_linear_data)
    logging.info("Encoding %s -> %s", input_path, output_path)
    logging.debug("Command: %s", format_command(command))
    if args.dry_run:
        print(format_command(command))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, capture_output=True, text=True, shell=isinstance(command, str))
    if result.stdout:
        logging.debug(result.stdout.strip())
    if result.stderr:
        logging.debug(result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"Encoder failed for '{input_path}' with exit code {result.returncode}: {result.stderr.strip()}")

    validate_ktx2(output_path, command)


def validate_ktx2(path, command):
    if not path.exists():
        raise RuntimeError(f"Encoder reported success but did not create '{path}'. Command: {format_command(command)}")

    with path.open("rb") as stream:
        header = stream.read(len(KTX2_MAGIC))
    if header == KTX2_MAGIC:
        return

    if header.startswith(DDS_MAGIC):
        raise RuntimeError(
            f"'{path}' is a DDS file, not KTX2. This installed nvcompress.exe writes DDS containers even "
            "when the output name ends in .ktx2. Use nvtt_export.exe, toktx, or a custom command that "
            "writes a real KTX2 file."
        )
    raise RuntimeError(f"'{path}' is not a KTX2 file; header bytes were {header!r}")


def format_command(command):
    return command if isinstance(command, str) else subprocess.list2cmdline(command)


def choose_output_path(args, source_path, source_uri, image_index, output_gltf_dir, used_paths):
    if args.texture_output_dir:
        output_dir = Path(args.texture_output_dir)
        if not output_dir.is_absolute():
            output_dir = output_gltf_dir / output_dir
        destination = output_dir / f"{source_path.stem}.ktx2"
    else:
        decoded = Path(decode_uri(source_uri))
        destination = output_gltf_dir / decoded.with_suffix(".ktx2")

    if destination not in used_paths:
        used_paths.add(destination)
        return destination

    destination = destination.with_name(f"{destination.stem}_{image_index}{destination.suffix}")
    used_paths.add(destination)
    return destination


def convert_gltf(args):
    apply_fast_preset(args)

    input_gltf = Path(args.input_gltf)
    if input_gltf.suffix.lower() != ".gltf":
        raise SystemExit("This script reads JSON .gltf files only; .glb files are not supported.")
    if not input_gltf.exists():
        raise SystemExit(f"Input glTF does not exist: {input_gltf}")

    if args.in_place:
        output_gltf = input_gltf
    elif args.output_gltf:
        output_gltf = Path(args.output_gltf)
    else:
        output_gltf = input_gltf.with_name(f"{input_gltf.stem}_ktx2.gltf")

    if output_gltf.suffix.lower() != ".gltf":
        raise SystemExit("Output path must end in .gltf")

    tool_path = resolve_tool(args)
    logging.info("Input glTF: %s", input_gltf)
    logging.info("Output glTF: %s", output_gltf)
    if tool_path:
        logging.info("Texture tool: %s", tool_path)

    with input_gltf.open("r", encoding="utf-8-sig") as stream:
        gltf = json.load(stream)

    images = get_list(gltf, "images")
    if not images:
        logging.warning("No images found.")
        return 0

    input_gltf_dir = input_gltf.parent
    output_gltf_dir = output_gltf.parent
    normal_images, linear_images = classify_texture_images(gltf)
    converted_indices = set()
    used_output_paths = set()
    conversion_records = []
    encode_records = []

    for image_index, image in enumerate(images):
        uri = image.get("uri")
        if not isinstance(uri, str) or not uri:
            logging.warning("Skipping %s: image has no external URI.", image_name(image, image_index))
            continue
        if is_data_uri(uri):
            logging.warning("Skipping %s: data URIs are not converted.", image_name(image, image_index))
            continue
        if image.get("bufferView") is not None:
            logging.warning("Skipping %s: bufferView-backed images are not converted.", image_name(image, image_index))
            continue

        source_path = uri_to_path(input_gltf_dir, uri)
        if not source_path.exists():
            logging.warning("Skipping %s: source image does not exist: %s", image_name(image, image_index), source_path)
            continue

        output_path = choose_output_path(args, source_path, uri, image_index, output_gltf_dir, used_output_paths)
        output_uri = path_to_uri(output_path, output_gltf_dir, args.allow_absolute_uris)

        if output_path.exists() and not args.force:
            validate_ktx2(output_path, ["existing", str(output_path)])
            logging.info("Keeping existing %s", output_path)
        else:
            encode_records.append(
                {
                    "source_path": source_path,
                    "output_path": output_path,
                    "is_normal_map": image_index in normal_images,
                    "is_linear_data": image_index in linear_images,
                }
            )

        conversion_records.append(
            {
                "image_index": image_index,
                "image": image,
                "output_uri": output_uri,
            }
        )
        converted_indices.add(image_index)

    run_encoder_jobs(args, tool_path, encode_records)

    if not converted_indices:
        logging.warning("No images were converted.")
        return 0

    if not args.dry_run:
        for record in conversion_records:
            image = record["image"]
            image["uri"] = record["output_uri"]
            image["mimeType"] = "image/ktx2"
            image.pop("bufferView", None)

        update_textures_for_basisu(gltf, converted_indices, keep_fallback_source=args.keep_fallback_source)
        add_unique_extension(gltf, "extensionsUsed", KHR_TEXTURE_BASISU)
        if args.extension_required:
            add_unique_extension(gltf, "extensionsRequired", KHR_TEXTURE_BASISU)
        else:
            remove_unique_extension(gltf, "extensionsRequired", KHR_TEXTURE_BASISU)

        output_gltf.parent.mkdir(parents=True, exist_ok=True)
        with output_gltf.open("w", encoding="utf-8") as stream:
            json.dump(gltf, stream, indent=2)
            stream.write("\n")

    logging.info("Converted %d image(s).", len(converted_indices))
    return len(converted_indices)


def run_encoder_jobs(args, tool_path, records):
    if not records:
        return

    if args.jobs <= 1 or args.dry_run:
        for record in records:
            run_encoder(
                args,
                tool_path,
                record["source_path"],
                record["output_path"],
                record["is_normal_map"],
                record["is_linear_data"],
            )
        return

    logging.info("Encoding %d image(s) with %d parallel job(s).", len(records), args.jobs)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs)
    futures = [
        executor.submit(
            run_encoder,
            args,
            tool_path,
            record["source_path"],
            record["output_path"],
            record["is_normal_map"],
            record["is_linear_data"],
        )
        for record in records
    ]
    try:
        for future in concurrent.futures.as_completed(futures):
            future.result()
    except Exception:
        for future in futures:
            future.cancel()
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            executor.shutdown(wait=False)
        raise
    else:
        executor.shutdown(wait=True)


def update_textures_for_basisu(gltf, converted_indices, keep_fallback_source):
    for texture in get_list(gltf, "textures"):
        if not isinstance(texture, dict):
            continue

        source = texture.get("source")
        basis_source = None
        if isinstance(source, int) and source in converted_indices:
            basis_source = source

        extensions = texture.setdefault("extensions", {})
        if not isinstance(extensions, dict):
            extensions = {}
            texture["extensions"] = extensions

        existing_basis = extensions.get(KHR_TEXTURE_BASISU)
        if isinstance(existing_basis, dict) and isinstance(existing_basis.get("source"), int):
            if existing_basis["source"] in converted_indices:
                basis_source = existing_basis["source"]

        if basis_source is None:
            if not extensions:
                texture.pop("extensions", None)
            continue

        extensions[KHR_TEXTURE_BASISU] = {"source": basis_source}
        if not keep_fallback_source:
            texture.pop("source", None)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Convert external images referenced by a JSON .gltf file to KTX2 and update texture "
            "metadata to use KHR_texture_basisu."
        )
    )
    parser.add_argument("input_gltf", nargs="?", help="Input .gltf file")
    parser.add_argument("output_gltf", nargs="?", help="Output .gltf file. Defaults to <input>_ktx2.gltf")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input .gltf JSON.")
    parser.add_argument("--texture-output-dir", help="Directory for generated .ktx2 files, relative to output .gltf.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing .ktx2 files.")
    parser.add_argument("--dry-run", action="store_true", help="Print encoder commands and do not write files.")
    parser.add_argument("--allow-absolute-uris", action="store_true", help="Allow absolute image URIs in output JSON.")

    parser.add_argument(
        "--tool",
        choices=("auto", "nvtt_export", "nvcompress", "custom"),
        default="auto",
        help="Texture encoder to run. auto uses nvtt_export.exe when available.",
    )
    parser.add_argument("--tool-path", help="Explicit path to nvtt_export.exe or nvcompress.exe.")
    parser.add_argument(
        "--command-template",
        help="Custom command with {input}, {output}, and {format} placeholders. Used with --tool custom.",
    )
    parser.add_argument(
        "--encoder-arg",
        action="append",
        default=[],
        help="Additional argument passed to the encoder. Repeat for multiple arguments.",
    )
    parser.add_argument("--format", default="bc7", help="Encoder format. For KTX2, nvtt_export supports uastc/etc1s-rgb/etc1s-rgba.")
    parser.add_argument(
        "--normal-format",
        help="Encoder format for tangent-space normal-map images. Defaults to --format, except for "
             "a source that is already two-channel (a BC5 / RG DDS or KTX2), which is encoded as bc5 "
             "so it stays two-channel. Setting this overrides both.",
    )
    parser.add_argument("--quality", choices=("fastest", "fast", "normal", "production", "highest"), default="normal")
    parser.add_argument("--zcmp", type=int, default=5, help="KTX2 Zstandard supercompression level for nvtt_export.")
    parser.add_argument("--no-zcmp", dest="zcmp", action="store_const", const=None, help="Skip KTX2 Zstandard supercompression.")
    parser.add_argument("--fast", action="store_true", help="Iteration preset: use fastest quality and skip Zstandard compression.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel encoder processes to run for new textures.")
    parser.add_argument("--no-mips", dest="mips", action="store_false", help="Disable mipmap generation.")
    parser.add_argument("--no-cuda", action="store_true", help="Disable CUDA in nvtt_export.")
    parser.set_defaults(mips=True)

    parser.add_argument(
        "--keep-fallback-source",
        action="store_true",
        help="Keep texture.source instead of moving all converted textures to KHR_texture_basisu only.",
    )
    parser.add_argument(
        "--optional-extension",
        dest="extension_required",
        action="store_false",
        help="Add KHR_texture_basisu to extensionsUsed only, not extensionsRequired.",
    )
    parser.set_defaults(extension_required=True)
    parser.add_argument("--print-tools", action="store_true", help="Print discovered NVIDIA Texture Tools paths and exit.")
    parser.add_argument("--log", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log), format="%(levelname)s: %(message)s")

    if args.print_tools:
        print_discovered_tools()
        return 0

    if not args.input_gltf:
        raise SystemExit("input_gltf is required unless --print-tools is used.")
    if args.in_place and args.output_gltf:
        raise SystemExit("Use either --in-place or output_gltf, not both.")

    try:
        convert_gltf(args)
    except Exception as exc:
        logging.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
