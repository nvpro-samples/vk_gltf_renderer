from __future__ import annotations

import os
from pathlib import Path


def benchmark_dir() -> Path:
    """Directory containing the benchmark scripts and .cfg files."""
    return Path(__file__).resolve().parent


def project_root() -> Path:
    """vk_gltf_renderer repository root."""
    return benchmark_dir().parent.parent


def resolve_input_path(path_str: str, root: Path | None = None) -> Path:
    """Resolve scene/HDR paths relative to the repo root."""
    root = root or project_root()
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def resolve_cfg_path(path_str: str) -> Path:
    """Resolve .cfg under utils/benchmark/ or the repo root."""
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    for base in (benchmark_dir(), project_root()):
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate
    return (benchmark_dir() / path).resolve()


def resolve_output_dir(path_str: str) -> Path:
    """Default output is utils/benchmark/output/."""
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (benchmark_dir() / path).resolve()


def resolve_hdr_path(hdr_str: str, root: Path | None = None) -> str | None:
    """Resolve HDR under resources/; fall back to std_env.hdr when missing."""
    if not hdr_str:
        return None
    root = root or project_root()
    candidates = [
        resolve_input_path(hdr_str, root),
        root / "resources" / hdr_str,
        root / "resources" / Path(hdr_str).name,
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    for fallback in ("std_env.hdr", "env3.hdr"):
        path = root / "resources" / fallback
        if path.is_file():
            print(f"Warning: HDR '{hdr_str}' not found, using {path}")
            return str(path)
    print(f"Warning: HDR '{hdr_str}' not found; executable will search resource dirs")
    return str(candidates[0])


def find_executable() -> str | None:
    root = project_root()
    candidates = [
        root / "_bin/Release/vk_gltf_renderer.exe",
        root / "_bin/Release/vk_gltf_renderer",
        root / "_bin/Debug/vk_gltf_renderer.exe",
        root / "_bin/Debug/vk_gltf_renderer",
    ]
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    return None


def resolve_executable(executable: str | None) -> str | None:
    if executable is None:
        return find_executable()
    path = Path(executable)
    if not path.is_file() or not os.access(path, os.X_OK):
        return None
    return str(path.resolve())


def load_scenes_file(path: str, scenes_root: str) -> dict[str, str]:
    scenes: dict[str, str] = {}
    root = Path(scenes_root)
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, rel = line.split(None, 1) if " " in line else (Path(line).stem, line)
            scenes[name] = str((root / rel).resolve())
    return scenes
