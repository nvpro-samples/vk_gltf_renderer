from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_paths import load_scenes_file, project_root, resolve_cfg_path, resolve_hdr_path, resolve_input_path, resolve_output_dir
from benchmark_results import parse_benchmark, parse_headless_summary, save_headless_csv, save_to_csv


@dataclass(frozen=True)
class SceneInput:
    name: str
    path: str


@dataclass
class HeadlessRun:
    label: str
    log_path: Path
    exit_code: int
    summary: dict[str, str] | None


def run_benchmark(
    executable: str,
    benchmark_file: str,
    scene_path: str,
    hdr_path: str | None,
    output_log: str,
    extra_args: list[str] | None = None,
) -> int:
    command = [
        executable,
        "--size",
        "1920",
        "1080",
        "--benchmark",
        "1",
        "--sequencefile",
        benchmark_file,
        "--scenefile",
        scene_path,
    ]
    if hdr_path:
        command.extend(["--hdrfile", hdr_path])
    if extra_args:
        command.extend(extra_args)
    with open(output_log, "w", encoding="utf-8") as log_file:
        return subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False).returncode


def _resolve_scenes(args: argparse.Namespace) -> list[SceneInput]:
    root = project_root()
    if args.scenes_file:
        scenes_root = str(resolve_input_path(args.scenes_root, root))
        scenes = load_scenes_file(str(resolve_input_path(args.scenes_file, root)), scenes_root)
        return [SceneInput(name, path) for name, path in scenes.items()]
    scene_path = str(resolve_input_path(args.scene, root))
    return [SceneInput(Path(args.scene).stem, scene_path)]


def run_matrix(args: argparse.Namespace, executable: str) -> int:
    benchmark_cfg = str(resolve_cfg_path(args.benchmark))
    if not os.path.isfile(benchmark_cfg):
        print(f"Benchmark config not found: {benchmark_cfg}")
        return 1

    hdr_resolved = resolve_hdr_path(args.hdr, project_root())
    out_dir = resolve_output_dir(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_cwd = os.getcwd()
    os.chdir(out_dir)
    all_results: list[dict[str, Any]] = []
    exit_code = 0
    try:
        for scene in _resolve_scenes(args):
            if not os.path.exists(scene.path):
                print(f"Skipping missing scene: {scene.path}")
                exit_code = 1
                continue
            output_prefix = f"benchmark_{scene.name.replace(' ', '_')}"
            output_log = f"{output_prefix}.log"
            print(f"Running benchmark for {scene.name} ...")
            print(f"  cfg: {benchmark_cfg}")
            print(f"  scene: {scene.path}")
            if hdr_resolved:
                print(f"  hdr: {hdr_resolved}")
            rc = run_benchmark(
                executable,
                benchmark_cfg,
                scene.path,
                hdr_resolved,
                output_log,
                extra_args=args.extra_args.split() if args.extra_args else None,
            )
            if rc != 0:
                print(f"  Warning: process exited with code {rc}")
                exit_code = rc
            with open(output_log, encoding="utf-8") as file:
                all_results.extend(parse_benchmark(file.read(), scene.name))

        csv_path = args.csv_name
        save_to_csv(all_results, csv_path)
        print(f"Wrote {out_dir / csv_path} ({len(all_results)} sequences)")
    finally:
        os.chdir(prev_cwd)
    return exit_code


def run_headless_timing(
    executable: str,
    scene_path: str,
    hdr_path: str | None,
    frames: int,
    spp_list: list[int],
    output_dir: Path,
    extra_args: list[str] | None = None,
) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for spp in spp_list:
        label = f"{Path(scene_path).stem}_spp{spp}"
        run = _run_single_headless(executable, scene_path, hdr_path, frames, spp, output_dir, extra_args)
        row = {
            "label": label,
            "frames": str(frames),
            "ptSamples": str(spp),
            "exit_code": str(run.exit_code),
            "log": str(run.log_path),
        }
        if run.summary:
            row.update(run.summary)
            extra = ""
            if "spp_per_sec" in run.summary:
                extra = (
                    f"  throughput_MSps={run.summary['throughput_MSps']}  "
                    f"spp/s={run.summary['spp_per_sec']}"
                )
            print(
                f"  wall_ms={run.summary['wall_ms']}  ms/frame={run.summary['ms_per_frame']}  "
                f"effective_spp={run.summary['effective_spp']}{extra}"
            )
        else:
            print("  Warning: headless summary not found in log (rebuild vk_gltf_renderer?)")
            _try_add_fallback_wall_time(row, run.log_path)
        results.append(row)
    return results


def _run_single_headless(
    executable: str,
    scene_path: str,
    hdr_path: str | None,
    frames: int,
    spp: int,
    output_dir: Path,
    extra_args: list[str] | None,
) -> HeadlessRun:
    label = f"{Path(scene_path).stem}_spp{spp}"
    log_path = output_dir / f"headless_{label}.log"
    command = [
        executable,
        "--headless",
        "--size",
        "1920",
        "1080",
        "--frames",
        str(frames),
        "--maxFrames",
        str(frames),
        "--ptSamples",
        str(spp),
        "--ptAdaptiveSampling",
        "0",
        "--renderSystem",
        "0",
        "--envSystem",
        "1",
        "--scenefile",
        scene_path,
    ]
    if hdr_path:
        command.extend(["--hdrfile", hdr_path])
    if extra_args:
        command.extend(extra_args)
    print(f"Headless run: {label} ({frames} frames, {spp} spp/frame) ...")
    with open(log_path, "w", encoding="utf-8") as log_file:
        completed = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    log_text = log_path.read_text(encoding="utf-8")
    return HeadlessRun(label=label, log_path=log_path, exit_code=completed.returncode, summary=parse_headless_summary(log_text))


def _try_add_fallback_wall_time(row: dict[str, str], log_path: Path) -> None:
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        if "headlessRun" in line and "->" in line:
            row["wall_ms"] = line.split("->", 1)[1].strip().replace(" ms", "")
            print(f"  fallback wall time: {row['wall_ms']}")
            break


def run_headless_command(args: argparse.Namespace, executable: str) -> int:
    scene = str(resolve_input_path(args.scene))
    hdr = resolve_hdr_path(args.hdr)
    if not os.path.isfile(scene):
        print(f"Scene not found: {scene}")
        return 1

    out_dir = resolve_output_dir(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = run_headless_timing(
        executable,
        scene,
        hdr,
        args.frames,
        args.spp,
        out_dir,
        args.extra_args.split() if args.extra_args else None,
    )
    csv_path = out_dir / args.csv_name
    save_headless_csv(rows, str(csv_path))
    print(f"Wrote {csv_path}")
    return 1 if any(int(row.get("exit_code", "0")) != 0 for row in rows) else 0
