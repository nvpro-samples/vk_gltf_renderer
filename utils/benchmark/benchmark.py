#!/usr/bin/env python3
"""
Command-line entry point for vk_gltf_renderer benchmarks.

Run from the repository root, e.g.:

  python utils/benchmark/benchmark.py headless --scene resources/shader_ball.gltf
  python utils/benchmark/benchmark.py run quick.cfg --scene resources/shader_ball.gltf
"""

from __future__ import annotations

import argparse
import sys

from benchmark_paths import resolve_executable
from benchmark_results import compare_csv, compare_headless_logs
from benchmark_runner import run_headless_command, run_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vk_gltf_renderer benchmarks (utils/benchmark/).")
    sub = parser.add_subparsers(dest="command", required=True)

    headless_p = sub.add_parser(
        "headless",
        help="Headless timing: N frames at 1 and/or 5 spp (for version A/B comparison)",
    )
    headless_p.add_argument("--scene", type=str, required=True, help="Scene .gltf path")
    headless_p.add_argument("--hdr", type=str, default="std_env.hdr")
    headless_p.add_argument("--frames", type=int, default=500, help="App frames and maxFrames (must match)")
    headless_p.add_argument(
        "--spp",
        type=int,
        nargs="+",
        default=[1, 5],
        help="Path tracer samples per pixel per frame (e.g. 1 5)",
    )
    headless_p.add_argument("--output-dir", type=str, default="output")
    headless_p.add_argument("--csv-name", type=str, default="headless_results.csv")
    headless_p.add_argument("--executable", type=str, default=None)
    headless_p.add_argument("--extra-args", type=str, default="")

    headless_cmp_p = sub.add_parser("headless-compare", help="Compare headless summaries from two log files")
    headless_cmp_p.add_argument("baseline_log", type=str)
    headless_cmp_p.add_argument("candidate_log", type=str)

    run_p = sub.add_parser("run", help="Run scripted sequencer benchmark and write CSV")
    run_p.add_argument(
        "benchmark",
        type=str,
        help="Benchmark .cfg (e.g. quick.cfg or matrix.cfg in utils/benchmark/)",
    )
    run_p.add_argument("--scene", type=str, help="Single scene .gltf path")
    run_p.add_argument("--scenes-file", type=str, help="Text file: 'name relative/path.gltf' per line")
    run_p.add_argument("--scenes-root", type=str, default=".", help="Root directory for scenes-file paths")
    run_p.add_argument(
        "--hdr",
        type=str,
        default="std_env.hdr",
        help="HDR environment file (searched under resources/; falls back to std_env.hdr)",
    )
    run_p.add_argument("--csv-name", type=str, default="benchmark_results.csv")
    run_p.add_argument("--output-dir", type=str, default="output")
    run_p.add_argument("--executable", type=str, default=None)
    run_p.add_argument("--extra-args", type=str, default="", help="Extra CLI args passed to the executable")

    cmp_p = sub.add_parser("compare", help="Compare baseline vs candidate CSV")
    cmp_p.add_argument("baseline_csv", type=str)
    cmp_p.add_argument("candidate_csv", type=str)
    cmp_p.add_argument("--output", type=str, default="benchmark_compare.csv")
    cmp_p.add_argument(
        "--regression-threshold-pct",
        type=float,
        default=5.0,
        help="Flag regression if candidate GPU time is slower by more than this percent",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "compare":
        return compare_csv(
            args.baseline_csv,
            args.candidate_csv,
            args.output,
            args.regression_threshold_pct,
        )

    if args.command == "headless-compare":
        return compare_headless_logs(args.baseline_log, args.candidate_log)

    executable = resolve_executable(args.executable)
    if not executable:
        print("Executable not found. Build vk_gltf_renderer or pass --executable.")
        return 1

    if args.command == "headless":
        return run_headless_command(args, executable)

    if not args.scene and not args.scenes_file:
        print("run requires --scene or --scenes-file")
        return 1

    print(f"Using executable: {executable}")
    return run_matrix(args, executable)


if __name__ == "__main__":
    sys.exit(main())
