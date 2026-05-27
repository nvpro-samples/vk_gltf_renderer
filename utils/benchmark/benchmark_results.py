from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

JSON_PREFIX = "BENCHMARK_JSON "


def iter_benchmark_records(log_text: str) -> Iterable[dict[str, Any]]:
    """Yield versioned machine-readable benchmark records from app logs."""
    for line in log_text.splitlines():
        start = line.find(JSON_PREFIX)
        if start < 0:
            continue
        payload = line[start + len(JSON_PREFIX) :].strip()
        try:
            record = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if record.get("schema") == 1 and isinstance(record.get("type"), str):
            yield record


def _record_value(record: dict[str, Any], key: str) -> str:
    value = record[key]
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _format_number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _parse_int(value: str | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _add_measured_window_from_progress(summary: dict[str, str], records: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Derive post-warmup timing for older logs that only stored total wall time."""
    if "measured_frames" in summary and "total_wall_ms" in summary:
        return summary

    progress = [
        record
        for record in records
        if record.get("type") == "headless_progress"
        and isinstance(record.get("app_frame"), int)
        and isinstance(record.get("elapsed_ms"), (int, float))
    ]
    if len(progress) < 2:
        return summary

    progress.sort(key=lambda record: int(record["app_frame"]))
    start = progress[0]
    end = progress[-1]
    start_frame = int(start["app_frame"])
    end_frame = int(end["app_frame"])
    start_elapsed = float(start["elapsed_ms"])
    end_elapsed = float(end["elapsed_ms"])
    measured_frames = end_frame - start_frame
    total_wall_ms = _parse_float(summary.get("total_wall_ms") or summary.get("wall_ms"), 0.0)
    measured_wall_ms = total_wall_ms - start_elapsed if total_wall_ms > end_elapsed else end_elapsed - start_elapsed
    if measured_frames <= 0 or measured_wall_ms <= 0.0:
        return summary

    result = dict(summary)
    result.setdefault("total_wall_ms", result.get("wall_ms", ""))
    result.setdefault("total_ms_per_frame", result.get("ms_per_frame", ""))
    result["warmup_frames"] = str(start_frame)
    result["measured_frames"] = str(measured_frames)
    result["wall_ms"] = _format_number(measured_wall_ms)
    result["ms_per_frame"] = _format_number(measured_wall_ms / measured_frames)

    pt_samples = max(_parse_int(result.get("ptSamples"), 1), 1)
    total_accum_frames = _parse_int(result.get("effective_spp"), 0) // pt_samples
    if total_accum_frames <= 0:
        frames = _parse_int(result.get("frames"), end_frame)
        max_frames = max(_parse_int(result.get("maxFrames"), frames), 0)
        total_accum_frames = min(frames, max_frames)
    measured_accum_frames = min(max(total_accum_frames - start_frame, 0), measured_frames)
    measured_effective_spp = measured_accum_frames * pt_samples
    result["measured_effective_spp"] = str(measured_effective_spp)

    width = _parse_int(result.get("resolution_w"), 0)
    height = _parse_int(result.get("resolution_h"), 0)
    measured_wall_sec = measured_wall_ms / 1000.0
    if width > 0 and height > 0 and measured_wall_sec > 0.0:
        samples = float(width * height * measured_effective_spp)
        result["throughput_MSps"] = _format_number(samples / measured_wall_sec / 1e6)
    if measured_wall_sec > 0.0:
        result["spp_per_sec"] = _format_number(measured_effective_spp / measured_wall_sec, 2)

    return result


def _parse_legacy_headless_summary(log_text: str) -> dict[str, str] | None:
    for line in log_text.splitlines():
        if "HEADLESS_SUMMARY" not in line:
            continue
        values = dict(re.findall(r"([A-Za-z_]+)=([^\s]+)", line))
        required = ["frames", "maxFrames", "ptSamples", "effective_spp", "wall_ms", "ms_per_frame"]
        if not all(key in values for key in required):
            continue

        result = {key: values[key] for key in required}
        for key in [
            "measured_effective_spp",
            "total_wall_ms",
            "total_ms_per_frame",
            "warmup_frames",
            "measured_frames",
            "throughput_MSps",
            "spp_per_sec",
            "samples_per_ms",
        ]:
            if key in values:
                result[key] = values[key]

        if "resolution" in values and "x" in values["resolution"]:
            width, height = values["resolution"].split("x", 1)
            result["resolution_w"] = width
            result["resolution_h"] = height
        if "throughput_GSps" in values and "throughput_MSps" not in result:
            result["throughput_MSps"] = str(float(values["throughput_GSps"]) * 1000.0)
        return result
    return None


def parse_headless_summary(log_text: str) -> dict[str, str] | None:
    records = list(iter_benchmark_records(log_text))
    for record in records:
        if record.get("type") != "headless_summary":
            continue
        fields = [
            "frames",
            "maxFrames",
            "ptSamples",
            "effective_spp",
            "measured_effective_spp",
            "resolution_w",
            "resolution_h",
            "wall_ms",
            "ms_per_frame",
            "total_wall_ms",
            "total_ms_per_frame",
            "warmup_frames",
            "measured_frames",
            "throughput_MSps",
            "spp_per_sec",
        ]
        summary = {key: _record_value(record, key) for key in fields if key in record}
        return _add_measured_window_from_progress(summary, records)

    legacy_summary = _parse_legacy_headless_summary(log_text)
    if not legacy_summary:
        return None
    return _add_measured_window_from_progress(legacy_summary, records)


def _parse_json_memory_records(log_text: str) -> dict[int, dict[str, dict[str, int]]]:
    records: dict[int, dict[str, dict[str, int]]] = {}
    for record in iter_benchmark_records(log_text):
        if record.get("type") != "sequence_memory":
            continue
        benchmark_id = int(record.get("id", 0))
        memory: dict[str, dict[str, int]] = {}
        for sample in record.get("memory", []):
            category = str(sample.get("category", "Unknown"))
            memory[category] = {
                "Host Used": int(sample.get("host_used", 0)),
                "Device Used": int(sample.get("device_used", 0)),
                "Device Allocated": int(sample.get("device_allocated", 0)),
            }
        records[benchmark_id] = memory
    return records


def _parse_legacy_memory_records(log_text: str) -> dict[int, dict[str, dict[str, int]]]:
    benchmark_adv_pattern = re.compile(r"BENCHMARK_ADV (\d+) \{")
    memory_pattern = re.compile(
        r"Memory (\w+); Host used\s+(\d+); Device Used\s+(\d+); Device Allocated\s+(\d+);"
    )
    records: dict[int, dict[str, dict[str, int]]] = {}
    benchmark_adv_sections = re.split(benchmark_adv_pattern, log_text)[1:]
    for i in range(0, len(benchmark_adv_sections), 2):
        benchmark_id = int(benchmark_adv_sections[i])
        benchmark_content = benchmark_adv_sections[i + 1]
        memory_data: dict[str, dict[str, int]] = {}
        for match in memory_pattern.finditer(benchmark_content):
            memory_type = match.group(1).strip()
            memory_data[memory_type] = {
                "Host Used": int(match.group(2)),
                "Device Used": int(match.group(3)),
                "Device Allocated": int(match.group(4)),
            }
        records[benchmark_id] = memory_data
    return records


def parse_benchmark(log_text: str, scene_name: str) -> list[dict[str, Any]]:
    benchmark_pattern = re.compile(r'ParameterSequence\s+(\d+)\s+"([^"]+)"\s*=')
    timer_pattern = re.compile(r'Timer\s+"([^"]+)"\s*;\s*GPU;\s*avg\s+(\d+);.*?CPU;\s*avg\s+(\d+);')
    benchmark_sections = re.split(benchmark_pattern, log_text)[1:]
    benchmark_data: dict[int, dict[str, Any]] = {}

    for i in range(0, len(benchmark_sections), 3):
        benchmark_id = int(benchmark_sections[i])
        benchmark_name = benchmark_sections[i + 1].strip()
        benchmark_content = benchmark_sections[i + 2]

        timers: dict[str, dict[str, float]] = {}
        for match in timer_pattern.finditer(benchmark_content):
            stage = match.group(1).strip()
            vk_time = float(match.group(2)) / 1000.0
            cpu_time = float(match.group(3)) / 1000.0
            timers[stage] = {"VK": vk_time, "CPU": cpu_time}

        benchmark_data[benchmark_id] = {
            "scene": scene_name,
            "id": benchmark_id,
            "name": benchmark_name,
            "timers": timers,
            "memory": {},
        }

    memory_records = _parse_json_memory_records(log_text) or _parse_legacy_memory_records(log_text)
    for benchmark_id, memory_data in memory_records.items():
        if benchmark_id in benchmark_data:
            benchmark_data[benchmark_id]["memory"] = memory_data

    return list(benchmark_data.values())


def primary_timer_ms(benchmark: dict[str, Any], device: str, preferred_stages: list[str] | None = None) -> float | None:
    timers = benchmark.get("timers", {})
    if not timers:
        return None
    for stage in preferred_stages or []:
        if stage in timers:
            return timers[stage].get(device)
    return sum(t.get(device, 0.0) for t in timers.values())


def save_to_csv(benchmarks: list[dict[str, Any]], filename: str) -> None:
    stages = sorted({stage for benchmark in benchmarks for stage in benchmark["timers"]})
    memory_types = sorted({mtype for benchmark in benchmarks for mtype in benchmark.get("memory", {})})
    fieldnames = ["Scene", "Benchmark ID", "Benchmark Name", "Primary GPU ms", "Primary CPU ms", "Scene VRAM peak MB"]
    fieldnames += [f"{stage} VK ms" for stage in stages] + [f"{stage} CPU ms" for stage in stages]
    for mtype in memory_types:
        fieldnames += [f"{mtype} Device Used", f"{mtype} Device Allocated"]

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for benchmark in benchmarks:
            preferred = ["GltfRenderer::onRender", "PathTracer::onRender", "Rasterizer::onRender"]
            gpu_ms = primary_timer_ms(benchmark, "VK", preferred)
            cpu_ms = primary_timer_ms(benchmark, "CPU", preferred)
            scene_peak = benchmark.get("memory", {}).get("Scene", {}).get("Device Allocated")
            scene_peak_mb = f"{scene_peak / (1024 * 1024):.2f}" if scene_peak is not None else "N/A"

            row: dict[str, Any] = {
                "Scene": benchmark["scene"],
                "Benchmark ID": benchmark["id"],
                "Benchmark Name": benchmark["name"],
                "Primary GPU ms": f"{gpu_ms:.4f}" if gpu_ms is not None else "N/A",
                "Primary CPU ms": f"{cpu_ms:.4f}" if cpu_ms is not None else "N/A",
                "Scene VRAM peak MB": scene_peak_mb,
            }
            for stage in stages:
                row[f"{stage} VK ms"] = benchmark["timers"].get(stage, {}).get("VK", "N/A")
                row[f"{stage} CPU ms"] = benchmark["timers"].get(stage, {}).get("CPU", "N/A")
            for mtype in memory_types:
                row[f"{mtype} Device Used"] = benchmark["memory"].get(mtype, {}).get("Device Used", "N/A")
                row[f"{mtype} Device Allocated"] = benchmark["memory"].get(mtype, {}).get("Device Allocated", "N/A")
            writer.writerow(row)


def save_headless_csv(rows: list[dict[str, str]], path: str) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def compare_csv(
    baseline_csv: str,
    candidate_csv: str,
    output_csv: str,
    regression_threshold_pct: float,
    vram_regression_threshold_mb: float = 64.0,
) -> int:
    baseline = {(row["Scene"], row["Benchmark Name"]): row for row in load_csv_rows(baseline_csv)}
    candidate = {(row["Scene"], row["Benchmark Name"]): row for row in load_csv_rows(candidate_csv)}
    keys = sorted(set(baseline) | set(candidate))
    fieldnames = [
        "Scene",
        "Benchmark Name",
        "Baseline GPU ms",
        "Candidate GPU ms",
        "GPU delta %",
        "Baseline VRAM peak MB",
        "Candidate VRAM peak MB",
        "VRAM delta MB",
        "Regression",
    ]
    regressions = 0
    gpu_regressions = 0
    vram_regressions = 0

    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for key in keys:
            baseline_row = baseline.get(key)
            candidate_row = candidate.get(key)
            baseline_gpu = float(baseline_row["Primary GPU ms"]) if baseline_row and baseline_row["Primary GPU ms"] != "N/A" else None
            candidate_gpu = (
                float(candidate_row["Primary GPU ms"]) if candidate_row and candidate_row["Primary GPU ms"] != "N/A" else None
            )
            baseline_vram = (
                float(baseline_row["Scene VRAM peak MB"])
                if baseline_row and baseline_row["Scene VRAM peak MB"] != "N/A"
                else None
            )
            candidate_vram = (
                float(candidate_row["Scene VRAM peak MB"])
                if candidate_row and candidate_row["Scene VRAM peak MB"] != "N/A"
                else None
            )

            gpu_delta = ""
            row_regressed = False
            if baseline_gpu is not None and candidate_gpu is not None and baseline_gpu > 0:
                delta_pct = ((candidate_gpu - baseline_gpu) / baseline_gpu) * 100.0
                gpu_delta = f"{delta_pct:+.2f}"
                if delta_pct > regression_threshold_pct:
                    row_regressed = True
                    gpu_regressions += 1

            vram_delta = ""
            if baseline_vram is not None and candidate_vram is not None:
                vram_delta_val = candidate_vram - baseline_vram
                vram_delta = f"{vram_delta_val:+.2f}"
                if vram_delta_val > vram_regression_threshold_mb:
                    row_regressed = True
                    vram_regressions += 1

            regression = "yes" if row_regressed else "no"
            if row_regressed:
                regressions += 1

            writer.writerow(
                {
                    "Scene": key[0],
                    "Benchmark Name": key[1],
                    "Baseline GPU ms": baseline_row["Primary GPU ms"] if baseline_row else "N/A",
                    "Candidate GPU ms": candidate_row["Primary GPU ms"] if candidate_row else "N/A",
                    "GPU delta %": gpu_delta or "N/A",
                    "Baseline VRAM peak MB": baseline_row["Scene VRAM peak MB"] if baseline_row else "N/A",
                    "Candidate VRAM peak MB": candidate_row["Scene VRAM peak MB"] if candidate_row else "N/A",
                    "VRAM delta MB": vram_delta or "N/A",
                    "Regression": regression,
                }
            )

    print(
        f"Comparison written to {output_csv} "
        f"({regressions} regressed rows: {gpu_regressions} GPU > {regression_threshold_pct}%, "
        f"{vram_regressions} VRAM > {vram_regression_threshold_mb} MB)"
    )
    return 1 if regressions > 0 else 0


def compare_headless_logs(baseline_log: str, candidate_log: str) -> int:
    baseline = parse_headless_summary(Path(baseline_log).read_text(encoding="utf-8"))
    candidate = parse_headless_summary(Path(candidate_log).read_text(encoding="utf-8"))
    if not baseline or not candidate:
        print("Could not parse HEADLESS_SUMMARY from one or both logs.")
        return 1

    baseline_wall = float(baseline["wall_ms"])
    candidate_wall = float(candidate["wall_ms"])
    delta_pct = ((candidate_wall - baseline_wall) / baseline_wall * 100.0) if baseline_wall > 0 else 0.0
    direction = "same" if abs(delta_pct) < 0.005 else ("slower" if delta_pct > 0 else "faster")
    baseline_total = float(baseline.get("total_wall_ms", baseline["wall_ms"]))
    candidate_total = float(candidate.get("total_wall_ms", candidate["wall_ms"]))
    total_delta = ((candidate_total - baseline_total) / baseline_total * 100.0) if baseline_total > 0 else 0.0

    baseline_frames = baseline.get("measured_frames", baseline["frames"])
    candidate_frames = candidate.get("measured_frames", candidate["frames"])
    baseline_warmup = baseline.get("warmup_frames", "0")
    candidate_warmup = candidate.get("warmup_frames", "0")
    print(
        f"Baseline: {baseline['ptSamples']} spp/frame  measured_wall_ms={baseline_wall:.3f}  "
        f"ms/frame={baseline['ms_per_frame']}  measured_frames={baseline_frames}  warmup_frames={baseline_warmup}"
    )
    print(
        f"Candidate: {candidate['ptSamples']} spp/frame  measured_wall_ms={candidate_wall:.3f}  "
        f"ms/frame={candidate['ms_per_frame']}  measured_frames={candidate_frames}  warmup_frames={candidate_warmup}"
    )
    print(f"Measured delta: {delta_pct:+.2f}% wall time ({direction})")
    if "total_wall_ms" in baseline or "total_wall_ms" in candidate:
        print(
            f"Total wall: baseline {baseline_total:.3f} ms  candidate {candidate_total:.3f} ms  "
            f"delta={total_delta:+.2f}%"
        )

    if "throughput_MSps" in baseline and "throughput_MSps" in candidate:
        baseline_msps = float(baseline["throughput_MSps"])
        candidate_msps = float(candidate["throughput_MSps"])
        msps_delta = ((candidate_msps - baseline_msps) / baseline_msps * 100.0) if baseline_msps > 0 else 0.0
        print(f"Measured throughput: baseline {baseline_msps:.3f} MSps  candidate {candidate_msps:.3f} MSps  delta={msps_delta:+.2f}%")
    if "spp_per_sec" in baseline and "spp_per_sec" in candidate:
        baseline_sps = float(baseline["spp_per_sec"])
        candidate_sps = float(candidate["spp_per_sec"])
        sps_delta = ((candidate_sps - baseline_sps) / baseline_sps * 100.0) if baseline_sps > 0 else 0.0
        print(f"Measured accum rate: baseline {baseline_sps:.2f} spp/s  candidate {candidate_sps:.2f} spp/s  delta={sps_delta:+.2f}%")
    return 0
