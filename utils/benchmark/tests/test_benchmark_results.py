from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_results import compare_csv, parse_benchmark, parse_headless_summary  # noqa: E402


class BenchmarkResultTests(unittest.TestCase):
    def test_parse_headless_summary_prefers_json_record(self) -> None:
        log_text = (
            "HEADLESS_SUMMARY frames=1 maxFrames=1 ptSamples=1 effective_spp=1 "
            "resolution=16x16 wall_ms=999.0 ms_per_frame=999.0\n"
            'BENCHMARK_JSON {"schema":1,"type":"headless_summary","frames":500,'
            '"maxFrames":500,"ptSamples":5,"effective_spp":2500,"resolution_w":1920,'
            '"resolution_h":1080,"wall_ms":123.456,"ms_per_frame":0.247,'
            '"throughput_MSps":42.5,"spp_per_sec":20.25}\n'
        )

        summary = parse_headless_summary(log_text)

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["frames"], "500")
        self.assertEqual(summary["ptSamples"], "5")
        self.assertEqual(summary["wall_ms"], "123.456")
        self.assertEqual(summary["throughput_MSps"], "42.5")

    def test_parse_benchmark_uses_json_memory_record(self) -> None:
        log_text = (
            'ParameterSequence 0 "Path tracer" = {\n'
            '  Timer "GltfRenderer::onRender"; GPU; avg 12000; CPU; avg 3400;\n'
            "}\n"
            'BENCHMARK_JSON {"schema":1,"type":"sequence_memory","id":0,"memory":['
            '{"category":"Scene","host_used":0,"device_used":100,"device_allocated":200},'
            '{"category":"PathTracer","host_used":0,"device_used":30,"device_allocated":40}]}\n'
        )

        rows = parse_benchmark(log_text, "shader_ball")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timers"]["GltfRenderer::onRender"]["VK"], 12.0)
        self.assertEqual(rows[0]["memory"]["Scene"]["Device Allocated"], 200)
        self.assertEqual(rows[0]["memory"]["PathTracer"]["Device Used"], 30)

    def test_parse_headless_summary_legacy_fallback(self) -> None:
        log_text = (
            "HEADLESS_SUMMARY frames=500 maxFrames=500 ptSamples=1 effective_spp=500 "
            "resolution=1920x1080 wall_ms=12345.678 ms_per_frame=24.691 "
            "throughput_MSps=84.0 spp_per_sec=40.49\n"
        )

        summary = parse_headless_summary(log_text)

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["frames"], "500")
        self.assertEqual(summary["resolution_w"], "1920")
        self.assertEqual(summary["spp_per_sec"], "40.49")

    def test_parse_headless_summary_derives_post_warmup_measurement(self) -> None:
        log_text = (
            'BENCHMARK_JSON {"schema":1,"type":"headless_progress","app_frame":1,'
            '"frames":11,"elapsed_ms":100.0,"ms_per_frame":100.0,"percent":9.09}\n'
            'BENCHMARK_JSON {"schema":1,"type":"headless_progress","app_frame":11,'
            '"frames":11,"elapsed_ms":600.0,"ms_per_frame":54.545,"percent":100.0}\n'
            'BENCHMARK_JSON {"schema":1,"type":"headless_summary","frames":11,'
            '"maxFrames":11,"ptSamples":1,"effective_spp":11,"resolution_w":10,'
            '"resolution_h":10,"wall_ms":1000.0,"ms_per_frame":90.909,'
            '"throughput_MSps":0.001,"spp_per_sec":11.0}\n'
        )

        summary = parse_headless_summary(log_text)

        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["total_wall_ms"], "1000")
        self.assertEqual(summary["warmup_frames"], "1")
        self.assertEqual(summary["measured_frames"], "10")
        self.assertEqual(summary["measured_effective_spp"], "10")
        self.assertEqual(summary["wall_ms"], "900")
        self.assertEqual(summary["ms_per_frame"], "90")
        self.assertEqual(summary["throughput_MSps"], "0.001")
        self.assertEqual(summary["spp_per_sec"], "11.11")

    def test_compare_csv_flags_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = root / "baseline.csv"
            candidate = root / "candidate.csv"
            output = root / "diff.csv"
            fieldnames = ["Scene", "Benchmark Name", "Primary GPU ms", "Scene VRAM peak MB"]
            with baseline.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({"Scene": "s", "Benchmark Name": "b", "Primary GPU ms": "10.0", "Scene VRAM peak MB": "1.0"})
            with candidate.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({"Scene": "s", "Benchmark Name": "b", "Primary GPU ms": "11.0", "Scene VRAM peak MB": "2.0"})

            rc = compare_csv(str(baseline), str(candidate), str(output), 5.0)

            self.assertEqual(rc, 1)
            with output.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["Regression"], "yes")
            self.assertEqual(rows[0]["GPU delta %"], "+10.00")


if __name__ == "__main__":
    unittest.main()
