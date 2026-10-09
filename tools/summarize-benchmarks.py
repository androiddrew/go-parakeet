"""Create a compact, reviewable release benchmark record from raw local reports."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reports", type=Path, default=Path("artifacts/release"))
parser.add_argument("--output", type=Path, default=Path("validation/benchmarks.json"))
parser.add_argument("--gpu-name", required=True, help="GPU identified for the measured run")
args = parser.parse_args()
cpu = json.loads(subprocess.check_output(["lscpu", "--json"], text=True))
hardware = {x["field"]: x["data"] for x in cpu["lscpu"]}
first_report = json.loads((args.reports / "bench-cpu-fp32.json").read_text())
first_run = first_report["runs"][0]
result = {"cpu": hardware.get("Model name:"), "logical_cpus": hardware.get("CPU(s):"),
          "gpu": args.gpu_name, "runtime": first_report["runtime"]["runtime_version"], "go": first_report["go_version"],
          "window_seconds": first_run["summary"]["window_samples"]/16000,
          "overlap_seconds": first_run["summary"]["overlap_samples"]/16000,
          "method": f"sequential processes; no race detector; first request, {first_report['warmup']} warmups, {first_report['repeat']} warm measured requests per duration; model loading excluded from request timings",
          "memory_method": "Linux RSS/HWM includes native allocations and mapped pages; Go heap reported separately; GPU per-process memory sampled every 100 ms",
          "results": []}
for name in ("bench-cpu-fp32", "bench-cpu-int8", "bench-cuda-fp32"):
    report = json.loads((args.reports / f"{name}.json").read_text())
    if report.get("gpu_sampling_errors"):
        raise RuntimeError("incomplete GPU measurements")
    if report["runtime"]["runtime_version"] != result["runtime"] or report["seed_sha256"] != first_report["seed_sha256"]:
        raise RuntimeError("reports use different runtimes or inputs")
    row = {"provider": report["provider"], "precision": report["precision"], "threads": report["threads"],
           "seed_sha256": report["seed_sha256"], "runtime_version": report["runtime"]["runtime_version"],
           "model_load_seconds": report["runtime"]["model_seconds"],
           "gpu_sampled_peak_mib": report["gpu_sampled_peak_mib"], "durations": []}
    for seconds in sorted({r["summary"]["audio_seconds"] for r in report["runs"]}):
        runs = [r for r in report["runs"] if r["summary"]["audio_seconds"] == seconds]
        first = next(r for r in runs if r["phase"] == "first")
        warm = [r for r in runs if r["phase"] == "warm"]
        entry = {"audio_seconds": seconds, "first_seconds": first["wall_seconds"],
                 "first_chunk_seconds": first["first_chunk_seconds"], "warm_count": len(warm),
                 "warm_median_seconds": statistics.median(r["wall_seconds"] for r in warm),
                 "warm_min_seconds": min(r["wall_seconds"] for r in warm),
                 "warm_max_seconds": max(r["wall_seconds"] for r in warm),
                 "warm_median_first_chunk_seconds": statistics.median(r["first_chunk_seconds"] for r in warm),
                 "peak_process_rss_mib": max(r["memory_after_request"]["peak_rss_bytes"] for r in runs)/1048576,
                 "final_process_rss_mib": warm[-1]["memory_after_request"]["rss_bytes"]/1048576,
                 "max_observed_go_heap_mib": max(r["memory_after_request"]["go_heap_bytes"] for r in runs)/1048576}
        row["durations"].append(entry)
        print(f"{name} {seconds:g}s: median {entry['warm_median_seconds']:.3f}s, first chunk {entry['warm_median_first_chunk_seconds']:.3f}s, RSS peak {entry['peak_process_rss_mib']:.1f} MiB")
    result["results"].append(row)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
