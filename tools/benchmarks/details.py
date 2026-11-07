#!/usr/bin/env python3
"""Human-readable startup, memory and sustained-decode evidence from the archive."""

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from protocol import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--directory", type=Path, default=ROOT / "docs/benchmarks-2026-11-07"
    )
    a = p.parse_args()
    data = json.loads(gzip.decompress((a.directory / "raw.json.gz").read_bytes()))
    lines = [
        "# Additional measurements",
        "",
        "All times are synchronous API times. Startup uses warm filesystem/driver caches; it is not a cold-install or cold-disk claim. TTFT includes preallocated-cache reset. Median values below exclude a pair if either process block was confounded.",
        "",
        "| Model | Track | Load Rust / MLX (s) | Process start to ready Rust / MLX (s) | P512 request TTFT Rust / MLX (ms) |",
        "|---|---|---:|---:|---:|",
    ]
    for name in ["timing-primary", "timing-variants"]:
        groups = defaultdict(list)
        for block in data["timings"][name]["blocks"]:
            groups[block["model"], block["track"]].append(block)
        for (model, track), blocks in groups.items():
            bad = {r["block"] for r in blocks if r["confounded"]}
            values = {
                engine: [
                    r for r in blocks if r["engine"] == engine and r["block"] not in bad
                ]
                for engine in ["rust", "mlx"]
            }
            load = {
                e: np.median([r["ready"]["load_seconds"] for r in rs])
                for e, rs in values.items()
            }
            ready = {
                e: np.median([r["ready"]["process_start_to_ready_seconds"] for r in rs])
                for e, rs in values.items()
            }
            ttft = {
                e: np.median(
                    [
                        run["request_ttft_seconds"]
                        for r in rs
                        for c in r["cells"]
                        if c["workload"] == "prose-p512-d64"
                        for run in c["runs"]
                    ]
                )
                * 1000
                for e, rs in values.items()
            }
            lines.append(
                f"| {model} | {track} | {load['rust']:.3f} / {load['mlx']:.3f} | {ready['rust']:.3f} / {ready['mlx']:.3f} | {ttft['rust']:.1f} / {ttft['mlx']:.1f} |"
            )
    lines += [
        "",
        "## Memory",
        "",
        "One separate inference trial per configuration, after readiness: P512/D64, context 4096. OS charge, RSS and Metal counters are different accounts. Do not add them or infer a whole-system RAM winner from one column. [Method](../memory-method.md).",
        "",
        "| Model | Track | OS charged peak Rust / MLX GiB | RSS peak Rust / MLX GiB | Rust Metal retained span GiB | MLX active / peak / cache GiB |",
        "|---|---|---:|---:|---:|---:|",
    ]
    groups = defaultdict(dict)
    for r in data["memory"]["rows"]:
        groups[r["model"], r["track"]][r["engine"]] = r
    for (model, track), engines in groups.items():
        rust, mlx = engines["rust"], engines["mlx"]

        def pair(key):
            return f"{rust[key] / 2**30:.2f} / {mlx[key] / 2**30:.2f}"

        rm, mm = rust["allocator_metrics"], mlx["allocator_metrics"]
        lines.append(
            f"| {model} | {track} | {pair('peak_os_charged_footprint_bytes')} | {pair('peak_rss_after_ready_bytes')} | {rm['metal_allocated_bytes'] / 2**30:.2f} | {mm['metal_active_bytes'] / 2**30:.2f} / {mm['metal_peak_bytes'] / 2**30:.2f} / {mm['metal_cache_bytes'] / 2**30:.2f} |"
        )
    lines += [
        "",
        "## 512-step continuation",
        "",
        "Qwen3.5 0.8B 4-bit, matched F32, P512 followed by D512. Two balanced blocks, one warmup and two measured requests per process (four measured runs per engine); this is a small endurance experiment. Teacher-forced history and position ranges are identical. Values are median tokens/s within each 64-step window, so increasing context remains visible.",
        "",
        "| Consumed positions | Rust tok/s | MLX tok/s |",
        "|---|---:|---:|",
    ]
    records = data["timings"]["timing-endurance"]["blocks"]
    for window in range(8):
        rates = {e: [] for e in ["rust", "mlx"]}
        for record in records:
            if record["confounded"]:
                continue
            for run in record["cells"][0]["runs"]:
                duration = sum(
                    run["decode_step_seconds"][window * 64 : (window + 1) * 64]
                )
                rates[record["engine"]].append(64 / duration)
        lines.append(
            f"| {513 + window * 64}–{576 + window * 64} | {np.median(rates['rust']):.1f} | {np.median(rates['mlx']):.1f} |"
        )
    counts = defaultdict(set)
    for timing in data["timings"].values():
        for record in timing["blocks"]:
            if record["engine"] == "rust":
                for cell in record["cells"]:
                    counts[
                        record["model"], record["track"], record["ready"]["context"]
                    ].add(cell["metal_allocated_bytes"])
    lines += [
        "",
        f"Rust retained Metal allocation counts remained constant across workload cells in all {len(counts)} model/precision/context configurations (each had one distinct retained-byte value). The real MLX session test also checks the count before/after 128 decode steps. These counters establish bounded retained resources, not zero allocator churn or a bound on all driver memory.",
        "",
        "## Evidence scope",
        "",
        "The primary matrix has four independent balanced blocks and eight measured observations per engine/cell; the endurance subset has two blocks. Only the BF16 matched-F32 block 0 pair was excluded for swap growth. Raw observations are retained, including that pair. No outlier trimming was applied. Other applications and dynamic GPU clocks remain uncontrolled; confidence intervals describe this run/configuration, not hardware-wide performance.",
        "",
        "The upstream llama.cpp table uses a 260K-parameter F32-weight GGUF with F32 caches and implementation-default internal precision. Its maximum observed logit difference was about 0.009 and argmax agreed. It is a small integration/dispatch comparison, not evidence for billion-parameter parity. Quantized Qwen GGUF failed its numerical gate and is excluded.",
        "",
        "[Throughput and confidence intervals](results.md) · [Compact raw JSON (gzip)](raw.json.gz)",
    ]
    (a.directory / "details.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
