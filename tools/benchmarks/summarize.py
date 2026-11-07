#!/usr/bin/env python3
"""Summaries preserve raw blocks and label precision/thermal limitations."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def distribution(values):
    return dict(
        samples=len(values),
        median=float(np.median(values)) if values else None,
        p95=float(np.quantile(values, 0.95)) if values else None,
    )


def summarize(records):
    groups = defaultdict(dict)
    for record in records:
        for cell in record["cells"]:
            key = (record["model"], record["track"], cell["workload"])
            groups[key][(record["block"], record["engine"])] = (record, cell)
    output = []
    for (model, track, workload), items in groups.items():
        engines = sorted({engine for _, engine in items})
        reference = next(e for e in engines if e != "rust")
        metrics = {}
        for metric in [
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
            "request_ttft_seconds",
            "reset_seconds",
        ]:
            values = {
                engine: [
                    run[metric]
                    for (block, e), (record, cell) in items.items()
                    if e == engine
                    for run in cell["runs"]
                ]
                for engine in engines
            }
            paired = []
            valid_values = {engine: [] for engine in engines}
            confounded = []
            for block in sorted({b for b, _ in items}):
                if (block, "rust") not in items or (block, reference) not in items:
                    continue
                rust, ref = items[block, "rust"], items[block, reference]
                if rust[0]["confounded"] or ref[0]["confounded"]:
                    confounded.append(block)
                    continue
                valid_values["rust"].extend(r[metric] for r in rust[1]["runs"])
                valid_values[reference].extend(r[metric] for r in ref[1]["runs"])
                a = np.median([r[metric] for r in rust[1]["runs"]])
                b = np.median([r[metric] for r in ref[1]["runs"]])
                paired.append(np.log(a / b if "per_second" in metric else b / a))
            rng = np.random.default_rng(42)
            if len(paired) >= 2:
                means = np.mean(
                    rng.choice(paired, size=(10000, len(paired)), replace=True), axis=1
                )
                ratio = float(np.exp(np.mean(paired)))
                interval = np.exp(np.quantile(means, [0.025, 0.975])).tolist()
            else:
                ratio = None
                interval = None
            metrics[metric] = dict(
                engines={e: distribution(v) for e, v in values.items()},
                valid_paired_engines={
                    e: distribution(v) for e, v in valid_values.items()
                },
                rust_speed_ratio=ratio,
                paired_bootstrap95=interval,
                valid_paired_blocks=len(paired),
                confounded_blocks=confounded,
            )
        output.append(
            dict(model=model, track=track, workload=workload, metrics=metrics)
        )
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directories", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    records = [
        json.loads(line)
        for directory in a.directories
        for line in (directory / "blocks.jsonl").read_text().splitlines()
    ]
    report = summarize(records)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                schema_version=2,
                ratio_definition="Rust/reference throughput; >1 favors Rust; confounded blocks excluded from paired ratio",
                results=report,
            ),
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Benchmark results",
        "",
        "Synchronous full-host-logit boundary; teacher-forced decode. Native is not precision matched. Table medians and ratios exclude confounded pairs. Small-N CIs describe this machine/run only.",
        "",
        "| Model | Track | Workload | Prefill Rust / ref tok/s | Decode Rust / ref tok/s | Decode ratio [95% CI] | Valid pairs |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for row in report:
        m = row["metrics"]
        pp = m["prefill_tokens_per_second"]
        tg = m["decode_tokens_per_second"]
        ref = next(e for e in tg["engines"] if e != "rust")
        ratio = tg["rust_speed_ratio"]
        ci = tg["paired_bootstrap95"]
        ratios = (
            "unavailable"
            if ratio is None
            else f"{ratio:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"
        )
        lines.append(
            f"| {row['model']} | {row['track']} | {row['workload']} | {pp['valid_paired_engines']['rust']['median']:.1f} / {pp['valid_paired_engines'][ref]['median']:.1f} | {tg['valid_paired_engines']['rust']['median']:.1f} / {tg['valid_paired_engines'][ref]['median']:.1f} | {ratios} | {tg['valid_paired_blocks']} |"
        )
    lines.extend(
        [
            "",
            "Ratios are geometric means of paired block-median ratios, not ratios of pooled medians. Each primary cell has eight observations per engine before confound exclusion. Raw distributions, all samples, startup, TTFT and memory remain in the JSON evidence.",
        ]
    )
    a.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
