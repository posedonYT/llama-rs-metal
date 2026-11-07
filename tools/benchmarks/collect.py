#!/usr/bin/env python3
"""Publish compact evidence without model weights, absolute home paths or logits."""

import argparse
import gzip
import hashlib
import json
import platform
from pathlib import Path

from protocol import ROOT


def portable(value):
    if isinstance(value, str):
        return value.replace(str(ROOT), ".")
    if isinstance(value, list):
        return [portable(x) for x in value]
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=ROOT / "bench-results/mlx-v2")
    p.add_argument("--output", type=Path, default=ROOT / "docs/benchmarks-2026-11-07")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timings = {}
    for directory in sorted(args.source.glob("timing-*")):
        # Preliminary and pre-optimization runs are diagnostics, not final results.
        if directory.name in [
            "timing-initial-full-forward",
            "timing-before-packed-decode",
        ]:
            continue
        if not (directory / "blocks.jsonl").exists():
            continue
        timings[directory.name] = dict(
            metadata=json.loads((directory / "metadata.json").read_text()),
            workloads={
                p.name: json.loads(p.read_text())
                for p in directory.glob("*-workloads.json")
            },
            blocks=[
                json.loads(line)
                for line in (directory / "blocks.jsonl").read_text().splitlines()
            ],
        )
    gates = {
        str(p.relative_to(args.source)): json.loads(p.read_text())
        for p in sorted(args.source.rglob("report.json"))
        if "memory" not in p.parts
    }
    payload = portable(
        dict(
            schema_version=2,
            analysis={
                "python": platform.python_version(),
                "requirements": (ROOT / "tools/benchmarks/requirements.lock")
                .read_text()
                .splitlines(),
                "summarize_sha256": hashlib.sha256(
                    (ROOT / "tools/benchmarks/summarize.py").read_bytes()
                ).hexdigest(),
                "cli_smoke_binary_sha256": hashlib.sha256(
                    (ROOT / "target/release/llama-cli").read_bytes()
                ).hexdigest(),
            },
            timings=timings,
            verification=gates,
            memory=json.loads((args.source / "memory/report.json").read_text())
            if (args.source / "memory/report.json").exists()
            else None,
            chat_smoke=json.loads((args.source / "chat-smoke.json").read_text())
            if (args.source / "chat-smoke.json").exists()
            else None,
        )
    )
    encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
    (args.output / "raw.json.gz").write_bytes(gzip.compress(encoded, mtime=0))


if __name__ == "__main__":
    main()
