#!/usr/bin/env python3
"""Balanced isolated benchmark blocks. Raw observations are never discarded."""

import argparse
import datetime
import json
import os
import platform
import re
import subprocess
from pathlib import Path

import psutil

from protocol import ROOT, Worker, command, worker_identity
from verify import CORPORA, file_hash


def host():
    result = json.loads(
        subprocess.check_output([ROOT / "target/host_probe"], text=True)
    )
    swap = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
    used = re.search(r"used = ([\d.]+)([MG])", swap)
    result.update(
        swap=swap,
        swap_used_bytes=float(used[1]) * (1024**2 if used[2] == "M" else 1024**3),
        load_average=os.getloadavg(),
        available_ram_bytes=psutil.virtual_memory().available,
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest", type=Path, default=ROOT / "tools/benchmarks/models.json"
    )
    p.add_argument("--models", default="qwen35-08-4bit")
    p.add_argument("--tracks", default="matched_f32,native")
    p.add_argument("--lengths", default="128,512,2048")
    p.add_argument("--styles", default="prose,code,russian")
    p.add_argument("--decode", type=int, default=64)
    p.add_argument("--context", type=int, default=4096)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repetitions", type=int, default=2)
    p.add_argument("--output", type=Path, default=ROOT / "bench-results/mlx-v2/timing")
    p.add_argument(
        "--verification", type=Path, default=ROOT / "bench-results/mlx-v2/verification"
    )
    p.add_argument(
        "--gguf-verification",
        type=Path,
        default=ROOT / "bench-results/mlx-v2/gguf-verification/report.json",
    )
    p.add_argument(
        "--gguf", type=Path, help="run separate GGUF track instead of MLX manifest"
    )
    args = p.parse_args()
    if (
        os.environ.get("LLAMA_METAL_PROFILE")
        or os.environ.get("MTL_DEBUG_LAYER")
        or os.environ.get("MTL_SHADER_VALIDATION")
    ):
        raise ValueError("unset profiling and Metal validation for timing")
    if args.blocks < 2 or args.warmups < 1 or args.repetitions < 1 or args.decode < 1:
        raise ValueError(
            "need at least two balanced blocks and positive repetitions/decode"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    models = [
        m
        for m in json.loads(args.manifest.read_text())["models"]
        if m["id"] in args.models.split(",")
    ]
    if args.gguf:
        gate = json.loads(args.gguf_verification.read_text())
        if (
            not gate["passed"]
            or gate["sha256"] != file_hash(args.gguf)
            or gate.get("workers")
            != {e: worker_identity(e) for e in ["rust", "llama.cpp"]}
        ):
            raise ValueError("same-GGUF correctness gate is missing, failed or changed")
        models = [
            dict(
                id=args.gguf.stem,
                path=str(args.gguf),
                files={
                    args.gguf.name: dict(
                        sha256=file_hash(args.gguf), bytes=args.gguf.stat().st_size
                    )
                },
            )
        ]
    tools = [
        *Path(__file__).parent.glob("*.py"),
        ROOT / "docs/benchmark-protocol.md",
        ROOT / "target/release/examples/benchmark_worker",
        ROOT / "target/llama_benchmark_worker",
    ]
    metadata = dict(
        schema_version=2,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        host=host(),
        platform=platform.platform(),
        compiler={
            c: subprocess.check_output(cmd, text=True).strip()
            for c, cmd in {
                "rust": ["rustc", "--version"],
                "cpp": ["c++", "--version"],
            }.items()
        },
        tools={
            str(path.relative_to(ROOT)): file_hash(path)
            for path in tools
            if path.is_file()
        },
        llama_cpp_revision=(ROOT / "target/llama-cpp-revision.txt").read_text().strip(),
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output / "blocks.jsonl").open("w") as raw:
        for model in models:
            identity = model
            if not args.gguf:
                gate = json.loads(
                    (args.verification / model["id"] / "report.json").read_text()
                )
                if not gate["passed"]:
                    raise ValueError(f"correctness gate failed: {model['id']}")
                if gate.get("workers") != {
                    e: worker_identity(e) for e in ["rust", "mlx"]
                }:
                    raise ValueError(
                        "worker changed since verification; rebuild and rerun the numerical gate"
                    )
                # Ensure a passed report cannot authorize timing different weights.
                for name, entry in gate["files"].items():
                    if file_hash(ROOT / model["path"] / name) != entry["sha256"]:
                        raise ValueError(f"checkpoint changed after gate: {name}")
                identity = {**model, "files": gate["files"]}
            path = ROOT / model["path"]
            reference = "llama.cpp" if args.gguf else "mlx"
            workloads = []
            with Worker(
                command(reference, path, not bool(args.gguf), args.context),
                args.output / f"{model['id']}-tokenize.log",
            ) as worker:
                for style in args.styles.split(","):
                    ids = worker.request(
                        dict(action="tokenize", text=CORPORA[style] * 512, special=True)
                    )["ids"]
                    for length in map(int, args.lengths.split(",")):
                        prompt, decode = (
                            ids[:length],
                            ids[length : length + args.decode],
                        )
                        if (
                            len(prompt) != length
                            or len(decode) != args.decode
                            or length + args.decode > args.context
                        ):
                            raise ValueError("invalid corpus length/context")
                        workloads.append(
                            dict(
                                id=f"{style}-p{length}-d{args.decode}",
                                prompt_ids=prompt,
                                decode_ids=decode,
                            )
                        )
            (args.output / f"{model['id']}-workloads.json").write_text(
                json.dumps(dict(model=identity, workloads=workloads), indent=2) + "\n"
            )
            tracks = ["gguf"] if args.gguf else args.tracks.split(",")
            for track in tracks:
                if track not in ["matched_f32", "native", "gguf"]:
                    raise ValueError(track)
                for block in range(args.blocks):
                    order = (
                        ["rust", reference] if block % 2 == 0 else [reference, "rust"]
                    )
                    for engine in order:
                        prefix = f"{model['id']}-{track}-b{block}-{engine}"
                        before = host()
                        with Worker(
                            command(engine, path, track == "matched_f32", args.context),
                            args.output / f"{prefix}.log",
                        ) as worker:
                            cells = []
                            for workload in workloads:
                                result = worker.request(
                                    {
                                        **workload,
                                        "action": "benchmark",
                                        "warmups": args.warmups,
                                        "repetitions": args.repetitions,
                                    }
                                )
                                cells.append(dict(workload=workload["id"], **result))
                            ready = worker.ready
                        after = host()
                        reasons = []
                        if max(before["thermal_state"], after["thermal_state"]) >= 2:
                            reasons.append("serious_or_critical_thermal")
                        if before["low_power_mode"] or after["low_power_mode"]:
                            reasons.append("low_power_mode")
                        if after["swap_used_bytes"] > before["swap_used_bytes"]:
                            reasons.append("swap_growth")
                        record = dict(
                            model=model["id"],
                            track=track,
                            block=block,
                            engine=engine,
                            order=order,
                            command=worker.command,
                            ready=ready,
                            host_before=before,
                            host_after=after,
                            confounded=bool(reasons),
                            confound_reasons=reasons,
                            sampled_peak_rss_bytes=worker.peak_rss,
                            cells=cells,
                        )
                        raw.write(json.dumps(record) + "\n")
                        raw.flush()
                        print(
                            json.dumps(
                                dict(
                                    model=model["id"],
                                    track=track,
                                    block=block,
                                    engine=engine,
                                    confounded=reasons,
                                    cells=len(cells),
                                )
                            ),
                            flush=True,
                        )


if __name__ == "__main__":
    main()
