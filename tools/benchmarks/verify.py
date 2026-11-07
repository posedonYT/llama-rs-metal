#!/usr/bin/env python3
"""Correctness gate before timing: same checkpoint, F32 policy, IDs and state evolution."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from protocol import ROOT, Worker, command, worker_identity

TEXTS = [
    "Hello, world!",
    "Привет, мир!",
    "你好世界",
    "café cafe\u0301",
    "  a   b\n\nend\t",
    "1234567890 3.14159",
    "It's we're I'LL I've",
    "🐢🚀🙂",
    "a\r\nb\n",
    "<|im_start|>user\nhello<|im_end|>\n",
]
CORPORA = {
    "prose": "A compiler turns source code into instructions. Its parser builds syntax trees, and its optimizer preserves program behavior while reducing unnecessary work. ",
    "code": "def stable_softmax(values):\n    maximum = max(values)\n    weights = [math.exp(x - maximum) for x in values]\n    return [x / sum(weights) for x in weights]\n\n",
    "russian": "На компьютерах с общей памятью процессор и графический ускоритель могут обращаться к одним данным. Скорость зависит от размера модели, контекста и организации вычислений. ",
}


def file_hash(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def compare(reference, actual, atol, rtol):
    a = np.fromfile(reference["path"], dtype="<f4")
    b = np.fromfile(actual["path"], dtype="<f4")
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid trace shape or nonfinite logits")
    difference = a.astype(np.float64) - b.astype(np.float64)
    error = np.abs(difference)
    a_top = np.argsort(a)[-5:][::-1]
    b_top = np.argsort(b)[-5:][::-1]
    return dict(
        max_abs=float(error.max()),
        rmse=float(np.sqrt(np.mean(difference * difference))),
        normalized_rmse=float(
            np.sqrt(np.mean(difference * difference))
            / max(np.sqrt(np.mean(a.astype(np.float64) ** 2)), 1e-12)
        ),
        within_tolerance=bool(np.all(error <= atol + rtol * np.abs(a))),
        top1_equal=reference["top1"] == actual["top1"],
        top5_reference=a_top.tolist(),
        top5_actual=b_top.tolist(),
        top1_margin=float(a[a_top[0]] - a[a_top[1]]),
    )


def run_model(
    model, out, lengths, atol, rtol, context, decode_count, partitions, styles
):
    path = ROOT / model["path"]
    files = [
        p
        for p in path.iterdir()
        if p.suffix in [".safetensors", ".json", ".jinja"] and p.is_file()
    ]
    identity = {
        p.name: dict(bytes=p.stat().st_size, sha256=file_hash(p)) for p in files
    }
    cases = []
    reference = {}
    token_checks = []
    # Workers run sequentially: two copies of weights never compete for unified memory.
    with Worker(command("mlx", path, True, context), out / "mlx.log") as worker:
        mlx_ready = worker.ready
        tokens = {}
        for text in TEXTS:
            token_checks.append(
                dict(
                    text=text,
                    ids=worker.request(
                        dict(action="tokenize", text=text, special=True)
                    )["ids"],
                )
            )
        for style in styles:
            text = CORPORA[style]
            tokens[style] = worker.request(
                dict(action="tokenize", text=text * 128, special=True)
            )["ids"]
        for length in lengths:
            for style in styles:
                ids = tokens[style]
                prompt, decode = ids[:length], ids[length : length + decode_count]
                if len(prompt) != length or len(decode) != decode_count:
                    raise ValueError("corpus is shorter than the workload")
                case = dict(
                    id=f"{style}-p{length}-d{decode_count}",
                    style=style,
                    prompt_ids=prompt,
                    decode_ids=decode,
                )
                cases.append(case)
                request = dict(
                    action="verify",
                    prompt_ids=prompt,
                    decode_ids=decode,
                    state=style == "prose" and length == max(lengths),
                )
                reference[case["id"]] = worker.request(
                    {
                        **request,
                        "output_dir": str(
                            (out / "traces" / case["id"] / "mlx").resolve()
                        ),
                    }
                )
    comparisons = []
    with Worker(command("rust", path, True, context), out / "rust.log") as worker:
        rust_ready = worker.ready
        for check in token_checks:
            actual = worker.request(
                dict(action="tokenize", text=check["text"], special=True)
            )["ids"]
            check["equal"] = actual == check["ids"]
            if not check["equal"]:
                check["rust_ids"] = actual
        for case in cases:
            base = dict(
                action="verify",
                prompt_ids=case["prompt_ids"],
                decode_ids=case["decode_ids"],
                state=case["style"] == "prose"
                and len(case["prompt_ids"]) == max(lengths),
            )
            for partition in partitions:
                name = "batch" if partition is None else f"chunk-{partition}"
                request = {
                    **base,
                    "output_dir": str((out / "traces" / case["id"] / name).resolve()),
                }
                if partition is not None:
                    request["chunk_size"] = partition
                actual = worker.request(request)
                diffs = [
                    compare(a, b, atol, rtol)
                    for a, b in zip(reference[case["id"]]["traces"], actual["traces"])
                ]
                states = []
                expected_states = reference[case["id"]].get("states", [])
                actual_states = actual.get("states", [])
                if len(expected_states) != len(actual_states):
                    raise ValueError("state tensor count mismatch")
                for a, b in zip(expected_states, actual_states):
                    if (a["layer"], a["kind"], a["shape"]) != (
                        b["layer"],
                        b["kind"],
                        b["shape"],
                    ):
                        raise ValueError("state layout mismatch")
                    av = np.fromfile(a["path"], dtype="<f4")
                    bv = np.fromfile(b["path"], dtype="<f4")
                    error = np.abs(av.astype(np.float64) - bv.astype(np.float64))
                    states.append(
                        dict(
                            layer=a["layer"],
                            kind=a["kind"],
                            shape=a["shape"],
                            max_abs=float(error.max()),
                            within_tolerance=bool(
                                np.isfinite(bv).all()
                                and np.all(error <= atol + rtol * np.abs(av))
                            ),
                        )
                    )
                comparisons.append(
                    dict(
                        states=states,
                        case=case["id"],
                        partition=name,
                        traces=diffs,
                        passed=len(diffs) == decode_count + 1
                        and all(
                            d["within_tolerance"] and d["top1_equal"] for d in diffs
                        )
                        and all(s["within_tolerance"] for s in states),
                    )
                )
    return dict(
        schema_version=2,
        workers={e: worker_identity(e) for e in ["rust", "mlx"]},
        model=model,
        files=identity,
        precision="matched_f32",
        integer_codes="unchanged",
        float_parameters_and_affine_metadata="cast_f32",
        atol=atol,
        rtol=rtol,
        reference=mlx_ready,
        rust=rust_ready,
        tokenizer=token_checks,
        workloads=cases,
        comparisons=comparisons,
        passed=all(c["equal"] for c in token_checks)
        and all(c["passed"] for c in comparisons),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest", type=Path, default=ROOT / "tools/benchmarks/models.json"
    )
    p.add_argument("--models", help="comma-separated manifest IDs")
    p.add_argument(
        "--output", type=Path, default=ROOT / "bench-results/mlx-v2/verification"
    )
    p.add_argument("--lengths", default="32,129")
    p.add_argument("--decode", type=int, default=4)
    p.add_argument("--partitions", default="batch,1,17")
    p.add_argument("--styles", default="prose,code,russian")
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--atol", type=float, default=0.01)
    p.add_argument("--rtol", type=float, default=0.001)
    p.add_argument("--skip-build", action="store_true")
    args = p.parse_args()
    if not args.skip_build:
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "llama-rs",
                "--example",
                "benchmark_worker",
            ],
            cwd=ROOT,
            check=True,
        )
    selected = set(args.models.split(",")) if args.models else None
    models = [
        m
        for m in json.loads(args.manifest.read_text())["models"]
        if selected is None or m["id"] in selected
    ]
    if selected and {m["id"] for m in models} != selected:
        raise ValueError("unknown manifest model ID")
    reports = []
    for model in models:
        out = args.output / model["id"]
        out.mkdir(parents=True, exist_ok=True)
        report = run_model(
            model,
            out,
            [int(x) for x in args.lengths.split(",")],
            args.atol,
            args.rtol,
            args.context,
            args.decode,
            [None if x == "batch" else int(x) for x in args.partitions.split(",")],
            args.styles.split(","),
        )
        (out / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        worst = max(d["max_abs"] for c in report["comparisons"] for d in c["traces"])
        print(
            json.dumps(dict(model=model["id"], passed=report["passed"], max_abs=worst)),
            flush=True,
        )
        reports.append(
            dict(
                model=model["id"],
                passed=report["passed"],
                report=str(out / "report.json"),
            )
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.json").write_text(json.dumps(reports, indent=2) + "\n")
    if not all(r["passed"] for r in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
