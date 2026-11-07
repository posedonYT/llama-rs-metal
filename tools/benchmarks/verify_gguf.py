#!/usr/bin/env python3
"""Same-GGUF gate against pinned upstream; never equate GGUF and MLX quants."""

import argparse
import json
from pathlib import Path

from protocol import ROOT, Worker, command, worker_identity
from verify import CORPORA, compare, file_hash

p = argparse.ArgumentParser()
p.add_argument("model", type=Path)
p.add_argument(
    "--output", type=Path, default=ROOT / "bench-results/mlx-v2/gguf-verification"
)
p.add_argument("--lengths", default="128,512,2048")
p.add_argument("--context", type=int, default=4096)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
cases = []
reference = {}
with Worker(
    command("llama.cpp", a.model, False, a.context), a.output / "reference.log"
) as w:
    ready = w.ready
    for length in map(int, a.lengths.split(",")):
        ids = w.request(
            dict(action="tokenize", text=CORPORA["prose"] * 512, special=True)
        )["ids"]
        case = dict(
            id=f"prose-p{length}-d4",
            prompt_ids=ids[:length],
            decode_ids=ids[length : length + 4],
        )
        cases.append(case)
        reference[case["id"]] = w.request(
            {
                **case,
                "action": "verify",
                "output_dir": str((a.output / case["id"] / "cpp").resolve()),
            }
        )
comparisons = []
with Worker(command("rust", a.model, False, a.context), a.output / "rust.log") as w:
    for case in cases:
        actual = w.request(
            {
                **case,
                "action": "verify",
                "output_dir": str((a.output / case["id"] / "rust").resolve()),
            }
        )
        diffs = [
            compare(x, y, 0.2, 0.01)
            for x, y in zip(reference[case["id"]]["traces"], actual["traces"])
        ]
        comparisons.append(
            dict(
                case=case["id"],
                traces=diffs,
                passed=len(diffs) == 5
                and all(d["within_tolerance"] and d["top1_equal"] for d in diffs),
            )
        )
report = dict(
    workers={e: worker_identity(e) for e in ["rust", "llama.cpp"]},
    reference=ready,
    model=str(a.model),
    sha256=file_hash(a.model),
    atol=0.2,
    rtol=0.01,
    precision="F32 caches; implementation-default operation precision",
    workloads=cases,
    comparisons=comparisons,
    passed=all(c["passed"] for c in comparisons),
)
(a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        dict(
            passed=report["passed"],
            max_abs=max(d["max_abs"] for c in comparisons for d in c["traces"]),
        )
    )
)
if not report["passed"]:
    raise SystemExit(1)
