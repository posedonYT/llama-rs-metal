#!/usr/bin/env python3
"""Compare tokenizer IDs and final logits against a pinned, optional llama.cpp oracle."""
import argparse
import hashlib
import json
import math
import pathlib
import struct
import subprocess
import tempfile


def floats(path):
    data = path.read_bytes()
    return struct.unpack(f"<{len(data)//4}f", data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=pathlib.Path)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--tokens", type=pathlib.Path, help="use explicit token IDs instead of tokenizing")
    p.add_argument("--atol", type=float, default=0.001)
    p.add_argument("--output", type=pathlib.Path)
    args = p.parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent
    subprocess.run(["cargo", "build", "--release", "-p", "llama-rs", "--example", "dump_logits"], cwd=root, check=True)
    prompt = "@" + str(args.tokens.resolve()) if args.tokens else args.prompt
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for engine, binary in [("reference", root / "target/reference"), ("rust", root / "target/release/examples/dump_logits")]:
            with (tmp / f"{engine}.log").open("w") as log:
                result = subprocess.run([str(binary), str(args.model.resolve()), prompt, str(tmp / f"{engine}.bin"), str(tmp / f"{engine}.tokens")], stdout=log, stderr=log)
            if result.returncode:
                raise RuntimeError((tmp / f"{engine}.log").read_text())
        ref, rust = floats(tmp / "reference.bin"), floats(tmp / "rust.bin")
        tokens_match = (tmp / "reference.tokens").read_text() == (tmp / "rust.tokens").read_text()
        finite = all(math.isfinite(x) for x in ref + rust)
        if len(ref) != len(rust) or not finite:
            raise RuntimeError("logit count mismatch or nonfinite logits")
        diffs = [abs(a - b) for a, b in zip(ref, rust)]
        report = {"reference_revision": "72ba0238e1a65defbffaa384caadd84121c75f1d", "model": str(args.model), "model_sha256": hashlib.file_digest(args.model.open("rb"), "sha256").hexdigest(), "token_ids_equal": tokens_match, "max_abs": max(diffs), "rmse": math.sqrt(sum(d*d for d in diffs)/len(diffs)), "top1_equal": max(range(len(ref)), key=ref.__getitem__) == max(range(len(rust)), key=rust.__getitem__), "atol": args.atol}
        report["passed"] = tokens_match and report["max_abs"] <= args.atol and report["top1_equal"]
        text = json.dumps(report, indent=2)
        print(text)
        if args.output:
            args.output.write_text(text + "\n")
        if not report["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
