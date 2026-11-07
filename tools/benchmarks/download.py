#!/usr/bin/env python3
"""Download explicitly pinned checkpoints from the benchmark manifest."""

import argparse
import json

from huggingface_hub import snapshot_download

from protocol import ROOT

p = argparse.ArgumentParser()
p.add_argument("models", nargs="*")
args = p.parse_args()
for model in json.loads((ROOT / "tools/benchmarks/models.json").read_text())["models"]:
    if args.models and model["id"] not in args.models:
        continue
    snapshot_download(
        repo_id=model["repo"],
        revision=model["revision"],
        local_dir=ROOT / model["path"],
        allow_patterns=["*.safetensors", "*.json", "*.jinja", "README.md"],
    )
