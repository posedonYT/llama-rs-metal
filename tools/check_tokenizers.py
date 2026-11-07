#!/usr/bin/env python3
"""Optional real-vocabulary Unicode/special-token comparison against the reference."""
import argparse
import json
import os
import pathlib
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument("models", nargs="+", type=pathlib.Path)
p.add_argument("--output", type=pathlib.Path)
args = p.parse_args()
root = pathlib.Path(__file__).resolve().parent.parent
subprocess.run(["cargo", "build", "--release", "-p", "llama-rs", "--example", "tokenize"], cwd=root, check=True)
cases = ["Hello, world!", "Привет, мир!", "你好世界", "café cafe\u0301", "  two   spaces\n\nend\t", "1234567890 3.14159", "It's we're I'LL I've", "🐢🚀🙂", "a\r\nb\n", "<|im_start|>user\nhello<|im_end|>\n"]
reports = []
with tempfile.TemporaryDirectory() as directory:
    directory = pathlib.Path(directory)
    for model in args.models:
        for text in cases:
            with (directory/"reference.log").open("w") as log:
                subprocess.run([str(root/"target/reference"), str(model.resolve()), text, str(directory/"unused.bin"), str(directory/"ref.tokens")], env={**os.environ, "TOKENIZER_ONLY":"1"}, stdout=log, stderr=log, check=True)
            rust = subprocess.check_output([str(root/"target/release/examples/tokenize"), str(model.resolve()), text], text=True)
            expected = (directory/"ref.tokens").read_text()
            if rust != expected:
                raise RuntimeError(f"tokenizer mismatch for {model}: {text!r}\nreference={expected}\nrust={rust}")
        reports.append({"model":str(model),"cases":len(cases),"passed":True})
text = json.dumps(reports, ensure_ascii=False, indent=2)
print(text)
if args.output:
    args.output.write_text(text+"\n")
