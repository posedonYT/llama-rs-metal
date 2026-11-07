"""Process isolation and line-delimited JSON transport shared by verification and timing."""

import hashlib
import json
import os
import select
import subprocess
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]


class Worker:
    def __init__(self, command, log_path, timeout=600):
        self.command = [str(x) for x in command]
        self.timeout = timeout
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("w")
        self.started = time.perf_counter()
        self.process = subprocess.Popen(
            self.command,
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
        )
        self.peak_rss = 0
        self.stop_sampling = threading.Event()

        def observe():
            process = psutil.Process(self.process.pid)
            while not self.stop_sampling.wait(0.1):
                try:
                    self.peak_rss = max(self.peak_rss, process.memory_info().rss)
                except psutil.NoSuchProcess:
                    return

        self.observer = threading.Thread(target=observe, daemon=True)
        self.observer.start()
        self.journal = self.log_path.with_suffix(".requests.jsonl").open("w")
        try:
            self.ready = self.read()
        except BaseException:
            self.close()
            raise
        if self.ready.get("type") != "ready":
            self.close()
            raise RuntimeError(f"invalid worker handshake: {self.ready}")
        self.ready["process_start_to_ready_seconds"] = (
            time.perf_counter() - self.started
        )

    def read(self):
        if not select.select([self.process.stdout], [], [], self.timeout)[0]:
            raise TimeoutError(f"worker timed out: {self.command}; see {self.log_path}")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                f"worker exited {self.process.poll()}: {self.log_path.read_text()[-8000:]}"
            )
        return json.loads(line)

    def request(self, value):
        self.journal.write(
            json.dumps(
                dict(request=value, monotonic_seconds=time.perf_counter()),
                ensure_ascii=False,
            )
            + "\n"
        )
        self.journal.flush()
        self.process.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = self.read()
        if not response.get("ok"):
            raise RuntimeError(f"worker error: {response}; see {self.log_path}")
        return response["result"]

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.stop_sampling.set()
        self.observer.join(timeout=1)
        self.journal.close()
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def command(engine, model, float32, context, python=None):
    if engine == "rust":
        result = [ROOT / "target/release/examples/benchmark_worker", model]
    elif engine == "mlx":
        result = [
            python or ROOT / "target/mlx-env/bin/python",
            ROOT / "tools/benchmarks/mlx_worker.py",
            model,
        ]
    elif engine == "llama.cpp":
        result = [ROOT / "target/llama_benchmark_worker", model]
    else:
        raise ValueError(engine)
    return result + (["--f32"] if float32 else []) + ["--context", str(context)]


def worker_identity(engine):
    """Bind a numerical gate to the actual measured executable/reference code."""
    path = {
        "rust": ROOT / "target/release/examples/benchmark_worker",
        "mlx": ROOT / "tools/benchmarks/mlx_worker.py",
        "llama.cpp": ROOT / "target/llama_benchmark_worker",
    }[engine]
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return dict(artifact=str(path.relative_to(ROOT)), sha256=digest)
