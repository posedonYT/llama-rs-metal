#!/usr/bin/env python3
"""Separate memory experiments: OS charged footprint + RSS + GPU allocators.
Memory probes do not run in the throughput experiments. Darwin rusage_info_v2
layout follows the macOS SDK sys/resource.h, with 16-byte UUID and 18 u64 fields.
"""

import argparse
import ctypes
import json
import threading
import time
from pathlib import Path

from protocol import ROOT, Worker, command, worker_identity

FIELDS = [
    "user_time",
    "system_time",
    "pkg_idle_wkups",
    "interrupt_wkups",
    "pageins",
    "wired_size",
    "resident_size",
    "phys_footprint",
    "proc_start_abstime",
    "proc_exit_abstime",
    "child_user_time",
    "child_system_time",
    "child_pkg_idle_wkups",
    "child_interrupt_wkups",
    "child_pageins",
    "child_elapsed_abstime",
    "diskio_bytesread",
    "diskio_byteswritten",
]


class RusageV2(ctypes.Structure):
    _fields_ = [
        ("uuid", ctypes.c_ubyte * 16),
        *[(name, ctypes.c_uint64) for name in FIELDS],
    ]


libproc = ctypes.CDLL("libproc.dylib", use_errno=True)
libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
libproc.proc_pid_rusage.restype = ctypes.c_int


def sample(pid):
    info = RusageV2()
    if libproc.proc_pid_rusage(pid, 2, ctypes.byref(info)) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    return dict(
        monotonic_seconds=time.monotonic(),
        os_charged_footprint_bytes=info.phys_footprint,
        rss_bytes=info.resident_size,
        wired_bytes=info.wired_size,
        pageins=info.pageins,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="qwen35-08-4bit,llama32-1b-4bit,qwen35-4b-4bit")
    p.add_argument("--output", type=Path, default=ROOT / "bench-results/mlx-v2/memory")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "tools/benchmarks/models.json").read_text())["models"]
    rows = []
    for model in manifest:
        if model["id"] not in a.models.split(","):
            continue
        directory = (
            "timing-primary"
            if model["id"] in ["qwen35-08-4bit", "llama32-1b-4bit"]
            else "timing-variants"
        )
        workloads = json.loads(
            (
                ROOT
                / "bench-results/mlx-v2"
                / directory
                / f"{model['id']}-workloads.json"
            ).read_text()
        )["workloads"]
        workload = next(w for w in workloads if w["id"] == "prose-p512-d64")
        gate = json.loads(
            (
                ROOT
                / "bench-results/mlx-v2/long-verification"
                / model["id"]
                / "report.json"
            ).read_text()
        )
        if not gate["passed"] or gate["workers"] != {
            e: worker_identity(e) for e in ["rust", "mlx"]
        }:
            raise ValueError("memory experiment requires the current numerical gate")
        for track in ["matched_f32", "native"]:
            for engine in ["rust", "mlx"]:
                samples = []
                stop = threading.Event()
                with Worker(
                    command(engine, ROOT / model["path"], track == "matched_f32", 4096),
                    a.output / f"{model['id']}-{track}-{engine}.log",
                ) as worker:
                    samples.append(sample(worker.process.pid))

                    def observe():
                        while not stop.wait(0.1):
                            try:
                                samples.append(sample(worker.process.pid))
                            except OSError:
                                return

                    observer = threading.Thread(target=observe, daemon=True)
                    observer.start()
                    try:
                        result = worker.request(
                            {
                                **workload,
                                "action": "benchmark",
                                "warmups": 1,
                                "repetitions": 1,
                            }
                        )
                    finally:
                        stop.set()
                        observer.join()
                    samples.append(sample(worker.process.pid))
                    row = dict(
                        model=model["id"],
                        track=track,
                        engine=engine,
                        context=4096,
                        workload=workload["id"],
                        ready=worker.ready,
                        peak_os_charged_footprint_bytes=max(
                            s["os_charged_footprint_bytes"] for s in samples
                        ),
                        peak_rss_after_ready_bytes=max(s["rss_bytes"] for s in samples),
                        samples=samples,
                        allocator_metrics={
                            k: v
                            for k, v in result.items()
                            if k not in ["runs", "warmup_runs"]
                        },
                    )
                rows.append(row)
                (a.output / "report.json").write_text(
                    json.dumps(
                        dict(
                            schema_version=1,
                            scope="after ready, including one warmup and one measured fixed workload; not throughput samples",
                            note="OS footprint, RSS and GPU allocator counters are distinct; no whole-system memory ranking",
                            rows=rows,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in [
                                "model",
                                "track",
                                "engine",
                                "peak_os_charged_footprint_bytes",
                                "peak_rss_after_ready_bytes",
                            ]
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
