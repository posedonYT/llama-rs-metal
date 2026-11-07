#!/usr/bin/env python3
"""Isolated MLX LM reference with the same synchronous host-logit API boundary."""

import argparse
import importlib.metadata
import json
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache


def floating_dtype(model):
    for name, value in tree_flatten(model.parameters()):
        if (
            "embed_tokens" in name
            and name.endswith(("scales", "weight"))
            and mx.issubdtype(value.dtype, mx.floating)
        ):
            return value.dtype
    raise ValueError("cannot identify embedding output precision")


class Reference:
    def __init__(self, path, float32, context):
        self.model, self.tokenizer = load(path)
        if float32:
            self.model.set_dtype(mx.float32)
        mx.eval(self.model.parameters())
        root = json.loads((Path(path) / "config.json").read_text())
        self.config = root.get("text_config", root)
        self.dtype = floating_dtype(self.model)
        self.context = context
        self.cache = None

    def reset(self):
        c = self.config
        self.cache = make_prompt_cache(self.model)
        for cache in self.cache:
            if isinstance(cache, KVCache):
                shape = (
                    1,
                    c.get("num_key_value_heads", c["num_attention_heads"]),
                    self.context,
                    c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
                )
                cache.keys = mx.zeros(shape, self.dtype)
                cache.values = mx.zeros(shape, self.dtype)
                cache.offset = 0
            elif isinstance(cache, ArraysCache):
                dk, kh = c["linear_key_head_dim"], c["linear_num_key_heads"]
                dv, vh = c["linear_value_head_dim"], c["linear_num_value_heads"]
                cache[0] = mx.zeros(
                    (1, c["linear_conv_kernel_dim"] - 1, 2 * dk * kh + dv * vh),
                    self.dtype,
                )
                cache[1] = mx.zeros((1, vh, dv, dk), mx.float32)
            else:
                raise ValueError(f"unsupported cache type: {type(cache)}")
        mx.eval(
            [
                value
                for _, value in tree_flatten([c.state for c in self.cache])
                if isinstance(value, mx.array)
            ]
        )
        mx.synchronize()

    def forward(self, ids):
        logits = self.model(mx.array([ids], mx.uint32), cache=self.cache)[
            :, -1, :
        ].astype(mx.float32)
        mx.eval(logits)
        # Match Rust's Session::prefill/decode: all final-token logits are host-visible.
        host = np.array(logits[0], dtype=np.float32, copy=True)
        if not np.isfinite(host).all():
            raise ValueError("nonfinite logits")
        return host

    def prefill(self, ids):
        # Match MLX-LM generate_step: evaluate cache state for all but the last
        # token, so lazy evaluation prunes the unused vocabulary projection.
        # Calling forward() on the entire prompt would benchmark a less efficient
        # path than the framework's actual generation implementation.
        offset = 0
        while len(ids) - offset > 1:
            count = min(2048, len(ids) - offset - 1)
            self.model(
                mx.array([ids[offset : offset + count]], mx.uint32), cache=self.cache
            )
            mx.eval([c.state for c in self.cache])
            offset += count
            mx.clear_cache()
        return self.forward(ids[offset:])

    def run(self, prompt, decode):
        start = time.perf_counter()
        self.reset()
        reset = time.perf_counter() - start
        start = time.perf_counter()
        logits = self.prefill(prompt)
        last = int(np.argmax(logits))
        prefill = time.perf_counter() - start
        latencies = []
        start = time.perf_counter()
        for token in decode:
            step = time.perf_counter()
            logits = self.forward([token])
            last = int(np.argmax(logits))
            latencies.append(time.perf_counter() - step)
        elapsed = time.perf_counter() - start
        return dict(
            reset_seconds=reset,
            prefill_seconds=prefill,
            request_ttft_seconds=reset + prefill,
            decode_seconds=elapsed,
            decode_step_seconds=latencies,
            prefill_tokens_per_second=len(prompt) / prefill,
            decode_tokens_per_second=len(decode) / elapsed,
            last_prediction=last,
        )

    def handle(self, request):
        action = request["action"]
        if action == "tokenize":
            return {
                "ids": self.tokenizer.encode(
                    request["text"], add_special_tokens=request.get("special", True)
                )
            }
        prompt, decode = request["prompt_ids"], request["decode_ids"]
        if not prompt or len(prompt) + len(decode) > self.context:
            raise ValueError("invalid workload/context")
        if action == "verify":
            directory = Path(request["output_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            self.reset()
            chunk = request.get("chunk_size", len(prompt))
            if "chunk_size" not in request:
                logits = self.prefill(prompt)
            else:
                for i in range(0, len(prompt), chunk):
                    logits = self.forward(prompt[i : i + chunk])
            traces = []

            def save(name, values):
                path = directory / name
                values.astype("<f4").tofile(path)
                traces.append(
                    dict(
                        path=str(path),
                        top1=int(np.argmax(values)),
                        count=int(values.size),
                    )
                )

            save("prefill.f32", logits)
            for i, token in enumerate(decode):
                save(f"decode-{i}.f32", self.forward([token]))
            states = []
            if request.get("state", False):
                for layer, cache in enumerate(self.cache):
                    arrays = (
                        [
                            (
                                "key",
                                cache.keys[0, :, : cache.offset].transpose(1, 0, 2),
                            ),
                            (
                                "value",
                                cache.values[0, :, : cache.offset].transpose(1, 0, 2),
                            ),
                        ]
                        if isinstance(cache, KVCache)
                        else [
                            ("convolution", cache[0][0].transpose(1, 0)),
                            ("recurrent", cache[1][0]),
                        ]
                    )
                    for kind, value in arrays:
                        path = directory / f"layer-{layer}-{kind}.f32"
                        np.array(value.astype(mx.float32)).astype("<f4").tofile(path)
                        states.append(
                            dict(
                                layer=layer,
                                kind=kind,
                                shape=list(value.shape),
                                path=str(path),
                            )
                        )
            return {"traces": traces, "states": states, "tokens": prompt + decode}
        if action == "benchmark":
            warmups = [
                self.run(prompt, decode) for _ in range(request.get("warmups", 2))
            ]
            mx.reset_peak_memory()
            runs = [
                self.run(prompt, decode) for _ in range(request.get("repetitions", 2))
            ]
            return dict(
                runs=runs,
                warmup_runs=warmups,
                metal_active_bytes=mx.get_active_memory(),
                metal_peak_bytes=mx.get_peak_memory(),
                metal_cache_bytes=mx.get_cache_memory(),
                process_peak_rss_bytes=resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                prefill_policy="mlx_lm_generate_step_2048_then_last",
                boundary="full_logits_host_greedy",
                cache_policy="preallocated_reset_outside_forward",
            )
        raise ValueError(f"unknown action {action}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--f32", action="store_true")
    parser.add_argument("--context", type=int, default=4096)
    args = parser.parse_args()
    start = time.perf_counter()
    ref = Reference(args.model, args.f32, args.context)
    print(
        json.dumps(
            dict(
                type="ready",
                protocol=1,
                engine="mlx",
                versions={
                    k: importlib.metadata.version(k)
                    for k in ["mlx", "mlx-lm", "numpy", "transformers"]
                },
                float_weights_f32=args.f32,
                activation_dtype=str(ref.dtype),
                context=args.context,
                load_seconds=time.perf_counter() - start,
            )
        ),
        flush=True,
    )
    for line in sys.stdin:
        try:
            result = ref.handle(json.loads(line))
            response = dict(ok=True, result=result)
        except Exception as error:
            response = dict(ok=False, error=str(error))
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
