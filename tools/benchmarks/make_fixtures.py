#!/usr/bin/env python3
"""Small deterministic *synthetic* MLX architectures, not pretrained checkpoints.
Uses the pinned official MLX-LM models as the oracle, including expert dispatch.
"""

import json

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5, qwen3_next
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer

from protocol import ROOT
from verify import CORPORA, TEXTS

out = ROOT / "models/mlx/fixtures"
out.mkdir(parents=True, exist_ok=True)
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.train_from_iterator(
    [*TEXTS, *CORPORA.values()],
    trainers.BpeTrainer(
        vocab_size=512,
        special_tokens=["<unk>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    ),
)
base = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=512,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    linear_num_value_heads=2,
    linear_num_key_heads=1,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    full_attention_interval=2,
    attention_bias=False,
    tie_word_embeddings=False,
    num_experts=4,
    num_experts_per_tok=2,
    decoder_sparse_step=1,
    shared_expert_intermediate_size=64,
    moe_intermediate_size=128,
    norm_topk_prob=True,
    rope_theta=10000.0,
    partial_rotary_factor=0.5,
    mlp_only_layers=[],
)
manifest = []
for kind, module in [("qwen3_5_moe", qwen3_5), ("qwen3_next", qwen3_next)]:
    for quant, normalize in [(False, True), (True, True), (True, False)]:
        mx.random.seed(42)
        cfg = {**base, "model_type": kind, "norm_topk_prob": normalize}
        if kind == "qwen3_5_moe":
            cfg["rope_parameters"] = dict(
                type="default", rope_theta=10000.0, partial_rotary_factor=0.5
            )
        name = f"synthetic-{kind}-{'mixed4-8' if quant else 'f32'}" + (
            "-unnormalized" if not normalize else ""
        )
        directory = out / name
        directory.mkdir(exist_ok=True)
        (directory / "config.json").write_text(json.dumps(cfg))
        tokenizer.save(str(directory / "tokenizer.json"))
        (directory / "tokenizer_config.json").write_text(
            json.dumps(
                dict(
                    tokenizer_class="PreTrainedTokenizerFast",
                    unk_token="<unk>",
                    model_max_length=4096,
                )
            )
        )
        # Model-specific Transformers classes may install a pretokenizer and add
        # special tokens. Persist that effective contract before sizing embeddings.
        effective = AutoTokenizer.from_pretrained(directory)
        effective.backend_tokenizer.save(str(directory / "tokenizer.json"))
        size = max(effective.get_vocab().values()) + 1
        cfg["vocab_size"] = ((size + 31) // 32) * 32
        model = module.Model(module.ModelArgs.from_dict(cfg))
        if quant:
            nn.quantize(
                model,
                group_size=32,
                bits=4,
                class_predicate=lambda path, layer: (
                    {"bits": 8, "group_size": 32}
                    if path.endswith(("mlp.gate", "shared_expert_gate"))
                    else isinstance(layer, (nn.Linear, nn.Embedding))
                    or hasattr(layer, "to_quantized")
                ),
            )
            cfg["quantization"] = dict(bits=4, group_size=32, mode="affine")
            for path, layer in model.named_modules():
                if getattr(layer, "bits", None) == 8:
                    cfg["quantization"][path] = dict(
                        bits=8, group_size=32, mode="affine"
                    )
        mx.eval(model.parameters())
        mx.save_safetensors(
            str(directory / "model.safetensors"), dict(tree_flatten(model.parameters()))
        )
        (directory / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        manifest.append(
            dict(
                id=name,
                path=str(directory.relative_to(ROOT)),
                synthetic=True,
                seed=42,
                oracle="mlx-lm==0.32.2",
            )
        )
(out / "manifest.json").write_text(
    json.dumps(dict(schema_version=1, models=manifest), indent=2) + "\n"
)
