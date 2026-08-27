# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

"""Generate non-degenerate Hugging Face fixtures for dense Qwen3.5/3.6."""

import argparse
import json
import pathlib
import sys

import torch
import transformers
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[4]
CONVERTER_DIR = REPO_ROOT / "Applications" / "CausalLM" / "res" / "qwen3_5"
DEFAULT_OUT = THIS_DIR.parent / "qwen3_5_tiny"
sys.path.insert(0, str(CONVERTER_DIR))

from weight_converter import save_qwen3_5_for_nntrainer  # noqa: E402


TINY_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": 31,
    "head_dim": 8,
    "hidden_act": "silu",
    "hidden_size": 64,
    "intermediate_size": 64,
    "layer_types": ["linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 8,
    "linear_num_key_heads": 8,
    "linear_num_value_heads": 8,
    "linear_value_head_dim": 8,
    "max_position_embeddings": 8,
    "num_attention_heads": 8,
    "num_hidden_layers": 2,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-6,
    "rope_parameters": {
        "partial_rotary_factor": 0.5,
        "rope_theta": 10000.0,
        "rope_type": "default",
    },
    "tie_word_embeddings": True,
    "use_cache": True,
    "vocab_size": 32,
}

INPUT_IDS = [1, 4, 2, 3]
N_GEN = 4

TINY_TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {
            "id": 31,
            "content": "<eos>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {
            "<unk>": 0,
            "hello": 1,
            "world": 2,
            **{f"tok{i}": i for i in range(3, 31)},
            "<eos>": 31,
        },
        "unk_token": "<unk>",
    },
}


def build_model(seed):
    torch.manual_seed(seed)
    config = Qwen3_5TextConfig(**TINY_CONFIG)
    model = Qwen3_5ForCausalLM(config)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("A_log"):
                values = torch.linspace(0.4, 1.2, parameter.numel())
                parameter.copy_(values.log().reshape_as(parameter))
            elif name.endswith("dt_bias"):
                parameter.copy_(
                    torch.linspace(-0.4, 0.2, parameter.numel()).reshape_as(parameter)
                )
            elif name.endswith("linear_attn.norm.weight"):
                parameter.copy_(
                    1.0
                    + 0.04
                    * torch.randn(
                        parameter.shape, generator=generator, dtype=parameter.dtype
                    )
                )
            elif name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                # Qwen3.5 regular RMSNorm parameters are zero-centered offsets.
                parameter.copy_(
                    0.04
                    * torch.randn(
                        parameter.shape, generator=generator, dtype=parameter.dtype
                    )
                )
            else:
                parameter.copy_(
                    0.08
                    * torch.randn(
                        parameter.shape, generator=generator, dtype=parameter.dtype
                    )
                )
    model.eval()
    return config, model


def run_forward(model):
    ids = torch.tensor([INPUT_IDS], dtype=torch.long)
    with torch.no_grad():
        return model(ids, use_cache=False).logits[0, -1].float().tolist()


def run_greedy(model, n):
    ids = torch.tensor([INPUT_IDS], dtype=torch.long)
    with torch.no_grad():
        output = model.generate(
            ids,
            max_new_tokens=n,
            do_sample=False,
            repetition_penalty=1.0,
            use_cache=False,
            eos_token_id=None,
        )
    return output[0, len(INPUT_IDS) :].tolist()[:n]


def write_configs(out_dir, binary_name):
    root_config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": dict(TINY_CONFIG),
        "tie_word_embeddings": True,
    }
    generation_config = {
        "bos_token_id": 0,
        "eos_token_id": 31,
        "do_sample": False,
        "top_k": 1,
        "top_p": 1.0,
        "temperature": 1.0,
    }
    nntr_config = {
        "bad_word_ids": [],
        "batch_size": 1,
        "embedding_dtype": "FP32",
        "fc_layer_dtype": "FP32",
        "init_seq_len": 4,
        "lmhead_dtype": "FP32",
        "max_seq_len": 8,
        "model_file_name": binary_name,
        "model_tensor_type": "FP32-FP32",
        "model_type": "CausalLM",
        "num_to_generate": 1,
        "tokenizer_file": "tokenizer.json",
    }
    for name, value in (
        ("config.json", root_config),
        ("generation_config.json", generation_config),
        ("nntr_config.json", nntr_config),
        ("tokenizer.json", TINY_TOKENIZER),
    ):
        with (out_dir / name).open("w", encoding="utf-8") as output:
            json.dump(value, output, indent=2)
            output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=N_GEN)
    args = parser.parse_args()

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    binary_name = "nntr_qwen3_5_tiny_fp32.bin"
    config, model = build_model(args.seed)
    with (out_dir / binary_name).open("wb") as output:
        save_qwen3_5_for_nntrainer(
            model.state_dict(), config, "float32", output
        )

    write_configs(out_dir, binary_name)
    values = {
        "input_ids.json": INPUT_IDS,
        "reference_logits.json": run_forward(model),
        "reference_tokens.json": run_greedy(model, args.n),
        "meta.json": {
            "seed": args.seed,
            "n_gen": args.n,
            "input_ids": INPUT_IDS,
            "logits_atol_fp32": 0.02,
            "logits_atol_q40": 2.0,
            "prefix_match_min": 2,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "reference": "Hugging Face Qwen3_5ForCausalLM",
        },
    }
    for name, value in values.items():
        with (out_dir / name).open("w", encoding="utf-8") as output:
            json.dump(value, output, indent=2)
            output.write("\n")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
