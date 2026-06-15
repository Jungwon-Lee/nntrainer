# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from utils import save_tensor, save_lora_or_weight, save_configs


SAMPLE_INPUT = (
    "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\nCurrent date: 2025-09-10\n\nReasoning: low\n\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    "<|end|><|start|>user<|message|>What is on-device AI?<|im_end|>\n<|end|><|start|>assistant"
)


def _save_weights(params, config, dtype, file):
    n_experts = config.num_local_experts

    def save(name, transpose=False):
        save_tensor(file, params[name], dtype, transpose=transpose)

    def save_proj(layer_name, proj_name):
        save_lora_or_weight(file, params, layer_name, proj_name, dtype, transpose=True)

    def save_attention(layer_name):
        for proj in ["q_proj", "k_proj", "v_proj"]:
            save_proj(layer_name, f"self_attn.{proj}")
            save(f"{layer_name}self_attn.{proj}.bias")
        save(f"{layer_name}self_attn.sinks")
        save_proj(layer_name, "self_attn.o_proj")
        save(f"{layer_name}self_attn.o_proj.bias")

    def save_feed_forward(layer_name):
        save(f"{layer_name}mlp.router.weight", transpose=True)
        save(f"{layer_name}mlp.router.bias")

        gate_up = params[f"{layer_name}mlp.experts.gate_up_proj"]
        gate_up_bias = params[f"{layer_name}mlp.experts.gate_up_proj_bias"]
        down = params[f"{layer_name}mlp.experts.down_proj"]
        down_bias = params[f"{layer_name}mlp.experts.down_proj_bias"]
        for n in range(n_experts):
            # gate_up_proj interleaves up (odd) and gate (even) columns.
            save_tensor(file, gate_up[..., 1::2][n], dtype)       # up_proj
            save_tensor(file, gate_up_bias[..., 1::2][n], dtype)
            save_tensor(file, gate_up[..., ::2][n], dtype)        # gate_proj
            save_tensor(file, gate_up_bias[..., ::2][n], dtype)
            save_tensor(file, down[n], dtype)                     # down_proj
            save_tensor(file, down_bias[n], dtype)

    save("model.embed_tokens.weight")

    for i in range(config.num_hidden_layers):
        layer_name = f"model.layers.{i}."
        save(f"{layer_name}input_layernorm.weight")
        save_attention(layer_name)
        save(f"{layer_name}post_attention_layernorm.weight")
        save_feed_forward(layer_name)

    save("model.norm.weight")
    save("lm_head.weight", transpose=True)


def convert(model_path, output_name, dtype, **kwargs):
    """Convert GPT-OSS 20B MoE weights to nntrainer binary format."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()

    with open(output_name, "wb") as f:
        _save_weights(model.state_dict(), config, dtype, f)

    print(f"Saved binary: {output_name}")

    nntr_extra = {
        "model_type": "CausalLM",
        "lmhead_dtype": "FP32" if dtype == "float32" else "FP16",
        "sample_input": SAMPLE_INPUT,
    }
    save_configs(output_name, dtype, hf_config=config, model_path=model_path, nntr_extra=nntr_extra)
