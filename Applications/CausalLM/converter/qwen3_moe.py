# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from utils import save_tensor, save_lora_or_weight, save_configs, make_chat_input

CHAT_QUESTION = "Give me a short introduction to large language model."
# Pre-formatted fallback used only when no chat_template is available.
SAMPLE_INPUT = (
    "<|im_start|>user\nGive me a short introduction to large language model."
    "<|im_end|>\n<|im_start|>assistant\n"
)


def _save_weights(params, config, dtype, file):
    n_experts = config.num_experts

    def save(name, transpose=False):
        save_tensor(file, params[name], dtype, transpose=transpose)

    def save_proj(layer_name, proj_name):
        save_lora_or_weight(file, params, layer_name, proj_name, dtype, transpose=True)

    def save_attention(layer_name):
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_proj(layer_name, f"self_attn.{proj}")
            norm_key = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if norm_key in params:
                save(norm_key)

    def save_feed_forward(layer_name):
        save(f"{layer_name}mlp.gate.weight", transpose=True)
        for expert_id in range(n_experts):
            for proj in ["up_proj", "gate_proj", "down_proj"]:
                save_proj(layer_name, f"mlp.experts.{expert_id}.{proj}")

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
    """Convert Qwen3 MoE weights to nntrainer binary format."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()

    with open(output_name, "wb") as f:
        _save_weights(model.state_dict(), config, dtype, f)

    print(f"Saved binary: {output_name}")

    nntr_extra = {
        "model_type": "CausalLM",
        "sample_input": SAMPLE_INPUT,
        "chat_input": make_chat_input(CHAT_QUESTION),
    }
    save_configs(output_name, dtype, hf_config=config, model_path=model_path, nntr_extra=nntr_extra)
