# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from utils import (
    tensor_to_numpy,
    save_tensor,
    save_lora_or_weight,
    get_tie_word_embeddings,
    get_safetensors_output_name,
    save_safetensors,
    save_configs,
)

SAMPLE_INPUT = (
    "<|im_start|>user\nGive me a short introduction to large language model."
    "<|im_end|>\n<|im_start|>assistant\n"
)


def _save_binary(params, n_layers, dtype, file, tie_word_embeddings=True):
    def save(tensor, transpose=False):
        save_tensor(file, tensor, dtype, transpose=transpose)

    def save_proj(layer_name, proj_name):
        save_lora_or_weight(file, params, layer_name, proj_name, dtype, transpose=True)

    def save_attention(layer_name):
        save(params[f"{layer_name}input_layernorm.weight"])
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_proj(layer_name, f"self_attn.{proj}")
            norm_key = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if norm_key in params:
                save(params[norm_key])

    def save_feed_forward(layer_name):
        save(params[f"{layer_name}post_attention_layernorm.weight"])
        # Keep existing nntrainer binary weight order: up, gate, down
        for proj in ["up_proj", "gate_proj", "down_proj"]:
            save_proj(layer_name, f"mlp.{proj}")

    save(params["model.embed_tokens.weight"])

    for i in range(n_layers):
        layer_name = f"model.layers.{i}."
        save_attention(layer_name)
        save_feed_forward(layer_name)

    save(params["model.norm.weight"])

    if not tie_word_embeddings:
        save(params["lm_head.weight"], transpose=True)


def _collect_safetensors(params, n_layers, dtype, tie_word_embeddings=True):
    """Collect (nntrainer_name, ndarray) pairs for safetensors export."""
    weights = []

    def add(name, tensor, transpose=False):
        weights.append((name, tensor_to_numpy(tensor, dtype, transpose=transpose)))

    def add_proj(nntr_name, layer_name, proj_name):
        prefix = f"{layer_name}{proj_name}"
        lora_key = f"{prefix}.lora_A.default.weight"
        if lora_key in params:
            add(nntr_name, params[f"{prefix}.base_layer.weight"], transpose=True)
        else:
            add(nntr_name, params[f"{prefix}.weight"], transpose=True)

    add("embedding0:Embedding", params["model.embed_tokens.weight"])

    for i in range(n_layers):
        hf = f"model.layers.{i}."
        p = f"layer{i}"

        add(f"{p}_attention_norm:gamma", params[f"{hf}input_layernorm.weight"])
        add_proj(f"{p}_wq:weight", hf, "self_attn.q_proj")

        q_norm = f"{hf}self_attn.q_norm.weight"
        if q_norm in params:
            add(f"{p}_q_norm:gamma", params[q_norm])

        add_proj(f"{p}_wk:weight", hf, "self_attn.k_proj")

        k_norm = f"{hf}self_attn.k_norm.weight"
        if k_norm in params:
            add(f"{p}_k_norm:gamma", params[k_norm])

        add_proj(f"{p}_wv:weight", hf, "self_attn.v_proj")
        add_proj(f"{p}_attention_out:weight", hf, "self_attn.o_proj")

        add(f"{p}_ffn_norm:gamma", params[f"{hf}post_attention_layernorm.weight"])
        add_proj(f"{p}_ffn_gate:weight", hf, "mlp.gate_proj")
        add_proj(f"{p}_ffn_up:weight", hf, "mlp.up_proj")
        add_proj(f"{p}_ffn_down:weight", hf, "mlp.down_proj")

    add("output_norm:gamma", params["model.norm.weight"])

    if not tie_word_embeddings:
        add("output_of_causallm:weight", params["lm_head.weight"], transpose=True)

    return weights


def convert(model_path, output_name, dtype, **kwargs):
    """Convert Qwen3 dense weights to nntrainer binary or safetensors format."""
    use_safetensors = kwargs.get("safetensors", False)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()

    tie = get_tie_word_embeddings(config)
    print(f"tie_word_embeddings: {tie}")

    params = model.state_dict()

    if use_safetensors:
        out_path = get_safetensors_output_name(output_name)
        weights = _collect_safetensors(params, config.num_hidden_layers, dtype, tie_word_embeddings=tie)
        save_safetensors(weights, out_path, dtype)
        effective_output = out_path
    else:
        with open(output_name, "wb") as f:
            _save_binary(params, config.num_hidden_layers, dtype, f, tie_word_embeddings=tie)
        print(f"Saved binary: {output_name}")
        effective_output = output_name

    nntr_extra = {
        "model_type": "CausalLM",
        "sample_input": SAMPLE_INPUT,
    }
    save_configs(effective_output, dtype, hf_config=config, model_path=model_path, nntr_extra=nntr_extra)
