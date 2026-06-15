# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from utils import save_tensor, save_lora_or_weight, save_configs

SAMPLE_INPUT = "Explain the concept of AI"
# Embedding variant — use a plain sentence to embed.
SAMPLE_INPUT_EMBEDDING = "This is an example sentence"


def _save_weights(params, config, dtype, file, save_lm_head=True):
    n_layers = config.num_hidden_layers

    def save(tensor, is_rms=False):
        save_tensor(file, tensor, dtype, add_one=is_rms)

    def save_proj(layer_name, proj_name):
        save_lora_or_weight(file, params, layer_name, proj_name, dtype, transpose=True)

    def save_attention(layer_name):
        save(params[f"{layer_name}input_layernorm.weight"], is_rms=True)
        save_proj(layer_name, "self_attn.q_proj")
        q_norm = f"{layer_name}self_attn.q_norm.weight"
        if q_norm in params:
            save(params[q_norm], is_rms=True)
        save_proj(layer_name, "self_attn.k_proj")
        k_norm = f"{layer_name}self_attn.k_norm.weight"
        if k_norm in params:
            save(params[k_norm], is_rms=True)
        save_proj(layer_name, "self_attn.v_proj")
        save_proj(layer_name, "self_attn.o_proj")

    def save_feed_forward(layer_name):
        save(params[f"{layer_name}post_attention_layernorm.weight"], is_rms=True)
        save(params[f"{layer_name}pre_feedforward_layernorm.weight"], is_rms=True)
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            save_proj(layer_name, f"mlp.{proj}")
        save(params[f"{layer_name}post_feedforward_layernorm.weight"], is_rms=True)

    save(params["model.embed_tokens.weight"])

    for i in range(n_layers):
        layer_name = f"model.layers.{i}."
        save_attention(layer_name)
        save_feed_forward(layer_name)

    save(params["model.norm.weight"], is_rms=True)

    if save_lm_head and "lm_head.weight" in params:
        save_tensor(file, params["lm_head.weight"], dtype, transpose=True)


def _save_embedding_model(model, dtype, file):
    """Save Gemma3-backed SentenceTransformer embedding model."""
    raw_params = model.state_dict()
    gemma_params = {
        key.removeprefix("0.auto_model.").removeprefix("0."): value
        for key, value in raw_params.items()
        if key.startswith("0.")
    }
    _save_weights(gemma_params, model[0].auto_model.config, dtype, file, save_lm_head=False)

    for module_name, module in model._modules.items():
        component = module.__class__.__name__
        if component in ["Transformer", "Pooling", "Normalize"]:
            continue
        if component != "Dense":
            raise NotImplementedError(f"Unsupported SentenceTransformer module: {module_name} ({component})")
        save_tensor(file, raw_params[f"{module_name}.linear.weight"], dtype, transpose=True)
        bias_key = f"{module_name}.linear.bias"
        if bias_key in raw_params:
            save_tensor(file, raw_params[bias_key], dtype)


def convert(model_path, output_name, dtype, **kwargs):
    """Convert Gemma3 weights to nntrainer binary format.

    Pass is_embedding=True (or --embedding_model CLI flag) to load as a
    SentenceTransformer embedding model instead of AutoModelForCausalLM.
    """
    is_embedding = kwargs.get("is_embedding", False)

    if is_embedding:
        from sentence_transformers import SentenceTransformer
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32
        model = SentenceTransformer(model_path, trust_remote_code=True, model_kwargs={"torch_dtype": torch_dtype})
        model.eval()
        hf_config = model[0].auto_model.config
        with open(output_name, "wb") as f:
            _save_embedding_model(model, dtype, f)
        nntr_extra = {
            "model_type": "embedding",
            "sample_input": SAMPLE_INPUT_EMBEDDING,
        }
    else:
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
        model.eval()
        save_lm_head = not getattr(hf_config, "tie_word_embeddings", False)
        with open(output_name, "wb") as f:
            _save_weights(model.state_dict(), hf_config, dtype, f, save_lm_head=save_lm_head)
        nntr_extra = {
            "model_type": "CausalLM",
            "sample_input": SAMPLE_INPUT,
        }

    print(f"Saved binary: {output_name}")
    save_configs(output_name, dtype, hf_config=hf_config, model_path=model_path, nntr_extra=nntr_extra)
