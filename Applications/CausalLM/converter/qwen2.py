# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from utils import (
    save_tensor,
    save_lora_or_weight,
    save_configs,
    get_tie_word_embeddings,
    make_chat_input,
)

# Plain user question; the chat_template is applied at inference time via chat_input.
CHAT_QUESTION = "Give me a short introduction to large language model."
# Pre-formatted fallback used only when no chat_template is available.
SAMPLE_INPUT = (
    "<|im_start|>user\nGive me a short introduction to large language model."
    "<|im_end|>\n<|im_start|>assistant\n"
)
# KaLM is an embedding model — use a plain sentence instead of a chat prompt.
SAMPLE_INPUT_KALM = "This is an example sentence"


def _save_weights(params, n_layers, dtype, file, layer_prefix_fn=None, save_lm_head=False):
    """Core Qwen2-architecture weight saver.

    layer_prefix_fn: callable(layer_idx) -> str. Defaults to standard HF naming.
    The KaLM-embedding model uses '0.auto_model.layers.{i}.' prefix instead.
    save_lm_head: save lm_head.weight separately (only when embeddings are NOT tied).
    """
    if layer_prefix_fn is None:
        layer_prefix_fn = lambda i: f"model.layers.{i}."

    def save(tensor, transpose=False):
        save_tensor(file, tensor, dtype, transpose=transpose)

    def save_proj(layer_name, proj_name):
        save_lora_or_weight(file, params, layer_name, proj_name, dtype, transpose=True)

    def save_attention(layer_name):
        save(params[f"{layer_name}input_layernorm.weight"])
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_proj(layer_name, f"self_attn.{proj}")
            if proj != "o_proj":
                save(params[f"{layer_name}self_attn.{proj}.bias"])

    def save_feed_forward(layer_name):
        save(params[f"{layer_name}post_attention_layernorm.weight"])
        # nntrainer's createMlp builds ffn_up before ffn_gate, so the binary must
        # store mlp weights in up, gate, down order (see models/transformer.cpp).
        for proj in ["up_proj", "gate_proj", "down_proj"]:
            save_proj(layer_name, f"mlp.{proj}")

    embed_key = "0.auto_model.embed_tokens.weight" if "0.auto_model.embed_tokens.weight" in params else "model.embed_tokens.weight"
    save(params[embed_key])

    for i in range(n_layers):
        layer_name = layer_prefix_fn(i)
        save_attention(layer_name)
        save_feed_forward(layer_name)

    norm_key = "0.auto_model.norm.weight" if "0.auto_model.norm.weight" in params else "model.norm.weight"
    save(params[norm_key])

    # For tied-embedding models (e.g. Qwen2-0.5B, Qwen2.5-0.5B/1.5B) lm_head shares
    # model.embed_tokens.weight and is not saved separately. Larger Qwen2.5 variants
    # (3B/7B/...) set tie_word_embeddings=false and ship a separate lm_head.weight.
    if save_lm_head and "lm_head.weight" in params:
        save(params["lm_head.weight"], transpose=True)


def convert(model_path, output_name, dtype, **kwargs):
    """Convert Qwen2 (or KaLM-embedding) weights to nntrainer binary format."""
    is_kalm = kwargs.get("is_kalm", False)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    if is_kalm:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path, trust_remote_code=True, model_kwargs={"torch_dtype": torch.float32})
        model.eval()
        layer_prefix_fn = lambda i: f"0.auto_model.layers.{i}."
        # KaLM is an embedding model — no lm_head.
        save_lm_head = False
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
        model.eval()
        layer_prefix_fn = None
        # Save lm_head only when embeddings are not tied (e.g. Qwen2.5-3B/7B/...).
        save_lm_head = not get_tie_word_embeddings(config)
        print(f"tie_word_embeddings: {not save_lm_head}")

    params = model.state_dict()

    with open(output_name, "wb") as f:
        _save_weights(params, config.num_hidden_layers, dtype, f,
                      layer_prefix_fn=layer_prefix_fn, save_lm_head=save_lm_head)

    print(f"Saved binary: {output_name}")

    nntr_extra = {
        "model_type": "embedding" if is_kalm else "CausalLM",
        "sample_input": SAMPLE_INPUT_KALM if is_kalm else SAMPLE_INPUT,
    }
    if is_kalm:
        nntr_extra["is_causal"] = False
    else:
        nntr_extra["chat_input"] = make_chat_input(CHAT_QUESTION)
    save_configs(output_name, dtype, hf_config=config, model_path=model_path,
                 nntr_extra=nntr_extra, sentence_transformer=is_kalm)
