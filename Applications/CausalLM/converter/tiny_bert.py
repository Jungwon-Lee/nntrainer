# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

import torch
from transformers import AutoConfig, AutoModel

from utils import save_tensor, save_configs, tensor_to_numpy, get_safetensors_output_name, save_safetensors

SAMPLE_INPUT = "This is a sentence"


def _save_weights(params, config, dtype, file):
    """Save BERT encoder weights in nntrainer layer order."""

    def save(name, transpose=False):
        save_tensor(file, params[name], dtype, transpose=transpose)

    # Embeddings
    save("embeddings.word_embeddings.weight")
    save("embeddings.token_type_embeddings.weight")
    save("embeddings.position_embeddings.weight")
    save("embeddings.LayerNorm.weight")
    save("embeddings.LayerNorm.bias")

    # Encoder layers
    for i in range(config.num_hidden_layers):
        p = f"encoder.layer.{i}."
        save(f"{p}attention.self.query.weight", transpose=True)
        save(f"{p}attention.self.query.bias")
        save(f"{p}attention.self.key.weight", transpose=True)
        save(f"{p}attention.self.key.bias")
        save(f"{p}attention.self.value.weight", transpose=True)
        save(f"{p}attention.self.value.bias")
        save(f"{p}attention.output.dense.weight", transpose=True)
        save(f"{p}attention.output.dense.bias")
        save(f"{p}attention.output.LayerNorm.weight")
        save(f"{p}attention.output.LayerNorm.bias")
        save(f"{p}intermediate.dense.weight", transpose=True)
        save(f"{p}intermediate.dense.bias")
        save(f"{p}output.dense.weight", transpose=True)
        save(f"{p}output.dense.bias")
        save(f"{p}output.LayerNorm.weight")
        save(f"{p}output.LayerNorm.bias")

    # Pooler (optional)
    if "pooler.dense.weight" in params:
        save("pooler.dense.weight", transpose=True)
        save("pooler.dense.bias")


def _collect_safetensors(params, config, dtype):
    """Collect (nntrainer_name, ndarray) pairs for safetensors export."""
    weights = []

    def add(name, hf_key, transpose=False):
        weights.append((name, tensor_to_numpy(params[hf_key], dtype, transpose=transpose)))

    add("embedding0:Embedding", "embeddings.word_embeddings.weight")
    add("position_embedding:Embedding", "embeddings.position_embeddings.weight")
    add("token_type_embedding:Embedding", "embeddings.token_type_embeddings.weight")
    add("embedding_norm:gamma", "embeddings.LayerNorm.weight")
    add("embedding_norm:beta", "embeddings.LayerNorm.bias")

    for i in range(config.num_hidden_layers):
        hf = f"encoder.layer.{i}."
        nn = f"layer{i}"
        sa = f"{hf}attention.self."
        add(f"{nn}_wq:weight", f"{sa}query.weight", transpose=True)
        add(f"{nn}_wq:bias", f"{sa}query.bias")
        add(f"{nn}_wk:weight", f"{sa}key.weight", transpose=True)
        add(f"{nn}_wk:bias", f"{sa}key.bias")
        add(f"{nn}_wv:weight", f"{sa}value.weight", transpose=True)
        add(f"{nn}_wv:bias", f"{sa}value.bias")
        add(f"{nn}_attention_out:weight", f"{hf}attention.output.dense.weight", transpose=True)
        add(f"{nn}_attention_out:bias", f"{hf}attention.output.dense.bias")
        add(f"{nn}_attention_norm:gamma", f"{hf}attention.output.LayerNorm.weight")
        add(f"{nn}_attention_norm:beta", f"{hf}attention.output.LayerNorm.bias")
        add(f"{nn}_ffn_fc1:weight", f"{hf}intermediate.dense.weight", transpose=True)
        add(f"{nn}_ffn_fc1:bias", f"{hf}intermediate.dense.bias")
        add(f"{nn}_ffn_down:weight", f"{hf}output.dense.weight", transpose=True)
        add(f"{nn}_ffn_down:bias", f"{hf}output.dense.bias")
        add(f"{nn}_ffn_norm:gamma", f"{hf}output.LayerNorm.weight")
        add(f"{nn}_ffn_norm:beta", f"{hf}output.LayerNorm.bias")

    return weights


def convert(model_path, output_name, dtype, **kwargs):
    """Convert a (multilingual tiny) BERT encoder to nntrainer binary format."""
    use_safetensors = kwargs.get("safetensors", False)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    params = model.state_dict()

    if use_safetensors:
        effective_output = get_safetensors_output_name(output_name)
        weights = _collect_safetensors(params, config, dtype)
        save_safetensors(weights, effective_output, dtype)
    else:
        effective_output = output_name
        with open(output_name, "wb") as f:
            _save_weights(params, config, dtype, f)
        print(f"Saved binary: {output_name}")

    nntr_extra = {
        "model_type": "embedding",
        "sample_input": SAMPLE_INPUT,
    }
    save_configs(effective_output, dtype, hf_config=config, model_path=model_path, nntr_extra=nntr_extra)
