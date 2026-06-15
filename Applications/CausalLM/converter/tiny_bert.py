# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

import torch
from transformers import AutoConfig, AutoModel

from utils import save_tensor, save_configs

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


def convert(model_path, output_name, dtype, **kwargs):
    """Convert a (multilingual tiny) BERT encoder to nntrainer binary format."""
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()

    with open(output_name, "wb") as f:
        _save_weights(model.state_dict(), config, dtype, f)

    print(f"Saved binary: {output_name}")

    nntr_extra = {
        "model_type": "embedding",
        "sample_input": SAMPLE_INPUT,
    }
    save_configs(output_name, dtype, hf_config=config, model_path=model_path, nntr_extra=nntr_extra)
