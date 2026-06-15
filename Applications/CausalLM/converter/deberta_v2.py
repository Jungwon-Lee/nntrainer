# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

import numpy as np
import torch
from pathlib import Path

from utils import save_configs

SAMPLE_INPUT = "This is an example sentence"


def _to_numpy(weight, dtype):
    return weight.detach().cpu().numpy().astype(dtype, copy=False)


def _save_weight(file, weight, dtype):
    _to_numpy(weight, dtype).tofile(file)


def _save_linear(file, params, prefix, dtype):
    _save_weight(file, params[f"{prefix}.weight"].transpose(0, 1), dtype)
    _save_weight(file, params[f"{prefix}.bias"], dtype)


def _save_weights(params, config, dtype, file):
    _save_weight(file, params["embeddings.word_embeddings.weight"], dtype)
    _save_weight(file, params["embeddings.LayerNorm.weight"], dtype)
    _save_weight(file, params["embeddings.LayerNorm.bias"], dtype)

    norm_rel_ebd = getattr(config, "norm_rel_ebd", "none").lower().split("|")
    norm_rel_ebd = [item.strip() for item in norm_rel_ebd]
    pos_att_type = getattr(config, "pos_att_type", None) or []
    uses_relative_bias = getattr(config, "relative_attention", False) and (
        "c2p" in pos_att_type or "p2c" in pos_att_type
    )
    saved_relative_embeddings = False

    for i in range(config.num_hidden_layers):
        layer_prefix = f"encoder.layer.{i}"
        attn_prefix = f"{layer_prefix}.attention"
        self_prefix = f"{attn_prefix}.self"

        _save_linear(file, params, f"{self_prefix}.query_proj", dtype)
        _save_linear(file, params, f"{self_prefix}.key_proj", dtype)
        _save_linear(file, params, f"{self_prefix}.value_proj", dtype)

        if uses_relative_bias and not saved_relative_embeddings:
            _save_weight(file, params["encoder.rel_embeddings.weight"], dtype)
            if "layer_norm" in norm_rel_ebd:
                _save_weight(file, params["encoder.LayerNorm.weight"], dtype)
                _save_weight(file, params["encoder.LayerNorm.bias"], dtype)
            saved_relative_embeddings = True

        _save_linear(file, params, f"{attn_prefix}.output.dense", dtype)
        _save_weight(file, params[f"{attn_prefix}.output.LayerNorm.weight"], dtype)
        _save_weight(file, params[f"{attn_prefix}.output.LayerNorm.bias"], dtype)
        _save_linear(file, params, f"{layer_prefix}.intermediate.dense", dtype)
        _save_linear(file, params, f"{layer_prefix}.output.dense", dtype)
        _save_weight(file, params[f"{layer_prefix}.output.LayerNorm.weight"], dtype)
        _save_weight(file, params[f"{layer_prefix}.output.LayerNorm.bias"], dtype)


def _strip_encoder_prefix(params):
    if "embeddings.word_embeddings.weight" in params:
        return params
    prefixes = (
        "0.auto_model.",
        "0.auto_model.deberta.",
        "auto_model.",
        "auto_model.deberta.",
        "deberta.",
    )
    for prefix in prefixes:
        key = prefix + "embeddings.word_embeddings.weight"
        if key in params:
            return {name[len(prefix):]: value for name, value in params.items() if name.startswith(prefix)}
    raise KeyError("Cannot find DeBERTa V2 encoder weights in state_dict")


def _load(model_path):
    from transformers import DebertaV2Model
    try:
        model = DebertaV2Model.from_pretrained(model_path)
        params = model.deberta.state_dict() if hasattr(model, "deberta") else model.state_dict()
        return model.config, _strip_encoder_prefix(params)
    except Exception:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model_path, trust_remote_code=True)
        auto_model = getattr(st[0], "auto_model", None)
        if auto_model is None:
            raise AttributeError("first SentenceTransformer module has no auto_model")
        return auto_model.config, _strip_encoder_prefix(st.state_dict())


def convert(model_path, output_name, dtype, **kwargs):
    """Convert DeBERTa V2 weights to nntrainer binary format."""
    np_dtype = np.float16 if dtype == "float16" else np.float32
    config, params = _load(model_path)

    output_path = Path(output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad(), output_path.open("wb") as f:
        _save_weights(params, config, np_dtype, f)

    print(f"Saved binary: {output_name}")

    nntr_extra = {
        "model_type": "embedding",
        "sample_input": SAMPLE_INPUT,
    }
    save_configs(output_name, dtype, hf_config=config, model_path=model_path,
                 nntr_extra=nntr_extra, sentence_transformer=True)
