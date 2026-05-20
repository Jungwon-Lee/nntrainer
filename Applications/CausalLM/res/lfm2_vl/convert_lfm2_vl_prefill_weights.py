#!/usr/bin/env python3
"""Convert LFM2-VL language-model weights for prefill validation."""

import argparse
import json
from pathlib import Path

import numpy as np
import safetensors.torch
import torch


TEXT_PREFIX = "model.language_model."


def load_state_dict(model_path):
    model_path = Path(model_path)
    if model_path.is_dir():
        model_path = model_path / "model.safetensors"
    if model_path.suffix == ".safetensors":
        return safetensors.torch.load_file(str(model_path))
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def load_config(model_path):
    model_path = Path(model_path)
    config_path = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def save_weight(file, tensor, transpose=False):
    array = tensor.detach().cpu().to(torch.float32).numpy()
    if transpose:
        array = array.T
    array.astype(np.float32).tofile(file)


def save_conv_kernel(file, tensor):
    # HF Conv1d stores [hidden, 1, kernel] in old-to-current order for the
    # padded cross-correlation. nntrainer causal_conv1d expects
    # [kernel, hidden] as current, previous, previous-2.
    array = tensor.detach().cpu().to(torch.float32)[:, 0, :].flip(-1).T.numpy()
    array.astype(np.float32).tofile(file)


def convert(model_path, output_path):
    state = load_state_dict(model_path)
    cfg = load_config(model_path)
    text_cfg = cfg["text_config"] if "text_config" in cfg else cfg
    layer_types = text_cfg["layer_types"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        for layer_id, layer_type in enumerate(layer_types):
            prefix = TEXT_PREFIX + f"layers.{layer_id}."
            if layer_type == "full_attention":
                save_weight(file, state[prefix + "operator_norm.weight"])
                attn = prefix + "self_attn."
                save_weight(file, state[attn + "q_proj.weight"], transpose=True)
                save_weight(file, state[attn + "q_layernorm.weight"])
                save_weight(file, state[attn + "k_proj.weight"], transpose=True)
                save_weight(file, state[attn + "k_layernorm.weight"])
                save_weight(file, state[attn + "v_proj.weight"], transpose=True)
                save_weight(file, state[attn + "out_proj.weight"], transpose=True)
            else:
                save_weight(file, state[prefix + "operator_norm.weight"])
                conv = prefix + "conv."
                save_weight(file, state[conv + "in_proj.weight"], transpose=True)
                if text_cfg.get("conv_bias", False):
                    save_weight(file, state[conv + "in_proj.bias"])
                save_conv_kernel(file, state[conv + "conv.weight"])
                if text_cfg.get("conv_bias", False):
                    save_weight(file, state[conv + "conv.bias"])
                save_weight(file, state[conv + "out_proj.weight"], transpose=True)
                if text_cfg.get("conv_bias", False):
                    save_weight(file, state[conv + "out_proj.bias"])

            save_weight(file, state[prefix + "ffn_norm.weight"])
            mlp = prefix + "feed_forward."
            # nntrainer loads these FC weights in layer-name order
            # (ffn_gate before ffn_up), while createMlp computes
            # swiglu(gate, up). Save HF w1 before w3 to preserve
            # silu(w1(x)) * w3(x).
            save_weight(file, state[mlp + "w1.weight"], transpose=True)
            save_weight(file, state[mlp + "w3.weight"], transpose=True)
            save_weight(file, state[mlp + "w2.weight"], transpose=True)

        save_weight(file, state[TEXT_PREFIX + "embedding_norm.weight"])
        save_weight(file, state[TEXT_PREFIX + "embed_tokens.weight"], transpose=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.model, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
