#!/usr/bin/env python3
"""Convert LFM2-VL SigLIP2 vision tower weights to nntrainer order."""

import argparse
import json
from pathlib import Path

import numpy as np
import safetensors.torch
import torch


VISION_PREFIX = "model.vision_tower.vision_model."


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


def save_weight(file, tensor, dtype=np.float32, transpose=False):
    array = tensor.detach().cpu().to(torch.float32).numpy()
    if transpose:
        array = array.T
    array.astype(dtype).tofile(file)


def get_config(model_path):
    model_path = Path(model_path)
    config_path = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def convert(model_path, output_path, dtype=np.float32, include_projector=False):
    state = load_state_dict(model_path)
    cfg = get_config(model_path)
    vision_cfg = cfg["vision_config"] if "vision_config" in cfg else cfg
    num_layers = vision_cfg["num_hidden_layers"]

    with open(output_path, "wb") as file:
        emb = VISION_PREFIX + "embeddings."
        save_weight(file, state[emb + "patch_embedding.weight"], dtype, transpose=True)
        save_weight(file, state[emb + "patch_embedding.bias"], dtype)
        save_weight(file, state[emb + "position_embedding.weight"].unsqueeze(0).unsqueeze(0), dtype)

        for layer_id in range(num_layers):
            prefix = VISION_PREFIX + f"encoder.layers.{layer_id}."
            save_weight(file, state[prefix + "layer_norm1.weight"], dtype)
            save_weight(file, state[prefix + "layer_norm1.bias"], dtype)

            attn = prefix + "self_attn."
            save_weight(file, state[attn + "q_proj.weight"], dtype, transpose=True)
            save_weight(file, state[attn + "q_proj.bias"], dtype)
            save_weight(file, state[attn + "k_proj.weight"], dtype, transpose=True)
            save_weight(file, state[attn + "k_proj.bias"], dtype)
            save_weight(file, state[attn + "v_proj.weight"], dtype, transpose=True)
            save_weight(file, state[attn + "v_proj.bias"], dtype)
            save_weight(file, state[attn + "out_proj.weight"], dtype, transpose=True)
            save_weight(file, state[attn + "out_proj.bias"], dtype)

            save_weight(file, state[prefix + "layer_norm2.weight"], dtype)
            save_weight(file, state[prefix + "layer_norm2.bias"], dtype)

            mlp = prefix + "mlp."
            save_weight(file, state[mlp + "fc1.weight"], dtype, transpose=True)
            save_weight(file, state[mlp + "fc1.bias"], dtype)
            save_weight(file, state[mlp + "fc2.weight"], dtype, transpose=True)
            save_weight(file, state[mlp + "fc2.bias"], dtype)

        save_weight(file, state[VISION_PREFIX + "post_layernorm.weight"], dtype)
        save_weight(file, state[VISION_PREFIX + "post_layernorm.bias"], dtype)

        if include_projector:
            projector = "model.multi_modal_projector."
            if cfg.get("projector_use_layernorm", False):
                save_weight(file, state[projector + "layer_norm.weight"], dtype)
                save_weight(file, state[projector + "layer_norm.bias"], dtype)
            save_weight(file, state[projector + "linear_1.weight"], dtype, transpose=True)
            if cfg.get("projector_bias", True):
                save_weight(file, state[projector + "linear_1.bias"], dtype)
            save_weight(file, state[projector + "linear_2.weight"], dtype, transpose=True)
            if cfg.get("projector_bias", True):
                save_weight(file, state[projector + "linear_2.bias"], dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model directory or model.safetensors path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["float32"], default="float32")
    parser.add_argument(
        "--include-projector",
        action="store_true",
        help="Append LFM2-VL multi_modal_projector weights after the vision tower.",
    )
    args = parser.parse_args()

    convert(args.model, args.output, np.float32, args.include_projector)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
