# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

import numpy as np
import torch

from utils import save_configs

# ViT consumes an image file rather than text.
SAMPLE_INPUT = "sample.png"


def _load_state_dict(model_path):
    if model_path.endswith(".safetensors"):
        import safetensors.torch
        state_dict = safetensors.torch.load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    if isinstance(state_dict, dict):
        for key in ("state_dict", "model"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def _save(file, weight, dtype, transpose=False):
    arr = weight.detach().cpu().numpy() if not isinstance(weight, np.ndarray) else weight
    if transpose and arr.ndim >= 2:
        arr = arr.T
    arr.astype(dtype).tofile(file)


def convert(model_path, output_name, dtype, **kwargs):
    """Convert timm ViT weights to nntrainer binary format.

    model_path: path to a .safetensors or .bin checkpoint file (not an HF hub ID).
    """
    np_dtype = np.float16 if dtype == "float16" else np.float32

    print(f"Loading model from: {model_path}")
    state_dict = _load_state_dict(model_path)

    # Infer number of transformer blocks instead of hard-coding 12.
    num_layers = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith("norm1.weight"))
    if num_layers == 0:
        raise ValueError("Could not detect transformer blocks in state_dict. Check model_path.")

    dim = state_dict["patch_embed.proj.bias"].shape[0]

    print(f"Converting to: {output_name}  (layers={num_layers}, dim={dim})")

    with open(output_name, "wb") as f:
        # 1. Patch embedding (Conv2D — no transpose)
        _save(f, state_dict["patch_embed.proj.weight"], np_dtype, transpose=False)
        if "patch_embed.proj.bias" in state_dict:
            _save(f, state_dict["patch_embed.proj.bias"], np_dtype)

        # 2. Position embedding — reshape to [1, 1, seq_len, dim]
        pos_embed = state_dict["pos_embed"].unsqueeze(1)
        _save(f, pos_embed, np_dtype, transpose=False)

        # 3. Transformer blocks
        for i in range(num_layers):
            p = f"blocks.{i}."

            _save(f, state_dict[f"{p}norm1.weight"], np_dtype)
            _save(f, state_dict[f"{p}norm1.bias"], np_dtype)

            qkv_w = state_dict[f"{p}attn.qkv.weight"]
            qkv_b = state_dict[f"{p}attn.qkv.bias"]
            for chunk_w, chunk_b in [
                (qkv_w[:dim], qkv_b[:dim]),
                (qkv_w[dim:2*dim], qkv_b[dim:2*dim]),
                (qkv_w[2*dim:], qkv_b[2*dim:]),
            ]:
                _save(f, chunk_w, np_dtype, transpose=True)
                _save(f, chunk_b, np_dtype)

            _save(f, state_dict[f"{p}attn.proj.weight"], np_dtype, transpose=True)
            _save(f, state_dict[f"{p}attn.proj.bias"], np_dtype)

            _save(f, state_dict[f"{p}norm2.weight"], np_dtype)
            _save(f, state_dict[f"{p}norm2.bias"], np_dtype)

            _save(f, state_dict[f"{p}mlp.fc1.weight"], np_dtype, transpose=True)
            _save(f, state_dict[f"{p}mlp.fc1.bias"], np_dtype)
            _save(f, state_dict[f"{p}mlp.fc2.weight"], np_dtype, transpose=True)
            _save(f, state_dict[f"{p}mlp.fc2.bias"], np_dtype)

        # 4. Final normalization
        _save(f, state_dict["norm.weight"], np_dtype)
        _save(f, state_dict["norm.bias"], np_dtype)

        # 5. Attention pool (optional — present when global_pool="map")
        if "attn_pool.latent" in state_dict:
            _save(f, state_dict["attn_pool.latent"], np_dtype, transpose=False)
            _save(f, state_dict["attn_pool.q.weight"], np_dtype, transpose=True)
            _save(f, state_dict["attn_pool.q.bias"], np_dtype)
            _save(f, state_dict["attn_pool.kv.weight"], np_dtype, transpose=True)
            _save(f, state_dict["attn_pool.kv.bias"], np_dtype)
            _save(f, state_dict["attn_pool.proj.weight"], np_dtype, transpose=True)
            _save(f, state_dict["attn_pool.proj.bias"], np_dtype)
            _save(f, state_dict["attn_pool.norm.weight"], np_dtype)
            _save(f, state_dict["attn_pool.norm.bias"], np_dtype)
            _save(f, state_dict["attn_pool.mlp.fc1.weight"], np_dtype, transpose=True)
            _save(f, state_dict["attn_pool.mlp.fc1.bias"], np_dtype)
            _save(f, state_dict["attn_pool.mlp.fc2.weight"], np_dtype, transpose=True)
            _save(f, state_dict["attn_pool.mlp.fc2.bias"], np_dtype)

    print(f"Saved binary: {output_name}")

    patch_size = kwargs.get("patch_size", 16)
    img_size = kwargs.get("img_size", 224)
    num_patches = (img_size // patch_size) ** 2
    nntr_extra = {
        "model_type": "Model",
        "patch_size": patch_size,
        "img_size": img_size,
        "num_patches": num_patches,
        "num_classes": 0,
        "num_to_generate": 0,
        "init_seq_len": num_patches,
        "max_seq_len": num_patches * 5,
        "skip_tokenizer": True,
        "sample_input": SAMPLE_INPUT,
    }
    # ViT loads from a checkpoint file, not an HF model directory — skip config.json / generation_config.json.
    save_configs(output_name, dtype, hf_config=None, model_path=None, nntr_extra=nntr_extra)
