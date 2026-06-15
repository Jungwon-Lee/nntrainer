# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

import json
import os
import shutil
import struct

import numpy as np


SAFETENSORS_DTYPE_MAP = {
    "float32": "F32",
    "float16": "F16",
}


def tensor_to_numpy(tensor, dtype, transpose=False, add_one=False):
    """Convert a torch tensor to a contiguous numpy array.

    transpose: permute the first two dims (PyTorch OI -> nntrainer IO).
    add_one:   add 1.0 before casting (Gemma3 RMSNorm convention).
    """
    if transpose:
        tensor = tensor.permute(1, 0)
    arr = tensor.detach().cpu().numpy()
    if add_one:
        arr = arr + 1.0
    return np.ascontiguousarray(arr.astype(dtype))


def write_array(file, arr):
    """Write a numpy array to an open binary file."""
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr.tofile(file)


def save_tensor(file, tensor, dtype, transpose=False, add_one=False):
    """Convert tensor and write to file."""
    write_array(file, tensor_to_numpy(tensor, dtype, transpose=transpose, add_one=add_one))


def has_lora(params, layer_name, proj_name):
    return f"{layer_name}{proj_name}.lora_A.default.weight" in params


def save_lora_or_weight(file, params, layer_name, proj_name, dtype, transpose=True):
    """Save projection weight. If LoRA weights exist, save base + lora_A + lora_B."""
    prefix = f"{layer_name}{proj_name}"
    if has_lora(params, layer_name, proj_name):
        save_tensor(file, params[f"{prefix}.base_layer.weight"], dtype, transpose=transpose)
        save_tensor(file, params[f"{prefix}.lora_A.default.weight"], dtype, transpose=transpose)
        save_tensor(file, params[f"{prefix}.lora_B.default.weight"], dtype, transpose=transpose)
    else:
        save_tensor(file, params[f"{prefix}.weight"], dtype, transpose=transpose)


def get_tie_word_embeddings(config):
    return getattr(config, "tie_word_embeddings", True)


def get_safetensors_output_name(output_name):
    if output_name.endswith(".bin"):
        return output_name[:-4] + ".safetensors"
    if output_name.endswith(".safetensors"):
        return output_name
    return output_name + ".safetensors"


def save_safetensors(weights, output_path, dtype):
    """Write (name, ndarray) pairs to a safetensors file."""
    if dtype not in SAFETENSORS_DTYPE_MAP:
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")

    safetensors_dtype = SAFETENSORS_DTYPE_MAP[dtype]
    offset = 0
    tensor_meta = {}
    raw_buffers = []

    for name, arr in weights:
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        tensor_meta[name] = {
            "dtype": safetensors_dtype,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        raw_buffers.append(arr.tobytes(order="C"))
        offset += nbytes

    header = {"__metadata__": {"format": "pt"}}
    header.update(tensor_meta)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad

    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for buf in raw_buffers:
            f.write(buf)

    print(f"Saved safetensors: {output_path}")
    print(f"Tensor data size: {offset / 1e9:.2f} GB")


# ---------------------------------------------------------------------------
# Config saving helpers
# ---------------------------------------------------------------------------

def make_chat_input(text):
    """Build a chat_input payload for nntr_config.json.

    nntrainer applies the model's chat_template to chat_input at inference time.
    sample_input (plain text) is only used as a fallback when no chat_template
    is available, so causal models should provide both.
    """
    return {"messages": [{"role": "user", "content": text}]}


def _dtype_tag(dtype):
    return "FP32" if dtype == "float32" else "FP16"


def _default_nntr_config(output_name, dtype, hf_config=None):
    """Build the default nntr_config dict from common fields."""
    tag = _dtype_tag(dtype)
    max_seq = getattr(hf_config, "max_position_embeddings", 2048) if hf_config else 2048
    output_dir = os.path.dirname(os.path.abspath(output_name))
    return {
        "model_type": "CausalLM",
        "model_tensor_type": f"{tag}-{tag}",
        "model_file_name": os.path.basename(output_name),
        "fc_layer_dtype": tag,
        "embedding_dtype": tag,
        "lora_rank": 0,
        "lora_alpha": 0,
        "lora_target": [],
        "bad_word_ids": [],
        "fsu": False,
        "fsu_lookahead": 2,
        "num_to_generate": 512,
        "init_seq_len": 1024,
        "max_seq_len": max_seq,
        "batch_size": 1,
        "tokenizer_file": os.path.join(output_dir, "tokenizer.json"),
        "sample_input": "",
    }


def _resolve_local_dir(model_path):
    """Return a local directory for model_path, downloading config-only files if it
    is a HuggingFace hub id. Returns None if it cannot be resolved."""
    if os.path.isdir(model_path):
        return model_path
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(
            model_path,
            allow_patterns=[
                "modules.json",
                "config_sentence_transformers.json",
                "*/config.json",
                "*/sentence_bert_config.json",
            ],
        )
    except Exception as e:
        print(f"Warning: could not fetch SentenceTransformer module configs ({e})")
        return None


def save_st_module_configs(model_path, output_dir):
    """Copy a SentenceTransformer's modules.json and per-module config.json into
    output_dir, preserving the module subdirectory layout.

    The nntrainer runtime reads modules.json and a config.json per module
    (models/sentence_transformer.cpp), so only those small files are copied — not
    the duplicate transformer weights. Returns the copied modules.json path, or
    None if the source has no modules.json.
    """
    src = _resolve_local_dir(model_path)
    if src is None:
        return None

    modules_src = os.path.join(src, "modules.json")
    if not os.path.isfile(modules_src):
        return None

    shutil.copy(modules_src, os.path.join(output_dir, "modules.json"))

    with open(modules_src) as f:
        modules = json.load(f)

    for module in modules:
        rel_path = module.get("path", "")
        if not rel_path:
            continue
        cfg_src = os.path.join(src, rel_path, "config.json")
        if os.path.isfile(cfg_src):
            dst_dir = os.path.join(output_dir, rel_path)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(cfg_src, os.path.join(dst_dir, "config.json"))

    return os.path.join(output_dir, "modules.json")


def save_configs(output_name, dtype, hf_config=None, model_path=None, nntr_extra=None,
                 sentence_transformer=False):
    """Save config.json, generation_config.json, tokenizer files, and nntr_config.json.

    Args:
        output_name: path to the converted weight file (used to derive output dir and filename).
        dtype:       data type string ("float32" or "float16").
        hf_config:   loaded HuggingFace PretrainedConfig object (or None for timm ViT).
        model_path:  HuggingFace model id or local directory. Used to load the
                     generation config and tokenizer (incl. chat_template).
        nntr_extra:  dict of model-specific fields that override the defaults in nntr_config.json.
        sentence_transformer: if True, copy modules.json + per-module config.json and
                     set module_config_path (required for embedding models run as a
                     SentenceTransformer pipeline by the nntrainer runtime).
    """
    output_dir = os.path.dirname(os.path.abspath(output_name))
    os.makedirs(output_dir, exist_ok=True)

    # -- config.json ----------------------------------------------------------
    if hf_config is not None:
        hf_config.save_pretrained(output_dir)
        print(f"Saved config.json: {os.path.join(output_dir, 'config.json')}")

    # -- generation_config.json -----------------------------------------------
    if model_path is not None:
        try:
            from transformers import GenerationConfig
            gen_config = GenerationConfig.from_pretrained(model_path)
            gen_config.save_pretrained(output_dir)
            print(f"Saved generation_config.json: {os.path.join(output_dir, 'generation_config.json')}")
        except Exception:
            pass  # not all models ship a generation_config

    # -- tokenizer (tokenizer.json, tokenizer_config.json, chat_template) ------
    # save_pretrained re-emits the chat_template (embedded in tokenizer_config.json
    # or as a standalone chat_template.jinja), so it follows the model automatically
    # even when the source only ships the template file.
    tokenizer = None
    if model_path is not None:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            tokenizer.save_pretrained(output_dir)
            print(f"Saved tokenizer files to: {output_dir}")
        except Exception as e:
            print(f"Warning: could not save tokenizer ({e})")

    # -- SentenceTransformer module configs (modules.json + per-module config) -
    module_config_path = None
    if sentence_transformer and model_path is not None:
        module_config_path = save_st_module_configs(model_path, output_dir)
        if module_config_path is not None:
            print(f"Saved SentenceTransformer module configs to: {output_dir}")
        else:
            print("Warning: no modules.json found; module_config_path not set")

    # -- nntr_config.json -----------------------------------------------------
    nntr = _default_nntr_config(output_name, dtype, hf_config=hf_config)
    if nntr_extra:
        nntr.update(nntr_extra)
    if module_config_path is not None:
        nntr["module_config_path"] = module_config_path

    # Render sample_input from chat_input using the model's own chat_template, so
    # the fallback prompt is already correctly formatted for this exact model
    # (matches what the runtime produces from chat_input at inference time).
    chat_input = nntr.get("chat_input")
    if tokenizer is not None and chat_input and getattr(tokenizer, "chat_template", None):
        messages = chat_input["messages"] if isinstance(chat_input, dict) else chat_input
        try:
            nntr["sample_input"] = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(f"Warning: could not render sample_input from chat_template ({e})")

    nntr_path = os.path.join(output_dir, "nntr_config.json")
    with open(nntr_path, "w") as f:
        json.dump(nntr, f, indent=4)
    print(f"Saved nntr_config.json: {nntr_path}")
