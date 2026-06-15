# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file weight_converter.py
# @brief Unified weight conversion script for all CausalLM models.
# @author  Jungwon-Lee <jungone.lee@samsung.com>
#
# This script lives in Applications/CausalLM/converter/.
# Converted weights and config JSON files default to the sibling res/ directory.
#
# Supported models (--model_type):
#   qwen2         Qwen2 (e.g. Qwen/Qwen2-0.5B)
#   qwen3         Qwen3 dense (e.g. Qwen/Qwen3-0.6B, Qwen3-4B)
#   qwen3_moe     Qwen3 MoE (e.g. Qwen/Qwen3-30B-A3B)
#   gemma3        Gemma3 causal LM (config model_type "gemma3" or "gemma3_text")
#   gemma3_emb    Gemma3 SentenceTransformer embedding model
#   gpt_oss       GPT-OSS 20B MoE  (HF model_type may differ; use --model_type)
#   kalm          KaLM-embedding (SentenceTransformer, Qwen2 backbone)
#   deberta_v2    DeBERTa V2 encoder / SentenceTransformer
#   bert          (multilingual tiny) BERT encoder
#   vit           timm ViT  (--model_path is a .safetensors/.bin file, not an HF id)
#
# Outputs are saved to the same directory as --output_name.
# Default output directory is the sibling res/ directory.
# Along with the weight binary, three JSON files are always written:
#   config.json            HuggingFace model config (HF models only)
#   generation_config.json HuggingFace generation config (if available)
#   nntr_config.json       nntrainer runtime config
#
# Usage:
#   # output_name defaults to res/nntr_<model>_<dtype>.bin
#   python weight_converter.py --model_path Qwen/Qwen3-0.6B
#
#   # explicit output path
#   python weight_converter.py --model_path Qwen/Qwen3-0.6B --output_name /tmp/out.bin
#
#   # safetensors output (Qwen3 only)
#   python weight_converter.py --model_path Qwen/Qwen3-0.6B --safetensors
#
#   # timm ViT (model_path is a checkpoint file, not an HF hub id)
#   python weight_converter.py --model_path ./model.safetensors --model_type vit

import argparse
import importlib
import os
import re
import sys

# This script lives in Applications/CausalLM/converter/.
# Make the sibling converter modules importable when run directly as a script.
_CONVERTER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _CONVERTER_DIR)

# Default output directory: the sibling res/ directory.
_DEFAULT_OUTPUT_DIR = os.path.normpath(os.path.join(_CONVERTER_DIR, os.pardir, "res"))

# model_type string -> converter module name
_MODULE_MAP = {
    "qwen2":      "qwen2",
    "qwen3":      "qwen3",
    "qwen3_moe":  "qwen3_moe",
    "gemma3":      "gemma3",
    "gemma3_text": "gemma3",  # text-only Gemma3 (Gemma3ForCausalLM) reports this
    "gemma3_emb":  "gemma3",
    "gpt_oss":    "gpt_oss",
    "kalm":       "qwen2",
    "deberta-v2": "deberta_v2",
    "deberta_v2": "deberta_v2",
    "bert":       "tiny_bert",
    "vit":        "vit",
}

# Extra kwargs forwarded to converter.convert() depending on model_type
_EXTRA_KWARGS = {
    "gemma3_emb": {"is_embedding": True},
    "kalm":       {"is_kalm": True},
}


def _detect_model_type(model_path):
    """Try to read model_type from HuggingFace config."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return config.model_type
    except Exception:
        return None


def _default_output_name(model_path, model_type, dtype, safetensors=False):
    """Derive a default output path under res/<model_name>/."""
    # Use the last path component of model_path and sanitize it.
    basename = os.path.basename(model_path.rstrip("/\\"))
    if not basename:
        basename = model_type
    # lowercase, replace spaces and slashes with hyphens
    basename = re.sub(r"[^a-zA-Z0-9.\-_]", "-", basename).lower()
    dtype_tag = "fp32" if dtype == "float32" else "fp16"
    ext = ".safetensors" if safetensors else ".bin"
    filename = f"nntr_{basename}_{dtype_tag}{ext}"
    # Place each model's outputs in its own res/<model_name>/ subdirectory.
    return os.path.join(_DEFAULT_OUTPUT_DIR, basename, filename)


def main():
    parser = argparse.ArgumentParser(
        description="Unified nntrainer weight converter for CausalLM/res models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model id, local model directory, or checkpoint file (ViT).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help=(
            "Output file path (.bin or .safetensors). "
            "Defaults to res/nntr_<model>_<dtype>.bin. "
            "config.json, generation_config.json, and nntr_config.json are "
            "always written to the same directory."
        ),
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Weight dtype written to the output file.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help=(
            "Override model type detection. "
            "Choices: " + ", ".join(sorted(_MODULE_MAP)) + ". "
            "Required for gpt_oss and vit."
        ),
    )
    parser.add_argument(
        "--safetensors",
        action="store_true",
        help="Save in safetensors format (Qwen3 only).",
    )
    # ViT-specific geometry (used for nntr_config.json)
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="(ViT only) Patch size in pixels.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="(ViT only) Input image size in pixels.",
    )
    args = parser.parse_args()

    # Resolve model_type
    model_type = args.model_type
    if model_type is None:
        model_type = _detect_model_type(args.model_path)
        if model_type is None:
            parser.error(
                "Could not detect model_type automatically. "
                "Please specify --model_type explicitly."
            )
        print(f"Detected model_type: {model_type}")

    model_type = model_type.lower()
    module_name = _MODULE_MAP.get(model_type)
    if module_name is None:
        parser.error(
            f"Unsupported model_type '{model_type}'. "
            "Supported: " + ", ".join(sorted(_MODULE_MAP))
        )

    # Resolve output path
    output_name = args.output_name
    if output_name is None:
        output_name = _default_output_name(
            args.model_path, model_type, args.data_type, safetensors=args.safetensors
        )
        print(f"Output: {output_name}")

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_name))
    os.makedirs(output_dir, exist_ok=True)

    module = importlib.import_module(module_name)
    extra = dict(_EXTRA_KWARGS.get(model_type, {}))
    extra["safetensors"] = args.safetensors
    if model_type == "vit":
        extra["patch_size"] = args.patch_size
        extra["img_size"] = args.img_size

    module.convert(
        model_path=args.model_path,
        output_name=output_name,
        dtype=args.data_type,
        **extra,
    )


if __name__ == "__main__":
    main()
