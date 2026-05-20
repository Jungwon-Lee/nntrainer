#!/usr/bin/env python3
"""Prepare a staged end-to-end nntrainer LFM2-VL model directory."""

import argparse
import json
from pathlib import Path

import numpy as np

from convert_lfm2_vl_embedding_merge_weights import convert as convert_embedding
from convert_lfm2_vl_prefill_weights import convert as convert_prefill
from convert_lfm2_vl_vision_weights import convert as convert_vision


VISION_BIN = "nntr_siglip2_vision_fp32.bin"
MERGE_BIN = "nntr_lfm2_vl_embedding_merge_fp32.bin"
PREFILL_BIN = "nntr_lfm2_vl_prefill_fp32.bin"
WRAPPER_BIN = "lfm2_vl_e2e_wrapper.bin"


def load_config(model_path):
    model_path = Path(model_path)
    config_path = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-input", required=True)
    parser.add_argument("--sequence-length", type=int, default=115)
    parser.add_argument("--num-to-generate", type=int, default=0)
    parser.add_argument("--max-patches", type=int, default=1024)
    parser.add_argument("--max-image-tokens", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.model)
    cfg["architectures"] = ["Lfm2VlForConditionalGeneration"]
    with open(output_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(cfg, config_file, indent=2)

    nntr_cfg = {
        "model_tensor_type": "FP32-FP32",
        "model_file_name": WRAPPER_BIN,
        "vision_model_file_name": VISION_BIN,
        "embedding_merge_model_file_name": MERGE_BIN,
        "prefill_model_file_name": PREFILL_BIN,
        "decode_embedding_file": str((output_dir / MERGE_BIN).resolve()),
        "model_type": "Model",
        "embedding_dtype": "FP32",
        "fc_layer_dtype": "FP32",
        "batch_size": 1,
        "max_patches": args.max_patches,
        "max_image_tokens": args.max_image_tokens,
        "sample_input": args.sample_input,
        "init_seq_len": args.sequence_length,
        "max_seq_len": args.sequence_length + args.num_to_generate,
        "num_to_generate": args.num_to_generate,
        "skip_tokenizer": True,
    }
    with open(output_dir / "nntr_config.json", "w", encoding="utf-8") as config_file:
        json.dump(nntr_cfg, config_file, indent=2)

    convert_vision(args.model, output_dir / VISION_BIN, np.float32, include_projector=True)
    print(f"Wrote {output_dir / VISION_BIN}")
    convert_embedding(args.model, output_dir / MERGE_BIN)
    print(f"Wrote {output_dir / MERGE_BIN}")
    convert_prefill(args.model, output_dir / PREFILL_BIN)
    print(f"Wrote {output_dir / PREFILL_BIN}")
    print(f"Prepared {output_dir}")


if __name__ == "__main__":
    main()
