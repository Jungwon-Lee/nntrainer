#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

"""Stream Qwen3.6-35B-A3B safetensors into an NNTrainer Q4_0 model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

COMMON_DIR = Path(__file__).resolve().parents[2] / "qwen3_5"
sys.path.insert(0, str(COMMON_DIR))

from converter_common import (  # noqa: E402
    ModelWriter,
    ShardedTensorStore,
    as_fp32,
    copy_runtime_files,
    detect_text_prefix,
    write_attention,
    write_delta_net,
    write_nntr_config,
)


def _text_config(config):
    return getattr(config, "text_config", config)


def save_qwen3_5_moe_for_nntrainer(params, config, dtype, output):
    """Write an in-memory HF MoE model in NNTrainer FP32 allocation order."""
    config = _text_config(config)
    layer_types = list(config.layer_types)
    heads = config.num_attention_heads
    head_dim = config.head_dim

    def save(tensor, transpose=False, add_one=False):
        tensor = tensor.detach().cpu().float()
        if transpose:
            tensor = tensor.transpose(0, 1)
        if add_one:
            tensor = tensor + 1.0
        np.asarray(tensor.contiguous().numpy(), dtype=dtype).tofile(output)

    def save_attention_layer(prefix):
        q = params[f"{prefix}.self_attn.q_proj.weight"]
        hidden = q.shape[1]
        q = q.reshape(heads, 2, head_dim, hidden)
        q_gate = np.concatenate(
            (q[:, 0].reshape(-1, hidden), q[:, 1].reshape(-1, hidden)), axis=0
        )
        np.asarray(q_gate.T, dtype=dtype).tofile(output)
        save(params[f"{prefix}.self_attn.q_norm.weight"], add_one=True)
        save(params[f"{prefix}.self_attn.k_proj.weight"], transpose=True)
        save(params[f"{prefix}.self_attn.k_norm.weight"], add_one=True)
        save(params[f"{prefix}.self_attn.v_proj.weight"], transpose=True)
        save(params[f"{prefix}.self_attn.o_proj.weight"], transpose=True)

    def save_delta_layer(prefix):
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
            save(params[f"{prefix}.linear_attn.{name}.weight"], transpose=True)
        conv = params[f"{prefix}.linear_attn.conv1d.weight"]
        save(conv[:, 0, :].flip(-1).transpose(0, 1))
        save(params[f"{prefix}.linear_attn.dt_bias"])
        save(params[f"{prefix}.linear_attn.A_log"])
        save(params[f"{prefix}.linear_attn.norm.weight"])
        save(params[f"{prefix}.linear_attn.out_proj.weight"], transpose=True)

    def save_moe(prefix):
        save(params[f"{prefix}.mlp.gate.weight"], transpose=True)
        gate_up = params[f"{prefix}.mlp.experts.gate_up_proj"]
        down = params[f"{prefix}.mlp.experts.down_proj"]
        for expert in range(config.num_experts):
            save(gate_up[expert], transpose=True)
            save(down[expert], transpose=True)
        shared_gate = params[f"{prefix}.mlp.shared_expert.gate_proj.weight"]
        shared_up = params[f"{prefix}.mlp.shared_expert.up_proj.weight"]
        shared_gate_up = torch.cat((shared_gate, shared_up), dim=0)
        save(shared_gate_up, transpose=True)
        save(
            params[f"{prefix}.mlp.shared_expert.down_proj.weight"],
            transpose=True,
        )
        save(params[f"{prefix}.mlp.shared_expert_gate.weight"], transpose=True)

    save(params["model.embed_tokens.weight"])
    for layer_id, layer_type in enumerate(layer_types):
        prefix = f"model.layers.{layer_id}"
        save(params[f"{prefix}.input_layernorm.weight"], add_one=True)
        if layer_type == "linear_attention":
            save_delta_layer(prefix)
        elif layer_type == "full_attention":
            save_attention_layer(prefix)
        else:
            raise ValueError(f"unsupported layer type: {layer_type}")
        save(params[f"{prefix}.post_attention_layernorm.weight"], add_one=True)
        save_moe(prefix)

    save(params["model.norm.weight"], add_one=True)
    if not config.tie_word_embeddings:
        save(params["lm_head.weight"], transpose=True)


def write_moe(writer, prefix, text_config):
    """Write router, routed experts and sigmoid-gated shared expert."""
    hidden = text_config["hidden_size"]
    intermediate = text_config["moe_intermediate_size"]
    experts = text_config["num_experts"]

    writer.write_q4_weight(f"{prefix}.mlp.gate.weight")
    gate_up_name = f"{prefix}.mlp.experts.gate_up_proj"
    down_name = f"{prefix}.mlp.experts.down_proj"
    for expert in range(experts):
        writer.write_q4_expert(gate_up_name, expert, 2 * intermediate, hidden)
        writer.write_q4_expert(down_name, expert, hidden, intermediate)

    writer.write_q4_weight(f"{prefix}.mlp.shared_expert.gate_proj.weight")
    writer.write_q4_weight(f"{prefix}.mlp.shared_expert.up_proj.weight")
    writer.write_q4_weight(f"{prefix}.mlp.shared_expert.down_proj.weight")

    shared_gate_name = f"{prefix}.mlp.shared_expert_gate.weight"
    shared_gate = as_fp32(writer.store.read(shared_gate_name))
    if shared_gate.shape != (1, hidden):
        raise ValueError(f"{shared_gate_name}: {shared_gate.shape} != (1, {hidden})")
    writer.write_fp32(shared_gate_name, shared_gate)


def convert(model_dir, output_file, target_isa, chunk_rows):
    with (model_dir / "config.json").open(encoding="utf-8") as config_file:
        root_config = json.load(config_file)
    architecture = root_config.get("architectures", [None])[0]
    if architecture not in (
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForCausalLM",
    ):
        raise ValueError(f"unsupported architecture: {architecture}")
    text_config = root_config.get("text_config", root_config)
    layer_types = text_config["layer_types"]
    if len(layer_types) != text_config["num_hidden_layers"]:
        raise ValueError("layer_types size must match num_hidden_layers")

    interleave = 4 if target_isa == "arm" else 8
    with ShardedTensorStore(model_dir) as store, output_file.open("wb") as output:
        prefix = detect_text_prefix(store)
        writer = ModelWriter(output, store, interleave, chunk_rows)
        writer.write_q4_weight(f"{prefix}.embed_tokens.weight", repack=False)

        for layer_id, layer_type in enumerate(layer_types):
            print(f"layer {layer_id}/{len(layer_types) - 1}: {layer_type}")
            layer_prefix = f"{prefix}.layers.{layer_id}"
            writer.write_fp32_weight(
                f"{layer_prefix}.input_layernorm.weight", add_one=True
            )
            if layer_type == "linear_attention":
                write_delta_net(writer, layer_prefix, text_config)
            elif layer_type == "full_attention":
                write_attention(writer, layer_prefix, text_config)
            else:
                raise ValueError(f"unsupported layer type: {layer_type}")
            writer.write_fp32_weight(
                f"{layer_prefix}.post_attention_layernorm.weight", add_one=True
            )
            write_moe(writer, layer_prefix, text_config)

        writer.write_fp32_weight(f"{prefix}.norm.weight", add_one=True)
        if not text_config.get("tie_word_embeddings", False):
            writer.write_q4_weight("lm_head.weight")
        print(f"wrote {output.tell():,} bytes to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="local HF snapshot directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-isa", choices=("arm", "x86"), default="arm")
    parser.add_argument("--chunk-rows", type=int, default=64)
    parser.add_argument("--init-seq-len", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--num-to-generate", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.init_seq_len > args.max_seq_len:
        raise ValueError("init-seq-len cannot exceed max-seq-len")
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_file = output_dir / f"nntr_qwen3.6-35b-a3b-q4_0-{args.target_isa}.bin"
    if output_file.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output_file}; pass --force")

    output_dir.mkdir(parents=True, exist_ok=True)
    convert(model_dir, output_file, args.target_isa, args.chunk_rows)
    copy_runtime_files(model_dir, output_dir)
    write_nntr_config(output_dir, output_file, args)


if __name__ == "__main__":
    main()
