#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

"""Shared streaming and Q4_0 helpers for Qwen3.5-family converters."""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


QK4_0 = 32
Q4_0_BLOCK_BYTES = 18


class ShardedTensorStore:
    """Read slices from a local sharded safetensors checkpoint."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        single_path = self.model_dir / "model.safetensors"
        if index_path.exists():
            with index_path.open(encoding="utf-8") as index_file:
                self.weight_map = json.load(index_file)["weight_map"]
        elif single_path.exists():
            with safe_open(single_path, framework="pt", device="cpu") as src:
                self.weight_map = {name: single_path.name for name in src.keys()}
        else:
            raise FileNotFoundError(
                "model.safetensors.index.json or model.safetensors is required"
            )

        self._shard_name = None
        self._context = None
        self._reader = None

    def __contains__(self, name):
        return name in self.weight_map

    def _open_for(self, name):
        if name not in self.weight_map:
            raise KeyError(f"checkpoint tensor not found: {name}")
        shard_name = self.weight_map[name]
        if shard_name != self._shard_name:
            self.close()
            self._context = safe_open(
                self.model_dir / shard_name, framework="pt", device="cpu"
            )
            self._reader = self._context.__enter__()
            self._shard_name = shard_name
        return self._reader

    def shape(self, name):
        return tuple(self._open_for(name).get_slice(name).get_shape())

    def read(self, name, selection=None):
        tensor_slice = self._open_for(name).get_slice(name)
        if selection is None:
            selection = tuple(slice(None) for _ in tensor_slice.get_shape())
        return tensor_slice[selection]

    def close(self):
        if self._context is not None:
            self._context.__exit__(None, None, None)
        self._shard_name = None
        self._context = None
        self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def as_fp32(tensor):
    """Return a contiguous CPU float32 numpy array."""
    if isinstance(tensor, np.ndarray):
        return np.ascontiguousarray(tensor, dtype=np.float32)
    return tensor.detach().to(device="cpu", dtype=torch.float32).numpy()


def quantize_q4_0(array):
    """Quantize a row-major FP32 matrix to canonical GGML Q4_0 blocks."""
    matrix = np.asarray(array, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] % QK4_0 != 0:
        raise ValueError(f"Q4_0 matrix must be 2-D with K % 32 == 0: {matrix.shape}")

    blocks = matrix.reshape(-1, QK4_0)
    max_indices = np.argmax(np.abs(blocks), axis=1)
    signed_max = blocks[np.arange(blocks.shape[0]), max_indices]
    scale = signed_max / -8.0
    inverse_scale = np.zeros_like(scale)
    np.divide(1.0, scale, out=inverse_scale, where=scale != 0.0)

    quant = np.trunc(blocks * inverse_scale[:, None] + 8.5)
    quant = np.clip(quant, 0, 15).astype(np.uint8)
    packed = quant[:, :16] | (quant[:, 16:] << 4)
    scale_bytes = scale.astype("<f2").view(np.uint8).reshape(-1, 2)
    return np.concatenate((scale_bytes, packed), axis=1).tobytes()


def repack_q4_0(canonical, rows, columns, interleave):
    """Repack canonical blocks to NNTrainer q4_0x4/q4_0x8 layout."""
    if rows % interleave != 0 or columns % QK4_0 != 0:
        raise ValueError(
            f"Q4_0 repack needs N % {interleave} == 0 and K % 32 == 0, "
            f"got N={rows}, K={columns}"
        )

    block_count = columns // QK4_0
    source = np.frombuffer(canonical, dtype=np.uint8).reshape(
        rows, block_count, Q4_0_BLOCK_BYTES
    )
    scales = source[:, :, :2]
    quants = source[:, :, 2:]
    group_count = rows // interleave

    scale_groups = scales.reshape(group_count, interleave, block_count, 2)
    scale_groups = scale_groups.transpose(0, 2, 1, 3).reshape(
        group_count, block_count, 2 * interleave
    )
    quant_groups = quants.reshape(group_count, interleave, block_count, 2, 8)
    quant_groups = quant_groups.transpose(0, 2, 3, 1, 4).reshape(
        group_count, block_count, 16 * interleave
    )
    quant_groups ^= 0x88
    return np.concatenate((scale_groups, quant_groups), axis=2).tobytes()


class ModelWriter:
    """Write tensors in graph allocation order."""

    def __init__(self, output, store, interleave, chunk_rows):
        self.output = output
        self.store = store
        self.interleave = interleave
        self.chunk_rows = max(interleave, chunk_rows // interleave * interleave)

    def write_fp32(self, label, tensor, add_one=False):
        array = as_fp32(tensor)
        if add_one:
            array = array + 1.0
        print(f"FP32 {label}: {tuple(array.shape)}")
        self.output.write(np.asarray(array, dtype="<f4").tobytes())

    def write_fp32_weight(self, name, add_one=False):
        self.write_fp32(name, self.store.read(name), add_one)

    def write_q4_rows(self, label, rows, columns, read_rows, repack=True):
        if rows % self.interleave != 0 or columns % QK4_0 != 0:
            raise ValueError(
                f"{label}: Q4_0 shape ({rows}, {columns}) is incompatible "
                f"with interleave={self.interleave}"
            )
        print(f"Q4_0 {label}: ({rows}, {columns})")
        for start in range(0, rows, self.chunk_rows):
            end = min(rows, start + self.chunk_rows)
            array = as_fp32(read_rows(start, end))
            expected = (end - start, columns)
            if array.shape != expected:
                raise ValueError(f"{label}[{start}:{end}] {array.shape} != {expected}")
            canonical = quantize_q4_0(array)
            if repack:
                canonical = repack_q4_0(
                    canonical, end - start, columns, self.interleave
                )
            self.output.write(canonical)

    def write_q4_weight(self, name, repack=True):
        shape = self.store.shape(name)
        if len(shape) != 2:
            raise ValueError(f"{name}: expected a matrix, got {shape}")
        self.write_q4_rows(
            name,
            shape[0],
            shape[1],
            lambda start, end: self.store.read(
                name, (slice(start, end), slice(None))
            ),
            repack,
        )

    def write_q4_expert(self, name, expert, rows, columns):
        shape = self.store.shape(name)
        if shape[0] <= expert or tuple(shape[1:]) != (rows, columns):
            raise ValueError(
                f"{name}: expected [experts, {rows}, {columns}], got {shape}"
            )
        self.write_q4_rows(
            f"{name}[{expert}]",
            rows,
            columns,
            lambda start, end: self.store.read(
                name, (expert, slice(start, end), slice(None))
            ),
        )


def detect_text_prefix(store):
    """Find the text-backbone prefix used by text-only or multimodal weights."""
    for prefix in ("model.language_model", "model"):
        if f"{prefix}.embed_tokens.weight" in store:
            return prefix
    raise KeyError("could not find the Qwen text-backbone prefix")


def write_attention(writer, prefix, text_config):
    """Write gated full-attention weights."""
    hidden = text_config["hidden_size"]
    heads = text_config["num_attention_heads"]
    head_dim = text_config["head_dim"]
    q_name = f"{prefix}.self_attn.q_proj.weight"
    expected_q_shape = (2 * heads * head_dim, hidden)
    if writer.store.shape(q_name) != expected_q_shape:
        raise ValueError(
            f"{q_name}: {writer.store.shape(q_name)} != {expected_q_shape}"
        )

    # HF stores [head0 query|gate, head1 query|gate, ...]. NNTrainer's split
    # layer expects [all queries | all gates].
    for half, half_name in ((0, "query"), (1, "gate")):
        for head in range(heads):
            source_start = head * 2 * head_dim + half * head_dim
            writer.write_q4_rows(
                f"{q_name}:{half_name}:head{head}",
                head_dim,
                hidden,
                lambda start, end, base=source_start: writer.store.read(
                    q_name, (slice(base + start, base + end), slice(None))
                ),
            )

    writer.write_fp32_weight(f"{prefix}.self_attn.q_norm.weight", add_one=True)
    writer.write_q4_weight(f"{prefix}.self_attn.k_proj.weight")
    writer.write_fp32_weight(f"{prefix}.self_attn.k_norm.weight", add_one=True)
    writer.write_q4_weight(f"{prefix}.self_attn.v_proj.weight")
    writer.write_q4_weight(f"{prefix}.self_attn.o_proj.weight")


def write_delta_net(writer, prefix, text_config):
    """Write Gated DeltaNet projections and FP32 recurrent parameters."""
    writer.write_q4_weight(f"{prefix}.linear_attn.in_proj_qkv.weight")
    writer.write_q4_weight(f"{prefix}.linear_attn.in_proj_z.weight")
    for projection in ("in_proj_b", "in_proj_a"):
        name = f"{prefix}.linear_attn.{projection}.weight"
        writer.write_fp32(f"{name}:transposed", as_fp32(writer.store.read(name)).T)

    conv_name = f"{prefix}.linear_attn.conv1d.weight"
    conv = as_fp32(writer.store.read(conv_name))
    expected = (
        2
        * text_config["linear_num_key_heads"]
        * text_config["linear_key_head_dim"]
        + text_config["linear_num_value_heads"]
        * text_config["linear_value_head_dim"],
        1,
        text_config["linear_conv_kernel_dim"],
    )
    if conv.shape != expected:
        raise ValueError(f"{conv_name}: {conv.shape} != {expected}")
    writer.write_fp32(conv_name, conv[:, 0, ::-1].T.copy())
    writer.write_fp32_weight(f"{prefix}.linear_attn.dt_bias")
    writer.write_fp32_weight(f"{prefix}.linear_attn.A_log")
    writer.write_fp32_weight(f"{prefix}.linear_attn.norm.weight")
    writer.write_q4_weight(f"{prefix}.linear_attn.out_proj.weight")


def copy_runtime_files(model_dir, output_dir):
    """Copy tokenizer and model metadata required by the CausalLM runner."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        source = model_dir / name
        destination = output_dir / name
        if source.exists() and source.resolve() != destination.resolve():
            shutil.copy2(source, destination)


def write_nntr_config(output_dir, model_file, args):
    """Write a portable Q4_0 runtime configuration."""
    nntr_config = {
        "model_type": "CausalLM",
        "model_tensor_type": "Q4_0-FP32",
        "model_file_name": model_file.name,
        "fc_layer_dtype": "Q4_0",
        "embedding_dtype": "Q4_0",
        "lmhead_dtype": "Q4_0",
        "bad_word_ids": [],
        "batch_size": 1,
        "fsu": False,
        "init_seq_len": args.init_seq_len,
        "max_seq_len": args.max_seq_len,
        "num_to_generate": args.num_to_generate,
        "tokenizer_file": "tokenizer.json",
        "sample_input": (
            "<|im_start|>user\nGive me a short introduction to large language "
            "models.<|im_end|>\n<|im_start|>assistant\n"
        ),
    }
    with (output_dir / "nntr_config.json").open("w", encoding="utf-8") as file:
        json.dump(nntr_config, file, indent=2)
        file.write("\n")
