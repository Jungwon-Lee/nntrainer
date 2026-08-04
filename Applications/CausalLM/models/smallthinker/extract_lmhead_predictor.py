#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract SmallThinker's LM-head predictor into NNTrainer FP32 row order."""

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract fc1/fc2 weights from SmallThinker's model_lm_head.pt. "
            "The output contains fc1 [unit, hidden] followed by "
            "fc2 [vocab, unit], both contiguous FP32."
        )
    )
    parser.add_argument("predictor", type=Path, help="model_lm_head.pt path")
    parser.add_argument("output", type=Path, help="output raw FP32 file")
    return parser.parse_args()


def require_matrix(state: dict, name: str) -> torch.Tensor:
    if name not in state:
        raise KeyError(f"predictor does not contain {name}")

    value = state[name]
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional tensor")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def main() -> None:
    args = parse_args()
    state = torch.load(args.predictor, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("predictor must contain a state dictionary")

    fc1 = require_matrix(state, "fc1.weight")
    fc2 = require_matrix(state, "fc2.weight")
    if fc1.shape[0] != fc2.shape[1]:
        raise ValueError(
            "predictor dimension mismatch: "
            f"fc1={tuple(fc1.shape)}, fc2={tuple(fc2.shape)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        output.write(fc1.numpy().tobytes(order="C"))
        output.write(fc2.numpy().tobytes(order="C"))

    print(f"fc1.weight: {tuple(fc1.shape)}")
    print(f"fc2.weight: {tuple(fc2.shape)}")
    print(f"predictor_unit: {fc1.shape[0]}")
    print(f"wrote: {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
