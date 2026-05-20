#!/usr/bin/env python3
"""Convert LFM2-VL token embedding weights for merge validation."""

import argparse
from pathlib import Path

import numpy as np
import safetensors.torch
import torch


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


def convert(model_path, output_path):
    state = load_state_dict(model_path)
    embedding = state["model.language_model.embed_tokens.weight"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embedding.detach().cpu().to(torch.float32).numpy().astype(np.float32).tofile(
        output_path
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    convert(args.model, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
