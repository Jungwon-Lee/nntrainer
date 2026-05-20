#!/usr/bin/env python3
"""Pack nntrainer SigLIP2 vision raw output into an NPZ for comparison."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-patches", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=768)
    args = parser.parse_args()

    input_path = Path(args.input_dir) / "vision_last_hidden_state.f32"
    values = np.fromfile(input_path, dtype=np.float32)
    expected = args.batch_size * args.max_patches * args.hidden_size
    if values.size != expected:
        raise ValueError(
            f"Unexpected value count for {input_path}: "
            f"got {values.size}, expected {expected}"
        )

    array = values.reshape(args.batch_size, args.max_patches, args.hidden_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, vision_last_hidden_state=array)
    print(f"Wrote {output_path}")
    print(f"vision_last_hidden_state: shape={array.shape}, dtype={array.dtype}")


if __name__ == "__main__":
    main()
