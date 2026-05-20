#!/usr/bin/env python3
"""Pack nntrainer LFM2-VL embedding-merge raw output into NPZ."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=1024)
    args = parser.parse_args()

    input_path = Path(args.input_dir) / "inputs_embeds_after_image_merge.f32"
    values = np.fromfile(input_path, dtype=np.float32)
    expected = args.batch_size * args.sequence_length * args.hidden_size
    if values.size != expected:
        raise ValueError(
            f"Unexpected merged embedding value count: "
            f"got {values.size}, expected {expected}"
        )

    array = values.reshape(args.batch_size, args.sequence_length, args.hidden_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, inputs_embeds_after_image_merge=array)
    print(f"Wrote {output_path}")
    print(f"inputs_embeds_after_image_merge: shape={array.shape}, dtype={array.dtype}")


if __name__ == "__main__":
    main()
