#!/usr/bin/env python3
"""Pack nntrainer LFM2-VL prefill raw output into NPZ."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=115)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument(
        "--key",
        default="prefill_logits",
        choices=["prefill_logits", "hidden", "norm"],
    )
    args = parser.parse_args()

    file_name = {
        "prefill_logits": "prefill_logits.f32",
        "hidden": "prefill_hidden.f32",
        "norm": "prefill_norm.f32",
    }[args.key]
    input_path = Path(args.input_dir) / file_name
    values = np.fromfile(input_path, dtype=np.float32)

    if args.key == "prefill_logits":
        expected = args.batch_size * args.vocab_size
        array = values.reshape(args.batch_size, 1, args.vocab_size)
    else:
        expected = args.batch_size * args.sequence_length * args.hidden_size
        array = values.reshape(args.batch_size, args.sequence_length, args.hidden_size)

    if values.size != expected:
        raise ValueError(
            f"Unexpected {args.key} value count: got {values.size}, expected {expected}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **{args.key: array})
    print(f"Wrote {output_path}")
    print(f"{args.key}: shape={array.shape}, dtype={array.dtype}")


if __name__ == "__main__":
    main()
