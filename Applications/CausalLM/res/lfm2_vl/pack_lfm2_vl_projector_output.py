#!/usr/bin/env python3
"""Pack nntrainer LFM2-VL projector raw output into HF image_features shape."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-image-tokens", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--downsample-factor", type=int, default=2)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    values = np.fromfile(input_dir / "image_features.f32", dtype=np.float32)
    expected = args.batch_size * args.max_image_tokens * args.hidden_size
    if values.size != expected:
        raise ValueError(
            f"Unexpected image_features value count: "
            f"got {values.size}, expected {expected}"
        )

    full = values.reshape(args.batch_size, args.max_image_tokens, args.hidden_size)
    spatial_shapes = np.fromfile(input_dir / "spatial_shapes.f32", dtype=np.float32)
    spatial_shapes = spatial_shapes.reshape(args.batch_size, 2).astype(np.int64)

    features = []
    for batch_idx, (height, width) in enumerate(spatial_shapes):
        token_count = (height // args.downsample_factor) * (
            width // args.downsample_factor
        )
        features.append(full[batch_idx, :token_count])

    image_features = np.concatenate(features, axis=0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, image_features=image_features)
    print(f"Wrote {output_path}")
    print(f"image_features: shape={image_features.shape}, dtype={image_features.dtype}")


if __name__ == "__main__":
    main()
