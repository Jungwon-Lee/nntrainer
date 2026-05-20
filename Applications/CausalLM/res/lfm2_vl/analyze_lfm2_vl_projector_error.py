#!/usr/bin/env python3
"""Split LFM2-VL projector error into vision and C++ projector components."""

import argparse

import numpy as np
import torch
from transformers import AutoModelForImageTextToText


def stats(name, lhs, rhs):
    diff = np.abs(lhs - rhs)
    idx = np.unravel_index(diff.argmax(), diff.shape)
    print(
        f"{name}: max_abs={diff.max():.8g}, mean_abs={diff.mean():.8g}, "
        f"p99={np.quantile(diff, 0.99):.8g}, idx={idx}, "
        f"lhs={lhs[idx]:.8g}, rhs={rhs[idx]:.8g}"
    )


def project(model, hidden_states, spatial_shapes, pixel_attention_mask):
    projector = model.model.multi_modal_projector
    lengths = pixel_attention_mask.sum(dim=1)
    features = []
    with torch.no_grad():
        for idx in range(hidden_states.shape[0]):
            height, width = spatial_shapes[idx]
            feature = hidden_states[idx, : lengths[idx]]
            feature = feature.reshape(1, height, width, -1)
            features.append(projector(feature).reshape(-1, projector.linear_2.out_features))
    return torch.cat(features).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--nntr-vision-raw", required=True)
    parser.add_argument("--nntr-projector", required=True)
    args = parser.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, device_map=None
    ).eval()
    reference = np.load(args.reference, allow_pickle=False)
    spatial_shapes = torch.tensor(reference["spatial_shapes"], dtype=torch.long)
    pixel_attention_mask = torch.tensor(reference["pixel_attention_mask"])

    ref_vision = torch.tensor(reference["vision_last_hidden_state"])
    nntr_vision = np.fromfile(args.nntr_vision_raw, dtype=np.float32)
    nntr_vision = nntr_vision.reshape(ref_vision.shape)
    nntr_vision = torch.tensor(nntr_vision)

    py_from_ref = project(model, ref_vision, spatial_shapes, pixel_attention_mask)
    py_from_nntr = project(model, nntr_vision, spatial_shapes, pixel_attention_mask)
    nntr_projector = np.load(args.nntr_projector, allow_pickle=False)["image_features"]

    stats("reference_dump_vs_hf_projector(reference_vision)", reference["image_features"], py_from_ref)
    stats("hf_projector(reference_vision)_vs_hf_projector(nntr_vision)", py_from_ref, py_from_nntr)
    stats("hf_projector(nntr_vision)_vs_nntr_projector(nntr_vision)", py_from_nntr, nntr_projector)
    stats("reference_dump_vs_nntr_projector", reference["image_features"], nntr_projector)


if __name__ == "__main__":
    main()
