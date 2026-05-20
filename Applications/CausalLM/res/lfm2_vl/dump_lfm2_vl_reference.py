#!/usr/bin/env python3
"""Dump Hugging Face LFM2-VL intermediate tensors for staged validation."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def require_pillow():
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Pillow is required to load or generate validation images. "
            "Install it in the reference environment, for example: "
            "`conda install -n nntrainer pillow`."
        ) from error
    return Image


def make_test_image(width=384, height=256):
    """Create a deterministic RGB image with enough structure for resize tests."""
    Image = require_pillow()
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = (x + 3 * y) % 256
    rgb[..., 1] = (2 * x + y) % 256
    rgb[..., 2] = (5 * x + 7 * y) % 256
    return Image.fromarray(rgb, "RGB")


def tensor_to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32 if value.is_floating_point() else value.dtype).numpy()
    return np.asarray(value)


def get_input_embeddings(model):
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    if hasattr(model, "model") and hasattr(model.model, "get_input_embeddings"):
        return model.model.get_input_embeddings()
    raise AttributeError("Could not find input embedding module")


def compute_image_features(model, inputs):
    if not all(key in inputs for key in ("pixel_values", "spatial_shapes", "pixel_attention_mask")):
        return None, None

    target = model.model if hasattr(model, "model") else model
    if not hasattr(target, "get_image_features"):
        return None, None

    image_outputs = target.get_image_features(
        pixel_values=inputs["pixel_values"],
        spatial_shapes=inputs["spatial_shapes"],
        pixel_attention_mask=inputs["pixel_attention_mask"],
        return_dict=True,
    )

    vision_last_hidden_state = getattr(image_outputs, "last_hidden_state", None)
    pooler_output = getattr(image_outputs, "pooler_output", None)
    if isinstance(pooler_output, (list, tuple)):
        image_features = torch.cat([item.reshape(-1, item.shape[-1]) for item in pooler_output], dim=0)
    else:
        image_features = pooler_output

    return vision_last_hidden_state, image_features


def merge_image_features(model, input_ids, inputs_embeds, image_features):
    if image_features is None:
        return None

    config = getattr(model, "config", None)
    image_token_id = getattr(config, "image_token_id", None)
    if image_token_id is None and hasattr(model, "model"):
        image_token_id = getattr(model.model.config, "image_token_id", None)
    if image_token_id is None:
        raise AttributeError("Could not find image_token_id in model config")

    mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
    expected = int(mask.sum().item())
    actual = int(image_features.numel())
    if expected != actual:
        raise RuntimeError(
            f"Image features and image tokens do not match: "
            f"token slots={expected}, feature values={actual}"
        )
    return inputs_embeds.masked_scatter(mask, image_features.to(inputs_embeds.device, inputs_embeds.dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local model directory")
    parser.add_argument("--image", help="Optional image path. A deterministic test image is used if omitted.")
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map[args.dtype]

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    model.eval()

    Image = require_pillow()
    image = Image.open(args.image).convert("RGB") if args.image else make_test_image()
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

    with torch.no_grad():
        input_embeddings = get_input_embeddings(model)
        inputs_embeds = input_embeddings(inputs["input_ids"])
        vision_last_hidden_state, image_features = compute_image_features(model, inputs)
        merged_embeds = merge_image_features(model, inputs["input_ids"], inputs_embeds, image_features)

        outputs = model(**inputs, use_cache=False, logits_to_keep=args.max_new_tokens)
        logits = outputs.logits

    dump = {}
    metadata = {
        "model": args.model,
        "prompt": args.prompt,
        "dtype": args.dtype,
        "device": args.device,
        "image_size": list(image.size),
    }
    dump["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))

    for key, value in inputs.items():
        array = tensor_to_numpy(value)
        if array is not None:
            dump[key] = array

    dump["inputs_embeds"] = tensor_to_numpy(inputs_embeds)
    if vision_last_hidden_state is not None:
        dump["vision_last_hidden_state"] = tensor_to_numpy(vision_last_hidden_state)
    if image_features is not None:
        dump["image_features"] = tensor_to_numpy(image_features)
    if merged_embeds is not None:
        dump["inputs_embeds_after_image_merge"] = tensor_to_numpy(merged_embeds)
    dump["prefill_logits"] = tensor_to_numpy(logits)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **dump)

    print(f"Wrote {output_path}")
    for key in sorted(dump):
        if key == "metadata_json":
            print(f"{key}: {dump[key].item()}")
        else:
            print(f"{key}: shape={dump[key].shape}, dtype={dump[key].dtype}")


if __name__ == "__main__":
    main()
