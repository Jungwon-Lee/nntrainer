#!/usr/bin/env python3
"""Dump a local LFM2-VL processor implementation for parity checks.

This intentionally does not call Hugging Face's Lfm2VlProcessor or
Lfm2VlImageProcessor. It mirrors the preprocessing contract needed by the
nntrainer port: chat-template text expansion, SigLIP2 NaFlex image resize,
patchification, spatial-shape reporting, and padding masks.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2.functional as tvF
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer


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
    Image = require_pillow()
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = (x + 3 * y) % 256
    rgb[..., 1] = (2 * x + y) % 256
    rgb[..., 2] = (5 * x + 7 * y) % 256
    return Image.fromarray(rgb, "RGB")


def round_by_factor(number, factor):
    return round(number / factor) * factor


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio

    return best_ratio


def target_ratios(min_tiles, max_tiles):
    ratios = [
        (w, h)
        for n in range(min_tiles, max_tiles + 1)
        for w in range(1, n + 1)
        for h in range(1, n + 1)
        if min_tiles <= w * h <= max_tiles
    ]
    return sorted(set(ratios), key=lambda item: item[0] * item[1])


def smart_resize(height, width, downsample_factor, min_image_tokens, max_image_tokens, encoder_patch_size):
    total_factor = encoder_patch_size * downsample_factor
    min_pixels = min_image_tokens * encoder_patch_size**2 * downsample_factor**2
    max_pixels = max_image_tokens * encoder_patch_size**2 * downsample_factor**2

    h_bar = max(total_factor, round_by_factor(height, total_factor))
    w_bar = max(total_factor, round_by_factor(width, total_factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(total_factor, math.floor(height / beta / total_factor) * total_factor)
        w_bar = max(total_factor, math.floor(width / beta / total_factor) * total_factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / total_factor) * total_factor
        w_bar = math.ceil(width * beta / total_factor) * total_factor

    return w_bar, h_bar


def is_image_too_large(height, width, max_image_tokens, encoder_patch_size, downsample_factor, max_pixels_tolerance):
    total_factor = encoder_patch_size * downsample_factor
    h_bar = max(encoder_patch_size, round_by_factor(height, total_factor))
    w_bar = max(encoder_patch_size, round_by_factor(width, total_factor))
    return h_bar * w_bar > max_image_tokens * encoder_patch_size**2 * downsample_factor**2 * max_pixels_tolerance


def split_to_tiles(image, num_tiles_height, num_tiles_width):
    _, channels, height, width = image.shape
    tile_h = height // num_tiles_height
    tile_w = width // num_tiles_width
    tiles = []
    for row in range(num_tiles_height):
        for col in range(num_tiles_width):
            y0 = row * tile_h
            x0 = col * tile_w
            tiles.append(image[:, :, y0 : y0 + tile_h, x0 : x0 + tile_w])
    return tiles


def resize_and_split(image, cfg):
    _, _, height, width = image.shape
    min_tiles = cfg["min_tiles"] if cfg["do_image_splitting"] else 1
    max_tiles = cfg["max_tiles"] if cfg["do_image_splitting"] else 1
    new_width, new_height = smart_resize(
        height,
        width,
        cfg["downsample_factor"],
        cfg["min_image_tokens"],
        cfg["max_image_tokens"],
        cfg["encoder_patch_size"],
    )

    large = is_image_too_large(
        height,
        width,
        cfg["max_image_tokens"],
        cfg["encoder_patch_size"],
        cfg["downsample_factor"],
        cfg["max_pixels_tolerance"],
    )

    if large and cfg["do_image_splitting"]:
        ratios = target_ratios(min_tiles, max_tiles)
        grid_width, grid_height = find_closest_aspect_ratio(
            width / height, ratios, width, height, cfg["tile_size"]
        )
        resized = tvF.resize(
            image,
            [cfg["tile_size"] * grid_height, cfg["tile_size"] * grid_width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        crops = split_to_tiles(resized, grid_height, grid_width)
        if cfg["use_thumbnail"] and grid_width * grid_height != 1:
            thumbnail = tvF.resize(
                image,
                [new_height, new_width],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            crops.append(thumbnail)
        return crops, grid_height, grid_width, [new_height, new_width]

    resized = tvF.resize(
        image,
        [new_height, new_width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return [resized], 1, 1, [new_height, new_width]


def convert_image_to_patches(images, patch_size):
    batch_size, num_channels, image_height, image_width = images.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patched_image = images.reshape(
        batch_size,
        num_channels,
        num_patches_height,
        patch_size,
        num_patches_width,
        patch_size,
    )
    patched_image = patched_image.permute(0, 2, 4, 3, 5, 1)
    return patched_image.reshape(batch_size, num_patches_height * num_patches_width, -1)


def process_image(image, cfg):
    image_tensor = tvF.pil_to_tensor(image).unsqueeze(0)
    crops, rows, cols, resized_image_size = resize_and_split(image_tensor, cfg)
    mean = torch.tensor(cfg["image_mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(cfg["image_std"], dtype=torch.float32).view(1, 3, 1, 1)
    max_num_patches = max(
        cfg["max_image_tokens"] * cfg["downsample_factor"] ** 2,
        (cfg["tile_size"] // cfg["encoder_patch_size"]) ** 2 if cfg["do_image_splitting"] else 0,
    )

    pixel_values = []
    pixel_attention_mask = []
    spatial_shapes = []
    for crop in crops:
        crop = crop.to(torch.float32) * cfg["rescale_factor"]
        crop = (crop - mean) / std
        _, _, height, width = crop.shape
        patch_h = height // cfg["encoder_patch_size"]
        patch_w = width // cfg["encoder_patch_size"]
        patches = convert_image_to_patches(crop, cfg["encoder_patch_size"]).squeeze(0)

        mask = torch.ones((max_num_patches,), dtype=torch.int32)
        if patches.shape[0] < max_num_patches:
            padding = torch.zeros((max_num_patches - patches.shape[0], patches.shape[1]), dtype=patches.dtype)
            patches = torch.cat([patches, padding], dim=0)
            mask[patches.shape[0] - padding.shape[0] :] = 0

        pixel_values.append(patches)
        pixel_attention_mask.append(mask)
        spatial_shapes.append([patch_h, patch_w])

    return {
        "pixel_values": torch.stack(pixel_values),
        "pixel_attention_mask": torch.stack(pixel_attention_mask),
        "spatial_shapes": torch.tensor(spatial_shapes, dtype=torch.int64),
        "image_rows": rows,
        "image_cols": cols,
        "image_size": resized_image_size,
    }


def tokens_for_image(image_size, encoder_patch_size, downsample_factor):
    image_height, image_width = image_size
    patches_h = math.ceil((image_height // encoder_patch_size) / downsample_factor)
    patches_w = math.ceil((image_width // encoder_patch_size) / downsample_factor)
    return patches_h * patches_w


def tokens_per_tile(tile_size, encoder_patch_size, downsample_factor):
    num_patches = tile_size // encoder_patch_size
    downsampled_patches = math.ceil(num_patches / downsample_factor)
    return downsampled_patches * downsampled_patches


def build_image_tokens(rows, cols, image_size, cfg, token_cfg):
    pieces = []
    if cfg["use_image_special_tokens"]:
        pieces.append(token_cfg["image_start_token"])

    multi_tile = rows > 1 or cols > 1
    if multi_tile:
        per_tile = tokens_per_tile(cfg["tile_size"], cfg["encoder_patch_size"], cfg["downsample_factor"])
        for row in range(rows):
            for col in range(cols):
                if cfg["use_image_special_tokens"]:
                    pieces.append(f"<|img_row_{row + 1}_col_{col + 1}|>")
                pieces.append(token_cfg["image_token"] * per_tile)
        if cfg["use_thumbnail"]:
            if cfg["use_image_special_tokens"]:
                pieces.append(token_cfg["image_thumbnail_token"])
            pieces.append(
                token_cfg["image_token"]
                * tokens_for_image(image_size, cfg["encoder_patch_size"], cfg["downsample_factor"])
            )
    else:
        pieces.append(
            token_cfg["image_token"]
            * tokens_for_image(image_size, cfg["encoder_patch_size"], cfg["downsample_factor"])
        )

    if cfg["use_image_special_tokens"]:
        pieces.append(token_cfg["image_end_token"])
    return "".join(pieces)


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_processor_config(model_path):
    config_path = Path(model_path) / "processor_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as config_file:
            processor_config = json.load(config_file)
        image_cfg = processor_config["image_processor"]
    else:
        image_cfg = {
            "do_image_splitting": True,
            "downsample_factor": 2,
            "encoder_patch_size": 16,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "max_image_tokens": 256,
            "max_pixels_tolerance": 2.0,
            "max_tiles": 10,
            "min_image_tokens": 64,
            "min_tiles": 2,
            "rescale_factor": 1 / 255,
            "tile_size": 512,
            "use_thumbnail": True,
        }
    image_cfg.setdefault("use_image_special_tokens", True)
    return image_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local model directory")
    parser.add_argument("--image", help="Optional image path. A deterministic test image is used if omitted.")
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--raw-output-dir",
        help="Optional directory for C++ vision runner inputs: pixel_values.f32, "
        "pixel_attention_mask.f32, spatial_shapes.f32.",
    )
    args = parser.parse_args()

    Image = require_pillow()
    image = Image.open(args.image).convert("RGB") if args.image else make_test_image()
    cfg = load_processor_config(args.model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token_cfg = {
        "image_token": getattr(tokenizer, "image_token", "<image>"),
        "image_start_token": getattr(tokenizer, "image_start_token", "<|image_start|>"),
        "image_end_token": getattr(tokenizer, "image_end_token", "<|image_end|>"),
        "image_thumbnail_token": getattr(tokenizer, "image_thumbnail_token", "<|img_thumbnail|>"),
    }

    image_inputs = process_image(image, cfg)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    image_tokens = build_image_tokens(
        image_inputs.pop("image_rows"),
        image_inputs.pop("image_cols"),
        image_inputs.pop("image_size"),
        cfg,
        token_cfg,
    )
    prompt = prompt.replace(token_cfg["image_token"], image_tokens, 1)
    text_inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")

    dump = {}
    dump["metadata_json"] = np.asarray(
        json.dumps(
            {
                "model": args.model,
                "prompt": args.prompt,
                "image_size": list(image.size),
                "processor": "local_lfm2_vl_candidate",
            },
            sort_keys=True,
        )
    )
    for key, value in image_inputs.items():
        dump[key] = tensor_to_numpy(value)
    for key, value in text_inputs.items():
        dump[key] = tensor_to_numpy(value)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **dump)

    if args.raw_output_dir:
        raw_dir = Path(args.raw_output_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        dump["pixel_values"].astype(np.float32).tofile(raw_dir / "pixel_values.f32")
        dump["pixel_attention_mask"].astype(np.float32).tofile(raw_dir / "pixel_attention_mask.f32")
        dump["spatial_shapes"].astype(np.float32).tofile(raw_dir / "spatial_shapes.f32")
        dump["input_ids"].astype(np.float32).tofile(raw_dir / "input_ids.f32")
        dump["attention_mask"].astype(np.float32).tofile(raw_dir / "attention_mask.f32")

    print(f"Wrote {output_path}")
    for key in sorted(dump):
        if key == "metadata_json":
            print(f"{key}: {dump[key].item()}")
        else:
            print(f"{key}: shape={dump[key].shape}, dtype={dump[key].dtype}")


if __name__ == "__main__":
    main()
