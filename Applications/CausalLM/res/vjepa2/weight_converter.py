## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
##
## @file weight_converter.py
## @brief Weight conversion for V-JEPA 2.1 ViT-B/16 (video) encoder.
## @author Jijoong Moon <jijoong.moon@samsung.com>
##
## Converts the V-JEPA 2.1 encoder weights into the nntrainer binary format
## expected by the VJEPA2ViT model graph.
##
## Supports two input formats:
##   1. Standalone V-JEPA 2.1 checkpoint (.pt):
##      - Downloaded from https://dl.fbaipublicfiles.com/vjepa2
##      - Contains: {checkpoint_key: {encoder_state_dict}, "predictor": ...}
##      - Keys: patch_embed.proj.weight, blocks.{i}.norm1.weight, ...
##
##   2. Merged VoRA safetensors checkpoint (.safetensors):
##      - Full model checkpoint (vision_tower + merger + language_model)
##      - Keys prefixed with: model.vision_tower.model.
##      - e.g., model.vision_tower.model.patch_embed.proj.weight
##
## The byte order of the output exactly follows the order in which VJEPA2ViT
## creates its weight-bearing layers (see models/vjepa2_vit/vjepa2_vit.cpp):
##
##   patch_embed/proj          (FC: weight, bias)
##   for each of 12 blocks:
##     layer{i}_attention_norm (LN: weight, bias)   <- blocks.{i}.norm1
##     layer{i}_qkv_q          (FC: weight, bias)   <- blocks.{i}.attn.qkv[0:768]
##     layer{i}_qkv_k          (FC: weight, bias)   <- blocks.{i}.attn.qkv[768:1536]
##     layer{i}_qkv_v          (FC: weight, bias)   <- blocks.{i}.attn.qkv[1536:2304]
##     layer{i}_attention_out  (FC: weight, bias)   <- blocks.{i}.attn.proj
##     layer{i}_ffn_norm       (LN: weight, bias)   <- blocks.{i}.norm2
##     layer{i}_ffn_up         (FC: weight, bias)   <- blocks.{i}.mlp.fc1
##     layer{i}_ffn_down       (FC: weight, bias)   <- blocks.{i}.mlp.fc2
##   output_norm               (LN: weight, bias)   <- norms_block.{last}
##
## Notes:
##   - The 3D tubelet patch embedding (Conv3d) is non-overlapping, hence exactly
##     equivalent to a Linear over the flattened tubelet. The Conv3d weight
##     [embed, in_ch, kT, kH, kW] is reshaped to [embed, in_ch*kT*kH*kW] (C
##     order, matching VJEPA2ViT::patchify's (c, kt, kh, kw) layout) and then
##     transposed to nntrainer's [in, out] FC layout.
##   - The (video) modality embedding is a constant added to every token, so it
##     is folded into the patch-embed bias here. The image modality path
##     (patch_embed_img / img_mod_embed) is intentionally not exported.
##   - 3D RoPE carries no weights.

import argparse
import numpy as np
import torch

try:
    from safetensors.torch import load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def load_encoder_state_from_pt(model_path, checkpoint_key):
    """Load the encoder sub state-dict from a standalone .pt checkpoint."""
    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and checkpoint_key in sd:
        sd = sd[checkpoint_key]
    cleaned = {}
    for k, v in sd.items():
        k = k.replace("module.", "").replace("backbone.", "")
        cleaned[k] = v
    return cleaned


def load_encoder_state_from_safetensors(model_path, vision_prefix=None):
    """Load the encoder state-dict from a merged VoRA safetensors checkpoint.

    In the merged checkpoint, vision tower keys are typically prefixed with
    something like 'model.vision_tower.model.' or 'vision_tower.model.'.
    This function strips the prefix to get the bare encoder keys.

    If vision_prefix is not specified, it will be auto-detected by looking
    for 'patch_embed.proj.weight' in the key names.
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError(
            "safetensors package is required for .safetensors input. "
            "Install with: pip install safetensors"
        )

    sd = load_file(model_path)

    # Auto-detect vision prefix if not specified
    if vision_prefix is None:
        # Look for patch_embed.proj.weight in the keys
        for k in sd.keys():
            if "patch_embed.proj.weight" in k:
                # Extract prefix (everything before patch_embed)
                idx = k.index("patch_embed.proj.weight")
                vision_prefix = k[:idx]
                break

        if vision_prefix is None:
            # Print available keys for debugging
            print("Could not auto-detect vision tower prefix. Available keys (first 30):")
            for i, k in enumerate(sorted(sd.keys())):
                if i >= 30:
                    break
                print(f"  {k} {tuple(sd[k].shape)}")
            raise RuntimeError(
                "Could not find vision tower keys in safetensors. "
                "Use --vision_prefix to specify the key prefix."
            )

    print(f"  Using vision_prefix: '{vision_prefix}'")

    # Extract and strip prefix
    encoder_sd = {}
    for k, v in sd.items():
        if k.startswith(vision_prefix):
            bare_key = k[len(vision_prefix):]
            encoder_sd[bare_key] = v if isinstance(v, torch.Tensor) else torch.tensor(v)

    # Verify we got the expected keys
    if "patch_embed.proj.weight" not in encoder_sd:
        raise RuntimeError(
            f"After stripping prefix '{vision_prefix}', 'patch_embed.proj.weight' "
            f"not found. Got {len(encoder_sd)} keys. Check --vision_prefix."
        )

    return encoder_sd


def save_weight(weight, dtype, file, transpose=False):
    """Save a tensor to nntrainer format (optionally transposing OI -> IO)."""
    if isinstance(weight, np.ndarray):
        array = weight
    else:
        # Convert to float32 first to handle bfloat16 tensors (numpy doesn't
        # support bfloat16 natively)
        array = weight.detach().cpu().float().numpy()
    if transpose and array.ndim >= 2:
        array = array.T
    array.astype(dtype).tofile(file)


def convert(sd, output_path, dtype, cfg):
    dim = cfg["hidden_size"]
    num_layers = cfg["num_hidden_layers"]
    final_norm_key = cfg["final_norm_key"]

    # Verify essential keys
    if "patch_embed.proj.weight" not in sd:
        print("  [warn] 'patch_embed.proj.weight' not found. Available keys:")
        for k in list(sd.keys())[:20]:
            print("    ", k, tuple(sd[k].shape) if hasattr(sd[k], 'shape') else type(sd[k]))
        raise RuntimeError("Required key 'patch_embed.proj.weight' not found in state dict.")

    print(f"Writing nntrainer weights to: {output_path}")
    with open(output_path, "wb") as f:
        # 1. Patch embedding (Conv3d -> FC), modality embed folded into bias.
        pw = sd["patch_embed.proj.weight"]            # [embed, in_ch, kT, kH, kW]
        pw = pw.reshape(pw.shape[0], -1)              # [embed, in_ch*kT*kH*kW]
        save_weight(pw, dtype, f, transpose=True)     # -> [in, out]

        pb = sd["patch_embed.proj.bias"].clone()       # [embed]
        if "video_mod_embed" in sd:
            pb = pb + sd["video_mod_embed"].reshape(-1)
            print("  folded video_mod_embed into patch-embed bias")
        save_weight(pb, dtype, f)

        # 2. Transformer blocks.
        for i in range(num_layers):
            p = f"blocks.{i}."

            # norm1 -> attention_norm
            save_weight(sd[p + "norm1.weight"], dtype, f)
            save_weight(sd[p + "norm1.bias"], dtype, f)

            # fused qkv -> q, k, v (each transposed + bias)
            qkv_w = sd[p + "attn.qkv.weight"]          # [3*dim, dim]
            qkv_b = sd[p + "attn.qkv.bias"]            # [3*dim]
            for s in range(3):
                save_weight(qkv_w[s * dim:(s + 1) * dim, :], dtype, f, transpose=True)
                save_weight(qkv_b[s * dim:(s + 1) * dim], dtype, f)

            # attn.proj -> attention_out
            save_weight(sd[p + "attn.proj.weight"], dtype, f, transpose=True)
            save_weight(sd[p + "attn.proj.bias"], dtype, f)

            # norm2 -> ffn_norm
            save_weight(sd[p + "norm2.weight"], dtype, f)
            save_weight(sd[p + "norm2.bias"], dtype, f)

            # mlp.fc1 -> ffn_up, mlp.fc2 -> ffn_down
            save_weight(sd[p + "mlp.fc1.weight"], dtype, f, transpose=True)
            save_weight(sd[p + "mlp.fc1.bias"], dtype, f)
            save_weight(sd[p + "mlp.fc2.weight"], dtype, f, transpose=True)
            save_weight(sd[p + "mlp.fc2.bias"], dtype, f)

            print(f"  layer {i + 1}/{num_layers} done")

        # 3. Final norm (norms_block[-1] in the reference forward).
        save_weight(sd[final_norm_key + ".weight"], dtype, f)
        save_weight(sd[final_norm_key + ".bias"], dtype, f)

    print("Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert V-JEPA 2.1 encoder weights to nntrainer format")
    parser.add_argument("--input", type=str, required=True,
                        help="Input checkpoint (.pt or .safetensors)")
    parser.add_argument("--output", type=str, default="nntr_vjepa2_vitb_fp32.bin",
                        help="Output nntrainer weight file")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"])
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument("--checkpoint_key", type=str, default="ema_encoder",
                        help="Top-level state_dict key holding the encoder (for .pt only)")
    parser.add_argument("--final_norm_key", type=str, default="norms_block.3",
                        help="Key of the final LayerNorm (norms_block[-1])")
    parser.add_argument("--vision_prefix", type=str, default=None,
                        help="Vision tower key prefix in safetensors "
                             "(auto-detected if not specified)")
    args = parser.parse_args()

    dtype = {"float32": np.float32, "float16": np.float16}[args.dtype]
    cfg = dict(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        final_norm_key=args.final_norm_key,
    )

    if args.input.endswith(".safetensors"):
        print(f"Loading from safetensors: {args.input}")
        sd = load_encoder_state_from_safetensors(args.input, args.vision_prefix)
    elif args.input.endswith(".pt") or args.input.endswith(".bin"):
        print(f"Loading from PyTorch checkpoint ('{args.checkpoint_key}'): {args.input}")
        sd = load_encoder_state_from_pt(args.input, args.checkpoint_key)
    else:
        raise ValueError(f"Unsupported file format: {args.input}. Use .pt, .bin, or .safetensors")

    print(f"  Found {len(sd)} encoder keys")
    convert(sd, args.output, dtype, cfg)
