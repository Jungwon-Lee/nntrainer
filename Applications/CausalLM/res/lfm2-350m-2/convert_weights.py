"""Auto-generated weight conversion script."""
import torch
import sys
import os
from transformers import AutoModel, AutoModelForCausalLM
from my_vora_omni.src.model import Lfm2VLVJepa21BModel
from my_vora_omni.src.processor import Lfm2VLVJepa21BProcessor

# Weight mapping (HF key, transform)
WEIGHT_MAP = []
def make_weight_map(layer_types):
    # WEIGHT_MAP.append(("embed_tokens.weight", "none"))

    for i, layer_type in enumerate(layer_types):
        if layer_type == 'full_attention' or layer_type == 'attention':
            WEIGHT_MAP.extend([
                (f"layers.{i}.operator_norm.weight", "none"),
                (f"layers.{i}.self_attn.q_proj.weight", "transpose"),
                (f"layers.{i}.self_attn.q_layernorm.weight", "none"),
                (f"layers.{i}.self_attn.k_proj.weight", "transpose"),
                (f"layers.{i}.self_attn.k_layernorm.weight", "none"),
                (f"layers.{i}.self_attn.v_proj.weight", "transpose"),
                (f"layers.{i}.self_attn.out_proj.weight", "transpose"),
                (f"layers.{i}.ffn_norm.weight", "none"),
                (f"layers.{i}.feed_forward.w1.weight", "transpose"),
                (f"layers.{i}.feed_forward.w3.weight", "transpose"),
                (f"layers.{i}.feed_forward.w2.weight", "transpose"),
            ])
        elif layer_type == 'conv':
            WEIGHT_MAP.extend([
                (f"layers.{i}.operator_norm.weight", "none"),
                (f"layers.{i}.conv.in_proj.weight", "transpose"),
                (f"layers.{i}.conv.conv.weight", "conv_causal"),
                (f"layers.{i}.conv.out_proj.weight", "transpose"),
                (f"layers.{i}.ffn_norm.weight", "none"),
                (f"layers.{i}.feed_forward.w1.weight", "transpose"),
                (f"layers.{i}.feed_forward.w3.weight", "transpose"),
                (f"layers.{i}.feed_forward.w2.weight", "transpose"),
            ])
        else:
            raise ValueError(f'Unsupported Layer type: {layer_type}')

    WEIGHT_MAP.append(("embedding_norm.weight", "none"))
    WEIGHT_MAP.append(("embed_tokens.weight", "transpose"))
    return WEIGHT_MAP

layer_types = [
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv"
]

def convert(model_path, output_path, WEIGHT_MAP, dtype='float32'):
    target = torch.float32 if dtype == 'float32' else torch.float16
    model = Lfm2VLVJepa21BModel.from_pretrained(model_path, torch_dtype=torch.float32)
    model = model.model.language_model

    # model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
    sd = model.state_dict()

    with open(output_path, 'wb') as f:
        for hf_key, transform in WEIGHT_MAP:
            print(f"Save keys: {hf_key} in {dtype}")
            t = sd[hf_key].to(target)
            if transform == 'transpose' and t.dim() == 2:
                # Standard FC weight: [out, in] -> [in, out]
                t = t.t().contiguous()
            elif transform == 'conv_causal':
                # Causal depthwise conv weight conversion for CausalConv1DLayer.
                #
                # PyTorch DepthwiseConv1d weight shape: [filters, 1, kernel_size]
                #   = [1536, 1, 3]
                # Kernel dim semantics with left-pad=2, no bias:
                #   output[t, f] = w[f,0,2]*x[t] + w[f,0,1]*x[t-1] + w[f,0,0]*x[t-2]
                #   (kernel index 2 = current, 1 = t-1, 0 = t-2)
                #
                # CausalConv1DLayer weight shape: [1, 1, kernel_size, filters]
                #   = [1, 1, 3, 1536]
                # Memory layout: [w0_f0..w0_f1535, w1_f0..w1_f1535, w2_f0..w2_f1535]
                #   where w0 = weight for current (= PyTorch kernel[2]),
                #         w1 = weight for t-1    (= PyTorch kernel[1]),
                #         w2 = weight for t-2    (= PyTorch kernel[0])
                #
                # Conversion: [F, 1, K] -> squeeze -> [F, K]
                #              -> flip kernel axis -> [F, K_reversed]
                #              -> transpose -> [K, F]  (matches [1,1,K,F] layout)
                t = t.to(torch.float32)   # always save conv weights in fp32
                t = t.squeeze(1)          # [1536, 3]
                t = t.flip(1)             # [1536, 3]  kernel reversed: [cur, t-1, t-2]
                t = t.t().contiguous()    # [3, 1536]  = [K, F]
            f.write(t.cpu().numpy().tobytes())

    print(f'Saved {len(WEIGHT_MAP)} weight tensors to {output_path}')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python convert_weights.py <model_path> <output.bin> [float32|float16]')
        sys.exit(1)
    dtype = sys.argv[3] if len(sys.argv) > 3 else 'float32'

    weight_map = make_weight_map(layer_types)
    convert(sys.argv[1], sys.argv[2] + "model.bin", weight_map, dtype)

    weight_map = [("embed_tokens.weight", "none")]
    convert(sys.argv[1], sys.argv[2] + 'embed.bin', weight_map, dtype)
