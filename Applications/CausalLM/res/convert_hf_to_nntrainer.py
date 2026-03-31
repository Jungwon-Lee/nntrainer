#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# @file convert_hf_to_nntrainer.py
# @brief Unified weight conversion script for HuggingFace models to nntrainer format
# @author Jungwon-Lee <jungone.lee@samsung.com>
#
# This script converts HuggingFace models to nntrainer binary format, supporting both
# CausalLM (text generation) and Sentence Transformer/Embedding (text encoding) models.

import argparse

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.models import (
    Qwen3ForCausalLM,
    Qwen3MoeModel,
    Gemma3ForCausalLM,
    Gemma3Model,
)

def save_model_for_nntrainer(model, config, dtype, output_path):
    """Dispatch to appropriate model-specific converter based on model type."""
    print(f"[Info] Detected {type(model).__name__}")
    
    # Dispatch to appropriate converter based on model type
    if isinstance(model, Qwen3ForCausalLM):
        from qwen3.weight_converter import save_qwen3_for_nntrainer as save_model
    elif isinstance(model, Qwen3MoeModel):
        from qwen3.weight_converter import save_qwen3_moe_for_nntrainer as save_model
    elif isinstance(model, Gemma3ForCausalLM):
        from gemma3.weight_converter import save_gemma3_for_nntrainer as save_model
    else:
        raise NotImplementedError(
            f"Model type '{type(model).__name__}' is not supported yet. "
        )

    # Save weights using the dispatched converter
    with open(output_path, "wb") as f_model:
        save_model(model.state_dict(), config, dtype, f_model)
    print(f"[Success] Model weights saved to: {output_path}")


def add_padding_for_embedding(model, padding_size=32):
    def padding(original_tensor):
        padding_needed = (padding_size - (original_tensor.size(0) % padding_size)) % padding_size
        padded_tensor = torch.nn.functional.pad(original_tensor.T, (0, padding_needed), mode='constant', value=0).T

        return padded_tensor

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding) and module.weight.size(0) % padding_size != 0:
            print(f"[Padding] Embedding '{name}' is not divided by {padding_size}.")

            padded_weight = padding(module.weight)
            new_embedding = torch.nn.Embedding(*padded_weight.shape)
            new_embedding.weight.data.copy_(padded_weight)

            print(f"[Padding] Add padding for Embedding '{name}' ({module.weight.size(0)} -> {new_embedding.weight.size(0)})\n")
            model.set_submodule(name, new_embedding)

    return model


def main():
    """Main function to parse arguments and execute model conversion."""
    parser = argparse.ArgumentParser(description="Convert HuggingFace models to nntrainer format")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the HuggingFace model directory")
    parser.add_argument("--output_path", type=str, required=True, 
                        help="Path where the converted weights will be saved")
    parser.add_argument("--embedding", action='store_true', 
                        help="Load model as Sentence Transformer/Embedding model using AutoModel (no LM head).")
    parser.add_argument("--padding", action='store_true', 
                        help="Add padding to embedding layers for Q4_0 quantization")
    parser.add_argument("--data_type", type=str, default="float32", choices=["float32"], 
                        help="Data type for weight conversion (default: float32)")
    
    args = parser.parse_args()
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Info] Using device: {device}")
    
    # Load model configuration and model
    print(f"[Info] Loading model from: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)

    if args.embedding:
        model = AutoModel.from_pretrained(
            args.model_path,
            dtype=args.data_type,
            trust_remote_code=True
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=args.data_type,
        trust_remote_code=True
    ).to(device)
    model.eval()
    print(f"[Info] Model loaded successfully")
    
    # Add padding for Q4_0 quantization if requested
    if args.padding:
        print("[Info] Adding padding for embedding layers...")
        model = add_padding_for_embedding(model)
    
    # Save model in nntrainer format
    save_model_for_nntrainer(model, config, args.data_type, args.output_path)


if __name__ == "__main__":
    '''
    Usage:
    python convert_hf_to_nntrainer.py --model_path /path/to/model --output_path output.bin [options]

    Examples:
    # Convert a CausalLM model (Qwen3)
    python convert_hf_to_nntrainer.py --model_path ./qwen3-0.6b --output_path qwen3_0.6b_fp32.bin

    # Convert a Sentence Transformer model (Qwen3-embedding) with --embedding flag
    python convert_hf_to_nntrainer.py --model_path ./qwen3-0.6b --output_path qwen3_0.6b_embedding_fp32.bin --embedding

    # Convert with padding for Q4_0 quantization
    python convert_hf_to_nntrainer.py --model_path ./gemma3-270m --output_path gemma3_270m_fp32.bin --padding

    Model Types:
    - CausalLM (default): Loads with AutoModelForCausalLM, includes LM head for text generation
    - Embedding (--embedding flag): Loads with AutoModel, encoder-only for text embeddings
    '''
    main()
