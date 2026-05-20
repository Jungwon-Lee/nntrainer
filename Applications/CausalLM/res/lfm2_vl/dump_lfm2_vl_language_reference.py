#!/usr/bin/env python3
"""Dump HF LFM2 language hidden states from merged multimodal embeddings."""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText
from transformers.models.lfm2.modeling_lfm2 import create_causal_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = np.load(args.reference)
    inputs_embeds = torch.from_numpy(reference["inputs_embeds_after_image_merge"]).to(torch.float32)
    attention_mask = torch.from_numpy(reference["attention_mask"]).to(torch.long)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32
    )
    model.eval()
    language_model = model.model.language_model
    config = language_model.config

    with torch.no_grad():
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        causal_mask = create_causal_mask(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        linear_attention = attention_mask if inputs_embeds.shape[1] != 1 else None
        hidden_states = inputs_embeds
        position_embeddings = language_model.rotary_emb(hidden_states, position_ids=position_ids)

        outputs = {"hidden_00": hidden_states.cpu().numpy().astype(np.float32)}
        for layer_id, decoder_layer in enumerate(language_model.layers[: config.num_hidden_layers]):
            layer_mask = (
                causal_mask
                if config.layer_types[layer_id] == "full_attention"
                else linear_attention
            )
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
            )
            outputs[f"hidden_{layer_id + 1:02d}"] = (
                hidden_states.cpu().numpy().astype(np.float32)
            )

        norm = language_model.embedding_norm(hidden_states)
        outputs["norm"] = norm.cpu().numpy().astype(np.float32)
        outputs["prefill_logits"] = model.lm_head(norm[:, -1:, :]).cpu().numpy().astype(np.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **outputs)
    print(f"Wrote {output_path}")
    for key, value in outputs.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")


if __name__ == "__main__":
    main()
