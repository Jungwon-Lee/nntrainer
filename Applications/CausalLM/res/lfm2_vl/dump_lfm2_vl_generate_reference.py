#!/usr/bin/env python3
"""Run greedy HF language generation from merged LFM2-VL embeddings."""

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-tokens", type=int, default=16)
    args = parser.parse_args()

    reference = np.load(args.reference)
    inputs_embeds = torch.from_numpy(
        reference["inputs_embeds_after_image_merge"]
    ).to(torch.float32)
    attention_mask = torch.from_numpy(reference["attention_mask"]).to(torch.long)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.float32
    )
    model.eval()

    ids = []
    with torch.no_grad():
        outputs = model.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
        )
        logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])[:, -1, :]
        next_id = int(logits.argmax(dim=-1).item())
        ids.append(next_id)
        past_key_values = outputs.past_key_values

        for _ in range(args.num_tokens - 1):
            input_ids = torch.tensor([[next_id]], dtype=torch.long)
            outputs = model.model.language_model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = model.lm_head(outputs.last_hidden_state[:, -1:, :])[:, -1, :]
            next_id = int(logits.argmax(dim=-1).item())
            ids.append(next_id)
            past_key_values = outputs.past_key_values

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids_array = np.asarray(ids, dtype=np.int64)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, generated_ids=ids_array)

    print(f"Wrote {output_path}")
    print("ids:", ids)
    print("text:", repr(tokenizer.decode(ids, skip_special_tokens=False)))
    print("clean_text:", repr(tokenizer.decode(ids, skip_special_tokens=True)))


if __name__ == "__main__":
    main()
