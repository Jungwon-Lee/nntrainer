#!/usr/bin/env python3
"""Decode generated token ids emitted by Lfm2VlPrefill."""

import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--ids-file", default="generated_ids.f32")
    args = parser.parse_args()

    ids_path = Path(args.input_dir) / args.ids_file
    ids = np.fromfile(ids_path, dtype=np.float32).astype(np.int64).tolist()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    clean_text = tokenizer.decode(ids, skip_special_tokens=True)

    print("ids:", ids)
    print("text:", repr(text))
    print("clean_text:", repr(clean_text))


if __name__ == "__main__":
    main()
