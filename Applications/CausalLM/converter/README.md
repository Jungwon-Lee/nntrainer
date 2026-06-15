# NNTrainer CausalLM Weight Converter

Unified weight conversion tool that converts HuggingFace / timm models into the
NNTrainer weight format. A single entry point (`weight_converter.py`) dispatches
to per-model converter modules based on the model type.

## Layout

```
Applications/CausalLM/
├── converter/                 # conversion code (this directory)
│   ├── weight_converter.py    # unified entry point
│   ├── utils.py               # shared helpers (tensor I/O, safetensors, config saving)
│   ├── qwen2.py               # Qwen2 + KaLM-embedding
│   ├── qwen3.py               # Qwen3 dense (binary + safetensors)
│   ├── qwen3_moe.py           # Qwen3 MoE
│   ├── gemma3.py              # Gemma3 causal + embedding
│   ├── gpt_oss.py             # GPT-OSS 20B MoE
│   ├── deberta_v2.py          # DeBERTa V2 encoder
│   └── vit.py                 # timm ViT
└── res/                       # default output location (weights + config JSON)
```

## Supported models

| `--model_type` | Description | Loader | Notes |
|----------------|-------------|--------|-------|
| `qwen2`        | Qwen2 (e.g. `Qwen/Qwen2-0.5B`) | `AutoModelForCausalLM` | QKV bias, tied embedding |
| `qwen3`        | Qwen3 dense (e.g. `Qwen/Qwen3-0.6B`, `Qwen3-4B`) | `AutoModelForCausalLM` | Q/K norm, supports safetensors output |
| `qwen3_moe`    | Qwen3 MoE (e.g. `Qwen/Qwen3-30B-A3B`) | `AutoModelForCausalLM` | gate routing + per-expert FFN |
| `gemma3`       | Gemma3 causal LM | `AutoModelForCausalLM` | RMSNorm (+1), pre/post FFN norm |
| `gemma3_emb`   | Gemma3 embedding model | `SentenceTransformer` | embedding variant |
| `gpt_oss`      | GPT-OSS 20B MoE | `AutoModelForCausalLM` | attn bias + sink, split `gate_up_proj` |
| `kalm`         | KaLM-embedding | `SentenceTransformer` | Qwen2 backbone |
| `deberta_v2`   | DeBERTa V2 encoder | `DebertaV2Model` / `SentenceTransformer` | encoder-only, relative attention |
| `vit`          | timm ViT | checkpoint file | `--model_path` is a `.safetensors`/`.bin` file |

If `--model_type` is omitted, it is auto-detected from the HuggingFace config
(`config.model_type`). `gpt_oss` and `vit` usually need to be specified explicitly.

## Usage

```bash
# From Applications/CausalLM/
# Auto-detect model type; output defaults to res/<model>/nntr_<model>_<dtype>.bin
python converter/weight_converter.py --model_path Qwen/Qwen3-0.6B

# Explicit output path
python converter/weight_converter.py \
    --model_path Qwen/Qwen3-0.6B \
    --output_name /tmp/nntr_qwen3.bin

# float16 weights
python converter/weight_converter.py --model_path Qwen/Qwen2-0.5B --data_type float16

# safetensors output (Qwen3 only)
python converter/weight_converter.py --model_path Qwen/Qwen3-0.6B --safetensors

# Embedding variants
python converter/weight_converter.py --model_path <kalm_id>   --model_type kalm
python converter/weight_converter.py --model_path <gemma_id>  --model_type gemma3_emb

# GPT-OSS (non-standard model_type → specify explicitly)
python converter/weight_converter.py --model_path . --model_type gpt_oss

# timm ViT — model_path is a checkpoint file, not an HF hub id
python converter/weight_converter.py \
    --model_path ./model.safetensors --model_type vit \
    --patch_size 16 --img_size 224
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | (required) | HF model id, local directory, or checkpoint file (ViT) |
| `--output_name` | `res/<model>/nntr_<model>_<dtype>.bin` | Output weight file path |
| `--data_type` | `float32` | `float32` or `float16` |
| `--model_type` | auto-detect | Override model type detection |
| `--safetensors` | off | Save weights in safetensors format (Qwen3 only) |
| `--patch_size` | `16` | (ViT only) patch size in pixels |
| `--img_size` | `224` | (ViT only) input image size in pixels |

## Output files

Every conversion writes the weight file plus companion files into the **same
directory** as `--output_name`. By default each model gets its own subdirectory,
`res/<model_name>/`, so all of a model's artifacts stay grouped together:

| File | Source | When |
|------|--------|------|
| `nntr_<model>_<dtype>.bin` / `.safetensors` | converted weights | always |
| `nntr_config.json` | generated runtime config | always |
| `config.json` | `hf_config.save_pretrained()` | HF models |
| `generation_config.json` | `GenerationConfig.from_pretrained()` | if the model ships one |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | `AutoTokenizer.save_pretrained()` | HF models |
| `chat_template.jinja` | tokenizer | if the model defines a chat template |

The tokenizer (including the **chat template**) is re-emitted via
`save_pretrained`, so it follows the model automatically even when the source
repo only ships the standalone template file. This keeps the output directory
self-contained — `nntr_config.json`'s `tokenizer_file` points to the
`tokenizer.json` saved alongside it.

ViT loads from a checkpoint file (`--model_path` is not an HF directory), so it
skips `config.json` / `generation_config.json` / tokenizer saving and sets
`skip_tokenizer: true` in `nntr_config.json`.

## Adding a new model

1. Create `converter/<model>.py` exposing:

   ```python
   def convert(model_path, output_name, dtype, **kwargs):
       ...
   ```

2. Use the shared helpers in `utils.py`:
   - `save_tensor(file, tensor, dtype, transpose=False, add_one=False)`
   - `save_lora_or_weight(file, params, layer_name, proj_name, dtype, transpose=True)`
   - `save_safetensors(weights, output_path, dtype)` (for safetensors output)
   - `save_configs(output_name, dtype, hf_config=, model_path=, nntr_extra=)`

3. Define a module-level `SAMPLE_INPUT` constant and pass it through `nntr_extra`.

4. Register the model type in `_MODULE_MAP` (and `_EXTRA_KWARGS` if it needs
   extra flags) in `weight_converter.py`.

## Verification

There is no automatic numerical test bundled here. To validate a new or changed
converter, compare its byte output against a known-good reference:

```bash
python converter/weight_converter.py --model_path <id> --output_name new.bin
cmp old.bin new.bin   # no output means the files are identical
```
