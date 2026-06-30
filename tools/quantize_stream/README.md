# Quick.AI Tools

## Streaming Quantizer

`quick_dot_ai_quantize_stream` converts an FP32 Quick.AI/NNTrainer model binary
to another dtype without loading the full model into memory. It is intended for
large transformer checkpoints where the original `quick_dot_ai_quantize` path
may require too much RAM.

Build:

```bash
meson setup build
ninja -C build quick_dot_ai_quantize_stream
```

Run:

```bash
./build/quick_dot_ai_quantize_stream <model_dir> <output_dir> [output_bin] [options]
```

Example:

```bash
./build/quick_dot_ai_quantize_stream \
  ./res/smallthinker/SmallThinker-4BA0.6B-Instruct \
  /tmp/smallthinker-q4 \
  --dtype Q4_0
```

The helper script builds the target if needed:

```bash
./scripts/quantize_stream.sh <model_dir> <output_dir> --dtype Q4_0
```

### Options

- `--dtype <type>`: set FC, embedding, and LM head dtype together.
- `--fc_dtype <type>`: set FC/projection dtype only. Default: `Q4_0`.
- `--embd_dtype <type>`: set embedding dtype only. Default: `FP32`.
- `--lmhead_dtype <type>`: set LM head dtype only. Default: `FP32`.
- `--output_bin <name>`: override the output `.bin` filename.
- `-h`, `--help`: print usage.

Supported dtypes:

- `FP32`
- `Q4_0`
- `Q4_K`
- `Q6_K`

### Supported Architectures

The stream quantizer reads `config.json` and selects a model recipe from
`architectures[0]`.

Currently supported:

- `LlamaForCausalLM`
- `Qwen2ForCausalLM`
- `Qwen3ForCausalLM`
- `Qwen3MoeForCausalLM`
- `Gemma3ForCausalLM`
- `Gemma3TextModel`
- `GptOssForCausalLM`
- `GptOssCachedSlimCausalLM`
- `SmallThinkerForCausalLM`

### Files

- `tools/quantize_stream.h`: shared types and interfaces.
- `tools/quantize_stream.cpp`: CLI, config parsing, metadata copy, and generic
  tensor streaming/quantization.
- `tools/quantize_stream_models.cpp`: model-specific tensor order recipes.

### Adding A Model

If a model reuses an existing tensor order, add only a new `registry.add(...)`
entry in `tools/quantize_stream_models.cpp`.

If a model has a new tensor order:

1. Add a new layer writer function in `tools/quantize_stream_models.cpp`.
2. Use `TensorWriter` helpers such as `copyFp32Tensor`,
   `writeTransposedMatrix`, and `quantizeFcWithBias`.
3. Register the model architecture with `registerBuiltInRecipes()`.
4. Rebuild and run `./build/quick_dot_ai_quantize_stream --help`.

The input binary must match the exact weight order requested by the Quick.AI
model builder for that architecture. Unsupported architectures fail explicitly
instead of guessing the layout.
