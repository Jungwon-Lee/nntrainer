# Dense Qwen3.5/Qwen3.6 text models

This directory contains the text-only `Qwen3_5CausalLM` weight converter. It
supports official checkpoints whose architecture is
`Qwen3_5ForConditionalGeneration` or `Qwen3_5ForCausalLM`, including
`Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.6-27B`. Vision and MTP weights are omitted.

Use the 0.8B checkpoint as the small real-weight validation before attempting
the 27B dense model or the 35B-A3B MoE model:

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir Qwen3.5-0.8B

python3 Applications/CausalLM/res/qwen3_5/weight_converter.py \
  Qwen3.5-0.8B \
  --output-dir qwen3.5-0.8b-q4 \
  --target-isa arm \
  --init-seq-len 64 \
  --max-seq-len 256 \
  --num-to-generate 32
```

The converter memory-maps safetensors shards and writes Q4_0 a few output rows
at a time. Use `--target-isa arm` for q4_0x4 and `--target-isa x86` for
q4_0x8. The generated directory contains a portable `nntr_config.json`.
The narrow DeltaNet decay and beta projections remain FP32 because the dense
model's 16/48 output rows do not satisfy Q4_0's block-size constraint.

Before downloading a real checkpoint, run the non-zero tiny-model comparison.
It covers a DeltaNet layer, a gated full-attention layer with partial RoPE, the
dense SwiGLU MLP, FP32 loading, greedy decoding, and Q4_0 conversion:

```bash
meson setup build -Denable-transformer=true -Denable-test=true
ninja -C build Applications/CausalLM/unittest_causallm_models \
  Applications/CausalLM/nntr_quantize

NNTR_NUM_THREADS=1 \
NNTR_QUANTIZE_BIN="$PWD/build/Applications/CausalLM/nntr_quantize" \
build/Applications/CausalLM/unittest_causallm_models \
  --gtest_filter='Qwen3_5DifferentialTest.*'
```

The checked-in reference logits and tokens are produced by Transformers'
`Qwen3_5ForCausalLM`, not by another NNTrainer model.
