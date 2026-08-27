# Qwen3.6-35B-A3B text-only Q4_0 conversion

This converter produces the weight order used by NNTrainer's
`Qwen3_5MoeCausalLM` implementation. It covers the language backbone only:
Gated DeltaNet, gated full attention, routed experts, the shared expert, and
the untied LM head. The vision encoder and MTP head are intentionally omitted.

The official checkpoint is about 72 GB in BF16 and includes vision weights.
Allow roughly 95 GB of free disk while keeping the source snapshot and the
text-only Q4_0 output together. The converter memory-maps shards and slices the
large expert tensors, so peak conversion RAM is controlled by `--chunk-rows`.

## 1. Download the checkpoint

```bash
python3 -m pip install "huggingface_hub[cli]" safetensors torch numpy
hf download Qwen/Qwen3.6-35B-A3B \
  --local-dir /models/Qwen3.6-35B-A3B
```

## 2. Convert for the target CPU

Android/ARM:

```bash
python3 weight_converter.py /models/Qwen3.6-35B-A3B \
  --output-dir /models/Qwen3.6-35B-A3B-nntrainer-q4 \
  --target-isa arm
```

x86:

```bash
python3 weight_converter.py /models/Qwen3.6-35B-A3B \
  --output-dir /models/Qwen3.6-35B-A3B-nntrainer-q4-x86 \
  --target-isa x86
```

Q4_0 FC data is ISA-repacked (`q4_0x4` on ARM, `q4_0x8` on x86), so use the
output matching the runtime CPU. Embedding data stays in canonical Q4_0 lookup
layout. Small norms, DeltaNet convolution coefficients/state parameters, and
the narrow DeltaNet decay/beta projections and shared-expert scalar gate remain
FP32.

The output directory receives the converted `.bin`, runtime metadata, and an
initial `nntr_config.json` with a deliberately small 64-token prefill and
256-token maximum context. Increase those limits only after the first smoke
test succeeds.

## 3. Run the dense prerequisite and MoE smoke tests

Validate the shared dense Qwen3.5/Qwen3.6 backbone before the MoE extension:

```bash
meson setup build -Denable-transformer=true -Denable-test=true
ninja -C build Applications/CausalLM/unittest_causallm_models \
  Applications/CausalLM/nntr_quantize

NNTR_NUM_THREADS=1 \
NNTR_QUANTIZE_BIN="$PWD/build/Applications/CausalLM/nntr_quantize" \
build/Applications/CausalLM/unittest_causallm_models \
  --gtest_filter='Qwen3_5DifferentialTest.*'
```

Then validate routed/shared experts on the same hybrid backbone:

```bash
NNTR_NUM_THREADS=1 \
NNTR_QUANTIZE_BIN="$PWD/build/Applications/CausalLM/nntr_quantize" \
build/Applications/CausalLM/unittest_causallm_models \
  --gtest_filter='Qwen3_5MoeDifferentialTest.*:Qwen3_5Moe/*'
```

Then run the converted checkpoint:

```bash
./build/Applications/CausalLM/nntr_causallm \
  /models/Qwen3.6-35B-A3B-nntrainer-q4
```

Conversion does not prove model-level numerical parity by itself. Before a
long generation run, verify the tiny FP32/Q4_0 unit test and compare a short
prompt's logits or generated prefix against Hugging Face on the same tokens.
